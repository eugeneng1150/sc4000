"""V3: two-fold screening, controlled coverage/data comparisons, confirmation and export."""
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time
import zipfile
import numpy as np
import pandas as pd

from study_engine import GIB,VERSION,digest,write_json,metadata,release_memory
from study_common import execute,folder_bytes,frame_score,aligned_predictions,paired_summary,validate_config
from study_plan import (default_config,manifest,coverage_plan,followup_plan,tabular_plan,
                        effective_config,profile)


def make_bundle(output,reserve_gib=1.):
    output=Path(output);target=output.parent/(output.name+'_results.zip')
    required=folder_bytes(output)*1.05+reserve_gib*GIB
    if shutil.disk_usage(output.parent).free<required:
        raise OSError('Insufficient free disk to build a safe checkpoint ZIP; saved result files remain intact')
    temporary=target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file() and not path.name.endswith('.tmp'):
                archive.write(path,path.relative_to(output))
    temporary.replace(target)
    return target


def restore_archive(archive_path,output,limit_gib):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise ValueError('Restore into an empty result directory only')
    with zipfile.ZipFile(archive_path) as archive:
        if sum(i.file_size for i in archive.infolist())>limit_gib*GIB:
            raise ValueError('Archive exceeds output limit')
        for item in archive.infolist():
            dest=(output/item.filename).resolve()
            if output.resolve() not in dest.parents:raise ValueError('Unsafe archive path')
        if archive.testzip() is not None:raise ValueError('Damaged checkpoint ZIP')
        archive.extractall(output)


def scientific_config(cfg):
    resources={'worker_ram_gib','system_reserve_gib','scratch_limit_gib','output_limit_gib',
               'disk_reserve_gib','job_minutes','extraction_minutes','session_hours','gnn_job_minutes',
               'inference_minutes','checkpoint_minutes','sample_submission','retry_failed','clean_scratch'}
    # Probe thresholds affect scientific eligibility, so they remain part of run identity.
    return {k:v for k,v in cfg.items() if k not in resources}


def trial_dir(directory,fold,exp,seed):
    return Path(directory)/'folds'/str(fold)/'trials'/exp/str(seed)


def finished(directory,exp,cfg,seeds=None,action='trial'):
    seeds=cfg['seeds'] if seeds is None else seeds
    marker='status.json' if action=='trial' else 'holdout_status.json'
    result='result.json' if action=='trial' else 'holdout_result.json'
    return all((trial_dir(directory,f,exp,s)/result).exists() and
               (trial_dir(directory,f,exp,s)/marker).exists() and
               json.loads((trial_dir(directory,f,exp,s)/marker).read_text())['status']=='ok'
               for f in cfg['folds'] for s in seeds)


def prediction_sets(directory,names,cfg,seeds,split):
    return {fold:aligned_predictions(Path(directory)/'folds'/str(fold),names,seeds,split)
            for fold in cfg['folds']}


def method_scores(directory,names,cfg,seeds,collection):
    frames=prediction_sets(directory,names,cfg,seeds,'dev');tile=collection.startswith('tile')
    return {name:[frame_score(frames[f][name],tile) for f in cfg['folds']] for name in names}


def select_coverage(directory,catalog,cfg,collection):
    path=Path(directory)/'coverage_lock.json'
    if path.exists():return json.loads(path.read_text())
    candidates=[e for e in catalog if e['name'].startswith('G_full_') and
                finished(directory,e['name'],cfg,[cfg['seeds'][0]])]
    if not candidates:raise RuntimeError('No complete coverage candidate; inspect failures before resuming')
    values=method_scores(directory,[e['name'] for e in candidates],cfg,[cfg['seeds'][0]],collection)
    # Tie break toward smaller graph budgets; labels only come from development folds.
    best=max(candidates,key=lambda e:(np.mean(values[e['name']]),-e['profile']['nodes']))
    lock=dict(experiment=best['name'],profile=best['profile'],fold_scores=values,
              source='development folds, screening seed only',selected_mean=float(np.mean(values[best['name']])) )
    write_json(path,lock);return lock


def confirmation_plan(directory,catalog,cfg,collection,coverage):
    path=Path(directory)/'confirmation_plan.json'
    if path.exists():return json.loads(path.read_text())
    complete=[e['name'] for e in catalog if finished(directory,e['name'],cfg,[cfg['seeds'][0]])]
    required=['B0_basic_hgb','T_full_small']
    if coverage:
        base=cfg['graph_budgets'][0];chosen=coverage['profile']['nodes']
        required += [f'G_reference_n{base}',f'G_full_n{base}',f'G_reference_n{chosen}',f'G_full_n{chosen}']
        # Confirm the fixed 2×2 corners: data changes can be separated and their interaction measured.
        max_c=max(cfg['data_config_counts']);max_g=0 if 0 in cfg['data_graph_counts'] else max(cfg['data_graph_counts'])
        for g,c in [(24,64),(24,max_c),(max_g,64),(max_g,max_c)]:
            tag=f'g{g or "all"}_c{c}'
            required += [f'G_full_n{chosen}' if (g,c)==(24,64) else f'G_data_{tag}',f'H_matched_{tag}']
        required += [f'G_masks_n{chosen}',f'G_context_n{chosen}',f'G_no_messages_n{chosen}',f'G_connected_n{chosen}']
    scores=method_scores(directory,complete,cfg,[cfg['seeds'][0]],collection)
    for neural in [False,True]:
        group=[n for n in complete if n.startswith('G')==neural]
        required += sorted(group,key=lambda n:np.mean(scores[n]),reverse=True)[:cfg['confirmation_top_k']]
    all_names={e['name'] for e in catalog}
    required=list(dict.fromkeys(n for n in required if n in all_names))
    plan=dict(names=required,screened_complete=complete,screening_fold_scores=scores,
              note='All specified controls plus development-selected leaders; confirmation is not independent selection data')
    write_json(path,plan);return plan


def lock_selection(directory,catalog,cfg,collection,names):
    path=Path(directory)/'selection_lock.json'
    if path.exists():return json.loads(path.read_text())
    complete=[n for n in names if finished(directory,n,cfg)]
    if 'B0_basic_hgb' not in complete:raise RuntimeError('Baseline must finish every fold/seed')
    if set(complete)!=set(names):raise RuntimeError('Confirmation is incomplete; no final selection was made')
    pred=prediction_sets(directory,complete,cfg,cfg['seeds'],'dev');tile=collection.startswith('tile')
    values={n:[frame_score(pred[f][n],tile) for f in cfg['folds']] for n in complete}
    tabular=max([n for n in complete if not n.startswith('G')],key=lambda n:np.mean(values[n]))
    neural=[n for n in complete if n.startswith('G')]
    gnn=max(neural,key=lambda n:np.mean(values[n])) if neural else None
    candidates=[]
    for base in list(dict.fromkeys(['B0_basic_hgb',tabular])):
        for weight in ([0.,.25,.5,.75,1.] if gnn else [0.]):
            components={base:1-weight}
            if gnn:components[gnn]=weight
            components={k:v for k,v in components.items() if v>0}
            fold_scores=[]
            for fold in cfg['folds']:
                frame=pred[fold][base].copy()
                frame['prediction']=sum(w*pred[fold][n].prediction for n,w in components.items())
                fold_scores.append(frame_score(frame,tile))
            delta=np.array(fold_scores)-np.array(values['B0_basic_hgb'])
            candidates.append(dict(components=components,fold_scores=fold_scores,mean_score=float(np.mean(fold_scores)),
                                   eligible=bool(np.min(delta)>=-cfg['fold_regression_tolerance'])))
    selected=max([c for c in candidates if c['eligible']],key=lambda c:c['mean_score'])
    lock=dict(components=selected['components'],winner=tabular,gnn=gnn,confirmed=complete,
        seeds=cfg['seeds'],folds=cfg['folds'],dev_score=selected['mean_score'],fold_scores=selected['fold_scores'],
        method_fold_scores=values,blend_grid=candidates,
        source='development only; each fold uses only models trained without that fold',
        fold_regression_tolerance=cfg['fold_regression_tolerance'],
        locked_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    write_json(path,lock);return lock


def factor_contrasts(graphs,cfg,coverage,seeds):
    """Explicit paired changes; each row states the one experimental factor changed."""
    pairs=[]
    def add(factor,reference,candidate):
        if reference!=candidate:pairs.append((factor,reference,candidate))
    for candidate,factor,reference in [
        ('T_full_small','tabular features','B0_basic_hgb'),
        ('T_more_graphs','training graphs','T_full_small'),
        ('T_more_configs','training configurations','T_full_small'),
        ('T_more_both','training configurations','T_more_graphs'),
        ('T_more_both','training graphs','T_more_configs'),
        ('T_deeper_hgb','HGB capacity','T_more_graphs'),
        ('T_extra_trees','tabular estimator','T_more_graphs')]:add(factor,reference,candidate)
    if coverage:
        chosen=coverage['profile']['nodes'];base=cfg['graph_budgets'][0]
        reference=f'G_reference_n{chosen}';full=f'G_full_n{chosen}'
        for n in cfg['graph_budgets']:
            for recipe in ['reference','full']:
                add('graph coverage',f'G_{recipe}_n{base}',f'G_{recipe}_n{n}')
        for recipe,factor in [('masks','mask features'),('context','context features'),('full','masks and context')]:
            add(factor,reference,f'G_{recipe}_n{chosen}')
        add('message passing',f'G_no_messages_n{chosen}',full)
        add('connected sampling',full,f'G_connected_n{chosen}')
        def name(model,g,c):
            return full if model=='G_data' and (g,c)==(24,64) else f'{model}_g{g or "all"}_c{c}'
        # Include fixed endpoint contrasts even when intermediate sizes are screening-only.
        for model in ['G_data','H_matched']:
            for g in cfg['data_graph_counts']:
                for c in cfg['data_config_counts']:
                    add('training configurations',name(model,g,64),name(model,g,c))
                    add('training graphs',name(model,24,c),name(model,g,c))
        for g in cfg['data_graph_counts']:
            for c in cfg['data_config_counts']:
                add('GNN versus matched HGB',name('H_matched',g,c),name('G_data',g,c))
    available=set(graphs.experiment);rows=[]
    for factor,reference,candidate in dict.fromkeys(pairs):
        if {reference,candidate}<=available:
            table=paired_summary(graphs[graphs.experiment.isin([reference,candidate])],
                                 seeds,cfg['bootstrap_repeats'],reference)
            if table.empty:continue
            for row in table[table.experiment==candidate].to_dict('records'):
                rows.append(dict(factor=factor,**row))
    return pd.DataFrame(rows)


def summarize(directory,cfg,collection):
    directory=Path(directory);all_results=[];catalog_path=directory/'experiment_plan.json'
    catalog=json.loads(catalog_path.read_text()) if catalog_path.exists() else []
    for p in directory.glob('folds/*/trials/*/*/result.json'):
        status=p.parent/'status.json'
        if status.exists() and json.loads(status.read_text())['status']=='ok':
            row=json.loads(p.read_text());row.update(json.loads(status.read_text()));row['fold']=int(p.parents[3].name)
            all_results.append(row)
    if all_results:pd.DataFrame(all_results).to_csv(directory/'all_trials.csv',index=False)
    for split in ['dev','holdout']:
        frames=[]
        for p in directory.glob(f'folds/*/trials/*/*/{split}_graphs.csv'):
            marker=p.parent/('status.json' if split=='dev' else 'holdout_status.json')
            if marker.exists() and json.loads(marker.read_text())['status']=='ok':
                f=pd.read_csv(p);f['fold']=int(p.parents[3].name);frames.append(f)
        if not frames:continue
        graphs=pd.concat(frames,ignore_index=True);graphs.to_csv(directory/f'{split}_graph_scores.csv',index=False)
        graphs.groupby(['experiment','fold','seed']).score.mean().reset_index().to_csv(directory/f'{split}_fold_seed_scores.csv',index=False)
        # Screening and confirmed comparisons are labelled separately; never mix seed counts.
        for scope,seeds in [('screening',[cfg['seeds'][0]]),('confirmed',cfg['seeds'])]:
            names=[e['name'] for e in catalog if finished(directory,e['name'],cfg,seeds,'trial' if split=='dev' else 'holdout')]
            subset=graphs[graphs.experiment.isin(names)&graphs.seed.isin(seeds)]
            if subset.empty:continue
            reduced=subset.groupby(['experiment','graph','seed'],as_index=False).score.mean()
            refs=['B0_basic_hgb',f'G_reference_n{cfg["graph_budgets"][0]}',f'G_full_n{cfg["graph_budgets"][0]}']
            lock_path=directory/'coverage_lock.json'
            if lock_path.exists():refs += [json.loads(lock_path.read_text())['experiment'],'H_matched_g24_c64']
            for ref in dict.fromkeys(refs):
                if ref in names:
                    paired_summary(reduced,seeds,cfg['bootstrap_repeats'],ref).to_csv(
                        directory/f'{split}_{scope}_vs_{ref}.csv',index=False)
            coverage=json.loads(lock_path.read_text()) if lock_path.exists() else None
            contrasts=factor_contrasts(reduced,cfg,coverage,seeds)
            if not contrasts.empty:
                contrasts.to_csv(directory/f'{split}_{scope}_factor_contrasts.csv',index=False)
    lock_path=directory/'selection_lock.json'
    if not lock_path.exists():return
    lock=json.loads(lock_path.read_text());names=list(dict.fromkeys(['B0_basic_hgb',lock['winner']]+list(lock['components'])))
    if not all(finished(directory,n,cfg,action='holdout') for n in names):return
    by_fold=prediction_sets(directory,names,cfg,cfg['seeds'],'holdout');pred={}
    for name in names:
        first=by_fold[cfg['folds'][0]][name].copy()
        for fold in cfg['folds'][1:]:
            if not first[['graph','config_id','runtime']].equals(by_fold[fold][name][['graph','config_id','runtime']]):
                raise ValueError('Holdout labels/IDs differ across folds')
        first['prediction']=np.mean([by_fold[f][name].prediction.to_numpy() for f in cfg['folds']],axis=0);pred[name]=first
    blend=pred['B0_basic_hgb'].copy();blend['prediction']=sum(w*pred[n].prediction for n,w in lock['components'].items())
    rows=[]
    for name,frame in [('baseline',pred['B0_basic_hgb']),('tabular',pred[lock['winner']]),('selected',blend)]:
        frame.to_csv(directory/f'{name}_holdout_predictions.csv.gz',index=False,compression='gzip')
        for graph,g in frame.groupby('graph'):
            from study_engine import scores
            rows.append(dict(experiment=name,graph=graph,seed=0,**scores(g.runtime.to_numpy(),g.prediction.to_numpy(),collection.startswith('tile'))))
    table=pd.DataFrame(rows);table.to_csv(directory/'system_holdout_graphs.csv',index=False)
    paired_summary(table,[0],cfg['bootstrap_repeats'],'baseline').to_csv(directory/'system_holdout_gains.csv',index=False)


def write_summary(output,cfg,states):
    output=Path(output)
    write_json(output/'run_state.json',dict(collections=states,config=cfg))
    lines=['# V3 checkpoint','', 'Status refers to completed workers, not a promised runtime or leaderboard gain.',
           'Development folds choose profiles, data recipes and blends. Official valid is an already-inspected audit.',
           'Screening uses the first seed on every fold; key controls and finalists use all seeds.',
           'Fold-specific models only evaluate their own development fold. Test/audit ensembles average all fold/seed models.',
           '','| Collection | Status |','|---|---|']
    lines += [f'| {c} | {state} |' for c,state in states.items()]
    lines += ['', '## Interpretation', '',
        '- `experiment_plan.json`: exact stages, graph caps, training sizes and feature recipes.',
        '- `probes/`: label-free admission pilots; failed budgets are explicit resource exclusions.',
        '- `coverage_lock.json`: development-selected graph budget, before the training-data sweep.',
        '- `confirmation_plan.json`: repeated comparisons, separate from one-seed screening.',
        '- `selection_lock.json`: selected model/blend, including per-fold regression checks.',
        '- `preparations/`: per-profile/fold manifests and actual retained-node/edge/isolation diagnostics.',
        '- `*_confirmed_vs_*.csv`: paired comparisons with all seeds. Screening tables are exploratory.',
        '- `*_factor_contrasts.csv`: explicit coverage, data-size, component and matched-estimator changes.',
        '- `system_holdout_gains.csv`: ensemble audit results, not official Kaggle scores.',
        '- `submission_*.csv`: all test candidates scored, using template-sized ranked outputs.',
        '', '## Recovery', '',
        'Only ZIPs successfully downloaded to your computer survive deletion of local Colab storage.',
        'Browser download requests can be blocked or missed during disconnection. No Google Drive backup is enabled.',
        'Resume a trusted v3 ZIP with unchanged scientific settings and package versions. V2 ZIPs are not v3 checkpoints.',
        'Completed jobs are reused; self-contained GNN epoch checkpoints retain optimizer, RNG and best-model state.',
        'A session-budget stop is incomplete work. Re-run the same notebook; it does not require new experiment choices.',
        '', '## Limits', '',
        'Larger training sets change both information and optimizer steps under a fixed epoch limit.',
        'The configurable-node seeds are nested by budget; full retained graphs need not be strict subsets.',
        'Intervals resample whole graphs and are exploratory, without multiple-comparison correction.',
        'The two folds are training variations, not independent copies of the holdout.',
        'No guarantee of an OOM-free run or of reaching top 10%.', '']
    (output/'RUN_SUMMARY.md').write_text('\n'.join(lines))


def run_study(root,output,scratch,engine_path,cfg,on_checkpoint=None):
    validate_config(cfg)
    if not cfg['folds'] or len(set(cfg['folds']))!=len(cfg['folds']):raise ValueError('Distinct folds required')
    if max(cfg['data_config_counts']+[cfg['large_fit_configs']])>cfg['train_pool_configs']:raise ValueError('Training pool is too small')
    if 24 not in cfg['data_graph_counts'] or 64 not in cfg['data_config_counts']:raise ValueError('Keep the 24 graph / 64 configuration control')
    root,output,scratch=map(lambda p:Path(p).resolve(),[root,output,scratch]);engine_path=Path(engine_path).resolve()
    if any(a==b or a in b.parents or b in a.parents for a,b in [(root,scratch),(output,scratch),(root,output)]):
        raise ValueError('Data, scratch and output directories must be separate')
    output.mkdir(parents=True,exist_ok=True);scratch.mkdir(parents=True,exist_ok=True)
    marker=scratch/'.v3_owned'
    if not marker.exists() and any(scratch.iterdir()):raise ValueError('Refusing nonempty unmarked scratch directory')
    marker.touch();versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','torch','joblib','psutil']}
    sources={p.name:digest(p.read_text()) for p in engine_path.parent.glob('study_*.py')}
    signature=digest(dict(cfg=scientific_config(cfg),versions=versions,sources=sources))
    identity=output/'run_identity.json'
    if identity.exists() and json.loads(identity.read_text())['signature']!=signature:
        raise ValueError('Code/settings/packages changed, or this is a v2 checkpoint. Use the matching v3 code/settings.')
    write_json(identity,dict(version=VERSION,signature=signature,versions=versions,config=cfg,sources=sources))
    for path in engine_path.parent.glob('study_*.py'):
        target=output/path.name
        if target.exists() and target.read_bytes()!=path.read_bytes():raise ValueError('Archived code differs')
        if not target.exists():shutil.copy2(path,target)
    deadline=time.monotonic()+cfg['session_hours']*3600;last_checkpoint=time.monotonic()
    states={c:'pending' for c in cfg['collections']};active=None

    def checkpoint(force=False):
        nonlocal last_checkpoint
        if not force and time.monotonic()-last_checkpoint<cfg['checkpoint_minutes']*60:return
        write_summary(output,cfg,states)
        archive=make_bundle(output,cfg['disk_reserve_gib'])
        write_json(output/'last_checkpoint.json',dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),archive=archive.name,
                   note='ZIP created; browser/computer receipt cannot be verified by the notebook'))
        if on_checkpoint:on_checkpoint(archive)
        last_checkpoint=time.monotonic()
        print(f'Checkpoint ZIP: {archive}',flush=True)

    def worker(job,status,limits=None):
        if folder_bytes(output)>cfg['output_limit_gib']*GIB:raise OSError('Output budget reached; preserve/download results')
        result=execute(job,engine_path,limits or cfg,deadline,status,heartbeat=checkpoint)
        checkpoint();return result

    def cleanup():
        work=scratch/'work'
        if marker.exists() and work.exists():shutil.rmtree(work)

    def prepare_for(collection,directory,fold,exp):
        effective=effective_config(cfg,exp,collection);m=manifest(root,collection,cfg,fold)
        manifest_path=directory/'manifests'/f'fold_{fold}.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text())!=m:raise ValueError('Source manifest changed')
        write_json(manifest_path,m)
        geometry=exp.get('profile') if exp['model']=='gnn' else None
        key=digest(dict(manifest=m,geometry=geometry,version=VERSION))[:16]
        work=scratch/'work';cache=work/'cache_identity.json'
        if cache.exists() and json.loads(cache.read_text())['key']==key and (work/'ready.json').exists():return effective,work,True
        cleanup();work.mkdir(exist_ok=True)
        prep=directory/'preparations'/f'fold_{fold}_{key}';prep.mkdir(parents=True,exist_ok=True)
        status=prep/'status.json';status.unlink(missing_ok=True)
        ok=worker(dict(action='prepare',cfg=effective,root=str(root),manifest=m,scratch=str(work),output=str(prep)),status)
        if ok:write_json(cache,dict(key=key))
        return effective,work,ok

    def run_jobs(collection,directory,catalog,seeds,action='trial'):
        ok_all=True
        jobs=[(fold,e,seed) for fold in cfg['folds'] for e in catalog for seed in seeds]
        jobs.sort(key=lambda x:(x[0],json.dumps(x[1].get('profile') if x[1]['model']=='gnn' else None,sort_keys=True),x[1]['name'],x[2]))
        for fold,exp,seed in jobs:
            path=trial_dir(directory,fold,exp['name'],seed);path.mkdir(parents=True,exist_ok=True)
            marker_path=path/('status.json' if action=='trial' else 'holdout_status.json')
            if marker_path.exists() and json.loads(marker_path.read_text())['status']=='ok':continue
            if time.monotonic()>=deadline:return False
            effective,work,ready=prepare_for(collection,directory,fold,exp)
            if not ready:ok_all=False;continue
            print(f'{collection} | {exp["stage"]} | fold {fold} | {exp["name"]} | seed {seed} | {action}',flush=True)
            ok=worker(dict(action=action,cfg=effective,experiment=exp,seed=seed,collection=collection,
                           scratch=str(work),output=str(path)),marker_path)
            ok_all=ok_all and bool(ok)
            if ok is None:return False
        return ok_all

    try:
        # Verify every fold/source before allowing completed-job shortcuts.
        for collection in cfg['collections']:
            directory=output/collection.replace(':','_');directory.mkdir(exist_ok=True)
            for fold in cfg['folds']:
                m=manifest(root,collection,cfg,fold);p=directory/'manifests'/f'fold_{fold}.json'
                if p.exists() and json.loads(p.read_text())!=m:raise ValueError('Source manifest changed; do not mix data versions')
                write_json(p,m)
        for collection in cfg['collections']:
            active=collection;directory=output/collection.replace(':','_')
            if (directory/'complete.json').exists():states[collection]='complete (resumed)';continue
            if time.monotonic()>=deadline:states[collection]='incomplete: session budget';continue
            catalog=tabular_plan(collection,cfg);coverage=None
            if collection in cfg['gnn_collections']:
                states[collection]='profiling graph budgets'
                # Probe the largest training graphs by three independent static dimensions, with no runtime labels.
                m=json.loads((directory/'manifests'/f'fold_{cfg["folds"][0]}.json').read_text())
                train=[root/r['relative_path'] for r in m['records'] if r['split']=='train']
                stats=[(p,metadata(p,'node_feat')[0][0],metadata(p,'node_config_feat')[0][1],metadata(p,'edge_index')[0][0]) for p in train]
                probe_paths=list(dict.fromkeys(str(max(stats,key=lambda x:x[k])[0]) for k in [1,2,3]))[:cfg['probe_graphs']]
                eligible=[]
                for nodes in cfg['graph_budgets']:
                    exp=next(e for e in coverage_plan(cfg) if e['name']==f'G_full_n{nodes}')
                    effective=effective_config(cfg,exp,collection);destination=directory/'probes'/str(nodes);destination.mkdir(parents=True,exist_ok=True)
                    result=worker(dict(action='probe',cfg=effective,paths=probe_paths,output=str(destination)),destination/'status.json',
                                  dict(cfg,worker_ram_gib=cfg['probe_ram_gib']))
                    if result:eligible.append(nodes)
                    if result is None:break
                write_json(directory/'resource_eligibility.json',dict(eligible_nodes=eligible,attempted_nodes=cfg['graph_budgets'],
                    note='Budgets are excluded on pilot resource failure, not on held-out accuracy'))
                if len(eligible)<1 or cfg['graph_budgets'][0] not in eligible:
                    states[collection]='incomplete: reference probe failed or budget ended';continue
                first=[e for e in coverage_plan(cfg) if e['profile']['nodes'] in eligible]
                catalog += first;write_json(directory/'experiment_plan.json',catalog)
                states[collection]='screening graph coverage on all development folds'
                run_jobs(collection,directory,first,[cfg['seeds'][0]])
                if time.monotonic()>=deadline:states[collection]='incomplete: coverage screening';continue
                if not all(finished(directory,e['name'],cfg,[cfg['seeds'][0]]) for e in first):
                    states[collection]='incomplete: coverage screening failures';continue
                coverage=select_coverage(directory,first,cfg,collection)
                catalog += followup_plan(cfg,coverage['profile'])
            write_json(directory/'experiment_plan.json',catalog)
            remaining=[e for e in catalog if not finished(directory,e['name'],cfg,[cfg['seeds'][0]])]
            states[collection]='screening training-data grid, controls and tabular candidates'
            run_jobs(collection,directory,remaining,[cfg['seeds'][0]])
            if not all(finished(directory,e['name'],cfg,[cfg['seeds'][0]]) for e in catalog):
                states[collection]='incomplete: screening jobs (inspect logs)';continue
            plan=confirmation_plan(directory,catalog,cfg,collection,coverage)
            confirmed=[e for e in catalog if e['name'] in plan['names']]
            states[collection]='three-seed confirmation'
            run_jobs(collection,directory,confirmed,cfg['seeds'])
            if not all(finished(directory,e['name'],cfg) for e in confirmed):
                states[collection]='incomplete: confirmation';continue
            lock=lock_selection(directory,catalog,cfg,collection,plan['names'])
            print(f'LOCKED on development folds: {lock["components"]}',flush=True)
            states[collection]='auditing confirmed comparisons'
            run_jobs(collection,directory,confirmed,cfg['seeds'],'holdout')
            if all(finished(directory,e['name'],cfg,action='holdout') for e in confirmed):
                write_json(directory/'complete.json',dict(status='complete'));states[collection]='complete'
            else:states[collection]='incomplete: audit'
            summarize(directory,cfg,collection);cleanup();checkpoint(force=True)
        if cfg['generate_submissions'] and all((output/c.replace(':','_')/'complete.json').exists() for c in cfg['collections']):
            from study_submission import export_submissions
            export_submissions(root,output,engine_path,cfg,deadline,
                               lambda job,engine,limits,until,status:worker(job,status))
        elif cfg['generate_submissions']:
            write_json(output/'submission_status.json',dict(status='pending',reason='Required stages incomplete'))
    except KeyboardInterrupt:
        if active:states[active]='incomplete: interrupted'
        print('Interrupted: creating a recovery archive.',flush=True)
    except Exception as exc:
        if active:states[active]=f'incomplete: {type(exc).__name__}: {exc}'
        write_json(output/'controller_error.json',dict(type=type(exc).__name__,error=str(exc)))
        raise
    finally:
        for collection in cfg['collections']:
            directory=output/collection.replace(':','_')
            if directory.exists():
                try:summarize(directory,cfg,collection)
                except Exception as exc:write_json(directory/'summary_error.json',dict(error=str(exc)))
        cleanup();write_summary(output,cfg,states);checkpoint(force=True);release_memory()
    return output.parent/(output.name+'_results.zip')
