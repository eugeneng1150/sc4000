"""Run the fixed target/learner/objective study, preserving the complete v4 baseline."""
from pathlib import Path
import json
import shutil
import time
import zipfile
import numpy as np
import pandas as pd
from study_engine import write_json, digest, GIB, scores
from study_common import paired_summary
from study_plan import effective_config
from study_submission import preflight, validate_ranking
from v4_core import read, check_data_identity, blend
import v5_core as core
from v5_scheduler import Scheduler, Pause, ResourceFailure, snapshot


def run(root, output, scratch, runtime, cfg, on_checkpoint=None, enforce_versions=True):
    root,output,scratch,runtime=map(lambda p:Path(p).resolve(),[root,output,scratch,runtime])
    for a,b in [(root,output),(root,scratch),(output,scratch)]:
        if a==b or a in b.parents or b in a.parents:
            raise ValueError('Data, results and scratch must be separate')
    ref=output/'reference_v4'
    core.archive_identity(output,runtime,cfg,enforce_versions)
    complete_cfg=dict(cfg,collections=cfg['submission_collections'])
    preflight(root,complete_cfg)
    check_data_identity(root,output,complete_cfg)
    scratch.mkdir(parents=True,exist_ok=True)
    marker=scratch/'.v5_owned'
    if not marker.exists() and any(scratch.iterdir()):
        raise ValueError('Refusing to clean unmarked scratch')
    marker.touch()
    state=read(output/'run_state.json') if (output/'run_state.json').exists() else {'collections':{}}
    state.update(status='running',config=cfg)
    deadline=time.monotonic()+cfg['session_hours']*3600
    last_checkpoint=time.monotonic()
    max_fits=len(cfg['collections'])*len(cfg['folds'])*(3+2*(len(cfg['seeds'])-1))

    def progress():
        count=sum(read(p).get('status')=='ok' for p in output.glob('*/folds/*/trials/*/*/status.json'))
        value=dict(completed_new_fits=count,maximum_new_fits_before_exclusions=max_fits,
            collections=state['collections'],note='Includes matched XGBoost confirmation control; pilots/audit/inference separate')
        write_json(output/'progress.json',value)
        return count

    def checkpoint(force=False):
        nonlocal last_checkpoint
        if not force and time.monotonic()-last_checkpoint<cfg['checkpoint_minutes']*60:
            return
        write_json(output/'run_state.json',state)
        progress()
        write_json(output/'last_checkpoint.json',dict(
            created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            note='Only committed work included; download request is not confirmed delivery'))
        archive=snapshot(output,cfg['disk_reserve_gib'])
        print(f'ZIP READY: {archive} ({archive.stat().st_size:,} bytes)',flush=True)
        if on_checkpoint:
            on_checkpoint(archive)
        last_checkpoint=time.monotonic()

    scheduler=Scheduler(output,scratch,runtime/'v5_worker.py',cfg,deadline,checkpoint)
    print(f'RESUME: {progress()} new fits complete; upper bound {max_fits}. '
          f'CPU capacity {scheduler.cpus}, max workers {min(2,cfg["max_parallel"])}. V4/GNN fits reused.',flush=True)

    def prepare(c,fold,exp):
        m=core.manifest(root,ref,c,fold,cfg)
        settings=effective_config(cfg,exp,c)
        settings['gnn_collections']=[]
        key=digest(m)[:16]
        work=scratch/'work'
        if (work/'cache.json').exists() and read(work/'cache.json')['key']==key and (work/'ready.json').exists():
            return settings,work,m
        if work.exists():
            shutil.rmtree(work)
        work.mkdir()
        prep=output/c.replace(':','_')/'preparations'/key
        prep.mkdir(parents=True,exist_ok=True)
        write_json(prep/'manifest.json',m)
        (prep/'status.json').unlink(missing_ok=True)
        job=dict(action='prepare',root=str(root),manifest=m,cfg=settings,scratch=str(work),
                 label=f'PREPARE {c} fold {fold}')
        scheduler.batch([dict(job=job,status=prep/'status.json',
                             estimate=min(2.,cfg['worker_ram_gib']))],parallel=False)
        write_json(work/'cache.json',dict(key=key))
        return settings,work,m

    def admission(c,fold,exp,settings,work,m):
        p=output/c.replace(':','_')/'admission'/str(fold)/exp['name']
        p.mkdir(parents=True,exist_ok=True)
        decision=p/'decision.json'
        if decision.exists():
            result=read(decision)
            print(f'REUSE PILOT: {c} fold {fold} {exp["name"]}; '
                  f'admitted={result["admitted"]}, '
                  f'fit ~{result.get("fit_minutes",0):.1f} min, '
                  f'reserved {result.get("estimate_gib",0):.2f} GiB',flush=True)
            return result
        _,source=core.tabular_reference(ref,c)
        feature_count=read(Path(source['directory'])/'folds'/str(fold)/'trials'/
                           source['experiment']['name']/str(cfg['seeds'][0])/'result.json')['feature_count']
        # Upper bound includes all prepared training records; exact row match checked in worker.
        rows=sum(min(len(r['indices']),exp['configs']) for r in m['records'] if r['split']=='train')
        matrix=rows*feature_count*4/GIB
        static=max(.5, matrix*4+.5)
        if matrix*1024>cfg['max_matrix_mib'] or static>cfg['worker_ram_gib']:
            result=dict(admitted=False,reason='Static matrix/memory bound',matrix_gib=matrix)
            write_json(decision,result)
            return result
        pilots=[]
        try:
            for iterations in cfg['pilot_iterations']:
                dest=p/str(iterations)
                job=dict(action='pilot',cfg=settings,experiment=exp,seed=cfg['seeds'][0],
                    expected_training=str(core.expected_training(ref,c,fold,cfg['seeds'][0])),
                    collection=c,scratch=str(work),pilot_iterations=iterations,
                    label=f'PILOT {c} {exp["name"]} fold {fold}, {iterations} trees')
                scheduler.batch([dict(job=job,status=dest/'status.json',estimate=static)],parallel=False)
                pilots.append(dict(read(dest/'pilot.json'),**read(dest/'status.json')))
        except ResourceFailure as exc:
            result=dict(admitted=False,reason='Measured pilot memory guard',detail=str(exc))
            write_json(decision,result)
            return result
        # Two measured checkpoints expose some histogram/tree-memory growth.
        low,high=pilots[0],pilots[-1]
        remaining=max(0,exp['iterations']-high['iterations'])
        per_tree=max(0.,(high['peak_rss_gib']-low['peak_rss_gib'])/
                     max(1,high['iterations']-low['iterations']))
        model_growth=max(p['model_bytes']/p['iterations']/GIB for p in pilots)
        forecast=max(p['peak_rss_gib'] for p in pilots)+remaining*max(per_tree,model_growth)
        reservation=forecast*cfg['pilot_memory_multiplier']+cfg['pilot_memory_add_gib']
        seconds=max(p['fit_seconds']/p['iterations'] for p in pilots)*exp['iterations']
        result=dict(admitted=reservation<=cfg['worker_ram_gib'],estimate_gib=reservation,
            measured_peak_gib=max(p['peak_rss_gib'] for p in pilots),fit_minutes=seconds/60,
            reason='Measured full-row pilots plus growth/headroom; not a hard peak guarantee',
            rows=high['rows'],features=high['features'],pilots=pilots)
        write_json(decision,result)
        print(f'ESTIMATE {c} {exp["name"]} fold {fold}: fit ~{seconds/60:.1f} min '
              f'(rough range {seconds/120:.1f}–{seconds/30:.1f}); '
              f'reserved {reservation:.2f} GiB; admitted={result["admitted"]}',flush=True)
        return result

    def fits(c,exps,seeds,action='trial'):
        admitted={e['name']:True for e in exps}
        for fold in cfg['folds']:
            needed=[]
            for exp in exps:
                for seed in seeds:
                    p=output/c.replace(':','_')/'folds'/str(fold)/'trials'/exp['name']/str(seed)
                    status=p/('status.json' if action=='trial' else 'holdout_status.json')
                    if status.exists() and read(status).get('status')=='ok':
                        print(f'SKIP {c} {exp["name"]} fold {fold} seed {seed} {action}',flush=True)
                    else:
                        needed.append((exp,seed,p,status))
            if not needed:
                continue
            settings,work,m=prepare(c,fold,exps[0])
            decisions={}
            for exp,_,_,_ in needed:
                if exp['name'] not in decisions:
                    decisions[exp['name']]=admission(c,fold,exp,settings,work,m)
                    admitted[exp['name']] &= decisions[exp['name']]['admitted']
            tasks=[]
            for exp,seed,p,status in needed:
                estimate=decisions[exp['name']]
                if not estimate['admitted']:
                    print('RESOURCE EXCLUSION:',c,fold,exp['name'],estimate['reason'],flush=True)
                    continue
                job=dict(action=action,cfg=effective_config(settings,exp,c),experiment=exp,seed=seed,
                    collection=c,scratch=str(work),model_directory=str(p),
                    expected_training=str(core.expected_training(ref,c,fold,seed)),
                    label=f'{c} | {exp["name"]} | fold {fold} seed {seed} | {action}')
                tasks.append(dict(job=job,status=status,estimate=estimate['estimate_gib'],
                                  fit_minutes=estimate['fit_minutes'] if action=='trial' else 0.))
            scheduler.batch(tasks,parallel=True)
            print(f'PROGRESS: {progress()} new fits complete; upper bound {max_fits}.',flush=True)
        return [e for e in exps if admitted[e['name']] and all(
            (output/c.replace(':','_')/'folds'/str(f)/'trials'/e['name']/str(s)/'status.json').exists()
            and read(output/c.replace(':','_')/'folds'/str(f)/'trials'/e['name']/str(s)/'status.json')['status']=='ok'
            for f in cfg['folds'] for s in seeds)]

    try:
        for c in cfg['collections']:
            d=output/c.replace(':','_')
            d.mkdir(exist_ok=True)
            if (d/'complete.json').exists():
                state['collections'][c]='complete'
                print('SKIP complete collection:',c,flush=True)
                continue
            catalog=core.plans(ref,c,cfg)
            write_json(d/'experiment_plan.json',catalog)
            state['collections'][c]='screening'
            admitted=fits(c,catalog,[cfg['seeds'][0]])
            screen=core.choose(ref,output,c,[e['name'] for e in admitted],cfg,[cfg['seeds'][0]])
            write_json(d/'screening_selection.json',screen)
            names=core.finalists(screen,[e['name'] for e in admitted])
            selected=[e for e in admitted if e['name'] in names]
            state['collections'][c]='confirmation'
            confirmed=fits(c,selected,cfg['seeds']) if selected else []
            lock=core.choose(ref,output,c,[e['name'] for e in confirmed],cfg,cfg['seeds'])
            if (d/'selection_lock.json').exists() and read(d/'selection_lock.json')!=lock:
                raise ValueError('Frozen selection differs')
            write_json(d/'selection_lock.json',lock)
            print(f'LOCKED {c}: {lock["components"]}; mean dev gain vs v4 {lock["gain_vs_v4"]:+.5f}',flush=True)
            state['collections'][c]='audit'
            if confirmed:
                fits(c,confirmed,cfg['seeds'],'holdout')
            audit(ref,output,c,confirmed,cfg)
            write_json(d/'complete.json',dict(status='complete',finalists=[e['name'] for e in confirmed]))
            state['collections'][c]='complete'
            checkpoint(True)
        print('ALL MODEL TRAINING COMPLETE. Remaining: test prediction, CSV export, report, final ZIP.',flush=True)
        state['status']='test inference'
        export(root,ref,output,cfg,scheduler)
        state['status']='complete'
    except (Pause,KeyboardInterrupt) as exc:
        state.update(status='paused',reason=str(exc) or 'Interrupted')
        print('PAUSED. Completed fits are saved; unfinished fits restart on resume.',flush=True)
    except Exception as exc:
        state.update(status='error',reason=str(exc))
        write_json(output/'controller_error.json',dict(error=str(exc)))
        raise
    finally:
        # Scheduler has stopped/reaped all active workers before control reaches here.
        for path in [scratch/'work',output/'.inflight']:
            if path.exists():
                shutil.rmtree(path)
        write_json(output/'run_state.json',state)
        from v5_report import generate
        try:
            generate(output)
        except Exception as exc:
            write_json(output/'report_error.json',dict(error=str(exc)))
            print('Report failed; saved models are intact:',exc,flush=True)
        checkpoint(True)
    return output.parent/(output.name+'_results.zip')


def audit(ref,output,c,experiments,cfg):
    d=Path(output)/c.replace(':','_')
    lock=read(d/'selection_lock.json')
    names=list(dict.fromkeys([*lock['incumbent'],*lock['components'],*[e['name'] for e in experiments]]))
    frames=core.prediction_frames(ref,output,c,names,cfg,cfg['seeds'],'holdout')
    systems={'v4_selected':lock['incumbent'],'v5_selected':lock['components']}
    systems.update({e['name']:{e['name']:1.} for e in experiments})
    rows=[]
    for name,components in systems.items():
        members=[blend(frames[f],components) for f in cfg['folds']]
        keys=['graph','config_id','runtime']
        if any(not m[keys].equals(members[0][keys]) for m in members[1:]):
            raise ValueError('Audit alignment failed')
        frame=members[0].copy()
        frame['prediction']=np.mean([m.prediction.to_numpy() for m in members],axis=0)
        for graph,g in frame.groupby('graph'):
            rows.append(dict(experiment=name,graph=graph,seed=0,
                             **scores(g.runtime.to_numpy(),g.prediction.to_numpy())))
    table=pd.DataFrame(rows)
    table.to_csv(d/'system_audit_graphs.csv',index=False)
    paired_summary(table,[0],cfg['bootstrap_repeats'],'v4_selected').to_csv(d/'system_audit_gains.csv',index=False)


def export(root,ref,output,cfg,scheduler):
    template=pd.read_csv(cfg['sample_submission'],dtype=str)
    original=pd.read_csv(Path(ref)/'submission_selected.csv',dtype=str)
    tests=read(Path(ref)/'reference_v3/test_manifest.json')['tests']
    if not original.ID.equals(template.ID) or original.ID.tolist()!=[r['id'] for r in tests]:
        raise ValueError('Submission template/reference mismatch')
    locks={c:read(Path(output)/c.replace(':','_')/'selection_lock.json') for c in cfg['collections']}
    rows=[];provenance=[]
    for i,r in enumerate(tests):
        c=r['collection'];lock=locks.get(c)
        with zipfile.ZipFile(Path(root)/r['relative_path']) as z:
            fp=[(x.filename,x.CRC,x.file_size) for x in z.infolist()]
        if json.loads(json.dumps(fp))!=r['fingerprint']:
            raise ValueError('Test archive changed')
        chosen=original.iloc[i].TopConfigs
        if lock and lock['changed']:
            p=Path(output)/'test_predictions'/f'{i:05d}'
            catalog=core.sources(ref,output,c)
            job=dict(action='inference',cfg=cfg,path=str(Path(root)/r['relative_path']),
                     sources={n:catalog[n] for n in lock['components']},
                     components=lock['components'],label=f'TEST {i+1}/{len(tests)} {r["id"]}',**r)
            # Inference remains sequential and bounded; its GNN forward pass is not retraining.
            scheduler.batch([dict(job=job,status=p/'status.json',estimate=cfg['worker_ram_gib'])],parallel=False)
            pred=read(p/'prediction.json')
            if pred['ID']!=r['id']:
                raise ValueError('Prediction ID mismatch')
            validate_ranking(pred['selected'],pred['count'],len(r['template_values']),False,r['template_values'])
            chosen=';'.join(map(str,pred['selected']))
        rows.append(dict(ID=r['id'],TopConfigs=chosen))
        provenance.append(dict(ID=r['id'],origin='v5 selection' if lock and lock['changed'] else 'v4 exact row',
                               components=json.dumps(lock['components']) if lock else 'v4 tile unchanged'))
    for name,frame in [('v4_reference',original),('selected',pd.DataFrame(rows))]:
        if not frame.ID.equals(template.ID) or frame.isna().any().any():
            raise ValueError('Invalid export')
        p=Path(output)/f'submission_{name}.csv'
        temp=p.with_suffix('.csv.tmp');frame.to_csv(temp,index=False);temp.replace(p)
    pd.DataFrame(provenance).to_csv(Path(output)/'submission_provenance.csv',index=False)
    write_json(Path(output)/'submission_status.json',dict(status='complete',rows=len(rows),
        changed_collections=[c for c,l in locks.items() if l['changed']],
        note='Local candidate; no automatic Kaggle submission'))
