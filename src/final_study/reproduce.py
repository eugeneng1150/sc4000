"""Load compatible models or train exact selected recipes; no model archive required."""
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import shutil
import time
import zipfile
import numpy as np
import pandas as pd
from study_engine import digest, write_json, scores, GIB
from study_submission import preflight, locate_template, ranking, validate_ranking
from v4_core import read, sha, restore
from v5_scheduler import Scheduler, Pause, snapshot

VERSION='final-reproduction-1'


def unpack_spec(payload, expected_sha):
    if hashlib.sha256(payload).hexdigest()!=expected_sha:raise ValueError('Reproduction pack hash mismatch')
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        if sum(i.file_size for i in z.infolist())>150*1024**2:raise ValueError('Unexpected specification size')
        contract=json.loads(z.read('contract.json'))
    if contract['version']!=VERSION:raise ValueError('Incompatible specification')
    return contract


def expected_rows(contract,spec):
    return [contract['training_rows'][k] for k in contract['trainings'][spec['training']]]


def plan(contract,run_ablations=False,ablation_seeds=(42,)):
    if not set(ablation_seeds)<={42,137,2026}:raise ValueError('Use recorded ablation seeds 42, 137, 2026')
    return [s for s in contract['jobs'] if set(s['roles'])&{'selected','baseline'} or
            (run_ablations and s['seed'] in ablation_seeds)]


def defaults():
    return dict(cpu_threads=2, max_parallel=1, worker_ram_gib=5., system_reserve_gib=2.,
        scratch_limit_gib=4., output_limit_gib=12., disk_reserve_gib=4.,
        extraction_minutes=180,job_minutes=120,inference_minutes=120,
        session_hours=8.,checkpoint_minutes=30,monitor_seconds=.5,retry_failed=True)


def validate_versions(contract,run_ablations,strict=True):
    packages=['numpy','pandas','scipy','scikit-learn','torch','joblib']
    if run_ablations:packages.append('xgboost')
    actual={p:importlib.metadata.version(p) for p in packages}
    bad={p:(contract['versions'][p],v) for p,v in actual.items() if v!=contract['versions'][p]}
    if strict and bad:raise ValueError(f'Dependency mismatch (expected, actual): {bad}. Run the dependency cell and restart the session.')
    # Optional XGBoost is checked above, but enabling ablations must not invalidate
    # already fitted HGB/GNN models from the same fixed specification.
    return {p:v for p,v in actual.items() if p != "xgboost"}


def model_dir(output,spec):
    return Path(output)/'models'/spec['collection'].replace(':','_')/'folds'/str(spec['fold'])/'trials'/spec['experiment']['name']/str(spec['seed'])


def model_name(spec):return 'model.pt' if spec['experiment']['model']=='gnn' else 'model.joblib'


def receipt_valid(directory,signature):
    p=Path(directory)/'model_receipt.json'
    if not p.exists():return False
    receipt=read(p)
    return receipt.get('signature')==signature and bool(receipt.get('files')) and all(
        (Path(directory)/name).is_file() and sha(Path(directory)/name)==value for name,value in receipt['files'].items())


def record_model(directory,spec,signature,origin):
    directory=Path(directory)
    names=[model_name(spec),'training_manifest.json','result.json']
    if not all((directory/n).is_file() for n in names):raise ValueError('Incomplete fit cannot be reused')
    if (directory/'history.csv').exists():names.append('history.csv')
    write_json(directory/'model_receipt.json',dict(signature=signature,origin=origin,
        files={n:sha(directory/n) for n in names}))


def import_model(archive,directory,spec,signature):
    """Copy only hash-identified model artifacts; never extract arbitrary archive code."""
    if not archive:return False
    archive=Path(archive);directory=Path(directory)
    if not archive.exists():raise FileNotFoundError(archive)
    prefix=spec['archive_directory'];needed=model_name(spec)
    def consume(read_bytes,exists):
        if not exists(prefix+'/'+needed):return False
        for name,expected in spec['archive_files'].items():
            if not exists(prefix+'/'+name):return False
            data=read_bytes(prefix+'/'+name)
            if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Archive artifact differs: '+prefix+'/'+name)
        directory.mkdir(parents=True,exist_ok=True)
        for name in spec['archive_files']:
            temp=directory/(name+'.tmp');temp.write_bytes(read_bytes(prefix+'/'+name));temp.replace(directory/name)
        record_model(directory,spec,signature,'verified historical archive')
        return True
    if archive.is_dir():return consume(lambda p:(archive/p).read_bytes(),lambda p:(archive/p).is_file())
    with zipfile.ZipFile(archive) as z:
        names=set(z.namelist());return consume(z.read,lambda p:p in names)


def validate_data(root,contract,jobs):
    """Bind fixed split/configuration manifests to actual raw NPZ identities before fitting."""
    root=Path(root);checked=set()
    for key in dict.fromkeys(s['manifest'] for s in jobs):
        m=contract['manifests'][key]
        sets={split:{r['graph'] for r in m['records'] if r['split']==split} for split in ['train','dev','holdout']}
        if any(sets[a]&sets[b] for a,b in [('train','dev'),('train','holdout'),('dev','holdout')]):
            raise ValueError('Graph leakage in recorded split')
        for r in m['records']:
            rel=Path(r['relative_path'])
            if rel.is_absolute() or '..' in rel.parts:raise ValueError('Invalid data path')
            if (str(rel),r['source']) in checked:continue
            path=root/rel
            with zipfile.ZipFile(path) as z:identity=[(i.filename,i.CRC,i.file_size) for i in z.infolist()]
            if digest(identity)!=r['source']:raise ValueError(f'Raw data differs from recorded experiment: {path}')
            if not r['indices'] or min(r['indices'])<0 or max(r['indices'])>=r['count']:raise ValueError('Invalid sampled candidate IDs')
            checked.add((str(rel),r['source']))
    print(f'Validated {len(checked)} raw graphs against recorded split/configuration manifests.',flush=True)


def aggregate_audit(output,contract,jobs):
    """Rank each member before blending, with fold-safe development evaluation."""
    output=Path(output);frames={};fit_rows=[]
    for s in jobs:
        d=model_dir(output,s);result=read(d/'result.json')
        fit_rows.append(dict(collection=s['collection'],fold=s['fold'],seed=s['seed'],
            experiment=s['experiment']['name'],dev_score=result['dev_score'],roles=','.join(s['roles']),
            provenance=read(d/'model_receipt.json')['origin']))
        for split in ['dev','holdout']:
            p=d/'audit'/f'{split}_predictions.csv.gz'
            if not p.exists():raise ValueError('Audit is incomplete: '+str(p))
            f=pd.read_csv(p).sort_values(['graph','config_id']).reset_index(drop=True)
            f['prediction']=f.groupby('graph').prediction.rank(method='average')/f.groupby('graph').prediction.transform('count')
            frames.setdefault((s['collection'],s['fold'],s['experiment']['name'],split),[]).append(f)
    def average(items):
        out=items[0].copy()
        for f in items[1:]:
            if not out[['graph','config_id','runtime']].equals(f[['graph','config_id','runtime']]):raise ValueError('Prediction candidates or labels differ')
        out['prediction']=np.mean([f.prediction.to_numpy() for f in items],axis=0)
        return out
    means={k:average(v) for k,v in frames.items()};metrics=[]
    for c,components in contract['selected'].items():
        for fold in contract['inference_cfg']['folds']:
            for split in ['dev','holdout']:
                base=means[c,fold,'B0_basic_hgb',split]
                selected=base.copy();selected['prediction']=0.
                for name,weight in components.items():
                    f=means[c,fold,name,split]
                    if not base[['graph','config_id','runtime']].equals(f[['graph','config_id','runtime']]):raise ValueError('Blend candidates differ')
                    selected['prediction']+=weight*f.prediction
                for label,frame in [('baseline',base),('selected',selected)]:
                    for graph,g in frame.groupby('graph'):
                        metrics.append(dict(collection=c,fold=fold,split=split,system=label,graph=graph,
                            **scores(g.runtime.to_numpy(),g.prediction.to_numpy(),c.startswith('tile'))))
    out=output/'evaluation';out.mkdir(exist_ok=True)
    method_rows=[]
    for (c,fold,name,split),frame in means.items():
        for graph,g in frame.groupby('graph'):
            method_rows.append(dict(collection=c,fold=fold,experiment=name,split=split,graph=graph,
                seeds=len(frames[c,fold,name,split]),**scores(g.runtime.to_numpy(),g.prediction.to_numpy(),c.startswith('tile'))))
    pd.DataFrame(method_rows).to_csv(out/'method_graph_scores.csv',index=False)
    pd.DataFrame(fit_rows).to_csv(out/'fit_results.csv',index=False)
    f=pd.DataFrame(metrics);f.to_csv(out/'per_fold_graph_scores.csv',index=False)
    # Aggregate HOLDOUT predictions across folds before scoring, matching the historical ensemble.
    rows=[]
    for c,components in contract['selected'].items():
        methods={n:average([means[c,fold,n,'holdout'] for fold in contract['inference_cfg']['folds']]) for n in ['B0_basic_hgb',*components]}
        base=methods['B0_basic_hgb'];selected=base.copy()
        selected['prediction']=sum(w*methods[n].prediction for n,w in components.items())
        values={}
        for label,frame in [('baseline',base),('selected',selected)]:
            values[label]=np.mean([scores(g.runtime.to_numpy(),g.prediction.to_numpy(),c.startswith('tile'))['score'] for _,g in frame.groupby('graph')])
        delta=values['selected']-values['baseline']
        rows.append(dict(collection=c,metric='1 - top5 regret' if c.startswith('tile') else 'Kendall tau',
            baseline=values['baseline'],selected=values['selected'],absolute_gain=delta,
            relative_gain_pct=100*delta/values['baseline'] if values['baseline']>=.1 else None,
            graphs=base.graph.nunique(),provenance='current model inference on reused audit graphs'))
    pd.DataFrame(rows).to_csv(out/'current_audit.csv',index=False)
    return rows


def run(root,output,scratch,runtime,contract,resources=None,model_archive='',run_ablations=False,
        ablation_seeds=(42,),sample_submission='',on_checkpoint=None,strict_versions=True):
    root,output,scratch,runtime=[Path(p).resolve() for p in [root,output,scratch,runtime]]
    for a,b in [(root,output),(root,scratch),(output,scratch),(scratch,runtime)]:
        if a==b or a in b.parents or b in a.parents:raise ValueError('Data, output, runtime and scratch paths must be separate')
    output.mkdir(parents=True,exist_ok=True);scratch.mkdir(parents=True,exist_ok=True)
    marker=scratch/'.final_owned'
    if not marker.exists() and any(scratch.iterdir()):raise ValueError('Refusing to clean unowned scratch')
    marker.touch()
    cfg=dict(defaults(),**(resources or {}));jobs=plan(contract,run_ablations,ablation_seeds)
    versions=validate_versions(contract,run_ablations,strict_versions)
    identity=dict(contract=digest(contract),versions=versions,
        sources={p.name:sha(p) for p in sorted(runtime.glob('*.py')) if p.name not in {'story.py','final_report.py'}},cpu_threads=cfg['cpu_threads'])
    ip=output/'reproduction_identity.json'
    if ip.exists() and read(ip)!=identity:raise ValueError('Recipe/code/environment changed; use a new RUN_NAME')
    write_json(ip,identity)
    complete_cfg=dict(contract['inference_cfg'],collections=list(contract['selected']),sample_submission=sample_submission,cpu_threads=cfg['cpu_threads'])
    preflight(root,complete_cfg);validate_data(root,contract,jobs)
    state={'status':'running','required_fits':len(jobs),'completed_fits':0,'run_ablations':run_ablations}
    deadline=time.monotonic()+cfg['session_hours']*3600;last=time.monotonic()
    def checkpoint(force=False):
        nonlocal last
        if not force and time.monotonic()-last<cfg['checkpoint_minutes']*60:return
        write_json(output/'run_state.json',state)
        path=snapshot(output,cfg['disk_reserve_gib']);last=time.monotonic()
        print('CHECKPOINT READY:',path,'| bytes:',path.stat().st_size,flush=True)
        if on_checkpoint:on_checkpoint(path)
    scheduler=Scheduler(output,scratch,runtime/'final_worker.py',cfg,deadline,checkpoint)
    signatures={s['id']:digest(dict(identity=identity,spec=s)) for s in jobs}
    def run_job(job,status):
        scheduler.batch([dict(job=job,status=status,estimate=min(cfg['worker_ram_gib'],4.))],parallel=False)
    try:
        # Import/check before extraction: a fully fitted run should do no training preparation.
        for s in jobs:
            d=model_dir(output,s);sig=signatures[s['id']]
            if receipt_valid(d,sig):print('REUSE',s['id'],flush=True)
            elif import_model(model_archive,d,s,sig):print('IMPORTED',s['id'],flush=True)
            else:
                (d/'status.json').unlink(missing_ok=True)
                (d/'model_receipt.json').unlink(missing_ok=True)
            state['completed_fits']+=receipt_valid(d,sig)
        print(f'MODELS: {state["completed_fits"]}/{len(jobs)} reusable; {len(jobs)-state["completed_fits"]} fits needed.',flush=True)
        # Group compatible preparations; one bounded scratch group at a time.
        groups={}
        for s in jobs:
            settings=dict(s['cfg'],**cfg,max_excluded_fraction=0.)
            geometry={k:settings.get(k) for k in ['gnn_collections','gnn_nodes','gnn_config_nodes','gnn_edges','gnn_sampler','opcode_bins','depth_bins','edge_bins','edge_cap']}
            key=digest(dict(manifest=s['manifest'],geometry=geometry))[:20]
            groups.setdefault(key,[]).append((s,settings))
        for key,members in groups.items():
            needed=[]
            for s,settings in members:
                d=model_dir(output,s);sig=signatures[s['id']]
                ap=d/'audit'/'receipt.json'
                audit_ok=receipt_valid(d,sig) and ap.exists() and read(ap).get('model_files')==read(d/'model_receipt.json')['files'] and all(
                    (d/'audit'/n).is_file() and sha(d/'audit'/n)==h for n,h in read(ap).get('files',{}).items()) and bool(read(ap).get('files'))
                if not audit_ok:needed.append((s,settings))
            if not needed:continue
            work=scratch/'work';cached=work/'cache.json'
            if not cached.exists() or read(cached).get('key')!=key or not (work/'ready.json').exists():
                if work.exists():shutil.rmtree(work)
                work.mkdir();prep=output/'preparations'/key
                (prep/'status.json').unlink(missing_ok=True)
                s,settings=needed[0]
                run_job(dict(action='prepare',root=str(root),manifest=contract['manifests'][s['manifest']],cfg=settings,
                    scratch=str(work),label=f'PREPARE {s["collection"]} fold {s["fold"]}'),prep/'status.json')
                write_json(cached,dict(key=key))
            for s,settings in needed:
                d=model_dir(output,s);d.mkdir(parents=True,exist_ok=True);sig=signatures[s['id']]
                expected=d/'expected_training.json';write_json(expected,expected_rows(contract,s))
                job=dict(cfg=settings,experiment=s['experiment'],seed=s['seed'],collection=s['collection'],
                    scratch=str(work),expected_training=str(expected),model_directory=str(d))
                if not receipt_valid(d,sig):
                    run_job(dict(job,action='trial',label='TRAIN '+s['id']),d/'status.json')
                    record_model(d,s,sig,'retrained from raw data');state['completed_fits']+=1
                    print(f'FITS COMPLETE: {state["completed_fits"]}/{len(jobs)}',flush=True)
                audit=d/'audit';(audit/'status.json').unlink(missing_ok=True)
                run_job(dict(job,action='audit',label='EVALUATE '+s['id']),audit/'status.json')
                files={p.name:sha(p) for p in audit.iterdir() if p.name in ['dev_predictions.csv.gz','holdout_predictions.csv.gz','audit_result.json']}
                write_json(audit/'receipt.json',dict(model_files=read(d/'model_receipt.json')['files'],files=files))
            shutil.rmtree(work)
            checkpoint()
        aggregate_audit(output,contract,jobs)
        state['status']='predicting';checkpoint(True)
        template=pd.read_csv(locate_template(root,sample_submission),dtype=str);predictions=[]
        recipes={(s['collection'],s['experiment']['name']):s['experiment'] for s in jobs}
        for index,row in enumerate(template.itertuples(index=False)):
            c,stem=row.ID.rsplit(':',1);components=contract['selected'][c]
            path=root.joinpath(*c.split(':'),'test',stem+'.npz')
            with zipfile.ZipFile(path) as z:raw_identity=[(i.filename,i.CRC,i.file_size) for i in z.infolist()]
            sources={n:dict(directory=str(output/'models'/c.replace(':','_')),experiment=recipes[c,n]) for n in ['B0_basic_hgb',*components]}
            # Each inference result is bound to raw graph + template + exact model bytes.
            model_hashes={s['id']:read(model_dir(output,s)/'model_receipt.json')['files'] for s in jobs if s['collection']==c and set(s['roles'])&{'selected','baseline'}}
            signature=digest(dict(raw=raw_identity,template=row.TopConfigs,models=model_hashes,components=components))
            dest=output/'test_predictions'/f'{index:05d}';cache=dest/'receipt.json'
            good=cache.exists() and read(cache).get('signature')==signature and all((dest/n).exists() and sha(dest/n)==h for n,h in read(cache).get('files',{}).items()) and bool(read(cache).get('files'))
            if not good:
                (dest/'status.json').unlink(missing_ok=True)
                run_job(dict(action='inference',cfg=complete_cfg,sources=sources,components=components,
                    collection=c,path=str(path),id=row.ID,template_values=[int(v) for v in row.TopConfigs.split(';')],
                    label=f'PREDICT {index+1}/{len(template)} {row.ID}'),dest/'status.json')
                write_json(cache,dict(signature=signature,files={n:sha(dest/n) for n in ['prediction.json','ranks.npz']}))
            result=read(dest/'prediction.json')
            if result['ID']!=row.ID:raise ValueError('Prediction ID mismatch')
            with np.load(dest/'ranks.npz') as ranks:
                result['baseline']=ranking(ranks['B0_basic_hgb'],len(row.TopConfigs.split(';')),c.startswith('tile'),list(map(int,row.TopConfigs.split(';'))))
            for kind in ['baseline','selected']:
                validate_ranking(result[kind],result['count'],len(row.TopConfigs.split(';')),c.startswith('tile'),list(map(int,row.TopConfigs.split(';'))))
            predictions.append(result)
        for kind,name in [('baseline','submission_baseline.csv'),('selected','submission_final.csv')]:
            frame=pd.DataFrame([dict(ID=r['ID'],TopConfigs=';'.join(map(str,r[kind]))) for r in predictions])
            if not frame.ID.equals(template.ID):raise ValueError('Submission order differs')
            temp=output/(name+'.tmp');frame.to_csv(temp,index=False);temp.replace(output/name)
        actual=sha(output/'submission_final.csv')
        write_json(output/'submission_provenance.json',dict(rows=len(template),sha256=actual,
            historical_sha256=contract['historical_submission_sha256'],byte_identical_to_historical=actual==contract['historical_submission_sha256'],
            source='fresh full-candidate inference from loaded/retrained models',identity=identity,
            selected=contract['selected'],reported_historical_kaggle_score=.567,
            current_kaggle_score=None,note='A new CSV has no measured Kaggle score until submitted.'))
        state['status']='complete'
    except (KeyboardInterrupt,Pause) as exc:
        state.update(status='paused',reason=str(exc));print('PAUSED: completed models are saved. Resume with the same settings.',flush=True)
    except Exception as exc:
        state.update(status='failed',reason=str(exc));raise
    finally:
        # Scheduler has stopped/reaped workers before propagating errors.
        write_json(output/'run_state.json',state);checkpoint(True)
    return state
