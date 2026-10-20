"""Resumable, bounded continuation of v3, with auditable fallback decisions."""
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
from study_engine import write_json,digest,GIB,scores
from study_common import execute,folder_bytes,paired_summary
from study_controller import make_bundle
from study_plan import effective_config
from study_submission import validate_ranking,preflight
from v4_core import (read,sha,archive_identity,check_data_identity,plans,expanded_manifest,resources,choose,predictions,blend)


class Pause(Exception): pass


def complete(directory,exp,cfg,seeds):
    for f in cfg['folds']:
        for s in seeds:
            p=Path(directory)/'folds'/str(f)/'trials'/exp/str(s)
            if not (p/'result.json').exists() or not (p/'status.json').exists() or read(p/'status.json')['status']!='ok':return False
    return True


def run(root,output,scratch,runtime,cfg,on_checkpoint=None,enforce_versions=True):
    root,output,scratch,runtime=map(lambda x:Path(x).resolve(),[root,output,scratch,runtime])
    for a,b in [(root,output),(root,scratch),(output,scratch)]:
        if a==b or a in b.parents or b in a.parents:raise ValueError('Data, output and scratch must be separate')
    ref=output/'reference_v3'
    archive_identity(output,cfg,runtime,enforce_versions)
    preflight(root,cfg)
    check_data_identity(root,output,cfg)
    scratch.mkdir(parents=True,exist_ok=True);marker=scratch/'.v4_owned'
    if not marker.exists() and any(scratch.iterdir()):raise ValueError('Refusing unmarked scratch directory')
    marker.touch()
    deadline=time.monotonic()+cfg['session_hours']*3600;last_checkpoint=time.monotonic()
    state=read(output/'run_state.json') if (output/'run_state.json').exists() else {'collections':{},'config':cfg}
    state['config']=cfg;state['status']='running';active=None
    max_recipes=sum(len(plans(ref,c,cfg))+(c not in cfg['gnn_collections']) for c in cfg['collections'])
    max_fits=max_recipes*len(cfg['folds'])+len(cfg['collections'])*len(cfg['folds'])*(len(cfg['seeds'])-1)
    def progress():
        completed=sum(read(p).get('status')=='ok' for p in output.glob('*/folds/*/trials/*/*/status.json'))
        value=dict(completed_new_fits=completed,maximum_new_fits_before_exclusions=max_fits,
                   note='Upper bound shrinks with resource exclusions and unsuccessful screening; excludes pilots/audit/inference',
                   collections=state['collections'])
        write_json(output/'progress.json',value)
        return completed
    print(f'RESUME CHECK: {progress()} new fits complete; at most {max_fits} planned before resource/screening exclusions. V3 fits are reused.',flush=True)

    def checkpoint(force=False):
        nonlocal last_checkpoint
        if not force and time.monotonic()-last_checkpoint<cfg['checkpoint_minutes']*60:return
        write_json(output/'run_state.json',state)
        progress()
        # Timestamp belongs to this ZIP, not the previous one.
        write_json(output/'last_checkpoint.json',dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                    note='ZIP creation/request is not proof of browser receipt'))
        archive=make_bundle(output,cfg['disk_reserve_gib'])
        print(f'ZIP READY: {archive} ({archive.stat().st_size:,} bytes)',flush=True)
        if on_checkpoint:on_checkpoint(archive)
        last_checkpoint=time.monotonic()

    def worker(job,status):
        if status.exists() and read(status)['status']=='ok':
            print('SKIP completed:',status.relative_to(output),flush=True);return True
        if time.monotonic()>=deadline:raise Pause('Session budget; resume this v4 checkpoint')
        if folder_bytes(output)>cfg['output_limit_gib']*GIB:raise OSError('Output budget exceeded')
        ok=execute(job,runtime/'v4_worker.py',cfg,deadline,status,heartbeat=checkpoint)
        checkpoint()
        if ok is None:raise Pause('Session budget; completed jobs retained')
        if not ok:
            why=read(status).get('status')
            failure=Path(job['output'])/'worker_failure.json'
            memory_error=failure.exists() and read(failure).get('kind')=='resource'
            if job['action']=='pilot' and (why in ['memory_guard','system_memory_guard','disk_guard'] or memory_error):return False
            raise RuntimeError(f'{why}: inspect {status.with_suffix(".log")}; checkpoint preserved. No automatic scientific exclusion.')
        return True

    def prepare(c,exp,f):
        m=expanded_manifest(root,ref,c,f,exp,cfg)
        effective=effective_config(cfg,exp,c)
        key=digest(dict(manifest=m,profile=exp.get('profile') if exp['model']=='gnn' else None))[:16]
        work=scratch/'work';cache=work/'cache.json'
        if cache.exists() and read(cache).get('key')==key and (work/'ready.json').exists():return effective,work,m
        if work.exists():shutil.rmtree(work)
        work.mkdir()
        prep=output/c.replace(':','_')/'preparations'/key;prep.mkdir(parents=True,exist_ok=True)
        write_json(prep/'manifest.json',m)
        # Scratch is ephemeral: successful preparation cannot be skipped after its cache disappeared.
        (prep/'status.json').unlink(missing_ok=True)
        print(f'PREPARE {c}, fold {f}: {sum(r["split"]=="train" for r in m["records"])} training graphs',flush=True)
        worker(dict(action='prepare',root=str(root),manifest=m,cfg=effective,scratch=str(work),output=str(prep)),prep/'status.json')
        write_json(cache,dict(key=key))
        return effective,work,m

    def jobs(c,exp,seeds,action='trial'):
        d=output/c.replace(':','_')
        total=len(cfg['folds'])*len(seeds)
        for i,(f,s) in enumerate(( (f,s) for f in cfg['folds'] for s in seeds),1):
            p=d/'folds'/str(f)/'trials'/exp['name']/str(s);p.mkdir(parents=True,exist_ok=True)
            status=p/('status.json' if action=='trial' else 'holdout_status.json')
            if status.exists() and read(status)['status']=='ok':
                print(f'SKIP {c} {exp["name"]} fold {f} seed {s} {action} [{i}/{total}]',flush=True);continue
            effective,work,m=prepare(c,exp,f)
            if exp['model']=='gnn' and (p/'history.csv').exists():
                history=pd.read_csv(p/'history.csv');print(f'RESUME GNN after epoch {history.epoch.max()}',flush=True)
            print(f'RUN {c} | {exp["name"]} | fold {f} seed {s} | {action} [{i}/{total}]',flush=True)
            worker(dict(action=action,cfg=effective,experiment=exp,seed=s,collection=c,scratch=str(work),output=str(p)),status)
            if action=='trial':print(f'PROGRESS: {progress()} new fits complete (upper bound {max_fits}; audit/inference separate)',flush=True)
        return True

    def admission(c,exp):
        d=output/c.replace(':','_');p=d/'admission'/exp['name'];p.mkdir(parents=True,exist_ok=True)
        verdict=p/'decision.json'
        if verdict.exists():return read(verdict)['admitted']
        checks=[resources(ref,c,exp,expanded_manifest(root,ref,c,f,exp,cfg),cfg) for f in cfg['folds']]
        if not all(x['admitted'] for x in checks):
            write_json(verdict,dict(admitted=False,checks=checks,reason='Static resource bound'));print('RESOURCE EXCLUSION',c,exp['name'],flush=True);return False
        if exp['model']!='gnn':
            effective,work,_=prepare(c,exp,cfg['folds'][0])
            ok=worker(dict(action='pilot',cfg=effective,experiment=exp,seed=cfg['seeds'][0],collection=c,
                           scratch=str(work),output=str(p)),p/'status.json')
            status=read(p/'status.json')
            if not ok or status.get('peak_rss_gib',0)>cfg['worker_ram_gib']*cfg['pilot_memory_fraction']:
                write_json(verdict,dict(admitted=False,checks=checks,reason='Measured pilot memory guard',pilot=status));return False
            estimate=read(p/'pilot.json')
            print(f'PILOT {exp["name"]}: {status["peak_rss_gib"]:.2f} GiB; rough fit {estimate["rough_full_fit_minutes"]:.1f} min/job (excludes preparation/evaluation)',flush=True)
        write_json(verdict,dict(admitted=True,checks=checks));return True

    try:
        # Validate original sources before any successful-job shortcuts.
        for c in cfg['collections']:
            rd=ref/c.replace(':','_');oldexp=read(rd/'experiment_plan.json')[0]
            for f in cfg['folds']:
                # graph_limit=original cap retains original source records only.
                n=sum(r['split']=='train' for r in read(rd/'manifests'/f'fold_{f}.json')['records'])
                expanded_manifest(root,ref,c,f,dict(oldexp,graph_limit=n),cfg)
        for c in cfg['collections']:
            active=c;d=output/c.replace(':','_');d.mkdir(exist_ok=True)
            if (d/'complete.json').exists():
                state['collections'][c]='complete (resumed)';print('SKIP completed collection:',c,flush=True);continue
            catalog=plans(ref,c,cfg);admitted=[];seen=set()
            state['collections'][c]='screening'
            for exp in catalog:
                m=expanded_manifest(root,ref,c,cfg['folds'][0],exp,cfg)
                sample=digest(m)
                if sample in seen:continue
                seen.add(sample)
                if admission(c,exp):
                    admitted.append(exp);jobs(c,exp,[cfg['seeds'][0]])
            if c not in cfg['gnn_collections'] and admitted:
                for largest in sorted(admitted,key=lambda e:len(expanded_manifest(root,ref,c,cfg['folds'][0],e,cfg)['records']),reverse=True):
                    more=dict(largest,name=largest['name'].replace('c256','c384'),configs=384,stage='training configurations')
                    catalog.append(more)
                    if admission(c,more):
                        admitted.append(more);jobs(c,more,[cfg['seeds'][0]]);break
            write_json(d/'experiment_plan.json',catalog)
            screened=choose(ref,output,c,[e['name'] for e in admitted],cfg,[cfg['seeds'][0]])
            write_json(d/'screening_selection.json',screened)
            # One new finalist: highest eligible blend containing a new recipe.
            eligible=[r for r in screened['candidates'] if r['eligible'] and set(r['components']) & {e['name'] for e in admitted}]
            finalists=[]
            if eligible:
                leader=max(eligible,key=lambda r:r['mean_score'])
                finalists=[e for e in admitted if e['name'] in leader['components']]
            state['collections'][c]='confirmation'
            for e in finalists:jobs(c,e,cfg['seeds'])
            lock=choose(ref,output,c,[e['name'] for e in finalists],cfg,cfg['seeds'])
            if (d/'selection_lock.json').exists() and read(d/'selection_lock.json')!=lock:raise ValueError('Frozen selection changed')
            write_json(d/'selection_lock.json',lock)
            print(f'LOCKED {c}: {lock["components"]}; mean dev gain vs v3 {lock["gain_vs_v3"]:+.5f}',flush=True)
            state['collections'][c]='audit'
            for e in finalists:jobs(c,e,cfg['seeds'],'holdout')
            audit(ref,output,c,finalists,cfg)
            write_json(d/'complete.json',dict(status='complete',finalists=[e['name'] for e in finalists]))
            state['collections'][c]='complete';checkpoint(True)
        state['status']='test inference'
        export(root,ref,output,cfg,worker)
        state['status']='complete'
    except (Pause,KeyboardInterrupt) as exc:
        state['status']='paused';state['reason']=str(exc) or 'Interrupted'
        print('PAUSED: resume the same v4 output or its downloaded ZIP.',flush=True)
    except Exception as exc:
        state['status']='error';state['reason']=str(exc)
        write_json(output/'controller_error.json',dict(error=str(exc),collection=active));raise
    finally:
        work=scratch/'work'
        if marker.exists() and work.exists():shutil.rmtree(work)
        write_json(output/'run_state.json',state)
        from v4_report import generate
        try:generate(ref,output,root)
        except Exception as exc:
            write_json(output/'report_error.json',dict(error=str(exc)))
            print('REPORT GENERATION FAILED:',str(exc),'— saved training results remain intact.',flush=True)
        checkpoint(True)
    return output.parent/(output.name+'_results.zip')


def audit(ref,output,c,finalists,cfg):
    rd=Path(ref)/c.replace(':','_');nd=Path(output)/c.replace(':','_');lock=read(nd/'selection_lock.json')
    names=list(dict.fromkeys(list(lock['incumbent'])+[n for n in lock['components'] if not n.startswith('V4_')]))
    old=predictions(rd,names,cfg,cfg['seeds'],'holdout')
    new=predictions(nd,[e['name'] for e in finalists],cfg,cfg['seeds'],'holdout') if finalists else {f:{} for f in cfg['folds']}
    all_frames={f:dict(old[f],**new[f]) for f in cfg['folds']}
    systems={'v3_selected':lock['incumbent'],'v4_selected':lock['components']}
    systems.update({e['name']:{e['name']:1.} for e in finalists})
    rows=[]
    for name,components in systems.items():
        frames=[blend(all_frames[f],components) for f in cfg['folds']]
        for frame in frames[1:]:
            if not frames[0][['graph','config_id','runtime']].equals(frame[['graph','config_id','runtime']]):raise ValueError('Audit fold alignment failed')
        frame=frames[0].copy();frame['prediction']=np.mean([f.prediction.to_numpy() for f in frames],axis=0)
        for graph,g in frame.groupby('graph'):
            rows.append(dict(experiment=name,graph=graph,seed=0,**scores(g.runtime.to_numpy(),g.prediction.to_numpy(),c.startswith('tile'))))
    table=pd.DataFrame(rows);table.to_csv(nd/'system_audit_graphs.csv',index=False)
    paired_summary(table,[0],cfg['bootstrap_repeats'],'v3_selected').to_csv(nd/'system_audit_gains.csv',index=False)


def export(root,ref,output,cfg,worker):
    ref,output=Path(ref),Path(output)
    template=pd.read_csv(cfg['sample_submission'],dtype=str)
    baseline=pd.read_csv(ref/'submission_selected.csv',dtype=str)
    if not template.ID.equals(baseline.ID):raise ValueError('V3/template IDs differ')
    manifest=read(ref/'test_manifest.json');tests=manifest['tests']
    if template.ID.tolist()!=[r['id'] for r in tests]:raise ValueError('V3 test manifest IDs differ')
    rows=[];tabrows=[];provenance=[]
    locks={c:read(output/c.replace(':','_')/'selection_lock.json') for c in cfg['collections']}
    for i,r in enumerate(tests):
        c=r['collection'];lock=locks[c];original=baseline.iloc[i].TopConfigs
        # Validate fingerprints even when copying old rows.
        import zipfile
        with zipfile.ZipFile(Path(root)/r['relative_path']) as z:fp=[(x.filename,x.CRC,x.file_size) for x in z.infolist()]
        if json.loads(json.dumps(fp))!=r['fingerprint']:raise ValueError('Test source differs from v3')
        if not lock['changed']:
            chosen=original;origin='v3 exact row'
        else:
            sources={}
            for name in lock['components']:
                directory=(output if name.startswith('V4_') else ref)/c.replace(':','_')
                exp=next(e for e in read(directory/'experiment_plan.json') if e['name']==name)
                sources[name]=dict(directory=str(directory),experiment=exp)
            p=output/'test_predictions'/f'{i:05d}';p.mkdir(parents=True,exist_ok=True)
            print(f'TEST {i+1}/{len(tests)}: {r["id"]}',flush=True)
            worker(dict(action='inference',cfg=cfg,sources=sources,components=lock['components'],path=str(Path(root)/r['relative_path']),
                        output=str(p),**r),p/'status.json')
            pred=read(p/'prediction.json')
            if pred['ID']!=r['id']:raise ValueError('Inference ID mismatch')
            validate_ranking(pred['selected'],pred['count'],len(r['template_values']),c.startswith('tile'),r['template_values'])
            chosen=';'.join(map(str,pred['selected']));origin='v4 locked selection'
        rows.append(dict(ID=r['id'],TopConfigs=chosen))
        tabrows.append(dict(ID=r['id'],TopConfigs=original if c in cfg['gnn_collections'] else chosen))
        provenance.append(dict(ID=r['id'],origin=origin,components=json.dumps(lock['components'],sort_keys=True)))
    for name,frame in [('v3_reference',baseline),('tabular_update_only',pd.DataFrame(tabrows)),('selected',pd.DataFrame(rows))]:
        if not frame.ID.equals(template.ID) or frame.isna().any().any():raise ValueError('Invalid final submission')
        path=output/f'submission_{name}.csv';temp=path.with_suffix('.csv.tmp');frame.to_csv(temp,index=False);temp.replace(path)
    pd.DataFrame(provenance).to_csv(output/'submission_provenance.csv',index=False)
    write_json(output/'submission_status.json',dict(status='complete',rows=len(rows),
        changed_collections=[c for c,l in locks.items() if l['changed']],note='Candidate, not submitted to Kaggle'))
