"""Freeze completed study evidence and the exact submitted ranking; never fit models."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
EXPECTED_SUBMISSION = 'dc037f378ddb9d0892c283a26a7f31170ac6cc01e937e4d14f2847296ff80312'
COLLECTIONS = ['layout:nlp:default','layout:nlp:random','layout:xla:default','layout:xla:random','tile:xla']


def read(p): return json.loads(p.read_text())
def sha(data): return hashlib.sha256(data).hexdigest()
def json_bytes(x): return (json.dumps(x,indent=2)+'\n').encode()
def csv_bytes(rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=keys);w.writeheader();w.writerows(rows)
    return s.getvalue().encode()


def freeze(source, target):
    source=Path(source).resolve(); ref=source/'reference_v4'; old=ref/'reference_v3'
    assert read(source/'run_state.json')['status']=='complete'
    assert read(source/'submission_status.json')['status']=='complete'
    assert sha((source/'submission_selected.csv').read_bytes())==EXPECTED_SUBMISSION
    files={}; provenance=[]
    def copy(path,name):
        content=path.read_bytes();files[name]=content
        provenance.append(dict(artifact=name,source=str(path.relative_to(source)),sha256=sha(content)))
    for name in ['submission_selected.csv','submission_provenance.csv','run_identity.json','progress.json']:
        copy(source/name,name)
    for name in ['dataset_inventory','coverage_sources','component_ablation_sources',
                 'coverage_score_sources','training_grid_sources','final_system_sources','v4_trials']:
        copy(ref/'report'/f'{name}.csv','tables/'+name+'.csv')
    for name in ['trial_results','audit_gains','selection_candidates','worker_resources']:
        copy(source/'report'/f'{name}.csv','tables/'+name+'.csv')
    # Keep exact underlying code for source inspection, never import it during report replay.
    for path in sorted(source.glob('*.py')):
        copy(path,'archived_code/'+path.name)
    test_manifest=read(old/'test_manifest.json')['tests']
    rows=list(csv.DictReader(io.StringIO(files['submission_selected.csv'].decode())))
    assert [r['ID'] for r in rows]==[r['id'] for r in test_manifest]
    constraints=[]
    for index,(r,t) in enumerate(zip(rows,test_manifest)):
        ids=list(map(int,r['TopConfigs'].split(';')))
        assert len(ids)==len(set(ids))==len(t['template_values']) and min(ids)>=0
        prediction=read(old/'test_predictions'/f'{index:05d}'/'prediction.json')
        assert prediction['ID']==r['ID']
        count=prediction['count']
        assert all(0<=v<count for v in ids[:min(count,len(ids))])
        if t['collection'].startswith('layout'):
            assert len(ids)<=count
            if len(ids)==count:assert sorted(ids)==list(range(count))
        elif len(ids)>count:
            assert ids[count:]==t['template_values'][count:]==list(range(count,len(ids)))
        constraints.append(dict(ID=r['ID'],collection=t['collection'],length=len(ids),candidate_count=count,
                                ranking_sha256=sha(r['TopConfigs'].encode())))
    files['tables/submission_constraints.csv']=csv_bytes(constraints)
    models=[];sizes=[];histories=[];recipes=[];decisions=[];summary=[];stats=[]
    for c in COLLECTIONS:
        key=c.replace(':','_'); catalog={}
        for base in [old,ref,source]:
            path=base/key/'experiment_plan.json'
            if path.exists():
                for e in read(path):catalog[e['name']]=(base/key,e)
        lock_path=source/key/'selection_lock.json' if c.startswith('layout') else old/key/'selection_lock.json'
        lock=read(lock_path);copy(lock_path,'locks/'+key+'.json')
        if c.startswith('layout'):
            audit=list(csv.DictReader(io.StringIO((source/key/'system_audit_gains.csv').read_text())))
            chosen=next(r for r in audit if r['experiment']=='v5_selected')
            summary.append(dict(collection=c,dev_gain=lock['gain_vs_v4'],**chosen))
        for name,weight in lock['components'].items():
            directory,e=catalog[name]
            recipes.append(dict(collection=c,name=name,weight=weight,experiment=e))
            for fold in [0,1]:
                for seed in [42,137,2026]:
                    d=directory/'folds'/str(fold)/'trials'/name/str(seed)
                    model=d/('model.pt' if e['model']=='gnn' else 'model.joblib')
                    assert model.is_file(),model
                    models.append(dict(collection=c,experiment=name,weight=weight,fold=fold,seed=seed,
                        path=str(model.relative_to(source)),bytes=model.stat().st_size,sha256=sha(model.read_bytes())))
                    records=read(d/'training_manifest.json');counts=[len(r['config_ids']) for r in records]
                    sizes.append(dict(collection=c,experiment=name,weight=weight,fold=fold,seed=seed,
                        graphs=len(records),configurations=sum(counts),min_configs=min(counts),max_configs=max(counts)))
                    if e['model']=='gnn':
                        h=list(csv.DictReader(io.StringIO((d/'history.csv').read_text())))
                        result=read(d/'result.json')
                        for r in h:histories.append(dict(collection=c,experiment=name,fold=fold,seed=seed,
                            selected_epoch=result['best_epoch'],**r))
        # Record actual split counts, including small XLA audit sets.
        for fold in [0,1]:
            m=read(old/key/'manifests'/f'fold_{fold}.json')
            for split in ['train','dev','holdout']:
                records=[r for r in m['records'] if r['split']==split]
                stats.append(dict(collection=c,fold=fold,split=split,graphs=len(records),
                    candidate_pool_rows=sum(len(r['indices']) for r in records)))
    files['tables/model_inventory.csv']=csv_bytes(models)
    files['tables/selected_training_sizes.csv']=csv_bytes(sizes)
    files['tables/selected_gnn_histories.csv']=csv_bytes(histories)
    files['tables/selected_audit_summary.csv']=csv_bytes(summary)
    files['tables/reference_split_inventory.csv']=csv_bytes(stats)
    files['selected_recipes.json']=json_bytes(recipes)
    files['scientific_config.json']=json_bytes(read(source/'run_state.json')['config'])
    files['reported_kaggle_scores.json']=json_bytes([
        dict(label='Basic baseline',score=.40,precision='approximate',source='user-reported'),
        dict(label='Tabular system',score=.53,precision='approximate',source='user-reported'),
        dict(label='Initial hybrid',score=.55,precision='approximate',source='user-reported'),
        dict(label='Data/training extension',score=.558,precision='reported to three decimals',source='user-reported'),
        dict(label='Final selected system',score=.567,precision='reported to three decimals',source='user-reported')])
    files['source_provenance.json']=json_bytes(provenance)
    manifest=dict(frozen_date='2026-10-21',description='Final evidence replay; not training or fresh test inference',
        submission_sha256=EXPECTED_SUBMISSION,submission_rows=894,public_score_user_reported=.567,
        selected_model_files=len(models),source_run_signature=read(source/'run_identity.json')['signature'],
        files={k:dict(sha256=sha(v),bytes=len(v)) for k,v in files.items()})
    files['freeze_manifest.json']=json_bytes(manifest)
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,data in sorted(files.items()):
            i=zipfile.ZipInfo(name,date_time=(2026,10,21,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(i,data)
    (target.parent/'freeze_manifest.json').write_bytes(files['freeze_manifest.json'])
    print(target, target.stat().st_size,'bytes;',len(models),'saved selected-model files')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--target',default=str(HERE.parents[1]/'artifacts/final/frozen_evidence.zip'))
    a=p.parse_args();freeze(a.source,a.target)
