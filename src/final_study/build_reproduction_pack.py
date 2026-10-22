"""Curate label-free manifests and exact recipes; no weights or raw data required at runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent

def read(p): return json.loads(Path(p).read_text())
def digest(v): return hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def build(source, target):
    source = Path(source).resolve()
    bases = [source/'reference_v4/reference_v3', source/'reference_v4', source]
    old = bases[0]
    selected = {}
    for c in read(old/'run_state.json')['config']['collections']:
        lock = source/c.replace(':','_')/'selection_lock.json'
        if not lock.exists(): lock = old/c.replace(':','_')/'selection_lock.json'
        selected[c] = read(lock)['components']
    manifests, jobs = {}, []
    for base in bases:
        for result in sorted(base.glob('*/folds/*/trials/*/*/result.json')):
            d = result.parent
            if read(result).get('status') != 'ok': continue
            job = read(d/'status.job.json')
            c, fold = job['collection'], int(d.parents[2].name)
            exp, cfg = job['experiment'], job['cfg']
            reference = read(old/c.replace(':','_')/'manifests'/f'fold_{fold}.json')
            if base == old:
                manifest = reference
            else:
                dev = {r['graph'] for r in reference['records'] if r['split']=='dev'}
                candidates = [read(p) for p in (base/c.replace(':','_')/'preparations').glob('*/manifest.json')]
                candidates = [m for m in candidates if {r['graph'] for r in m['records'] if r['split']=='dev'}==dev]
                if not candidates: raise ValueError(f'Missing preparation manifest: {d}')
                # Multiple resource/data settings may have preparations; identify the exact training records.
                expected = read(d/'training_manifest.json')
                needed = {r['graph'] for r in expected}
                candidates = [m for m in candidates if needed <= {r['graph'] for r in m['records'] if r['split']=='train'}]
                manifest = min(candidates, key=lambda m: len(m['records']))
            key = digest(manifest); manifests[key] = manifest
            model = d/('model.pt' if exp['model']=='gnn' else 'model.joblib')
            roles = []
            if exp['name'] in selected[c]: roles.append('selected')
            if exp['name']=='B0_basic_hgb': roles.append('baseline')
            if not roles: roles=['ablation']
            jobs.append(dict(id=f'{c}/{fold}/{exp["name"]}/{job["seed"]}', collection=c, fold=fold,
                seed=job['seed'], experiment=exp, cfg=cfg, manifest=key, roles=roles,
                expected_training=read(d/'training_manifest.json'),
                archive_directory=str(d.relative_to(source)),
                archive_files={p.name: sha(p) for p in d.iterdir() if p.is_file() and p.name in
                    ['model.pt','model.joblib','result.json','training_manifest.json','history.csv']},
                historical_dev_score=read(result)['dev_score']))
    assert len({j['id'] for j in jobs})==len(jobs)
    assert sum('selected' in j['roles'] for j in jobs)==60
    assert sum('baseline' in j['roles'] for j in jobs)==30
    training_rows, trainings = {}, {}
    for job in jobs:
        refs=[]
        for row in job.pop('expected_training'):
            key=digest(row);training_rows[key]=row;refs.append(key)
        key=digest(refs);trainings[key]=refs;job['training']=key
    contract = dict(training_rows=training_rows, trainings=trainings, version='final-reproduction-1', selected=selected, jobs=jobs,
        versions=read(source/'run_identity.json')['versions'],
        inference_cfg=read(source/'run_state.json')['config'],
        manifests=manifests, historical_submission_sha256=sha(source/'submission_selected.csv'),
        note='Exact historical inputs and recipes. No model weights, raw arrays or runtime labels.')
    files={'contract.json':json.dumps(contract, separators=(',',':')).encode()}
    # Early model-family experiments are exploratory, not directly comparable to later splits.
    legacy=HERE.parents[1]/'archive/legacy/figures/experiment_results_summary_from_log.csv'
    files['early_model_exploration.csv']=legacy.read_bytes()
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,data in files.items():
            info=zipfile.ZipInfo(name,date_time=(2026,10,21,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,data)
    print(target, target.stat().st_size, 'bytes;',len(jobs),'available fits;',len(manifests),'unique manifests')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--target',default=str(HERE.parents[1]/'artifacts/final/reproduction_spec.zip'))
    a=p.parse_args();build(a.source,a.target)
