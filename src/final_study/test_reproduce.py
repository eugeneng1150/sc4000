"""Real synthetic fits, missing-model recovery, archive reuse and fixed training manifests."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import zipfile
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
RUNTIME=Path(tempfile.mkdtemp(prefix='sc4000-final-runtime-test-'))
for name in ['engine','gnn','common','plan','submission']:
    shutil.copy2(REPO/'src/ablation_v3'/f'{name}.py',RUNTIME/f'study_{name}.py')
for name in ['v4_core','v4_worker']:shutil.copy2(REPO/'src/ablation_v4'/f'{name}.py',RUNTIME/f'{name}.py')
for name in ['v5_worker','v5_scheduler']:shutil.copy2(REPO/'src/ablation_v5'/f'{name}.py',RUNTIME/f'{name}.py')
for name in ['reproduce','final_worker']:shutil.copy2(HERE/f'{name}.py',RUNTIME/f'{name}.py')
sys.path.insert(0,str(RUNTIME))
import numpy as np
import pandas as pd
import reproduce as r
import study_engine as engine
import study_plan as plan
from study_gnn import catalog
from final_worker import training_rows,canonical


def synthetic(path,tile=False,seed=0,candidates=10):
    rng=np.random.default_rng(seed);n=14;c=candidates
    data=dict(node_feat=rng.uniform(0,3,(n,140)).astype(np.float32),node_opcode=np.arange(n)%7,
        edge_index=np.column_stack([np.arange(1,n),np.arange(n-1)]))
    if tile:
        x=rng.uniform(0,5,(c,24)).astype(np.float32)
        data.update(config_feat=x,config_runtime=2+x[:,0],config_runtime_normalizers=np.ones(c))
    else:
        x=rng.integers(-1,6,(c,n,18)).astype(np.float32)
        data.update(node_config_feat=x,node_config_ids=np.arange(n),config_runtime=10+x[:,0,0])
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**data)


def fixture(root):
    cfg=plan.default_config();cfg.update(folds=[0],seeds=[42],hgb_iterations=3,fit_configs=8,
        train_pool_configs=10,eval_configs=10,gnn_nodes=14,gnn_config_nodes=14,gnn_edges=100,
        gnn_train_graphs=3,gnn_configs=8,gnn_epochs=2,gnn_patience=2,gnn_batch=4,
        gnn_early_configs=8,gnn_predict_batch=4,inference_batch=4,opcode_bins=8,depth_bins=8,edge_bins=8,
        cpu_threads=1,max_excluded_fraction=0.,max_graph_features_mib=8,max_matrix_mib=8)
    contract=dict(version=r.VERSION,jobs=[],manifests={},training_rows={},trainings={},selected={},
        inference_cfg=cfg,versions={},historical_submission_sha256='not-a-historical-run')
    import importlib.metadata
    for p in ['numpy','pandas','scipy','scikit-learn','torch','joblib','xgboost']:contract['versions'][p]=importlib.metadata.version(p)
    template=[]
    for ci,c in enumerate(engine.COLLECTIONS):
        tile=c.startswith('tile');records=[]
        for i in range(6):
            split='train' if i<3 else 'dev' if i<5 else 'holdout'
            name=f'graph_{ci}_{i}';rel=Path(*c.split(':'))/('valid' if split=='holdout' else 'train')/(name+'.npz')
            synthetic(root/rel,tile,ci*10+i)
            with zipfile.ZipFile(root/rel) as z:identity=[(v.filename,v.CRC,v.file_size) for v in z.infolist()]
            records.append(dict(split=split,graph=name,family=name,relative_path=str(rel),count=10,
                indices=list(range(10)),source=engine.digest(identity),fortran=False,row_bytes=96 if tile else 14*18*4))
        m=dict(collection=c,fold=0,records=records);key=engine.digest(m);contract['manifests'][key]=m
        base=next(e for e in engine.experiments(tile) if e['name']=='B0_basic_hgb')
        selected=dict(base,name='percentile',target='percentile')
        settings=dict(cfg,gnn_collections=[])
        if c.startswith('layout:xla'):
            selected=dict(next(e for e in catalog() if e['name']=='G_full'),name='G_selected',profile=dict(nodes=14,config_nodes=14,edges=100,sampler='uniform'),train_graphs=3,configs=8)
            settings=dict(cfg,gnn_collections=[c])
        contract['selected'][c]={base['name']:.5,selected['name']:.5}
        for exp,roles in [(base,['baseline','selected']),(selected,['selected'])]:
            selected_cfg=settings if exp['model']=='gnn' else dict(cfg,gnn_collections=[])
            rows=training_rows(records,exp,42,selected_cfg);refs=[]
            for row in rows:
                k=engine.digest(row);contract['training_rows'][k]=row;refs.append(k)
            k=engine.digest(refs);contract['trainings'][k]=refs
            contract['jobs'].append(dict(id=f'{c}/0/{exp["name"]}/42',collection=c,fold=0,seed=42,
                experiment=exp,cfg=selected_cfg,manifest=key,roles=roles,training=k,
                archive_directory=f'{c.replace(":","_")}/folds/0/trials/{exp["name"]}/42',archive_files={}))
        synthetic(root.joinpath(*c.split(':'),'test','test.npz'),tile,99+ci,3 if tile else 10)
        template.append(dict(ID=c+':test',TopConfigs=';'.join(map(str,range(5 if tile else 10)))))
    # Optional objective controls share the same exact examples, without becoming final members.
    control=next(s for s in contract['jobs'] if s['collection']=='layout:nlp:default' and 'baseline' in s['roles'])
    for kind in ['xgb_reg','xgb_rank']:
        spec=json.loads(json.dumps(control));spec['roles']=['ablation']
        spec['experiment'].update(name=kind,model=kind,iterations=3)
        spec['cfg'].update(xgb_depth=2,xgb_learning_rate=.1,xgb_max_bin=16,xgb_pairs=2)
        spec['id']=f'layout:nlp:default/0/{kind}/42'
        spec['archive_directory']=f'layout_nlp_default/folds/0/trials/{kind}/42'
        contract['jobs'].append(spec)
    path=root/'sample_submission.csv';pd.DataFrame(template).to_csv(path,index=False)
    return contract,path


class ReproductionTests(unittest.TestCase):
    def test_recorded_all_training_ids_are_reconstructible(self):
        with zipfile.ZipFile(REPO/'artifacts/final/reproduction_spec.zip') as z:c=json.loads(z.read('contract.json'))
        self.assertEqual(len(r.plan(c)),90)
        self.assertEqual(sum('selected' in j['roles'] for j in r.plan(c)),60)
        for spec in c['jobs']:
            records=c['manifests'][spec['manifest']]['records']
            self.assertEqual(canonical(training_rows(records,spec['experiment'],spec['seed'],spec['cfg'])),
                             canonical(r.expected_rows(c,spec)),spec['id'])

    def test_cold_train_warm_resume_missing_model_and_archive(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);root=p/'data';contract,template=fixture(root);out=p/'run'
            resources=dict(cpu_threads=1,worker_ram_gib=1.,system_reserve_gib=.1,disk_reserve_gib=.01,
                session_hours=1,checkpoint_minutes=100,monitor_seconds=.05)
            args=dict(root=root,output=out,scratch=p/'scratch',runtime=RUNTIME,contract=contract,
                resources=resources,sample_submission=str(template),strict_versions=False)
            self.assertEqual(r.run(**args)['status'],'complete')
            submitted=(out/'submission_final.csv').read_bytes()
            hashes={s['id']:r.sha(r.model_dir(out,s)/r.model_name(s)) for s in r.plan(contract)}
            events=(out/'scheduler_events.jsonl').read_text()
            self.assertEqual(r.run(**args)['status'],'complete')
            self.assertEqual(events,(out/'scheduler_events.jsonl').read_text())
            self.assertEqual(submitted,(out/'submission_final.csv').read_bytes())
            # Missing one valid model must retrain exactly that fit, not silently replay predictions.
            victim=contract['jobs'][0];(r.model_dir(out,victim)/r.model_name(victim)).unlink()
            self.assertEqual(r.run(**args)['status'],'complete')
            for s in r.plan(contract):
                if s['id']!=victim['id']:self.assertEqual(hashes[s['id']],r.sha(r.model_dir(out,s)/r.model_name(s)))
            # Enabling optional ablations reuses the completed final models rather than invalidating them.
            args['run_ablations']=True
            self.assertEqual(r.run(**args)['status'],'complete')
            self.assertEqual(submitted,(out/'submission_final.csv').read_bytes())
            for s in r.plan(contract):self.assertEqual(hashes[s['id']],r.sha(r.model_dir(out,s)/r.model_name(s)))
            # Construct a verified historical-style archive from the actual synthetic fitted models.
            archive=p/'models.zip'
            with zipfile.ZipFile(archive,'w') as z:
                for s in contract['jobs']:
                    d=r.model_dir(out,s);names=[r.model_name(s),'result.json','training_manifest.json']
                    s['archive_files']={n:r.sha(d/n) for n in names}
                    for n in names:z.write(d/n,s['archive_directory']+'/'+n)
            other=p/'imported'
            args.update(output=other,scratch=p/'scratch2',contract=contract,model_archive=archive)
            self.assertEqual(r.run(**args)['status'],'complete')
            self.assertNotIn('TRAIN ',(other/'scheduler_events.jsonl').read_text())
            self.assertEqual(submitted,(other/'submission_final.csv').read_bytes())
            # Same archive path but changed data must be rejected before any fit/reuse.
            raw=next(root.glob('layout/xla/default/train/*.npz'));synthetic(raw,False,999)
            with self.assertRaisesRegex(ValueError,'Raw data differs'):r.run(**args)

    def test_notebook_defaults_and_embedded_implementations(self):
        import nbformat
        n=nbformat.read(REPO/'notebooks/SC4000_Final.ipynb',as_version=4);nbformat.validate(n)
        src='\n'.join(c.source for c in n.cells if c.cell_type=='code')
        self.assertIn('RUN_MODE = "reproduce"',src);self.assertIn('RUN_ABLATIONS = False',src)
        self.assertIn('MODEL_ARCHIVE = ""',src);self.assertNotIn('final_report.export_submission(',src)
        for name in ['reproduce','final_worker','story']:
            self.assertIn(repr((HERE/(name+'.py')).read_text()),src)
        for c in n.cells:
            if c.cell_type=='code':ast.parse(c.source)

if __name__=='__main__':unittest.main(verbosity=2)
