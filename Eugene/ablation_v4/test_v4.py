"""Real synthetic tabular fits/inference plus protocol/resume invariants; no Kaggle training."""
import ast
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
HERE=Path(__file__).resolve().parent
RUNTIME=Path(tempfile.mkdtemp(prefix='sc4000-v4-tests-runtime-'))
for name in ['engine','gnn','common','plan','submission','controller']:
    shutil.copy2(HERE.parent/'ablation_v3'/(name+'.py'),RUNTIME/('study_'+name+'.py'))
for p in HERE.glob('v4_*.py'):shutil.copy2(p,RUNTIME/p.name)
sys.path.insert(0,str(RUNTIME))
import numpy as np
import pandas as pd
import study_engine as engine
import study_plan as plan
import v4_core as core
import v4_controller as controller
import v4_worker as worker


def synthetic(p,seed):
    rng=np.random.default_rng(seed);n,c=12,16
    x=rng.uniform(0,10,(c,24)).astype(np.float32)
    p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p,node_feat=rng.uniform(0,4,(n,140)).astype(np.float32),
        node_opcode=np.arange(n,dtype=np.int64)%7,edge_index=np.column_stack([np.arange(1,n),np.arange(n-1)]),
        config_feat=x,config_runtime_normalizers=np.ones(c),config_runtime=2+x[:,0])


def reference(root,ref):
    cfg=plan.default_config();cfg.update(collections=['tile:xla'],gnn_collections=[],folds=[0,1],seeds=[42],
        dev_graphs=2,holdout_graphs=2,tile_train_graphs=3,tile_more_graphs=3,train_pool_configs=16,
        fit_configs=16,eval_configs=16,cpu_threads=1,bootstrap_repeats=20,system_reserve_gib=.05,
        disk_reserve_gib=.01)
    for split,n in [('train',9),('valid',2),('test',1)]:
        for i in range(n):synthetic(root/'tile/xla'/split/f'{split}_{i}.npz',i+len(split)*20)
    template=root/'sample_submission.csv';pd.DataFrame([dict(ID='tile:xla:test_0',TopConfigs='0;1;2;3;4')]).to_csv(template,index=False)
    cfg['sample_submission']=str(template)
    ref.mkdir();d=ref/'tile_xla';d.mkdir()
    catalog=[dict(e,iterations=3,min_leaf=2) for e in plan.tabular_plan('tile:xla',cfg) if e['name'] in ['B0_basic_hgb','T_deeper_hgb']]
    engine.write_json(d/'experiment_plan.json',catalog)
    trials=[]
    for fold in cfg['folds']:
        m=plan.manifest(root,'tile:xla',cfg,fold)
        engine.write_json(d/'manifests'/f'fold_{fold}.json',m)
        scratch=root.parent/f'prep_{fold}';prep=root.parent/f'prep_report_{fold}';prep.mkdir()
        engine.prepare(root,m,dict(cfg,gnn_collections=[]),scratch,prep)
        for e in catalog:
            p=d/'folds'/str(fold)/'trials'/e['name']/'42'
            job=dict(cfg=plan.effective_config(cfg,e,'tile:xla'),experiment=e,seed=42,collection='tile:xla',scratch=str(scratch),output=str(p))
            engine.run_trial(job);engine.evaluate_holdout(job)
            for name in ['status.json','holdout_status.json']:engine.write_json(p/name,dict(status='ok',peak_rss_gib=.1,wall_seconds=.1))
            trials.append(dict(core.read(p/'result.json'),peak_rss_gib=.1,wall_seconds=.1,fold=fold))
    pd.DataFrame(trials).to_csv(d/'all_trials.csv',index=False)
    engine.write_json(d/'selection_lock.json',dict(components={'T_deeper_hgb':1.},winner='T_deeper_hgb',gnn=None))
    engine.write_json(d/'complete.json',dict(status='complete'))
    engine.write_json(ref/'run_state.json',dict(config=cfg))
    versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','joblib','psutil']}
    engine.write_json(ref/'run_identity.json',dict(version='3.0.0',versions=versions))
    engine.write_json(ref/'submission_status.json',dict(status='complete'))
    path=root/'tile/xla/test/test_0.npz'
    with zipfile.ZipFile(path) as z:fp=[(i.filename,i.CRC,i.file_size) for i in z.infolist()]
    r=dict(id='tile:xla:test_0',collection='tile:xla',relative_path='tile/xla/test/test_0.npz',template_values=list(range(5)),fingerprint=fp)
    engine.write_json(ref/'test_manifest.json',dict(tests=[r]))
    p=root.parent/'reference_test';p.mkdir()
    worker.inference(dict(cfg=cfg,collection='tile:xla',path=str(path),id=r['id'],template_values=r['template_values'],output=str(p),
        components={'T_deeper_hgb':1.},sources={'T_deeper_hgb':dict(directory=str(d),experiment=next(e for e in catalog if e['name']=='T_deeper_hgb'))}))
    pd.DataFrame([dict(ID=r['id'],TopConfigs=';'.join(map(str,core.read(p/'prediction.json')['selected'])))]).to_csv(ref/'submission_selected.csv',index=False)
    rows=[]
    for n in ['baseline','tabular','selected']:
        for i in range(2):rows.append(dict(experiment=n,graph=f'valid_{i}',seed=0,score=.9))
    pd.DataFrame(rows).to_csv(d/'system_holdout_graphs.csv',index=False)
    return cfg


class ProtocolTests(unittest.TestCase):
    def test_selection_rejects_one_fold_regression(self):
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);d=t/'ref/layout_xla_default';d.mkdir(parents=True)
            engine.write_json(d/'selection_lock.json',dict(components={'T':.5,'G':.5},winner='T',gnn='G'))
            cfg=dict(folds=[0,1],gnn_collections=['layout:xla:default'],minimum_gain_layout=.002,
                     minimum_gain_tile=.0005,fold_regression_tolerance=0.)
            def frame(x):return pd.DataFrame(dict(graph=['g'],config_id=[0],runtime=[1.],prediction=[x]))
            old={f:{'T':frame(.5),'G':frame(.5)} for f in cfg['folds']}
            new={0:{'V4_G_longer':frame(.9)},1:{'V4_G_longer':frame(.4)}}
            with patch.object(core,'predictions',side_effect=[old,new]),patch.object(core,'frame_score',side_effect=lambda f,t:float(f.prediction.mean())):
                result=core.choose(t/'ref',t/'out','layout:xla:default',['V4_G_longer'],cfg,[42])
            self.assertFalse(result['changed'])
            self.assertTrue(any(r['gain_vs_v3']>0 and not r['eligible'] for r in result['candidates']))

    def test_gnn_epoch_resume_and_inference(self):
        try:import torch
        except ImportError:self.skipTest('PyTorch unavailable; run with temporary PyTorch PYTHONPATH')
        import study_gnn
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);root=t/'data';cfg=plan.default_config()
            cfg.update(collections=['layout:xla:default'],cpu_threads=1,gnn_epochs=1,gnn_patience=10,
                gnn_nodes=16,gnn_config_nodes=12,gnn_edges=128,gnn_train_graphs=3,gnn_configs=8,
                gnn_batch=4,gnn_predict_batch=4,gnn_early_configs=8,dev_graphs=2,holdout_graphs=1,
                train_pool_configs=16,eval_configs=16,max_excluded_fraction=0.)
            for split,nfiles in [('train',9),('valid',1),('test',1)]:
                for i in range(nfiles):
                    p=root/'layout/xla/default'/split/f'{split}_{i}.npz';p.parent.mkdir(parents=True,exist_ok=True)
                    rng=np.random.default_rng(i+len(split));x=rng.integers(-1,6,(16,12,18)).astype(np.float32)
                    np.savez_compressed(p,node_feat=rng.uniform(0,4,(12,140)).astype(np.float32),
                        node_opcode=np.arange(12)%7,edge_index=np.column_stack([np.arange(1,12),np.arange(11)]),
                        node_config_ids=np.arange(12),node_config_feat=x,config_runtime=20+x[:,0,0]+.1*x[:,1,0])
            exp=plan.gnn_exp('V4_G_longer','G_full',nodes=16,graphs=3,configs=8,stage='training budget')
            settings=plan.effective_config(cfg,exp,'layout:xla:default')
            m=plan.manifest(root,'layout:xla:default',settings,0);prep=t/'prep';prep.mkdir()
            engine.prepare(root,m,settings,t/'scratch',prep)
            d=t/'model';p=d/'folds/0/trials/V4_G_longer/42'
            job=dict(action='trial',cfg=settings,experiment=exp,seed=42,collection='layout:xla:default',scratch=str(t/'scratch'),output=str(p))
            worker.run(job)
            first=pd.read_csv(p/'history.csv')
            self.assertEqual(first.epoch.tolist(),[1])
            job['cfg']=dict(settings,gnn_epochs=2);worker.run(job)
            after=pd.read_csv(p/'history.csv');self.assertEqual(after.epoch.tolist(),[1,2])
            pd.testing.assert_frame_equal(first,after.iloc[:1].reset_index(drop=True))
            worker.run(dict(job,action='holdout'))
            dest=t/'inference';dest.mkdir()
            worker.inference(dict(cfg=dict(settings,folds=[0],seeds=[42]),collection='layout:xla:default',
                path=str(root/'layout/xla/default/test/test_0.npz'),id='layout:xla:default:test_0',
                template_values=list(range(16)),output=str(dest),components={'V4_G_longer':1.},
                sources={'V4_G_longer':dict(directory=str(d),experiment=exp)}))
            self.assertEqual(sorted(core.read(dest/'prediction.json')['selected']),list(range(16)))

    def test_notebook_compiles_and_embeds_exact_modules(self):
        notebook=json.loads((HERE.parent/'SC4000_V4_Focused_Scaling_and_Report.ipynb').read_text())
        import nbformat
        nbformat.validate(nbformat.from_dict(notebook))
        for c in notebook['cells']:
            if c['cell_type']=='code':ast.parse(''.join(c['source']))
        texts=[''.join(c['source']) for c in notebook['cells']]
        for p in HERE.glob('v4_*.py'):self.assertTrue(any(repr(p.read_text()) in s for s in texts),p.name)

    def test_restore_guards(self):
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);z=t/'bad.zip'
            with zipfile.ZipFile(z,'w') as f:f.writestr('../escape','bad')
            with self.assertRaises(ValueError):core.restore(z,t/'out')
            self.assertFalse((t/'escape').exists())
            with zipfile.ZipFile(z,'w') as f:f.writestr('safe','ok')
            core.restore(z,t/'out')
            with self.assertRaises(ValueError):core.restore(z,t/'out')
            z.write_text('truncated')
            with self.assertRaises(ValueError):core.restore(z,t/'out2')

    def test_real_v3_incumbent_available_without_retraining(self):
        ref=HERE.parents[1]/'results/sc4000_gnn_v3_results (65)'
        if not ref.exists():self.skipTest('Local checkpoint 65 unavailable')
        cfg=core.defaults(ref)
        for c in cfg['collections']:
            choice=core.choose(ref,Path('/unused'),c,[],cfg,cfg['seeds'])
            self.assertTrue(choice['eligible']);self.assertGreaterEqual(choice['gain_vs_v3'],-1e-10)
            self.assertIn(choice['incumbent'],[r['components'] for r in choice['candidates']])

    def test_end_to_end_resume_and_changed_inference(self):
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);root=t/'data';ref=t/'ref';reference(root,ref)
            out=t/'v4';out.mkdir();shutil.copytree(ref,out/'reference_v3')
            cfg=core.defaults(out/'reference_v3')
            cfg.update(tile_graph_ladder=[5],minimum_gain_tile=10.,session_hours=1.,checkpoint_minutes=999,
                       pilot_iterations=2,worker_ram_gib=3.,max_matrix_mib=128)
            # Nesting, labels and reserved dev graphs.
            exp=core.plans(ref,'tile:xla',cfg)[0]
            old=core.read(ref/'tile_xla/manifests/fold_0.json')
            bigger=core.expanded_manifest(root,ref,'tile:xla',0,exp,cfg)
            self.assertTrue(all(r in bigger['records'] for r in old['records']))
            self.assertEqual(sum(r['split']=='train' for r in bigger['records']),5)
            constrained=dict(cfg,max_matrix_mib=.0001)
            self.assertFalse(core.resources(ref,'tile:xla',exp,bigger,constrained)['admitted'])
            archive=controller.run(root,out,t/'scratch',RUNTIME,cfg,enforce_versions=False)
            self.assertEqual(core.read(out/'run_state.json')['status'],'complete')
            self.assertEqual((out/'submission_selected.csv').read_bytes(),(ref/'submission_selected.csv').read_bytes())
            before={str(p.relative_to(out)):core.sha(p) for p in out.glob('*/folds/*/trials/*/*/status.json')}
            self.assertTrue(before)
            controller.run(root,out,t/'scratch',RUNTIME,cfg,enforce_versions=False)
            after={str(p.relative_to(out)):core.sha(p) for p in out.glob('*/folds/*/trials/*/*/status.json')}
            self.assertEqual(before,after)
            self.assertFalse((t/'scratch/work').exists())
            self.assertFalse((out/'report_error.json').exists())
            self.assertEqual(len(list((out/'report').glob('0*.png'))),6)
            with self.assertRaises(ValueError):
                controller.run(root,out,t/'scratch',RUNTIME,dict(cfg,minimum_gain_tile=9.),enforce_versions=False)
            restored=t/'restored';core.restore(archive,restored)
            self.assertTrue((restored/'reference_v3/submission_selected.csv').exists())
            controller.run(root,restored,t/'restored_scratch',RUNTIME,cfg,enforce_versions=False)
            self.assertEqual(core.read(restored/'run_state.json')['status'],'complete')
            restored_hashes={str(p.relative_to(restored)):core.sha(p) for p in restored.glob('*/folds/*/trials/*/*/status.json')}
            self.assertEqual(before,restored_hashes)
            # Exercise a changed collection through real full-candidate inference/export.
            lock=core.read(out/'tile_xla/selection_lock.json');lock.update(changed=True,components={exp['name']:1.})
            engine.write_json(out/'tile_xla/selection_lock.json',lock)
            def inline(job,status):
                worker.run(job);engine.write_json(status,dict(status='ok'));return True
            controller.export(root,out/'reference_v3',out,cfg,inline)
            pred=core.read(out/'test_predictions/00000/prediction.json')
            self.assertEqual(len(set(pred['selected'])),5)
            self.assertTrue((out/'test_predictions/00000/ranks.npz').exists())
            self.assertEqual(core.read(out/'submission_status.json')['rows'],1)
            # Dataset identity mismatch blocks reuse.
            synthetic(root/'tile/xla/train/train_0.npz',999)
            with self.assertRaises(ValueError):core.expanded_manifest(root,ref,'tile:xla',0,dict(exp,graph_limit=0),cfg)


if __name__=='__main__':
    try:unittest.main(verbosity=2)
    finally:shutil.rmtree(RUNTIME)
