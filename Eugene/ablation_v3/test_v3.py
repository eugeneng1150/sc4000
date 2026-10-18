"""Integration checks for v3. Uses synthetic NPZs, never competition training."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
import zipfile
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
for name in ['engine','gnn','common','plan','submission','controller']:
    spec=importlib.util.spec_from_file_location('study_'+name,HERE/(name+'.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
import study_engine as engine,study_gnn as gnn,study_plan as plan,study_controller as controller,study_submission as submission,study_common as common


def synthetic(path,tile=False,seed=1):
    rng=np.random.default_rng(seed);n,c=32,16
    data=dict(node_feat=rng.uniform(0,4,(n,140)).astype(np.float32),node_opcode=np.arange(n,dtype=np.int64)%7,
              edge_index=np.column_stack([np.arange(1,n),np.arange(n-1)]))
    if tile:
        x=rng.uniform(0,10,(c,24)).astype(np.float32);normalizer=np.linspace(1,2,c)
        data.update(config_feat=x,config_runtime_normalizers=normalizer,config_runtime=(2+x[:,0])*normalizer)
    else:
        x=rng.integers(-1,6,(c,n,18)).astype(np.float32)
        data.update(node_config_ids=np.arange(n),node_config_feat=x,
                    config_runtime=20+x[:,0,0]+.2*x[:,1,1]+rng.uniform(0,.1,c))
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**data)
    return data


class V3Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.cfg=plan.default_config()
    def tearDown(self):self.temp.cleanup()

    def test_notebook_embeds_exact_sources(self):
        import nbformat
        notebook=nbformat.read(HERE.parent/'SC4000_V3_Coverage_and_Data.ipynb',as_version=4)
        nbformat.validate(notebook);names=[]
        for c in notebook.cells:
            if c.cell_type!='code':continue
            compile(c.source,'notebook','exec')
            if c.source.startswith('# Embedded implementation:'):
                name=c.source.splitlines()[0].split(': ')[1]
                self.assertEqual(ast.literal_eval(ast.parse(c.source).body[0].value),(HERE/name).read_text());names.append(name)
        self.assertEqual(names,['engine.py','gnn.py','common.py','plan.py','submission.py','controller.py'])

    def test_graph_and_configuration_budgets_are_separate(self):
        stages=plan.coverage_plan(self.cfg)
        self.assertEqual(len(stages),8)
        self.assertEqual({(e['train_graphs'],e['configs']) for e in stages},{(24,64)})
        follow=plan.followup_plan(self.cfg,plan.profile(2048))
        for e in follow:
            if e['model']=='gnn':self.assertEqual(e['profile']['nodes'],2048)
        data=[e for e in follow if e['stage']=='training_data'];matched=[e for e in follow if e.get('matched') and e['model']=='hgb']
        self.assertEqual(len(data),8);self.assertEqual(len(matched),9)
        self.assertEqual({e['profile']['sampler'] for e in data},{'uniform'})
        ids=[engine.training_indices(384,'graph',42,dict(self.cfg,gnn_configs=n),True) for n in [64,128,256]]
        self.assertTrue(set(ids[0])<set(ids[1])<set(ids[2]))

    def test_fold_disjointness_and_label_independent_manifest(self):
        folder=self.root/'layout/xla/default'
        for split,number in [('train',14),('valid',2)]:
            for i in range(number):synthetic(folder/split/f'bert_{split}_{i}.npz',seed=i+1)
        cfg=dict(self.cfg,dev_graphs=2,eval_configs=8)
        manifests=[plan.manifest(self.root,'layout:xla:default',cfg,f) for f in cfg['folds']]
        sets=[]
        for m in manifests:
            splits={s:{r['graph'] for r in m['records'] if r['split']==s} for s in ['train','dev','holdout']}
            self.assertFalse(splits['train']&splits['dev']);self.assertFalse(splits['train']&splits['holdout']);sets.append(splits)
        self.assertFalse(sets[0]['dev']&sets[1]['dev'])
        self.assertEqual(sets[0]['holdout'],sets[1]['holdout'])
        path=folder/'train/bert_train_1.npz'
        with np.load(path) as z:data={k:z[k] for k in z.files}
        data['config_runtime']*=100;np.savez_compressed(path,**data)
        changed=plan.manifest(self.root,'layout:xla:default',cfg,0)
        self.assertEqual([r['indices'] for r in manifests[0]['records']],[r['indices'] for r in changed['records']])
        self.assertNotEqual(manifests[0],changed)

    def test_strict_sampling_and_connected_control(self):
        path=self.root/'graph.npz';synthetic(path)
        context=engine.graph_context(path,self.cfg);local=engine.layout_context(path,context,self.cfg);samples=[]
        for n in [8,16]:
            cfg=plan.effective_config(self.cfg,plan.gnn_exp('x','G_full',n),'layout:xla:default')
            graph=gnn.compact_graph(context,local,cfg);samples.append(graph)
            self.assertLessEqual(len(graph['gnn_numeric']),n);self.assertLessEqual(len(graph['gnn_src']),n*8)
            self.assertTrue(np.isfinite(graph['gnn_numeric']).all())
        self.assertTrue(set(samples[0]['gnn_positions'])<set(samples[1]['gnn_positions']))
        cfg=plan.effective_config(self.cfg,plan.gnn_exp('x','G_full',16,sampler='connected'),'layout:xla:default')
        graph=gnn.compact_graph(context,local,cfg)
        self.assertLessEqual(len(graph['gnn_numeric']),16)
        self.assertTrue(len(graph['gnn_src'])>0)
        self.assertEqual(len(graph['gnn_positions']),12)

    def test_safe_archive_restore_and_disk_guard(self):
        output=self.root/'out';output.mkdir();(output/'result.json').write_text('{"ok":true}')
        archive=controller.make_bundle(output,0)
        restored=self.root/'restored';controller.restore_archive(archive,restored,1)
        self.assertEqual((restored/'result.json').read_text(),'{"ok":true}')
        with self.assertRaises(ValueError):controller.restore_archive(archive,restored,1)
        evil=self.root/'evil.zip'
        with zipfile.ZipFile(evil,'w') as z:z.writestr('../escape.txt','bad')
        with self.assertRaises(ValueError):controller.restore_archive(evil,self.root/'unsafe',1)
        with self.assertRaises(OSError):controller.make_bundle(output,1000000)
        self.assertTrue(archive.exists())

    def test_development_selection_never_blends_other_fold_models(self):
        directory=self.root/'selection';directory.mkdir();cfg=dict(self.cfg,seeds=[42],folds=[0,1])
        # Each fold's dev graphs are deliberately different. Cross-fold ID alignment would fail.
        for fold in cfg['folds']:
            for name,pred in [('B0_basic_hgb',[1,2,3]),('T_candidate',[3,2,1])]:
                d=controller.trial_dir(directory,fold,name,42);d.mkdir(parents=True)
                engine.write_json(d/'status.json',dict(status='ok'));engine.write_json(d/'result.json',dict(status='ok'))
                pd.DataFrame(dict(graph=[f'graph_{fold}']*3,config_id=[0,1,2],runtime=[1,2,3],prediction=pred)).to_csv(d/'dev_predictions.csv.gz',index=False)
        lock=controller.lock_selection(directory,[],cfg,'layout:xla:default',['B0_basic_hgb','T_candidate'])
        self.assertEqual(lock['components'],{'B0_basic_hgb':1.})
        self.assertEqual(lock['fold_scores'],[1.,1.])
        # A fabricated better holdout file cannot change the existing dev lock.
        engine.write_json(directory/'holdout.json',dict(winner='T_candidate'))
        self.assertEqual(controller.lock_selection(directory,[],cfg,'layout:xla:default',[])['components'],{'B0_basic_hgb':1.})

    def test_factor_contrasts_match_graphs_and_seeds(self):
        values={'G_full_n2048':.2,'G_data_gall_c64':.3,'G_data_g24_c256':.25,
                'G_data_gall_c256':.4,'H_matched_gall_c256':.35}
        rows=[dict(experiment=n,graph=g,seed=s,score=v+offset)
              for n,v in values.items() for g,offset in [('a',0),('b',.1)] for s in self.cfg['seeds']]
        table=controller.factor_contrasts(pd.DataFrame(rows),dict(self.cfg,bootstrap_repeats=20),
                                         {'profile':plan.profile(2048)},self.cfg['seeds'])
        graphs=table[(table.reference=='G_full_n2048')&(table.experiment=='G_data_gall_c64')].iloc[0]
        self.assertEqual(graphs.factor,'training graphs');self.assertAlmostEqual(graphs.mean_paired_gain,.1)
        matched=table[table.factor=='GNN versus matched HGB'].iloc[0]
        self.assertAlmostEqual(matched.mean_paired_gain,.05);self.assertEqual(matched.graphs,2)
        endpoint=table[(table.reference=='G_data_gall_c64')&(table.experiment=='G_data_gall_c256')].iloc[0]
        self.assertEqual(endpoint.factor,'training configurations')

    def test_live_checkpoint_heartbeat_and_running_resume(self):
        worker=self.root/'worker.py';worker.write_text('import time\ntime.sleep(1)\n')
        status=self.root/'status.json';engine.write_json(status,{'status':'running'})
        cfg=dict(self.cfg,cpu_threads=1,disk_reserve_gib=0,system_reserve_gib=0)
        beats=[]
        ok=common.execute({'action':'trial'},worker,cfg,time.monotonic()+30,status,
                          heartbeat=lambda:beats.append(json.loads(status.read_text())['status']))
        self.assertTrue(ok);self.assertTrue(beats);self.assertEqual(set(beats),{'running'})
        self.assertEqual(json.loads(status.read_text())['status'],'ok')
        # A successful restored worker is skipped without another heartbeat or process.
        self.assertTrue(common.execute({'action':'trial'},worker,cfg,time.monotonic()+30,status,
                                        heartbeat=lambda:self.fail('Finished worker was rerun')))

    def test_complete_pipeline_and_restore(self):
        collections=['layout:xla:default','tile:xla'];data_root=self.root/'data'
        template=[]
        for collection in collections:
            for split,number in [('train',10),('valid',2),('test',1)]:
                for i in range(number):synthetic(data_root.joinpath(*collection.split(':'),split,f'bert_{split}_{i}.npz'),collection.startswith('tile'),i+1)
            template.append(dict(ID=collection+':bert_test_0',TopConfigs=';'.join(map(str,range(5 if collection.startswith('tile') else 16)))))
        template_path=self.root/'sample_submission.csv';pd.DataFrame(template).to_csv(template_path,index=False)
        runtime=self.root/'runtime';runtime.mkdir()
        for name in ['engine','gnn','common','plan','submission','controller']:shutil.copy2(HERE/(name+'.py'),runtime/('study_'+name+'.py'))
        cfg=dict(self.cfg,collections=collections,gnn_collections=[collections[0]],folds=[0,1],seeds=[42],
                 graph_budgets=[8,16],data_graph_counts=[24,0],data_config_counts=[64,128],
                 dev_graphs=2,holdout_graphs=2,eval_configs=8,train_pool_configs=384,
                 fit_configs=8,large_fit_configs=12,layout_train_graphs=4,tile_train_graphs=4,
                 layout_more_graphs=8,tile_more_graphs=8,gnn_epochs=2,gnn_patience=2,gnn_batch=4,
                 gnn_early_configs=4,gnn_predict_batch=4,hgb_iterations=2,extra_trees=3,
                 bootstrap_repeats=10,cpu_threads=1,session_hours=.5,disk_reserve_gib=.01,
                 system_reserve_gib=.1,probe_ram_gib=2.,sample_submission=str(template_path))
        callbacks=[];output=self.root/'results';scratch=self.root/'scratch'
        archive=controller.run_study(data_root,output,scratch,runtime/'study_engine.py',cfg,on_checkpoint=lambda p:callbacks.append(p.stat().st_size))
        self.assertTrue(archive.exists());self.assertGreaterEqual(len(callbacks),3)
        self.assertEqual(json.loads((output/'submission_status.json').read_text())['status'],'complete')
        for collection in collections:
            directory=output/collection.replace(':','_')
            self.assertTrue((directory/'complete.json').exists())
            self.assertFalse((directory/'summary_error.json').exists())
            self.assertTrue((directory/'system_holdout_gains.csv').exists())
            self.assertTrue((directory/'dev_screening_factor_contrasts.csv').exists())
            self.assertTrue((directory/'holdout_confirmed_factor_contrasts.csv').exists())
        self.assertEqual(list(scratch.iterdir()),[scratch/'.v3_owned'])
        for kind in ['baseline','tabular','selected']:
            f=pd.read_csv(output/f'submission_{kind}.csv');self.assertEqual(f.ID.tolist(),[r['ID'] for r in template])
        directory=output/'layout_xla_default';lock=json.loads((directory/'coverage_lock.json').read_text());n=lock['profile']['nodes']
        # Force a selected-profile GNN at raw inference even if the tiny dataset picks HGB.
        job=dict(cfg=cfg,collection=collections[0],path=str(data_root/'layout/xla/default/test/bert_test_0.npz'),
                 directory=str(directory),lock=dict(winner='T_full_small',components={f'G_full_n{n}':1.}),
                 id=template[0]['ID'],template_values=list(range(16)),output=str(self.root/'inference'))
        submission.infer_graph(job);first=json.loads((self.root/'inference/prediction.json').read_text())
        job['cfg']=dict(cfg,inference_batch=3,gnn_predict_batch=2);submission.infer_graph(job)
        second=json.loads((self.root/'inference/prediction.json').read_text());self.assertEqual(first['selected'],second['selected'])
        # Matched full-feature HGB and GNN use exactly the same graph/config IDs.
        g=controller.trial_dir(directory,0,f'G_full_n{n}',42)/'training_manifest.json'
        h=controller.trial_dir(directory,0,'H_matched_g24_c64',42)/'training_manifest.json'
        self.assertEqual(json.loads(g.read_text()),json.loads(h.read_text()))
        original={str(p.relative_to(output)):p.read_bytes() for p in output.glob('*/folds/*/trials/*/*/result.json')}
        restored=self.root/'restored';controller.restore_archive(archive,restored,12)
        controller.run_study(data_root,restored,self.root/'scratch2',runtime/'study_engine.py',cfg)
        self.assertEqual(original,{str(p.relative_to(restored)):p.read_bytes() for p in restored.glob('*/folds/*/trials/*/*/result.json')})
        with self.assertRaisesRegex(ValueError,'changed'):
            controller.run_study(data_root,restored,self.root/'scratch2',runtime/'study_engine.py',dict(cfg,gnn_epochs=3))

    def test_epoch_checkpoint_is_self_contained(self):
        collection='layout:xla:default';root=self.root/'data'
        for split,count in [('train',10),('valid',2)]:
            for i in range(count):synthetic(root.joinpath(*collection.split(':'),split,f'bert_{split}_{i}.npz'),seed=i+1)
        cfg=dict(self.cfg,dev_graphs=2,eval_configs=8,train_pool_configs=16,gnn_epochs=1,gnn_patience=3,
                 gnn_batch=4,gnn_predict_batch=4,gnn_early_configs=4,cpu_threads=1,gnn_nodes=8,gnn_config_nodes=6,gnn_edges=64)
        m=plan.manifest(root,collection,cfg,0);scratch=self.root/'features';out=self.root/'out'
        engine.prepare(root,m,cfg,scratch,self.root/'prep')
        exp=plan.gnn_exp('G_reference_n8','G0_reference',8)
        job=dict(cfg=cfg,experiment=exp,seed=42,collection=collection,scratch=str(scratch),output=str(out))
        gnn.train(job)
        # Only latest.pt survives from the active trial in this simulated recovery.
        recovered=self.root/'recovered';recovered.mkdir();shutil.copy2(out/'latest.pt',recovered/'latest.pt')
        job['cfg']=dict(cfg,gnn_epochs=2);job['output']=str(recovered);gnn.train(job)
        job['output']=str(self.root/'fresh');gnn.train(job)
        a=gnn.load_model(recovered/'model.pt');b=gnn.load_model(self.root/'fresh/model.pt')
        for key,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[key],rtol=0,atol=0)

if __name__=='__main__':unittest.main(verbosity=2)
