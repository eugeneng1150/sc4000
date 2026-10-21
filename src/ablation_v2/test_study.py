"""Synthetic validation of the actual embedded study pipeline; no Kaggle data needed.

Run: /opt/anaconda3/bin/python src/ablation_v2/test_study.py
"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import warnings

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


engine = load('study_engine', HERE / 'engine.py')
controller = load('study_controller', HERE / 'controller.py')
gnn = load('study_gnn', HERE / 'gnn.py')
submission = load('study_submission', HERE / 'submission.py')


def synthetic(path, tile=False, seed=1):
    rng = np.random.default_rng(seed)
    n, c = 8, 24
    # Consumer -> producer, the official NPZ convention.
    data = dict(node_feat=rng.uniform(0, 4, (n, 140)).astype(np.float32),
                node_opcode=np.array([1, 2, 3, 4, 1, 2, 3, 4], dtype=np.int32),
                edge_index=np.column_stack([np.arange(1,n), np.arange(n-1)]))
    if tile:
        configs = rng.uniform(0, 10, (c,24)).astype(np.float32)
        normalizer = np.linspace(1, 2, c)
        runtime = (2 + configs[:,0]) * normalizer
        data.update(config_feat=configs, config_runtime_normalizers=normalizer)
    else:
        configs = rng.integers(-1, 6, (c,n,18)).astype(np.float32)
        configs[:, :, 4:6] = -1
        configs[1] = configs[0]  # repeated config, slightly different measurement
        runtime = 10 + configs[:,0,0] + .2 * configs[:,1,1] + rng.uniform(0,.1,c)
        data.update(node_config_feat=configs, node_config_ids=np.arange(n))
    data['config_runtime'] = runtime
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    return data


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = dict(controller.default_config(),gnn_collections=[],generate_submissions=False,optional_screening=False)

    def tearDown(self):
        self.temp.cleanup()

    def test_streaming_rows_and_fortran_rejection(self):
        path = self.root / 'array.npz'
        source = np.arange(200*3*18, dtype=np.float32).reshape(200,3,18)
        np.savez_compressed(path, node_config_feat=source)
        ids = [0,4,99,199]
        got = list(engine.selected_rows(path, 'node_config_feat', ids, chunk_bytes=32))
        np.testing.assert_array_equal(np.stack([a for _,a in got]), source[ids])
        with self.assertRaises(ValueError):
            list(engine.selected_rows(path, 'node_config_feat', [2,1]))
        np.savez_compressed(path, node_config_feat=np.asfortranarray(source))
        with self.assertRaisesRegex(ValueError, 'Fortran'):
            list(engine.selected_rows(path, 'node_config_feat', [1]))

    def test_operation_assignment_and_padding(self):
        path = self.root / 'layout.npz'
        data = synthetic(path)
        context = engine.graph_context(path, self.cfg)
        local = engine.layout_context(path, context, self.cfg)
        # Swapping whole rows keeps pooled values identical, but changes operation association.
        first = data['node_config_feat'][2].copy()
        second = first.copy()
        second[[0,1]] = second[[1,0]]
        a = engine.layout_features(first, context, local, self.cfg)
        b = engine.layout_features(second, context, local, self.cfg)
        np.testing.assert_allclose(a['basic'], b['basic'])
        np.testing.assert_allclose(a['position'], b['position'])
        self.assertGreater(np.abs(a['opcode']-b['opcode']).sum(), 0)
        # Correct producer-to-consumer depth, independent of edge-list convention.
        np.testing.assert_array_equal(context['depth'], np.arange(8))
        empty = engine.layout_features(np.full((8,18), -1), context, local, self.cfg)
        self.assertTrue(all(np.isfinite(v).all() for v in empty.values()))
        self.assertTrue(np.all(empty['position'].reshape(18,7)[:,0] == 1))

    def test_tile_normalization_and_metrics(self):
        path = self.root / 'tile.npz'
        data = synthetic(path, tile=True)
        record = dict(relative_path='tile.npz', indices=[0,2,5], count=24)
        engine.extract_graph(self.root, record, 'tile:xla', self.cfg, self.root/'compact.npz')
        with np.load(self.root/'compact.npz') as compact:
            np.testing.assert_allclose(compact['y'],
                (data['config_runtime']/data['config_runtime_normalizers'])[[0,2,5]])
        self.assertEqual(engine.scores(np.arange(1.,8.), -np.arange(1.,8.))['score'], -1.)
        self.assertEqual(engine.scores(np.arange(1.,8.), np.zeros(7))['score'], 0.)
        self.assertEqual(engine.scores(np.arange(1.,8.), np.arange(1.,8.), True)['score'], 1.)

    def test_manifest_is_label_independent_and_disjoint(self):
        folder = self.root/'layout/xla/default'
        for split, number in [('train',9),('valid',3)]:
            for i in range(number):
                synthetic(folder/split/f'bert_{split}_{i}.npz', seed=i+1)
        m = engine.make_manifest(self.root, 'layout:xla:default', self.cfg)
        sets = {s: {r['graph'] for r in m['records'] if r['split']==s} for s in ['train','dev','holdout']}
        self.assertFalse(sets['train'] & sets['dev'])
        self.assertFalse((sets['train']|sets['dev']) & sets['holdout'])
        path = folder/'train/bert_train_1.npz'
        with np.load(path) as z:
            arrays = {k:z[k] for k in z.files}
        arrays['config_runtime'] *= 100
        np.savez_compressed(path, **arrays)
        changed = engine.make_manifest(self.root, 'layout:xla:default', self.cfg)
        self.assertEqual([r['indices'] for r in m['records']], [r['indices'] for r in changed['records']])
        self.assertNotEqual(m, changed)  # stale result detection includes labels' CRC

    def test_ablation_catalog_and_no_identifier_features(self):
        for tile in [False,True]:
            catalog = engine.experiments(tile)
            self.assertEqual(len({e['name'] for e in catalog}), len(catalog))
            full = next(e for e in catalog if e['name']=='P_full')
            for exp in catalog:
                self.assertNotIn('ids',exp['blocks'])
                if exp['name'].startswith('A_without_'):
                    removed = exp['name'].removeprefix('A_without_')
                    self.assertEqual(exp['blocks'], [b for b in full['blocks'] if b != removed])
        self.assertGreater(len(engine.experiments(False)), 20)

    def test_all_model_families(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(64,10)).astype(np.float32)
        y = x[:,0] + .1*x[:,1]
        groups = np.repeat(np.arange(4),16)
        cfg = dict(self.cfg, hgb_iterations=2, extra_trees=3, mlp_epochs=2, pairs_per_graph=16)
        with warnings.catch_warnings(), engine.threadpool_limits(limits=1):
            warnings.simplefilter('ignore')
            for kind in ['hgb','extra_trees','ridge','pairwise','mlp']:
                bundle = engine.fit_model(x,y,np.ones(64),groups,dict(model=kind),42,cfg)
                prediction = engine.predict(bundle,x)
                self.assertEqual(prediction.shape,(64,))
                self.assertTrue(np.isfinite(prediction).all())

    def test_resource_stop_and_checkpoint(self):
        worker = self.root/'slow.py'
        worker.write_text('import time\ntime.sleep(30)\n')
        status = self.root/'status.json'
        cfg = dict(self.cfg, job_minutes=.001, system_reserve_gib=.01, disk_reserve_gib=.01)
        ok = controller.execute(dict(action='trial'), worker, cfg, controller.time.monotonic()+10, status)
        self.assertFalse(ok)
        self.assertEqual(json.loads(status.read_text())['status'], 'timeout')
        self.assertTrue(status.with_suffix('.log').exists())

    def test_memory_watchdog_and_sampled_matrix_limit(self):
        worker = self.root/'allocate.py'
        worker.write_text('import time\nx=bytearray(80*1024**2)\ntime.sleep(15)\n')
        status = self.root/'memory_status.json'
        cfg = dict(self.cfg, worker_ram_gib=.02, system_reserve_gib=.01, disk_reserve_gib=.01)
        ok = controller.execute(dict(action='trial'), worker, cfg, controller.time.monotonic()+10, status)
        self.assertFalse(ok)
        self.assertEqual(json.loads(status.read_text())['status'], 'memory_guard')
        path = self.root/'layout.npz'
        synthetic(path)
        record = dict(relative_path='layout.npz', indices=[0,1], count=24)
        with self.assertRaisesRegex(MemoryError, 'Compact graph feature matrix'):
            engine.extract_graph(self.root,record,'layout:xla:default',
                                 dict(self.cfg,max_graph_features_mib=.00001),self.root/'unused.npz')

    def test_notebook_is_self_contained_and_matches_modules(self):
        import ast
        import nbformat
        notebook = nbformat.read(HERE.parents[1]/'notebooks/experiments'/'SC4000_Comprehensive_Ablations.ipynb',as_version=4)
        nbformat.validate(notebook)
        embedded = []
        for i,cell in enumerate(notebook.cells):
            if cell.cell_type != 'code':
                continue
            compile(cell.source, f'notebook-cell-{i}', 'exec')
            if cell.source.startswith('# Embedded independent implementation:'):
                name = cell.source.splitlines()[0].split(': ')[1]
                value = ast.literal_eval(ast.parse(cell.source).body[0].value)
                self.assertEqual(value,(HERE/name).read_text())
                embedded.append(name)
        self.assertEqual(embedded,['engine.py','gnn.py','submission.py','controller.py'])

    def test_complete_study_and_resume(self):
        data_root = self.root/'data'
        for collection in ['layout:xla:default','tile:xla']:
            for split, number in [('train',7),('valid',2)]:
                for i in range(number):
                    synthetic(data_root.joinpath(*collection.split(':'),split,f'bert_{split}_{i}.npz'),
                              tile=collection.startswith('tile'), seed=i+int(split=='valid')*100)
        runtime = self.root/'runtime'; runtime.mkdir()
        shutil.copy2(HERE/'engine.py',runtime/'study_engine.py')
        cfg = dict(self.cfg, collections=['layout:xla:default','tile:xla'], seeds=[42,137],
                   layout_train_graphs=4,tile_train_graphs=4, dev_graphs=2,holdout_graphs=2,
                   train_pool_configs=24,fit_configs=16,eval_configs=16,gnn_configs=8,
                   hgb_iterations=2,extra_trees=3,mlp_epochs=2,pairs_per_graph=8,
                   bootstrap_repeats=20,cpu_threads=1,session_hours=.5,
                   disk_reserve_gib=.01,system_reserve_gib=.1)
        out, scratch = self.root/'results', self.root/'scratch'
        bundle = controller.run_study(data_root,out,scratch,runtime/'study_engine.py',cfg)
        self.assertTrue(bundle.exists())
        self.assertEqual(list(scratch.iterdir()), [scratch/'.sc4000_owned_scratch'])
        mtimes = {str(p):p.stat().st_mtime_ns for p in out.glob('*/trials/*/*/result.json')}
        for collection in cfg['collections']:
            directory = out/collection.replace(':','_')
            self.assertTrue((directory/'complete.json').exists())
            lock = json.loads((directory/'selection_lock.json').read_text())
            self.assertIn('B0_basic_hgb',lock['confirmed'])
            gains = pd.read_csv(directory/'holdout_paired_gains.csv')
            self.assertEqual(gains[gains.experiment=='B0_basic_hgb'].mean_paired_gain.iloc[0],0.)
            all_trials = pd.read_csv(directory/'all_trials.csv')
            self.assertEqual(set(all_trials.status), {'ok'})
            self.assertTrue((directory/'locked_systems_holdout_scores.csv').exists())
        controller.run_study(data_root,out,scratch,runtime/'study_engine.py',cfg)
        self.assertEqual(mtimes, {str(p):p.stat().st_mtime_ns for p in out.glob('*/trials/*/*/result.json')})
        with self.assertRaisesRegex(ValueError, 'changed'):
            controller.run_study(data_root,out,scratch,runtime/'study_engine.py',dict(cfg,fit_configs=15))

    def test_gnn_bounds_capacity_and_message_ablation(self):
        import torch
        path=self.root/'graph.npz';data=synthetic(path)
        cfg=dict(self.cfg,gnn_nodes=5,gnn_config_nodes=4,gnn_edges=2)
        context=engine.graph_context(path,cfg);local=engine.layout_context(path,context,cfg)
        compact=gnn.compact_graph(context,local,cfg)
        self.assertLessEqual(len(compact['gnn_numeric']),5)
        self.assertEqual(len(compact['gnn_config_nodes']),4)
        self.assertLessEqual(len(compact['gnn_src']),2)
        for key,value in compact.items():
            np.testing.assert_array_equal(value,gnn.compact_graph(context,local,cfg)[key])
        graph=tuple(torch.from_numpy(compact[k]) for k in
                    ['gnn_numeric','gnn_opcode','gnn_src','gnn_dst','gnn_config_nodes'])
        layout=torch.from_numpy(data['node_config_feat'][:2,compact['gnn_positions']].astype(np.int8))
        summary=torch.rand(2,7)
        catalog={e['name']:e for e in gnn.catalog()}
        counts=[]
        for name in ['G0_reference','G_add_masks','G_add_context','G_full']:
            torch.manual_seed(9);model=gnn.Ranker(140,7,catalog[name]);model.eval()
            counts.append(sum(p.numel() for p in model.parameters()))
            model(graph,layout,summary).sum().backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertEqual(len(set(counts)),1)
        model=gnn.Ranker(140,7,catalog['G_no_messages']);model.eval()
        empty=(*graph[:2],torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long),graph[-1])
        torch.testing.assert_close(model(graph,layout,summary),model(empty,layout,summary))
        # Reference ignores the global summary; the context addition can use it.
        torch.manual_seed(2);reference=gnn.Ranker(140,7,catalog['G0_reference']).eval()
        torch.testing.assert_close(reference(graph,layout,summary),reference(graph,layout,summary*0))
        logits=torch.tensor([.3,-.4],requires_grad=True)
        good=gnn.loss_function(logits,torch.tensor([1.,-1.]),'pairwise')
        bad=gnn.loss_function(-logits,torch.tensor([1.,-1.]),'pairwise')
        self.assertLess(good.item(),bad.item())

    def test_full_gnn_pipeline_checkpoint_and_streamed_export(self):
        import torch
        collection='layout:xla:default'
        data_root=self.root/'data';folder=data_root/'layout/xla/default'
        for split,number in [('train',7),('valid',2),('test',2)]:
            for i in range(number):synthetic(folder/split/f'bert_{split}_{i}.npz',seed=i+1)
        template=self.root/'sample_submission.csv'
        pd.DataFrame([dict(ID=f'{collection}:bert_test_{i}',TopConfigs=';'.join(map(str,range(24))))
                      for i in range(2)]).to_csv(template,index=False)
        runtime=self.root/'runtime';runtime.mkdir()
        for name in ['engine','gnn','submission']:
            shutil.copy2(HERE/f'{name}.py',runtime/f'study_{name}.py')
        cfg=dict(self.cfg,collections=[collection],gnn_collections=[collection],seeds=[42],
                 layout_train_graphs=4,dev_graphs=2,holdout_graphs=2,train_pool_configs=24,
                 fit_configs=16,eval_configs=16,gnn_train_graphs=3,gnn_configs=8,
                 gnn_nodes=6,gnn_config_nodes=4,gnn_edges=4,gnn_batch=4,gnn_predict_batch=5,
                 gnn_epochs=2,gnn_early_configs=8,hgb_iterations=2,cpu_threads=1,
                 bootstrap_repeats=10,session_hours=.5,disk_reserve_gib=.01,system_reserve_gib=.1,
                 clean_scratch=False,generate_submissions=True,sample_submission=str(template))
        out=self.root/'results';scratch=self.root/'scratch'
        self.assertEqual(submission.preflight(data_root,cfg)['rows'],2)
        controller.run_study(data_root,out,scratch,runtime/'study_engine.py',cfg)
        directory=out/collection.replace(':','_');compact=scratch/directory.name
        self.assertTrue((directory/'complete.json').exists())
        self.assertEqual(json.loads((out/'submission_status.json').read_text())['status'],'complete')
        for name in ['baseline','selected']:
            frame=pd.read_csv(out/f'submission_{name}.csv')
            self.assertEqual(list(frame.ID),list(pd.read_csv(template).ID))
            for value in frame.TopConfigs:self.assertEqual(sorted(map(int,value.split(';'))),list(range(24)))
        for reference in ['G0_reference','G_full','B_matched_hgb']:
            self.assertTrue((directory/f'holdout_paired_vs_{reference}.csv').exists())
        records=json.loads((compact/'prepared.json').read_text())
        matched=next(e for e in controller.core_experiments(collection,cfg) if e['name']=='B_matched_hgb')
        train_records=engine.training_records(records,matched,42,cfg)
        actual=json.loads((directory/'trials/G0_reference/42/training_manifest.json').read_text())
        expected=[dict(graph=r['graph'],config_ids=np.asarray(r['indices'])[
                  engine.training_indices(len(r['indices']),r['graph'],42,cfg,True)].tolist()) for r in train_records]
        self.assertEqual(actual,expected)
        # Resume epoch 1 and prove it yields the same checkpoint as uninterrupted epoch 2.
        experiment=gnn.catalog()[0]
        job=dict(cfg=dict(cfg,gnn_epochs=1),scratch=str(compact),output=str(self.root/'resumed'),
                 experiment=experiment,seed=42,collection=collection)
        gnn.train(job);job['cfg']=cfg;gnn.train(job)
        fresh=gnn.load_model(directory/'trials/G0_reference/42/model.pt')
        resumed=gnn.load_model(self.root/'resumed/model.pt')
        for key,value in fresh.state_dict().items():torch.testing.assert_close(value,resumed.state_dict()[key],rtol=0,atol=0)
        # Force the GNN inference path even if dev selects the baseline in this tiny fixture.
        infer=dict(cfg=cfg,collection=collection,path=str(folder/'test/bert_test_0.npz'),
                   directory=str(directory),lock=dict(components={'G_full':1.}),
                   id=f'{collection}:bert_test_0',template_values=list(range(24)),output=str(self.root/'infer'))
        submission.infer_graph(infer)
        first=json.loads((self.root/'infer/prediction.json').read_text())
        infer['cfg']=dict(cfg,inference_batch=3,gnn_predict_batch=2)
        submission.infer_graph(infer)
        second=json.loads((self.root/'infer/prediction.json').read_text())
        self.assertEqual(first['selected'],second['selected'])
        self.assertEqual(first['baseline'],second['baseline'])
        # Compact evaluation and raw streamed inference use exactly the same features.
        source=folder/'test/bert_test_0.npz'
        record=dict(relative_path=str(source.relative_to(data_root)),indices=list(range(24)),count=24)
        engine.extract_graph(data_root,record,collection,cfg,self.root/'test_compact.npz')
        with np.load(self.root/'test_compact.npz') as data:
            graph,layout,summary=gnn.arrays(data)
            pred=gnn.predict_arrays(gnn.load_model(directory/'trials/G_full/42/model.pt'),graph,layout,summary,5)
        self.assertEqual(first['selected'],np.argsort(pred,kind='stable').tolist())
        mtimes={str(p):p.stat().st_mtime_ns for p in out.glob('*/trials/*/*/result.json')}
        controller.run_study(data_root,out,scratch,runtime/'study_engine.py',cfg)
        self.assertEqual(mtimes,{str(p):p.stat().st_mtime_ns for p in out.glob('*/trials/*/*/result.json')})

    def test_submission_validation_and_short_tile_contract(self):
        self.assertEqual(submission.ranking(np.array([2.,1.]),5,True,[0,1,2,3,4]),[1,0,2,3,4])
        # Some official layout templates request 1,000 of 1,001 available configurations.
        self.assertEqual(submission.ranking(np.array([3.,1.,2.]),2,False,[0,1]),[1,2])
        with self.assertRaises(ValueError):submission.ranking(np.array([2.,1.]),5,False,[0,1,2,3,4])
        with self.assertRaises(ValueError):submission.ranking(np.array([2.,1.]),5,True,[0,1,6,7,8])
        with self.assertRaises(ValueError):submission.validate_ranking([0,0],2,2,False,[0,1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
