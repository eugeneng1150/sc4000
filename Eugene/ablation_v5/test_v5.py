"""Synthetic real fits plus scheduler/resume/selection contracts; no competition training."""
import ast
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
HERE=Path(__file__).resolve().parent
RUNTIME=Path(tempfile.mkdtemp(prefix='sc4000-v5-runtime-'))
for name in ['engine','gnn','common','plan','submission','controller']:
    shutil.copy2(HERE.parent/'ablation_v3'/(name+'.py'),RUNTIME/('study_'+name+'.py'))
for name in ['v4_core','v4_worker']:
    shutil.copy2(HERE.parent/'ablation_v4'/(name+'.py'),RUNTIME/(name+'.py'))
for p in HERE.glob('v5_*.py'):
    shutil.copy2(p,RUNTIME/p.name)
sys.path.insert(0,str(RUNTIME))
import numpy as np
import pandas as pd
import study_engine as engine
import study_plan as plan
import v5_core as core
import v5_worker as worker
import v5_scheduler as scheduling
import v5_controller as controller
from v4_core import read,restore,sha


def synthetic(p,seed,tile=False):
    rng=np.random.default_rng(seed);n,c=12,20
    p.parent.mkdir(parents=True,exist_ok=True)
    data=dict(node_feat=rng.uniform(0,4,(n,140)).astype(np.float32),
        node_opcode=np.arange(n)%7,edge_index=np.column_stack([np.arange(1,n),np.arange(n-1)]))
    if tile:
        x=rng.uniform(0,5,(c,24)).astype(np.float32)
        data.update(config_feat=x,config_runtime=2+x[:,0],config_runtime_normalizers=np.ones(c))
    else:
        x=rng.integers(-1,6,(c,n,18)).astype(np.float32)
        data.update(node_config_ids=np.arange(n),node_config_feat=x,config_runtime=20+x[:,0,0]+.1*x[:,1,0])
    np.savez_compressed(p,**data)


def reference(root,ref):
    c='layout:nlp:random';key=c.replace(':','_')
    cfg=plan.default_config()
    cfg.update(collections=[c],gnn_collections=[],folds=[0,1],seeds=[42,137,2026],
        dev_graphs=2,holdout_graphs=2,layout_train_graphs=4,layout_more_graphs=4,
        train_pool_configs=16,fit_configs=12,eval_configs=16,cpu_threads=1,hgb_iterations=6,
        bootstrap_repeats=20,system_reserve_gib=.01,disk_reserve_gib=.01)
    for split,count in [('train',9),('valid',2),('test',1)]:
        for i in range(count):
            synthetic(root/'layout/nlp/random'/split/f'{split}_{i}.npz',i+len(split)*20)
    synthetic(root/'tile/xla/test/tile_test.npz',92,True)
    tests=[]
    for collection,stem,values in [(c,'test_0',list(range(20))),('tile:xla','tile_test',list(range(5)))]:
        path=root.joinpath(*collection.split(':'),'test',stem+'.npz')
        with zipfile.ZipFile(path) as z:
            fp=[(i.filename,i.CRC,i.file_size) for i in z.infolist()]
        tests.append(dict(id=collection+':'+stem,collection=collection,
            relative_path=str(path.relative_to(root)),template_values=values,fingerprint=fp))
    template=root/'sample_submission.csv'
    pd.DataFrame([dict(ID=r['id'],TopConfigs=';'.join(map(str,r['template_values']))) for r in tests]).to_csv(template,index=False)
    cfg['sample_submission']=str(template)
    ref3=ref/'reference_v3';d=ref3/key;d.mkdir(parents=True)
    exp=dict(next(e for e in engine.experiments() if e['name']=='P_full'),
             name='T_full_small',configs=12,min_leaf=2,train_graphs=0)
    exp.pop('iterations',None)  # Real XLA default reference inherits cfg['hgb_iterations'].
    engine.write_json(d/'experiment_plan.json',[exp])
    for fold in cfg['folds']:
        m=plan.manifest(root,c,cfg,fold)
        engine.write_json(d/'manifests'/f'fold_{fold}.json',m)
        scratch=root.parent/f'ref_scratch_{fold}';prep=root.parent/f'ref_prep_{fold}';prep.mkdir()
        engine.prepare(root,m,dict(cfg,gnn_collections=[]),scratch,prep)
        for seed in cfg['seeds']:
            p=d/'folds'/str(fold)/'trials'/exp['name']/str(seed)
            job=dict(cfg=plan.effective_config(cfg,exp,c),experiment=exp,seed=seed,collection=c,
                     scratch=str(scratch),output=str(p))
            engine.run_trial(job);engine.evaluate_holdout(job)
            for fn in ['status.json','holdout_status.json']:
                engine.write_json(p/fn,dict(status='ok',peak_rss_gib=.1,wall_seconds=.1))
    lock=dict(components={exp['name']:1.},winner=exp['name'],gnn=None)
    engine.write_json(d/'selection_lock.json',lock)
    engine.write_json(d/'complete.json',dict(status='complete'))
    versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','joblib','psutil']}
    engine.write_json(ref3/'run_identity.json',dict(version='3.0.0',versions=versions))
    engine.write_json(ref3/'run_state.json',dict(config=cfg))
    engine.write_json(ref3/'submission_status.json',dict(status='complete'))
    engine.write_json(ref3/'test_manifest.json',dict(tests=tests))
    engine.write_json(ref/'run_identity.json',dict(version='4.0.0',versions=versions))
    engine.write_json(ref/'run_state.json',dict(config=cfg,status='complete'))
    engine.write_json(ref/'submission_status.json',dict(status='complete',rows=2))
    engine.write_json(ref/key/'selection_lock.json',dict(components=lock['components'],changed=False))
    engine.write_json(ref/key/'complete.json',dict(status='complete'))
    prediction=root.parent/'ref_prediction';prediction.mkdir()
    worker.run(dict(action='inference',cfg=cfg,collection=c,path=str(root/tests[0]['relative_path']),
        id=tests[0]['id'],template_values=list(range(20)),output=str(prediction),
        components=lock['components'],sources={exp['name']:dict(directory=str(d),experiment=exp)}))
    frame=pd.read_csv(template,dtype=str)
    frame.loc[0,'TopConfigs']=';'.join(map(str,read(prediction/'prediction.json')['selected']))
    frame.to_csv(ref/'submission_selected.csv',index=False)
    frame.to_csv(ref3/'submission_selected.csv',index=False)
    return cfg


class V5Tests(unittest.TestCase):
    def test_real_ranker_direction_and_grouping(self):
        rng=np.random.default_rng(42)
        x=rng.uniform(0,1,(90,3)).astype(np.float32)
        y=x[:,0]*5
        groups=np.repeat(np.arange(3),30)
        cfg=dict(xgb_depth=3,xgb_learning_rate=.2,xgb_max_bin=32,xgb_pairs=4,cpu_threads=1)
        exp=dict(model='xgb_rank',iterations=25)
        bundle=worker.fit_model(x,y,np.ones(90),groups,exp,42,cfg)
        prediction=worker.predict(bundle,x)
        self.assertGreater(engine.scores(y,prediction)['score'],.8)
        with self.assertRaises(ValueError):
            worker.fit_model(x,y,np.ones(90),groups[::-1],exp,42,cfg)

    def test_admission_reserves_unallocated_memory_and_cpu(self):
        self.assertTrue(scheduling.can_start(2.,[dict(estimate=2.,rss=.5)],6.,2.,2,1,2))
        self.assertFalse(scheduling.can_start(3.,[dict(estimate=3.,rss=.5)],6.,2.,2,1,2))
        self.assertFalse(scheduling.can_start(1.,[dict(estimate=1.,rss=.5)],8.,2.,1,1,2))

    def scheduler_fixture(self,base):
        output=base/'out';output.mkdir()
        scratch=base/'scratch';scratch.mkdir()
        p=base/'worker.py'
        p.write_text('import argparse,json,time\nfrom pathlib import Path\n'
            'p=argparse.ArgumentParser();p.add_argument("--job");a=p.parse_args()\n'
            'j=json.loads(Path(a.job).read_text());time.sleep(.35)\n'
            'Path(j["output"],"done.json").write_text(json.dumps({"ok":True}))\n')
        cfg=dict(cpu_threads=1,max_parallel=2,worker_ram_gib=1.,system_reserve_gib=.01,
                 extraction_minutes=1,inference_minutes=1,job_minutes=1,retry_failed=False,
                 monitor_seconds=.02,scratch_limit_gib=1.,output_limit_gib=1.,disk_reserve_gib=.01)
        s=scheduling.Scheduler(output,scratch,p,cfg,time.monotonic()+30)
        s.cpus=2
        tasks=[dict(job=dict(action='trial',cfg=cfg,label=f'job {i}'),
                    status=output/f'j{i}'/'status.json',estimate=.05) for i in range(2)]
        return s,tasks

    def test_actual_parallel_jobs_and_pressure_fallback(self):
        with tempfile.TemporaryDirectory() as t:
            s,tasks=self.scheduler_fixture(Path(t))
            # Force one low-memory observation while two processes are active.
            pressure=[False]
            def memory():
                if len(s.active)==2 and all(a['rss']>0 for a in s.active) and not pressure[0]:
                    pressure[0]=True
                    return 0.
                return 8.
            with patch.object(scheduling,'available_gib',side_effect=memory):
                s.batch(tasks)
            events=[json.loads(x) for x in (s.output/'scheduler_events.jsonl').read_text().splitlines()]
            self.assertTrue(any(e['event']=='start' and e['workers']==2 for e in events))
            self.assertTrue(any(e['event']=='sequential_fallback' for e in events))
            self.assertTrue(all(read(x['status'])['status']=='ok' for x in tasks))
            before=len(events);s.batch(tasks)
            self.assertEqual(len((s.output/'scheduler_events.jsonl').read_text().splitlines()),before)

    def test_interruption_checkpoint_excludes_live_artifacts(self):
        with tempfile.TemporaryDirectory() as t:
            s,tasks=self.scheduler_fixture(Path(t))
            def interrupt():
                if s.active:
                    archive=scheduling.snapshot(s.output,.001)
                    with zipfile.ZipFile(archive) as z:
                        self.assertFalse(any('.inflight/' in n for n in z.namelist()))
                    raise KeyboardInterrupt()
            s.heartbeat=interrupt
            with self.assertRaises(KeyboardInterrupt):
                s.batch(tasks)
            self.assertFalse(s.active)
            self.assertEqual(read(tasks[0]['status'])['status'],'interrupted')
            s.heartbeat=None;s.batch(tasks)
            self.assertTrue(all(read(x['status'])['status']=='ok' for x in tasks))

    def test_finalist_control_and_fold_guard(self):
        screen={'candidates':[dict(components={'V5_X_pairwise':.5,'G':.5},
            eligible=True,mean_score=.8)]}
        self.assertEqual(set(core.finalists(screen,['V5_X_pairwise','V5_X_regression'])),
                         {'V5_X_pairwise','V5_X_regression'})
        with tempfile.TemporaryDirectory() as t:
            ref=Path(t)/'ref';d=ref/'layout_nlp_random'
            engine.write_json(d/'selection_lock.json',dict(components={'T':1.}))
            def frame(x):return pd.DataFrame(dict(graph=['g'],config_id=[0],runtime=[1.],prediction=[x]))
            frames={0:{'T':frame(.5),'V5_H_percentile':frame(.9)},
                    1:{'T':frame(.5),'V5_H_percentile':frame(.4)}}
            cfg=dict(folds=[0,1],fold_regression_tolerance=0.,minimum_gain_layout=.002)
            with patch.object(core,'prediction_frames',return_value=frames),patch.object(core,'tabular_reference',return_value=('T',{})),patch.object(core,'frame_score',side_effect=lambda f,t:float(f.prediction.mean())):
                result=core.choose(ref,Path(t)/'out','layout:nlp:random',['V5_H_percentile'],cfg,[42])
            self.assertFalse(result['changed'])
            self.assertTrue(any(r['gain_vs_v4']>0 and not r['eligible'] for r in result['candidates']))

    def test_end_to_end_fits_export_and_restored_resume(self):
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);root=t/'data';output=t/'out';ref=output/'reference_v4'
            original=reference(root,ref)
            cfg=core.defaults(ref)
            cfg.update(collections=original['collections'],submission_collections=[*original['collections'],'tile:xla'],
                xgb_iterations=8,xgb_depth=2,xgb_max_bin=16,pilot_iterations=[2,4],
                minimum_gain_layout=-2.,fold_regression_tolerance=2., # exercise accepted/export path, NOT default protocol
                session_hours=.5,worker_ram_gib=3.,system_reserve_gib=.01,disk_reserve_gib=.01,
                pilot_memory_multiplier=1.1,pilot_memory_add_gib=.1,monitor_seconds=.02)
            archive=controller.run(root,output,t/'scratch',RUNTIME,cfg,enforce_versions=False)
            self.assertEqual(read(output/'run_state.json')['status'],'complete')
            self.assertEqual(read(output/'submission_status.json')['rows'],2)
            current=pd.read_csv(output/'submission_selected.csv',dtype=str)
            old=pd.read_csv(ref/'submission_selected.csv',dtype=str)
            self.assertEqual(current.iloc[1].TopConfigs,old.iloc[1].TopConfigs)
            self.assertEqual(sorted(map(int,current.iloc[0].TopConfigs.split(';'))),list(range(20)))
            all_jobs=list(output.glob('*/folds/*/trials/*/*/status.job.json'))
            self.assertTrue(all(read(p)['experiment']['model']!='gnn' for p in all_jobs))
            results={str(p.relative_to(output)):sha(p) for p in output.glob('*/folds/*/trials/*/*/result.json')}
            restored=t/'restored'
            restore(archive,restored)
            identity=read(restored/'run_identity.json')
            identity['sources'].update(core.ESTIMATOR_FIX_PREVIOUS_SOURCES)
            identity['signature']=engine.digest({k:identity[k] for k in
                ['config','versions','sources','reference_digest']})
            engine.write_json(restored/'run_identity.json',identity)
            controller.run(root,restored,t/'scratch2',RUNTIME,cfg,enforce_versions=False)
            self.assertEqual(results,{str(p.relative_to(restored)):sha(p) for p in restored.glob('*/folds/*/trials/*/*/result.json')})
            self.assertTrue((restored/'estimator_fix_migration.json').exists())
            # Old absolute paths recorded in job logs are not reused on restore.
            self.assertEqual((output/'submission_selected.csv').read_bytes(),(restored/'submission_selected.csv').read_bytes())
            self.assertTrue((restored/'report/01_target_and_learner_screening.png').exists())

    def test_estimator_upgrade_rejects_scientific_changes(self):
        from copy import deepcopy
        value=dict(config={'hgb_iterations':180},versions={'numpy':'fixed'},
                   reference_digest='reference',sources=dict(core.ESTIMATOR_FIX_PREVIOUS_SOURCES,
                   **{'v5_worker.py':'unchanged-worker'}))
        previous=dict(value,version=core.VERSION,signature=engine.digest(value))
        current=deepcopy(value)
        current['sources'].update({'v5_core.py':'new-core','v5_controller.py':'new-controller'})
        self.assertTrue(core.estimator_fix_compatible(previous,current))
        for key,changed in [('config',{'hgb_iterations':300}),('versions',{'numpy':'other'}),
                            ('reference_digest','different-reference'),
                            ('sources',dict(current['sources'],**{'v5_worker.py':'different-worker'}))]:
            with self.subTest(key=key):
                self.assertFalse(core.estimator_fix_compatible(previous,dict(current,**{key:changed})))
        self.assertFalse(core.estimator_fix_compatible(dict(previous,signature='invalid'),current))

    def test_notebook_syntax_and_embedded_sources(self):
        import nbformat
        notebook=json.loads((HERE.parent/'SC4000_V5_Ranking_and_Targets.ipynb').read_text())
        nbformat.validate(nbformat.from_dict(notebook))
        texts=[''.join(c['source']) for c in notebook['cells']]
        for c in notebook['cells']:
            if c['cell_type']=='code':ast.parse(''.join(c['source']))
        for p in HERE.glob('v5_*.py'):
            self.assertTrue(any(repr(p.read_text()) in text for text in texts),p.name)


if __name__=='__main__':
    try:
        unittest.main(verbosity=2)
    finally:
        shutil.rmtree(RUNTIME,ignore_errors=True)
