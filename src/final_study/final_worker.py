"""Isolated train/evaluate/predict worker for the final reproducible workflow."""
import argparse
import json
from pathlib import Path
import traceback
import numpy as np
import study_engine as engine
from threadpoolctl import threadpool_limits


def training_rows(records, exp, seed, cfg):
    gnn = exp['model']=='gnn'
    selector = dict(matched=True, graph_fraction=1.) if gnn else exp
    return [dict(graph=r['graph'],config_ids=np.asarray(r['indices'])[
        engine.training_indices(len(r['indices']),r['graph'],seed,cfg,
            matched=True if gnn else exp.get('matched',False),
            fraction=1. if gnn else exp['row_fraction'])].tolist())
        for r in engine.training_records(records,selector,seed,cfg)]


def canonical(rows):
    return sorted((r['graph'],tuple(r['config_ids'])) for r in rows)


def run(job):
    from v5_worker import install
    install()  # Adds optional XGBoost learners; HGB/GNN recipes are unchanged.
    action=job['action'];out=Path(job['output']);out.mkdir(parents=True,exist_ok=True)
    if action=='prepare':
        engine.prepare(job['root'],job['manifest'],dict(job['cfg'],max_excluded_fraction=0.),job['scratch'],out)
        return
    if action=='inference':
        from v4_worker import inference
        inference(job)
        return
    records=json.loads((Path(job['scratch'])/'prepared.json').read_text())
    exp,cfg,seed=job['experiment'],job['cfg'],job['seed']
    expected=json.loads(Path(job['expected_training']).read_text())
    if canonical(training_rows(records,exp,seed,cfg))!=canonical(expected):
        raise ValueError('Training graph/configuration IDs differ from the recorded recipe')
    if action=='trial':
        engine.run_trial(job)
        actual=json.loads((out/'training_manifest.json').read_text())
        if canonical(actual)!=canonical(expected):raise ValueError('Fitted training IDs differ')
    elif action=='audit':
        directory=Path(job['model_directory'])
        with threadpool_limits(limits=cfg['cpu_threads']):
            if exp['model']=='gnn':
                import torch
                from study_gnn import load_model,evaluate
                torch.set_num_threads(cfg['cpu_threads'])
                model=load_model(directory/'model.pt')
                evaluator=lambda split:evaluate(model,job,records,split)
            else:
                import joblib
                bundle=joblib.load(directory/'model.joblib')
                evaluator=lambda split:engine.evaluate(bundle,job['scratch'],records,split,
                    job['collection'],out,seed,exp['name'])
            values={split:evaluator(split) for split in ['dev','holdout']}
        engine.write_json(out/'audit_result.json',dict(status='ok',scores=values))
    else:raise ValueError(action)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args()
    job=json.loads(Path(a.job).read_text())
    try:run(job)
    except Exception as exc:
        engine.write_json(Path(job['output'])/'worker_failure.json',dict(
            kind='resource' if isinstance(exc,MemoryError) else 'error',error=str(exc)))
        traceback.print_exc();raise SystemExit(1)
