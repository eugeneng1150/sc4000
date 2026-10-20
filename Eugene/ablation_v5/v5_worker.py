"""Isolated CPU workers. Never fit a GNN; reuse frozen feature construction."""
import argparse
import json
from pathlib import Path
import time
import traceback
import joblib
import numpy as np
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
import study_engine as engine
from study_engine import write_json

ORIGINAL_FIT = engine.fit_model
ORIGINAL_PREDICT = engine.predict


def fit_model(x, y, weights, groups, exp, seed, cfg):
    if exp['model'] not in ['xgb_reg', 'xgb_rank']:
        return ORIGINAL_FIT(x, y, weights, groups, exp, seed, cfg)
    from xgboost import XGBRegressor, XGBRanker
    args = dict(n_estimators=exp['iterations'], max_depth=cfg['xgb_depth'],
        learning_rate=cfg['xgb_learning_rate'], max_bin=cfg['xgb_max_bin'],
        tree_method='hist', device='cpu', n_jobs=cfg['cpu_threads'],
        subsample=1., colsample_bytree=1., min_child_weight=1.,
        reg_lambda=1., random_state=seed, verbosity=0)
    if exp['model'] == 'xgb_reg':
        model = XGBRegressor(objective='reg:squarederror', **args)
        model.fit(x, y, sample_weight=weights)
    else:
        if np.any(np.diff(groups) < 0):
            raise ValueError('Query IDs must be sorted')
        relevance = np.empty(len(y), np.float32)
        unique, starts, counts = np.unique(groups, return_index=True, return_counts=True)
        for start, count in zip(starts, counts):
            # Faster runtime => higher relevance. Equal runtimes keep equal labels.
            relevance[start:start+count] = 1-(rankdata(y[start:start+count])-1)/max(1,count-1)
        # One equal weight per query, not per candidate. Regression has graph-normalized rows.
        model = XGBRanker(objective='rank:pairwise', lambdarank_pair_method='mean',
            lambdarank_num_pair_per_sample=cfg['xgb_pairs'], lambdarank_normalization=True,
            disable_default_eval_metric=True, **args)
        model.fit(x, relevance, qid=groups, sample_weight=np.ones(len(unique)))
    return dict(model=model, scaler=None, experiment=exp)


def predict(bundle, x):
    value = ORIGINAL_PREDICT(bundle, x)
    # Entire existing pipeline expects smaller predictions to mean faster.
    return -value if bundle['experiment']['model'] == 'xgb_rank' else value


def install():
    engine.fit_model = fit_model
    engine.predict = predict


def verify_rows(job):
    records = json.loads((Path(job['scratch'])/'prepared.json').read_text())
    exp, cfg, seed = job['experiment'], job['cfg'], job['seed']
    actual = [dict(graph=r['graph'], config_ids=np.asarray(r['indices'])[
        engine.training_indices(len(r['indices']),r['graph'],seed,cfg,
                                exp.get('matched',False),exp['row_fraction'])].tolist())
        for r in engine.training_records(records, exp, seed, cfg)]
    expected = json.loads(Path(job['expected_training']).read_text())
    canonical = lambda rows: sorted((r['graph'], tuple(r['config_ids'])) for r in rows)
    if canonical(actual) != canonical(expected):
        raise ValueError('Training examples differ from the selected reference; stop the controlled comparison')


def pilot(job):
    verify_rows(job)
    cfg, exp = job['cfg'], dict(job['experiment'], iterations=job['pilot_iterations'])
    records = json.loads((Path(job['scratch'])/'prepared.json').read_text())
    start = time.monotonic()
    with threadpool_limits(limits=cfg['cpu_threads']):
        x,y,w,g = engine.training_data(job['scratch'], records, exp, job['seed'], cfg)
        fit_start = time.monotonic()
        bundle = fit_model(x,y,w,g,exp,job['seed'],cfg)
        fit_seconds = time.monotonic()-fit_start
        p = Path(job['output'])/'pilot_model.joblib'
        joblib.dump(bundle, p, compress=0)
        size = p.stat().st_size
        p.unlink()
    write_json(Path(job['output'])/'pilot.json', dict(rows=len(x),features=x.shape[1],
        iterations=exp['iterations'],fit_seconds=fit_seconds,
        preparation_seconds=fit_start-start,model_bytes=size))


def run(job):
    install()
    action = job['action']
    if action == 'prepare':
        engine.prepare(job['root'],job['manifest'],job['cfg'],job['scratch'],job['output'])
    elif action == 'pilot':
        pilot(job)
    elif action == 'trial':
        verify_rows(job)
        if job['experiment']['model'] == 'gnn':
            raise ValueError('V5 never trains a GNN')
        engine.run_trial(job)
    elif action == 'holdout':
        bundle = joblib.load(Path(job['model_directory'])/'model.joblib')
        records = json.loads((Path(job['scratch'])/'prepared.json').read_text())
        with threadpool_limits(limits=job['cfg']['cpu_threads']):
            score = engine.evaluate(bundle, job['scratch'], records, 'holdout',
                job['collection'], job['output'], job['seed'], job['experiment']['name'])
        write_json(Path(job['output'])/'holdout_result.json',dict(status='ok',score=score))
    elif action == 'inference':
        from v4_worker import inference
        inference(job)
    else:
        raise ValueError(action)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',required=True)
    args=parser.parse_args()
    try:
        run(json.loads(Path(args.job).read_text()))
    except Exception as exc:
        job=json.loads(Path(args.job).read_text())
        write_json(Path(job['output'])/'worker_failure.json',
                   dict(kind='resource' if isinstance(exc,MemoryError) else 'error',error=str(exc)))
        traceback.print_exc()
        raise SystemExit(1)
