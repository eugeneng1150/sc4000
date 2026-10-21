import importlib.metadata

import json

import os

from pathlib import Path

import shutil

import subprocess

import sys

import time

import zipfile

import numpy as np

import pandas as pd

import psutil

from study_engine import (BLOCKS, COLLECTIONS, GIB, VERSION, digest, experiments,
                          make_manifest, release_memory, scores, stable_seed, write_json)

def default_config():
    return dict(
        collections=COLLECTIONS.copy(), seeds=[42, 137, 2026], split_seed=729,
        layout_train_graphs=40, tile_train_graphs=160, dev_graphs=12, holdout_graphs=24,
        train_pool_configs=384, fit_configs=256, eval_configs=1024,
        opcode_bins=128, depth_bins=64, edge_bins=64, edge_cap=100000,
        hgb_iterations=180, extra_trees=80, mlp_epochs=60, pairs_per_graph=256,
        cpu_threads=2, max_graph_features_mib=96, max_matrix_mib=256,
        worker_ram_gib=5.0, system_reserve_gib=2.0, scratch_limit_gib=2.0,
        output_limit_gib=4.0, disk_reserve_gib=3.0,
        job_minutes=25, extraction_minutes=150, session_hours=9.0,
        bootstrap_repeats=1000, max_excluded_fraction=.1,
        clean_scratch=True, retry_failed=False,
        gnn_collections=['layout:xla:default', 'layout:xla:random'],
        gnn_nodes=512, gnn_config_nodes=384, gnn_edges=4096,
        gnn_train_graphs=24, gnn_configs=64, gnn_batch=16, gnn_predict_batch=32,
        gnn_epochs=20, gnn_patience=5, gnn_early_configs=128, gnn_job_minutes=120,
        inference_batch=128, inference_minutes=90, generate_submissions=True,
        sample_submission='', optional_screening=True,
    )

def scientific_config(cfg):
    resource = {'worker_ram_gib', 'system_reserve_gib', 'scratch_limit_gib', 'output_limit_gib',
                'disk_reserve_gib', 'job_minutes', 'extraction_minutes', 'session_hours',
                'clean_scratch', 'retry_failed', 'gnn_job_minutes', 'inference_minutes', 'sample_submission'}
    return {k: v for k, v in cfg.items() if k not in resource}

def validate_config(cfg):
    if not cfg['seeds'] or len(set(cfg['seeds'])) != len(cfg['seeds']):
        raise ValueError('Use distinct training seeds')
    if cfg['fit_configs'] > cfg['train_pool_configs']:
        raise ValueError('fit_configs cannot exceed train_pool_configs')
    for name in ['fit_configs', 'eval_configs', 'train_pool_configs', 'layout_train_graphs',
                 'tile_train_graphs', 'dev_graphs', 'holdout_graphs', 'hgb_iterations', 'cpu_threads',
                 'gnn_nodes','gnn_config_nodes','gnn_edges','gnn_train_graphs','gnn_configs',
                 'gnn_batch','gnn_predict_batch','gnn_epochs','gnn_patience','gnn_early_configs','inference_batch']:
        if cfg[name] < 1:
            raise ValueError(f'{name} must be positive')
    if cfg['gnn_configs'] > cfg['train_pool_configs']:
        raise ValueError('GNN configurations exceed training pool')
    if set(cfg['gnn_collections']) - {'layout:xla:default','layout:xla:random'}:
        raise ValueError('This GNN protocol targets the two XLA layout collections')
    if set(cfg['collections']) - set(COLLECTIONS):
        raise ValueError('Unknown collection')
    if cfg['worker_ram_gib'] + cfg['system_reserve_gib'] > psutil.virtual_memory().total / GIB:
        raise ValueError('Worker cap plus system reserve exceeds machine RAM')

def folder_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())

def terminate(process):
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for p in children:
            p.terminate()
        parent.terminate()
        _, alive = psutil.wait_procs(children + [parent], timeout=3)
        for p in alive:
            p.kill()
    except psutil.NoSuchProcess:
        pass
    except (psutil.AccessDenied, PermissionError):
        # Still reap our own worker if host process enumeration is restricted.
        process.terminate()
    process.wait(timeout=10)

def execute(job, engine_path, cfg, deadline, status_path, heartbeat=None):
    status_path = Path(status_path)
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if previous['status'] == 'ok':
            return True
        resumable_timeout = (previous['status']=='timeout' and job.get('experiment',{}).get('model')=='gnn'
                             and (status_path.parent/'latest.pt').exists())
        if not cfg['retry_failed'] and previous['status'] not in ['session_budget', 'interrupted', 'running'] and not resumable_timeout:
            return False
    if time.monotonic() >= deadline:
        return None
    if shutil.disk_usage(status_path.parent).free < cfg['disk_reserve_gib'] * GIB:
        write_json(status_path, dict(status='disk_guard', detail='Disk reserve reached'))
        return False
    job_path = status_path.with_suffix('.job.json')
    log_path = status_path.with_suffix('.log')
    write_json(job_path, job)
    environment = os.environ.copy()
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        environment[name] = str(cfg['cpu_threads'])
    start, peak, reason = time.monotonic(), 0, None
    maximum = cfg['extraction_minutes'] if job['action'] == 'prepare' else cfg['job_minutes']
    if job.get('experiment', {}).get('model') == 'gnn':
        maximum = cfg['gnn_job_minutes']
    if job['action']=='probe':maximum=cfg.get('probe_minutes',5)
    if job['action'] == 'inference':
        maximum = cfg['inference_minutes']
    write_json(status_path,dict(status='running'))
    with log_path.open('w') as log:
        process = subprocess.Popen([sys.executable, str(engine_path), '--job', str(job_path)],
                                   stdout=log, stderr=subprocess.STDOUT, env=environment)
        try:
            last_message = start
            while process.poll() is None:
                try:
                    parent = psutil.Process(process.pid)
                    rss = sum(p.memory_info().rss for p in [parent] + parent.children(recursive=True)
                              if p.is_running())
                    peak = max(peak, rss)
                except psutil.NoSuchProcess:
                    continue
                elapsed = time.monotonic() - start
                if rss > cfg['worker_ram_gib'] * GIB:
                    reason = 'memory_guard'
                elif psutil.virtual_memory().available < cfg['system_reserve_gib'] * GIB:
                    reason = 'system_memory_guard'
                elif elapsed > maximum * 60:
                    reason = 'timeout'
                elif time.monotonic() >= deadline:
                    reason = 'session_budget'
                elif shutil.disk_usage(status_path.parent).free < cfg['disk_reserve_gib'] * GIB:
                    reason = 'disk_guard'
                if reason:
                    terminate(process)
                    break
                if time.monotonic() - last_message >= 45:
                    print(f'  still running: {elapsed/60:.1f} min, worker {rss/GIB:.2f} GiB', flush=True)
                    last_message = time.monotonic()
                if heartbeat:heartbeat()
                time.sleep(.5)
        except BaseException:
            terminate(process)
            write_json(status_path, dict(status='interrupted', peak_rss_gib=peak/GIB))
            raise
    status = reason or ('ok' if process.returncode == 0 else 'failed')
    write_json(status_path, dict(status=status, peak_rss_gib=peak/GIB,
                                wall_seconds=time.monotonic()-start, exit_code=process.returncode))
    release_memory()
    if status != 'ok':
        print(f'  {status}: {log_path}', flush=True)
    return None if status == 'session_budget' else status == 'ok'

def aligned_predictions(output, names, seeds, split):
    """One graph/config ID alignment for all ranks; reject mismatches instead of intersecting."""
    result = {}
    for name in names:
        frames = []
        for seed in seeds:
            path = Path(output) / 'trials' / name / str(seed) / f'{split}_predictions.csv.gz'
            frame = pd.read_csv(path).sort_values(['graph', 'config_id']).reset_index(drop=True)
            if frames and not frame[['graph', 'config_id', 'runtime']].equals(frames[0][['graph', 'config_id', 'runtime']]):
                raise ValueError('Prediction IDs or labels differ across seeds')
            frame['rank'] = frame.groupby('graph').prediction.rank(method='average', pct=True)
            frames.append(frame)
        mean = frames[0][['graph', 'config_id', 'runtime']].copy()
        mean['prediction'] = np.mean([f['rank'].to_numpy() for f in frames], axis=0)
        if result:
            first = next(iter(result.values()))
            if not mean[['graph', 'config_id', 'runtime']].equals(first[['graph', 'config_id', 'runtime']]):
                raise ValueError('Prediction IDs differ between methods')
        result[name] = mean
    return result

def frame_score(frame, tile):
    return float(np.mean([scores(g.runtime.to_numpy(), g.prediction.to_numpy(), tile)['score']
                          for _, g in frame.groupby('graph')]))

def paired_summary(graphs, seeds, repeats, reference='B0_basic_hgb'):
    if graphs.empty:
        return pd.DataFrame()
    baseline = graphs[graphs.experiment == reference][['graph', 'seed', 'score']].rename(columns={'score':'baseline'})
    results = []
    for name, frame in graphs.groupby('experiment'):
        paired = frame.merge(baseline, on=['graph','seed'], validate='one_to_one')
        paired['delta'] = paired.score - paired.baseline
        # Average repeats within graph; bootstrap independent graphs, not configs/seeds.
        by_graph = paired.groupby('graph').agg(delta=('delta','mean'), seeds=('seed','nunique'))
        by_graph = by_graph[by_graph.seeds == len(seeds)]
        if by_graph.empty:
            continue
        d = by_graph.delta.to_numpy()
        rng = np.random.default_rng(551)
        boot = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(repeats)])
        low, high = np.percentile(boot, [2.5,97.5])
        base_mean = float(paired.baseline.mean())
        candidate_mean = float(paired.score.mean())
        results.append(dict(experiment=name, reference=reference, baseline_score=base_mean,
                            candidate_score=candidate_mean, mean_paired_gain=float(d.mean()),
                            score_points=100*float(d.mean()),
                            relative_gain_pct=(100*float(d.mean())/base_mean if base_mean > .01 else None),
                            graphs=len(d),
                            graph_win_fraction=float(np.mean(d > 0)),
                            bootstrap_low=float(low), bootstrap_high=float(high),
                            uncertainty_note='exploratory graph bootstrap; no multiplicity correction'))
    return pd.DataFrame(results)
