"""Sequential, resumable orchestration for the standalone Colab experiment notebook."""
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


def execute(job, engine_path, cfg, deadline, status_path):
    status_path = Path(status_path)
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if previous['status'] == 'ok':
            return True
        resumable_timeout = (previous['status']=='timeout' and job.get('experiment',{}).get('model')=='gnn'
                             and (status_path.parent/'latest.pt').exists())
        if not cfg['retry_failed'] and previous['status'] not in ['session_budget', 'interrupted'] and not resumable_timeout:
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
    if job['action'] == 'inference':
        maximum = cfg['inference_minutes']
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


def read_results(output):
    rows = []
    for path in Path(output).glob('trials/*/*/result.json'):
        status = path.parent / 'status.json'
        if status.exists() and json.loads(status.read_text()).get('status') == 'ok':
            row = json.loads(path.read_text())
            row.update(json.loads(status.read_text()))
            rows.append(row)
    return pd.DataFrame(rows)


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


def core_experiments(collection, cfg):
    tabular = experiments(collection.startswith('tile'))
    core = [e for e in tabular if e['name'] in ['B0_basic_hgb','B1_masked_hgb','P0_position','P_full']
            or e['name'].startswith('A_without_')]
    if collection in cfg['gnn_collections']:
        from study_gnn import catalog
        # GNN reference/additions run before secondary tabular removals.
        core = core[:1] + [dict(core[0],name='B_matched_hgb',matched=True)] + catalog() + core[1:]
    return core


def lock_selection(output, cfg, collection, names):
    path = Path(output) / 'selection_lock.json'
    if path.exists():
        return json.loads(path.read_text())
    rows = read_results(output)
    complete = [n for n in names if set(rows[rows.experiment == n].seed) == set(cfg['seeds'])]
    if set(names) != set(complete):
        raise ValueError('Every core comparison must finish all seeds before selection/holdout')
    summaries = rows[rows.experiment.isin(complete)].groupby('experiment').dev_score.agg(['mean','std'])
    summaries.sort_values('mean',ascending=False).to_csv(Path(output)/'confirmation_dev_summary.csv')
    pred = aligned_predictions(output, complete, cfg['seeds'], 'dev')
    tile = collection.startswith('tile')
    # Select actual seed ensembles, rather than assuming the best single-seed mean is best.
    tabular = max([n for n in complete if not n.startswith('G')],key=lambda n: frame_score(pred[n],tile))
    neural = [n for n in complete if n.startswith('G')]
    gnn = max(neural,key=lambda n: frame_score(pred[n],tile)) if neural else None
    candidates = [dict(components={'B0_basic_hgb':1.},dev_score=frame_score(pred['B0_basic_hgb'],tile))]
    for base in list(dict.fromkeys(['B0_basic_hgb',tabular])):
        for weight in ([0.,.25,.5,.75,1.] if gnn else [0.]):
            components = {base:1-weight}
            if gnn:
                components[gnn] = weight
            components = {n:w for n,w in components.items() if w > 0}
            blend = pred[base].copy()
            blend['prediction'] = sum(w*pred[n].prediction for n,w in components.items())
            candidates.append(dict(components=components,dev_score=frame_score(blend,tile)))
    # Stable max keeps B0 on exact ties. Holdout is never consulted here.
    selected = max(candidates,key=lambda c:c['dev_score'])
    lock = dict(winner=tabular, gnn=gnn, components=selected['components'],
                dev_score=selected['dev_score'], confirmed=complete,seeds=cfg['seeds'],
                blend_grid=candidates,source='development predictions only',
                locked_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    write_json(path,lock)
    return lock


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


def summarize_collection(output, cfg, collection):
    output = Path(output)
    result = read_results(output)
    if not result.empty:
        result.to_csv(output / 'all_trials.csv', index=False)
    for split in ['dev', 'holdout']:
        paths = list(output.glob(f'trials/*/*/{split}_graphs.csv'))
        if not paths:
            continue
        frames = []
        for p in paths:
            marker = p.parent / ('holdout_status.json' if split == 'holdout' else 'status.json')
            if marker.exists() and json.loads(marker.read_text()).get('status') == 'ok':
                frames.append(pd.read_csv(p))
        if not frames:
            continue
        graphs = pd.concat(frames, ignore_index=True)
        graphs.to_csv(output / f'{split}_graph_scores.csv', index=False)
        graphs.groupby(['experiment','seed']).score.mean().reset_index().to_csv(output / f'{split}_seed_scores.csv', index=False)
        paired_summary(graphs, cfg['seeds'], cfg['bootstrap_repeats']).to_csv(output / f'{split}_paired_gains.csv', index=False)
        for reference in ['G0_reference','G_full','B_matched_hgb']:
            if reference in set(graphs.experiment):
                paired_summary(graphs,cfg['seeds'],cfg['bootstrap_repeats'],reference).to_csv(
                    output/f'{split}_paired_vs_{reference}.csv',index=False)
        graphs.groupby(['experiment','family']).score.agg(['mean','count']).to_csv(output / f'{split}_family_scores.csv')
    lock_path = output / 'selection_lock.json'
    if not lock_path.exists():
        return
    lock = json.loads(lock_path.read_text())
    names = list(dict.fromkeys(['B0_basic_hgb', lock['winner']] + list(lock['components'])))
    if not all((output / 'trials' / n / str(s) / 'holdout_result.json').exists() for n in names for s in cfg['seeds']):
        return
    pred = aligned_predictions(output, names, cfg['seeds'], 'holdout')
    b, p = pred['B0_basic_hgb'], pred[lock['winner']]
    blend = b.copy()
    blend['prediction'] = sum(w*pred[n].prediction for n,w in lock['components'].items())
    rows = []
    for name, frame in [('baseline_seed_ensemble', b), ('winner_seed_ensemble', p), ('locked_blend', blend)]:
        frame.to_csv(output / f'{name}_holdout_predictions.csv.gz', index=False, compression='gzip')
        for graph, group in frame.groupby('graph'):
            rows.append(dict(system=name, graph=graph,
                             **scores(group.runtime.to_numpy(), group.prediction.to_numpy(), collection.startswith('tile'))))
    table = pd.DataFrame(rows)
    table.to_csv(output / 'locked_systems_holdout_graphs.csv', index=False)
    table.groupby('system').score.mean().to_csv(output / 'locked_systems_holdout_scores.csv')
    paired_summary(table.rename(columns={'system':'experiment'}).assign(seed=0),[0],
                   cfg['bootstrap_repeats'],'baseline_seed_ensemble').to_csv(
                       output/'locked_systems_holdout_gains.csv',index=False)


def make_bundle(output):
    output = Path(output)
    target = output.parent / f'{output.name}_results.zip'
    temp = target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=2) as z:
        for path in output.rglob('*'):
            if path.is_file() and not path.name.endswith('.tmp'):
                z.write(path, path.relative_to(output))
    temp.replace(target)
    return target


def summary_report(output, states, cfg):
    output = Path(output)
    text = ['# SC4000 experiment run', '',
            'These are local sampled-validation scores, not Kaggle leaderboard scores.',
            'Development graphs were held out from official train. Official valid was reserved for the final audit.',
            'The selected methods/blend were locked using development scores before the holdout was evaluated.',
            'All trials use independently implemented methods. See notebook references for inspiration.', '',
            '| Collection | Status |', '|---|---|']
    for collection, state in states.items():
        text.append(f'| {collection} | {state} |')
    text += ['', 'Collection status above describes required comparisons and the holdout audit.']
    for stage in ['submission','optional']:
        path=output/f'{stage}_status.json'
        state=json.loads(path.read_text()) if path.exists() else {'status':'pending' if cfg['generate_submissions' if stage=='submission' else 'optional_screening'] else 'disabled'}
        text += ['', f'{stage.capitalize()} stage: `{json.dumps(state)}`']
    text += ['', '## Reading the results', '',
             '- `all_trials.csv`: development scores, model size, time and monitored worker peak RAM.',
             '- `holdout_paired_gains.csv`: paired gains against HGB, averaged within graph across seeds.',
             '- `holdout_paired_vs_G0_reference.csv`: additions versus the fixed GraphSAGE reference.',
             '- `holdout_paired_vs_G_full.csv`: removals versus the full GNN; compare signs carefully.',
             '- `submission_baseline.csv`, `submission_selected.csv`: full test predictions when export finishes.',
             '- `locked_systems_holdout_scores.csv`: seed ensembles and the development-selected blend.',
             '- `coverage.json`, `manifest.json`: exact graphs/configurations and data source fingerprints.',
             '- `extraction_failures.json`, worker status/log files: skipped or failed work. Do not ignore exclusions.',
             '- Models/predictions are retained for review; raw data and temporary feature matrices are excluded.',
             '', '## Limits', '',
             'Three seeds vary training configuration subsamples and model randomness, not graph splits.',
             'Layout tau-b is exact on the fixed sampled candidates. Tile top-5 regret is relative to the sampled candidates.',
             'Scores do not establish full-test/public-leaderboard performance. No Kaggle submission is made.',
             'Graph names are disjoint; related architecture families may still occur in both sets.',
             'Confidence intervals are exploratory graph bootstraps, not corrected for the many comparisons.',
             'HGB early stopping is disabled to avoid row-wise validation leakage within a graph.',
             'The MLP uses documented weighted resampling; model-family comparisons change more than one training detail.',
             'Both XLA layout collections use a bounded GraphSAGE reference, additions and removal controls.',
             'The GNN graph is sampled: inspect node/configurable-node/edge coverage before interpreting improvements.',
             'Relative gain percentages are omitted when the reference score is near zero or negative; prefer absolute tau changes.',
             'Optional tabular screening runs only after the final selection is frozen and cannot change it.',
             'Submission CSVs rank all test candidates using the audited fitted models; they are not retrained on holdout.',
             'Resource limits stop workers proactively, but cannot guarantee protection from an OS kill or Colab reset.',
             'A downloaded ZIP can restore progress; local files alone do not survive runtime deletion.', '']
    (output / 'RUN_SUMMARY.md').write_text('\n'.join(text))
    write_json(output / 'run_state.json', dict(collections=states, config=cfg))


def run_study(root, output, scratch_root, engine_path, cfg):
    validate_config(cfg)
    root, output, scratch_root = Path(root).resolve(), Path(output).resolve(), Path(scratch_root).resolve()
    if scratch_root == output or scratch_root == root or root in scratch_root.parents or output in scratch_root.parents:
        raise ValueError('Scratch must be separate from source data and saved outputs')
    if scratch_root in root.parents or scratch_root in output.parents:
        raise ValueError('Scratch cannot contain data or saved outputs')
    output.mkdir(parents=True,exist_ok=True);scratch_root.mkdir(parents=True,exist_ok=True)
    marker = scratch_root/'.sc4000_owned_scratch'
    if not marker.exists() and any(scratch_root.iterdir()):
        raise ValueError('Refusing to use a nonempty unmarked scratch directory')
    marker.touch()
    packages = ['numpy','pandas','scipy','scikit-learn','joblib','psutil']
    if cfg['gnn_collections']:
        packages.append('torch')
    versions = {n:importlib.metadata.version(n) for n in packages}
    modules = {'engine':digest(Path(engine_path).read_text()),'controller':digest(Path(__file__).read_text())}
    for name in ['gnn','submission']:
        path = Path(engine_path).parent/f'study_{name}.py'
        if path.exists():
            modules[name] = digest(path.read_text())
    signature = digest(dict(config=scientific_config(cfg),modules=modules,versions=versions))
    identity = output/'run_identity.json'
    if identity.exists() and json.loads(identity.read_text())['signature'] != signature:
        raise ValueError('Code, scientific settings or package versions changed. Use a new RUN_NAME to avoid mixing results.')
    write_json(identity,dict(signature=signature,versions=versions,config=cfg,version=VERSION))
    deadline = time.monotonic()+cfg['session_hours']*3600
    states = {c:'pending' for c in cfg['collections']}
    active = None

    def prepare_collection(collection):
        directory=output/collection.replace(':','_');directory.mkdir(exist_ok=True)
        scratch=scratch_root/directory.name
        manifest=make_manifest(root,collection,cfg)
        previous=directory/'manifest.json'
        if previous.exists() and json.loads(previous.read_text()) != manifest:
            raise ValueError(f'{collection}: source manifest changed. Use a new RUN_NAME.')
        write_json(previous,manifest)
        if not (scratch/'ready.json').exists():
            (directory/'prepare_status.json').unlink(missing_ok=True)
        ok=execute(dict(action='prepare',root=str(root),manifest=manifest,cfg=cfg,
                        scratch=str(scratch),output=str(directory)),engine_path,cfg,deadline,directory/'prepare_status.json')
        return directory,scratch,ok

    def fit(collection,directory,scratch,exp,seed,action='trial'):
        trial=directory/'trials'/exp['name']/str(seed);trial.mkdir(parents=True,exist_ok=True)
        if folder_bytes(output) > cfg['output_limit_gib']*GIB:
            return False
        job=dict(action=action,cfg=cfg,scratch=str(scratch),collection=collection,
                 experiment=exp,seed=seed,output=str(trial))
        return execute(job,engine_path,cfg,deadline,trial/('status.json' if action=='trial' else 'holdout_status.json'))

    def cleanup(scratch):
        if cfg['clean_scratch'] and marker.exists() and scratch.parent == scratch_root and scratch.exists():
            shutil.rmtree(scratch)

    try:
        # All required comparisons precede optional exploratory searches.
        for collection in cfg['collections']:
            active=collection
            directory=output/collection.replace(':','_');directory.mkdir(exist_ok=True)
            scratch=scratch_root/directory.name
            core=core_experiments(collection,cfg)
            write_json(directory/'experiment_plan.json',core)
            # Always check source identity, including when all work is already complete.
            manifest=make_manifest(root,collection,cfg)
            previous=directory/'manifest.json'
            if previous.exists() and json.loads(previous.read_text()) != manifest:
                raise ValueError(f'{collection}: source manifest changed. Use a new RUN_NAME.')
            write_json(previous,manifest)
            if (directory/'complete.json').exists():
                states[collection]='complete (resumed)';continue
            if time.monotonic() >= deadline:
                states[collection]='incomplete: session budget';continue
            print(f'\n{collection}: {len(core)} required methods × {len(cfg["seeds"])} seeds',flush=True)
            directory,scratch,ok=prepare_collection(collection)
            if not ok:
                states[collection]='incomplete: extraction/resource limit';continue
            complete=True
            for exp in core:
                for seed in cfg['seeds']:
                    print(f'{exp["name"]}, seed {seed}',flush=True)
                    ok=fit(collection,directory,scratch,exp,seed)
                    complete=complete and bool(ok)
                    if ok is None: break
                if ok is None: break
            if not complete:
                states[collection]='incomplete: required trials (inspect status/logs)'
                cleanup(scratch);continue
            lock=lock_selection(directory,cfg,collection,[e['name'] for e in core])
            print(f'Development selection locked: {lock["components"]}; auditing reserved graphs',flush=True)
            audited=True
            for exp in core:
                for seed in cfg['seeds']:
                    ok=fit(collection,directory,scratch,exp,seed,'holdout')
                    audited=audited and bool(ok)
                    if ok is None:break
                if ok is None:break
            states[collection]='complete' if audited else 'incomplete: holdout jobs'
            if audited:write_json(directory/'complete.json',dict(status='complete'))
            summarize_collection(directory,cfg,collection)
            cleanup(scratch)
            summary_report(output,states,cfg);print(f'Checkpoint: {make_bundle(output)}',flush=True)

        # Full test export is ahead of optional model/feature searches.
        if cfg['generate_submissions']:
            from study_submission import export_submissions
            if all((output/c.replace(':','_')/'complete.json').exists() for c in cfg['collections']):
                export_submissions(root,output,engine_path,cfg,deadline,execute)
            else:
                write_json(output/'submission_status.json',dict(status='pending',reason='Required core audit incomplete'))
        if cfg['optional_screening'] and all(v.startswith('complete') for v in states.values()):
            submission_status=output/'submission_status.json'
            can_screen=not cfg['generate_submissions'] or (submission_status.exists() and
                json.loads(submission_status.read_text())['status']=='complete')
            if can_screen:
                for collection in cfg['collections']:
                    if time.monotonic() >= deadline:break
                    active=collection
                    directory,scratch,ok=prepare_collection(collection)
                    if not ok:continue
                    core_names={e['name'] for e in core_experiments(collection,cfg)}
                    optional=[e for e in experiments(collection.startswith('tile')) if e['name'] not in core_names]
                    write_json(directory/'optional_plan.json',optional)
                    for exp in optional:
                        print(f'Exploratory dev-only: {exp["name"]}',flush=True)
                        if fit(collection,directory,scratch,exp,cfg['seeds'][0]) is None:break
                    cleanup(scratch)
    except KeyboardInterrupt:
        if active:states[active]='incomplete: interrupted'
        print('Interrupted. Saving completed work; rerun with the same settings to resume.',flush=True)
    except Exception as exc:
        if active:states[active]=f'incomplete: {type(exc).__name__}: {exc}'
        write_json(output/'controller_error.json',dict(type=type(exc).__name__,error=str(exc)))
        raise
    finally:
        optional_total, optional_done, optional_failed = 0, 0, 0
        for collection in cfg['collections']:
            directory=output/collection.replace(':','_')
            core_names={e['name'] for e in core_experiments(collection,cfg)}
            for exp in experiments(collection.startswith('tile')):
                if exp['name'] in core_names:continue
                optional_total+=1
                status=directory/'trials'/exp['name']/str(cfg['seeds'][0])/'status.json'
                if status.exists():
                    if json.loads(status.read_text())['status']=='ok':optional_done+=1
                    else:optional_failed+=1
            if directory.exists():
                try:summarize_collection(directory,cfg,collection)
                except Exception as exc:write_json(directory/'summary_error.json',dict(error=str(exc)))
            cleanup(scratch_root/directory.name)
        write_json(output/'optional_status.json',dict(
            status=('disabled' if not cfg['optional_screening'] else 'complete' if optional_done==optional_total else 'incomplete'),
            completed=optional_done,total=optional_total,failed_or_interrupted=optional_failed,
            interpretation='development-only exploration; does not alter the locked selection'))
        summary_report(output,states,cfg)
        bundle=make_bundle(output);release_memory()
        print(f'\nResults bundle: {bundle}\nRaw data retained; temporary feature matrices removed.',flush=True)
    return bundle
