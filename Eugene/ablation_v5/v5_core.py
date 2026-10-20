"""Fixed v5 protocol: immutable v4 benchmark, matched rows, development selection."""
import importlib.metadata
import json
from pathlib import Path
import shutil
import numpy as np
from study_engine import digest, write_json
from study_common import aligned_predictions, frame_score
from v4_core import read, sha, restore, validate_reference, expanded_manifest, blend

VERSION = '5.0.0'
XGBOOST_VERSION = '3.0.5'


def validate_v4(ref):
    ref = Path(ref)
    if read(ref/'run_identity.json')['version'] != '4.0.0':
        raise ValueError('Use the completed v4 ZIP as V4_REFERENCE')
    if read(ref/'run_state.json')['status'] != 'complete':
        raise ValueError('The v4 study is incomplete')
    if read(ref/'submission_status.json')['status'] != 'complete':
        raise ValueError('V4 submission is incomplete')
    validate_reference(ref/'reference_v3')
    for c in read(ref/'run_state.json')['config']['collections']:
        if not (ref/c.replace(':', '_')/'complete.json').exists():
            raise ValueError(f'Incomplete reference collection: {c}')
    return read(ref/'run_state.json')['config']


def import_reference(source, output):
    output = Path(output)
    ref = output/'reference_v4'
    if ref.exists():
        validate_v4(ref)
        return ref
    source = Path(source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError('Upload completed v4 results and set V4_REFERENCE')
    output.mkdir(parents=True, exist_ok=True)
    temporary = output/'.reference_import'
    if temporary.exists():
        raise ValueError(f'Interrupted import: inspect {temporary} before retrying')
    if source.is_file():
        restore(source, temporary, 12)
    else:
        if source == output.resolve() or source in output.resolve().parents or output.resolve() in source.parents:
            raise ValueError('Reference and output must be separate')
        if any(p.is_symlink() for p in source.rglob('*')):
            raise ValueError('Reference must not contain symlinks')
        shutil.copytree(source, temporary)
    validate_v4(temporary)
    temporary.rename(ref)
    return ref


def defaults(ref):
    cfg = dict(validate_v4(ref))
    cfg['submission_collections'] = list(cfg['collections'])
    cfg.update(version=VERSION, collections=['layout:nlp:default', 'layout:nlp:random',
        'layout:xla:default', 'layout:xla:random'], cpu_threads=1, max_parallel=2,
        session_hours=8., checkpoint_minutes=30, worker_ram_gib=5.,
        system_reserve_gib=2., scratch_limit_gib=4., output_limit_gib=10.,
        disk_reserve_gib=4., max_matrix_mib=1024, max_excluded_fraction=0.,
        pilot_iterations=[10, 30], pilot_memory_multiplier=1.5,
        pilot_memory_add_gib=.25, minimum_gain_layout=.002,
        fold_regression_tolerance=0., xgb_iterations=300, xgb_depth=6,
        xgb_learning_rate=.04, xgb_max_bin=128, xgb_pairs=4,
        checkpoint_score=.558, retry_failed=False, monitor_seconds=.5)
    return cfg


def sources(ref, output, c):
    """Resolve each selected model to its original artifact directory."""
    ref, output = Path(ref), Path(output)
    key = c.replace(':', '_')
    result = {}
    for base in [ref/'reference_v3', ref, output]:
        p = base/key/'experiment_plan.json'
        if p.exists():
            for e in read(p):
                result[e['name']] = dict(directory=str(base/key), experiment=e)
    return result


def tabular_reference(ref, c):
    lock = read(Path(ref)/c.replace(':', '_')/'selection_lock.json')
    catalog = sources(ref, Path(ref)/'.unused', c)
    tabs = [n for n in lock['components'] if catalog[n]['experiment']['model'] != 'gnn']
    if len(tabs) != 1:
        raise ValueError('V5 expects one selected tabular member per collection')
    return tabs[0], catalog[tabs[0]]


def plans(ref, c, cfg):
    name, source = tabular_reference(ref, c)
    base = dict(source['experiment'])
    if base['model'] != 'hgb':
        raise ValueError('The HGB target control requires an HGB reference')
    shared = dict(base, reference=name)
    return [
        dict(shared, name='V5_H_percentile', stage='target', target='percentile'),
        dict(shared, name='V5_X_regression', stage='learner', model='xgb_reg',
             target='centered_log', iterations=cfg['xgb_iterations']),
        dict(shared, name='V5_X_pairwise', stage='objective', model='xgb_rank',
             target='centered_log', iterations=cfg['xgb_iterations']),
    ]


def manifest(root, ref, c, fold, cfg):
    name, source = tabular_reference(ref, c)
    if name.startswith('V4_'):
        return expanded_manifest(root, Path(ref)/'reference_v3', c, fold,
                                 source['experiment'], cfg)
    return read(Path(ref)/'reference_v3'/c.replace(':', '_')/'manifests'/f'fold_{fold}.json')


def expected_training(ref, c, fold, seed):
    _, source = tabular_reference(ref, c)
    return Path(source['directory'])/'folds'/str(fold)/'trials'/source['experiment']['name']/str(seed)/'training_manifest.json'


def prediction_frames(ref, output, c, names, cfg, seeds, split='dev'):
    catalog = sources(ref, output, c)
    return {f: {n: aligned_predictions(Path(catalog[n]['directory'])/'folds'/str(f),
                                       [n], seeds, split)[n] for n in names}
            for f in cfg['folds']}


def choose(ref, output, c, new_names, cfg, seeds):
    incumbent = read(Path(ref)/c.replace(':', '_')/'selection_lock.json')['components']
    names = list(dict.fromkeys([*incumbent, *new_names]))
    frames = prediction_frames(ref, output, c, names, cfg, seeds)
    base = [frame_score(blend(frames[f], incumbent), False) for f in cfg['folds']]
    candidates = [incumbent]
    tab, _ = tabular_reference(ref, c)
    for name in new_names:
        # Fixed small grid: new tabular model + current complete system.
        for w in [.25, .5, .75, 1.]:
            components = {n: (1-w)*v for n, v in incumbent.items() if (1-w)*v}
            components[name] = w
            candidates.append(components)
        # Explicitly replace only the tabular member, preserving the GNN weight.
        if len(incumbent) > 1:
            candidates.append({**{n:v for n,v in incumbent.items() if n != tab}, name:incumbent[tab]})
    rows, seen = [], set()
    for components in candidates:
        identity = digest(components)
        if identity in seen:
            continue
        seen.add(identity)
        values = [frame_score(blend(frames[f], components), False) for f in cfg['folds']]
        delta = np.asarray(values)-base
        same = components == incumbent
        rows.append(dict(components=components, fold_scores=values,
            mean_score=float(np.mean(values)), gain_vs_v4=float(delta.mean()),
            eligible=bool(same or (delta.min() >= -cfg['fold_regression_tolerance']
                                  and delta.mean() >= cfg['minimum_gain_layout']))))
    best = max((r for r in rows if r['eligible']), key=lambda r:r['mean_score'])
    return dict(**best, incumbent=incumbent, incumbent_fold_scores=base,
                candidates=rows, seeds=seeds, changed=best['components'] != incumbent,
                source='Development only; reused folds, exploratory selection')


def finalists(screening, admitted):
    rows = [r for r in screening['candidates'] if r['eligible'] and set(r['components']) & set(admitted)]
    if not rows:
        return []
    winner = max(rows, key=lambda r:r['mean_score'])
    chosen = [n for n in winner['components'] if n in admitted]
    # Keep the regression/ranking bridge if either XGBoost arm is promoted.
    if any(n.startswith('V5_X_') for n in chosen):
        chosen += [n for n in ['V5_X_regression', 'V5_X_pairwise'] if n in admitted]
    return list(dict.fromkeys(chosen))


def archive_identity(output, runtime, cfg, enforce_versions=True):
    output, runtime = Path(output), Path(runtime)
    ref = output/'reference_v4'
    old = read(ref/'run_identity.json')['versions']
    versions = {n:importlib.metadata.version(n) for n in old}
    if enforce_versions and versions != old:
        raise ValueError('Scientific dependencies must match v4; run dependency cell and restart if requested')
    versions.update({n:importlib.metadata.version(n) for n in ['xgboost', 'threadpoolctl', 'matplotlib']})
    if enforce_versions and versions['xgboost'] != XGBOOST_VERSION:
        raise ValueError('Use the pinned XGBoost version')
    for p in runtime.glob('study_*.py'):
        archived = ref/'reference_v3'/p.name
        if archived.exists() and sha(archived) != sha(p):
            raise ValueError('Frozen feature/model dependency changed: '+p.name)
    reference_digest = digest({str(p.relative_to(ref)):sha(p) for p in ref.rglob('*') if p.is_file()})
    excluded = {'session_hours', 'checkpoint_minutes', 'sample_submission', 'worker_ram_gib',
        'system_reserve_gib', 'job_minutes', 'extraction_minutes', 'inference_minutes',
        'retry_failed', 'max_parallel', 'monitor_seconds'}
    value = dict(config={k:v for k,v in cfg.items() if k not in excluded},
                 versions=versions, sources={p.name:sha(p) for p in runtime.glob('*.py')},
                 reference_digest=reference_digest)
    signature = digest(value)
    p = output/'run_identity.json'
    if p.exists() and read(p)['signature'] != signature:
        raise ValueError('V5 code, reference, versions or scientific settings changed; do not mix runs')
    write_json(p, dict(version=VERSION, signature=signature, **value))
    for p in runtime.glob('*.py'):
        shutil.copy2(p, output/p.name)
