"""V4 protocol, immutable reference import, strict manifests and selection."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
import pandas as pd
from study_engine import digest, write_json, metadata, balanced_order, family, stable_seed, GIB
from study_common import aligned_predictions, frame_score
from study_plan import effective_config

VERSION = '4.0.0'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def restore(archive, destination, limit_gib=12):
    """Validate before extraction; never overlay a populated result directory."""
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f'{destination} already has results. Leave RESTORE_ZIP empty to resume them; no files overwritten.')
    if not zipfile.is_zipfile(archive):
        raise ValueError('Upload is incomplete or not a ZIP. Wait for upload, then verify file size against your computer.')
    with zipfile.ZipFile(archive) as z:
        if sum(i.file_size for i in z.infolist()) > limit_gib * GIB: raise ValueError('Archive exceeds output limit')
        names = z.namelist()
        if len(names) != len(set(names)): raise ValueError('Duplicate ZIP entries')
        for i in z.infolist():
            target = (destination / i.filename).resolve()
            if destination.resolve() not in target.parents or (i.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Unsafe archive member')
        if z.testzip(): raise ValueError('ZIP integrity check failed')
        z.extractall(destination)
    print(f'RESTORED and CRC checked: {archive} -> {destination}', flush=True)


def validate_reference(ref):
    ref = Path(ref)
    identity = read(ref / 'run_identity.json')
    if identity.get('version') != '3.0.0': raise ValueError('V3_REFERENCE must be a completed v3 run')
    cfg = read(ref / 'run_state.json')['config']
    if read(ref / 'submission_status.json')['status'] != 'complete': raise ValueError('V3 submission is incomplete')
    for c in cfg['collections']:
        d = ref / c.replace(':', '_')
        if not (d / 'complete.json').exists(): raise ValueError(f'Incomplete v3 collection: {c}')
        lock = read(d / 'selection_lock.json')
        for name in set(lock['components']) | {lock['winner']}:
            for f in cfg['folds']:
                for s in cfg['seeds']:
                    p = d / 'folds' / str(f) / 'trials' / name / str(s)
                    if read(p / 'status.json')['status'] != 'ok': raise ValueError(f'Incomplete reference fit {p}')
                    for fn in ['dev_predictions.csv.gz', 'holdout_predictions.csv.gz', 'training_manifest.json']:
                        if not (p / fn).exists(): raise FileNotFoundError(p / fn)
    return cfg


def import_reference(source, output):
    output = Path(output); ref = output / 'reference_v3'
    if ref.exists():
        validate_reference(ref)
        print('REUSE embedded reference_v3; no v3 retraining.', flush=True)
        return ref
    source = Path(source).expanduser().resolve()
    if not source.exists(): raise FileNotFoundError('Set V3_REFERENCE to checkpoint 65 ZIP or extracted folder')
    temporary = output / '.reference_import'
    if temporary.exists():
        raise ValueError(f'Previous reference import was interrupted: inspect/rename {temporary} before retrying')
    output.mkdir(parents=True, exist_ok=True)
    size = source.stat().st_size if source.is_file() else sum(p.stat().st_size for p in source.rglob('*') if p.is_file())
    if shutil.disk_usage(output).free < size * 3 + 4 * GIB: raise OSError('Insufficient space to import and back up v3')
    if source.is_file(): restore(source, temporary)
    else:
        if source == output.resolve() or output.resolve() in source.parents or source in output.resolve().parents:
            raise ValueError('Reference and output must be separate directories')
        if any(p.is_symlink() for p in source.rglob('*')): raise ValueError('Symlinks are not accepted in a reference')
        shutil.copytree(source, temporary)
    validate_reference(temporary)
    temporary.rename(ref)
    return ref


def defaults(ref):
    cfg = dict(validate_reference(ref))
    cfg.update(version=VERSION, session_hours=8., checkpoint_minutes=30,
               max_matrix_mib=1024, output_limit_gib=8., scratch_limit_gib=4.,
               max_excluded_fraction=0., job_minutes=90, extraction_minutes=180,
               gnn_job_minutes=120, gnn_epochs=40, gnn_patience=10,
               tile_graph_ladder=[640,1280,2560,5120,0], confirmation_top_k=1,
               minimum_gain_layout=.002, minimum_gain_tile=.0005,
               fold_regression_tolerance=0., pilot_iterations=10,
               matrix_peak_multiplier=4., pilot_memory_fraction=.85)
    # Produce the cheaper tabular evidence before the longer GNN continuation.
    cfg['collections'] = [c for c in cfg['collections'] if c not in cfg['gnn_collections']] + [c for c in cfg['collections'] if c in cfg['gnn_collections']]
    return cfg


def plans(ref, c, cfg):
    d = Path(ref) / c.replace(':', '_'); lock = read(d / 'selection_lock.json')
    catalog = {e['name']: e for e in read(d / 'experiment_plan.json')}
    if c in cfg['gnn_collections']:
        original = catalog[lock['gnn']]
        return [dict(original, name='V4_G_longer', stage='training budget',
                     reference=original['name'], epochs=cfg['gnn_epochs'], patience=cfg['gnn_patience'])]
    base = catalog[lock['winner']]
    counts = cfg['tile_graph_ladder'] if c.startswith('tile') else [0]
    return [dict(base, name=f'V4_T_g{g or "all"}_c256', stage='training graphs',
                 graph_limit=g, train_graphs=0, configs=256, reference=base['name']) for g in counts]


def expanded_manifest(root, ref, c, fold, exp, cfg):
    """Preserve exact v3 train subset, dev IDs, pool sampling and archive CRCs."""
    root = Path(root); d = Path(ref) / c.replace(':','_')
    old = read(d / 'manifests' / f'fold_{fold}.json')
    result = json.loads(json.dumps(old))
    if exp['model'] != 'gnn':
        existing = [r for r in old['records'] if r['split'] == 'train']
        used = {r['graph'] for r in old['records']}
        paths = balanced_order(list(root.joinpath(*c.split(':'), 'train').glob('*.npz')), cfg['split_seed'])
        additional = [p for p in paths if p.stem not in used]
        limit = exp.get('graph_limit', 0)
        take = len(additional) if limit == 0 else max(0, limit - len(existing))
        for p in additional[:take]:
            key = 'config_feat' if c.startswith('tile') else 'node_config_feat'
            shape, fortran, dtype = metadata(p, key)
            ids = np.sort(np.random.default_rng(stable_seed(f'{cfg["split_seed"]}/{c}/{p.stem}'))
                          .choice(shape[0], min(shape[0],cfg['train_pool_configs']), replace=False))
            with zipfile.ZipFile(p) as z: fingerprint = [(i.filename,i.CRC,i.file_size) for i in z.infolist()]
            result['records'].append(dict(split='train',graph=p.stem,family=family(p.stem),
                relative_path=str(p.relative_to(root)),count=shape[0],indices=ids.tolist(),source=digest(fingerprint),
                fortran=bool(fortran),row_bytes=int(np.prod(shape[1:]))*dtype.itemsize))
    for r in result['records']:
        p = root / r['relative_path']
        with zipfile.ZipFile(p) as z: fingerprint = [(i.filename,i.CRC,i.file_size) for i in z.infolist()]
        if digest(fingerprint) != r['source']: raise ValueError(f'Data differs from reference: {p}')
    train = {r['graph'] for r in result['records'] if r['split']=='train'}
    dev = {r['graph'] for r in result['records'] if r['split']=='dev'}
    if train & dev: raise ValueError('Training/development leakage')
    return result


def resources(ref, c, exp, manifest, cfg):
    train = [r for r in manifest['records'] if r['split']=='train']
    rows = sum(min(len(r['indices']),exp['configs']) for r in train)
    if exp['model']=='gnn':
        return dict(admitted=True,graphs=len(train),rows=rows,reason='Reuses v3 winning bounded graph geometry; worker guards apply')
    d = Path(ref)/c.replace(':','_')/'folds'/str(cfg['folds'][0])/'trials'/exp['reference']/str(cfg['seeds'][0])
    features = read(d/'result.json')['feature_count']
    matrix = rows*features*4
    peak = .5 + cfg['matrix_peak_multiplier']*matrix/GIB
    admitted = matrix <= cfg['max_matrix_mib']*1024**2 and peak <= cfg['worker_ram_gib']*cfg['pilot_memory_fraction']
    return dict(admitted=bool(admitted),graphs=len(train),rows=rows,features=features,
                estimated_matrix_mib=matrix/1024**2,estimated_peak_gib=peak,
                reason='Conservative static estimate; measured pilot still required' if admitted else 'Excluded by predeclared matrix/memory estimate')


def predictions(directory, names, cfg, seeds, split='dev'):
    return {f: aligned_predictions(Path(directory)/'folds'/str(f), names, seeds, split) for f in cfg['folds']}


def blend(frames, components):
    first = frames[next(iter(components))].copy()
    keys = ['graph','config_id','runtime']
    for n in components:
        if not first[keys].equals(frames[n][keys]): raise ValueError('Blend IDs/labels differ')
    first['prediction'] = sum(w*frames[n].prediction for n,w in components.items())
    return first


def choose(ref, output, c, names, cfg, seeds):
    """Development-only grid with v3 selected system as the incumbent."""
    rd = Path(ref)/c.replace(':','_'); nd = Path(output)/c.replace(':','_')
    lock = read(rd/'selection_lock.json')
    old_names = list(dict.fromkeys(list(lock['components'])+[lock['winner']]+([lock['gnn']] if lock.get('gnn') else [])))
    old = predictions(rd,old_names,cfg,seeds)
    new = predictions(nd,names,cfg,seeds) if names else {f:{} for f in cfg['folds']}
    all_frames = {f:dict(old[f],**new[f]) for f in cfg['folds']}
    tile = c.startswith('tile')
    base = [frame_score(blend(old[f],lock['components']),tile) for f in cfg['folds']]
    candidates = [dict(lock['components'])]
    if c in cfg['gnn_collections']:
        for n in [lock['gnn']]+names:
            for w in [0.,.25,.5,.75,1.]:
                candidates.append({k:v for k,v in {lock['winner']:1-w,n:w}.items() if v})
    else: candidates += [{n:1.} for n in names]
    rows = []
    for components in candidates:
        values = [frame_score(blend(all_frames[f],components),tile) for f in cfg['folds']]
        delta = np.array(values)-base
        incumbent = components == lock['components']
        eligible = incumbent or (delta.min() >= -cfg['fold_regression_tolerance'] and
                    delta.mean() >= cfg['minimum_gain_tile' if tile else 'minimum_gain_layout'])
        rows.append(dict(components=components,fold_scores=values,mean_score=float(np.mean(values)),
                         gain_vs_v3=float(delta.mean()),eligible=bool(eligible)))
    best = max([r for r in rows if r['eligible']],key=lambda r:r['mean_score'])
    return dict(**best,incumbent=lock['components'],incumbent_fold_scores=base,candidates=rows,
                seeds=seeds,source='development only; reused folds, exploratory selection',
                changed=best['components']!=lock['components'])


def archive_identity(output, cfg, runtime, enforce_versions=True):
    output, runtime = Path(output), Path(runtime)
    ref = output/'reference_v3'
    for p in runtime.glob('study_*.py'):
        archived=ref/p.name
        if archived.exists() and sha(archived)!=sha(p):
            raise ValueError(f'V3 feature/model implementation differs: {p.name}')
    versions = {n:importlib.metadata.version(n) for n in read(ref/'run_identity.json')['versions']}
    if enforce_versions and versions != read(ref/'run_identity.json')['versions']:
        raise ValueError('Package versions differ from v3. Run the pinned dependency cell and restart the kernel if requested.')
    excluded = {'session_hours','checkpoint_minutes','sample_submission','worker_ram_gib','system_reserve_gib',
                'job_minutes','gnn_job_minutes','extraction_minutes','inference_minutes','retry_failed'}
    sources = {p.name:sha(p) for p in runtime.glob('*.py')}
    reference = {str(p.relative_to(ref)):sha(p) for p in ref.rglob('*') if p.is_file()}
    signature = digest(dict(config={k:v for k,v in cfg.items() if k not in excluded},versions=versions,
                            sources=sources,reference=reference))
    path = output/'run_identity.json'
    if path.exists() and read(path)['signature'] != signature: raise ValueError('V4 code/data/settings changed. Do not mix checkpoints.')
    write_json(path,dict(version=VERSION,signature=signature,versions=versions,config=cfg,sources=sources,
                         reference_digest=digest(reference)))
    for p in runtime.glob('*.py'): shutil.copy2(p,output/p.name)


def check_data_identity(root,output,cfg):
    """Include additions/removals as well as changed archive entries on every resume."""
    root=Path(root);files={}
    for c in cfg['collections']:
        for split in ['train','valid','test']:
            for p in sorted(root.joinpath(*c.split(':'),split).glob('*.npz')):
                with zipfile.ZipFile(p) as z:
                    files[str(p.relative_to(root))]=[(i.filename,i.CRC,i.file_size) for i in z.infolist()]
    value=dict(files=files,template=sha(cfg['sample_submission']))
    value=json.loads(json.dumps(value))
    p=Path(output)/'dataset_identity.json'
    if p.exists() and read(p)!=value:raise ValueError('Competition data/template changed since v4 began')
    write_json(p,value)
