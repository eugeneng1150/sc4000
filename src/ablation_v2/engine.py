"""Independent, bounded-memory SC4000 experiments. Embedded in the Colab notebook.

Only original NPZ files and compact sampled feature matrices are read. No full
configuration array is decompressed into memory or an expanded disk cache.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback
import zipfile

import joblib
import numpy as np
import pandas as pd
import psutil
from scipy.stats import kendalltau, rankdata
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

VERSION = '2.0.0'
GIB = 1024 ** 3
COLLECTIONS = ['layout:xla:default', 'layout:xla:random',
               'layout:nlp:default', 'layout:nlp:random', 'tile:xla']
BLOCKS = ['basic', 'masked', 'position', 'opcode', 'depth', 'edge', 'topology', 'wl']


def stable_seed(value):
    return int(hashlib.sha256(str(value).encode()).hexdigest()[:8], 16)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str))
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def release_memory():
    gc.collect()
    if sys.platform == 'linux':
        try:
            import ctypes
            ctypes.CDLL('libc.so.6').malloc_trim(0)
        except (OSError, AttributeError):
            pass


def guard(required=0, reserve=1.5 * GIB):
    if psutil.virtual_memory().available < required + reserve:
        raise MemoryError(f'Insufficient available RAM for {required / GIB:.2f} GiB allocation')


def header(stream):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, order, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version in ((2, 0), (3, 0)):
        shape, order, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f'Unsupported NPY version: {version}')
    if np.dtype(dtype).hasobject:
        raise ValueError('Object arrays are not allowed')
    return tuple(shape), order, np.dtype(dtype)


def metadata(path, key):
    with zipfile.ZipFile(path) as z, z.open(key + '.npy') as f:
        return header(f)


def exact_read(f, size):
    data = bytearray(size)
    view = memoryview(data)
    position = 0
    while position < size:
        count = f.readinto(view[position:])
        if not count:
            raise EOFError('Truncated NPZ member')
        position += count
    return data


def selected_rows(path, key, indices, chunk_bytes=16 * 1024 ** 2):
    """Yield one selected row at a time; discard skipped bytes in bounded chunks.

    Fortran arrays fail explicitly: silently loading their full payload is unsafe.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) and (np.any(np.diff(indices) <= 0) or indices[0] < 0):
        raise ValueError('Indices must be sorted, unique and nonnegative')
    with zipfile.ZipFile(path) as z, z.open(key + '.npy') as f:
        shape, order, dtype = header(f)
        if order:
            raise ValueError('Fortran-order configuration array: safe streaming requires C order')
        if len(indices) and indices[-1] >= shape[0]:
            raise IndexError('Configuration index out of range')
        size = math.prod(shape[1:]) * dtype.itemsize
        guard(size * 12)
        position = 0
        for index in indices:
            skip = (int(index) - position) * size
            while skip:
                chunk = f.read(min(skip, chunk_bytes))
                if not chunk:
                    raise EOFError('Truncated NPZ while skipping rows')
                skip -= len(chunk)
            raw = exact_read(f, size)
            row = np.frombuffer(raw, dtype=dtype).reshape(shape[1:])
            yield int(index), row
            del row, raw
            position = int(index) + 1


def read_small(path, key, limit=256 * 1024 ** 2):
    shape, _, dtype = metadata(path, key)
    size = math.prod(shape) * dtype.itemsize
    if size > limit:
        raise MemoryError(f'Static member {key} exceeds {limit / 1024**2:.0f} MiB guard')
    guard(size * 3)
    with np.load(path, allow_pickle=False) as data:
        return data[key]


def family(stem):
    for name in ['resnet', 'bert', 'albert', 'inception', 'efficientnet', 'transformer',
                 'retinanet', 'mask_rcnn', 'mnasnet', 'alexnet', 'magenta', 'brax',
                 'ncf', 'xception', 'electra', 'trax', 'unet', 'mlperf']:
        if name in stem.lower():
            return name
    if len(stem) >= 24 and all(c in '0123456789abcdef' for c in stem[:24]):
        return 'hashed_graph'
    return stem.split('_')[0].split('.')[0].split('-')[0]


def balanced_order(paths, seed):
    rng = np.random.default_rng(seed)
    groups = {}
    for path in sorted(paths):
        groups.setdefault(family(path.stem), []).append(path)
    for values in groups.values():
        rng.shuffle(values)
    names = sorted(groups)
    rng.shuffle(names)
    result = []
    while any(groups.values()):
        for name in names:
            if groups[name]:
                result.append(groups[name].pop())
    return result


def make_manifest(root, collection, cfg):
    folder = Path(root).joinpath(*collection.split(':'))
    train = sorted((folder / 'train').glob('*.npz'))
    valid = sorted((folder / 'valid').glob('*.npz'))
    if len(train) < 5 or not valid:
        raise ValueError(f'{collection}: need >=5 training graphs and >=1 official validation graph')
    if {p.stem for p in train} & {p.stem for p in valid}:
        raise ValueError('Graph names overlap between train and official validation')
    tile = collection.startswith('tile')
    order = balanced_order(train, cfg['split_seed'])
    ndev = min(cfg['dev_graphs'], max(2, len(order) // 5))
    dev = order[:ndev]
    fit = order[ndev:ndev + (cfg['tile_train_graphs'] if tile else cfg['layout_train_graphs'])]
    holdout = balanced_order(valid, cfg['split_seed'] + 1)[:cfg['holdout_graphs']]
    records = []
    for split, paths in [('train', fit), ('dev', dev), ('holdout', holdout)]:
        for path in sorted(paths):
            # ZIP CRCs bind resume to the source payload, independent of mount path/mtime.
            with zipfile.ZipFile(path) as z:
                source = [(i.filename, i.CRC, i.file_size) for i in z.infolist()]
            key = 'config_feat' if tile else 'node_config_feat'
            shape, order, dtype = metadata(path, key)
            n = shape[0]
            cap = cfg['train_pool_configs'] if split == 'train' else cfg['eval_configs']
            ids = np.sort(np.random.default_rng(stable_seed(f'{cfg["split_seed"]}/{collection}/{path.stem}'))
                          .choice(n, min(n, cap), replace=False))
            records.append(dict(split=split, graph=path.stem, family=family(path.stem),
                                relative_path=str(path.relative_to(root)), count=n,
                                indices=ids.tolist(), source=digest(source),
                                fortran=bool(order), row_bytes=math.prod(shape[1:]) * dtype.itemsize))
    return dict(collection=collection, records=records,
                split_note='dev graphs from official train; holdout from official valid; no test labels')


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    if not a.size:
        return np.zeros(4, dtype=np.float32)
    return np.array([a.mean(), a.std(), a.min(), a.max()], dtype=np.float32)


def hash_counts(values, bins):
    return np.bincount(np.asarray(values, dtype=np.uint64).ravel().astype(np.int64) % bins,
                       minlength=bins).astype(np.float32)


def graph_context(path, cfg):
    op = read_small(path, 'node_opcode').astype(np.int64)
    feat = read_small(path, 'node_feat').astype(np.float32)
    edge = read_small(path, 'edge_index').astype(np.int64)
    n = len(op)
    if not n or edge.ndim != 2 or edge.shape[1] != 2 or feat.shape[0] != n:
        raise ValueError('Invalid graph arrays')
    if edge.size and (edge.min() < 0 or edge.max() >= n):
        raise ValueError('Edge node index out of range')
    # Official NPZ convention: [consumer, producer], opposite to common edge lists.
    consumer, producer = edge[:, 0], edge[:, 1]
    indegree = np.bincount(consumer, minlength=n)
    outdegree = np.bincount(producer, minlength=n)
    # Kahn traversal using sorted CSR adjacency, avoiding Python lists per edge.
    order = np.argsort(producer, kind='stable')
    targets = consumer[order]
    offsets = np.concatenate([[0], np.cumsum(outdegree)])
    left = indegree.copy()
    queue = np.empty(n, dtype=np.int64)
    roots = np.flatnonzero(left == 0)
    queue[:len(roots)] = roots
    tail, head = len(roots), 0
    depth = np.zeros(n, dtype=np.int64)
    while head < tail:
        u = queue[head]
        head += 1
        for v in targets[offsets[u]:offsets[u + 1]]:
            depth[v] = max(depth[v], depth[u] + 1)
            left[v] -= 1
            if left[v] == 0:
                queue[tail] = v
                tail += 1
    feat = np.nan_to_num(feat, nan=0, posinf=0, neginf=0)
    signed = np.sign(feat) * np.log1p(np.abs(feat))
    # Fixed width preserves compatibility across graphs; upstream schema is checked per collection.
    basic = np.concatenate([np.log1p([n, len(edge)]), signed.mean(0), signed.std(0),
                            signed.min(0), signed.max(0), hash_counts(op, 32) / n]).astype(np.float32)
    topology = np.concatenate([stats(np.log1p(indegree)), stats(np.log1p(outdegree)),
                               stats(np.log1p(depth)), [np.mean(left > 0)]]).astype(np.float32)
    labels = op.copy() % 64
    wl_parts = []
    for _ in range(2):
        incoming = np.bincount(consumer, weights=(labels[producer] + 1) * 37,
                               minlength=n).astype(np.int64)
        outgoing = np.bincount(producer, weights=(labels[consumer] + 1) * 53,
                               minlength=n).astype(np.int64)
        labels = (labels * 1009 + incoming * 9176 + outgoing * 6361) % 64
        wl_parts.append(hash_counts(labels, 64) / n)
    # This bounded hashed neighborhood signature is our implementation, not exact WL isomorphism testing.
    return dict(op=op, numeric=signed, edge=edge, depth=depth, basic=basic, topology=topology,
                wl=np.concatenate(wl_parts), nodes=n, cycle_fraction=float(np.mean(left > 0)))


def layout_context(path, context, cfg):
    ids = read_small(path, 'node_config_ids').astype(np.int64)
    n = context['nodes']
    if not len(ids) or ids.min() < 0 or ids.max() >= n or len(np.unique(ids)) != len(ids):
        raise ValueError('Invalid configurable node IDs')
    lookup = np.full(n, -1, dtype=np.int64)
    lookup[ids] = np.arange(len(ids))
    pairs = lookup[context['edge']]
    pairs = pairs[np.all(pairs >= 0, axis=1)]
    # Use a deterministic, configuration-independent edge cap.
    if len(pairs) > cfg['edge_cap']:
        chosen = np.sort(np.random.default_rng(913).choice(len(pairs), cfg['edge_cap'], replace=False))
        pairs = pairs[chosen]
    d = context['depth'][ids]
    bins = np.minimum(3, (d * 4 // max(1, int(context['depth'].max()) + 1)))
    return dict(ids=ids, opcode=context['op'][ids], depth=bins, pairs=pairs)


def layout_features(row, context, local, cfg):
    a = np.asarray(row)
    if a.shape != (len(local['ids']), 18):
        raise ValueError(f'Expected layout shape (configurable_nodes,18), got {a.shape}')
    if not np.isfinite(a).all() or np.any(a != np.floor(a)) or a.min() < -1 or a.max() > 5:
        raise ValueError('Layout values must be categorical indices -1..5')
    mask = a != -1
    zero = np.where(mask, a, 0).astype(np.float32)
    basic_config = np.concatenate([zero.mean(0), zero.std(0), zero.min(0), zero.max(0),
                                   [1 - mask.mean(), np.log1p(len(a))]])
    count = mask.sum(0)
    means = zero.sum(0) / np.maximum(1, count)
    variance = (zero ** 2).sum(0) / np.maximum(1, count) - means ** 2
    minimum = np.where(mask, a, 6).min(0)
    maximum = np.where(mask, a, -1).max(0)
    minimum[count == 0], maximum[count == 0] = 0, 0
    masked = np.concatenate([means, np.sqrt(np.maximum(0, variance)), minimum, maximum,
                             count / len(a)]).astype(np.float32)
    position = np.zeros((18, 7), dtype=np.float32)
    opcode_hist = np.zeros(cfg['opcode_bins'], dtype=np.float32)
    depth_hist = np.zeros(cfg['depth_bins'], dtype=np.float32)
    # Include padding as its own category; never treat axis ID as a physical magnitude here.
    categories = a.astype(np.int64) + 1
    for p in range(18):
        token = categories[:, p]
        position[p] = np.bincount(token, minlength=7) / len(a)
        opcode_hist += hash_counts(local['opcode'] * 1009 + p * 37 + token * 9176, cfg['opcode_bins'])
        depth_hist += hash_counts(local['depth'] * 1009 + p * 37 + token * 9176, cfg['depth_bins'])
    denom = max(1, len(a) * 18)
    pairs = local['pairs']
    edge_hist = np.zeros(cfg['edge_bins'], dtype=np.float32)
    if len(pairs):
        # Categorical co-occurrence, not an assertion of physical layout compatibility.
        for p in [0, 6, 12]:
            codes = categories[pairs[:, 0], p] * 7 + categories[pairs[:, 1], p]
            edge_hist += hash_counts(codes + p * 1009, cfg['edge_bins'])
        edge_hist /= len(pairs) * 3
    return dict(basic=np.concatenate([context['basic'], basic_config]).astype(np.float32),
                masked=masked, position=position.ravel(), opcode=opcode_hist / denom,
                depth=depth_hist / denom, edge=edge_hist,
                topology=context['topology'], wl=context['wl'])


def tile_features(row, context, cfg):
    a = np.asarray(row, dtype=np.float32).ravel()
    if len(a) != 24:
        raise ValueError(f'Expected 24 tile features, got {len(a)}')
    a = np.nan_to_num(a, nan=0, posinf=0, neginf=0)
    signed = np.sign(a) * np.log1p(np.abs(a))
    # Tile-specific transformed values and interactions; layout categories do not apply.
    interactions = np.outer(signed[:8], signed[:8])[np.triu_indices(8, 1)]
    return dict(basic=np.concatenate([context['basic'], a, stats(a)]).astype(np.float32),
                masked=signed, position=interactions.astype(np.float32),
                opcode=np.zeros(1, np.float32), depth=np.zeros(1, np.float32),
                edge=np.zeros(1, np.float32), topology=context['topology'], wl=context['wl'])


def extract_graph(root, record, collection, cfg, target):
    path = Path(root) / record['relative_path']
    context = graph_context(path, cfg)
    tile = collection.startswith('tile')
    local = None if tile else layout_context(path, context, cfg)
    gnn_graph = None
    if collection in cfg.get('gnn_collections', []):
        from study_gnn import compact_graph
        gnn_graph = compact_graph(context, local, cfg)
    key = 'config_feat' if tile else 'node_config_feat'
    runtime = read_small(path, 'config_runtime').astype(np.float64)
    if tile:
        normalizer = read_small(path, 'config_runtime_normalizers').astype(np.float64)
        if np.any(normalizer <= 0) or len(normalizer) != len(runtime):
            raise ValueError('Invalid tile runtime normalizers')
        runtime /= normalizer
    ids = np.asarray(record['indices'], dtype=np.int64)
    if len(runtime) != record['count'] or not np.isfinite(runtime[ids]).all() or np.any(runtime[ids] <= 0):
        raise ValueError('Invalid runtime labels')
    arrays, fingerprints = None, []
    layouts = None
    if gnn_graph is not None:
        size = len(ids)*len(gnn_graph['gnn_positions'])*18
        guard(size*3)
        layouts = np.empty((len(ids),len(gnn_graph['gnn_positions']),18),np.int8)
    for j, (index, row) in enumerate(selected_rows(path, key, ids)):
        features = tile_features(row, context, cfg) if tile else layout_features(row, context, local, cfg)
        if arrays is None:
            required = len(ids) * sum(v.size for v in features.values()) * 4
            if required > cfg['max_graph_features_mib'] * 1024**2:
                raise MemoryError('Compact graph feature matrix exceeds configured limit')
            guard(required * 3)
            arrays = {k: np.empty((len(ids), len(v)), np.float32) for k, v in features.items()}
        for name, values in features.items():
            if not np.isfinite(values).all():
                raise ValueError('Non-finite features')
            arrays[name][j] = values
        fingerprints.append(hashlib.blake2b(np.ascontiguousarray(row).tobytes(), digest_size=16).hexdigest())
        if layouts is not None:
            layouts[j] = row[gnn_graph['gnn_positions']].astype(np.int8)
    if arrays is None:
        raise ValueError('No configurations selected')
    if gnn_graph is not None:
        arrays.update(gnn_graph)
        arrays['gnn_layout'] = layouts
    np.savez_compressed(target, **arrays, y=runtime[ids], ids=ids,
                        fingerprints=np.array(fingerprints, dtype='U32'))
    return dict(sampled_configs=len(ids), unique_configs=len(set(fingerprints)),
                cycle_fraction=context['cycle_fraction'],
                gnn_coverage=None if gnn_graph is None else gnn_graph['gnn_coverage'].tolist())


def prepare(root, manifest, cfg, scratch, output):
    """Compact artifacts only, with checkpoints per graph and no raw-array cache."""
    scratch, output = Path(scratch), Path(output)
    scratch.mkdir(parents=True, exist_ok=True)
    failures, prepared = [], []
    for k, record in enumerate(manifest['records']):
        target = scratch / f'g{k:05d}.npz'
        status = scratch / f'g{k:05d}.json'
        signature = digest(dict(record=record, version=VERSION, features={
            key: cfg[key] for key in ['opcode_bins', 'depth_bins', 'edge_bins', 'edge_cap']}))
        try:
            if target.exists() and status.exists() and json.loads(status.read_text()).get('signature') == signature:
                details = json.loads(status.read_text())['details']
            else:
                print(f'Extract {k+1}/{len(manifest["records"])} {record["split"]} {record["graph"]}', flush=True)
                details = extract_graph(root, record, manifest['collection'], cfg, target)
                write_json(status, dict(signature=signature, details=details))
            prepared.append(dict(record, artifact=target.name, **details))
        except Exception as exc:
            target.unlink(missing_ok=True)
            failures.append(dict(graph=record['graph'], split=record['split'], error=str(exc)))
            print(f'Graph excluded for every experiment: {record["graph"]}: {exc}', flush=True)
        release_memory()
        if sum(p.stat().st_size for p in scratch.glob('*.npz')) > cfg['scratch_limit_gib'] * GIB:
            raise MemoryError('Compact scratch feature disk limit reached')
    write_json(output / 'extraction_failures.json', failures)
    write_json(output / 'coverage.json', prepared)
    write_json(scratch / 'prepared.json', prepared)
    counts = {s: sum(r['split'] == s for r in prepared) for s in ['train', 'dev', 'holdout']}
    if counts['train'] < 3 or counts['dev'] < 2 or counts['holdout'] < 1:
        raise ValueError(f'Insufficient shared coverage after extraction: {counts}')
    if len(failures) / max(1, len(manifest['records'])) > cfg['max_excluded_fraction']:
        raise ValueError('Too many excluded graphs; inspect extraction_failures.json')
    write_json(scratch / 'ready.json', dict(counts=counts, complete=True))


def experiments(tile=False):
    def trial(name, blocks, **kwargs):
        return dict(name=name, blocks=blocks, model='hgb', target='centered_log',
                    weighting='graph', dedup=False, row_fraction=1., graph_fraction=1.,
                    **kwargs) if not kwargs else dict(dict(name=name, blocks=blocks, model='hgb',
                    target='centered_log', weighting='graph', dedup=False,
                    row_fraction=1., graph_fraction=1.), **kwargs)
    base = ['basic']
    full = ['basic', 'masked', 'position', 'topology', 'wl'] if tile else [
        'basic', 'masked', 'position', 'opcode', 'depth', 'edge']
    result = [trial('B0_basic_hgb', base), trial('B1_masked_hgb', ['basic', 'masked']),
              trial('P0_position', ['basic', 'masked', 'position']), trial('P_full', full)]
    if not tile:
        result += [trial('P_opcode', ['basic', 'masked', 'position', 'opcode']),
                   trial('P_depth', ['basic', 'masked', 'position', 'depth']),
                   trial('P_opcode_depth', ['basic', 'masked', 'position', 'opcode', 'depth']),
                   trial('P_edge', ['basic', 'masked', 'position', 'edge'])]
    result += [trial('S_topology', base + ['topology']), trial('S_wl', base + ['wl']),
               trial('P_full_static', full + [b for b in ['topology', 'wl'] if b not in full])]
    if tile:
        # Avoid a duplicate trial: P_full already contains all tile feature blocks.
        result = [r for r in result if r['name'] != 'P_full_static']
    for block in full:
        if block != 'basic':
            result.append(trial('A_without_' + block, [b for b in full if b != block]))
    result += [trial('C_unweighted', full, weighting='none'),
               trial('C_family_weight', full, weighting='family'),
               trial('C_log_target', full, target='log'),
               trial('C_percentile_target', full, target='percentile'),
               trial('C_deduplicate', full, dedup=True),
               trial('C_half_configs', full, row_fraction=.5),
               trial('C_half_graphs', full, graph_fraction=.5),
               trial('M_extra_trees', full, model='extra_trees'),
               trial('M_mlp', full, model='mlp'),
               trial('M_linear_regression', full, model='ridge'),
               trial('M_linear_pairwise', full, model='pairwise')]
    return result


def feature_matrix(data, blocks):
    return np.concatenate([data[b] for b in blocks], axis=1).astype(np.float32, copy=False)


def training_records(records, exp, seed, cfg):
    records = [r for r in records if r['split'] == 'train']
    if exp.get('matched'):
        records = sorted(records,key=lambda r: stable_seed(f'gnn-train/{r["graph"]}'))[:cfg['gnn_train_graphs']]
    if exp['graph_fraction'] < 1:
        order = sorted(records, key=lambda r: stable_seed(f'{seed}/{r["graph"]}'))
        records = order[:max(2, math.ceil(len(order) * exp['graph_fraction']))]
    return records


def training_indices(n, graph, seed, cfg, matched=False, fraction=1.):
    cap = cfg['gnn_configs'] if matched else cfg['fit_configs']
    take = min(n,max(1,int(cap*fraction)))
    return np.sort(np.random.default_rng(stable_seed(f'{seed}/{graph}')).permutation(n)[:take])


def training_data(scratch, records, exp, seed, cfg):
    records = training_records(records,exp,seed,cfg)
    xs, ys, weights, groups = [], [], [], []
    family_counts = {}
    for r in records:
        family_counts[r['family']] = family_counts.get(r['family'], 0) + 1
    total_bytes = 0
    for group, record in enumerate(records):
        with np.load(Path(scratch) / record['artifact'], allow_pickle=False) as data:
            n = len(data['y'])
            selected = training_indices(n,record['graph'],seed,cfg,exp.get('matched',False),exp['row_fraction'])
            x = feature_matrix(data, exp['blocks'])[selected]
            y = data['y'][selected].astype(np.float64)
            if exp['dedup']:
                _, inverse = np.unique(data['fingerprints'][selected], return_inverse=True)
                unique_ids = np.unique(inverse)
                x = np.stack([x[np.flatnonzero(inverse == j)[0]] for j in unique_ids])
                y = np.array([np.median(y[inverse == j]) for j in unique_ids])
            if exp['target'] == 'percentile':
                y = (rankdata(y) - 1) / max(1, len(y) - 1)
            else:
                y = np.log1p(y)
                if exp['target'] == 'centered_log':
                    y -= np.median(y)
            w = np.ones(len(y), np.float64)
            if exp['weighting'] in ['graph', 'family']:
                w /= len(y)
            if exp['weighting'] == 'family':
                w /= family_counts[record['family']]
            total_bytes += x.nbytes
            if total_bytes > cfg['max_matrix_mib'] * 1024**2:
                raise MemoryError('Training matrix exceeds configured bound')
            guard(total_bytes * 5)
            xs.append(x); ys.append(y); weights.append(w)
            groups.append(np.full(len(y), group, np.int32))
    x, y, w, g = np.concatenate(xs), np.concatenate(ys), np.concatenate(weights), np.concatenate(groups)
    return x, y, w / w.mean(), g


def fit_model(x, y, weights, groups, exp, seed, cfg):
    kind = exp['model']
    scaler = None
    if kind in ['ridge', 'pairwise', 'mlp']:
        scaler = StandardScaler()
        x = scaler.fit_transform(x).astype(np.float32)
    if kind == 'hgb':
        model = HistGradientBoostingRegressor(max_iter=cfg['hgb_iterations'], learning_rate=.06,
                    max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=.1,
                    early_stopping=False, random_state=seed)
        model.fit(x, y, sample_weight=weights)
    elif kind == 'extra_trees':
        model = ExtraTreesRegressor(n_estimators=cfg['extra_trees'], max_depth=14,
                                     min_samples_leaf=8, n_jobs=cfg['cpu_threads'], random_state=seed)
        model.fit(x, y, sample_weight=weights)
    elif kind == 'ridge':
        model = Ridge(alpha=10.)
        model.fit(x, y, sample_weight=weights)
    elif kind == 'mlp':
        model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=cfg['mlp_epochs'],
                             batch_size=256, early_stopping=False, random_state=seed)
        # sklearn versions differ in weight support. Weighted resampling is explicit and stable.
        rng = np.random.default_rng(seed)
        rows = rng.choice(len(y), len(y), replace=True, p=weights / weights.sum())
        model.fit(x[rows], y[rows])
    elif kind == 'pairwise':
        # A separate linear comparator, not the existing GNN or a novelty claim.
        rng = np.random.default_rng(seed)
        left, right, pair_weights = [], [], []
        for group in np.unique(groups):
            ids = np.flatnonzero(groups == group)
            i = rng.choice(ids, cfg['pairs_per_graph'])
            j = rng.choice(ids, cfg['pairs_per_graph'])
            use = y[i] != y[j]
            i, j = i[use], j[use]
            left.extend(i); right.extend(j)
            pair_weights.extend(np.repeat(weights[ids].mean(), len(i)))
        left, right = np.asarray(left, dtype=int), np.asarray(right, dtype=int)
        if not len(left):
            raise ValueError('No unequal-runtime training pairs')
        difference = x[left] - x[right]
        labels = y[left] > y[right]
        # Mirrored pairs guarantee both classes and force antisymmetric supervision.
        difference = np.concatenate([difference, -difference])
        labels = np.concatenate([labels, ~labels])
        pw = np.tile(pair_weights, 2).astype(float)
        model = SGDClassifier(loss='log_loss', alpha=.001, max_iter=300, tol=1e-4,
                               fit_intercept=False, random_state=seed)
        model.fit(difference, labels, sample_weight=pw / pw.mean())
    else:
        raise ValueError(kind)
    return dict(model=model, scaler=scaler, experiment=exp)


def predict(bundle, x):
    if bundle['scaler'] is not None:
        x = bundle['scaler'].transform(x)
    if bundle['experiment']['model'] == 'pairwise':
        return bundle['model'].decision_function(x)
    return bundle['model'].predict(x)


def scores(y, pred, tile=False):
    """Exact tau-b on the sampled set; never call these Kaggle/public LB scores."""
    if not np.isfinite(pred).all():
        raise ValueError('Non-finite predictions')
    tau = kendalltau(y, pred, variant='b').statistic
    tau = 0. if not np.isfinite(tau) else float(tau)
    order = np.argsort(pred, kind='stable')
    regret = float(y[order[:5]].min() / y.min() - 1)
    return dict(score=1 - regret if tile else tau, kendall_tau_b=tau,
                top5_regret=regret, distinct_predictions=int(len(np.unique(pred))))


def evaluate(bundle, scratch, records, split, collection, output, seed, exp_name):
    rows, prediction_rows = [], []
    for record in records:
        if record['split'] != split:
            continue
        with np.load(Path(scratch) / record['artifact'], allow_pickle=False) as data:
            x = feature_matrix(data, bundle['experiment']['blocks'])
            y, ids = data['y'], data['ids']
            pred = predict(bundle, x)
            metric = scores(y, pred, collection.startswith('tile'))
            rows.append(dict(collection=collection, split=split, graph=record['graph'],
                             family=record['family'], experiment=exp_name, seed=seed,
                             configurations=len(y), total_configurations=record['count'], **metric))
            prediction_rows.append(pd.DataFrame(dict(graph=record['graph'], config_id=ids,
                                                      runtime=y, prediction=pred)))
    if not rows:
        raise ValueError(f'No {split} graphs')
    pd.DataFrame(rows).to_csv(Path(output) / f'{split}_graphs.csv', index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(Path(output) / f'{split}_predictions.csv.gz',
                                                      index=False, compression='gzip')
    return float(np.mean([r['score'] for r in rows]))


def run_trial(job):
    if job['experiment']['model'] == 'gnn':
        from study_gnn import train
        return train(job)
    output = Path(job['output']); output.mkdir(parents=True, exist_ok=True)
    cfg, exp, seed = job['cfg'], job['experiment'], job['seed']
    records = json.loads((Path(job['scratch']) / 'prepared.json').read_text())
    start = time.monotonic()
    with threadpool_limits(limits=cfg['cpu_threads']):
        x, y, w, groups = training_data(job['scratch'], records, exp, seed, cfg)
        fit_start = time.monotonic()
        bundle = fit_model(x, y, w, groups, exp, seed, cfg)
        fit_seconds = time.monotonic() - fit_start
        train_rows, feature_count = x.shape
        del x, y, w, groups
        release_memory()
        dev = evaluate(bundle, job['scratch'], records, 'dev', job['collection'], output, seed, exp['name'])
        joblib.dump(bundle, output / 'model.joblib', compress=3)
    write_json(output / 'result.json', dict(status='ok', experiment=exp['name'], seed=seed,
               collection=job['collection'], dev_score=dev, train_rows=train_rows,
               feature_count=feature_count, fit_seconds=fit_seconds,
               total_seconds=time.monotonic()-start, version=VERSION))


def evaluate_holdout(job):
    if job['experiment']['model'] == 'gnn':
        from study_gnn import holdout
        return holdout(job)
    bundle = joblib.load(Path(job['output']) / 'model.joblib')
    records = json.loads((Path(job['scratch']) / 'prepared.json').read_text())
    with threadpool_limits(limits=job['cfg']['cpu_threads']):
        score = evaluate(bundle, job['scratch'], records, 'holdout', job['collection'],
                         job['output'], job['seed'], job['experiment']['name'])
    write_json(Path(job['output']) / 'holdout_result.json', dict(status='ok', score=score))


def worker(job_path):
    job = json.loads(Path(job_path).read_text())
    try:
        if job['action'] == 'prepare':
            prepare(job['root'], job['manifest'], job['cfg'], job['scratch'], job['output'])
        elif job['action'] == 'trial':
            run_trial(job)
        elif job['action'] == 'holdout':
            evaluate_holdout(job)
        elif job['action'] == 'inference':
            from study_submission import infer_graph
            infer_graph(job)
        else:
            raise ValueError(job['action'])
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    sys.exit(worker(args.job))
