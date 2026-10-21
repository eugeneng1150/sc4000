"""Predeclared v3 experiments and graph-disjoint development folds."""
import json
from pathlib import Path
import zipfile
import numpy as np
from study_engine import (COLLECTIONS,balanced_order,metadata,stable_seed,digest,family,
                          experiments)
from study_common import default_config as previous_defaults


def default_config():
    cfg=previous_defaults()
    cfg.update(version='3.0.0',folds=[0,1],dev_graphs=12,split_seed=729,
        graph_budgets=[512,1024,2048,4096],data_graph_counts=[24,40,0],data_config_counts=[64,128,256],
        gnn_epochs=20,gnn_patience=5,gnn_batch=16,gnn_predict_batch=16,
        gnn_sampler='uniform',max_matrix_mib=512,scratch_limit_gib=4.,output_limit_gib=12.,
        disk_reserve_gib=4.,probe_minutes=8,probe_ram_gib=3.5,probe_graphs=3,
        layout_more_graphs=128,tile_more_graphs=320,large_fit_configs=384,
        confirmation_top_k=2,checkpoint_minutes=30,session_hours=9.,
        fold_regression_tolerance=.02,sample_submission='',generate_submissions=True)
    return cfg


def profile(nodes,sampler='uniform'):
    return dict(nodes=int(nodes),config_nodes=int(nodes*3//4),edges=int(nodes*8),sampler=sampler)


def effective_config(cfg,exp,collection):
    result=dict(cfg)
    p=exp.get('profile')
    if p:
        result.update(gnn_nodes=p['nodes'],gnn_config_nodes=p['config_nodes'],
                      gnn_edges=p['edges'],gnn_sampler=p['sampler'])
    if exp['model']=='gnn' or exp.get('matched'):
        result.update(gnn_train_graphs=exp.get('train_graphs',24) or 1000000,
                      gnn_configs=exp.get('configs',64))
    else:
        result['fit_configs']=exp.get('configs',256)
    if exp['model']!='gnn':result['gnn_collections']=[]
    return result


def manifest(root,collection,cfg,fold):
    root=Path(root);folder=root.joinpath(*collection.split(':'))
    order=balanced_order(list((folder/'train').glob('*.npz')),cfg['split_seed'])
    ndev=min(cfg['dev_graphs'],max(2,len(order)//5))
    if len(order)<ndev*len(cfg['folds'])+3:
        raise ValueError('Insufficient training graphs for disjoint development folds')
    # Each fold has a disjoint development block. Training may include another fold's dev graphs;
    # those models are NEVER used to predict that other fold during selection.
    slot=cfg['folds'].index(fold)
    dev=order[slot*ndev:(slot+1)*ndev]
    available=[p for p in order if p not in dev]
    if collection in cfg['gnn_collections']:
        train=available
    else:
        cap=cfg['tile_more_graphs'] if collection.startswith('tile') else cfg['layout_more_graphs']
        train=available[:max(cap,cfg['tile_train_graphs'] if collection.startswith('tile') else cfg['layout_train_graphs'])]
    holdout=balanced_order(list((folder/'valid').glob('*.npz')),cfg['split_seed']+1)[:cfg['holdout_graphs']]
    if {p.stem for p in order}&{p.stem for p in holdout}:raise ValueError('Train/valid graph name overlap')
    records=[]
    for split,paths in [('train',train),('dev',dev),('holdout',holdout)]:
        for path in sorted(paths):
            key='config_feat' if collection.startswith('tile') else 'node_config_feat'
            shape,fortran,dtype=metadata(path,key);n=shape[0]
            cap=cfg['train_pool_configs'] if split=='train' else cfg['eval_configs']
            # Fold-independent ordering gives the same candidates whenever a graph has the same role.
            ids=np.sort(np.random.default_rng(stable_seed(f'{cfg["split_seed"]}/{collection}/{path.stem}'))
                        .choice(n,min(n,cap),replace=False))
            with zipfile.ZipFile(path) as z:source=[(i.filename,i.CRC,i.file_size) for i in z.infolist()]
            records.append(dict(split=split,graph=path.stem,family=family(path.stem),
                relative_path=str(path.relative_to(root)),count=n,indices=ids.tolist(),
                source=digest(source),fortran=bool(fortran),row_bytes=int(np.prod(shape[1:]))*dtype.itemsize))
    return dict(collection=collection,fold=fold,records=records,
                split_note='graph-disjoint development within fold; official valid is an already-inspected audit')


def gnn_exp(name,recipe,nodes=512,sampler='uniform',graphs=24,configs=64,stage='coverage',**changes):
    from study_gnn import catalog
    exp=dict(next(e for e in catalog() if e['name']==recipe))
    exp.update(name=name,profile=profile(nodes,sampler),train_graphs=graphs,configs=configs,stage=stage,**changes)
    return exp


def coverage_plan(cfg):
    result=[]
    for nodes in cfg['graph_budgets']:
        result.extend([gnn_exp(f'G_reference_n{nodes}','G0_reference',nodes),
                       gnn_exp(f'G_full_n{nodes}','G_full',nodes)])
    return result


def followup_plan(cfg,selected_profile):
    n=selected_profile['nodes'];result=[]
    for suffix,recipe in [('masks','G_add_masks'),('context','G_add_context'),('no_messages','G_no_messages')]:
        result.append(gnn_exp(f'G_{suffix}_n{n}',recipe,n,stage='component_controls'))
    result.append(gnn_exp(f'G_connected_n{n}','G_full',n,sampler='connected',stage='sampler_control'))
    full=next(e for e in experiments(False) if e['name']=='P_full')
    for graphs in cfg['data_graph_counts']:
        for count in cfg['data_config_counts']:
            tag=f'g{graphs or "all"}_c{count}'
            if graphs!=24 or count!=64:
                result.append(gnn_exp(f'G_data_{tag}','G_full',n,graphs=graphs,configs=count,stage='training_data'))
            result.append(dict(full,name=f'H_matched_{tag}',train_graphs=graphs,configs=count,
                               matched=True,stage='matched_training_data'))
    return result


def tabular_plan(collection,cfg):
    tile=collection.startswith('tile');catalog=experiments(tile)
    base=next(e for e in catalog if e['name']=='B0_basic_hgb')
    full=next(e for e in catalog if e['name']=='P_full')
    small=cfg['tile_train_graphs'] if tile else cfg['layout_train_graphs']
    large=cfg['tile_more_graphs'] if tile else cfg['layout_more_graphs']
    def trial(name,exp,graphs,configs,**changes):
        return dict(exp,name=name,train_graphs=graphs,configs=configs,stage='tabular',**changes)
    return [trial('B0_basic_hgb',base,small,cfg['fit_configs']),
            trial('T_full_small',full,small,cfg['fit_configs']),
            trial('T_more_graphs',full,large,cfg['fit_configs']),
            trial('T_more_configs',full,small,cfg['large_fit_configs']),
            trial('T_more_both',full,large,cfg['large_fit_configs']),
            trial('T_deeper_hgb',full,large,cfg['fit_configs'],iterations=300,leaves=63,min_leaf=10,learning_rate=.04),
            trial('T_extra_trees',full,large,cfg['fit_configs'],model='extra_trees')]
