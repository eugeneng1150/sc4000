"""CPU GraphSAGE reference, controlled additions, and bounded graph preparation.

Retains the existing project's two-layer, width-32 GraphSAGE approach. Strict
sampling, corrected producer-to-consumer edges, masks and shared manifests mean
this is a controlled reimplementation, not an exact historical reproduction.
"""
import copy
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from study_engine import (read_small, stable_seed, scores, write_json, release_memory,
                          training_records, training_indices)

GNN_COLLECTIONS = ['layout:xla:default', 'layout:xla:random']


def catalog():
    reference = dict(model='gnn', blocks=[], layers=2, width=32, masks=False,
                     context=False, message=True, opcode=True, numeric=True,
                     loss='pairwise', matched=True)
    def exp(name, **changes):
        return dict(reference, name=name, **changes)
    return [exp('G0_reference'), exp('G_add_masks', masks=True),
            exp('G_add_context', context=True), exp('G_full', masks=True, context=True),
            exp('G_no_messages', masks=True, context=True, message=False),
            exp('G_no_opcode_embedding', masks=True, context=True, opcode=False),
            exp('G_no_numeric', masks=True, context=True, numeric=False),
            exp('G_regression', masks=True, context=True, loss='regression'),
            exp('G_one_layer', masks=True, context=True, layers=1)]


def compact_graph(context, local, cfg):
    """Strict node AND edge bounds, before PyTorch allocation; no adjacency lists."""
    ids = local['ids']
    rng = np.random.default_rng(193)
    count = min(len(ids), cfg['gnn_config_nodes'], cfg['gnn_nodes'])
    # Fixed permutations give nested configurable seeds when only the budget increases.
    positions = np.sort(np.random.default_rng(193).permutation(len(ids))[:count])
    if cfg.get('gnn_sampler','uniform') == 'connected':
        # Separate ablation: choose seeds by graph distance to a fixed anchor.
        edges=context['edge'];n=context['nodes']
        distance=np.full(n,n+1,np.int64);distance[ids[positions[0]]]=0
        for hop in range(min(n,cfg['gnn_nodes'])):
            frontier=distance==hop
            touched=edges[np.any(frontier[edges],axis=1)].ravel()
            if not len(touched):break
            new=np.unique(touched[distance[touched]>hop+1]);distance[new]=hop+1
            if np.sum(distance[ids]<=hop+1)>=count:break
        tie=np.random.default_rng(193).permutation(len(ids))
        positions=np.sort(tie[np.argsort(distance[ids[tie]],kind='stable')[:count]])
    seeds = ids[positions]
    keep = np.sort(seeds)
    edges = context['edge']  # [consumer, producer]
    for _ in range(2):
        if len(keep) >= cfg['gnn_nodes']:
            break
        mask = np.zeros(context['nodes'], dtype=bool)
        mask[keep] = True
        candidates = np.unique(edges[np.any(mask[edges], axis=1)].ravel())
        candidates = candidates[~mask[candidates]]
        budget = min(len(candidates), cfg['gnn_nodes']-len(keep))
        chosen = rng.choice(candidates, budget, replace=False)
        keep = np.sort(np.concatenate([keep,chosen]))
    lookup = np.full(context['nodes'], -1, np.int64)
    lookup[keep] = np.arange(len(keep))
    mapped = lookup[edges]
    mapped = mapped[np.all(mapped >= 0,axis=1)]
    before = len(mapped)
    if before > cfg['gnn_edges']:
        chosen = np.sort(rng.choice(before,cfg['gnn_edges'],replace=False))
        mapped = mapped[chosen]
    numeric = context['numeric'][keep].copy()
    numeric = (numeric-numeric.mean(0))/(numeric.std(0)+1e-6)
    config_nodes = lookup[seeds]
    degree=np.bincount(mapped.ravel(),minlength=len(keep)) if len(mapped) else np.zeros(len(keep),int)
    isolated=float(np.mean(degree==0))
    assert len(keep) <= cfg['gnn_nodes'] and len(mapped) <= cfg['gnn_edges']
    return dict(gnn_isolated_fraction=np.array([isolated],np.float32),gnn_numeric=numeric.astype(np.float32), gnn_opcode=context['op'][keep].clip(0,255),
                gnn_src=mapped[:,1], gnn_dst=mapped[:,0],
                gnn_config_nodes=config_nodes, gnn_positions=positions,
                gnn_coverage=np.array([context['nodes'],len(keep),len(ids),len(positions),
                                       len(edges),before,len(mapped)],dtype=np.int64))


class Ranker(nn.Module):
    def __init__(self, numeric_dim, summary_dim, exp):
        super().__init__()
        self.exp = exp
        h = exp['width']
        self.embedding = nn.Embedding(256,16)
        # Same capacity for reference/mask/context additions: unused inputs are zeros.
        self.project = nn.Linear(numeric_dim+16+18+18,h)
        self.layers = nn.ModuleList([nn.Linear(2*h,h) for _ in range(exp['layers'])])
        self.head = nn.Sequential(nn.Linear(h*4+summary_dim,h),nn.ReLU(),nn.Dropout(.05),nn.Linear(h,1))

    def forward(self, graph, layout, summary):
        numeric, opcode, src, dst, config_nodes = graph
        batch, n = len(layout),len(numeric)
        features = numeric if self.exp['numeric'] else torch.zeros_like(numeric)
        op = self.embedding(opcode)
        if not self.exp['opcode']:
            op = torch.zeros_like(op)
        values = torch.zeros((batch,n,18),dtype=torch.float32)
        masks = torch.zeros_like(values)
        valid = layout != -1
        values[:,config_nodes] = torch.where(valid,layout,0).float()/5.
        if self.exp['masks']:
            masks[:,config_nodes] = valid.float()
        h = F.relu(self.project(torch.cat([features.unsqueeze(0).expand(batch,-1,-1),
                    op.unsqueeze(0).expand(batch,-1,-1),values,masks],dim=-1)))
        degree = torch.bincount(dst,minlength=n).clamp_min(1).float()
        for layer in self.layers:
            neighbors = torch.zeros_like(h)
            if self.exp['message'] and len(src):
                neighbors.index_add_(1,dst,h[:,src])
                neighbors /= degree[None,:,None]
            h = h + F.relu(layer(torch.cat([h,neighbors],dim=-1)))
        selected = h[:,config_nodes]
        pooled = torch.cat([h.mean(1),h.max(1).values,selected.mean(1),selected.max(1).values],dim=-1)
        if not self.exp['context']:
            summary = torch.zeros_like(summary)
        return self.head(torch.cat([pooled,summary],dim=-1)).flatten()


def arrays(data):
    graph = tuple(torch.from_numpy(np.asarray(data[k]).copy()) for k in
                  ['gnn_numeric','gnn_opcode','gnn_src','gnn_dst','gnn_config_nodes'])
    # Histograms remain small, finite and label independent.
    summary = np.concatenate([data[k] for k in ['position','opcode','depth','edge']],axis=1).astype(np.float32)
    return graph, data['gnn_layout'].copy(),summary


def loss_function(pred,y,mode):
    if mode == 'regression':
        return F.smooth_l1_loss(pred,y)
    left,right = torch.triu_indices(len(pred),len(pred),offset=1)
    difference = y[left]-y[right]
    eligible = difference.abs() > .02
    if not eligible.any():
        return F.smooth_l1_loss(pred,y)
    return F.relu(.1 - difference[eligible].sign() * (pred[left[eligible]]-pred[right[eligible]])).mean()


def atomic_torch(path,obj):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    torch.save(obj,temp)
    temp.replace(path)


def load_model(path):
    saved = torch.load(path,map_location='cpu',weights_only=True)
    model = Ranker(saved['numeric_dim'],saved['summary_dim'],saved['experiment'])
    model.load_state_dict(saved['state'])
    model.eval()
    return model


def predict_arrays(model,graph,layout,summary,batch):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0,len(layout),batch):
            predictions.append(model(graph,torch.from_numpy(layout[start:start+batch]),
                                      torch.from_numpy(summary[start:start+batch])).numpy())
    return np.concatenate(predictions)


def evaluate(model,job,records,split,save=True,early=False):
    rows,frames = [],[]
    for record in records:
        if record['split'] != split:
            continue
        with np.load(Path(job['scratch'])/record['artifact'],allow_pickle=False) as data:
            graph,layout,summary = arrays(data)
            y,ids = data['y'],data['ids']
            if early:
                chosen = np.sort(np.random.default_rng(331).choice(len(y),min(len(y),job['cfg']['gnn_early_configs']),replace=False))
                layout,summary,y,ids = layout[chosen],summary[chosen],y[chosen],ids[chosen]
            pred = predict_arrays(model,graph,layout,summary,job['cfg']['gnn_predict_batch'])
            rows.append(dict(collection=job['collection'],split=split,graph=record['graph'],family=record['family'],
                experiment=job['experiment']['name'],seed=job['seed'],configurations=len(y),
                total_configurations=record['count'],**scores(y,pred)))
            if save:
                frames.append(pd.DataFrame(dict(graph=record['graph'],config_id=ids,runtime=y,prediction=pred)))
    if not rows:
        raise ValueError('No evaluation graphs')
    if save:
        pd.DataFrame(rows).to_csv(Path(job['output'])/f'{split}_graphs.csv',index=False)
        pd.concat(frames).to_csv(Path(job['output'])/f'{split}_predictions.csv.gz',index=False,compression='gzip')
    return float(np.mean([r['score'] for r in rows]))


def train(job):
    cfg,exp,seed = job['cfg'],job['experiment'],job['seed']
    torch.set_num_threads(cfg['cpu_threads'])
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    output = Path(job['output']);output.mkdir(parents=True,exist_ok=True)
    records = json.loads((Path(job['scratch'])/'prepared.json').read_text())
    training = training_records(records,dict(matched=True,graph_fraction=1.),seed,cfg)
    with np.load(Path(job['scratch'])/training[0]['artifact']) as data:
        graph,layout,summary = arrays(data)
        numeric_dim,summary_dim = graph[0].shape[1],summary.shape[1]
    del graph,layout,summary
    model = Ranker(numeric_dim,summary_dim,exp)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    start = time.monotonic()
    best,stale,first_epoch,best_epoch = -float('inf'),0,1,0
    history = []
    best_state = None
    previous_seconds = 0.
    checkpoint = output/'latest.pt'
    if checkpoint.exists():
        state = torch.load(checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        best,stale,first_epoch,best_epoch = state['best'],state['stale'],state['epoch']+1,state['best_epoch']
        torch.set_rng_state(state['rng'])
        history = state['history']
        previous_seconds = state.get('elapsed_seconds',0.)
        best_state=state['best_state']
        atomic_torch(output/'model.pt',dict(state=best_state,experiment=exp,
                     numeric_dim=numeric_dim,summary_dim=summary_dim))
    train_ids = []
    for record in training:
        ids = training_indices(len(record['indices']),record['graph'],seed,cfg,matched=True)
        train_ids.append(dict(graph=record['graph'],config_ids=np.asarray(record['indices'])[ids].tolist()))
    write_json(output/'training_manifest.json',train_ids)
    for epoch in range(first_epoch,cfg['gnn_epochs']+1):
        if stale >= cfg['gnn_patience']:
            break
        rng = np.random.default_rng(seed+epoch*1009)
        losses=[]
        for index in rng.permutation(len(training)):
            record = training[index]
            with np.load(Path(job['scratch'])/record['artifact'],allow_pickle=False) as data:
                graph,layout,summary = arrays(data)
                ids = training_indices(len(data['y']),record['graph'],seed,cfg,matched=True)
                target = np.log1p(data['y'][ids])
                target = (target-np.median(target)).astype(np.float32)
                order = rng.permutation(len(ids))
                model.train()
                for pos in range(0,len(order),cfg['gnn_batch']):
                    batch = order[pos:pos+cfg['gnn_batch']]
                    chosen=ids[batch]
                    pred=model(graph,torch.from_numpy(layout[chosen]),torch.from_numpy(summary[chosen]))
                    loss=loss_function(pred,torch.from_numpy(target[batch]),exp['loss'])
                    if not torch.isfinite(loss):
                        raise ValueError('Non-finite GNN loss')
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(),5.)
                    optimizer.step();losses.append(float(loss.detach()))
        dev=evaluate(model,job,records,'dev',save=False,early=True)
        if dev > best + 1e-4:
            best,stale,best_epoch=dev,0,epoch
            best_state=copy.deepcopy(model.state_dict())
            atomic_torch(output/'model.pt',dict(state=best_state,experiment=exp,
                         numeric_dim=numeric_dim,summary_dim=summary_dim))
        else:
            stale+=1
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),early_dev_score=dev,best_epoch=best_epoch))
        pd.DataFrame(history).to_csv(output/'history.csv',index=False)
        atomic_torch(checkpoint,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                     best=best,stale=stale,epoch=epoch,best_epoch=best_epoch,
                     history=history,rng=torch.get_rng_state(),best_state=best_state,
                     elapsed_seconds=previous_seconds+time.monotonic()-start))
        print(f'{exp["name"]} seed {seed} epoch {epoch}: dev={dev:.4f}, best={best:.4f}',flush=True)
    fit_seconds=previous_seconds+time.monotonic()-start
    model=load_model(output/'model.pt')
    dev=evaluate(model,job,records,'dev')
    write_json(output/'result.json',dict(status='ok',experiment=exp['name'],seed=seed,
               collection=job['collection'],dev_score=dev,train_rows=sum(len(r['config_ids']) for r in train_ids),
               feature_count=numeric_dim+36,parameters=sum(p.numel() for p in model.parameters()),
               best_epoch=best_epoch,epochs=len(history),fit_seconds=fit_seconds,
               total_seconds=previous_seconds+time.monotonic()-start,version='3.0.0'))


def holdout(job):
    torch.set_num_threads(job['cfg']['cpu_threads'])
    records=json.loads((Path(job['scratch'])/'prepared.json').read_text())
    score=evaluate(load_model(Path(job['output'])/'model.pt'),job,records,'holdout')
    write_json(Path(job['output'])/'holdout_result.json',dict(status='ok',score=score))


def probe(job):
    """Label-free capacity pilot: one full-size batch, real retained graph, dummy targets."""
    from study_engine import graph_context,layout_context,layout_features,selected_rows,metadata
    cfg=job['cfg'];torch.set_num_threads(cfg['cpu_threads']);torch.manual_seed(42)
    rows=[]
    for filename in job['paths']:
        path=Path(filename);context=graph_context(path,cfg);local=layout_context(path,context,cfg)
        compact=compact_graph(context,local,cfg)
        graph=tuple(torch.from_numpy(compact[k].copy()) for k in
                    ['gnn_numeric','gnn_opcode','gnn_src','gnn_dst','gnn_config_nodes'])
        n=metadata(path,'node_config_feat')[0][0]
        layouts=[];summaries=[]
        for _,row in selected_rows(path,'node_config_feat',list(range(min(n,cfg['gnn_batch'])))):
            features=layout_features(row,context,local,cfg)
            layouts.append(row[compact['gnn_positions']].astype(np.int8))
            summaries.append(np.concatenate([features[k] for k in ['position','opcode','depth','edge']]))
        while len(layouts)<cfg['gnn_batch']:
            layouts.append(layouts[0]);summaries.append(summaries[0])
        layout=torch.from_numpy(np.stack(layouts));summary=torch.from_numpy(np.stack(summaries))
        exp=catalog()[3];model=Ranker(graph[0].shape[1],summary.shape[1],exp)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        start=time.monotonic()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss=loss_function(model(graph,layout,summary),torch.linspace(-1,1,len(layout)),'pairwise')
            loss.backward();optimizer.step()
        rows.append(dict(graph=path.stem,seconds_per_batch=(time.monotonic()-start)/3,
                         coverage=compact['gnn_coverage'].tolist(),isolated_fraction=float(compact['gnn_isolated_fraction'][0])))
        del model,optimizer,graph,layout,summary,context,local,compact
        release_memory()
    write_json(Path(job['output'])/'probe.json',rows)
