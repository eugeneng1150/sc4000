"""Isolated v4 workers; existing v3 estimators and feature definitions are unchanged."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
from study_engine import write_json
import study_engine as engine


def pilot(job):
    records = json.loads((Path(job['scratch'])/'prepared.json').read_text())
    exp = dict(job['experiment'],iterations=job['cfg']['pilot_iterations'])
    with threadpool_limits(limits=job['cfg']['cpu_threads']):
        x,y,w,g = engine.training_data(job['scratch'],records,exp,job['seed'],job['cfg'])
        start = time.monotonic()
        engine.fit_model(x,y,w,g,exp,job['seed'],job['cfg'])
    elapsed = time.monotonic()-start
    write_json(Path(job['output'])/'pilot.json',dict(rows=len(x),features=x.shape[1],iterations=exp['iterations'],
        fit_seconds=elapsed,rough_full_fit_minutes=elapsed*job['experiment'].get('iterations',300)/exp['iterations']/60,
        note='Linear extrapolation only; excludes extraction, development evaluation and test inference'))


def inference(job):
    """Infer only selected members, retaining full rank vectors for future reuse."""
    import joblib
    from study_submission import ranking
    from study_plan import effective_config
    cfg=job['cfg'];tile=job['collection'].startswith('tile');path=Path(job['path'])
    members=[];profiles={}
    for name,source in job['sources'].items():
        exp=source['experiment']
        for fold in cfg['folds']:
            for seed in cfg['seeds']:
                p=Path(source['directory'])/'folds'/str(fold)/'trials'/name/str(seed)
                if exp['model']=='gnn':
                    import torch
                    from study_gnn import load_model
                    torch.set_num_threads(cfg['cpu_threads'])
                    model=load_model(p/'model.pt');key=engine.digest(exp['profile'])
                    profiles[key]=effective_config(cfg,exp,job['collection'])
                else: model=joblib.load(p/'model.joblib');key=None
                members.append((name,model,key))
    n=engine.metadata(path,'config_feat' if tile else 'node_config_feat')[0][0]
    engine.guard(n*8*(len(members)+8))
    predictions=np.empty((len(members),n),np.float64)
    context=engine.graph_context(path,cfg);local=None if tile else engine.layout_context(path,context,cfg)
    bounded={};graphs={}
    if profiles:
        from study_gnn import compact_graph,predict_arrays
        for key,settings in profiles.items():
            bounded[key]=compact_graph(context,local,settings)
            graphs[key]=tuple(torch.from_numpy(bounded[key][k].copy()) for k in
                ['gnn_numeric','gnn_opcode','gnn_src','gnn_dst','gnn_config_nodes'])
    features=[];layouts={k:[] for k in profiles};indices=[]
    def flush():
        data={k:np.stack([r[k] for r in features]) for k in features[0]};xs={}
        if profiles:
            summary=np.concatenate([data[k] for k in ['position','opcode','depth','edge']],axis=1)
            batch={k:np.stack(v) for k,v in layouts.items()}
        for i,(name,model,key) in enumerate(members):
            if key is not None: pred=predict_arrays(model,graphs[key],batch[key],summary,cfg['gnn_predict_batch'])
            else:
                blocks=tuple(model['experiment']['blocks'])
                if blocks not in xs: xs[blocks]=np.concatenate([data[b] for b in blocks],axis=1)
                pred=engine.predict(model,xs[blocks])
            predictions[i,indices]=pred
        features.clear();indices.clear()
        for v in layouts.values():v.clear()
    with threadpool_limits(limits=cfg['cpu_threads']):
        for index,row in engine.selected_rows(path,'config_feat' if tile else 'node_config_feat',np.arange(n)):
            features.append(engine.tile_features(row,context,cfg) if tile else engine.layout_features(row,context,local,cfg))
            for key,g in bounded.items():layouts[key].append(row[g['gnn_positions']].astype(np.int8))
            indices.append(index)
            if len(indices)>=cfg['inference_batch']:flush()
        if indices:flush()
    by_method={name:np.mean([rankdata(predictions[i],method='average')/n for i,(m,_,_) in enumerate(members) if m==name],axis=0)
               for name in job['sources']}
    selected=sum(w*by_method[n] for n,w in job['components'].items())
    output=Path(job['output'])
    np.savez_compressed(output/'ranks.npz',**by_method)
    write_json(output/'prediction.json',dict(ID=job['id'],count=n,components=job['components'],
        selected=ranking(selected,len(job['template_values']),tile,job['template_values'])))


def run(job):
    if job['action']=='pilot':pilot(job)
    elif job['action']=='inference':inference(job)
    elif job['action']=='prepare':engine.prepare(job['root'],job['manifest'],job['cfg'],job['scratch'],job['output'])
    elif job['action']=='trial':engine.run_trial(job)
    elif job['action']=='holdout':engine.evaluate_holdout(job)
    else:raise ValueError(job['action'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True)
    args=parser.parse_args()
    job=json.loads(Path(args.job).read_text())
    try:run(job)
    except MemoryError as exc:
        write_json(Path(job['output'])/'worker_failure.json',dict(kind='resource',error=str(exc)))
        traceback.print_exc();sys.exit(1)
    except Exception:traceback.print_exc();sys.exit(1)
