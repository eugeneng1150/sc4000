"""Full-candidate streamed prediction with frozen development-selected models."""
import json
from pathlib import Path
import time
import zipfile

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

from study_engine import (metadata, graph_context, layout_context, layout_features, tile_features,
                          selected_rows, predict, guard, digest, write_json, COLLECTIONS)


def ranking(prediction, count, tile, template_values):
    n = len(prediction)
    if not n or not np.isfinite(prediction).all():
        raise ValueError('Empty or nonfinite predictions')
    ordered = np.argsort(prediction,kind='stable')[:min(n,count)].tolist()
    if not tile and count > n:
        raise ValueError('Layout template requests more candidates than exist')
    if count > n:
        # Preserve ONLY the supplied template suffix, never invent candidate IDs.
        # The competition template has 0;1;2;3;4 for some tile graphs with <5 candidates.
        expected = list(range(n,count))
        if not tile or template_values != list(range(count)):
            raise ValueError('Unrecognized short-tile template contract')
        ordered += expected
    validate_ranking(ordered,n,count,tile,template_values)
    return ordered


def validate_ranking(values,n,count,tile,template_values):
    if len(values)!=count or len(set(values))!=count:
        raise ValueError('Ranking length/uniqueness mismatch')
    prefix=values[:min(n,count)]
    if any(v<0 or v>=n for v in prefix):
        raise ValueError('Invalid candidate in ranking')
    if not tile and (count>n or (count==n and sorted(values)!=list(range(n)))):
        raise ValueError('Layout ranking is not a full permutation')
    if count>n and (not tile or template_values!=list(range(count)) or values[n:]!=template_values[n:]):
        raise ValueError('Short tile suffix differs from original template')


def infer_graph(job):
    from study_plan import effective_config
    cfg=job['cfg'];tile=job['collection'].startswith('tile')
    path=Path(job['path']);directory=Path(job['directory']);lock=job['lock']
    names=list(dict.fromkeys(['B0_basic_hgb',lock['winner']]+list(lock['components'])))
    catalog={e['name']:e for e in json.loads((directory/'experiment_plan.json').read_text())}
    members=[];profiles={}
    for name in names:
        for fold in cfg['folds']:
            for seed in cfg['seeds']:
                model_dir=directory/'folds'/str(fold)/'trials'/name/str(seed)
                if catalog[name]['model']=='gnn':
                    import torch
                    from study_gnn import load_model
                    torch.set_num_threads(cfg['cpu_threads'])
                    model=load_model(model_dir/'model.pt');key=digest(catalog[name]['profile'])
                    profiles[key]=effective_config(cfg,catalog[name],job['collection'])
                else:model=joblib.load(model_dir/'model.joblib');key=None
                members.append((name,model,key))
    n=metadata(path,'config_feat' if tile else 'node_config_feat')[0][0]
    guard(n*8*(len(members)+8));predictions=np.empty((len(members),n),np.float64)
    context=graph_context(path,cfg);local=None if tile else layout_context(path,context,cfg)
    bounded={};graphs={}
    if profiles:
        from study_gnn import compact_graph,predict_arrays
        for key,settings in profiles.items():
            bounded[key]=compact_graph(context,local,settings)
            graphs[key]=tuple(torch.from_numpy(bounded[key][k].copy()) for k in
                          ['gnn_numeric','gnn_opcode','gnn_src','gnn_dst','gnn_config_nodes'])
    features=[];layouts={key:[] for key in profiles};indices=[]
    def flush():
        data={k:np.stack([row[k] for row in features]) for k in features[0]};xs={}
        if profiles:
            summary=np.concatenate([data[k] for k in ['position','opcode','depth','edge']],axis=1)
            batch_layout={k:np.stack(v) for k,v in layouts.items()}
        for member,(name,model,key) in enumerate(members):
            if key is not None:pred=predict_arrays(model,graphs[key],batch_layout[key],summary,cfg['gnn_predict_batch'])
            else:
                blocks=tuple(model['experiment']['blocks'])
                if blocks not in xs:xs[blocks]=np.concatenate([data[k] for k in blocks],axis=1)
                pred=predict(model,xs[blocks])
            predictions[member,indices]=pred
        features.clear();indices.clear()
        for values in layouts.values():values.clear()
    with threadpool_limits(limits=cfg['cpu_threads']):
        for index,row in selected_rows(path,'config_feat' if tile else 'node_config_feat',np.arange(n)):
            features.append(tile_features(row,context,cfg) if tile else layout_features(row,context,local,cfg))
            for key,graph in bounded.items():layouts[key].append(row[graph['gnn_positions']].astype(np.int8))
            indices.append(index)
            if len(indices)>=cfg['inference_batch']:flush()
        if indices:flush()
    by_method={name:np.mean([rankdata(predictions[i],method='average')/n for i,(member,_,_) in enumerate(members)
                           if member==name],axis=0) for name in names}
    chosen=sum(w*by_method[name] for name,w in lock['components'].items());count=len(job['template_values'])
    result=dict(ID=job['id'],count=n,components=lock['components'],folds=cfg['folds'],seeds=cfg['seeds'],
                gnn_coverage={k:v['gnn_coverage'].tolist() for k,v in bounded.items()},
                baseline=ranking(by_method['B0_basic_hgb'],count,tile,job['template_values']),
                tabular=ranking(by_method[lock['winner']],count,tile,job['template_values']),
                selected=ranking(chosen,count,tile,job['template_values']))
    write_json(Path(job['output'])/'prediction.json',result)


def locate_template(root,explicit):
    if explicit:
        path=Path(explicit)
        if not path.is_file():raise FileNotFoundError(path)
        return path
    for folder in [root,root.parent,root.parent.parent,root.parent.parent.parent]:
        for name in ['sample_submission.csv','sample_submission_Eugene.csv']:
            if (folder/name).is_file():return folder/name
    raise FileNotFoundError('Set SAMPLE_SUBMISSION to the competition template CSV path')


def preflight(root,cfg):
    """Check the complete test/template contract before spending hours on training."""
    root=Path(root)
    template=pd.read_csv(locate_template(root,cfg['sample_submission']),dtype=str)
    if list(template.columns)!=['ID','TopConfigs'] or template.ID.duplicated().any() or template.isna().any().any():
        raise ValueError('Invalid sample submission template')
    if {value.rsplit(':',1)[0] for value in template.ID}!=set(cfg['collections']):
        raise ValueError('Template must cover exactly the configured collections')
    short,truncated=0,0
    for row in template.itertuples(index=False):
        collection,stem=row.ID.rsplit(':',1)
        if collection not in COLLECTIONS or '/' in stem or '\\' in stem:
            raise ValueError('Invalid template ID')
        tile=collection.startswith('tile')
        path=root.joinpath(*collection.split(':'),'test',stem+'.npz')
        shape,fortran,_=metadata(path,'config_feat' if tile else 'node_config_feat')
        if not shape[0] or fortran:
            raise ValueError(f'{row.ID}: empty or non-streamable test configurations')
        values=[int(v) for v in row.TopConfigs.split(';')]
        validate_ranking(values,shape[0],len(values),tile,values)
        short+=tile and len(values)>shape[0]
        truncated+=not tile and len(values)<shape[0]
    return dict(rows=len(template),short_tile_rows=short,layout_rows_truncated_to_template=truncated)


def export_submissions(root,output,engine_path,cfg,deadline,execute):
    template_path=locate_template(root,cfg['sample_submission'])
    template=pd.read_csv(template_path,dtype=str)
    if list(template.columns)!=['ID','TopConfigs'] or template.ID.duplicated().any() or template.isna().any().any():
        raise ValueError('Invalid sample submission template')
    actual_collections={row.rsplit(':',1)[0] for row in template.ID}
    if actual_collections!=set(cfg['collections']):
        raise ValueError('Template collections and configured collections differ')
    tests=[]
    for row in template.itertuples(index=False):
        collection,stem=row.ID.rsplit(':',1)
        if '/' in stem or '\\' in stem or collection not in COLLECTIONS:
            raise ValueError('Invalid test ID')
        path=root.joinpath(*collection.split(':'),'test',stem+'.npz')
        with zipfile.ZipFile(path) as z:
            fingerprint=[(i.filename,i.CRC,i.file_size) for i in z.infolist()]
        tests.append(dict(id=row.ID,collection=collection,relative_path=str(path.relative_to(root)),
                          template_values=[int(v) for v in row.TopConfigs.split(';')],fingerprint=fingerprint))
    # JSON round trip normalizes tuples for resume comparison.
    manifest=json.loads(json.dumps(dict(template_sha=digest(template_path.read_text()),tests=tests)))
    marker=output/'test_manifest.json'
    if marker.exists() and json.loads(marker.read_text())!=manifest:
        raise ValueError('Test inputs/template changed; use a new RUN_NAME')
    write_json(marker,manifest)
    successes=[]
    for i,record in enumerate(tests):
        directory=output/record['collection'].replace(':','_')
        lock=json.loads((directory/'selection_lock.json').read_text())
        destination=output/'test_predictions'/f'{i:05d}';destination.mkdir(parents=True,exist_ok=True)
        job=dict(action='inference',cfg=cfg,lock=lock,directory=str(directory),
                 path=str(root/record['relative_path']),output=str(destination),**record)
        print(f'Test graph {i+1}/{len(tests)}: {record["id"]}',flush=True)
        ok=execute(job,engine_path,cfg,deadline,destination/'status.json')
        if ok:successes.append(json.loads((destination/'prediction.json').read_text()))
        if ok is None:break
    if len(successes)!=len(tests):
        write_json(output/'submission_status.json',dict(status='incomplete',completed=len(successes),total=len(tests)))
        return
    for kind in ['baseline','tabular','selected']:
        rows=[]
        for record,result in zip(tests,successes):
            if result['ID']!=record['id']:raise ValueError('Prediction ID mismatch')
            validate_ranking(result[kind],result['count'],len(record['template_values']),
                             record['collection'].startswith('tile'),record['template_values'])
            rows.append(dict(ID=result['ID'],TopConfigs=';'.join(map(str,result[kind]))))
        frame=pd.DataFrame(rows)
        if not frame.ID.equals(template.ID):raise ValueError('Submission order mismatch')
        target=output/f'submission_{kind}.csv';temp=target.with_suffix('.csv.tmp')
        frame.to_csv(temp,index=False);temp.replace(target)
    write_json(output/'submission_status.json',dict(status='complete',rows=len(tests),
               template=str(template_path.name),full_candidate_inference=True,
               layout_truncated_to_template=sum(not r['collection'].startswith('tile') and len(r['template_values'])<s['count'] for r,s in zip(tests,successes)),
               short_tile_template_rows=sum(len(r['template_values'])>s['count'] for r,s in zip(tests,successes)),
               fitted_models='same frozen fold/seed models used in audit; no holdout retraining'))
