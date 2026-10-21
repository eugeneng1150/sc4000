"""Read-only frozen evidence replay: charts, tables, archive verification and exact CSV export."""
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SHORT={'layout:nlp:default':'NLP default','layout:nlp:random':'NLP random',
       'layout:xla:default':'XLA default','layout:xla:random':'XLA random','tile:xla':'Tile XLA'}
COLORS=['#286f8e','#d27b32','#3a8c6c','#835a9b','#65737e','#a44848']


def sha(data):return hashlib.sha256(data).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def unpack(payload,expected_sha,dest):
    if sha(payload)!=expected_sha:raise ValueError('Frozen evidence package hash mismatch')
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        if sum(i.file_size for i in z.infolist())>20*1024**2:
            raise ValueError('Unexpected evidence package size')
        manifest=json.loads(z.read('freeze_manifest.json'))
        if set(z.namelist())!=set(manifest['files'])|{'freeze_manifest.json'}:
            raise ValueError('Unexpected package contents')
        for name in z.namelist():
            rel=PurePosixPath(name)
            if rel.is_absolute() or '..' in rel.parts:raise ValueError('Invalid package path')
            data=z.read(name)
            if name!='freeze_manifest.json' and sha(data)!=manifest['files'][name]['sha256']:
                raise ValueError('Artifact hash mismatch: '+name)
            path=dest/name
            if path.exists() and path.read_bytes()!=data:
                raise ValueError('Existing evidence differs; choose a new OUTPUT_DIR: '+str(path))
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    return manifest


def validate_submission(evidence):
    evidence=Path(evidence);data=(evidence/'submission_selected.csv').read_bytes()
    manifest=read(evidence/'freeze_manifest.json')
    if sha(data)!=manifest['submission_sha256']:raise ValueError('Frozen submission changed')
    rows=list(csv.DictReader(io.StringIO(data.decode())))
    constraints=list(csv.DictReader(io.StringIO((evidence/'tables/submission_constraints.csv').read_text())))
    if len(rows)!=manifest['submission_rows'] or len(rows)!=len(constraints):raise ValueError('Row count mismatch')
    if len({r['ID'] for r in rows})!=len(rows):raise ValueError('Duplicate IDs')
    for r,t in zip(rows,constraints):
        ids=list(map(int,r['TopConfigs'].split(';')));n=int(t['candidate_count']);k=int(t['length'])
        if r['ID']!=t['ID'] or len(ids)!=k or len(set(ids))!=k:raise ValueError('Ranking identity/length mismatch')
        if any(v<0 or v>=n for v in ids[:min(k,n)]):raise ValueError('Candidate out of bounds')
        if t['collection'].startswith('layout'):
            if k>n or (k==n and sorted(ids)!=list(range(n))):raise ValueError('Invalid layout ranking')
        elif k>n and ids[n:]!=list(range(n,k)):raise ValueError('Invalid short-tile padding')
        if sha(r['TopConfigs'].encode())!=t['ranking_sha256']:raise ValueError('Ranking changed')
    return dict(rows=len(rows),sha256=sha(data),collections=len({r['collection'] for r in constraints}))


def verify_model_archive(path,evidence):
    """Optional streaming hash check; does not unpickle or execute archived code."""
    path=Path(path);models=pd.read_csv(Path(evidence)/'tables/model_inventory.csv')
    def checksum(stream):
        h=hashlib.sha256()
        for chunk in iter(lambda:stream.read(1024**2),b''):h.update(chunk)
        return h.hexdigest()
    if path.is_dir():
        for row in models.itertuples():
            with (path/row.path).open('rb') as f:actual=checksum(f)
            if actual!=row.sha256:raise ValueError('Model differs: '+row.path)
    else:
        with zipfile.ZipFile(path) as z:
            for row in models.itertuples():
                with z.open(row.path) as f:actual=checksum(f)
                if actual!=row.sha256:raise ValueError('Model differs: '+row.path)
    return f'Verified {len(models)} selected model files; no models loaded or fitted.'


def export_submission(evidence,output):
    validation=validate_submission(evidence)
    path=Path(output)/'submission_final.csv';data=(Path(evidence)/'submission_selected.csv').read_bytes()
    if path.exists() and path.read_bytes()!=data:raise ValueError('Refusing to replace a different submission')
    path.write_bytes(data)
    (Path(output)/'submission_verification.json').write_text(json.dumps(validation,indent=2)+'\n')
    return path


def generate(evidence,output):
    evidence=Path(evidence);output=Path(output);figdir=output/'figures';tabdir=output/'tables'
    figdir.mkdir(parents=True,exist_ok=True);tabdir.mkdir(exist_ok=True)
    load=lambda n:pd.read_csv(evidence/'tables'/f'{n}.csv')
    captions=[]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'figure.dpi':110,'savefig.dpi':180,'pdf.fonttype':42})
    def save(fig,name,caption):
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(figdir/f'{name}.{ext}',bbox_inches='tight')
        plt.close(fig);captions.append(f'**{name}.** {caption}')
    def table(frame,name):frame.to_csv(tabdir/f'{name}.csv',index=False)

    recipes=read(evidence/'selected_recipes.json')
    description=[]
    for r in recipes:
        e=r['experiment']
        description.append(dict(collection=SHORT[r['collection']],model=r['name'],weight=r['weight'],
            family=e['model'],target=e.get('target',e.get('loss')),node_budget=e.get('profile',{}).get('nodes'),
            configs_per_graph=e.get('configs'),training_graph_cap=e.get('train_graphs')))
    table(pd.DataFrame(description),'final_model_recipe')
    table(load('selected_training_sizes'),'actual_training_sizes')
    table(load('selected_audit_summary'),'target_blend_audit')
    table(load('component_ablation_sources'),'component_ablations')

    fig,ax=plt.subplots(figsize=(11,4));ax.set_xlim(0,11);ax.set_ylim(0,4);ax.axis('off')
    nodes=[(1,2,'NPZ graphs +\nconfigurations'),(3.4,2,'Graph-disjoint\ndevelopment folds'),
        (6,3,'Masked / categorical\ntabular features → HGB'),(6,1,'Bounded graph + layout\npairwise GNN (XLA)'),
        (9,2,'Frozen rank blend\nper collection'),(9,.2,'894-row submission')]
    for x,y,text in nodes:ax.text(x,y,text,ha='center',va='center',fontsize=10,
        bbox=dict(boxstyle='round,pad=.55',facecolor='#edf4f6',edgecolor='#467187'))
    for a,b in [((1.9,2),(2.3,2)),((4.5,2.2),(4.7,2.9)),((4.5,1.8),(4.7,1.1)),
                ((7.3,3),(8,2.3)),((7.3,1),(8,1.7)),((9,1.5),(9,.6))]:
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#476071'))
    save(fig,'01_pipeline','Training workflow that produced the frozen system. This notebook replays its evidence; it does not rerun this training pipeline.')

    inv=load('dataset_inventory');cov=load('coverage_sources')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    cs=list(SHORT);axes[0].boxplot([inv[inv.collection==c].nodes for c in cs],showfliers=False)
    axes[0].set_xticks(range(1,6),[SHORT[c] for c in cs],rotation=25,ha='right')
    axes[0].set_yscale('log');axes[0].set_ylabel('Nodes per supplied training graph (log)')
    for i,c in enumerate(['layout:xla:default','layout:xla:random']):
        g=cov[(cov.collection==c)&(cov.sampler=='uniform')].groupby('nodes_budget')[['node_fraction','edge_fraction']].median()
        for col,style in [('node_fraction','-'),('edge_fraction','--')]:
            axes[1].plot(g.index,100*g[col],style,marker='o',color=COLORS[i],label=SHORT[c]+' '+col.split('_')[0])
    axes[1].set_xlabel('Node budget');axes[1].set_ylabel('Median retained fraction (%)');axes[1].legend(fontsize=8)
    save(fig,'02_data_and_coverage','Left: supplied training graph inventory; boxplot outliers hidden. Right: uniform-sampled audit graph coverage; edge denominator is the pre-capped context, not necessarily the original complete graph. Node/configurable-node/edge caps co-vary.')
    table(inv.groupby('collection').agg(graphs=('graph','count'),median_nodes=('nodes','median'),max_nodes=('nodes','max')).reset_index(),'dataset_summary')

    f=load('component_ablation_sources');fig,axes=plt.subplots(1,2,figsize=(11,3.8))
    for ax,c in zip(axes,['layout:xla:default','layout:xla:random']):
        g=f[f.collection==c].reset_index(drop=True)
        for i,r in g.iterrows():ax.plot([r.bootstrap_low,r.bootstrap_high],[i,i],color=COLORS[0]);ax.plot(r.mean_paired_gain,i,'o',color=COLORS[0])
        ax.axvline(0,color='grey',ls='--');ax.set_yticks(range(len(g)),g.factor)
        ax.set_title(SHORT[c]);ax.set_xlabel('Paired development tau change')
    save(fig,'03_component_ablations','Confirmed component comparisons against their specified controls. Exploratory graph-bootstrap intervals, not multiplicity-corrected. All intervals shown cross zero; do not claim independently established gains for masks or context. Not a cumulative waterfall.')

    f=load('coverage_score_sources');fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for ax,c in zip(axes,['layout:xla:default','layout:xla:random']):
        for label,color in [('reference',COLORS[0]),('full',COLORS[1])]:
            g=f[(f.collection==c)&f.experiment.str.startswith('G_'+label+'_')].copy()
            g['nodes']=g.experiment.str.extract(r'n(\d+)$')[0].astype(int)
            a=g.groupby('nodes').score.mean();ax.plot(a.index,a,'o-',color=color,label=label)
        ax.set_title(SHORT[c]);ax.set_xlabel('Node budget');ax.set_ylabel('Development tau');ax.legend()
    save(fig,'04_graph_budget','Seed-42 screening averaged over two development folds. Larger sampled graphs did not consistently improve scores. Other graph caps co-vary; missing settings are not zero.')

    f=load('training_grid_sources');fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for ax,c in zip(axes,['layout:xla:default','layout:xla:random']):
        g=f[f.collection==c].pivot(index='graphs',columns='configs',values='score').reindex(index=[24,40,0],columns=[64,128,256])
        im=ax.imshow(np.ma.masked_invalid(g.to_numpy()),cmap='viridis')
        fig.colorbar(im,ax=ax,label='Development tau')
        for i in range(3):
            for j in range(3):
                v=g.iloc[i,j];color='black' if pd.notna(v) and im.norm(v)>.65 else 'white'
                ax.text(j,i,f'{v:.3f}' if pd.notna(v) else 'not run',ha='center',va='center',color=color)
        ax.set_xticks(range(3),[64,128,256]);ax.set_yticks(range(3),['24','40','all eligible'])
        ax.set_xlabel('Configurations per training graph');ax.set_ylabel('Training graphs');ax.set_title(SHORT[c])
    save(fig,'05_training_data','Seed-42 data-budget screening, two-fold mean at the selected graph budget. “All eligible” excludes each fold’s development graphs. More examples also increase optimiser steps, so this is not compute-matched.')

    f=load('trial_results');f=f[f.seed==42].copy();f['collection']=f.collection.str.replace('_',':',n=2)
    names=['HGB_control','V5_H_percentile','V5_X_regression','V5_X_pairwise']
    labels=['HGB\ncontrol','HGB\npercentile','XGB\nregression','XGB\npairwise']
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,c in zip(axes.flat,list(SHORT)[:4]):
        g=f[f.collection==c].pivot(index='experiment',columns='fold',values='dev_score').reindex(names)
        for i,fold in enumerate([0,1]):ax.bar(np.arange(4)+(i-.5)*.35,g[fold],.35,label=f'Fold {fold}',color=COLORS[i])
        ax.set_xticks(range(4),labels);ax.set_title(SHORT[c]);ax.set_ylabel('Development tau');ax.legend(fontsize=8)
    save(fig,'06_target_and_objective','Matched example IDs per fold/seed. HGB control versus percentile isolates target transformation; XGB regression versus pairwise isolates objective within its fixed learner settings. HGB versus XGB changes learner/capacity. Seed-42 screening only; no XGB model was selected.')

    base=load('final_system_sources');audit=load('selected_audit_summary');rows=[]
    for c in SHORT:
        basic=float(base[(base.collection==c)&(base.system=='baseline')].score.iloc[0])
        if c.startswith('layout'):
            selected=float(audit[audit.collection==c].candidate_score.iloc[0]);n=int(audit[audit.collection==c].graphs.iloc[0])
        else:
            r=base[(base.collection==c)&(base.system=='selected')].iloc[0];selected=float(r.score);n=int(r.graphs)
        rows.append(dict(collection=c,baseline=basic,final_selected=selected,audit_graphs=n))
    comparison=pd.DataFrame(rows);table(comparison,'baseline_and_final_audit')
    fig,axes=plt.subplots(1,2,figsize=(11,4),gridspec_kw={'width_ratios':[3,1]})
    for ax,collections in zip(axes,[list(SHORT)[:4],['tile:xla']]):
        g=comparison.set_index('collection').loc[collections];x=np.arange(len(g))
        ax.bar(x-.18,g.baseline,.36,color=COLORS[0],label='Basic HGB baseline')
        ax.bar(x+.18,g.final_selected,.36,color=COLORS[2],label='Frozen final system')
        ax.set_xticks(x,[SHORT[c] for c in collections],rotation=20,ha='right');ax.legend(fontsize=8)
    axes[0].set_ylabel('Mean sampled audit Kendall tau');axes[1].set_ylabel('Mean sampled 1 − top-5 regret')
    save(fig,'07_final_audit','Previously inspected official-validation audit: 20 graphs per NLP collection, 7 per XLA layout, 24 Tile. Baseline and selected are saved ensembles. Metrics differ between panels; do not average them into a Kaggle estimate. Reuse makes this exploratory evidence, not a new untouched test.')

    f=pd.DataFrame(read(evidence/'reported_kaggle_scores.json'));table(f,'reported_public_scores')
    fig,ax=plt.subplots(figsize=(9,3.8));bars=ax.bar(range(len(f)),f.score,color=[COLORS[4]]*4+[COLORS[2]])
    ax.set_xticks(range(len(f)),['Basic\nbaseline','Tabular\nsystem','Initial\nhybrid','Data/training\nextension','Final selected\nsystem'])
    for b,r in zip(bars,f.itertuples()):ax.text(b.get_x()+b.get_width()/2,b.get_height()+.008,('≈' if r.precision=='approximate' else '')+f'{r.score:.3f}',ha='center')
    ax.set_ylim(0,.64);ax.set_ylabel('Public Kaggle score (user-reported)')
    save(fig,'08_public_scores','User-reported public scores; first three are approximate. Final 0.567 versus 0.558: +0.009 absolute (+1.61% relative). Verify exact scores and current rank with a screenshot before report submission. No inferred leaderboard percentile.')

    f=load('selected_gnn_histories');fig,axes=plt.subplots(2,2,figsize=(11,7))
    diagnostics=[]
    for row,c in enumerate(['layout:xla:default','layout:xla:random']):
        for i,((fold,seed),g) in enumerate(f[f.collection==c].groupby(['fold','seed'])):
            g=g.sort_values('epoch');best=int(g.selected_epoch.iloc[0]);selected=g[g.epoch==best].iloc[0]
            label=f'Fold {fold}, seed {seed}'
            for ax,col in zip(axes[row],['loss','early_dev_score']):
                ax.plot(g.epoch,g[col],color=COLORS[i],label=label,alpha=.85,linewidth=1.4)
                ax.plot(best,selected[col],'o',color=COLORS[i],markersize=4)
            diagnostics.append(dict(collection=c,fold=fold,seed=seed,selected_epoch=best,
                epochs_run=int(g.epoch.max()),first_loss=g.loss.iloc[0],last_loss=g.loss.iloc[-1],
                selected_dev_tau=selected.early_dev_score,last_dev_tau=g.early_dev_score.iloc[-1]))
        axes[row,0].set_ylabel('Mean training batch loss');axes[row,1].set_ylabel('Early-development Kendall tau')
        for ax in axes[row]:ax.set_title(SHORT[c]);ax.set_xlabel('Epoch');ax.legend(fontsize=7,ncol=2)
    table(pd.DataFrame(diagnostics),'gnn_learning_diagnostics')
    save(fig,'09_gnn_learning_curves','Actual selected GNN histories; dots mark saved selected epochs. Left: mean training batch loss (pairwise objective with regression fallback on batches lacking eligible pairs). Right: ranking score on the smaller early-stopping development subset, not validation loss or the full selection metric. Individual fold/seed curves are unsmoothed; no claim of absence of overfitting.')

    f=load('worker_resources');f=f[f.label.str.endswith('| trial')]
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    axes[0].plot(range(1,len(f)+1),f.wall_seconds/60,'o',color=COLORS[0]);axes[0].set_ylabel('Worker wall time (minutes)')
    axes[1].plot(range(1,len(f)+1),f.peak_rss_gib,'o',color=COLORS[2]);axes[1].set_ylabel('Worker peak RSS (GiB)')
    for ax in axes:ax.set_xlabel('Completed target/objective training job')
    save(fig,'10_training_resources','Measured workers in the target/objective study. Parallel job times are not total elapsed time; extraction, inference and backups are additional. Worker RSS excludes the rest of the Colab session.')
    (output/'FIGURE_CAPTIONS.md').write_text('# Figure captions\n\n'+'\n\n'.join(captions)+'\n')
    return figdir
