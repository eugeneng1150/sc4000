"""Report figures from saved evidence; never retrain to draw a chart."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from v4_core import read

COLORS=['#5975a4','#d38a38','#32866b']


def generate(ref,output,root=None,kaggle=None):
    ref,output=Path(ref),Path(output)
    dest=output/'report';dest.mkdir(parents=True,exist_ok=True)
    cfg=read(ref/'run_state.json')['config'];collections=cfg['collections']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.dpi':110,'savefig.dpi':180,'pdf.fonttype':42})
    captions=[]
    def save(fig,name,caption):
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(dest/f'{name}.{ext}',bbox_inches='tight')
        plt.close(fig);captions.append(f'**{name}.** {caption}')
    def table(rows,name):
        pd.DataFrame(rows).to_csv(dest/f'{name}.csv',index=False)

    fig,ax=plt.subplots(figsize=(10,4));ax.set_xlim(0,10);ax.set_ylim(0,4);ax.axis('off')
    boxes=[(1,2,'NPZ graphs +\nconfigurations'),(3.3,2,'Graph-disjoint folds\nFixed configuration pools'),
           (5.8,3,'Tabular features\nHGB'),(5.8,1,'Bounded graph sampling\nGNN (XLA layout)'),
           (8.5,2,'Development-selected\nrank ensemble'),(8.5,.35,'Audit + full-candidate\ntest rankings')]
    for x,y,label in boxes:ax.text(x,y,label,ha='center',va='center',bbox=dict(boxstyle='round,pad=.5',fc='#edf3f8',ec='#55718c'))
    for start,end in [((1.9,2),(2.2,2)),((4.45,2.2),(4.7,2.9)),((4.45,1.8),(4.65,1.1)),
                      ((6.85,3),(7.4,2.3)),((6.85,1),(7.4,1.7)),((8.5,1.55),(8.5,.8))]:
        ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',color='#44586a'))
    save(fig,'01_pipeline','V3 pipeline. Sampling precedes model fitting. Development selects the ensemble; official validation is an already-inspected audit. V4 retains these definitions and expands specified training sets.')

    inventory=dest/'dataset_inventory.csv'
    if root and not inventory.exists():
        from study_engine import metadata
        rows=[]
        for c in collections:
            for p in sorted(Path(root).joinpath(*c.split(':'),'train').glob('*.npz')):
                rows.append(dict(collection=c,graph=p.stem,nodes=metadata(p,'node_feat')[0][0],edges=metadata(p,'edge_index')[0][0]))
        if rows:table(rows,'dataset_inventory')
    coverage=[]
    for c in collections:
        for p in (ref/c.replace(':','_')/'preparations').glob('*/coverage.json'):
            job=p.parent/'status.job.json'
            if not job.exists():continue
            settings=read(job)['cfg']
            for r in read(p):
                v=r.get('gnn_coverage')
                if v and r['split']=='holdout':
                    coverage.append(dict(collection=c,graph=r['graph'],nodes_budget=settings['gnn_nodes'],
                        sampler=settings.get('gnn_sampler','uniform'),node_fraction=v[1]/max(v[0],1),edge_fraction=v[6]/max(v[4],1)))
    cov=pd.DataFrame(coverage)
    if len(cov):
        cov=cov.drop_duplicates(['collection','graph','nodes_budget','sampler']);cov.to_csv(dest/'coverage_sources.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    if inventory.exists():
        inv=pd.read_csv(inventory)
        axes[0].boxplot([inv[inv.collection==c].nodes for c in collections],tick_labels=[c.replace('layout:','') for c in collections],showfliers=False)
        axes[0].set_yscale('log');axes[0].tick_params(axis='x',rotation=35);axes[0].set_ylabel('Nodes per training graph (log scale)')
    else:axes[0].text(.5,.5,'Dataset inventory pending\nRun with DATA_ROOT to populate',ha='center',transform=axes[0].transAxes)
    if len(cov):
        for c in cov.collection.unique():
            g=cov[(cov.collection==c)&(cov.sampler=='uniform')].groupby('nodes_budget')[['node_fraction','edge_fraction']].median()
            for col,style in [('node_fraction','-'),('edge_fraction','--')]:axes[1].plot(g.index,100*g[col],style,marker='o',label=c.replace('layout:','')+' '+col.split('_')[0])
        axes[1].legend(fontsize=8)
    axes[1].set_xlabel('Node budget (other graph caps co-vary)');axes[1].set_ylabel('Median retained fraction (%)')
    save(fig,'02_dataset_and_coverage','Left: all supplied official training graphs, boxplot outliers hidden. Right: uniform-sampled XLA audit graphs, deduplicated across folds; edge fractions are relative to the pre-capped edge context, not necessarily the original full graph. Coverage values do not establish the cause of model performance.')

    xlas=[c for c in collections if c in cfg['gnn_collections']]
    fig,axes=plt.subplots(1,max(1,len(xlas)),figsize=(11,4),squeeze=False);source=[]
    for ax,c in zip(axes[0],xlas):
        p=ref/c.replace(':','_')/'dev_confirmed_factor_contrasts.csv'
        if not p.exists():continue
        f=pd.read_csv(p);f=f[f.factor.isin(['mask features','context features','masks and context','message passing'])].copy()
        f['collection']=c;source+=f.to_dict('records')
        y=np.arange(len(f));ax.scatter(f.mean_paired_gain,y,color=COLORS[2])
        for j,r in enumerate(f.itertuples()):ax.plot([r.bootstrap_low,r.bootstrap_high],[j,j],color=COLORS[2])
        ax.axvline(0,color='grey',ls='--');ax.set_yticks(y,f.factor);ax.set_title(c.replace('layout:',''));ax.set_xlabel('Paired development tau change')
    if not xlas:axes[0,0].text(.5,.5,'No XLA GNN evidence in this fixture',ha='center')
    table(source,'component_ablation_sources')
    save(fig,'03_component_ablations','Three-seed confirmed v3 development comparisons, paired by graph; exploratory 95% graph-bootstrap intervals, no multiplicity correction. Masks/context compare with the reference GNN; message passing compares full with no-message. These are different specified controls, not a cumulative improvement waterfall.')

    fig,axes=plt.subplots(1,max(1,len(xlas)),figsize=(10,3.8),squeeze=False);source=[]
    for ax,c in zip(axes[0],xlas):
        f=pd.read_csv(ref/c.replace(':','_')/'dev_fold_seed_scores.csv')
        f=f[(f.seed==cfg['seeds'][0])&f.experiment.str.match(r'G_(full|reference)_n\d+$')].copy()
        f['collection']=c;source+=f.to_dict('records')
        for recipe in ['reference','full']:
            g=f[f.experiment.str.startswith('G_'+recipe+'_')].copy();g['nodes']=g.experiment.str.extract(r'n(\d+)$')[0].astype(int)
            g=g.groupby('nodes').score.mean();ax.plot(g.index,g.values,marker='o',label=recipe)
        ax.set_title(c.replace('layout:',''));ax.set_xlabel('Node budget');ax.set_ylabel('Mean development tau');ax.legend()
    table(source,'coverage_score_sources')
    save(fig,'04_coverage_scores','Seed-42 screening, mean of two development folds. Node, configurable-node and edge budgets change together; training graphs/configurations are fixed. Missing budgets are not zero scores. Lines connect tested settings, not a fitted scaling law.')

    fig,axes=plt.subplots(1,max(1,len(xlas)),figsize=(10,4),squeeze=False);source=[]
    for ax,c in zip(axes[0],xlas):
        d=ref/c.replace(':','_');f=pd.read_csv(d/'dev_fold_seed_scores.csv');f=f[f.seed==cfg['seeds'][0]]
        n=read(d/'coverage_lock.json')['profile']['nodes'];grid=np.full((3,3),np.nan)
        for i,g in enumerate([24,40,0]):
            for j,count in enumerate([64,128,256]):
                name=f'G_full_n{n}' if (g,count)==(24,64) else f'G_data_g{g or "all"}_c{count}'
                values=f[f.experiment==name].score
                if len(values)==len(cfg['folds']):grid[i,j]=values.mean()
                source.append(dict(collection=c,graphs=g,configs=count,score=grid[i,j],experiment=name))
        im=ax.imshow(np.ma.masked_invalid(grid),cmap='viridis');fig.colorbar(im,ax=ax,label='Mean development tau')
        for i in range(3):
            for j in range(3):
                value=grid[i,j]
                color='#17202a' if np.isfinite(value) and im.norm(value)>.6 else 'white'
                ax.text(j,i,f'{value:.3f}' if np.isfinite(value) else 'missing',ha='center',va='center',color=color)
        ax.set_xticks(range(3),[64,128,256]);ax.set_yticks(range(3),['24','40','all eligible']);ax.set_xlabel('Configurations per training graph');ax.set_ylabel('Training graphs');ax.set_title(c.replace('layout:',''))
    table(source,'training_grid_sources')
    save(fig,'05_training_data_grid','Seed-42 screening over two development folds at each collection’s selected graph budget. “All” excludes that fold’s development graphs. More examples also change optimiser steps under the fixed epoch cap; this is not a compute-matched experiment.')

    systems=[]
    for c in collections:
        f=pd.read_csv(ref/c.replace(':','_')/'system_holdout_graphs.csv')
        for name,g in f.groupby('experiment'):systems.append(dict(collection=c,system=name,score=g.score.mean(),graphs=g.graph.nunique()))
    table(systems,'final_system_sources');systems=pd.DataFrame(systems)
    fig,axes=plt.subplots(1,3,figsize=(13,4),gridspec_kw={'width_ratios':[2,1,1.1]})
    for ax,cs in [(axes[0],[c for c in collections if c.startswith('layout')]),(axes[1],[c for c in collections if c.startswith('tile')])]:
        x=np.arange(len(cs))
        for j,name in enumerate(['baseline','tabular','selected']):
            vals=[systems[(systems.collection==c)&(systems.system==name)].score.iloc[0] for c in cs]
            ax.bar(x+(j-1)*.24,vals,.24,label=name,color=COLORS[j])
        ax.set_xticks(x,[c.replace('layout:','') for c in cs],rotation=35);ax.set_title('Local audit');ax.legend(fontsize=8)
    axes[0].set_ylabel('Mean sampled Kendall tau');axes[1].set_ylabel('Mean 1 − sampled top-5 regret')
    if kaggle is not None:write_json_local(dest/'kaggle_scores.json',kaggle)
    reported=read(dest/'kaggle_scores.json') if (dest/'kaggle_scores.json').exists() else {'baseline':.40,'tabular':.53,'selected':.55}
    axes[2].bar(list(reported),list(reported.values()),color=COLORS[:len(reported)]);axes[2].set_ylim(0,max(.65,max(reported.values())+.05));axes[2].tick_params(axis='x',rotation=35)
    axes[2].set_title('Kaggle: user-reported\napproximate scores');axes[2].set_ylabel('Leaderboard score')
    table([dict(system=k,score=v,status='user-reported; verify exact score and leaderboard split') for k,v in reported.items()],'kaggle_sources')
    save(fig,'06_final_comparison','V3 audit ensembles and separately reported Kaggle scores. Local metrics and Kaggle are not interchangeable; do not average local panels. Kaggle values are rounded conversation reports, not independently verified. Confirm exact scores before final report submission.')

    setup=[];decisions=[];v4rows=[];training_sizes=[]
    for c in collections:
        d=ref/c.replace(':','_');lock=read(d/'selection_lock.json')
        fit=pd.read_csv(d/'all_trials.csv')
        for name,weight in lock['components'].items():
            for fold in cfg['folds']:
                for seed in cfg['seeds']:
                    path=d/'folds'/str(fold)/'trials'/name/str(seed)/'training_manifest.json'
                    if path.exists():
                        records=read(path);counts=[len(r['config_ids']) for r in records]
                        training_sizes.append(dict(collection=c,method=name,ensemble_weight=weight,fold=fold,seed=seed,
                            training_graphs=len(records),training_configurations=sum(counts),
                            min_configs_per_graph=min(counts),max_configs_per_graph=max(counts)))
        setup.append(dict(collection=c,folds=len(cfg['folds']),confirmation_seeds=len(cfg['seeds']),
            fit_jobs=len(fit),worker_peak_gib=fit.peak_rss_gib.max(),worker_fit_hours=fit.wall_seconds.sum()/3600,
            baseline='B0_basic_hgb',selected=json.dumps(lock['components']),max_epochs=cfg['gnn_epochs'],
            worker_cap_gib=cfg['worker_ram_gib'],matrix_cap_mib=cfg['max_matrix_mib']))
        p=d/'dev_confirmed_factor_contrasts.csv'
        if p.exists():
            f=pd.read_csv(p)
            for r in f.to_dict('records'):
                verdict='positive exploratory interval' if r['bootstrap_low']>0 else 'negative exploratory interval' if r['bootstrap_high']<0 else 'inconclusive interval'
                decisions.append(dict(collection=c,**r,interpretation=verdict,decision='See selection_lock.json; interval alone was not the selection rule'))
        nd=output/c.replace(':','_')
        for p in nd.glob('folds/*/trials/*/*/result.json'):
            row=read(p);row.update(collection=c,fold=int(p.parents[3].name));v4rows.append(row)
    table(setup,'table_1_setup_and_resources');table(decisions,'table_2_findings_and_decisions')
    table(training_sizes,'table_1_selected_training_sizes')
    if v4rows:
        table(v4rows,'v4_trials');f=pd.DataFrame(v4rows)
        groups=list(f[f.seed==cfg['seeds'][0]].groupby('collection'))
        fig,axes=plt.subplots(1,len(groups),figsize=(max(6,3.5*len(groups)),4),squeeze=False)
        for ax,(c,g) in zip(axes[0],groups):
            means=g.groupby('experiment').dev_score.mean();ax.plot(means.index,means.values,'o')
            ax.tick_params(axis='x',rotation=75);ax.set_title(c.replace('layout:',''))
            ax.set_ylabel('Development 1−regret' if c.startswith('tile') else 'Development tau')
        save(fig,'A1_v4_screening','New seed-42 experiments, one panel per collection. Raw fold/seed values are in v4_trials.csv. Main figures 1–6 retain v3 evidence. New audit tables are under each v4 collection.')
        audit_rows=[]
        for c in collections:
            p=output/c.replace(':','_')/'system_audit_gains.csv'
            if p.exists():
                a=pd.read_csv(p);a['collection']=c;audit_rows+=a.to_dict('records')
        if audit_rows:table(audit_rows,'v4_audit_gains')
        histories=[]
        for c in xlas:
            for p in (output/c.replace(':','_')).glob('folds/*/trials/*/*/history.csv'):
                h=pd.read_csv(p);h['collection']=c;h['fold']=p.parents[3].name;h['seed']=p.parent.name;h['experiment']=p.parents[1].name
                histories+=h.to_dict('records')
        if histories:
            table(histories,'v4_learning_histories');hist=pd.DataFrame(histories)
            fig,axes=plt.subplots(1,len(xlas),figsize=(10,3.5),squeeze=False)
            for ax,c in zip(axes[0],xlas):
                for (fold,seed),g in hist[hist.collection==c].groupby(['fold','seed']):
                    ax.plot(g.epoch,g.early_dev_score,label=f'fold {fold}, seed {seed}')
                ax.set_title(c.replace('layout:',''));ax.set_xlabel('Epoch');ax.set_ylabel('Early-stopping development tau')
                if len(hist[hist.collection==c]):ax.legend(fontsize=7)
            save(fig,'A2_v4_learning_curves','Early-stopping scores on the fixed smaller development configuration subset. Do not compare their absolute values with the full development evaluation or Kaggle.')
    (dest/'CAPTIONS.md').write_text('# Report figure captions\n\n'+'\n\n'.join(captions)+'\n')
    (dest/'REPORT_GUIDE.md').write_text('''# Report evidence guide

Six numbered figures and two CSV tables support a roughly ten-page report.
Suggested space: introduction 1 page; data/EDA 1; preprocessing/methods 2;
evaluation protocol 1; results/ablations 3; discussion/limitations/conclusion 2.
Check the course rules for whether references count toward the limit.

These are evidence assets, not the finished report. Figures 1–6 use completed v3
results even if v4 is paused. V4 tables and appendix plots are added as fits finish.
Do not describe screening as three-seed evidence or a confidence interval crossing
zero as proof of no effect. Development folds were reused for selection, so they
are not independent final tests. Official validation was previously inspected.
Do not pool layout and tile metrics into an alleged Kaggle score. Exact public/
private leaderboard scores and filenames need verification. Training subsets and
configuration caps must be reported. V4 may exclude larger sets for resources;
such exclusions are not negative accuracy results. Worker-fit time excludes data
preparation, inference, controller overhead and lost work from runtime resets.

Preprocessing: inspect embedded study_engine.py and study_gnn.py for exact feature
definitions. Tile runtimes use supplied normalizers; tabular features include
graph context and configuration transforms; layout features encode masks,
positions, opcodes, depth and edges. GNN numeric features are standardized within
the sampled graph. Targets/ranking losses differ by estimator and are recorded
in experiment_plan.json. Do not imply a universal scaler shared by all models.
''')
    return dest


def write_json_local(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
