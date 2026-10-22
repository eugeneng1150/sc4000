"""Question-led historical figures, with explicit protocols and a separate current-run view."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from final_report import SHORT, read

COLORS=['#276B8E','#C77529','#328469','#8655A2']

def gains(frame, before, after):
    f=frame.copy();f['absolute_gain']=f[after]-f[before]
    # Relative tau changes near zero obscure effect sizes rather than explain them.
    f['relative_gain_pct']=np.where(f[before]>=.1,100*f.absolute_gain/f[before],np.nan)
    return f


def generate(evidence, output, contract):
    evidence,output=Path(evidence),Path(output);figdir=output/'figures';tables=output/'tables'
    figdir.mkdir(parents=True,exist_ok=True);tables.mkdir(exist_ok=True)
    load=lambda name:pd.read_csv(evidence/'tables'/f'{name}.csv')
    captions=[]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170,'pdf.fonttype':42})
    def save(fig,name,caption):
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(figdir/(name+'.'+ext),bbox_inches='tight')
        plt.close(fig);captions.append(f'**{name}.** {caption}')
    def table(f,name):f.to_csv(tables/(name+'.csv'),index=False)

    fig,ax=plt.subplots(figsize=(11,4));ax.axis('off');ax.set_xlim(0,10);ax.set_ylim(0,4)
    nodes=[(1,2,'Raw graphs +\ncompiler configurations'),(3.4,2,'Fixed graph splits\nBounded preprocessing'),
           (6,3,'Compact features\nGradient boosting'),(6,1,'Sampled graph\nGraphSAGE ranker'),
           (8.8,2,'Within-graph ranks\nFixed collection blend'),(8.8,.25,'Submission CSV')]
    for x,y,t in nodes:ax.text(x,y,t,ha='center',va='center',bbox=dict(boxstyle='round,pad=.55',fc='#EDF4F6',ec='#426E84'))
    for a,b in [((1.9,2),(2.25,2)),((4.5,2.25),(4.9,2.9)),((4.5,1.75),(4.9,1.1)),((7.15,3),(7.75,2.2)),((7.15,1),(7.75,1.8)),((8.8,1.5),(8.8,.6))]:
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#426E84'))
    save(fig,'01_method','Raw-data reproduction workflow. Compatible saved models are loaded; missing models are trained before evaluation and prediction. Blend recipes remain fixed.')

    # A single shared protocol for the model-family chart; no early MLP scores mixed in.
    jobs=[s for s in contract['jobs'] if s['seed']==42 and s['archive_directory'].startswith('reference_v4/reference_v3/')]
    rows=[]
    for c in ['layout:xla:default','layout:xla:random']:
        gnn=next(n for n in contract['selected'][c] if n.startswith('G_data'))
        matched=gnn.replace('G_data_','H_matched_')
        for name,label in [('B0_basic_hgb','Basic HGB'),(matched,'Matched-data HGB'),(gnn,'GraphSAGE')]:
            for s in jobs:
                if s['collection']==c and s['experiment']['name']==name:
                    rows.append(dict(collection=c,model=label,experiment=name,fold=s['fold'],score=s['historical_dev_score']))
    f=pd.DataFrame(rows);table(f,'model_family_comparison')
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for ax,c in zip(axes,['layout:xla:default','layout:xla:random']):
        labels=['Basic HGB','Matched-data HGB','GraphSAGE']
        for fold,marker in [(0,'o'),(1,'s')]:
            g=f[(f.collection==c)&(f.fold==fold)].set_index('model').reindex(labels)
            ax.plot(range(3),g.score,marker+'--',label=f'Development fold {fold}',alpha=.9)
        ax.set_xticks(range(3),labels);ax.set_title(SHORT[c]);ax.set_ylabel('Development Kendall tau');ax.legend(fontsize=8)
    save(fig,'02_model_families','Historical seed-42 exploration, common development folds. Matched-data HGB and GraphSAGE share graph/configuration budgets; architecture/features differ. Basic HGB is a separate pipeline baseline. Fold disagreement is visible; this is not proof of universal GNN superiority.')

    f=load('component_ablation_sources');table(gains(f,'baseline_score','candidate_score'),'component_changes')
    fig,axes=plt.subplots(1,2,figsize=(11,3.7))
    labels={'mask features':'Add validity masks','context features':'Add context summaries','masks and context':'Add both','message passing':'Retain message passing'}
    for ax,c in zip(axes,['layout:xla:default','layout:xla:random']):
        g=f[f.collection==c].reset_index(drop=True)
        for i,r in g.iterrows():
            ax.plot([r.bootstrap_low,r.bootstrap_high],[i,i],color=COLORS[0]);ax.plot(r.mean_paired_gain,i,'o',color=COLORS[0])
        ax.axvline(0,color='grey',ls='--');ax.set_yticks(range(len(g)),[labels.get(x,x) for x in g.factor]);ax.invert_yaxis()
        ax.set_title(SHORT[c]);ax.set_xlabel('Change in development tau (candidate − control)')
    save(fig,'03_components','Historical three-seed confirmed comparisons. Each row has its own named control in the source table. Bootstrap intervals cross zero: these comparisons do not establish clear standalone gains. They are not additive steps in a waterfall.')

    cov=load('coverage_score_sources');grid=load('training_grid_sources')
    fig,axes=plt.subplots(2,2,figsize=(11,7.2))
    for col,c in enumerate(['layout:xla:default','layout:xla:random']):
        ax=axes[0,col]
        for name,label,color in [('reference','Reference GraphSAGE',COLORS[0]),('full','Masks + context GraphSAGE',COLORS[1])]:
            g=cov[(cov.collection==c)&cov.experiment.str.startswith('G_'+name+'_')].copy()
            g['nodes']=g.experiment.str.extract(r'n(\d+)$')[0].astype(int);means=g.groupby('nodes').score.mean()
            ax.plot(means.index,means,'o-',label=label,color=color)
        ax.set_title(SHORT[c]+' — graph coverage');ax.set_xlabel('Sampled node budget');ax.set_ylabel('Mean development tau');ax.legend(fontsize=8)
        ax=axes[1,col];g=grid[grid.collection==c]
        for count,color in zip([64,128,256],COLORS):
            means=g[g.configs==count].set_index('graphs').reindex([24,40,0])
            ax.plot(range(3),means.score,'o-',color=color,label=f'{count} configurations/graph')
        ax.set_xticks(range(3),['24','40','All eligible']);ax.set_xlabel('Training graphs');ax.set_ylabel('Mean development tau');ax.set_title(SHORT[c]+' — training-data budget');ax.legend(fontsize=8)
    save(fig,'04_coverage_and_data','Historical seed-42 screening, averaged over two development folds. Top: node/configurable-node/edge caps change together. Bottom: graph/configuration counts also change optimizer work. Larger budgets did not consistently improve ranking; this is not a compute-matched scaling experiment.')
    table(cov,'graph_budget_screening');table(grid,'training_data_screening')

    f=load('trial_results');f=f[f.seed==42].copy();f['collection']=f.collection.str.replace('_',':',n=2)
    names=['HGB_control','V5_H_percentile','V5_X_regression','V5_X_pairwise']
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,c in zip(axes.flat,list(SHORT)[:4]):
        g=f[f.collection==c].pivot(index='experiment',columns='fold',values='dev_score').reindex(names)
        for i,fold in enumerate([0,1]):ax.bar(np.arange(4)+(i-.5)*.36,g[fold],.36,color=COLORS[i],label=f'Fold {fold}')
        ax.set_xticks(range(4),['HGB\nlog-runtime','HGB\npercentile','XGB\nregression','XGB\npairwise']);ax.set_title(SHORT[c]);ax.set_ylabel('Development Kendall tau');ax.legend(fontsize=8)
    save(fig,'05_target_and_objective','Historical matched-example screening. HGB log-runtime versus percentile changes target; XGB regression versus pairwise changes objective. HGB versus XGB also changes learner/capacity. Final blend improvements require the separate three-seed confirmation evidence.')
    table(f,'target_screening');table(gains(load('selected_audit_summary'),'baseline_score','candidate_score'),'blend_changes')

    sources=load('final_system_sources');audit=load('selected_audit_summary');rows=[]
    for c in SHORT:
        b=sources[(sources.collection==c)&(sources.system=='baseline')].iloc[0]
        a=audit[audit.collection==c]
        final=float(a.candidate_score.iloc[0]) if len(a) else float(sources[(sources.collection==c)&(sources.system=='selected')].score.iloc[0])
        rows.append(dict(collection=c,metric='1 - top5 regret' if c.startswith('tile') else 'Kendall tau',baseline=float(b.score),selected=final,graphs=int(b.graphs)))
    f=gains(pd.DataFrame(rows),'baseline','selected');table(f,'historical_final_audit')
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),gridspec_kw={'width_ratios':[4,1.3]})
    for ax,collections in zip(axes,[list(SHORT)[:4],['tile:xla']]):
        g=f.set_index('collection').loc[collections];x=np.arange(len(g))
        ax.bar(x-.18,g.baseline,.36,color=COLORS[0],label='Basic HGB baseline');ax.bar(x+.18,g.selected,.36,color=COLORS[2],label='Selected system')
        ax.set_xticks(x,[SHORT[c] for c in collections],rotation=20,ha='right');ax.legend(fontsize=8)
    axes[0].set_ylabel('Mean audit Kendall tau');axes[1].set_ylabel('Mean audit 1 − top-5 regret')
    save(fig,'06_final_results','Historical baseline-to-final comparison on reused official-validation audit graphs. Layout and Tile have different metrics. This is a whole-system comparison, not an isolated ablation; do not average the panels or interpret them as a Kaggle score.')

    recipe=[]
    for r in read(evidence/'selected_recipes.json'):
        e=r['experiment'];recipe.append(dict(collection=SHORT[r['collection']],model=r['name'],weight=r['weight'],family=e['model'],target=e.get('target',e.get('loss')),nodes=e.get('profile',{}).get('nodes'),configs=e.get('configs'),graph_cap=e.get('train_graphs')))
    table(pd.DataFrame(recipe),'selected_recipe');table(load('selected_training_sizes'),'actual_training_sizes')
    table(pd.DataFrame(read(evidence/'reported_kaggle_scores.json')),'reported_public_scores')
    inv=load('dataset_inventory');table(inv.groupby('collection').agg(graphs=('graph','count'),median_nodes=('nodes','median'),max_nodes=('nodes','max')).reset_index(),'dataset_inventory')
    table(load('reference_split_inventory'),'split_inventory')
    # Learning curves are supporting diagnostics, not a proof of generalization.
    history=load('selected_gnn_histories');fig,axes=plt.subplots(2,2,figsize=(11,7))
    for row,c in enumerate(['layout:xla:default','layout:xla:random']):
        for (fold,seed),g in history[history.collection==c].groupby(['fold','seed']):
            for ax,key in zip(axes[row],['loss','early_dev_score']):
                line=ax.plot(g.epoch,g[key],label=f'Fold {fold}, seed {seed}',alpha=.8)[0]
                best=g[g.epoch==int(g.selected_epoch.iloc[0])].iloc[0];ax.plot(best.epoch,best[key],'o',color=line.get_color(),markersize=4)
        for ax in axes[row]:ax.set_title(SHORT[c]);ax.set_xlabel('Epoch');ax.legend(fontsize=7,ncol=2)
        axes[row,0].set_ylabel('Mean training batch loss');axes[row,1].set_ylabel('Early-development Kendall tau')
    save(fig,'S1_learning_curves','Historical selected GNNs. Training loss and early-development tau use different units. Dots mark selected epochs. No per-epoch validation loss was saved, and these curves do not prove absence of overfitting.')
    table(history,'selected_gnn_histories')
    (output/'FIGURE_CAPTIONS.md').write_text('# Historical experiment figure captions\n\n'+'\n\n'.join(captions)+'\n')
    return figdir


def current_results(output):
    output=Path(output);p=output/'evaluation/current_audit.csv'
    if not p.exists():return None
    f=pd.read_csv(p);fig,axes=plt.subplots(1,2,figsize=(11,3.8),gridspec_kw={'width_ratios':[4,1.3]})
    for ax,mask in zip(axes,[f.collection.str.startswith('layout'),f.collection.str.startswith('tile')]):
        g=f[mask];x=np.arange(len(g));ax.bar(x-.18,g.baseline,.36,label='Current baseline',color=COLORS[0]);ax.bar(x+.18,g.selected,.36,label='Current selected',color=COLORS[2])
        ax.set_xticks(x,[SHORT[c] for c in g.collection],rotation=20,ha='right');ax.set_ylabel(g.metric.iloc[0]);ax.legend(fontsize=8)
    fig.suptitle('Current run — fresh audit predictions (reused validation graphs)');fig.tight_layout()
    dest=output/'evaluation/current_results.png';fig.savefig(dest,dpi=170,bbox_inches='tight');plt.close(fig)
    return dest
