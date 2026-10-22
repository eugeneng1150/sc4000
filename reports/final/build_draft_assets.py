"""Render report assets from frozen presentation CSVs; never train or reselect models."""
from pathlib import Path
import csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'draft_assets'
OUT.mkdir(exist_ok=True)
SHORT = {'layout:nlp:default':'NLP default', 'layout:nlp:random':'NLP random',
         'layout:xla:default':'XLA default', 'layout:xla:random':'XLA random', 'tile:xla':'Tile'}
def rows(name):
    with (ROOT/'tables'/f'{name}.csv').open() as f:
        return list(csv.DictReader(f))
def table(name, spec, head, body, caption, label):
    text = '\\begin{table}[t]\n\\centering\n\\begin{tabular}{'+spec+'}\n\\toprule\n'
    text += ' & '.join(head)+r' \\'+'\n\\midrule\n'
    text += '\n'.join(' & '.join(r)+r' \\' for r in body)
    text += '\n\\bottomrule\n\\end{tabular}\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\end{table}\n'
    (OUT/f'{name}.tex').write_text(text)

table('dataset_table','@{}lrrr@{}',['Collection','Graphs','Median $|V|$','Max $|V|$'],
      [[SHORT[r['collection']],r['graphs'],f"{float(r['median_nodes']):,.0f}",f"{int(r['max_nodes']):,}"] for r in rows('dataset_inventory')],
      'Graph counts in the official training data. Counts describe the supplied collection, not every model\'s training subset. Graph identities may overlap across collections.', 'tab:data')

family = rows('model_family_comparison')
body=[]
for c in ['layout:xla:default','layout:xla:random']:
    for model,lab in [('Basic HGB','Basic HGB'),('Matched-data HGB','Matched HGB'),('GraphSAGE','GraphSAGE')]:
        scores=sorted((int(r['fold']),float(r['score'])) for r in family if r['collection']==c and r['model']==model)
        body.append([SHORT[c].replace('XLA ','') if model=='Basic HGB' else '',lab,*[f'{v:.4f}' for _,v in scores],f'{np.mean([v for _,v in scores]):.4f}'])
table('family_table','@{}llrrr@{}',['XLA','Model','Fold 0','Fold 1','Mean'],body,
      'Model-family screening at seed 42 on common development folds. Matched HGB shares GraphSAGE training examples/budgets; basic HGB uses its separate baseline budget. All values are Kendall tau.', 'tab:family')

target=rows('target_screening')
body=[]
for c in list(SHORT)[:4]:
    vals=[]
    for e in ['HGB_control','V5_H_percentile','V5_X_regression','V5_X_pairwise']:
        vals.append(f"{np.mean([float(r['dev_score']) for r in target if r['collection']==c and r['experiment']==e]):.4f}")
    body.append([SHORT[c].replace('default','def.').replace('random','rand.'),*vals])
table('target_table','@{}lrrrr@{}',['Collection','H-log','H-pct','X-reg','X-pair'],body,
      'Mean development tau over two folds at seed 42. H-log/H-pct compare HGB targets; X-reg/X-pair compare XGBoost objectives. Training examples match within each collection. These standalone screen results differ from confirmed ensemble gains.', 'tab:targets')

recipe=rows('selected_recipe')
body=[]
for c in ['NLP default','NLP random','XLA default','XLA random','Tile XLA']:
    subset=[r for r in recipe if r['collection']==c]
    vals=[sum(float(r['weight']) for r in subset if r['family']==family and r['target']==target) for family,target in [('hgb','centered_log'),('hgb','percentile'),('gnn','pairwise')]]
    body.append([c.replace('Tile XLA','Tile'),*[f'{100*v:g}\\%' if v else '--' for v in vals]])
table('recipe_table','@{}lrrr@{}',['Collection','HGB log','HGB pct','GNN'],body,
      'Final rank-blend weights. Each component averages recorded seeds and both folds for test inference. NLP random uses expanded-data HGB; GNN node budgets are 2,048 for XLA default and 512 for XLA random. No XGBoost model is selected.', 'tab:recipe')

body=[]
for r in rows('historical_final_audit'):
    body.append([SHORT[r['collection']],r['graphs'],f"{float(r['baseline']):.4f}",f"{float(r['selected']):.4f}",f"+{float(r['absolute_gain']):.4f}"])
table('final_table','@{}lrrrr@{}',['Collection','$n$','Basic','Final','$\\Delta$'],body,
      'Whole-system comparison on reused audit graphs. Layout rows report mean Kendall tau; Tile reports one minus top-five regret. These different metrics must not be averaged into a Kaggle score. Gains are absolute.', 'tab:final')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                    'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':180})
BLUE='#236485'; ORANGE='#BB6B2B'; GREEN='#357C65'
fig,ax=plt.subplots(figsize=(7,1.65))
ax.set(xlim=(0,10),ylim=(0,3));ax.axis('off')
boxes=[(.9,1.5,'Graph +\nconfiguration'),(3.2,1.5,'Validate and\nconstruct inputs'),
       (5.6,2.35,'Tabular features\nHGB / target study'),(5.6,.65,'Sampled graph\nGraphSAGE'),
       (8.65,1.5,'Normalised ranks\nCollection blend\nOrdered candidates')]
for x,y,s in boxes:
    ax.text(x,y,s,ha='center',va='center',fontsize=8.8,bbox=dict(boxstyle='round,pad=.4',fc='#EFF5F7',ec=BLUE))
for a,b in [((1.8,1.5),(2.22,1.5)),((4.13,1.8),(4.55,2.2)),((4.13,1.2),(4.55,.8)),((6.75,2.2),(7.5,1.8)),((6.65,.8),(7.5,1.2))]:
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color=BLUE,lw=1.2))
fig.tight_layout(pad=.3);fig.savefig(OUT/'pipeline.pdf',bbox_inches='tight');plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(7,2.75),sharey=True)
data=rows('component_changes')
labels={'mask features':'Add masks','context features':'Add context','masks and context':'Add both','message passing':'Retain messages'}
for ax,c in zip(axes,['layout:xla:default','layout:xla:random']):
    subset=[r for r in data if r['collection']==c]
    for i,r in enumerate(subset):
        ax.plot([float(r['bootstrap_low']),float(r['bootstrap_high'])],[i,i],color=BLUE,lw=2)
        ax.plot(float(r['mean_paired_gain']),i,'o',color=BLUE,ms=5)
    ax.axvline(0,color='#777777',ls='--',lw=.8)
    ax.set_title(SHORT[c],fontsize=10);ax.set_yticks(range(4),[labels[r['factor']] for r in subset])
    ax.set_xlabel('Change in development tau');ax.set_xlim(-.085,.055);ax.set_xticks([-.08,-.04,0,.04]);ax.grid(axis='x',alpha=.15)
axes[0].invert_yaxis()
fig.tight_layout(pad=.6,w_pad=1.5);fig.savefig(OUT/'components.pdf',bbox_inches='tight');plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(7,4.7))
coverage=rows('graph_budget_screening');data=rows('training_data_screening')
for col,c in enumerate(['layout:xla:default','layout:xla:random']):
    ax=axes[0,col]
    for key,label,color in [('reference','Reference',BLUE),('full','Masks + context',ORANGE)]:
        group=defaultdict(list)
        for r in coverage:
            if r['collection']==c and r['experiment'].startswith('G_'+key+'_'):
                group[int(r['experiment'].split('_n')[-1])].append(float(r['score']))
        xs=sorted(group);ax.plot(xs,[np.mean(group[x]) for x in xs],'o-',label=label,color=color,ms=4)
    ax.set_title(SHORT[c]+': graph coverage',fontsize=10)
    ax.set_xticks([512,1024,2048,4096],['512','1,024','2,048','4,096'])
    ax.tick_params(axis='x',labelsize=8);ax.set_xlabel('Sampled nodes');ax.set_ylabel('Development tau');ax.legend(fontsize=8)
    ax=axes[1,col]
    for count,color in zip([64,128,256],[BLUE,ORANGE,GREEN]):
        subset={int(r['graphs']):float(r['score']) for r in data if r['collection']==c and int(r['configs'])==count}
        ax.plot(range(3),[subset[k] for k in [24,40,0]],'o-',color=color,label=f'{count} configs',ms=4)
    ax.set_title(SHORT[c]+': training quantity',fontsize=10)
    ax.set_xticks(range(3),['24','40','All']);ax.set_xlabel('Training graphs');ax.set_ylabel('Development tau');ax.legend(fontsize=8,loc='upper left')
for ax in axes.flat: ax.grid(alpha=.15)
fig.tight_layout(pad=.65,w_pad=1.1,h_pad=1.5);fig.savefig(OUT/'scaling.pdf',bbox_inches='tight');plt.close(fig)
print('Wrote five source-derived tables and three vector figures to',OUT)
