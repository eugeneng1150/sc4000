"""Small report pack with source tables; no inference of missing leaderboard results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from v4_core import read
from v5_core import tabular_reference


def generate(output):
    output=Path(output)
    dest=output/'report';dest.mkdir(exist_ok=True)
    rows=[];decisions=[];pilots=[]
    for d in sorted(output.glob('layout_*')):
        control_name,control=tabular_reference(output/'reference_v4',d.name.replace('_',':',2))
        for p in Path(control['directory']).glob(f'folds/*/trials/{control_name}/*/result.json'):
            r=read(p)
            rows.append({**r,'collection':d.name,'fold':int(p.parents[3].name),
                         'experiment':'HGB_control','source_experiment':control_name,'reused':True})
        for p in d.glob('folds/*/trials/*/*/result.json'):
            r=read(p)
            rows.append({**r,'collection':d.name,'fold':int(p.parents[3].name),'reused':False})
        for p in d.glob('admission/*/*/decision.json'):
            r=read(p)
            pilots.append(dict(collection=d.name,fold=p.parents[1].name,
                               experiment=p.parent.name,**{k:v for k,v in r.items() if k!='pilots'}))
        for name in ['screening_selection','selection_lock']:
            p=d/(name+'.json')
            if p.exists():
                lock=read(p)
                for r in lock['candidates']:
                    decisions.append(dict(collection=d.name,phase=name,
                        components=json.dumps(r['components']),mean_score=r['mean_score'],
                        gain_vs_v4=r['gain_vs_v4'],eligible=r['eligible'],
                        fold_0=r['fold_scores'][0],fold_1=r['fold_scores'][-1]))
    trials=pd.DataFrame(rows);choices=pd.DataFrame(decisions);admissions=pd.DataFrame(pilots)
    trials.to_csv(dest/'trial_results.csv',index=False)
    choices.to_csv(dest/'selection_candidates.csv',index=False)
    admissions.to_csv(dest/'resource_estimates.csv',index=False)
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','pdf']:
            fig.savefig(dest/f'{name}.{ext}',dpi=170,bbox_inches='tight')
        plt.close(fig)
    if not trials.empty:
        frame=trials[trials.seed==42]
        collections=list(frame.collection.unique())
        fig,axes=plt.subplots(len(collections),1,figsize=(9,3*len(collections)),squeeze=False)
        for ax,c in zip(axes.flat,collections):
            subset=frame[frame.collection==c]
            table=subset.pivot(index='experiment',columns='fold',values='dev_score')
            table.plot.bar(ax=ax,rot=0)
            ax.tick_params(axis='x',labelsize=8)
            ax.set_title(c+' — seed-42 screening')
            ax.set_ylabel('Development Kendall tau')
            ax.set_xlabel('')
        save(fig,'01_target_and_learner_screening')
    locks=[(d.name,read(d/'selection_lock.json')) for d in output.glob('layout_*') if (d/'selection_lock.json').exists()]
    if locks:
        fig,ax=plt.subplots(figsize=(9,4))
        ax.bar([c for c,_ in locks],[v['gain_vs_v4'] for _,v in locks],color='#287c8e')
        ax.axhline(0,color='black',linewidth=.7)
        ax.set_ylabel('Selected development gain vs v4')
        ax.set_title('Frozen selection; zero means the v4 system was retained')
        ax.tick_params(axis='x',rotation=15)
        save(fig,'02_selected_system_gain')
    logs=output/'scheduler_events.jsonl'
    events=[json.loads(s) for s in logs.read_text().splitlines()] if logs.exists() else []
    done=pd.DataFrame([e for e in events if e['event']=='finish'])
    if not done.empty:
        done.to_csv(dest/'worker_resources.csv',index=False)
        tasks=done[done.label.str.endswith('| trial')]
        if not tasks.empty:
            fig,axes=plt.subplots(1,2,figsize=(11,4))
            axes[0].plot(np.arange(1,len(tasks)+1),tasks.wall_seconds/60,'o')
            axes[0].set_ylabel('Worker wall time (minutes)')
            axes[1].plot(np.arange(1,len(tasks)+1),tasks.peak_rss_gib,'o')
            axes[1].set_ylabel('Measured worker peak RSS (GiB)')
            for ax in axes:
                ax.set_xlabel('Finished training job')
            save(fig,'03_actual_training_resources')
    audit=[pd.read_csv(p).assign(collection=p.parent.name) for p in output.glob('layout_*/system_audit_gains.csv')]
    if audit:
        pd.concat(audit,ignore_index=True).to_csv(dest/'audit_gains.csv',index=False)
    (dest/'CAPTIONS.md').write_text(
        '# V5 evidence scope\n\n'
        '01: Individual-model development tau on the two fixed folds, seed 42. '
        'HGB_control is reused from the selected reference tabular model. '
        'Compare HGB percentile with that control; compare XGBoost '
        'regression with pairwise ranking. The models use identical example IDs per fold/seed. '
        'XGBoost and HGB differ in implementation/capacity; their direct comparison does not isolate loss.\n\n'
        '02: Development-selected system gain against v4. The zero-fold-regression rule and '
        'minimum gain of 0.002 are pragmatic thresholds, not significance tests. '
        'Full candidate fold scores, including rejected positive averages, are in selection_candidates.csv.\n\n'
        '03: Actual worker time and peak RSS; parallel worker times must not be summed as elapsed wall time. '
        'Pilots predict fitting only; extraction, evaluation, inference and backups are extra. '
        'Scheduling events include concurrency and sequential fallback.\n\n'
        'Repeated seeds may produce identical tree predictions; do not treat repetitions as independent graphs. '
        'Development reuse is adaptive; official validation is an already-inspected audit. '
        'Audit bootstrap intervals are exploratory and not multiplicity corrected.\n\n'
        'V4 Kaggle score 0.558 is user-reported. No v5 Kaggle score is known automatically. '
        'The original six main figures and source tables remain in reference_v4/report/; '
        'their archived leaderboard panel predates the 0.558 observation.\n')
    return dest
