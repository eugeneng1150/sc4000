"""Build the standalone final presentation notebook from the immutable evidence pack."""
import base64
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
cells=[]
def cell(kind,text,**metadata):
    text=text.strip()+'\n'
    c=dict(cell_type=kind,id=hashlib.sha256(text.encode()).hexdigest()[:12],metadata=metadata,source=text.splitlines(keepends=True))
    if kind=='code':c.update(outputs=[],execution_count=None)
    cells.append(c)
def md(s):cell('markdown',s)
def code(s,**kwargs):cell('code',s,**kwargs)
md('''# SC4000 — Learning to rank compiler configurations
## Frozen final system and experimental evidence

**Final public Kaggle score: 0.567 (user-reported). Frozen on 21 October 2026.**

We rank compiler configurations using collection-specific histogram gradient boosting
models and sampled graph neural networks, combining their predictions through rank
ensembles. A within-graph percentile target provides a useful complementary HGB model.

This is the consolidated final notebook. The exploratory notebooks contain the search
history; this notebook explains the final method and recreates its evidence in one place.
Its default run **does not train, tune, download the dataset, or perform fresh model
inference**. It loads an embedded, hash-checked evidence pack, generates report figures,
and exports the exact previously evaluated submission. The full checkpoint remains the
saved-model archive; its selected model files can be verified optionally below.

**Outputs:** `submission_final.csv`, ten PNG/PDF figures, supporting CSV tables,
figure captions, and a report artifact ZIP. The submission hash must be
`dc037f378ddb9d0892c283a26a7f31170ac6cc01e937e4d14f2847296ff80312`.

No other notebook or uploaded ZIP is needed for the default evidence replay.
Keep the complete model checkpoint separately: the compact embedded pack is not a model backup.
''')
md('''## 1. Run settings and frozen evidence

Run from top to bottom on Colab CPU or locally with NumPy, pandas and Matplotlib.
No package upgrades are performed. Those plotting packages are separate from the exact
scientific versions recorded for the trained model archive.

`MODEL_ARCHIVE` is optional: leave it empty to skip checking the full saved-model archive.
Set it to your complete downloaded ZIP or extracted results folder to verify all selected
models without loading pickle files. `DOWNLOAD_OUTPUTS` controls only the final browser
request; verify that the file actually reaches your computer.
''')
code('''from pathlib import Path
import os, sys, json, base64, hashlib, io, zipfile, types
BASE_DIR = Path("/content") if Path("/content").exists() else Path.cwd()
OUTPUT_DIR = BASE_DIR / "sc4000_final_frozen"
MODEL_ARCHIVE = ""  # optional complete saved-model ZIP or extracted results folder
DOWNLOAD_OUTPUTS = False
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(OUTPUT_DIR / ".matplotlib")
from IPython.display import display, Image, Markdown
''')
payload=(HERE/'frozen_evidence.zip').read_bytes();digest=hashlib.sha256(payload).hexdigest()
module=(HERE/'final_report.py').read_text()
code(f'''# Self-contained audited report code and compact frozen evidence; no training calls.
REPORT_SOURCE = {module!r}
EVIDENCE_B64 = {base64.b64encode(payload).decode()!r}
EVIDENCE_SHA256 = {digest!r}
final_report = types.ModuleType("sc4000_final_report")
exec(compile(REPORT_SOURCE, "embedded_final_report.py", "exec"), final_report.__dict__)
EVIDENCE_DIR = OUTPUT_DIR / "evidence"
freeze = final_report.unpack(base64.b64decode(EVIDENCE_B64), EVIDENCE_SHA256, EVIDENCE_DIR)
print("Frozen submission:", freeze["submission_sha256"])
print("Rows:", freeze["submission_rows"], "| selected model files:", freeze["selected_model_files"])
print("Public score:", freeze["public_score_user_reported"], "(user-reported)")
''',jupyter={'source_hidden':True})
code('''import pandas as pd
import numpy as np
FIGURES = final_report.generate(EVIDENCE_DIR, OUTPUT_DIR)
def table(name):
    return pd.read_csv(OUTPUT_DIR / "tables" / (name + ".csv"))
def evidence_table(name):
    return pd.read_csv(EVIDENCE_DIR / "tables" / (name + ".csv"))
def show_figure(name):
    display(Image(filename=str(FIGURES / (name + ".png"))))
print("Report figures and source tables regenerated from saved evidence.")
''')
md('''## 2. Task, data and evaluation

The input is a computation graph and candidate compiler configurations. The output is
an ordering of configuration IDs, with faster predicted candidates first. There are five
collections: NLP default/random layout, XLA default/random layout, and XLA tiling. These
are dataset groups, not model families. Each has its own selected predictor or blend.

The saved training inventory below describes all supplied training graphs. Actual model
fits use bounded subsets, documented later. Layout graphs can be large, and the number
of configurations makes fully materializing their arrays expensive.

For local **layout** evaluation we use Kendall's tau between runtimes and predicted
costs: higher is better. For **Tile** we use one minus top-five regret on sampled
candidates, where regret is `fastest_runtime_among_predicted_top5 / best_runtime - 1`.
Local candidate subsets and graph counts differ from full Kaggle evaluation; the metrics
must not be averaged or treated as interchangeable with the public score.

The final CSV follows each sample-submission row's required output length. A layout row
need not contain every possible candidate when the template requests fewer. Short Tile
rows retain the established template-padding convention.
''')
code('''display(table("dataset_summary"))
show_figure("02_data_and_coverage")
''')
md('''## 3. Preprocessing and representation

**Graph-level splitting.** Two fixed development folds reserve disjoint development graph
blocks. Within each fold, its development graphs are excluded from training. A graph may
be used for training in the other fold; that other model is not used to score its development
fold. Official validation graphs form an audit set that has already been inspected during
exploration. We do not describe it as a new untouched test set.

**Configuration sampling and memory bounds.** Fixed label-independent candidate pools
and recorded training-ID manifests make comparisons repeatable. Selected configuration
rows are streamed from NPZ members in bounded chunks; full layout tensors are not loaded
at once. Graph/configuration caps, matrix guards, isolated workers and scratch cleanup
keep the training workflow within the intended 12–13 GB Colab CPU environment. Limits
can exclude an experiment; exclusion does not mean that model performed badly.

**Layout features.** The `-1` sentinel is distinguished from valid axis categories. We
compute masked statistics and validity fractions, per-position categorical histograms,
hashed opcode/layout and depth/layout summaries, edge-category co-occurrences, and compact
graph topology features. Axis IDs in categorical histograms are categories, not physical
sizes. The HGB feature blocks used by each selected model are recorded in the recipe JSON.

**Tile features.** The 24 numeric configuration inputs use raw summaries, signed-log
transforms and selected pair interactions alongside graph summaries. Layout padding
semantics are not imposed on Tile inputs.

**Targets and weights.** Original HGB models predict `log(1 + runtime)` centered by the
training graph's median. The alternative target is `(rank(runtime) - 1)/(m - 1)` within
the selected training configurations of each graph (with denominator at least one).
Ties use average ranks. Graph weighting prevents configuration-rich graphs from dominating
simply through their row counts. Validation runtimes are never used to construct training
labels or label-independent features.
''')
code('''show_figure("01_pipeline")
display(evidence_table("reference_split_inventory").query("split != 'train'"))
''')
md('''## 4. Models and the frozen final recipe

**Baseline:** basic-summary histogram gradient boosting (`B0_basic_hgb`). A richer
feature HGB provides the stronger tabular comparison. Differences in training size and
capacity are documented; the complete pipeline comparison is not a single-component ablation.

**Sampled GNN:** a 16-dimensional opcode embedding, standardized numeric node features,
configuration values and optional validity masks feed a width-32 network with two directed
mean-neighbour message-passing layers and residual updates. Mean/max pooling over all
retained nodes and configurable nodes is combined with optional context summaries. A
pairwise hinge objective encourages lower predicted cost for faster configurations;
batches without eligible runtime differences fall back to smooth-L1 regression.

The selected XLA default GNN uses a 2,048-node cap and 256 training configurations per
graph; the selected XLA random GNN uses 512 nodes and 128 configurations. Both use all
eligible training graphs for their fold. These caps do not imply complete graph retention.

**Final ensemble:** each saved fold/seed predictor scores candidates; scores are converted
to within-graph normalized ranks before averaging and blending. The weights below are
fixed by development selection. They are not weights learned from Kaggle test labels.
No XGBoost model was retained in the final system.

The table keeps original experiment IDs so every entry can be traced to a saved model.
A training graph cap of zero means all eligible graphs, not zero examples. The actual
counts in the second table are the authoritative record of what each model saw.
''')
code('''display(table("final_model_recipe"))
sizes = table("actual_training_sizes")
display(sizes.groupby(["collection", "experiment", "fold"], as_index=False).agg(
    training_graphs=("graphs", "first"), configurations=("configurations", "first"),
    min_configs=("min_configs", "min"), max_configs=("max_configs", "max")))
''')
md('''## 5. What the ablations establish

### Components and graph structure

We tested validity masks, context summaries, message passing, graph budgets and training
budgets. These are distinct comparisons, not cumulative improvements that can be added.
The component plot uses graph-paired comparisons and exploratory bootstrap intervals.
**All displayed mask/context/message-passing intervals cross zero.** Therefore this
experiment does not establish a clear standalone benefit for those components. A final
model can contain a component without its individual gain having been proven.

Increasing the sampled graph budget did not consistently improve development scores.
Changing training graph/configuration counts sometimes helped, but the effect varied by
collection and fold. More training examples also change the number of optimizer steps;
we do not claim a compute-matched scaling law. Coverage limits are a plausible constraint,
not a demonstrated cause of poorer GNN performance.
''')
code('''show_figure("03_component_ablations")
show_figure("04_graph_budget")
show_figure("05_training_data")
''')
md('''### Training-data expansion and target/objective comparisons

Expanding tabular training data helped NLP random in the earlier development selection;
the existing NLP default and Tile predictors were retained. Larger Tile settings met
resource exclusions, so they cannot be called negative performance results. Longer GNN
training did not pass the fold-consistency selection rule.

The final controlled study kept graph/configuration IDs matched to the selected tabular
reference and compared: (1) percentile-target HGB; (2) XGBoost regression; and (3) grouped
XGBoost pairwise ranking. HGB control versus percentile changes the target. XGBoost
regression versus pairwise changes the objective within fixed XGBoost settings. Comparing
HGB directly with XGBoost also changes learner/capacity.

Screening used seed 42 on two folds. Promising models received seeds 137 and 2026. A
small predefined blend grid was eligible only if its mean development gain was at least
0.002 and neither fold regressed. This is a practical selection rule, not a significance
test. Some deterministic tree predictions are identical across seeds; repeated seeds
are not independent graph replications.

**Result:** percentile HGB complements the incumbent on NLP default, NLP random and XLA
default. In NLP random, the percentile model alone is slightly worse on mean development
score, but its blend improves. No new candidate was selected for XLA random. Tile was
carried forward. The study finished with 36 new fits; 56 was only the initial upper bound.
''')
code('''show_figure("06_target_and_objective")
display(table("target_blend_audit")[["collection", "dev_gain", "baseline_score",
    "candidate_score", "mean_paired_gain", "graphs", "bootstrap_low", "bootstrap_high"]])
''')
md('''## 6. Final results and interpretation

The table and audit chart compare the frozen system with the basic HGB baseline. The
stronger tabular comparison is also retained in the underlying `final_system_sources`
table. In particular, the XLA random hybrid does not outperform the earlier tabular
ensemble on every audit comparison; selection was made on development data, not by
reselecting whichever model later looked best on the audit set.

All three accepted final updates improve both development folds and their reused audit
scores. Audit gains versus the preceding selected system are **+0.0423** (NLP default),
**+0.0118** (NLP random), and **+0.0592** (XLA default). XLA audit sets contain only seven
graphs; NLP sets contain twenty. Bootstrap intervals are exploratory, not corrected for
multiple comparisons, and the audit has been inspected before.

The user-reported public score increased from **0.558 to 0.567**, an absolute gain of
**0.009**, approximately **1.61% relative**. Three collections changed together, so we
cannot attribute that exact public gain to a single collection. Earlier baseline/tabular/
hybrid scores in the chart are approximate conversation reports, not independently verified
measurements. Add the Kaggle screenshot and exact ranking position before submitting the
report. We do not infer a top-10% ranking from the score.
''')
code('''display(table("baseline_and_final_audit"))
show_figure("07_final_audit")
show_figure("08_public_scores")
''')
md('''## 7. Training behaviour and resource constraints

These curves come from the **GNN checkpoints actually used in the final ensemble**,
not the rejected longer-training candidates. Each line is one fold/seed; dots mark saved
selected epochs. The left panel is training loss; the right panel is development ranking
score on the smaller early-stopping subset. They have different units and should not be
plotted as two comparable losses.

A falling training loss with declining development tau can indicate overfitting. A
plateau indicates little additional validation benefit. Early stopping limits training
past the best observed checkpoint, but the curves do **not** prove absence of overfitting.
There is no recorded per-epoch validation-loss series to reconstruct honestly, and these
GNN curves do not diagnose the entire HGB/GNN ensemble.

In the saved curves, XLA random training loss generally decreases while development
scores fluctuate and often peak before the last epoch. XLA default is less stable,
with several low or near-zero development trajectories. These observations support
checkpoint selection and explicit uncertainty, not a claim of uniformly successful
GNN training or no overfitting.

The resource chart shows measured worker peaks and wall times for the target/objective
study. Parallel jobs were admitted only after pilots and memory/CPU checks, with sequential
fallback. Summed worker time is not total elapsed time. Streamed preparation and repeated
feature extraction can remain expensive even when worker RAM use is small.
''')
code('''show_figure("09_gnn_learning_curves")
display(table("gnn_learning_diagnostics"))
show_figure("10_training_resources")
''')
md('''## 8. Freeze, provenance and submission export

The compact evidence pack includes measured results, selected recipes, underlying source
code for inspection, and hashes for all **60 selected saved model files**. It does not
contain the model weights. Keep the complete checkpoint ZIP as the model archive.

The following cell optionally verifies that archive by streaming each selected file's
hash. It does not load model objects. Reproducing fresh predictions from raw NPZs remains
a separate saved-model inference task using the archived code and exact dependency
versions; exporting the saved submission below is explicitly **prediction replay**, not
fresh inference or retraining.
''')
code('''if MODEL_ARCHIVE:
    print(final_report.verify_model_archive(MODEL_ARCHIVE, EVIDENCE_DIR))
else:
    print("Full model archive check skipped; compact evidence integrity was verified.")
print("Original trained-model dependency versions:")
display(final_report.read(EVIDENCE_DIR / "run_identity.json")["versions"])
SUBMISSION = final_report.export_submission(EVIDENCE_DIR, OUTPUT_DIR)
print("Verified submission:", SUBMISSION)
print(final_report.validate_submission(EVIDENCE_DIR))
''')
md('''## 9. Report and presentation handoff

Organize the report around **one solution**, not notebook version numbers:

1. Task, challenges, group contributions and reported leaderboard result.
2. Preprocessing, feature design, bounded GNN and collection-specific rank blends.
3. Baselines and controlled ablations; explain both positive and inconclusive results.
4. Final audit/public results, computational constraints and limitations.
5. Conclusions and references.

The supplied course brief requires the AAAI format, **at most ten pages including
references**, a presentation of at most ten minutes, and the prediction CSV or notebook
with saved models to check the reported result. A new full training sweep is not required
by that brief. Keep source code and the saved-model archive for possible checking.

For the ten-page report, prioritize figures 01, 03, 06, 07 and 09; combine or omit the
remaining figures according to space. Tables/figures need explanatory text. Use the
caption file for metric definitions and evidence limits. Complete the team contribution
statement and exact leaderboard screenshot/rank outside this automated evidence pack.

**Attribution:** HGB, GNN message passing, ranking losses and percentile transforms are
established methods; we do not claim to have invented them. Our project contribution is
the implemented representation/combination and the empirical investigation. Acknowledge
any participant code consulted as a reference in the final report; the course prohibits
directly using participant solutions. The repository's research notes retain the reading
trail. Do not claim unproven standalone mask/context gains or novelty of standard methods.

**Stopping decision:** freeze the 0.567 submission and focus on a convincing, reproducible
presentation of the evidence rather than another speculative sweep toward 0.590.
''')
code('''# Small report deliverable: figures, tables, captions, verification and exact CSV.
# Full model archive is deliberately separate and must be retained separately.
REPORT_ZIP = OUTPUT_DIR / "sc4000_final_report_artifacts.zip"
with zipfile.ZipFile(REPORT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for folder in [OUTPUT_DIR / "figures", OUTPUT_DIR / "tables"]:
        for path in sorted(folder.glob("*")):
            if path.is_file(): archive.write(path, str(path.relative_to(OUTPUT_DIR)))
    for name in ["FIGURE_CAPTIONS.md", "submission_final.csv", "submission_verification.json"]:
        archive.write(OUTPUT_DIR / name, name)
    archive.write(EVIDENCE_DIR / "freeze_manifest.json", "freeze_manifest.json")
print("Report artifacts:", REPORT_ZIP)
print("Kaggle file:", SUBMISSION)
if DOWNLOAD_OUTPUTS:
    try:
        from google.colab import files
        files.download(str(REPORT_ZIP))
        print("Download requested. Verify completion on your computer.")
    except ImportError:
        print("Outside Colab: copy the files from the paths above.")
''')
notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
    language_info=dict(name='python',version='3.11'),colab=dict(name='SC4000_Final.ipynb')),
    nbformat=4,nbformat_minor=5)
target=HERE.parent/'SC4000_Final.ipynb';target.write_text(json.dumps(notebook,indent=1)+'\n')
print(target,len(cells),'cells')
