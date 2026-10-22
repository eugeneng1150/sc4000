"""Build the standalone raw-data reproduction and experimental narrative notebook."""
import base64
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
cells=[]
def cell(kind,text,**metadata):
    text=text.strip()+'\n'
    c=dict(cell_type=kind,id=hashlib.sha256(text.encode()).hexdigest()[:12],metadata=metadata,source=text.splitlines(keepends=True))
    if kind=='code':c.update(outputs=[],execution_count=None)
    cells.append(c)
def md(s):cell('markdown',s)
def code(s,**kwargs):cell('code',s,**kwargs)

def runtime_sources():
    result={}
    for name in ['engine','gnn','common','plan','submission']:
        result['study_'+name+'.py']=(REPO/'src/ablation_v3'/f'{name}.py').read_text()
    for name in ['v4_core','v4_worker']:
        result[name+'.py']=(REPO/'src/ablation_v4'/f'{name}.py').read_text()
    for name in ['v5_worker','v5_scheduler']:
        result[name+'.py']=(REPO/'src/ablation_v5'/f'{name}.py').read_text()
    for name in ['final_worker','reproduce','final_report','story']:
        result[name+'.py']=(HERE/f'{name}.py').read_text()
    return result

md('''# SC4000 — Ranking compiler configurations under limited resources
## From a boosted-tree baseline to a collection-specific graph–tabular ensemble

**Research question:** how can we improve configuration rankings while staying within a
standard Colab CPU session's approximately 12–13 GB RAM?

Our investigation starts with a basic boosted-tree baseline, explores compact features
and graph learning, tests representation/data/objective changes, and combines models
where their predictions complement one another. The final selected submission achieved
**0.567 on the public leaderboard (user-reported)**. That score describes the historical
submission, not an unsubmitted rerun.

**This notebook is executable from raw competition data without a model ZIP.** It contains
all source code, fixed recipes, graph splits, configuration sampling IDs and seeds.
Compatible models are loaded; missing models are trained. Predictions are generated from
those models, not copied from the historical CSV. A model archive is an optional shortcut.

Default execution reproduces **60 selected fits + 30 basic-baseline fits**, across two
folds and three seeds. It does not search for a different final model. The optional ablation
switch reruns recorded experiments without changing the selected recipe. A fresh CPU run
can require multiple sessions: inspect the job count, retain backups and resume completed fits.

**Reading guide:** each investigation follows *question → experiment → observation → decision*.
Historical figures are explicitly labelled. Current-run evaluation appears separately at
section 7. Saved presentation outputs do not claim that a new full training run has completed.
''')
md('''## 0. Execution settings and reproducibility

Use `RUN_MODE = "reproduce"` for the normal load-or-train workflow. `"review"` only
regenerates the historical narrative and figures; it deliberately produces no new submission.

Set `DATA_ROOT` to extracted competition data, or enable the existing Kaggle download helper
with your own credentials and competition access. Dataset access is required for training
and fresh inference even when models are available. No private paths or uploaded model ZIP
are required. Model training uses the recorded bounded subsets, not silently all supplied data.

A complete **historical v5** archive can be supplied as `MODEL_ARCHIVE`. New final-run backups
instead go in `RESTORE_RUN_ZIP`; restore once into an empty output directory, then clear it
when resuming in the same runtime. Existing incompatible runs require a different `RUN_NAME`.
''')
code('''from pathlib import Path
import os, sys, json, base64, hashlib, io, zipfile, shutil, subprocess, importlib.metadata
RUN_MODE = "reproduce"  # "review" renders historical evidence only
RUN_ABLATIONS = False
ABLATION_SEEDS = [42]  # [42, 137, 2026] reruns all recorded confirmations too
DATA_ROOT = ""
SAMPLE_SUBMISSION = ""
DOWNLOAD_IF_MISSING = False
MODEL_ARCHIVE = ""  # optional complete historical v5 ZIP or extracted folder
RESTORE_RUN_ZIP = ""  # optional new final-reproduction backup, restored ONCE
RUN_NAME = "sc4000_final_reproduction"
AUTO_DOWNLOAD_CHECKPOINTS = True
SESSION_HOURS = 8.0
CHECKPOINT_MINUTES = 30
WORKER_RAM_GIB = 5.0
SYSTEM_RESERVE_GIB = 2.0
CPU_THREADS = 2
BASE_DIR = Path("/content") if Path("/content").exists() else Path.cwd()
OUTPUT_DIR = BASE_DIR / RUN_NAME
SCRATCH_DIR = BASE_DIR / ("." + RUN_NAME + "_scratch")
RUNTIME_DIR = BASE_DIR / ("." + RUN_NAME + "_runtime")
STORY_DIR = BASE_DIR / (RUN_NAME + "_historical_report")
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(RUNTIME_DIR / "matplotlib")
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[key] = str(CPU_THREADS)
if RUN_MODE not in {"reproduce", "review"}: raise ValueError("Choose reproduce or review")
''')
payload=(REPO/'artifacts/final/frozen_evidence.zip').read_bytes()
spec=(REPO/'artifacts/final/reproduction_spec.zip').read_bytes()
code(f'''# Embedded source and small experiment specification. Neither archive contains weights or raw data.
SOURCES = {runtime_sources()!r}
EVIDENCE_B64 = {base64.b64encode(payload).decode()!r}
EVIDENCE_SHA256 = {hashlib.sha256(payload).hexdigest()!r}
SPEC_B64 = {base64.b64encode(spec).decode()!r}
SPEC_SHA256 = {hashlib.sha256(spec).hexdigest()!r}
SPEC_BYTES = base64.b64decode(SPEC_B64)
if hashlib.sha256(SPEC_BYTES).hexdigest() != SPEC_SHA256: raise ValueError("Specification checksum failed")
with zipfile.ZipFile(io.BytesIO(SPEC_BYTES)) as z:
    CONTRACT = json.loads(z.read("contract.json"))
for name, source in SOURCES.items():
    (RUNTIME_DIR / name).write_text(source)
sys.path.insert(0, str(RUNTIME_DIR))
print("Embedded implementation and exact experimental specification ready.")
''',jupyter={'source_hidden':True})
md('''### Dependencies

Reproduction uses the recorded scientific versions. If packages already imported in this
session are upgraded, restart the session and Run all again; keep the data and results.
Installing pinned dependencies can take several minutes. Review mode uses the available
plotting environment and does not load trained model objects.
''')
code('''if RUN_MODE == "reproduce":
    wanted = dict(CONTRACT["versions"])
    if not RUN_ABLATIONS: wanted.pop("xgboost", None)
    changed = []
    for package, version in wanted.items():
        try: installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: installed = None
        if installed != version: changed.append(package)
    ordinary = [f"{p}=={wanted[p]}" for p in changed if p != "torch"]
    if ordinary:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir", *ordinary])
    if "torch" in changed:
        args = [sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir"]
        if sys.platform == "linux": args += ["--index-url", "https://download.pytorch.org/whl/cpu"]
        subprocess.check_call(args + ["torch==" + wanted["torch"]])
    modules = {"scikit-learn": "sklearn"}
    if any(modules.get(p,p) in sys.modules for p in changed):
        raise RuntimeError("Dependencies updated. Restart session, then Run all; keep files/settings.")
    print("Scientific versions:", wanted)
else:
    print("REVIEW ONLY: historical figures; no models loaded, trained or submitted.")
import numpy as np
import pandas as pd
from IPython.display import display, Image, Markdown
import final_report, story
EVIDENCE_DIR = STORY_DIR / "evidence"
freeze = final_report.unpack(base64.b64decode(EVIDENCE_B64), EVIDENCE_SHA256, EVIDENCE_DIR)
FIGURES = story.generate(EVIDENCE_DIR, STORY_DIR, CONTRACT)
def table(name): return pd.read_csv(STORY_DIR / "tables" / (name + ".csv"))
def show(name): display(Image(filename=str(FIGURES / (name + ".png"))))
print("Historical figures regenerated. These are recorded experiment results, not new-run metrics.")
''')
md('''## 1. Problem and evaluation: what does a good prediction mean?

A computation graph is a program; each candidate configuration chooses compiler settings
for that same program. We predict a **ranking**, with faster configurations first. The
five collections—NLP default/random layout, XLA default/random layout and XLA tiling—are
dataset groups, not model families.

**Layout:** local evaluation uses Kendall tau-b between measured runtime and predicted
cost, averaged over graphs; higher is better. **Tile:** local evaluation uses one minus
top-five regret: `1 - (best runtime among predicted top five / best runtime - 1)`.
These sampled local metrics are not the aggregate Kaggle score and must not be averaged together.

**What is Kendall’s tau, and why use it?** It measures how closely our predicted ordering
matches the actual fastest-to-slowest ordering. For each pair of configurations, it asks:
if A is actually faster than B, did our model also place A before B? Correctly ordered
pairs increase the score; reversed pairs decrease it. Tau-b accounts for tied values.

For an actual order **A → B → C**, with no ties:

| Predicted order | Interpretation | Kendall’s tau |
|---|---|---|
| A → B → C | All three pairs are correct | 1 |
| A → C → B | Two pairs correct, one reversed | approximately 0.33 |
| C → B → A | All three pairs reversed | −1 |

Higher is better; values near zero indicate little net agreement in pair ordering.
**Tau = 0.5 does not mean 50% accuracy.** This metric evaluates ordering rather than
how close predicted runtimes are in seconds.

Kendall’s tau is an established statistical metric used in the original
[TPUGraphs research](https://papers.nips.cc/paper/2023/file/ded1a89e2b3b925444ada973af66336e-Paper-Datasets_and_Benchmarks.pdf),
the dataset behind this competition. We adopt it because the task requires ranking
compiler configurations; it is not our proposed modelling contribution. Our implementation
uses [SciPy’s Kendall tau-b](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html).
The local mean tau reported here remains distinct from the historical aggregate Kaggle score of **0.567**.

We reserve graph-disjoint development data within each of two folds. A graph can train the
other fold's model, but that model never scores its development fold. Official validation
is an already-inspected audit, not a new untouched test. XLA audit sets have only seven graphs.

**Constraint:** configuration tensors can be enormous even when model parameters are small.
We stream sampled NPZ rows and bound graph size, feature matrices, scratch disk and worker RAM.
The inventory below describes the supplied data; it does not mean every model trains on all of it.
''')
code('''display(table("dataset_inventory"))
display(table("split_inventory"))
# Illustrative ranking example, not a measured experiment.
example = pd.DataFrame({"configuration": [0,1,2,3], "runtime": [12.,8.,15.,10.],
                        "predicted_cost": [.7,.1,.9,.4]})
display(example)
print("Predicted order:", example.sort_values("predicted_cost").configuration.tolist())
''')
md('''## 2. Preprocessing and a clear baseline

**Question:** can an inexpensive model give us a useful reference before using graph learning?

The baseline is **basic-summary histogram gradient boosting**, `B0_basic_hgb`. It is separate
from the *reference GraphSAGE* used later for component ablations. We reproduce both baseline
and selected-model recipes from their exact recorded training graph/configuration IDs.

**Preprocessing:** fixed label-independent candidate pools; graph-level splits; bounded row
extraction; numeric graph summaries; and within-graph targets. Original HGB fits graph-centred
`log(1 + runtime)`, with per-graph weighting. Layout representations distinguish padding from
valid axis categories. Richer models add masked statistics, categorical position histograms,
opcode/depth associations, edge co-occurrences and compact topology/WL summaries. Tile uses
its numeric configuration features and signed-log/interactions. Training labels and feature
transforms never use development/audit runtimes to fit a model.

**Why these features?** A graph summary alone is identical for every configuration of that
graph. Configuration-aware features provide information that can distinguish its candidates.
The feature bundle is tested as a bundle; we do not claim every constituent independently helps.
''')
code('''show("01_method")
baselines = [s for s in CONTRACT["jobs"] if "baseline" in s["roles"] and s["fold"]==0 and s["seed"]==42]
display(pd.DataFrame([{"collection":s["collection"], "model":s["experiment"]["name"],
    "features":", ".join(s["experiment"]["blocks"]), "target":s["experiment"]["target"],
    "trees":s["experiment"].get("iterations",s["cfg"]["hgb_iterations"])} for s in baselines]))
''')
md('''## 3. Exploring model families: do we need graph learning?

We explored three main directions:

| Direction | Motivation | Evidence and decision |
|---|---|---|
| MLP on compact features | A neural predictor without graph message passing | Earlier exploratory results are retained below; their protocol differs from later studies. |
| Gradient boosting | Nonlinear feature interactions with manageable CPU cost | Remains a strong component across collections and the sole final Tile predictor. |
| GraphSAGE | Learn how neighbouring operations interact with configuration choices | Retained in both XLA layout blends; performance varies across collections/folds. |
| Gated GraphSAGE | Learn how much each neighbour contributes | Code exists on an earlier branch, but no verified run result was recovered. It is not presented as a measured success or failure. |

The selected GraphSAGE uses opcode embeddings, numeric/configuration inputs, two width-32
mean-neighbour message-passing layers with residual updates, graph/configurable-node pooling
and a pairwise ranking loss. It is an established architecture, not a new invention.

The chart below uses the **same later development folds** to compare GraphSAGE and a matched-data
HGB control. The basic HGB is a separate baseline with its own budget. We show both folds so
instability is visible. We do not combine the earlier MLP scores into this controlled chart.
''')
code('''show("02_model_families")
display(table("model_family_comparison"))
with zipfile.ZipFile(io.BytesIO(SPEC_BYTES)) as z:
    early = pd.read_csv(z.open("early_model_exploration.csv"))
display(Markdown("**Earlier exploration only — different protocol; do not compare numerically with the chart above.**"))
display(early.pivot(index="experiment", columns="collection", values="ranking_score").round(4))
''')
md('''**Decision:** keep both compact tabular models and graph learning under investigation.
Neither model family is assumed to win on every collection. The next experiments target
specific possible limitations rather than replacing the whole system blindly.

## 4. Controlled improvements: what worked and what did not?
### 4.1 Representation and message passing

**Question:** do explicit validity masks and context summaries improve the reference GNN?
We compare additions with their named reference and separately examine removal controls.

**Observation:** the displayed confirmed intervals cross zero. These results do not establish
clear standalone gains for masks/context. A selected model can contain them without proving
that each component is independently beneficial. Removal tests ask a different question from
introducing a novel component. The source table names each control so these effects are not
mistakenly added together.

**Decision:** report the uncertainty; retain the development-selected recipe, without inventing
a cumulative improvement story.
''')
code('''show("03_components")
display(table("component_changes")[["collection","factor","reference","experiment",
    "baseline_score","candidate_score","absolute_gain","relative_gain_pct","bootstrap_low","bootstrap_high"]])
''')
md('''### 4.2 Graph coverage and training-data budgets

**Motivation:** severe graph sampling may discard useful structure, and limited training data
may restrict generalization. These are hypotheses, not established explanations of weak scores.

**Experiment:** increase sampled graph budgets; separately vary training graph counts and
configurations per graph. Follow-up work expands tabular training and tests longer GNN training.

**Observation:** larger sampled graphs and more data do not consistently help. Tabular expansion
was accepted for NLP random; existing NLP default/Tile models remained preferable under the
selection rule. Longer GNN training was not accepted. Large resource-excluded Tile settings
were not accuracy failures—they were not successfully evaluated.

**Decision:** use the selected bounded recipes. More retained structure is not automatically
better, and more examples also change optimizer work. These are not compute-matched experiments.
''')
code('''show("04_coverage_and_data")
display(table("training_data_screening").round(4))
''')
md('''### 4.3 Target transformation and ranking objectives

**Motivation:** configurations need to be ordered within each graph. Runtime scale varies across
graphs, so we test whether changing the target helps the learner focus on relative performance.

**Controlled tests:** HGB graph-centred log-runtime versus within-graph percentile target;
XGBoost regression versus grouped pairwise ranking with fixed XGBoost capacity. HGB versus XGBoost
also changes learner/capacity and is not an isolated objective ablation. Faster configurations
have higher relevance for XGBoost ranking; its output is negated for our lower-is-faster pipeline.

**Observation:** percentile-target HGB contributes to accepted blends on three collections.
Neither screened XGBoost alternative enters the final system. NLP random illustrates why model
complementarity matters: the percentile model alone is slightly worse on mean development score,
yet blending it with the incumbent improves the combined ranking.

**Decision:** add percentile-target predictions where the blend improves both development folds.
Do not conclude that all pairwise rankers are inferior from one fixed screening setup.
''')
code('''show("05_target_and_objective")
display(table("target_screening")[["collection","experiment","fold","seed","dev_score"]])
''')
md('''## 5. Selecting complementary models and quantifying improvement

A small predefined blend grid was screened on development predictions and promising candidates
confirmed with three seeds. Accepted final updates required mean development gain of at least
0.002 and no regression in either fold. Repeated seeds are not independent graph replications;
some deterministic tree predictions coincide.

Each member's predictions become normalized within-graph ranks before averaging across seeds/folds
and applying the fixed collection weights below. Development scoring uses only the appropriate
fold; the final test ensemble uses both folds. The final recipes are locked: this reproduction
notebook does not retune them on the reused audit or the leaderboard.

**Percentages:** tables show before/after scores and absolute differences. Relative change is
`100 × (after − before) / before`, suppressed for baseline tau below 0.1 where it is misleading.
These are relative score changes, not accuracy percentage points. Ablation gains are not additive.

The target-enhanced NLP-default blend moves reused audit tau from 0.4113 to 0.4536: +0.0423,
about +10.29% relative. This is a blend improvement, not an isolated causal estimate for the target.
The reported public improvement 0.558 → 0.567 is +0.009 (+1.61% relative), with three collections
changing together. It cannot be attributed to one collection from the aggregate score alone.
''')
code('''display(table("selected_recipe"))
display(table("blend_changes")[["collection","dev_gain","baseline_score","candidate_score",
    "absolute_gain","relative_gain_pct","graphs","bootstrap_low","bootstrap_high"]])
display(table("actual_training_sizes").groupby(["collection","experiment","fold"],as_index=False).agg(
    graphs=("graphs","first"),configurations=("configurations","first")))
''')
md('''## 6. Reproduce: load compatible models or train missing models

The executable workflow below validates raw data against fixed split manifests, verifies model
identity and model-file hashes, and trains missing baseline/selected fits. A model-file name alone
is not sufficient for reuse. An optional historical archive is checked against known hashes before
its model objects are loaded. Current-run evaluations always come from model inference.

Feature extraction runs in isolated workers with bounded scratch storage. Only one preparation
group is retained at a time; it is removed after use. Worker RAM and system reserves are monitored.
A resource failure stops the run rather than silently dropping a required final model. A clean
runtime may need several sessions; completed fits are resumable, while an interrupted fit restarts.

`RUN_ABLATIONS=True` additionally runs the recorded successful comparison recipes for the requested
seeds. Seed 42 reruns screening-style comparisons; all three seeds include the recorded confirmations.
It does not recreate resource-excluded jobs or claim to rerun the early MLP/gated branch experiments.
The optional suite never changes the final blend. Its fit-level and per-graph results are saved for
inspection; only matched settings/seeds should be used for causal comparison.

Checkpoints are written every configured interval, at boundaries and on handled interruption.
A browser download request is not confirmed delivery. Keep a completed backup before disconnecting.
''')
code('''if RUN_MODE == "reproduce":
    import reproduce
    selected_jobs = reproduce.plan(CONTRACT, RUN_ABLATIONS, ABLATION_SEEDS)
    print("Selected + baseline fits:", 90)
    print("Total requested fits:", len(selected_jobs), "(existing verified models will be reused)")
    print("Ablations:", RUN_ABLATIONS, "| seeds:", ABLATION_SEEDS)
    display(pd.DataFrame([{"collection":s["collection"], "role":", ".join(s["roles"]),
        "model":s["experiment"]["name"]} for s in selected_jobs]).groupby(["collection","role"]).size().rename("fits").reset_index())
else:
    print("REVIEW: no training plan is executed.")
''')
# Reuse the proven data-locator/download helper, without requiring previous study outputs.
old=json.loads((REPO/'notebooks/experiments/SC4000_V3_Coverage_and_Data.ipynb').read_text())
locator=next(''.join(c['source']) for c in old['cells'] if c['cell_type']=='code' and ''.join(c['source']).startswith('def locate_data():'))
locator=locator[:locator.index('CONFIG["sample_submission"]')]
code('''if RUN_MODE == "reproduce":
'''+''.join('    '+line+'\n' for line in locator.splitlines())+'''    from study_submission import locate_template
    SAMPLE_SUBMISSION = str(locate_template(NPZ_ROOT, SAMPLE_SUBMISSION))
    if RESTORE_RUN_ZIP:
        from v4_core import restore
        restore(RESTORE_RUN_ZIP, OUTPUT_DIR, 12)
        print("Restored completed work. Clear RESTORE_RUN_ZIP before rerunning this cell in the same runtime.")
    print("Data:", NPZ_ROOT)
    print("Template:", SAMPLE_SUBMISSION)
''')
code('''RUN_STATE = {"status":"review_only"}
def download_checkpoint(path):
    print("Backup:", path, "| bytes:", Path(path).stat().st_size)
    if AUTO_DOWNLOAD_CHECKPOINTS:
        try:
            from google.colab import files
            files.download(str(path))
        except ImportError: print("Outside Colab: copy this backup from its path.")
        except Exception as exc: print("Download request failed:", exc, "— use the Files sidebar.")
    print("Verify the completed ZIP on your computer; local Colab files are temporary.")

if RUN_MODE == "reproduce":
    resources = dict(session_hours=SESSION_HOURS, checkpoint_minutes=CHECKPOINT_MINUTES,
        worker_ram_gib=WORKER_RAM_GIB, system_reserve_gib=SYSTEM_RESERVE_GIB, cpu_threads=CPU_THREADS)
    RUN_STATE = reproduce.run(NPZ_ROOT, OUTPUT_DIR, SCRATCH_DIR, RUNTIME_DIR, CONTRACT,
        resources=resources, model_archive=MODEL_ARCHIVE, run_ablations=RUN_ABLATIONS,
        ablation_seeds=ABLATION_SEEDS, sample_submission=SAMPLE_SUBMISSION,
        on_checkpoint=download_checkpoint)
    print(RUN_STATE)
else:
    print("REVIEW ONLY: historical figures are available; no fresh submission is generated.")
''')
md('''## 7. Results: historical evidence and this execution

### Historical baseline-to-final comparison

The following figure compares the basic HGB pipeline with the final selected system on the
previously inspected audit graphs. It is a whole-system comparison. Layout and Tile use different
metrics; the panels must not be averaged into an estimated Kaggle score. The final model is not
reselected according to which result looks best on this audit.

Our conclusion is conditional: the chosen combination improves the recorded overall baseline
comparison, but not every proposed change helps. Small audit sets, repeated inspection, sampled
candidate evaluation and multiple tested alternatives limit the strength of generalization claims.
''')
code('''show("06_final_results")
display(table("historical_final_audit"))
display(table("reported_public_scores"))
''')
md('''### Current execution

This section only reports newly evaluated predictions from the models in this run. It does not
substitute the historical score when training is incomplete. Different hardware/library builds
can affect retraining; a new submission's Kaggle score remains unknown until evaluated there.
If every historical model is reused under its recorded environment, the output hash can be
compared with the frozen CSV, but the notebook never forces new predictions to match that CSV.
''')
code('''if RUN_MODE == "reproduce":
    if (OUTPUT_DIR / "evaluation/current_audit.csv").exists():
        display(pd.read_csv(OUTPUT_DIR / "evaluation/current_audit.csv"))
        current_plot = story.current_results(OUTPUT_DIR)
        if current_plot: display(Image(filename=str(current_plot)))
    if RUN_STATE.get("status") == "complete":
        display(json.loads((OUTPUT_DIR / "submission_provenance.json").read_text()))
        print("Fresh Kaggle submission:", OUTPUT_DIR / "submission_final.csv")
        print("Fresh baseline submission:", OUTPUT_DIR / "submission_baseline.csv")
    else:
        print("Run incomplete:", RUN_STATE, "— resume before using a submission.")
else:
    print("No current-run metrics: this execution rendered historical evidence only.")
''')
md('''## 8. What we learned, limitations and supporting diagnostics

| Finding | Decision |
|---|---|
| Compact boosted trees provide a strong practical reference | Retain them throughout comparisons and in the final system. |
| GraphSAGE contributes complementary predictions on XLA | Use development-selected blends rather than replacing every collection with a GNN. |
| Mask/context standalone gains are inconclusive | Report uncertainty; do not claim each addition independently improves accuracy. |
| Larger graph/data/training budgets are not uniformly better | Keep selected resource-bounded recipes and disclose untested resource exclusions. |
| Percentile-target predictions improve selected blends | Preserve the accepted target-based components; screened XGBoost alternatives are not selected. |

The curves below are **historical selected-GNN diagnostics**, not new training output. Training
loss and early-development tau have different units. There is no recorded validation-loss series,
and these plots cannot prove absence of overfitting. Current GNN histories, when trained here,
are saved beside their model under `models/.../history.csv`.

For the written report, follow these questions and decisions rather than version numbers.
The six main figures are organized in that order; use only those needed within the page limit.
Underlying tables retain original experiment IDs for traceability. Standard HGB, GraphSAGE and
ranking methods are not claimed as inventions. Research notes in the repository acknowledge
prior work; the gated branch is described as implemented but unverified, not invented evidence.
''')
code('''show("S1_learning_curves")
print("Historical figures/tables/captions:", STORY_DIR)
print("Current models, evaluation, provenance and submissions:", OUTPUT_DIR)
print("Keep the completed final-run backup for resuming; the small evidence pack has no model weights.")
''')
notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
    language_info=dict(name='python',version='3.11'),colab=dict(name='SC4000_Final.ipynb'),
    reproducibility=dict(default='raw-data load-or-train',historical_score=.567,model_archive_required=False)),
    nbformat=4,nbformat_minor=5)
target=REPO/'notebooks/SC4000_Final.ipynb';target.write_text(json.dumps(notebook,indent=1)+'\n')
print(target,len(cells),'cells',target.stat().st_size,'bytes')
