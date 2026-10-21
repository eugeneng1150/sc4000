"""Build a self-contained Colab notebook from the reviewed study source modules."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESTINATION = HERE.parents[1]/'notebooks/experiments'/'SC4000_Comprehensive_Ablations.ipynb'


def cell(kind, content):
    value = dict(cell_type=kind, metadata={}, source=content.strip().splitlines(keepends=True),
                 id=hashlib.sha256(content.encode()).hexdigest()[:12])
    if kind == 'code':
        value.update(execution_count=None, outputs=[])
    return value


cells = []


def md(content):
    cells.append(cell('markdown', content))


def code(content):
    cells.append(cell('code', content))


md('''
# SC4000: CPU GNN, baseline and ablation study

**For standard Colab CPU RAM (about 12–13 GB), using local storage only.**

Run all cells in order. The notebook trains the GNN reference and eight controlled variants
on **both XLA layout collections**, plus tabular baselines/ablations on all five collections.
It repeats required comparisons with three seeds, locks model/blend choices on development
data, audits reserved graphs, and exports full baseline and selected-method submission CSVs. Download the final results ZIP
and share that one file for analysis. No repository checkout is required.

**Read before starting:**

1. Use a fresh CPU runtime. Avoid keeping a previous notebook's models/data in memory.
2. If the data already exists, set `DATA_ROOT` below. It must contain `layout/` and `tile/`,
   or a parent containing `npz_all/npz/`. Otherwise enable the download option.
3. The default sweep can take hours. The exact duration depends on graph sizes and Colab.
   The run stops after a configurable nine-hour session budget and exports partial results.
4. Local results can resume while the same files survive. A deleted/reset Colab VM loses
   local files. Download checkpoints to preserve them; upload a checkpoint ZIP to resume
   in a fresh VM with the same code, packages, settings and source data.
5. This is an end-to-end **bounded CPU GNN study**, including full test prediction.
   It does not upload to Kaggle or claim a public-leaderboard score. All required comparisons
   and submissions take priority over optional tabular exploration. If the session budget
   is reached, the ZIP explicitly marks unfinished stages; rerunning resumes automatically.

There are two fixed references: **B0**, the HGB system baseline, and **G0**, a two-layer
width-32 GraphSAGE model based on your existing GNN architecture. G0 is not deliberately
weakened to manufacture gains. Strict sampling and shared splits make this a controlled
reimplementation, rather than an exact rerun of an older model. Improvements are measured,
not guaranteed; a baseline can remain the selected method if additions do not help.
''')
md('''
## 1. Settings

Defaults target 5 GiB for each isolated worker, leave 2 GiB available system RAM, and use
two CPU threads. The monitor terminates a worker that exceeds its RAM/time/disk limit,
records the reason, and proceeds. Bounds reduce OOM risk but cannot prevent every sudden
allocation spike or Colab runtime termination.

Only one graph and one configuration row are decoded at a time. There is **no expanded
raw-array cache**. Compact feature tables are reused for the current collection, then
deleted; model/results files and source NPZ data are retained.
''')
code('''
from pathlib import Path
import os
import sys

DATA_ROOT = ""  # e.g. "/content/predict-ai-model-runtime/npz_all/npz"
DOWNLOAD_IF_MISSING = True
RUN_NAME = "sc4000_gnn_ablation_v2"
RESTORE_ZIP = ""  # optional: "/content/sc4000_gnn_ablation_v2_results.zip"
AUTO_DOWNLOAD_FINAL_ZIP = True
SAMPLE_SUBMISSION = ""  # autodetected; otherwise path to the competition template CSV

# Scientific settings. Changing them requires a new RUN_NAME.
SEEDS = [42, 137, 2026]
COLLECTION_ORDER = ["layout:xla:default", "layout:xla:random",
                    "layout:nlp:default", "layout:nlp:random", "tile:xla"]
LAYOUT_TRAIN_GRAPHS = 40
TILE_TRAIN_GRAPHS = 160
DEVELOPMENT_GRAPHS = 12
HOLDOUT_GRAPHS = 24
TRAIN_POOL_CONFIGS = 384
FIT_CONFIGS = 256
EVALUATION_CONFIGS = 1024
GNN_TRAIN_GRAPHS = 24
GNN_TRAIN_CONFIGS = 64
GNN_MAX_NODES = 512
GNN_MAX_CONFIGURABLE_NODES = 384
GNN_MAX_EDGES = 4096
GNN_EPOCHS = 20
GNN_PATIENCE = 5

# Resource settings. These may change when resuming the same run.
SESSION_HOURS = 9.0
WORKER_RAM_GIB = 5.0
SYSTEM_RESERVE_GIB = 2.0
JOB_MINUTES = 25
EXTRACTION_MINUTES = 150
GNN_JOB_MINUTES = 120
INFERENCE_MINUTES = 90

BASE_DIR = Path("/content") if Path("/content").exists() else Path.cwd()
OUTPUT_DIR = BASE_DIR / RUN_NAME
SCRATCH_DIR = BASE_DIR / ("." + RUN_NAME + "_scratch")
RUNTIME_DIR = BASE_DIR / ("." + RUN_NAME + "_runtime")
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "2"
os.environ["MPLCONFIGDIR"] = str(RUNTIME_DIR / "matplotlib")
''')
md('''
## 2. Install missing dependencies and write the embedded study code

No competitor implementation, model weights or predictions are used. Package versions,
code hashes, data fingerprints and all settings are included in the results bundle.
''')
code('''
import importlib.util
import subprocess

packages = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "sklearn": "scikit-learn",
            "joblib": "joblib", "psutil": "psutil", "threadpoolctl": "threadpoolctl",
            "matplotlib": "matplotlib", "torch": "torch>=2.6"}
missing = [package for module, package in packages.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir", *missing])
''')
for file, module in [(name+'.py', 'study_'+name+'.py') for name in ['engine','gnn','submission','controller']]:
    source = (HERE / file).read_text()
    assert "'''" not in source, 'Embedded raw string delimiter occurs in module'
    code(f'# Embedded independent implementation: {file}\n'
         f"SOURCE = r'''{source}'''\n"
         f'(RUNTIME_DIR / {module!r}).write_text(SOURCE)\ndel SOURCE\n')
code('''
import importlib
import json
import shutil
import zipfile
import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, str(RUNTIME_DIR))
import study_engine
import study_gnn
import study_submission
import study_controller
for module in [study_engine, study_gnn, study_submission, study_controller]:
    importlib.reload(module)
from study_controller import default_config, run_study

CONFIG = default_config()
CONFIG.update(seeds=SEEDS, collections=COLLECTION_ORDER,
    layout_train_graphs=LAYOUT_TRAIN_GRAPHS, tile_train_graphs=TILE_TRAIN_GRAPHS,
    dev_graphs=DEVELOPMENT_GRAPHS, holdout_graphs=HOLDOUT_GRAPHS,
    train_pool_configs=TRAIN_POOL_CONFIGS, fit_configs=FIT_CONFIGS,
    eval_configs=EVALUATION_CONFIGS, session_hours=SESSION_HOURS,
    worker_ram_gib=WORKER_RAM_GIB, system_reserve_gib=SYSTEM_RESERVE_GIB,
    job_minutes=JOB_MINUTES, extraction_minutes=EXTRACTION_MINUTES,
    gnn_train_graphs=GNN_TRAIN_GRAPHS, gnn_configs=GNN_TRAIN_CONFIGS,
    gnn_nodes=GNN_MAX_NODES, gnn_config_nodes=GNN_MAX_CONFIGURABLE_NODES,
    gnn_edges=GNN_MAX_EDGES, gnn_epochs=GNN_EPOCHS, gnn_patience=GNN_PATIENCE,
    gnn_job_minutes=GNN_JOB_MINUTES, inference_minutes=INFERENCE_MINUTES,
    sample_submission=SAMPLE_SUBMISSION)
study_controller.validate_config(CONFIG)
print(f"RAM total: {psutil.virtual_memory().total / 1024**3:.1f} GiB")
print(f"RAM available: {psutil.virtual_memory().available / 1024**3:.1f} GiB")
print(f"Disk available: {shutil.disk_usage(BASE_DIR).free / 1024**3:.1f} GiB")
print("Output:", OUTPUT_DIR)
''')
md('''
## 3. Locate data or download it once

If downloading, first accept the competition rules on Kaggle. Store `KAGGLE_USERNAME`
and `KAGGLE_KEY` in Colab Secrets, or use an existing authenticated Kaggle session.
The notebook does not print or include credentials in the results.

Source competition files are required disk space, not disposable model cache. They are
not deleted. For a local CPU study, do not load the complete dataset into a Python object.
''')
code('''
def locate_npz_root():
    bases = ([Path(DATA_ROOT)] if DATA_ROOT else []) + [
        BASE_DIR / "predict-ai-model-runtime", BASE_DIR / "data", BASE_DIR,
    ]
    for base in bases:
        for path in [base, base / "npz_all/npz", base / "npz"]:
            if (path / "layout").is_dir() and (path / "tile").is_dir():
                return path.resolve()
    return None

NPZ_ROOT = locate_npz_root()
if NPZ_ROOT is None and DOWNLOAD_IF_MISSING:
    from importlib.metadata import version, PackageNotFoundError
    try:
        installed = tuple(int(x) for x in version("kagglehub").split(".")[:3])
    except (PackageNotFoundError, ValueError):
        installed = (0, 0, 0)
    if installed < (0, 4, 3):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir", "kagglehub>=0.4.3,<2"])
    try:
        from google.colab import userdata
        for key in ["KAGGLE_USERNAME", "KAGGLE_KEY"]:
            if not os.environ.get(key):
                try:
                    value = userdata.get(key)
                    if value:
                        os.environ[key] = value
                except Exception:
                    pass
    except ImportError:
        pass
    import kagglehub
    if shutil.disk_usage(BASE_DIR).free < 15 * 1024**3:
        raise RuntimeError("Less than 15 GiB free before download. Free space or provide existing data first.")
    print("Downloading competition source files; this can take time and substantial disk space.")
    kagglehub.competition_download("predict-ai-model-runtime", output_dir=str(BASE_DIR / "predict-ai-model-runtime"))
    NPZ_ROOT = locate_npz_root()
if NPZ_ROOT is None:
    raise FileNotFoundError("Set DATA_ROOT to the extracted folder containing layout/ and tile/.")
print("NPZ source:", NPZ_ROOT)
TEMPLATE_PATH = study_submission.locate_template(NPZ_ROOT, SAMPLE_SUBMISSION)
CONFIG["sample_submission"] = str(TEMPLATE_PATH)
print("Submission template:", TEMPLATE_PATH)
print("Test/template preflight:", study_submission.preflight(NPZ_ROOT, CONFIG))
''')
md('''
## 4. Restore an optional checkpoint and inspect the experiment plan

For a new run leave `RESTORE_ZIP` blank. To resume after losing a VM, upload a previously
downloaded results ZIP through the Colab Files panel, set its path in section 1, and run
again with the original settings. Models in the ZIP are Python joblib files: restore only
your own trusted notebook output. The original NPZ data must also be available again.

**Validation protocol:** development graphs come from the official training split;
reserved audit graphs come from the official validation split. Candidate configuration
IDs are sampled without labels and fixed in a manifest. No graph occurs in both splits
within a collection, although related architecture families can. Existing work may already
have used some official validation graphs; this is an untouched holdout for this run only,
not a claim that the project has never inspected those graphs.
''')
code('''
if RESTORE_ZIP:
    archive_path = Path(RESTORE_ZIP).resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if any(OUTPUT_DIR.iterdir()):
        raise ValueError("Restore into an empty output directory to avoid overwriting results.")
    with zipfile.ZipFile(archive_path) as archive:
        if sum(item.file_size for item in archive.infolist()) > CONFIG["output_limit_gib"] * 1024**3:
            raise ValueError("Checkpoint exceeds configured output size limit.")
        for item in archive.infolist():
            destination = (OUTPUT_DIR / item.filename).resolve()
            if OUTPUT_DIR.resolve() not in destination.parents:
                raise ValueError("Unsafe checkpoint path.")
        archive.extractall(OUTPUT_DIR)
    print("Restored:", archive_path)

plan = []
for collection in CONFIG["collections"]:
    trials = study_controller.core_experiments(collection, CONFIG)
    for trial in trials:
        plan.append(dict(collection=collection, **trial))
plan_df = pd.DataFrame(plan)
display(plan_df[["collection", "name", "model", "blocks"]])
print("Required fits:", len(plan_df) * len(SEEDS))
print("Every listed method runs all three seeds; optional tabular screening follows submissions.")
print("GNN additions share architecture capacity, training IDs, graph samples and budgets.")
''')
md('''
## What the experiments test

| Group | Experiments | Interpretation |
|---|---|---|
| System reference | `B0_basic_hgb` | Basic summaries + fixed histogram gradient boosting |
| Matched training-data control | `B_matched_hgb` | HGB trained on exactly the GNN graph/configuration IDs |
| GNN reference | `G0_reference` | Two-layer width-32 GraphSAGE, opcode embedding, numeric node features, pairwise loss |
| GNN additions | `G_add_masks`, `G_add_context`, `G_full` | Explicit padding masks; global categorical position/opcode/depth/edge summaries; both |
| GNN removals | `G_no_messages`, `G_no_opcode_embedding`, `G_no_numeric` | Remove learned message aggregation, opcode embedding, or numeric node inputs from G_full |
| GNN training/architecture controls | `G_regression`, `G_one_layer` | Replace pairwise objective with regression; reduce to one message layer |
| Padding / values | `B1_masked_hgb` | Add masks and valid-only layout summaries; tile uses signed-log values |
| Position | `P0_position` | Positional layout histograms; tile uses bounded config interactions |
| Operation and graph context | `P_opcode`, `P_depth`, `P_opcode_depth`, `P_edge`, `P_full` | Preserve which operations/graph positions receive which layouts |
| Structural references | `S_topology`, `S_wl`, `P_full_static` | Configuration-invariant graph statistics and bounded neighborhood fingerprints |
| Remove one feature block | `A_without_*` | Test whether each block helps the full feature set |
| Training controls | `C_*` | Weights, target transforms, sampled duplicate aggregation, half data |
| Model comparisons | `M_*` | Extra Trees, small MLP, linear regression and pairwise linear ranking |

The positional/category representation is independently designed. Layout edge features
are categorical co-occurrences, not a physical memory-compatibility calculation. The `wl`
block is a bounded hashed neighborhood signature, not an exact graph-isomorphism test.
`config_id` is retained only for joins and never used as a predictive feature.

Duplicate handling groups **only sampled training rows** by raw configuration equality and
uses their median runtime; evaluation IDs remain untouched. Primary experiments train
on the same candidate IDs for a given seed. Other seeds vary the training subsample and
model randomness; they are not three independent graph splits. Half-data controls are
reported separately. Model-family comparisons are not pure feature ablations.

No new ranking novelty is claimed: the project's existing GNN already uses pairwise loss.
The optional linear pairwise method is a separate comparator.

**GNN interpretation:** G0 and the additions have identical parameter counts. Disabled
mask/context inputs are zero-filled, so gains are not just from a larger prediction head.
Each graph uses deterministic, label-independent sampling with hard node, configurable-node
and edge caps. The context addition summarizes the full configurable-node set, so it may
recover information omitted by graph sampling. Coverage is recorded per graph.
`G_no_opcode_embedding` retains opcode-aware global summaries; `G_no_messages` retains
edge-aware summaries. These test the learned components, not removal of all such information.
The one-layer and regression controls deliberately change architecture/objective.

GNN training uses up to 24 graphs × 64 configurations; B0 uses up to 40 × 256. The matched
HGB control separates training-data volume from model differences. All methods use the
same fixed development and holdout candidates. Early stopping uses a fixed subset of
128 development candidates per graph; selection uses the complete sampled development set.
Checkpoints retain model, optimizer and RNG state after each epoch. No full graph/config cache
or GPU is required. The scope is both XLA layout collections; NLP and tile use tabular models.
''')
md('''
## 5. Run the full study

This cell performs repeated core experiments, development-only selection, holdout evaluation,
full test inference, paired comparisons and checkpoint export without manual model choices.
Optional tabular screening runs last, on development data only, and cannot alter the frozen
selection after the audit. Final CSVs use the same audited fitted models, without holdout retraining.
It displays progress and worker RAM. If you interrupt it, completed jobs remain usable.
The same cell resumes successful jobs rather than retraining them. Temporary features
may need to be rebuilt after cleanup. Failures remain recorded and are not silently
replaced by a different experiment.

After each completed collection a fresh results ZIP is available beside the output
folder. You can download it from the Files sidebar while the next collection runs.
An interrupted run is labelled incomplete; partial results are never presented as a full sweep.
''')
code('''
# Save the exact implementation with the evidence, so results can be audited later.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for name in ["study_engine.py", "study_gnn.py", "study_submission.py", "study_controller.py"]:
    existing = OUTPUT_DIR / name
    if existing.exists() and existing.read_bytes() != (RUNTIME_DIR / name).read_bytes():
        raise ValueError("Saved implementation differs. Use a new RUN_NAME; do not overwrite old evidence.")
    if not existing.exists():
        shutil.copy2(RUNTIME_DIR / name, existing)

STUDY_RETURNED = False
try:
    RESULTS_ZIP = run_study(NPZ_ROOT, OUTPUT_DIR, SCRATCH_DIR,
                            RUNTIME_DIR / "study_engine.py", CONFIG)
    STUDY_RETURNED = True
finally:
    # Include partial evidence even when an exception stops a collection.
    if OUTPUT_DIR.exists():
        RESULTS_ZIP = study_controller.make_bundle(OUTPUT_DIR)
    if AUTO_DOWNLOAD_FINAL_ZIP and not STUDY_RETURNED and "RESULTS_ZIP" in globals():
        try:
            from google.colab import files
            files.download(str(RESULTS_ZIP))
        except (ImportError, OSError):
            print("Download manually from the Files sidebar:", RESULTS_ZIP)
''')
md('''
## 6. Read the results and export figures

Layout uses exact Kendall tau-b on the sampled candidates; tile uses normalized runtimes
and top-five regret relative to the sampled candidates. These are local diagnostics,
not the full official competition score. Constant predictions receive a diagnostic tau
of zero rather than disappearing from the average.

The holdout is evaluated only after model and blend choices have been saved in
`selection_lock.json`. The bootstrap resamples whole graphs after averaging paired
seed differences within each graph. With many experiments and few graphs, these are
exploratory intervals, not proof of statistical significance.

Use `holdout_paired_vs_G0_reference.csv` for adding masks/context to the GNN reference,
`holdout_paired_vs_G_full.csv` for removal controls, and `holdout_paired_gains.csv` for
comparisons against HGB. These include absolute changes, score points (100 × change),
relative percentage changes only when the reference is positive and above 0.01, and
paired graph-bootstrap intervals. A tau increase from 0.40 to 0.44 is **+0.04 tau**
(or +10% relative tau), not “4% more accurate.” Negative and uncertain gains remain visible.
Feature interactions mean the gain of the full model need not equal individual gains.
''')
code('''
import matplotlib.pyplot as plt

state = json.loads((OUTPUT_DIR / "run_state.json").read_text())
display(pd.DataFrame(list(state["collections"].items()), columns=["collection", "status"]))
for collection in CONFIG["collections"]:
    directory = OUTPUT_DIR / collection.replace(":", "_")
    summary = directory / "holdout_paired_gains.csv"
    print("\\n", collection)
    if summary.exists() and summary.stat().st_size > 2:
        table = pd.read_csv(summary).sort_values("mean_paired_gain")
        display(table)
        fig, ax = plt.subplots(figsize=(9, max(3, len(table) * .35)))
        ax.errorbar(table.mean_paired_gain, np.arange(len(table)),
                    xerr=np.maximum(0, np.vstack([table.mean_paired_gain-table.bootstrap_low,
                                    table.bootstrap_high-table.mean_paired_gain])), fmt="o", capsize=3)
        ax.set_yticks(np.arange(len(table)), table.experiment)
        ax.axvline(0, color="gray", linestyle="--")
        ax.set_xlabel("Paired holdout score change versus B0 (higher is better)")
        ax.set_title(collection + " — exploratory graph-bootstrap intervals")
        fig.tight_layout()
        fig.savefig(directory / "holdout_paired_gains.png", dpi=150)
        plt.show()
        plt.close(fig)
    else:
        print("No completed holdout comparisons yet. Inspect all_trials.csv and logs.")
for collection in CONFIG["gnn_collections"]:
    directory = OUTPUT_DIR / collection.replace(":", "_")
    for reference in ["G0_reference", "G_full", "B_matched_hgb"]:
        path = directory / f"holdout_paired_vs_{reference}.csv"
        if path.exists() and path.stat().st_size > 2:
            print(collection, "compared with", reference)
            display(pd.read_csv(path))
status_path = OUTPUT_DIR / "submission_status.json"
if status_path.exists():
    print("Submission export:", json.loads(status_path.read_text()))
for path in OUTPUT_DIR.glob("submission_*.csv"):
    print("CSV:", path)
RESULTS_ZIP = study_controller.make_bundle(OUTPUT_DIR)
print("Final ZIP including charts:", RESULTS_ZIP)
''')
md('''
## 7. Download this one file

Share **the results ZIP**, not just screenshots or the winning score. It contains per-graph
scores, all sampled predictions, repeat variability, feature/model definitions, timings,
observed worker RAM, graph manifests, failures, selected models and the exact source.
It also contains `submission_baseline.csv` and `submission_selected.csv` when full test
inference finishes. In the rare tile graphs with fewer candidates than the template's five
slots, all real candidates come first and the original template suffix is preserved and
logged. All candidates are scored, then each row is truncated to its template length.
For this dataset, two layout files have 1,001 candidates but the template requests 1,000.
CSV validation checks IDs, order, uniqueness, index bounds and full permutations whenever
the template requests every candidate.
It excludes raw competition data and temporary feature tables.

The ZIP produced in section 5 already contains all numerical evidence. This final copy
also includes the charts. If the browser blocks downloads, use the Files sidebar.
''')
code('''
print(RESULTS_ZIP)
print(f"Bundle size: {RESULTS_ZIP.stat().st_size / 1024**2:.1f} MiB")
if AUTO_DOWNLOAD_FINAL_ZIP:
    try:
        from google.colab import files
        files.download(str(RESULTS_ZIP))
    except ImportError:
        print("Outside Colab: copy the ZIP from the path above.")
''')
md('''
## References and attribution

These sources motivated questions about representation and ranking. No competitor code,
weights or predictions are embedded here. Independent implementation does not establish
global novelty; cite these inspirations in the course report.

- [1st-place write-up](https://www.kaggle.com/competitions/predict-ai-model-runtime/writeups/viet-br-connection-1st-place-solution-for-the-goog): preprocessing and configuration comparison.
- [2nd-place write-up](https://www.kaggle.com/competitions/predict-ai-model-runtime/writeups/latenciaga-2nd-place-solution-for-the-google-fast-): ranking objectives and validation variability.
- [3rd-place author repository](https://github.com/jafluri/kaggle_tpu_graph): meaningful features and repeated training.
- [5th-place write-up](https://www.kaggle.com/competitions/predict-ai-model-runtime/discussion/456093): dimension correspondence.
- [Official data description](https://www.kaggle.com/competitions/predict-ai-model-runtime/data): array meanings, edge direction and tile runtime normalization.
- [PyTorch checkpoints](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html): model, optimizer and RNG state are retained for epoch-level recovery.
- [SciPy Kendall tau-b](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html): tie-aware sampled-set diagnostic.
- [HGB documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html): sample weights and early-stopping settings.
- [Colab resource limits](https://research.google.com/colaboratory/faq.html): resources and session duration are not guaranteed.
- [KaggleHub download documentation](https://github.com/Kaggle/kagglehub): direct download to a chosen directory.
''')

notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                 language_info=dict(name='python', version='3.11'),
                 colab=dict(name=DESTINATION.name, provenance=[])), nbformat=4, nbformat_minor=5)
DESTINATION.write_text(json.dumps(notebook, indent=1) + '\n')
print(DESTINATION)
