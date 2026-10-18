"""Generate the standalone, locally backed-up v3 Colab notebook."""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
TARGET=HERE.parent/'SC4000_V3_Coverage_and_Data.ipynb'
cells=[]
def cell(kind,source):
    item=dict(cell_type=kind,metadata={},source=source.strip().splitlines(keepends=True),id=hashlib.sha256(source.encode()).hexdigest()[:12])
    if kind=='code':item.update(outputs=[],execution_count=None)
    cells.append(item)
def md(text):cell('markdown',text)
def code(text):cell('code',text)

md('''
# SC4000 v3 — graph coverage and training data in one run

This notebook contains **both experiments**:

1. Show the same GNN more of each graph, while keeping its training examples fixed.
2. At the development-selected graph budget, give it more training graphs and configurations.

It also tests masks/context, message passing, connected-neighbourhood sampling, matched-data
HGB controls, larger tabular training sets, a deeper HGB and Extra Trees. It chooses models
and blends automatically and exports full Kaggle candidate CSVs. You do not choose experiments
between stages. Everything needed is embedded here; no repository checkout is required.
The GNN coverage/data study runs on XLA layout default and random. NLP layout default/random
and tile receive the expanded tabular study. Submissions cover all five collections.

**Target:** standard Colab CPU RAM (approximately 12–13 GB), local data and temporary files.
No Google Drive mount is used. Checkpoint ZIPs are requested as downloads to your computer
every 30 minutes and after each collection, including while a long worker runs. The browser
must allow multiple downloads. **A download request is not proof that the ZIP reached your
computer. A disconnected browser cannot be relied on to receive new backups.** Check Downloads.

A nine-hour session budget does not mean the study finishes in nine hours. This is an extensive
staged study and can require more than one session. Resume the latest downloaded **v3** ZIP
with this same notebook/settings. Completed jobs are reused; GNNs resume at epoch checkpoints.
V2 results remain your historical benchmark and are not compatible v3 recovery archives.

The reported v2 Kaggle score was 0.45936, with a user-reported top-10% target around 0.59.
No improvement, completion time or ranking is guaranteed. The v3 baseline is retrained under
its own folds/sampling; it does not reproduce the exact v2 submission automatically.
''')
md('''
## 1. Set paths and resource limits

For a fresh run, leave `RESTORE_ZIP` empty. It is an **input backup to restore**, not the output
path. Output is created automatically at `/content/sc4000_gnn_v3_results.zip` in Colab.
After a runtime reset, upload your most recent downloaded v3 ZIP and enter its actual path.
Use the same `RUN_NAME`. Local files from the deleted VM cannot be recovered without a copy.

Training batches stay at 16 configurations across graph budgets. We do not silently reduce
batch size, because that would change which pairs the ranking loss sees. A label-free pilot
rejects a graph budget if it exceeds the pilot resource limit; rejected budgets are logged.
''')
code('''
from pathlib import Path
import os, sys
DATA_ROOT = ""  # folder containing layout/ and tile/, or parent containing npz_all/npz
SAMPLE_SUBMISSION = ""  # autodetect beside competition data
DOWNLOAD_IF_MISSING = True
RUN_NAME = "sc4000_gnn_v3"
RESTORE_ZIP = ""  # e.g. "/content/sc4000_gnn_v3_results.zip"
AUTO_DOWNLOAD_CHECKPOINTS = True
CHECKPOINT_MINUTES = 30
SESSION_HOURS = 9.0
WORKER_RAM_GIB = 5.0
SYSTEM_RESERVE_GIB = 2.0
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
## 2. Dependencies and embedded implementation

Only missing dependencies are installed, without a pip download cache. A restored run requires
matching scientific-library versions; the saved identity prevents mixing incompatible results.
Only restore your own trusted archives: model artifacts include Python joblib files.
''')
code('''
import importlib.util, subprocess
packages = {"numpy":"numpy", "pandas":"pandas", "scipy":"scipy", "sklearn":"scikit-learn",
            "torch":"torch>=2.6", "joblib":"joblib", "psutil":"psutil",
            "threadpoolctl":"threadpoolctl", "matplotlib":"matplotlib"}
missing = [package for module,package in packages.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable,"-m","pip","install","--quiet","--no-cache-dir",*missing])
''')
for name in ['engine','gnn','common','plan','submission','controller']:
    source=(HERE/(name+'.py')).read_text()
    assert "'''" not in source
    code(f'# Embedded implementation: {name}.py\nSOURCE = r\'\'\'{source}\'\'\'\n'
         f'(RUNTIME_DIR / "study_{name}.py").write_text(SOURCE)\ndel SOURCE')
code('''
import importlib, json, shutil, zipfile
import numpy as np
import pandas as pd
import psutil
sys.path.insert(0,str(RUNTIME_DIR))
import study_engine, study_gnn, study_common, study_plan, study_submission, study_controller
for module in [study_engine,study_gnn,study_common,study_plan,study_submission,study_controller]:
    importlib.reload(module)
CONFIG = study_plan.default_config()
CONFIG.update(session_hours=SESSION_HOURS,checkpoint_minutes=CHECKPOINT_MINUTES,
              worker_ram_gib=WORKER_RAM_GIB,system_reserve_gib=SYSTEM_RESERVE_GIB)
print(f"Available RAM: {psutil.virtual_memory().available/1024**3:.1f} GiB")
print(f"Available disk: {shutil.disk_usage(BASE_DIR).free/1024**3:.1f} GiB")
print("Local output:", OUTPUT_DIR)
print("Automatic ZIP downloads:", AUTO_DOWNLOAD_CHECKPOINTS)
if not AUTO_DOWNLOAD_CHECKPOINTS:
    print("WARNING: local-only progress will be lost if the VM is deleted; download ZIPs manually.")
''')
md('''
## 3. Find the competition data and check the test template

Download requires accepting the competition rules on Kaggle. Credentials can be stored as
`KAGGLE_USERNAME` and `KAGGLE_KEY` in Colab Secrets; they are not printed or archived.
Original competition files remain on disk. The notebook creates no expanded raw configuration
cache. Prefer at least 20 GiB free after extracting the source data, with more space for a
large run and simultaneous old/new ZIPs. Disk guards do not guarantee against quota changes.
''')
code('''
def locate_data():
    bases=([Path(DATA_ROOT)] if DATA_ROOT else [])+[BASE_DIR/"predict-ai-model-runtime",BASE_DIR/"data",BASE_DIR]
    for base in bases:
        for path in [base,base/"npz_all/npz",base/"npz"]:
            if (path/"layout").is_dir() and (path/"tile").is_dir():return path.resolve()
    return None
NPZ_ROOT=locate_data()
if NPZ_ROOT is None and DOWNLOAD_IF_MISSING:
    from importlib.metadata import version,PackageNotFoundError
    try:installed=tuple(int(x) for x in version("kagglehub").split(".")[:3])
    except (PackageNotFoundError,ValueError):installed=(0,0,0)
    if installed<(0,4,3):
        subprocess.check_call([sys.executable,"-m","pip","install","--quiet","--no-cache-dir","kagglehub>=0.4.3,<2"])
    try:
        from google.colab import userdata
        for name in ["KAGGLE_USERNAME","KAGGLE_KEY"]:
            if not os.environ.get(name):
                try:
                    value=userdata.get(name)
                    if value:os.environ[name]=value
                except Exception:pass
    except ImportError:pass
    if shutil.disk_usage(BASE_DIR).free<25*1024**3:
        raise OSError("Need more free disk before downloading data and saving experiments.")
    import kagglehub
    kagglehub.competition_download("predict-ai-model-runtime",output_dir=str(BASE_DIR/"predict-ai-model-runtime"))
    NPZ_ROOT=locate_data()
if NPZ_ROOT is None:raise FileNotFoundError("Set DATA_ROOT to the extracted competition data.")
CONFIG["sample_submission"]=str(study_submission.locate_template(NPZ_ROOT,SAMPLE_SUBMISSION))
print("Data:", NPZ_ROOT)
print("Test/template preflight:", study_submission.preflight(NPZ_ROOT,CONFIG))
''')
md('''
## 4. Restore a checkpoint and inspect the automatic plan

Leave the restore path empty on the first run. After a VM reset, upload a **v3** ZIP and run all
cells with its path. Restoring requires an empty output directory. If the current runtime still
has your results, leave `RESTORE_ZIP` empty and rerun section 6 to continue directly.

The two development folds use different held-out training graphs. A model only scores its
own development fold; we never validate it on another fold's graphs that it trained on.
Official valid has already been inspected during v2, so v3 calls it an **audit**, not an
untouched final test. Architecture families can still cross graph splits.
''')
code('''
if RESTORE_ZIP:
    with zipfile.ZipFile(RESTORE_ZIP) as archive:
        identity=json.loads(archive.read("run_identity.json"))
        if identity.get("version")!="3.0.0":raise ValueError("Use a v3 checkpoint; v2 is a separate study.")
    study_controller.restore_archive(RESTORE_ZIP,OUTPUT_DIR,CONFIG["output_limit_gib"])
    print("Restored:",RESTORE_ZIP)
print("Folds:",CONFIG["folds"],"| confirmation seeds:",CONFIG["seeds"])
print("Graph node budgets:",CONFIG["graph_budgets"])
print("GNN training graph counts:",CONFIG["data_graph_counts"],"(0 means all available training graphs)")
print("Configurations per training graph:",CONFIG["data_config_counts"])
print("Checkpoint download interval:",CONFIG["checkpoint_minutes"],"minutes")
plan=study_plan.coverage_plan(CONFIG)+study_plan.followup_plan(CONFIG,study_plan.profile(2048))
display(pd.DataFrame(plan)[["stage","name","model","train_graphs","configs","profile"]])
print("The table illustrates follow-up experiments at 2,048 nodes; the actual budget is selected automatically.")
''')
md('''
## 5. What runs, and why

| Stage | Experiment | What stays fixed |
|---|---|---|
| Capacity pilots | Real large graphs, full training batch, forward/backward; no runtime labels | Resource thresholds; batch size 16 |
| Graph coverage | G0 and G_full at 512/1,024/2,048/4,096 nodes | 24 training graphs × 64 configurations, architecture, loss and evaluation candidates |
| Component controls | Masks, context and no message passing at the selected budget | Graph budget and training data |
| Sampler control | Connected configurable seeds versus uniform configurable seeds | Caps, model recipe and training-data IDs |
| Training-data grid | 24 / 40 / all training graphs × 64 / 128 / 256 configurations | Selected graph budget and full GNN recipe |
| Matched HGB | Same graph/configuration IDs for each training-data recipe | Full tabular representation and HGB recipe |
| Tabular development | Basic/full features, more graphs, more configurations, both, deeper HGB, Extra Trees | Explicit per-experiment settings |
| Confirmation | Key controls, the data-grid corners and development leaders repeated across 3 seeds | Same 2 graph folds |
| Selection and audit | Freeze a development-selected model/blend, then audit confirmed methods | No audit-based model selection |
| Test export | Baseline, tabular-only and selected system CSVs | Frozen fitted models; no audit retraining |

All candidates are screened with seed 42 on **both folds**. Key comparisons and selected leaders
then receive seeds 137 and 2026 too. Screening and confirmed results are saved separately.
The 2×2 training-data corners isolate more graphs, more configurations and their combination;
intermediate grid points remain available for diagnosis and can qualify as finalists.

A graph budget must pass a pilot before training. Exclusions remain explicit. The same physical
configuration batch is retained, so the pairwise loss is not silently changed by microbatching.
Sampling uses a fixed configurable-node permutation to make its seed prefixes nested. The full
retained graphs need not be strict subsets. The 512-node control is a v3 reference under this
sampling implementation, not a reconstruction of v2's exact sampled nodes.

More data also means more optimizer updates under a fixed epoch cap. This is a practical
training-data study, not a compute-matched scaling-law experiment. Early stopping uses a
fixed development subset; candidate selection uses the complete sampled development set.

Selected blends must improve the mean development score and avoid a drop larger than 0.02
on either fold relative to B0. Baseline-only is always a candidate. This is a development rule,
not a guarantee that every audit or leaderboard slice improves.
''')
md('''
## 6. Run everything and download checkpoints automatically

Allow this browser page to download multiple files. Check your computer's Downloads folder
for successive copies of `sc4000_gnn_v3_results.zip`. Your browser may append `(1)`, `(2)`, etc.
The most recent successfully downloaded copy is the recovery point.

A ZIP is created and its download requested every 30 minutes, after each collection and at
normal stop/interruption. It includes completed trials and self-contained active GNN epoch
checkpoints. It excludes competition data and temporary feature matrices. On a runtime reset,
work after the last downloaded checkpoint can be lost. No keep-alive or runtime-limit bypass
is used. The notebook cannot confirm that your browser finished saving a file.
''')
code('''
def download_checkpoint(path):
    import datetime
    print("ZIP ready at",datetime.datetime.now(datetime.timezone.utc).isoformat(),":",path)
    print("Check that the download reaches your computer. Colab local storage is temporary.")
    if AUTO_DOWNLOAD_CHECKPOINTS:
        try:
            from google.colab import files
            files.download(str(path))
        except ImportError:
            print("Outside Colab: copy this ZIP from the displayed path.")
        except Exception as exc:
            print("Browser download request failed:",type(exc).__name__,str(exc))
            print("Download the ZIP manually from the Colab Files sidebar.")

RESULTS_ZIP=study_controller.run_study(NPZ_ROOT,OUTPUT_DIR,SCRATCH_DIR,
                                     RUNTIME_DIR/"study_engine.py",CONFIG,
                                     on_checkpoint=download_checkpoint)
''')
md('''
## 7. Review completeness and results

Graph coverage and training-data comparisons are separate in the saved plan. Use confirmed
comparisons for report claims and retain negative results. A single-seed screening gain is not
three-seed evidence. Layout uses sampled Kendall tau-b; tile uses sampled top-five regret.
These local metrics are not Kaggle leaderboard scores. Bootstrap intervals are exploratory.
`*_factor_contrasts.csv` identifies the factor changed and its paired gain, including matched
GNN/HGB comparisons. Relative percentages are omitted when the reference score is too low
or negative to make them useful. Screening and confirmed tables remain separate.
''')
code('''
import matplotlib.pyplot as plt
state=json.loads((OUTPUT_DIR/"run_state.json").read_text())
display(pd.DataFrame(state["collections"].items(),columns=["collection","status"]))
for collection in CONFIG["collections"]:
    directory=OUTPUT_DIR/collection.replace(":","_")
    print("\\n",collection)
    lock=directory/"selection_lock.json"
    if lock.exists():print("Selected:",json.loads(lock.read_text())["components"])
    path=directory/"system_holdout_gains.csv"
    if path.exists():display(pd.read_csv(path))
    path=directory/"holdout_confirmed_factor_contrasts.csv"
    if path.exists():display(pd.read_csv(path))
    path=directory/"all_trials.csv"
    if path.exists():
        trials=pd.read_csv(path)
        print("Completed fits:",len(trials),"| observed maximum worker GiB:",round(trials.peak_rss_gib.max(),3))
        screen=trials[trials.seed==CONFIG["seeds"][0]]
        plot=screen[screen.experiment.str.startswith("G_full_n")].groupby("experiment").dev_score.mean()
        if len(plot):
            fig,ax=plt.subplots(figsize=(7,3));plot.plot.bar(ax=ax)
            ax.set_ylabel("Mean development tau across folds — screening seed")
            ax.set_title(collection+": graph-budget screening");fig.tight_layout()
            fig.savefig(directory/"coverage_screening.png",dpi=150);plt.show();plt.close(fig)
submission_status=OUTPUT_DIR/"submission_status.json"
if submission_status.exists():print("Submission status:",json.loads(submission_status.read_text()))
for path in OUTPUT_DIR.glob("submission_*.csv"):print("Candidate CSV:",path)
RESULTS_ZIP=study_controller.make_bundle(OUTPUT_DIR,CONFIG["disk_reserve_gib"])
download_checkpoint(RESULTS_ZIP)
''')
md('''
## 8. What to send back

Send the most recent **results ZIP**, including partial results if the run stopped. It contains
code/settings, graph/configuration manifests, resource pilots, selection decisions, training
histories, models, scores, prediction IDs, failures and submission CSVs when complete.
You do not need to select the next experiment manually. Resume until the required stages
and submission export are complete, or share partial evidence if a resource failure blocks work.

`submission_baseline.csv`, `submission_tabular.csv` and `submission_selected.csv` are candidate
files. The notebook does not upload to Kaggle. All models use only their fold's training graphs;
the final test ensemble averages fold and seed models. It does not retrain on the audit labels.

## Attribution

This is an independent extension of the project's v2 implementation. Established methods are
not new inventions. The prior GNN already had message passing, numeric node features, opcode
embeddings and pairwise loss. The research question here is whether retaining more graph
context and using more labelled training examples improves the controlled references.

- [GraphSAGE paper](https://arxiv.org/abs/1706.02216): neighbourhood sampling and aggregation.
- [1st-place authors](https://github.com/thanhhau097/google_fast_or_slow): representation/preprocessing inspiration.
- [2nd-place authors](https://github.com/dmitrii-khizbullin/kaggle_latenciaga): ranking and configuration comparisons.
- [3rd-place authors](https://github.com/jafluri/kaggle_tpu_graph): representation and seed variability.
- [5th-place write-up](https://www.kaggle.com/competitions/predict-ai-model-runtime/discussion/456093): dimension semantics.
- [Official competition data](https://www.kaggle.com/competitions/predict-ai-model-runtime/data): NPZ schema and normalization.
- [PyTorch checkpoint guide](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html): training-state recovery.
- [Colab FAQ](https://research.google.com/colaboratory/faq.html): temporary runtimes and resource availability.

No competitor solution code, weights or predictions are embedded. Cite inspiration and distinguish
new project experiments from global method novelty. The notebook does not guarantee top 10%.
''')
notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
    language_info=dict(name='python',version='3.11'),colab=dict(name=TARGET.name,provenance=[])),nbformat=4,nbformat_minor=5)
TARGET.write_text(json.dumps(notebook,indent=1)+'\n');print(TARGET)
