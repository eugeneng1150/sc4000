"""Build the complete Colab notebook with frozen dependencies embedded."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
TARGET=HERE.parents[1]/'notebooks/experiments'/'SC4000_V5_Ranking_and_Targets.ipynb'
cells=[]


def cell(kind,text):
    source=text.strip()+'\n'
    value=dict(cell_type=kind,id=hashlib.sha256(source.encode()).hexdigest()[:12],
               metadata={},source=source.splitlines(keepends=True))
    if kind=='code':
        value.update(outputs=[],execution_count=None)
    cells.append(value)


def md(text):cell('markdown',text)
def code(text):cell('code',text)


md('''# SC4000 v5 — targets and boosted-tree ranking

**Reference: your completed v4 selected submission, user-reported Kaggle score 0.558.**
This self-contained notebook tests three tabular changes on the four layout collections:
percentile-target HGB, XGBoost regression, and grouped XGBoost pairwise ranking.
It reuses the existing features, training examples, development folds and trained GNNs.
Tile is carried forward unchanged. There is **no GNN training**.

**2026-10-21 repair:** the HGB pilot estimator handles inherited tree counts.
Original v5 checkpoints are upgraded automatically only when scientific settings,
reference, dependencies and all other runtime sources match. Completed fits are reused.

Run all on standard Colab CPU. No repository checkout or Drive mount is needed.
On the first run upload the completed **v4 ZIP**. Later, a **v5 ZIP** is sufficient:
it embeds the v4 and v3 references. Only use your own trusted model archives.

At most two isolated jobs run together after measured pilots and resource checks.
Each uses one CPU thread. Parallelism is conditional, not a guarantee of 2× speed.
The notebook pauses at its session budget and can resume completed jobs.
It does not submit anything to Kaggle.
''')
md('''## 1. Paths and resource settings

First run: set `V4_REFERENCE` to the uploaded completed v4 ZIP and leave `RESTORE_ZIP` empty.
After a runtime reset: upload the latest completed **v5** ZIP and set `RESTORE_ZIP`.
On the same runtime: clear `RESTORE_ZIP` and reuse the existing output directory.
Do not restore over a populated directory or change scientific settings mid-study.
''')
code('''from pathlib import Path
import os, sys, json, zipfile, shutil, subprocess, importlib.metadata
V4_REFERENCE = "/content/sc4000_gnn_v4_results.zip"
RESTORE_ZIP = ""  # v5 checkpoint ONLY, after a reset
DATA_ROOT = ""
SAMPLE_SUBMISSION = ""
DOWNLOAD_IF_MISSING = True
RUN_NAME = "sc4000_gnn_v5"
AUTO_DOWNLOAD_CHECKPOINTS = True
MAX_PARALLEL_JOBS = 2  # automatic admission; set 1 to force sequential execution
SESSION_HOURS = 8.0   # session budget, not a total-run ETA
CHECKPOINT_MINUTES = 30
WORKER_RAM_GIB = 5.0
SYSTEM_RESERVE_GIB = 2.0
BASE_DIR = Path("/content") if Path("/content").exists() else Path.cwd()
OUTPUT_DIR = BASE_DIR / RUN_NAME
SCRATCH_DIR = BASE_DIR / ("." + RUN_NAME + "_scratch")
RUNTIME_DIR = BASE_DIR / ("." + RUN_NAME + "_runtime")
RUNTIME_DIR.mkdir(parents=True,exist_ok=True)
for key in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
    os.environ[key] = "1"
os.environ["MPLCONFIGDIR"] = str(RUNTIME_DIR/"matplotlib")
''')
md('''## 2. Match archived dependencies

The original scientific versions are required to load and reuse the saved models.
XGBoost is pinned to 3.0.5 and runs on CPU. Downloads use `--no-cache-dir`.
If an already-imported package changes, **Restart session**, then **Run all**.
Restarting the session is different from deleting the runtime and its files.
''')
code('''def archive_metadata(path, member):
    path=Path(path)
    if path.is_dir():return json.loads((path/member).read_text())
    if not path.exists():raise FileNotFoundError(f"Upload the backup and check the exact filename: {path}")
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Incomplete upload or invalid ZIP: {path} ({path.stat().st_size:,} bytes). Compare with your local file.")
    with zipfile.ZipFile(path) as z:return json.loads(z.read(member))

if RESTORE_ZIP:
    meta=archive_metadata(RESTORE_ZIP,"run_identity.json")
    if meta.get("version")!="5.0.0":raise ValueError("RESTORE_ZIP must be v5; use V4_REFERENCE for the first run.")
elif (OUTPUT_DIR/"reference_v4").exists():
    meta=archive_metadata(OUTPUT_DIR/"reference_v4","run_identity.json")
else:
    meta=archive_metadata(V4_REFERENCE,"run_identity.json")
    if meta.get("version")!="4.0.0":raise ValueError("V4_REFERENCE must be a completed v4 archive.")
versions=dict(meta["versions"])
versions.update(xgboost="3.0.5")
versions.setdefault("matplotlib","3.10.6")
versions.setdefault("threadpoolctl","3.6.0")
changed=[]
for name,wanted in versions.items():
    try:installed=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:installed=None
    if installed!=wanted:changed.append(name)
ordinary=[f"{n}=={versions[n]}" for n in changed if n not in ["torch","xgboost"]]
if ordinary:subprocess.check_call([sys.executable,"-m","pip","install","--quiet","--no-cache-dir",*ordinary])
if "xgboost" in changed:
    # numpy/scipy are installed above; CPU execution does not need CUDA/NCCL packages.
    subprocess.check_call([sys.executable,"-m","pip","install","--quiet","--no-cache-dir","--no-deps","xgboost==3.0.5"])
if "torch" in changed:
    subprocess.check_call([sys.executable,"-m","pip","install","--quiet","--no-cache-dir",
        "--index-url","https://download.pytorch.org/whl/cpu","torch=="+versions["torch"]])
modules={"scikit-learn":"sklearn"}
if any(modules.get(n,n) in sys.modules for n in changed):
    raise RuntimeError("Dependencies updated. Restart session, then Run all. Keep uploaded files and these settings.")
print("Dependencies:",versions)
''')
md('''## 3. Embedded, version-checked implementation

Feature extraction and the previous models remain unchanged. V5 adds isolated tree
workers, a scheduler, controlled selection, and a compact report.
''')
for name in ['engine','gnn','common','plan','submission','controller']:
    text=(HERE.parent/'ablation_v3'/(name+'.py')).read_text()
    code(f'(RUNTIME_DIR/"study_{name}.py").write_text({text!r})')
for name in ['v4_core','v4_worker']:
    text=(HERE.parent/'ablation_v4'/(name+'.py')).read_text()
    code(f'(RUNTIME_DIR/"{name}.py").write_text({text!r})')
for name in ['v5_core','v5_worker','v5_scheduler','v5_controller','v5_report']:
    text=(HERE/(name+'.py')).read_text()
    code(f'(RUNTIME_DIR/"{name}.py").write_text({text!r})')
code('''sys.path.insert(0,str(RUNTIME_DIR))
import pandas as pd
import v5_core, v5_controller, v5_report, study_common, study_submission
from v4_core import restore
if RESTORE_ZIP:restore(RESTORE_ZIP,OUTPUT_DIR,12)
REFERENCE_DIR=v5_core.import_reference(V4_REFERENCE,OUTPUT_DIR)
CONFIG=v5_core.defaults(REFERENCE_DIR)
CONFIG.update(max_parallel=MAX_PARALLEL_JOBS,session_hours=SESSION_HOURS,
              checkpoint_minutes=CHECKPOINT_MINUTES,worker_ram_gib=WORKER_RAM_GIB,
              system_reserve_gib=SYSTEM_RESERVE_GIB)
if MAX_PARALLEL_JOBS not in [1,2]:raise ValueError("Choose one or two workers.")
study_common.validate_config(CONFIG)
print("Reference:",REFERENCE_DIR)
print("Results:",OUTPUT_DIR)
print("Completed new fits:",len(list(OUTPUT_DIR.glob("*/folds/*/trials/*/*/result.json"))))
''')
md('''## 4. Competition data

Use local Colab storage. Existing files are reused. If missing, the data-discovery
cell can download using your existing Kaggle credentials/Colab Secrets after you
have accepted the competition rules. It does not mount Drive.
''')
old=json.loads((HERE.parents[1]/'notebooks/experiments'/'SC4000_V3_Coverage_and_Data.ipynb').read_text())
data_cell=next(''.join(c['source']) for c in old['cells'] if c['cell_type']=='code' and
              'def locate_data' in ''.join(c['source']))
code(data_cell.replace('study_submission.preflight(NPZ_ROOT,CONFIG)',
                       'study_submission.preflight(NPZ_ROOT,dict(CONFIG,collections=CONFIG["submission_collections"]))'))
md('''## 5. Controlled experiments and selection

| Arm | Target/objective | Comparison |
|---|---|---|
| Saved selected HGB | Centred log runtime / squared error | Reused benchmark |
| V5_H_percentile | Within-graph runtime percentile / squared error | Target change only |
| V5_X_regression | Centred log runtime / squared error | Tree-library change |
| V5_X_pairwise | Within-graph pairwise logistic ranking | Same XGBoost capacity and rows as regression |

Training graph/configuration IDs must match the selected tabular reference for each
fold and seed. NLP random therefore uses the expanded v4 recipe; other collections
use their retained recipes. No new features, data scaling, GNN fitting or tile search.
Higher relevance means faster for XGBoost ranking; its predictions are sign-reversed
before entering our lower-is-faster evaluation pipeline. Ties are preserved.
The ranking arm uses equal query weights; regression uses normalized graph weights.
This objective comparison is a learning formulation comparison, not equal mathematical
weighting of every pair and every regression residual.

Screen three arms × two folds × four collections = **24 new fits**. Only the best
eligible new recipe gets seeds 137 and 2026. If either XGBoost arm qualifies, confirm
its matched counterpart too so the objective comparison remains available:
**at most 56 new fits total**, often fewer. Pilots, audit and inference are separate.
Identical predictions across seeds are reported as such, not independent evidence.

Keep v4 unless the development mean improves by at least 0.002 and neither fold
regresses. Test a fixed small grid of new-model/current-system rank blends, plus
replacement of just the existing tabular member in an XLA hybrid. No unrestricted
weight search. Freeze selection before audit. Audit labels never promote a model.
Development reuse is adaptive; the official validation audit was inspected previously.
''')
code('''plan=[]
for c in CONFIG["collections"]:
    control,source=v5_core.tabular_reference(REFERENCE_DIR,c)
    for e in v5_core.plans(REFERENCE_DIR,c,CONFIG):
        plan.append(dict(collection=c,control=control,experiment=e["name"],
                         model=e["model"],configs=e["configs"],stage=e["stage"]))
display(pd.DataFrame(plan))
print("24 screening fits; at most 32 additional confirmation fits.")
print("Feature preparation and 10/30-tree full-row pilots precede fitting.")
''')
md('''## 6. How parallel admission and estimates work

1. Extract one collection/fold's compact features sequentially. Share them read-only
   among jobs; delete the scratch cache when moving to another fold/stage and on exit.
2. Run each model's 10-tree and 30-tree pilots sequentially on its full training matrix.
   Estimate full fit time and memory growth, then add 50% memory headroom plus 0.25 GiB.
3. Launch at most two jobs with one thread each only when CPU capacity, current free
   RAM, outstanding memory reservations and the 2 GiB system reserve permit it.
   Otherwise run sequentially. A one-core allocation always runs sequentially.
4. Monitor RSS, system memory, disk and session/job limits. If memory pressure appears
   while paired, stop one job and retry it sequentially; successful jobs remain saved.
   An unexpected error stops with diagnostics rather than becoming a poor model score.

Pilots are estimates, not peak-memory or timing guarantees. A small pilot may miss
later allocation spikes; guards reduce risk but cannot guarantee avoidance of OOM.
Time estimates cover fitting; extraction, evaluation, GNN **prediction** for changed
XLA blends, and ZIP creation take additional time. Two jobs may contend for memory bandwidth.
The earlier 4–8-hour estimate was provisional; use this run's measured estimates.

Only completed worker outputs enter ZIP checkpoints. Unfinished fits restart from
their beginning after reset; completed fits are skipped. Raw-data caches are not retained.
''')
md('''## 7. Run and download checkpoints

The cell automatically screens, confirms, locks, audits, predicts and exports.
You will see `ESTIMATE`, `QUEUE`, worker counts, actual time/memory, `SKIP`,
`LOCKED`, and finally `ALL MODEL TRAINING COMPLETE`.

ZIPs are created every 30 minutes, at collection boundaries, and on handled stops.
Downloads may queue in Colab until the cell ends. **Verify a completed file in your
computer Downloads**; `ZIP READY` proves only the Colab archive exists. You can also
download the completed `.zip` from Files. Never download `.zip.tmp`.
Stopping once allows worker cleanup and a backup; deleting the VM cannot do this.
''')
code('''def download_checkpoint(path):
    print("Backup:",path,"| bytes:",Path(path).stat().st_size)
    if AUTO_DOWNLOAD_CHECKPOINTS:
        try:
            from google.colab import files
            files.download(str(path))
        except ImportError:print("Outside Colab: copy the displayed ZIP.")
        except Exception as exc:print("Download request failed:",exc,"— use the Files sidebar.")
    print("Verify the completed download on your computer.")

RESULTS_ZIP=v5_controller.run(NPZ_ROOT,OUTPUT_DIR,SCRATCH_DIR,RUNTIME_DIR,CONFIG,
                             on_checkpoint=download_checkpoint)
''')
md('''## 8. Review and final download

Only `status: complete` with a completed submission export means the study has finished.
`paused` needs another session. An excluded experiment is not a negative accuracy result.
Keep the v4 submission if no candidate passes selection; do not change the rule after seeing scores.

Upload **`submission_selected.csv`** from the v5 results root to Kaggle.
`submission_v4_reference.csv` is the unchanged benchmark.
`submission_provenance.csv` is supporting evidence, not a Kaggle submission.
Send the final ZIP back for review. It includes models, predictions, timing/resource
records, frozen selections, source tables and figures, plus the complete earlier reference.
''')
code('''state=v5_core.read(OUTPUT_DIR/"run_state.json")
print("Study:",state["status"])
display(pd.DataFrame(state["collections"].items(),columns=["collection","status"]))
for c in CONFIG["collections"]:
    p=OUTPUT_DIR/c.replace(":","_")
    if (p/"selection_lock.json").exists():
        lock=v5_core.read(p/"selection_lock.json")
        print(c,lock["components"],"| dev gain vs v4:",lock["gain_vs_v4"])
    if (p/"system_audit_gains.csv").exists():display(pd.read_csv(p/"system_audit_gains.csv"))
if (OUTPUT_DIR/"submission_status.json").exists():print(v5_core.read(OUTPUT_DIR/"submission_status.json"))
REPORT_DIR=v5_report.generate(OUTPUT_DIR)
print("New report:",REPORT_DIR)
print("Earlier six figures:",REFERENCE_DIR/"report")
from v5_scheduler import snapshot
RESULTS_ZIP=snapshot(OUTPUT_DIR,CONFIG["disk_reserve_gib"])
download_checkpoint(RESULTS_ZIP)
''')
md('''## Attribution and interpretation

Gradient boosting and pairwise ranking are established methods. This notebook tests
their value with our existing compact features and sampled-GNN system; it does not
claim to invent a new learning algorithm. No competitor code, weights or predictions
are incorporated.

Primary implementation references:
[XGBoost learning to rank](https://xgboost.readthedocs.io/en/release_3.0.0/tutorials/learning_to_rank.html)
and [Python API](https://xgboost.readthedocs.io/en/release_3.0.0/python/python_api.html).
Graph IDs define query groups, and selection uses actual development Kendall tau.
No inference of a Kaggle score or guarantee of 0.59 is made from local metrics.
''')
TARGET.write_text(json.dumps(dict(cells=cells,metadata=dict(
    kernelspec=dict(display_name='Python 3',language='python',name='python3'),
    language_info=dict(name='python'),colab=dict(name=TARGET.name,provenance=[])),
    nbformat=4,nbformat_minor=5),indent=1)+'\n')
print(TARGET)
