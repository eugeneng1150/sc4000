"""Build a self-contained v4 notebook; embedded v3 engines remain unchanged."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
TARGET=HERE.parent/'SC4000_V4_Focused_Scaling_and_Report.ipynb'
cells=[]
def cell(kind,text):
    source=text.strip()+'\n'
    value=dict(cell_type=kind,id=hashlib.sha256(source.encode()).hexdigest()[:12],metadata={},source=source.splitlines(keepends=True))
    if kind=='code':value.update(outputs=[],execution_count=None)
    cells.append(value)
def md(s):cell('markdown',s)
def code(s):cell('code',s)

md('''# SC4000 v4 — focused scaling and report figures

**Benchmark: your completed v3 selected submission (~0.55, user-reported).**
This notebook keeps that hybrid system, expands the winning tabular training sets,
tests one focused XLA GNN training-budget change, and exports the six report figures
and two evidence tables. It does not rerun v3's coverage/component sweep.

Everything is embedded; no Git checkout is required. Use standard Colab **CPU** RAM.
No Google Drive is used. The first run needs your **completed v3 checkpoint 65**;
later resumes need only the latest **v4** ZIP (which includes the v3 reference).
Only load your own trusted model archives: joblib/PyTorch artifacts contain executable
Python serialization. The notebook never submits a file to Kaggle.

Run all cells. Screening, confirmation, selection, audit, inference and charts proceed
automatically. A session budget can pause this multi-session task. After a reset you
must restore a completed local ZIP; automatic browser download requests do not guarantee
receipt. Confirm your Downloads folder. No leaderboard or completion-time guarantee.
''')
md('''## 1. Inputs

First run: upload checkpoint 65 and set `V3_REFERENCE` to its exact ZIP filename.
Leave `RESTORE_ZIP` empty. A local extracted folder also works outside Colab.

Resume after reset: upload a **v4** ZIP and set `RESTORE_ZIP`; `V3_REFERENCE` is ignored.
Resume on the same VM: leave `RESTORE_ZIP` empty and keep the existing output folder.
Never change `RUN_NAME` or scientific settings mid-study. The notebook does not overwrite
nonempty output folders during restore. Incomplete uploads are diagnosed before extraction.
''')
code('''from pathlib import Path
import os, sys, json, zipfile, shutil, subprocess, importlib.metadata
V3_REFERENCE = "/content/sc4000_gnn_v3_results (65).zip"
RESTORE_ZIP = ""  # A v4 checkpoint only; e.g. /content/sc4000_gnn_v4_results.zip
DATA_ROOT = ""
SAMPLE_SUBMISSION = ""
DOWNLOAD_IF_MISSING = True
RUN_NAME = "sc4000_gnn_v4"
AUTO_DOWNLOAD_CHECKPOINTS = True
SESSION_HOURS = 8.0  # Per session, not a promised total duration
CHECKPOINT_MINUTES = 30
WORKER_RAM_GIB = 5.0
SYSTEM_RESERVE_GIB = 2.0
BASE_DIR = Path('/content') if Path('/content').exists() else Path.cwd()
OUTPUT_DIR = BASE_DIR / RUN_NAME
SCRATCH_DIR = BASE_DIR / ('.' + RUN_NAME + '_scratch')
RUNTIME_DIR = BASE_DIR / ('.' + RUN_NAME + '_runtime')
RUNTIME_DIR.mkdir(parents=True,exist_ok=True)
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='2'
os.environ['MPLCONFIGDIR']=str(RUNTIME_DIR/'matplotlib')
''')
md('''## 2. Match the saved model's dependencies

The v3 archive records scientific package versions. Matching them avoids loading old HGB
models into incompatible scikit-learn versions and prevents silently changing the study.
Installations use `--no-cache-dir`. If packages already imported by Colab change, this cell
asks for **Restart session**, then **Run all**; do not delete the runtime or its files.
''')
code('''def archive_metadata(path, member):
    path=Path(path)
    if path.is_dir(): return json.loads((path/member).read_text())
    if not path.exists(): raise FileNotFoundError(f'Upload your backup and set its exact path: {path}')
    if not zipfile.is_zipfile(path):
        raise ValueError(f'Not a complete ZIP: {path} ({path.stat().st_size:,} bytes). Wait for upload; compare size with your local file.')
    with zipfile.ZipFile(path) as z: return json.loads(z.read(member))

if RESTORE_ZIP:
    meta=archive_metadata(RESTORE_ZIP,'run_identity.json')
    if meta.get('version')!='4.0.0':raise ValueError('RESTORE_ZIP must be v4. Put checkpoint 65 in V3_REFERENCE for a fresh run.')
    versions=meta['versions']
elif (OUTPUT_DIR/'reference_v3').exists():
    versions=archive_metadata(OUTPUT_DIR/'reference_v3','run_identity.json')['versions']
else:
    meta=archive_metadata(V3_REFERENCE,'run_identity.json')
    if meta.get('version')!='3.0.0':raise ValueError('V3_REFERENCE must be a v3 archive')
    versions=meta['versions']
changed=[]
for name,wanted in versions.items():
    try: installed=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: installed=None
    if installed!=wanted: changed.append(name)
ordinary=[f'{n}=={versions[n]}' for n in changed if n!='torch']
if ordinary:subprocess.check_call([sys.executable,'-m','pip','install','--quiet','--no-cache-dir',*ordinary])
if 'torch' in changed:
    subprocess.check_call([sys.executable,'-m','pip','install','--quiet','--no-cache-dir',
                          '--index-url','https://download.pytorch.org/whl/cpu','torch=='+versions['torch']])
for name in ['matplotlib','threadpoolctl']:
    try:
        installed=importlib.metadata.version(name)
        if name=='matplotlib' and installed!='3.10.6':
            subprocess.check_call([sys.executable,'-m','pip','install','--quiet','--no-cache-dir','matplotlib==3.10.6'])
            changed.append(name)
    except importlib.metadata.PackageNotFoundError:
        package='matplotlib==3.10.6' if name=='matplotlib' else name
        subprocess.check_call([sys.executable,'-m','pip','install','--quiet','--no-cache-dir',package])
module_names={'scikit-learn':'sklearn'}
if any(module_names.get(n,n) in sys.modules for n in changed):
    raise RuntimeError('Dependencies updated. Use Runtime > Restart session, then Run all. Keep uploaded files and these settings.')
print('Pinned scientific versions:',versions)
''')
md('''## 3. Embedded implementation

The six `study_*` modules below are the unchanged v3 feature/model implementation.
V4 adds a separate controller and protocol; v3 source files and results are preserved.
''')
for name in ['engine','gnn','common','plan','submission','controller']:
    source=(HERE.parent/'ablation_v3'/(name+'.py')).read_text()
    code(f'# Frozen v3 dependency: {name}\n(RUNTIME_DIR / "study_{name}.py").write_text({source!r})')
for name in ['v4_core','v4_worker','v4_controller','v4_report']:
    source=(HERE/(name+'.py')).read_text()
    code(f'# V4 module: {name}\n(RUNTIME_DIR / "{name}.py").write_text({source!r})')
code('''sys.path.insert(0,str(RUNTIME_DIR))
import importlib, pandas as pd
import v4_core, v4_controller, v4_report, study_submission, study_common
for module in [v4_core,v4_controller,v4_report]:importlib.reload(module)
if RESTORE_ZIP:v4_core.restore(RESTORE_ZIP,OUTPUT_DIR)
REFERENCE_DIR=v4_core.import_reference(V3_REFERENCE,OUTPUT_DIR)
CONFIG=v4_core.defaults(REFERENCE_DIR)
CONFIG.update(session_hours=SESSION_HOURS,checkpoint_minutes=CHECKPOINT_MINUTES,
              worker_ram_gib=WORKER_RAM_GIB,system_reserve_gib=SYSTEM_RESERVE_GIB)
study_common.validate_config(CONFIG)
print('Reference:',REFERENCE_DIR)
print('Output:',OUTPUT_DIR)
print('Existing completed v4 fits:',len(list(OUTPUT_DIR.glob('*/folds/*/trials/*/*/result.json'))))
''')
md('''## 4. Locate competition data

`DATA_ROOT` may point to the folder containing `layout/` and `tile/`, or its parent.
If missing, Kaggle authentication is read from the environment or Colab Secrets.
The account must already have accepted the competition rules. Data stays on local storage.
''')
# Reuse the tested v3 data discovery/download cell, with CONFIG already defined above.
old=json.loads((HERE.parent/'SC4000_V3_Coverage_and_Data.ipynb').read_text())
data_cell=next(''.join(c['source']) for c in old['cells'] if c['cell_type']=='code' and 'def locate_data' in ''.join(c['source']))
code(data_cell)
md('''## 5. Generate the report pack now (no training)

This works from v3 evidence even if new training is postponed. Six figures are exported as
PNG and PDF with source CSVs and captions. The dataset chart reads only NPZ shape metadata.
The component intervals use confirmed results; coverage/data grids are explicitly screening.
Kaggle values below are approximate user reports: replace them with verified exact values
before final report submission. Local audit and Kaggle metrics remain separate.
''')
code('''REPORT_DIR=v4_report.generate(REFERENCE_DIR,OUTPUT_DIR,NPZ_ROOT,
    kaggle={'baseline':0.40,'tabular':0.53,'selected':0.55})
print('Report figures, captions and tables:',REPORT_DIR)
from IPython.display import display, Image
display(Image(filename=str(REPORT_DIR/'06_final_comparison.png')))
''')
md('''## 6. Fixed experimental plan

| Track | New experiments | Control |
|---|---|---|
| NLP default/random | All eligible training graphs at 256 then 384 configs | V3 deeper HGB at 128 graphs |
| Tile | 640 / 1,280 / 2,560 / 5,120 / all eligible graphs at 256 configs, then 384 at largest size admitting both counts | V3 deeper HGB at 320 graphs |
| XLA default/random | Same winning GNN, graph geometry and training examples; 40 epochs / patience 10 | V3 20 epochs / patience 5 |
| XLA blend | 0 / .25 / .5 / .75 / 1 GNN weights using saved dev predictions | Exact v3 selected blend |

Each size must pass a matrix estimate and a measured 10-iteration HGB pilot. A skipped
size is logged with the resource reason; **“all data” is not promised on standard RAM**.
Training graphs/configuration samples are nested relative to v3. Dev graphs stay excluded.
The 384-config comparison uses the same graphs as its 256-config control.

Screen with seed 42 on both folds; confirm only the strongest eligible new recipe per
collection with seeds 137 and 2026. The GNN change is a combined training-budget change,
not an architecture ablation, and is trained fresh with the same seed. Existing v3 fits
are never rerun. Interrupted v4 GNN jobs resume their saved epoch/optimizer/RNG state.

A new system must improve the development mean by at least .002 (layout) or .0005
(tile) and must not reduce either fold relative to **v3 selected**. Otherwise keep v3.
These are predeclared pragmatic thresholds, not significance tests. Development reuse
can overfit; official valid is an already-inspected audit. Audit labels never choose
the replacement. No automatic refit on dev/audit data is done after selection.

One worker, 5 GiB worker cap, 2 GiB system reserve, 1 GiB matrix bound, 4 GiB scratch,
8 GiB output, 4 GiB disk reserve. Pilots must leave extra memory headroom. The worker
guard is a monitored threshold, not a guarantee against sudden allocation spikes.
Scratch is removed at stage changes/exit; source NPZs and checkpoint models are retained.
''')
code('''plan=[]
for c in CONFIG['collections']:
    for e in v4_core.plans(REFERENCE_DIR,c,CONFIG):
        plan.append({'collection':c,'experiment':e['name'],'model':e['model'],
                     'graphs':e.get('graph_limit',e.get('train_graphs')) or 'all eligible',
                     'configs':e['configs'],'factor':e['stage']})
display(pd.DataFrame(plan))
print('Plus one 384-config trial per tabular collection at its largest graph set admitting both configuration counts.')
print('Two screening folds per recipe; up to one new recipe per collection gets all three seeds.')
''')
md('''## 7. Run the continuation and request checkpoint downloads

The cell prints `SKIP`, `PREPARE`, `PILOT`, `RUN`, `LOCKED`, and `TEST`. Pilot timing
is only an approximate fit cost; it excludes feature extraction and test inference.
The study may need multiple Colab sessions. It does not keep a runtime alive artificially.

Every 30 minutes, at collection boundaries and on handled interruption, the notebook
creates a complete replacement ZIP and requests a browser download. **A “ZIP READY”
line proves the local archive was created, not that your computer received it.** Colab
downloads can queue until a cell stops. You can download the completed `.zip` from Files;
never use `.zip.tmp`. Match the byte size and verify it opens locally before disconnecting.
The reference v3 archive is embedded, so v4 backups are several hundred MB.

If you stop the cell once, let cleanup/checkpoint finish. A killed VM cannot run cleanup.
New VM: upload the latest completed v4 ZIP, set `RESTORE_ZIP`, then Run all. Same VM:
clear `RESTORE_ZIP`, leave the output folder intact and rerun this cell.
''')
code('''def download_checkpoint(path):
    print('Backup path:',path,'| bytes:',Path(path).stat().st_size)
    if AUTO_DOWNLOAD_CHECKPOINTS:
        try:
            from google.colab import files
            files.download(str(path))
        except ImportError:print('Outside Colab: copy the ZIP from this path.')
        except Exception as exc:print('Download request failed:',exc,'— download manually from Files.')
    print('Verify a completed ZIP in your computer Downloads; a request is not delivery.')

RESULTS_ZIP=v4_controller.run(NPZ_ROOT,OUTPUT_DIR,SCRATCH_DIR,RUNTIME_DIR,CONFIG,
                             on_checkpoint=download_checkpoint)
''')
md('''## 8. Review status, export figures and download the final bundle

`complete` means all planned/admitted work and submission export finished. `paused`
means restore/resume is required. Errors preserve a checkpoint and identify the log.
Resource exclusions are listed explicitly. The report pack is useful even when paused.

Final CSVs: `submission_v3_reference.csv` (unchanged benchmark),
`submission_tabular_update_only.csv` (preserves XLA v3 rows), and
`submission_selected.csv` (all development-accepted changes). They may be identical
if nothing passes the replacement rule. Do not submit until submission status says complete.
`submission_provenance.csv` records which collection/model produced each row.
''')
code('''state=v4_core.read(OUTPUT_DIR/'run_state.json')
print('Overall:',state['status'])
display(pd.DataFrame(state['collections'].items(),columns=['collection','status']))
for c in CONFIG['collections']:
    d=OUTPUT_DIR/c.replace(':','_')
    if (d/'selection_lock.json').exists():
        lock=v4_core.read(d/'selection_lock.json')
        print(c,lock['components'],'| development gain:',lock['gain_vs_v3'])
    if (d/'system_audit_gains.csv').exists():display(pd.read_csv(d/'system_audit_gains.csv'))
    for p in d.glob('admission/*/decision.json'):
        if not v4_core.read(p)['admitted']:print('Resource exclusion:',p.parent.name,v4_core.read(p)['reason'])
if (OUTPUT_DIR/'submission_status.json').exists():print(v4_core.read(OUTPUT_DIR/'submission_status.json'))
REPORT_DIR=v4_report.generate(REFERENCE_DIR,OUTPUT_DIR,NPZ_ROOT)
from study_controller import make_bundle
RESULTS_ZIP=make_bundle(OUTPUT_DIR,CONFIG['disk_reserve_gib'])
download_checkpoint(RESULTS_ZIP)
print('Send back this results ZIP. It contains code, models, predictions, decisions, six report figures and source tables.')
''')
md('''## Report structure and attribution

Use approximately six main figures: pipeline; dataset/coverage; component ablations;
coverage scores; training-data grids; final comparison. Two compact tables describe
setup/resources and findings/decisions. Appendix plots add v4 results. `report/CAPTIONS.md`
and `report/REPORT_GUIDE.md` explain evidence scope and limitations. This is not the final
ten-page report. V2 provenance/earlier experiments remain in the project evidence folder.

The established GraphSAGE architecture, embeddings and pairwise loss are not new inventions.
This is an independent extension of the project's v3 implementation. Competitor code,
weights and predictions are not included. Cite the project's prior-work record and the
original GraphSAGE paper when writing the report. No improvement to 0.59 is guaranteed.
''')
TARGET.write_text(json.dumps(dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
    language_info=dict(name='python'),colab=dict(name=TARGET.name,provenance=[])),nbformat=4,nbformat_minor=5),indent=1)+'\n')
print(TARGET)
