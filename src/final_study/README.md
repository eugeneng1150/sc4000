# Reproducible final notebook

Open [SC4000_Final.ipynb](../../notebooks/SC4000_Final.ipynb). It tells one
experimental story and **loads compatible models or trains missing ones from raw
competition data**. No historical model ZIP is required.

## Execution modes

- `RUN_MODE = "reproduce"` (default): train/load the fixed selected models and
  basic baseline, evaluate them and run full-candidate test inference. There are
  **90 required fits**: 60 selected members and 30 baseline members. Two folds and
  three seeds are retained; this is not a new model search.
- `RUN_MODE = "review"`: regenerate historical figures and tables only. This mode
  produces no new Kaggle submission and does not load model objects.
- `RUN_ABLATIONS = True`: add the recorded successful comparison recipes.
  With `ABLATION_SEEDS = [42]`, the complete plan has **286 fits**, including the
  default 90; with all three seeds it has **450 fits**. Reusable fits are skipped.
  This is optional and potentially much longer than final-model reproduction.
  Resource-excluded jobs and early MLP/gated-GraphSAGE experiments are not rerun.

The checked-in saved outputs were executed in **review mode** so readers can see
historical evidence without training. Source defaults remain `reproduce`. Current
metrics and new submission provenance are generated only by the reproduction path.

## Run from raw data

1. Open the standalone notebook in Colab CPU. Set `DATA_ROOT` to the extracted
   competition data; the locator accepts a parent of `npz_all/npz`. Set
   `SAMPLE_SUBMISSION` if autodetection cannot find the competition template.
2. Alternatively enable `DOWNLOAD_IF_MISSING` with your own Kaggle credentials
   and accepted competition access. The notebook does not provide credentials/data.
3. Run the dependency cell. It matches the recorded scientific versions. If it
   requests a session restart, restart without deleting files, then Run all again.
4. Keep `RUN_ABLATIONS=False` for the final system plus baseline. Models are fitted
   in isolated sequential workers with resource guards. A fresh CPU run can span
   multiple sessions; no fixed completion time is promised.
5. Keep completed checkpoint downloads. Same runtime: leave the results directory
   intact and rerun. New runtime: upload the final-run backup, set `RESTORE_RUN_ZIP`
   and restore into an empty results directory; clear this setting after restoring.

Default worker cap is 5 GiB with a 2 GiB system reserve, bounded scratch and
per-job/session limits. Data, scratch and output directories must be separate.
There is no expanded raw-array cache. Scratch groups are removed after use.
A required model resource failure stops the run instead of substituting a model.
An interrupted fit restarts; completed fits and completed graph predictions resume.

## Optional existing models

`MODEL_ARCHIVE` accepts the complete historical v5 archive or extracted directory.
Known artifact hashes are checked before model loading; only required model files
are copied, not archive code. Missing models are trained. Locally that archive is
`local/results/sc4000_gnn_v5_results (11)/` from the repository root.

New reproduction runs use receipts binding recipe, recorded split/configuration
IDs, runtime source and dependency versions to model hashes. An incompatible run
identity requires a new `RUN_NAME`; corrupt/missing fit artifacts trigger a refit.
The historical full checkpoint and the new final-run backup are different formats:
use `MODEL_ARCHIVE` for the former and `RESTORE_RUN_ZIP` for the latter.

## Outputs and interpretation

Under `sc4000_final_reproduction/`:

- `models/`: trained/imported models, training manifests, receipts and GNN histories.
- `evaluation/current_audit.csv`: baseline/current selected comparison with absolute
  changes and relative changes where the reference score is at least 0.1.
- `evaluation/fit_results.csv`, `method_graph_scores.csv`: current individual fit
  and per-graph results, including requested optional experiments. Match protocol,
  data and seeds before interpreting a difference as a controlled ablation.
- `submission_final.csv`, `submission_baseline.csv`: **fresh model predictions**.
- `submission_provenance.json`: model/source/environment identity and comparison
  with the frozen CSV hash. No new Kaggle score is assigned without submission.
- `run_state.json`, logs and periodic `<RUN_NAME>_results.zip`: progress/recovery.

Under `sc4000_final_reproduction_historical_report/`: six main figures, one
supporting learning-curve figure, tables and captions. Historical evidence and
current-run metrics are explicitly separated. The historical score is **0.567
(user-reported)**; retraining does not guarantee byte-identical predictions or the
same public score on every platform. The prior submitted CSV is preserved under
[artifacts/final](../../artifacts/final/README.md).

The narrative follows baseline → model families → controlled improvements →
complementarity → final results. Earlier MLP results use a different protocol.
Gated GraphSAGE has recovered code but no verified run outcome. Mask/context
intervals cross zero; larger graph/data budgets do not consistently help. Percentile
targets help selected blends. Audit data was previously inspected; its metrics
are distinct from the public score. Tau ratios near zero are suppressed.

## Implementation and checks

- `build_reproduction_pack.py`: curates fixed recipes, label-free data manifests,
  training configuration IDs and optional archive hashes from the historical results.
  The resulting 5.9 MiB `artifacts/final/reproduction_spec.zip` is embedded in the
  notebook and contains no weights or raw data. Ordinary users do not rebuild it.
- `reproduce.py`, `final_worker.py`: orchestration, cache validation, exact training
  example checks, resource-isolated fitting/evaluation and submission export.
- `story.py`: question-led figures and tables with explicit evidence scope.
- `final_report.py`: retained historical archive integrity helpers; its CSV replay
  function is **not called** by the new final notebook.
- `build_notebook.py`: embeds all sources/specifications and clears saved outputs.

```sh
python src/final_study/test_final.py
python src/final_study/test_reproduce.py
python src/final_study/build_notebook.py
```

Tests require the scientific dependencies, including PyTorch. Process-monitoring
integration tests require an environment that permits inspecting owned workers.
Synthetic tests exercise real HGB/percentile/GNN fits, optional XGBoost controls,
warm resume, missing-model recovery, optional archive reuse, short-Tile padding
and changed-data rejection.
The 450 archived training selections are checked against their recorded IDs.
Full competition training is not part of the test suite.
