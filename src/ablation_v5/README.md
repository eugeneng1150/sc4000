# V5: controlled targets and tree ranking

Open [SC4000_V5_Ranking_and_Targets.ipynb](../../notebooks/experiments/SC4000_V5_Ranking_and_Targets.ipynb)
in Colab CPU. The notebook embeds all code. See [the protocol](../../docs/evidence/v5_plan.md).

## Run

1. Upload the completed `sc4000_gnn_v4_results.zip`.
2. Set `V4_REFERENCE` to that exact `/content/...zip` path. Leave `RESTORE_ZIP=""`.
3. Run all. If dependency installation requests it, restart the session without
   deleting files, then Run all again.
4. Verify completed checkpoint downloads on your computer.
5. Wait for the study and submission status to say `complete`. Upload
   `submission_selected.csv` from the **v5 root** to Kaggle.

After a runtime reset, upload the latest completed **v5 ZIP**, set `RESTORE_ZIP`,
and Run all. The archive includes the v4/v3 references. On the same runtime,
leave `RESTORE_ZIP` empty and reuse the output directory. Completed fits print
SKIP; interrupted tree fits restart from the beginning. Scientific config,
dependencies, source, reference and dataset identities are validated.

## Experiment

Four layout collections; three new arms: percentile HGB, XGBoost regression,
XGBoost pairwise ranking. Existing feature extraction and per-fold/seed training
graph/configuration IDs are reused and explicitly checked. No new features,
GNN training, tile training, all-data refitting or automatic Kaggle submission.
Changed XLA submissions can require GNN forward inference from saved weights.

24 seed-42 screening fits. Confirm at most one new leader per collection;
when it is an XGBoost arm, also confirm its paired objective control.
At most 56 fits total (usually fewer), plus full-row pilots, audit and inference.
The full fixed blend grid and rejected candidates are saved. Selection must
gain at least 0.002 mean development tau and cannot regress on either fold.
Tile and unchanged collection test rows are copied exactly from v4.

This protocol is not a novelty claim. XGBoost ranking is an established method;
the project evaluates it on its independently implemented feature representation.
Reused development folds and an already-inspected audit limit generalisation
claims. Repeated deterministic tree fits are not independent evidence.

## Parallel execution

Feature extraction is sequential. All models for a fold share one read-only
compact cache. Each model gets sequential 10/30-tree pilots on its complete
training matrix. Admission extrapolates tree memory growth and applies a 1.5×
factor plus 0.25 GiB; the per-worker cap defaults to 5 GiB.

The scheduler uses CPU affinity/quota where available, current available RAM,
unallocated reservations for running jobs, and a 2 GiB system reserve. It runs
at most **two workers with one thread each**. Set `MAX_PARALLEL_JOBS=1` to force
sequential execution. Pairing is reassessed dynamically; pilot-based queue
estimates show sequential and resource-admitted paired fit time.

If paired jobs trigger memory pressure, one is stopped and retried sequentially.
An unresolved single-worker memory failure or an unexpected error stops the run
with diagnostics. Runtime guards cannot prevent every abrupt allocation/OOM.
Parallelism can contend for bandwidth; no 2× speedup or total-time promise.

Workers write into private `.inflight` directories. Only successful results are
committed to normal result directories, with success status written last.
Checkpoints exclude `.inflight`, so a running model is never archived as complete.
Scratch is removed on stage changes and exit; source data and completed models
are preserved. ZIPs use atomic replacement; browser delivery must be checked.

## Outputs

- `submission_selected.csv`, `submission_v4_reference.csv`, provenance and status.
- Model bundles, exact training manifests, dev/audit predictions and test ranks.
- `selection_lock.json`, screening candidates, admission decisions and pilot records.
- `scheduler_events.jsonl`: concurrency, timings, peaks, estimates and fallback events.
- Three new PNG/PDF figures and CSV sources in `report/`; previous six main figures
  and earlier evidence remain in `reference_v4/report/`.
- Source code, version metadata and a complete resumable results ZIP.

Build: `python3 src/ablation_v5/build_notebook.py`.
Test with an environment containing XGBoost 3.0.5 and the scientific dependencies:
`python src/ablation_v5/test_v5.py`.

### Resuming after `KeyError: 'iterations'` (2026-10-21)

Use the updated notebook. The repair affects pilot resource estimation only; it
preserves training settings and completed fits. Original v5 checkpoints are
recognized through exact source hashes and upgraded with an audit record.
Download the last completed checkpoint before resetting a runtime, restore it into
an empty results directory in the updated notebook, then run all cells in order.
For an intact results directory use `RESTORE_ZIP = ""` and rerun setup before training.
Expected message: `CHECKPOINT UPGRADE: HGB estimator fix; completed fits and scientific settings preserved.`
