# CPU GNN development

`dev` starts from `eugene-graph`. `graph.ipynb` still targets only
`layout:xla:default` and `layout:xla:random`. The main notebook is the reference
baseline; this change does not claim improved Kaggle performance.

## What changed

- Stream `node_config_feat.npy` out of each compressed source once, then use a
  read-only memory map. Batches no longer decompress the whole configuration
  array. Initial extraction needs enough disk space for uncompressed arrays.
- Keep prepared static graphs in a bounded CPU LRU cache (256 MiB by default).
  The budget excludes disk-backed configuration pages and the current batch/model.
- Copy only the requested configurations and retained configurable nodes into
  each batch. Cache and tensor construction preserve the old model inputs.
- Use 16 training configurations per batch, 32 for prediction, and at most four
  CPU threads by default. Benchmark these settings on the actual CPU; they are
  starting points, not universal optima. Keep batch size 4 for strict comparisons
  of the loading changes with the old training setup.
- Log startup, graph lookup, batch construction, model step, training and
  validation times. Save the best validation weights and restore them after
  training; stop after five validation checks without improvement.
- Sample validation configurations uniformly without consulting labels. This
  diagnostic is deliberately different from the previous runtime-stratified
  sample. Prediction ties contribute zero to sampled concordance. Rerun the
  baseline on identical validation IDs before comparing scores. Use the official
  full-configuration metric for final model selection.
- Remove runtime MAE from the GNN report: its pairwise scores have no calibrated
  runtime scale. Handle an empty configurable-node pooling mask with zeros.

## Run

Open `Eugene/graph.ipynb` from the `dev` branch in Colab, choose a CPU runtime,
and run the cells. The notebook is self-contained; it needs the competition data
and the existing Colab Kaggle secrets for the download cell. With local data,
skip the Colab download cell. Python, PyTorch, NumPy, pandas, scikit-learn and
tqdm are required. The notebook installs missing small dependencies, but PyTorch
must already be installed.

The current experiment uses up to 24 family-stratified training graphs, 64
configurations per graph, 20 epochs, validation every epoch, and patience of five
validation checks. Inspect `gnn_runs/<collection>/history.csv` and
`best.pt` under the working directory. These filenames are replaced on another
run for that collection: copy a run's outputs or change `GNN_OUTPUT_DIR` before
running another experiment. Disk cache keys include source path, size and mtime.
Delete `.gnn_cache` if you replace source contents without changing those fields.

The experiment prints available and selected graph counts per filename-inferred
family. The five validation files and seeded uniform configuration samples match
the previous runs. `validation_manifest.json` records exact IDs and source identity;
`training_manifest.json` and `training_coverage.csv` record training coverage.
Selection uses training filenames only. The limit is capped by available files.

The last cell evaluates saved main ensemble members if
`models/ensemble_members_by_collection.joblib` exists, or if
`BASELINE_ARTIFACT_PATH` points to that trusted file. Those fitted artifacts are
not in the repository. With no saved artifact, the notebook trains two labelled
HGB reference models (simple summary and WL fingerprint) using this repository's
original feature functions and HGB settings, on the same training files and
configuration IDs as the GNN. These references do not reproduce the top-17%
submission. An explicitly requested missing/incompatible artifact fails visibly.
Saved rank ensembles are averaged within the common validation sample; full-set
rank averaging can differ. Final decisions need full-configuration evaluation.

`comparison_summary.csv` reports GNN and baseline/reference scores;
`comparison_by_graph.csv` shows family-specific performance;
`comparison_predictions.csv` saves aligned predictions and configuration IDs.
Send the combined `gnn_runs/comparison_summary.csv`, the two `history.csv` files,
and both `training_coverage.csv` files after running. Five validation graphs and
checkpoint selection on those same graphs make this a preliminary comparison.
No submission is modified automatically. The other three collections remain in
the main notebook.

Configurable nodes are always retained, so graphs with
more than 512 such nodes exceed the neighbourhood target. Lower batch size if
needed for RAM. The cache is local to the runtime and may need rebuilding after
a Colab session ends.

## Verification

Run `python Eugene/test_graph_pipeline.py` from the repository root. It exercises
the notebook definitions directly using a synthetic compressed graph. It checks
exact equality of old/cached input tensors, predictions and gradients; cache
reuse and zero-capacity eviction; label-independent validation sampling; finite
two-epoch CPU training; saved best-checkpoint restoration; prediction; deterministic
family coverage; exact feature values under batching; fixed validation IDs;
reference-model comparisons; and loaded ensemble-member comparisons.

In the development environment, 32 batches of synthetic input construction took
0.242 seconds through repeated compressed reads and 0.001 seconds through the
warm cache. This excludes extraction, forward/backward computation and validation;
it is **not an end-to-end training speedup estimate**. Actual competition training
and leaderboard improvement have not been measured.

The changes were developed from this repository's pipeline. No competitor code,
model weights or predictions were imported. Further model experiments should
record and cite any externally inspired ideas separately.
