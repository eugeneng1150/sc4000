# Comprehensive CPU GNN and ablation study

Open [SC4000_Comprehensive_Ablations.ipynb](../../notebooks/experiments/SC4000_Comprehensive_Ablations.ipynb)
in Google Colab, choose a CPU runtime and run all cells. It embeds its implementation;
no repository checkout is needed. Defaults target standard 12–13 GB RAM and local storage.

Set `DATA_ROOT`, or use the built-in Kaggle download. `SAMPLE_SUBMISSION` is autodetected
beside the data; set it explicitly if needed. Credentials belong in Colab Secrets and are
not copied into outputs. No submission is uploaded automatically.

## Experiments and references

- **B0:** fixed HGB system baseline on all five collections.
- **G0:** a two-layer, width-32 GraphSAGE reference on both XLA layout collections,
  derived from the existing project's architecture. It is a controlled reimplementation,
  not an exact historical reproduction or an intentionally weak comparator.
- **GNN additions:** explicit padding masks, global categorical context, and both.
- **GNN controls:** remove message aggregation, opcode embedding or numeric node inputs;
  replace pairwise loss with regression; reduce to one message layer.
- **Matched HGB:** exactly the same training graph/configuration IDs as the GNN.
- **Tabular additions/removals:** masks, positional and graph-associated layout features;
  tile transformations/interactions and graph features.
- **Optional exploration:** remaining tabular feature variants, weights, targets,
  duplicate handling, half-data controls, Extra Trees, MLP, ridge and linear ranking.

The default plan has **192 required fits** (64 method/collection combinations × three seeds),
including 54 GNN fits, followed by up to 85 optional tabular screening fits.
Every required method runs three seeds before development-only selection, reserved-graph
audit and full test prediction. Optional tabular screening comes last and cannot alter the
frozen selection after holdout results are available. NLP layout and tile use tabular models;
the GNN scope is the two XLA layout collections.

G0 and its feature additions have equal parameter counts. Unused mask/context channels
are zeroed. The context addition sees summary statistics over all configurable nodes;
it can recover information missed by graph sampling. Removing opcode embedding retains
opcode-aware context; removing message aggregation retains edge-aware context. These
controls test learned components, not removal of every source of that information.

Development graphs come from official train; official valid is the reserved audit for this
run. Existing project work may already have inspected official valid. Within a collection,
graph names are disjoint; related model families can cross splits. Configuration IDs are
label-independent and shared at evaluation. Three seeds vary training configuration
samples and randomness, not graph splits. IDs and runtime are not predictive features.

Layout scores are exact tau-b on the fixed sampled candidate sets. Tile scores use normalized
runtime and top-five regret relative to sampled candidates. These are diagnostics, not the
competition leaderboard score. Final CSVs rank all test candidates using the same fitted
models that were audited; no holdout retraining occurs.

## Report evidence

The ZIP includes:

- `holdout_paired_gains.csv`: all required methods compared with B0.
- `holdout_paired_vs_G0_reference.csv`: GNN additions compared with G0.
- `holdout_paired_vs_G_full.csv`: removal controls compared with the full GNN.
- `holdout_paired_vs_B_matched_hgb.csv`: comparison with the matched-data HGB.
- `locked_systems_holdout_scores.csv`: baseline and selected system ensembles.
- Training histories, checkpoints, training IDs, per-graph scores and sampled predictions.
- Graph/configuration manifests, retained-node/edge coverage, timings, worker peak RAM,
  failures, model definitions, package versions and code fingerprints.
- `submission_baseline.csv` and `submission_selected.csv` when inference completes.

Paired tables include absolute changes, score points and relative percentages when the
reference is positive and above 0.01. Prefer “+0.04 tau” to ambiguous “4% more accurate.”
Graph-bootstrap intervals average seeds within graph and are exploratory, without multiple
comparison correction. Negative or uncertain results are retained; gains are not promised.

## Memory and recovery

One isolated worker runs at a time, with two CPU threads, a 5 GiB RSS limit and 2 GiB system
reserve. GNN graphs have hard caps of 512 nodes, 384 configurable nodes and 4,096 edges.
Training defaults to 24 graphs × 64 configurations, batch 16, at most 20 epochs, patience 5.
Inference batches GNN configurations in groups of 32. These bounds reduce risk but cannot
guarantee protection against abrupt allocation spikes, OS kills or Colab runtime deletion.

NPZ configuration rows stream in ascending order. No expanded raw-array cache is created.
Only bounded sampled feature tables are reused, then removed from this study's marked
scratch directory. Source data, saved models and results are retained. Static arrays,
training matrices, scratch storage and disk reserves have checks; excluded graphs are
excluded consistently across methods and excessive exclusions stop a collection.

The nine-hour session budget exports partial results if reached. Completed jobs are reused
on rerun; GNN checkpoints save optimizer/model/RNG state each epoch. Failed/resource-stopped
jobs are logged; `retry_failed=True` permits explicit retry after adjusting resource limits.
A session-budget stop is automatically resumable, as is a GNN timeout with an epoch checkpoint. Scientific changes require a new run name.

Local files do not survive VM deletion. Download the checkpoint ZIP from the Colab Files
sidebar to preserve it. `RESTORE_ZIP` restores your own trusted output into an empty result
directory; code, scientific settings, dependencies and source data must match. Features may
need rebuilding. The notebook cannot restore files that were never downloaded.

Short tile rows preserve the original template suffix only when the supplied template uses
the expected sequential five-slot pattern. All real candidates are ranked first. Those rows
are counted in `submission_status.json`; no unknown placeholder convention is invented.
Two actual layout files contain 1,001 candidates while the template requests 1,000 ranks.
Every candidate is scored, then the best template-sized set is exported. The validator
requires a full permutation when the template requests all candidates.

## Maintain and validate

Edit `engine.py`, `gnn.py`, `submission.py` or `controller.py`, then regenerate:

```bash
python3 src/ablation_v2/build_notebook.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python src/ablation_v2/test_study.py
```

Tests require NumPy, pandas, SciPy, sklearn, joblib, psutil, PyTorch and nbformat. They cover
streaming, masking, edge direction, normalization, graph/configuration splits, hard graph
bounds, matched training samples, checkpoint recovery, identical compact/streamed inference,
submission validation, resource watchdogs, full synthetic studies and resume. Synthetic
checks do not measure competition accuracy or prove the real default sweep fits one session.
