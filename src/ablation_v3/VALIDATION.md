# V3 implementation validation — 18 October 2026

This records software checks, not a completed competition experiment. No v3 leaderboard
score, improvement over v2 or Colab completion-time estimate has been measured.

## Automated checks

Final result: **10 tests passed in 217.219 seconds**, including the complete workflow and
restore test with the final factor-comparison tables enabled.

The test suite in [test_v3.py](test_v3.py) covers:

- Notebook schema, compilation and byte-for-byte equality of all six embedded modules.
- Separate graph-coverage and training-data budgets, with nested training configuration IDs.
- Disjoint development graphs across folds, train/development separation within each fold,
  label-independent candidate selection and source-change detection.
- Sampled node/edge caps, nested configurable-node prefixes and connected sampling.
- ZIP restoration, traversal rejection, nonempty-destination rejection and preserving an old
  archive when there is insufficient disk for replacement.
- Development-only selection, correct fold-specific prediction alignment and baseline fallback.
- Explicit factor contrasts with paired graph/seed comparisons and known numeric gains.
- Live-worker checkpoint heartbeats, resuming `running` jobs and skipping completed jobs.
- End-to-end synthetic XLA layout and tile workflows: two folds, coverage selection, data
  grid, matched HGB/GNN training IDs, confirmation, audit, three candidate submission files,
  inference batch invariance, ZIP restoration and scientific-identity rejection.
- Restoring only an active GNN's `latest.pt`: the continued best model exactly matches an
  uninterrupted two-epoch run, including when its separate best-model file is missing.

The integration fixture uses one seed, small graphs and two epochs to check workflow
correctness. It does not execute the full default three-seed competition study.

## Actual competition data checks

The v3 submission preflight passed all **894 rows** in the local competition template,
including 16 short tile rows and two layout rows requiring the template's 1,000-entry limit.
Actual manifests contain the following graph counts per fold:

| Collection | Training pool | Development | Audit |
|---|---:|---:|---:|
| XLA layout default | 49 | 12 | 7 |
| XLA layout random | 57 | 12 | 7 |
| NLP layout default | 128 | 12 | 20 |
| NLP layout random | 128 | 12 | 20 |
| Tile | 320 | 12 | 24 |

These are available pools; individual experiments use their specified subsets. Therefore
"all graphs" in the XLA training-data grid means 49 or 57 graphs, respectively, in this data.

A real large XLA training graph, `inference_mlperf_ssd_1200_batch_1`, was also exercised
with three optimizer steps, a full batch of 16 configurations and dummy probe targets:

| Node cap | Retained nodes / 25,544 | Configurable nodes / 3,143 | Retained edges / 33,522 | Isolated fraction |
|---|---:|---:|---:|---:|
| 512 | 512 | 384 | 132 | 0.5117 |
| 4,096 | 4,096 | 3,072 | 1,390 | 0.4304 |

This verifies that increased caps can retain more structure on real data. It does not
establish an accuracy gain or represent median coverage across the collection. Both samples
still omit substantial structure; the notebook records coverage and isolation for the run.

## Local environment and limits

Checks used macOS, scientific packages from `/opt/anaconda3/bin/python`, and PyTorch 2.14.1
installed in an isolated temporary directory. The local real-graph probe with two Torch CPU
threads exited with native signal 11; rerunning with one CPU thread passed. Synthetic
integration checks and the successful real probes therefore used one thread. The target
Colab Linux default is two threads and has not been validated here on a full real-data run.

Process watchdog tests need permission to enumerate their child processes on this macOS
host. Browser download delivery cannot be validated by these local tests. Users must verify
that ZIPs actually appear on their computer; there is no Drive backup.
