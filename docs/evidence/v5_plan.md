# V5 protocol: target and boosted-tree ranking controls

Date: 20 October 2026. Authorised scope: build a comprehensive continuation
notebook, with automatic resource estimates and parallel execution when feasible.
Benchmark: completed v4 selected submission, user-reported Kaggle score 0.558.
This is an experiment plan, not a report of new model gains.

## Questions

1. Does per-graph runtime percentile improve HGB over its centred-log target?
2. Does replacing HGB with histogram XGBoost regression help using the same features?
3. Within XGBoost, does pairwise ordering improve over regression?
4. Does a new tabular model improve the existing selected system, including its
   GNN contribution on XLA?

The percentile experiment follows an earlier exploratory project result.
Ranking is established prior work, implemented through the official library.
No competitor implementation, model weights or predictions are incorporated.
No dimension-aware feature extension is included in this experiment.

## Controlled design

Use the four layout collections. Keep tile unchanged. Reuse the v4 selected
tabular recipe's features and exact per-fold/seed training graph/configuration
IDs. For NLP random this is the expanded v4 dataset. Keep the two development
folds, runtime labels and evaluation candidates unchanged.

Reuse the HGB control. Fit HGB percentile plus two XGBoost arms: centred-log
regression and grouped pairwise ranking. XGBoost arms share 300 histogram
boosting rounds, depth 6, learning rate 0.04, 128 bins, full row/column sampling,
and one CPU thread. The ranking arm compares configurations only within each
graph, gives faster configurations higher relevance, preserves ties, and
negates output scores to match the existing lower-is-faster pipeline.
Equal query weights in ranking and graph-normalized sample weights in regression
are related fairness choices, not mathematically identical losses.

The HGB versus XGBoost comparison changes learner implementation/capacity.
Only the XGBoost comparison holds the tree settings fixed while changing the
learning formulation. No claim that pairwise logistic exactly optimises tau.
Primary references: [XGBoost ranking documentation](https://xgboost.readthedocs.io/en/release_3.0.0/tutorials/learning_to_rank.html)
and [official Python API](https://xgboost.readthedocs.io/en/release_3.0.0/python/python_api.html).

Screen 24 new fits: 3 arms × 2 folds × 4 collections at seed 42.
Confirm only the best eligible new arm using seeds 137 and 2026; if XGBoost wins,
confirm the matched XGBoost counterpart too. Upper bound: 56 fits, excluding
pilots, audit and prediction. This preserves the objective comparison without
confirming every unsuccessful candidate.

Evaluate a fixed .25/.5/.75/1 new-model weight against the existing selected
system. On XLA also evaluate replacement of only the tabular member while
preserving the GNN weight. Promotion requires mean development gain at least
0.002 and no regression on either fold. Freeze before audit. No retrospective
relaxation, broad hyperparameter search or training on audit labels.

## Resource scheduler

Run sequential full-row 10/30-tree pilots for each arm and fold. Estimate
fit time, peak-memory growth and serialized model growth; reserve the forecast
times 1.5 plus 0.25 GiB. Admit only within the 5 GiB worker cap.
These are conservative heuristics, not hard peak bounds.

At most two workers, one CPU thread each. Pair only with sufficient effective
CPU capacity and current available memory after running-job reservations and
the 2 GiB system reserve. Sequential extraction avoids concurrent large NPZ
decompression. All jobs in a fold share read-only compact features. Preparation
is repeated only when changing fold/stage or when scratch was lost.

Record pilot estimates, actual wall times/RSS and launch/fallback decisions.
On paired memory pressure, stop one worker and retry sequentially. On unresolved
single-worker pressure, pause/stop with diagnostic state rather than claim a
scientific negative result. CPU/memory estimates cannot ensure a speedup or
prevent every abrupt OOM.

Only committed successful artifacts enter the periodic ZIP; live worker files
are excluded. Completed work is skipped on resume; interrupted fits restart.
Runtime and scratch directories are separate from the immutable reference.
Keep local Colab storage, browser downloads, memory/disk monitoring and no Drive.

## Deliverables and interpretation

A self-contained Colab notebook, selected and reference CSVs, provenance,
checkpoints/models, development/audit predictions, timing/resource records,
three new report figures and source tables. Existing report figures stay in
the embedded v4 reference. No new GNN fitting is scheduled, although changed
XLA blends may need saved-GNN forward inference.

Adaptive development reuse limits conclusions. Audit graphs are not newly
untouched data; seed/fold repetitions are not independent graph samples.
Deterministic repeats must not be presented as independent confirmations.
Retain the 0.558 candidate when nothing qualifies. No promise of 0.59.
