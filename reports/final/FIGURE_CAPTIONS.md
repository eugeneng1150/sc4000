# Figure captions

**01_pipeline.** Training workflow that produced the frozen system. This notebook replays its evidence; it does not rerun this training pipeline.

**02_data_and_coverage.** Left: supplied training graph inventory; boxplot outliers hidden. Right: uniform-sampled audit graph coverage; edge denominator is the pre-capped context, not necessarily the original complete graph. Node/configurable-node/edge caps co-vary.

**03_component_ablations.** Confirmed component comparisons against their specified controls. Exploratory graph-bootstrap intervals, not multiplicity-corrected. All intervals shown cross zero; do not claim independently established gains for masks or context. Not a cumulative waterfall.

**04_graph_budget.** Seed-42 screening averaged over two development folds. Larger sampled graphs did not consistently improve scores. Other graph caps co-vary; missing settings are not zero.

**05_training_data.** Seed-42 data-budget screening, two-fold mean at the selected graph budget. “All eligible” excludes each fold’s development graphs. More examples also increase optimiser steps, so this is not compute-matched.

**06_target_and_objective.** Matched example IDs per fold/seed. HGB control versus percentile isolates target transformation; XGB regression versus pairwise isolates objective within its fixed learner settings. HGB versus XGB changes learner/capacity. Seed-42 screening only; no XGB model was selected.

**07_final_audit.** Previously inspected official-validation audit: 20 graphs per NLP collection, 7 per XLA layout, 24 Tile. Baseline and selected are saved ensembles. Metrics differ between panels; do not average them into a Kaggle estimate. Reuse makes this exploratory evidence, not a new untouched test.

**08_public_scores.** User-reported public scores; first three are approximate. Final 0.567 versus 0.558: +0.009 absolute (+1.61% relative). Verify exact scores and current rank with a screenshot before report submission. No inferred leaderboard percentile.

**09_gnn_learning_curves.** Actual selected GNN histories; dots mark saved selected epochs. Left: mean training batch loss (pairwise objective with regression fallback on batches lacking eligible pairs). Right: ranking score on the smaller early-stopping development subset, not validation loss or the full selection metric. Individual fold/seed curves are unsmoothed; no claim of absence of overfitting.

**10_training_resources.** Measured workers in the target/objective study. Parallel job times are not total elapsed time; extraction, inference and backups are additional. Worker RSS excludes the rest of the Colab session.
