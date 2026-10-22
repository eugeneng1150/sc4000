# Historical experiment figure captions

**01_method.** Raw-data reproduction workflow. Compatible saved models are loaded; missing models are trained before evaluation and prediction. Blend recipes remain fixed.

**02_model_families.** Historical seed-42 exploration, common development folds. Matched-data HGB and GraphSAGE share graph/configuration budgets; architecture/features differ. Basic HGB is a separate pipeline baseline. Fold disagreement is visible; this is not proof of universal GNN superiority.

**03_components.** Historical three-seed confirmed comparisons. Each row has its own named control in the source table. Bootstrap intervals cross zero: these comparisons do not establish clear standalone gains. They are not additive steps in a waterfall.

**04_coverage_and_data.** Historical seed-42 screening, averaged over two development folds. Top: node/configurable-node/edge caps change together. Bottom: graph/configuration counts also change optimizer work. Larger budgets did not consistently improve ranking; this is not a compute-matched scaling experiment.

**05_target_and_objective.** Historical matched-example screening. HGB log-runtime versus percentile changes target; XGB regression versus pairwise changes objective. HGB versus XGB also changes learner/capacity. Final blend improvements require the separate three-seed confirmation evidence.

**06_final_results.** Historical baseline-to-final comparison on reused official-validation audit graphs. Layout and Tile have different metrics. This is a whole-system comparison, not an isolated ablation; do not average the panels or interpret them as a Kaggle score.

**S1_learning_curves.** Historical selected GNNs. Training loss and early-development tau use different units. Dots mark selected epochs. No per-epoch validation loss was saved, and these curves do not prove absence of overfitting.
