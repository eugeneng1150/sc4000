# Completed target and ranking study: evidence and stopping decision

Assessment date: 2026-10-21. Working evidence, not the final report.

## Sources and completion

Inspected `local/results/sc4000_gnn_v5_results (11)/` and the executed notebook on
`origin/baseline` at commit `8aa8132`. Their displayed selections and audit scores
agree. Run state and all four layout collections are complete; 36 new fits have
successful statuses, versus a pre-screening upper bound of 56. Tile is carried
forward. All 36 new training manifests match the corresponding incumbent training
examples, including graph and configuration IDs. Submission has 894 rows with
identical ordered IDs to the v4 reference. Exactly 42 rankings changed: 17 NLP
default, 17 NLP random, 8 XLA default; all other rows are identical.

Selected CSV SHA-256:
`dc037f378ddb9d0892c283a26a7f31170ac6cc01e937e4d14f2847296ff80312`.

The user reports public Kaggle score **0.567**, versus **0.558** for v4: +0.009
absolute, approximately +1.61% relative. These are user-reported scores, not a
fresh leaderboard verification. They are distinct from per-collection Kendall tau.

## What was selected

Weights below are the saved prediction-blend weights.

| Collection | Final components | Development tau gain vs v4 | Reused holdout audit tau, v4 → selected |
|---|---|---:|---|
| NLP default | 25% original deeper HGB + 75% percentile-target HGB | +0.066477 | 0.411263 → 0.453579 |
| NLP random | 50% all-graph HGB + 50% percentile-target HGB | +0.005016 | 0.672493 → 0.684280 |
| XLA default | 37.5% original HGB + 37.5% saved GNN + 25% percentile-target HGB | +0.029271 | 0.035175 → 0.094348 |
| XLA random | Existing 25% deeper HGB + 75% saved GNN | 0 | 0.271156 → 0.271156 |
| Tile XLA | Existing deeper HGB | Not retested | Carried forward |

All three accepted updates improve both development folds after three-seed
confirmation. Percentile-target HGB is the only new model type included in the
selected submission. GNNs remain in both XLA blends; no GNN was retrained for this
study. Both XGBoost regression and pairwise ranking were screened, but neither
entered the final selected system. That is a result for these controlled settings,
not proof that XGBoost or ranking objectives cannot work on this task.

The NLP random percentile model alone slightly loses on mean development score;
the 50/50 blend improves. The final conclusion therefore concerns complementary
predictions and target design, not universal superiority of a replacement model.

## Strength and limits of the evidence

Selected audit gains are +0.042316 (NLP default), +0.011787 (NLP random), and
+0.059173 (XLA default). Their graph bootstrap intervals are respectively
[0.022174, 0.063437], [0.006799, 0.018001], and [0.031275, 0.092724]. They improve
15/20, 18/20 and 7/7 audit graphs, respectively. However, these holdout graphs have
been examined in earlier studies; the bootstrap is exploratory and uncorrected
for multiple comparisons. XLA has only seven audit graphs. Do not call these a
new untouched test set or claim definitive significance/no overfitting.

Report absolute tau differences instead of the misleadingly large relative
percentage for XLA default, whose reference audit score is close to zero. Do not
attribute the exact public score gain to one collection: three changed together.
Some deterministic HGB fits yield identical predictions across seeds, so seed
count should not be presented as three independent data replications.

## Decision given remaining time

Freeze this selected submission and its complete checkpoint as the final system.
There is no supported promise that another sweep would close the remaining 0.023
to the user's stated 0.590 target. Prioritize the coherent final notebook and
approximately ten-page report: preprocessing and resource constraints, baseline,
feature/component ablations, graph/data scaling, target/objective comparisons,
final collection-specific blends, and limitations. The final notebook should load
saved evidence/models with no default experiment sweep. A selected-model retrain
can remain optional if course reproducibility requirements require it.

Available v5 figures are under the results folder's `report/`: target/learner
screening, selected-system development gain, and measured training resources
(PNG/PDF). Earlier figures are preserved under `reference_v4/report/`. GNN
histories contain training loss and early-development tau; do not label tau as
validation loss or claim the curves alone establish absence of overfitting.
