# Final notebook: raw-data reproduction and coherent experiment narrative

The user requested a final notebook that is independently reproducible, with
load-or-train behavior and a clear explanation of the experiments. This supersedes
the earlier evidence-replay-only presentation plan. The selected model recipes
and historical 0.567 submission remain frozen; no new search or leaderboard claim
is introduced.

## Narrative

1. Problem, five collections, ranking metrics and CPU/RAM constraints.
2. Preprocessing and the explicitly defined basic HGB baseline.
3. Model-family exploration: MLP, engineered-feature boosting and GraphSAGE.
4. Controlled representation, graph/data-budget and target/objective experiments.
5. Complementarity, development selection, fixed blends and improvement measures.
6. Reproducible fitting, evaluation, prediction and submission export.
7. Historical versus current-run results, limitations and learning diagnostics.

Six main figures each answer a question. Historical MLP results are shown in a
separate table because their protocol differs. Gated GraphSAGE implementation was
found in `graph_improvement` at `4cd4356`, but no verified executed result was
recovered. It is not labelled a successful/failed measured GAT experiment.

Ablation tables show the actual control, candidate, before/after scores, absolute
change and relative percentage where the reference is at least 0.1. Tau changes
near zero are not presented as inflated percentages. Model additions, removal
controls, whole-system gains and reported Kaggle progression are distinct evidence.
The final recipe contains masks/context, but the displayed confirmed standalone
comparisons remain inconclusive. Larger graph/data budgets do not universally help.
Percentile targets improve selected blends; no XGBoost model is selected.

## Reproduction contract

The standalone notebook embeds the original scientific implementations, 16 unique
raw-data split manifests and 450 recorded successful fit specifications. The compact
reproduction pack stores no fitted weights, raw arrays or runtime labels. Its
training IDs reproduce the recorded sampling decisions from ordinary competition
data. Original module source hashes and trained-model metadata remain inspectable.

The default plan is 90 fits (60 selected plus 30 baseline), not the full search.
Compatible model receipts or an optional known historical archive avoid fitting.
The source-model archive is not required. New fresh predictions are never replaced
by the historical CSV. Runtime environment, input identities and model bytes are
recorded; byte identity with the historical CSV is reported, not forced.

The optional ablation switch includes 286 fits total with seed 42, or 450 with all
three recorded seeds. It reuses the default models and never changes the locked
blend. It covers recorded successful later comparisons; resource-excluded jobs and
early MLP/gated experiments remain historical context. Newly computed individual
fit/per-graph tables must be compared with matched protocols and seeds.

Workers run sequentially with memory/disk/session guards. Feature extraction uses
one bounded scratch group at a time and removes it after use. Completed fits and
test graphs resume; an interrupted fit restarts. A final-run backup is independent
of the optional historical model archive. There is no promise that a cold run
finishes in a single Colab session or reproduces exactly 0.567 on other platforms.

## Validation scope

- All 450 training selections reconstructed exactly from their recipes/manifests.
- All 450 archived fit artifact sets verified against their stored hashes.
- The 887 raw files needed by the default plan matched their archived NPZ identities.
- Original scientific feature/GNN source modules remain byte-identical.
- The notebook's 30 cells executed in review mode, with labelled historical outputs;
  its source default remains `RUN_MODE="reproduce"`.
- Real synthetic HGB, percentile-HGB and GNN training validated cold execution,
  warm reuse, missing-model repair, archive-only model reuse and fresh prediction.
- Optional synthetic XGBoost regression/pairwise controls passed: enabling
  ablations reused the completed final model plan and preserved its submission.
- Raw-data mutation is rejected before fitting; submission ordering and short-Tile
  template padding are checked.

No full competition retraining was performed locally for this notebook overhaul.
All seven automated tests passed (four archive-integrity tests and three
reproduction tests, including the synthetic end-to-end workflow).
Synthetic tests establish executable workflow contracts, not a new public score,
a real-data runtime estimate or guaranteed OOM prevention.
