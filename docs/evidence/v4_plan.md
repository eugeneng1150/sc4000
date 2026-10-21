# V4: focused scaling and report evidence

V3's selected submission (user-reported Kaggle score approximately 0.55) is the
benchmark. The tabular-only score was approximately 0.53; v4 preserves the winning
hybrid and does not replace it indiscriminately. Exact leaderboard scores still
need verification for the report. A score of 0.59 is a target, not a guarantee.

## Training plan

- Import completed v3 checkpoint 65 as an immutable reference. A v4 checkpoint
  includes that reference, so a fresh runtime needs only the latest v4 archive.
- Preserve v3's graph-disjoint development folds, feature definitions, configuration
  pools and seeds. Official validation is an already-inspected audit, never used
  to select replacements. Keep the v3 training graphs as a subset of larger sets.
- NLP: test the winning deeper HGB with all eligible training graphs at 256,
  then 384 configurations per graph. Development graphs remain excluded.
- Tile: test 640, 1,280, 2,560, 5,120 and all eligible training graphs at 256
  configurations. Only resource-admitted sizes run. Test 384 configurations at
  the largest size admitting both configuration counts; compare it to that same size at 256 configurations.
- XLA: reuse the winning GNN geometry/data recipe; screen one longer-training
  recipe (40 epochs, patience 10 versus v3's 20/5). Train from the same seed,
  not from a differently stopped checkpoint. This changes the training budget,
  not a single architectural component. No repeated coverage sweep.
- Screen new recipes with seed 42 on both folds. Confirm the best improving
  new recipe per collection with seeds 137 and 2026. Re-evaluate a small blend
  grid using saved development predictions. A replacement must improve the
  mean by a predeclared margin and must not reduce either fold's score relative
  to the v3 selected system. This rule does not ensure leaderboard improvement.
- Audit only new finalists after locking. Keep unchanged collection test rows
  byte-for-byte from v3. Export v3 reference, tabular-update-only, and final
  selected candidate CSVs, with source/selection records for every collection.

## Resources and recovery

Standard Colab CPU, local storage, one worker, two threads, 5 GiB worker limit,
2 GiB system reserve, bounded matrix/scratch/output, and no persistent raw-array
cache. Admit each candidate using a conservative matrix estimate, then a measured
10-iteration HGB pilot on its actual training set. Report extrapolated fit time
as a rough estimate, excluding extraction/inference, not a completion promise.
Guard every training/inference worker. Stop resource-ineligible candidates
explicitly; unexpected failures stop with a recoverable checkpoint rather than
silently becoming scientific negative results. Restore safely and print skipped
jobs, next stage and progress. Session limits pause work; they are not total-run
limits. Browser delivery must be verified by the user.

## Report pack (available before v4 training)

Six numbered figures, PNG and PDF, with source tables and captions:

1. Preprocessing/model/ensemble pipeline, including sampling points.
2. Dataset graph sizes (when competition files are available) and retained
   node/edge fractions from v3 preparation records; state coverage scope.
3. Component ablations with paired graph-bootstrap intervals from confirmed
   comparisons; clearly separate masks, context, their combination and messages.
4. Coverage-budget development scores at 512/1,024/2,048/4,096 nodes, explicitly
   labelled seed-42 screening; node, configurable-node and edge caps co-vary.
5. Training-graph/configuration grids, labelled seed-42 screening. Separate
   panels for XLA collections; missing/excluded results are not plotted as zero.
6. Final local system comparison, with layout/tile on separate axes, and a
   separate user-reported Kaggle panel. Do not average local metrics as Kaggle.

Two compact tables: experimental setup/resources; findings/decisions. Appendices
include v4 scaling curves and learning histories. Save v3 and v4 findings
separately. Fold/seed repetitions are not independent audit graphs; intervals
are exploratory, not corrected for multiple comparisons. Preserve negative and
inconclusive findings. No causal claim that every performance change comes from
one component when training budget, sample composition or model selection changes.

The ten-page report can be written from v2/v3 alone. V4 adds scaling evidence;
it is not required to make existing work reportable. This document is the plan,
not the final course report. Course page-count/reference rules require checking
when drafting the final report.
