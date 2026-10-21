# Focused v4 continuation and report pack

Open [SC4000_V4_Focused_Scaling_and_Report.ipynb](../../notebooks/experiments/SC4000_V4_Focused_Scaling_and_Report.ipynb)
in Colab CPU. It is self-contained. The agreed plan, including six report figures,
is in [evidence/v4_plan.md](../../docs/evidence/v4_plan.md).

## First run

1. Upload your completed v3 checkpoint 65 ZIP.
2. Set `V3_REFERENCE` to its exact `/content/...zip` path; leave `RESTORE_ZIP` empty.
3. Run all cells. Use the existing competition download/authentication setup.
4. If dependency installation requests a kernel restart, restart the session
   without deleting its files, then Run all again.
5. Verify that periodic backup ZIPs actually finish downloading to your computer.

V4 first creates six PNG/PDF figures from v3 without retraining. It then runs NLP
and tile scaling before XLA, so the cheaper track is not delayed behind GNN work.
The new result directory is `sc4000_gnn_v4`, separate from v3.

## Experiments

- NLP: all eligible training graphs, preserving development graphs and the exact
  original training subset/configuration pools; 256 then 384 configurations.
- Tile: 640, 1,280, 2,560, 5,120 and all eligible graphs, subject to static matrix
  bounds and a measured HGB pilot. One 384-config comparison at the largest size
  admitted for both configuration counts. No claim that full tile data fits RAM.
- XLA: winning v3 GNN recipe, same geometry/data, 40 epochs and patience 10.
  Fresh same-seed fit; no repeat of v3 coverage/architecture sweeps. Existing
  selected GNN/tabular predictions are also tested in a small development blend grid.
- Two-fold seed-42 screening; at most one new finalist per collection gets the
  other two seeds. This is at most 44 new fits before resource/screening exclusions
  with the default plan, plus pilots, audit and inference. It is not a runtime ETA.
- Replacement requires positive predeclared mean improvement and no loss on either
  development fold against v3 selected. Otherwise retain the exact v3 test rows.
- Lock selection before auditing new finalists. No training on audit labels and
  no post-selection all-data refit that would invalidate the measured comparison.

On the supplied real-data headers, fold 0 has 186/195 eligible NLP graphs.
Static estimates admit tile 640 and 1,280 at 256 configurations (approximately
419/860 MiB feature matrices). 2,560+ exceed the 1 GiB matrix bound. These are
estimates; Colab pilots and runtime guards determine actual admission.

## Outputs

- `report/01_*` through `06_*`: six main figures in PNG/PDF, source CSVs, captions.
- `report/table_1_*`, `table_2_*`: setup/resources and findings/decisions.
- V4 appendix plots, raw fold/seed results, learning histories and audit tables.
- `submission_v3_reference.csv`: original selected submission.
- `submission_tabular_update_only.csv`: accepted NLP/tile changes, original XLA rows.
- `submission_selected.csv`: all development-accepted changes.
- `submission_provenance.csv`: origin and model weights for each row.
- New-model per-graph rank vectors, model checkpoints, manifests, resource admission
  records, logs, frozen selections and complete source code.

Kaggle chart values 0.40/0.53/0.55 are explicitly approximate user reports.
Verify the exact scores and leaderboard split before submitting the final report.
Local sampled metrics are not official leaderboard scores. Three seeds/two folds
are not six independent copies of audit graphs. Intervals are exploratory.

## Recovery and limitations

V4 ZIPs include the immutable v3 reference (about 490 MiB uncompressed for checkpoint
65); do not delete it to shrink the backup. After reset, upload the latest **v4** ZIP,
set `RESTORE_ZIP`, and run all. A populated output directory is never overwritten.
On the same VM, clear `RESTORE_ZIP` and rerun the training cell to reuse existing work.

Completed jobs explicitly print SKIP. GNN continuation prints its saved epoch.
Package versions, code, scientific settings, reference artifacts and data archive
fingerprints are checked. Dataset additions/removals also invalidate a resume.
Unexpected failures stop with a checkpoint rather than being labelled poor accuracy.
Use the saved error/log to diagnose; `retry_failed` can be enabled for a resolved
transient failure without changing the scientific experiment identity.

Limits: one worker, two threads, 5 GiB worker threshold, 2 GiB system reserve,
1 GiB feature-matrix bound, 4 GiB scratch, 8 GiB output. Scratch is cleaned on stage
changes/exit; competition NPZs and trained model files remain. Monitoring cannot
prevent every allocation spike. Session budget is eight hours, not total study time.
No Drive mount, runtime keep-alive, Kaggle upload or automatic delivery guarantee.

Build and test:

```bash
python3 src/ablation_v4/build_notebook.py
python src/ablation_v4/test_v4.py
```

The builder embeds the unchanged `ablation_v3` modules plus the four `v4_*` modules.
Changing code after a run starts intentionally invalidates its resume identity.
