# V3 overnight checkpoint comparison — 19 October 2026

The user identified `sc4000_gnn_v3_results_old` as the checkpoint from before the
overnight run and `sc4000_gnn_v3_results (24)` as the checkpoint downloaded afterward.
Both are extracted directories now stored under the project’s `local/results/` folder. This investigation reads those
artifacts; it does not change the notebook, training implementation or checkpoints.

## Finding

The overnight run recomputed previously completed work. It was not merely spending
nine additional hours on new XLA-default experiments.

| Evidence | Before (`old`) | After (`24`) |
|---|---:|---:|
| Completed training fits | 135 | 147 |
| Other recorded training status | 1 running | 1 session-budget stop |
| Completed holdout evaluations | 0 | 0 |
| Controller error file | Absent | Absent |
| Embedded checkpoint timestamp (UTC) | 2026-10-18 10:53:56 | 2026-10-19 00:39:49 |

The timestamp is the value stored in `last_checkpoint.json`; that file is written
after ZIP creation, so it can describe the preceding archive rather than the exact
ZIP containing it. It is not proof of browser download completion.

All 135 previously completed trial paths still exist afterward, but **none of their
completion records are unchanged**. Every corresponding `result.json` has new
`fit_seconds` and `total_seconds` values. All other result fields are identical.
The same jobs account for **6.464 hours of worker wall time** in the after snapshot,
versus 6.666 hours in the before snapshot. This excludes preparation and other work.

Example: `layout_xla_default/folds/0/trials/G_full_n2048/42` was already complete
after seven epochs. Its fit time changes from 98.506 seconds to 93.856 seconds,
and its after-run log starts at epoch 1. Its scores and selected epoch are identical,
consistent with repeating the deterministic training recipe.

Only 12 additional fits are complete afterward. The full XLA-default confirmation
plan has 115 of 120 fits complete; five remain. Initial screening also includes fits
outside the confirmation set, explaining the total of 147 completed fits.

## What was checked

- Code/source hashes, package versions, configuration and scientific identity are
  identical across snapshots.
- Archived worker jobs use the same `/content/sc4000_gnn_v3/...` output paths.
- A read-only comparison asserting preservation of previously completed worker
  records failed: 135 changed, zero preserved. No completed paths were missing.
- The actual `execute()` implementation archived in `old` was invoked against all
  135 completed status files with subprocess creation replaced by a failing guard.
  It skipped all 135, launched zero workers and changed zero status files.
- The controller also checks each completed trial marker before feature preparation
  and worker launch. Cleanup removes marked scratch features, not trial checkpoints.

## Interpretation and unresolved cause

The artifacts establish repeated training. They do not establish which notebook
interaction caused it. The leading explanation is that the completed results were
not restored into the active output directory before the overnight study began.
Possibilities include leaving `RESTORE_ZIP` empty, not executing the restore cell,
or manually extracting results outside the active output directory. These remain
hypotheses, not findings about what the user did.

The matching run identity proves matching code/settings, not shared execution history.
A single `status: ok` file also cannot prove successful restoration: a fresh run can
produce a new successful status at the same path. Earlier conversational reassurance
based on that evidence alone was too strong.

## Recovery guidance

Use the most recent valid checkpoint, or preserve the current Colab results if they
have advanced beyond it. Do not restore `old` over a newer run.

For a fresh runtime, upload the ZIP, set its exact `RESTORE_ZIP` path and execute the
restore cell before training. Confirm its `Restored: ...` output and saved completed
jobs under the active `/content/sc4000_gnn_v3` folder. Merely uploading a ZIP does not
extract it or enable restoration.

For resumption from `(24)`, the expected first unfinished fit is
`G_reference_n2048`, fold 1, seed 2026, whose saved checkpoint contains epoch 9.
The user subsequently reported that exact first training job after preparation.
That is consistent with using `(24)`; checking that the new worker log begins at
epoch 10 provides direct evidence of epoch continuation.

No training-code changes were made in this diagnosis.

## Subsequent checkpoint `(38)`: successful continuation verified

The user supplied `local/results/sc4000_gnn_v3_results (38)` after resuming from `(24)`.
Comparison verifies actual preservation, rather than inferring resume from filenames:

- All 147 completed training status files from `(24)` are byte-identical in `(38)`.
- All 147 corresponding `result.json` files are also byte-identical.
- Exactly the five outstanding fits completed, bringing XLA-default training to 152
  successful fits with no non-success training status remaining.
- The interrupted `G_reference_n2048`, fold 1, seed 2026 worker log starts at epoch
  10 and continues through epoch 12, directly verifying continuation after the saved
  epoch 9 rather than restarting from epoch 1.
- Holdout evaluation has 42 successful jobs and one running job, out of 120 required
  evaluations. XLA default has not yet written its collection-complete marker.
- The locked development-selected blend is 50% `T_full_small` and 50%
  `G_data_gall_c256`. This is a selection result, not a measured holdout improvement.
- The other collections remain pending. Submission CSVs have not been generated.
- Scientific identity remains unchanged, and no controller error file is present.

The embedded `last_checkpoint.json` retains the earlier timestamp in this snapshot,
consistent with that marker being updated after archive creation. Actual trial and
evaluation contents establish that `(38)` contains newer progress.
