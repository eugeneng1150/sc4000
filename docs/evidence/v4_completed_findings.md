# Focused scaling study: completed findings

Reviewed 20 October 2026. This is supporting evidence, not the final course report.

## Source and completion

Source: [uploaded v4 archive](../../local/results/sc4000_gnn_v4_results.zip).
The ZIP passes its CRC integrity check and contains 9,779 entries. Its
`last_checkpoint.json` records `2026-10-20T07:39:58Z`.
`run_state.json` marks all five collections and the study complete.
`progress.json` records 24 completed new fits; 44 was the upper bound before
screening and resource exclusions, not a required total.
`submission_status.json` records 894 submission rows.

The user reports submitting `submission_selected.csv` and receiving **0.558**
on Kaggle. This score is user-reported, not independently verified. The earlier
v3 score was supplied approximately as 0.55, so an exact leaderboard increment
or percentage improvement cannot be calculated from the available precision.
The distance from 0.558 to the user's target of 0.590 is 0.032; the current
top-10% cutoff has not been rechecked.

## What actually changed in the submission

Direct CSV comparison finds **17 changed rows, all in `layout:nlp:random`**.
The other 877 rows are unchanged. The archive's `submission_v3_reference.csv`
is byte-identical to checkpoint 65's selected submission.
The v4 `submission_selected.csv` and `submission_tabular_update_only.csv`
are byte-identical in this run; submitting both would duplicate predictions.

Selected CSV SHA-256:
`2f2fd4fe46502e7332abea7e43792393f93aeee0f255155f7c6b5a2e893245b9`

## Development selection

Sources inside the ZIP: each collection's `selection_lock.json` and
`screening_selection.json`.

| Collection | Result | Interpretation |
|---|---|---|
| NLP random | Replaced `T_deeper_hgb` with `V4_T_gall_c384` | Mean development tau 0.707918 → 0.724171, gain +0.016253. Both folds improved: +0.018909 and +0.013597. |
| NLP default | Retained `T_deeper_hgb` | The expanded-data candidate improved the confirmed mean by +0.008408, but fold 1 declined by 0.000572. It failed the predeclared zero-fold-regression rule. |
| Tile | Retained `T_deeper_hgb` | Seed-42 screening at 640 graphs underperformed the incumbent at both 256 and 384 configurations. Larger settings were resource-excluded. |
| XLA default | Retained 50% `T_full_small` + 50% `G_data_gall_c256` | The longer-training candidates did not pass screening. A 50/50 blend with the longer GNN gained +0.002216 on average at seed 42 but failed the fold guard. |
| XLA random | Retained 25% `T_deeper_hgb` + 75% `G_data_gall_c128` | The 25/75 blend using the longer GNN lost 0.017651 at seed-42 screening. No new candidate was promoted. |

The printed `+0.00000` values describe retained selections compared with
themselves. They do not mean all candidate scores were identical or that every
candidate had a nonpositive average gain. The NLP-default outcome illustrates
the conservatism of the selection rule. Do not relax that rule retrospectively
and present the changed selection as part of the original protocol.

The confirmed new NLP HGB fits produced identical aggregate development scores
across the three seeds within each fold. These repetitions should not be
presented as three independent demonstrations of stochastic robustness.

## Audit evidence

Source: `layout_nlp_random/system_audit_gains.csv`.
The selected NLP-random model improves mean audit tau from **0.646578 to
0.672493**, a paired gain of **+0.025915** (about 4.01% relative to this local
reference score). It improves 17 of 20 audit graphs. The exploratory
graph-bootstrap interval is **[0.012170, 0.042647]**, without correction for
multiple comparisons. This is local audit evidence, not a Kaggle improvement
estimate. The official validation set was already inspected in earlier work;
it is not a newly untouched test set.

For NLP default, the unselected candidate's audit gain is +0.012258 with an
interval [-0.001465, 0.030434]. This does not override the development selection.
The remaining selected systems have zero audit change because they were reused.

## Limits of the scaling experiment

Tile 1,280-graph preparation reached a measured pilot peak of **4.787 GiB**,
above the admission threshold of 85% of the 5 GiB worker cap (4.25 GiB).
It was excluded before full fitting. The 2,560-, 5,120- and all-training-graph
settings failed static resource bounds. These are feasibility exclusions, not
evidence that those larger models would predict poorly.

Sources: `tile_xla/admission/*/decision.json`.

## Decision and next direction

Keep the 0.558 submission as the current user-reported benchmark. This run
supports a useful NLP-random data-scaling gain, but does not establish that
scaling helps every collection or that model capacity/data are universally
exhausted. The modest overall improvement is consistent with changing only
one collection; do not equate its local tau gain with aggregate Kaggle gain.

The next controlled investigation should prioritise the representation and
target experiments in [the research note](../research/post_v4_improvement_research.md),
using this selected system as the comparator. Avoid another broad coverage
sweep or automatically extending training across every collection.

The ZIP includes six main report figures plus two appendix figures, their
source tables and captions. Its `report/kaggle_scores.json` still contains
the older rounded 0.40/0.53/0.55 values. Before final-report export, add the
user-reported v4 score as a distinct observation; the current archived figure
must not be described as already displaying 0.558. The archive itself was
left unchanged during this review.
