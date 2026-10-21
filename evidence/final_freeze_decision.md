# Final freeze and notebook decision

Decision on 2026-10-21: stop further experiments at the user-reported public score
0.567. Preserve the selected submission rather than start another sweep toward
0.590. The evidence for this choice is in `v5_completed_findings.md`.

Overhaul `Eugene/SC4000_Final.ipynb` on `baseline`; its former recipe was obsolete.
Keep exploration notebooks and their execution outputs as provenance. Present one
final method, ablations, selected blends, constraints and conclusions. No merge to
`main` or rebase is needed before this work. Merge the reviewed final state into
`main` later; do not rewrite the shared baseline history merely to tidy it.

The final notebook is a self-contained evidence/prediction replay. It explains
preprocessing and model training, regenerates report figures and exports exactly
the frozen CSV. It does not retrain or run fresh saved-model inference by default.
The compact embedded pack does not contain model weights. Keep the full checkpoint
separately; sixty selected weights are recorded by path and SHA-256 and may be
verified through an optional archive setting.

Freeze identity: submission SHA-256
`dc037f378ddb9d0892c283a26a7f31170ac6cc01e937e4d14f2847296ff80312`.
894 submission rows. Final score is user-reported; exact leaderboard rank and
screenshots remain a human report input, not a value inferred from the score.

Report constraints from the supplied course brief: AAAI template, at most ten
pages including references, report + presentation assessed, CSV/notebook used to
check reported results. No additional full training sweep is mandated. Separate
local sampled audit metrics from the public score; prior audit reuse and small
XLA audit sets must be disclosed. Standard model/objective methods are not claimed
as inventions. Referenced participant ideas/code must be acknowledged accurately.
