# V5 validation

20 October 2026.

## Executed

Seven tests passed in **35.673 seconds** using `/opt/anaconda3/bin/python`
and XGBoost **3.0.5**, installed without dependencies into a temporary directory.
Worker process inspection required execution outside the macOS filesystem sandbox.

- Actual pairwise XGBoost training: correct lower-is-faster prediction direction,
  positive synthetic ordering quality, and rejection of unsorted query groups.
- Admission accounting: current memory plus unallocated worker reservations;
  refusal of two one-thread workers on a one-core allocation.
- Actual two-process execution with injected memory pressure: a worker was stopped,
  retried sequentially and completed; already successful tasks were skipped.
- Handled interruption: workers were reaped, in-flight artifacts excluded from the
  ZIP, and interrupted jobs could run again.
- Confirmation control selection and rejection of positive average gains with a
  regressing fold under the default guard.
- Full synthetic study with real HGB/XGBoost fits, pilots, confirmation, audit,
  prediction, CSV/provenance export, and figures. A forced permissive selection
  threshold was used **only in this test** to exercise the export path.
  The unchanged tile row was preserved; layout output was a valid permutation.
  Restoring into a different directory preserved model-result hashes and reused
  completed training/prediction jobs.
- Notebook schema validation, parsing of all code cells, and exact embedded-source
  matching for the v5 modules.

The synthetic run used its installed scientific package versions with
`enforce_versions=False`; production defaults enforce the archived versions.

## Real uploaded reference check

Imported `results/sc4000_gnn_v4_results.zip` into a temporary directory with CRC
validation. All selected reference training manifests exist for both folds and
all three seeds. Frozen `study_*` source hashes exactly match the v3 code embedded
inside the supplied v4 archive. Resolved controls:

| Collection | Selected tabular control | Configuration cap |
|---|---|---:|
| NLP default | T_deeper_hgb | 256 |
| NLP random | V4_T_gall_c384 | 384 |
| XLA default | T_full_small | 256 |
| XLA random | T_deeper_hgb | 256 |

No competition model was trained or submitted during validation.

## Limits

The tests establish workflow and selected correctness/resource contracts, not
Colab runtime, real-data XGBoost peak RAM, leaderboard gains, or guaranteed OOM
prevention. Full-row Colab pilots determine admission and estimates. Actual
parallel speedup depends on CPU allocation and contention.
The test suite uses tabular reference fixtures; saved-GNN forward inference is
delegated to the unchanged v4 worker, whose GNN path was validated in v4.
