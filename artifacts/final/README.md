# Frozen artifacts

- [submission_final.csv](submission_final.csv): exact selected Kaggle submission,
  894 rows; reported public score **0.567**.
- [frozen_evidence.zip](frozen_evidence.zip): compact supporting evidence, archived
  implementation, model-file hashes and prediction replay data. **No model weights.**
- [freeze_manifest.json](freeze_manifest.json): evidence identity and artifact hashes.

Submission SHA-256:
`dc037f378ddb9d0892c283a26a7f31170ac6cc01e937e4d14f2847296ff80312`.

The [final notebook](../../notebooks/SC4000_Final.ipynb) embeds this pack
as historical evidence. Its reproduction mode generates fresh predictions
from loaded/retrained models and compares their hash with this frozen submission. Full trained models remain in the separately backed-up
checkpoint, locally `local/results/sc4000_gnn_v5_results (11)/` from the repo root.
Do not replace that checkpoint with this smaller evidence archive.

`reproduction_spec.zip` contains exact recipes, raw-data identity/split manifests,
training configuration IDs and model hashes for load-or-train reproduction. It
contains no weights or raw data and is embedded in the final notebook.
