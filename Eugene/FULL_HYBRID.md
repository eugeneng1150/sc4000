# Full baseline + GNN submission workflow

Open `SC4000_Full_Hybrid.ipynb` on the `dev` branch in Colab and run top to bottom
on CPU. Data download is skipped when a supported data directory already exists.
The competition sample submission is required to define IDs and output lengths.

The notebook retrains the main-derived five-collection baseline, including its
experiment selection and final refits. It saves the fitted members, then trains
GNNs only for `layout:xla:default` and `layout:xla:random`. CPU data caching is
shared with the standalone graph notebook. Full baseline training costs more
than the preceding small GNN/reference experiments; no unmeasured runtime is promised.

GNN recipes follow the best previously observed diagnostics: 24 family-stratified
training files for default; the original first eight files for random; 64 training
configurations per graph; up to 20 epochs with validation each epoch and patience
five. These settings were selected using the same five validation graphs, so their
reported scores are preliminary. A comparison against the newly fitted baseline
uses identical validation IDs and the same sampled concordance metric. Saved
ensemble ranks are averaged within that sample, not over the entire configuration
space. Use broader validation and official scoring before claiming a reliable gain.

Outputs in `full_run/`:

- `submission_baseline.csv`: five-collection baseline rebuilt in this run.
- `submission_hybrid.csv`: both XLA layout collections replaced by GNN predictions.
- `validation_comparison.csv`: common-sample GNN/baseline comparison.
- `models/ensemble_members_by_collection.joblib`: keep these fitted baseline members.
- `gnn_runs/`: best GNN checkpoints, training histories, coverage and aligned predictions.
- `run_metadata.json` and `submission_checks.json`: package versions, settings and checks.

The hybrid always swaps both requested collections and prints a message if a GNN
fails to beat the rebuilt baseline on the diagnostic sample. Every untargeted
row must equal the corresponding baseline row exactly. Export checks enforce
template ID order, unique valid configuration indices and required lengths; no
fake indices are inserted. The notebook does not submit to Kaggle automatically.

The original top-17% artifacts were lost. The other three collections retain the
main code's recipe and the current run's baseline predictions, but their numerical
predictions are not guaranteed to match the historical submission after retraining.
Keep both CSVs and compare their Kaggle scores as an experiment.

The final cell downloads the hybrid CSV and `full_run_backup.zip` in Colab. If
browser download permissions block multiple downloads, download them from the
Colab file sidebar. Download the backup before ending the runtime. Existing
`full_run/` outputs are overwritten if the notebook is rerun there.

Verification: `python Eugene/test_full_hybrid.py` exercises all five baseline fits,
both GNN fits, saving/loading artifacts, common-sample comparison and submission
export on tiny synthetic data with reduced iteration counts. It checks that the
other three collections are unchanged and rejects duplicate indices. It does not
measure competition accuracy. No competitor code, weights or predictions are used.
