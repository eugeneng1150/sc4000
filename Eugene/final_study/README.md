# Frozen final system

Use [`../SC4000_Final.ipynb`](../SC4000_Final.ipynb). Public score **0.567** is
user-reported. The notebook presents one methodology, regenerates ten figures
(PNG/PDF) and source tables, and exports the exact frozen CSV. All outputs are
saved in the checked-in notebook, so figures are visible without executing it.

## Run

Run all cells. NumPy, pandas and Matplotlib are the only scientific dependencies
for report replay. No dataset, GPU, model fitting, dependency upgrade or other
notebook is needed. The compact evidence pack is embedded and SHA-256 checked.

Output directory: `/content/sc4000_final_frozen` in Colab, otherwise
`sc4000_final_frozen` under the working directory. Outputs:

- `submission_final.csv`: **prediction replay**, byte-identical to the submitted CSV.
- `figures/01_pipeline` through `10_training_resources`: PNG and PDF.
- `tables/`: machine-readable supporting results.
- `FIGURE_CAPTIONS.md`: units, comparisons and evidence limitations.
- `sc4000_final_report_artifacts.zip`: compact report deliverables, not model weights.

Set `DOWNLOAD_OUTPUTS = True` to request the report ZIP at the end. Otherwise use
the Files sidebar. Verify the download completes. No repeated checkpoint downloads
are needed because this notebook does not train.

## Frozen submission and model archive

The checked-in [`submission_final.csv`](submission_final.csv) has 894 rows and SHA-256:

`dc037f378ddb9d0892c283a26a7f31170ac6cc01e937e4d14f2847296ff80312`

Keep the **complete v5 checkpoint** separately; do not replace it with the smaller
report ZIP. Sixty selected model files (five collections × selected members × two
folds × three seeds) are indexed with individual hashes in the evidence pack.
Optional `MODEL_ARCHIVE` accepts that complete ZIP or extracted folder and verifies
the selected weights without unpickling. The default notebook does not perform new
predictions; fresh saved-model inference requires the archived code, raw data and
matching scientific dependencies. It does not require another search sweep.

`frozen_evidence.zip` contains supporting tables, selected recipes, histories,
submission provenance and the archived implementation for inspection. It contains
**no model weights**. `freeze_manifest.json` records every embedded artifact hash.
Study versions remain in provenance IDs only; the report narrative follows model
choices and evidence rather than separate notebook versions.

## Rebuild and validation

`freeze_evidence.py` curates a completed results directory and refuses a submission
with a different expected hash. `build_notebook.py` embeds the evidence and report
code; rebuilding clears saved notebook outputs. Execute the rebuilt notebook to
save figures in its output cells. `test_final.py` checks exact replay, tamper
rejection, repeatability, notebook embedding and model-file hash verification.

Validation on 2026-10-21: the notebook was executed top-to-bottom in a fresh local
Jupyter kernel; all 23 cells completed, all 10 figure pairs were generated, and the
exported submission hash matched. All 60 actual selected model files were verified
against the local complete checkpoint. Four focused integrity tests passed.

The selected GNN curves show training loss and early-development **tau**, not
validation loss. Do not claim they prove absence of overfitting. Component mask/
context bootstrap intervals cross zero; report inconclusive evidence honestly.

## Course handoff

The supplied brief assesses the report and presentation; the CSV/notebook checks
the reported prediction result. Use the AAAI template and at most ten pages
including references. Capture the exact Kaggle score/rank screenshot and complete
team contributions and source acknowledgements before submission. No new training
sweep is required by that brief. Retain source and weights for possible checking.
