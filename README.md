# SC4000 — Predict AI Model Runtime

The frozen final system scored **0.567** on Kaggle (user-reported). It combines
collection-specific boosted trees and GNN rankers. Experiments are complete;
the final notebook presents the evidence and reproduces the submitted predictions.

## Start here

- **[Final notebook](notebooks/SC4000_Final.ipynb)** — methodology, preprocessing,
  ablations, selected models, limitations and saved figure outputs. Run all cells
  to regenerate the report assets and exact submission; no training sweep is needed.
- **[Final submission CSV](artifacts/final/submission_final.csv)** — the frozen
  894-row file for Kaggle. This is the selected submission, renamed for handoff.
- **[Report figures and tables](reports/final/README.md)** — ten PNG/PDF figure pairs,
  supporting CSVs and captions, ready to use in the report.
- **[Run and verification guide](src/final_study/README.md)** — dependencies,
  optional model-archive verification and replay limitations.
- **[Freeze decision](docs/evidence/final_freeze_decision.md)** and
  **[completed findings](docs/evidence/v5_completed_findings.md)** — why this system
  was selected and what the results establish.

The final notebook replays saved predictions; it does **not** train models or run
fresh inference by default. It needs NumPy, pandas and Matplotlib, with no raw
competition data. Its embedded evidence pack contains no model weights. Preserve
the complete checkpoint separately for model verification or future inference.
Local audit metrics and the reported Kaggle score are different evidence sources.

## Repository map

| Location | Contents |
|---|---|
| [`notebooks/`](notebooks/README.md) | Final deliverable and historical experiment notebooks |
| [`src/`](src/README.md) | Notebook builders, study implementations and checks |
| [`artifacts/final/`](artifacts/final/README.md) | Frozen CSV, compact evidence archive and hash manifest |
| [`reports/final/`](reports/final/README.md) | Curated final figures, tables and captions |
| [`docs/`](docs/README.md) | Decisions, research, methodology drafts and reference papers |
| [`archive/legacy/`](archive/legacy/README.md) | Earlier approaches retained for provenance |
| `local/` | Downloaded checkpoints, output copies, snapshots and scratch files; ignored by Git |
| `data/` | Raw competition data; ignored by Git |

On this workstation the complete final checkpoint is
`local/results/sc4000_gnn_v5_results (11)/`. Keep an independent backup: Git does
not store it. Paths and model identifiers **inside** checkpoint archives retain
their original names so resume and provenance checks remain valid.

## Rebuild or check

From the repository root, in a Python environment with the scientific dependencies:

```sh
python src/final_study/test_final.py
python src/final_study/build_notebook.py
```

The builder regenerates the final notebook and clears its saved outputs. Running
the checked-in notebook is sufficient for report replay; rebuilding is optional.
Study-specific dependencies and checks are documented beside each experiment in
`src/`. The experimental notebooks embed their runtime code for Colab portability.

For the course handoff, write one coherent methodology and results narrative.
Use the figure captions to distinguish supported gains, inconclusive ablations
and resource exclusions. Retain the exact Kaggle score/rank screenshot and team
contributions alongside the report. See the [organization record](docs/repository_organization.md)
for the old-to-new path map.
