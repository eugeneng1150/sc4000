# SC4000 — Predict AI Model Runtime

A collection-specific boosted-tree/GraphSAGE ensemble has a screenshot-documented
public Kaggle score of **0.56715** and private score of **0.49624** from a
post-deadline submission; no official leaderboard position is claimed.
The final notebook explains the investigation and can
**retrain the selected system from raw competition data without a model ZIP**.

## Start here

- **[Report draft](reports/final/report_draft.pdf)** — the 10-page report, including
  references. [LaTeX source](reports/final/report_draft.tex) and the
  [Overleaf bundle](reports/final/report_draft_source.zip) are included.
- **[Final notebook](notebooks/SC4000_Final.ipynb)** — baseline, model exploration,
  controlled ablations, selection, load-or-train execution and fresh submission export.
- **[Execution and recovery guide](src/final_study/README.md)** — data setup,
  optional model reuse, dependency pinning, resources and checkpoints.
- **[Report figures and tables](reports/final/README.md)** — six main historical
  figures plus supporting diagnostics, organized around experimental questions.
- **[Previously submitted CSV](artifacts/final/submission_final.csv)** — the frozen
  894-row submission associated with the reported 0.567; new predictions are saved
  separately by the notebook.
- **[Findings](docs/evidence/v5_completed_findings.md)** and
  **[reproduction design](docs/evidence/reproducible_final_notebook.md)** — evidence,
  decisions and the difference between historical results and a new execution.

Default `Run all` loads verified models or trains missing ones: 60 selected fits
plus 30 basic-baseline fits, then evaluation and test prediction. Raw competition
data is required. The notebook includes code, recipes, splits, configuration IDs
and seeds. A full historical model archive is optional and accelerates fitting.
Fresh CPU reproduction can require multiple sessions.

`RUN_ABLATIONS=False` avoids rerunning the historical search by default. Explicitly
enable the recorded comparison suite when needed. `RUN_MODE="review"` renders the
historical figures only and produces no new submission. Checked-in saved outputs
are labelled historical review outputs; source defaults use reproduction.

## Repository map

| Location | Contents |
|---|---|
| [`notebooks/`](notebooks/README.md) | Final notebook and historical experiments |
| [`src/`](src/README.md) | Builders, implementations and tests |
| [`artifacts/final/`](artifacts/final/README.md) | Frozen submission/evidence and portable reproduction specification |
| [`reports/final/`](reports/final/README.md) | Current narrative's historical figures, tables and captions |
| [`docs/`](docs/README.md) | Findings, research, methodology drafts and references |
| [`archive/legacy/`](archive/legacy/README.md) | Earlier approaches and report drafts |
| `local/` | Downloaded checkpoints, local outputs and scratch; ignored by Git |
| `data/` | Raw competition data; ignored by Git |

On the original workstation the complete historical model archive is
`local/results/sc4000_gnn_v5_results (11)/`. Keep independent backups; Git does not
store trained weights. The compact evidence and reproduction packs contain no weights.

Local audit scores and public Kaggle scores are different evidence. Historical
results are retained; freshly trained models receive their own evaluation and
provenance. No new leaderboard result is assumed from a rerun.
