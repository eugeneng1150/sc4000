# Repository organization — 21 October 2026

The final system is frozen. This cleanup separates the course handoff from
experimental history and local runtime files. It does not change model recipes,
training implementations, predictions or checkpoint identities.

## Location changes

| Previous location | Current location |
|---|---|
| `Eugene/SC4000_Final.ipynb` | `notebooks/SC4000_Final.ipynb` |
| `Eugene/SC4000_Comprehensive_Ablations.ipynb`, `SC4000_V3_*`, `SC4000_V4_*`, `SC4000_V5_*` | `notebooks/experiments/` |
| `Eugene/ablation_study/` | `src/ablation_v2/` |
| `Eugene/ablation_v3/`, `ablation_v4/`, `ablation_v5/` | Corresponding directories under `src/` |
| `Eugene/final_study/` code | `src/final_study/` |
| `Eugene/final_study/` CSV, evidence ZIP and manifest | `artifacts/final/` |
| `evidence/` findings and decisions | `docs/evidence/` |
| Prior-work and post-v4 research notes | `docs/research/` |
| `Eugene/Papers/` | `docs/references/` |
| `Eugene/gnn_architecture.tex` | `docs/methodology/gnn_architecture.tex` |
| Earlier notebooks, ranker, report draft and figures | `archive/legacy/` |
| `results/` and root checkpoint ZIP | `local/results/` |
| `output/` | `local/outputs/` |
| `Eugene/Output_1.ipynb`, `Output_2.ipynb` | `local/notebook_snapshots/` |
| Local course brief, scratch files and cache | `local/course/`, `local/tmp/`, `local/cache/` |

Final figures, tables and captions were copied from the successfully executed
final notebook output into `reports/final/`. Raw data remains in `data/`.
Both `data/` and `local/` are ignored by Git. No full checkpoint or fitted model
was deleted; keep independent backups of these local files.

The prior uncommitted architecture edits and findings were retained. The complete
v2 source, earlier extracted ranker and research notes are now included in their
appropriate directories. Unique local notebook snapshots were preserved rather
than discarded as duplicates.

## Removed files

- `Eugene/test_graph_pipeline.py`: orphaned test for an already absent `graph.ipynb`.
- `Eugene/README.md`: superseded by the root README and directory indexes.
- macOS `.DS_Store` metadata in the retired directory.

The tracked removed files remain available in Git history. The earlier hybrid
notebook and its matching test remain together under `archive/legacy/`.

## Path compatibility and verification

Builders and tests resolve notebooks and frozen artifacts from their new paths.
Relative documentation links were updated. Runtime module contents and checkpoint
internal paths are unchanged, preserving source-hash identity checks.

The frozen CSV, evidence ZIP, notebooks (including saved outputs), scientific
runtime modules and edited architecture draft were verified against their hashes
from before the move. Four final replay/integrity checks and the four experimental
notebook embedding checks passed. All 60 selected model-file hashes were also
verified at the relocated checkpoint path, without loading or fitting models.
All five builders were exercised in a temporary
copy so the saved notebook outputs were not overwritten. The historical v4/v5
notebooks retain their run-specific settings; builders use fresh-run defaults.

No competition training was rerun for this cleanup. The latest full checkpoint is
`local/results/sc4000_gnn_v5_results (11)/`; the portable final evidence archive is
`artifacts/final/frozen_evidence.zip` and does not contain the fitted weights.

## Branch handoff

Commit the organized state on `baseline` and review a PR from `baseline` to `main`.
At cleanup time `main` was an ancestor of `baseline`, so no rebase or preliminary
merge was needed. The cleanup does not merge branches or rewrite history.
