# Supporting code

| Directory | Role |
|---|---|
| [final_study](final_study/README.md) | Load-or-train reproduction, historical narrative, evidence packaging and final notebook builder |
| [ablation_v2](ablation_v2/README.md) | Comprehensive baseline and component ablations |
| [ablation_v3](ablation_v3/README.md) | Graph coverage and training-data study |
| [ablation_v4](ablation_v4/README.md) | Focused scaling and report generation |
| [ablation_v5](ablation_v5/README.md) | Ranking, target transformations and resource-aware scheduling |

Run builders from the repository root: `python src/<study>/build_notebook.py`.
They write to `notebooks/` (final) or `notebooks/experiments/` (studies).
Rebuilding clears saved cell outputs. Each study README and validation record
explains its dependencies and checks. Versioned runtime modules are preserved
because checkpoints identify the exact implementation used to train models.
