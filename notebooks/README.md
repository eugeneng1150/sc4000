# Notebooks

Start with [SC4000_Final.ipynb](SC4000_Final.ipynb). It contains the frozen final
method and saved outputs; running it regenerates evidence figures and the exact
submission without retraining.

The [experiment notebooks](experiments/) retain the exploration history:

| Notebook | Question studied |
|---|---|
| [Comprehensive ablations](experiments/SC4000_Comprehensive_Ablations.ipynb) | Baselines, features and GNN component controls |
| [Coverage and data](experiments/SC4000_V3_Coverage_and_Data.ipynb) | Graph sampling budgets and training-data scaling |
| [Focused scaling](experiments/SC4000_V4_Focused_Scaling_and_Report.ipynb) | Larger tabular training sets and longer GNN training |
| [Ranking and targets](experiments/SC4000_V5_Ranking_and_Targets.ipynb) | Target transformations and boosted-tree ranking |

These are reproducibility records, not prerequisites for running the final notebook.
Their builders and runtime implementations live in [src](../src/README.md).
Older approaches are in [archive/legacy](../archive/legacy/README.md).
