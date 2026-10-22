# Notebooks

Start with [SC4000_Final.ipynb](SC4000_Final.ipynb). Its default run loads compatible
models or trains missing selected/baseline models from raw competition data, then
evaluates and generates a fresh submission. No model ZIP is required. Review mode
renders the historical evidence without training. See the
[execution guide](../src/final_study/README.md).

| Historical notebook | Question studied |
|---|---|
| [Comprehensive ablations](experiments/SC4000_Comprehensive_Ablations.ipynb) | Baselines, features and GNN component controls |
| [Coverage and data](experiments/SC4000_V3_Coverage_and_Data.ipynb) | Graph sampling and training-data budgets |
| [Focused scaling](experiments/SC4000_V4_Focused_Scaling_and_Report.ipynb) | Larger tabular sets and longer GNN training |
| [Ranking and targets](experiments/SC4000_V5_Ranking_and_Targets.ipynb) | Target transformations and boosted-tree ranking |

These preserve exploration history. Running them is not a prerequisite for the
standalone final notebook. Their implementations are in [src](../src/README.md);
older approaches are in [archive/legacy](../archive/legacy/README.md).
