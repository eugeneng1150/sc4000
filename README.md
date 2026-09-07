# SC4000 - Predict AI Model Runtime

This repository contains project work for Kaggle's **Google - Fast or Slow? Predict AI Model Runtime** competition.

The competition is about building a learned cost model for compiler optimization. Given a machine learning computation graph and many possible compiler configurations, the goal is to rank the configurations from fastest predicted runtime to slowest predicted runtime.

## Competition Goal

Machine learning models can be represented as computation graphs:

- A node represents a tensor operation, such as matrix multiplication, convolution, reshape, or add.
- An edge represents a tensor flowing between operations.
- A compiler configuration changes how the compiler optimizes or lays out the graph.

Actually benchmarking every compiler configuration is expensive. The model should predict which configurations are likely to run fastest, so the compiler can try better candidates first.

The submission file contains rows like:

```csv
ID,TopConfigs
tile:xla:example_file,3;7;1;5;0
layout:xla:random:example_file,10;2;8;1;0;...
```

`TopConfigs` is a semicolon-separated list of configuration indices ordered from fastest predicted runtime to slowest predicted runtime.

## Dataset Collections

The dataset is split into five official collections:

```text
tile:xla
layout:xla:default
layout:xla:random
layout:nlp:default
layout:nlp:random
```

These names combine:

- `tile` or `layout`: the compiler optimization task.
- `xla` or `nlp`: the graph source/family.
- `default` or `random`: the layout configuration source. This only applies to `layout`.

## `layout` vs `tile`

### `layout`

`layout` configurations control how tensors are arranged in physical memory.

For layout collections:

- The search space can be very large.
- Each graph can have thousands to 100k candidate configurations.
- The submission generally ranks all configurations.
- The ranking quality matters across the whole list.

Layout collections:

```text
layout:xla:default
layout:xla:random
layout:nlp:default
layout:nlp:random
```

### `tile`

`tile` configurations control tile sizes for fused subgraphs.

For the tile collection:

- The graphs are usually smaller.
- Only the top few predicted configurations are most important.
- The goal is to place a very fast configuration near the top, especially in the top 5.

Tile collection:

```text
tile:xla
```

## `xla` vs `nlp`

### `xla`

`xla` refers to general XLA HLO computation graphs. These can come from different model families such as ResNet, BERT, Inception, MLPerf workloads, SSD, Transformer, and others.

Collections using `xla`:

```text
tile:xla
layout:xla:default
layout:xla:random
```

### `nlp`

`nlp` refers to NLP model graphs, mostly BERT-style workloads.

Collections using `nlp`:

```text
layout:nlp:default
layout:nlp:random
```

## `default` vs `random`

This only applies to the `layout` collections.

`default` configurations are based around compiler-generated or default-style layout choices.

`random` configurations are randomly generated layout choices.

These can behave differently, so notebooks should normally treat `layout:xla:default` and `layout:xla:random` as separate collections instead of combining them.

## Collaboration Rule

Multiple people may use this GitHub repository with Google Colab. To avoid overwriting each other's notebooks, outputs, and reports, each person should create their own top-level folder.

Recommended structure:

```text
sc4000/
  Eugene/
    SC4000_Eugene.ipynb
    README.md
    experiment_report.tex
  Member_1/
    SC4000_Member_1.ipynb
    README.md
  Member_2/
    SC4000_Member_2.ipynb
    README.md
  data/
    npz_all/
    sample_submission.csv
```

Do not edit another person's folder unless you are intentionally collaborating on their work.

Use a clear notebook name:

```text
SC4000_<name>.ipynb
```

If you generate outputs such as models, reports, or submissions, keep them inside your own folder:

```text
<name>/models/
<name>/submissions/
```

The dataset folder should stay shared at the repository root:

```text
data/
```

This avoids duplicating the large Kaggle data in every person's folder.

## Data Location

Locally, the dataset should be at:

```text
data/
```

The `.npz` files should be under:

```text
data/npz_all/npz
```

In Colab, the common expected location is:

```text
/content/data/npz_all/npz
```

Individual notebooks should implement path discovery so they can find the data whether they are run from the repository root, from a personal folder, or from Colab. Supporting the original Kaggle folder name `predict-ai-model-runtime/` as a fallback is useful, but the documented repository folder name is `data/`.

## Baseline Modelling Ideas

A practical first modelling pipeline is:

1. Load each `.npz` graph file.
2. Inspect graph arrays, configuration arrays, and runtime labels.
3. Build one row per graph/configuration pair.
4. Extract graph-level and configuration-level features.
5. Train one model per collection.
6. Predict runtime or centered log runtime for each test configuration.
7. Sort configuration indices by predicted speed.
8. Export `submission.csv`.

Useful baseline model families:

- MLP over tabular graph/configuration summaries.
- Random forest or boosted tree regressors.
- XGBoost or LightGBM if dependency setup is acceptable.
- Graph neural networks if compute and project scope allow deeper modelling.

## Important Modelling Issues

This dataset has strong imbalance:

- `tile:xla` has many more graph files than the layout collections.
- Some graph families, such as BERT, ResNet, and MLPerf workloads, appear much more often.
- Layout graphs can have very large numbers of configurations.
- The fastest configurations matter more than random slow configurations.

Good notebooks should consider:

- training separate models per collection,
- family-stratified file sampling,
- configuration sampling for large layout graphs,
- runtime-aware or top-k-aware sampling,
- ranking-aware validation metrics,
- validation across enough graph files to reduce noise.

## Competition Link

Kaggle competition:

https://www.kaggle.com/competitions/predict-ai-model-runtime
