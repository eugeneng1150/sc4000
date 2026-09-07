# SC4000 - Predict AI Model Runtime

This project works on Kaggle's **Google - Fast or Slow? Predict AI Model Runtime** competition.

The goal is to predict which compiler configuration will make an AI model graph run fastest. The final output is not just a runtime value; it is a ranking of configurations from fastest to slowest for each test graph.

## Competition Task

AI models can be represented as computation graphs:

- A node represents a tensor operation, such as matrix multiplication, convolution, reshape, or add.
- An edge represents a tensor flowing between operations.
- A compiler configuration changes how the compiler optimizes the graph.

For each graph, the model must rank candidate compiler configurations by predicted runtime.

The submission file has this format:

```csv
ID,TopConfigs
tile:xla:example_file,3;7;1;5;0
layout:xla:random:example_file,10;2;8;1;0;...
```

`TopConfigs` is a semicolon-separated list of configuration indices, ordered from fastest predicted runtime to slowest predicted runtime.

## Dataset Collections

The dataset is split into five collections:

```text
tile:xla
layout:xla:default
layout:xla:random
layout:nlp:default
layout:nlp:random
```

These names combine three ideas: the optimization type, the graph family, and the configuration source.

## `layout` vs `tile`

### `layout`

`layout` configurations control how tensors are arranged in physical memory.

For layout collections:

- The search space is large.
- The submission should rank all configurations.
- Kaggle evaluates the full ranking using Kendall Tau correlation.

Collections:

```text
layout:xla:default
layout:xla:random
layout:nlp:default
layout:nlp:random
```

### `tile`

`tile` configurations control tile sizes for fused subgraphs.

For the tile collection:

- The search space is smaller.
- Only the first 5 predicted configurations are used for scoring.
- The goal is to include a very fast configuration in the top 5.

Collection:

```text
tile:xla
```

## `xla` vs `nlp`

### `xla`

`xla` refers to general XLA HLO computation graphs.

XLA is Google's compiler system for optimizing machine learning computations. These graphs may come from different model families, such as CNNs, Transformers, BERT, SSD, and other ML workloads.

Collections using `xla`:

```text
tile:xla
layout:xla:default
layout:xla:random
```

### `nlp`

`nlp` refers to graphs from NLP models, mostly BERT-style model workloads.

Collections using `nlp`:

```text
layout:nlp:default
layout:nlp:random
```

## `default` vs `random`

This only applies to the `layout` collections.

### `default`

`default` configurations are based around compiler-generated or default-style layout choices.

### `random`

`random` configurations are randomly generated layout choices.

These two groups can behave differently, so the first modelling approach should treat each collection separately.

## Recommended Modelling Strategy

Start with a simple, explainable ranking pipeline before attempting graph neural networks.

Recommended first version:

1. Load all `.npz` files.
2. Inspect array keys, shapes, and runtime labels.
3. Create one training row per graph/configuration pair.
4. Extract simple graph and configuration features.
5. Train a tabular model to predict runtime.
6. Sort configurations by predicted runtime for each test graph.
7. Write `submission.csv`.

Good baseline models:

- LightGBM
- XGBoost
- RandomForestRegressor

A full graph neural network may perform better, but it is harder to implement, slower to train, and more difficult to debug. For this project, a clean ranking baseline is a better first milestone.

## Notebook Workflow

Main notebook:

```text
SC4000.ipynb
```

Current intended sections:

1. Load Kaggle credentials.
2. Download the competition dataset.
3. Inspect the dataset structure and `.npz` contents.
4. Build features.
5. Train baseline model.
6. Generate ranked predictions.
7. Export `submission.csv`.

The notebook is developed locally in this repository, but it should remain compatible with Google Colab or another cloud runtime.

## Data Location

In Colab, the dataset is downloaded to:

```text
/content/predict-ai-model-runtime
```

The `.npz` files are under:

```text
/content/predict-ai-model-runtime/npz_all/npz
```

Expected structure:

```text
npz_all/npz/
  tile/xla/
    train/
    valid/
    test/
  layout/xla/default/
    train/
    valid/
    test/
  layout/xla/random/
    train/
    valid/
    test/
  layout/nlp/default/
    train/
    valid/
    test/
  layout/nlp/random/
    train/
    valid/
    test/
```

## Useful Link

Kaggle competition:

https://www.kaggle.com/competitions/predict-ai-model-runtime
