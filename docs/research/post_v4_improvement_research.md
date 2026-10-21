# Research: improvement directions after the focused scaling study

Research date: 20 October 2026. This is a research and experiment-design note, not a claim that the proposed methods have been implemented or improved the score. The running v4 study should finish unchanged. No training code, notebook, saved model or submission was changed by this research.

## Recommendation

The strongest next direction is **better configuration representation and task-specific supervision**, with the existing GNN/tabular ensemble retained as the strong comparator. More data is a reasonable experiment already running in v4; the evidence does not justify assuming that repeatedly expanding data or graphs will close the score gap.

First repeat the inexpensive percentile-target comparison on the stronger final HGB recipe. Then test a small grouped tree-ranking experiment and independently designed semantic features. Treat graph compression, segment training and new GNN collections as later projects with substantially greater implementation and compute cost.

The user reported approximate Kaggle scores of **0.40 baseline, 0.53 tabular and 0.55 selected**, with a target around **0.59**. These are conversation-supplied rounded values, not independently verified leaderboard records. The selected and tabular submissions differ only on the 16 XLA rows; that supports preserving the XLA contribution in the selected submission, but does not identify the gain from each individual GNN or collection. The completed artifacts are in [checkpoint 65](<../../local/results/sc4000_gnn_v3_results (65)>).

## What our evidence actually supports

| Observation | Interpretation and limit | Local evidence |
|---|---|---|
| Rich tabular features improved NLP default and random over the basic HGB under the v2 controlled comparison: approximately +0.0658 and +0.1419 tau. V3 confirmed development comparisons also favour the richer bundle. | Configuration-aware representations are a supported direction. These are bundle effects, not an isolated benefit from every feature. | [V2 findings, section 8](../evidence/gnn_ablation_v2_findings.md); [v3 NLP default](<../../local/results/sc4000_gnn_v3_results (65)/layout_nlp_default/dev_confirmed_vs_B0_basic_hgb.csv>), [NLP random](<../../local/results/sc4000_gnn_v3_results (65)/layout_nlp_random/dev_confirmed_vs_B0_basic_hgb.csv>). |
| On v2 XLA random, full GNN pairwise loss scored 0.186747; its regression replacement scored −0.057819. Retaining pairwise loss gave +0.244566 tau, exploratory graph-bootstrap interval [0.12108, 0.37761]. | This supports the ranking objective for that specific GNN setup. It does not prove every ranker beats every regressor. The saved contrast file expresses the reverse sign, ablation minus full model. | [Removal controls](../../local/results/sc4000_gnn_ablation_v2_results/layout_xla_random/holdout_paired_vs_G_full.csv), [interpretation](../evidence/gnn_ablation_v2_findings.md). |
| Increasing GNN training graphs helped some comparisons, but not all. In v3 XLA random at 64 configurations, the confirmed development contrast from 24 to all eligible training graphs was about +0.0450; at 256 configurations the corresponding gain was about +0.0018 with an interval crossing zero. | Data scaling can help, but its effect depends on the configuration budget and recipe. More graphs is not a universal fix. | [V3 confirmed factor contrasts](<../../local/results/sc4000_gnn_v3_results (65)/layout_xla_random/dev_confirmed_factor_contrasts.csv>). |
| Larger graph-coverage settings did not consistently improve the saved comparisons. The local XLA audit preferred the tabular system to the selected blend, while the user-reported Kaggle ordering was reversed. | Preserve uncertainty and avoid tuning a large search on seven XLA audit graphs. Neither a bigger graph nor a higher local point estimate is sufficient evidence of generalisation. | [XLA default contrasts](<../../local/results/sc4000_gnn_v3_results (65)/layout_xla_default/holdout_confirmed_factor_contrasts.csv>), [random contrasts](<../../local/results/sc4000_gnn_v3_results (65)/layout_xla_random/holdout_confirmed_factor_contrasts.csv>), system holdout files in the same directories. |

The official TPUGraphs work distinguishes whole-list ordering for layout from best-runtime-among-top-K for tile, and reports that ranking losses and early combination of graph/configuration information helped its baselines. These are findings from the authors' experiments, not results for our sampled models. [TPUGraphs paper, sections 4–5](https://proceedings.neurips.cc/paper_files/paper/2023/file/ded1a89e2b3b925444ada973af66336e-Paper-Datasets_and_Benchmarks.pdf).

## 1. First: revisit the cheap target experiment on the strong model

The current selected HGB fits a graph-centred log-runtime target. Earlier optional experiments already tested a per-graph percentile target; this idea is **not new or untested in the project**. It received only a one-seed exploratory comparison under the older setup. [Current target construction](../../src/ablation_v3/engine.py), [v2 optional-stage qualification](../evidence/gnn_ablation_v2_findings.md).

The saved seed-42 development scores for percentile target versus `P_full` were:

| Collection | Centred log | Percentile | Difference |
|---|---:|---:|---:|
| XLA default | 0.072956 | 0.176837 | +0.103881 |
| XLA random | 0.364464 | 0.270613 | −0.093851 |
| NLP default | 0.361097 | 0.380053 | +0.018956 |
| NLP random | 0.583396 | 0.601078 | +0.017682 |
| Tile | 0.988821 | 0.990692 | +0.001871 |

Sources: `dev_seed_scores.csv` for [XLA default](../../local/results/sc4000_gnn_ablation_v2_results/layout_xla_default/dev_seed_scores.csv), [XLA random](../../local/results/sc4000_gnn_ablation_v2_results/layout_xla_random/dev_seed_scores.csv), [NLP default](../../local/results/sc4000_gnn_ablation_v2_results/layout_nlp_default/dev_seed_scores.csv), [NLP random](../../local/results/sc4000_gnn_ablation_v2_results/layout_nlp_random/dev_seed_scores.csv), and [tile](../../local/results/sc4000_gnn_ablation_v2_results/tile_xla/dev_seed_scores.csv).

**Proposed controlled test:** hold the final HGB features, graph/configuration IDs, development folds, seeds and tree settings fixed; change only centred-log versus percentile target. Start with the two NLP collections and XLA default. Include the existing target as a control even where the earlier result looked encouraging. Confirm only reproducible gains across the two folds and additional seeds. CPU cost should be similar in order to the existing HGB fit with the same matrix; measure it rather than promise an ETA.

Percentile targets remove runtime-distance information. That could help a whole-order task but can also hurt, as the XLA-random result demonstrates. Tile requires a separate decision based on top-five regret, not the tiny old development gain alone.

## 2. Grouped boosted-tree ranking, with a learner control

The current tree models optimise regression. The optional v2 pairwise model was a **linear** difference classifier; its weak performance does not settle whether nonlinear tree ranking helps. The GNN already uses within-graph pairwise hinge loss, so “introduce ranking” would be an inaccurate description of the project. [Implemented learners and loss](../../src/ablation_v3/engine.py), [GNN loss](../../src/ablation_v3/gnn.py).

XGBoost provides `rank:pairwise`, an unscaled pairwise logistic objective, and `rank:ndcg`, which applies NDCG-related weighting. It requires query groups; here each graph is a query and configurations are its candidates. Its `mean` pair construction samples pairs across a query, whereas `topk` emphasises its leading candidates. Larger ranker scores mean greater relevance. [Official XGBoost learning-to-rank documentation](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html).

**Proposed layout test:** use the same compact features and training IDs for (a) current HGB, (b) XGBoost histogram regression and (c) XGBoost histogram pairwise ranking. Match the two XGBoost arms' tree capacity and iteration budget. Comparing (b) with (c) isolates the objective within one learner; comparing HGB directly with the ranker would also change implementation and tree-growth behaviour. Encode faster configurations as higher relevance, group strictly within graph, handle ties consistently, and evaluate actual mean per-graph Kendall tau. Pairwise logistic is a surrogate, not exact tau optimisation.

For tile, LightGBM's LambdaRank can focus on a top cutoff and requires nonnegative integer relevance labels with explicit gains. A cutoff near five is only a hypothesis: NDCG@5 rewards the whole leading list, whereas our tile criterion depends on its single best runtime. Do not claim the objectives are equivalent. [Official LightGBM parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html).

A tile pilot should compare regression and ranking in the same tree library using the same data, with relevance bins derived from **training** runtime ranks. Stop if exact top-five regret fails to improve. No new graph-message tensors are needed, but tree training still has dataset/histogram memory overhead; retain pilots and worker caps. Avoid an unrestricted sweep across both libraries, several losses and many hyperparameters.

The reported second-place team found ranking losses useful and used different losses for different collections; it also reported unstable XLA validation and expensive GPU training. This motivates controlled loss choices, not reproducing its training recipe or assuming its gains transfer. [Authors' repository](https://github.com/dmitrii-khizbullin/kaggle_latenciaga/blob/master/README.md).

## 3. Semantic configuration features rather than more generic summaries

Our code already has separate padding masks, positional categorical histograms, opcode/depth-conditioned counts, edge co-occurrences, topology and WL summaries. Recommending those again would repeat completed work. However, most pooled layout summaries do not explicitly join a configuration's chosen axis with that axis's physical dimension size. The GNN's local layout input is still the 18 axis IDs scaled by five, alongside masks and a categorical global context. [Layout features](../../src/ablation_v3/engine.py), [GNN input](../../src/ablation_v3/gnn.py).

OpenXLA specifies that axis numbers are labels, and that the minor-to-major permutation describes memory order. Consequently, an axis ID alone is not a physical size or a universally comparable quantity across shapes. [Official shape/layout semantics](https://openxla.org/xla/shapes).

The TPUGraphs schema provides output dimension sizes in node-feature positions 21–26 and the output layout in configuration positions 0–5. Tile configurations separately expose kernel, output and input bounds. A layout block of all −1 also has a default-compiler-strategy meaning; it should not be interpreted as a real axis permutation. [Official dataset feature schema](https://github.com/google-research-datasets/tpu_graphs#features).

**Independently designed hypotheses to test separately:**

- **Layout semantic summaries:** for valid output permutations, reorder output dimension sizes into physical layout order; aggregate bounded log-sizes, aspect ratios and alignment/padding proxies, optionally conditioned on opcode. Start with output tensors where the NPZ mapping is explicit. Input/kernel dimension correspondence must be verified before extending this recipe; do not infer it just from matching numeric indices.
- **Tile interaction summaries:** the present explicit pairwise interactions use only the first eight configuration fields, while all 24 raw fields are included. Add a small block based on output bounds and verified output shape: tile-to-shape ratios, tile volumes and boundary-remainder indicators. This is added representation, not fixing missing raw columns.
- **GNN categorical control:** replace scalar layout axis IDs with a small categorical encoding while preserving sampled nodes, edges, data, loss and training budget. First test categorical versus scalar input; only then add dimension-aware features. Report changed parameter count if an embedding changes model capacity.

TPU documentation describes padding/alignment effects involving 8-by-128 chunks for the relevant hardware family. Third-place authors also used dimension transformations motivated by hardware alignment. These support trying compact features, **not** claiming that any simple padding formula predicts real runtime or that applying modulo 128 to every column is correct. [Google TPU performance guide](https://docs.cloud.google.com/tpu/docs/performance-guide?hl=en), [third-place author repository](https://github.com/jafluri/kaggle_tpu_graph).

Start with HGB plus one new semantic block, at fixed data and objective, before placing it inside a GNN. Extraction is bounded per graph/configuration and can avoid storing full configuration tensors, but throughput and validity still require a real-data pilot. This extends the strongest local feature trajectory at lower initial engineering cost than a new graph architecture.

## 4. Preserve useful structure: later, not the immediate next sweep

Uniform and connected sampling and multiple graph budgets have already been compared. More retained nodes alone therefore is not an unexplored recommendation. A different question is whether we can retain information from the full graph without backpropagating through it all.

Graph Segment Training divides large graphs into segments and backpropagates through sampled segments; its stronger variant uses historical embeddings and methods to mitigate their staleness. The authors evaluated it on TPUGraphs and report memory/efficiency advantages. [Original GST paper](https://arxiv.org/abs/2305.12322).

For this project, GST is a substantial implementation change: segment caches, configuration dependence, stale-state recovery and inference all need validation. Memory efficiency does not imply cheap CPU runtime. A simpler independently designed compression pilot could retain configurable nodes and aggregate nonconfigurable regions, but dropping or contracting paths changes graph semantics and must be audited. Neither is justified as “the fix” by the current coverage results.

A GNN pilot on NLP or tile is also a legitimate future direction; no local comparison proves tabular superiority there. Official baselines include GNNs for both tasks, including early configuration injection for tile. [Official baseline implementation overview](https://github.com/google-research-datasets/tpu_graphs#running-baseline-models). However, the XLA model cannot simply be reused unchanged for tile's global configuration features. Pilot the observed graph-size distribution and full training/inference cost before authorising another extensive branch.

## 5. Improve decisions before multiplying models

Use saved out-of-fold predictions to measure per-graph failures, seed variance and complementarity at almost no new training cost. Rank averaging is already used; merely proposing an ensemble is not a new method. Examine whether candidate gains come from a broad set of graphs or one outlier, and whether the candidate improves the actual **combined** system. [V3 submission implementation](../../src/ablation_v3/submission.py), [v2 examples of ensemble reversals](../evidence/gnn_ablation_v2_findings.md).

For tile, inspect whether rank averaging discards a fast candidate found by an individual model. A fixed small candidate-selection alternative, such as a validated union/reranking rule, can be tested on saved development predictions. This is an untested project hypothesis, not a known guaranteed improvement. Avoid searching hundreds of blend weights or choosing per-graph rules from audit labels.

The third-place authors reported substantial repeat-to-repeat variation, and the second-place authors also described XLA instability. These are reasons to repeat promising experiments, not estimates of our own variance. [Third-place repository](https://github.com/jafluri/kaggle_tpu_graph), [second-place repository](https://github.com/dmitrii-khizbullin/kaggle_latenciaga/blob/master/README.md).

## A bounded next sequence

| Order | Experiment | Hold fixed | Cost category and promotion condition |
|---|---|---|---|
| 0 | Finish v4; inspect actual gains and saved prediction complementarity | Running study and current submissions | No additional training for analysis; choose the next question from observed failures. |
| 1 | Centred-log versus percentile HGB on selected layout collections | Features, IDs, folds, seeds, learner and tree settings | Small relative to GNN work; screen two folds, confirm only promising changes. |
| 2 | One semantic feature block with the selected HGB | Target, data, folds and learner | Extra extraction plus HGB fits; reject invalid/missing shape mappings before training. |
| 3 | Grouped boosted-tree regression versus ranking | Same library, features, IDs, folds and capacity | Pilot bounded histogram fits; require actual tau/top-five gains, not proxy-metric gains alone. |
| 4 | One small GNN representation or alternative-loss control | Winning graph budget, data, architecture except stated factor | Several GNN fits; proceed only if earlier stages or failure analysis justify the cost. |
| Later | GST/compression or new NLP/tile GNN branch | Explicit separate protocol | Larger research/engineering commitment; not part of the current v4 run. |

Change one factor at a time for attribution; only combine independently promising changes afterwards. Compare against the selected system, not only the weak baseline. Use the existing development folds for comparability, but recognise repeated adaptive reuse: freeze a small candidate set and, where feasible, use a genuinely uninspected graph audit for the next round. Previously inspected official validation cannot become untouched again. Report graph-level paired changes and seed sensitivity, keep full evaluation candidate sets where feasible, and separate local results from Kaggle.

## Source and provenance limits

Only official documentation, original papers and author-owned repositories were used as external evidence. The cited libraries' stable documentation was checked on the research date; pin and verify the actual implementation version before any future experiment. Kaggle's fifth-place write-up redirected to a page with no readable body in this session, so no new detailed claims were based on it. Reported competition placements are the authors' descriptions, not independently verified leaderboard rows.

Implementation observations above refer to the current local helper files at the cited paths. The numerical findings refer to the saved result bundles. This research did not perform a new byte-for-byte identity audit between every current helper and every archived runtime; future implementation should record the exact source identity it uses. Semantic ratios must join each configuration with its relevant node/tensor dimensions, not with an unrelated graph-wide average.

The research uses ideas for independent controlled experiments; no competitor model weights, predictions or solution code were copied into the project. None of these sources establishes that the proposed changes will raise this submission from 0.55 to 0.59.
