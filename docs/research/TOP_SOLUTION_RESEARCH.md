# Ideas from leading Fast or Slow solutions

Research date: 17 October 2026. Scope: conceptual inspiration for our own experiments; no competitor code, weights, or predictions were copied, downloaded, or integrated.

## Sources and rank qualification

The first-party repositories identify their solutions as first, second, and third place; the fifth-place author also has a Kaggle writeup with that placement. These are **reported competition placements**, not verified public-leaderboard positions. The leaderboard URL returned no readable rows in this research session, so this note does not claim exact public scores or public ranks. Directly opening several Kaggle writeups returned empty bodies; searchable excerpts from Kaggle itself and readable author-owned GitHub READMEs supplied the evidence below. References to findings are the competitors' reports, not independently reproduced results.

| Reported place | What the original source reports | Lesson worth testing |
|---|---|---|
| 1st — Viet-Br Connection | Layout preprocessing distinguishes padding, and the solution includes graph compression. Their writeup emphasizes feature scaling and explicitly comparing configurations inside the network. Their repository describes ensembles of 5–20 models per collection. | Preserve configuration information and audit preprocessing before increasing model size. Our implementation need not use their attention block or ensemble. [Author repository](https://github.com/thanhhau097/google_fast_or_slow), [original writeup](https://www.kaggle.com/competitions/predict-ai-model-runtime/writeups/viet-br-connection-1st-place-solution-for-the-goog). |
| 2nd — Latenciaga | Their GraphSAGE system used different losses across collections. They report that ranking losses substantially outperformed element-wise losses; configurations from the same graph were grouped for training. They also report duplicate configuration measurements and expensive training: roughly 20 A100 hours per collection. | Test whether the learning objective fits the ranking task; investigate repeated measurements. A compact experiment is more suitable for our CPU constraints than reproducing their training recipe. [Author repository and technical explanation](https://github.com/dmitrii-khizbullin/kaggle_latenciaga). |
| 3rd — Duck / jafluri | The author describes additional graph encodings, transforms for wide-range features, and dimension features motivated by TPU alignment. They report substantial seed variation, including score differences around 0.1, and advise repeated training. | Representation and repeatability deserve explicit experiments. Their reported variability is not a measured property of our pipeline. [Author repository](https://github.com/jafluri/kaggle_tpu_graph), [original writeup](https://www.kaggle.com/competitions/predict-ai-model-runtime/writeups/duck-3rd-place-solution-write-up). |
| 5th — knshnb | Their writeup preserves correspondence among dimension features using dimension tokens; a transformer embeds them before graph processing. They train using a pairwise ranking loss. | Respect dimension/configuration semantics when forming features. We can investigate cheap categorical summaries without adopting their transformer architecture. [Original writeup](https://www.kaggle.com/competitions/predict-ai-model-runtime/discussion/456093), [author repository](https://github.com/knshnb/kaggle-tpu-graph-5th-place). |

The evaluation distinguishes layout ordering (Kendall tau) from tile selection (best runtime among the top five predicted configurations). Therefore a good generic runtime regression score is not sufficient evidence of improvement on the actual task. [Official competition evaluation](https://www.kaggle.com/competitions/predict-ai-model-runtime/overview).

## What this suggests for our project

The following are **our untested hypotheses**, not claims that a winner used these exact features, nor claims of research novelty.

Inspection of cell 16 in [SC4000_Eugene.ipynb](../../archive/legacy/SC4000_Eugene.ipynb) shows two potentially useful gaps. The graph fingerprints are shared by every configuration of one graph. They can help cross-graph interactions, but cannot alone distinguish that graph's candidates. Existing local layout features use scalar means and products with degree/depth/opcode values; they do not explicitly encode the categorical association between a node's operation and its layout. Replacing padding with zero for aggregation can also obscure the distinction between missing positions and valid dimension zero.

**Recommended first experiment: compact configuration-aware layout features.** Keep the basic HGB regressor fixed and add a small independently designed feature block:

- Count layout categories at each position, separately by operation type (opcode), retaining a separate padding mask. Use bounded hashing or a training-derived small vocabulary to control feature count.
- Summarize the same positional categories within coarse graph-depth or degree groups. This tests whether where a configuration occurs matters beyond its overall histogram.
- As a later extension, summarize layout-pattern pairs across directed edges between configurable nodes. Call these categorical co-occurrence features; do not claim physical memory compatibility unless tensor-dimension correspondence has actually been established.

This is a focused question: **does preserving which operation receives which layout improve ranking beyond pooled summaries?** It is closer to the decision being predicted than simply adding more configuration-invariant graph statistics. There is no guarantee of a gain; extra sparse features can hurt generalization or cost more than they help.

A second, separate experiment can compare a compact ranking model with a regression model using the same features and graph groups. The project already has `pairwise_margin_ranking_loss` in [gnn_final.py](../../archive/legacy/gnn_final.py), so this would extend controlled comparison to the pooled-feature branch rather than introduce ranking to the entire project. Do not conflate a feature change, learner change, and loss change into one claimed contribution. A ranking objective alone is an established method, not novel merely because our implementation is independent. Prioritize layout first; improvements to layout do not imply improvements to tile top-five selection.

## Fixed baseline and ablations

Use the existing `simple_summary_ablation` feature profile with a fixed HGB configuration as the primary baseline, and retain the pooled MLP as a secondary reference. The name of an experiment does not determine whether it can serve as a baseline. Keep the current selected ensemble as an additional strong system-level reference if evaluating a later blend.

| Experiment | Change from the primary baseline | Question |
|---|---|---|
| B0 | None: basic summaries + HGB | What does the fixed reference achieve? |
| B1 | Padding-aware summaries only | Does distinguishing absent dimensions help? |
| P1 | B1 + opcode-conditioned positional layout counts | Does operation/layout association help? |
| P2 | P1 + graph-position-conditioned counts | Does graph location add information? |
| A1 | P2 with opcode conditioning removed | Is the association useful beyond more features? |
| A2 | P2 with graph-position conditioning removed | Does that structural context contribute? |

The feature baseline should retain exactly the same training files, sampled configurations, budget, target, and HGB settings across comparisons. Audit feature definitions and padding first; if an actual preprocessing error is fixed, apply the correction to a clearly versioned baseline too rather than presenting a bug fix as a new learning method.

Evaluate with graph-disjoint development and held-out sets. Reserve the final holdout before exploring these ideas; do not repeatedly tune on it. Keep the official validation set intact for any reported official-split comparison. Use full configuration sets where practical, or a fixed label-independent validation sample; training-only runtime-stratified sampling is a separate choice. Repeat promising runs with several seeds, report graph-level paired score differences and per-collection results, and retain preprocessing/training/inference time and memory measurements. If bootstrapping uncertainty, resample graphs rather than treating correlated configurations as independent observations.

For a report, a defensible claim would be: “Motivated by prior work on configuration-aware representations, we designed compact categorical layout summaries and evaluated their incremental effect under a fixed boosting baseline.” Claim improvement only after the measurements support it. Attribute the motivation to these sources; independent implementation does not make established ranking or encoding ideas newly invented.

No model training or implementation changes were made by this research task.
