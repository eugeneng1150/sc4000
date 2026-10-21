# GNN ablation v2: findings, decisions and proposed next experiments

Recorded: 18 October 2026. This is a working evidence and decision record, **not the final course report**.

## 1. Outcome and evidence status

The user reported Kaggle scores of **0.41065 for the baseline submission** and **0.45936 for the selected submission**. This is an absolute increase of **0.04871**, or **11.86% relative to the baseline**. These numbers were supplied in conversation; a Kaggle submission-history export or screenshot has not been independently checked. Confirm the submission filenames, evaluation phase and leaderboard split before using them in the final report. Do not silently attribute 0.41065 to a particular older notebook.

The local evidence supports a more specific conclusion: the selected system improves the sampled holdout score for both NLP collections and XLA random, but worsens XLA default and tile. The clearest feature-improvement evidence comes from the NLP tabular experiments. GNN effects vary by collection and feature combination.

The user also reported a current position around **top 15.7%**, and a top-10% score near **0.59**. These are user-reported leaderboard observations, not independently verified standings. Reaching 0.59 from 0.45936 requires another **0.13064**, approximately **28.44% relative**. We cannot infer that a graph-size change will achieve this.

The result bundle is an extracted directory at [sc4000_gnn_ablation_v2_results](../../local/results/sc4000_gnn_ablation_v2_results). The experiment notebook is [SC4000_Comprehensive_Ablations.ipynb](../../notebooks/experiments/SC4000_Comprehensive_Ablations.ipynb), committed on branch `baseline` as `65df150`.

### Completion checks

| Check | Observed result |
|---|---:|
| Collections completed | 5/5 |
| Required training jobs | 192 |
| Optional development-only training jobs | 85 |
| Total successful training jobs | 277 |
| Successful holdout evaluation jobs | 192 |
| Successful test-graph inference jobs | 894 |
| Recorded extraction exclusions | 0 |
| Rows / unique IDs in each submission CSV | 894 / 894 |

All inspected worker status files reported `ok`. Both submission CSVs are present. No rerun is needed to complete v2.

Sources: [run summary](../../local/results/sc4000_gnn_ablation_v2_results/RUN_SUMMARY.md), [run state and configuration](../../local/results/sc4000_gnn_ablation_v2_results/run_state.json), [submission status](../../local/results/sc4000_gnn_ablation_v2_results/submission_status.json), [optional-stage status](../../local/results/sc4000_gnn_ablation_v2_results/optional_status.json).

## 2. Why this study was built

The earlier workflow did not give a sufficiently clear controlled comparison of a fixed reference against individual additions. The objective was to obtain evidence for statements such as “adding feature X changed score Y by Z,” instead of presenting only a final model score.

The user wanted one extensive Colab workflow, with no repeated cycle of running a small experiment, sending outputs back and editing the notebook. Constraints were standard approximately 12–13 GB CPU RAM and local storage. The solution therefore prioritised bounded memory, fixed experiment definitions, automatic model selection, checkpoints and one reviewable results bundle.

Two distinct reference points were needed:

| Reference | Purpose | Important qualification |
|---|---|---|
| `B0_basic_hgb` | System baseline: fixed pooled-summary features and HGB | A documented v2 baseline, not an exact reproduction of every historical final model |
| `G0_reference` | Fixed two-layer, width-32 GraphSAGE reference for incremental GNN comparisons | Derived from the existing project architecture, with shared v2 splits and strict sampling; not an exact historical rerun |
| `B_matched_hgb` | HGB trained on the same graph/configuration IDs as the GNN | Separates training-data volume from some model differences; it does not equate their representations or objectives |

The term “reference” does **not** mean an intentionally weak model. It means a version held fixed so that additions can be measured.

## 3. Provenance: established methods, inspiration and our implementation

The statement “the reference already contains them” refers to **our own G0/project architecture**. Message passing, opcode embeddings, numeric node features and pairwise ranking were already present before the new mask/context additions. Showing that these existing components are useful is an ablation result; it is not evidence that we newly invented or introduced them in v2.

GraphSAGE and ranking losses are established methods. For example, the [original GraphSAGE paper](https://arxiv.org/abs/1706.02216) describes learning representations by sampling and aggregating neighbourhood features. Its experiments are not evidence of performance on this competition.

At the user's request, leading competition solutions provided conceptual inspiration about padding, configuration-aware features, ranking objectives and seed variability. The v2 implementation was written independently; competitor solution code, model weights and predictions were not integrated. Independent implementation does not establish global research novelty.

The attribution record is [TOP_SOLUTION_RESEARCH.md](../research/TOP_SOLUTION_RESEARCH.md). It distinguishes the competitors' reported findings from our hypotheses and notes limitations in retrieving their write-ups. That document is a pre-experiment research note: its “untested” statements describe that stage, not the current results.

Our defensible contribution is the independently implemented, CPU-bounded representation and experiment workflow, together with controlled empirical evidence in this project. Cite the prior work that motivated it. Do not describe standard GraphSAGE, embeddings or pairwise loss as original inventions.

## 4. Evaluation protocol and scope

Development graphs were separated from official training data. Official validation graphs were reserved for the v2 audit. The project had inspected official validation in earlier work, so “reserved” applies to this run's selection procedure; it is not a claim that these graphs were historically unseen by the project.

| Collection | Tabular training graphs | Development graphs | Holdout graphs | GNN included? |
|---|---:|---:|---:|---|
| XLA layout, default | 40 | 12 | 7 | Yes |
| XLA layout, random | 40 | 12 | 7 | Yes |
| NLP layout, default | 40 | 12 | 20 | No |
| NLP layout, random | 40 | 12 | 20 | No |
| XLA tile | 160 | 12 | 24 | No |

The GNN used up to 24 training graphs and 64 configurations per graph. Standard tabular fits used up to 256 configurations per training graph, drawn from a saved pool of up to 384. Evaluation used at most 1,024 fixed, label-independent configurations per graph. All methods in a collection shared evaluation IDs and labels.

Required methods used seeds **42, 137 and 2026**. Seeds change training configuration samples and model randomness; they are **not three independent graph splits**. Related architecture families can occur across splits even though graph names are disjoint. Default/random XLA collections also share holdout graph identities, so their results should not be treated as independent sets of graph evidence.

GNN early stopping used a fixed subset of up to 128 development configurations per graph, at most 20 epochs and patience 5. Final development comparisons used the full saved development samples. Selection did not use holdout scores.

### Metrics and aggregation

- Layout: exact Kendall tau-b on each sampled candidate set, then an unweighted mean across graphs. Constant predictions receive diagnostic tau zero.
- Tile: runtime divided by its normalizer; the diagnostic is `1 - top5_regret`, where regret is the best runtime among the predicted top five divided by the best sampled runtime, minus one.
- These are local diagnostics. Do not average layout and tile results into an alleged official Kaggle score.
- Per-method ablation tables average each seed's per-graph score. System tables first average prediction ranks across seeds, then score that ensemble. These are different operations and need not produce the same score.
- Paired intervals resample graphs after averaging seed differences within each graph. They are exploratory bootstrap intervals, without correction for multiple comparisons. Seven XLA holdout graphs provide limited uncertainty estimates.

The final submissions score all test candidates with the already-audited fitted models. They do not retrain on the holdout or automatically use all available training graphs. Two layout template rows request 1,000 ranks from 1,001 candidates; 16 short tile rows preserve the original template's five-slot suffix. These cases are logged in submission metadata.

## 5. What was tested, and why

### GNN experiments on both XLA layout collections

| Experiment | Change / question |
|---|---|
| `G0_reference` | Two-layer GraphSAGE, opcode embedding, numeric node inputs, zero-filled layout values and pairwise ranking objective |
| `G_add_masks` | Add explicit valid-position masks: does distinguishing padding from valid dimension zero help? |
| `G_add_context` | Add categorical position/opcode/depth/edge summaries: does global configuration context help? |
| `G_full` | Add both masks and context: do they complement one another? |
| `G_no_messages` | Remove learned neighbour aggregation from G_full: does message passing contribute? |
| `G_no_opcode_embedding` | Remove the learned opcode embedding from G_full |
| `G_no_numeric` | Remove numeric node inputs from G_full |
| `G_regression` | Replace pairwise ranking with regression in G_full |
| `G_one_layer` | Reduce G_full from two message layers to one |

G0 and the mask/context additions have equal parameter counts: unused channels are zero-filled. Their graph samples, training IDs and budgets are shared. This avoids attributing gains simply to a larger prediction head.

The context summaries cover the full configurable-node set, while the learned GNN uses a sampled graph. Context can therefore add information absent from the sampled graph, not just transform existing information. `G_no_opcode_embedding` still has opcode-aware context, and `G_no_messages` still has edge-aware context. Their names describe removal of learned components, not removal of every source of that information.

### Tabular experiments on all five collections

| Group | Purpose |
|---|---|
| `B0_basic_hgb` | Fixed HGB and basic summaries |
| `B1_masked_hgb` | Add padding-aware layout summaries; tile uses signed-log transformed values |
| `P0_position` | Add positional categorical layout histograms; tile uses configuration interactions |
| `P_full` | Add the full tested representation; layout includes opcode/depth/edge-conditioned summaries, while tile includes graph features |
| `A_without_*` | Remove one block from P_full while retaining the remaining recipe |
| `B_matched_hgb` | Match GNN training-data IDs on XLA collections |
| Optional `S_*`, `P_*`, `C_*`, `M_*` | Additional feature combinations, weights, targets, duplicate handling, data-size controls and model families |

Optional model families were Extra Trees, a small MLP, ridge and linear pairwise ranking. Optional experiments ran only on the first seed and only on development data, **after selection, holdout evaluation and submission export**. Their results did not change the frozen submission and should not be described as confirmed three-seed holdout gains.

## 6. Why these particular final combinations were selected

Selection ranked configurations within each graph for each seed, then averaged those ranks across seeds. It chose the best tabular and GNN seed ensembles on development data. Candidate blends used GNN weights `{0, 0.25, 0.5, 0.75, 1}` with either B0 or the selected tabular model. Baseline-only was included, and exact ties favoured it. The best development score was frozen before holdout evaluation.

Consequently, the winner need not be the method with the highest mean individual-seed score in `confirmation_dev_summary.csv`. The actual ensemble prediction is what was selected. This explains apparent differences between that summary and `selection_lock.json`.

| Collection | Frozen selected system | Development ensemble score | Decision rationale at selection time |
|---|---|---:|---|
| XLA default | 50% `P0_position` + 50% `G_no_numeric` | 0.159400 | Better development score than tabular alone (0.133285), GNN alone (0.135245), or B0 (0.072583) |
| XLA random | 25% `A_without_depth` + 75% `G_full` | 0.397873 | Better development score than tabular alone (0.363930), GNN alone (0.374171), or B0 (0.327216) |
| NLP default | `A_without_opcode` | 0.352379 | Best eligible development ensemble under the frozen selection procedure |
| NLP random | `A_without_masked` | 0.589472 | Best eligible development ensemble under the frozen selection procedure |
| Tile | `A_without_topology` | 0.993652 | Best eligible development ensemble under the frozen selection procedure |

This records why a choice was reasonable using the information available then. It does not retroactively label every choice optimal after seeing the holdout.

Sources: selection locks for [XLA default](../../local/results/sc4000_gnn_ablation_v2_results/layout_xla_default/selection_lock.json), [XLA random](../../local/results/sc4000_gnn_ablation_v2_results/layout_xla_random/selection_lock.json), [NLP default](../../local/results/sc4000_gnn_ablation_v2_results/layout_nlp_default/selection_lock.json), [NLP random](../../local/results/sc4000_gnn_ablation_v2_results/layout_nlp_random/selection_lock.json), [tile](../../local/results/sc4000_gnn_ablation_v2_results/tile_xla/selection_lock.json).

## 7. System-level held-out results

These are prediction-rank ensembles across three seeds, not means of standalone seed scores.

| Collection | B0 ensemble | Selected system | Paired change | Exploratory 95% graph-bootstrap interval | Graphs improved |
|---|---:|---:|---:|---|---:|
| XLA default | 0.079451 | -0.008841 | -0.088292 | [-0.302946, 0.058375] | 3/7 |
| XLA random | 0.163779 | 0.237660 | +0.073882 | [-0.000927, 0.151620] | 4/7 |
| NLP default | 0.243613 | 0.295223 | +0.051610 | [0.022607, 0.080696] | 14/20 |
| NLP random | 0.341036 | 0.478588 | +0.137552 | [0.097853, 0.180009] | 19/20 |
| Tile | 0.890712 | 0.841098 | -0.049614 | [-0.136694, 0.012696] | 6/24 |

The strongest system-level local evidence is on NLP. XLA random improves on average but its interval slightly crosses zero. XLA default and tile degrade on average. This mixed holdout outcome can coexist with an overall Kaggle improvement: the samples and metrics differ, and the aggregate score alone cannot attribute component effects.

Sources: paired system tables for [XLA default](../../local/results/sc4000_gnn_ablation_v2_results/layout_xla_default/locked_systems_holdout_gains.csv), [XLA random](../../local/results/sc4000_gnn_ablation_v2_results/layout_xla_random/locked_systems_holdout_gains.csv), [NLP default](../../local/results/sc4000_gnn_ablation_v2_results/layout_nlp_default/locked_systems_holdout_gains.csv), [NLP random](../../local/results/sc4000_gnn_ablation_v2_results/layout_nlp_random/locked_systems_holdout_gains.csv), [tile](../../local/results/sc4000_gnn_ablation_v2_results/tile_xla/locked_systems_holdout_gains.csv).

### Selection failures and informative counterexamples

- XLA default: `inception_v3_batch_128_train` falls from approximately 0.368 with B0 to -0.310 with the selected blend. A large loss on one of seven graphs heavily affects the average. This is a diagnostic observation, not a reason to exclude that graph.
- XLA random: the selected tabular ensemble alone scores **0.288629**, higher than the frozen blend's **0.237660** on holdout. The blend was better on development, so its apparent complementarity did not fully transfer.
- Tile: `A_without_topology` averages **0.908232** when scoring seeds separately, but its rank ensemble scores **0.841098**. Averaging ranks need not preserve the best top-five candidates; ensemble scoring must be evaluated directly.

Do not replace frozen choices using these holdout findings and then call the revised score an untouched final evaluation. Such changes would start a new model-development round.

## 8. Feature evidence and GNN evidence

### Stronger tabular representations

With HGB, training samples and other primary settings fixed, P_full versus B0 gives:

| Collection | B0 mean seed score | P_full mean seed score | Change | Exploratory interval |
|---|---:|---:|---:|---|
| NLP default | 0.235771 | 0.301596 | +0.065824 | [0.04298, 0.08858] |
| NLP random | 0.332216 | 0.474148 | +0.141932 | [0.10453, 0.18200] |

The gain is positive for every seed in both collections. These comparisons support the **feature bundle**; they do not assign the entire gain to any one histogram or mask. Individual blocks may overlap in information, and feature effects need not add linearly.

Sources: [NLP default method gains](../../local/results/sc4000_gnn_ablation_v2_results/layout_nlp_default/holdout_paired_gains.csv), [NLP random method gains](../../local/results/sc4000_gnn_ablation_v2_results/layout_nlp_random/holdout_paired_gains.csv).

### Additions relative to the fixed GNN reference

These scores average per-seed graph scores, rather than scoring an ensemble.

| Collection | G0 | G0 + masks | G0 + context | G_full: masks + context |
|---|---:|---:|---:|---:|
| XLA default | -0.014477 | 0.008391 | -0.050524 | -0.065379 |
| XLA random | 0.169145 | 0.110541 | 0.124073 | 0.186747 |

Specific conclusions:

- Default masks: **+0.022867 tau**, interval **[0.012369, 0.032442]**, improvement on 6/7 graphs after averaging seeds. This helps relative to a weak-performing G0, but the resulting score remains near zero. Two of three seed means improve; do not claim every seed improves.
- Random full additions: **+0.017602 tau**, or **10.41% relative to G0**, interval **[-0.031202, 0.063596]**. An observed average gain with substantial uncertainty, not an established robust improvement. Two of three seed means improve.
- Context alone is worse than G0 on both collections. Masks alone hurt on random. The combination's behaviour shows that feature interactions matter.
- All tested default GNN methods score below the mean B0 score, despite some improving on G0.

Sources: [default comparisons with G0](../../local/results/sc4000_gnn_ablation_v2_results/layout_xla_default/holdout_paired_vs_G0_reference.csv), [random comparisons with G0](../../local/results/sc4000_gnn_ablation_v2_results/layout_xla_random/holdout_paired_vs_G0_reference.csv).

### Removing components from G_full on XLA random

| Removed / changed component | Ablated score | G_full minus ablation | Exploratory interval for that difference |
|---|---:|---:|---|
| Message aggregation | 0.106924 | +0.079823 | [0.03056, 0.13341] |
| Numeric node inputs | -0.019234 | +0.205981 | [0.08313, 0.34394] |
| Opcode embedding | 0.082844 | +0.103903 | [0.02011, 0.20089] |
| Second message layer | 0.120869 | +0.065878 | [0.01567, 0.11900] |
| Pairwise loss replaced by regression | -0.057819 | +0.244566 | [0.12108, 0.37761] |

These support the contribution of components **conditional on this full model and collection**. They do not establish new inventions or gains from adding those components to G0, which already has them. Results are not universal: removing numeric inputs improves the default full GNN.

Source: [XLA random removal controls](../../local/results/sc4000_gnn_ablation_v2_results/layout_xla_random/holdout_paired_vs_G_full.csv). The saved file records ablation minus G_full; signs and interval endpoints above are reversed to express the benefit of retaining a component.

### Matched-data comparison

On XLA random, G_full exceeds matched-data HGB by **0.072957 tau**, but the interval **[-0.050676, 0.181293]** crosses zero. On default, G_full is **0.158675 below** matched-data HGB, interval **[-0.311681, -0.056172]**. There is no general GNN-superiority result under the current sampling/training regime.

Sources: [default matched-data comparison](../../local/results/sc4000_gnn_ablation_v2_results/layout_xla_default/holdout_paired_vs_B_matched_hgb.csv), [random matched-data comparison](../../local/results/sc4000_gnn_ablation_v2_results/layout_xla_random/holdout_paired_vs_B_matched_hgb.csv).

## 9. Resource results and graph-coverage limitation

The largest recorded worker RSS across monitored jobs was **0.8794 GiB**, against a configured **5 GiB** worker threshold. This is sampled process memory, not total Colab RAM or a guarantee that every instantaneous spike was captured.

Summing recorded worker wall times gives approximately **4.18 hours**. This excludes unmeasured download, notebook overhead, plotting and archive work; it is not an exact end-to-end runtime. The earlier conversational 10–20 hour estimate was conservative and should not replace these observed measurements.

The GNN caps were **512 total nodes, 384 configurable nodes and 4,096 edges**, with a training batch of 16 configurations. Hard caps prevented retaining every configurable node in large graphs.

Across the seven XLA holdout graphs, in each collection:

| Retention measure | Mean | Median |
|---|---:|---:|
| Nodes retained / original nodes | approximately 6.1% | approximately 4.2% |
| Configurable nodes retained / original configurable nodes | approximately 69.7% | 100% |
| Edges retained / original edges | approximately 3.5% | approximately 1.5% |

The 100% median configurable-node retention matters: the GNN often retained all configurable nodes while losing much of their surrounding graph. A few large graphs retained far fewer configurable nodes; the holdout minimum was approximately 21.6%.

These are per-graph ratios averaged or medianed over graphs, not ratios of aggregate node/edge totals. The retained-edge fraction includes edges lost because an endpoint was discarded; simply increasing the edge cap cannot restore those edges. The node cap, seed distribution and neighbourhood selection also matter.

**Hypothesis, not established cause:** the sparse retained neighbourhoods may restrict what message passing can learn. This run did not vary coverage while holding everything else fixed, so it cannot establish that sampling caused the weak default GNN results.

Sources: [default coverage](../../local/results/sc4000_gnn_ablation_v2_results/layout_xla_default/coverage.json), [random coverage](../../local/results/sc4000_gnn_ablation_v2_results/layout_xla_random/coverage.json), worker status files and the archived [GNN implementation](../../local/results/sc4000_gnn_ablation_v2_results/study_gnn.py).

## 10. Reproducibility checks and historical comparisons

The assessment checked:

1. Required, optional, holdout and test inference completion counts and status values.
2. Disjoint train/development/holdout graph names within each collection.
3. GNN training manifests contain only graphs assigned to train.
4. Holdout graph/configuration IDs and labels align across the baseline, selected tabular method and selected blend components, for every seed.
5. Recomputed baseline and selected-system ensemble scores from saved prediction files match the saved system scores.
6. All four archived implementation modules match the local v2 helper sources byte-for-byte at assessment time.

These checks support the integrity of these comparisons; they are not proof that every possible scientific or implementation issue has been excluded.

Earlier artifacts in [full_run_backup](../../local/results/full_run_backup) use a different protocol; its GNN validation table has five graphs per XLA collection, while v2 has seven. Its metadata also describes a rebuilt baseline, not recovered historical top-17% fitted artifacts. Earlier summary files use other model/feature choices. Do not compare their local scores directly to v2 as if everything else were fixed.

A stronger Kaggle score can establish a better result between the specific submitted files on that leaderboard split. It does not establish the cause, guarantee a private-score improvement, or retrospectively make historical local-validation scores comparable.

## 11. Decisions supported now

| Decision | Rationale | Status |
|---|---|---|
| Preserve the v2 selected submission and its model-selection record | User-reported Kaggle gain and complete archived evidence | Completed; no rewriting of its selection history |
| Use richer tabular representations as a strong comparator in the next round | Consistent NLP improvements and strong XLA-random tabular results | Recommended |
| Describe GNN effects per collection and per control | Additions/removals have different effects across collections | Required for accurate interpretation |
| Treat the full-GNN gain on random as uncertain | Its G0 comparison interval crosses zero | Required qualification |
| Investigate graph coverage | Severe neighbourhood reduction, with observed RAM headroom | Proposed experiment, not a proven fix |
| Investigate validation stability and ensemble behaviour | Some development-selected improvements reverse on holdout | Proposed |
| Separate optional exploratory results from confirmed ablations | Optional methods only received one-seed development screening | Required qualification |
| Preserve raw results and keep this note separate from the final report | Enables audit and later revision without overwriting evidence | This document implements that decision |

A defensible eventual report statement is:

> Our development-selected system improved the reported Kaggle score from 0.41065 to 0.45936, an 11.86% relative increase. Controlled ablations showed benefits from richer tabular representations on NLP collections, while GNN improvements depended on the collection and feature combination.

Do not write “the GNN caused an 11.86% improvement,” “every addition helped,” “all collections improved,” or “the 0.0176 GNN gain is statistically established.” Do not call a tau increase “accuracy percentage points.”

## 12. Proposed next round toward the reported 0.59 target

This section is a recommendation, **not an implemented v3 notebook or an additional completed experiment**. The target requires substantially more than the previous absolute Kaggle gain. No rank or score improvement is guaranteed. The leaderboard page did not expose readable rows during the check, so the 0.59 threshold remains the user's estimate.

### A. Protect the next run and freeze its evaluation plan

V2 uses local storage. Its epoch checkpoints and ZIP are not durable against deletion of the Colab VM unless downloaded elsewhere. `RESTORE_ZIP` is an optional input to recovery, not the output location or an automatic external backup. The current extracted results are complete, so this observation does not imply missing v2 evidence.

Before another long run, recommend backing up completed experiment checkpoints/results to persistent storage, while retaining dataset reads and temporary features locally. Google Drive mounting requires user authorization; no persistent-backup implementation or authorization is implied by this note.

Use additional graph-disjoint development folds within official train for the next selection stage. Keep the v2 holdout as an **already-inspected benchmark**, not a newly untouched test. Reserve genuinely unused graph labels where feasible for a fresh audit, and document any prior exposure. Avoid repeatedly choosing configurations from public leaderboard feedback.

### B. Calibrate memory and test graph coverage first

Start with a few large representative graphs and measure actual forward/backward peak RSS and throughput. Do not infer safe batch sizes by linearly scaling the 0.8794 GiB whole-run maximum.

Suggested node-cap candidates are **512 (v2 control), 1,024, 2,048 and 4,096**. Increase configurable-node and edge budgets coherently; just raising `gnn_nodes` while leaving configurable nodes capped at 384 tests a narrower question. Preserve the v2 sample as a nested subset where practical, and record actual node/configurable-node/edge coverage for every setting.

Keep model width, loss, training graph/configuration IDs, development candidates, seed set and effective configuration batch fixed for the primary coverage ablation. Reduce physical batch size only if profiling requires it. If batching changes, use an appropriate control: ordinary gradient accumulation does not automatically recreate pairwise comparisons between configurations in different microbatches.

Test improved connected-neighbourhood selection as a **separate** factor after the size experiment. The current sampler spreads configurable seed nodes across the graph, so increasing graph size and changing the sampling method simultaneously would obscure which change helped. Record connected components, retained-neighbour counts and isolated-node fraction as diagnostics, not only raw size.

Use G0 and G_full as fixed recipes on both XLA collections. Developments in GraphSAGE motivate neighbourhood sampling, but the original paper does not establish the best sampling strategy for this runtime-ranking task. Citation: [Hamilton et al., GraphSAGE](https://arxiv.org/abs/1706.02216).

### C. Test training-data coverage separately

The GNN currently learns from 24 graphs × 64 configurations, and the tabular models are also trained on subsets. After selecting a safe graph budget on development data, test more training graphs and configuration counts such as 64, 128 and 256, with all candidate IDs drawn from train only. Include matched-data HGB controls.

Do not present an improvement from both more graph context and more labelled training data as the isolated effect of either change. Larger data is a hypothesis, not a guaranteed gain: matched-data HGB already outperformed larger-data B0 on the default holdout average.

### D. Develop the strong tabular route and choose blends carefully

Carry forward the successful configuration-aware features. Compare the current HGB against suitable boosted-tree candidates and increased training coverage under the same graph folds and resource bounds. The project already has older XGBoost/LightGBM workflows; this is a controlled comparison to perform, not a measured v2 result or a claim that a particular library will win.

Only blend when development folds show useful complementarity. Include tabular-only, GNN-only and baseline-only endpoints. Evaluate the actual ensemble's metric, particularly tile top-five selection. The random holdout result (tabular 0.2886 versus blended 0.2377) shows why adding a GNN should not be automatic.

### E. Promotion criteria

Promote a candidate based on repeated graph-development performance, seed variation, per-collection regressions, measured resource use and a frozen evaluation protocol. Keep the v2 submission as the comparison point. Report any prospective leaderboard gain separately from local component effects.

The most useful next claim to test is: **“Increasing retained graph context improves the same GNN under fixed training data and model settings.”** If it fails, the experiment still narrows the explanation, and the tabular/data-coverage route remains available. Reaching top 10% likely requires evaluating several such improvements rather than assuming one memory setting will close a 0.13064 score gap.

## 13. V3 implementation follow-up — 18 October 2026

The prospective coverage and training-data studies are now implemented together in
[SC4000_V3_Coverage_and_Data.ipynb](../../notebooks/experiments/SC4000_V3_Coverage_and_Data.ipynb).
See the [protocol and recovery guide](../../src/ablation_v3/README.md) and
[implementation validation](../../src/ablation_v3/VALIDATION.md). This records an implemented
experiment, **not a measured v3 accuracy improvement**; all v2 findings above remain historical.

The user chose periodic browser downloads of recovery ZIPs, with local Colab data and scratch
storage, rather than Google Drive. Recovery depends on the latest ZIP actually saved on the
user's computer. The new notebook includes both stages, matched-data controls, two development
folds, three-seed confirmation of key comparisons and automatic selection/submission export.
It may require multiple sessions, but does not require manual model choices between stages.
