# V3: graph coverage and training-data study

Run [SC4000_V3_Coverage_and_Data.ipynb](../../notebooks/experiments/SC4000_V3_Coverage_and_Data.ipynb)
in a standard CPU Colab runtime. It is self-contained and includes both requested questions:
show the same GNN more graph structure, then train it with more graphs/configurations.
V2 and its evidence are preserved separately.

The GNN coverage/data study targets XLA layout default and random, where v2 showed the
coverage limitation. NLP layout default/random and tile receive the expanded tabular study.
Final submissions cover all five collections.

## Automatic workflow

1. Check every test/template row before spending time on training.
2. Build two graph-disjoint development folds within official train. Each model evaluates
   only its own development fold; cross-fold training graphs cannot leak into selection.
3. Probe real large training graphs with a full configuration batch and no runtime labels.
   Test 512, 1,024, 2,048 and 4,096 node budgets, with configurable-node and edge caps growing
   alongside them. Reject resource-unsafe budgets explicitly.
4. Screen G0 and G_full at each admitted graph budget, fixing 24 training graphs and
   64 configurations. Select a budget using mean development scores across both folds.
5. At that fixed budget, screen a full 3×3 training-data grid: 24, 40 or all training graphs,
   and 64, 128 or 256 configurations. Include full-feature HGB on exactly the same IDs.
6. Include masks/context, no-message and connected-sampler controls. Screen tabular feature,
   graph-count, configuration-count, deeper HGB and Extra Trees candidates on all collections.
7. Repeat key coverage controls, the 2×2 data-grid corners, component/sampler controls,
   baseline/full-feature HGB and development leaders across all three seeds.
8. Freeze selection from development predictions. Blend candidates include baseline-only,
   tabular-only and GNN-only; selected blends cannot regress by more than 0.02 on either
   development fold versus B0. This does not guarantee audit/leaderboard improvement.
9. Audit confirmed comparisons on official valid, then export baseline, tabular and selected
   submission CSVs with all test candidates scored.

Defaults screen 186 training jobs if all graph budgets pass. Confirmation adds further runs,
with the exact count depending on the selected budget and leaders. Budget for multiple sessions;
nine hours is the per-session stop, not a completion estimate. Screening uses seed 42 on both
folds. Confirmed comparisons add seeds 137 and 2026. Their tables are explicitly separate.

The official validation set was inspected in v2. It is an already-inspected audit, not a new
untouched holdout. Three seeds and two folds are training variations, not independent copies
of the holdout graphs. Source NPZ fingerprints and scientific settings bind resumption.

## Memory and disk

One isolated worker, two CPU threads, 5 GiB worker RSS threshold and 2 GiB system reserve.
Pilots have a more conservative 3.5 GiB worker threshold. Sampled configurations stream from
NPZs. Only one fold/profile's compact features is retained, with a 4 GiB scratch budget;
training matrices are bounded at 512 MiB. Source data stays intact, and this study's marked
scratch directory is cleaned between profiles and on exit.

Larger graphs retain a training configuration batch of 16. The workflow does not silently
microbatch pairwise loss, because that would change its pairs. Configurable-node seed prefixes
are nested; complete retained node sets need not be. Connected sampling is an independent
control. The v3 512-node reference uses v3 sampling, not the exact historical v2 sample.

Checkpoints and saved models have a 12 GiB output budget. ZIP creation verifies local free
space before creating a replacement archive and preserves the previous ZIP on failure.
Monitoring and admission checks reduce OOM/disk risk but cannot eliminate sudden spikes,
Colab termination or storage failures.

## Downloads and recovery

Per user preference, **no Google Drive backup** is enabled. The notebook requests a browser
download of `sc4000_gnn_v3_results.zip` every 30 minutes, after each collection and on normal
exit/interruption. The interval is checked during long-running workers as well as between jobs.
Allow multiple downloads and check your computer's Downloads folder. Repeated copies may
have browser-added suffixes such as `(1)`.

A ZIP actually saved on your computer survives runtime deletion. A download request alone
is not proof of receipt; a disconnected browser may receive nothing. Work since the last
successful download can be lost. No runtime keep-alive or policy bypass is used.

After a reset, upload the latest v3 ZIP, set `RESTORE_ZIP` to its actual `/content/...` path,
keep the same `RUN_NAME` and scientific settings, and run all cells. The output directory
must be empty before restoration. Dataset files must be downloaded/provided again.
If the existing VM still has results, leave `RESTORE_ZIP` empty and rerun the study cell.

GNN epoch checkpoints include the current model, optimizer, RNG, history and **best model
state**, so the checkpoint can resume even if the separate best-model file was not captured
in an interrupted backup. Successful jobs are skipped. Workers interrupted by a deleted
runtime have `running` status and are eligible to resume. Version/identity changes fail
explicitly rather than silently mixing evidence. Restore only your own trusted model ZIPs.

## Outputs

- `experiment_plan.json`: model/stage/profile/data-size definitions.
- `resource_eligibility.json`, `probes/`: admitted budgets and pilot measurements.
- `preparations/`: actual sampled coverage and isolated-node fraction for each fold/profile.
- `coverage_lock.json`: chosen coverage before the data grid starts.
- `confirmation_plan.json`: scientific controls and leaders repeated across seeds.
- `selection_lock.json`: final development-only model/blend selection and per-fold checks.
- `*_screening_vs_*.csv`, `*_confirmed_vs_*.csv`: separately labelled paired comparisons.
- `*_factor_contrasts.csv`: explicit changes in coverage, training graphs, configurations,
  features, message passing and GNN versus HGB with matched training IDs. Each includes score
  differences, relative percentage gains when the reference score supports that calculation,
  and exploratory graph-bootstrap intervals.
- `system_holdout_gains.csv`: audit of the actual fold/seed ensembles.
- `submission_baseline.csv`, `submission_tabular.csv`, `submission_selected.csv`.
- Models, training histories/IDs, prediction IDs, logs, settings, source and run state.

The v3 baseline is newly fitted under v3 folds and sampling. It is not automatically the
0.41065 v2 baseline submission. Archive the existing v2 selected CSV as the external
competition benchmark. V3 does not submit to Kaggle or guarantee a 0.59 score/top-10% rank.

## Development

```bash
python3 src/ablation_v3/build_notebook.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python src/ablation_v3/test_v3.py
```

The notebook embeds `engine.py`, `gnn.py`, `common.py`, `plan.py`, `submission.py` and
`controller.py`. Tests need the scientific Python packages plus PyTorch and nbformat.
Process-watchdog integration tests require process enumeration permissions on macOS.
