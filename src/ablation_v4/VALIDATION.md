# V4 validation — 20 October 2026

Six tests passed, including real synthetic model fits/inference:

1. Synthetic tabular pipeline: resource pilot, two-fold screening, incumbent
   fallback, audit summaries, complete CSV export, report generation, ZIP restore,
   unchanged completed-job hashes on resume, changed-model full-candidate inference,
   configuration-change rejection and changed-data rejection.
2. GNN: train epoch 1, resume through epoch 2 with identical prior history,
   evaluate audit data, and export a valid full-candidate permutation. Tests use
   the same feature/model functions as the notebook.
3. Notebook schema and compilation of every code cell; exact embedding of v4 modules.
4. Archive checks: truncated file, path traversal and nonempty restore destination.
5. All five real v3 collections: existing selected ensemble remains an available
   candidate using saved predictions, without refitting any v3 model.
6. Selection guard: a higher average with a loss on one fold is rejected.

Final six-test run completed in 16.302 seconds. It also restored the ZIP into a
different directory and reran the controller there: completed fit status hashes
were preserved, and the run completed without retraining those fits.

Additional read-only real-data validation:

- Compared frozen v3 source modules with checkpoint 65: all six byte-identical.
- Generated and visually inspected all six main figures from checkpoint 65.
- Read actual training archive headers/fingerprints. Fold-0 NLP expansion:
  186 and 195 graphs. Tile 640/1,280 static matrix estimates: 419.5/859.5 MiB;
  2,560/5,120/all exceed configured limits and are excluded before preparation.
- The report preview is in `output/v4_report_preview/report/`.

Scope: local macOS tests use one CPU thread and temporary PyTorch 2.14.1.
The Colab notebook instead pins the package versions recorded in checkpoint 65
(including its CPU PyTorch build), with two worker threads. No full v4 competition
training has been run locally; real Colab memory/runtime and leaderboard gains
are not established by synthetic tests. Browser download delivery must be checked
on the user's computer. The new notebook does not promise 0.59 or top 10%.
