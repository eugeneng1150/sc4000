# Validation performed on 17 October 2026

- Full synthetic suite: **13 tests passed** in 142 seconds.
- After submission-template and reporting fixes, reran the affected end-to-end GNN,
  submission validation and notebook embedding checks: **3 tests passed** in 65 seconds.
- Both tabular and GNN synthetic workflows exercised isolated workers, development-only
  selection, holdout scoring, paired tables and unchanged-run resume.
- GNN checks exercised all nine variants, equal parameter counts for feature additions,
  hard graph bounds, disabled message aggregation, matched HGB training IDs, exact
  epoch-checkpoint recovery and compact-versus-streamed prediction agreement.
- Resource checks verified that memory and timeout limits terminate workers and record
  explicit status files. Notebook schema, Python syntax and embedded/source equality passed.
- Real data smoke check: `inference_mlperf_ssd_1200_batch_1.npz`, with 25,544 nodes,
  3,143 configurable nodes and 33,522 edges. Three configurations streamed successfully;
  the GNN retained 512 nodes, 384 configurable nodes and 139 induced edges and produced
  finite predictions. This is a data-path/memory-bound check, not an accuracy evaluation.
- Preflight verified all **894** supplied template rows against local test NPZ headers:
  **16** tile rows require preservation of the supplied five-slot template suffix;
  **2** layout rows request 1,000 ranks from 1,001 candidates. Both cases are handled
  explicitly and counted in export metadata.

Local validation used Python 3.13 with PyTorch 2.14.1 and the installed scientific Python
packages. It did not execute the full 192-fit competition study, establish leaderboard
performance, or measure the full run's duration on standard Colab. Those measurements
will be saved by the user's Colab run. A nine-hour stop is a session budget, not a completion
or accuracy guarantee; unfinished stages remain labelled incomplete and resumable.
