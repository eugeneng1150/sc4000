SC4000 full report draft - 22 October 2026

Main source: report_draft.tex
Bibliography: report_references.bib
Compiled document: report_draft.pdf (10 pages including references)
Supporting figures, submission screenshot and generated tables: draft_assets/

Build from this directory:
  tectonic report_draft.tex
or use pdfLaTeX + BibTeX in Overleaf with report_draft.tex as the main document.
The source bundle contains every file needed to compile; it does not contain
raw competition data, model checkpoints or the submitted predictions.

Template provenance:
  Official AAAI 2026 author kit, CameraReady/LaTeX/aaai2026.sty and .bst
  https://aaai.org/wp-content/uploads/2025/07/AuthorKit26-1.zip
  Downloaded 21 October 2026; style and bibliography files are unmodified.
  The draft option omits the AAAI publication copyright notice.
  The explicit OT1 selection makes Tectonic use the template's PSNFSS Times
  fonts instead of silently substituting Latin Modern under its TU default.
  No lecturer-specific template year has been confirmed. The course PDF
  requires AAAI format and at most ten pages including references.

Draft completion items:
  - Affiliation details, if required; author names and uppercase matriculation
    numbers have been supplied and added to the report.
  - Confirmed final rank and competitor count for relative ranking.
These are deliberately visible in the draft and have not been invented.

Evidence sources:
  draft_assets/kaggle_submission_scores.png is the supplied Kaggle screenshot.
  It shows public score 0.56715 and private score 0.49624 for
  submission_selected.csv, marked Complete (after deadline). It does not
  establish a leaderboard position.
  Tables and figures are regenerated from reports/final/tables CSVs by
  build_draft_assets.py. Model descriptions were checked against the current
  ablation_v3/v4/v5 source and the fixed reproduction contract. Older
  feature-bundle results come from gnn_ablation_v2_findings.md; longer-training
  and tabular-scaling findings from v4_completed_findings.md; blend confirmation
  from v5_completed_findings.md. Early MLP scores use the legacy experiment log
  and are explicitly separated from later controlled comparisons.

No training, model reselection, commit, push or public release was performed.
The historical submitted CSV was not changed. The report describes historical evidence, not a new training run. The source
and PDF were updated on 22 October 2026. The compiled PDF has 10 pages,
including all references. Contribution statements follow the user-provided roles.

Sources were checked against original papers, official library documentation,
and author-owned competition repositories. Stable documentation may describe
newer library releases; numerical claims and implementation details come from
the recorded local versions and code, not the current documentation defaults.

For local asset regeneration (not required to compile the source bundle):
  python build_draft_assets.py
Requires the evidence CSVs under tables/, NumPy and Matplotlib. This only
generates document figures and tables; it never runs model fitting.
