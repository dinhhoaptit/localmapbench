# JOSS to-do list (everything needed)

Order matters: **public repo timeline** is a hard gate. Start the GitHub clock early.

## A. Pre-submission gates (desk-reject if missing)

- [ ] Create public GitHub repository (browse/clone/issues without login)
- [ ] Keep **> 6 months** of public, iterative development before JOSS submission (not a one-day dump)
- [ ] OSI-approved LICENSE file in repo root (MIT/BSD/Apache/GPL-compatible as preferred)
- [ ] Evidence of research use (this project’s experiments / preprint / technical report / Zenodo methods archive; external users if any)
- [ ] Feature-complete scope: reusable library, not “run these scripts once”

## B. Package the software (`localmapbench` or similar)

- [ ] Standard Python layout: `src/localmapbench/` (or package name) + `pyproject.toml`
- [ ] Public API modules, e.g.:
  - feature maps (Raw, Binary/Soft, Sliding, Pairwise, PLS, SIR)
  - nested CV runner + 1-SE largest-\(k\) rule
  - suitability diagnostic
  - regime generators (positive-control suite)
- [ ] Move experiment drivers to `examples/` or `scripts/` (not the only interface)
- [ ] `pip install -e .` works on a clean environment
- [ ] Pin / document dependencies (`numpy`, `pandas`, `scikit-learn`, …)
- [ ] Optional: publish to TestPyPI / PyPI

## C. Documentation

- [ ] Root `README.md`: what it does, install, quickstart (≤10 lines to first result)
- [ ] `docs/` or MkDocs/Sphinx: API overview
- [ ] Installation instructions verified on a second machine / clean venv
- [ ] Example notebooks or scripts: one synthetic regime + one public dataset
- [ ] `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, support/contact expectations
- [ ] `CITATION.cff` (and/or bib entry) for citing the software
- [ ] `CHANGELOG.md` with versioned releases

## D. Tests and quality

- [ ] Unit tests for maps, fold isolation (no leakage), \(k\)-selection helpers
- [ ] Small golden-file / smoke tests (not full \(R=30\) MC in CI)
- [ ] Documented manual test procedure if some checks are interactive
- [ ] CI on GitHub Actions (pytest on push/PR)
- [ ] Lint/format optional but helpful (`ruff`/`black`)

## E. Open development practices (especially solo author)

- [ ] Iterative commits over months (features, fixes, docs)
- [ ] Tagged releases (`v0.1.0`, `v0.2.0`, …)
- [ ] Use Issues for bugs/enhancements (even if self-filed)
- [ ] Prefer small PRs over silent rewrites (optional for solo, still good signal)

## F. JOSS paper (in the same repo)

- [ ] Add `paper.md` (JOSS template sections only; ~750–1750 words)
  - Summary  
  - Statement of need  
  - State of the field (compare to related tools; build-vs-contribute)  
  - Software design  
  - Research impact statement  
  - AI usage disclosure  
  - Acknowledgements  
  - References  
- [ ] Add `paper.bib`
- [ ] **Do not** center the paper on Monte Carlo “which map wins” results
- [ ] Reframe MC ladder as **software validation / positive controls**
- [ ] Disclose related publications (LSSP submission/rejection, any preprint) in submission notes
- [ ] Figures: few or none; no large result tables in `paper.md`

## G. Archive and submit

- [ ] Make a GitHub Release matching the reviewed version
- [ ] Deposit that version on **Zenodo** (or Figshare) → software DOI
- [ ] Fill JOSS submission form (repo URL, version, review domains)
- [ ] Respond to pre-review / review issues on GitHub within JOSS timelines
- [ ] After acceptance: final tagged release + Zenodo DOI update as instructed

## H. Optional but helpful

- [ ] pyOpenSci-style docs polish
- [ ] Minimal logo / diagram of nested-CV workflow in docs (not required in paper)
- [ ] Demo data subset in-repo; full UCI copies via download script (license clarity)
- [ ] Badge: docs, CI, license, Zenodo DOI

## Explicit non-goals for JOSS

- Do not submit the LSSP/CompStat LaTeX PDF as the JOSS article  
- Do not claim novelty of nested CV itself  
- Do not lead with “maps win when designed to win” as a scientific discovery  

## Suggested timeline

| When | What |
|------|------|
| Week 0 | Public GitHub + LICENSE + package skeleton + README |
| Weeks 1–8 | API, tests, CI, examples, iterative commits |
| Months 2–6+ | Continue public development; draft `paper.md`; gather impact evidence |
| After ≥6 months public | Submit to JOSS |

**Earliest realistic JOSS submission:** after the 6-month public-history gate is satisfied (unless an existing public repo already covers that history).
