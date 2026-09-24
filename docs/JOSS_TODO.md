# JOSS to-do list (everything needed)

Order matters: **public repo timeline** is a hard gate.

## A. Pre-submission gates

- [x] Local git + iterative commits
- [x] Public GitHub: https://github.com/dinhhoaptit/localmapbench
- [ ] Keep **> 6 months** public iterative development before JOSS submission
- [x] OSI MIT LICENSE
- [x] Research-use evidence path: prior Monte Carlo study + frozen tables in `software/` (cite in paper impact; preprint/Zenodo optional)
- [x] Feature-complete library API (v0.2.0) — continue refining until submission

## B. Package

- [x] `src/localmapbench/` + `pyproject.toml`
- [x] Maps, nested CV, diagnostic, regimes
- [x] Examples under `examples/`
- [x] `pip install -e .` + deps including scipy
- [ ] Optional: TestPyPI / PyPI publish

## C. Documentation

- [x] Root README quickstart
- [x] `docs/api.md`
- [x] Examples: blocks + suitability
- [x] CONTRIBUTING, CODE_OF_CONDUCT, CITATION.cff, CHANGELOG

## D. Tests and quality

- [x] Unit tests (maps, k-selection, regimes, nested Raw, diagnostic)
- [x] GitHub Actions CI
- [x] Manual procedure in `docs/api.md`

## E. Open development

- [x] Iterative commits (ongoing for 6 months)
- [x] Tag `v0.2.0` release
- [x] Seed GitHub issues
- [ ] Continue monthly commits until submission window

## F. JOSS paper

- [x] Draft `paper.md` + `paper.bib`
- [x] Validation framing (not MC leaderboard)
- [ ] Polish wording after external feedback; update impact with DOI when available

## G. Archive and submit (after ≥6 months)

- [ ] GitHub Release for reviewed version
- [ ] Zenodo DOI
- [ ] JOSS submission form
- [ ] Respond to review
- [ ] Final tag + Zenodo update

## H. Optional

- [ ] Broader docs site
- [ ] Demo data download helpers
- [x] CI badge in README
