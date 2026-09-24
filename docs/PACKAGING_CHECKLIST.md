# Packaging checklist (before SoftwareX / JOSS / JSS)

Source experiments today live in `../IDA/`. For a software venue, promote a **named package** rather than a zip of scripts.

## Proposed package name

`localmapbench` (working title) — local-regression feature-map benchmarking under nested CV.

## Minimum viable package

- [ ] `pyproject.toml` / installable module (`pip install -e .`)
- [ ] Public API: feature maps, nested CV runner, 1-SE \(k\) rule, suitability diagnostic, regime generators
- [ ] CLI or notebook demos for one synthetic regime + one public dataset
- [ ] Unit tests for maps + fold isolation (no test leakage)
- [ ] Frozen small golden outputs for CI (not full \(R=30\) grids)
- [ ] LICENSE (MIT/BSD/Apache), CITATION.cff, CHANGELOG
- [ ] Docs: README + API page + “reproduce paper tables” section
- [ ] Zenodo archive of tagged release → DOI for Data/Software Availability

## Manuscript shape (SoftwareX-style)

1. Motivation: confounding of map vs \(k\); need matched protocol  
2. Software description / architecture  
3. Validation suite (five regimes as **positive controls**)  
4. Illustrative applications (workflow, not leaderboard)  
5. Impact / reuse / limitations  
6. Short related software comparison (what exists; what this adds)

Keep MC sensitivity grids in **supplement** or as optional long-running scripts, not as the abstract’s main claim.

## Explicitly reframe LSSP “expected findings”

Write one sentence in abstract/intro:

> The simulation ladder is a positive-control suite: each regime is designed so that a correctly specified map should dominate; the software is validated when that ranking is recovered under matched nested CV and broken under fixed-\(k\) or shuffled-order ablations.

That turns the editor’s critique into the validation story.
