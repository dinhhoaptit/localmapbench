# localmapbench

[![CI](https://github.com/dinhhoaptit/localmapbench/actions/workflows/ci.yml/badge.svg)](https://github.com/dinhhoaptit/localmapbench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Matched **nested cross-validation** benchmarking of feature preprocessors for **local regression**.

Public repository: https://github.com/dinhhoaptit/localmapbench  
Prepared for [*Journal of Open Source Software*](https://joss.theoj.org/) (submit after ≥6 months of public development).

## Why

Neighborhood size $k$ and the feature map interact. Comparing maps at a fixed small $k$ confounds representation quality with smoothing. `localmapbench` evaluates maps under shared nested folds, train-only fitting, and a one-standard-error largest-$k$ rule.

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Quickstart

```python
from localmapbench import evaluate_maps_nested, make_regime, summarize_results
from localmapbench.protocol import DEFAULT_MAPS

X, y = make_regime("blocks", n=120, d=8, seed=0)
maps = {"Raw": DEFAULT_MAPS["Raw"], "Binary": DEFAULT_MAPS["Binary"]}
summary = summarize_results(
    evaluate_maps_nested(X, y, maps=maps, n_splits=3, random_state=0)
)
print(summary)
```

Or:

```bash
python examples/quickstart_blocks.py
python examples/suitability_demo.py
pytest -q
```

## Features

- Feature maps: Raw, Binary/Soft blocks, Sliding, Pairwise, PLS, SIR
- Nested CV protocol with 1-SE largest-$k$ selection
- Five positive-control regimes
- Train-only suitability diagnostic
- Research scripts and frozen tables under `software/` (migration archive)

## Citation

See [CITATION.cff](CITATION.cff). JOSS `paper.md` is included for the forthcoming software paper.

## License

MIT — see [LICENSE](LICENSE).

## Author

Hoa Dinh Nguyen (Posts and Telecommunications Institute of Technology) — `hoand@ptit.edu.vn`
