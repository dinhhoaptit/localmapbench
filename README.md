# localmapbench

Matched **nested cross-validation** benchmarking of feature preprocessors for **local regression**.

This repository is being prepared for a [*Journal of Open Source Software*](https://joss.theoj.org/) submission. Public development starts here to satisfy JOSS open-history requirements.

## Why this exists

Neighborhood size \(k\) and the feature map interact. Comparing maps at a fixed small \(k\) confounds representation quality with smoothing. `localmapbench` provides a shared nested-CV protocol, train-only preprocessing, and a one-standard-error largest-\(k\) rule so differences are attributable to the map.

## Status

**v0.1.0 (alpha)** — package skeleton + license. Core maps and runners are being migrated from the research scripts under `software/`.

## Install (development)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Quick check

```python
import localmapbench
print(localmapbench.__version__)
```

## Roadmap

1. Public GitHub history (this repo)  
2. Migrate feature maps, nested-CV runner, diagnostic, and regime generators into `src/localmapbench/`  
3. Tests + CI, examples, docs  
4. JOSS `paper.md` after ≥6 months of public iterative development  

## License

MIT — see [LICENSE](LICENSE).

## Author

Hoa Dinh Nguyen (Posts and Telecommunications Institute of Technology) — `hoand@ptit.edu.vn`
