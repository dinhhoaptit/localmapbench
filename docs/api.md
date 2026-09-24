# API overview

Install:

```bash
pip install -e ".[dev]"
```

## High-level entry points

| Function | Module | Role |
|----------|--------|------|
| `make_regime(name, n, d, seed)` | `regimes` | Positive-control synthetic data |
| `evaluate_maps_nested(X, y, ...)` | `protocol` | Outer nested CV over feature maps |
| `summarize_results(results)` | `protocol` | Mean RMSE by map |
| `select_optimum_k(X, y, ...)` | `optimum_k_selection` | 1-SE largest-$k$ on a training fold |
| `suitability_diagnostic(X, y, ...)` | `diagnostic` | Train-only patterned-map screen |

Default maps in `DEFAULT_MAPS`: Raw, Binary, Soft, Sliding, Pairwise, PLS, SIR.

## Regimes

`blocks`, `scattered`, `linear_index`, `local_max`, `pairwise`.

## Manual test procedure

```bash
pytest -q
python examples/quickstart_blocks.py
python examples/suitability_demo.py
```

Expected: tests pass; Blocks quickstart shows Binary RMSE below Raw on average.
