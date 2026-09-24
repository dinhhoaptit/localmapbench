"""localmapbench: nested-CV benchmarking of feature maps for local regression."""

from localmapbench.diagnostic import suitability_diagnostic
from localmapbench.optimum_k_selection import select_optimum_k
from localmapbench.protocol import (
    DEFAULT_MAPS,
    evaluate_maps_nested,
    summarize_results,
)
from localmapbench.regimes import REGIME_GENERATORS, make_regime

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "make_regime",
    "REGIME_GENERATORS",
    "evaluate_maps_nested",
    "summarize_results",
    "DEFAULT_MAPS",
    "select_optimum_k",
    "suitability_diagnostic",
]
