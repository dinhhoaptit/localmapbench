"""Quickstart: compare Raw vs Binary on the Blocks positive-control regime."""

from __future__ import annotations

from localmapbench import evaluate_maps_nested, make_regime, summarize_results
from localmapbench.protocol import DEFAULT_MAPS


def main() -> None:
    X, y = make_regime("blocks", n=120, d=8, seed=0)
    maps = {"Raw": DEFAULT_MAPS["Raw"], "Binary": DEFAULT_MAPS["Binary"]}
    results = evaluate_maps_nested(
        X, y, maps=maps, n_splits=3, random_state=0, max_candidates=6, n_repeats=1
    )
    summary = summarize_results(results)
    print("Mean outer-fold RMSE (Blocks regime):")
    for name, rmse in sorted(summary.items(), key=lambda kv: kv[1]):
        print(f"  {name:8s}  {rmse:.4f}")


if __name__ == "__main__":
    main()
