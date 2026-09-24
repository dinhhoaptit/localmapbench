"""Train-only suitability diagnostic on two synthetic regimes."""

from __future__ import annotations

from localmapbench import make_regime, suitability_diagnostic


def main() -> None:
    for name in ("blocks", "scattered"):
        X, y = make_regime(name, n=100, d=8, seed=1)
        diag = suitability_diagnostic(X, y, include_sliding=True, random_state=1)
        print(f"\nRegime: {name}")
        print(f"  vs_raw (Binary): {diag['vs_raw']:+.3f}  recommend={diag['recommend_binary']}")
        print(
            f"  vs_raw (Sliding): {diag['vs_raw_sliding']:+.3f}  "
            f"recommend={diag['recommend_sliding']}"
        )


if __name__ == "__main__":
    main()
