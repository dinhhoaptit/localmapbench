"""Train-only 1-SE largest-k selection on one synthetic regime."""

from __future__ import annotations

from localmapbench import make_regime, select_optimum_k


def main() -> None:
    X, y = make_regime("blocks", n=120, d=8, seed=0)
    sel = select_optimum_k(
        X,
        y,
        model="linear",
        n_splits=3,
        n_repeats=1,
        max_candidates=6,
        random_state=0,
    )
    print(
        f"Selected k: {sel.k_star}  "
        f"(feasible band {sel.k_min}..{sel.k_max}, rule {sel.rule})"
    )
    print(f"Risk-minimizing k: {sel.k0_argmin}")
    print(f"Acceptable set: {sel.acceptable}")


if __name__ == "__main__":
    main()
