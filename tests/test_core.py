"""Unit tests for maps, regimes, k-selection, and nested protocol."""

from __future__ import annotations

import numpy as np

import localmapbench
from localmapbench.binary_kernel_transform import (
    generate_pattern_kernels,
    greedy_select_kernels,
    transform,
)
from localmapbench.optimum_k_selection import select_optimum_k
from localmapbench.protocol import DEFAULT_MAPS, evaluate_maps_nested, summarize_results
from localmapbench.regimes import make_regime


def test_version():
    assert localmapbench.__version__ == "0.2.0"


def test_binary_kernels_and_transform():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    y = X[:, :2].sum(axis=1)
    bank = generate_pattern_kernels(6)
    assert bank.ndim == 2 and bank.shape[1] == 6
    selected = greedy_select_kernels(X, y, bank, k_nn=5, max_kernels=4).kernels
    Phi = transform(X, selected)
    assert Phi.shape == (40, len(selected))


def test_select_optimum_k_train_only():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 4))
    y = X[:, 0] + 0.1 * rng.normal(size=80)
    sel = select_optimum_k(
        X, y, model="linear", n_splits=3, n_repeats=1, max_candidates=5, random_state=1
    )
    assert sel.k_min <= sel.k_star <= sel.k_max
    assert sel.k_star in sel.candidates or sel.k_star in sel.acceptable


def test_make_regime_shapes():
    for name in ("blocks", "scattered", "linear_index", "local_max", "pairwise"):
        X, y = make_regime(name, n=60, d=8, seed=2)
        assert X.shape == (60, 8)
        assert y.shape == (60,)


def test_nested_raw_only_fast():
    X, y = make_regime("scattered", n=90, d=6, seed=3)
    results = evaluate_maps_nested(
        X,
        y,
        maps={"Raw": DEFAULT_MAPS["Raw"]},
        n_splits=2,
        random_state=3,
        max_candidates=4,
        n_repeats=1,
    )
    summary = summarize_results(results)
    assert "Raw" in summary
    assert summary["Raw"] > 0


def test_suitability_diagnostic_runs():
    X, y = make_regime("blocks", n=80, d=8, seed=4)
    out = localmapbench.suitability_diagnostic(
        X, y, include_sliding=False, random_state=4
    )
    assert "vs_raw" in out
    assert "recommend_binary" in out
