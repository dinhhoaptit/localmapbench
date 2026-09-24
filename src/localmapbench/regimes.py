"""Positive-control synthetic regimes for validating feature-map choices."""

from __future__ import annotations

from typing import Callable

import numpy as np

from localmapbench.binary_kernel_transform import (
    generate_pattern_kernels,
    transform as binary_transform,
)


def _standardize_s(s: np.ndarray) -> np.ndarray:
    return (s - s.mean()) / (s.std() + 1e-12)


def _spread_indices(d: int, k: int) -> np.ndarray:
    k = min(k, d)
    if k == 1:
        return np.array([0], dtype=int)
    return np.unique(np.linspace(0, d - 1, num=k, dtype=int))


def make_blocks(
    n: int,
    d: int,
    rng: np.random.Generator,
    *,
    x_noise: float = 0.2,
    y_noise: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Response depends on multi-scale block averages (binary-map control)."""
    t = rng.uniform(0.0, 1.0, size=n)
    positions = np.linspace(0.0, 1.0, d)
    X = (
        t[:, None]
        + 0.35 * np.sin(2 * np.pi * (positions[None, :] + t[:, None]))
        + 0.25 * np.cos(4 * np.pi * positions[None, :] * (0.5 + t[:, None]))
    )
    X = X + rng.normal(0.0, x_noise, size=X.shape)
    bank = generate_pattern_kernels(d)
    picks = sorted(
        {
            0,
            min(1, len(bank) - 1),
            len(bank) // 3,
            (2 * len(bank)) // 3,
            len(bank) - 1,
        }
    )
    Phi_true = binary_transform(X, bank[picks])
    weights = np.linspace(1.0, -0.6, len(picks))
    s = _standardize_s(Phi_true @ weights)
    y = np.sin(2 * np.pi * s) + 0.15 * s + rng.normal(0.0, y_noise, size=n)
    return X, y


def make_scattered(
    n: int,
    d: int,
    rng: np.random.Generator,
    *,
    y_noise: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """Few distant relevant coordinates; patterned adjacency should not help."""
    X = rng.normal(0.0, 1.0, size=(n, d))
    i1, i2 = int(_spread_indices(d, 2)[0]), int(_spread_indices(d, 2)[-1])
    if i1 == i2 and d > 1:
        i2 = d - 1
    s = np.sin(2.0 * np.pi * X[:, i1]) + 0.7 * np.tanh(X[:, i2])
    y = _standardize_s(s) + rng.normal(0.0, y_noise, size=n)
    return X, y


def make_linear_index(
    n: int,
    d: int,
    rng: np.random.Generator,
    *,
    y_noise: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """y = f(beta^T x) with spread-out loadings (PLS/SIR control)."""
    X = rng.normal(0.0, 1.0, size=(n, d))
    idx = _spread_indices(d, min(4, d))
    beta = np.zeros(d)
    beta[idx] = rng.normal(0.0, 1.0, size=len(idx))
    beta /= np.linalg.norm(beta) + 1e-12
    s = X @ beta
    y = s**3 + rng.normal(0.0, y_noise, size=n)
    return X, y


def make_local_max(
    n: int,
    d: int,
    rng: np.random.Generator,
    *,
    y_noise: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """Local moving-maximum response (sliding-filter control)."""
    X = rng.normal(0.0, 1.0, size=(n, d))
    win = 5 if d >= 5 else max(2, d // 2)
    ma = np.stack(
        [X[:, j : j + win].mean(axis=1) for j in range(max(1, d - win + 1))],
        axis=1,
    )
    s = ma.max(axis=1)
    y = np.sin(2.0 * np.pi * _standardize_s(s)) + rng.normal(0.0, y_noise, size=n)
    return X, y


def make_pairwise(
    n: int,
    d: int,
    rng: np.random.Generator,
    *,
    y_noise: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """Product and absolute-difference interactions (pairwise control)."""
    X = rng.normal(0.0, 1.0, size=(n, d))
    idx = _spread_indices(d, min(4, d))
    while len(idx) < 4:
        idx = np.append(idx, len(idx) % d)
    s = X[:, idx[0]] * X[:, idx[1]] + 0.8 * np.abs(X[:, idx[2]] - X[:, idx[3]])
    y = _standardize_s(s) + rng.normal(0.0, y_noise, size=n)
    return X, y


REGIME_GENERATORS: dict[str, Callable[..., tuple[np.ndarray, np.ndarray]]] = {
    "blocks": make_blocks,
    "scattered": make_scattered,
    "linear_index": make_linear_index,
    "local_max": make_local_max,
    "pairwise": make_pairwise,
}


def make_regime(
    name: str,
    n: int = 200,
    d: int = 8,
    seed: int = 0,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a named positive-control regime."""
    if name not in REGIME_GENERATORS:
        raise KeyError(f"Unknown regime {name!r}; choose from {sorted(REGIME_GENERATORS)}")
    rng = np.random.default_rng(seed)
    return REGIME_GENERATORS[name](n, d, rng, **kwargs)
