"""
Feature seriation / column ordering for patterned binary-kernel transforms.

Pattern masks assume adjacent columns are related. These heuristics reorder
tabular features so that similar (and optionally similarly y-relevant) columns
sit near each other.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform


def _abs_corr_matrix(X: np.ndarray) -> np.ndarray:
    """Pairwise |Pearson| correlation; NaNs -> 0; diagonal forced to 1."""
    d = X.shape[1]
    # Guard zero-variance columns
    Xc = X.copy()
    for j in range(d):
        if np.std(Xc[:, j]) < 1e-12:
            Xc[:, j] = 0.0
    C = np.corrcoef(Xc, rowvar=False)
    if C.ndim == 0:
        C = np.array([[1.0]])
    C = np.nan_to_num(np.abs(C), nan=0.0)
    np.fill_diagonal(C, 1.0)
    return C


def _feature_y_abs_corr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    out = np.zeros(d, dtype=float)
    y_std = float(np.std(y))
    if y_std < 1e-12:
        return out
    for j in range(d):
        xj = X[:, j]
        if np.std(xj) < 1e-12:
            continue
        c = np.corrcoef(xj, y)[0, 1]
        out[j] = 0.0 if not np.isfinite(c) else abs(float(c))
    return out


def order_native(d: int) -> np.ndarray:
    return np.arange(d, dtype=int)


def order_corr_abs_y(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Descending |corr(x_j, y)|."""
    return np.argsort(_feature_y_abs_corr(X, y))[::-1]


def order_spectral(X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    """Fiedler (spectral) seriation on |feature-feature correlation| affinity.

    y is unused; kept for a uniform (X, y) API.
    """
    del y
    S = _abs_corr_matrix(X)
    d = S.shape[0]
    if d <= 2:
        return np.arange(d, dtype=int)
    deg = S.sum(axis=1)
    L = np.diag(deg) - S
    # Symmetric eigendecomposition; Fiedler = 2nd smallest eigenvector
    vals, vecs = np.linalg.eigh(L)
    # vals ascending
    fiedler = vecs[:, 1]
    return np.argsort(fiedler)


def order_hierarchical(X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    """Hierarchical clustering leaf order on distance = 1 - |corr|."""
    del y
    S = _abs_corr_matrix(X)
    d = S.shape[0]
    if d <= 2:
        return np.arange(d, dtype=int)
    dist = np.clip(1.0 - S, 0.0, None)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    Z = optimal_leaf_ordering(Z, condensed)
    return leaves_list(Z).astype(int)


def order_nn_tour(X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    """Greedy nearest-neighbor path (open TSP) on distance = 1 - |corr|."""
    del y
    S = _abs_corr_matrix(X)
    d = S.shape[0]
    if d <= 2:
        return np.arange(d, dtype=int)
    dist = np.clip(1.0 - S, 0.0, None)
    np.fill_diagonal(dist, np.inf)
    # Start from feature with highest mean similarity (lowest mean dist)
    mean_d = np.mean(np.where(np.isfinite(dist), dist, 0.0), axis=1)
    start = int(np.argmin(mean_d))
    used = np.zeros(d, dtype=bool)
    path = [start]
    used[start] = True
    for _ in range(d - 1):
        cur = path[-1]
        cand = np.where(~used)[0]
        nxt = int(cand[np.argmin(dist[cur, cand])])
        path.append(nxt)
        used[nxt] = True
    return np.asarray(path, dtype=int)


def order_y_aware_spectral(
    X: np.ndarray, y: np.ndarray, lam: float = 1.0
) -> np.ndarray:
    """Spectral seriation with y-aware affinity.

    Affinity blends feature similarity with closeness of |corr to y|:
      A_ij = |corr_ij| * exp(-lam * ||r_i - r_j||)
    then Fiedler order on the Laplacian of A.
    """
    S = _abs_corr_matrix(X)
    r = _feature_y_abs_corr(X, y)
    d = S.shape[0]
    if d <= 2:
        return np.arange(d, dtype=int)
    dr = np.abs(r[:, None] - r[None, :])
    A = S * np.exp(-lam * dr)
    np.fill_diagonal(A, 1.0)
    deg = A.sum(axis=1)
    L = np.diag(deg) - A
    _, vecs = np.linalg.eigh(L)
    return np.argsort(vecs[:, 1])


def order_y_aware_nn(
    X: np.ndarray, y: np.ndarray, lam: float = 1.0
) -> np.ndarray:
    """NN tour on distance = (1-|corr|) + lam * ||r_i-r_j||."""
    S = _abs_corr_matrix(X)
    r = _feature_y_abs_corr(X, y)
    d = S.shape[0]
    if d <= 2:
        return np.arange(d, dtype=int)
    dist = np.clip(1.0 - S, 0.0, None) + lam * np.abs(r[:, None] - r[None, :])
    np.fill_diagonal(dist, np.inf)
    mean_d = np.mean(np.where(np.isfinite(dist), dist, 0.0), axis=1)
    start = int(np.argmin(mean_d))
    used = np.zeros(d, dtype=bool)
    path = [start]
    used[start] = True
    for _ in range(d - 1):
        cur = path[-1]
        cand = np.where(~used)[0]
        nxt = int(cand[np.argmin(dist[cur, cand])])
        path.append(nxt)
        used[nxt] = True
    return np.asarray(path, dtype=int)


OrderFn = Callable[..., np.ndarray]

# Name -> factory(X, y) -> order indices
ORDERING_METHODS: dict[str, OrderFn] = {
    "native": lambda X, y: order_native(X.shape[1]),
    "corr_abs_y": order_corr_abs_y,
    "spectral": order_spectral,
    "hierarchical": order_hierarchical,
    "nn_tour": order_nn_tour,
    "y_aware_spectral": order_y_aware_spectral,
    "y_aware_nn": order_y_aware_nn,
}


def apply_order(X: np.ndarray, order: np.ndarray) -> np.ndarray:
    return X[:, order]


def reverse_order(order: np.ndarray) -> np.ndarray:
    return order[::-1].copy()


def select_best_orientation(
    X: np.ndarray,
    y: np.ndarray,
    order: np.ndarray,
    score_fn: Callable[[np.ndarray, np.ndarray], float],
) -> np.ndarray:
    """Choose order vs reversed order by a train-only score (higher is better)."""
    s_fwd = score_fn(X[:, order], y)
    s_rev = score_fn(X[:, order[::-1]], y)
    return order if s_fwd >= s_rev else order[::-1]


def kendall_tau_order(true_order: np.ndarray, est_order: np.ndarray) -> float:
    """Kendall tau between two permutations of 0..d-1 (as rank sequences)."""
    d = len(true_order)
    # Position of each feature id in estimated order
    pos = np.empty(d, dtype=int)
    pos[est_order] = np.arange(d)
    ranks = pos[true_order]
    concord = 0
    total = 0
    for i in range(d):
        for j in range(i + 1, d):
            total += 1
            if (ranks[i] - ranks[j]) * (i - j) > 0:
                concord += 1
            elif (ranks[i] - ranks[j]) * (i - j) < 0:
                pass
            else:
                concord += 0.5
    if total == 0:
        return 1.0
    # tau = (C - D) / total; C + D = total => tau = 2C/total - 1
    return float(2.0 * concord / total - 1.0)
