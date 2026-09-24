"""
Binary patterned-mask feature transform for tabular regression.

Each kernel is a fixed 0/1 mask over a fixed feature order. The transform is
L1-normalized block aggregation (not CNN convolution). Kernels are selected
greedily to maximize local continuity of y in the transformed space; prediction
uses distance-weighted local linear regression in that space.

Limitations: expressivity is limited to linear combinations of block-sums;
performance depends on a meaningful feature order. Greedy selection is
suboptimal vs exhaustive 2^K search but is the feasible realization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors


def generate_pattern_kernels(
    d: int, max_order: int | None = None
) -> np.ndarray:
    """Generate patterned binary masks of shape (K, d).

    For block size m = 1..M with 2m <= d, produce m+1 phase-shifted period-2m
    masks (m ones, m zeros). Append the all-ones mask. Cap M at floor(d/2).
    """
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")
    M = d // 2 if max_order is None else min(int(max_order), d // 2)
    kernels: list[np.ndarray] = []
    for m in range(1, M + 1):
        period = 2 * m
        base = np.zeros(d, dtype=float)
        for i in range(d):
            if (i % period) < m:
                base[i] = 1.0
        for shift in range(m + 1):
            kernels.append(np.roll(base, shift))
    kernels.append(np.ones(d, dtype=float))
    # Drop exact duplicates (can occur for small d / large shifts wrapping)
    uniq: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for k in kernels:
        key = tuple(k.tolist())
        if key not in seen:
            seen.add(key)
            uniq.append(k)
    return np.asarray(uniq, dtype=float)


def transform(X: np.ndarray, kernels: np.ndarray) -> np.ndarray:
    """Apply L1-normalized mask aggregations: Phi[i,j] = (k_j · x_i) / ||k_j||_1."""
    if kernels.ndim != 2:
        raise ValueError("kernels must be 2D (K, d)")
    if X.shape[1] != kernels.shape[1]:
        raise ValueError(
            f"X dim {X.shape[1]} != kernel dim {kernels.shape[1]}"
        )
    norms = kernels.sum(axis=1)
    norms = np.maximum(norms, 1e-12)
    return (X @ kernels.T) / norms


def alignment_score(
    Phi: np.ndarray,
    y: np.ndarray,
    k_nn: int,
    exclude_self: bool = True,
) -> float:
    """Local continuity: negative mean variance of y among k_nn neighbors in Phi-space.

    Higher (less negative / closer to zero) is better.
    """
    n = len(y)
    if n < 2:
        return 0.0
    k_fetch = min(k_nn + (1 if exclude_self else 0), n)
    k_use = min(k_nn, n - (1 if exclude_self else 0))
    if k_use < 1:
        return 0.0

    nn = NearestNeighbors(n_neighbors=k_fetch, algorithm="auto")
    nn.fit(Phi)
    dists, idxs = nn.kneighbors(Phi, return_distance=True)

    vars_list: list[float] = []
    for i in range(n):
        d = dists[i]
        ix = idxs[i]
        if exclude_self:
            keep = d > 1e-15
            if not np.any(keep):
                keep = np.ones_like(d, dtype=bool)
                keep[0] = False
            ix = ix[keep][:k_use]
        else:
            ix = ix[:k_use]
        if len(ix) < 2:
            continue
        vars_list.append(float(np.var(y[ix])))
    if not vars_list:
        return 0.0
    return -float(np.mean(vars_list))


def spearman_distance_alignment(
    Phi: np.ndarray,
    y: np.ndarray,
    max_pairs: int = 5000,
    rng: np.random.Generator | None = None,
) -> float:
    """Spearman correlation between pairwise Phi distances and |Δy| (diagnostic)."""
    n = len(y)
    if n < 3:
        return 0.0
    rng = rng or np.random.default_rng(0)
    # Sample unordered pairs
    n_pairs = min(max_pairs, n * (n - 1) // 2)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    if len(i) < 10:
        return 0.0
    d_phi = np.linalg.norm(Phi[i] - Phi[j], axis=1)
    d_y = np.abs(y[i] - y[j])
    # Rank correlation without scipy
    r_phi = d_phi.argsort().argsort().astype(float)
    r_y = d_y.argsort().argsort().astype(float)
    r_phi -= r_phi.mean()
    r_y -= r_y.mean()
    denom = np.sqrt((r_phi**2).sum() * (r_y**2).sum())
    if denom < 1e-12:
        return 0.0
    return float((r_phi * r_y).sum() / denom)


@dataclass
class GreedyResult:
    kernels: np.ndarray
    indices: list[int]
    scores: list[float]
    final_score: float


def greedy_select_kernels(
    X: np.ndarray,
    y: np.ndarray,
    kernels: np.ndarray,
    k_nn: int,
    eps: float = 1e-6,
    max_kernels: int = 16,
) -> GreedyResult:
    """Greedily add masks that improve local continuity of y in Phi-space."""
    K = len(kernels)
    remaining = list(range(K))
    selected: list[int] = []
    score_hist: list[float] = []
    # Empty set: use a constant feature so NN is well-defined / baseline
    best_score = alignment_score(np.zeros((len(X), 1)), y, k_nn)
    score_hist.append(best_score)

    while remaining and len(selected) < max_kernels:
        best_idx = None
        best_cand_score = best_score
        for idx in remaining:
            cand = selected + [idx]
            Phi = transform(X, kernels[cand])
            s = alignment_score(Phi, y, k_nn)
            if s > best_cand_score + eps:
                best_cand_score = s
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
        best_score = best_cand_score
        score_hist.append(best_score)

    if not selected:
        # Fall back to all-ones if present, else first kernel
        ones = np.where(np.all(kernels == 1.0, axis=1))[0]
        fallback = int(ones[0]) if len(ones) else 0
        selected = [fallback]
        Phi = transform(X, kernels[selected])
        best_score = alignment_score(Phi, y, k_nn)
        score_hist.append(best_score)

    return GreedyResult(
        kernels=kernels[selected],
        indices=selected,
        scores=score_hist,
        final_score=best_score,
    )


def neighbor_weights(
    dists: np.ndarray,
    mode: str = "inverse",
    power: float = 2.0,
    bandwidth_factor: float | None = 1.0,
    eps: float = 1e-12,
) -> np.ndarray:
    d = np.asarray(dists, dtype=float)
    if mode == "uniform":
        return np.ones_like(d)
    if mode == "inverse":
        return 1.0 / (np.power(d, power) + eps)
    if mode == "gaussian":
        med = float(np.median(d)) if len(d) else 1.0
        if med < eps:
            med = 1.0
        factor = 1.0 if bandwidth_factor is None else float(bandwidth_factor)
        h = max(factor * med, eps)
        return np.exp(-((d / h) ** 2))
    raise ValueError(f"Unknown mode {mode}")


def _weighted_mean(yn: np.ndarray, w: np.ndarray) -> float:
    ww = w / (w.sum() + 1e-12)
    return float(np.dot(ww, yn))


def n_params_poly2(d: int) -> int:
    """Parameters in local quadratic: 1 + d + d(d+1)/2."""
    return 1 + d + d * (d + 1) // 2


def _quadratic_design(delta: np.ndarray) -> np.ndarray:
    """Rows: [1, delta, unique quadratic monomials]."""
    n_loc, d = delta.shape
    cols = [np.ones(n_loc), delta]
    quads = []
    for i in range(d):
        for j in range(i, d):
            quads.append(delta[:, i] * delta[:, j])
    if quads:
        cols.append(np.column_stack(quads))
    return np.column_stack(cols)


def _local_linear_at_query(
    Xn: np.ndarray,
    yn: np.ndarray,
    xq: np.ndarray,
    w: np.ndarray,
    ridge_alpha: float = 0.0,
) -> float:
    """Fit weighted local linear (or ridge) at query; return prediction at xq."""
    n_loc, n_dim = Xn.shape
    if n_loc < 2 or n_dim == 0:
        return _weighted_mean(yn, w)
    delta = Xn - xq
    sw = np.sqrt(np.maximum(w, 0.0))
    A = np.column_stack([np.ones(n_loc), delta])
    Aw = sw[:, None] * A
    yw = sw * yn
    if ridge_alpha > 0:
        ATA = Aw.T @ Aw
        ATy = Aw.T @ yw
        pen = np.ones(n_dim + 1) * ridge_alpha
        pen[0] = 0.0
        ATA = ATA + np.diag(pen)
        try:
            coeff = np.linalg.solve(ATA, ATy)
        except np.linalg.LinAlgError:
            return _weighted_mean(yn, w)
        return float(coeff[0])
    try:
        coeff, _, rank, _ = np.linalg.lstsq(Aw, yw, rcond=None)
        if rank < min(n_dim + 1, n_loc):
            return _weighted_mean(yn, w)
        return float(coeff[0])
    except np.linalg.LinAlgError:
        return _weighted_mean(yn, w)


def _local_poly2_at_query(
    Xn: np.ndarray,
    yn: np.ndarray,
    xq: np.ndarray,
    w: np.ndarray,
    ridge_alpha: float = 1.0,
) -> float:
    """Weighted local quadratic regression centered at the query."""
    n_loc, d = Xn.shape
    need = n_params_poly2(d)
    if n_loc < max(need, 3):
        return _local_linear_at_query(Xn, yn, xq, w, ridge_alpha=ridge_alpha)
    delta = Xn - xq
    A = _quadratic_design(delta)
    sw = np.sqrt(np.maximum(w, 0.0))
    Aw = sw[:, None] * A
    yw = sw * yn
    p = A.shape[1]
    ATA = Aw.T @ Aw
    ATy = Aw.T @ yw
    pen = np.ones(p) * float(ridge_alpha)
    pen[0] = 0.0
    ATA = ATA + np.diag(pen)
    try:
        coeff = np.linalg.solve(ATA, ATy)
        return float(coeff[0])
    except np.linalg.LinAlgError:
        return _local_linear_at_query(Xn, yn, xq, w, ridge_alpha=ridge_alpha)


def _local_kernel_ridge_at_query(
    Xn: np.ndarray,
    yn: np.ndarray,
    xq: np.ndarray,
    w: np.ndarray,
    ridge_alpha: float = 1.0,
) -> float:
    """RBF kernel ridge on the local neighborhood."""
    from sklearn.kernel_ridge import KernelRidge

    if len(yn) < 3:
        return _weighted_mean(yn, w)
    # Scale-free gamma from local pairwise distances
    if len(yn) >= 2:
        dif = Xn - Xn.mean(axis=0)
        med2 = float(np.median(np.sum(dif * dif, axis=1))) + 1e-12
        gamma = 1.0 / (2.0 * med2)
    else:
        gamma = 1.0
    try:
        model = KernelRidge(alpha=max(ridge_alpha, 1e-6), kernel="rbf", gamma=gamma)
        model.fit(Xn, yn, sample_weight=np.maximum(w, 1e-12))
        pred = model.predict(xq.reshape(1, -1))
        val = float(pred[0])
        return val if np.isfinite(val) else _weighted_mean(yn, w)
    except Exception:
        return _weighted_mean(yn, w)


def _local_svr_at_query(
    Xn: np.ndarray,
    yn: np.ndarray,
    xq: np.ndarray,
    w: np.ndarray,
) -> float:
    """RBF SVR on the local neighborhood."""
    from sklearn.svm import SVR

    if len(yn) < 3:
        return _weighted_mean(yn, w)
    try:
        model = SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")
        model.fit(Xn, yn, sample_weight=np.maximum(w, 1e-12))
        pred = model.predict(xq.reshape(1, -1))
        val = float(pred[0])
        return val if np.isfinite(val) else _weighted_mean(yn, w)
    except Exception:
        return _weighted_mean(yn, w)


LOCAL_MODEL_NAMES = (
    "mean",
    "linear",
    "ridge",
    "poly2",
    "kernel_ridge",
    "svr",
)


def predict_local(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k_pred: int,
    model: str = "linear",
    weight_mode: str = "inverse",
    power: float = 2.0,
    bandwidth_factor: float | None = 1.0,
    ridge_alpha: float = 1.0,
) -> np.ndarray:
    """Local neighborhood estimators in feature space.

    model:
      - "mean": distance-weighted local constant (Nadaraya–Watson)
      - "linear": distance-weighted local linear regression
      - "ridge": distance-weighted local ridge (penalized slopes)
      - "poly2": weighted local quadratic polynomial
      - "kernel_ridge": local RBF kernel ridge
      - "svr": local RBF SVR
    """
    model = model.lower()
    if model not in LOCAL_MODEL_NAMES:
        raise ValueError(f"Unknown local model {model}")

    n_train, n_dim = X_train.shape
    n_test = X_test.shape[0]
    k_fetch = min(max(int(k_pred), 1), n_train)

    nn = NearestNeighbors(n_neighbors=k_fetch, algorithm="auto")
    nn.fit(X_train)
    dists, idxs = nn.kneighbors(X_test, return_distance=True)

    preds = np.zeros(n_test, dtype=float)
    for i in range(n_test):
        d = dists[i]
        ix = idxs[i]
        Xn = X_train[ix]
        yn = y_train[ix]
        w = neighbor_weights(d, weight_mode, power, bandwidth_factor)
        xq = X_test[i]

        if model == "mean" or len(ix) < 2:
            preds[i] = _weighted_mean(yn, w)
        elif model == "linear":
            if len(ix) < n_dim + 1:
                preds[i] = _weighted_mean(yn, w)
            else:
                preds[i] = _local_linear_at_query(Xn, yn, xq, w, ridge_alpha=0.0)
        elif model == "ridge":
            preds[i] = _local_linear_at_query(
                Xn, yn, xq, w, ridge_alpha=float(ridge_alpha)
            )
        elif model == "poly2":
            preds[i] = _local_poly2_at_query(
                Xn, yn, xq, w, ridge_alpha=float(ridge_alpha)
            )
        elif model == "kernel_ridge":
            preds[i] = _local_kernel_ridge_at_query(
                Xn, yn, xq, w, ridge_alpha=float(ridge_alpha)
            )
        else:  # svr
            preds[i] = _local_svr_at_query(Xn, yn, xq, w)
    return preds


def predict_local_linear(
    Phi_train: np.ndarray,
    y_train: np.ndarray,
    Phi_test: np.ndarray,
    k_pred: int,
    weight_mode: str = "inverse",
    power: float = 2.0,
    bandwidth_factor: float | None = 1.0,
) -> np.ndarray:
    """Distance-weighted local linear regression (wrapper around predict_local)."""
    return predict_local(
        Phi_train,
        y_train,
        Phi_test,
        k_pred=k_pred,
        model="linear",
        weight_mode=weight_mode,
        power=power,
        bandwidth_factor=bandwidth_factor,
    )


def predict_local_mean(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k_pred: int,
    weight_mode: str = "inverse",
    power: float = 2.0,
    bandwidth_factor: float | None = 1.0,
) -> np.ndarray:
    """Distance-weighted local constant / Nadaraya–Watson."""
    return predict_local(
        X_train,
        y_train,
        X_test,
        k_pred=k_pred,
        model="mean",
        weight_mode=weight_mode,
        power=power,
        bandwidth_factor=bandwidth_factor,
    )


def predict_local_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k_pred: int,
    ridge_alpha: float = 1.0,
    weight_mode: str = "inverse",
    power: float = 2.0,
    bandwidth_factor: float | None = 1.0,
) -> np.ndarray:
    """Distance-weighted local ridge regression."""
    return predict_local(
        X_train,
        y_train,
        X_test,
        k_pred=k_pred,
        model="ridge",
        weight_mode=weight_mode,
        power=power,
        bandwidth_factor=bandwidth_factor,
        ridge_alpha=ridge_alpha,
    )


def k_grid_for_dim(d: int, n: int) -> list[int]:
    """Candidate neighborhood sizes spanning small-k to large-k regimes."""
    raw = [
        d + 1,
        d + 2,
        d + 4,
        2 * d,
        3 * d,
        5 * d,
        10,
        15,
        20,
        30,
        50,
        80,
        max(3, int(np.sqrt(n))),
        max(3, int(n * 0.05)),
        max(3, int(n * 0.1)),
    ]
    ks = sorted({int(k) for k in raw if 2 <= int(k) < n})
    return ks


def default_k_nn(d: int, n: int, mult: int = 2, cap: int = 40) -> int:
    """Default neighborhood size ≈ 2d, capped, and < n."""
    return int(max(3, min(cap, mult * d, n - 1)))
