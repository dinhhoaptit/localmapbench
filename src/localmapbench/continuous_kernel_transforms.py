"""
Continuous-valued and order-free alternatives to binary patterned kernels.

1) SoftMaskTransform
   Continuous masks w in (0,1)^d, L1-normalized aggregations.
   Initialized from binary pattern kernels (greedy-selected), then logits
   are fine-tuned by minimizing train Ridge-CV RMSE (supervised, fast).

2) SlidingConvTransform
   After features are ordered (seriation), apply 1D sliding filters of
   lengths L in {3,5,...}. For each filter, mean/max pool over positions
   to get compact features. Filter weights are learned the same way.

3) PairwiseModel
   Order-free bilinear features: products x_i x_j and |x_i-x_j| for top
   pairs ranked by supervised |corr| with y (among top marginal features).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from localmapbench.binary_kernel_transform import (
    alignment_score,
    default_k_nn,
    generate_pattern_kernels,
    greedy_select_kernels,
    transform as binary_transform,
)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p) - np.log(1.0 - p)


def soft_mask_transform(X: np.ndarray, logits: np.ndarray) -> np.ndarray:
    """Phi = X @ W.T / ||W||_1 with W = sigmoid(logits), shape (n, K)."""
    W = _sigmoid(logits)
    norms = np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
    Wn = W / norms
    return X @ Wn.T


def sliding_conv_features(
    X: np.ndarray,
    filters: dict[int, np.ndarray],
) -> np.ndarray:
    """Apply sliding filters; pool mean+max over positions per filter.

    filters: map length L -> array (n_filters, L)
    Output dim = 2 * sum_L n_filters(L)
    """
    n, d = X.shape
    cols: list[np.ndarray] = []
    for L, W in filters.items():
        if W.size == 0:
            continue
        W = np.asarray(W, dtype=float)
        if W.ndim == 1:
            W = W.reshape(1, -1)
        n_f, Lw = W.shape
        assert Lw == L
        if d < L:
            # Degenerate: global weighted sum with truncated/padded filter
            w = W[:, :d] if d > 0 else W
            # mean over available dims
            for j in range(n_f):
                ww = w[j]
                s = float(np.sum(np.abs(ww))) + 1e-12
                agg = (X * (ww / s)).sum(axis=1)
                cols.append(agg)
                cols.append(agg.copy())
            continue
        n_pos = d - L + 1
        # responses[j, t, i] expensive; compute per filter
        for j in range(n_f):
            resp = np.zeros((n, n_pos), dtype=float)
            wj = W[j]
            for t in range(n_pos):
                resp[:, t] = X[:, t : t + L] @ wj
            cols.append(resp.mean(axis=1))
            cols.append(resp.max(axis=1))
    if not cols:
        return np.zeros((n, 1), dtype=float)
    return np.column_stack(cols)


def _ridge_cv_rmse(Phi: np.ndarray, y: np.ndarray, n_splits: int = 3) -> float:
    if Phi.ndim != 2 or Phi.shape[1] == 0:
        return float(np.std(y) + 1.0)
    Phi_s = StandardScaler().fit_transform(Phi)
    kf = KFold(n_splits=min(n_splits, len(y)), shuffle=True, random_state=0)
    rmses = []
    for tr, te in kf.split(Phi_s):
        model = RidgeCV(alphas=(0.1, 1.0, 10.0))
        model.fit(Phi_s[tr], y[tr])
        pred = model.predict(Phi_s[te])
        rmses.append(float(np.sqrt(np.mean((pred - y[te]) ** 2))))
    return float(np.mean(rmses))


@dataclass
class SoftMaskModel:
    logits: np.ndarray  # (K, d)
    init_binary: np.ndarray
    train_loss: float

    def transform(self, X: np.ndarray) -> np.ndarray:
        return soft_mask_transform(X, self.logits)

    @property
    def weights(self) -> np.ndarray:
        return _sigmoid(self.logits)


@dataclass
class SlidingConvModel:
    filters: dict[int, np.ndarray]  # L -> (n_filters, L)
    train_loss: float

    def transform(self, X: np.ndarray) -> np.ndarray:
        return sliding_conv_features(X, self.filters)


def fit_soft_masks(
    X: np.ndarray,
    y: np.ndarray,
    max_order: int = 6,
    max_kernels: int = 8,
    n_iters: int = 40,
    random_state: int = 42,
) -> SoftMaskModel:
    """Init from greedy binary masks; fine-tune continuous logits via Ridge-CV."""
    n, d = X.shape
    k_nn = default_k_nn(min(d, 20), n)
    bank = generate_pattern_kernels(d, max_order=min(max_order, max(1, d // 2)))
    greedy = greedy_select_kernels(X, y, bank, k_nn=k_nn, max_kernels=max_kernels)
    init = greedy.kernels.copy()
    if len(init) == 0:
        init = np.ones((1, d), dtype=float)
    logits0 = _logit(init)

    flat0 = logits0.ravel().copy()
    shape = logits0.shape

    def pack(v: np.ndarray) -> np.ndarray:
        return soft_mask_transform(X, v.reshape(shape))

    def objective(v: np.ndarray) -> float:
        Phi = pack(v)
        # Mild L2 on logits to keep masks smooth / near init
        reg = 1e-3 * float(np.mean((v - flat0) ** 2))
        return _ridge_cv_rmse(Phi, y) + reg

    # Continuity warm-start diagnostic (not optimized directly — Ridge-CV is primary)
    _ = alignment_score(binary_transform(X, init), y, k_nn)

    res = minimize(
        objective,
        flat0,
        method="L-BFGS-B",
        options={"maxiter": n_iters, "ftol": 1e-4, "disp": False},
    )
    logits = res.x.reshape(shape)
    loss = float(res.fun)
    return SoftMaskModel(logits=logits, init_binary=init, train_loss=loss)


def fit_sliding_conv(
    X: np.ndarray,
    y: np.ndarray,
    lengths: tuple[int, ...] = (3, 5),
    n_filters: int = 4,
    n_iters: int = 50,
    random_state: int = 42,
) -> SlidingConvModel:
    """Learn sliding 1D filters on an already-ordered feature matrix."""
    rng = np.random.default_rng(random_state)
    n, d = X.shape
    lengths = tuple(L for L in lengths if L >= 1)
    if not lengths:
        lengths = (3,)

    # Parameter packing: concatenate all filter weights
    pieces: list[tuple[int, int, int]] = []  # (L, n_f, offset)
    vals: list[float] = []
    for L in lengths:
        n_f = n_filters
        # Prefer odd-symmetric-ish init: center bump
        W0 = rng.normal(0.0, 0.1, size=(n_f, L))
        for j in range(n_f):
            W0[j] += np.exp(-0.5 * ((np.arange(L) - (L - 1) / 2) / max(L / 3, 1)) ** 2)
        pieces.append((L, n_f, len(vals)))
        vals.extend(W0.ravel().tolist())
    flat0 = np.asarray(vals, dtype=float)

    def unpack(v: np.ndarray) -> dict[int, np.ndarray]:
        filters: dict[int, np.ndarray] = {}
        for L, n_f, off in pieces:
            size = n_f * L
            filters[L] = v[off : off + size].reshape(n_f, L)
        return filters

    def objective(v: np.ndarray) -> float:
        Phi = sliding_conv_features(X, unpack(v))
        reg = 1e-3 * float(np.mean(v**2))
        return _ridge_cv_rmse(Phi, y) + reg

    res = minimize(
        objective,
        flat0,
        method="L-BFGS-B",
        options={"maxiter": n_iters, "ftol": 1e-4, "disp": False},
    )
    filters = unpack(res.x)
    return SlidingConvModel(filters=filters, train_loss=float(res.fun))


# ---------------------------------------------------------------------------
# 3) Pairwise / bilinear features (order-free)
# ---------------------------------------------------------------------------


@dataclass
class PairwiseModel:
    """Selected feature pairs and which interaction types to emit."""

    pairs: list[tuple[int, int]]
    use_product: bool = True
    use_absdiff: bool = True
    include_raw: bool = False

    def transform(self, X: np.ndarray) -> np.ndarray:
        cols: list[np.ndarray] = []
        if self.include_raw:
            cols.append(X)
        for i, j in self.pairs:
            if self.use_product:
                cols.append(X[:, i] * X[:, j])
            if self.use_absdiff:
                cols.append(np.abs(X[:, i] - X[:, j]))
        if not cols:
            return np.zeros((X.shape[0], 1), dtype=float)
        return np.column_stack(cols)


def _abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return 0.0 if not np.isfinite(c) else abs(float(c))


def fit_pairwise_features(
    X: np.ndarray,
    y: np.ndarray,
    max_pairs: int = 30,
    top_features: int | None = None,
    use_product: bool = True,
    use_absdiff: bool = True,
    include_raw: bool = False,
) -> PairwiseModel:
    """Select top pairs by supervised |corr| of product and/or |diff| with y.

    For large d, only form pairs among the top_features columns by |corr(x_j,y)|.
    """
    n, d = X.shape
    if top_features is None:
        top_features = min(d, max(12, int(np.ceil(np.sqrt(2 * max_pairs) + 3))))
    top_features = min(d, max(2, int(top_features)))

    feat_scores = np.array([_abs_corr(X[:, j], y) for j in range(d)])
    ranked = np.argsort(feat_scores)[::-1][:top_features]

    scored: list[tuple[float, int, int]] = []
    for a in range(len(ranked)):
        for b in range(a + 1, len(ranked)):
            i, j = int(ranked[a]), int(ranked[b])
            score = 0.0
            if use_product:
                score = max(score, _abs_corr(X[:, i] * X[:, j], y))
            if use_absdiff:
                score = max(score, _abs_corr(np.abs(X[:, i] - X[:, j]), y))
            scored.append((score, i, j))
    scored.sort(key=lambda t: t[0], reverse=True)
    pairs = [(i, j) for _, i, j in scored[:max_pairs]]
    if not pairs and d >= 2:
        pairs = [(0, 1)]
    return PairwiseModel(
        pairs=pairs,
        use_product=use_product,
        use_absdiff=use_absdiff,
        include_raw=include_raw,
    )
