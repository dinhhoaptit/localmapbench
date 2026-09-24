"""
Data-driven supervised projections for tabular regression (suggestion 4).

- PLS: partial least squares (sklearn)
- SIR: sliced inverse regression (Li, 1991)
- MLPEncoder: small MLP trained to predict y; bottleneck hidden layer is Phi
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


@dataclass
class LinearProjectionModel:
    """Phi = (X - mean) @ components  (components: d x k)."""

    mean_: np.ndarray
    components_: np.ndarray  # (d, k)
    name: str = "projection"

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) @ self.components_


@dataclass
class MLPEncoderModel:
    """Bottleneck activations of a fitted MLPRegressor."""

    scaler: StandardScaler
    mlp: MLPRegressor
    bottleneck_layer: int  # index into coefs_ for bottleneck output
    name: str = "mlp_encoder"

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        a = Xs
        # Forward through layers 0..bottleneck_layer inclusive
        for li in range(self.bottleneck_layer + 1):
            z = a @ self.mlp.coefs_[li] + self.mlp.intercepts_[li]
            # Hidden layers use activation; output layer of MLP is identity for regression
            if li < len(self.mlp.coefs_) - 1:
                act = self.mlp.activation
                if act == "relu":
                    a = np.maximum(z, 0.0)
                elif act == "tanh":
                    a = np.tanh(z)
                elif act == "logistic":
                    a = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
                else:
                    a = z
            else:
                a = z
        return a


def fit_pls(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int | None = None,
) -> LinearProjectionModel:
    n, d = X.shape
    k = n_components if n_components is not None else min(8, d, max(1, n // 20))
    k = int(max(1, min(k, d, n - 1)))
    mean = X.mean(axis=0)
    Xc = X - mean
    pls = PLSRegression(n_components=k, scale=False)
    pls.fit(Xc, y)
    # x_rotations_ maps X to X_scores
    comps = np.asarray(pls.x_rotations_, dtype=float)
    return LinearProjectionModel(mean_=mean, components_=comps, name="pls")


def fit_sir(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int | None = None,
    n_slices: int = 10,
) -> LinearProjectionModel:
    """Sliced inverse regression: leading eigenvectors of Cov(E[x|y])."""
    n, d = X.shape
    k = n_components if n_components is not None else min(8, d, max(1, n // 20))
    k = int(max(1, min(k, d, n - 1)))
    mean = X.mean(axis=0)
    Xc = X - mean

    # Standardize for SIR numerical stability
    std = Xc.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    Z = Xc / std

    n_slices = int(max(2, min(n_slices, n // 5, n)))
    # Equal-frequency slices on y
    order = np.argsort(y)
    edges = np.linspace(0, n, n_slices + 1, dtype=int)
    slice_means = []
    weights = []
    for s in range(n_slices):
        idx = order[edges[s] : edges[s + 1]]
        if len(idx) == 0:
            continue
        slice_means.append(Z[idx].mean(axis=0))
        weights.append(len(idx) / n)
    M = np.zeros((d, d), dtype=float)
    for w, mu in zip(weights, slice_means):
        M += w * np.outer(mu, mu)

    # Cov(Z) ~ I if well standardized; still use generalized eigen vs cov for safety
    Sigma = np.cov(Z, rowvar=False)
    Sigma = Sigma + 1e-6 * np.eye(d)
    # Solve M v = lambda Sigma v
    try:
        # Whitened eigen: Sigma^{-1/2} M Sigma^{-1/2}
        evals_s, evecs_s = np.linalg.eigh(Sigma)
        evals_s = np.maximum(evals_s, 1e-10)
        half = evecs_s * np.sqrt(1.0 / evals_s)
        Mw = half.T @ M @ half
        evals, evecs = np.linalg.eigh(Mw)
        # ascending eigh -> take largest k
        idx = np.argsort(evals)[::-1][:k]
        Vw = evecs[:, idx]
        # back to Z-space then to X-space: dir = Sigma^{-1/2} Vw, then /std
        Vz = half @ Vw
        comps = (Vz.T / std).T  # (d, k) applying to (X-mean)
    except np.linalg.LinAlgError:
        comps = np.eye(d, k)

    # Orthonormalize columns in X-metric roughly
    q, _ = np.linalg.qr(comps)
    comps = q[:, :k]
    return LinearProjectionModel(mean_=mean, components_=comps, name="sir")


def fit_mlp_encoder(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int | None = None,
    hidden: int = 32,
    max_iter: int = 300,
    random_state: int = 42,
) -> MLPEncoderModel:
    """Train MLP x -> hidden -> bottleneck(k) -> y; return bottleneck as Phi."""
    n, d = X.shape
    k = n_components if n_components is not None else min(8, d, max(2, n // 50))
    k = int(max(1, min(k, d, 16)))
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    # layers: d -> hidden -> k -> 1
    mlp = MLPRegressor(
        hidden_layer_sizes=(hidden, k),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=random_state,
    )
    mlp.fit(Xs, y)
    # coefs_: [d->h, h->k, k->1]; bottleneck is after layer index 1
    return MLPEncoderModel(
        scaler=scaler, mlp=mlp, bottleneck_layer=1, name="mlp_encoder"
    )


def default_n_components(d: int, n: int) -> int:
    return int(max(2, min(8, d, max(2, n // 30))))
