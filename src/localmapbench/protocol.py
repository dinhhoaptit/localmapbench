"""Matched nested cross-validation protocol for local regression feature maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from localmapbench.binary_kernel_transform import (
    default_k_nn,
    generate_pattern_kernels,
    greedy_select_kernels,
    predict_local,
    transform as binary_transform,
)
from localmapbench.continuous_kernel_transforms import (
    fit_pairwise_features,
    fit_sliding_conv,
    fit_soft_masks,
)
from localmapbench.optimum_k_selection import select_optimum_k
from localmapbench.supervised_projections import default_n_components, fit_pls, fit_sir


MapFitter = Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass
class FoldResult:
    fold: int
    map_name: str
    rmse: float
    k_star: int


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def fit_raw_map(
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    del y_tr
    return X_tr, X_te


def fit_binary_map(
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    d = X_tr.shape[1]
    bank = generate_pattern_kernels(d)
    selected = greedy_select_kernels(
        X_tr, y_tr, bank, k_nn=default_k_nn(d, len(X_tr))
    ).kernels
    return binary_transform(X_tr, selected), binary_transform(X_te, selected)


def fit_soft_map(
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    model = fit_soft_masks(X_tr, y_tr)
    return model.transform(X_tr), model.transform(X_te)


def fit_sliding_map(
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    model = fit_sliding_conv(X_tr, y_tr)
    return model.transform(X_tr), model.transform(X_te)


def fit_pairwise_map(
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    model = fit_pairwise_features(X_tr, y_tr)
    return model.transform(X_tr), model.transform(X_te)


def fit_pls_map(
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    k = default_n_components(X_tr.shape[1], X_tr.shape[0])
    model = fit_pls(X_tr, y_tr, n_components=k)
    return model.transform(X_tr), model.transform(X_te)


def fit_sir_map(
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    k = default_n_components(X_tr.shape[1], X_tr.shape[0])
    model = fit_sir(X_tr, y_tr, n_components=k)
    return model.transform(X_tr), model.transform(X_te)


DEFAULT_MAPS: dict[str, MapFitter] = {
    "Raw": fit_raw_map,
    "Binary": fit_binary_map,
    "Soft": fit_soft_map,
    "Sliding": fit_sliding_map,
    "Pairwise": fit_pairwise_map,
    "PLS": fit_pls_map,
    "SIR": fit_sir_map,
}


def evaluate_maps_nested(
    X: np.ndarray,
    y: np.ndarray,
    *,
    maps: dict[str, MapFitter] | None = None,
    n_splits: int = 3,
    random_state: int = 0,
    local_model: str = "linear",
    max_candidates: int = 8,
    n_repeats: int = 1,
) -> list[FoldResult]:
    """Outer nested CV: fit map + select k on train only; score on test fold."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    maps = maps or DEFAULT_MAPS
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results: list[FoldResult] = []

    for fold, (tr, te) in enumerate(kf.split(X)):
        scaler = StandardScaler().fit(X[tr])
        X_tr = scaler.transform(X[tr])
        X_te = scaler.transform(X[te])
        y_tr, y_te = y[tr], y[te]

        for name, fitter in maps.items():
            F_tr, F_te = fitter(X_tr, y_tr, X_te)
            sel = select_optimum_k(
                F_tr,
                y_tr,
                model=local_model,
                n_splits=min(3, max(2, len(tr) // 20)),
                n_repeats=n_repeats,
                random_state=random_state + fold,
                max_candidates=max_candidates,
            )
            y_hat = predict_local(
                F_tr,
                y_tr,
                F_te,
                k_pred=sel.k_star,
                model=local_model,
            )
            results.append(
                FoldResult(
                    fold=fold,
                    map_name=name,
                    rmse=_rmse(y_te, y_hat),
                    k_star=int(sel.k_star),
                )
            )
    return results


def summarize_results(results: list[FoldResult]) -> dict[str, float]:
    """Mean RMSE by map name."""
    names = sorted({r.map_name for r in results})
    out: dict[str, float] = {}
    for name in names:
        vals = [r.rmse for r in results if r.map_name == name]
        out[name] = float(np.mean(vals))
    return out
