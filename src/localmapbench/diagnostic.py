"""Train-only suitability screen for patterned feature maps."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from localmapbench.binary_kernel_transform import (
    alignment_score,
    default_k_nn,
    generate_pattern_kernels,
    greedy_select_kernels,
    predict_local,
    transform as binary_transform,
)
from localmapbench.continuous_kernel_transforms import fit_sliding_conv
from localmapbench.optimum_k_selection import select_optimum_k


def _inner_rmse(
    F: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int,
    n_splits: int = 3,
) -> float:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    errs: list[float] = []
    for tr, te in kf.split(F):
        sel = select_optimum_k(
            F[tr],
            y[tr],
            model="linear",
            n_splits=min(3, max(2, len(tr) // 15)),
            n_repeats=1,
            random_state=random_state,
            max_candidates=6,
        )
        y_hat = predict_local(F[tr], y[tr], F[te], k_pred=sel.k_star, model="linear")
        errs.append(float(np.sqrt(np.mean((y[te] - y_hat) ** 2))))
    return float(np.mean(errs))


def suitability_diagnostic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    shuffle_seed: int = 0,
    random_state: int = 0,
    include_sliding: bool = True,
) -> dict:
    """Train-only screen: does a patterned map beat Raw under matched opt-k?

    Returns continuity scores, relative risk lifts, and recommend_* flags.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    Xs = StandardScaler().fit_transform(X)
    n, d = Xs.shape
    k_nn = default_k_nn(d, n)
    bank = generate_pattern_kernels(d)

    greedy_nat = greedy_select_kernels(Xs, y, bank, k_nn=k_nn)
    Phi_nat = binary_transform(Xs, greedy_nat.kernels)
    s_nat = alignment_score(Phi_nat, y, k_nn)

    rng = np.random.default_rng(shuffle_seed)
    perm = rng.permutation(d)
    Xs_shuf = Xs[:, perm]
    greedy_shuf = greedy_select_kernels(Xs_shuf, y, bank, k_nn=k_nn)
    Phi_shuf = binary_transform(Xs_shuf, greedy_shuf.kernels)
    s_shuf = alignment_score(Phi_shuf, y, k_nn)

    rmse_raw = _inner_rmse(Xs, y, random_state=random_state)
    rmse_bin = _inner_rmse(Phi_nat, y, random_state=random_state + 1)
    vs_raw = (rmse_raw - rmse_bin) / max(rmse_raw, 1e-12)

    out: dict = {
        "alignment_native": float(s_nat),
        "alignment_shuffled": float(s_shuf),
        "adjacency_gain": float(s_nat - s_shuf),
        "rmse_raw": float(rmse_raw),
        "rmse_binary": float(rmse_bin),
        "vs_raw": float(vs_raw),
        "recommend_binary": bool(vs_raw > 0),
    }

    if include_sliding:
        slide = fit_sliding_conv(Xs, y)
        Phi_slide = slide.transform(Xs)
        rmse_slide = _inner_rmse(Phi_slide, y, random_state=random_state + 2)
        vs_slide = (rmse_raw - rmse_slide) / max(rmse_raw, 1e-12)
        out.update(
            {
                "rmse_sliding": float(rmse_slide),
                "vs_raw_sliding": float(vs_slide),
                "recommend_sliding": bool(vs_slide > 0),
                "recommend_patterned": bool(vs_raw > 0 or vs_slide > 0),
            }
        )
    else:
        out["recommend_patterned"] = bool(vs_raw > 0)

    return out
