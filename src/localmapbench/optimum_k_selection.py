"""
General optimum-k selection for local neighborhood regressors.

Method (train-only; no test leakage)
------------------------------------
1. Feasibility band
   - k_min = d+1 for local linear / ridge (so the local design is identifiable);
     k_min = 3 for local mean.
   - k_max = min(floor(gamma * n_train), n_train - 1) with gamma=0.25 by default
     (avoid neighborhoods so large that the fit is nearly global).
   - Candidate grid: union of {k_min, ..., geometric/linear ladder, asymptotic
     pilot} clipped to [k_min, k_max].

2. Asymptotic pilot (optional anchor, not the final choice)
   For Nadaraya–Watson-type smoothers in d dimensions, the MSE-optimal rate is
       k ~ c * n^{4/(4+d)}
   We use this only to ensure the grid covers the theoretically relevant scale;
   the data decide the final k.

3. Repeated K-fold CV risk curve (primary criterion)
   On the training set only, for each candidate k, estimate
       R(k) = mean_over_repeats_and_folds RMSE(y, yhat_k)
   with the *same* local model / weights that will be used at test time.
   Also record SE(k) across fold-level RMSEs (stability).

4. Smooth the risk curve
   Apply a short moving-average on log-k (or on the ordered grid index) to
   suppress grid jitter without changing the global shape.

5. Selection rule (1-SE + bias–variance preference)
   Let k0 = argmin_k R_smooth(k).
   Define the acceptable set
       A = { k : R_smooth(k) <= R_smooth(k0) + SE(k0) }.
   Among k in A, choose the **largest** k.
   Rationale: within statistical noise of the best risk, a larger neighborhood
   is the more regularized / stabler local fit (classic smoothing preference),
   which generalizes better on heterogeneous real tabular data.

6. Space- and model-specific k
   Select k separately for each feature space (Raw vs Phi) and each local
   model (mean / linear / ridge). Do not reuse a Raw-tuned k on Phi.

7. Outer evaluation
   Nested usage: outer split -> select k on outer-train -> predict outer-test.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from localmapbench.binary_kernel_transform import k_grid_for_dim, predict_local


@dataclass
class KSelectionResult:
    k_star: int
    k_min: int
    k_max: int
    candidates: list[int]
    risk_mean: list[float]
    risk_se: list[float]
    risk_smooth: list[float]
    k0_argmin: int
    acceptable: list[int]
    pilot_k: int
    model: str
    n: int
    d: int
    rule: str = "1se_largest"

    def to_dict(self) -> dict:
        return asdict(self)


def asymptotic_pilot_k(n: int, d: int, c: float = 1.0) -> int:
    """MSE-rate pilot k ~ c * n^{4/(4+d)} for NW-type smoothers."""
    d_eff = max(int(d), 1)
    k = int(np.ceil(c * (n ** (4.0 / (4.0 + d_eff)))))
    return int(k)


def feasible_k_bounds(
    n: int,
    d: int,
    model: str = "linear",
    gamma_max: float = 0.25,
    inner_n: int | None = None,
) -> tuple[int, int]:
    from localmapbench.binary_kernel_transform import n_params_poly2

    model = model.lower()
    if model in {"linear", "ridge"}:
        k_min = max(d + 1, 3)
    elif model == "poly2":
        k_min = max(n_params_poly2(d) + 2, d + 2, 8)
    elif model in {"kernel_ridge", "svr"}:
        k_min = max(15, 2 * d, 8)
        # Nonlinear local fits are O(k^2..k^3) per query — keep k modest
        gamma_max = min(gamma_max, 0.12)
    else:
        k_min = 3
    hard_cap = n - 1
    if model in {"kernel_ridge", "svr", "poly2"}:
        hard_cap = min(hard_cap, 120)
    # Nested CV uses smaller train folds; k must fit those neighborhoods.
    if inner_n is not None:
        hard_cap = min(hard_cap, max(int(inner_n) - 1, 3))
    k_max = int(min(max(k_min, int(np.floor(gamma_max * n))), hard_cap))
    if k_max < k_min:
        k_max = min(hard_cap, k_min)
        k_min = min(k_min, k_max)
    return int(k_min), int(k_max)


def inner_train_size(n: int, n_splits: int) -> int:
    """Smallest training-fold size under KFold(n_splits)."""
    return int(n - np.ceil(n / float(max(int(n_splits), 1))))


def build_k_candidates(
    n: int,
    d: int,
    model: str = "linear",
    gamma_max: float = 0.25,
    max_candidates: int | None = None,
    n_splits: int = 5,
) -> list[int]:
    k_min, k_max = feasible_k_bounds(
        n, d, model, gamma_max, inner_n=inner_train_size(n, n_splits)
    )
    base = k_grid_for_dim(d, n)
    pilot = asymptotic_pilot_k(n, d)
    n_log = 8 if model in {"kernel_ridge", "svr", "poly2"} else 12
    if k_max > k_min:
        log_grid = np.unique(
            np.round(
                np.exp(
                    np.linspace(np.log(k_min), np.log(max(k_max, k_min + 1)), n_log)
                )
            ).astype(int)
        )
    else:
        log_grid = np.array([k_min], dtype=int)
    cand = sorted(
        {
            int(k)
            for k in list(base) + list(log_grid) + [pilot, k_min, k_max]
            if k_min <= int(k) <= k_max
        }
    )
    if not cand:
        cand = [k_min]
    if max_candidates is not None and len(cand) > max_candidates:
        # Keep endpoints + evenly spaced subset
        idx = np.unique(
            np.round(np.linspace(0, len(cand) - 1, max_candidates)).astype(int)
        )
        cand = [cand[i] for i in idx]
    return cand


def _smooth_risk(risk: np.ndarray, window: int = 3) -> np.ndarray:
    """Simple moving average on the ordered candidate index."""
    if len(risk) == 0:
        return risk
    w = max(1, min(int(window), len(risk)))
    if w == 1:
        return risk.copy()
    pad = w // 2
    x = np.pad(risk, (pad, pad), mode="edge")
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="valid")[: len(risk)]


def select_optimum_k(
    X: np.ndarray,
    y: np.ndarray,
    model: str = "linear",
    weight_mode: str = "inverse",
    ridge_alpha: float = 1.0,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
    gamma_max: float = 0.25,
    smooth_window: int = 3,
    rule: str = "1se_largest",
    candidates: list[int] | None = None,
    max_val_points: int | None = None,
    max_candidates: int | None = None,
) -> KSelectionResult:
    """Select optimum neighborhood size k on a training set.

    Parameters
    ----------
    rule :
      - \"1se_largest\": among k within 1 SE of best smoothed risk, take largest
      - \"argmin_smooth\": pure argmin of smoothed CV risk
      - \"argmin_raw\": argmin of unsmoothed CV risk
    max_val_points :
      Optional cap on validation points per fold (speeds nonlinear locals).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    model = model.lower()
    if model in {"kernel_ridge", "svr", "poly2"} and max_candidates is None:
        max_candidates = 7
    if model in {"kernel_ridge", "svr"} and max_val_points is None:
        max_val_points = 150
    if model == "poly2" and max_val_points is None:
        max_val_points = 250

    inner_n = inner_train_size(n, n_splits)
    k_min, k_max = feasible_k_bounds(
        n, d, model, gamma_max, inner_n=inner_n
    )
    cand = candidates if candidates is not None else build_k_candidates(
        n,
        d,
        model,
        gamma_max,
        max_candidates=max_candidates,
        n_splits=n_splits,
    )
    cand = [k for k in cand if k_min <= k <= k_max]
    if not cand:
        cand = [k_min]
    pilot = int(np.clip(asymptotic_pilot_k(n, d), k_min, k_max))

    fold_rmses: list[list[float]] = [[] for _ in cand]

    rng = np.random.default_rng(random_state)
    for rep in range(n_repeats):
        kf = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=int(rng.integers(0, 10**9)),
        )
        for tr, va in kf.split(X):
            X_tr, X_va = X[tr], X[va]
            y_tr, y_va = y[tr], y[va]
            if max_val_points is not None and len(va) > max_val_points:
                sub = rng.choice(len(va), size=max_val_points, replace=False)
                X_va = X_va[sub]
                y_va = y_va[sub]
            for j, k in enumerate(cand):
                if k >= len(y_tr):
                    fold_rmses[j].append(float("inf"))
                    continue
                pred = predict_local(
                    X_tr,
                    y_tr,
                    X_va,
                    k_pred=k,
                    model=model,
                    weight_mode=weight_mode,
                    ridge_alpha=ridge_alpha,
                )
                rmse = float(np.sqrt(mean_squared_error(y_va, pred)))
                fold_rmses[j].append(rmse)

    risk_mean = np.array([float(np.mean(v)) for v in fold_rmses], dtype=float)
    risk_se = np.array(
        [
            float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
            for v in fold_rmses
        ],
        dtype=float,
    )
    risk_smooth = _smooth_risk(risk_mean, window=smooth_window)

    finite = np.isfinite(risk_smooth)
    if finite.any():
        k0_idx = int(np.flatnonzero(finite)[int(np.argmin(risk_smooth[finite]))])
    else:
        k0_idx = 0
    k0 = int(cand[k0_idx])
    threshold = float(risk_smooth[k0_idx] + risk_se[k0_idx])
    acceptable = [int(cand[i]) for i, r in enumerate(risk_smooth) if r <= threshold]

    if rule == "argmin_raw":
        k_star = int(cand[int(np.argmin(risk_mean))])
    elif rule == "argmin_smooth":
        k_star = k0
    else:
        k_star = int(max(acceptable)) if acceptable else k0

    return KSelectionResult(
        k_star=k_star,
        k_min=k_min,
        k_max=k_max,
        candidates=[int(k) for k in cand],
        risk_mean=[float(x) for x in risk_mean],
        risk_se=[float(x) for x in risk_se],
        risk_smooth=[float(x) for x in risk_smooth],
        k0_argmin=k0,
        acceptable=acceptable,
        pilot_k=pilot,
        model=model,
        n=n,
        d=d,
        rule=rule,
    )


def select_k_and_model(
    X: np.ndarray,
    y: np.ndarray,
    models: tuple[str, ...] = ("mean", "linear", "ridge"),
    **kwargs,
) -> tuple[str, KSelectionResult, dict[str, KSelectionResult]]:
    """Jointly choose local model and its optimum k by CV risk at k_star."""
    results: dict[str, KSelectionResult] = {}
    best_model = models[0]
    best_res: KSelectionResult | None = None
    best_risk = np.inf
    for m in models:
        res = select_optimum_k(X, y, model=m, **kwargs)
        results[m] = res
        # Risk at selected k (use smoothed risk on the selected candidate)
        idx = res.candidates.index(res.k_star)
        r = res.risk_smooth[idx]
        if not np.isfinite(r):
            continue
        if r < best_risk:
            best_risk = r
            best_model = m
            best_res = res
    if best_res is None:
        best_model = models[0]
        best_res = results[best_model]
    return best_model, best_res, results
