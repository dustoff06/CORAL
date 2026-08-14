"""Core routines for CORAL / PCA / ZCA comparison.

This module is a compact application-facing extraction of the hard-constrained
CORAL implementation used in the manuscript. It operates on continuous
variables from a pandas DataFrame, standardizes them, constructs the sample
correlation matrix, runs CORAL, PCA, and ZCA, and returns transformed data and
comparison diagnostics.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

try:
    import autograd.numpy as anp
    import pymanopt
    from pymanopt.manifolds import Euclidean, Oblique, Product, Stiefel
    from pymanopt.optimizers import TrustRegions
    _HAS_PYMANOPT = True
except ImportError:
    anp = None
    pymanopt = None
    Euclidean = Oblique = Product = Stiefel = TrustRegions = None
    _HAS_PYMANOPT = False


# -----------------------------------------------------------------------------
# Linear algebra
# -----------------------------------------------------------------------------

def _symmetrize(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("R must be a square matrix.")
    return 0.5 * (R + R.T)


def _validate_spd_correlation(R: np.ndarray, eig_tol: float = 1e-12) -> np.ndarray:
    R = _symmetrize(R)
    if not np.allclose(np.diag(R), 1.0, atol=1e-8, rtol=0.0):
        warnings.warn(
            "R does not have an exact unit diagonal. Fidelity values are "
            "correlations only when R is a correlation matrix.",
            RuntimeWarning,
        )
    evals = np.linalg.eigvalsh(R)
    if evals.min() <= eig_tol:
        raise ValueError(
            "The selected-variable correlation matrix is not positive definite. "
            f"Minimum eigenvalue={evals.min():.3e}. Remove redundant variables "
            "or use a larger sample; CORAL is intentionally not regularizing R "
            "silently."
        )
    return R


def matrix_sqrt_and_inv_sqrt(R: np.ndarray, eig_tol: float = 1e-12):
    R = _validate_spd_correlation(R, eig_tol=eig_tol)
    evals, evecs = np.linalg.eigh(R)
    Rsqrt = (evecs * np.sqrt(evals)) @ evecs.T
    Rinv_sqrt = (evecs * (1.0 / np.sqrt(evals))) @ evecs.T
    return Rsqrt, Rinv_sqrt


def offdiag_stats(Sigma: np.ndarray):
    Sigma = np.asarray(Sigma, dtype=float)
    p = Sigma.shape[0]
    if p < 2:
        return 0.0, 0.0
    mask = ~np.eye(p, dtype=bool)
    vals = np.abs(Sigma[mask])
    return float(vals.max()), float(vals.mean())


def squared_decorrelation(Sigma: np.ndarray) -> float:
    Sigma = np.asarray(Sigma, dtype=float)
    off = Sigma - np.diag(np.diag(Sigma))
    return 0.5 * float(np.sum(off ** 2))


def _random_oblique(p: int, rng: np.random.Generator) -> np.ndarray:
    W = rng.normal(size=(p, p))
    norms = np.linalg.norm(W, axis=0)
    bad = norms <= np.finfo(float).eps
    while np.any(bad):
        W[:, bad] = rng.normal(size=(p, int(bad.sum())))
        norms = np.linalg.norm(W, axis=0)
        bad = norms <= np.finfo(float).eps
    return W / norms


def _random_orthogonal(p: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(p, p))
    Q, Rq = np.linalg.qr(A)
    signs = np.sign(np.diag(Rq))
    signs[signs == 0] = 1.0
    return Q * signs


def _require_pymanopt():
    if not _HAS_PYMANOPT:
        raise ImportError(
            'CORAL requires autograd and pymanopt. Install with: '
            'pip install "pymanopt[autograd]"'
        )


# -----------------------------------------------------------------------------
# PCA: sign-invariant, source-matched benchmark
# -----------------------------------------------------------------------------

def _bottleneck_assignment(abs_corr: np.ndarray):
    A = np.asarray(abs_corr, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("abs_corr must be square.")

    thresholds = np.unique(A.ravel())[::-1]
    for threshold in thresholds:
        forbidden = A < threshold
        rows, cols = linear_sum_assignment(forbidden.astype(float))
        if not np.any(forbidden[rows, cols]):
            big = 1e6
            cost = np.where(forbidden, big, -A)
            rows, cols = linear_sum_assignment(cost)
            return rows, cols, float(threshold)
    raise RuntimeError("No perfect PCA assignment found.")


def _align_pca_assignment(T_raw, C_raw, row_ind, col_ind):
    p = T_raw.shape[1]
    component_for_source = np.full(p, -1, dtype=int)
    component_for_source[row_ind] = col_ind
    if np.any(component_for_source < 0):
        raise RuntimeError("Incomplete PCA assignment.")

    T_aligned = np.zeros_like(T_raw)
    fidelity = np.zeros(p)
    signs = np.ones(p)

    for source in range(p):
        component = component_for_source[source]
        raw_corr = C_raw[source, component]
        sign = 1.0 if raw_corr >= 0 else -1.0
        signs[source] = sign
        T_aligned[:, source] = sign * T_raw[:, component]
        fidelity[source] = abs(raw_corr)

    return {
        "T_aligned": T_aligned,
        "component_for_source": component_for_source,
        "orientation_signs": signs,
        "fidelity": fidelity,
        "min_fidelity": float(fidelity.min()),
        "mean_fidelity": float(fidelity.mean()),
        "assignment_total": float(fidelity.sum()),
    }


def pca_fidelity_sign_invariant(R: np.ndarray):
    """Full PCA, R-normalized to unit transformed variance.

    Two favorable source-component assignments are computed:
    * bottleneck: maximizes minimum absolute source fidelity;
    * sum-optimal: maximizes mean absolute source fidelity.

    The transformation exposed as T_aligned is the bottleneck-aligned version,
    because CORAL's defining guarantee is a minimum per-variable fidelity.
    """
    R = _validate_spd_correlation(R)

    eigvals, V = np.linalg.eigh(R)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    V = V[:, order]
    if np.any(eigvals <= 0):
        raise ValueError("PCA benchmark requires positive eigenvalues.")

    T_raw = V @ np.diag(1.0 / np.sqrt(eigvals))
    C_raw = R @ T_raw
    abs_corr = np.abs(C_raw)

    rows_sum, cols_sum = linear_sum_assignment(-abs_corr)
    sum_optimal = _align_pca_assignment(T_raw, C_raw, rows_sum, cols_sum)
    sum_optimal["C_aligned"] = R @ sum_optimal["T_aligned"]
    sum_optimal["Sigma_aligned"] = (
        sum_optimal["T_aligned"].T @ R @ sum_optimal["T_aligned"]
    )

    rows_bot, cols_bot, threshold = _bottleneck_assignment(abs_corr)
    bottleneck = _align_pca_assignment(T_raw, C_raw, rows_bot, cols_bot)
    bottleneck["C_aligned"] = R @ bottleneck["T_aligned"]
    bottleneck["Sigma_aligned"] = (
        bottleneck["T_aligned"].T @ R @ bottleneck["T_aligned"]
    )
    bottleneck["bottleneck_threshold"] = threshold

    return {
        "T_raw": T_raw,
        "C_raw": C_raw,
        "eigenvalues": eigvals,
        "bottleneck": bottleneck,
        "sum_optimal": sum_optimal,
        "T_aligned": bottleneck["T_aligned"],
        "Sigma_aligned": bottleneck["Sigma_aligned"],
        "fidelity": bottleneck["fidelity"],
        "min_fidelity": bottleneck["min_fidelity"],
        "mean_fidelity": bottleneck["mean_fidelity"],
        "best_min_fidelity": bottleneck["min_fidelity"],
        "best_mean_fidelity": sum_optimal["mean_fidelity"],
    }


# -----------------------------------------------------------------------------
# ZCA
# -----------------------------------------------------------------------------

def zca_benchmark(R: np.ndarray, variable_names=None):
    R = _validate_spd_correlation(R)
    Rsqrt, Rinv_sqrt = matrix_sqrt_and_inv_sqrt(R)
    T = Rinv_sqrt
    Sigma_T = T.T @ R @ T
    fidelity = np.diag(Rsqrt)
    max_off, mean_off = offdiag_stats(Sigma_T)
    result = {
        "T": T,
        "Sigma_T": Sigma_T,
        "fidelity": fidelity,
        "min_fidelity": float(fidelity.min()),
        "mean_fidelity": float(fidelity.mean()),
        "max_offdiag": max_off,
        "mean_offdiag": mean_off,
        "decorr_value": squared_decorrelation(Sigma_T),
    }
    if variable_names is not None:
        result["fidelity_by_variable"] = dict(zip(variable_names, fidelity))
    return result


# -----------------------------------------------------------------------------
# CORAL hard-constrained primal
# -----------------------------------------------------------------------------

def _alm_penalty_term(viol, lam, rho):
    shifted = lam + rho * viol
    positive = anp.maximum(0.0, shifted)
    return (anp.sum(positive ** 2) - anp.sum(lam ** 2)) / (2.0 * rho)


def _alm_multiplier_update(viol, lam, rho):
    return np.maximum(0.0, lam + rho * np.asarray(viol, dtype=float))


def solve_primal_hard(
    R,
    rho_min,
    n_outer=15,
    rho0=10.0,
    rho_growth=2.0,
    tol=1e-4,
    n_starts=100,
    max_iterations=300,
    seed=None,
):
    """Dense CORAL primal using ALM on the oblique manifold."""
    _require_pymanopt()
    R = _validate_spd_correlation(R)
    p = R.shape[0]
    Rsqrt, Rinv_sqrt = matrix_sqrt_and_inv_sqrt(R)
    rng = np.random.default_rng(seed)

    if not (0.0 <= rho_min <= 1.0):
        raise ValueError("rho_min must lie in [0, 1].")
    if n_starts < 1:
        raise ValueError("n_starts must be at least 1.")

    def run_once(W0):
        lam = np.zeros(p)
        rho_pen = float(rho0)
        W = W0.copy()
        max_viol = np.inf

        for outer in range(n_outer):
            def cost(W, lam=lam.copy(), rho_pen=rho_pen):
                WtW = W.T @ W
                offdiag_sq = anp.sum(WtW ** 2) - anp.sum(anp.diag(WtW) ** 2)
                decorr = 0.5 * offdiag_sq
                fidelity = anp.diag(Rsqrt @ W)
                viol = rho_min - fidelity
                return decorr + _alm_penalty_term(viol, lam, rho_pen)

            manifold = Oblique(p, p)

            @pymanopt.function.autograd(manifold)
            def wrapped_cost(W):
                return cost(W)

            problem = pymanopt.Problem(manifold, wrapped_cost)
            optimizer = TrustRegions(verbosity=0, max_iterations=max_iterations)
            result = optimizer.run(problem, initial_point=W)
            W = result.point

            fidelity = np.diag(Rsqrt @ W)
            viol = rho_min - fidelity
            max_viol = max(0.0, float(np.max(viol)))
            lam = _alm_multiplier_update(viol, lam, rho_pen)
            if max_viol <= tol:
                break
            rho_pen *= rho_growth

        return W, max_viol, outer + 1

    candidates = []
    for start in range(n_starts):
        W0 = _random_oblique(p, rng)
        W, max_viol, n_used = run_once(W0)
        T = Rinv_sqrt @ W
        Sigma_T = T.T @ R @ T
        decorr_value = squared_decorrelation(Sigma_T)
        feasible = max_viol <= tol
        rank_key = (0 if feasible else 1, 0.0 if feasible else max_viol, decorr_value)
        candidates.append((rank_key, T, Sigma_T, max_viol, n_used, decorr_value, start))

    candidates.sort(key=lambda x: x[0])
    _, T, Sigma_T, max_viol, n_used, decorr_value, best_start = candidates[0]
    fidelity = np.diag(R @ T)
    max_off, mean_off = offdiag_stats(Sigma_T)
    start_objectives = np.array([c[5] for c in candidates], dtype=float)

    return {
        "T": T,
        "Sigma_T": Sigma_T,
        "fidelity": fidelity,
        "fid": fidelity,  # backward-compatible alias
        "min_fidelity": float(fidelity.min()),
        "mean_fidelity": float(fidelity.mean()),
        "max_offdiag": max_off,
        "mean_offdiag": mean_off,
        "feasible": bool(max_viol <= tol),
        "max_violation": float(max_viol),
        "n_outer_used": int(n_used),
        "decorr_value": float(decorr_value),
        "obj_std": float(start_objectives.std(ddof=0)),
        "start_objectives": start_objectives,
        "best_start": int(best_start),
    }


# -----------------------------------------------------------------------------
# Optional exact-decorrelation threshold rho_star(R)
# -----------------------------------------------------------------------------

def solve_rho_star_exact(
    R,
    n_outer=20,
    rho0=10.0,
    rho_growth=2.0,
    tol=1e-6,
    n_starts=100,
    max_iterations=500,
    seed=None,
):
    """Best-found common fidelity under exact decorrelation.

    This is a non-convex multistart calculation. The achieved value is a
    constructive lower bound on rho_star(R), not a proof of global optimality.
    """
    _require_pymanopt()
    R = _validate_spd_correlation(R)
    p = R.shape[0]
    Rsqrt, Rinv_sqrt = matrix_sqrt_and_inv_sqrt(R)
    rng = np.random.default_rng(seed)

    def run_once(Q0, gamma0):
        lam = np.zeros(p)
        rho_pen = float(rho0)
        Q, gamma = Q0.copy(), float(gamma0)
        max_viol = np.inf

        for outer in range(n_outer):
            def cost(Qt, lam=lam.copy(), rho_pen=rho_pen):
                Q, gamma_arr = Qt
                gamma_scalar = gamma_arr[0]
                fidelity = anp.diag(Rsqrt @ Q)
                viol = gamma_scalar - fidelity
                return -gamma_scalar + _alm_penalty_term(viol, lam, rho_pen)

            manifold = Product([Stiefel(p, p), Euclidean(1)])

            @pymanopt.function.autograd(manifold)
            def wrapped_cost(Q, gamma_arr):
                return cost((Q, gamma_arr))

            problem = pymanopt.Problem(manifold, wrapped_cost)
            optimizer = TrustRegions(verbosity=0, max_iterations=max_iterations)
            result = optimizer.run(problem, initial_point=(Q, np.array([gamma])))
            Q, gamma_arr = result.point
            gamma = float(gamma_arr[0])

            fidelity = np.diag(Rsqrt @ Q)
            viol = gamma - fidelity
            max_viol = max(0.0, float(np.max(viol)))
            lam = _alm_multiplier_update(viol, lam, rho_pen)
            if max_viol <= tol:
                break
            rho_pen *= rho_growth

        return Q, gamma, max_viol, outer + 1

    candidates = []
    for _ in range(n_starts):
        Q0 = _random_orthogonal(p, rng)
        gamma0 = float(np.min(np.diag(Rsqrt @ Q0)))
        Q, gamma, max_viol, n_used = run_once(Q0, gamma0)
        T = Rinv_sqrt @ Q
        Sigma_T = T.T @ R @ T
        fidelity = np.diag(R @ T)
        feasible = max_viol <= tol
        rank_key = (0 if feasible else 1, 0.0 if feasible else max_viol, -gamma)
        candidates.append((rank_key, Q, T, Sigma_T, fidelity, gamma, max_viol, n_used))

    candidates.sort(key=lambda x: x[0])
    _, Q, T, Sigma_T, fidelity, gamma, max_viol, n_used = candidates[0]
    zca_lower = float(np.min(np.diag(Rsqrt)))
    trace_upper = float(np.trace(Rsqrt) / p)

    return {
        "rho_star": float(gamma),
        "Q": Q,
        "T": T,
        "Sigma_T": Sigma_T,
        "fidelity": fidelity,
        "feasible": bool(max_viol <= tol),
        "max_violation": float(max_viol),
        "n_outer_used": int(n_used),
        "zca_lower_bound": zca_lower,
        "trace_upper_bound": trace_upper,
        "orthogonality_error": float(np.max(np.abs(Q.T @ Q - np.eye(p)))),
        "decorrelation_error": float(np.max(np.abs(Sigma_T - np.eye(p)))),
    }


# -----------------------------------------------------------------------------
# DataFrame-facing comparison API
# -----------------------------------------------------------------------------

@dataclass
class CORALComparison:
    selected_columns: list[str]
    retained_index: pd.Index
    standardized_data: pd.DataFrame
    correlation: pd.DataFrame
    summary: pd.DataFrame
    fidelity: pd.DataFrame
    transformations: dict[str, pd.DataFrame]
    transformed_correlations: dict[str, pd.DataFrame]
    transformed_data: dict[str, pd.DataFrame]
    raw_results: dict
    diagnostics: dict


def prepare_continuous_dataframe(
    df: pd.DataFrame,
    columns: Sequence[str],
    missing: str = "drop",
):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    columns = list(columns)
    if len(columns) < 2:
        raise ValueError("Select at least two continuous variables.")
    missing_cols = [c for c in columns if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns not found: {missing_cols}")

    X = df.loc[:, columns].copy()
    non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise TypeError(f"Selected variables must be numeric: {non_numeric}")

    X = X.replace([np.inf, -np.inf], np.nan)
    if missing == "drop":
        X = X.dropna(axis=0, how="any")
    elif missing == "error" and X.isna().any().any():
        raise ValueError("Missing values are present in the selected variables.")
    elif missing not in {"drop", "error"}:
        raise ValueError("missing must be 'drop' or 'error'.")

    if len(X) < 3:
        raise ValueError("At least three complete observations are required.")
    if len(X) <= len(columns):
        raise ValueError(
            f"Need more complete observations than selected variables for a "
            f"full-rank sample correlation matrix (n={len(X)}, p={len(columns)})."
        )

    sd = X.std(axis=0, ddof=1)
    zero_sd = sd[sd <= np.finfo(float).eps].index.tolist()
    if zero_sd:
        raise ValueError(f"Zero-variance variables cannot be analyzed: {zero_sd}")

    Z = (X - X.mean(axis=0)) / sd
    R = Z.corr().to_numpy(dtype=float)
    R = _validate_spd_correlation(R)
    return X, Z, R


def _method_frames(Z: pd.DataFrame, T: np.ndarray, columns: list[str], prefix: str):
    values = Z.to_numpy(dtype=float) @ T
    out_cols = [f"{prefix}_{c}" for c in columns]
    return pd.DataFrame(values, index=Z.index, columns=out_cols)


def _matrix_frame(A: np.ndarray, columns: list[str]):
    return pd.DataFrame(A, index=columns, columns=columns)


def compare_coral_pca_zca(
    df: pd.DataFrame,
    columns: Sequence[str],
    rho_min: float = 0.95,
    n_starts: int = 20,
    n_outer: int = 15,
    max_iterations: int = 300,
    tol: float = 1e-4,
    seed: Optional[int] = 0,
    missing: str = "drop",
    estimate_rho_star: bool = False,
    rho_star_starts: int = 20,
):
    """Run CORAL, sign-invariant PCA, and ZCA on selected DataFrame columns."""
    columns = list(columns)
    X, Z, R = prepare_continuous_dataframe(df, columns, missing=missing)

    coral = solve_primal_hard(
        R,
        rho_min=rho_min,
        n_starts=n_starts,
        n_outer=n_outer,
        max_iterations=max_iterations,
        tol=tol,
        seed=seed,
    )
    pca = pca_fidelity_sign_invariant(R)
    zca = zca_benchmark(R, variable_names=columns)

    methods = {
        "CORAL": {
            "T": coral["T"],
            "Sigma": coral["Sigma_T"],
            "fidelity": coral["fidelity"],
            "min_fidelity": coral["min_fidelity"],
            "mean_fidelity": coral["mean_fidelity"],
            "feasible": coral["feasible"],
            "max_violation": coral["max_violation"],
        },
        "PCA": {
            "T": pca["T_aligned"],
            "Sigma": pca["Sigma_aligned"],
            "fidelity": pca["fidelity"],
            "min_fidelity": pca["best_min_fidelity"],
            # The manuscript benchmark reports the favorable sum-optimal mean.
            "mean_fidelity": pca["best_mean_fidelity"],
            "feasible": True,
            "max_violation": 0.0,
        },
        "ZCA": {
            "T": zca["T"],
            "Sigma": zca["Sigma_T"],
            "fidelity": zca["fidelity"],
            "min_fidelity": zca["min_fidelity"],
            "mean_fidelity": zca["mean_fidelity"],
            "feasible": True,
            "max_violation": 0.0,
        },
    }

    summary_rows = []
    fidelity_data = {"Variable": columns}
    transformations = {}
    transformed_correlations = {}
    transformed_data = {}

    base_max, base_mean = offdiag_stats(R)
    summary_rows.append({
        "Method": "Original",
        "Min source fidelity": 1.0,
        "Mean source fidelity": 1.0,
        "Max |offdiag|": base_max,
        "Mean |offdiag|": base_mean,
        "D2": squared_decorrelation(R),
        "Fidelity constraint met": True,
    })

    for name, result in methods.items():
        max_off, mean_off = offdiag_stats(result["Sigma"])
        summary_rows.append({
            "Method": name,
            "Min source fidelity": result["min_fidelity"],
            "Mean source fidelity": result["mean_fidelity"],
            "Max |offdiag|": max_off,
            "Mean |offdiag|": mean_off,
            "D2": squared_decorrelation(result["Sigma"]),
            "Fidelity constraint met": (
                bool(result["min_fidelity"] >= rho_min - tol) if name == "CORAL" else np.nan
            ),
        })
        fidelity_data[name] = result["fidelity"]
        transformations[name] = _matrix_frame(result["T"], columns)
        transformed_correlations[name] = _matrix_frame(result["Sigma"], columns)
        transformed_data[name] = _method_frames(Z, result["T"], columns, name)

    summary = pd.DataFrame(summary_rows).set_index("Method")
    fidelity = pd.DataFrame(fidelity_data).set_index("Variable")

    evals = np.linalg.eigvalsh(R)
    diagnostics = {
        "n_original": int(len(df)),
        "n_complete": int(len(X)),
        "n_dropped": int(len(df) - len(X)),
        "p": int(len(columns)),
        "min_eigenvalue": float(evals.min()),
        "condition_number": float(evals.max() / evals.min()),
        "rho_min": float(rho_min),
        "coral_n_starts": int(n_starts),
        "coral_max_violation": float(coral["max_violation"]),
        "pca_mean_note": (
            "PCA minimum fidelity is from the bottleneck assignment; PCA mean "
            "fidelity is from the separately sum-optimal assignment."
        ),
    }

    rho_star = None
    if estimate_rho_star:
        rho_star = solve_rho_star_exact(
            R,
            n_starts=rho_star_starts,
            seed=seed,
        )
        diagnostics.update({
            "rho_star_best_found": float(rho_star["rho_star"]),
            "rho_star_zca_lower": float(rho_star["zca_lower_bound"]),
            "rho_star_trace_upper": float(rho_star["trace_upper_bound"]),
        })

    return CORALComparison(
        selected_columns=columns,
        retained_index=X.index,
        standardized_data=Z,
        correlation=_matrix_frame(R, columns),
        summary=summary,
        fidelity=fidelity,
        transformations=transformations,
        transformed_correlations=transformed_correlations,
        transformed_data=transformed_data,
        raw_results={"CORAL": coral, "PCA": pca, "ZCA": zca, "rho_star": rho_star},
        diagnostics=diagnostics,
    )
