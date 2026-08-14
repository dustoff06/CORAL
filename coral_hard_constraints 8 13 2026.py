"""
CORAL: hard-constrained solvers and theoretical diagnostics
=======================================================================

This revision keeps the original Augmented Lagrangian (ALM) primal and dual
solvers, and adds:

1. Sign-invariant PCA benchmarking (bottleneck + sum-optimal assignment).
2. Direct estimation of rho_star(R), the largest common source fidelity
   compatible with exact decorrelation, via the T = R^{-1/2} Q, Q in O(p)
   characterization.
3. Exact-support sparse solvers (SLSQP, forbidden entries removed from the
   parameter vector rather than penalized).
4. ZCA benchmark (closed-form: T = R^{-1/2}, i.e. Q = I in the exact-
   decorrelator family -- no optimization required).
5. Sparse PCA benchmark (sklearn's SparsePCA, fit on synthetic data
   reproducing R via Cholesky factorization, rescaled to the same R-metric
   unit-variance normalization used throughout).

Core dependencies:
    numpy, scipy

ALM and rho_star dependencies:
    autograd, pymanopt

Sparse PCA benchmark dependency:
    scikit-learn (optional; guarded import, skipped with a message if absent)

Install the manifold stack with, for example:
    pip install "pymanopt[autograd]"
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize

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

try:
    from sklearn.decomposition import SparsePCA

    _HAS_SKLEARN = True
except ImportError:
    SparsePCA = None
    _HAS_SKLEARN = False


# ==========================================================================
# Linear-algebra helpers
# ==========================================================================


def _symmetrize(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("R must be a square matrix.")
    return 0.5 * (R + R.T)


def _validate_spd_correlation(R: np.ndarray, eig_tol: float = 1e-12) -> np.ndarray:
    R = _symmetrize(R)
    if not np.allclose(np.diag(R), 1.0, atol=1e-8, rtol=0.0):
        warnings.warn(
            "R does not have an exact unit diagonal. The routines will use it "
            "as supplied, but source-fidelity values are correlations only when "
            "R is a correlation matrix.",
            RuntimeWarning,
        )
    evals = np.linalg.eigvalsh(R)
    if evals.min() <= eig_tol:
        raise ValueError(
            f"R must be positive definite. Minimum eigenvalue={evals.min():.3e}."
        )
    return R


def matrix_sqrt_and_inv_sqrt(R: np.ndarray, eig_tol: float = 1e-12):
    """Symmetric positive-definite square root and inverse square root."""
    R = _validate_spd_correlation(R, eig_tol=eig_tol)
    evals, evecs = np.linalg.eigh(R)
    if evals.min() <= eig_tol:
        raise ValueError("R is not numerically positive definite.")
    Rsqrt = (evecs * np.sqrt(evals)) @ evecs.T
    Rinv_sqrt = (evecs * (1.0 / np.sqrt(evals))) @ evecs.T
    return Rsqrt, Rinv_sqrt


def make_correlation_matrix(p: int, seed: int = 1) -> np.ndarray:
    """Synthetic correlation matrix matching the manuscript construction."""
    if p < 2:
        raise ValueError("p must be at least 2.")
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(p, p))
    S = A @ A.T / p + 0.5 * np.eye(p)
    d = np.diag(S)
    R = S / np.sqrt(np.outer(d, d))
    np.fill_diagonal(R, 1.0)
    return _validate_spd_correlation(R)


def offdiag_stats(Sigma: np.ndarray):
    """Return maximum and mean absolute off-diagonal entries."""
    Sigma = np.asarray(Sigma, dtype=float)
    p = Sigma.shape[0]
    mask = ~np.eye(p, dtype=bool)
    vals = np.abs(Sigma[mask])
    return float(vals.max()), float(vals.mean())


def _squared_decorrelation(Sigma: np.ndarray) -> float:
    off = Sigma - np.diag(np.diag(Sigma))
    return 0.5 * float(np.sum(off**2))


def _random_oblique(p: int, rng: np.random.Generator) -> np.ndarray:
    W = rng.normal(size=(p, p))
    norms = np.linalg.norm(W, axis=0)
    bad = norms <= np.finfo(float).eps
    while np.any(bad):
        W[:, bad] = rng.normal(size=(p, bad.sum()))
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
            "This routine requires autograd and pymanopt. Install with "
            "`pip install \"pymanopt[autograd]\"`."
        )


def _require_sklearn():
    if not _HAS_SKLEARN:
        raise ImportError(
            "This routine requires scikit-learn. Install with `pip install scikit-learn`."
        )


# ==========================================================================
# PCA benchmark: sign-invariant optimal assignment
# ==========================================================================


def _bottleneck_assignment(abs_corr: np.ndarray):
    """
    Maximize the minimum assigned absolute correlation.

    Among assignments attaining the optimal bottleneck threshold, maximize the
    total absolute correlation as a secondary criterion.
    """
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
    """Reorder and sign-orient PCA components into source-variable order."""
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
        "C_aligned": None,
        "component_for_source": component_for_source,
        "orientation_signs": signs,
        "fidelity": fidelity,
        "min_fidelity": float(fidelity.min()),
        "mean_fidelity": float(fidelity.mean()),
        "assignment_total": float(fidelity.sum()),
    }


def pca_fidelity_sign_invariant(R: np.ndarray):
    """
    Compute sign-invariant PCA fidelity under two favorable assignments.

    bottleneck    : maximizes the minimum assigned fidelity (fairest
                    benchmark for CORAL's declared floor).
    sum_optimal   : maximizes total assigned fidelity (fairest benchmark
                    for mean interpretability).

    Top-level fidelity/min_fidelity/mean_fidelity fields use the bottleneck
    assignment, since CORAL's defining guarantee is a minimum per-variable
    fidelity.
    """
    R = _validate_spd_correlation(R)
    p = R.shape[0]

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

    rows_bot, cols_bot, bottleneck_threshold = _bottleneck_assignment(abs_corr)
    bottleneck = _align_pca_assignment(T_raw, C_raw, rows_bot, cols_bot)
    bottleneck["C_aligned"] = R @ bottleneck["T_aligned"]
    bottleneck["Sigma_aligned"] = (
        bottleneck["T_aligned"].T @ R @ bottleneck["T_aligned"]
    )
    bottleneck["bottleneck_threshold"] = bottleneck_threshold

    return {
        "T_raw": T_raw,
        "C_raw": C_raw,
        "eigenvalues": eigvals,
        "bottleneck": bottleneck,
        "sum_optimal": sum_optimal,
        "T_aligned": bottleneck["T_aligned"],
        "C_aligned": bottleneck["C_aligned"],
        "Sigma_aligned": bottleneck["Sigma_aligned"],
        "component_for_source": bottleneck["component_for_source"],
        "orientation_signs": bottleneck["orientation_signs"],
        "fidelity": bottleneck["fidelity"],
        "min_fidelity": bottleneck["min_fidelity"],
        "mean_fidelity": bottleneck["mean_fidelity"],
        "assignment_total": bottleneck["assignment_total"],
        "best_min_fidelity": bottleneck["min_fidelity"],
        "best_mean_fidelity": sum_optimal["mean_fidelity"],
    }


# ==========================================================================
# ZCA benchmark: closed-form, Q = I case of the exact-decorrelator family
# ==========================================================================


def zca_benchmark(R: np.ndarray, variable_names=None):
    """
    Closed-form ZCA fidelity benchmark.

    ZCA whitening is the exact decorrelator T = R^{-1/2} corresponding to
    Q = I in the family T = R^{-1/2} Q, Q in O(p). Its per-variable source
    fidelity is diag(R^{1/2}), computed directly from the eigendecomposition
    -- no optimization, no multistart, no non-convexity.
    """
    R = _validate_spd_correlation(R)
    Rsqrt, Rinv_sqrt = matrix_sqrt_and_inv_sqrt(R)

    T_zca = Rinv_sqrt
    Sigma_zca = T_zca.T @ R @ T_zca  # should equal I to numerical precision
    fid = np.diag(Rsqrt)  # = diag(R @ T_zca)

    off = Sigma_zca - np.diag(np.diag(Sigma_zca))
    max_off = float(np.max(np.abs(off)))

    result = {
        "T": T_zca,
        "Sigma_T": Sigma_zca,
        "fidelity": fid,
        "min_fidelity": float(fid.min()),
        "mean_fidelity": float(fid.mean()),
        "max_offdiag": max_off,  # sanity check, confirms exact decorrelation
    }
    if variable_names is not None:
        result["fidelity_by_variable"] = dict(zip(variable_names, fid))
    return result


# ==========================================================================
# Sparse PCA benchmark
# ==========================================================================


def sparse_pca_benchmark(
    R: np.ndarray,
    alpha_grid=(0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
    n_samples: int = 5000,
    seed: int = 2,
    variable_names=None,
):
    """
    Sparse PCA fidelity/decorrelation benchmark across a sweep of L1 penalties.

    sklearn's SparsePCA operates on a data matrix, not R directly, so
    synthetic observations reproducing R exactly (via Cholesky factorization)
    are generated first. Each component is rescaled to satisfy
    t_j' R t_j = 1 (the same R-metric normalization used throughout) before
    computing fidelity and residual correlation, so results are comparable
    to CORAL's and ZCA's on equal footing.

    Sparse PCA components, like PCA components, have arbitrary sign and no
    intrinsic one-to-one correspondence to source variables. Fidelity is
    therefore computed under the SAME sign-invariant bottleneck (best
    possible minimum fidelity) and sum-optimal (best possible mean fidelity)
    assignments used for PCA in pca_fidelity_sign_invariant(), rather than
    raw index-order matching -- an earlier version of this function used raw
    index order and could report large negative "fidelities" that were
    artifacts of sign/assignment mismatch rather than genuine misalignment
    with the source variables.

    Loading sparsity here is a total nonzero-count budget, not a declared
    support pattern -- unlike solve_primal_support_hard's exact support, this
    comparison is descriptive rather than support-identical unless the
    caller separately checks that the resulting nonzero pattern coincides
    with a specific declared mask.

    Returns a list of dicts, one per alpha, each with:
        alpha, nnz, min_fidelity, mean_fidelity, max_offdiag, mean_offdiag
    (min_fidelity/mean_fidelity are the bottleneck- and sum-optimal-assignment
    values respectively, matching the PCA benchmark's field semantics.)
    """
    _require_sklearn()
    R = _validate_spd_correlation(R)
    p = R.shape[0]

    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(R)
    Z = rng.normal(size=(n_samples, p))
    X = Z @ L.T  # X has covariance ~ R

    results = []
    for alpha in alpha_grid:
        spca = SparsePCA(n_components=p, alpha=alpha, random_state=0, max_iter=500)
        spca.fit(X)
        T = spca.components_.T  # p x p, columns = sparse loadings

        Tn = np.zeros_like(T)
        for j in range(p):
            v = T[:, j]
            norm = np.sqrt(max(float(v @ R @ v), 0.0))
            Tn[:, j] = v / norm if norm > 1e-10 else v

        nnz = int(np.sum(np.abs(T) > 1e-8))

        # Sign-invariant, optimally-assigned fidelity (same treatment as PCA).
        C_raw = R @ Tn
        abs_corr = np.abs(C_raw)

        rows_sum, cols_sum = linear_sum_assignment(-abs_corr)
        sum_optimal = _align_pca_assignment(Tn, C_raw, rows_sum, cols_sum)

        rows_bot, cols_bot, _ = _bottleneck_assignment(abs_corr)
        bottleneck = _align_pca_assignment(Tn, C_raw, rows_bot, cols_bot)

        # Decorrelation is reported on the RAW (un-reassigned) transformation:
        # reassigning/reflecting columns to optimize fidelity does not change
        # which variables are mixed together, only which source each column
        # is compared against, so Sigma is computed from Tn directly.
        Sigma = Tn.T @ R @ Tn
        off = Sigma - np.diag(np.diag(Sigma))
        max_off = float(np.max(np.abs(off)))
        mean_off = float(np.mean(np.abs(off[np.triu_indices(p, k=1)])))

        entry = {
            "alpha": alpha,
            "nnz": nnz,
            "min_fidelity": bottleneck["min_fidelity"],
            "mean_fidelity": sum_optimal["mean_fidelity"],
            "max_offdiag": max_off,
            "mean_offdiag": mean_off,
        }
        if variable_names is not None:
            entry["fidelity_by_variable"] = dict(
                zip(variable_names, bottleneck["fidelity"])
            )
        results.append(entry)

    return results

def sparse_pca_at_matched_density(
    R: np.ndarray,
    target_nnz: int,
    alpha_grid=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0),
    n_samples: int = 5000,
    seed: int = 2,
):
    """
    Convenience wrapper: run sparse_pca_benchmark across alpha_grid and
    return the single result whose nonzero count is closest to target_nnz,
    for a fairer side-by-side comparison against a declared exact-support
    CORAL solve at a known density (e.g. the 110 free entries in the p=18
    |R_ij| > 0.15 support).

    This still matches on TOTAL nonzero count, not on WHICH entries are
    nonzero (sklearn's SparsePCA has no mechanism to enforce a declared
    support pattern), so it remains a descriptive rather than
    support-identical comparison.
    """
    results = sparse_pca_benchmark(R, alpha_grid=alpha_grid, n_samples=n_samples, seed=seed)
    best = min(results, key=lambda r: abs(r["nnz"] - target_nnz))
    return best, results


# ==========================================================================
# Shared ALM machinery
# ==========================================================================


def _alm_penalty_term(viol, lam, rho):
    """
    Augmented Lagrangian term for inequality constraints g(x) <= 0.

    viol contains g_j(x). The Hestenes-Powell-Rockafellar form is
        (1/(2*rho)) * sum(max(0, lam + rho*viol)^2 - lam^2).
    """
    bracket = anp.maximum(0.0, lam + rho * viol)
    return anp.sum(bracket**2 - lam**2) / (2.0 * rho)


def _alm_multiplier_update(viol, lam, rho):
    return np.maximum(0.0, lam + rho * viol)


# ==========================================================================
# PRIMAL: declared fidelity floor, minimize decorrelation
# ==========================================================================


def solve_primal_hard(
    R,
    rho_min,
    off_graph_mask=None,
    lam_sp=0.0,
    n_outer=15,
    rho0=10.0,
    rho_growth=2.0,
    tol=1e-4,
    n_starts=100,
    max_iterations=300,
    seed=None,
):
    """
    Dense hard-fidelity primal solved by ALM on the Oblique manifold.

    Constraints
    -----------
    e_j' R t_j >= rho_min       ALM to tolerance
    t_j' R t_j = 1              exact manifold constraint

    The optional off_graph_mask / lam_sp arguments are preserved only for
    backward compatibility. For new sparsity analyses, use
    solve_primal_support_hard(), which enforces forbidden coefficients exactly.
    """
    _require_pymanopt()
    R = _validate_spd_correlation(R)
    p = R.shape[0]
    Rsqrt, Rinv_sqrt = matrix_sqrt_and_inv_sqrt(R)
    rng = np.random.default_rng(seed)

    if not (0.0 <= rho_min <= 1.0):
        raise ValueError("rho_min must lie in [0, 1].")

    def run_once(W0):
        lam = np.zeros(p)
        rho_pen = float(rho0)
        W = W0.copy()
        max_viol = np.inf

        for outer in range(n_outer):
            def cost(W, lam=lam.copy(), rho_pen=rho_pen):
                WtW = W.T @ W
                offdiag_sq = anp.sum(WtW**2) - anp.sum(anp.diag(WtW) ** 2)
                decorr = 0.5 * offdiag_sq
                fid = anp.diag(Rsqrt @ W)
                viol = rho_min - fid
                alm_term = _alm_penalty_term(viol, lam, rho_pen)

                sp_term = 0.0
                if lam_sp > 0 and off_graph_mask is not None:
                    Tmat = Rinv_sqrt @ W
                    sp_term = anp.sum((Tmat * off_graph_mask) ** 2)

                return decorr + alm_term + lam_sp * sp_term

            manifold = Oblique(p, p)

            @pymanopt.function.autograd(manifold)
            def wrapped_cost(W):
                return cost(W)

            problem = pymanopt.Problem(manifold, wrapped_cost)
            optimizer = TrustRegions(verbosity=0, max_iterations=max_iterations)
            result = optimizer.run(problem, initial_point=W)
            W = result.point

            fid = np.diag(Rsqrt @ W)
            viol = rho_min - fid
            max_viol = max(0.0, float(np.max(viol)))
            lam = _alm_multiplier_update(viol, lam, rho_pen)

            if max_viol <= tol:
                break
            rho_pen *= rho_growth

        return W, max_viol, outer + 1

    candidates = []
    for _ in range(n_starts):
        W0 = _random_oblique(p, rng)
        W, max_viol, n_used = run_once(W0)
        T = Rinv_sqrt @ W
        Sigma_T = T.T @ R @ T
        decorr_val = _squared_decorrelation(Sigma_T)
        feasible = max_viol <= tol
        rank_key = (0 if feasible else 1, 0.0 if feasible else max_viol, decorr_val)
        candidates.append((rank_key, T, Sigma_T, max_viol, n_used, decorr_val))

    candidates.sort(key=lambda x: x[0])
    _, T, Sigma_T, max_viol, n_used, decorr_val = candidates[0]
    fid = np.diag(R @ T)
    start_objectives = np.array([c[5] for c in candidates], dtype=float)

    return {
        "T": T,
        "Sigma_T": Sigma_T,
        "fid": fid,
        "feasible": bool(max_viol <= tol),
        "max_violation": float(max_viol),
        "n_outer_used": int(n_used),
        "decorr_value": float(decorr_val),
        "obj_std": float(start_objectives.std(ddof=0)),
        "start_objectives": start_objectives,
    }


# ==========================================================================
# DUAL: declared decorrelation budget, maximize fidelity floor
# ==========================================================================


def solve_dual_hard(
    R,
    epsilon,
    n_outer=15,
    rho0=10.0,
    rho_growth=2.0,
    tol=1e-4,
    n_starts=5,
    max_iterations=300,
    seed=None,
):
    """
    Hard-constrained dual solved by ALM.

    Maximize t subject to
        sum_{i<j} (t_i' R t_j)^2 <= epsilon
        e_j' R t_j >= t for all j
        t_j' R t_j = 1 for all j.
    """
    _require_pymanopt()
    R = _validate_spd_correlation(R)
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative.")

    p = R.shape[0]
    Rsqrt, Rinv_sqrt = matrix_sqrt_and_inv_sqrt(R)
    rng = np.random.default_rng(seed)

    def decorr_of(W):
        WtW = W.T @ W
        offdiag_sq = anp.sum(WtW**2) - anp.sum(anp.diag(WtW) ** 2)
        return 0.5 * offdiag_sq

    def run_once(W0, t0):
        lam_fid = np.zeros(p)
        lam_decorr = 0.0
        rho_pen = float(rho0)
        W, t = W0.copy(), float(t0)
        max_viol = np.inf

        for outer in range(n_outer):
            def cost(Wt, lam_fid=lam_fid.copy(), lam_decorr=lam_decorr, rho_pen=rho_pen):
                W, t_arr = Wt
                t_scalar = t_arr[0]
                fid = anp.diag(Rsqrt @ W)
                viol_fid = t_scalar - fid
                alm_fid = _alm_penalty_term(viol_fid, lam_fid, rho_pen)

                decorr = decorr_of(W)
                viol_decorr = anp.array([decorr - epsilon])
                alm_decorr = _alm_penalty_term(
                    viol_decorr, anp.array([lam_decorr]), rho_pen
                )
                return -t_scalar + alm_fid + alm_decorr

            manifold = Product([Oblique(p, p), Euclidean(1)])

            @pymanopt.function.autograd(manifold)
            def wrapped_cost(W, t_arr):
                return cost((W, t_arr))

            problem = pymanopt.Problem(manifold, wrapped_cost)
            optimizer = TrustRegions(verbosity=0, max_iterations=max_iterations)
            result = optimizer.run(problem, initial_point=(W, np.array([t])))
            W, t_arr = result.point
            t = float(t_arr[0])

            fid = np.diag(Rsqrt @ W)
            viol_fid = t - fid
            decorr_val = float(decorr_of(W))
            viol_decorr = decorr_val - epsilon
            max_viol = max(
                0.0,
                float(np.max(viol_fid)),
                float(viol_decorr),
            )

            lam_fid = _alm_multiplier_update(viol_fid, lam_fid, rho_pen)
            lam_decorr = float(
                _alm_multiplier_update(
                    np.array([viol_decorr]), np.array([lam_decorr]), rho_pen
                )[0]
            )

            if max_viol <= tol:
                break
            rho_pen *= rho_growth

        return W, t, max_viol, outer + 1

    candidates = []
    for _ in range(n_starts):
        W0 = _random_oblique(p, rng)
        t0 = min(0.0, float(np.min(np.diag(Rsqrt @ W0))))
        W, t, max_viol, n_used = run_once(W0, t0)
        T = Rinv_sqrt @ W
        Sigma_T = T.T @ R @ T
        feasible = max_viol <= tol
        rank_key = (0 if feasible else 1, 0.0 if feasible else max_viol, -t)
        candidates.append((rank_key, T, Sigma_T, t, max_viol, n_used))

    candidates.sort(key=lambda x: x[0])
    _, T, Sigma_T, t, max_viol, n_used = candidates[0]
    fid = np.diag(R @ T)
    decorr_value = _squared_decorrelation(Sigma_T)

    return {
        "T": T,
        "Sigma_T": Sigma_T,
        "fid": fid,
        "t_achieved": float(t),
        "decorr_value": float(decorr_value),
        "feasible": bool(max_viol <= tol),
        "max_violation": float(max_viol),
        "n_outer_used": int(n_used),
    }


# ==========================================================================
# rho_star(R): maximum common fidelity under exact decorrelation
# ==========================================================================


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
    """
    Numerically estimate rho_star(R).

    Every exact decorrelator is T = R^{-1/2} Q with Q in O(p). This routine
    solves
        maximize t
        subject to diag(R^{1/2} Q) >= t,
                   Q'Q = I.

    Orthogonality is exact through the Stiefel manifold. Fidelity inequalities
    use ALM. Because the max-min problem is non-convex, the returned rho_star is
    a multistart best-found estimate rather than a global certificate.
    """
    _require_pymanopt()
    R = _validate_spd_correlation(R)
    p = R.shape[0]
    Rsqrt, Rinv_sqrt = matrix_sqrt_and_inv_sqrt(R)
    rng = np.random.default_rng(seed)

    def run_once(Q0, t0):
        lam = np.zeros(p)
        rho_pen = float(rho0)
        Q, t = Q0.copy(), float(t0)
        max_viol = np.inf

        for outer in range(n_outer):
            def cost(Qt, lam=lam.copy(), rho_pen=rho_pen):
                Q, t_arr = Qt
                t_scalar = t_arr[0]
                fid = anp.diag(Rsqrt @ Q)
                viol = t_scalar - fid
                return -t_scalar + _alm_penalty_term(viol, lam, rho_pen)

            manifold = Product([Stiefel(p, p), Euclidean(1)])

            @pymanopt.function.autograd(manifold)
            def wrapped_cost(Q, t_arr):
                return cost((Q, t_arr))

            problem = pymanopt.Problem(manifold, wrapped_cost)
            optimizer = TrustRegions(verbosity=0, max_iterations=max_iterations)
            result = optimizer.run(problem, initial_point=(Q, np.array([t])))
            Q, t_arr = result.point
            t = float(t_arr[0])

            fid = np.diag(Rsqrt @ Q)
            viol = t - fid
            max_viol = max(0.0, float(np.max(viol)))
            lam = _alm_multiplier_update(viol, lam, rho_pen)

            if max_viol <= tol:
                break
            rho_pen *= rho_growth

        return Q, t, max_viol, outer + 1

    candidates = []
    for _ in range(n_starts):
        Q0 = _random_orthogonal(p, rng)
        t0 = float(np.min(np.diag(Rsqrt @ Q0)))
        Q, t, max_viol, n_used = run_once(Q0, t0)
        T = Rinv_sqrt @ Q
        Sigma_T = T.T @ R @ T
        fid = np.diag(R @ T)
        feasible = max_viol <= tol
        rank_key = (0 if feasible else 1, 0.0 if feasible else max_viol, -t)
        candidates.append((rank_key, Q, T, Sigma_T, fid, t, max_viol, n_used))

    candidates.sort(key=lambda x: x[0])
    _, Q, T, Sigma_T, fid, t, max_viol, n_used = candidates[0]

    lower_bound = float(np.min(np.diag(Rsqrt)))
    upper_bound = float(np.trace(Rsqrt) / p)

    return {
        "rho_star": float(t),
        "Q": Q,
        "T": T,
        "Sigma_T": Sigma_T,
        "fid": fid,
        "feasible": bool(max_viol <= tol),
        "max_violation": float(max_viol),
        "n_outer_used": int(n_used),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "orthogonality_error": float(np.max(np.abs(Q.T @ Q - np.eye(p)))),
        "decorrelation_error": float(np.max(np.abs(Sigma_T - np.eye(p)))),
        "global_certified": False,
    }


def rho_star_subset_upper_bound(R, variable_names=None):
    """
    Rigorous subset upper bound for rho_star(R).

    For every nonempty subset S of columns of A = R^(1/2),
        rho_star(R) <= ||A[:, S]||_* / |S|,
    where ||.||_* is the nuclear norm. Taking the minimum over all subsets
    produces an upper bound at least as strong as trace(R^(1/2)) / p.

    Exhaustive enumeration is practical for WDI (p=8) and Wine (p=13); it is
    NOT tractable for p=50 (2^50 subsets) -- for larger systems, use only the
    full-set trace bound returned in solve_rho_star_exact's upper_bound field.
    """
    R = np.asarray(R, dtype=float)
    R = 0.5 * (R + R.T)
    p = R.shape[0]
    Rsqrt, _ = matrix_sqrt_and_inv_sqrt(R)

    trace_bound = float(np.trace(Rsqrt) / p)
    best_bound = trace_bound
    best_subset = tuple(range(p))
    best_nuclear_norm = float(np.trace(Rsqrt))

    for k in range(1, p + 1):
        for S in combinations(range(p), k):
            A_S = Rsqrt[:, S]
            singular_values = np.linalg.svd(A_S, compute_uv=False)
            nuclear_norm = float(np.sum(singular_values))
            bound = nuclear_norm / k
            if bound < best_bound:
                best_bound = bound
                best_subset = S
                best_nuclear_norm = nuclear_norm

    subset_names = (
        list(best_subset) if variable_names is None
        else [variable_names[j] for j in best_subset]
    )

    return {
        "upper_bound": best_bound,
        "trace_bound": trace_bound,
        "subset_indices": best_subset,
        "subset_names": subset_names,
        "subset_size": len(best_subset),
        "nuclear_norm": best_nuclear_norm,
    }


# ==========================================================================
# Exact-support sparse parameterization and diagnostics
# ==========================================================================


@dataclass(frozen=True)
class SupportParameterization:
    allowed_mask: np.ndarray
    indices_by_column: tuple
    slices_by_column: tuple
    n_parameters: int


def _prepare_allowed_mask(allowed_mask: np.ndarray, p: int) -> np.ndarray:
    mask = np.asarray(allowed_mask, dtype=bool)
    if mask.shape != (p, p):
        raise ValueError(f"allowed_mask must have shape {(p, p)}.")
    mask = mask.copy()
    np.fill_diagonal(mask, True)
    if np.any(mask.sum(axis=0) == 0):
        raise ValueError("Every transformation column needs at least one allowed entry.")
    return mask


def support_from_threshold(R: np.ndarray, threshold: float) -> np.ndarray:
    """Allowed support using |R_ij| > threshold, with the diagonal always allowed."""
    R = _validate_spd_correlation(R)
    if threshold < 0:
        raise ValueError("threshold must be nonnegative.")
    allowed = np.abs(R) > threshold
    np.fill_diagonal(allowed, True)
    return allowed


def allowed_mask_from_off_graph_mask(off_graph_mask: np.ndarray) -> np.ndarray:
    """Convert a legacy forbidden-entry mask into an allowed-entry mask."""
    off_graph_mask = np.asarray(off_graph_mask)
    if off_graph_mask.ndim != 2 or off_graph_mask.shape[0] != off_graph_mask.shape[1]:
        raise ValueError("off_graph_mask must be square.")
    allowed = ~off_graph_mask.astype(bool)
    np.fill_diagonal(allowed, True)
    return allowed


def _make_support_parameterization(allowed_mask: np.ndarray) -> SupportParameterization:
    p = allowed_mask.shape[0]
    indices = []
    slices = []
    cursor = 0
    for j in range(p):
        idx = np.flatnonzero(allowed_mask[:, j])
        indices.append(idx)
        slices.append(slice(cursor, cursor + len(idx)))
        cursor += len(idx)
    return SupportParameterization(
        allowed_mask=allowed_mask,
        indices_by_column=tuple(indices),
        slices_by_column=tuple(slices),
        n_parameters=cursor,
    )


def _unpack_support(theta: np.ndarray, param: SupportParameterization) -> np.ndarray:
    p = param.allowed_mask.shape[0]
    T = np.zeros((p, p), dtype=float)
    for j, (idx, sl) in enumerate(zip(param.indices_by_column, param.slices_by_column)):
        T[idx, j] = theta[sl]
    return T


def _pack_support(T: np.ndarray, param: SupportParameterization) -> np.ndarray:
    theta = np.empty(param.n_parameters, dtype=float)
    for j, (idx, sl) in enumerate(zip(param.indices_by_column, param.slices_by_column)):
        theta[sl] = T[idx, j]
    return theta


def _random_support_point(
    R: np.ndarray, param: SupportParameterization, rng: np.random.Generator
) -> np.ndarray:
    """Random exact-support T with t_j' R t_j = 1 for every column."""
    p = R.shape[0]
    T = np.zeros((p, p), dtype=float)
    for j, idx in enumerate(param.indices_by_column):
        u = rng.normal(size=len(idx))
        G = R[np.ix_(idx, idx)]
        norm2 = float(u @ G @ u)
        while norm2 <= np.finfo(float).eps:
            u = rng.normal(size=len(idx))
            norm2 = float(u @ G @ u)
        T[idx, j] = u / np.sqrt(norm2)
    return T


def support_dof_diagnostic(allowed_mask: np.ndarray):
    """
    Degrees-of-freedom diagnostic for a declared support.

    Generic dimension heuristic, not an infeasibility proof for a particular
    R. Specially aligned correlation matrices can admit sparse exact
    decorrelators even when the zero restrictions outnumber rotational
    degrees of freedom.
    """
    mask = np.asarray(allowed_mask, dtype=bool)
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError("allowed_mask must be square.")
    p = mask.shape[0]
    mask = _prepare_allowed_mask(mask, p)
    n_free = int(mask.sum())
    n_zero = int(p * p - n_free)
    dim_orthogonal = int(p * (p - 1) // 2)
    symmetric_equations = int(p * (p + 1) // 2)
    offdiag_allowed = int(mask.sum() - p)
    offdiag_total = int(p * (p - 1))
    return {
        "p": p,
        "n_free_entries": n_free,
        "n_forced_zero_entries": n_zero,
        "allowed_offdiag_entries": offdiag_allowed,
        "total_offdiag_entries": offdiag_total,
        "offdiag_density": offdiag_allowed / offdiag_total,
        "dim_exact_decorrelator_family": dim_orthogonal,
        "n_exact_decorrelation_equations": symmetric_equations,
        "generic_zero_overconstraint": n_zero > dim_orthogonal,
        "generic_equation_overconstraint": n_free < symmetric_equations,
    }


def _support_constraint_diagnostics(R, T, rho_min=None):
    Sigma = T.T @ R @ T
    unit_error = float(np.max(np.abs(np.diag(Sigma) - 1.0)))
    off = Sigma - np.diag(np.diag(Sigma))
    max_off = float(np.max(np.abs(off)))
    mean_off = float(np.mean(np.abs(off[~np.eye(T.shape[1], dtype=bool)])))
    fid = np.diag(R @ T)
    fid_violation = 0.0
    if rho_min is not None:
        fid_violation = max(0.0, float(rho_min - np.min(fid)))
    return Sigma, fid, unit_error, max_off, mean_off, fid_violation


def solve_primal_support_hard(
    R,
    rho_min,
    allowed_mask,
    n_starts=100,
    max_iterations=2000,
    ftol=1e-12,
    constraint_tol=1e-6,
    seed=None,
):
    """
    Sparse CORAL frontier with exact support and hard fidelity constraints.

    Forbidden coefficients are removed from the parameter vector, so support
    is exact by construction. SLSQP minimizes the smooth squared-decorrelation
    objective subject to unit-variance equalities and fidelity inequalities.

    This problem is non-convex. The best multistart solution is a numerical
    best-found solution and is not a global-optimality certificate.
    """
    R = _validate_spd_correlation(R)
    p = R.shape[0]
    allowed = _prepare_allowed_mask(allowed_mask, p)
    param = _make_support_parameterization(allowed)
    rng = np.random.default_rng(seed)

    if not (0.0 <= rho_min <= 1.0):
        raise ValueError("rho_min must lie in [0, 1].")

    def objective(theta):
        T = _unpack_support(theta, param)
        Sigma = T.T @ R @ T
        return _squared_decorrelation(Sigma)

    def unit_eq(theta):
        T = _unpack_support(theta, param)
        return np.diag(T.T @ R @ T) - 1.0

    def fidelity_ineq(theta):
        T = _unpack_support(theta, param)
        return np.diag(R @ T) - rho_min

    constraints = [
        {"type": "eq", "fun": unit_eq},
        {"type": "ineq", "fun": fidelity_ineq},
    ]

    candidates = []
    for start in range(n_starts):
        T0 = _random_support_point(R, param, rng)

        if start == 0 and np.all(np.diag(allowed)):
            T0 = np.eye(p)

        theta0 = _pack_support(T0, param)
        result = minimize(
            objective,
            theta0,
            method="SLSQP",
            constraints=constraints,
            options={"maxiter": max_iterations, "ftol": ftol, "disp": False},
        )

        T = _unpack_support(result.x, param)
        Sigma, fid, unit_err, max_off, mean_off, fid_viol = _support_constraint_diagnostics(
            R, T, rho_min=rho_min
        )
        support_err = float(np.max(np.abs(T[~allowed]))) if np.any(~allowed) else 0.0
        max_violation = max(unit_err, fid_viol, support_err)
        feasible = max_violation <= constraint_tol
        rank_key = (
            0 if feasible else 1,
            0.0 if feasible else max_violation,
            _squared_decorrelation(Sigma),
        )
        candidates.append(
            {
                "rank_key": rank_key,
                "result": result,
                "T": T,
                "Sigma_T": Sigma,
                "fid": fid,
                "unitvar_error": unit_err,
                "support_error": support_err,
                "fidelity_violation": fid_viol,
                "max_violation": max_violation,
                "max_offdiag": max_off,
                "mean_offdiag": mean_off,
                "decorr_value": _squared_decorrelation(Sigma),
                "feasible": feasible,
            }
        )

    candidates.sort(key=lambda d: d["rank_key"])
    best = candidates[0].copy()
    best.pop("rank_key")
    best["optimizer_success"] = bool(best["result"].success)
    best["optimizer_message"] = str(best["result"].message)
    best["n_starts"] = int(n_starts)
    best["global_certified"] = False
    best["allowed_mask"] = allowed
    best["dof"] = support_dof_diagnostic(allowed)
    best["start_decorr_values"] = np.array([c["decorr_value"] for c in candidates])
    best.pop("result")
    return best


def solve_support_squared_floor_hard(
    R,
    allowed_mask,
    n_starts=100,
    max_iterations=3000,
    ftol=1e-12,
    constraint_tol=1e-6,
    seed=None,
):
    """
    Estimate the exact-support floor for CORAL's squared decorrelation objective.

    No fidelity constraint imposed. Non-convex; returned value is a multistart
    best-found estimate, not a global-optimality certificate.
    """
    R = _validate_spd_correlation(R)
    p = R.shape[0]
    allowed = _prepare_allowed_mask(allowed_mask, p)
    param = _make_support_parameterization(allowed)
    rng = np.random.default_rng(seed)

    def objective(theta):
        T = _unpack_support(theta, param)
        return _squared_decorrelation(T.T @ R @ T)

    def unit_eq(theta):
        T = _unpack_support(theta, param)
        return np.diag(T.T @ R @ T) - 1.0

    constraints = [{"type": "eq", "fun": unit_eq}]
    candidates = []

    for start in range(n_starts):
        T0 = np.eye(p) if start == 0 else _random_support_point(R, param, rng)
        theta0 = _pack_support(T0, param)
        result = minimize(
            objective,
            theta0,
            method="SLSQP",
            constraints=constraints,
            options={"maxiter": max_iterations, "ftol": ftol, "disp": False},
        )

        T = _unpack_support(result.x, param)
        Sigma, fid, unit_err, max_off, mean_off, _ = _support_constraint_diagnostics(R, T)
        support_err = float(np.max(np.abs(T[~allowed]))) if np.any(~allowed) else 0.0
        max_violation = max(unit_err, support_err)
        feasible = max_violation <= constraint_tol
        decorr_value = _squared_decorrelation(Sigma)
        rank_key = (
            0 if feasible else 1,
            0.0 if feasible else max_violation,
            decorr_value,
        )
        candidates.append(
            {
                "rank_key": rank_key,
                "result": result,
                "T": T,
                "Sigma_T": Sigma,
                "fid": fid,
                "squared_floor_estimate": decorr_value,
                "max_offdiag": max_off,
                "mean_offdiag": mean_off,
                "unitvar_error": unit_err,
                "support_error": support_err,
                "max_violation": max_violation,
                "feasible": feasible,
            }
        )

    candidates.sort(key=lambda d: d["rank_key"])
    best = candidates[0].copy()
    best.pop("rank_key")
    best["optimizer_success"] = bool(best["result"].success)
    best["optimizer_message"] = str(best["result"].message)
    best["n_starts"] = int(n_starts)
    best["global_certified"] = False
    best["exact_decorrelator_found"] = bool(
        best["squared_floor_estimate"] <= constraint_tol
    )
    best["allowed_mask"] = allowed
    best["dof"] = support_dof_diagnostic(allowed)
    best["start_squared_values"] = np.array(
        [c["squared_floor_estimate"] for c in candidates]
    )
    best.pop("result")
    return best


def solve_support_floor_hard(
    R,
    allowed_mask,
    n_starts=100,
    max_iterations=3000,
    ftol=1e-12,
    constraint_tol=1e-6,
    seed=None,
):
    """
    Estimate the support-induced min-max residual-correlation floor.

    Non-convex; returned floor is a multistart best-found value, not a proof
    of the global minimum.
    """
    R = _validate_spd_correlation(R)
    p = R.shape[0]
    allowed = _prepare_allowed_mask(allowed_mask, p)
    param = _make_support_parameterization(allowed)
    rng = np.random.default_rng(seed)
    pairs = np.array([(i, j) for i in range(p) for j in range(i + 1, p)], dtype=int)

    def unpack_x(x):
        theta = x[:-1]
        u = float(x[-1])
        return _unpack_support(theta, param), u

    def objective(x):
        return float(x[-1])

    def unit_eq(x):
        T, _ = unpack_x(x)
        return np.diag(T.T @ R @ T) - 1.0

    def corr_epigraph_ineq(x):
        T, u = unpack_x(x)
        Sigma = T.T @ R @ T
        c = Sigma[pairs[:, 0], pairs[:, 1]]
        return np.concatenate((u - c, u + c, np.array([u])))

    constraints = [
        {"type": "eq", "fun": unit_eq},
        {"type": "ineq", "fun": corr_epigraph_ineq},
    ]

    candidates = []
    for start in range(n_starts):
        T0 = _random_support_point(R, param, rng)
        if start == 0:
            T0 = np.eye(p)
        Sigma0 = T0.T @ R @ T0
        off0 = Sigma0 - np.diag(np.diag(Sigma0))
        u0 = float(np.max(np.abs(off0))) + 1e-4
        x0 = np.concatenate((_pack_support(T0, param), np.array([u0])))

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            constraints=constraints,
            options={"maxiter": max_iterations, "ftol": ftol, "disp": False},
        )

        T, u = unpack_x(result.x)
        Sigma, fid, unit_err, max_off, mean_off, _ = _support_constraint_diagnostics(R, T)
        c = Sigma[pairs[:, 0], pairs[:, 1]]
        epi_viol = max(
            0.0,
            float(np.max(c - u)),
            float(np.max(-c - u)),
            float(-u),
        )
        support_err = float(np.max(np.abs(T[~allowed]))) if np.any(~allowed) else 0.0
        max_violation = max(unit_err, epi_viol, support_err)
        feasible = max_violation <= constraint_tol
        rank_key = (
            0 if feasible else 1,
            0.0 if feasible else max_violation,
            max_off,
        )
        candidates.append(
            {
                "rank_key": rank_key,
                "result": result,
                "T": T,
                "Sigma_T": Sigma,
                "fid": fid,
                "floor_estimate": max_off,
                "epigraph_u": float(u),
                "unitvar_error": unit_err,
                "support_error": support_err,
                "epigraph_violation": epi_viol,
                "max_violation": max_violation,
                "mean_offdiag": mean_off,
                "feasible": feasible,
            }
        )

    candidates.sort(key=lambda d: d["rank_key"])
    best = candidates[0].copy()
    best.pop("rank_key")
    best["optimizer_success"] = bool(best["result"].success)
    best["optimizer_message"] = str(best["result"].message)
    best["n_starts"] = int(n_starts)
    best["global_certified"] = False
    best["exact_decorrelator_found"] = bool(best["floor_estimate"] <= constraint_tol)
    best["allowed_mask"] = allowed
    best["dof"] = support_dof_diagnostic(allowed)
    best["start_floor_values"] = np.array([c["floor_estimate"] for c in candidates])
    best.pop("result")
    return best


# ==========================================================================
# Reporting helpers
# ==========================================================================


def print_pca_benchmark(label: str, R: np.ndarray):
    pca = pca_fidelity_sign_invariant(R)
    print(
        f"{label}: best possible minimum fidelity = {pca['best_min_fidelity']:.4f} "
        f"(bottleneck assignment); best possible mean fidelity = "
        f"{pca['best_mean_fidelity']:.4f} (sum-optimal assignment)"
    )
    return pca


def print_zca_benchmark(label: str, R: np.ndarray):
    zca = zca_benchmark(R)
    print(
        f"{label}: ZCA min fidelity = {zca['min_fidelity']:.4f}, "
        f"mean fidelity = {zca['mean_fidelity']:.4f} "
        f"(max offdiag = {zca['max_offdiag']:.2e}, confirms exact decorrelation)"
    )
    return zca


def print_sparse_pca_benchmark(label: str, R: np.ndarray, target_nnz=None, **kwargs):
    if not _HAS_SKLEARN:
        print(f"{label}: [skipped -- scikit-learn not installed]")
        return None

    if target_nnz is not None:
        best, all_results = sparse_pca_at_matched_density(R, target_nnz, **kwargs)
        print(
            f"{label}: sparse PCA at matched density "
            f"(target nnz={target_nnz}, actual nnz={best['nnz']}, alpha={best['alpha']}): "
            f"min fidelity={best['min_fidelity']:.4f}, mean fidelity={best['mean_fidelity']:.4f}, "
            f"max offdiag={best['max_offdiag']:.4f}, mean offdiag={best['mean_offdiag']:.4f}"
        )
        return best, all_results
    else:
        results = sparse_pca_benchmark(R, **kwargs)
        print(f"{label}: sparse PCA sweep")
        print(f"{'alpha':>8}{'nnz':>8}{'MinFid':>10}{'MeanFid':>10}{'MaxOff':>10}{'MeanOff':>10}")
        for r in results:
            print(
                f"{r['alpha']:>8.2f}{r['nnz']:>8d}{r['min_fidelity']:>10.4f}"
                f"{r['mean_fidelity']:>10.4f}{r['max_offdiag']:>10.4f}{r['mean_offdiag']:>10.4f}"
            )
        return results


def print_support_diagnostic(label: str, allowed_mask: np.ndarray):
    d = support_dof_diagnostic(allowed_mask)
    print(
        f"{label}: free={d['n_free_entries']}, forced zeros={d['n_forced_zero_entries']}, "
        f"offdiag density={100*d['offdiag_density']:.1f}%, "
        f"dim O(p)={d['dim_exact_decorrelator_family']}, "
        f"exact equations={d['n_exact_decorrelation_equations']}"
    )
    return d


# ==========================================================================
# Validation / manuscript rerun
# ==========================================================================


if __name__ == "__main__":
    print("=" * 78)
    print("CORAL REVISION: PCA/ZCA/SPARSE-PCA BENCHMARKS, rho_star, EXACT-SUPPORT SPARSITY")
    print("=" * 78)

    # ---------------------------------------------------------- synthetic p=6
    R6 = make_correlation_matrix(p=6, seed=1)
    print("\nSYNTHETIC p=6")
    print_pca_benchmark("PCA", R6)
    print_zca_benchmark("ZCA", R6)
    # Sparse PCA is a sparsity-CONSTRAINED method and is only reported where
    # CORAL is also evaluated under a declared sparsity constraint (see the
    # p=18 exact-support ablation below). R6 has no declared support, so
    # there is nothing for a sparse-PCA nonzero-count target to be matched
    # against; comparing it here would manufacture a comparison rather than
    # report one that actually exists.

    if _HAS_PYMANOPT:
        print("\nHard primal frontier")
        print(f"{'rho_min':>8}{'MaxOffDiag':>14}{'MinFidelity':>14}{'Feasible':>10}{'MaxViol':>12}")
        for rho_min in [0.30, 0.50, 0.70, 0.85, 0.95, 0.99]:
            res = solve_primal_hard(R6, rho_min, n_starts=100, n_outer=20, seed=0)
            max_off, _ = offdiag_stats(res["Sigma_T"])
            print(
                f"{rho_min:>8.2f}{max_off:>14.4f}{res['fid'].min():>14.4f}"
                f"{str(res['feasible']):>10}{res['max_violation']:>12.2e}"
            )

        print("\nDirect exact-decorrelation fidelity threshold rho_star(R)")
        rho_star6 = solve_rho_star_exact(R6, n_starts=100, n_outer=25, seed=0)
        print(
            f"rho_star(best found)={rho_star6['rho_star']:.6f}, "
            f"bounds=[{rho_star6['lower_bound']:.6f}, {rho_star6['upper_bound']:.6f}], "
            f"decorrelation error={rho_star6['decorrelation_error']:.2e}"
        )
    else:
        print("\n[Skipping ALM and rho_star runs: install pymanopt[autograd].]")

    # --------------------------------------------------------- synthetic p=18
    R18 = make_correlation_matrix(p=18, seed=1)
    allowed18 = support_from_threshold(R18, threshold=0.15)
    dof18 = support_dof_diagnostic(allowed18)
    print("\nSYNTHETIC p=18")
    print_pca_benchmark("PCA", R18)
    print_zca_benchmark("ZCA", R18)

    if _HAS_PYMANOPT:
        print("\nHard primal frontier, p=18")
        print(f"{'rho_min':>8}{'MaxOffDiag':>14}{'MeanOffDiag':>14}{'MinFidelity':>14}{'Feasible':>10}{'MaxViol':>12}")
        for rho_min in [0.30, 0.50, 0.70, 0.85, 0.95, 0.99]:
            res = solve_primal_hard(
                R18,
                rho_min,
                n_starts=100,
                n_outer=20,
                seed=1800 + int(round(100 * rho_min)),
            )
            max_off, mean_off = offdiag_stats(res["Sigma_T"])
            print(
                f"{rho_min:>8.2f}{max_off:>14.4f}{mean_off:>14.4f}"
                f"{res['fid'].min():>14.4f}{str(res['feasible']):>10}"
                f"{res['max_violation']:>12.2e}"
            )



    print_support_diagnostic("Declared |R_ij| > 0.15 support", allowed18)
    # Sparse PCA is reported HERE, and only here: p=18 is the sole system
    # with a declared sparsity ablation (allowed18), so it is the only
    # system where CORAL's own exact-support solve gives sparse PCA a
    # legitimate, matched target to be compared against. See the R6, R50,
    # WDI, and Wine blocks for the same reasoning stated at each skip point.
    print_sparse_pca_benchmark(
        "Sparse PCA (matched to declared support density)",
        R18,
        target_nnz=dof18["n_free_entries"],
    )

    print("\nExact-support sparse frontier")
    print(f"{'rho_min':>8}{'MaxOffDiag':>14}{'MeanOffDiag':>14}{'MinFidelity':>14}{'Feasible':>10}")
    for rho_min in [0.00, 0.30, 0.50, 0.70, 0.85, 0.95, 0.99]:
        res = solve_primal_support_hard(
            R18,
            rho_min,
            allowed18,
            n_starts=100,
            seed=100 + int(round(100 * rho_min)),
        )
        print(
            f"{rho_min:>8.2f}{res['max_offdiag']:>14.4f}{res['mean_offdiag']:>14.4f}"
            f"{res['fid'].min():>14.4f}{str(res['feasible']):>10}"
        )

    print("\nExact-support squared-objective floor")
    sq_floor18 = solve_support_squared_floor_hard(R18, allowed18, n_starts=100, seed=0)
    print(
        f"best-found squared floor={sq_floor18['squared_floor_estimate']:.6f}, "
        f"max offdiag={sq_floor18['max_offdiag']:.6f}, "
        f"mean offdiag={sq_floor18['mean_offdiag']:.6f}, "
        f"global certified={sq_floor18['global_certified']}"
    )

    print("\nExact-support min-max floor")
    floor18 = solve_support_floor_hard(R18, allowed18, n_starts=100, seed=0)
    print(
        f"best-found worst-pair floor={floor18['floor_estimate']:.6f}, "
        f"mean offdiag={floor18['mean_offdiag']:.6f}, "
        f"unit-var error={floor18['unitvar_error']:.2e}, "
        f"support error={floor18['support_error']:.2e}, "
        f"global certified={floor18['global_certified']}"
    )

    if _HAS_PYMANOPT:
        rho_star18 = solve_rho_star_exact(R18, n_starts=10, n_outer=25, seed=0)
        print(
            f"rho_star p=18 (best found)={rho_star18['rho_star']:.6f}, "
            f"bounds=[{rho_star18['lower_bound']:.6f}, {rho_star18['upper_bound']:.6f}]"
        )

    # --------------------------------------------------------- synthetic p=50
    import time

    R50 = make_correlation_matrix(p=50, seed=1)

    print("\nSYNTHETIC p=50: DENSE SCALABILITY CHECK")

    base_max50, base_mean50 = offdiag_stats(R50)
    print(f"Baseline correlation: max |offdiag|={base_max50:.4f}, mean |offdiag|={base_mean50:.4f}")

    print_pca_benchmark("PCA", R50)
    print_zca_benchmark("ZCA", R50)
    # No sparse-PCA comparison here -- see note at the R6 block above.
    # p=50 has no declared support ablation, so sparse PCA has no matched
    # target to compare against.

    if _HAS_PYMANOPT:
        print("\nHard primal frontier, p=50")
        print(
            f"{'rho_min':>8}{'MaxOffDiag':>14}{'MeanOffDiag':>14}{'MinFidelity':>14}"
            f"{'Feasible':>10}{'MaxViol':>12}{'Seconds':>12}"
        )

        total_start = time.perf_counter()
        for rho_min in [0.50, 0.70, 0.85, 0.95, 0.99]:
            row_start = time.perf_counter()
            res = solve_primal_hard(
                R50, rho_min, n_starts=5, n_outer=20, seed=5000 + int(round(100 * rho_min))
            )
            elapsed = time.perf_counter() - row_start
            max_off, mean_off = offdiag_stats(res["Sigma_T"])
            print(
                f"{rho_min:>8.2f}{max_off:>14.4f}{mean_off:>14.4f}{res['fid'].min():>14.4f}"
                f"{str(res['feasible']):>10}{res['max_violation']:>12.2e}{elapsed:>12.2f}"
            )
        primal_total = time.perf_counter() - total_start
        print(f"\nTotal p=50 primal frontier time: {primal_total:.2f} seconds")

        print("\nDirect exact-decorrelation fidelity threshold rho_star(R), p=50")
        rho_start = time.perf_counter()
        rho_star50 = solve_rho_star_exact(R50, n_starts=100, n_outer=25, seed=50)
        rho_elapsed = time.perf_counter() - rho_start
        print(
            f"rho_star p=50 (best found)={rho_star50['rho_star']:.6f}, "
            f"bounds=[{rho_star50['lower_bound']:.6f}, {rho_star50['upper_bound']:.6f}], "
            f"decorrelation error={rho_star50['decorrelation_error']:.2e}, time={rho_elapsed:.2f} sec"
        )
        print(
            "NOTE: p=50 upper bound is the full-set trace bound only -- exhaustive "
            "subset enumeration (2^50 subsets) is not tractable at this dimension."
        )
        print(
            f"Certified numerical interval using attained solution as lower bound: "
            f"[{rho_star50['rho_star']:.6f}, {rho_star50['upper_bound']:.6f}]"
        )
    else:
        print("\n[Skipping p=50 ALM and rho_star runs: install pymanopt[autograd].]")

    # -------------------------------------------------------------- WDI 2019
    import os
    import pandas as pd

    wdi_path = "wdi_p8_2019.csv"
    if os.path.exists(wdi_path):
        df = pd.read_csv(wdi_path)
        cols_wdi = [c for c in df.columns if c not in ("Country Name", "Country Code")]
        Xw = df[cols_wdi].values
        Xw = (Xw - Xw.mean(axis=0)) / Xw.std(axis=0, ddof=1)
        Rw = np.corrcoef(Xw.T)
        np.fill_diagonal(Rw, 1.0)
        p_w = Rw.shape[0]

        rho_wdi = solve_rho_star_exact(
            Rw, n_starts=100, n_outer=30, tol=1e-8, max_iterations=750, seed=12345
        )
        print("\nWDI exact-decorrelation fidelity threshold")
        print(
            f"rho_star WDI (best found)={rho_wdi['rho_star']:.6f}, "
            f"bounds=[{rho_wdi['lower_bound']:.6f}, {rho_wdi['upper_bound']:.6f}], "
            f"gap={rho_wdi['upper_bound'] - rho_wdi['rho_star']:.6f}, "
            f"decorrelation error={rho_wdi['decorrelation_error']:.2e}"
        )

        rho_wdi_ub = rho_star_subset_upper_bound(Rw, variable_names=cols_wdi)
        print(f"WDI stronger subset upper bound={rho_wdi_ub['upper_bound']:.6f}")
        print(f"Binding subset (k={rho_wdi_ub['subset_size']}): {rho_wdi_ub['subset_names']}")
        print(f"Certified rho_star interval: [{rho_wdi['rho_star']:.6f}, {rho_wdi_ub['upper_bound']:.6f}]")

        print(f"\nWDI 2019, p={p_w}, n={df.shape[0]}")
        pca_w = print_pca_benchmark("PCA", Rw)
        print_zca_benchmark("ZCA", Rw)
        # No sparse-PCA comparison here -- WDI has no declared support
        # ablation, so there is no matched nonzero-count target to compare
        # against (see note in the R6 block above).
        print("PCA fidelity by source variable:")
        for name, fid, comp in zip(cols_wdi, pca_w["fidelity"], pca_w["component_for_source"]):
            print(f"  {name}: {fid:.4f}  (component {comp + 1})")

        if _HAS_PYMANOPT:
            print("\nWDI hard primal")
            print(f"{'rho_min':>8}{'Hard MinFid':>14}{'Hard MaxOff':>14}{'Feasible':>10}{'MaxViol':>12}")
            for rho_min in [0.50, 0.70, 0.85, 0.95, 0.99]:
                res = solve_primal_hard(Rw, rho_min, n_starts=5, n_outer=20, seed=0)
                max_off, _ = offdiag_stats(res["Sigma_T"])
                print(
                    f"{rho_min:>8.2f}{res['fid'].min():>14.4f}{max_off:>14.4f}"
                    f"{str(res['feasible']):>10}{res['max_violation']:>12.2e}"
                )

            print("\nWDI dual")
            print(f"{'epsilon':>8}{'AchievedT':>14}{'MinFidelity':>14}{'DecorrVal':>12}{'Feasible':>10}")
            for epsilon in [0.01, 0.05, 0.10, 0.20, 0.40]:
                res = solve_dual_hard(Rw, epsilon, n_starts=100, n_outer=20, seed=0)
                print(
                    f"{epsilon:>8.3f}{res['t_achieved']:>14.4f}{res['fid'].min():>14.4f}"
                    f"{res['decorr_value']:>12.4f}{str(res['feasible']):>10}"
                )
    else:
        print(f"\n[Skipping WDI: '{wdi_path}' not found in working directory.]")

    # -------------------------------------------------------------- Wine
    try:
        from sklearn.datasets import load_wine

        d = load_wine()
        Xv = d.data
        Xv = (Xv - Xv.mean(axis=0)) / Xv.std(axis=0, ddof=1)
        Rv = np.corrcoef(Xv.T)
        np.fill_diagonal(Rv, 1.0)
        p_v = Rv.shape[0]

        rho_wine = solve_rho_star_exact(
            Rv, n_starts=100, n_outer=30, tol=1e-8, max_iterations=750, seed=12345
        )
        print("\nWine exact-decorrelation fidelity threshold")
        print(
            f"rho_star Wine (best found)={rho_wine['rho_star']:.6f}, "
            f"bounds=[{rho_wine['lower_bound']:.6f}, {rho_wine['upper_bound']:.6f}], "
            f"gap={rho_wine['upper_bound'] - rho_wine['rho_star']:.6f}, "
            f"decorrelation error={rho_wine['decorrelation_error']:.2e}"
        )

        rho_wine_ub = rho_star_subset_upper_bound(Rv, variable_names=list(d.feature_names))
        print(f"Wine stronger subset upper bound={rho_wine_ub['upper_bound']:.6f}")
        print(f"Binding subset (k={rho_wine_ub['subset_size']}): {rho_wine_ub['subset_names']}")
        print(f"Certified rho_star interval: [{rho_wine['rho_star']:.6f}, {rho_wine_ub['upper_bound']:.6f}]")

        print(f"\nWINE, p={p_v}, n={d.data.shape[0]}")
        pca_v = print_pca_benchmark("PCA", Rv)
        print_zca_benchmark("ZCA", Rv)
        # No sparse-PCA comparison here -- Wine has no declared support
        # ablation, so there is no matched nonzero-count target to compare
        # against (see note in the R6 block above).
        print("PCA fidelity by source variable:")
        for name, fid, comp in zip(d.feature_names, pca_v["fidelity"], pca_v["component_for_source"]):
            print(f"  {name}: {fid:.4f}  (component {comp + 1})")

        if _HAS_PYMANOPT:
            print("\nWine hard primal")
            print(f"{'rho_min':>8}{'Hard MinFid':>14}{'Hard MaxOff':>14}{'Feasible':>10}{'MaxViol':>12}")
            for rho_min in [0.50, 0.70, 0.85, 0.95, 0.99]:
                res = solve_primal_hard(Rv, rho_min, n_starts=100, n_outer=20, seed=0)
                max_off, _ = offdiag_stats(res["Sigma_T"])
                print(
                    f"{rho_min:>8.2f}{res['fid'].min():>14.4f}{max_off:>14.4f}"
                    f"{str(res['feasible']):>10}{res['max_violation']:>12.2e}"
                )

            print("\nWine dual")
            print(f"{'epsilon':>8}{'AchievedT':>14}{'MinFidelity':>14}{'DecorrVal':>12}{'Feasible':>10}")
            for epsilon in [0.01, 0.05, 0.10, 0.20, 0.40]:
                res = solve_dual_hard(Rv, epsilon, n_starts=100, n_outer=20, seed=0)
                print(
                    f"{epsilon:>8.3f}{res['t_achieved']:>14.4f}{res['fid'].min():>14.4f}"
                    f"{res['decorr_value']:>12.4f}{str(res['feasible']):>10}"
                )
    except ImportError:
        print("\n[Skipping wine: scikit-learn not installed.]")