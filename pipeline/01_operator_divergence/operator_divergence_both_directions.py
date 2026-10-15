#!/usr/bin/env python3
"""
operator_divergence_both_directions.py — the FC-versus-EC energy ratio for
every subject, in BOTH directions of the transition, at native length.

WHY
  The published cohort batches computed the ratio in opposite directions:
  HCP from the high-variance state to the quiescent one; ADNI and the
  epilepsy cohort from quiescent to high-variance. For a directed EC
  operator the direction changes the energy, so the cross-cohort
  comparison mixed two quantities. This recomputes all of them both ways.

WHAT IT REPRODUCES EXACTLY
  The batch scripts' own computation (run_adni_nctn_batch.py,
  run_hcp_aud_batch.py): one GraphNet estimate per subject (ridge 1,
  prior 1), normalisation to spectral radius 0.9, B = I, horizon 10, and
  define_states copied verbatim (quiescent = mean of the lowest-variance
  fifth of timepoints, high-variance = mean of the highest, unit norm).

GATE
  With --published, the published batch values in the published direction
  must be reproduced to a relative tolerance of 1e-6 for every subject:
    --convention quiescent_first  (ADNI, epilepsy): E_rest_to_demand,
        E_demand_to_rest, E_fc, teleport_ratio
    --convention highvar_first    (HCP): E_craving_to_rest,
        E_rest_to_craving, E_fc, teleport_ratio
  HCP's published file uses anonymised IDs: pass --id-key with the
  restricted subject key (real_id, anon_id). No identifiers are written.

SELF-CONTAINED: the NeuroSim functions used are vendored below, verbatim,
with source checksums. Saves every 10 subjects and resumes.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import argparse
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

HORIZON = 10
TARGET_RHO = 0.9
RHOS = (0.8, 0.9, 0.95)
# ==========================================================================
# VENDORED NEUROSIM CODE (corrected package version). Source checksums:
#   neurosim/connectivity.py  sha256 2ae501d6f7f3a44ccbe6864566cb3ec7a88626cb95e4c9d923978ad9d5becee6
#     functions: functional_connectivity, ridge_effective_connectivity, graph_laplacian, graphnet_effective_connectivity
#   neurosim/physics.py       sha256 31dc44b64a5ce53201cb191efcc2192f2b00bc0aa4b33760ae1242aeeb416493
#     functions: normalise_matrix, compute_gramian_doubling, minimum_energy
#   neurosim/loader.py        sha256 1387b643754d9e3cc0986d28d34c7bd68d6ed2b7fbef52b26b1f5e73d4f528f8
#     functions: load_connectome, from_arrays
# ==========================================================================

# ---- from neurosim/connectivity.py ----
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge

def functional_connectivity(X: NDArray, detrend: bool = True) -> NDArray:
    """Compute undirected Functional Connectivity via Pearson correlation.

    Provided for direct comparison with Effective Connectivity. **Not**
    recommended as the ``A`` matrix in NeuroSim's physics engine due to
    symmetric structure (see module docstring).

    Parameters
    ----------
    X : (N, T) array  - BOLD time series, N regions * T timepoints.
    detrend : bool    - Remove linear trend before correlation (recommended).

    Returns
    -------
    FC : (N, N) ndarray  - Symmetric correlation matrix with zero diagonal.
    """
    X = np.asarray(X, dtype=float)
    if detrend:
        from scipy.signal import detrend as sp_detrend
        X = sp_detrend(X, axis=1)
    FC = np.corrcoef(X)
    np.fill_diagonal(FC, 0.0)
    return FC


def ridge_effective_connectivity(
    X: NDArray,
    alpha: float = 1.0,
    lag: int = 1,
) -> NDArray:
    """Estimate directed Effective Connectivity via MVAR Ridge Regression.

    Regresses X[t+lag] onto X[t] for every region independently, recovering
    the first-order Multivariate AutoRegressive (MVAR) transition matrix.
    This is a computationally efficient approximation to Dynamic Causal
    Modelling that recovers causal directionality.

    Parameters
    ----------
    X     : (N, T) ndarray - BOLD time series.
    alpha : float          - L2 regularisation strength (Ridge penalty).
    lag   : int            - Temporal lag in TR steps (default=1).

    Returns
    -------
    EC : (N, N) ndarray  - Asymmetric effective connectivity matrix.
                           EC[i, j] = causal influence of region j on region i.
    """
    X = np.asarray(X, dtype=float)
    N, T = X.shape
    X_past   = X[:, :-lag].T   # (T-lag, N)
    X_future = X[:, lag:].T    # (T-lag, N)

    EC = np.zeros((N, N))
    model = Ridge(alpha=alpha, fit_intercept=False)
    for i in range(N):
        model.fit(X_past, X_future[:, i])
        EC[i, :] = model.coef_

    return EC


def graph_laplacian(SC: NDArray, normalised: bool = True) -> NDArray:
    """Compute the Graph Laplacian of the structural connectome.

    L_sc = D - SC   (unnormalised)
    L_sc = I - D^{-1/2} SC D^{-1/2}   (normalised)

    Parameters
    ----------
    SC         : (N, N) ndarray - Structural connectome (symmetric, non-negative).
    normalised : bool           - Use normalised Laplacian (recommended).

    Returns
    -------
    L : (N, N) ndarray  - Graph Laplacian.
    """
    SC = np.asarray(SC, dtype=float)
    SC = (SC + SC.T) / 2.0  # enforce symmetry
    np.fill_diagonal(SC, 0.0)

    D = np.diag(SC.sum(axis=1))
    L = D - SC

    if normalised:
        d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(D), 1e-12)))
        L = d_inv_sqrt @ L @ d_inv_sqrt

    return L


def graphnet_effective_connectivity(
    X: NDArray,
    SC: NDArray,
    lambda_ridge: float = 1.0,
    lambda_graph: float = 1.0,
    max_iter: int = 3000,
    tol: float = 1e-6,
    lag: int = 1,
) -> NDArray:
    """Estimate EC with Graph Laplacian regularisation (GraphNet).

    Minimises the objective::

        J(A) = ||X_{t+1} - A X_t||²_F
               + λ_ridge · ||A||²_F
               + λ_graph · Tr(A^T L_sc A)

    via **FISTA** (Beck & Teboulle, 2009) - proximal gradient descent with
    Nesterov momentum. The combined quadratic penalty (Ridge + GraphNet) has
    closed-form proximal operator, enabling fast convergence.

    CORRECTED (see CHANGELOG): two fixes applied to the original version.
    (1) max_iter raised from 500 to 3000 -- 500 was insufficient for
    lower-T/higher-N problems (e.g. ADNI, N=400, T=197), confirmed via
    direct convergence-trajectory checks showing genuine, still-ongoing
    convergence at 500 iterations, not a stall. (2) The convergence check
    was reordered: A is now assigned from A_new BEFORE the tolerance
    check, not after. The original ordering meant that whenever the loop
    broke, it returned the value from BEFORE the update that actually
    satisfied convergence -- for problems that converged in very few
    iterations (confirmed on HCP and UNAM data), this meant the function
    silently returned the unmodified ridge-only starting point,
    regardless of lambda_graph, since the one update that would have
    incorporated the graph term was computed and then discarded on every
    call.

    Parameters
    ----------
    X            : (N, T) ndarray - BOLD time series.
    SC           : (N, N) ndarray - Structural connectome (DTI).
    lambda_ridge : float          - L2 / Ridge penalty weight.
    lambda_graph : float          - Graph Laplacian penalty weight.
    max_iter     : int            - Maximum FISTA iterations.
    tol          : float          - Convergence tolerance (||ΔA||_F / ||A||_F).
    lag          : int            - Temporal lag (default=1 TR).

    Returns
    -------
    EC : (N, N) ndarray  - GraphNet-regularised effective connectivity.
    """
    X = np.asarray(X, dtype=float)
    SC = np.asarray(SC, dtype=float)
    N, T = X.shape
    X_t = X[:, :-lag]
    X_tp1 = X[:, lag:]
    L = graph_laplacian(SC, normalised=True)

    XXT = X_t @ X_t.T
    YXT = X_tp1 @ X_t.T
    Reg = lambda_ridge * np.eye(N) + lambda_graph * L

    # FISTA setup
    A = ridge_effective_connectivity(X, alpha=lambda_ridge, lag=lag)
    A_prev = A.copy()
    t_k = 1.0
    eigmax_XXT = np.max(np.abs(np.linalg.eigvalsh(XXT)))
    step = 1.0 / (2.0 * eigmax_XXT + 2.0 * np.max(np.abs(np.linalg.eigvalsh(Reg))) + 1e-8)

    for k in range(max_iter):
        Y = A + ((t_k - 1) / (t_k + 1)) * (A - A_prev)
        grad_data = 2.0 * (Y @ XXT - YXT)
        grad_reg = 2.0 * (Reg @ Y.T).T
        A_new = Y - step * (grad_data + grad_reg)

        A_prev = A.copy()
        A = A_new
        if np.linalg.norm(A - A_prev, 'fro') / (np.linalg.norm(A_prev, 'fro') + 1e-12) < tol:
            break
        t_k = (1.0 + np.sqrt(1.0 + 4.0 * t_k**2)) / 2.0

    return A

# ---- from neurosim/physics.py ----
import math
import warnings
from typing import Callable, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigvals, solve_discrete_lyapunov

def normalise_matrix(
    A: NDArray,
    target_rho: float = 0.9,
) -> NDArray:
    """Normalise adjacency matrix so its spectral radius equals target_rho.

    A stable DT-LTI system requires ρ(A) < 1. This function rescales A so
    that ρ(A_norm) = target_rho exactly, preserving the directional
    asymmetry of the matrix (eigenvalue ratios are unchanged).

    Parameters
    ----------
    A : (N, N) array-like
        Raw connectivity matrix (structural or effective). May be unstable.
    target_rho : float
        Desired spectral radius after normalisation (default 0.9).

    Returns
    -------
    A_norm : (N, N) ndarray
        Rescaled matrix with ρ(A_norm) = target_rho.

    Raises
    ------
    ValueError
        If the matrix has a near-zero spectral radius (degenerate input).

    Notes
    -----
    The stability guard (ρ ≥ 1 check) belongs in ``compute_gramian_doubling``,
    not here. ``normalise_matrix`` is designed to *fix* unstable matrices.
    """
    A = np.asarray(A, dtype=float)
    rho = np.max(np.abs(eigvals(A)))
    if rho < 1e-12:
        raise ValueError(
            f"Matrix has near-zero spectral radius (ρ = {rho:.2e}). "
            "Check your input - the matrix may be all-zeros or degenerate."
        )
    return A * (target_rho / rho)


def compute_gramian_doubling(
    A: NDArray,
    B: NDArray,
    T: int,
) -> NDArray:
    """Compute the Discrete Finite-Horizon Reachability Gramian via Doubling.

    Implements the iterative squaring (Van Loan) approach to compute::

        W_T = Σ_{k=0}^{T-1} A^k B B^T (A^T)^k

    in O(log T . N³) rather than O(T . N³).

    Parameters
    ----------
    A : (N, N) ndarray
        Normalised effective connectivity matrix (ρ(A) < 1 required).
    B : (N, M) ndarray
        Input matrix. For full-rank control, use ``np.eye(N)``.
    T : int
        Finite time horizon (number of TR steps).

    Returns
    -------
    W_T : (N, N) ndarray
        Positive semi-definite Reachability Gramian.

    Raises
    ------
    ValueError
        If ρ(A) ≥ 1 (unstable system). Call ``normalise_matrix()`` first.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    N = A.shape[0]

    rho = np.max(np.abs(eigvals(A)))
    if rho >= 1.0:
        raise ValueError(
            f"Spectral radius ρ(A) = {rho:.4f} ≥ 1. "
            "System is unstable. Call normalise_matrix() first."
        )

    # Binary-decomposition doubling
    W       = np.zeros((N, N))
    Phi_acc = np.eye(N)      # A^{accumulated bits}
    Phi_j   = A.copy()       # A^{2^j}
    Q_j     = B @ B.T        # BBᵀ doubled at each step

    iters = math.ceil(math.log2(T)) + 1 if T > 1 else 1
    t = T

    for _ in range(iters):
        if t & 1:
            W      += Phi_acc @ Q_j @ Phi_acc.T
            Phi_acc = Phi_acc @ Phi_j
        Q_j   = Q_j + Phi_j @ Q_j @ Phi_j.T
        Phi_j = Phi_j @ Phi_j
        t >>= 1
        if t == 0:
            break

    return (W + W.T) / 2.0


def minimum_energy(
    A: NDArray,
    B: NDArray,
    x0: NDArray,
    xT: NDArray,
    T: int,
    rcond: float = 1e-10,
) -> Tuple[float, NDArray]:
    """Compute the minimum control energy for a state transition.

    The optimal (minimum-norm) control input that steers the DT-LTI system
    from state x0 to xT in exactly T steps has energy::

        E* = (xT - A^T x0)^T  W_T^{-1}  (xT - A^T x0)

    Parameters
    ----------
    A  : (N, N) ndarray - normalised connectivity matrix.
    B  : (N, M) ndarray - input matrix.
    x0 : (N,) ndarray  - initial brain state.
    xT : (N,) ndarray  - target brain state.
    T  : int           - finite horizon (TR steps).
    rcond : float      - pseudo-inverse cutoff for ill-conditioned Gramians.

    Returns
    -------
    energy : float       - Scalar minimum control energy E*.
    u_opt  : (N,) ndarray - Optimal first-step control input.
    """
    W_T   = compute_gramian_doubling(A, B, T)
    A_T   = np.linalg.matrix_power(A, T)
    delta = xT - A_T @ x0

    cond = np.linalg.cond(W_T)
    if cond > 1e12:
        warnings.warn(
            f"Gramian condition number {cond:.2e} is very large. "
            "Energy estimate may be unreliable.",
            RuntimeWarning,
        )

    W_inv  = np.linalg.pinv(W_T, rcond=rcond)
    energy = float(delta @ W_inv @ delta)
    u_opt  = B.T @ W_inv @ delta

    return energy, u_opt

# ---- from neurosim/loader.py ----
import os, warnings
from pathlib import Path
import numpy as np

def load_connectome(connectome_path, normalise=False, threshold_pct=None):
    connectome_path = Path(connectome_path)
    if not connectome_path.exists():
        raise FileNotFoundError(f"Connectome file not found: {connectome_path}")
    suffix = connectome_path.suffix.lower()
    if suffix == ".npy":
        SC = np.load(str(connectome_path))
    elif suffix == ".npz":
        data = np.load(str(connectome_path))
        SC = data["SC"] if "SC" in data else data[list(data.keys())[0]]
    elif suffix in (".csv", ".tsv", ".txt"):
        sep = "\t" if suffix == ".tsv" else ","
        try:
            import pandas as pd
            SC = pd.read_csv(connectome_path, sep=sep, header=None).values.astype(float)
        except ImportError:
            SC = np.loadtxt(str(connectome_path), delimiter=sep)
    else:
        try: SC = np.loadtxt(str(connectome_path))
        except: raise ValueError(f"Cannot read connectome file with extension '{suffix}'.")
    if SC.ndim != 2 or SC.shape[0] != SC.shape[1]:
        raise ValueError(f"Connectome must be a square matrix. Got shape {SC.shape}.")
    SC = (SC + SC.T) / 2.0
    np.fill_diagonal(SC, 0.0)
    if threshold_pct is not None:
        nonzero = SC[SC > 0]
        if len(nonzero) > 0:
            SC[SC < np.percentile(nonzero, threshold_pct)] = 0.0
    if normalise:
        mx = SC.max()
        if mx > 0: SC = SC / mx
    return SC.astype(float)


def from_arrays(X, SC=None, subject_id="unknown", validate=True):
    X = np.asarray(X, dtype=float)
    if validate:
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (N_regions, T_timepoints). Got shape {X.shape}.")
        N, T = X.shape
        if T < N:
            warnings.warn(f"X has shape ({N}, {T}) — T < N. Check orientation.", UserWarning, stacklevel=2)
        if np.any(np.isnan(X)):
            raise ValueError("X contains NaN values.")
        if np.any(np.isinf(X)):
            raise ValueError("X contains infinite values.")
    N, T = X.shape
    if SC is not None:
        SC = np.asarray(SC, dtype=float)
        if validate:
            if SC.shape != (N, N):
                raise ValueError(f"SC shape {SC.shape} does not match X's N_regions={N}.")
            if not np.allclose(SC, SC.T, atol=1e-6):
                warnings.warn("SC is not symmetric. Enforcing symmetry.", UserWarning, stacklevel=2)
            SC = (SC + SC.T) / 2.0
            np.fill_diagonal(SC, 0.0)
    return {"X": X, "SC": SC, "subject_id": subject_id, "N": N, "T": T}

# ==========================================================================
# END OF VENDORED CODE
# ==========================================================================


def energy_direct(A, x0, x1, T, rcond=1e-10):
    """Finite-horizon minimum energy by direct summation, B = I:
    W_T = sum_{t<T} A^t (A^t)^T,  E = (x1 - A^T x0)^T pinv(W_T) (x1 - A^T x0).
    Same quantity as minimum_energy(), but defined for UNSTABLE A as well,
    which the fixed-scale symmetric part can be. Checked against
    minimum_energy() on the stable estimate for every subject before use."""
    N = A.shape[0]; W = np.zeros((N, N)); P = np.eye(N)
    for _ in range(T):
        W += P @ P.T
        P = A @ P
    delta = x1 - np.linalg.matrix_power(A, T) @ x0
    return float(delta @ np.linalg.pinv(W, rcond=rcond) @ delta)


def define_states(X):
    """Verbatim from run_adni_nctn_batch.py / run_hcp_aud_batch.py (names
    neutralised): quiescent = lowest-variance fifth, high-variance = highest."""
    N, T = X.shape
    n = max(1, T // 5)
    tvar = X.var(axis=0)
    idx = np.argsort(tvar)
    x_q = X[:, idx[:n]].mean(axis=1)
    x_h = X[:, idx[-n:]].mean(axis=1)
    x_q /= (np.linalg.norm(x_q) + 1e-8)
    x_h /= (np.linalg.norm(x_h) + 1e-8)
    return x_q, x_h


def one(job):
    sid, ts_path, sc_path, extensions = job      # flag travels WITH the job: a global set in
                                                 # main() is not inherited under spawn/forkserver
    try:
        X_raw = np.load(ts_path)
        SC = load_connectome(sc_path)
        n_sc = SC.shape[0]
        # the batches stored (T, N) and transposed; resolve against the
        # REGION COUNT so a file stored (N, T) is not silently transposed
        X_in = X_raw.T if X_raw.shape[1] == n_sc else (X_raw if X_raw.shape[0] == n_sc else None)
        if X_in is None:
            raise ValueError(f"timeseries {X_raw.shape} has no axis of {n_sc} regions")
        data = from_arrays(X=X_in, SC=SC, subject_id=sid, validate=True)
        X, SC, N, T = data["X"], data["SC"], data["N"], data["T"]
        EC = graphnet_effective_connectivity(X, SC, lambda_ridge=1.0, lambda_graph=1.0)
        A = normalise_matrix(EC, target_rho=TARGET_RHO)
        A_fc = normalise_matrix(functional_connectivity(X), target_rho=TARGET_RHO)
        B = np.eye(N)
        x_q, x_h = define_states(X)
        e = {}
        for op, M in (("ec", A), ("fc", A_fc)):
            e[f"E_{op}_q2h"] = float(minimum_energy(M, B, x_q, x_h, T=HORIZON)[0])
            e[f"E_{op}_h2q"] = float(minimum_energy(M, B, x_h, x_q, T=HORIZON)[0])
        extra = {}
        if extensions:
            # Added 2026-10-03 (round-8 review), rule fixed before any run:
            # (1) symmetry within one estimator: EC against its symmetrised
            #     form (EC + EC^T)/2, normalised identically;
            # (2) spectral radius 0.8, 0.9, 0.95 for FC/EC and EC_sym/EC.
            # Read as: symmetry alone matters if EC_sym/EC exceeds one in every
            # subject; normalisation does not drive the operator result if FC/EC
            # exceeds one in every subject at every radius, with the cohort
            # ordering unchanged.
            # (3) added 2026-10-03, round-9 review, rule fixed before any run:
            #     the symmetrised comparison above renormalises the symmetric
            #     matrix, so it changes the normalising factor as well as
            #     symmetry. S_fixed = (Ar + Ar^T)/2 removes direction from the
            #     already-normalised estimate WITHOUT renormalising. Then
            #     log(Sr/Ar) = log(S_fixed/Ar) + log(Sr/S_fixed): direction
            #     removed at fixed scale, plus the effect of rescaling. Read as:
            #     removing direction at fixed scale raises energy if
            #     S_fixed/EC > 1 in every subject.
            EC_sym = 0.5 * (EC + EC.T)
            FC = functional_connectivity(X)
            for r in RHOS:
                Ar, Fr_, Sr = (normalise_matrix(M, target_rho=r) for M in (EC, FC, EC_sym))
                S_fixed = 0.5 * (Ar + Ar.T)
                tag = f"r{int(round(r * 100)):03d}"
                extra[f"rho_symfixed_{tag}"] = float(np.max(np.abs(np.linalg.eigvalsh(S_fixed))))
                for d, (x0, x1) in (("h2q", (x_h, x_q)), ("q2h", (x_q, x_h))):
                    ee = float(minimum_energy(Ar, B, x0, x1, T=HORIZON)[0])
                    ef = float(minimum_energy(Fr_, B, x0, x1, T=HORIZON)[0])
                    es = float(minimum_energy(Sr, B, x0, x1, T=HORIZON)[0])
                    chk = energy_direct(Ar, x0, x1, HORIZON)
                    if abs(chk - ee) > 1e-6 * abs(ee):      # guard: same quantity as minimum_energy
                        raise RuntimeError(f"energy_direct disagrees with minimum_energy: {chk} vs {ee}")
                    ex = energy_direct(S_fixed, x0, x1, HORIZON)
                    extra[f"fc_over_ec_{d}_{tag}"] = ef / (ee + 1e-12)
                    extra[f"sym_over_ec_{d}_{tag}"] = es / (ee + 1e-12)
                    extra[f"symfixed_over_ec_{d}_{tag}"] = ex / (ee + 1e-12)
        return dict(subject_id=sid, N=N, T=T, **e,
                    ratio_q2h=e["E_fc_q2h"] / (e["E_ec_q2h"] + 1e-12),
                    ratio_h2q=e["E_fc_h2q"] / (e["E_ec_h2q"] + 1e-12), **extra, error="")
    except Exception as ex:
        return dict(subject_id=sid, error=f"{type(ex).__name__}: {str(ex)[:120]}")


MAP = {"quiescent_first": {"E_rest_to_demand": "E_ec_q2h", "E_demand_to_rest": "E_ec_h2q",
                           "E_fc": "E_fc_q2h", "teleport_ratio": "ratio_q2h"},
       "highvar_first":   {"E_craving_to_rest": "E_ec_h2q", "E_rest_to_craving": "E_ec_q2h",
                           "E_fc": "E_fc_h2q", "teleport_ratio": "ratio_h2q"}}


def gate(df, published, convention, id_key):
    pub = pd.read_csv(published)
    pub["k"] = pub.subject_id.astype(str).str.replace("^sub-", "", regex=True)
    d = df.copy(); d["k"] = d.subject_id.astype(str).str.replace("^sub-", "", regex=True)
    if id_key:
        key = pd.read_csv(id_key).astype(str)
        key["real_id"] = key.real_id.str.replace("^sub-", "", regex=True)   # same normalisation as d
        n_before = len(d)
        d = d.merge(key.rename(columns={"real_id": "k"}), on="k").drop(columns="k").rename(columns={"anon_id": "k"})
        print(f"  ID key matched {len(d)} of {n_before} subjects")
    # numeric IDs: compare without leading zeros (epilepsy files write 04221, the
    # published file 4221)
    for t in (pub, d):
        num = t["k"].str.fullmatch(r"\d+")
        t.loc[num, "k"] = t.loc[num, "k"].str.lstrip("0").replace("", "0")
    m = d.merge(pub, on="k", suffixes=("", "_pub"))
    print(f"\nGATE against {os.path.basename(published)} ({convention}): {len(m)} subjects matched of {len(d)}")
    if len(m) == 0:
        print("  ** FAIL — no subject matched the published file; check IDs and the --id-key **")
        return False
    if len(m) < len(d):
        print(f"  note: {len(d) - len(m)} subjects have no published row and are not gated")
    ok = True
    for pc, oc in MAP[convention].items():
        rel = (m[oc] - m[pc]).abs() / m[pc].abs()
        good = bool((rel <= 1e-6).all()); ok &= good
        print(f"   {pc:18s} vs {oc:10s} max relative difference {rel.max():.2e}   {'pass' if good else 'FAIL'}")
    print("  " + ("pass — the published batch values are reproduced exactly" if ok else
                  "** FAIL — inputs, settings or code differ from the published batch; nothing below is comparable **"))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ts-dir", required=True); ap.add_argument("--ts-suffix", required=True)
    ap.add_argument("--sc-dir", help="per-subject connectomes"); ap.add_argument("--sc-suffix")
    ap.add_argument("--sc-file", help="ONE connectome for every subject (epilepsy cohort's template)")
    ap.add_argument("--cohort", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-workers", type=int, default=0)
    ap.add_argument("--published"); ap.add_argument("--convention", choices=list(MAP))
    ap.add_argument("--id-key", help="restricted subject key (real_id, anon_id), HCP only")
    ap.add_argument("--extensions", action="store_true",
                    help="also compute EC vs symmetrised EC, and both ratios at spectral radius 0.8/0.9/0.95")
    a = ap.parse_args()
    files = sorted(f for f in os.listdir(a.ts_dir) if f.endswith(a.ts_suffix))
    jobs = []
    for f in files:
        sid = f[:-len(a.ts_suffix)]
        sc = a.sc_file or os.path.join(a.sc_dir, sid + a.sc_suffix)
        if os.path.exists(sc):
            jobs.append((sid, os.path.join(a.ts_dir, f), sc, a.extensions))
    rows = []
    if os.path.exists(a.out):
        prev = pd.read_csv(a.out)
        if a.extensions and "symfixed_over_ec_h2q_r090" not in prev.columns:
            # saved rows lack the requested analyses: recompute them rather than
            # report "0 to run" and silently return the old results
            print(f"  {a.out} lacks the current extension columns: recomputing every subject")
            prev = prev.iloc[0:0]
        rows = prev.to_dict("records"); done = {str(r["subject_id"]) for r in rows}
        jobs = [j for j in jobs if str(j[0]) not in done]; print(f"RESUMING: {len(done)} done, {len(jobs)} to run")
    print(f"{a.cohort}: {len(jobs)} subjects to run" + ("  [extensions ON]" if a.extensions else ""))
    def save():
        pd.DataFrame(rows).to_csv(a.out + ".tmp", index=False); os.replace(a.out + ".tmp", a.out)
    t0 = time.time()
    if jobs:
        with Pool(a.n_workers or max(1, min(cpu_count(), len(jobs)))) as pool:
            for i, r in enumerate(pool.imap_unordered(one, jobs), 1):
                rows.append(r)
                if i % 10 == 0: save()
                if i % 20 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)}  {(time.time() - t0) / 60:.1f} min", flush=True)
    save()
    df = pd.DataFrame(rows); bad = df[df.error.fillna("") != ""]
    if len(bad): print(f"  {len(bad)} subjects failed:", bad.error.unique()[:3])
    ok = df[df.error.fillna("") == ""]
    print(f"\n{a.cohort}: {len(ok)} subjects")
    for d_, lab in (("q2h", "quiescent -> high-variance"), ("h2q", "high-variance -> quiescent")):
        r = ok[f"ratio_{d_}"]
        print(f"   {lab:28s} median FC/EC ratio {r.median():7.2f}  (IQR {r.quantile(.25):.2f}-{r.quantile(.75):.2f}), above 1 in {(r > 1).mean():.0%}")
    if a.extensions and "fc_over_ec_h2q_r090" in ok:
        print("\n  EXTENSIONS (high-variance -> quiescent; median, and fraction of subjects above one)")
        for r in ("r080", "r090", "r095"):
            f_, y = ok[f"fc_over_ec_h2q_{r}"], ok[f"sym_over_ec_h2q_{r}"]
            print(f"   radius {int(r[1:]) / 100:.2f}:  FC/EC {f_.median():7.2f} (>1 in {(f_ > 1).mean():.0%})   "
                  f"EC_sym/EC {y.median():6.2f} (>1 in {(y > 1).mean():.0%})")
        if "symfixed_over_ec_h2q_r090" in ok:
            print("\n  DECOMPOSITION (h->q): symmetrised at FIXED scale vs renormalised")
            for r in ("r080", "r090", "r095"):
                x, y, rs = ok[f"symfixed_over_ec_h2q_{r}"], ok[f"sym_over_ec_h2q_{r}"], ok[f"rho_symfixed_{r}"]
                share = (np.log(x) / np.log(y)).median()
                print(f"   radius {int(r[1:]) / 100:.2f}:  fixed-scale {x.median():7.2f} (>1 in {(x > 1).mean():.0%})   "
                      f"renormalised {y.median():7.2f}   share at fixed scale {share:.2f}   "
                      f"spectral radius of fixed-scale symmetric part: median {rs.median():.2f}, >1 in {(rs > 1).mean():.0%}")
        dev = (ok.fc_over_ec_h2q_r090 - ok.ratio_h2q).abs().max()
        print(f"   check: radius-0.9 FC/EC equals the main ratio to {dev:.1e}")
    if a.published:
        sys.exit(0 if gate(ok, a.published, a.convention, a.id_key) else 1)


if __name__ == "__main__":
    main()
