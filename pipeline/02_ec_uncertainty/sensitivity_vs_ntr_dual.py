#!/usr/bin/env python3
"""
Does estimator sensitivity SCALE with the parameter-to-observation ratio?

WHY THIS IS THE LOAD-BEARING EXPERIMENT
---------------------------------------
Three independent manipulations already point the same way: estimator
penalties, bootstrap block length and recording length are each inert
where observations outnumber parameters and consequential where they do
not. But each was measured at one or two values of that ratio, and two
points make a contrast, not a law.

To claim that the ratio GOVERNS the behaviour rather than merely
distinguishing two cohorts, sensitivity has to scale with it across a
continuum, measured WITHIN one cohort so that population, atlas,
scanner and pipeline are all held fixed.

This script produces that continuum. It truncates a cohort's recordings
to a series of lengths and, at each, recomputes rank stability across
the estimator's penalty grid. Sensitivity at a given length is the
spread across that grid.

    sensitivity rises monotonically with N/T  ->  the ratio governs.
        A dose-response relationship, not a two-group difference.

    it does not                               ->  the principle claim
        fails and the paper reverts to reporting a contrast. This is a
        real possible outcome and is written into the stop criteria.

WHAT IS MEASURED
----------------
At each truncation length:
  - median rank stability at the published setting
  - the spread of median rank stability across the penalty grid, which
    is the sensitivity measure
  - the within-subject spread across the grid, which is the same
    quantity computed per subject and then summarised, and is the more
    conservative of the two

Both are reported because they can disagree, and a claim resting on the
more favourable one alone would not survive review.

DESIGN NOTES
------------
**Truncation, not subsampling.** Taking the first n timepoints keeps the
sampling rate fixed and varies only the sample count, which is what N/T
is about. The stride alternative confounds rate with count and is
available in operator_divergence_vs_length.py for comparison.

**Array orientation is resolved against the region count**, never by
comparing dimensions. Below N timepoints a size heuristic silently
transposes, and this script deliberately goes below N.

**Block length follows the square-root rule per length**, as the rest of
the paper does, so it is not held fixed at a value appropriate to the
untruncated data.

COST
----
lengths x penalties x subjects x resamples. Parallelism is over
(subject, length, penalty) triples, since none of those alone exceeds
the core count.

USAGE
-----
    python3 sensitivity_vs_ntr.py \\
        --ts-dir <timeseries> --ts-suffix _native_timeseries.npy \\
        --sc-dir <connectomes> --sc-suffix _SC_SIFT2_410.csv \\
        --cohort HCP --lengths 4800,2400,1200,600,400,300,200 \\
        --n-subjects 20 --n-boot 25 --n-workers 78 \\
        --out sensitivity_vs_ntr_HCP.csv
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import itertools
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

# ==========================================================================
# VENDORED NEUROSIM CODE — copied verbatim, so this script does not depend on
# whichever connectivity.py / physics.py / loader.py the server happens to hold.
# A drift check before vendoring found the project copy of connectivity.py to be
# the UNCORRECTED estimator (max_iter 500; convergence check before the update,
# returning the ridge-only start) with no block_bootstrap_ec at all. The code
# below is the corrected NeuroSim package version. Source checksums:
#   neurosim/connectivity.py  sha256 2ae501d6f7f3a44ccbe6864566cb3ec7a88626cb95e4c9d923978ad9d5becee6
#     functions: ridge_effective_connectivity, graph_laplacian, graphnet_effective_connectivity, block_bootstrap_ec
#   neurosim/physics.py       sha256 31dc44b64a5ce53201cb191efcc2192f2b00bc0aa4b33760ae1242aeeb416493
#     functions: normalise_matrix, average_controllability
#   neurosim/loader.py        sha256 1387b643754d9e3cc0986d28d34c7bd68d6ed2b7fbef52b26b1f5e73d4f528f8
#     functions: load_connectome, from_arrays
# The package-definition column must still reproduce the published run exactly;
# that gate is the final word on whether this is the code behind it.
# ==========================================================================

# ---- from neurosim/connectivity.py ----
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge

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


def block_bootstrap_ec(
    X: NDArray,
    SC: NDArray,
    ec_func: "callable" = None,
    n_boot: int = 200,
    block_length: int = 15,
    seed: int = 0,
    **ec_kwargs,
) -> NDArray:
    """Moving-block bootstrap over BOLD time series, re-estimating EC each draw.

    Per Phase 0 plan (Path B): naive timepoint bootstrap is invalid for BOLD
    because of temporal autocorrelation. This resamples contiguous blocks of
    ``block_length`` TRs (with replacement) to preserve short-range temporal
    structure, concatenates them back to length T, and re-estimates EC on
    each resampled series.

    Parameters
    ----------
    X            : (N, T) ndarray - original BOLD time series.
    SC           : (N, N) ndarray - structural connectome (passed through
                   to ``ec_func``, e.g. for the GraphNet Laplacian prior).
    ec_func      : callable(X, SC, **ec_kwargs) -> (N, N) ndarray. Defaults
                   to ``graphnet_effective_connectivity`` if not given.
    n_boot       : int   - number of bootstrap resamples (B). Phase 0
                   default: start at 200, check convergence of the
                   rank-stability metric before scaling up.
    block_length : int   - block length in TRs (Phase 0 default: 15,
                   ≈ sqrt(T) rule of thumb for T≈197-200).
    seed         : int   - RNG seed for reproducibility.
    **ec_kwargs  : passed through to ec_func.

    Returns
    -------
    EC_boot : (n_boot, N, N) ndarray - one EC estimate per resample.

    Notes
    -----
    Compute cost is n_boot × (cost of ec_func). Benchmark on 2-3 subjects
    before launching the full batch (Phase 0 compute-budget check).

    Uses graphnet_effective_connectivity's corrected version by default
    (see that function's docstring) -- this function required no changes
    itself, but every call it makes now benefits from the fix.
    """
    if ec_func is None:
        ec_func = graphnet_effective_connectivity

    X = np.asarray(X, dtype=float)
    N, T = X.shape
    rng = np.random.default_rng(seed)

    n_blocks = int(np.ceil(T / block_length))
    max_start = T - block_length
    if max_start < 0:
        raise ValueError(
            f"block_length ({block_length}) exceeds series length ({T})."
        )

    EC_boot = np.zeros((n_boot, N, N))
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_length) for s in starts])[:T]
        X_resampled = X[:, idx]
        EC_boot[b] = ec_func(X_resampled, SC, **ec_kwargs)

    return EC_boot

# ---- from neurosim/physics.py ----
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


def average_controllability(A: NDArray) -> NDArray:
    """Trace of the infinite-horizon Gramian per node (Gu et al., 2015).

    Returns
    -------
    ac : (N,) ndarray - Average controllability for each node.
    """
    A = np.asarray(A, dtype=float)
    N = A.shape[0]
    W_inf = solve_discrete_lyapunov(A, np.eye(N))
    return np.diag(W_inf)

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

# The same grid used for the single-length sensitivity analysis, so the
# two are directly comparable. lambda_graph = 0 removes the structural
# prior entirely.
LAMBDA_RIDGE = [0.01, 1.0, 100.0]
LAMBDA_GRAPH = [0.0, 1.0, 100.0]
TARGET_RHO = 0.9

# --- added 2026-09-29: score the SAME resamples under both definitions of
# average controllability. Package (row-based) = regions most easily driven;
# Gu et al. (2015) (column-based) = regions that drive. The package column is
# unchanged, so it must reproduce the published sensitivity_vs_ntr_HCP.csv.
from scipy.linalg import solve_discrete_lyapunov


def gu_average_controllability(A_norm):
    """Gu et al. (2015) on an ALREADY-normalised A: diag of the Gramian
    solved for the transposed operator, sum_t ||A^t e_j||^2."""
    return np.diag(solve_discrete_lyapunov(A_norm.T, np.eye(A_norm.shape[0])))


def mean_pairwise_jaccard(ranks, k=5):
    tops = [set(np.where(r < k)[0]) for r in ranks]
    B = len(tops)
    v = [len(tops[i] & tops[j]) / len(tops[i] | tops[j]) for i in range(B) for j in range(i + 1, B)]
    return float(np.mean(v)) if v else np.nan


def mean_pairwise_tau(ranks, max_pairs=150, seed=0):
    B = ranks.shape[0]
    pairs = [(i, j) for i in range(B) for j in range(i + 1, B)]
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(seed)
        pairs = [pairs[k] for k in rng.choice(len(pairs), max_pairs,
                                              replace=False)]
    v = [kendalltau(ranks[i], ranks[j]).correlation for i, j in pairs]
    v = [x for x in v if np.isfinite(x)]
    return float(np.mean(v)) if v else np.nan


def process_one(args):
    sid, ts_path, sc_path, cohort, L, lr, lg, n_boot = args
    try:
        X_raw = np.load(ts_path)
        SC_raw = load_connectome(sc_path)
        N_nodes = SC_raw.shape[0]
        # Orientation from the region count. A size heuristic fails below
        # N timepoints, which is exactly where this script operates.
        if X_raw.shape[0] == N_nodes:
            X_o = X_raw
        elif X_raw.shape[1] == N_nodes:
            X_o = X_raw.T
        else:
            raise ValueError(f"X {X_raw.shape} matches neither dim of SC")
        Xt = X_o[:, :L]
        data = from_arrays(X=Xt, SC=SC_raw, subject_id=sid, validate=True)
        X, SC, N = data["X"], data["SC"], data["N"]
        T = X.shape[1]
        bl = max(2, int(round(np.sqrt(T))))
        if T // bl < 4:
            return dict(subject_id=sid, cohort=cohort, requested_length=L,
                        T=T, N=N, lambda_ridge=lr, lambda_graph=lg,
                        mean_tau=np.nan, error="too few blocks")
        ens = block_bootstrap_ec(
            X, SC, ec_func=graphnet_effective_connectivity,
            n_boot=n_boot, block_length=bl, seed=0,
            lambda_ridge=lr, lambda_graph=lg)
        ranks, ranks_gu = [], []
        for k in range(ens.shape[0]):
            A = normalise_matrix(ens[k], target_rho=TARGET_RHO)
            ranks.append(np.argsort(np.argsort(-average_controllability(A))))
            ranks_gu.append(np.argsort(np.argsort(-gu_average_controllability(A))))
        ranks, ranks_gu = np.asarray(ranks), np.asarray(ranks_gu)
        return dict(subject_id=sid, cohort=cohort, requested_length=L,
                    T=T, N=N, N_over_T=N / T, block_length=bl,
                    lambda_ridge=lr, lambda_graph=lg, n_boot=n_boot,
                    mean_tau=mean_pairwise_tau(ranks),
                    mean_tau_gu=mean_pairwise_tau(ranks_gu),
                    jaccard_top5=mean_pairwise_jaccard(ranks),
                    jaccard_top5_gu=mean_pairwise_jaccard(ranks_gu))
    except Exception as e:
        return dict(subject_id=sid, cohort=cohort, requested_length=L,
                    lambda_ridge=lr, lambda_graph=lg, mean_tau=np.nan,
                    error=str(e)[:90])


def summarise(df, cohort):
    """Per-length summary. Penalty sensitivity (spread across the nine
    settings) is computed ONLY at lengths where all nine were run: with one
    setting the spread is zero by construction, and an earlier version of
    this summary then reported a series of zeros as "monotone" — a false
    verdict. compare_ntr_rerun.py is the authoritative report."""
    ok = df.dropna(subset=["mean_tau"])
    print("\n" + "=" * 78)
    print(f"TRUNCATION SERIES — {cohort}   (authoritative report: compare_ntr_rerun.py)")
    print("=" * 78)
    if not len(ok):
        print("  nothing usable")
        return
    rows = []
    print(f"\n  {'T':>6} {'N/T':>7} {'n':>4} {'tau pkg (1,1)':>14} {'tau driver':>11} "
          f"{'settings':>9} {'within-subj spread':>19}")
    for T, g in ok.groupby("T"):
        pub = g[(g.lambda_ridge == 1.0) & (g.lambda_graph == 1.0)]
        n_set = g.groupby(["lambda_ridge", "lambda_graph"]).ngroups
        within = np.nan
        if n_set == 9:
            piv = g.pivot_table(index="subject_id", columns=["lambda_ridge", "lambda_graph"], values="mean_tau")
            within = (piv.max(axis=1) - piv.min(axis=1)).median()
        ntr = g.N_over_T.iloc[0]
        rows.append(dict(cohort=cohort, T=T, N_over_T=ntr, n_subjects=g.subject_id.nunique(),
                         tau_published=pub.mean_tau.median() if len(pub) else np.nan,
                         tau_driver=pub.mean_tau_gu.median() if (len(pub) and "mean_tau_gu" in pub) else np.nan,
                         n_settings=n_set, within_subject_spread=within))
        print(f"  {T:>6} {ntr:>7.3f} {g.subject_id.nunique():>4} {rows[-1]['tau_published']:>14.4f} "
              f"{rows[-1]['tau_driver']:>11.4f} {n_set:>9} "
              f"{('%.4f' % within) if n_set == 9 else 'not run':>19}")
    s = pd.DataFrame(rows).sort_values("N_over_T")
    full = s.dropna(subset=["within_subject_spread"])
    print("\n  PENALTY-SENSITIVITY DOSE-RESPONSE")
    if len(full) >= 4:
        rho, p = spearmanr(full.N_over_T, full.within_subject_spread)
        print(f"    within-subject spread vs N/T over {len(full)} full-grid lengths: "
              f"rho = {rho:+.3f}, p = {p:.4f}, monotone: {full.within_subject_spread.is_monotonic_increasing}")
    else:
        print(f"    not tested: the full grid was run at {len(full)} length(s); "
              "a dose-response needs at least four. No verdict.")
    return s


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ts-dir", required=True)
    ap.add_argument("--ts-suffix", required=True)
    ap.add_argument("--sc-dir", required=True)
    ap.add_argument("--sc-suffix", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--lengths", required=True,
                    help="Comma-separated truncation lengths, longest first.")
    ap.add_argument("--n-subjects", type=int, default=0,
                    help="0 = all subjects, as the published run (238)")
    ap.add_argument("--n-boot", type=int, default=25,
                    help="25 = the published run; keep it")
    ap.add_argument("--checkpoint-every", type=int, default=20,
                    help="save completed cells every N; re-run the same command to resume")
    ap.add_argument("--published-setting-only", action="store_true",
                    help="only ridge 1, prior 1: about one-ninth of the cost; "
                         "answers stability but not penalty sensitivity")
    ap.add_argument("--n-workers", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]
    files = sorted(f for f in os.listdir(args.ts_dir)
                   if f.endswith(args.ts_suffix))
    if args.n_subjects:
        files = files[:args.n_subjects]
    subs = []
    for f in files:
        sid = f[:-len(args.ts_suffix)]
        sc = os.path.join(args.sc_dir, sid + args.sc_suffix)
        if os.path.exists(sc):
            subs.append((sid, os.path.join(args.ts_dir, f), sc))
    grid = list(itertools.product(LAMBDA_RIDGE, LAMBDA_GRAPH))
    if args.published_setting_only:
        grid = [(1.0, 1.0)]
    jobs = [(sid, ts, sc, args.cohort, L, lr, lg, args.n_boot)
            for (sid, ts, sc) in subs for L in lengths for (lr, lg) in grid]
    print(f"{len(subs)} subjects x {len(lengths)} lengths x {len(grid)} "
          f"settings = {len(jobs)} cells, {args.n_boot} resamples each")
    if not jobs:
        raise SystemExit("No subject had both a timeseries and a connectome.")

    # RESUME: cells already in --out are skipped. Every cell seeds its own
    # resampling (seed 0), independent of order, so resuming cannot change
    # any value. Keys are normalised so numeric subject IDs read back from
    # CSV still match.
    def key(sid, L, lr, lg):
        return (str(sid), int(L), float(lr), float(lg))
    rows = []
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out)
        rows = prev.to_dict("records")
        done = {key(r["subject_id"], r["requested_length"], r["lambda_ridge"], r["lambda_graph"]) for r in rows}
        before = len(jobs)
        jobs = [j for j in jobs if key(j[0], j[4], j[5], j[6]) not in done]
        print(f"  RESUMING: {before - len(jobs)} cells already in {args.out}, {len(jobs)} to run")

    def checkpoint():
        tmp = args.out + ".tmp"
        pd.DataFrame(rows).to_csv(tmp, index=False)
        os.replace(tmp, args.out)            # atomic: never a half-written file

    n_workers = args.n_workers or max(1, min(cpu_count(), len(jobs)))
    print(f"  {n_workers} workers; saving every {args.checkpoint_every} cells")
    t0 = time.time()
    if jobs:
        with Pool(n_workers) as pool:
            for i, r in enumerate(pool.imap_unordered(process_one, jobs), 1):
                rows.append(r)
                if i % args.checkpoint_every == 0:
                    checkpoint()
                if i % 50 == 0 or i == len(jobs):
                    el = time.time() - t0
                    print(f"  {i}/{len(jobs)}  elapsed {el/60:.1f} min"
                          f"  eta {el/i*(len(jobs)-i)/60:.1f} min", flush=True)
    checkpoint()
    df = pd.DataFrame(rows)
    s = summarise(df, args.cohort)
    if s is not None:
        s.to_csv(args.out.replace(".csv", "_summary.csv"), index=False)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
