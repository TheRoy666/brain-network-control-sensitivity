"""
Empirical check: does the iteration-0 bug pattern (confirmed on original
point-estimate timeseries for HCP/UNAM) ALSO hold on bootstrap-resampled
data -- i.e., was Path B's 200-resample bootstrap for these two cohorts
also effectively Ridge EC throughout, or does resampling change the
picture?

Replicates block_bootstrap_ec's EXACT resampling logic (contiguous
block resampling with replacement), not an approximation of it, so this
result is a faithful test of what Path B's real bootstrap actually did.

Tests multiple resamples per subject, multiple subjects per cohort --
one lucky/unlucky draw isn't enough to call this confirmed either way.
"""
import os
import numpy as np
from nctsens.loader import load_connectome, from_arrays
from nctsens.connectivity import graph_laplacian, ridge_effective_connectivity

# =====================================================================
HCP_TS_DIR = "<<< FILL IN: same value as before >>>"
HCP_SC_DIR = "<<< FILL IN: same value as before >>>"
UNAM_TS_DIR = "<<< FILL IN: same value as before >>>"
UNAM_SC_FILE = "<<< FILL IN: same value as before >>>"
# =====================================================================

HCP_SUBJECTS = os.environ.get("HCP_SUBJECTS", "").split()   # subject IDs, space-separated
UNAM_SUBJECTS = os.environ.get("UNAM_SUBJECTS", "").split()   # subject IDs, space-separated
N_RESAMPLES_PER_SUBJECT = 10


def graphnet_traced(X, SC, lambda_ridge=1.0, lambda_graph=1.0, max_iter=3000, tol=1e-6, lag=1):
    """Same traced logic as before, using the CORRECTED max_iter=3000
    default -- this checks the bug's behavior under the fix's own
    iteration budget, not the original 500."""
    N, T = X.shape
    X_t, X_tp1 = X[:, :-lag], X[:, lag:]
    L = graph_laplacian(SC, normalised=True)
    XXT, YXT = X_t @ X_t.T, X_tp1 @ X_t.T
    Reg = lambda_ridge * np.eye(N) + lambda_graph * L
    A = ridge_effective_connectivity(X, alpha=lambda_ridge, lag=lag)
    A_prev = A.copy()
    t_k = 1.0
    step = 1.0 / (2 * np.max(np.abs(np.linalg.eigvalsh(XXT)))
                  + 2 * np.max(np.abs(np.linalg.eigvalsh(Reg))) + 1e-8)
    for k in range(max_iter):
        Y = A + ((t_k - 1) / (t_k + 1)) * (A - A_prev)
        A_new = Y - step * (2 * (Y @ XXT - YXT) + 2 * (Reg @ Y.T).T)
        if np.linalg.norm(A_new - A, 'fro') / (np.linalg.norm(A, 'fro') + 1e-12) < tol:
            return k
        A_prev = A.copy()
        A = A_new
        t_k = (1 + np.sqrt(1 + 4 * t_k ** 2)) / 2
    return max_iter


def block_resample(X, block_length, seed):
    """EXACT replica of block_bootstrap_ec's resampling logic."""
    N, T = X.shape
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / block_length))
    max_start = T - block_length
    starts = rng.integers(0, max_start + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_length) for s in starts])[:T]
    return X[:, idx]


def check_cohort(name, subjects, ts_pattern_fn, sc_source, block_length, is_shared_template=False):
    print(f"\n=== {name} ===")
    all_k = []
    for sub in subjects:
        try:
            X_raw = np.load(ts_pattern_fn(sub))
            SC_raw = sc_source if is_shared_template else load_connectome(sc_source(sub))
            data = from_arrays(X=X_raw.T, SC=SC_raw, subject_id=sub, validate=True)
            X, SC = data["X"], data["SC"]

            k_orig = graphnet_traced(X, SC)
            print(f"  {sub} (original, unresampled): broke at iteration {k_orig}")

            ks_resampled = []
            for b in range(N_RESAMPLES_PER_SUBJECT):
                X_boot = block_resample(X, block_length, seed=b)
                k = graphnet_traced(X_boot, SC)
                ks_resampled.append(k)
            print(f"  {sub} ({N_RESAMPLES_PER_SUBJECT} resamples): "
                  f"iterations={ks_resampled}")
            all_k.extend(ks_resampled)
        except Exception as e:
            print(f"  {sub}: FAILED -- {e}")

    if all_k:
        print(f"\n  -> across all resamples: median={np.median(all_k):.0f}, "
              f"all near-zero (<=2): {all(k<=2 for k in all_k)}, "
              f"max={max(all_k)}")


if __name__ == "__main__":
    check_cohort(
        "HCP-AUD (Path B resampled)",
        HCP_SUBJECTS,
        lambda sub: f"{HCP_TS_DIR}/{sub}_native_timeseries.npy",
        lambda sub: f"{HCP_SC_DIR}/{sub}_SC_SIFT2_410.csv",
        block_length=69,  # sqrt(4800) rounded, matches real HCP block_length_used
    )
    check_cohort(
        "UNAM-TLE (Path B resampled)",
        UNAM_SUBJECTS,
        lambda sub: f"{UNAM_TS_DIR}/sub-{sub}_dk68_native_timeseries.npy",
        np.loadtxt(UNAM_SC_FILE, delimiter=",") if UNAM_SC_FILE and "FILL IN" not in UNAM_SC_FILE else None,
        block_length=14,  # sqrt(200) rounded, matches real UNAM block_length_used
        is_shared_template=True,
    )

    print("\n" + "="*60)
    print("If both cohorts show 'all near-zero (<=2): True' across ALL")
    print("resamples, the inference is confirmed: Path B's bootstrap for")
    print("HCP/UNAM was effectively Ridge EC throughout, same as the")
    print("point estimates. If any resample shows a large iteration")
    print("count, that specific draw's bootstrap EC was genuinely closer")
    print("to true GraphNet convergence -- meaning the bootstrap ensemble")
    print("may be a MIX, not uniformly ridge-only like the point estimate.")
    print("="*60)
