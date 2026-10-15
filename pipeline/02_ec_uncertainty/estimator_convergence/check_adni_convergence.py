"""
ADNI-specific follow-up: the original bug doesn't apply here (loop runs
to max_iter, never breaks early) -- this checks instead whether the
estimate has practically stabilized by iteration 500, or is still
changing meaningfully. Uses the SAME paths as the previous script --
just copy in the same ADNI_TS_DIR / ADNI_SC_DIR values you already used.
"""
import os
import numpy as np
from nctsens.loader import load_connectome, from_arrays
from nctsens.connectivity import graph_laplacian, ridge_effective_connectivity

# =====================================================================
ADNI_TS_DIR = "<<< SAME VALUE YOU USED LAST TIME >>>"
ADNI_SC_DIR = "<<< SAME VALUE YOU USED LAST TIME >>>"
# =====================================================================

ADNI_SUBJECTS = os.environ.get("ADNI_SUBJECTS", "").split()   # subject IDs, space-separated


def graphnet_check_stability(X, SC, lambda_ridge=1.0, lambda_graph=1.0, max_iter=500, lag=1):
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
    A_at_250 = None
    for k in range(max_iter):
        Y = A + ((t_k - 1) / (t_k + 1)) * (A - A_prev)
        A_new = Y - step * (2 * (Y @ XXT - YXT) + 2 * (Reg @ Y.T).T)
        A_prev = A.copy()
        A = A_new
        if k == 249:
            A_at_250 = A.copy()
        t_k = (1 + np.sqrt(1 + 4 * t_k ** 2)) / 2
    # Relative change from iteration 250 to iteration 500 -- large means
    # still moving a lot even late; small means practically settled.
    rel_change = np.linalg.norm(A - A_at_250, 'fro') / (np.linalg.norm(A_at_250, 'fro') + 1e-12)
    return rel_change


print("=== ADNI convergence stability check (250 -> 500 iterations) ===")
for sub in ADNI_SUBJECTS:
    try:
        X_raw = np.load(f"{ADNI_TS_DIR}/sub-{sub}_schaefer400_native_timeseries.npy")
        SC_raw = load_connectome(f"{ADNI_SC_DIR}/sub-{sub}_schaefer400_connectome.csv")
        data = from_arrays(X=X_raw.T, SC=SC_raw, subject_id=sub, validate=True)
        rel_change = graphnet_check_stability(data["X"], data["SC"])
        print(f"  sub-{sub}: relative change from iter 250->500 = {rel_change:.6f}")
    except Exception as e:
        print(f"  sub-{sub}: FAILED -- {e}")

print()
print("If relative change is small (e.g. < 0.01), the estimate has")
print("practically settled well before 500 iterations -- the result is")
print("likely usable as-is, just not hitting the strict 1e-6 threshold.")
print("If it's still large, the estimate may genuinely still be moving,")
print("and would need more iterations or a loosened tolerance to trust.")
