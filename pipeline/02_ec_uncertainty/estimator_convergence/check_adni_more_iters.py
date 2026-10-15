"""
Simple follow-up: does ADNI's estimate actually converge if given more
iterations, or does it keep moving indefinitely? Same paths as before,
just copy them in again.
"""
import os
import numpy as np
from nctsens.loader import load_connectome, from_arrays
from nctsens.connectivity import graph_laplacian, ridge_effective_connectivity

# =====================================================================
ADNI_TS_DIR = "<<< SAME VALUE AS BEFORE >>>"
ADNI_SC_DIR = "<<< SAME VALUE AS BEFORE >>>"
# =====================================================================

SUB = os.environ["ADNI_SUBJECT"]  # one subject, quick check


def graphnet_run(X, SC, lambda_ridge=1.0, lambda_graph=1.0, max_iter=3000, lag=1):
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
    checkpoints = {}
    for k in range(max_iter):
        Y = A + ((t_k - 1) / (t_k + 1)) * (A - A_prev)
        A_new = Y - step * (2 * (Y @ XXT - YXT) + 2 * (Reg @ Y.T).T)
        A_prev = A.copy()
        A = A_new
        if k+1 in [500, 1000, 1500, 2000, 2500, 3000]:
            checkpoints[k+1] = A.copy()
        t_k = (1 + np.sqrt(1 + 4 * t_k ** 2)) / 2
    return checkpoints


X_raw = np.load(f"{ADNI_TS_DIR}/sub-{SUB}_schaefer400_native_timeseries.npy")
SC_raw = load_connectome(f"{ADNI_SC_DIR}/sub-{SUB}_schaefer400_connectome.csv")
data = from_arrays(X=X_raw.T, SC=SC_raw, subject_id=SUB, validate=True)

print(f"Running sub-{SUB} out to 3000 iterations, checking every 500...")
checkpoints = graphnet_run(data["X"], data["SC"])

prev_k, prev_A = None, None
for k in sorted(checkpoints.keys()):
    A = checkpoints[k]
    if prev_A is not None:
        rel_change = np.linalg.norm(A - prev_A, 'fro') / (np.linalg.norm(prev_A, 'fro') + 1e-12)
        print(f"  iter {prev_k} -> {k}: relative change = {rel_change:.6f}")
    prev_k, prev_A = k, A

print()
print("If the relative change keeps shrinking toward zero, it's genuinely")
print("converging, just slowly -- more iterations is the fix.")
print("If it stays roughly flat (not shrinking), something else is going")
print("on and more iterations alone won't solve it.")
