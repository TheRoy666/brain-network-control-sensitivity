"""
Real-cost check for HCP and UNAM's GraphNet convergence -- informs
whether/how to plan a full rerun for the paper. One subject each,
checkpointed out to a generous 5000 iterations.
"""
import os
import numpy as np
import time
from nctsens.loader import load_connectome, from_arrays
from nctsens.connectivity import graph_laplacian, ridge_effective_connectivity

# =====================================================================
HCP_TS_DIR = "<<< SAME AS BEFORE >>>"
HCP_SC_DIR = "<<< SAME AS BEFORE >>>"
UNAM_TS_DIR = "<<< SAME AS BEFORE >>>"
UNAM_SC_FILE = "<<< SAME AS BEFORE >>>"
# =====================================================================


def graphnet_checkpoints(X, SC, lambda_ridge=1.0, lambda_graph=1.0, max_iter=5000, lag=1,
                          checkpoints=(500, 1000, 2000, 3000, 4000, 5000)):
    N, T = X.shape
    X_t, X_tp1 = X[:, :-lag], X[:, lag:]
    L = graph_laplacian(SC, normalised=True)
    XXT, YXT = X_t @ X_t.T, X_tp1 @ X_t.T
    Reg = lambda_ridge * np.eye(N) + lambda_graph * L
    A = ridge_effective_connectivity(X, alpha=lambda_ridge, lag=lag)
    A_prev = A.copy()
    t_k = 1.0
    step = 1.0 / (2*np.max(np.abs(np.linalg.eigvalsh(XXT))) + 2*np.max(np.abs(np.linalg.eigvalsh(Reg))) + 1e-8)
    saved = {}
    t0 = time.time()
    for k in range(max_iter):
        Y = A + ((t_k-1)/(t_k+1)) * (A - A_prev)
        A_new = Y - step * (2*(Y@XXT-YXT) + 2*(Reg@Y.T).T)
        A_prev = A.copy(); A = A_new
        if k+1 in checkpoints:
            saved[k+1] = (A.copy(), time.time()-t0)
        t_k = (1+np.sqrt(1+4*t_k**2))/2
    return saved


def report(name, X, SC):
    print(f"\n=== {name} ===")
    saved = graphnet_checkpoints(X, SC)
    prev_k, prev_A = None, None
    for k in sorted(saved.keys()):
        A, elapsed = saved[k]
        if prev_A is not None:
            rel = np.linalg.norm(A-prev_A,'fro')/(np.linalg.norm(prev_A,'fro')+1e-12)
            print(f"  iter {prev_k}->{k}: rel_change={rel:.6f}, elapsed at {k}={elapsed:.1f}s")
        prev_k, prev_A = k, A


HCP_SUBJECT, UNAM_SUBJECT = os.environ["HCP_SUBJECT"], os.environ["UNAM_SUBJECT"]
X_raw = np.load(f"{HCP_TS_DIR}/{HCP_SUBJECT}_native_timeseries.npy")
SC_raw = load_connectome(f"{HCP_SC_DIR}/{HCP_SUBJECT}_SC_SIFT2_410.csv")
data = from_arrays(X=X_raw.T, SC=SC_raw, subject_id=HCP_SUBJECT, validate=True)
report("HCP subject", data["X"], data["SC"])

A_template = np.loadtxt(UNAM_SC_FILE, delimiter=",")
X_raw = np.load(f"{UNAM_TS_DIR}/sub-{UNAM_SUBJECT}_dk68_native_timeseries.npy")
data = from_arrays(X=X_raw.T, SC=A_template, subject_id=UNAM_SUBJECT, validate=True)
report("UNAM subject", data["X"], data["SC"])

print("\nUse the elapsed-time-at-checkpoint numbers to extrapolate real")
print("full-cohort cost once a stable relative-change threshold is found.")
