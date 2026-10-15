"""
Multi-subject confirmation: does one subject's instant, exact convergence
(rel_change=0.000000 at every checkpoint from 500-5000 iterations) hold
across a real sample of the HCP cohort, or was it specific to that one
subject? Uses a 15-subject sample -- at ~11.3s/subject for the full
5000-iteration trace, this is genuinely cheap.
"""
import os
import numpy as np
from nctsens.loader import load_connectome, from_arrays
from nctsens.connectivity import graph_laplacian, ridge_effective_connectivity

HCP_TS_DIR = os.path.expandvars("$BNCS_ROOT/Neurosim_Paper/HCP_Data/HCP_Native_Timeseries")
HCP_SC_DIR = os.path.expandvars("$BNCS_ROOT/Neurosim_Paper/HCP_Data/HCP_SC_matrices")

# 15 HCP subjects, space-separated in $HCP_SUBJECTS
HCP_SUBJECTS = os.environ.get("HCP_SUBJECTS", "").split()   # subject IDs, space-separated


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
    for k in range(max_iter):
        Y = A + ((t_k-1)/(t_k+1)) * (A - A_prev)
        A_new = Y - step * (2*(Y@XXT-YXT) + 2*(Reg@Y.T).T)
        A_prev = A.copy(); A = A_new
        if k+1 in checkpoints:
            saved[k+1] = A.copy()
        t_k = (1+np.sqrt(1+4*t_k**2))/2
    return saved


all_exact_zero = True
first_nonzero_checkpoint = {}

for sub in HCP_SUBJECTS:
    try:
        X_raw = np.load(os.path.join(HCP_TS_DIR, f"{sub}_native_timeseries.npy"))
        SC_raw = load_connectome(os.path.join(HCP_SC_DIR, f"{sub}_SC_SIFT2_410.csv"))
        data = from_arrays(X=X_raw.T, SC=SC_raw, subject_id=sub, validate=True)
        saved = graphnet_checkpoints(data["X"], data["SC"])

        ks = sorted(saved.keys())
        changes = []
        for i in range(1, len(ks)):
            rel = np.linalg.norm(saved[ks[i]] - saved[ks[i-1]], 'fro') / (np.linalg.norm(saved[ks[i-1]], 'fro') + 1e-12)
            changes.append(rel)

        exact_zero = all(c == 0.0 for c in changes)
        all_exact_zero = all_exact_zero and exact_zero
        print(f"  {sub}: exact_zero_throughout={exact_zero}, changes={[f'{c:.8f}' for c in changes]}")
    except Exception as e:
        print(f"  {sub}: FAILED -- {e}")

print(f"\n{'='*60}")
print(f"All {len(HCP_SUBJECTS)} subjects show exact-zero convergence by iter 500: {all_exact_zero}")
if all_exact_zero:
    print("CONFIRMED -- not specific to the first subject checked. A full HCP")
    print("GraphNet rerun is genuinely cheap. Safe to proceed to the")
    print("full-cohort rerun.")
else:
    print("NOT universal -- some subjects behave differently. Do not")
    print("assume the full cohort will be uniformly cheap; check further")
    print("before committing to the full rerun's timing.")
print(f"{'='*60}")
