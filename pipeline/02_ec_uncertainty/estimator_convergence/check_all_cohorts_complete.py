"""
COMPLETE, ready-to-run scope check across all four cohort/atlas
combinations. Fill in ONLY the path variables marked "<<< FILL IN >>>"
near the top -- everything else is already written.

Checks, per subject: how many FISTA iterations actually ran before the
(current, unpatched) function's convergence check broke -- k_broke=0
means essentially pure ridge output was returned; a large k_broke means
something close to genuine GraphNet convergence was actually reached
before the bug discarded the final update.

Does NOT modify connectivity.py. Safe to run against the current,
unpatched installation.
"""
import os
import numpy as np
from nctsens.loader import load_connectome, from_arrays
from nctsens.connectivity import graph_laplacian, ridge_effective_connectivity

# =====================================================================
# FILL IN THESE FOUR DIRECTORY PATHS -- everything else is ready to go
# =====================================================================
HCP_TS_DIR = "<<< FILL IN: e.g. $BNCS_ROOT/Neurosim_Paper/HCP_Data/timeseries >>>"
HCP_SC_DIR = "<<< FILL IN: e.g. $BNCS_ROOT/Neurosim_Paper/HCP_Data/HCP_SC_matrices >>>"

ADNI_TS_DIR = "<<< FILL IN: e.g. $BNCS_ROOT/Neurosim_Paper/ADNI_Data/timeseries >>>"
ADNI_SC_DIR = "<<< FILL IN: e.g. $BNCS_ROOT/Neurosim_Paper/ADNI_Data/connectomes >>>"

UNAM_TS_DIR = "<<< FILL IN: e.g. $BNCS_ROOT/Epilepsy/derivatives/timeseries >>>"
UNAM_SC_FILE = "<<< FILL IN: e.g. $BNCS_ROOT/Epilepsy/derivatives/timeseries/template_SC_dk68.csv >>>"
# =====================================================================

# Already-confirmed real subject IDs from earlier this session -- no
# need to look these up again.
HCP_SUBJECTS = os.environ.get("HCP_SUBJECTS", "").split()   # subject IDs, space-separated
ADNI_SUBJECTS = os.environ.get("ADNI_SUBJECTS", "").split()   # subject IDs, space-separated
UNAM_SUBJECTS = os.environ.get("UNAM_SUBJECTS", "").split()   # subject IDs, space-separated


def graphnet_traced(X, SC, lambda_ridge=1.0, lambda_graph=1.0, max_iter=500, tol=1e-6, lag=1):
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


def check_hcp():
    print("\n=== HCP-AUD ===")
    results = []
    for sub in HCP_SUBJECTS:
        try:
            X_raw = np.load(f"{HCP_TS_DIR}/{sub}_native_timeseries.npy")
            SC_raw = load_connectome(f"{HCP_SC_DIR}/{sub}_SC_SIFT2_410.csv")
            data = from_arrays(X=X_raw.T, SC=SC_raw, subject_id=sub, validate=True)
            k = graphnet_traced(data["X"], data["SC"])
            results.append(k)
            print(f"  {sub}: broke at iteration {k}")
        except Exception as e:
            print(f"  {sub}: FAILED -- {e}")
    if results:
        print(f"  -> median iteration: {np.median(results):.0f}, "
              f"all near-zero (<=2): {all(k<=2 for k in results)}")


def check_adni(atlas_label, ts_suffix, sc_suffix):
    print(f"\n=== ADNI-{atlas_label} ===")
    results = []
    for sub in ADNI_SUBJECTS:
        try:
            X_raw = np.load(f"{ADNI_TS_DIR}/sub-{sub}{ts_suffix}")
            SC_raw = load_connectome(f"{ADNI_SC_DIR}/sub-{sub}{sc_suffix}")
            data = from_arrays(X=X_raw.T, SC=SC_raw, subject_id=sub, validate=True)
            k = graphnet_traced(data["X"], data["SC"])
            results.append(k)
            print(f"  sub-{sub}: broke at iteration {k}")
        except Exception as e:
            print(f"  sub-{sub}: FAILED -- {e}")
    if results:
        print(f"  -> median iteration: {np.median(results):.0f}, "
              f"all near-zero (<=2): {all(k<=2 for k in results)}")


def check_unam():
    print("\n=== UNAM-TLE ===")
    A_template = np.loadtxt(UNAM_SC_FILE, delimiter=",")
    results = []
    for sub in UNAM_SUBJECTS:
        try:
            X_raw = np.load(f"{UNAM_TS_DIR}/sub-{sub}_dk68_native_timeseries.npy")
            data = from_arrays(X=X_raw.T, SC=A_template, subject_id=sub, validate=True)
            k = graphnet_traced(data["X"], data["SC"])
            results.append(k)
            print(f"  sub-{sub}: broke at iteration {k}")
        except Exception as e:
            print(f"  sub-{sub}: FAILED -- {e}")
    if results:
        print(f"  -> median iteration: {np.median(results):.0f}, "
              f"all near-zero (<=2): {all(k<=2 for k in results)}")


if __name__ == "__main__":
    check_hcp()
    check_adni("Schaefer400", "_schaefer400_native_timeseries.npy", "_schaefer400_connectome.csv")
    check_adni("TianS3", "_tians3_native_timeseries.npy", "_tians3_connectome.csv")
    check_unam()
    print("\n" + "="*60)
    print("If every cohort shows 'all near-zero (<=2): True', the bug")
    print("triggered essentially immediately everywhere -- reporting")
    print("Ridge EC honestly as the method is clean and accurate.")
    print("If any cohort shows False, that cohort is heterogeneous and")
    print("needs a closer look before deciding how to describe it.")
    print("="*60)
