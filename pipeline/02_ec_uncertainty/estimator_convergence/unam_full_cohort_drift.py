"""
Full 62-subject UNAM drift-consistency check -- is the first subject's
growing-relative-change pattern (across 500-5000 iterations) universal,
common, or a one-off? Cheap enough (~0.4s/subject at 5000 iterations
per the single-subject check) to just run the whole cohort, not a
subsample.
"""
import os
import numpy as np
from nctsens.loader import from_arrays
from nctsens.connectivity import graph_laplacian, ridge_effective_connectivity

# Paths were hardcoded to one lab machine in the original version, which
# meant this script silently failed for anyone else. Now CLI arguments.
import argparse as _argparse
_ap = _argparse.ArgumentParser(description=__doc__,
                               formatter_class=_argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--ts-dir", required=True,
                 help="Directory of UNAM dk68 timeseries .npy files")
_ap.add_argument("--sc-file", required=True,
                 help="Shared ENIGMA DK-68 template SC (UNAM has no individual DTI)")
_args = _ap.parse_args()
UNAM_TS_DIR = _args.ts_dir
UNAM_SC_FILE = _args.sc_file


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


A_template = np.loadtxt(UNAM_SC_FILE, delimiter=",")

ts_files = sorted(f for f in os.listdir(UNAM_TS_DIR) if f.endswith("_dk68_native_timeseries.npy"))
subjects = [f.split("_")[0].replace("sub-", "") for f in ts_files]
print(f"Found {len(subjects)} UNAM subjects\n")

pattern_counts = {"growing": 0, "shrinking": 0, "flat_zero": 0, "other": 0}
max_final_change = 0.0

for sub in subjects:
    try:
        X_raw = np.load(os.path.join(UNAM_TS_DIR, f"sub-{sub}_dk68_native_timeseries.npy"))
        data = from_arrays(X=X_raw.T, SC=A_template, subject_id=sub, validate=True)
        saved = graphnet_checkpoints(data["X"], data["SC"])

        ks = sorted(saved.keys())
        changes = []
        for i in range(1, len(ks)):
            rel = np.linalg.norm(saved[ks[i]] - saved[ks[i-1]], 'fro') / (np.linalg.norm(saved[ks[i-1]], 'fro') + 1e-12)
            changes.append(rel)

        max_final_change = max(max_final_change, changes[-1])

        if all(c == 0.0 for c in changes):
            pattern = "flat_zero"
        elif all(changes[i] <= changes[i-1] + 1e-12 for i in range(1, len(changes))):
            pattern = "shrinking"
        elif all(changes[i] >= changes[i-1] - 1e-12 for i in range(1, len(changes))):
            pattern = "growing"
        else:
            pattern = "other"
        pattern_counts[pattern] += 1
        print(f"  sub-{sub}: {pattern}, changes={[f'{c:.6f}' for c in changes]}")
    except Exception as e:
        print(f"  sub-{sub}: FAILED -- {e}")

print(f"\n{'='*60}")
print(f"Pattern counts across {len(subjects)} subjects: {pattern_counts}")
print(f"Largest final-checkpoint relative change seen: {max_final_change:.6f}")
print("If 'growing' is common (not just the first subject), this is a real,")
print("consistent characteristic of UNAM's optimization at this")
print("dimensionality -- worth naming honestly, though likely still too")
print("small in magnitude to change any actual result.")
print(f"{'='*60}")
