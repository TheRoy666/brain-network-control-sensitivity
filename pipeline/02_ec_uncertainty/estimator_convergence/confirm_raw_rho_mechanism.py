"""
Formally confirms (or refutes) that UNAM's long-open raw_rho pinning
anomaly is explained by the Ridge-EC finding -- checks whether
ridge_effective_connectivity's own spectral radius matches the
documented ~1.0 pinning value directly, on real data.
"""
import os
import numpy as np
import pandas as pd
from nctsens.loader import from_arrays
from nctsens.connectivity import ridge_effective_connectivity

UNAM_TS_DIR = "<<< SAME AS BEFORE >>>"
UNAM_SC_FILE = "<<< SAME AS BEFORE >>>"

SUBJECTS = os.environ.get("UNAM_SUBJECTS", "").split()   # subject IDs, space-separated

A_template = np.loadtxt(UNAM_SC_FILE, delimiter=",")

print("Documented raw_rho from unam_results.csv: mean=0.999995, std=0.000116")
print()
print("Ridge-only raw_rho, computed directly:")
for sub in SUBJECTS:
    X_raw = np.load(f"{UNAM_TS_DIR}/sub-{sub}_dk68_native_timeseries.npy")
    data = from_arrays(X=X_raw.T, SC=A_template, subject_id=sub, validate=True)
    A_ridge = ridge_effective_connectivity(data["X"], alpha=1.0, lag=1)
    raw_rho = np.max(np.abs(np.linalg.eigvals(A_ridge)))
    print(f"  sub-{sub}: {raw_rho:.6f}")

print()
print("If these match the documented ~1.0 pinning closely, the mechanism")
print("is formally confirmed: raw_rho's anomaly was always Ridge EC's")
print("own characteristic spectral radius at this dimensionality, not")
print("an unexplained artifact.")
