"""
Standalone confirmation check -- uses the exact same X/SC loading logic
already proven working in diagnose_lambda_graph_sweep.py, on one real
subject, to confirm directly whether graphnet_effective_connectivity is
actually returning the ridge-only starting point regardless of
lambda_graph.
"""
import os
import numpy as np
from nctsens.loader import from_arrays
from nctsens.connectivity import ridge_effective_connectivity, graphnet_effective_connectivity

SC_FILE = os.path.expandvars("$BNCS_ROOT/Epilepsy/derivatives/timeseries/template_SC_dk68.csv")
TS_DIR = os.path.expandvars("$BNCS_ROOT/Epilepsy/derivatives/timeseries")
SUBJECT = os.environ["UNAM_SUBJECT"]

A_template = np.loadtxt(SC_FILE, delimiter=",")
X_raw = np.load(f"{TS_DIR}/sub-{SUBJECT}_dk68_native_timeseries.npy")
data = from_arrays(X=X_raw.T, SC=A_template, subject_id=SUBJECT, validate=True)
X, SC = data["X"], data["SC"]

A_ridge = ridge_effective_connectivity(X, alpha=1.0, lag=1)
A_graphnet = graphnet_effective_connectivity(X, SC, lambda_ridge=1.0, lambda_graph=0.5)

print("Identical to ridge-only:", np.allclose(A_ridge, A_graphnet))
print(f"Max abs difference: {np.max(np.abs(A_ridge - A_graphnet)):.10f}")
