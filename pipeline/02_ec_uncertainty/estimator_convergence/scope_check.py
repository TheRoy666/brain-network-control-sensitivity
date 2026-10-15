"""
Checks whether the ridge-only-equivalence bug is universal across a
range of lambda_ridge/lambda_graph combinations, or specific to the
one pair already confirmed -- tells us how likely this is to have
affected real production runs using different parameter choices.
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

print(f"{'lambda_ridge':>12} {'lambda_graph':>12} {'identical_to_ridge':>20}")
for lr in [1.0, 0.5, 0.1, 0.01]:
    A_ridge_lr = ridge_effective_connectivity(X, alpha=lr, lag=1)
    for lg in [1.0, 0.5, 0.1, 0.01, 0.001, 0.0]:
        A_gn = graphnet_effective_connectivity(X, SC, lambda_ridge=lr, lambda_graph=lg)
        identical = np.allclose(A_ridge_lr, A_gn)
        print(f"{lr:>12} {lg:>12} {str(identical):>20}")
