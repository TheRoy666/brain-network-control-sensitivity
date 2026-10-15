"""
Checks, empirically, on real data from each cohort, whether the
ridge-only-equivalence bug actually triggered at production parameters
(lambda_ridge=1.0, lambda_graph=1.0) -- confirmed to trigger on at
least one real UNAM subject, confirmed NOT universal on synthetic data,
so this must be checked per-cohort, not assumed either way.

Run this BEFORE patching connectivity.py, using the current (buggy)
installed version -- this checks what actually happened in the results
already reported, not what would happen after a fix.
"""
import numpy as np
from nctsens.loader import load_connectome, from_arrays
from nctsens.connectivity import ridge_effective_connectivity, graphnet_effective_connectivity

def check_cohort(name, ts_paths, sc_paths_or_template, is_shared_template=False):
    print(f"\n=== {name} ===")
    n_affected = 0
    for i, ts_path in enumerate(ts_paths):
        X_raw = np.load(ts_path)
        SC_raw = sc_paths_or_template if is_shared_template else load_connectome(sc_paths_or_template[i])
        data = from_arrays(X=X_raw.T, SC=SC_raw, subject_id=f"check{i}", validate=True)
        X, SC = data["X"], data["SC"]

        A_ridge = ridge_effective_connectivity(X, alpha=1.0, lag=1)
        A_graphnet = graphnet_effective_connectivity(X, SC, lambda_ridge=1.0, lambda_graph=1.0)
        affected = np.allclose(A_ridge, A_graphnet)
        n_affected += affected
        print(f"  subject {i}: bug triggered = {affected}")
    print(f"  -> {n_affected}/{len(ts_paths)} subjects affected in this sample")
    return n_affected, len(ts_paths)

# Fill these in with real paths for a handful (5-10) of real subjects
# from EACH cohort, using files already known to exist from earlier work.

# Example shape -- adjust paths to your actual server layout:
# hcp_ts = ["/path/to/HCP/timeseries/<HCP subject ID>_native_timeseries.npy", ...]
# hcp_sc = ["/path/to/HCP/SC_matrices/<HCP subject ID>_SC_SIFT2_410.csv", ...]
# check_cohort("HCP-AUD", hcp_ts, hcp_sc)

# adni_s400_ts = [...]
# adni_s400_sc = [...]
# check_cohort("ADNI-Schaefer400", adni_s400_ts, adni_s400_sc)

# adni_tians3_ts = [...]
# adni_tians3_sc = [...]
# check_cohort("ADNI-TianS3", adni_tians3_ts, adni_tians3_sc)

# unam_template = np.loadtxt("/path/to/template_SC_dk68.csv", delimiter=",")
# unam_ts = [...]
# check_cohort("UNAM", unam_ts, unam_template, is_shared_template=True)
