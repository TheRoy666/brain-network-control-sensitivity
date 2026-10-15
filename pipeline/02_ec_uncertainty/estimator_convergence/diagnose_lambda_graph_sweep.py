#!/usr/bin/env python3
"""
UNAM_TLE -- diagnostic sweep: does reducing GraphNet's structural-prior
weight (lambda_graph) unpin raw_rho from ~1.0?

raw_rho here is the SPECTRAL RADIUS (dominant eigenvalue magnitude) of
the raw estimated EC matrix -- not a correlation coefficient. A value
pinned near exactly 1.0 sits at the precise boundary between a stable
and unstable discrete-time linear system, for every subject, which is
the actual anomaly being investigated.

Tests the Path A/B thread's hypothesis directly: the shared ENIGMA
template SC (identical across all 62 subjects) may be forcing the same
eigenstructure onto every subject's EC estimate via the Laplacian
regularization term, regardless of their actual BOLD signal.

Does NOT modify connectivity.py -- calls graphnet_effective_connectivity
with its existing, already-exposed lambda_graph parameter at several
values.

CORRECTED from the original version: the sweep variable was named
LAMBDA_RIDGE_VALUES and printed as "lambda_ridge=", but was actually
being passed into lambda_graph= in the function call the whole time --
lambda_ridge was silently fixed at 1.0, not swept. The computation was
therefore already testing the right thing; only the variable name,
print label, and saved CSV column were mislabeled. Fixed here so the
saved results aren't misread as testing the wrong parameter.

Usage:
    python diagnose_lambda_graph_sweep.py \
        --sc-file /path/to/template_SC_dk68.csv \
        --ts-dir  /path/to/derivatives/timeseries \
        --subjects 373 4001 4004 ...
"""
import argparse
import os
import numpy as np
import pandas as pd

from nctsens.loader import from_arrays
from nctsens.connectivity import graphnet_effective_connectivity

LAMBDA_GRAPH_VALUES = [1.0, 0.5, 0.1, 0.01, 0.001, 0.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc-file", required=True)
    ap.add_argument("--ts-dir", required=True)
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--out", default="lambda_graph_sweep_results.csv")
    args = ap.parse_args()

    A_template = np.loadtxt(args.sc_file, delimiter=",")

    rows = []
    for sub in args.subjects:
        ts_path = os.path.join(args.ts_dir, f"sub-{sub}_dk68_native_timeseries.npy")
        X_raw = np.load(ts_path)
        data = from_arrays(X=X_raw.T, SC=A_template, subject_id=sub, validate=True)
        X, SC = data["X"], data["SC"]

        print(f"\n=== sub-{sub} ===")
        for lam_g in LAMBDA_GRAPH_VALUES:
            EC = graphnet_effective_connectivity(
                X, SC, lambda_ridge=1.0, lambda_graph=lam_g)
            raw_rho = float(np.max(np.abs(np.linalg.eigvals(EC))))
            ec_asym = float(np.linalg.norm(EC - EC.T, "fro") / np.linalg.norm(EC, "fro"))
            rows.append({"subject": sub, "lambda_graph": lam_g,
                        "raw_rho": raw_rho, "ec_asymmetry": ec_asym})
            print(f"  lambda_graph={lam_g:6.3f}  raw_rho={raw_rho:.6f}  "
                  f"ec_asymmetry={ec_asym:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print(f"\n{'='*60}")
    print("INTERPRETATION:")
    print("If raw_rho moves substantially away from 1.0 as lambda_graph")
    print("decreases -> confirms the shared-template-prior hypothesis.")
    print("If raw_rho stays pinned near 1.0 regardless of lambda_graph")
    print("-> the prior weight isn't the mechanism, look elsewhere")
    print("  (e.g. lambda_ridge, the BOLD data itself, or the estimator's")
    print("  numerics at this dimension).")
    print(f"{'='*60}")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
