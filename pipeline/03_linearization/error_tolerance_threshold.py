#!/usr/bin/env python3
"""
error_tolerance_threshold.py — the linearization threshold defined by an
error tolerance instead of by the shape of the error curve (Supplementary S8
"Error tolerance"; Results 2.8).

For each subject, relative endpoint error is averaged over horizon and target
direction at every coupling, giving one error value per subject, displacement
and coupling. Coupling is then summarized per subject in three ways: the best
(lowest-error) coupling, the worst, and the mean over couplings. At each
displacement we take the fraction of subjects whose summarized error is within
the tolerance; the threshold is the displacement at which that fraction falls
through one half. The crossing is the last descent, interpolated in log
displacement, as for the critical displacement
(localize_critical_displacement.crossing_point).

    python3 pipeline/03_linearization/error_tolerance_threshold.py [--results results]

Prints one row per arm and coupling summary, for tolerances of 5% and 10%;
"none" means the fraction never falls below one half in the swept range.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from localize_critical_displacement import crossing_point  # noqa: E402

ARMS = ("HCP", "ADNI-400", "ADNI-450")
SUMMARIES = (("best", "min"), ("mean", "mean"), ("worst", "max"))


def tolerance_thresholds(sweep, tolerances=(0.05, 0.10)):
    """Threshold displacement per coupling summary and tolerance, from one fine sweep."""
    per = sweep.groupby(["subject_id", "target_scale", "w_EE"]).rel_reachability_error.mean().reset_index()
    out = {}
    for name, agg in SUMMARIES:
        s = per.groupby(["subject_id", "target_scale"]).rel_reachability_error.agg(agg).reset_index()
        ts = sorted(s.target_scale.unique())
        for tol in tolerances:
            frac = [float((s[s.target_scale == t].rel_reachability_error <= tol).mean()) for t in ts]
            out[(name, tol)] = crossing_point(ts, frac, 0.5)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    res = {}
    for arm in ARMS:
        f = os.path.join(a.results, "03_linearization", "fine_sweep", f"wc_sweep_finets_{arm}.csv.gz")
        res[arm] = tolerance_thresholds(pd.read_csv(f))
    fmt = lambda v: "none" if v is None or np.isnan(v) else f"{v:.4f}"
    print(f"{'coupling':10s} {'tolerance':>9s}  " + "  ".join(f"{arm:>9s}" for arm in ARMS))
    for name, _ in SUMMARIES:
        for tol in (0.05, 0.10):
            print(f"{name:10s} {tol:9.2f}  " + "  ".join(f"{fmt(res[arm][(name, tol)]):>9s}" for arm in ARMS))
    h, d = res["HCP"][("mean", 0.05)], res["ADNI-400"][("mean", 0.05)]
    print(f"\n5% tolerance, mean coupling: HCP and ADNI-400 differ by {100 * abs(h - d) / d:.1f}%")


if __name__ == "__main__":
    main()
