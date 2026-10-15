#!/usr/bin/env python3
"""
check_subject_coupling.py — why is one (subject, coupling) pair missing
from the fine linear-validity sweep?

THE QUESTION
------------
In ADNI-TianS3, subject ADNI_582 contributes four of the five
coupling levels; w_EE = 3.675 is absent. The sweep drops a pair for
exactly two reasons, in check_stable_fixed_point():

    1. find_fixed_point() raises RuntimeError   -> solver did not converge
    2. the Jacobian's largest real eigenvalue >= 0 -> fixed point unstable

Its stored output records neither which reason applied nor any detail;
the run log prints only a count of skipped pairs.

WHAT THE STORED DATA ALREADY SUGGEST
------------------------------------
This subject has the lowest stability margin of all 49 at w = 3.5062
(+0.0269), its margin falls faster at each step (-0.0036, -0.0046,
-0.0070), and the two subjects nearest it (+0.0286, +0.0290) lost about
0.024 in the final step, most of their remaining margin. Instability or
solver failure near the bifurcation is therefore the leading explanation.
This script tests it directly rather than inferring it.

HOW IT AVOIDS A SECOND IMPLEMENTATION
-------------------------------------
It imports load_and_scale_real_coupling, build_network and
check_stable_fixed_point from the sweep script itself, so the connectome
scaling, the network and the stability test are exactly those the sweep
used. It first reproduces a STORED margin (w = 3.5062, +0.0269) as a
sanity gate: if that does not match, the connectome or scaling differs
and nothing else it prints should be trusted.

USAGE
    cd pipeline/03_linearization
    python3 check_subject_coupling.py \\
        --sc <ADNI connectome dir>/ADNI_582_tians3_connectome.csv
"""
import argparse
import sys

import numpy as np

from wc_linear_validity_sweep_real_coupling import (
    load_and_scale_real_coupling, build_network, check_stable_fixed_point)

TIANS3_GRID = [3.0, 3.1687, 3.3375, 3.5062, 3.675]
STORED = {3.0: 0.0421, 3.1687: 0.0385, 3.3375: 0.0339, 3.5062: 0.0269}


def probe(N, w, C):
    net = build_network(N, w, C)
    try:
        r = check_stable_fixed_point(net)
    except Exception as e:                       # anything the sweep would not catch
        return dict(w=w, status=f"ERROR {type(e).__name__}: {str(e)[:60]}")
    if r is None:
        return dict(w=w, status="SOLVER DID NOT CONVERGE (RuntimeError)")
    E, I, max_re, stable = r
    return dict(w=w, status="stable" if stable else "UNSTABLE",
                max_real_eig=float(max_re), margin=float(-max_re))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sc", required=True, help="the subject's TianS3 connectome CSV")
    ap.add_argument("--fine", type=int, default=12,
                    help="extra points between 3.5062 and 3.675 to locate a crossing")
    a = ap.parse_args()

    C, N = load_and_scale_real_coupling(a.sc)   # returns (scaled matrix, N), as at the sweep's call site
    print(f"connectome {a.sc}\n  N = {N} regions (expected 450 for TianS3)\n")
    if N != 450:
        print("  ** unexpected region count — is this the TianS3 connectome? **")

    print("SANITY GATE — reproduce the stored margins")
    ok = True
    for w in TIANS3_GRID[:4]:
        r = probe(N, w, C)
        m = r.get("margin", np.nan)
        match = np.isfinite(m) and abs(m - STORED[w]) < 5e-4
        ok &= match
        print(f"  w={w:<7} {r['status']:10s} margin {m:+.4f}   stored {STORED[w]:+.4f}"
              f"   {'match' if match else 'MISMATCH'}")
    if not ok:
        print("\n  ** stored margins not reproduced: different connectome, scaling"
              " or code version. Do not interpret what follows. **")
        sys.exit(1)

    print("\nTHE MISSING LEVEL")
    r = probe(N, 3.675, C)
    print(f"  w=3.675  -> {r['status']}"
          + (f", margin {r['margin']:+.4f}" if "margin" in r else ""))

    print("\nWHERE DOES IT CHANGE? fine scan 3.5062 -> 3.675")
    last_ok = 3.5062
    for w in np.linspace(3.5062, 3.675, a.fine + 2)[1:]:
        r = probe(N, round(float(w), 4), C)
        tag = r["status"] + (f"  margin {r['margin']:+.5f}" if "margin" in r else "")
        print(f"  w={w:.4f}  {tag}")
        if r["status"] == "stable":
            last_ok = w
    print(f"\n  last stable coupling found: {last_ok:.4f}")

    print("\nWHAT TO WRITE")
    print("  If 3.675 is UNSTABLE: 'one subject's fixed point lost stability at")
    print("  the highest coupling level; it had the smallest stability margin of")
    print("  any subject at the level below.'")
    print("  If the SOLVER DID NOT CONVERGE while nearby couplings are stable:")
    print("  'the fixed-point solver failed to converge for one subject at the")
    print("  highest coupling level, close to its bifurcation.' Both are legitimate")
    print("  exclusions under the sweep's stated rule; the paper should say which.")


if __name__ == "__main__":
    main()
