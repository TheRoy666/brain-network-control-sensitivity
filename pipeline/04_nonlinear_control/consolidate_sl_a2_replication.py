#!/usr/bin/env python3
"""
Item 2, follow-up (1 of 3): consolidate the Stuart-Landau A2 result as a
THIRD-MODEL REPLICATION for Section 3.

WHY THIS EXISTS
---------------
The beta sweep returned a null (linearization-bias direction is not
predicted by nonlinearity structure), and the whole SL run was
initially written off as a null because of it. That was wrong. The
FHN-comparable condition (beta=+1.0, gamma=0) is a genuine POSITIVE
replication of Section 3's existing claim -- "neither naive nor LTV
linearization dominates consistently; the nonlinear solution is never
worse" -- in a third independent model.

That claim previously rested on two models (WC, FHN). Stuart-Landau
makes three, and it is the strongest of the three methodologically:
it is the Hopf normal form, its Jacobian is analytic at every state,
and its limit cycle is closed-form rather than numerically integrated
and spline-interpolated. So the third model is not merely another data
point, it is the one with the fewest numerical error sources.

This script extracts the comparison cleanly so it can go straight into
Section 3 without re-deriving it from the raw sweep each time.

SCOPE NOTE -- state this in the paper, do not overclaim
-------------------------------------------------------
Three models agreeing on the QUALITATIVE A2 result is not the same as
the generality claim that was tested and failed. The failed claim was
that the DIRECTION of linearization bias is predictable from the
structure of the nonlinearity. That remains a null. What replicates is
the weaker, separate statement that neither linear approximation
dominates and the nonlinear ceiling always wins. Keep the two apart.

USAGE
-----
    python3 consolidate_sl_a2_replication.py \
        --sl-results sl_a2_results.csv \
        --fhn-results fhn_a2_broader_grid_results.csv \
        --out sl_a2_replication_summary
"""
import argparse
import os

import numpy as np
import pandas as pd


def summarise(df, naive_col="naive_err", ltv_col="ltv_err",
              ceil_col="ceiling_err", label=""):
    """Compute the three A2 statistics on rows where naive is defined."""
    d = df[df[naive_col].notna()].copy()
    if len(d) == 0:
        return None
    d["ltv_wins"] = d[ltv_col] < d[naive_col]
    best_lin = np.minimum(d[naive_col].values, d[ltv_col].values)
    ceil = d[ceil_col].values
    valid = ceil > 0
    return {
        "label": label,
        "n": len(d),
        "ltv_wins": int(d.ltv_wins.sum()),
        "ltv_win_rate": float(d.ltv_wins.mean()),
        "median_naive": float(d[naive_col].median()),
        "median_ltv": float(d[ltv_col].median()),
        "median_ceiling": float(np.median(ceil)),
        "median_ceiling_improvement": float(np.median(best_lin[valid] / ceil[valid])),
        "ceiling_never_worse_than_best_linear":
            bool(np.all(ceil[valid] <= best_lin[valid])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sl-results", required=True)
    ap.add_argument("--fhn-results", required=True)
    ap.add_argument("--anchor-beta", type=float, default=1.0,
                    help="The SL condition directly comparable to FHN. "
                         "beta=1.0, gamma=0 is the reference supercritical "
                         "case. NOTE: with gamma=0 beta scales out of the "
                         "dynamics entirely, so ALL supercritical beta values "
                         "are the same system up to amplitude -- this choice "
                         "is a normalisation, not a cherry-pick.")
    ap.add_argument("--out", default="sl_a2_replication_summary")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    sl = pd.read_csv(args.sl_results)
    fhn = pd.read_csv(args.fhn_results)

    sl_sweep = sl[sl.condition == "beta_sweep"]
    anchor = sl_sweep[np.isclose(sl_sweep.beta, args.anchor_beta)
                      & np.isclose(sl_sweep.gamma, 0.0)]

    rows = []
    r_fhn = summarise(fhn, label="FitzHugh-Nagumo (stable starts)")
    r_sl = summarise(anchor, label=f"Stuart-Landau (beta={args.anchor_beta:+.1f})")
    r_sl_all = summarise(sl_sweep, label="Stuart-Landau (all beta conditions)")
    for r in [r_fhn, r_sl, r_sl_all]:
        if r:
            rows.append(r)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.out, "a2_three_model_comparison.csv"), index=False)

    print("=" * 78)
    print("SECTION 3 — A2 RESULT ACROSS MODELS")
    print("=" * 78)
    print(f"{'model':<38} {'LTV wins':>12} {'ceiling gain':>14}")
    for r in rows:
        print(f"{r['label']:<38} {r['ltv_wins']:>4}/{r['n']:<3} "
              f"({r['ltv_win_rate']:>4.0%}) {r['median_ceiling_improvement']:>12.0f}x")

    print()
    print("  Wilson-Cowan is the third arm; its numbers live in the existing")
    print("  A2 analysis, not in either file read here. Pull them in before")
    print("  writing the three-model sentence.")
    print()

    print("=" * 78)
    print("WHAT REPLICATES, AND WHAT DOES NOT")
    print("=" * 78)
    print("  REPLICATES (now three models): neither naive nor LTV")
    print("  linearization dominates consistently, and the nonlinear")
    print("  solution is never worse than the better of the two.")
    for r in rows:
        print(f"    {r['label']}: ceiling never worse than best linear = "
              f"{r['ceiling_never_worse_than_best_linear']}")
    print()
    print("  DOES NOT REPLICATE / remains a NULL: the direction of")
    print("  linearization bias is NOT predicted by the structure of the")
    print("  nonlinearity. Supercritical vs subcritical median ltv/naive")
    print("  across four phases: 0.645 vs 0.626 — no separation.")
    print()
    print("  These are two DIFFERENT claims. Section 3 gets the positive;")
    print("  the generality cap stays at Wilson-Cowan because of the null.")
    print()

    # Methodological differences that belong in the methods section.
    print("=" * 78)
    print("METHODOLOGICAL DIFFERENCES BETWEEN ARMS — disclose, do not bury")
    print("=" * 78)
    print("  1. SL uses diffusive coupling (standard for Stuart-Landau; keeps")
    print("     the fixed point at the origin independent of C). WC uses")
    print("     direct additive drive. Real difference between arms.")
    print("  2. SL's Jacobian is analytic at every state and its limit cycle")
    print("     is closed-form. FHN integrates a nominal orbit numerically,")
    print("     saves it, and interpolates with cubic splines. SL therefore")
    print("     carries strictly fewer numerical error sources.")
    print("  3. SL's phase axis has only HALF-period symmetry, not full")
    print("     rotational invariance: the dynamics are rotationally")
    print("     symmetric but the control matrix G=[[1],[0]] injects along a")
    print("     fixed direction, breaking it. Phases 0.0/0.5 agree and")
    print("     0.25/0.75 agree, but 0.0 and 0.25 differ ~10x in naive_err.")
    print(f"\nSaved to {args.out}/")


if __name__ == "__main__":
    main()
