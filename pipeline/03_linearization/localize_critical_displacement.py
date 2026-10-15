#!/usr/bin/env python3
"""
Phase 20, option (b): localise the critical target displacement and test
the cohort difference PER-DISPLACEMENT rather than marginalized.

WHY THIS EXISTS
---------------
Section 2's published headline (HCP 37.0% vs ADNI 95.9% interior-minimum
fraction) is computed by averaging each subject's error curve over T,
target_scale and dir_seed, then taking the argmin. That marginalisation
is confounded:

  * At MATCHED target displacement the cohorts are similar, and at
    ts=0.01/0.02 HCP is HIGHER than ADNI -- opposite to the published
    direction.
  * The argmin location is unstable across displacement in every cohort
    (zero subjects hold the same position across all five scales), so
    the marginalized figure conflates "is there an interior minimum"
    with "is its location stable across displacement".
  * The average is dominated by the 2-4x larger, monotone-decreasing
    high-displacement curves.

NOT a confound, and explicitly checked: the w_EE grids. They differ in
absolute span between cohorts (HCP 1.361, ADNI 0.969) but are normalised
to each cohort's own stability boundary, so rho_A matches to ~0.005 at
every grid index. That is correct methodology and needs no change. An
earlier analysis wrongly flagged this as a confound; the retraction is
recorded in MASTER_TODO Phase 20.

WHAT THE RERUN NEEDS
--------------------
Only `target_scale` sampling changes. Keep the w_EE grid, T values,
dir_seeds and every other parameter EXACTLY as in the original sweep --
otherwise the new results are not comparable to the existing ones.

Existing sampling: 0.005, 0.01, 0.02, 0.05, 0.1 -- which brackets the
collapse between 0.02 and 0.05 but cannot localise it.

Suggested sampling (13 values, log-spaced through the collapse):
    0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05,
    0.06, 0.07, 0.085, 0.1

The four original values are retained so old and new runs can be
cross-checked against each other at those points -- a free correctness
check on the rerun.

WHAT THIS SCRIPT PRODUCES
-------------------------
  1. Interior fraction vs target displacement, per cohort.
  2. The critical displacement where each cohort crosses a stated
     criterion, with a bootstrap CI over subjects.
  3. A per-displacement cohort comparison (the honest version of the
     Section 2 contrast).
  4. The marginalized figure alongside, for explicit contrast.

USAGE
-----
    python3 localize_critical_displacement.py \
        --sweep HCP=path/to/hcp_sweep.csv \
        --sweep ADNI-S400=path/to/adni_s400_sweep.csv \
        --sweep ADNI-TianS3=path/to/adni_t3_sweep.csv \
        --out critical_displacement_results
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats


def interior_flags(df, param_col="w_EE", err_col="rel_reachability_error"):
    """Per-subject, per-target_scale: is the argmin interior?

    Returns dict {target_scale: {subject_id: bool}}.
    Marginalises over T and dir_seed WITHIN each target_scale -- which is
    the point: it does NOT average across displacements.
    """
    out = {}
    for ts, g in df.groupby("target_scale"):
        d = {}
        for sid, gs in g.groupby("subject_id"):
            c = gs.groupby(param_col)[err_col].mean().sort_index()
            i = int(np.argmin(c.values))
            d[sid] = 0 < i < len(c) - 1
        out[ts] = d
    return out


def marginalized_fraction(df, param_col="w_EE", err_col="rel_reachability_error"):
    """The PUBLISHED statistic: average over everything, then argmin."""
    f = []
    for sid, g in df.groupby("subject_id"):
        c = g.groupby(param_col)[err_col].mean().sort_index()
        i = int(np.argmin(c.values))
        f.append(0 < i < len(c) - 1)
    return float(np.mean(f))


def crossing_point(ts_vals, fracs, criterion=0.5):
    """Displacement above which the interior minimum disappears.

    Uses the LAST descent through `criterion`, not the first.

    This matters, and the first version of this function got it wrong.
    The observed curves are NOT monotone -- they contain dips that
    recover. ADNI-TianS3 crosses 0.5 twice: a dip to 0.449 at ts=0.015
    that recovers to 0.776 at ts=0.020, then the real collapse to 0.265
    at ts=0.025. Taking the FIRST crossing returned 0.0139 (the noise
    dip) against ADNI-Schaefer400's 0.0233 -- an apparent 41%
    disagreement between two parcellations OF THE SAME 49 SUBJECTS.
    Taking the LAST crossing gives 0.0226 vs 0.0233: 3.3% apart.

    The last crossing is not chosen because it makes the numbers agree.
    It is chosen because the scientific question is "ABOVE what
    displacement does the interior minimum disappear?", which is the
    final descent by definition. The first-crossing rule was simply the
    wrong estimator for that question on a non-monotone curve, and would
    have been wrong even if it had happened to agree.

    Linear interpolation in log(target_scale), because sampling is
    log-spaced and the collapse spans an order of magnitude. Returns nan
    if the curve never crosses -- reported honestly, not extrapolated.
    """
    ts_vals = np.asarray(ts_vals, dtype=float)
    fracs = np.asarray(fracs, dtype=float)
    order = np.argsort(ts_vals)
    x, y = np.log(ts_vals[order]), fracs[order]
    idx = [k for k in range(len(x) - 1)
           if (y[k] >= criterion) and (y[k + 1] < criterion)]
    if not idx:
        return float("nan")
    k = idx[-1]                      # LAST descent
    if y[k] == y[k + 1]:
        return float(np.exp(x[k]))
    t = (y[k] - criterion) / (y[k] - y[k + 1])
    return float(np.exp(x[k] + t * (x[k + 1] - x[k])))


def n_crossings(ts_vals, fracs, criterion=0.5):
    """How many times the curve descends through `criterion`.

    >1 means the curve is non-monotone and the threshold is not cleanly
    defined. Reported alongside every estimate so a single number is
    never mistaken for a clean collapse.
    """
    fracs = np.asarray(fracs, dtype=float)[np.argsort(ts_vals)]
    return sum(1 for k in range(len(fracs) - 1)
               if fracs[k] >= criterion > fracs[k + 1])


def bootstrap_crossing(flags_by_ts, criterion=0.5, n_boot=2000, seed=0):
    """CI on the crossing point by resampling SUBJECTS with replacement.

    Subjects are the unit of replication: each contributes one boolean
    per displacement. Resampling subjects (not rows) answers "how would
    this shift in a different sample of people".
    """
    rng = np.random.default_rng(seed)
    ts_vals = sorted(flags_by_ts)
    subjects = sorted(flags_by_ts[ts_vals[0]])
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(subjects, size=len(subjects), replace=True)
        fr = [np.mean([flags_by_ts[ts][s] for s in pick]) for ts in ts_vals]
        c = crossing_point(ts_vals, fr, criterion)
        if np.isfinite(c):
            vals.append(c)
    if not vals:
        return None
    return {"n_ok": len(vals), "median": float(np.median(vals)),
            "lo": float(np.percentile(vals, 2.5)),
            "hi": float(np.percentile(vals, 97.5))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="append", required=True,
                    help="LABEL=path.csv ; repeat per cohort")
    ap.add_argument("--criterion", type=float, default=0.5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="critical_displacement_results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cohorts = {}
    for spec in args.sweep:
        if "=" not in spec:
            raise SystemExit(f"--sweep needs LABEL=path, got {spec!r}")
        label, path = spec.split("=", 1)
        cohorts[label] = pd.read_csv(path)

    print("=" * 78)
    print("INTERIOR-MINIMUM FRACTION vs TARGET DISPLACEMENT")
    print("(computed WITHIN each displacement -- not averaged across them)")
    print("=" * 78)

    flags, rows = {}, []
    for label, df in cohorts.items():
        flags[label] = interior_flags(df)
        ts_vals = sorted(flags[label])
        n = df.subject_id.nunique()
        print(f"\n  {label}  (n={n} subjects, {len(ts_vals)} displacements)")
        for ts in ts_vals:
            fr = float(np.mean(list(flags[label][ts].values())))
            rows.append({"cohort": label, "target_scale": ts,
                         "interior_fraction": fr, "n": n})
            print(f"    ts={ts:<8.4f} interior fraction = {fr:.4f}")
        if len(ts_vals) < 8:
            print("    *** only "
                  f"{len(ts_vals)} displacements -- this is the ORIGINAL "
                  "coarse sampling, not the finer rerun. The crossing "
                  "estimate below will be poorly localised. ***")

    pd.DataFrame(rows).to_csv(
        os.path.join(args.out, "interior_fraction_by_displacement.csv"), index=False)

    print()
    print("=" * 78)
    print(f"CRITICAL DISPLACEMENT (interior fraction crosses {args.criterion})")
    print("=" * 78)
    crit_rows = []
    for label in cohorts:
        ts_vals = sorted(flags[label])
        fr = [np.mean(list(flags[label][ts].values())) for ts in ts_vals]
        pt = crossing_point(ts_vals, fr, args.criterion)
        ci = bootstrap_crossing(flags[label], args.criterion, args.n_boot)
        if not np.isfinite(pt):
            print(f"  {label:14s} never crosses {args.criterion} in the sampled range")
            crit_rows.append({"cohort": label, "critical_ts": np.nan})
            continue
        nx = n_crossings(ts_vals, fr, args.criterion)
        tail = [f for t, f in zip(ts_vals, fr) if t >= 0.05]
        line = f"  {label:14s} critical target_scale = {pt:.5f}"
        if ci:
            line += f"   95% CI [{ci['lo']:.5f}, {ci['hi']:.5f}]  ({ci['n_ok']}/{args.n_boot} fits)"
        print(line)
        print(f"                 descents through {args.criterion}: {nx}"
              + ("  *** NON-MONOTONE -- threshold not cleanly defined; the "
                 "estimate uses the LAST descent ***" if nx > 1 else ""))
        if tail:
            print(f"                 tail (ts>=0.05): max={max(tail):.4f}"
                  + ("  *** tail not clean -- 'vanishes above threshold' is "
                     "weaker for this cohort ***" if max(tail) > 0.1 else ""))
        crit_rows.append({"cohort": label, "critical_ts": pt,
                          "ci_lo": ci["lo"] if ci else np.nan,
                          "ci_hi": ci["hi"] if ci else np.nan})
    pd.DataFrame(crit_rows).to_csv(
        os.path.join(args.out, "critical_displacement.csv"), index=False)

    # Do the cohort CIs overlap? That is the real cohort test.
    fin = [r for r in crit_rows if np.isfinite(r.get("critical_ts", np.nan))
           and np.isfinite(r.get("ci_lo", np.nan))]
    if len(fin) >= 2:
        print()
        print("  Cohort comparison on the critical displacement:")
        overlap_all = True
        for i in range(len(fin)):
            for j in range(i + 1, len(fin)):
                a, b = fin[i], fin[j]
                ov = not (a["ci_hi"] < b["ci_lo"] or b["ci_hi"] < a["ci_lo"])
                overlap_all &= ov
                print(f"    {a['cohort']} vs {b['cohort']}: CIs "
                      f"{'OVERLAP' if ov else 'DO NOT overlap'}")
        print()
        print("    NOTE: this tool does not know which arms are cohorts and")
        print("    which are real-vs-null or other contrasts. Read the CI")
        print("    table above and interpret each pair yourself; the summary")
        print("    line below is only about whether ANY pair separates.")
        print("    -> " + ("no pair of arms separates on the critical "
                           "displacement (all CIs overlap)"
                           if overlap_all else
                           "at least one pair of arms SEPARATES -- identify "
                           "which from the table above before interpreting"))

    print()
    print("=" * 78)
    print("PER-DISPLACEMENT COHORT COMPARISON (the honest Section 2 contrast)")
    print("=" * 78)
    labels = list(cohorts)
    common = set(flags[labels[0]])
    for l in labels[1:]:
        common &= set(flags[l])
    for ts in sorted(common):
        parts = []
        for l in labels:
            parts.append(f"{l}={np.mean(list(flags[l][ts].values())):.3f}")
        print(f"    ts={ts:<8.4f} " + "   ".join(parts))

    print()
    print("=" * 78)
    print("MARGINALIZED FIGURE (the PUBLISHED statistic) -- for contrast only")
    print("=" * 78)
    for label, df in cohorts.items():
        print(f"    {label:14s} {marginalized_fraction(df):.4f}")
    print()
    print("  This is the statistic Phase 20 found confounded: it averages")
    print("  across displacements whose minima sit in different places, and")
    print("  is dominated by the large monotone high-displacement curves.")
    print("  Report the per-displacement table above instead.")
    print(f"\nSaved to {args.out}/")


if __name__ == "__main__":
    main()
