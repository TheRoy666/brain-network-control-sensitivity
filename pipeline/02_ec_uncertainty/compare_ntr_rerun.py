#!/usr/bin/env python3
"""
compare_ntr_rerun.py — check the dual-scored truncation series against the
published run, then restate every claim of that analysis for DRIVER rankings.

GATE
  The estimator is deterministic and the bootstrap seed fixed, so each
  published mean_tau must be reproduced EXACTLY (1e-9) by the rerun's package
  column, cell by cell. Pilots and --published-setting-only runs are gated on
  the cells they contain (--allow-subset). A mismatch stops everything.

WHAT IT REPORTS, under both definitions
  1. the truncation series: median tau and top-5 Jaccard at the published
     setting, by recording length and timepoints per region
  2. penalty sensitivity by length: median within-subject spread across the
     nine settings (full-grid runs only)
  3. the matched-ratio comparison: HCP truncated to 200 timepoints (N/T 2.05)
     against ADNI at its native length (N/T 2.03, from the dual estimator sweep)
  4. the 600-timepoint dip: fraction of subjects lower at 600 than at both
     neighbouring lengths
  5. graph-penalty inertness: fraction of subjects whose tau is bit-identical
     between prior 0 and prior 1 at the same ridge setting

USAGE
  python3 compare_ntr_rerun.py --published sensitivity_vs_ntr_HCP.csv \\
      --rerun sensitivity_vs_ntr_HCP_dual.csv \\
      --adni graphnet_sensitivity_ADNI_Schaefer400_dual.csv graphnet_sensitivity_ADNI_TianS3_dual.csv
"""
import argparse
import sys

import numpy as np
import pandas as pd

KEYS = ["subject_id", "requested_length", "lambda_ridge", "lambda_graph"]
DEFS = (("package", "mean_tau", "jaccard_top5"), ("driver", "mean_tau_gu", "jaccard_top5_gu"))


def gate(pub, new, allow_subset):
    for d in (pub, new):
        d["subject_id"] = d["subject_id"].astype(str)
    if allow_subset:
        pub = pub.merge(new[KEYS], on=KEYS, how="inner")
        print(f"SUBSET MODE — gating {len(pub)} published cells present in the rerun")
    m = pub[KEYS + ["mean_tau"]].merge(new[KEYS + ["mean_tau"]], on=KEYS, how="left",
                                       suffixes=("_pub", "_new"), indicator=True)
    missing = int((m["_merge"] == "left_only").sum())
    both_nan = m.mean_tau_pub.isna() & m.mean_tau_new.isna()
    diff = (m.mean_tau_pub - m.mean_tau_new).abs()
    bad = m[(m["_merge"] == "both") & ~both_nan & ~(diff <= 1e-9)]
    print(f"GATE  cells {len(pub)}, missing {missing}, not reproduced {len(bad)}, "
          f"max |difference| {np.nanmax(diff.values) if diff.notna().any() else 0:.2e}")
    if missing or len(bad):
        print("  ** FAIL — check --n-boot 25, --n-subjects 0, the lengths and the inputs **")
        sys.exit(1)
    print("  pass — the rerun reproduces the published run exactly\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--published", required=True)
    ap.add_argument("--rerun", required=True, nargs="+",
                    help="one or more dual-scored files (e.g. the published-setting run and the full-grid run)")
    ap.add_argument("--adni", nargs="*", default=[], help="dual estimator-sweep CSVs for ADNI")
    ap.add_argument("--allow-subset", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pub = pd.read_csv(a.published)
    new = pd.concat([pd.read_csv(f) for f in a.rerun], ignore_index=True)
    new["subject_id"] = new["subject_id"].astype(str)
    dup = new.duplicated(KEYS)
    if dup.any():
        print(f"  {int(dup.sum())} cells appear in more than one rerun file — keeping one copy")
        new = new[~dup]
    for c in ("mean_tau_gu", "jaccard_top5", "jaccard_top5_gu"):
        if c not in new:
            sys.exit(f"rerun lacks {c}: not produced by sensitivity_vs_ntr_dual.py")
    gate(pub, new, a.allow_subset)

    pubset = new[(new.lambda_ridge == 1) & (new.lambda_graph == 1)]
    # penalty sensitivity only where the FULL grid was run at that length;
    # a single setting would give a spread of zero, which is not insensitivity
    grid_at = new.groupby("requested_length").apply(
        lambda g: g.groupby(["lambda_ridge", "lambda_graph"]).ngroups == 9)
    full_grid = bool(grid_at.any())
    rows = []
    for L, g in pubset.groupby("requested_length"):
        r = dict(length=int(L), T_over_N=round(float((g["T"] / g["N"]).median()), 2),
                 N_over_T=round(float(g.N_over_T.median()), 3), n=int(g.mean_tau_gu.notna().sum()))
        for lab, tc, jc in DEFS:
            r[f"tau_{lab}"] = round(float(g[tc].median()), 3)
            r[f"jaccard_{lab}"] = round(float(g[jc].median()), 3)
            if bool(grid_at.get(L, False)):
                pv = new[new.requested_length == L].pivot_table(index="subject_id",
                        columns=["lambda_ridge", "lambda_graph"], values=tc)
                r[f"spread_{lab}"] = float((pv.max(axis=1) - pv.min(axis=1)).median())
        rows.append(r)
    t = pd.DataFrame(rows).sort_values("length", ascending=False)
    print("1-2. TRUNCATION SERIES (published setting; spread over the nine settings where the full grid was run)\n")
    print(t.to_string(index=False)); print()

    if a.adni:
        print("3. MATCHED RATIO — HCP at 200 timepoints against ADNI at its native length\n")
        h = t[t.length == 200]
        for f in a.adni:
            d = pd.read_csv(f); p = d[(d.lambda_ridge == 1) & (d.lambda_graph == 1)]
            nt = float((d.N / d["T"]).median())
            for lab, tc, _ in DEFS:
                pv = d.pivot_table(index="subject_id", columns=["lambda_ridge", "lambda_graph"], values=tc)
                sp = float((pv.max(axis=1) - pv.min(axis=1)).median())
                hv = float(h[f"tau_{lab}"].iloc[0]) if len(h) else np.nan
                hs = float(h[f"spread_{lab}"].iloc[0]) if (len(h) and f"spread_{lab}" in h) else np.nan
                print(f"   {f.split('graphnet_sensitivity_')[-1][:-9]:14s} N/T {nt:.2f} {lab:8s}: "
                      f"ADNI tau {p[tc].median():.3f}, spread {sp:.3g}  |  HCP@200 tau {hv:.3f}, spread {hs:.3g}  "
                      f"->  tau ratio {hv / p[tc].median():.1f}x, spread ratio {sp / hs if hs else np.nan:.0f}x")
        print()

    print("4. THE 600-TIMEPOINT DIP — subjects lower at 600 than at BOTH 1200 and 400\n")
    for lab, tc, _ in DEFS:
        w = pubset.pivot_table(index="subject_id", columns="requested_length", values=tc)
        if {400, 600, 1200} <= set(w.columns):
            dip = ((w[600] < w[1200]) & (w[600] < w[400])).mean()
            print(f"   {lab:8s} {dip:.0%} of {len(w)} subjects")
    print()
    if full_grid:
        print("5. GRAPH-PENALTY INERTNESS — tau bit-identical between prior 0 and prior 1\n")
        for lab, tc, _ in DEFS:
            fr = []
            for (L, lr), g in new.groupby(["requested_length", "lambda_ridge"]):
                w = g.pivot_table(index="subject_id", columns="lambda_graph", values=tc)
                if {0.0, 1.0} <= set(w.columns):
                    fr.append(((w[0.0] - w[1.0]).abs() == 0).mean())
            print(f"   {lab:8s} identical in {np.mean(fr):.1%} of subject-length-ridge cells")
    if a.out:
        t.to_csv(a.out, index=False); print(f"\nwritten {a.out}")


if __name__ == "__main__":
    main()
