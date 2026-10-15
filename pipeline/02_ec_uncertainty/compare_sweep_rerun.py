#!/usr/bin/env python3
"""
compare_sweep_rerun.py — check a rerun sweep against the published one,
then summarise it under both controllability definitions.

The rerun scripts score the same bootstrap resamples twice:
  mean_tau, jaccard_top5         package definition (as published)
  mean_tau_gu, jaccard_top5_gu   Gu et al. (2015) definition (drivers)

GATE — the rerun is trusted only if it reproduces the published run.
The estimator is deterministic and the bootstrap seed is fixed, so every
published mean_tau must be reproduced EXACTLY (to 1e-9) by the rerun's
package column, cell by cell. Any mismatch means different code, inputs
or settings (for example the estimator sweep's default of 30 resamples
instead of the published 50), and nothing is summarised.

USAGE
  python3 compare_sweep_rerun.py estimator --published graphnet_sensitivity_HCP.csv \\
      --rerun graphnet_sensitivity_HCP_dual.csv
  python3 compare_sweep_rerun.py blocklength --published block_length_HCP.csv \\
      --rerun block_length_HCP_dual.csv
"""
import argparse
import sys

import numpy as np
import pandas as pd

KEYS = {"estimator": ["subject_id", "lambda_ridge", "lambda_graph", "lag"],
        "blocklength": ["subject_id", "block_length"]}
TOL = 1e-9
PUBLISHED_SETTING = (1.0, 1.0)


def gate(pub, new, keys, allow_subset=False):
    pub = pub.copy(); new = new.copy()
    for d in (pub, new):
        d["subject_id"] = d["subject_id"].astype(str)
    if allow_subset:        # pilot: check only the subjects the rerun contains
        pub = pub[pub.subject_id.isin(set(new.subject_id))]
        print(f"PILOT MODE — checking {pub.subject_id.nunique()} subjects only; "
              "the full run must pass without --allow-subset")
    m = pub[keys + ["mean_tau"]].merge(new[keys + ["mean_tau"]], on=keys, how="left",
                                       suffixes=("_pub", "_new"), indicator=True)
    missing = int((m["_merge"] == "left_only").sum())
    both_nan = m.mean_tau_pub.isna() & m.mean_tau_new.isna()
    diff = (m.mean_tau_pub - m.mean_tau_new).abs()
    bad = m[(m["_merge"] == "both") & ~both_nan & ~(diff <= TOL)]   # missing cells counted separately
    print(f"GATE  published cells {len(pub)}, found in rerun {len(pub) - missing}, "
          f"max |difference| {np.nanmax(diff.values) if diff.notna().any() else 0:.2e}")
    if missing or len(bad):
        print(f"  ** FAIL: {missing} published cells missing, {len(bad)} not reproduced **")
        print(bad.head(8).to_string())
        print("  Check --n-boot (published: 50), the grid, block lengths and the code version.")
        sys.exit(1)
    print("  pass — the rerun reproduces the published run exactly\n")


def estimator(new):
    rows = []
    for coh, d in new.groupby("cohort"):
        N, T = d.N.median(), d["T"].median()
        rec = {"cohort": coh, "parameters per observation": round(N / T, 2)}
        for tag, col in (("package", "mean_tau"), ("gu", "mean_tau_gu")):
            ok = d.dropna(subset=[col])
            pub = ok[(ok.lambda_ridge == PUBLISHED_SETTING[0]) & (ok.lambda_graph == PUBLISHED_SETTING[1])]
            grid = ok.groupby(["lambda_ridge", "lambda_graph", "lag"])[col].median()
            pv = ok.pivot_table(index="subject_id", columns=["lambda_ridge", "lambda_graph", "lag"], values=col)
            spread = (pv.max(axis=1) - pv.min(axis=1)).median()
            rec[f"median tau, published setting [{tag}]"] = round(pub[col].median(), 3)
            rec[f"grid range of median tau [{tag}]"] = f"{grid.min():.3f} – {grid.max():.3f}"
            rec[f"within-subject spread [{tag}]"] = f"{spread:.2g}"
            rec[f"cells reaching tau > 0.7 [{tag}]"] = f"{int((ok[col] > 0.7).sum())} / {len(ok)}"
        rows.append(rec)
    return pd.DataFrame(rows).set_index("cohort").T


def blocklength(new):
    out = []
    for coh, d in new.groupby("cohort"):
        for bl, g in d.groupby("block_length"):
            out.append(dict(cohort=coh, block_length=int(bl),
                            n=int(g.mean_tau.notna().sum()),
                            tau_package=round(g.mean_tau.median(), 3),
                            tau_gu=round(g.mean_tau_gu.median(), 3),
                            jaccard_package=round(g.jaccard_top5.median(), 3),
                            jaccard_gu=round(g.jaccard_top5_gu.median(), 3)))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=list(KEYS))
    ap.add_argument("--published", required=True)
    ap.add_argument("--rerun", required=True)
    ap.add_argument("--out", default=None, help="write the summary table as CSV")
    ap.add_argument("--allow-subset", action="store_true",
                    help="pilot runs: gate only the subjects present in the rerun")
    a = ap.parse_args()
    pub, new = pd.read_csv(a.published), pd.read_csv(a.rerun)
    for c in ("mean_tau_gu", "jaccard_top5", "jaccard_top5_gu"):
        if c not in new:
            sys.exit(f"rerun file lacks {c}: it was not produced by the dual-scoring script")
    gate(pub, new, KEYS[a.kind], a.allow_subset)
    t = estimator(new) if a.kind == "estimator" else blocklength(new)
    print(t.to_string())
    if a.out:
        t.to_csv(a.out); print(f"\nwritten {a.out}")


if __name__ == "__main__":
    main()
