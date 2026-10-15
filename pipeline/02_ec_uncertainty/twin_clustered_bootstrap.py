#!/usr/bin/env python3
"""
Recompute HCP confidence intervals with twin-pair clustering.

THE PROBLEM
-----------
The HCP sample is 238 individuals forming 119 complete same-sex twin
pairs, 80 monozygotic and 39 dizygotic. Every confidence interval we
report for that cohort was obtained by resampling the 238 individuals
independently.

That treats siblings as independent observations when they are not.
Monozygotic pairs share effectively all segregating genetic variation
and dizygotic pairs half, and the connectome and its derived metrics are
heritable. Resampling individuals therefore draws many samples that
contain both members of a pair and behaves as though it has 238
independent units when it has closer to 119. The consequence is
one-directional: intervals are too NARROW and any p-value too small.
Two referees raised it.

THE FIX
-------
Resample PAIRS with replacement, taking both members whenever a pair is
drawn. This is the standard cluster bootstrap and the correct unit here
is the family, not the person.

We report both so the size of the correction is visible rather than
merely asserted, and because a reader needs to know whether the
published intervals were materially wrong or only slightly optimistic.

An intraclass correlation is also computed per metric. It is what
determines the size of the correction: at ICC near zero, pairs carry no
shared information and the two bootstraps agree; at ICC near one, the
effective sample size approaches 119 and intervals widen by roughly the
square root of two. Reporting the ICC lets the reader judge whether the
correction is doing real work.

WHAT THIS DOES NOT FIX
----------------------
Clustering corrects the INTERVAL, not the POINT ESTIMATE. A median
computed over 238 non-independent individuals remains a median over 238
individuals; it is not reweighted. Nor does this address zygosity: MZ
and DZ pairs have different expected within-pair correlation, and a
cluster bootstrap treats them identically. A variance-components model
would do better and is out of scope here.

USAGE
-----
    python3 twin_clustered_bootstrap.py \\
        --values path_b_bootstrap_results.csv \\
        --value-cols kendall_tau_mean,jaccard_topk_mean \\
        --pairs aud_cohort_results_public.csv \\
        --pair-col TwinPairID --subject-col subject_id \\
        --out twin_clustered_ci.csv

--values may be any per-subject table. --pairs must map each subject to
a family identifier; if the two files use different identifier formats
the script reports the overlap and stops rather than silently analysing
whatever happened to match.
"""
import argparse

import numpy as np
import pandas as pd

N_BOOT = 10000


def icc_from_pairs(df, value_col, pair_col):
    """One-way ICC: between-pair variance over total variance.

    Uses only complete pairs. Returns nan if fewer than five are
    available, since the estimate is meaningless below that.
    """
    g = df.groupby(pair_col)[value_col]
    complete = [v.values for _, v in g if len(v) == 2 and np.isfinite(v).all()]
    if len(complete) < 5:
        return np.nan, len(complete)
    a = np.array(complete)
    pair_means = a.mean(axis=1)
    grand = a.mean()
    ms_between = 2 * np.sum((pair_means - grand) ** 2) / (len(a) - 1)
    ms_within = np.sum((a - pair_means[:, None]) ** 2) / len(a)
    denom = ms_between + ms_within
    if denom <= 0:
        return np.nan, len(a)
    return float((ms_between - ms_within) / denom), len(a)


def boot_ci(values, stat=np.median, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float)
    draws = [stat(rng.choice(v, size=len(v), replace=True))
             for _ in range(n_boot)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def boot_ci_clustered(df, value_col, pair_col, stat=np.median,
                      n_boot=N_BOOT, seed=0):
    """Resample PAIRS with replacement, taking both members each time."""
    rng = np.random.default_rng(seed)
    groups = [g[value_col].values for _, g in df.groupby(pair_col)]
    groups = [g for g in groups if len(g) and np.isfinite(g).all()]
    n = len(groups)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws.append(stat(np.concatenate([groups[i] for i in idx])))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--values", required=True)
    ap.add_argument("--value-cols", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--pair-col", default="TwinPairID")
    ap.add_argument("--subject-col", default="subject_id")
    ap.add_argument("--pairs-subject-col", default=None,
                    help="If the pairing file names subjects differently.")
    ap.add_argument("--strip-prefix", default=None,
                    help="Prefix to strip from the values file's subject ids "
                         "before merging, e.g. sub-")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    vals = pd.read_csv(args.values)
    pairs = pd.read_csv(args.pairs)
    psub = args.pairs_subject_col or args.subject_col

    vals["_sid"] = vals[args.subject_col].astype(str)
    if args.strip_prefix:
        vals["_sid"] = vals["_sid"].str.replace(f"^{args.strip_prefix}", "",
                                                regex=True)
    pairs["_sid"] = pairs[psub].astype(str)

    m = vals.merge(pairs[["_sid", args.pair_col]], on="_sid", how="inner")
    print(f"values file: {len(vals)} rows, pairing file: {len(pairs)} rows")
    print(f"merged: {len(m)} subjects, {m[args.pair_col].nunique()} families")
    if len(m) < 0.5 * len(vals):
        print("\n  Fewer than half the subjects matched. Identifier formats "
              "probably differ.")
        print(f"  values ids: {list(vals['_sid'].head(3))}")
        print(f"  pairs  ids: {list(pairs['_sid'].head(3))}")
        print("  Use --strip-prefix or --pairs-subject-col. Stopping rather "
              "than analysing a partial merge.")
        raise SystemExit(1)

    sizes = m.groupby(args.pair_col).size()
    print(f"  family sizes: {sizes.value_counts().to_dict()}")
    print(f"  complete pairs: {(sizes == 2).sum()}, singletons: {(sizes == 1).sum()}")

    rows = []
    print("\n" + "=" * 76)
    print("SUBJECT-LEVEL vs TWIN-PAIR-CLUSTERED BOOTSTRAP")
    print("=" * 76)
    for col in [c.strip() for c in args.value_cols.split(",")]:
        if col not in m:
            print(f"\n  {col}: not present, skipped")
            continue
        sub = m[[col, args.pair_col]].dropna()
        if len(sub) < 20:
            print(f"\n  {col}: only {len(sub)} usable rows, skipped")
            continue
        lo_i, hi_i = boot_ci(sub[col].values, n_boot=args.n_boot)
        lo_c, hi_c = boot_ci_clustered(sub, col, args.pair_col,
                                       n_boot=args.n_boot)
        icc, n_pairs = icc_from_pairs(sub, col, args.pair_col)
        w_i, w_c = hi_i - lo_i, hi_c - lo_c
        print(f"\n  {col}   median {np.median(sub[col]):.4f}   n = {len(sub)}")
        print(f"    independent  95% CI [{lo_i:.4f}, {hi_i:.4f}]  width {w_i:.4f}")
        print(f"    pair-clustered            [{lo_c:.4f}, {hi_c:.4f}]  width {w_c:.4f}")
        print(f"    widening factor {w_c / w_i if w_i else float('nan'):.2f}x"
              f"    ICC {icc:.3f} (from {n_pairs} complete pairs)")
        rows.append(dict(metric=col, n=len(sub), n_pairs=n_pairs,
                         median=float(np.median(sub[col])),
                         ci_lo_independent=lo_i, ci_hi_independent=hi_i,
                         ci_lo_clustered=lo_c, ci_hi_clustered=hi_c,
                         width_independent=w_i, width_clustered=w_c,
                         widening=w_c / w_i if w_i else np.nan, icc=icc))

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("\n" + "-" * 76)
    print("  A widening factor near 1 means the published intervals were")
    print("  already adequate and the objection, though correct in principle,")
    print("  does not change any reported number. A factor near 1.4 means the")
    print("  effective sample size is closer to the family count than to the")
    print("  individual count, and every HCP interval should be restated.")
    print("  The ICC column says which case you are in before you look at the")
    print("  widening, and the two should agree.")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
