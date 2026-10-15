#!/usr/bin/env python3
"""
Does head motion predict bootstrap rank stability within HCP?

WHY THIS SCRIPT EXISTS
----------------------
The within-HCP motion result reported in the manuscript (Spearman
rho = +0.17, p = 0.008, n = 238) was originally computed interactively
and never written to a script. The number is in the paper; the code that
produced it was not in the repository. This file reconstructs the
analysis so the result is reproducible from the deposited data.

Anyone re-running it should expect the published value. If they do not
get it, the discrepancy matters more than the original number.

THE ANALYSIS
------------
Within cohort, correlate each subject's mean framewise displacement
against the mean pairwise Kendall's tau of their bootstrap driver-node
rankings. HCP is single-site and single-scanner, so no scanner covariate
is available or needed; that is the reason this cohort is worth testing
separately from ADNI, where manufacturer must be controlled.

The direction of interest is POSITIVE. If more motion predicts HIGHER
rank stability, reproducibility cannot be read as evidence that the
ranking is correct, which is the claim the manuscript makes.

TWO CAUTIONS CARRIED FROM THE ORIGINAL ANALYSIS
-----------------------------------------------
**The HCP relationship is real but fragile.** It does not survive
trimming the tails of the tau distribution. That is reported here rather
than omitted, and it has a quantified explanation: HCP's tau is
compressed near its ceiling, with a coefficient of variation around
0.016 against ADNI's 0.578, and a compressed outcome attenuates any
correlation with it. The script computes both the full and trimmed
estimates so the fragility is visible.

**Absolute framewise displacement is NOT comparable across cohorts.**
HCP's median mean-FD is roughly 0.75 against ADNI's 0.15, but these are
computed over different run structures by different pipelines. Only the
WITHIN-cohort correlation is interpretable, and only that is reported.
Nothing here licenses a statement about which cohort moved more.

USAGE
-----
    python3 hcp_motion_vs_tau.py \\
        --pathb path_b_bootstrap_results.csv \\
        --motion HCP_rsfMRI_motion_FD.csv \\
        --out hcp_motion_vs_tau.csv
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats


def load_any(path):
    sep = "\t" if path.endswith((".tsv", ".txt")) else ","
    return pd.read_csv(path, sep=sep)


def pick(df, candidates, what):
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(f"Could not find a {what} column. Looked for "
                     f"{candidates}; available: {list(df.columns)[:12]}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pathb", required=True,
                    help="Per-subject bootstrap results with kendall_tau_mean")
    ap.add_argument("--motion", required=True,
                    help="Per-subject motion summary with a mean-FD column")
    ap.add_argument("--tau-col", default=None)
    ap.add_argument("--fd-col", default=None)
    ap.add_argument("--trim", type=float, default=0.05,
                    help="Fraction trimmed from each tail of tau for the "
                         "sensitivity estimate. 0 disables.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pb = load_any(args.pathb)
    fd = load_any(args.motion)

    if "status" in pb.columns:
        n0 = len(pb)
        pb = pb[pb.status == "ok"].copy()
        print(f"  kept {len(pb)}/{n0} subjects with status ok")

    tau_col = args.tau_col or pick(pb, ["kendall_tau_mean", "mean_tau",
                                        "tau"], "tau")
    fd_col = args.fd_col or pick(fd, ["Mean_FD", "mean_fd", "func_fd_mean",
                                      "FD_mean", "fd_mean"], "mean-FD")
    sid_pb = pick(pb, ["subject_id", "Subject", "subject"], "subject id")
    sid_fd = pick(fd, ["Subject", "subject_id", "subject"], "subject id")

    pb["_sid"] = pb[sid_pb].astype(str).str.replace(r"^sub-", "", regex=True)
    fd["_sid"] = fd[sid_fd].astype(str).str.replace(r"^sub-", "", regex=True)
    m = pb[["_sid", tau_col]].merge(fd[["_sid", fd_col]], on="_sid")
    m = m.dropna()
    print(f"  merged {len(m)} subjects "
          f"({pb['_sid'].nunique()} bootstrap, {fd['_sid'].nunique()} motion)")
    if len(m) < 0.5 * len(pb):
        raise SystemExit("Fewer than half the subjects matched; check the "
                         "identifier formats before interpreting anything.")

    print("\n" + "=" * 70)
    print("HCP: MOTION versus BOOTSTRAP RANK STABILITY")
    print("=" * 70)
    print(f"  tau:     median {m[tau_col].median():.4f}   "
          f"IQR [{m[tau_col].quantile(.25):.4f}, {m[tau_col].quantile(.75):.4f}]"
          f"   CV {m[tau_col].std() / m[tau_col].mean():.4f}")
    print(f"  mean FD: median {m[fd_col].median():.4f}   "
          f"IQR [{m[fd_col].quantile(.25):.4f}, {m[fd_col].quantile(.75):.4f}]")
    print("  NOTE: absolute FD is not comparable with other cohorts; it is")
    print("  computed over a different run structure by a different pipeline.")

    rho, p = stats.spearmanr(m[fd_col], m[tau_col])
    r, pp = stats.pearsonr(m[fd_col], m[tau_col])
    print(f"\n  Spearman rho = {rho:+.4f}   p = {p:.2e}   n = {len(m)}")
    print(f"  Pearson  r   = {r:+.4f}   p = {pp:.2e}")
    print(f"  direction: more motion -> "
          f"{'HIGHER' if rho > 0 else 'LOWER'} rank stability")

    rows = [dict(analysis="full", n=len(m), spearman_rho=rho, spearman_p=p,
                 pearson_r=r, pearson_p=pp,
                 tau_cv=m[tau_col].std() / m[tau_col].mean())]

    if args.trim > 0:
        lo, hi = m[tau_col].quantile([args.trim, 1 - args.trim])
        t = m[(m[tau_col] > lo) & (m[tau_col] < hi)]
        if len(t) > 20:
            rho_t, p_t = stats.spearmanr(t[fd_col], t[tau_col])
            print(f"\n  TRIMMED ({args.trim:.0%} from each tail of tau, "
                  f"n = {len(t)}):")
            print(f"    Spearman rho = {rho_t:+.4f}   p = {p_t:.3f}")
            print("    The relationship is expected to weaken here. HCP's tau")
            print("    is compressed near its ceiling, and trimming a")
            print("    compressed outcome attenuates any correlation with it.")
            print("    Report the direction as replicating and the effect as")
            print("    robust only where the outcome has room to vary.")
            rows.append(dict(analysis=f"trimmed_{args.trim}", n=len(t),
                             spearman_rho=rho_t, spearman_p=p_t,
                             pearson_r=np.nan, pearson_p=np.nan,
                             tau_cv=t[tau_col].std() / t[tau_col].mean()))

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("\n  Published values for comparison: rho = +0.17, p = 0.008,")
    print("  n = 238 full; rho = +0.10, p = 0.14 trimmed.")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
