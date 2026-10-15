#!/usr/bin/env python3
"""
compare_frozen_vs_published.py — does the FC/EC divergence still rise with
shorter recordings once the brain states are held fixed?

WHY THIS RUN EXISTS
-------------------
The published truncation analysis recomputed the two states from each
truncated recording. It therefore changed two things at once: the
operator estimates AND the transition being evaluated. Every reviewer
identified this. The frozen-state run holds the states at their
full-length values so that only the operator estimates change with T.

DECISION RULE — FIXED BEFORE THE RESULT EXISTS (2026-09-29)
-----------------------------------------------------------
Let R(T) be the median FC/EC energy ratio at T timepoints. The share of
the published rise that survives freezing the states is

      S = log( R_frozen(600) / R_frozen(4800) )
          / log( R_published(600) / R_published(4800) )

on the log scale, because the ratio is multiplicative. Then:

  S >= 0.75   the rise is mostly operator estimation. The "estimation
              quality" claim stands, now isolated, and is STRONGER.
  0.25 - 0.75 both matter. Restate: shorter recordings change both the
              operator estimates and the transition, each contributing.
  S <  0.25   the published rise was mostly the states moving. Restate:
              recording length changes which transition is evaluated;
              operator estimation accounts for little of the rise.

These thresholds are written down now, before any frozen result exists,
so the reading cannot drift toward whatever the data turn out to show.
Record S and its bootstrap interval whichever band it falls in.

SANITY GATE
-----------
At full length the frozen and recomputed states are the same states, and
the estimator is deterministic. So at T = 4800 the frozen run must
reproduce the published per-subject ratios to floating-point precision.
If it does not, the two runs used different inputs or settings, and the
comparison is void. The script stops.

USAGE
    python3 compare_frozen_vs_published.py \\
        --published operator_divergence_vs_length_HCP.csv \\
        --frozen    operator_divergence_frozen_states_HCP.csv
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

T_FULL, T_SHORT = 4800, 600
BANDS = [(0.75, "OPERATOR ESTIMATION DOMINATES — claim stands, now isolated"),
         (0.25, "BOTH CONTRIBUTE — restate as estimation plus state shift"),
         (-np.inf, "STATES DOMINATE — restate: length changes the transition")]


def share(pub, frz, rng=None):
    if rng is not None:
        ids = rng.choice(pub.index, len(pub.index), replace=True)
        pub, frz = pub.loc[ids], frz.loc[ids]
    num = np.log(frz[T_SHORT].median() / frz[T_FULL].median())
    den = np.log(pub[T_SHORT].median() / pub[T_FULL].median())
    return num / den


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--published", required=True)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    a = ap.parse_args()

    p = pd.read_csv(a.published); f = pd.read_csv(a.frozen)
    if "states" in f and set(f.states.unique()) != {"frozen"}:
        sys.exit("the --frozen file was not produced with --freeze-states")
    P = p.pivot_table(index="subject_id", columns="T_effective", values="teleport_ratio")
    F = f.pivot_table(index="subject_id", columns="T_effective", values="teleport_ratio")
    common = P.index.intersection(F.index)
    P, F = P.loc[common], F.loc[common]
    print(f"subjects in both runs: {len(common)} (published {len(p.subject_id.unique())}, "
          f"frozen {len(f.subject_id.unique())})")

    print("\nSANITY GATE — full length must match the published run exactly")
    d = (P[T_FULL] - F[T_FULL]).abs() / P[T_FULL].abs()
    print(f"  max relative difference at T={T_FULL}: {d.max():.2e}")
    if d.max() > 1e-9:
        print("  ** FAIL — the runs differ at full length, where they must agree.")
        print("     Different timeseries, connectomes, estimator settings or code.")
        print("     The comparison is void until this is resolved. **")
        sys.exit(1)
    print("  pass — same inputs, same settings")

    print(f"\n  {'T':>6} {'published':>10} {'frozen':>10}   frozen vs published, paired")
    for T in sorted(P.columns, reverse=True):
        if T not in F.columns:
            continue
        pm, fm = P[T].median(), F[T].median()
        line = f"  {T:>6} {pm:>10.2f} {fm:>10.2f}"
        if T != T_FULL:
            lr = np.log(F[T] / P[T])
            try:
                _, pw = wilcoxon(F[T], P[T])
            except ValueError:
                pw = np.nan
            line += f"   frozen lower in {(F[T] < P[T]).mean():4.0%}, median log-ratio {lr.median():+.3f}, p={pw:.1e}"
        print(line)

    rise_pub = P[T_SHORT] / P[T_FULL]; rise_frz = F[T_SHORT] / F[T_FULL]
    print(f"\n  rise {T_FULL}->{T_SHORT}: published {rise_pub.median():.2f}x "
          f"(rose in {(rise_pub > 1).mean():.0%}), frozen {rise_frz.median():.2f}x "
          f"(rose in {(rise_frz > 1).mean():.0%})")

    S = share(P, F)
    rng = np.random.default_rng(0)
    boot = np.array([share(P, F, rng) for _ in range(a.n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n  SHARE OF THE RISE SURVIVING FROZEN STATES:  S = {S:.3f}  "
          f"95% CI [{lo:.3f}, {hi:.3f}]")
    verdict = next(v for t, v in BANDS if S >= t)
    print(f"  VERDICT (thresholds fixed 2026-09-29): {verdict}")
    straddle = [t for t, _ in BANDS[:2] if lo < t < hi]
    if straddle:
        print(f"  NOTE: the interval straddles {straddle}; report the band AND "
              "that the evidence does not separate it from its neighbour.")


if __name__ == "__main__":
    main()
