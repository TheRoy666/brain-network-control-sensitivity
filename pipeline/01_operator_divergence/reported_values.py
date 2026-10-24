#!/usr/bin/env python3
"""
reported_values.py — every in-text number of the operator-divergence results
(Results 2.1–2.2, Discussion 3.1 and 3.5, Supplementary S8 "Operator choice")
recomputed from the deposited files in results/, and compared with the value
printed in the paper.

    python3 pipeline/01_operator_divergence/reported_values.py [--results results]

Each line shows the quantity, the value recomputed here, the value in the
paper and OK when the recomputed value rounds to the printed one. Values
that need data not deposited here (HCP per-subject energies, the HCP finite-
versus-infinite horizon factor) are listed at the end with the script that
prints them.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from reported import Report  # noqa: E402

COHORTS = ["HCP", "ADNI-400", "ADNI-450", "epilepsy"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    R = lambda p: os.path.join(a.results, p)
    rep = Report("01 Operator divergence")
    d = {c: pd.read_csv(R(f"01_operator_divergence/directions_{c}.csv")) for c in COHORTS}
    x1 = {c: pd.read_csv(R(f"01_operator_divergence/directions_ext_{c}.csv")) for c in COHORTS}
    x2 = {c: pd.read_csv(R(f"01_operator_divergence/directions_ext2_{c}.csv")) for c in COHORTS}

    # ---- 2.1 FC/EC ratio, both directions
    rep.section("2.1 FC versus EC control energy (high-variance -> quiescent, and reverse)")
    for c, h2q, q2h in (("HCP", 3.22, 4.45), ("ADNI-400", 9.09, 13.48), ("ADNI-450", 7.97, 12.38),
                        ("epilepsy", 65.4, 66.0)):
        rep.check(f"{c} median FC/EC, h->q", d[c].ratio_h2q.median(), h2q)
        rep.check(f"{c} median FC/EC, q->h", d[c].ratio_q2h.median(), q2h)
    allh = pd.concat([d[c].ratio_h2q for c in COHORTS])
    rep.check("every participant FC/EC > 1 (h->q), fraction", (allh > 1).mean(), 1.0, digits=2)
    rep.check("range across participants, min (h->q)", allh.min(), 1.7)
    rep.check("range across participants, max (h->q)", allh.max(), 189, digits=0)

    # ---- symmetrized EC at the same spectral radius (0.9)
    rep.section("2.1 Symmetrized EC, (A + A')/2 at the same spectral radius")
    for c, fold in (("HCP", 2.56), ("ADNI-400", 8.48), ("ADNI-450", 7.72), ("epilepsy", 142)):
        s = x1[c].sym_over_ec_h2q_r090
        rep.check(f"{c} median energy increase, symmetrized/EC", s.median(), fold, digits=2 if fold < 100 else 0)
        rep.check(f"{c} increase in every participant, fraction", (s > 1).mean(), 1.0, digits=2)
    for c in ("ADNI-400", "ADNI-450"):
        sh = 100 * np.median(np.log(x1[c].sym_over_ec_h2q_r090) / np.log(x1[c].fc_over_ec_h2q_r090))
        rep.check(f"{c} share of log FC/EC divergence reproduced (paper: 97-98% in ADNI)", sh, "97-98", kind="range", lo=96.5, hi=98.5)
    for c, share, col in (("HCP", 81, "h2q"), ("HCP", 88, "q2h")):
        sym = x1[c][f"sym_over_ec_{col}_r090"]; fc = x1[c][f"fc_over_ec_{col}_r090"]
        rep.check(f"{c} share of log FC/EC divergence reproduced, {col} (%)",
                  100 * np.median(np.log(sym) / np.log(fc)), share, digits=0,
                  note="median over subjects of log(sym/EC) / log(FC/EC)")
    rep.check("epilepsy symmetrization exceeds the divergence (share > 100%)",
              100 * np.median(np.log(x1["epilepsy"].sym_over_ec_h2q_r090) / np.log(x1["epilepsy"].fc_over_ec_h2q_r090)),
              ">100", kind="gt", ref=100)

    rep.section("S8 Symmetrizing the normalized estimate at fixed scale")
    for c, rho in (("HCP", 1.5),):
        rep.check(f"{c} median spectral radius after fixed-scale symmetrization", x2[c].rho_symfixed_r090.median(), rho, digits=1)
    for c in ("ADNI-400", "ADNI-450", "epilepsy"):
        rep.check(f"{c} median spectral radius after fixed-scale symmetrization (4.0-5.0)",
                  x2[c].rho_symfixed_r090.median(), "4.0-5.0", kind="range", lo=3.95, hi=5.05)
    for c in COHORTS:
        rep.check(f"{c} fixed-scale symmetrization raises energy, fraction",
                  (x2[c].symfixed_over_ec_h2q_r090 > 1).mean(), 1.0, digits=2)
        rep.check(f"{c} every symmetric matrix unstable (rho > 1), fraction", (x2[c].rho_symfixed_r090 > 1).mean(), 1.0, digits=2)

    # ---- normalization
    rep.section("2.1 / S8 Spectral radius 0.80 and 0.95")
    for c, lo, hi in (("HCP", 2.66, 3.47), ("ADNI-400", 6.43, 10.81), ("ADNI-450", 5.64, 9.37), ("epilepsy", 33.1, 163.7)):
        rep.check(f"{c} median FC/EC (h->q) at radius 0.80", x1[c].fc_over_ec_h2q_r080.median(), lo)
        rep.check(f"{c} median FC/EC (h->q) at radius 0.95", x1[c].fc_over_ec_h2q_r095.median(), hi)
        rep.check(f"{c} every ratio > 1 at 0.80 and 0.95, fraction",
                  np.mean([(x1[c][f"fc_over_ec_{dd}_r{r}"] > 1).mean() for r in ("080", "095") for dd in ("h2q", "q2h")]), 1.0, digits=2)
    order = lambda r, dd: list(pd.Series({c: x1[c][f"fc_over_ec_{dd}_r{r}"].median() for c in COHORTS}).sort_values().index)
    rep.check("order of cohorts unchanged across radii and directions",
              len({tuple(order(r, dd)) for r in ("080", "090", "095") for dd in ("h2q", "q2h")}) == 1, True)
    h = x1["HCP"].fc_over_ec_h2q_r095.median() / x1["HCP"].fc_over_ec_h2q_r080.median()
    e = x1["epilepsy"].fc_over_ec_h2q_r095.median() / x1["epilepsy"].fc_over_ec_h2q_r080.median()
    rep.check("3.1 HCP: ratio grows ~30% from 0.80 to 0.95 (%)", 100 * (h - 1), 30, digits=-1, note="about")
    rep.check("3.1 epilepsy: ratio grows ~fivefold from 0.80 to 0.95", e, 5, digits=0, note="about")

    rep.section("3.5 Energy of a move and of its reverse (EC)")
    for c in ("HCP", "ADNI-400", "ADNI-450"):
        rep.check(f"{c} median EC energy, h->q over q->h (about 1.5-fold)", (d[c].E_ec_h2q / d[c].E_ec_q2h).median(),
                  "~1.5", kind="range", lo=1.4, hi=1.7)
    rep.check("epilepsy median EC energy, h->q over q->h (barely differs)",
              (d["epilepsy"].E_ec_h2q / d["epilepsy"].E_ec_q2h).median(), "~1", kind="range", lo=0.9, hi=1.15)

    # ---- in- versus out-strength of the EC estimate
    rep.section("2.1 Incoming versus outgoing strength (manjunatha_hub_asymmetry_check.py)")
    io = {c: pd.read_csv(R(f"01_operator_divergence/in_out_strength_{c}.csv")) for c in COHORTS}
    for c in COHORTS:
        rep.check(f"{c}: subjects in the check", len(io[c]), 8, digits=0, note="a subsample, not the full cohort")
    for c in ("HCP", "epilepsy"):
        r = io[c].spearman_rho_in_vs_out_strength
        rep.check(f"{c}: in- and out-strength inversely related (median Spearman < 0)", r.median() < 0, True,
                  note=f"median rho {r.median():+.2f}; negative in {(r < 0).sum()} of {len(r)}")
    for c in ("ADNI-400", "ADNI-450"):
        r = io[c].spearman_rho_in_vs_out_strength
        rep.check(f"{c}: no inverse relation (no subject with rho < 0)", bool((r < 0).sum() == 0), True,
                  note=f"median rho {r.median():+.2f}")

    # ---- finite versus infinite horizon (deposited for ADNI)
    rep.section("2.1 Finite- versus infinite-horizon energy (ADNI)")
    for p, v in (("400", 32.2), ("450", 32.8)):     # the paper gives "32.2 and 32.8 in the two ADNI parcellations"
        f = pd.read_csv(R(f"05_clinical/adni_finite_vs_infinite_{p}.csv"))
        rep.check(f"ADNI-{p} median finite/infinite at T = 1", f[f["T"] == 1].ratio.median(), v)
        rep.check(f"ADNI-{p} finite > infinite at T = 1 in every subject", (f[f["T"] == 1].ratio > 1).mean(), 1.0, digits=2)
        m20 = f[f["T"] == 20].ratio.median()
        rep.check(f"ADNI-{p} median finite/infinite at T = 20 (converged, ~1)", m20, "~1", kind="range", lo=0.9, hi=1.2)

    # ---- 2.2 truncation
    rep.section("2.2 Recording length (HCP truncation, states recomputed and held fixed)")
    v = pd.read_csv(R("01_operator_divergence/operator_divergence_vs_length_HCP.csv"))
    v = v[v.stride == 1]
    full = v[v.T_effective == 4800].set_index("subject_id").teleport_ratio
    t600 = v[v.T_effective == 600].set_index("subject_id").teleport_ratio
    rep.check("HCP median FC/EC at 4800", full.median(), 3.22)
    rep.check("HCP median FC/EC at 600", t600.median(), 15.04)
    rep.check("increase in every participant, fraction", (t600 / full.reindex(t600.index) > 1).mean(), 1.0, digits=2)
    fz = pd.read_csv(R("01_operator_divergence/operator_divergence_frozen_states_HCP.csv"))
    F = fz.pivot(index="subject_id", columns="T_effective", values="teleport_ratio")   # states held at full length
    P = v.pivot(index="subject_id", columns="T_effective", values="teleport_ratio").reindex(F.index)
    rep.check("HCP median FC/EC at 600, states held fixed", F[600].median(), 9.32)
    rep.check("fixed: increase in every participant, fraction", (F[600] > F[4800]).mean(), 1.0, digits=2)
    rep.check("share of the log-scale rise preserved with states fixed (%)",
              100 * np.log(F[600].median() / F[4800].median()) / np.log(P[600].median() / P[4800].median()), 69, digits=0,
              note="as compare_frozen_vs_published.py; its bootstrap gives the 66-72% interval")
    rep.check("S8 median fold change, states fixed", (F[600] / F[4800]).median(), 2.88)
    rep.check("S8 median fold change, states recomputed", (P[600] / P[4800]).median(), 4.75)
    rep.check("S8 rise with states fixed, Wilcoxon p (600 against 4800)", stats.wilcoxon(F[600], F[4800]).pvalue, 8.5e-41, kind="sci")
    rep.check("S8 rise smaller with states fixed, subjects", int((F[600] / F[4800] < P[600] / P[4800]).sum()), 226, digits=0)
    rep.check("S8 fold change fixed against recomputed, paired Wilcoxon p", stats.wilcoxon(F[600] / F[4800], P[600] / P[4800]).pvalue,
              1.4e-39, kind="sci")
    share = np.log(F[600] / F[4800]) / np.log(P[600] / P[4800])
    rep.check("S8 more than half of the rise preserved, fraction of subjects (%)", 100 * (share > 0.5).mean(), 90, digits=0)
    st = pd.read_csv(R("01_operator_divergence/operator_divergence_stride_HCP.csv"))
    st200 = st[st.stride == 24]
    rep.check("subsampled to 200 (every 24th timepoint), median FC/EC", st200.teleport_ratio.median(), 11.92)
    asym = v.groupby("T_effective").ec_asymmetry.median()
    rep.check("S8 normalized EC asymmetry, median at 4800", asym.loc[4800], 1.28)
    rep.check("S8 normalized EC asymmetry, median at 600", asym.loc[600], 1.41)

    rep.section("2.2 Within ADNI (400 regions): truncation and scanner manufacturer")
    av = pd.read_csv(R("01_operator_divergence/operator_divergence_vs_length_ADNI-400.csv"))
    av = av[av.stride == 1]
    w = av.pivot_table(index="subject_id", columns="requested_length", values="teleport_ratio")
    cols = sorted(w.columns)          # requested lengths, e.g. 100, 150, native
    nat, mid, low = cols[-1], 150, 100
    rep.check("ratio rises from native to 150 timepoints, % of participants", 100 * (w[mid] > w[nat]).mean(), 98, digits=0)
    rep.check("ratio falls from 150 to 100 timepoints, % of participants", 100 * (w[low] < w[mid]).mean(), 94, digits=0)
    rep.check("participants with their maximum at 150 timepoints (%)", 100 * (w.idxmax(axis=1) == mid).mean(), 92, digits=0)
    res = pd.read_csv(R("05_clinical/adni_results_400.csv"))[["subject_id", "manufacturer"]]
    dm = d["ADNI-400"].merge(res, on="subject_id")
    for m, val in (("GE", 13.69), ("Siemens", 9.09), ("Philips", 7.78)):
        rep.check(f"native length, median FC/EC, {m}", dm[dm.manufacturer == m].ratio_h2q.median(), val)
    avm = av.merge(res, on="subject_id")
    orders = {L: tuple(avm[avm.requested_length == L].groupby("manufacturer").teleport_ratio.median().sort_values().index)
              for L in cols}
    rep.check("same manufacturer order at every truncation", len(set(orders.values())) == 1, True)
    q = pd.read_csv(R("cohort/adni_qc.csv")).merge(res, left_on="subject", right_on="subject_id")
    rep.check("GE recordings are the 200-timepoint ones", set(q[q.func_ntimepoints == 200].manufacturer) == {"GE"}, True)
    ts = q.groupby("manufacturer").func_tsnr.median()
    rep.check("tSNR ratio, Siemens/Philips median over GE (about five-fold)", ts[["Siemens", "Philips"]].min() / ts["GE"], 5, digits=0)

    rep.server_only([
        ("HCP finite/infinite at T = 1, median 11.7", "pipeline/01_operator_divergence/run_hcp_aud_batch.py (HCP per-subject results are not deposited; see docs/DATA_ACCESS.md)"),
    ])
    synthetic_chains(rep)
    sys.exit(rep.finish())


def synthetic_chains(rep, n_seeds=50):
    """Results 2.1 and Methods 4.6: simulated feedforward chains processed as the cohorts were."""
    import synthetic_chain_operator_check as SC
    rep.section("2.1 / 4.6 Simulated feedforward chains (synthetic_chain_operator_check.py, high-variance to quiescent)")
    r = {}
    for n in SC.LENGTHS:
        for seed in range(n_seeds):
            X, _, ops = SC.operators(n, seed)
            q, h = SC.states(X)
            for k in ("functional", "symmetrized"):
                for horizon in (SC.HORIZON, 3):
                    r.setdefault((n, k, horizon), []).append(SC.ratio(ops[k], ops["directed"], h, q, horizon))
    med = {key: float(np.median(v)) for key, v in r.items()}
    for k, lo, hi in (("functional", 2.28, 5.51), ("symmetrized", 1.25, 3.61)):
        rep.check(f"{k}: lowest median ratio over chain lengths", min(med[(n, k, SC.HORIZON)] for n in SC.LENGTHS), lo)
        rep.check(f"{k}: highest median ratio over chain lengths", max(med[(n, k, SC.HORIZON)] for n in SC.LENGTHS), hi)
    rep.check("ratio above one in every recording of 10 and 20 regions, both operators",
              all(min(r[(n, k, SC.HORIZON)]) > 1 for n in (10, 20) for k in ("functional", "symmetrized")), True)
    rep.check("three-step horizon: median ratio below one at 5 and 10 regions, both operators",
              all(med[(n, k, 3)] < 1 for n in (5, 10) for k in ("functional", "symmetrized")), True)


if __name__ == "__main__":
    main()
