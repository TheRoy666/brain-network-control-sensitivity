#!/usr/bin/env python3
"""
reported_values.py — every in-text number of the linearization results
(Results 2.8–2.10, Discussion 3.3, Methods 4.7, Supplementary S2, S6 and S8
"Linearization") recomputed from the deposited sweeps in results/ and
compared with the value printed in the paper. The critical displacement and
its interval use the functions of localize_critical_displacement.py, so the
estimator is the one the paper reports.

    python3 pipeline/03_linearization/reported_values.py [--results results] [--n-boot 2000]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import localize_critical_displacement as L  # noqa: E402
from error_tolerance_threshold import tolerance_thresholds  # noqa: E402
from reported import Report  # noqa: E402

ARMS = ["HCP", "ADNI-400", "ADNI-450"]


def fractions(flags):
    ts = sorted(flags)
    return ts, [float(np.mean(list(flags[t].values()))) for t in ts]


def crossing(flags):
    ts, fr = fractions(flags)
    return L.crossing_point(ts, fr), L.n_crossings(ts, fr)


def first_descent(ts, fr, c=0.5):
    x, y = np.log(np.asarray(ts)), np.asarray(fr)
    k = next(k for k in range(len(x) - 1) if y[k] >= c > y[k + 1])
    return float(np.exp(x[k] + (y[k] - c) / (y[k] - y[k + 1]) * (x[k + 1] - x[k])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    R = lambda p: os.path.join(a.results, "03_linearization", p)
    rep = Report("03 Validity of the linearization")
    S = {k: pd.read_csv(R(f"fine_sweep/wc_sweep_finets_{k}.csv.gz")) for k in ARMS}
    allr = pd.concat(S.values())

    rep.section("4.7 The sweep")
    rep.check("conditions in all", len(allr), 327405, digits=0)
    for k, n in (("HCP", 232050), ("ADNI-400", 47775), ("ADNI-450", 47580)):
        rep.check(f"{k} conditions", len(S[k]), n, digits=0)
    for k, hi in (("HCP", 4.361), ("ADNI-400", 3.969), ("ADNI-450", 3.675)):
        rep.check(f"{k} coupling grid, highest", S[k].w_EE.max(), hi)
        rep.check(f"{k} coupling grid, lowest", S[k].w_EE.min(), 3.0)
    rep.check("displacements", S["HCP"].target_scale.nunique(), 13, digits=0)
    rep.check("horizons", S["HCP"]["T"].nunique(), 5, digits=0)
    rep.check("ADNI-450 subjects lacking the highest coupling",
              int((S["ADNI-450"].groupby("subject_id").w_EE.nunique() < 5).sum()), 1, digits=0)

    rep.section("2.8 / 3.3 Error magnitude")
    e = allr.rel_reachability_error
    rep.check("median relative endpoint error (%)", 100 * e.median(), 3.4)
    rep.check("conditions above 10% error (%)", 100 * (e > 0.1).mean(), 4.1)
    rep.check("conditions above 20% error", int((e > 0.2).sum()), 0, digits=0)
    rep.check("S8 largest relative error", e.max(), 0.198)
    rep.check("median error at the largest displacement (%), HCP", 100 * S["HCP"][S["HCP"].target_scale == 0.1].rel_reachability_error.median(), 6.5,
              note=f"all three arms pooled: {100 * allr[allr.target_scale == 0.1].rel_reachability_error.median():.2f}")

    rep.section("2.8 Critical displacement (last descent through 0.5, 2000 subject resamples)")
    flags = {k: L.interior_flags(S[k]) for k in ARMS}
    for k, v, lo, hi in (("HCP", 0.02934, 0.02862, 0.03020), ("ADNI-400", 0.02333, 0.02236, 0.02513)):
        c, nd = crossing(flags[k])
        rep.check(f"{k} critical displacement", c, v)
        rep.check(f"{k} descents through 0.5", nd, 1, digits=0)
        ci = L.bootstrap_crossing(flags[k], n_boot=a.n_boot)
        # the interval depends on the order subjects are drawn in; pseudonymization reorders them
        rep.check(f"{k} 95% CI lower", ci["lo"], lo, kind="range", lo=lo - 4e-5, hi=lo + 4e-5, note="within 4e-5: resample order")
        rep.check(f"{k} 95% CI upper", ci["hi"], hi, kind="range", lo=hi - 4e-5, hi=hi + 4e-5, note="within 4e-5: resample order")
    ts, fr = fractions(flags["ADNI-450"])
    rep.check("ADNI-450 descents through 0.5", L.n_crossings(ts, fr), 2, digits=0)
    rep.check("S6 ADNI-450 first descent", first_descent(ts, fr), 0.01387)
    rep.check("S6 ADNI-450 last descent", L.crossing_point(ts, fr), 0.02256)
    rep.check("S6 first vs last descent differ (%)", 100 * (L.crossing_point(ts, fr) - first_descent(ts, fr)) / L.crossing_point(ts, fr), 38.5,
              note="relative to the last descent")
    c4 = crossing(flags["ADNI-400"])[0]
    rep.check("S6 the two ADNI parcellations agree under the last-descent rule (%)", 100 * abs(c4 - L.crossing_point(ts, fr)) / c4, 3.3)
    near = []
    for k in ("HCP", "ADNI-400"):
        c = crossing(flags[k])[0]
        t = min(S[k].target_scale.unique(), key=lambda t: abs(np.log(t / c)))
        near.append(S[k][S[k].target_scale == t].rel_reachability_error)
    rep.check("median error at the transition (%), grid displacement nearest each cohort's, HCP and ADNI-400",
              100 * pd.concat(near).median(), 2.4)

    rep.section("S6 Seeding: the first sweep against the rerun at the five shared displacements")
    dfr, dcr = [], []
    for k in ARMS:
        cs = L.interior_flags(pd.read_csv(R(f"coarse_sweep/wc_sweep_{k}.csv.gz")))
        shared = sorted(set(cs) & set(flags[k]))
        f1 = [np.mean(list(cs[t].values())) for t in shared]
        f2 = [np.mean(list(flags[k][t].values())) for t in shared]
        dfr.append(max(abs(x - y) for x, y in zip(f1, f2)))
        c1, c2 = L.crossing_point(shared, f1), L.crossing_point(shared, f2)
        dcr.append(100 * abs(c1 - c2) / c2)
    rep.check("interior-minimum fraction moved by up to", max(dfr), 0.082)
    rep.check("critical displacement agreed to within (%)", max(dcr), 3.0)

    rep.section("S8 Coupling dependence and error tolerance")
    hs = S["HCP"].groupby(["subject_id", "w_EE"]).rel_reachability_error.mean().groupby("w_EE").median()
    rep.check("HCP cohort median error, lowest coupling", hs.iloc[0], 0.0428)
    rep.check("HCP cohort median error, highest coupling", hs.iloc[-1], 0.0356)
    rep.check("falls monotonically with coupling", bool(np.all(np.diff(hs.values) < 0)), True)
    tol = {k: tolerance_thresholds(S[k]) for k in ("HCP", "ADNI-400")}
    for k in ("HCP", "ADNI-400"):
        rep.check(f"{k}: no 10% breach at best, worst or mean coupling",
                  all(np.isnan(tol[k][(c, 0.10)]) for c in ("best", "mean", "worst")), True)
    vals = [tol[k][(c, 0.05)] for k in tol for c in ("best", "mean", "worst")]
    rep.check("5% tolerance crossed from (lowest across cohort and coupling)", min(vals), 0.055)
    rep.check("5% tolerance crossed up to (highest)", max(vals), 0.078)
    rep.check("5% tolerance, cohort difference at mean coupling (%)",
              100 * abs(tol["HCP"][("mean", 0.05)] - tol["ADNI-400"][("mean", 0.05)]) / tol["ADNI-400"][("mean", 0.05)], 0.9,
              note="error_tolerance_threshold.py: 0.0648 against 0.0642")
    rep.check("5% tolerance, worst-case coupling not resolved (within one grid interval)",
              abs(tol["HCP"][("worst", 0.05)] - tol["ADNI-400"][("worst", 0.05)]) < 0.01, True)

    med = {k: S[k].groupby("target_scale").rel_reachability_error.median() for k in ARMS}
    rep.check("S4 connectome sweep: median error at 0.005, lowest arm (%)", 100 * min(m.loc[0.005] for m in med.values()), 2.5)
    rep.check("S4 connectome sweep: median error at 0.005, highest arm (%)", 100 * max(m.loc[0.005] for m in med.values()), 3.0)
    rep.check("S4 connectome sweep: median error at 0.01, every arm (%)", sorted({round(100 * m.loc[0.01], 1) for m in med.values()}) == [1.8], True)

    rep.section("2.9 Numerical error")
    nx = pd.concat([pd.read_csv(R(f"linearization_vs_numerical_error_{k}.csv")) for k in ARMS])
    rep.check("conditions", len(nx), 58500, digits=0)
    rep.check("participants (20 per arm; the ADNI arms share subjects)",
              sum(pd.read_csv(R(f"linearization_vs_numerical_error_{k}.csv")).subject_id.nunique() for k in ARMS), 60, digits=0)
    rep.check("median error on the linear system", nx.err_linear.median(), 8.0e-15, kind="sci")
    rep.check("maximum error on the linear system", nx.err_linear.max(), 4.2e-14, kind="sci")
    rep.check("largest Gramian condition number", nx.gramian_cond.max(), 5577, digits=0)
    rep.check("pseudoinverse threshold 1e-8 vs 1e-12 bit-identical, fraction",
              (nx.err_linear_rcond_1e8 == nx.err_linear_rcond_1e12).mean(), 1.0, digits=2)
    rep.check("S8 largest numerical share of the total error", (nx.err_linear / nx.err_nonlinear).max(), 4.2e-12, kind="sci")

    rep.section("2.9 / S8 Non-normality")
    thr = {k: None for k in ARMS}
    for k in ARMS:
        g = S[k].groupby(["subject_id", "target_scale", "w_EE"]).rel_reachability_error.mean().reset_index()
        out = {}
        for sid, s in g.groupby("subject_id"):
            fl = []
            for t, s2 in s.groupby("target_scale"):
                i = int(np.argmin(s2.sort_values("w_EE").rel_reachability_error.values))
                fl.append((t, 1.0 if 0 < i < len(s2) - 1 else 0.0))
            fl.sort()
            x, y = np.log([f[0] for f in fl]), np.array([f[1] for f in fl])
            idx = [j for j in range(len(x) - 1) if y[j] >= 0.5 > y[j + 1]]
            out[str(sid)] = float(np.exp(x[idx[-1]])) if idx else np.nan
        thr[k] = pd.Series(out, name="thr")
    nn = {k: pd.read_csv(R(f"non_normality_{k}.csv")) for k in ARMS}
    # measured at one operating point, the highest coupling (Results 2.9: "at one operating point under one norm")
    at_top = lambda k: nn[k][nn[k].w_EE == nn[k].w_EE.max()].set_index(nn[k][nn[k].w_EE == nn[k].w_EE.max()].subject_id.astype(str))
    subj = at_top("HCP").join(thr["HCP"]).dropna(subset=["thr"])
    for m, r, p in (("J_henrici", -0.057, 0.39), ("J_transient_amp", -0.053, 0.42), ("J_numerical_abscissa", -0.057, 0.38)):
        rr, pp = stats.spearmanr(subj[m], subj.thr)
        rep.check(f"HCP {m} vs subject transition: rho", rr, r)
        rep.check(f"HCP {m} vs subject transition: p", pp, p)
    sub4 = at_top("ADNI-400").join(thr["ADNI-400"]).dropna(subset=["thr"])
    rep.check("ADNI-400: all three non-significant",
              all(stats.spearmanr(sub4[m], sub4.thr)[1] > 0.05 for m in ("J_henrici", "J_transient_amp", "J_numerical_abscissa")), True)
    h = nn["HCP"]; top = h[h.w_EE == h.w_EE.max()]
    rep.check("numerical abscissa negative at every coupling below the highest (HCP), fraction",
              (h[h.w_EE < h.w_EE.max()].J_numerical_abscissa < 0).mean(), 1.0, digits=2)
    rep.check("positive at the highest coupling in all HCP subjects (count)", int((top.J_numerical_abscissa > 0).sum()), 238, digits=0)
    rep.check("HCP amplification at the highest coupling, median", top.J_transient_amp.median(), 1.36)
    rep.check("HCP amplification at the highest coupling, max", top.J_transient_amp.max(), 1.78)
    pos = [100 * (nn[k].J_numerical_abscissa > 0).mean() for k in ARMS]
    rep.check("abscissa positive across cohorts, lowest share (%)", min(pos), 20, digits=0)
    rep.check("abscissa positive across cohorts, highest share (%)", max(pos), 39, digits=0)
    rep.check("amplification across cohorts, max", max(nn[k].J_transient_amp.max() for k in ARMS), 2.90)
    for k in ARMS:
        b = nn[k][nn[k].w_EE == nn[k].w_EE.max()].set_index("subject_id")[["J_transient_amp"]]
        e = S[k][S[k].w_EE == S[k].w_EE.max()].groupby("subject_id").rel_reachability_error.mean()
        j = b.join(e.rename("e")).dropna()
        rr, pp = stats.spearmanr(j.J_transient_amp, j.e)
        if k == "HCP":
            rep.check("HCP boundary non-normality vs endpoint error: rho", rr, -0.320)
            rep.check("HCP boundary non-normality vs endpoint error: p", pp, 4.7e-7, kind="sci")
        else:
            rep.check(f"{k} same correlation negative", rr < 0, True)

    rep.section("2.10 / S2 Degree-preserving null")
    nulls = {k: L.interior_flags(pd.read_csv(R(f"fine_sweep/wc_null_finets_{k}.csv.gz"))) for k in ("HCP", "ADNI-400")}
    real = {k: crossing(flags[k])[0] for k in ("HCP", "ADNI-400")}
    null = {k: crossing(nulls[k])[0] for k in ("HCP", "ADNI-400")}
    rep.check("ADNI-400 rewired critical displacement", null["ADNI-400"], 0.02351)
    rep.check("HCP rewired critical displacement", null["HCP"], 0.02735)
    rep.check("ADNI-400 shift (%)", 100 * abs(null["ADNI-400"] - real["ADNI-400"]) / real["ADNI-400"], 0.8)
    rep.check("HCP shift (%)", 100 * abs(null["HCP"] - real["HCP"]) / real["HCP"], 6.8)
    dr, dn = real["HCP"] - real["ADNI-400"], null["HCP"] - null["ADNI-400"]
    rep.check("cohort difference, real", dr, 0.00601)
    rep.check("cohort difference, rewired", dn, 0.00384)
    rep.check("share surviving (%)", 100 * dn / dr, 64, digits=0)
    rep.check("rewired HCP about twice as close to real HCP as to real ADNI-400",
              abs(null["HCP"] - real["ADNI-400"]) / abs(null["HCP"] - real["HCP"]), 2, digits=0)
    dg = {k: pd.read_csv(R(f"fine_sweep/wc_null_finets_{k}_diagnostics.csv")) for k in ARMS}
    rep.check("ADNI-400 rewiring time, median (s)", dg["ADNI-400"].elapsed_s.median(), 1.37)
    rep.check("ADNI-450 rewiring time, median (s)", dg["ADNI-450"].elapsed_s.median(), 1.83)
    rep.check("HCP rewiring time, median (s)", dg["HCP"].elapsed_s.median(), 85.3)
    rep.check("HCP over ADNI-400 rewiring time (fold)", dg["HCP"].elapsed_s.median() / dg["ADNI-400"].elapsed_s.median(), 62, digits=0)
    inc = sorted(dg["HCP"][dg["HCP"].fraction_of_target < 1].fraction_of_target)
    rep.check("HCP subjects reaching the attempts ceiling", len(inc), 2, digits=0)
    rep.check("their share of target swaps, lower (%)", 100 * inc[0], 86.0)
    rep.check("their share of target swaps, upper (%)", 100 * inc[1], 92.9)
    rep.check("no subject hit the time budget", not any(dg[k].hit_time_budget.astype(bool).any() for k in ARMS), True)
    for k in ("ADNI-400", "ADNI-450"):
        rep.check(f"{k} rewired completely", bool((dg[k].fraction_of_target >= 1).all()), True)
    topo = {k: pd.read_csv(R(f"network_topology_{k}.csv")) for k in ARMS}
    for k, v in (("HCP", 0.805), ("ADNI-400", 0.501), ("ADNI-450", 0.541)):
        rep.check(f"{k} node-strength CV, mean", topo[k].strength_cv.mean(), v)
    for k, v in (("HCP", 2.525), ("ADNI-400", 1.883), ("ADNI-450", 2.171)):
        rep.check(f"{k} normalized rich club, top 10%, mean", topo[k].phi_norm_top10pct.mean(), v)
    rep.check("HCP higher at every threshold from 5 to 30%",
              all(topo["HCP"][f"phi_norm_top{p}pct"].mean() > max(topo[k][f"phi_norm_top{p}pct"].mean() for k in ("ADNI-400", "ADNI-450"))
                  for p in (5, 10, 15, 20, 25, 30)), True)

    rep.server_only([
        ("S6 HCP stability boundary 4.400 (spread 0.100) on two disjoint samples of twenty",
         "pipeline/03_linearization/real_coupling_boundary_recheck.py (needs the connectomes)"),
        ("S6 ADNI-450 subject: margin 0.008 at 3.597, no fixed point from 3.610",
         "pipeline/03_linearization/check_subject_coupling.py (needs the connectome)"),
        ("S2 combining-form search: no candidate beats the majority-class baseline",
         "pipeline/03_linearization/mechanism_search_v2.py (reads the fine sweep and the per-subject energies)"),
    ])
    sys.exit(rep.finish())


if __name__ == "__main__":
    main()
