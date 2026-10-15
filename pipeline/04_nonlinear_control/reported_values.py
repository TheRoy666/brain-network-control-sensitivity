#!/usr/bin/env python3
"""
reported_values.py — every in-text number of the nonlinear-control results
(Results 2.11, Discussion 3.4, Methods 4.8, Supplementary S4 and S8
"Nonlinear control") recomputed from the deposited files in results/ and
compared with the value printed in the paper.

    python3 pipeline/04_nonlinear_control/reported_values.py [--results results]

Constrained comparison: constrained_dominance_<model>.csv (one solve per
condition). Penalty sweep: matched_cost_<model>.csv (ten penalties x four
initializations per condition; condition-level columns repeat on every run).
A Wilson–Cowan condition is usable when the linear controller's relative
endpoint error is below one.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from reported import Report  # noqa: E402

MODELS = {"fhn": "FitzHugh–Nagumo", "sl": "Stuart–Landau", "wc": "Wilson–Cowan"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    R = lambda p: os.path.join(a.results, "04_nonlinear_control", p)
    rep = Report("04 Nonlinear optimal control")
    C = {m: pd.read_csv(R(f"constrained_dominance_{m}.csv")) for m in MODELS}
    C["wc"] = C["wc"].assign(usable=C["wc"].ltv_err < 1)
    regime = C["sl"].set_index("condition_id").regime
    runs = {m: pd.read_csv(R(f"matched_cost_{m}.csv")) for m in MODELS}
    cond = {m: runs[m].groupby("condition_id").first() for m in MODELS}
    cond["sl"] = cond["sl"].join(regime)

    rep.section("2.11 Constrained comparison (energy at most the linear controller's)")
    use = {"fhn": C["fhn"], "sl": C["sl"], "wc": C["wc"][C["wc"].usable]}
    for m, n in (("fhn", 40), ("sl", 120), ("wc", 20)):
        rep.check(f"{MODELS[m]}: lower-error feasible solution, count", int(use[m].dominates_ltv.sum()), n, digits=0)
        rep.check(f"{MODELS[m]}: conditions", len(use[m]), n, digits=0)
    for m, v in (("fhn", 0.0011), ("sl", 0.0003), ("wc", 0.0515)):
        rep.check(f"{MODELS[m]}: median terminal error relative to linear", (use[m].constrained_err / use[m].ltv_err).median(), v)
    rep.check("FitzHugh–Nagumo: median share of the linear energy used (%)",
              100 * (C["fhn"].constrained_energy / C["fhn"].ltv_energy).median(), 77, digits=0)
    rep.check("Stuart–Landau: median share of the linear energy used (%)", 100 * C["sl"].energy_frac.median(), 30, digits=0)
    allc = pd.concat([use[m] for m in MODELS])
    rep.check("solves in all", len(allc), 180, digits=0)
    rep.check("all feasible", bool(allc.feasible.astype(bool).all()), True)
    rep.check("all terminated successfully", bool(allc.converged.astype(bool).all()), True)
    for m, it in (("fhn", 16), ("sl", 16), ("wc", 156)):
        rep.check(f"4.8 {MODELS[m]}: median iterations", use[m].n_iter.median(), it, digits=0)
    rep.check("4.8 maximum iterations (the 180 included solves)", max(use[m].n_iter.max() for m in MODELS), 412, digits=0)
    viol = max(float(((C[m].constrained_energy - C[m].ltv_energy) / C[m].ltv_energy).max()) for m in MODELS)
    rep.check("4.8 largest constraint violation, fraction of the budget", viol, 5.7e-9, kind="sci")
    u = C["wc"][C["wc"].usable]
    rep.check("constrained peak amplitude, median (Wilson–Cowan, usable)", u.peak_amplitude.median(), 1.2)
    rep.check("constrained peak amplitude, maximum", u.peak_amplitude.max(), 16.5)

    rep.section("2.11 / S4 Penalty sweep (Pareto dominance over the time-varying linear solution)")
    usable = {m: (cond[m][cond[m].ltv_err < 1] if m == "wc" else cond[m]) for m in MODELS}
    for m, n, tot in (("fhn", 36, 40), ("sl", 110, 120), ("wc", 17, 20)):
        rep.check(f"{MODELS[m]}: dominated conditions", int(usable[m].dominated_ltv.astype(bool).sum()), n, digits=0)
        rep.check(f"{MODELS[m]}: of", len(usable[m]), tot, digits=0)
    missed = {m: int((use[m].dominates_ltv.astype(bool).values & ~usable[m].dominated_ltv.astype(bool).reindex(use[m].condition_id).values).sum())
              for m in MODELS}
    for m, n in (("fhn", 4), ("sl", 10), ("wc", 3)):
        rep.check(f"S4 conditions the sweep missed, {MODELS[m]}", missed[m], n, digits=0)
    rep.check("S4 conditions the sweep missed, in all", sum(missed.values()), 17, digits=0)
    for m, le, lq, uq, re_, rlq, ruq, fe, fr in (
            ("fhn", -3.21, -5.35, -1.08, -1.02, -1.54, -0.53, 24.7, 2.8),
            ("sl", -3.32, None, None, -0.80, None, None, 27.6, 2.2),
            ("wc", -0.78, None, None, -0.83, None, None, 2.2, 2.3)):
        d = usable[m][usable[m].dominated_ltv.astype(bool)]
        if m == "sl":   # S4: supercritical magnitudes are quoted at the anchor, not pooled over all six regimes
            d = d[d.regime == "supercritical_ref"]
        lE, lR = np.log(d.dom_energy_ratio_ltv), np.log(d.dom_err_ratio_ltv)
        rep.check(f"{MODELS[m]}: median log energy ratio", lE.median(), le)
        rep.check(f"{MODELS[m]}: median log error ratio", lR.median(), re_)
        rep.check(f"{MODELS[m]}: energy, times less", np.exp(-lE.median()), fe)
        rep.check(f"{MODELS[m]}: error, times lower", np.exp(-lR.median()), fr)
        if lq is not None:
            rep.check(f"{MODELS[m]}: log energy ratio IQR, lower", lE.quantile(.25), lq)
            rep.check(f"{MODELS[m]}: log energy ratio IQR, upper", lE.quantile(.75), uq)
            rep.check(f"{MODELS[m]}: log error ratio IQR, lower", lR.quantile(.25), rlq)
            rep.check(f"{MODELS[m]}: log error ratio IQR, upper", lR.quantile(.75), ruq)
    anchor = cond["sl"][cond["sl"].regime == "supercritical_ref"]
    da = anchor[anchor.dominated_ltv.astype(bool)]
    rep.check("Stuart–Landau, grid matched to FitzHugh–Nagumo (anchor): dominated", int(da.shape[0]), 17, digits=0)
    rep.check("Stuart–Landau anchor: energy, times less (paper text: 36.3)", np.exp(-np.log(da.dom_energy_ratio_ltv).median()), 36.3,
              note="36.3 is the median over all six regimes, which S4 says must not be pooled")
    sub = cond["sl"][cond["sl"].regime.str.startswith("sub")]
    ds = sub[sub.dominated_ltv.astype(bool)]
    rep.check("Stuart–Landau subcritical: dominated", len(ds), 59, digits=0)
    rep.check("Stuart–Landau subcritical: of", len(sub), 60, digits=0)
    rep.check("Stuart–Landau subcritical: median log energy ratio", np.log(ds.dom_energy_ratio_ltv).median(), -4.28)
    rep.check("Stuart–Landau subcritical: energy, times less", np.exp(-np.log(ds.dom_energy_ratio_ltv).median()), 72.5)
    sup = C["sl"][C["sl"].regime.str.startswith("super")]
    med = sup.groupby("regime")[["ltv_err", "ltv_energy"]].median()
    rep.check("S4 supercritical: absolute errors across the three differ by a factor of",
              med.ltv_err.max() / med.ltv_err.min(), 2, digits=0)
    rep.check("S4 supercritical: energies across the three differ by a factor of",
              med.ltv_energy.max() / med.ltv_energy.min(), 4, digits=0)
    sd = cond["sl"][cond["sl"].regime.str.startswith("super")].groupby("regime").dom_energy_ratio_ltv.median()
    rep.check("S4 supercritical: dominance energy ratios identical across the three (relative spread)",
              (sd.max() - sd.min()) / sd.min(), "<=0.07", kind="range", lo=0, hi=0.07)
    for m, v in (("fhn", 0.90), ("sl", 0.85)):
        d = cond[m] if m == "fhn" else anchor
        rep.check(f"S4 dominance rate, {MODELS[m]}", d.dominated_ltv.astype(bool).mean(), v)
    rep.check("S4 dominance rate, Wilson–Cowan usable", usable["wc"].dominated_ltv.astype(bool).mean(), 0.85)
    naive = cond["fhn"][np.isfinite(cond["fhn"].naive_energy)]
    rep.check("frozen-Jacobian solution dominated where computable, FitzHugh–Nagumo", int(naive.dominated_naive.astype(bool).sum()), 25, digits=0)
    rep.check("of", len(naive), 25, digits=0)
    rep.check("Wilson–Cowan dominated at face value", int(cond["wc"].dominated_ltv.astype(bool).sum()), 37, digits=0)
    rep.check("Wilson–Cowan conditions with linear relative error >= 1", int((cond["wc"].ltv_err >= 1).sum()), 20, digits=0)

    rep.section("2.11 / S4 / S8 Sweep diagnostics")
    r = runs["fhn"]
    spread = r[r.lambda_reg == 0].groupby("condition_id").err.agg(lambda x: (x.max() - x.min()) / x.min())
    rep.check("FitzHugh–Nagumo: terminal error spread across initializations, unpenalized, median (%)", 100 * spread.median(), 744, digits=0)
    for m, n in (("fhn", 26), ("wc", 11)):
        best = runs[m].loc[runs[m].groupby(["condition_id", "lambda_reg"]).err.idxmin()]
        nm = sum(not np.all(np.diff(g.sort_values("lambda_reg").energy.values) <= 0) for _, g in best.groupby("condition_id"))
        rep.check(f"{MODELS[m]}: energy-penalty frontier non-monotone (best over initializations), conditions", nm, n, digits=0)
    rep.check("Wilson–Cowan sweep: peak amplitude, maximum", runs["wc"].peak.max(), 4.4e5, kind="sci")
    rep.check("Wilson–Cowan sweep: peak amplitude, median", runs["wc"].peak.median(), 0.91)
    for m, n in (("fhn", 24), ("sl", 29), ("wc", 26)):
        rr = runs[m]
        cnt = ((rr.err < rr.ltv_err) & (rr.energy < rr.ltv_energy)).groupby(rr.condition_id).sum()
        dom = cond[m].dominated_ltv.astype(bool)
        rep.check(f"S4 dominating runs per dominated condition, {MODELS[m]}, median", cnt[dom].median(), n, digits=0,
                  note="Python rounds 24.5 to 24" if m == "fhn" else "")
    rep.check("S4 minimum dominating runs in a dominated condition",
              min(((runs[m].err < runs[m].ltv_err) & (runs[m].energy < runs[m].ltv_energy)).groupby(runs[m].condition_id).sum()[
                  cond[m].dominated_ltv.astype(bool)].min() for m in MODELS), 3, digits=0)
    w = cond["wc"]; f = w.ltv_err >= 1
    rep.check("S4 failing Wilson–Cowan conditions: median target distance", w[f].target_distance.median(), 0.45)
    rep.check("S4 other conditions: median target distance", w[~f].target_distance.median(), 1.32)
    rep.check("S4 failing: median absolute endpoint error", (w.ltv_err * w.target_distance)[f].median(), 1.68)
    rep.check("S4 others: median absolute endpoint error", (w.ltv_err * w.target_distance)[~f].median(), 0.23)
    # The two energies in this sentence were computed over all 40 Wilson–Cowan conditions;
    # the sentence attributes them to the 20 failing ones. Both scopes are printed.
    rr = runs["wc"]
    dom = rr[(rr.err < rr.ltv_err) & (rr.energy < rr.ltv_energy)]
    low = dom.groupby("condition_id").energy.min()
    rep.check("S8 failing conditions: lowest-energy dominating input, median energy", low.reindex(w[f].index).median(), 8.7e-7, kind="sci")
    rep.check("S8 failing conditions: linear method's energy, median", w[f].ltv_energy.median(), 761, digits=0)

    sys.exit(rep.finish())


if __name__ == "__main__":
    main()
