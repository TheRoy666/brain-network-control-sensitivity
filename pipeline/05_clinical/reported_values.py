#!/usr/bin/env python3
"""
reported_values.py — every in-text number of the exploratory clinical
contrasts (Results 2.12, Supplementary S1 and S3, Discussion 3.6)
recomputed from the deposited files in results/ and compared with the value
printed in the paper.

    python3 pipeline/05_clinical/reported_values.py [--results results]

The minimum detectable effects use statsmodels' two-sample t-test power
solver at alpha 0.05 and power 0.8 with the realized group sizes.
"""
import argparse
import os
import sys

import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from reported import Report  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    R = lambda p: os.path.join(a.results, "05_clinical", p)
    rep = Report("05 Exploratory clinical contrasts")

    rep.section("S1 Alcohol-use severity in HCP (group level; per-subject HCP results are not deposited)")
    h = pd.read_csv(R("hcp_aud_group_comparison.csv")).set_index("Severity_Label")
    for g, v in (("Social_Drinker", 0.282), ("Abuser", 0.235), ("Dependent", 0.284)):
        rep.check(f"{g}: median energy, craving -> rest (proxy reward network)", h.loc[g, "E_craving_to_rest_median"], v)
    for g, n in (("Social_Drinker", 196), ("Abuser", 26), ("Dependent", 16)):
        rep.check(f"{g}: n", int(h.loc[g, "N"]), n, digits=0)

    rep.section("S1 Cognitively normal versus MCI in ADNI")
    for p in ("400", "450"):
        hc = pd.read_csv(R(f"adni_harmony_comparison_{p}.csv")).set_index("metric")
        if p == "400":
            rep.check("smallest p before harmonization", hc.before_p.min(), 0.061)
            rep.check("on the demand -> rest energy", hc.before_p.idxmin() == "E_demand_to_rest", True)
            rep.check("Cliff's delta there", hc.loc["E_demand_to_rest", "before_delta"], 0.313)
            rest = hc.before_p.drop(hc.before_p.idxmin())
            rep.check("remaining five p-values, lowest", rest.min(), 0.18)
            rep.check("remaining five p-values, highest", rest.max(), 0.68)
            rep.check("smallest p after harmonization", hc.after_p.min(), 0.080)
            rep.check("metrics improved by harmonization", int(hc.p_improved.astype(bool).sum()), 1, digits=0)
        else:
            rep.check("ADNI-450 smallest p", hc.before_p.min(), 0.18)
        se = pd.read_csv(R(f"adni_site_effects_{p}.csv"))
        sig = se[se.significant.astype(bool)]
        if p == "400":
            rep.check("metrics with a significant site effect", len(sig), 4, digits=0)
            rep.check("eta-squared, lowest significant", sig.eta2.min(), 0.157, note="0.1564 rounds to 0.156")
            rep.check("eta-squared, highest", sig.eta2.max(), 0.251)
        else:
            rep.check("ADNI-450 eta-squared, highest", se.eta2.max(), 0.317)
    g4 = pd.read_csv(R("adni_results_400.csv"))
    rep.check("ADNI cognitively normal", int((g4.DX_Group == "CN").sum()), 25, digits=0)
    rep.check("ADNI mild cognitive impairment", int((g4.DX_Group == "MCI").sum()), 24, digits=0)

    rep.section("S1 Seizure-focus lateralization (pre-registered)")
    s = pd.read_csv(R("unam_lateralization_summary.csv")).set_index("test")
    pr = s.loc["primary_MTL_classification"]
    rep.check("correct", int(pr.correct), 13, digits=0)
    rep.check("patients", int(pr.n), 28, digits=0)
    rep.check("accuracy (%)", 100 * pr.accuracy, 46.4)
    rep.check("binomial p against 0.5", pr.p_value, 0.851)
    k = pd.read_csv(R("unam_lateralization_classification.csv"))
    rep.check("left-focus patients", int((k.TLEside == "Left-TLE").sum()), 17, digits=0)
    rep.check("right-focus patients", int((k.TLEside == "Right-TLE").sum()), 11, digits=0)
    rep.check("majority-class rate (%)", 100 * 17 / 28, 60.7)
    rep.check("against the majority rate: not significant", stats.binomtest(13, 28, 17 / 28).pvalue > 0.05, True)
    u = pd.read_csv(R("unam_results.csv"))
    p_sex = stats.mannwhitneyu(u[u.gender == "F"].ec_asymmetry, u[u.gender == "M"].ec_asymmetry).pvalue
    rep.check("covariate check: sex vs EC asymmetry survives Bonferroni over three covariates", p_sex < 0.05 / 3, True,
              note=f"Mann-Whitney p = {p_sex:.4f}")
    for g, acc, p in (("F", 40.0, 0.503), ("M", 62.5, 0.727)):
        d = k[k.gender == g]
        rep.check(f"stratum {g}: accuracy (%)", 100 * d.correct.mean(), acc)
        rep.check(f"stratum {g}: binomial p", stats.binomtest(int(d.correct.sum()), len(d), 0.5).pvalue, p)
    gc = pd.read_csv(R("unam_group_comparison.csv"))
    rep.check("Kruskal-Wallis p across six metrics, lowest", gc.p_value.min(), 0.43)
    rep.check("Kruskal-Wallis p across six metrics, highest", gc.p_value.max(), 0.99)
    lat = u[u.TLEside.isin(["Left-TLE", "Right-TLE"])]
    ct = pd.crosstab(lat.TLEside, lat.hasMTS)
    rep.check("MTS in all 62", int(u.hasMTS.sum()), 15, digits=0)
    rep.check("MTS, left focus", int(ct.loc["Left-TLE", 1]), 6, digits=0)
    rep.check("MTS, right focus", int(ct.loc["Right-TLE", 1]), 9, digits=0)
    rep.check("MTS share, left (%)", 100 * ct.loc["Left-TLE", 1] / ct.loc["Left-TLE"].sum(), 35.3)
    rep.check("MTS share, right (%)", 100 * ct.loc["Right-TLE", 1] / ct.loc["Right-TLE"].sum(), 81.8)
    odds, p = stats.fisher_exact(ct.loc[["Right-TLE", "Left-TLE"], [1, 0]].values)
    rep.check("Fisher exact odds ratio", odds, 8.25)
    rep.check("Fisher exact p", p, 0.024)
    rep.check("sex balanced between lateralized groups, Fisher p", stats.fisher_exact(pd.crosstab(lat.TLEside, lat.gender).values)[1], 1.00)

    rep.section("S3 Minimum detectable effects (alpha 0.05, power 0.8)")
    from statsmodels.stats.power import TTestIndPower
    pw = TTestIndPower()
    for lab, n1, n2, v in (("social drinkers vs abusers", 196, 26, 0.587), ("social drinkers vs dependent", 196, 16, 0.732),
                           ("CN vs MCI", 25, 24, 0.817), ("healthy vs left focus", 34, 17, 0.849),
                           ("healthy vs right focus", 34, 11, 0.994), ("left vs right focus", 17, 11, 1.126)):
        rep.check(f"Cohen's d, {lab}", pw.solve_power(nobs1=n1, ratio=n2 / n1, alpha=0.05, power=0.8), v)

    rep.server_only([
        ("Real reward-parcellation medians 0.257, 0.209, 0.256", "pipeline/05_clinical/run_hcp_aud_batch_real_reward.py"),
        ("Twin contrasts: 21 discordant pairs (12, 8, 1); 7 of 12 positive; -0.0616 vs -0.0624 with 3 of 8 positive in both",
         "pipeline/01_operator_divergence/run_hcp_aud_batch.py and pipeline/05_clinical/compare_real_reward_vs_original.py (pair-level HCP data are restricted)"),
        ("Age balance between lateralized groups (30.4 vs 29.2 years, Mann-Whitney p 0.48)",
         "pipeline/00_preprocessing/reported_values.py --participants <OpenNeuro ds004469 participants.tsv>"),
    ])
    sys.exit(rep.finish())


if __name__ == "__main__":
    main()
