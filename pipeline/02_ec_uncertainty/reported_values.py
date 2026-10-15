#!/usr/bin/env python3
"""
reported_values.py — every in-text number of the estimation-uncertainty
results (Results 2.3–2.7, Discussion 3.2, Methods 4.5, Supplementary S5, S6
and S8 "Driver rankings" and "Receiver ranking") recomputed from the
deposited files in results/ and compared with the value printed in the paper.

    python3 pipeline/02_ec_uncertainty/reported_values.py [--results results]

"Driver" is the Gu et al. (2015) ranking (columns *_gu); "receiver" is the
Gramian-diagonal ranking (columns without the suffix).
"""
import argparse
import os
import sys
from math import comb

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from reported import Report  # noqa: E402

ARMS = ["HCP", "ADNI-400", "ADNI-450"]


def auc(a, b):
    """Probability that a value from a exceeds one from b (ties count half)."""
    return stats.mannwhitneyu(a, b).statistic / (len(a) * len(b))


def expected_jaccard(N, k=5):
    return sum(comb(k, i) * comb(N - k, k - i) / comb(N, k) * i / (2 * k - i) for i in range(k + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    R = lambda p: os.path.join(a.results, p)
    rep = Report("02 Estimation uncertainty and driver rankings")
    c = {k: pd.read_csv(R(f"02_ec_uncertainty/ctrl_definition_comparison_{k}.csv")) for k in ARMS}
    c = {k: (v[v.error.isna()] if "error" in v else v) for k, v in c.items()}

    rep.section("2.3 Driver versus receiver ranking")
    for k, r in (("HCP", -0.52), ("ADNI-400", -0.09), ("ADNI-450", -0.25)):
        rep.check(f"{k} median within-subject rank correlation, receiver vs driver", c[k].rho_package_vs_gu.median(), r)
        rep.check(f"{k} median top-five overlap between the two rankings (rarely coincide)",
                  c[k].top5_overlap_package_vs_gu.median(), 0.0, digits=1)

    rep.section("2.4 Driver-ranking stability, 200 resamples")
    for k, t in (("HCP", 0.885), ("ADNI-400", 0.329), ("ADNI-450", 0.371)):
        rep.check(f"{k} median Kendall's tau (driver)", c[k].tau_gu.median(), t)
    rep.check("HCP tau range, min", c["HCP"].tau_gu.min(), 0.819)
    rep.check("HCP tau range, max", c["HCP"].tau_gu.max(), 0.919)
    rep.check("S8 most stable ADNI-400 subject", c["ADNI-400"].tau_gu.max(), 0.396)
    rep.check("S8 most stable ADNI-450 subject", c["ADNI-450"].tau_gu.max(), 0.461)
    for k in ("ADNI-400", "ADNI-450"):
        rep.check(f"no overlap HCP vs {k}: AUC", auc(c["HCP"].tau_gu, c[k].tau_gu), 1.000)
    for k, j, s in (("HCP", 0.484, 3.26), ("ADNI-400", 0.037, 0.35), ("ADNI-450", 0.037, 0.36)):
        J = c[k].jaccard_gu.median()
        rep.check(f"{k} median top-five Jaccard (driver)", J, j)
        rep.check(f"{k} top regions shared by two resamples, 10J/(1+J) at the median", 10 * J / (1 + J), s)
    rep.check("4.5 chance Jaccard for two random top-five sets, N = 400 to 450 (about 0.0065)", expected_jaccard(425),
              "~0.0065", kind="range", lo=expected_jaccard(450), hi=expected_jaccard(400),
              note=f"N = 400: {expected_jaccard(400):.4f}, N = 450: {expected_jaccard(450):.4f}")
    rep.check("ADNI overlap above chance", c["ADNI-400"].jaccard_gu.median(), ">chance", kind="gt", ref=expected_jaccard(400))

    rep.section("2.4 / 4.10 Twin-pair clustering, driver ranking (twin_clustered_bootstrap.py)")
    tw = pd.read_csv(R("02_ec_uncertainty/twin_clustered_ci_HCP_driver.csv")).set_index("metric")
    rep.check("intraclass correlation, tau", tw.loc["tau_gu", "icc"], 0.558)
    rep.check("intraclass correlation, top-five overlap", tw.loc["jaccard_gu", "icc"], 0.206)
    rep.check("clustered interval on median tau, lower", tw.loc["tau_gu", "ci_lo_clustered"], 0.881)
    rep.check("clustered interval on median tau, upper", tw.loc["tau_gu", "ci_hi_clustered"], 0.889)
    rep.check("clustered interval wider than the independent one (%)", 100 * (tw.loc["tau_gu", "widening"] - 1), 17, digits=0,
              note="the receiver-ranking run (twin_clustered_ci_HCP_receiver.csv) gives 17%")
    rep.check("complete pairs", int(tw.loc["tau_gu", "n_pairs"]), 119, digits=0)

    rep.section("2.4 Frässle and Stephan comparison (Supplementary Methods)")
    fs = pd.concat([pd.read_csv(R(f"02_ec_uncertainty/frassle_stephan_{k}.csv")).assign(arm=k) for k in ARMS])
    rep.check("CV falls with strength in every subject (count)", int((fs.spearman_rho_strength_vs_coefficient_of_variation < 0).sum()), 336, digits=0)
    rep.check("subjects analyzed", len(fs), 336, digits=0)
    rep.check("SD rises with strength in every subject, fraction", (fs.spearman_rho_strength_vs_raw_std > 0).mean(), 1.0, digits=2)
    for k, v in (("ADNI-400", -0.978), ("ADNI-450", -0.974)):
        rep.check(f"{k} median Spearman, strength vs CV", fs[fs.arm == k].spearman_rho_strength_vs_coefficient_of_variation.median(), v)

    rep.section("2.5 HCP truncated, published penalty setting (driver unless stated)")
    n = pd.read_csv(R("02_ec_uncertainty/ntr_published_setting.csv"))
    # matched-ratio comparisons use ADNI at the same setting and resample count (50), as compare_ntr_rerun.py does
    pub = {k: (lambda d: d[(d.lambda_ridge == 1) & (d.lambda_graph == 1)])(
        pd.read_csv(R(f"02_ec_uncertainty/graphnet_sensitivity_{k}.csv"))) for k in ("ADNI-400", "ADNI-450")}
    md = n.groupby("requested_length").median(numeric_only=True)
    for L, t in ((4800, 0.884), (1200, 0.873), (600, 0.661), (400, 0.456), (300, 0.391), (200, 0.593)):
        rep.check(f"median tau at {L} timepoints", md.loc[L, "mean_tau_gu"], t)
    for L, j in ((4800, 0.482), (600, 0.082)):
        rep.check(f"median top-five Jaccard at {L}", md.loc[L, "jaccard_top5_gu"], j)
    for L in (400, 300):
        rep.check(f"median top-five Jaccard at {L} (about 0.02)", md.loc[L, "jaccard_top5_gu"], 0.02, digits=2)
    rep.check("regions over timepoints at 200 (HCP)", md.loc[200, "N_over_T"], 2.05)
    rep.check("regions over timepoints in ADNI (400 regions, 197 timepoints)", 400 / 197, 2.03)
    rep.check("matched ratio: HCP top-five Jaccard at 200", md.loc[200, "jaccard_top5_gu"], 0.038)
    rep.check("matched ratio: ADNI-400 top-five Jaccard", c["ADNI-400"].jaccard_gu.median(), 0.037)
    j200 = n[n.requested_length == 200].jaccard_top5_gu
    aucs = [auc(j200, pub[k].jaccard_top5_gu) for k in ("ADNI-400", "ADNI-450")]
    rep.check("matched ratio: AUC for top-five overlap, lower", min(aucs), 0.51)
    rep.check("matched ratio: AUC for top-five overlap, upper", max(aucs), 0.55)
    t200 = n[n.requested_length == 200].mean_tau_gu
    rep.check("matched ratio: whole-ranking advantage over ADNI-450 (fold)", t200.median() / pub["ADNI-450"].mean_tau_gu.median(), 1.6)
    rep.check("matched ratio: whole-ranking advantage over ADNI-400 (fold)", t200.median() / pub["ADNI-400"].mean_tau_gu.median(), 1.8)
    for k in ("ADNI-400", "ADNI-450"):
        rep.check(f"matched ratio: AUC for tau vs {k}", auc(t200, pub[k].mean_tau_gu), 1.000)
    w = n.pivot(index="subject_id", columns="requested_length", values="mean_tau_gu")
    rep.check("stability rises from 300 to 200 in every participant, fraction", (w[200] > w[300]).mean(), 1.0, digits=2)
    t300 = n[n.requested_length == 300].mean_tau_gu
    rep.check("S8 at 300: AUC vs ADNI-400", auc(t300, pub["ADNI-400"].mean_tau_gu), 0.89)
    rep.check("S8 at 300: AUC vs ADNI-450", auc(t300, pub["ADNI-450"].mean_tau_gu), 0.64)
    rep.check("S8 at 300 and 400, truncated HCP top-five overlap below ADNI's",
              max(md.loc[300, "jaccard_top5_gu"], md.loc[400, "jaccard_top5_gu"]) < c["ADNI-400"].jaccard_gu.median(), True)

    rep.section("2.5 / S8 Penalty grid at 200 timepoints")
    g = pd.read_csv(R("02_ec_uncertainty/ntr_grid_T200.csv"))
    gm = g.groupby(["lambda_ridge", "lambda_graph"]).mean_tau_gu.median()
    rep.check("median driver tau varies across the nine settings (fold)", gm.max() / gm.min(), 1.2)
    G = {k: pd.read_csv(R(f"02_ec_uncertainty/graphnet_sensitivity_{k}.csv")).dropna(subset=["mean_tau_gu"]) for k in ARMS}
    am = G["ADNI-400"].groupby(["lambda_ridge", "lambda_graph"]).mean_tau_gu.median()
    rep.check("ADNI: same grid varies about fourfold", am.max() / am.min(), 4, digits=0)
    rep.check("S8 strongest prior lowers stability in truncated HCP", gm.loc[(1.0, 100.0)] < gm.loc[(1.0, 1.0)], True)
    pv = g.pivot_table(index=["subject_id", "lambda_ridge"], columns="lambda_graph", values="mean_tau")
    rep.check("S8 (receiver) at 200, prior 0 vs 1 leaves HCP cells unchanged (%)", 100 * (pv[0.0] == pv[1.0]).mean(), 99, digits=0)
    sp = lambda df, col: df.groupby("subject_id")[col].agg(lambda x: x.max() - x.min()).median()
    rep.check("truncated HCP is 3-4x less sensitive to the penalties than ADNI",
              min(sp(G[k], "mean_tau_gu") for k in ("ADNI-400", "ADNI-450")) / sp(g, "mean_tau_gu"), "3-4", kind="range", lo=2.95, hi=4.49)

    rep.section("2.6 Regularization (full length, nine settings)")
    rep.check("HCP median within-subject change across settings", sp(G["HCP"], "mean_tau_gu"), 3.8e-6, kind="sci")
    for k, s in (("ADNI-400", 0.31), ("ADNI-450", 0.35)):
        rep.check(f"{k} median within-subject spread", sp(G[k], "mean_tau_gu"), s)
    for k, lo, hi in (("ADNI-400", 0.100, 0.410), ("ADNI-450", 0.118, 0.472)):
        m = G[k].groupby(["lambda_ridge", "lambda_graph"]).mean_tau_gu.median()
        rep.check(f"{k} lowest cell median", m.min(), lo)
        rep.check(f"{k} highest cell median", m.max(), hi)
        rep.check(f"{k} rank of the setting used (of nine)", int(m.rank().loc[(1.0, 1.0)]), 4, digits=0)
    adni_cells = pd.concat([G["ADNI-400"], G["ADNI-450"]])
    rep.check("ADNI cells in all", len(adni_cells), 882, digits=0)
    rep.check("largest single ADNI cell", adni_cells.mean_tau_gu.max(), 0.541)
    rep.check("least stable HCP participant", c["HCP"].tau_gu.min(), 0.819)
    best = max(G[k].groupby(["lambda_ridge", "lambda_graph"]).mean_tau_gu.median().max() for k in ("ADNI-400", "ADNI-450"))
    rep.check("S8 HCP over ADNI at ADNI's most favourable setting (fold)",
              G["HCP"].groupby(["lambda_ridge", "lambda_graph"]).mean_tau_gu.median().loc[(1.0, 1.0)] / best, 1.9)
    hp = G["HCP"].pivot_table(index=["subject_id", "lambda_ridge"], columns="lambda_graph", values="mean_tau_gu")
    hr = G["HCP"].pivot_table(index=["subject_id", "lambda_ridge"], columns="lambda_graph", values="mean_tau")
    rep.check("S8 prior weight 0 vs 1 bit-identical in every HCP cell, driver", (hp[0.0] == hp[1.0]).mean(), 1.0, digits=2)
    rep.check("S8 same, receiver", (hr[0.0] == hr[1.0]).mean(), 1.0, digits=2)

    rep.section("2.7 Block length")
    B = {k: pd.read_csv(R(f"02_ec_uncertainty/block_length_{k}.csv")).dropna(subset=["mean_tau_gu"]) for k in ARMS}
    bm = {k: B[k].groupby("block_length")[["mean_tau_gu", "jaccard_top5_gu"]].median() for k in ARMS}
    rep.check("square-root rule block length, HCP", int(B["HCP"].sqrt_rule_bl.iloc[0]), 69, digits=0)
    rep.check("square-root rule block length, ADNI", int(B["ADNI-400"].sqrt_rule_bl.iloc[0]), 14, digits=0)
    for k, lo, hi in (("ADNI-400", 0.323, 0.329), ("ADNI-450", 0.358, 0.374)):
        rep.check(f"{k} median tau across block lengths, min", bm[k].mean_tau_gu.min(), lo)
        rep.check(f"{k} median tau across block lengths, max", bm[k].mean_tau_gu.max(), hi)
    rep.check("HCP tau at block length 5", bm["HCP"].loc[5, "mean_tau_gu"], 0.811)
    rep.check("HCP tau at block length 100", bm["HCP"].loc[100, "mean_tau_gu"], 0.887)
    rep.check("HCP top-five Jaccard at 5", bm["HCP"].loc[5, "jaccard_top5_gu"], 0.272)
    rep.check("HCP top-five Jaccard at 100", bm["HCP"].loc[100, "jaccard_top5_gu"], 0.485)
    shared = sorted(set(bm["HCP"].index) & set(bm["ADNI-400"].index))
    folds = [bm["HCP"].loc[L, "mean_tau_gu"] / bm[k].loc[L, "mean_tau_gu"] for L in shared for k in ("ADNI-400", "ADNI-450")]
    rep.check("HCP over ADNI at shared block lengths, min (fold)", min(folds), 2.3)
    rep.check("HCP over ADNI at shared block lengths, max (fold)", max(folds), 2.7)
    rep.check("AUC at every shared block length", min(auc(B["HCP"][B["HCP"].block_length == L].mean_tau_gu,
                                                          B[k][B[k].block_length == L].mean_tau_gu)
                                                      for L in shared for k in ("ADNI-400", "ADNI-450")), 1.000)

    rep.section("2.7 / S5 / S8 Receiver ranking and head motion")
    rep.check("receiver tau at 200 timepoints (HCP)", md.loc[200, "mean_tau"], 0.860)
    rep.check("receiver tau at full length, published penalty, 25 resamples", md.loc[4800, "mean_tau"], 0.868)
    hm = pd.read_csv(R("cohort/hcp_head_motion.csv")).rename(columns={"Subject": "subject_id"}).merge(c["HCP"], on="subject_id")
    q = pd.read_csv(R("cohort/adni_qc.csv")).rename(columns={"subject": "subject_id"})
    man = pd.read_csv(R("05_clinical/adni_results_400.csv"))[["subject_id", "manufacturer"]]
    r, p = stats.spearmanr(hm.Mean_FD, hm.tau_package)
    rep.check("HCP receiver: rho with mean FD", r, 0.17)
    r, p = stats.spearmanr(hm.Mean_FD, hm.tau_gu)
    rep.check("HCP driver: rho with mean FD", r, 0.02)
    rep.check("HCP driver: p", p, 0.70)
    for k, rr, rd, pd_ in (("ADNI-400", 0.43, 0.13, 0.37), ("ADNI-450", None, 0.06, 0.69)):
        m = q.merge(c[k], on="subject_id")
        if rr is not None:
            rep.check(f"{k} receiver: rho with mean FD", stats.spearmanr(m.func_fd_mean, m.tau_package)[0], rr)
            mm = m.merge(man, on="subject_id")
            X = np.column_stack([np.ones(len(mm)), pd.get_dummies(mm.manufacturer, drop_first=True).astype(float).values])
            res = lambda y: y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
            rep.check(f"{k} receiver: rho adjusted for manufacturer", stats.spearmanr(res(mm.func_fd_mean.values),
                                                                                     res(mm.tau_package.values))[0], 0.48)
        r, p = stats.spearmanr(m.func_fd_mean, m.tau_gu)
        rep.check(f"{k} driver: rho with mean FD", r, rd)
        rep.check(f"{k} driver: p", p, pd_)
    qp = q[q.QC_VERDICT == "PASS"]
    rep.check("ADNI: mean FD vs proportion of censored frames, rho", stats.spearmanr(qp.func_fd_mean, qp.func_fd_perc)[0], 0.91)
    t200r = n[n.requested_length == 200].mean_tau
    rep.check("S8 receiver at matched ratio: HCP over ADNI-400 (fold, about 13)",
              t200r.median() / pub["ADNI-400"].mean_tau.median(), 13, digits=0)
    gr = g.groupby("subject_id").mean_tau.agg(lambda x: x.max() - x.min()).median()
    ar = G["ADNI-400"].groupby("subject_id").mean_tau.agg(lambda x: x.max() - x.min()).median()
    rep.check("S8 receiver: penalty sensitivity, ADNI-400 over truncated HCP (fold, about 150)", ar / gr, 150, digits=-1,
              note="order of magnitude")
    wr = n.pivot(index="subject_id", columns="requested_length", values="mean_tau")
    rep.check("S8 every HCP subject less stable at 600 than at 1200 and 400, receiver",
              ((wr[600] < wr[1200]) & (wr[600] < wr[400])).mean(), 1.0, digits=2)
    rep.check("S8 none under the driver ranking", ((w[600] < w[1200]) & (w[600] < w[400])).mean(), 0.0, digits=2)

    rep.server_only([
        ("Receiver tau 0.857 with HCP subsampled to an effective TR of 2.88 s", "adni_quality_vs_taustability.py header; HCP stride runs need the HCP timeseries"),
        ("Within ADNI, no relationship of receiver stability with tSNR, age or six connectome-quality metrics",
         "pipeline/02_ec_uncertainty/adni_quality_vs_taustability.py (age and connectome QC are not deposited)"),
        ("Single fit at 410 regions: 6.20 s at 197 timepoints, 3.55 s at 1200",
         "pipeline/02_ec_uncertainty/estimator_convergence/check_hcp_unam_convergence.py and run logs"),
        ("Resample-count convergence (50, 100, 200, 400 on five subjects per cohort)",
         "pipeline/02_ec_uncertainty/run_ec_bootstrap_batch.py (convergence mode)"),
        ("nctpy 1.1.0 ranks correlate 0.98 with the receiver ranking on planted operators",
         "pipeline/02_ec_uncertainty/nctpy_definition_check.py (runs here)"),
    ])
    sys.exit(rep.finish())


if __name__ == "__main__":
    main()
