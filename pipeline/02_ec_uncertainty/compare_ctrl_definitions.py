#!/usr/bin/env python3
"""
compare_ctrl_definitions.py — do the rank-stability findings hold when
regions are ranked as DRIVERS rather than as RECEIVERS?

THE QUESTION
------------
The GraphNet estimator fits x(t+1) = A x(t), so A[i, j] is region j's
influence on region i. Two "average controllability" definitions then
differ for directed EC:

  package  physics.average_controllability = diag( sum_t A^t (A^t)^T )
           row-based: how strongly region j is REACHED by the network.
           This is what every published rank-stability result used.
  gu       Gu et al. (2015): trace of the Gramian with input at j alone,
           sum_t ||A^t e_j||^2 = diag( sum_t (A^t)^T A^t )
           column-based: how strongly an input AT region j spreads.
           This is what "driver node" means.

Both are identical for symmetric matrices. For directed EC they are not.

WHAT IT DOES
------------
Rescores the SAVED bootstrap EC ensembles ({subject}_EC_boot.npy) under
both definitions. No re-estimation. Both are scored through the package's
own driver_node_rank_stability(), so tau and Jaccard are computed the same
way for each.

SANITY GATE (per subject)
-------------------------
The package definition must reproduce the stored kendall_tau_mean and
jaccard_topk_mean from the published bootstrap results. If it does not,
the ensemble on disk is not the one behind the published numbers, and that
subject is flagged. The summary refuses to interpret a cohort with any
gate failure.

READING RULE — FIXED 2026-09-29, BEFORE ANY RESULT EXISTS
---------------------------------------------------------
(1) Does the HCP / ADNI dichotomy hold under the driver definition?
    AUC = probability that a random HCP subject's tau exceeds a random ADNI
    subject's tau, computed under each definition.
      AUC_gu >= 0.95   HOLDS       rank-stability section stands; report gu
                                   as primary, package as secondary
      0.75 - 0.95      WEAKENED    restate: the dichotomy is definition-
                                   dependent in strength
      < 0.75           DOES NOT HOLD  the section needs rethinking, not
                                   rewording
(2) Are the two definitions measuring the same thing on real EC?
    median, over subjects, of the Spearman correlation between the two
    definitions' mean scores.
      rho >= 0.7   AGREE        mostly a framing fix
      0.3 - 0.7    PARTIAL      report both; "driver" language follows gu
      < 0.3        DIFFERENT    different quantities; every "driver" claim
                                must rest on gu

USAGE
-----
  # once per cohort (resumable: re-run the same command to continue)
  python3 compare_ctrl_definitions.py run \\
      --ec-dir <dir of *_EC_boot.npy> \\
      --results path_b_bootstrap_results_s400.csv \\
      --label ADNI-400 --n-workers 40 --out ctrl_check/

  # then, across cohorts
  python3 compare_ctrl_definitions.py summarise --dir ctrl_check/ \\
      --high HCP --low ADNI-400 ADNI-450
"""
import os
# cap BLAS threads BEFORE numpy loads: parallel workers each multithreading
# internally cost this project a 70-140x slowdown before
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import argparse
import glob
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

# works both as the neurosim package and from a flat Scripts/ directory,
# as the published frozen-state run used
physics = driver_node_rank_stability = None


def _load_neurosim():
    """Imported only by 'run'; 'summarise' needs nothing but pandas.
    Works as the neurosim package or from a flat Scripts/ directory."""
    global physics, driver_node_rank_stability
    if physics is not None:
        return
    try:
        from nctsens import physics as _p
        from nctsens.connectivity import driver_node_rank_stability as _d
    except ImportError:
        from nctsens import physics as _p
        from nctsens.connectivity import driver_node_rank_stability as _d
    physics, driver_node_rank_stability = _p, _d

TOP_K = 5
TARGET_RHO = 0.9
GATE_TOL = 1e-6


def pkg_ctrl(A):
    """Exactly run_ec_bootstrap_batch.ctrl_func: normalise, then the
    package's average controllability."""
    _load_neurosim()
    return physics.average_controllability(physics.normalise_matrix(A, target_rho=TARGET_RHO))


def gu_ctrl(A):
    """Gu et al. (2015): column-based average controllability."""
    _load_neurosim()
    from scipy.linalg import solve_discrete_lyapunov
    An = physics.normalise_matrix(A, target_rho=TARGET_RHO)
    return np.diag(solve_discrete_lyapunov(An.T, np.eye(len(An))))


def check_pkg_matches_batch():
    """If run_ec_bootstrap_batch.py is importable, prove pkg_ctrl is its
    ctrl_func on a random directed matrix."""
    try:
        from run_ec_bootstrap_batch import ctrl_func
    except Exception:
        return "run_ec_bootstrap_batch.ctrl_func not importable; using the package directly"
    A = np.abs(np.random.default_rng(0).normal(size=(30, 30)))
    same = np.allclose(ctrl_func(A), pkg_ctrl(A))
    if not same:
        sys.exit("pkg_ctrl differs from run_ec_bootstrap_batch.ctrl_func — stop")
    return "pkg_ctrl verified identical to run_ec_bootstrap_batch.ctrl_func"


def one_subject(job):
    _load_neurosim()
    sub, path, stored_tau, stored_jac = job
    from scipy.stats import spearmanr
    t0 = time.time()
    try:
        EC = np.load(path)
        r_pkg = driver_node_rank_stability(EC, pkg_ctrl, top_k=TOP_K)
        r_gu = driver_node_rank_stability(EC, gu_ctrl, top_k=TOP_K)
        s_pkg = np.mean([pkg_ctrl(EC[b]) for b in range(len(EC))], axis=0)
        s_gu = np.mean([gu_ctrl(EC[b]) for b in range(len(EC))], axis=0)
        top_p = set(np.argsort(-s_pkg)[:TOP_K]); top_g = set(np.argsort(-s_gu)[:TOP_K])
        gate = (np.isfinite(stored_tau) and abs(r_pkg["kendall_tau_mean"] - stored_tau) <= GATE_TOL
                and abs(r_pkg["jaccard_topk_mean"] - stored_jac) <= GATE_TOL)
        return dict(subject_id=sub, n_boot=EC.shape[0], N=EC.shape[1],
                    tau_package=r_pkg["kendall_tau_mean"], jaccard_package=r_pkg["jaccard_topk_mean"],
                    tau_gu=r_gu["kendall_tau_mean"], jaccard_gu=r_gu["jaccard_topk_mean"],
                    rho_package_vs_gu=float(spearmanr(s_pkg, s_gu)[0]),
                    top5_overlap_package_vs_gu=len(top_p & top_g) / len(top_p | top_g),
                    stored_tau=stored_tau, stored_jaccard=stored_jac,
                    gate_pass=bool(gate), seconds=round(time.time() - t0, 1), error="")
    except Exception as e:
        return dict(subject_id=sub, gate_pass=False, error=f"{type(e).__name__}: {str(e)[:120]}")


def cmd_run(a):
    _load_neurosim()
    print(check_pkg_matches_batch())
    res = pd.read_csv(a.results)
    res = res[res.status == "ok"] if "status" in res else res
    stored = {str(r.subject_id): (r.kendall_tau_mean, r.jaccard_topk_mean) for r in res.itertuples()}
    files = sorted(glob.glob(os.path.join(a.ec_dir, "*_EC_boot.npy")))
    if not files:
        sys.exit(f"no *_EC_boot.npy in {a.ec_dir}")
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, f"ctrl_definition_comparison_{a.label}.csv")
    done = set(pd.read_csv(out).subject_id.astype(str)) if os.path.exists(out) else set()
    jobs = []
    for f in files:
        sub = os.path.basename(f).replace("_EC_boot.npy", "")
        if sub in done:
            continue
        t, j = stored.get(sub, (np.nan, np.nan))
        jobs.append((sub, f, t, j))
    missing = [s for s in (os.path.basename(f).replace("_EC_boot.npy", "") for f in files) if s not in stored]
    print(f"{a.label}: {len(files)} ensembles, {len(done)} already done, {len(jobs)} to run; "
          f"{len(missing)} without a stored result (gate cannot pass for those)")
    if a.max_subjects:
        jobs = jobs[:a.max_subjects]
    n = a.n_workers or max(1, (os.cpu_count() or 2) - 2)
    first = not os.path.exists(out)
    with Pool(n) as pool:
        for k, row in enumerate(pool.imap_unordered(one_subject, jobs), 1):
            pd.DataFrame([row]).to_csv(out, mode="a", header=first, index=False); first = False
            flag = "ok " if row.get("gate_pass") else "GATE" if not row.get("error") else "ERR "
            print(f"  [{k}/{len(jobs)}] {flag} {row['subject_id']}  "
                  f"tau pkg {row.get('tau_package', np.nan):.3f} gu {row.get('tau_gu', np.nan):.3f}  "
                  f"rho {row.get('rho_package_vs_gu', np.nan):+.2f}  {row.get('error', '')}", flush=True)
    d = pd.read_csv(out)
    print(f"\n{a.label}: {len(d)} subjects in {out}")
    print(f"  gate pass: {int(d.gate_pass.sum())}/{len(d)}   errors: {int((d.error.fillna('') != '').sum())}")
    cols = ["tau_package", "tau_gu", "jaccard_package", "jaccard_gu", "rho_package_vs_gu",
            "top5_overlap_package_vs_gu"]
    print(d[cols].median().round(3).to_string())


def auc(high, low):
    """P(random high-cohort value > random low-cohort value); ties count half."""
    h, l = np.asarray(high)[:, None], np.asarray(low)[None, :]
    return float(((h > l).sum() + 0.5 * (h == l).sum()) / (h.size * l.size))


def cmd_summarise(a):
    D = {}
    for f in sorted(glob.glob(os.path.join(a.dir, "ctrl_definition_comparison_*.csv"))):
        lab = os.path.basename(f)[len("ctrl_definition_comparison_"):-4]
        D[lab] = pd.read_csv(f)
    if not D:
        sys.exit(f"no comparison CSVs in {a.dir}")
    print("PER COHORT (medians)\n")
    bad = []
    for lab, d in D.items():
        noref = d[d.stored_tau.isna()] if "stored_tau" in d else d.iloc[0:0]
        if len(noref):
            print(f"  {lab}: {len(noref)} subject(s) with no published result "
                  f"({', '.join(noref.subject_id.astype(str))}) — excluded; they were not "
                  "in the published analysis, so there is nothing to reproduce")
            d = D[lab] = d[d.stored_tau.notna()]
        g = int(d.gate_pass.sum())
        print(f"  {lab:10s} n={len(d):3d}  gate {g}/{len(d)}   "
              f"tau pkg {d.tau_package.median():.3f} gu {d.tau_gu.median():.3f}   "
              f"Jaccard pkg {d.jaccard_package.median():.3f} gu {d.jaccard_gu.median():.3f}   "
              f"rho {d.rho_package_vs_gu.median():+.2f}   top-5 shared by definitions "
              f"{d.top5_overlap_package_vs_gu.median():.2f}")
        if g < len(d):
            bad.append(lab)
    if bad:
        print(f"\n  ** gate failures in {bad}: those subjects' ensembles do not reproduce the "
              "published results. Resolve before interpreting. **")
    print("\nREADING RULE (fixed 2026-09-29)\n")
    hi = D.get(a.high)
    for low in a.low:
        lo = D.get(low)
        if hi is None or lo is None:
            print(f"  {a.high} vs {low}: missing"); continue
        if a.high in bad or low in bad:
            print(f"  (1) {a.high} vs {low}: NOT INTERPRETED — gate failures in "
                  f"{[c for c in (a.high, low) if c in bad]}"); continue
        a_pkg = auc(hi.tau_package.dropna(), lo.tau_package.dropna())
        a_gu = auc(hi.tau_gu.dropna(), lo.tau_gu.dropna())
        verdict = ("HOLDS" if a_gu >= 0.95 else "WEAKENED" if a_gu >= 0.75 else "DOES NOT HOLD")
        print(f"  (1) {a.high} vs {low}: AUC package {a_pkg:.3f}, gu {a_gu:.3f}  ->  dichotomy under gu: {verdict}")
    ok = [d for lab, d in D.items() if lab not in bad]
    if not ok:
        print("  (2) NOT INTERPRETED — every cohort has gate failures"); return
    rho = pd.concat([d.rho_package_vs_gu for d in ok]).median()
    v2 = "AGREE" if rho >= 0.7 else "PARTIAL" if rho >= 0.3 else "DIFFERENT"
    print(f"  (2) median rho between definitions, cohorts passing the gate: {rho:+.3f}  ->  {v2}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run"); r.add_argument("--ec-dir", required=True); r.add_argument("--results", required=True)
    r.add_argument("--label", required=True); r.add_argument("--out", default="ctrl_check")
    r.add_argument("--n-workers", type=int, default=0); r.add_argument("--max-subjects", type=int, default=0)
    s = sp.add_parser("summarise"); s.add_argument("--dir", default="ctrl_check")
    s.add_argument("--high", default="HCP"); s.add_argument("--low", nargs="+", default=["ADNI-400", "ADNI-450"])
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_summarise(a)


if __name__ == "__main__":
    main()
