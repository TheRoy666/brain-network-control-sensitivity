#!/usr/bin/env python3
"""
Does the rank-stability dichotomy survive the estimator's
hyperparameters, or is it an artifact of one setting?

THE PROBLEM
-----------
The paper's lead reliability finding is that bootstrap rank agreement
for driver-node rankings is high in HCP (Kendall's tau 0.869) and near
chance in ADNI (approximately 0.07), with no overlap between the
distributions.

Both were computed with the GraphNet estimator at a single setting:
lambda_ridge = lambda_graph = 1.0, lag = 1. Those values were never
varied.

That matters more in ADNI than in HCP, because of an identifiability
arithmetic the manuscript does not currently state. The estimator fits
an N-by-N operator from N-by-T observations. In ADNI, N is 400 to 450
and T is about 197, so the number of free parameters exceeds the number
of observations by roughly a factor of two BEFORE regularisation. The
regulariser is not a refinement there; it is what makes the problem
solvable at all, and its weight determines the answer.

Four referees raised this independently. The threat is specific: if
sweeping the penalties recovers rank stability in ADNI, the finding
stops being a warning about short multi-site recordings and becomes a
statement about one poorly-chosen hyperparameter.

THE TEST
--------
Recompute bootstrap rank stability across a grid of lambda_ridge,
lambda_graph and lag, in both cohorts, holding everything else fixed.

    ADNI tau stays low across the grid   ->  the dichotomy is a property
                                             of the data regime. The
                                             finding stands.

    ADNI tau rises materially somewhere  ->  the dichotomy is at least
                                             partly a tuning artifact and
                                             the claim must be restated.

HCP is swept too, and for a reason. If HCP's tau is also sensitive to
the penalties, then the estimator is unstable everywhere and neither
cohort's number means what the paper says. If HCP is flat while ADNI
moves, that itself localises the problem to the low-T/N regime.

lambda_graph = 0 is included deliberately: it removes the structural
prior entirely, leaving ridge-only estimation. If ADNI's tau depends on
the structural prior, that is worth knowing, because the prior is the
same for every subject in a cohort and could impose shared structure
that inflates apparent reproducibility.

COST
----
Bootstrap cost scales as grid points x resamples x subjects. The full
analysis used 200 resamples; the convergence check reported in the paper
found that moving from 200 to 400 changed tau by 0.0003 to 0.0023, so a
smaller number is adequate for a comparison ACROSS settings even though
the headline numbers use 200. Defaults here are 50 resamples and 20
subjects per cohort, which is enough to detect a shift of the size that
would matter, and can be raised for any setting that looks interesting.

USAGE
-----
    python3 graphnet_sensitivity.py \\
        --ts-dir <timeseries> --ts-suffix _timeseries.npy \\
        --sc-dir <connectomes> --sc-suffix _schaefer400_connectome.csv \\
        --cohort ADNI-S400 --n-subjects 20 --n-boot 50 --n-workers 40 \\
        --out graphnet_sensitivity_ADNI_S400.csv
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import itertools
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from nctsens.connectivity import (graphnet_effective_connectivity, block_bootstrap_ec)
from nctsens.physics import normalise_matrix, average_controllability
from nctsens.loader import from_arrays, load_connectome

# DEFAULTS ARE DELIBERATELY COARSE. The question is narrow -- can ANY
# setting recover rank stability -- and a 5x5x2 grid at 50 resamples is
# roughly 50,000 EC estimates per cohort, which measured at 4-7 s each is
# 60-100 CPU-hours. That is not worth spending to answer a yes/no.
#
# Three decades spanning the published value, plus lambda_graph = 0 which
# removes the structural prior entirely, is enough to detect a recoverable
# setting. Widen with --lambda-ridge / --lambda-graph / --lags if any
# corner looks interesting.
LAMBDA_RIDGE = [0.01, 1.0, 100.0]
LAMBDA_GRAPH = [0.0, 1.0, 100.0]
LAGS = [1]

TARGET_RHO = 0.9


# --- added 2026-09-29: score the SAME resamples under both controllability
# definitions. The package's average_controllability is row-based (ranks
# regions most easily driven); Gu et al. (2015) is column-based (ranks
# regions that drive). They coincide for symmetric matrices only. The
# package columns are unchanged, so they must reproduce the published run.
from scipy.linalg import solve_discrete_lyapunov


def gu_average_controllability(A_norm):
    """Gu et al. (2015) average controllability of an ALREADY-normalised A:
    trace of the Gramian with input at node j alone, sum_t ||A^t e_j||^2."""
    return np.diag(solve_discrete_lyapunov(A_norm.T, np.eye(A_norm.shape[0])))


def mean_pairwise_jaccard(ranks, k=5):
    """Mean Jaccard of the top-k sets over all resample pairs; rank 0 = top."""
    tops = [set(np.where(r < k)[0]) for r in ranks]
    B = len(tops)
    vals = [len(tops[i] & tops[j]) / len(tops[i] | tops[j])
            for i in range(B) for j in range(i + 1, B)]
    return float(np.mean(vals)) if vals else np.nan


def mean_pairwise_tau(rank_matrix, max_pairs=200, seed=0):
    """Mean Kendall's tau across bootstrap pairs.

    All pairs is O(B^2); for B = 50 that is 1225, which is affordable,
    but the cap keeps the cost bounded if B is raised.
    """
    B = rank_matrix.shape[0]
    pairs = [(i, j) for i in range(B) for j in range(i + 1, B)]
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(seed)
        pairs = [pairs[k] for k in
                 rng.choice(len(pairs), max_pairs, replace=False)]
    vals = [kendalltau(rank_matrix[i], rank_matrix[j]).correlation
            for i, j in pairs]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


_CACHE = {}


def _load(ts_path, sc_path, sid):
    """Cached per worker: the same subject is now visited once per setting,
    so reloading and re-validating each time would dominate the cost."""
    key = (ts_path, sc_path)
    if key in _CACHE:
        return _CACHE[key]
    X_raw = np.load(ts_path)
    SC_raw = load_connectome(sc_path)
    N_nodes = SC_raw.shape[0]
    if X_raw.shape[0] == N_nodes:
        X_o = X_raw
    elif X_raw.shape[1] == N_nodes:
        X_o = X_raw.T
    else:
        raise ValueError(f"X {X_raw.shape} matches neither dim of SC "
                         f"{SC_raw.shape}")
    data = from_arrays(X=X_o, SC=SC_raw, subject_id=sid, validate=True)
    _CACHE.clear()          # one subject at a time; do not grow unbounded
    _CACHE[key] = data
    return data


def process_cell(args):
    """One (subject, setting) pair.

    Parallelising over subjects alone left workers idle whenever the
    subject count was below the core count, and made the wall clock equal
    to one subject's ENTIRE grid. At this granularity the work divides
    evenly regardless of how many subjects there are.
    """
    sid, ts_path, sc_path, cohort, n_boot, block_length, lr, lg, lag = args
    try:
        data = _load(ts_path, sc_path, sid)
        X, SC, N = data["X"], data["SC"], data["N"]
        T = X.shape[1]
        bl = block_length or max(2, int(round(np.sqrt(T))))
        ens = block_bootstrap_ec(
            X, SC, ec_func=graphnet_effective_connectivity,
            n_boot=n_boot, block_length=bl, seed=0,
            lambda_ridge=lr, lambda_graph=lg, lag=lag)
        ranks, ranks_gu = [], []
        for k in range(ens.shape[0]):
            A = normalise_matrix(ens[k], target_rho=TARGET_RHO)
            ranks.append(np.argsort(np.argsort(-average_controllability(A))))
            ranks_gu.append(np.argsort(np.argsort(-gu_average_controllability(A))))
        ranks, ranks_gu = np.asarray(ranks), np.asarray(ranks_gu)
        return dict(subject_id=sid, cohort=cohort, N=N, T=T, block_length=bl,
                    n_boot=n_boot, lambda_ridge=lr, lambda_graph=lg, lag=lag,
                    mean_tau=mean_pairwise_tau(ranks),
                    mean_tau_gu=mean_pairwise_tau(ranks_gu),
                    jaccard_top5=mean_pairwise_jaccard(ranks),
                    jaccard_top5_gu=mean_pairwise_jaccard(ranks_gu))
    except Exception as e:
        return dict(subject_id=sid, cohort=cohort, lambda_ridge=lr,
                    lambda_graph=lg, lag=lag, mean_tau=np.nan,
                    error=str(e)[:90])


def _unused_process_subject(args):
    sid, ts_path, sc_path, cohort, n_boot, block_length = args
    try:
        X_raw = np.load(ts_path)
        SC_raw = load_connectome(sc_path)
        N_nodes = SC_raw.shape[0]
        # Resolve orientation against the region count, never by size.
        if X_raw.shape[0] == N_nodes:
            X_o = X_raw
        elif X_raw.shape[1] == N_nodes:
            X_o = X_raw.T
        else:
            raise ValueError(f"X {X_raw.shape} matches neither dim of SC "
                             f"{SC_raw.shape}")
        data = from_arrays(X=X_o, SC=SC_raw, subject_id=sid, validate=True)
        X, SC, N = data["X"], data["SC"], data["N"]
        T = X.shape[1]
    except Exception as e:
        print(f"  load failed {sid}: {e}", flush=True)
        return []

    bl = block_length or max(2, int(round(np.sqrt(T))))
    rows = []
    for lr, lg, lag in itertools.product(LAMBDA_RIDGE, LAMBDA_GRAPH, LAGS):
        try:
            ens = block_bootstrap_ec(
                X, SC, ec_func=graphnet_effective_connectivity,
                n_boot=n_boot, block_length=bl, seed=0,
                lambda_ridge=lr, lambda_graph=lg, lag=lag)
            ranks = []
            for k in range(ens.shape[0]):
                A = normalise_matrix(ens[k], target_rho=TARGET_RHO)
                ac = average_controllability(A)
                ranks.append(np.argsort(np.argsort(-ac)))
            tau = mean_pairwise_tau(np.asarray(ranks))
            rows.append(dict(subject_id=sid, cohort=cohort, N=N, T=T,
                             block_length=bl, n_boot=n_boot,
                             lambda_ridge=lr, lambda_graph=lg, lag=lag,
                             mean_tau=tau))
        except Exception as e:
            rows.append(dict(subject_id=sid, cohort=cohort, N=N, T=T,
                             block_length=bl, n_boot=n_boot,
                             lambda_ridge=lr, lambda_graph=lg, lag=lag,
                             mean_tau=np.nan, error=str(e)[:90]))
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ts-dir", required=True)
    ap.add_argument("--ts-suffix", required=True)
    ap.add_argument("--sc-dir", required=True)
    ap.add_argument("--sc-suffix", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--n-subjects", type=int, default=0,
                    help="0 = all subjects. Parallelism is over "
                         "(subject, setting) pairs, so the full cohort costs "
                         "little more wall clock than a subsample.")
    ap.add_argument("--lags", default="1",
                    help="Comma-separated. Lag 2 doubles the cost and is the "
                         "least likely to matter; add it once lag 1 is read.")
    ap.add_argument("--n-boot", type=int, default=30,
                    help="30 gives 435 bootstrap pairs per cell, ample "
                         "for detecting a shift from 0.07. The headline "
                         "numbers use 200; this is a comparison ACROSS "
                         "settings, not a headline estimate.")
    ap.add_argument("--block-length", type=int, default=0,
                    help="0 = the square-root-of-T rule used in the paper.")
    ap.add_argument("--n-workers", type=int, default=0)
    ap.add_argument("--lambda-ridge", default=None,
                    help="Comma-separated override of the ridge grid.")
    ap.add_argument("--lambda-graph", default=None,
                    help="Comma-separated override. Include 0 to test the "
                         "estimator with no structural prior at all.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    global LAMBDA_RIDGE, LAMBDA_GRAPH, LAGS
    if args.lambda_ridge:
        LAMBDA_RIDGE = [float(x) for x in args.lambda_ridge.split(",")]
    if args.lambda_graph:
        LAMBDA_GRAPH = [float(x) for x in args.lambda_graph.split(",")]
    if args.lags:
        LAGS = [int(x) for x in args.lags.split(",")]

    files = sorted(f for f in os.listdir(args.ts_dir)
                   if f.endswith(args.ts_suffix))
    if args.n_subjects:
        files = files[:args.n_subjects]
    jobs = []
    for f in files:
        sid = f[:-len(args.ts_suffix)]
        sc = os.path.join(args.sc_dir, sid + args.sc_suffix)
        if os.path.exists(sc):
            jobs.append((sid, os.path.join(args.ts_dir, f), sc, args.cohort,
                         args.n_boot, args.block_length))
    grid = list(itertools.product(LAMBDA_RIDGE, LAMBDA_GRAPH, LAGS))
    cells = [(sid, ts, sc, coh, nb, bl, lr, lg, lag)
             for (sid, ts, sc, coh, nb, bl) in jobs
             for (lr, lg, lag) in grid]
    if not cells:
        raise SystemExit("No subject had both a timeseries and a connectome.")

    n_workers = args.n_workers or cpu_count()
    print(f"{len(jobs)} subjects x {len(grid)} settings = {len(cells)} cells, "
          f"{args.n_boot} resamples each, {n_workers} workers")
    print(f"  approx {len(cells) * args.n_boot} EC estimates total")

    import time
    t0 = time.time()
    rows = []
    with Pool(n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process_cell, cells, chunksize=1), 1):
            rows.append(r)
            if i % max(1, len(cells) // 20) == 0 or i == len(cells):
                el = time.time() - t0
                eta = el / i * (len(cells) - i)
                print(f"  {i}/{len(cells)}  elapsed {el/60:.1f} min  "
                      f"eta {eta/60:.1f} min", flush=True)
                # incremental write: an interrupted run keeps what it has
                pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    ok = df.dropna(subset=["mean_tau"])
    print("\n" + "=" * 76)
    print(f"GRAPHNET SENSITIVITY — {args.cohort}")
    print("=" * 76)
    print(f"  N = {ok.N.iloc[0]}, T = {ok['T'].median():.0f}, "
          f"N^2 / (N*T) = {ok.N.iloc[0] / ok['T'].median():.2f}"
          "   (>1 means more parameters than observations before regularisation)")

    pub = ok[(ok.lambda_ridge == 1.0) & (ok.lambda_graph == 1.0) & (ok.lag == 1)]
    if len(pub):
        print(f"\n  PUBLISHED SETTING (1.0, 1.0, lag 1): "
              f"median tau {pub.mean_tau.median():.4f}   n = {len(pub)}")

    print(f"\n  ACROSS THE WHOLE GRID: median tau "
          f"{ok.mean_tau.median():.4f}, range "
          f"[{ok.groupby(['lambda_ridge','lambda_graph','lag']).mean_tau.median().min():.4f}, "
          f"{ok.groupby(['lambda_ridge','lambda_graph','lag']).mean_tau.median().max():.4f}]")

    print("\n  median tau by setting (rows lambda_ridge, cols lambda_graph), lag 1:")
    l1 = ok[ok.lag == 1]
    if len(l1):
        print(l1.pivot_table(index="lambda_ridge", columns="lambda_graph",
                             values="mean_tau", aggfunc="median")
                .round(3).to_string())
    l2 = ok[ok.lag == 2]
    if len(l2):
        print("\n  same, lag 2:")
        print(l2.pivot_table(index="lambda_ridge", columns="lambda_graph",
                             values="mean_tau", aggfunc="median")
                .round(3).to_string())

    print("\n  lambda_graph = 0 removes the structural prior entirely:")
    for lg in sorted(ok.lambda_graph.unique()):
        g = ok[(ok.lambda_graph == lg) & (ok.lag == 1)]
        if len(g):
            print(f"    lambda_graph = {lg:>6}: median tau {g.mean_tau.median():.4f}")

    best = ok.groupby(["lambda_ridge", "lambda_graph", "lag"]).mean_tau.median()
    bi = best.idxmax()
    print("\n" + "-" * 76)
    print(f"  best setting: ridge {bi[0]}, graph {bi[1]}, lag {bi[2]} "
          f"-> median tau {best.max():.4f}")
    if len(pub):
        print(f"  published setting                       "
              f"-> median tau {pub.mean_tau.median():.4f}")
        print(f"  best minus published: {best.max() - pub.mean_tau.median():+.4f}")
    print("\n  Read this against the cohort's headline value. If tau stays low")
    print("  across the entire grid, the dichotomy is a property of the data")
    print("  regime. If any setting recovers substantial rank stability, the")
    print("  claim is at least partly about hyperparameter choice and must be")
    print("  restated. Note that picking the maximum over a grid is itself a")
    print("  selection, so a small rise at one corner is not evidence of a")
    print("  recoverable setting.")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
