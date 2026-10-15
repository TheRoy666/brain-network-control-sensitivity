#!/usr/bin/env python3
"""
Does the rank-stability contrast survive the bootstrap's block length?

THE PROBLEM
-----------
The moving-block bootstrap resamples contiguous runs of timepoints to
preserve temporal autocorrelation. Block length was set by the
square-root-of-T rule, which gives:

    HCP    T = 4800  ->  block length 69
    ADNI   T ~ 197   ->  block length 14

That is a five-fold difference between the two arms of the paper's
headline comparison, and it was chosen by a rule rather than validated.
Two referees raised it independently.

The concern is concrete. Block length trades off two errors: too short
and the resample destroys the autocorrelation the estimator depends on;
too long and the resamples become near-copies of each other, inflating
apparent agreement because there is little resampling variation left. If
rank stability depends on block length, and block length differs
systematically between cohorts, part of the dichotomy could be a
property of the bootstrap rather than of the data.

THE TEST
--------
Sweep block length in both cohorts and recompute rank stability, holding
everything else fixed.

    tau flat across block lengths      ->  the choice does not matter and
                                           the contrast is not a bootstrap
                                           artifact.

    tau rises with block length        ->  longer blocks inflate agreement,
                                           and HCP's 69 would be flattering
                                           HCP relative to ADNI's 14. The
                                           decisive check is then whether
                                           HCP at block length 14 still
                                           exceeds ADNI at 14.

That last comparison is the one that matters, and it is the reason HCP
must be swept over ADNI's range and not only its own. Running each
cohort at its own scale would leave exactly the confound the objection
names.

BLOCK LENGTHS
-------------
The defaults span from well below to well above both cohorts' rule-based
values, and include 14 and 69 explicitly so the published settings
appear in the sweep. For ADNI, block lengths above about 40 leave fewer
than five blocks per resample and are excluded automatically.

COST
----
Parallelism is over (subject, block length) pairs rather than over
subjects, because subject counts here are below core counts and
per-subject parallelism would leave most workers idle.

    ADNI:  49 subjects x 6 lengths x 50 resamples
    HCP:  238 subjects x 6 lengths x 50 resamples

USAGE
-----
    python3 bootstrap_block_length.py \\
        --ts-dir <timeseries> --ts-suffix _timeseries.npy \\
        --sc-dir <connectomes> --sc-suffix _schaefer400_connectome.csv \\
        --cohort ADNI-S400 --n-boot 50 --n-workers 39 \\
        --out block_length_ADNI_S400.csv
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from nctsens.connectivity import graphnet_effective_connectivity, block_bootstrap_ec
from nctsens.physics import normalise_matrix, average_controllability
from nctsens.loader import from_arrays, load_connectome

# Spans below and above both cohorts' rule-based values. 14 is ADNI's
# and 69 is HCP's; both appear so the published settings are in the sweep.
BLOCK_LENGTHS = [5, 14, 25, 40, 69, 100]
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
MIN_BLOCKS = 5          # fewer than this per resample is not a bootstrap


def mean_pairwise_tau(ranks, max_pairs=200, seed=0):
    B = ranks.shape[0]
    pairs = [(i, j) for i in range(B) for j in range(i + 1, B)]
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(seed)
        pairs = [pairs[k] for k in rng.choice(len(pairs), max_pairs,
                                              replace=False)]
    v = [kendalltau(ranks[i], ranks[j]).correlation for i, j in pairs]
    v = [x for x in v if np.isfinite(x)]
    return float(np.mean(v)) if v else np.nan


def load_subject(ts_path, sc_path, sid):
    """Orientation is resolved against the region count from SC, never by
    comparing array dimensions. A size heuristic silently transposes
    whenever T < N, which is the normal case in the shorter cohort."""
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
    return from_arrays(X=X_o, SC=SC_raw, subject_id=sid, validate=True)


def process_one(args):
    sid, ts_path, sc_path, cohort, n_boot, bl = args
    try:
        data = load_subject(ts_path, sc_path, sid)
        X, SC, N = data["X"], data["SC"], data["N"]
        T = X.shape[1]
        if T // bl < MIN_BLOCKS:
            return dict(subject_id=sid, cohort=cohort, N=N, T=T,
                        block_length=bl, n_boot=n_boot, mean_tau=np.nan,
                        n_blocks=T // bl,
                        error=f"only {T // bl} blocks; skipped")
        ens = block_bootstrap_ec(
            X, SC, ec_func=graphnet_effective_connectivity,
            n_boot=n_boot, block_length=bl, seed=0,
            lambda_ridge=1.0, lambda_graph=1.0)
        ranks, ranks_gu = [], []
        for k in range(ens.shape[0]):
            A = normalise_matrix(ens[k], target_rho=TARGET_RHO)
            ranks.append(np.argsort(np.argsort(-average_controllability(A))))
            ranks_gu.append(np.argsort(np.argsort(-gu_average_controllability(A))))
        ranks, ranks_gu = np.asarray(ranks), np.asarray(ranks_gu)
        return dict(subject_id=sid, cohort=cohort, N=N, T=T,
                    block_length=bl, n_boot=n_boot, n_blocks=T // bl,
                    sqrt_rule_bl=int(round(np.sqrt(T))),
                    mean_tau=mean_pairwise_tau(ranks),
                    mean_tau_gu=mean_pairwise_tau(ranks_gu),
                    jaccard_top5=mean_pairwise_jaccard(ranks),
                    jaccard_top5_gu=mean_pairwise_jaccard(ranks_gu))
    except Exception as e:
        return dict(subject_id=sid, cohort=cohort, block_length=bl,
                    n_boot=n_boot, mean_tau=np.nan, error=str(e)[:90])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ts-dir", required=True)
    ap.add_argument("--ts-suffix", required=True)
    ap.add_argument("--sc-dir", required=True)
    ap.add_argument("--sc-suffix", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--block-lengths", default=None,
                    help="Comma-separated. Default spans both cohorts' "
                         "rule-based values.")
    ap.add_argument("--n-subjects", type=int, default=0,
                    help="0 = all. Parallelism is over (subject, block "
                         "length) pairs, so the full cohort costs little "
                         "more wall clock than a subsample.")
    ap.add_argument("--n-boot", type=int, default=50)
    ap.add_argument("--n-workers", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bls = ([int(x) for x in args.block_lengths.split(",")]
           if args.block_lengths else BLOCK_LENGTHS)
    files = sorted(f for f in os.listdir(args.ts_dir)
                   if f.endswith(args.ts_suffix))
    if args.n_subjects:
        files = files[:args.n_subjects]
    subjects = []
    for f in files:
        sid = f[:-len(args.ts_suffix)]
        sc = os.path.join(args.sc_dir, sid + args.sc_suffix)
        if os.path.exists(sc):
            subjects.append((sid, os.path.join(args.ts_dir, f), sc))
    jobs = [(sid, ts, sc, args.cohort, args.n_boot, bl)
            for (sid, ts, sc) in subjects for bl in bls]
    print(f"{len(subjects)} subjects x {len(bls)} block lengths = "
          f"{len(jobs)} cells, {args.n_boot} resamples each")
    if not jobs:
        raise SystemExit("No subject had both a timeseries and a connectome.")

    n_workers = args.n_workers or min(cpu_count(), len(jobs))
    print(f"  {n_workers} workers")
    rows, t0 = [], time.time()
    with Pool(n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process_one, jobs), 1):
            rows.append(r)
            if i % 25 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  {i}/{len(jobs)}  elapsed {el/60:.1f} min  "
                      f"eta {el/i*(len(jobs)-i)/60:.1f} min", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    ok = df.dropna(subset=["mean_tau"])

    print("\n" + "=" * 74)
    print(f"BLOCK-LENGTH SENSITIVITY — {args.cohort}")
    print("=" * 74)
    if not len(ok):
        print("  no usable cells"); return
    print(f"  N = {ok.N.iloc[0]}, T = {ok['T'].median():.0f}, "
          f"square-root rule gives block length "
          f"{ok.sqrt_rule_bl.iloc[0] if 'sqrt_rule_bl' in ok else '?'}")
    print(f"\n  {'block':>7} {'blocks/resample':>17} {'n':>5} "
          f"{'median tau':>12} {'IQR':>22}")
    for bl, g in ok.groupby("block_length"):
        t = g.mean_tau
        print(f"  {bl:>7} {g.n_blocks.median():>17.0f} {len(g):>5} "
              f"{t.median():>12.4f} "
              f"[{t.quantile(.25):>9.4f}, {t.quantile(.75):>9.4f}]")

    med = ok.groupby("block_length").mean_tau.median()
    print(f"\n  range across block lengths: {med.min():.4f} to {med.max():.4f}"
          f"   spread {med.max() - med.min():.4f}")

    piv = ok.pivot_table(index="subject_id", columns="block_length",
                         values="mean_tau")
    if piv.shape[1] >= 2:
        w = (piv.max(axis=1) - piv.min(axis=1)).dropna()
        print(f"  within-subject spread: median {w.median():.4f}, "
              f"max {w.max():.4f}")

    print("\n  THE COMPARISON THAT MATTERS: read this cohort's value at block")
    print("  length 14 against the other cohort's value at 14. Comparing each")
    print("  cohort at its own rule-based length leaves exactly the confound")
    print("  the objection names.")
    if 14 in med.index:
        print(f"    {args.cohort} at block length 14: median tau {med.loc[14]:.4f}")
    if 69 in med.index:
        print(f"    {args.cohort} at block length 69: median tau {med.loc[69]:.4f}")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
