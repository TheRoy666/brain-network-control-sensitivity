#!/usr/bin/env python3
"""
Is the FC-versus-EC energy divergence an OPERATOR-CLASS effect or an
ESTIMATION-LENGTH effect?

THE PROBLEM
-----------
The manuscript's first claim is that operator choice changes control
energy by three to one hundred fold. The median teleportation ratio is:

    HCP           3.2x     T = 4800 timepoints
    ADNI         12-13x    T ~ 197
    UNAM-TLE       66x     T shortest, coarsest parcellation

The gradient tracks recording length across every cohort. Five referees
independently observed that operator class and estimation quality are
therefore not separable in these data, and that the claim leads the
abstract while the confound is admitted in the same breath.

Two readings are available and they imply opposite remedies:

  (a) OPERATOR CLASS. Symmetric operators genuinely misrepresent
      directed dynamics, and the effect is larger where the underlying
      connectivity is more directed. Remedy: use directed operators.

  (b) ESTIMATION LENGTH. The directed estimator becomes increasingly
      unstable as T/N worsens, inflating the divergence. Remedy: acquire
      longer scans; the operator choice may be secondary.

THE TEST
--------
Truncate HCP to ADNI's recording length and recompute the ratio, holding
everything else fixed: same subjects, same connectomes, same
parcellation, same estimator, same regularisation, same horizon, same
brain-state definitions.

    ratio stays near 3.2x   ->  reading (a). The divergence is a
                                property of the operator class, and the
                                cross-cohort gradient reflects genuine
                                differences in directedness.

    ratio rises toward 12x  ->  reading (b). The divergence is
                                substantially an estimation-regime
                                phenomenon, and the abstract's leading
                                claim must be restated.

Either outcome is publishable. (b) is arguably the more interesting,
because it would mean the framework's apparent sensitivity to operator
choice is partly a sensitivity to data adequacy -- which is the same
lesson the rank-stability result teaches, arriving by a second route.

DESIGN NOTES
------------
Truncation takes the FIRST n timepoints rather than a random window, so
the comparison is nested: the truncated series is a prefix of the full
one and no resampling variance is introduced. A stride option is also
provided, because truncation and subsampling confound differently --
truncation reduces sample count at fixed sampling rate, stride reduces
effective sampling rate at fixed duration. The rank-stability analysis
used both and they agreed; the same check applies here.

Brain states are recomputed from the truncated series rather than
carried over from the full one. Carrying them over would leak
information from timepoints the estimator never saw and would make the
truncated condition artificially favourable.

USAGE
-----
    python3 operator_divergence_vs_length.py \\
        --ts-dir <HCP timeseries> --ts-suffix _timeseries.npy \\
        --sc-dir <HCP SC matrices> --sc-suffix _SC_SIFT2_410.csv \\
        --lengths 4800,1200,400,200,197 \\
        --n-subjects 40 --n-workers 40 \\
        --out operator_divergence_vs_length_HCP.csv
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

try:                                    # repository layout
    from nctsens.connectivity import (graphnet_effective_connectivity,
                                       functional_connectivity)
    from nctsens.physics import minimum_energy, normalise_matrix
    from nctsens.loader import from_arrays, load_connectome
except ImportError:                     # flat layout, as the published run used
    from nctsens.connectivity import (graphnet_effective_connectivity,
                              functional_connectivity)
    from nctsens.physics import minimum_energy, normalise_matrix
    from nctsens.loader import from_arrays, load_connectome

TARGET_RHO = 0.9   # matches --target-rho default in run_hcp_aud_batch.py.
                   # An earlier version used 0.95, which rescales both
                   # operators differently and makes the energies
                   # incomparable with the published ratios.
HORIZON = 10


def define_states(X):
    """Rest and craving states, copied verbatim from run_hcp_aud_batch.py.

    Rest is the mean over the 20 per cent of timepoints with the LOWEST
    cross-regional variance; craving the mean over the highest 20 per
    cent. Both are unit-normalised.

    This must match the batch script exactly. An earlier version of this
    file used a mean-plus-or-minus-one-standard-deviation contrast, which
    is a different transition entirely and would have produced energies
    not comparable with the published ratios.

    States are recomputed per truncation length rather than carried over
    from the full series, so that no information from timepoints the
    estimator never saw can enter.
    """
    N, T = X.shape
    n = max(1, T // 5)
    tvar = X.var(axis=0)
    idx = np.argsort(tvar)
    x_rest = X[:, idx[:n]].mean(axis=1)
    x_craving = X[:, idx[-n:]].mean(axis=1)
    x_rest /= (np.linalg.norm(x_rest) + 1e-8)
    x_craving /= (np.linalg.norm(x_craving) + 1e-8)
    return x_rest, x_craving


def orient(X_raw, n_nodes):
    """Return (regions x time), resolved against the REGION COUNT.

    Never compare dimensions: a size heuristic silently transposes
    whenever timepoints fall below regions, which is exactly the regime
    this script truncates into. This is the fix Rhi applied before the
    published run; an earlier version of this file lacked it.
    """
    if X_raw.shape[0] == n_nodes:
        return X_raw
    if X_raw.shape[1] == n_nodes:
        return X_raw.T
    raise ValueError(f"neither dimension of X {X_raw.shape} matches {n_nodes} regions")


def process_subject(args):
    sid, ts_path, sc_path, lengths, stride, freeze = args
    try:
        X_raw = np.load(ts_path)
        SC_raw = load_connectome(sc_path)
    except Exception as e:
        print(f"Loading failed for {sid}: {e}")
        return []

    X_full = orient(X_raw, SC_raw.shape[0])
    # FROZEN STATES: computed once from the FULL recording, then reused at
    # every length, so that only the operator estimate changes with T.
    # float64 exactly as from_arrays() produces, so that at full length the
    # frozen states are bit-identical to the recomputed ones. If the stored
    # series is float32, computing states on it directly would differ by
    # ~1e-7 and break the full-length sanity gate.
    frozen = define_states(np.asarray(X_full, dtype=float)) if freeze else None

    rows = []
    for L in lengths:
        try:
            Xt = X_full[:, :L]
            if stride > 1:
                Xt = Xt[:, ::stride]
            data = from_arrays(X=Xt, SC=SC_raw, subject_id=sid, validate=True)
            X, SC, N = data["X"], data["SC"], data["N"]
            T_eff = X.shape[1]
            if T_eff < 20:
                continue

            EC = graphnet_effective_connectivity(X, SC, lambda_ridge=1.0,
                                                 lambda_graph=1.0)
            A_ec = normalise_matrix(EC, target_rho=TARGET_RHO)
            FC = functional_connectivity(X)
            A_fc = normalise_matrix(FC, target_rho=TARGET_RHO)

            B = np.eye(N)
            x_rest, x_craving = frozen if freeze else define_states(X)
            E_ec, _ = minimum_energy(A_ec, B, x_craving, x_rest, T=HORIZON)
            E_fc, _ = minimum_energy(A_fc, B, x_craving, x_rest, T=HORIZON)

            rows.append(dict(
                subject_id=sid, requested_length=L, T_effective=T_eff,
                N=N, stride=stride, states="frozen" if freeze else "recomputed",
                E_ec=float(E_ec), E_fc=float(E_fc),
                teleport_ratio=float(E_fc / (E_ec + 1e-12)),
                ec_asymmetry=float(np.linalg.norm(EC - EC.T, "fro")
                                   / (np.linalg.norm(EC, "fro") + 1e-12)),
                raw_rho=float(np.max(np.abs(np.linalg.eigvals(EC)))),
                T_over_N=float(T_eff / N),
            ))
        except Exception as e:
            print(f"Processing failed for {sid} at length {L}: {e}")
            continue
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ts-dir", required=True)
    ap.add_argument("--ts-suffix", required=True)
    ap.add_argument("--sc-dir", required=True)
    ap.add_argument("--sc-suffix", required=True)
    ap.add_argument("--lengths", default="4800,1200,400,200,197",
                    help="Timepoint counts to truncate to. Include the full "
                         "length so the untruncated ratio is reproduced.")
    ap.add_argument("--stride", type=int, default=1,
                    help="Subsample instead of truncating. Stride reduces "
                         "sampling rate at fixed duration; truncation reduces "
                         "sample count at fixed rate. They confound "
                         "differently and should agree.")
    ap.add_argument("--n-subjects", type=int, default=0,
                    help="0 = all subjects. The full run costs about the same "
                         "as one batch run, since it is dominated by the "
                         "full-length condition; truncated conditions are "
                         "cheaper. There is no good reason to subsample.")
    ap.add_argument("--n-workers", type=int, default=0)
    ap.add_argument("--freeze-states", action="store_true",
                    help="Compute the two states ONCE from the full recording "
                         "and reuse them at every length, so only the operator "
                         "estimate varies. Default off reproduces the "
                         "published run, where states are recomputed.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]
    files = sorted(f for f in os.listdir(args.ts_dir)
                   if f.endswith(args.ts_suffix))
    if args.n_subjects:
        files = files[:args.n_subjects]
    jobs = []
    for f in files:
        sid = f[:-len(args.ts_suffix)]
        sc = os.path.join(args.sc_dir, sid + args.sc_suffix)
        if os.path.exists(sc):
            jobs.append((sid, os.path.join(args.ts_dir, f), sc, lengths,
                         args.stride, args.freeze_states))
    print(f"{len(jobs)} subjects x {len(lengths)} lengths, stride {args.stride}, "
          f"states {'FROZEN at full length' if args.freeze_states else 'recomputed per length'}")
    if not jobs:
        raise SystemExit("No subject had both a timeseries and a connectome.")

    n_workers = args.n_workers or min(cpu_count(), len(jobs))
    rows = []
    with Pool(n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process_subject, jobs), 1):
            rows.extend(r)
            if i % 5 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print("\n" + "=" * 76)
    print("OPERATOR DIVERGENCE VERSUS RECORDING LENGTH")
    print("=" * 76)
    print(f"\n  {'T':>7} {'T/N':>7} {'n':>5} {'ratio median':>14} "
          f"{'IQR':>22} {'>1':>6}")
    for L, g in df.groupby("T_effective"):
        r = g.teleport_ratio
        print(f"  {L:>7} {g.T_over_N.iloc[0]:>7.2f} {len(g):>5} "
              f"{r.median():>14.2f} "
              f"[{r.quantile(.25):>8.2f}, {r.quantile(.75):>8.2f}] "
              f"{(r > 1).mean():>5.0%}")

    print("\n  PAIRED CHANGE (the design is within-subject: same subjects,")
    print("  same connectomes, only T varies)")
    piv = df.pivot_table(index="subject_id", columns="T_effective",
                         values="teleport_ratio")
    if piv.shape[1] >= 2:
        full_T, short_T = max(piv.columns), min(piv.columns)
        pair = piv[[full_T, short_T]].dropna()
        if len(pair):
            from scipy.stats import wilcoxon
            d = pair[short_T] - pair[full_T]
            lr = np.log(pair[short_T] / pair[full_T])
            try:
                _, pw = wilcoxon(pair[short_T], pair[full_T])
            except Exception:
                pw = np.nan
            print(f"    n paired = {len(pair)}   T {full_T} -> {short_T}")
            print(f"    ratio increased in {(d > 0).mean():.0%} of subjects")
            print(f"    median log-ratio change {lr.median():+.3f} "
                  f"(fold {np.exp(lr.median()):.2f}x)   Wilcoxon p = {pw:.2e}")
            print("    log-ratios are reported because fold-changes with small")
            print("    denominators are unstable as raw ratios.")

    print("\n  reference values from the manuscript:")
    print("    HCP full length   3.22x   (T = 4800)")
    print("    ADNI             12.4 to 13.5x   (T ~ 197)")
    print("    UNAM-TLE         66.0x")

    full = df[df.T_effective == df.T_effective.max()].teleport_ratio.median()
    short = df[df.T_effective == df.T_effective.min()].teleport_ratio.median()
    print("\n" + "-" * 76)
    print(f"  full length {full:.2f}x  ->  shortest {short:.2f}x   "
          f"(factor {short / full:.2f})")
    # No verdict is printed here. An earlier version judged the result from
    # the absolute ratio at the shortest length, which cannot tell a change
    # in the operator estimates from a change in the states being compared.
    # Run with --freeze-states and read the comparison from
    # compare_frozen_vs_published.py, whose decision rule was fixed in advance.
    print("\n  Interpretation: compare a --freeze-states run against the default")
    print("  run with compare_frozen_vs_published.py.")

    print("\n  Also inspect ec_asymmetry and raw_rho by length: if the estimated")
    print("  operator becomes more asymmetric or its spectral radius drifts as T")
    print("  falls, that is the estimator destabilising rather than the")
    print("  connectivity changing.")
    for c in ("ec_asymmetry", "raw_rho"):
        if c in df:
            print(f"\n  {c} by T:")
            for L, g in df.groupby("T_effective"):
                print(f"    T={L:>6}  median {g[c].median():.4f}")

    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
