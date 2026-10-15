#!/usr/bin/env python3
"""
Measure non-normality directly, and test it against the per-subject
displacement threshold.

WHY
---
The Discussion proposes non-normality as the mechanistic route linking
degree and weight statistics to where the linear approximation stops
behaving well. Four referees independently objected that we never
measure it: it is a hypothesis presented in a mechanism's clothing.

Either it becomes a finding or the paragraph goes. This script decides.

The hypothesis is specific and falsifiable. Asllani et al. (2018) showed
that in non-normal systems small disturbances are strongly amplified
during a transient even when the system is linearly stable, and that
eigenvalues become poor guides to behaviour. Lindmark and Altafini
(2020) built driver-node centrality on a characterisation of network
non-normality. If non-normality is the route, subjects with MORE
non-normal operators should depart from linear behaviour at SMALLER
displacements.

WHICH MATRIX TO MEASURE -- this is the load-bearing choice
---------------------------------------------------------
Non-normality is a property of a specific matrix, and there are three
candidates here, which are NOT interchangeable:

  (a) the coupling matrix C after normalisation. Symmetric by
      construction, therefore exactly normal, therefore useless. Any
      non-normality in this system comes from the DYNAMICS, not from
      the connectome's symmetry.

  (b) the continuous-time Jacobian J at the fixed point. This is the
      operator whose linearization the whole analysis rests on. It is
      non-normal even for symmetric C, because the excitatory and
      inhibitory blocks enter asymmetrically and with different time
      constants.

  (c) the discretised A = expm(J*dt). This is what the control
      computation actually uses.

We measure (b) and (c) and report both. (b) is the more interpretable
and is the primary; (c) is what the Gramian is built from. If they
disagree the discrepancy is itself informative.

Reporting (a) as well, purely to make explicit that it is trivially
normal and cannot be the source -- a reviewer will otherwise ask.

MEASURES
--------
Three standard and complementary quantifications, because no single
scalar captures non-normality and reporting only the most favourable one
would be indefensible:

  Henrici departure from normality (Frobenius):
      d_F(M) = sqrt( ||M||_F^2 - sum_i |lambda_i|^2 )
  normalised by ||M||_F so it is comparable across subjects and sizes.
  Zero exactly when M is normal.

  Eigenvector conditioning: kappa(V), the condition number of the
  eigenvector matrix. One for a normal matrix; large when eigenvectors
  are far from orthogonal. This is the quantity that governs how far
  eigenvalue intuition can be trusted.

  Transient amplification: the maximum of ||expm(M*t)||_2 over t, which
  exceeds one for a non-normal stable matrix and equals one for a normal
  stable one. This is the mechanism Asllani et al. describe, measured
  directly rather than inferred from a proxy.

WHAT WOULD FALSIFY THE HYPOTHESIS
---------------------------------
A null or positive correlation between non-normality and the per-subject
displacement threshold. We pre-state the prediction here: NEGATIVE
correlation (more non-normal -> smaller threshold). Report the result
either way.

Note the threshold is a fragile per-subject quantity -- the paper says
so -- and a null may reflect that fragility rather than an absent
relationship. Both possibilities are reported; the power to detect is
bounded by the outcome's own reliability.

USAGE
-----
    python3 measure_non_normality.py \\
        --sc-dir <connectomes> --sc-suffix _SC_SIFT2_410.csv \\
        --w-ee-values 3.0,3.3402,3.6805,4.0207,4.361 \\
        --sweep <the cohort's fine sweep csv> \\
        --out nonnormality_HCP.csv
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from scipy.linalg import expm

from nctsens import simulation
BASE_PARAMS = dict(w_IE=4.0, w_EI=3.0, w_II=2.0, c_E=-2.0, c_I=-2.0,
                   tau_E=10.0, tau_I=20.0)
DT = 2.0


def load_and_scale_real_coupling(sc_path):
    SC = np.loadtxt(sc_path, delimiter=",")
    N = SC.shape[0]
    np.fill_diagonal(SC, 0)
    SC_log = np.where(SC > 0, np.log1p(SC), 0)
    target = (0.075 / N) * 20 * (N - 1)
    return SC_log * (target / SC_log.sum(axis=1).mean()), N


def henrici(M):
    """Departure from normality, Frobenius, normalised by ||M||_F."""
    fro = np.linalg.norm(M, "fro")
    if fro == 0:
        return 0.0
    ev = np.linalg.eigvals(M)
    inside = fro ** 2 - np.sum(np.abs(ev) ** 2)
    return float(np.sqrt(max(inside, 0.0)) / fro)


def eigvec_conditioning(M):
    """kappa(V). 1 for a normal matrix; large when eigenvectors are
    far from orthogonal."""
    try:
        _, V = np.linalg.eig(M)
        return float(np.linalg.cond(V))
    except np.linalg.LinAlgError:
        return np.nan


def numerical_abscissa(M):
    """Largest eigenvalue of the symmetric part, (M + M^T)/2.

    This is the initial growth rate of ||expm(Mt)||. It decides whether
    the hypothesised mechanism can operate AT ALL:

      negative -> ||expm(Mt)|| decreases monotonically from t = 0, so NO
                  transient amplification is possible, however
                  ill-conditioned the eigenvectors are.
      positive -> transient growth occurs even though the system is
                  asymptotically stable. This is the Asllani et al.
                  regime.

    Reporting eigenvector conditioning without this would be misleading:
    a large kappa(V) with a negative numerical abscissa produces no
    transient growth, and it was exactly this combination that made the
    amplification measure return unity in testing.
    """
    sym = (M + M.T) / 2.0
    return float(np.linalg.eigvalsh(sym).max())


def transient_amplification(M, t_max=None, n_t=60):
    """max_t ||expm(M t)||_2.

    Equals 1 for a normal stable matrix. Exceeding 1 is precisely the
    transient growth Asllani et al. (2018) identify, so this is the
    measure closest to the hypothesised mechanism.
    """
    try:
        ev = np.linalg.eigvals(M)
        slowest = np.max(ev.real)
        if slowest >= 0:
            return np.nan          # not stable; amplification unbounded
        if t_max is None:
            t_max = min(50.0, 8.0 / abs(slowest))
        ts = np.linspace(0.0, t_max, n_t)
        return float(max(np.linalg.norm(expm(M * t), 2) for t in ts))
    except Exception:
        return np.nan


def process_subject(args):
    sid, path, w_ee_values = args
    try:
        C, N = load_and_scale_real_coupling(path)
    except Exception:
        return []

    # (a) the coupling matrix itself -- reported to make explicit that it
    # is symmetric and therefore trivially normal.
    c_henrici = henrici(C)
    c_asym = float(np.abs(C - C.T).max())

    rows = []
    for w_EE in w_ee_values:
        try:
            net = simulation.WilsonCowanNetwork(
                n_regions=N, C=C,
                node_params=dict(w_EE=w_EE, **BASE_PARAMS))
            E_star, I_star = net.find_fixed_point()
            J = net.jacobian_at(E_star, I_star)
            if np.linalg.eigvals(J).real.max() >= 0:
                continue
            G = simulation.input_jacobian_at(net, E_star, I_star)
            A, _ = simulation.discretize_system(J, G, DT)
        except Exception:
            continue

        rows.append(dict(
            subject_id=sid, N=N, w_EE=w_EE,
            coupling_henrici=c_henrici, coupling_max_asymmetry=c_asym,
            J_henrici=henrici(J),
            J_eigvec_cond=eigvec_conditioning(J),
            J_transient_amp=transient_amplification(J),
            J_numerical_abscissa=numerical_abscissa(J),
            A_henrici=henrici(A),
            A_eigvec_cond=eigvec_conditioning(A),
            J_max_real_eig=float(np.linalg.eigvals(J).real.max()),
        ))
    return rows


def per_subject_threshold(sweep_csv):
    """Per-subject displacement at which that subject's interior minimum
    disappears, using the same last-descent convention as the paper.

    This is a FRAGILE quantity and the paper says so. A subject whose
    curve never crosses is returned as nan rather than imputed.
    """
    d = pd.read_csv(sweep_csv)
    g = (d.groupby(["subject_id", "target_scale", "w_EE"])
           .rel_reachability_error.mean().reset_index())
    out = {}
    for sid, s in g.groupby("subject_id"):
        ts_all, has_int = [], []
        for ts, s2 in s.groupby("target_scale"):
            s2 = s2.sort_values("w_EE")
            a = int(np.argmin(s2.rel_reachability_error.values))
            ts_all.append(ts)
            has_int.append(1.0 if 0 < a < len(s2) - 1 else 0.0)
        ts_all = np.asarray(ts_all, float)
        has_int = np.asarray(has_int, float)
        order = np.argsort(ts_all)
        x, y = np.log(ts_all[order]), has_int[order]
        idx = [k for k in range(len(x) - 1) if y[k] >= 0.5 > y[k + 1]]
        out[str(sid)] = (float(np.exp(x[idx[-1]])) if idx else np.nan)
    return pd.Series(out, name="subject_threshold")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sc-dir", required=True)
    ap.add_argument("--sc-suffix", required=True)
    ap.add_argument("--w-ee-values", required=True)
    ap.add_argument("--sweep", default=None,
                    help="The cohort's fine sweep CSV. Without it the script "
                         "only measures non-normality and runs no test.")
    ap.add_argument("--n-subjects", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    w_ee = [float(x) for x in args.w_ee_values.split(",")]
    files = sorted(f for f in os.listdir(args.sc_dir)
                   if f.endswith(args.sc_suffix))
    if args.n_subjects:
        files = files[:args.n_subjects]
    jobs = [(f[:-len(args.sc_suffix)], os.path.join(args.sc_dir, f), w_ee)
            for f in files]
    print(f"{len(jobs)} subjects x {len(w_ee)} coupling values")

    n_workers = args.n_workers or min(cpu_count(), len(jobs))
    rows = []
    with Pool(n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process_subject, jobs), 1):
            rows.extend(r)
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print("\n" + "=" * 74)
    print("NON-NORMALITY")
    print("=" * 74)
    print(f"\n  coupling matrix C (symmetric by construction):")
    print(f"    Henrici departure: max {df.coupling_henrici.max():.3e}")
    print(f"    max |C - C^T|:     {df.coupling_max_asymmetry.max():.3e}")
    print("    -> confirms C is normal. Any non-normality is from the")
    print("       DYNAMICS, not from the connectome.")

    for m, lab in [("J_henrici", "Henrici departure, Jacobian"),
                   ("J_eigvec_cond", "eigenvector conditioning kappa(V)"),
                   ("J_numerical_abscissa", "numerical abscissa (decides if growth is possible)"),
                   ("J_transient_amp", "max transient amplification"),
                   ("A_henrici", "Henrici departure, discretised A")]:
        v = df[m].dropna()
        if len(v):
            print(f"\n  {lab}:")
            print(f"    median {v.median():.4f}   IQR [{v.quantile(.25):.4f}, "
                  f"{v.quantile(.75):.4f}]   range [{v.min():.4f}, {v.max():.4f}]")

    na = df.J_numerical_abscissa.dropna()
    if len(na):
        pos = (na > 0).mean()
        print(f"\n  DECISIVE: numerical abscissa positive in {pos:.1%} of "
              f"subject-coupling combinations.")
        if pos < 0.01:
            print("  It is negative essentially everywhere, so transient")
            print("  amplification cannot occur in these systems and the")
            print("  non-normality mechanism proposed in the Discussion CANNOT")
            print("  operate as described. Eigenvector conditioning may still be")
            print("  large, but without a positive numerical abscissa it produces")
            print("  no transient growth. Drop or restate the hypothesis.")
        else:
            print("  Transient amplification is possible in these systems, so")
            print("  the proposed mechanism is at least admissible. The")
            print("  correlation test below decides whether it operates.")

    if not args.sweep:
        print("\n  No --sweep given: measurement only, no hypothesis test.")
        return

    print("\n" + "=" * 74)
    print("THE TEST: does non-normality predict the displacement threshold?")
    print("=" * 74)
    print("  Pre-stated prediction: NEGATIVE correlation.")
    print("  More non-normal -> departs from linear behaviour sooner.\n")

    thr = per_subject_threshold(args.sweep)
    subj = df.groupby(df.subject_id.astype(str)).median(numeric_only=True)
    j = subj.join(thr, how="inner").dropna(subset=["subject_threshold"])
    print(f"  subjects with an estimable threshold: {len(j)}/{len(subj)}")
    if len(j) < 10:
        print("  Too few for a meaningful test. The threshold is a fragile")
        print("  per-subject quantity; this is itself a reportable result.")
        return

    from scipy.stats import spearmanr
    print(f"\n  {'measure':34s} {'rho':>8} {'p':>10}  direction")
    for m in ("J_henrici", "J_eigvec_cond", "J_numerical_abscissa",
              "J_transient_amp", "A_henrici"):
        if m not in j:
            continue
        v = j[[m, "subject_threshold"]].dropna()
        if len(v) < 10:
            continue
        rho, p = spearmanr(v[m], v.subject_threshold)
        mark = "as predicted" if (rho < 0 and p < 0.05) else (
            "OPPOSITE" if (rho > 0 and p < 0.05) else "null")
        print(f"  {m:34s} {rho:>+8.3f} {p:>10.4f}  {mark}")

    print("\n  Bonferroni threshold for 5 measures: p < 0.01")
    print("\n  A null here does not by itself refute the hypothesis: the")
    print("  per-subject threshold is unstable across target direction and")
    print("  grid endpoint, so the test is attenuated by outcome unreliability")
    print("  as well as by any true absence of effect. Report both readings.")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
