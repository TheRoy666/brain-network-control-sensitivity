#!/usr/bin/env python3
"""
Separate LINEARIZATION error from NUMERICAL CONTROL error.

WHY THIS EXPERIMENT EXISTS
--------------------------
The validity sweep computes a minimum-energy control input from the
linearized system and injects it into the full nonlinear system. The
resulting relative endpoint error is then interpreted as the cost of
linearizing.

That interpretation has an unexamined premise. The error could arise
from either of two entirely different sources:

  (1) MODEL MISMATCH. The linear model is a poor description of the
      nonlinear dynamics at that displacement. This is a statement
      about the approximation.

  (2) NUMERICAL CONTROL FAILURE. The minimum-energy input itself is
      inaccurately computed, because the controllability Gramian is
      ill-conditioned and its inversion amplifies error. This is a
      statement about our implementation, not about linearization.

Sun and Motter (2013) showed that (2) occurs routinely in network
control even where formal controllability criteria are satisfied, and
cannot be fixed by more numerical precision. Our own sweeps report
Gramian condition numbers rising from ~85 to ~130 across the coupling
grid, so (2) is not hypothetical here.

The distinction is decisive for what the paper can claim. If the error
is (1), "linear validity has a boundary" is a defensible framing. If it
is largely (2), the honest claim is "the unregularized minimum-energy
implementation becomes unreliable as the Gramian becomes ill-conditioned"
-- a real result, but a different paper.

THE TEST
--------
The minimum-energy solution is constructed so that the DISCRETE LINEAR
system reaches the target exactly. Any deviation when the same input is
applied to the LINEAR system is therefore pure numerical error:
Gramian conditioning, inversion, discretization.

So for each condition we compute the same input once and evaluate it
twice:

    err_linear    = ||x_linear(T)    - x_target|| / ||delta_xT||
    err_nonlinear = ||x_nonlinear(T) - x_target|| / ||delta_xT||

and read off:

    err_linear    ~ 0   ->  the sweep's error is MODEL MISMATCH
    err_linear    large ->  the sweep's error is NUMERICAL
    difference          ->  the part attributable to nonlinearity

We also record the Gramian condition number per condition, so the
numerical component can be related to conditioning directly.

USAGE
-----
Run from this directory with nctsens installed:

    python3 linearization_vs_numerical_error.py \\
        --sc-dir <connectomes> --sc-suffix _SC_SIFT2_410.csv \\
        --w-ee-values 3.0,3.3402,3.6805,4.0207,4.361 \\
        --n-subjects 20 --n-workers 20 \\
        --out linvsnum_HCP.csv

Defaults match the fine sweep. Keep them matched or the comparison is
not against the reported numbers.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import zlib
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from nctsens import simulation
from nctsens import physics
# --- matched to wc_linear_validity_sweep_real_coupling.py ---
BASE_PARAMS = dict(w_IE=4.0, w_EI=3.0, w_II=2.0, c_E=-2.0, c_I=-2.0,
                   tau_E=10.0, tau_I=20.0)
DT = 2.0  # MUST match the sweep. An earlier version used 0.1, which
          # makes every horizon 20x shorter and inflates endpoint error.
T_VALUES = [5, 20, 50, 100, 200]
TARGET_SCALES = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035,
                 0.04, 0.05, 0.06, 0.07, 0.085, 0.1]
N_TARGET_DIRECTIONS_PER_SEED = 3


def load_and_scale_real_coupling(sc_path):
    SC = np.loadtxt(sc_path, delimiter=",")
    N = SC.shape[0]
    np.fill_diagonal(SC, 0)
    SC_log = np.where(SC > 0, np.log1p(SC), 0)
    target_row_sum_mean = (0.075 / N) * 20 * (N - 1)
    return SC_log * (target_row_sum_mean / SC_log.sum(axis=1).mean()), N


def build_network(N, w_EE, C):
    return simulation.WilsonCowanNetwork(
        n_regions=N, C=C, node_params=dict(w_EE=w_EE, **BASE_PARAMS))


def simulate_discrete_linear(A, B, U, x0, T):
    """Propagate the DISCRETE LINEAR system under the same input.

    This is the system the minimum-energy solution was derived for, so
    the endpoint should equal the target up to numerical error alone.
    """
    # U is shape (T, M) and is indexed U[k], matching zero_order_hold in
    # nctsens.physics. Indexing it as U[:, k] silently takes a slice of
    # the wrong axis and fails only when T happens to differ from M.
    x = np.asarray(x0, dtype=float).copy()
    for k in range(T):
        x = A @ x + B @ U[k]
    return x


def process_subject(args):
    sid, path, w_ee_values = args
    try:
        C, N = load_and_scale_real_coupling(path)
    except Exception:
        return []

    rows = []
    for w_EE in w_ee_values:
        try:
            net = build_network(N, w_EE, C)
            E_star, I_star = net.find_fixed_point()
            J = net.jacobian_at(E_star, I_star)
            if np.linalg.eigvals(J).real.max() >= 0:
                continue
            G = simulation.input_jacobian_at(net, E_star, I_star)
            A, B = simulation.discretize_system(J, G, DT)
        except Exception:
            continue

        x_star = np.concatenate([E_star, I_star])

        for T in T_VALUES:
            # Gramian conditioning is a property of (A, B, T) alone, so
            # compute it once per horizon rather than per target.
            try:
                W = physics.compute_gramian_doubling(A, B, T)
                cond = float(np.linalg.cond(W))
                eigmin = float(np.min(np.linalg.eigvalsh(
                    (W + W.T) / 2.0)))
            except Exception:
                cond, eigmin = np.nan, np.nan

            for target_scale in TARGET_SCALES:
                for dir_seed in range(N_TARGET_DIRECTIONS_PER_SEED):
                    seed_key = (f"{sid}|{int(w_EE * 10)}|{T}|"
                                f"{int(target_scale * 1000)}|{dir_seed}")
                    drng = np.random.default_rng(
                        zlib.crc32(seed_key.encode("utf-8")) % (2**32))
                    direction = drng.standard_normal(N)
                    direction /= np.linalg.norm(direction)

                    delta_x0 = np.zeros(2 * N)
                    delta_xT = np.concatenate(
                        [target_scale * direction, np.zeros(N)])

                    try:
                        energy, U = physics.minimum_energy_trajectory(
                            A, B, delta_x0, delta_xT, T)
                    except Exception:
                        continue

                    movement = np.linalg.norm(delta_xT)

                    # The minimum-energy solution inverts the Gramian with
                    # a pseudo-inverse truncated at rcond=1e-10. That
                    # truncation is itself a numerical choice, and it is
                    # exactly where ill-conditioning would bite. Recompute
                    # at a stricter and a looser setting: if the endpoint
                    # error on the LINEAR system moves with rcond, the
                    # conditioning is doing real work.
                    err_lin_rcond = {}
                    for rc in (1e-8, 1e-12):
                        try:
                            _, U_rc = physics.minimum_energy_trajectory(
                                A, B, delta_x0, delta_xT, T, rcond=rc)
                            dx_rc = simulate_discrete_linear(
                                A, B, U_rc, delta_x0, T)
                            err_lin_rcond[rc] = (
                                np.linalg.norm(dx_rc - delta_xT) / movement)
                        except Exception:
                            err_lin_rcond[rc] = np.nan

                    # (a) the LINEAR system the solution was derived for
                    dx_lin = simulate_discrete_linear(A, B, U, delta_x0, T)
                    err_linear = np.linalg.norm(dx_lin - delta_xT) / movement

                    # (b) the true NONLINEAR system, as in the sweep
                    try:
                        u_func = physics.zero_order_hold(U, DT)
                        res = net.simulate_controlled(
                            u_func=u_func, t_span=(0.0, T * DT),
                            n_points=max(100, T), E0=E_star, I0=I_star)
                        realized = np.concatenate(
                            [res["E"][:, -1], res["I"][:, -1]])
                        err_nonlinear = (
                            np.linalg.norm(realized - (x_star + delta_xT))
                            / movement)
                    except Exception:
                        err_nonlinear = np.nan

                    rows.append(dict(
                        subject_id=sid, w_EE=w_EE, T=T,
                        target_scale=target_scale, dir_seed=dir_seed,
                        energy=energy,
                        gramian_cond=cond, gramian_min_eig=eigmin,
                        err_linear=err_linear,
                        err_linear_rcond_1e8=err_lin_rcond.get(1e-8, np.nan),
                        err_linear_rcond_1e12=err_lin_rcond.get(1e-12, np.nan),
                        err_nonlinear=err_nonlinear,
                        # the part attributable to nonlinearity alone
                        err_excess=err_nonlinear - err_linear,
                    ))
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sc-dir", required=True)
    ap.add_argument("--sc-suffix", required=True)
    ap.add_argument("--w-ee-values", required=True,
                    help="Comma-separated. MUST match the cohort's sweep grid.")
    ap.add_argument("--n-subjects", type=int, default=20,
                    help="A subsample is adequate: this measures a property "
                         "of the numerics, not a population effect.")
    ap.add_argument("--n-workers", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    w_ee = [float(x) for x in args.w_ee_values.split(",")]
    files = sorted(f for f in os.listdir(args.sc_dir)
                   if f.endswith(args.sc_suffix))[:args.n_subjects]
    jobs = [(f[:-len(args.sc_suffix)], os.path.join(args.sc_dir, f), w_ee)
            for f in files]
    print(f"{len(jobs)} subjects, w_EE {w_ee}, "
          f"{len(TARGET_SCALES)} displacements, {len(T_VALUES)} horizons")

    n_workers = args.n_workers or min(cpu_count(), len(jobs))
    rows = []
    with Pool(n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process_subject, jobs), 1):
            rows.extend(r)
            print(f"  {i}/{len(jobs)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print("\n" + "=" * 74)
    print("ERROR DECOMPOSITION")
    print("=" * 74)
    print(f"  rows: {len(df)}")
    print(f"\n  err_linear    (numerical only): median {df.err_linear.median():.6f}"
          f"   p95 {df.err_linear.quantile(.95):.6f}   max {df.err_linear.max():.6f}")
    print(f"  err_nonlinear (as reported):    median {df.err_nonlinear.median():.6f}"
          f"   p95 {df.err_nonlinear.quantile(.95):.6f}")
    share = (df.err_linear / df.err_nonlinear).replace([np.inf, -np.inf], np.nan)
    print(f"\n  numerical share of reported error: median {share.median():.1%}"
          f"   p95 {share.quantile(.95):.1%}")

    print("\n  by target displacement:")
    print(f"   {'ts':>7} {'err_linear':>12} {'err_nonlin':>12} {'num share':>10}")
    for ts, g in df.groupby("target_scale"):
        s = (g.err_linear / g.err_nonlinear).replace([np.inf, -np.inf], np.nan)
        print(f"   {ts:7.3f} {g.err_linear.median():12.6f} "
              f"{g.err_nonlinear.median():12.6f} {s.median():9.1%}")

    print("\n  by coupling (conditioning rises with w_EE):")
    print(f"   {'w_EE':>8} {'cond':>12} {'err_linear':>12} {'num share':>10}")
    for w, g in df.groupby("w_EE"):
        s = (g.err_linear / g.err_nonlinear).replace([np.inf, -np.inf], np.nan)
        print(f"   {w:8.4f} {g.gramian_cond.median():12.1f} "
              f"{g.err_linear.median():12.6f} {s.median():9.1%}")

    print("\n  pseudo-inverse truncation sensitivity (err on the LINEAR system):")
    for c, lab in [("err_linear_rcond_1e8", "rcond=1e-8 (looser)"),
                   ("err_linear", "rcond=1e-10 (as used)"),
                   ("err_linear_rcond_1e12", "rcond=1e-12 (stricter)")]:
        if c in df:
            print(f"    {lab:24s} median {df[c].median():.6f}   "
                  f"p95 {df[c].quantile(.95):.6f}")

    # Sanity gate: this run must reproduce the published sweep's
    # err_nonlinear distribution, or err_linear is being measured on
    # different solutions than the paper reports. DT is the parameter
    # most likely to break this and it must equal the sweep's DT.
    print(f"\n  SANITY: err_nonlinear median {df.err_nonlinear.median():.4f}, "
          f"p95 {df.err_nonlinear.quantile(.95):.4f}")
    print("  Compare against the published sweep for the same cohort. If these")
    print("  distributions disagree, this run is NOT measuring the reported")
    print("  solutions and the decomposition is invalid.")

    ok = df.err_linear.median() < 0.01 * df.err_nonlinear.median()
    print("\n" + "-" * 74)
    if ok:
        print("  VERDICT: numerical error is negligible against reported error.")
        print("  The sweep's error is MODEL MISMATCH. The linearization framing")
        print("  is supported.")
    else:
        print("  VERDICT: numerical error is NOT negligible. A material share of")
        print("  the reported error comes from ill-conditioned minimum-energy")
        print("  computation rather than from linearization. The claim must be")
        print("  restated as a property of the implementation, not of the")
        print("  linear approximation.")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
