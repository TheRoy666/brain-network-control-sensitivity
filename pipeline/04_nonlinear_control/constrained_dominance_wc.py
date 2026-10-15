#!/usr/bin/env python3
"""
Constrained dominance, Wilson-Cowan arm.

WHAT THIS ANSWERS
-----------------
The matched-cost analysis swept ten effort penalties from four
initialisations and declared a condition dominated if ANY resulting run
beat the linear solution on both energy and error. Four referees
observed that roughly forty attempts per condition makes the resulting
rate an upper bound inflated by search.

This script removes the search. Once per condition it solves

    minimise   || x(T) - x_target ||^2
    subject to integrated control energy <= E_ltv

from a single deterministic start. A feasible solution with lower
terminal error dominates by construction, so there is nothing to select
over.

WHAT DIFFERS IN THIS ARM, AND WHY IT MATTERS
--------------------------------------------
**This is a network, not a single node.** Ten regions, twenty states,
and a control input per region, so the decision variable is 25 x 10
rather than 25 x 1. The constraint and its gradient are unchanged in
form but act on a much larger vector.

**The original grid warm-started from U_NAIVE**, not from the LTV
solution as the other two arms did. That is why the tautology in this
arm guaranteed error no worse than NAIVE. The cold start used here
removes the question entirely.

**run_ltv_control in wc_oscillatory_linearization_comparison discards
the control sequence**, returning only its energy and final state. The
LTV leg is therefore reimplemented inline so the sequence is available;
any change to the original must be mirrored here.

**Deviation is measured from the free-running reference.** The target is
expressed relative to where the uncontrolled system would arrive, not
relative to the starting state, so the linear problem is posed on
x_target - x_natural. This is the convention of the original grid and is
preserved.

**Seeds without a stable reference phase are skipped**, as in the
original, so the realised condition count may fall below 8 x 5 = 40.

A NOTE ON THE EXISTING WC RESULTS
---------------------------------
The original WC A2 grid is partly reconstructed: 32 of its 40 rows were
recovered from printed output at one-decimal precision after an
execution limit truncated the run. Nothing from that file should be
quoted alongside output from this script.

USAGE
-----
Run from a directory holding wc_oscillatory_linearization_comparison.py,
adjoint_gradient.py and adjoint_gradient_wc_verification.py, with
nctsens installed:

    python3 constrained_dominance_wc.py \\
        --n-workers 40 --out constrained_dominance_wc.csv
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Matched to wc_a2_broader_grid.py so conditions correspond.
N = 10
DT = 2.0
T_STEPS = 25
PERIOD = 99.96
SEEDS = list(range(8))
SHIFTS_MS = [10, 20, 30, 40, 60]
MAXITER = 500


def control_energy(U, dt):
    return float(np.sum(U ** 2) * dt)


def ltv_control_sequence(net, t_arr, E_arr, I_arr, t_ref, x_target,
                         x_natural):
    """LTV leg, reimplemented to RETURN the control sequence.

    The original run_ltv_control computes U and then discards it,
    returning only energy and final state. Deviation is tracked relative
    to the moving reference, so the target deviation is
    x_target - x_natural rather than x_target - x_ref.
    """
    from nctsens import physics
    from nctsens import simulation
    import wc_oscillatory_linearization_comparison as wc_a2

    A_list, B_list = [], []
    for k in range(T_STEPS):
        E_k, I_k = wc_a2.state_at_time(t_arr, E_arr, I_arr, t_ref + k * DT)
        J_k = net.jacobian_at(E_k, I_k)
        G_k = simulation.input_jacobian_at(net, E_k, I_k)
        A_k, B_k = simulation.discretize_system(J_k, G_k, DT)
        A_list.append(A_k)
        B_list.append(B_k)
    return physics.minimum_energy_trajectory_ltv(
        A_list, B_list, np.zeros(2 * N), x_target - x_natural)


def run_condition(job):
    seed, shift_ms = job
    from nctsens import physics
    from nctsens import simulation
    import wc_oscillatory_linearization_comparison as wc_a2
    from adjoint_gradient import compute_adjoint_gradient
    from adjoint_gradient_wc_verification import (make_wc_jacobian_func,
                                                  make_wc_input_jacobian_func)

    row = dict(condition_id=f"seed{seed}|shift{shift_ms}",
               seed=seed, shift_ms=shift_ms)
    try:
        net = wc_a2.build_network(seed)
        t_arr, E_arr, I_arr = wc_a2.get_reference_trajectory(net, seed)
        t_ref, _ = wc_a2.find_stable_reference_phase(
            net, t_arr, E_arr, I_arr, PERIOD)
        if t_ref is None:
            row["error"] = "no stable reference phase; skipped"
            return row

        T_ms = T_STEPS * DT
        E_ref, I_ref = wc_a2.state_at_time(t_arr, E_arr, I_arr, t_ref)
        x_ref = np.concatenate([E_ref, I_ref])
        E_tgt, I_tgt = wc_a2.state_at_time(t_arr, E_arr, I_arr,
                                           t_ref + T_ms + shift_ms)
        x_target = np.concatenate([E_tgt, I_tgt])
        E_nat, I_nat = wc_a2.state_at_time(t_arr, E_arr, I_arr, t_ref + T_ms)
        x_natural = np.concatenate([E_nat, I_nat])
        target_distance = float(np.linalg.norm(x_target - x_natural))
        row["target_distance"] = target_distance

        def inject(U):
            u_func = physics.zero_order_hold(U, DT)
            r = net.simulate_controlled(u_func=u_func, t_span=(0.0, T_ms),
                                        n_points=max(100, T_STEPS),
                                        E0=E_ref, I0=I_ref)
            xf = np.concatenate([r["E"][:, -1], r["I"][:, -1]])
            return float(np.linalg.norm(xf - x_target) / target_distance)

        # --- naive LTI: Jacobian frozen at the reference state ---
        J0 = net.jacobian_at(E_ref, I_ref)
        G0 = simulation.input_jacobian_at(net, E_ref, I_ref)
        A, B = simulation.discretize_system(J0, G0, DT)
        _, U_naive = physics.minimum_energy_trajectory(
            A, B, np.zeros(2 * N), x_target - x_ref, T_STEPS)
        naive_err = inject(U_naive)
        naive_energy = control_energy(U_naive, DT)

        # --- LTV: the budget and the target ---
        _, U_ltv = ltv_control_sequence(net, t_arr, E_arr, I_arr, t_ref,
                                        x_target, x_natural)
        ltv_err = inject(U_ltv)
        budget = control_energy(U_ltv, DT)
        row.update(naive_err=naive_err, naive_energy=naive_energy,
                   ltv_err=ltv_err, ltv_energy=budget)

        jac_f = make_wc_jacobian_func(net)
        inp_f = make_wc_input_jacobian_func(net)

        def objective(u_flat):
            U = u_flat.reshape(T_STEPS, N)
            c, g, _ = compute_adjoint_gradient(net, x_ref, x_target, U, DT,
                                               jac_f, inp_f, lambda_reg=0.0)
            return c, g.flatten()

        def e_val(u):
            return float(np.sum(u ** 2) * DT)

        def e_grad(u):
            return 2.0 * u * DT

        res = minimize(
            objective, np.zeros(T_STEPS * N), jac=True, method="SLSQP",
            constraints=[{"type": "ineq",
                          "fun": lambda u: budget - e_val(u),
                          "jac": lambda u: -e_grad(u)}],
            options={"maxiter": MAXITER, "ftol": 1e-12})

        U = res.x.reshape(T_STEPS, N)
        used = control_energy(U, DT)
        err = inject(U)
        feasible = bool(used <= budget * (1 + 1e-6))
        row.update(
            constrained_err=err, constrained_energy=used,
            energy_frac=used / budget if budget else np.nan,
            peak_amplitude=float(np.max(np.abs(U))),
            feasible=feasible, converged=bool(res.success),
            n_iter=int(res.nit), solver_msg=str(res.message)[:60],
            dominates_ltv=bool(feasible and err < ltv_err),
            dominates_naive=bool(used <= naive_energy * (1 + 1e-6)
                                 and err < naive_err),
            # the linear controller is USABLE only if it beats no control
            ltv_usable=bool(ltv_err < 1.0))
    except Exception as exc:
        row.update(constrained_err=np.nan, error=str(exc)[:90])
    return row


def summarise(df):
    ok = df.dropna(subset=["constrained_err"])
    print("\n" + "=" * 76)
    print("CONSTRAINED DOMINANCE — WILSON-COWAN")
    print("=" * 76)
    print(f"  conditions {len(df)}   solved {len(ok)}", end="")
    if not len(ok):
        print("\n  nothing solved")
        return
    print(f"   feasible {int(ok.feasible.sum())}"
          f"   converged {int(ok.converged.sum())}")

    f = ok[ok.feasible]
    d = f.dominates_ltv.dropna()
    print(f"\n  DOMINATES LTV, all conditions: {int(d.sum())}/{len(d)}"
          f" = {d.mean():.0%}")

    print("\n  SPLIT BY WHETHER THE LINEAR CONTROLLER IS USABLE")
    print("  A relative error of 1.0 means the controlled endpoint is no")
    print("  closer to target than the uncontrolled one. Where the linear")
    print("  controller fails that test, dominating it is not informative:")
    print("  doing nothing dominates it too.")
    for lab, sub in [("linear usable (err < 1)", f[f.ltv_usable]),
                     ("linear fails  (err >= 1)", f[~f.ltv_usable])]:
        v = sub.dominates_ltv.dropna()
        if len(v):
            print(f"    {lab:26s} {int(v.sum())}/{len(v)} = {v.mean():.0%}"
                  f"   median target distance {sub.target_distance.median():.3f}")

    u = f[f.ltv_usable]
    if len(u):
        r = (u.constrained_err / u.ltv_err).replace([np.inf, -np.inf],
                                                    np.nan).dropna()
        if len(r):
            print(f"\n  among usable conditions, error ratio at or below the")
            print(f"  linear energy: median {r.median():.4f}"
                  f"   log-ratio median {np.log(r).median():+.3f}")
        print(f"  energy used as a fraction of budget: median "
              f"{u.energy_frac.median():.3f}")
        print(f"  peak control amplitude: median {u.peak_amplitude.median():.3g}"
              f"   max {u.peak_amplitude.max():.3g}")

    print("\n  sweep-based rates for comparison: 37/40 raw,"
          " 17/20 in usable conditions")
    print("  FitzHugh-Nagumo constrained result: 40/40")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-workers", type=int, default=0)
    ap.add_argument("--max-conditions", type=int, default=0,
                    help="Limit conditions, for smoke tests. 0 = all 40.")
    args = ap.parse_args()

    combos = [(s, sh) for s in SEEDS for sh in SHIFTS_MS]
    if args.max_conditions:
        combos = combos[:args.max_conditions]
    print(f"{len(combos)} conditions, ONE constrained optimisation each"
          f"   (N={N}, T_steps={T_STEPS})")

    n_workers = args.n_workers or min(cpu_count(), len(combos))
    print(f"  {n_workers} workers")
    rows = []
    with Pool(n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_condition, combos), 1):
            rows.append(r)
            print(f"  {i}/{len(combos)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    summarise(df)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
