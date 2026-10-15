#!/usr/bin/env python3
"""
Constrained dominance, Stuart-Landau arm.

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
terminal error dominates by construction: it cannot be spending more,
and it achieves less error. One optimisation, one answer, nothing to
select over.

DESIGN CHOICES
--------------
**One start, the zero control.** It is the only initialisation
guaranteed feasible for any positive budget, and the only one that
borrows nothing from either linear solution. Starting from a scaled
linear solution would import the information the comparison is meant to
be independent of; starting from several would reintroduce the
multiplicity this script exists to remove.

**The budget is the LTV solution's energy**, since LTV is the stronger
linear method and therefore the harder target. Dominance over the naive
solution is reported alongside but is not the headline.

**SLSQP with analytic gradients throughout.** The adjoint supplies the
error gradient; the energy gradient is 2*U*dt in closed form.

WHY STUART-LANDAU SPECIFICALLY
------------------------------
It is the cleanest of the three model arms: the limit cycle and the
Jacobian are both closed-form, so unlike the FitzHugh-Nagumo arm there
is no interpolated nominal orbit and no spline error entering the
comparison.

The beta sweep carries a degeneracy that governs how results may be
reported. With the quintic coefficient at zero, substituting a rescaled
state removes beta from the dynamics, so the three supercritical
conditions are one system up to rescaling. Absolute errors and energies
scale with beta while dimensionless ratios do not. Results are therefore
reported BY REGIME, with the supercritical result quoted once at the
beta = 1 anchor, and are never pooled across all six conditions.

USAGE
-----
Run from a directory holding stuart_landau_model.py and
adjoint_gradient.py, with nctsens installed:

    python3 constrained_dominance_sl.py \\
        --n-workers 40 --out constrained_dominance_sl.csv
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Matched to matched_cost_control_comparison_sl.py so conditions correspond.
DT = 0.5
T_TIME = 15.0
N_STEPS = int(T_TIME / DT)
G_CONST = np.array([[1.0], [0.0]])
OMEGA = 0.05
A_OSC = 0.2
PHASE_SHIFT_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5]
BETA_SWEEP_PHASES = [0.0, 0.125, 0.25, 0.375]
BETA_CONDITIONS = [
    {"beta": 2.0,  "gamma": 0.0, "regime": "supercritical_strong"},
    {"beta": 1.0,  "gamma": 0.0, "regime": "supercritical_ref"},
    {"beta": 0.5,  "gamma": 0.0, "regime": "supercritical_weak"},
    {"beta": -0.5, "gamma": 1.0, "regime": "subcritical_weak"},
    {"beta": -1.0, "gamma": 1.0, "regime": "subcritical_ref"},
    {"beta": -2.0, "gamma": 1.0, "regime": "subcritical_strong"},
]
MAXITER = 500


def control_energy(U, dt):
    return float(np.sum(U ** 2) * dt)


def make_net(beta, gamma):
    from stuart_landau_model import StuartLandauNetwork
    return StuartLandauNetwork(n_regions=1, C=np.zeros((1, 1)), a=A_OSC,
                               omega=OMEGA, beta=beta, gamma=gamma, G=0.0)


def orbit_state(net, phase_time):
    """Exact point on the limit cycle. Closed form, no interpolation."""
    angle = 2.0 * np.pi * (phase_time % net.period) / net.period
    return net.nominal_orbit(angle)


def run_condition(job):
    beta, gamma, regime, sp_frac, shift_frac = job
    from nctsens import physics
    from nctsens.simulation import discretize_system
    from adjoint_gradient import compute_adjoint_gradient

    net = make_net(beta, gamma)
    start_t = sp_frac * net.period
    shift_t = shift_frac * net.period
    label = f"{regime}|b{beta}"
    cid = f"{label}|{start_t:.4f}|{shift_t:.4f}"

    row = dict(condition_id=cid, condition_label=label, regime=regime,
               beta=beta, gamma=gamma, start_phase=start_t,
               phase_shift=shift_t)
    try:
        x0 = orbit_state(net, start_t)
        drift = orbit_state(net, start_t + T_TIME)
        target = orbit_state(net, start_t + shift_t)
        delta_x0 = np.zeros(2)
        delta_xT = target - drift

        def inject(U):
            def u_func(t):
                return np.array([U[min(int(t / DT), N_STEPS - 1), 0]])
            r = net.simulate_controlled(
                u_func=u_func, t_span=(0.0, N_STEPS * DT),
                n_points=max(100, N_STEPS),
                E0=np.array([x0[0]]), I0=np.array([x0[1]]))
            final = np.array([r["E"][0, -1], r["I"][0, -1]])
            return float(np.linalg.norm(final - target))

        # --- naive LTI: Jacobian frozen at the start state ---
        naive_err = naive_energy = np.nan
        J0 = net.jacobian_at(np.array([x0[0]]), np.array([x0[1]]))
        A, B = discretize_system(J0, G_CONST, DT)
        if np.max(np.abs(np.linalg.eigvals(A))) < 1:
            _, U_naive = physics.minimum_energy_trajectory(
                A, B, delta_x0, delta_xT, N_STEPS)
            naive_err = inject(U_naive)
            naive_energy = control_energy(U_naive, DT)

        # --- LTV along the nominal orbit: the budget and the target ---
        A_list, B_list = [], []
        for k in range(N_STEPS):
            sk = orbit_state(net, start_t + k * DT)
            Jk = net.jacobian_at(np.array([sk[0]]), np.array([sk[1]]))
            Ak, Bk = discretize_system(Jk, G_CONST, DT)
            A_list.append(Ak)
            B_list.append(Bk)
        _, U_ltv = physics.minimum_energy_trajectory_ltv(
            A_list, B_list, delta_x0, delta_xT)
        ltv_err = inject(U_ltv)
        budget = control_energy(U_ltv, DT)
        row.update(naive_err=naive_err, naive_energy=naive_energy,
                   ltv_err=ltv_err, ltv_energy=budget)

        def jac_f(x, u=None):
            return net.jacobian_at(x[0:1], x[1:2])

        def inp_f(x, u=None):
            return G_CONST

        def objective(u_flat):
            U = u_flat.reshape(N_STEPS, 1)
            c, g, _ = compute_adjoint_gradient(net, x0, target, U, DT,
                                               jac_f, inp_f, lambda_reg=0.0)
            return c, g.flatten()

        def e_val(u):
            return float(np.sum(u ** 2) * DT)

        def e_grad(u):
            return 2.0 * u * DT

        res = minimize(
            objective, np.zeros(N_STEPS), jac=True, method="SLSQP",
            constraints=[{"type": "ineq",
                          "fun": lambda u: budget - e_val(u),
                          "jac": lambda u: -e_grad(u)}],
            options={"maxiter": MAXITER, "ftol": 1e-12})

        U = res.x.reshape(N_STEPS, 1)
        used = control_energy(U, DT)
        err = inject(U)
        feasible = bool(used <= budget * (1 + 1e-6))
        row.update(
            constrained_err=err, constrained_energy=used,
            energy_frac=used / budget if budget else np.nan,
            feasible=feasible, converged=bool(res.success),
            n_iter=int(res.nit), solver_msg=str(res.message)[:60],
            dominates_ltv=bool(feasible and err < ltv_err),
            dominates_naive=bool(np.isfinite(naive_energy)
                                 and used <= naive_energy * (1 + 1e-6)
                                 and err < naive_err))
    except Exception as exc:
        row.update(constrained_err=np.nan, error=str(exc)[:90])
    return row


def summarise(df):
    ok = df.dropna(subset=["constrained_err"])
    print("\n" + "=" * 76)
    print("CONSTRAINED DOMINANCE — STUART-LANDAU")
    print("=" * 76)
    print(f"  conditions {len(df)}   solved {len(ok)}", end="")
    if len(ok):
        print(f"   feasible {int(ok.feasible.sum())}"
              f"   converged {int(ok.converged.sum())}")
    else:
        print("\n  nothing solved")
        return

    f = ok[ok.feasible]
    d = f.dominates_ltv.dropna()
    print(f"\n  DOMINATES LTV: {int(d.sum())}/{len(d)} = {d.mean():.0%}")
    print("    one constrained optimisation per condition, fixed cold start,")
    print("    feasible by construction AND lower terminal error")
    dn = f.dominates_naive.dropna()
    if len(dn):
        print(f"  DOMINATES NAIVE: {int(dn.sum())}/{len(dn)} = {dn.mean():.0%}")

    r = (f.constrained_err / f.ltv_err).replace([np.inf, -np.inf],
                                                np.nan).dropna()
    if len(r):
        print(f"\n  error ratio at or below the linear energy:"
              f"   median {r.median():.4f}"
              f"   log-ratio median {np.log(r).median():+.3f}")
    print(f"  energy used as a fraction of budget: median "
          f"{f.energy_frac.median():.3f}"
          f"   below 90%: {(f.energy_frac < 0.9).sum()}/{len(f)}")

    print("\n  BY REGIME — never pooled across all six, because with gamma = 0")
    print("  the three supercritical conditions are one system up to rescaling")
    for reg, g in f.groupby("regime"):
        v = g.dominates_ltv.dropna()
        print(f"    {reg:24s} {int(v.sum())}/{len(v)} = {v.mean():.0%}")
    anchor = f[f.regime == "supercritical_ref"]
    if len(anchor):
        v = anchor.dominates_ltv.dropna()
        print(f"\n  ANCHOR (beta = 1, gamma = 0), the FHN-comparable condition:"
              f" {int(v.sum())}/{len(v)}")
        print("  This is the row to quote against FitzHugh-Nagumo's 40/40.")

    print("\n  sweep-based rates for comparison: 110/120 overall,"
          " 17/20 at the anchor")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-workers", type=int, default=0)
    ap.add_argument("--max-conditions", type=int, default=0,
                    help="Limit conditions, for smoke tests. 0 = all 120.")
    args = ap.parse_args()

    combos = [(b["beta"], b["gamma"], b["regime"], sp, sh)
              for b in BETA_CONDITIONS
              for sp in BETA_SWEEP_PHASES
              for sh in PHASE_SHIFT_FRACTIONS]
    if args.max_conditions:
        combos = combos[:args.max_conditions]
    print(f"{len(combos)} conditions, ONE constrained optimisation each")

    n_workers = args.n_workers or min(cpu_count(), len(combos))
    print(f"  {n_workers} workers")
    rows = []
    with Pool(n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_condition, combos), 1):
            rows.append(r)
            if i % 20 == 0 or i == len(combos):
                print(f"  {i}/{len(combos)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    summarise(df)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
