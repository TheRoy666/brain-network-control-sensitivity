#!/usr/bin/env python3
"""
Path 1, Item 2 -- Stuart-Landau A2 grid.

PURPOSE
-------
Wilson-Cowan and FitzHugh-Nagumo disagree about the direction of
linearization bias. A third arbitrary model would yield a 2-vs-1 tally
and prove nothing. This script instead tests whether the direction is
PREDICTABLE from a structural property of the nonlinearity -- turning
the disagreement from the paper's biggest vulnerability into its most
interesting result.

Stuart-Landau makes that testable because its cubic coefficient `beta`
is a continuously tunable parameter spanning both criticality regimes:
  beta > 0, gamma = 0  -> supercritical Hopf (saturating cubic)
  beta < 0, gamma > 0  -> subcritical Hopf (self-amplifying cubic,
                          stabilised by the quintic)
WC (sigmoidal, saturating) and FHN (cubic, excitable) are two fixed
points in that space; here the property is swept rather than sampled.

MEASURED QUANTITY -- IDENTICAL TO fhn_a2_broader_grid.py
--------------------------------------------------------
For each condition, compute a control input under each approximation,
inject it into the TRUE nonlinear system, and measure
||final_state - target||:

  naive_err   : control from the frozen-LTI Gramian at x0
  ltv_err     : control from the time-varying Jacobian along the orbit
  ceiling_err : control from full nonlinear adjoint-gradient optimisation

Same three arms, same injection-and-measure protocol, same DT/T_TIME/
optimizer settings as the FHN grid, so results are directly comparable.

TWO DELIBERATE DEPARTURES FROM STRICT FHN REPLICATION
------------------------------------------------------
Both follow from a structural property of Stuart-Landau that was
verified numerically before this script was written, not assumed:

  On the SL limit cycle the frozen Jacobian has trace = -2a and
  det = omega^2 at EVERY phase -- confirmed identical across 8 phases,
  matching the analytic derivation.

1. FHN's 8-phase axis is DEGENERATE here. SL is rotationally symmetric,
   so every orbit point is equivalent up to rotation; 8 phases would
   give 8 identical results. FHN's phase axis is meaningful because a
   relaxation oscillator genuinely differs on its fast vs slow segments.
   A 4-phase DEGENERACY CHECK is run anyway (cheap) to confirm the
   symmetry argument holds end-to-end in the full nonlinear control
   problem, not merely in the Jacobian spectrum.

2. FHN's stable/unstable starting-phase distinction has NO SL analogue.
   trace = -2a < 0 and det = omega^2 > 0 on the cycle, so the frozen
   Jacobian is always stable and `naive` is always computable. FHN had
   3 unstable phases where naive was undefined; SL has none.

What survives as a genuine anchor is FHN's 5 phase-shift targets. The
budget freed by dropping the degenerate phase axis is spent on beta --
the axis that actually addresses the question.

INTERFACE NOTE
--------------
`physics.minimum_energy_trajectory`, `physics.minimum_energy_trajectory_ltv`
and `simulation.discretize_system` are called here with signatures taken
from fhn_a2_broader_grid.py's CALL SITES; their implementations were not
available when this was written. If anything fails unexpectedly, that is
the first place to look.

USAGE
-----
    python3 sl_a2_grid.py --out sl_a2_results.csv
"""
import argparse
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:
    from nctsens import physics
    from nctsens.simulation import discretize_system
    from adjoint_gradient import compute_adjoint_gradient
except ImportError as e:
    raise SystemExit(
        f"Import failed: {e}\n\n"
        "This script needs three modules that are NOT in this directory:\n"
        "  physics, simulation   -> in neurosim/\n"
        "  adjoint_gradient      -> in scripts/nonlinear_validity_regime/\n\n"
        "It must therefore be run FROM scripts/nonlinear_validity_regime/\n"
        "with nctsens installed, the same convention every other sweep\n"
        "script in this repo uses:\n\n"
        "  cp scripts/path1_upscale/sl_a2_grid.py \\\n"
        "     scripts/path1_upscale/stuart_landau_model.py \\\n"
        "     scripts/nonlinear_validity_regime/\n"
        "  cd pipeline/03_linearization\n"
        "  python3 sl_a2_grid.py --out sl_a2_results.csv\n"
    )
from stuart_landau_model import StuartLandauNetwork

# Matched to fhn_a2_broader_grid.py exactly so the arms are comparable.
DT = 0.5
T_TIME = 15.0
N_STEPS = int(T_TIME / DT)

# Control enters the fast (x) variable additively, exactly as FHN's
# enters v. Because entry is purely additive, df/du is u-independent --
# so a constant input Jacobian is CORRECT here, unlike WC where control
# passes through a sigmoid (see adjoint_gradient.py's docstring on that
# being a real, verified bug source).
G_CONST = np.array([[1.0], [0.0]])

OMEGA = 0.05
A_OSC = 0.2  # a > 0 -> limit cycle. Same for every condition.

PHASE_SHIFT_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5]  # matches FHN

# beta sweep: supercritical (saturating) through subcritical
# (self-amplifying). gamma must be > 0 wherever beta < 0 or the system
# diverges -- the model class enforces this rather than allowing it.
BETA_CONDITIONS = [
    {"beta": 2.0,  "gamma": 0.0, "regime": "supercritical_strong"},
    {"beta": 1.0,  "gamma": 0.0, "regime": "supercritical_ref"},   # FHN anchor
    {"beta": 0.5,  "gamma": 0.0, "regime": "supercritical_weak"},
    {"beta": -0.5, "gamma": 1.0, "regime": "subcritical_weak"},
    {"beta": -1.0, "gamma": 1.0, "regime": "subcritical_ref"},
    {"beta": -2.0, "gamma": 1.0, "regime": "subcritical_strong"},
]

DEGENERACY_CHECK_PHASES = [0.0, 0.25, 0.5, 0.75]  # fractions of period

# Phases for the beta sweep. The first run used phase 0 only, on the
# assumption that SL's rotational symmetry made phase irrelevant. THAT
# ASSUMPTION WAS WRONG and the degeneracy check caught it: phases 0.0
# and 0.5 agree, 0.25 and 0.75 agree, but 0.0 and 0.25 differ by nearly
# 10x in naive_err. The dynamics are rotationally invariant; the CONTROL
# PROBLEM is not, because G = [[1],[0]] injects only along x and so
# picks out a fixed direction on the circle. Only half-period symmetry
# survives. The beta sweep therefore spans a half period.
BETA_SWEEP_PHASES = [0.0, 0.125, 0.25, 0.375]


def make_net(beta, gamma):
    return StuartLandauNetwork(
        n_regions=1, C=np.zeros((1, 1)),
        a=A_OSC, omega=OMEGA, beta=beta, gamma=gamma, G=0.0,
    )


def inject_and_measure(net, x0, U, n_steps, target_absolute):
    """Identical to fhn_a2_broader_grid.inject_and_measure."""
    def u_func(t):
        idx = min(int(t / DT), n_steps - 1)
        return np.array([U[idx, 0]])
    result = net.simulate_controlled(
        u_func=u_func, t_span=(0.0, n_steps * DT), n_points=max(100, n_steps),
        E0=np.array([x0[0]]), I0=np.array([x0[1]]),
    )
    final_state = np.array([result["E"][0, -1], result["I"][0, -1]])
    return np.linalg.norm(final_state - target_absolute), final_state


def orbit_state(net, phase_time):
    """Point on the limit cycle at a given TIME along the orbit.

    Converted to angle internally. Unlike FHN -- which integrates a
    nominal orbit, saves it, and interpolates with cubic splines -- this
    is exact, removing interpolation error from the SL arm. That is a
    real methodological difference between arms and belongs in methods.
    """
    period = net.period
    angle = 2.0 * np.pi * (phase_time % period) / period
    return net.nominal_orbit(angle)


def run_one_combo(net, start_phase_time, phase_shift_time, label_extra):
    x0 = orbit_state(net, start_phase_time)
    drift_state = orbit_state(net, start_phase_time + T_TIME)
    target_absolute = orbit_state(net, start_phase_time + phase_shift_time)

    delta_x0 = np.zeros(2)
    delta_xT = target_absolute - drift_state

    row = dict(label_extra)
    row.update({"start_phase_time": start_phase_time,
                "phase_shift_time": phase_shift_time})

    # --- Naive: frozen Jacobian at x0 ---
    J0 = net.jacobian_at(x0[0:1], x0[1:2])
    A, B = discretize_system(J0, G_CONST, DT)
    rho_A = float(np.max(np.abs(np.linalg.eigvals(A))))
    row["rho_A_frozen"] = rho_A
    if rho_A < 1:
        energy_naive, U_naive = physics.minimum_energy_trajectory(
            A, B, delta_x0, delta_xT, N_STEPS)
        err_naive, _ = inject_and_measure(net, x0, U_naive, N_STEPS, target_absolute)
        row["naive_err"] = err_naive
        row["naive_energy"] = energy_naive
    else:
        row["naive_err"] = None
        row["naive_energy"] = None
        row["naive_undefined_reason"] = f"frozen A unstable (rho={rho_A:.3f})"

    # --- LTV: Jacobian along the true orbit ---
    A_list, B_list = [], []
    for k in range(N_STEPS):
        st_k = orbit_state(net, start_phase_time + k * DT)
        J_k = net.jacobian_at(st_k[0:1], st_k[1:2])
        A_k, B_k = discretize_system(J_k, G_CONST, DT)
        A_list.append(A_k)
        B_list.append(B_k)
    energy_ltv, U_ltv = physics.minimum_energy_trajectory_ltv(
        A_list, B_list, delta_x0, delta_xT)
    err_ltv, _ = inject_and_measure(net, x0, U_ltv, N_STEPS, target_absolute)
    row["ltv_err"] = err_ltv
    row["ltv_energy"] = energy_ltv

    # --- Nonlinear ceiling via adjoint gradient ---
    def sl_jacobian_func(x, u=None):
        # u-independent: control enters additively, so df/dx does not
        # depend on u. Same situation as FHN, unlike WC.
        return net.jacobian_at(x[0:1], x[1:2])

    def sl_input_jacobian_func(x, u=None):
        return G_CONST

    def objective_with_grad(u_flat):
        U = u_flat.reshape(N_STEPS, 1)
        cost, grad, _ = compute_adjoint_gradient(
            net, x0, target_absolute, U, DT,
            sl_jacobian_func, sl_input_jacobian_func,
        )
        return cost, grad.flatten()

    result = minimize(objective_with_grad, U_ltv.flatten(), jac=True,
                      method="L-BFGS-B", options={"maxiter": 100})
    U_ceiling = result.x.reshape(N_STEPS, 1)
    err_ceiling, _ = inject_and_measure(net, x0, U_ceiling, N_STEPS, target_absolute)
    row["ceiling_err"] = err_ceiling
    row["ceiling_n_evals"] = result.nfev

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sl_a2_results.csv")
    ap.add_argument("--skip-degeneracy-check", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    rows = []

    # ---- Degeneracy check: confirm the phase axis really is redundant ----
    if not args.skip_degeneracy_check:
        print("=" * 72)
        print("DEGENERACY CHECK -- is the phase axis genuinely redundant for SL?")
        print("Verified already for the Jacobian spectrum; this confirms it")
        print("end-to-end in the full nonlinear control problem.")
        print("=" * 72)
        net = make_net(1.0, 0.0)
        period = net.period
        shift = 0.3 * period
        deg_errs = []
        for frac in DEGENERACY_CHECK_PHASES:
            r = run_one_combo(net, frac * period, shift,
                              {"condition": "degeneracy_check",
                               "beta": 1.0, "gamma": 0.0,
                               "regime": "supercritical_ref",
                               "start_phase_frac": frac})
            rows.append(r)
            deg_errs.append((r["naive_err"], r["ltv_err"], r["ceiling_err"]))
            print(f"  phase_frac={frac}: naive={r['naive_err']:.6f}  "
                  f"ltv={r['ltv_err']:.6f}  ceiling={r['ceiling_err']:.3e}")
        arr = np.array([[e for e in t] for t in deg_errs], dtype=float)
        spread = np.nanmax(arr, axis=0) - np.nanmin(arr, axis=0)
        rel = spread / np.nanmean(arr, axis=0)
        print(f"\n  relative spread across phases: naive={rel[0]:.2%}, "
              f"ltv={rel[1]:.2%}, ceiling={rel[2]:.2%}")
        if np.nanmax(rel[:2]) < 0.02:
            print("  -> CONFIRMED degenerate. Dropping the phase axis is")
            print("     justified; report the symmetry argument in methods.")
        else:
            print("  -> NOT degenerate at the expected level. The rotational-")
            print("     symmetry argument does NOT carry through the nonlinear")
            print("     control problem. Reinstate the full phase sweep and")
            print("     do NOT report the phase axis as redundant.")
        print()

    # ---- Main sweep: beta x phase-shift ----
    print("=" * 72)
    print("BETA SWEEP -- does linearization-bias direction track the")
    print("structure of the nonlinearity?")
    print("=" * 72)
    for cond in BETA_CONDITIONS:
        net = make_net(cond["beta"], cond["gamma"])
        period = net.period
        r_star = net.analytical_limit_cycle_amplitude()
        print(f"\n  beta={cond['beta']:+.1f} gamma={cond['gamma']:.1f} "
              f"({cond['regime']}): r*={r_star:.4f}, period={period:.3f}")
        for ph_frac in BETA_SWEEP_PHASES:
            for frac in PHASE_SHIFT_FRACTIONS:
                row = run_one_combo(
                    net, ph_frac * period, frac * period,
                    {"condition": "beta_sweep", "beta": cond["beta"],
                     "gamma": cond["gamma"], "regime": cond["regime"],
                     "start_phase_frac": ph_frac,
                     "shift_frac": frac, "r_star": r_star})
                rows.append(row)
                naive = row.get("naive_err")
                print(f"    phase={ph_frac} shift={frac}: "
                      f"naive={naive if naive is None else f'{naive:.6f}'}  "
                      f"ltv={row['ltv_err']:.6f}  "
                      f"ceiling={row['ceiling_err']:.3e} "
                      f"[{time.time()-t_start:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    # ---- The actual question ----
    print()
    print("=" * 72)
    print("RESULT: direction of linearization bias vs. nonlinearity structure")
    print("=" * 72)
    sweep = df[df.condition == "beta_sweep"].copy()
    sweep = sweep[sweep.naive_err.notna()]
    sweep["ltv_wins"] = sweep.ltv_err < sweep.naive_err
    # RATIO, not win-count, is the primary statistic. The first run used
    # win-count alone and reported "no dependence on beta" while the
    # underlying ratios separated perfectly by sign(beta) -- a binary
    # over 5 points is too coarse to see a consistent magnitude effect.
    sweep["ltv_over_naive"] = sweep.ltv_err / sweep.naive_err
    for regime, g in sweep.groupby("regime", sort=False):
        print(f"  {regime:24s} beta={g.beta.iloc[0]:+.1f}: "
              f"median ltv/naive={g.ltv_over_naive.median():.3f}  "
              f"(LTV wins {g.ltv_wins.sum()}/{len(g)})")

    print()
    super_r = sweep[sweep.beta > 0].ltv_over_naive
    sub_r = sweep[sweep.beta < 0].ltv_over_naive
    print(f"  supercritical (beta>0): median ltv/naive = {super_r.median():.3f}")
    print(f"  subcritical  (beta<0): median ltv/naive = {sub_r.median():.3f}")
    # Clean separation means every supercritical cell on one side of 1.0
    # and every subcritical cell on the other -- that is the rule.
    clean = (super_r > 1.0).all() and (sub_r < 1.0).all()
    print(f"  clean separation about ratio=1.0 in EVERY cell: {clean}")
    super_win = super_r.median(); sub_win = sub_r.median()
    if clean:
        print("  -> DIRECTION FLIPS WITH THE SIGN OF beta, in every cell.")
        print("     This is the predictive rule: linearization-bias")
        print("     direction is determined by the structure of the")
        print("     nonlinearity, not by model identity. Headline of Item 2.")
    elif abs(super_win - sub_win) > 0.2:
        print("  -> partial dependence on beta. Suggestive, not a rule.")
    else:
        print("  -> NO dependence on beta. The WC/FHN disagreement is NOT")
        print("     explained by nonlinearity structure as parameterised")
        print("     here. Honest outcome: the models simply differ, and the")
        print("     generality claim stays capped at Wilson-Cowan.")

    print()
    print(f"  FHN comparison anchor (its stable-start combos): LTV beat naive")
    print(f"  in 18/25 (72%). The beta=+1.0 row above is the directly")
    print(f"  comparable SL condition.")
    print(f"\n  {len(df)} rows in {time.time()-t_start:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
