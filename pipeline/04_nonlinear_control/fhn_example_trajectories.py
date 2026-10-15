#!/usr/bin/env python3
"""
fhn_example_trajectories.py — trajectories and control signals for ONE
FitzHugh-Nagumo condition, for the phase-portrait panels of Figure 6.

constrained_dominance.py stores only summary numbers. This re-runs a single
condition through the SAME code path (same model, same linear controller,
same constrained SLSQP problem, same zero start) and keeps what the summary
discards: the state trajectories and the control inputs.

CHOICE OF CONDITION — by rule, not by eye
    the condition whose constrained/linear error ratio is closest to the
    median over all 40 conditions. A hand-picked exemplar invites the
    question of how typical it is; this one is typical by construction.

SANITY GATE
    the re-run must reproduce that condition's stored linear error and
    constrained error. If it does not, the code path has diverged and the
    trajectories are not those behind the reported numbers.

USAGE (from this directory)
    python3 fhn_example_trajectories.py \\
        --summary constrained_dominance_fhn.csv --out fhn_example.npz
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import constrained_dominance as cd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--orbit-dir", default=cd.ORBIT_DIR)
    ap.add_argument("--out", default="fhn_example.npz")
    a = ap.parse_args()

    s = pd.read_csv(a.summary)
    s = s[s.feasible & s.converged].copy()
    s["ratio"] = s.constrained_err / s.ltv_err
    row = s.iloc[(s.ratio - s.ratio.median()).abs().argsort().iloc[0]]
    print(f"condition {row.condition_id}: start phase {row.start_phase:.4f}, "
          f"shift {row.phase_shift:.2f}, ratio {row.ratio:.3e} "
          f"(median over {len(s)}: {s.ratio.median():.3e})")

    cd.ORBIT_DIR = a.orbit_dir
    from nctsens import physics
    from nctsens.simulation import discretize_system
    from adjoint_gradient import compute_adjoint_gradient
    import fhn_simulation as fhn

    params = dict(**fhn.FHN_DEFAULT_PARAMS); params["kappa"] = 0.0
    net = fhn.FitzHughNagumoNetwork(n_regions=1, C=np.zeros((1, 1)),
                                    I_ext=cd.I_EXT, node_params=params)
    period, v_i, w_i = cd.load_nominal_orbit()
    sp, sh = float(row.start_phase), float(row.phase_shift)
    x0 = cd.nominal_state(sp, period, v_i, w_i)
    drift = cd.nominal_state(sp + cd.T_TIME, period, v_i, w_i)
    target = cd.nominal_state(sp + sh, period, v_i, w_i)
    N, DT = cd.N_STEPS, cd.DT

    def simulate(U):
        u_func = lambda t: np.array([U[min(int(t / DT), N - 1), 0]])
        r = net.simulate_controlled(u_func=u_func, t_span=(0.0, N * DT),
                                    n_points=max(400, N),
                                    E0=np.array([x0[0]]), I0=np.array([x0[1]]))
        return r["E"][0], r["I"][0]

    A_list, B_list = [], []
    for k in range(N):
        sk = cd.nominal_state(sp + k * DT, period, v_i, w_i)
        Jk = net.jacobian_at(np.array([sk[0]]), np.array([sk[1]]))
        Ak, Bk = discretize_system(Jk, cd.G_CONST, DT)
        A_list.append(Ak); B_list.append(Bk)
    _, U_ltv = physics.minimum_energy_trajectory_ltv(A_list, B_list,
                                                     np.zeros(2), target - drift)
    E_budget = cd.control_energy(U_ltv, DT)

    def obj(u):
        c, g, _ = compute_adjoint_gradient(
            net, x0, target, u.reshape(N, 1), DT,
            lambda x, u=None: net.jacobian_at(x[0:1], x[1:2]),
            lambda x, u=None: cd.G_CONST, lambda_reg=0.0)
        return c, g.flatten()
    e_fun = lambda u: float(np.sum(u ** 2) * DT)
    res = minimize(obj, np.zeros(N), jac=True, method="SLSQP",
                   constraints=[{"type": "ineq", "fun": lambda u: E_budget - e_fun(u),
                                 "jac": lambda u: -2.0 * u * DT}],
                   options={"maxiter": cd.MAXITER, "ftol": 1e-12})
    U_con = res.x.reshape(N, 1)

    v0, w0 = simulate(np.zeros((N, 1)))
    vl, wl = simulate(U_ltv)
    vc, wc = simulate(U_con)
    err_l = float(np.linalg.norm(np.array([vl[-1], wl[-1]]) - target))
    err_c = float(np.linalg.norm(np.array([vc[-1], wc[-1]]) - target))

    print("\nSANITY GATE — reproduce the stored errors")
    ok = True
    for lab, got, want in [("linear", err_l, row.ltv_err),
                           ("constrained", err_c, row.constrained_err)]:
        rel = abs(got - want) / max(abs(want), 1e-12)
        match = rel < 1e-3
        ok &= match
        print(f"  {lab:12s} re-run {got:.6e}   stored {want:.6e}   "
              f"{'match' if match else 'MISMATCH'} (rel {rel:.1e})")
    if not ok:
        sys.exit("  ** stored errors not reproduced — do not use these trajectories **")

    tt = np.linspace(0, period, 800)
    np.savez(a.out, orbit_v=v_i(tt), orbit_w=w_i(tt), x0=x0, target=target,
             drift=drift, v_none=v0, w_none=w0, v_lin=vl, w_lin=wl,
             v_con=vc, w_con=wc, U_lin=U_ltv[:, 0], U_con=U_con[:, 0], dt=DT,
             E_budget=E_budget, E_con=cd.control_energy(U_con, DT),
             err_lin=err_l, err_con=err_c, condition=str(row.condition_id))
    print(f"\nsaved {a.out}: energy budget {E_budget:.4f}, "
          f"constrained energy {cd.control_energy(U_con, DT):.4f}")


if __name__ == "__main__":
    main()
