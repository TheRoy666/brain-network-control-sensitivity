#!/usr/bin/env python3
"""
Constrained dominance: one optimisation per condition, no multiplicity.

THE OBJECTION
-------------
The matched-cost analysis swept ten effort penalties from four
initialisations and declared a condition dominated if ANY of the
resulting runs beat the single linear solution on both energy and error.
Four referees independently observed that this gives the nonlinear
optimiser roughly forty opportunities to find a winner, so a dominance
rate estimated that way is an upper bound inflated by search.

The objection is correct in principle. It is also cheap to remove.

THE FIX
-------
Solve the constrained problem directly, once per condition:

    minimise   || x(T) - x_target ||^2
    subject to integrated control energy <= E_linear

A feasible solution with lower terminal error than the linear one is a
dominating solution by construction: it satisfies the energy constraint,
so it cannot be spending more, and it achieves lower error. One
optimisation, one answer, nothing to select over.

DESIGN CHOICES THAT MATTER
--------------------------
**One deterministic initialisation, declared here: the zero control.**
It is the only start that is guaranteed feasible (zero energy satisfies
any positive budget) and the only one that borrows nothing from either
linear solution. Starting from a scaled linear solution would import the
very information the comparison is meant to be independent of, and
starting from several would reintroduce the multiplicity this script
exists to remove.

**The constraint is on the LTV solution's energy**, since LTV is the
stronger of the two linear methods and therefore the harder target. The
naive comparison is reported alongside but is not the headline.

**SLSQP with analytic gradients for both objective and constraint.**
The adjoint supplies the error gradient; the energy gradient is
2*U*dt in closed form. No finite differencing.

WHAT EACH OUTCOME MEANS
-----------------------
    constrained solution beats the linear one  ->  dominance is
        established without any search. The multiplicity objection is
        answered rather than argued with.

    it does not  ->  the sweep-based rate WAS inflated by search, and the
        headline must be restated. That would be worth knowing and is
        the reason to run this rather than defend the original.

Note that a failure here does not prove no dominating solution exists;
SLSQP from a cold start can stall. The script therefore reports
convergence status per condition, and a non-converged run is recorded as
such rather than counted as a failure to dominate.

USAGE
-----
    python3 constrained_dominance.py \\
        --model fhn --orbit-dir fhn_nominal_orbit_data \\
        --n-workers 40 --out constrained_dominance_fhn.csv
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from scipy.optimize import minimize, NonlinearConstraint

MAXITER = 500

# FHN grid, matched to matched_cost_control_comparison.py
I_EXT = 0.5
DT = 0.5
T_TIME = 15.0
N_STEPS = int(T_TIME / DT)
STABLE_PHASES = [1.972697269726973, 5.918091809180918, 11.836183618361837,
                 21.6996699669967, 29.590459045904595]
UNSTABLE_PHASES = [0.0, 13.80888088808881, 35.50855085508551]
START_PHASES = STABLE_PHASES + UNSTABLE_PHASES
PHASE_SHIFT_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5]
G_CONST = np.array([[1.0], [0.0]])
ORBIT_DIR = "fhn_nominal_orbit_data"


def control_energy(U, dt):
    return float(np.sum(U ** 2) * dt)


def load_nominal_orbit():
    from scipy.interpolate import interp1d
    t = np.load(os.path.join(ORBIT_DIR, "nominal_t.npy"))
    v = np.load(os.path.join(ORBIT_DIR, "nominal_v.npy"))
    w = np.load(os.path.join(ORBIT_DIR, "nominal_w.npy"))
    period = t[-1] + (t[1] - t[0])
    return (period,
            interp1d(t, v, kind="cubic", fill_value="extrapolate"),
            interp1d(t, w, kind="cubic", fill_value="extrapolate"))


def nominal_state(phase, period, v_i, w_i):
    p = phase % period
    return np.array([v_i(p), w_i(p)])


def run_condition(job):
    start_phase, phase_shift, is_stable, orbit_dir = job
    global ORBIT_DIR
    ORBIT_DIR = orbit_dir

    from nctsens import physics
    from nctsens.simulation import discretize_system
    from adjoint_gradient import compute_adjoint_gradient
    import fhn_simulation as fhn

    params = dict(**fhn.FHN_DEFAULT_PARAMS)
    params["kappa"] = 0.0
    net = fhn.FitzHughNagumoNetwork(n_regions=1, C=np.zeros((1, 1)),
                                    I_ext=I_EXT, node_params=params)
    period, v_i, w_i = load_nominal_orbit()

    x0 = nominal_state(start_phase, period, v_i, w_i)
    drift = nominal_state(start_phase + T_TIME, period, v_i, w_i)
    target = nominal_state(start_phase + phase_shift, period, v_i, w_i)
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

    out = dict(condition_id=f"{start_phase:.4f}|{phase_shift:.4f}",
               start_phase=start_phase, phase_shift=phase_shift,
               is_stable_start=is_stable)

    # linear references
    U_naive = None
    if is_stable:
        J0 = net.jacobian_at(np.array([x0[0]]), np.array([x0[1]]))
        A, B = discretize_system(J0, G_CONST, DT)
        if np.max(np.abs(np.linalg.eigvals(A))) < 1:
            _, U_naive = physics.minimum_energy_trajectory(
                A, B, delta_x0, delta_xT, N_STEPS)
            out.update(naive_err=inject(U_naive),
                       naive_energy=control_energy(U_naive, DT))

    A_list, B_list = [], []
    for k in range(N_STEPS):
        sk = nominal_state(start_phase + k * DT, period, v_i, w_i)
        Jk = net.jacobian_at(np.array([sk[0]]), np.array([sk[1]]))
        Ak, Bk = discretize_system(Jk, G_CONST, DT)
        A_list.append(Ak); B_list.append(Bk)
    _, U_ltv = physics.minimum_energy_trajectory_ltv(
        A_list, B_list, delta_x0, delta_xT)
    E_budget = control_energy(U_ltv, DT)
    err_ltv = inject(U_ltv)
    out.update(ltv_err=err_ltv, ltv_energy=E_budget)

    def jac_f(x, u=None):
        return net.jacobian_at(x[0:1], x[1:2])

    def inp_f(x, u=None):
        return G_CONST

    def obj(u_flat):
        U = u_flat.reshape(N_STEPS, 1)
        c, g, _ = compute_adjoint_gradient(net, x0, target, U, DT,
                                           jac_f, inp_f, lambda_reg=0.0)
        return c, g.flatten()

    # energy constraint, analytic gradient
    def e_fun(u_flat):
        return float(np.sum(u_flat ** 2) * DT)

    def e_jac(u_flat):
        return 2.0 * u_flat * DT

    con = NonlinearConstraint(e_fun, -np.inf, E_budget, jac=e_jac)

    # ONE deterministic start: the zero control, which is always feasible
    U0 = np.zeros(N_STEPS)
    try:
        res = minimize(obj, U0, jac=True, method="SLSQP",
                       constraints=[{"type": "ineq",
                                     "fun": lambda u: E_budget - e_fun(u),
                                     "jac": lambda u: -e_jac(u)}],
                       options={"maxiter": MAXITER, "ftol": 1e-12})
        U = res.x.reshape(N_STEPS, 1)
        e_used = control_energy(U, DT)
        err = inject(U)
        out.update(constrained_err=err, constrained_energy=e_used,
                   energy_slack=E_budget - e_used,
                   feasible=bool(e_used <= E_budget * (1 + 1e-6)),
                   converged=bool(res.success), n_iter=int(res.nit),
                   status=str(res.message)[:60])
        out["dominates_ltv"] = bool(
            e_used <= E_budget * (1 + 1e-6) and err < err_ltv)
        if "naive_err" in out:
            out["dominates_naive"] = bool(
                e_used <= out["naive_energy"] * (1 + 1e-6)
                and err < out["naive_err"])
    except Exception as exc:
        out.update(constrained_err=np.nan, error=str(exc)[:80])
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orbit-dir", default="fhn_nominal_orbit_data")
    ap.add_argument("--n-workers", type=int, default=0)
    ap.add_argument("--max-conditions", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    global ORBIT_DIR
    ORBIT_DIR = args.orbit_dir
    for n in ("nominal_t.npy", "nominal_v.npy", "nominal_w.npy"):
        if not os.path.exists(os.path.join(ORBIT_DIR, n)):
            raise SystemExit(f"Missing {os.path.join(ORBIT_DIR, n)}.")
    period, _, _ = load_nominal_orbit()
    print(f"  nominal orbit period {period:.4f}")
    if abs(period - 39.4539) > 1e-3:
        raise SystemExit(f"Orbit period {period:.4f} does not match the "
                         f"published grid (39.4539).")

    combos = [(p, f * period, p in STABLE_PHASES, args.orbit_dir)
              for p in START_PHASES for f in PHASE_SHIFT_FRACTIONS]
    if args.max_conditions:
        combos = combos[:args.max_conditions]
    print(f"{len(combos)} conditions, ONE constrained optimisation each")

    n_workers = args.n_workers or min(cpu_count(), len(combos))
    rows = []
    with Pool(n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_condition, combos), 1):
            rows.append(r)
            if i % 10 == 0 or i == len(combos):
                print(f"  {i}/{len(combos)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    ok = df.dropna(subset=["constrained_err"])

    print("\n" + "=" * 76)
    print("CONSTRAINED DOMINANCE — one optimisation per condition")
    print("=" * 76)
    print(f"  conditions: {len(df)}   solved: {len(ok)}")
    if not len(ok):
        print("  nothing solved"); return

    feas = ok[ok.feasible]
    print(f"  feasible (energy within budget): {len(feas)}/{len(ok)}")
    print(f"  converged: {int(ok.converged.sum())}/{len(ok)}")

    d = feas.dominates_ltv.dropna()
    print(f"\n  DOMINATES LTV: {int(d.sum())}/{len(d)} = {d.mean():.0%}")
    print("    (feasible by construction AND lower terminal error,")
    print("     from a single cold-start optimisation)")
    if "dominates_naive" in feas:
        dn = feas.dominates_naive.dropna()
        if len(dn):
            print(f"  DOMINATES NAIVE: {int(dn.sum())}/{len(dn)} = {dn.mean():.0%}")

    r = (feas.constrained_err / feas.ltv_err).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r):
        lr = np.log(r)
        print(f"\n  error ratio to the linear solution, at or below its energy:")
        print(f"    median {r.median():.4f}   log-ratio median {lr.median():+.3f}"
              f"   IQR [{r.quantile(.25):.4f}, {r.quantile(.75):.4f}]")
    s = (feas.constrained_energy / feas.ltv_energy).dropna()
    if len(s):
        print(f"  energy actually used, as a fraction of budget: "
              f"median {s.median():.3f}")

    print("\n  For comparison, the sweep-based rate was 36/40 (90%) in this")
    print("  model. If the constrained rate is close, the sweep was not")
    print("  materially inflated by search. If it is much lower, the")
    print("  headline must be restated.")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
