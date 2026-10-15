#!/usr/bin/env python3
"""
Matched-cost comparison of naive-LTI, LTV and nonlinear control.

WHY THIS REPLACES THE EARLIER COMPARISON
----------------------------------------
The original A2 grids reported that the nonlinear solution was "never
worse than the better linear method", with median improvements of 748x
to 1563x. Six referees independently identified that comparison as
unfair, and inspection of the code confirms TWO separate reasons, either
of which alone invalidates the headline.

**Reason 1: mismatched objectives.** The nonlinear optimiser minimised
terminal tracking error with the effort penalty at zero
(`lambda_reg=0`). The linear methods are minimum-ENERGY solutions. Both
were then scored on terminal error only. The nonlinear method was
optimising the metric it was graded on, with effort free; the linear
methods were optimising something else entirely.

**Reason 2: warm-start guarantee.** The optimiser was initialised at
`U_ltv` and descends on terminal error. L-BFGS-B cannot increase its
objective, so `ceiling_err <= ltv_err` holds BY CONSTRUCTION. The
observed 25/25 is not evidence; it is arithmetic. Any claim of the form
"the nonlinear solution never lost" is, under warm-starting, a statement
about the starting point.

Neither the ceiling's control energy nor its peak amplitude was ever
recorded, in any A2 script, for any model. The comparison therefore
could not be repaired from the stored outputs and had to be re-run.

WHAT THIS SCRIPT DOES INSTEAD
-----------------------------
It abandons the single-number comparison and computes a TRADE-OFF
FRONTIER, which is the only fair way to compare controllers with
different objectives.

For each condition:

  1. Compute the naive-LTI and LTV solutions. Record terminal error,
     integrated control energy, and peak amplitude for each.

  2. Sweep the nonlinear optimiser's effort penalty `lambda_reg` across
     several decades. Each value yields one (energy, error) point. Taken
     together these trace the achievable frontier of the nonlinear
     method.

  3. From EVERY initialisation: cold (zeros), naive-warm, LTV-warm, and
     several random starts. Reporting the spread across initialisations
     addresses both the warm-start guarantee and the question of whether
     the optimiser finds a global optimum, which it is not guaranteed
     to do.

  4. Read off the two comparisons that are actually fair:

       AT MATCHED ENERGY  -- interpolate the nonlinear frontier to the
                             linear method's energy. Compare errors.
       AT MATCHED ERROR   -- interpolate to the linear method's error.
                             Compare energies.

If the linear solution lies ON the nonlinear frontier, linearization
costs nothing at that operating point. If it lies above, the vertical
distance is the cost of linearizing, measured honestly.

WHAT TO EXPECT
--------------
The nonlinear advantage will almost certainly SHRINK, possibly a lot.
That is the point. A smaller, defensible number is worth more than a
large one a referee can delete. If the advantage survives at matched
cost, the claim becomes genuinely strong.

USAGE
-----
Run from this directory with nctsens installed:

    python3 matched_cost_control_comparison.py \\
        --model fhn --out matched_cost_fhn.csv

    python3 matched_cost_control_comparison.py \\
        --model wc --out matched_cost_wc.csv

Grid and model parameters are taken from the corresponding original A2
script so the conditions match what was previously reported.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from scipy.optimize import minimize

# Effort penalties spanning "essentially free" to "effort dominates".
# The frontier is traced by sweeping this.
# Trimmed from a wider sweep for tractability. Each adjoint evaluation
# integrates the dynamics segment by segment, so cost scales as
# (conditions x initialisations x penalties x iterations). Six penalties
# spanning five decades is enough to trace the frontier.
# Widened after the first full run: the linear operating points fell
# outside the swept span in 11 of 40 conditions (8 above, 3 below), so the
# frontier could not be interpolated there.
LAMBDA_GRID = [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

# Initialisations. 'ltv' reproduces the original (guaranteed) setting and
# is retained ONLY so the guarantee can be demonstrated explicitly.
# 'ltv' reproduces the original (guaranteed) setting and is retained only
# so the guarantee can be shown explicitly. 'cold' is the honest test.
# One random start is enough to detect gross local-optimum sensitivity.
INITS = ["cold", "naive", "ltv", "rand0"]

MAXITER = 500   # raised: frontiers were non-monotone at 200, indicating
                # incomplete convergence rather than a real trade-off


def control_energy(U, dt):
    """Integrated squared control, the quantity minimum-energy control minimises."""
    return float(np.sum(U ** 2) * dt)


def peak_amplitude(U):
    return float(np.max(np.abs(U)))


def frontier_point(objective_with_grad, U_init, n_steps, n_ctrl,
                   inject_and_measure, dt):
    """One optimisation run. Returns error, energy, peak and diagnostics."""
    res = minimize(objective_with_grad, U_init.flatten(), jac=True,
                   method="L-BFGS-B", options={"maxiter": MAXITER})
    U = res.x.reshape(n_steps, n_ctrl)
    err = inject_and_measure(U)
    return dict(
        err=err,
        energy=control_energy(U, dt),
        peak=peak_amplitude(U),
        n_evals=int(res.nfev),
        converged=bool(res.success),
        final_grad_norm=float(np.linalg.norm(res.jac)) if res.jac is not None else np.nan,
    )


def interp_frontier(points, key_x, key_y, x_query):
    """Interpolate the (energy, error) frontier at a queried x.

    Points are sorted by x and only the Pareto-efficient ones retained,
    since a dominated point is not part of the achievable frontier.
    """
    pts = sorted(points, key=lambda p: p[key_x])
    pareto, best = [], np.inf
    for p in pts:
        if p[key_y] < best:
            pareto.append(p)
            best = p[key_y]
    if len(pareto) < 2:
        return np.nan
    xs = np.array([p[key_x] for p in pareto])
    ys = np.array([p[key_y] for p in pareto])
    if x_query < xs.min() or x_query > xs.max():
        return np.nan          # extrapolation is not evidence
    return float(np.interp(x_query, xs, ys))


def summarise(df):
    """All per-CONDITION, not per-row. An earlier version reported row
    counts, which inflated every n by the number of runs per condition."""
    per = df.groupby("condition_id").first()
    nC = len(per)
    print("\n" + "=" * 78)
    print(f"MATCHED-COST COMPARISON  ({nC} conditions)")
    print("=" * 78)

    print("\n1. PARETO DOMINANCE  <-- primary, assumption-free")
    print("   Does some nonlinear run beat the linear solution on BOTH energy")
    print("   and error simultaneously? No interpolation, no matched operating")
    print("   point, no frontier-coverage requirement.")
    for lin in ("naive", "ltv"):
        d = per[f"dominated_{lin}"].dropna()
        if not len(d):
            continue
        er = per[f"dom_energy_ratio_{lin}"].dropna()
        rr = per[f"dom_err_ratio_{lin}"].dropna()
        print(f"\n   {lin.upper()} dominated in {int(d.sum())}/{len(d)} conditions "
              f"({d.mean():.1%})")
        if len(er):
            print(f"     among dominated, the dominating point uses "
                  f"{er.median():.3f}x the energy (median)")
            print(f"     and achieves {rr.median():.3e}x the error (median)")

    print("\n2. THE WARM-START GUARANTEE")
    for init, note in [("ltv", "original setting -- arithmetic, not evidence"),
                       ("cold", "honest test")]:
        g = df[(df.init == init) & (df.lambda_reg == 0.0)]
        if len(g):
            g = g.dropna(subset=["err", "ltv_err"])
            print(f"   {init:5s} start, lambda=0: nonlinear err <= LTV err in "
                  f"{(g.err <= g.ltv_err).mean():.1%}   ({note})")

    print("\n3. SENSITIVITY TO INITIALISATION")
    g = df[df.lambda_reg == 0.0].groupby("condition_id").err
    sp = ((g.max() - g.min()) / g.min().replace(0, np.nan)).dropna()
    print(f"   relative spread in terminal error across initialisations:")
    print(f"     median {sp.median():.1%}   p95 {sp.quantile(.95):.1%}   max {sp.max():.1%}")
    print("   Large spread means the solver returns LOCAL optima. Any claim")
    print("   must say 'best solution found', not 'the nonlinear solution'.")

    print("\n4. FRONTIER QUALITY (is the optimiser converging?)")
    bad = 0
    for cid, g in df.groupby("condition_id"):
        b = (g.dropna(subset=["energy", "err"])
               .sort_values("err").groupby("lambda_reg").first())
        if len(b) > 2 and not b.sort_index().energy.is_monotonic_decreasing:
            bad += 1
    print(f"   non-monotone energy-vs-penalty frontiers: {bad}/{nC}")
    print("   A higher effort penalty should never yield MORE energy. Where it")
    print("   does, that condition's frontier is not converged.")

    print("\n5. MATCHED-COST RATIOS  (secondary: needs the linear operating")
    print("   point to lie inside the swept frontier)")
    for lin in ("naive", "ltv"):
        e = per[f"err_at_matched_energy_vs_{lin}"].dropna()
        n = per[f"energy_at_matched_err_vs_{lin}"].dropna()
        print(f"\n   versus {lin.upper()}:")
        print(f"     matched ENERGY: n={len(e)}/{nC} conditions", end="")
        if len(e):
            print(f"   error ratio median {e.median():.3e}"
                  f"   better in {(e < 1).mean():.0%}")
        else:
            print()
        print(f"     matched ERROR:  n={len(n)}/{nC} conditions", end="")
        if len(n):
            print(f"   energy ratio median {n.median():.3f}"
                  f"   cheaper in {(n < 1).mean():.0%}")
        else:
            print()

    print("\n6. THE ORIGINAL (UNFAIR) NUMBER, FOR CONTRAST")
    u = df[(df.init == "ltv") & (df.lambda_reg == 0.0)].dropna(
        subset=["err", "ltv_err"])
    if len(u):
        bl = np.minimum(u.naive_err.fillna(np.inf), u.ltv_err)
        r = (bl / u.err).replace([np.inf, -np.inf], np.nan).dropna()
        if len(r):
            print(f"   effort-free, LTV-warm improvement: median {r.median():.0f}x")
            print("   Two defects: the objective is unmatched, and the warm start")
            print("   makes the comparison a tautology. Report only alongside (1).")

    print("\n" + "-" * 78)
    print("READING THIS: (1) is the robust result and needs no assumptions.")
    print("If the linear solution is dominated, it is beaten on energy AND")
    print("error at once, which is stronger than any matched-cost ratio and")
    print("cannot be dismissed as an objective mismatch.")


# ----------------------------------------------------------------------
# FHN setup, copied verbatim from fhn_broader_grid.py so the conditions
# match exactly what was previously reported.
# ----------------------------------------------------------------------
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

ORBIT_DIR = "fhn_a2_data"   # overridden by --orbit-dir


def load_nominal_orbit():
    """Load the published nominal orbit.

    These arrays live in fhn_a2_data/ and are NOT in the public repo.
    They can be regenerated bit-identically with
    fhn_a2_generate_nominal_orbit.py, which is deterministic -- verified
    by comparison against the staged copy (789 samples, period 39.4539,
    arrays equal element-for-element). If the period differs from
    39.4539, the conditions will not match the published grid and the
    comparison against the earlier numbers is invalid.
    """
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


def run_condition(net, start_phase, phase_shift, period, v_i, w_i,
                  is_stable, lambda_grid, maxiter):
    from nctsens import physics
    from nctsens.simulation import discretize_system
    from adjoint_gradient import compute_adjoint_gradient

    x0 = nominal_state(start_phase, period, v_i, w_i)
    drift = nominal_state(start_phase + T_TIME, period, v_i, w_i)
    target_absolute = nominal_state(start_phase + phase_shift, period, v_i, w_i)
    delta_x0 = np.zeros(2)
    delta_xT = target_absolute - drift

    def inject(U):
        def u_func(t):
            return np.array([U[min(int(t / DT), N_STEPS - 1), 0]])
        r = net.simulate_controlled(
            u_func=u_func, t_span=(0.0, N_STEPS * DT),
            n_points=max(100, N_STEPS),
            E0=np.array([x0[0]]), I0=np.array([x0[1]]))
        final = np.array([r["E"][0, -1], r["I"][0, -1]])
        return float(np.linalg.norm(final - target_absolute))

    # --- naive LTI ---
    U_naive, naive = None, dict(err=np.nan, energy=np.nan, peak=np.nan)
    if is_stable:
        J0 = net.jacobian_at(np.array([x0[0]]), np.array([x0[1]]))
        A, B = discretize_system(J0, G_CONST, DT)
        if np.max(np.abs(np.linalg.eigvals(A))) < 1:
            e, U_naive = physics.minimum_energy_trajectory(
                A, B, delta_x0, delta_xT, N_STEPS)
            naive = dict(err=inject(U_naive), energy=control_energy(U_naive, DT),
                         peak=peak_amplitude(U_naive))

    # --- LTV ---
    A_list, B_list = [], []
    for k in range(N_STEPS):
        sk = nominal_state(start_phase + k * DT, period, v_i, w_i)
        Jk = net.jacobian_at(np.array([sk[0]]), np.array([sk[1]]))
        Ak, Bk = discretize_system(Jk, G_CONST, DT)
        A_list.append(Ak); B_list.append(Bk)
    _, U_ltv = physics.minimum_energy_trajectory_ltv(
        A_list, B_list, delta_x0, delta_xT)
    ltv = dict(err=inject(U_ltv), energy=control_energy(U_ltv, DT),
               peak=peak_amplitude(U_ltv))

    def jac_f(x, u=None):
        return net.jacobian_at(x[0:1], x[1:2])

    def inp_f(x, u=None):
        return G_CONST

    import zlib
    rng = np.random.default_rng(zlib.crc32(
        f"init|{start_phase:.6f}|{phase_shift:.6f}".encode()) % (2**32))
    scale = max(np.abs(U_ltv).max(), 1e-3)
    starts = {
        "cold":  np.zeros((N_STEPS, 1)),
        "ltv":   U_ltv.copy(),
        "rand0": rng.normal(0, scale, (N_STEPS, 1)),
    }
    if U_naive is not None:
        starts["naive"] = U_naive.copy()

    cid = f"{start_phase:.4f}|{phase_shift:.4f}"
    rows = []
    for init, U0 in starts.items():
        for lam in lambda_grid:
            def obj(u_flat, lam=lam):
                U = u_flat.reshape(N_STEPS, 1)
                c, g, _ = compute_adjoint_gradient(
                    net, x0, target_absolute, U, DT, jac_f, inp_f,
                    lambda_reg=lam)
                return c, g.flatten()
            try:
                r = frontier_point(obj, U0, N_STEPS, 1, inject, DT)
            except Exception as exc:
                r = dict(err=np.nan, energy=np.nan, peak=np.nan,
                         n_evals=np.nan, converged=False,
                         final_grad_norm=np.nan, error=str(exc)[:80])
            r.update(condition_id=cid, start_phase=start_phase,
                     phase_shift=phase_shift, is_stable_start=is_stable,
                     init=init, lambda_reg=lam,
                     naive_err=naive["err"], naive_energy=naive["energy"],
                     naive_peak=naive["peak"],
                     ltv_err=ltv["err"], ltv_energy=ltv["energy"],
                     ltv_peak=ltv["peak"])
            rows.append(r)
    return rows


def _worker(job):
    """Each worker builds its own network and orbit: neither is reliably
    picklable, and rebuilding is cheap relative to the optimisations."""
    start_phase, phase_shift, is_stable, lam_grid, maxiter, orbit_dir = job
    global ORBIT_DIR
    ORBIT_DIR = orbit_dir
    import fhn_simulation as fhn
    params = dict(**fhn.FHN_DEFAULT_PARAMS)
    params["kappa"] = 0.0
    net = fhn.FitzHughNagumoNetwork(n_regions=1, C=np.zeros((1, 1)),
                                    I_ext=I_EXT, node_params=params)
    period, v_i, w_i = load_nominal_orbit()
    return run_condition(net, start_phase, phase_shift, period, v_i, w_i,
                         is_stable, lam_grid, maxiter)


def add_dominance(df):
    """Is the linear solution PARETO-DOMINATED by some nonlinear point?

    This is the assumption-free comparison. A linear solution is dominated
    if some nonlinear run achieves BOTH lower energy AND lower terminal
    error. No interpolation, no frontier coverage requirement, no matched
    operating point. Either a dominating point exists or it does not.

    It matters because "minimum energy" is minimum energy for the
    LINEARIZED system. Injected into the true nonlinear system the
    solution is neither minimum-energy nor accurate for the actual
    problem, so its optimality guarantee does not transfer -- and it can
    be beaten on both axes at once.
    """
    for lin in ("naive", "ltv"):
        df[f"dominated_{lin}"] = np.nan
        df[f"dom_energy_ratio_{lin}"] = np.nan
        df[f"dom_err_ratio_{lin}"] = np.nan
    for cid, g in df.groupby("condition_id"):
        pts = g.dropna(subset=["energy", "err"])
        if not len(pts):
            continue
        for lin in ("naive", "ltv"):
            le = g[f"{lin}_energy"].iloc[0]
            lr = g[f"{lin}_err"].iloc[0]
            if not (np.isfinite(le) and np.isfinite(lr)):
                continue
            dom = pts[(pts.energy <= le) & (pts.err <= lr)]
            m = df.condition_id == cid
            df.loc[m, f"dominated_{lin}"] = float(len(dom) > 0)
            if len(dom):
                # among dominating points, the one with least energy
                b = dom.loc[dom.energy.idxmin()]
                df.loc[m, f"dom_energy_ratio_{lin}"] = b.energy / le
                df.loc[m, f"dom_err_ratio_{lin}"] = b.err / lr
    return df


def add_matched_columns(df):
    """Per condition, interpolate the nonlinear frontier at the linear
    methods' operating points. Uses the best point across ALL
    initialisations at each lambda, which is the achievable frontier."""
    for lin in ("naive", "ltv"):
        df[f"err_at_matched_energy_vs_{lin}"] = np.nan
        df[f"energy_at_matched_err_vs_{lin}"] = np.nan
    for cid, g in df.groupby("condition_id"):
        best = (g.dropna(subset=["err", "energy"])
                  .sort_values("err").groupby("lambda_reg").first().reset_index())
        pts = best[["energy", "err"]].to_dict("records")
        if len(pts) < 2:
            continue
        for lin in ("naive", "ltv"):
            le = g[f"{lin}_energy"].iloc[0]
            lr = g[f"{lin}_err"].iloc[0]
            if np.isfinite(le) and np.isfinite(lr) and lr > 0 and le > 0:
                e = interp_frontier(pts, "energy", "err", le)
                n = interp_frontier(pts, "err", "energy", lr)
                df.loc[df.condition_id == cid,
                       f"err_at_matched_energy_vs_{lin}"] = e / lr
                df.loc[df.condition_id == cid,
                       f"energy_at_matched_err_vs_{lin}"] = n / le
    return df


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lambda-grid", default=None,
                    help="Override the effort-penalty sweep, comma-separated.")
    ap.add_argument("--maxiter", type=int, default=MAXITER)
    ap.add_argument("--orbit-dir", default="fhn_a2_data",
                    help="Directory holding nominal_{t,v,w}.npy. On some "
                         "installs this is named fhn_nominal_orbit_data.")
    ap.add_argument("--n-workers", type=int, default=0,
                    help="0 = one per condition, capped at core count.")
    ap.add_argument("--max-conditions", type=int, default=0,
                    help="Limit conditions, for smoke tests. 0 = all 40.")
    args = ap.parse_args()

    lam_grid = ([float(x) for x in args.lambda_grid.split(",")]
                if args.lambda_grid else LAMBDA_GRID)

    global ORBIT_DIR
    ORBIT_DIR = args.orbit_dir
    for n in ("nominal_t.npy", "nominal_v.npy", "nominal_w.npy"):
        if not os.path.exists(os.path.join(ORBIT_DIR, n)):
            raise SystemExit(
                f"Missing {os.path.join(ORBIT_DIR, n)}.\n"
                f"Generate it with fhn_generate_nominal_orbit.py, or point "
                f"--orbit-dir at the directory that holds it.")

    import fhn_simulation as fhn
    params = dict(**fhn.FHN_DEFAULT_PARAMS)
    params["kappa"] = 0.0
    net = fhn.FitzHughNagumoNetwork(n_regions=1, C=np.zeros((1, 1)),
                                    I_ext=I_EXT, node_params=params)
    period, v_i, w_i = load_nominal_orbit()
    print(f"  nominal orbit: period {period:.4f}")
    if abs(period - 39.4539) > 1e-3:
        raise SystemExit(
            f"Orbit period {period:.4f} does not match the published grid "
            f"(39.4539). The conditions would not correspond to the results "
            f"this comparison is meant to correct. Regenerate the orbit.")

    combos = [(p, f * period, p in STABLE_PHASES)
              for p in START_PHASES for f in PHASE_SHIFT_FRACTIONS]
    if args.max_conditions:
        combos = combos[:args.max_conditions]
    print(f"{len(combos)} conditions x {len(lam_grid)} penalties "
          f"x up to {len(INITS)} initialisations")

    n_workers = args.n_workers or min(cpu_count(), len(combos))
    print(f"  {n_workers} workers")
    jobs = [(p, sh, st, lam_grid, args.maxiter, args.orbit_dir)
            for p, sh, st in combos]
    rows = []
    if n_workers > 1:
        with Pool(n_workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_worker, jobs), 1):
                rows.extend(r)
                print(f"  {i}/{len(jobs)}", flush=True)
    else:
        for i, j in enumerate(jobs, 1):
            rows.extend(_worker(j))
            print(f"  {i}/{len(jobs)}", flush=True)

    df = add_matched_columns(add_dominance(pd.DataFrame(rows)))
    df.to_csv(args.out, index=False)
    summarise(df)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
