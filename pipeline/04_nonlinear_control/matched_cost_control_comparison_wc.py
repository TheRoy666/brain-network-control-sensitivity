#!/usr/bin/env python3
"""
Matched-cost comparison, Wilson-Cowan arm.

This is the third and last model arm. See
matched_cost_control_comparison.py for why the original comparison was
invalid and what replaces it; the short version is that the nonlinear
optimiser minimised terminal error with effort unpriced while the linear
methods are minimum-ENERGY solutions, and it was warm-started from one of
them, so the finding that it never lost was arithmetic rather than
evidence.

THREE THINGS DIFFER IN THIS ARM AND ALL MATTER
----------------------------------------------
1. The warm start is U_NAIVE here, not U_ltv as in the FitzHugh-Nagumo
   and Stuart-Landau arms. So the tautology in this arm guaranteed
   ceiling_err <= naive_err, a different guarantee from the other two.
   The original grid's own column name, rel_err_ceiling_original_naive_
   warmstart, records this.

2. run_ltv_control in wc_oscillatory_linearization_comparison.py computes
   the LTV control sequence but returns only its energy and final state,
   discarding U. Energy is therefore available for LTV but the control
   itself is not, so peak amplitude cannot be recovered from that path.
   This script reimplements the LTV leg inline to keep U.

3. The existing WC A2 results file is NOT fully trustworthy: 32 of its
   40 rows carry the note "reconstructed from 1-decimal printed
   percentages (tool execution limit hit mid-run)" and only 8 are stored
   at full precision. This run replaces those numbers rather than
   extending them, and nothing from the old file should be quoted
   alongside these results.

WHAT IS AND IS NOT AT STAKE
---------------------------
Dominance has already been established in two models with matched-cost
methodology: FitzHugh-Nagumo, 36 of 40 conditions, and Stuart-Landau, 17
of 20 at the matched condition and 59 of 60 in the subcritical
conditions where the nonlinearity differs most. This arm is therefore
CONFIRMATORY. A third agreement strengthens the generality claim; a
disagreement would be more interesting still, and would mean the result
is regime-dependent in a way the other two models cannot reveal.

The system is small: N = 10 regions, 25 control steps. This is not the
410-region connectome used elsewhere in the paper, and the cost is
correspondingly modest.

USAGE
-----
Run from the directory holding wc_oscillatory_linearization_comparison.py
and adjoint_gradient.py, with nctsens installed:

    python3 matched_cost_control_comparison_wc.py \\
        --n-workers 40 --out matched_cost_wc.csv
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import zlib
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from nctsens import simulation
from nctsens import physics
import wc_oscillatory_linearization_comparison as wc_a2
from adjoint_gradient import compute_adjoint_gradient
from adjoint_gradient_wc_verification import (make_wc_jacobian_func,
                                              make_wc_input_jacobian_func)

# Matched to wc_a2_broader_grid.py so conditions correspond.
N = 10
DT = 2.0
T_STEPS = 25
PERIOD = 99.96
SEEDS = list(range(8))
SHIFTS_MS = [10, 20, 30, 40, 60]

LAMBDA_GRID = [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
INITS = ["cold", "naive", "ltv", "rand0"]
MAXITER = 500


def control_energy(U, dt):
    return float(np.sum(U ** 2) * dt)


def peak_amplitude(U):
    return float(np.max(np.abs(U)))


def ltv_control_sequence(net, t_arr, E_arr, I_arr, t_ref, T_steps,
                         x_target, x_natural, E_ref, I_ref):
    """LTV leg, reimplemented to RETURN the control sequence.

    wc_oscillatory_linearization_comparison.run_ltv_control computes U
    and then discards it, returning only energy and final state. The
    matched-cost comparison needs U itself, so the body is repeated here.
    Any change to the original must be mirrored.
    """
    A_list, B_list = [], []
    for k in range(T_steps):
        t_k = t_ref + k * DT
        E_k, I_k = wc_a2.state_at_time(t_arr, E_arr, I_arr, t_k)
        J_k = net.jacobian_at(E_k, I_k)
        G_k = simulation.input_jacobian_at(net, E_k, I_k)
        A_k, B_k = simulation.discretize_system(J_k, G_k, DT)
        A_list.append(A_k)
        B_list.append(B_k)
    delta_x0 = np.zeros(2 * N)
    delta_xT = x_target - x_natural
    energy, U = physics.minimum_energy_trajectory_ltv(
        A_list, B_list, delta_x0, delta_xT)
    return energy, U


def frontier_point(objective_with_grad, U_init, inject, dt):
    res = minimize(objective_with_grad, U_init.flatten(), jac=True,
                   method="L-BFGS-B", options={"maxiter": MAXITER})
    U = res.x.reshape(T_STEPS, N)
    return dict(err=inject(U), energy=control_energy(U, dt),
                peak=peak_amplitude(U), n_evals=int(res.nfev),
                converged=bool(res.success))


def interp_frontier(points, key_x, key_y, x_query):
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
        return np.nan
    return float(np.interp(x_query, xs, ys))


def run_condition(job):
    seed, shift_ms, lam_grid = job
    # Setup is three calls, matching wc_a2_broader_grid.py exactly. Seeds
    # with no stable reference phase are SKIPPED there, so they are
    # skipped here too; the condition count may therefore be below
    # 8 x 5 = 40, and the printed total reflects what actually ran.
    try:
        net = wc_a2.build_network(seed)
        t_arr, E_arr, I_arr = wc_a2.get_reference_trajectory(net, seed)
        t_ref, _ = wc_a2.find_stable_reference_phase(
            net, t_arr, E_arr, I_arr, PERIOD)
        if t_ref is None:
            return [dict(seed=seed, shift_ms=shift_ms,
                         condition_id=f"seed{seed}|shift{shift_ms}",
                         error="no stable reference phase; skipped")]
    except Exception as exc:
        return [dict(seed=seed, shift_ms=shift_ms,
                     condition_id=f"seed{seed}|shift{shift_ms}",
                     error=f"setup: {exc}"[:90])]

    T_ms = T_STEPS * DT
    E_ref, I_ref = wc_a2.state_at_time(t_arr, E_arr, I_arr, t_ref)
    x_ref = np.concatenate([E_ref, I_ref])
    E_tgt, I_tgt = wc_a2.state_at_time(t_arr, E_arr, I_arr,
                                       t_ref + T_ms + shift_ms)
    x_target = np.concatenate([E_tgt, I_tgt])
    E_nat, I_nat = wc_a2.state_at_time(t_arr, E_arr, I_arr, t_ref + T_ms)
    x_natural = np.concatenate([E_nat, I_nat])
    target_distance = float(np.linalg.norm(x_target - x_natural))

    def inject(U):
        u_func = physics.zero_order_hold(U, DT)
        r = net.simulate_controlled(u_func=u_func, t_span=(0.0, T_ms),
                                    n_points=max(100, T_STEPS),
                                    E0=E_ref, I0=I_ref)
        xf = np.concatenate([r["E"][:, -1], r["I"][:, -1]])
        return float(np.linalg.norm(xf - x_target) / target_distance)

    # --- naive: frozen Jacobian at the reference state ---
    J0 = net.jacobian_at(E_ref, I_ref)
    G0 = simulation.input_jacobian_at(net, E_ref, I_ref)
    A, B = simulation.discretize_system(J0, G0, DT)
    _, U_naive = physics.minimum_energy_trajectory(
        A, B, np.zeros(2 * N), x_target - x_ref, T_STEPS)
    naive = dict(err=inject(U_naive), energy=control_energy(U_naive, DT),
                 peak=peak_amplitude(U_naive))

    # --- LTV along the free-running reference ---
    _, U_ltv = ltv_control_sequence(net, t_arr, E_arr, I_arr, t_ref, T_STEPS,
                                    x_target, x_natural, E_ref, I_ref)
    ltv = dict(err=inject(U_ltv), energy=control_energy(U_ltv, DT),
               peak=peak_amplitude(U_ltv))

    jac_f = make_wc_jacobian_func(net)
    inp_f = make_wc_input_jacobian_func(net)

    rng = np.random.default_rng(
        zlib.crc32(f"wc|{seed}|{shift_ms}".encode()) % (2**32))
    scale = max(np.abs(U_ltv).max(), 1e-3)
    starts = {"cold": np.zeros((T_STEPS, N)), "naive": U_naive.copy(),
              "ltv": U_ltv.copy(),
              "rand0": rng.normal(0, scale, (T_STEPS, N))}

    cid = f"seed{seed}|shift{shift_ms}"
    rows = []
    for init, U0 in starts.items():
        for lam in lam_grid:
            def obj(u_flat, lam=lam):
                U = u_flat.reshape(T_STEPS, N)
                c, g, _ = compute_adjoint_gradient(
                    net, x_ref, x_target, U, DT, jac_f, inp_f,
                    lambda_reg=lam)
                return c, g.flatten()
            try:
                r = frontier_point(obj, U0, inject, DT)
            except Exception as exc:
                r = dict(err=np.nan, energy=np.nan, peak=np.nan,
                         n_evals=np.nan, converged=False,
                         error=str(exc)[:80])
            r.update(condition_id=cid, seed=seed, shift_ms=shift_ms,
                     init=init, lambda_reg=lam,
                     naive_err=naive["err"], naive_energy=naive["energy"],
                     naive_peak=naive["peak"], ltv_err=ltv["err"],
                     ltv_energy=ltv["energy"], ltv_peak=ltv["peak"],
                     target_distance=target_distance)
            rows.append(r)
    return rows


def add_dominance(df):
    for lin in ("naive", "ltv"):
        df[f"dominated_{lin}"] = np.nan
        df[f"dom_energy_ratio_{lin}"] = np.nan
        df[f"dom_err_ratio_{lin}"] = np.nan
    for cid, g in df.groupby("condition_id"):
        pts = g.dropna(subset=["energy", "err"])
        if not len(pts):
            continue
        for lin in ("naive", "ltv"):
            le, lr = g[f"{lin}_energy"].iloc[0], g[f"{lin}_err"].iloc[0]
            if not (np.isfinite(le) and np.isfinite(lr)):
                continue
            dom = pts[(pts.energy <= le) & (pts.err <= lr)]
            m = df.condition_id == cid
            df.loc[m, f"dominated_{lin}"] = float(len(dom) > 0)
            if len(dom):
                b = dom.loc[dom.energy.idxmin()]
                df.loc[m, f"dom_energy_ratio_{lin}"] = b.energy / le
                df.loc[m, f"dom_err_ratio_{lin}"] = b.err / lr
    return df


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lambda-grid", default=None)
    ap.add_argument("--n-workers", type=int, default=0)
    ap.add_argument("--max-conditions", type=int, default=0)
    args = ap.parse_args()

    lam_grid = ([float(x) for x in args.lambda_grid.split(",")]
                if args.lambda_grid else LAMBDA_GRID)
    combos = [(s, sh, lam_grid) for s in SEEDS for sh in SHIFTS_MS]
    if args.max_conditions:
        combos = combos[:args.max_conditions]
    print(f"{len(combos)} conditions x {len(lam_grid)} penalties "
          f"x {len(INITS)} initialisations   (N={N}, T_steps={T_STEPS})")

    n_workers = args.n_workers or min(cpu_count(), len(combos))
    print(f"  {n_workers} workers")
    rows = []
    with Pool(n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_condition, combos), 1):
            rows.extend(r)
            print(f"  {i}/{len(combos)}", flush=True)

    df = add_dominance(pd.DataFrame(rows))
    df.to_csv(args.out, index=False)
    ok = df.dropna(subset=["err"])
    per = df.groupby("condition_id").first()

    print("\n" + "=" * 78)
    print("MATCHED-COST COMPARISON — WILSON-COWAN")
    print("=" * 78)

    print("\n1. PARETO DOMINANCE  <-- primary, assumption-free")
    for lin in ("naive", "ltv"):
        d = per[f"dominated_{lin}"].dropna()
        if len(d):
            er = per[f"dom_energy_ratio_{lin}"].dropna()
            rr = per[f"dom_err_ratio_{lin}"].dropna()
            print(f"   {lin.upper():6s} dominated in {int(d.sum())}/{len(d)} "
                  f"({d.mean():.0%})", end="")
            if len(er):
                print(f"   dominating point: {er.median():.4f}x energy, "
                      f"{rr.median():.3f}x error")
            else:
                print()
    print("\n   For comparison, the other two arms:")
    print("     FitzHugh-Nagumo    36/40 (90%)  25.0x less energy, 2.8x lower error")
    print("     Stuart-Landau      17/20 (85%)  27.6x less energy, 2.2x lower error")
    print("     SL subcritical     59/60 (98%)  72.5x less energy, 2.4x lower error")

    print("\n2. THE WARM-START GUARANTEE — note this arm used U_NAIVE")
    for init, note in [("naive", "the original setting here: arithmetic, not evidence"),
                       ("cold", "honest test")]:
        g = ok[(ok.init == init) & (ok.lambda_reg == 0.0)].dropna(
            subset=["err", "naive_err"])
        if len(g):
            print(f"   {init:6s} start, lambda=0: err <= naive_err in "
                  f"{(g.err <= g.naive_err).mean():.0%}   ({note})")

    print("\n3. SENSITIVITY TO INITIALISATION")
    g = ok[ok.lambda_reg == 0.0].groupby("condition_id").err
    sp = ((g.max() - g.min()) / g.min().replace(0, np.nan)).dropna()
    if len(sp):
        print(f"   relative spread across initialisations: median {sp.median():.1%}"
              f"   max {sp.max():.1%}")

    print("\n4. FRONTIER QUALITY")
    bad = 0
    for cid, g in ok.groupby("condition_id"):
        b = g.sort_values("err").groupby("lambda_reg").first().sort_index()
        if len(b) > 2 and not b.energy.is_monotonic_decreasing:
            bad += 1
    print(f"   non-monotone energy-vs-penalty frontiers: {bad}/{per.shape[0]}")

    print("\n" + "-" * 78)
    print("This arm is CONFIRMATORY: dominance is already established in two")
    print("models with matched-cost methodology. Agreement strengthens the")
    print("generality claim; disagreement would mean the result is")
    print("regime-dependent, which the other two models cannot reveal.")
    print("\nNothing from the previous WC A2 grid should be quoted alongside")
    print("these numbers: 32 of its 40 rows were reconstructed from printed")
    print("output at one-decimal precision.")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
