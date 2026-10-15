#!/usr/bin/env python3
"""
Path 1, Item 1 -- mechanistic account of the Path A1 reachability-error
interior minimum.

THE PROBLEM THIS ATTACKS
------------------------
Three hypotheses were tested previously. Two were ruled out. The
survivor is qualitative and its two ingredients are CONFIRMED REAL:
as w_EE rises toward bifurcation,

    (a) Gramian conditioning worsens monotonically  (~85 -> ~130)
    (b) control energy falls monotonically

Two opposing monotonic trends can produce an interior minimum in their
combination. That story is plausible and the ingredients are verified.
What failed was the specific guessed combining function

    score(w) = energy(w) * (1 + conditioning(w))

which does not place its minimum at the observed w_EE ~ 3.68.

The failure is therefore a FUNCTIONAL-FORM problem, not an absence of
mechanism. Guessing one form at a time is a poor way to solve it. This
script searches a principled library of forms systematically and scores
each by whether it predicts the OBSERVED MINIMUM LOCATION.

WHY LOCATION, NOT CORRELATION
-----------------------------
A candidate can correlate strongly with the error curve while placing
its minimum somewhere else entirely -- both quantities are dominated by
their shared monotone components. Correlation would reward that. The
scientific claim is specifically "these two trends cross HERE, and that
is why the minimum is HERE," so the test must be positional. Candidates
are ranked by |predicted argmin - observed argmin| in w_EE.

WHY lambda_min IS IN THE LIBRARY ALONGSIDE cond
-----------------------------------------------
The original guess used the condition number. There is a specific
reason to suspect that is the wrong quantity. Reachability error arises
because a control input computed from the linear model is applied to the
nonlinear system; the resulting trajectory error is amplified by the
Gramian's SMALLEST eigenvalue (the least controllable direction),
which sets how badly a given input error propagates. The condition
number is a RATIO -- it can stay flat while lambda_min collapses, or
move for reasons driven entirely by lambda_max. The library therefore
contains both, so the data can distinguish them rather than the choice
being inherited from the first guess.

A FREE-EXPONENT FAMILY IS ALSO FITTED
-------------------------------------
    score(w) = energy(w)^alpha * amplifier(w)^beta

with (alpha, beta) fitted to place the minimum correctly. This is
diagnostic, not explanatory: if the best fit lands near interpretable
values (e.g. alpha=1, beta=1, or alpha=2, beta=1) that points at a
specific derivation; if it requires bizarre exponents, that is evidence
the two-trend story is incomplete and something else is missing. Either
outcome is informative and both are reported honestly.

NULL CHECK
----------
Reported alongside every result: how well would a RANDOM w_EE in the
swept range do? If the best candidate is not clearly better than
chance, the script says so plainly rather than presenting a ranked
table that implies success.

USAGE
-----
    # first, see what columns your sweep file actually has:
    python3 mechanism_functional_form_search.py --sweep <file.csv> --list-columns

    # then run with the mapping:
    python3 mechanism_functional_form_search.py \
        --sweep /path/to/wc_linear_validity_sweep_real_coupling_HCP_full.csv \
        --col-param w_EE \
        --col-error reachability_error \
        --col-energy control_energy \
        --col-cond gramian_cond \
        --col-lmin gramian_min_eig \
        --out mechanism_search_results
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _argmin_location(param, values):
    """Location of the minimum, refined by parabolic interpolation.

    Grid resolution otherwise floors the achievable accuracy: with a
    coarse w_EE grid, a candidate could be 'wrong' merely because the
    true minimum sits between samples. Parabolic refinement around the
    discrete argmin removes that artifact.
    """
    values = np.asarray(values, dtype=float)
    param = np.asarray(param, dtype=float)
    i = int(np.nanargmin(values))
    if i == 0 or i == len(values) - 1:
        return float(param[i]), True  # boundary -- flag it
    x0, x1, x2 = param[i - 1], param[i], param[i + 1]
    y0, y1, y2 = values[i - 1], values[i], values[i + 1]
    denom = (y0 - 2 * y1 + y2)
    if abs(denom) < 1e-15:
        return float(x1), False
    delta = 0.5 * (y0 - y2) / denom
    return float(x1 + delta * (x2 - x1)), False


def build_candidates(energy, cond, lmin):
    """Principled library. Each entry maps to a distinct mechanistic story."""
    eps = 1e-30
    inv_lmin = 1.0 / np.maximum(lmin, eps)

    return {
        # The original guess, retained as the baseline to beat.
        "energy * (1 + cond)": energy * (1.0 + cond),

        # Same story, no offset -- tests whether the +1 was doing work.
        "energy * cond": energy * cond,

        # Amplification set by the least-controllable direction, not the ratio.
        "energy / lambda_min": energy * inv_lmin,
        "energy^2 / lambda_min": energy ** 2 * inv_lmin,

        # Input error scales with ||u|| ~ sqrt(energy); amplification linear.
        "sqrt(energy) * cond": np.sqrt(np.maximum(energy, 0)) * cond,
        "sqrt(energy) / lambda_min": np.sqrt(np.maximum(energy, 0)) * inv_lmin,

        # Energy is quadratic in input; second-order remainder is quadratic
        # in state deviation -- motivates squaring one factor.
        "energy^2 * cond": energy ** 2 * cond,

        # Additive rather than multiplicative competition.
        "energy + cond": energy / np.nanmax(energy) + cond / np.nanmax(cond),
        "energy + 1/lambda_min": (energy / np.nanmax(energy)
                                  + inv_lmin / np.nanmax(inv_lmin)),

        # Log-domain: if the true relation is a product of powers, the
        # minimum of the sum of logs is the minimum of the product.
        "log(energy) + log(cond)": (np.log(np.maximum(energy, eps))
                                    + np.log(np.maximum(cond, eps))),
    }


def free_exponent_fit(param, error, energy, amplifier, observed_min):
    """Fit score = energy^alpha * amplifier^beta to place the minimum correctly."""
    eps = 1e-30
    log_e = np.log(np.maximum(energy, eps))
    log_a = np.log(np.maximum(amplifier, eps))

    def loss(p):
        alpha, beta = p
        score = alpha * log_e + beta * log_a  # log-domain; argmin is preserved
        loc, boundary = _argmin_location(param, score)
        if boundary:
            return 1e6
        return (loc - observed_min) ** 2

    best = None
    for a0 in [0.5, 1.0, 2.0]:
        for b0 in [0.5, 1.0, 2.0]:
            try:
                r = minimize(loss, [a0, b0], method="Nelder-Mead",
                             options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-8})
                if best is None or r.fun < best.fun:
                    best = r
            except Exception:
                continue
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--list-columns", action="store_true",
                    help="Print the file's columns and exit (run this first)")
    ap.add_argument("--col-param", default="w_EE")
    ap.add_argument("--col-error", default=None)
    ap.add_argument("--col-energy", default=None)
    ap.add_argument("--col-cond", default=None)
    ap.add_argument("--col-lmin", default=None)
    ap.add_argument("--group-by", default=None,
                    help="Optional subject column; if given, curves are "
                         "averaged across subjects before the search")
    ap.add_argument("--out", default="mechanism_search_results")
    args = ap.parse_args()

    df = pd.read_csv(args.sweep)

    if args.list_columns:
        print(f"Columns in {args.sweep}:")
        for c in df.columns:
            print(f"  {c:40s} dtype={df[c].dtype}")
        print(f"\n{len(df)} rows")
        return

    required = {"--col-error": args.col_error, "--col-energy": args.col_energy}
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise SystemExit(
            f"Must specify {', '.join(missing)}. "
            f"Run with --list-columns first to see what is available.")
    if args.col_cond is None and args.col_lmin is None:
        raise SystemExit("Specify at least one of --col-cond / --col-lmin.")

    os.makedirs(args.out, exist_ok=True)

    # Collapse to one curve per parameter value.
    agg_cols = [c for c in [args.col_error, args.col_energy,
                            args.col_cond, args.col_lmin] if c]
    curve = df.groupby(args.col_param)[agg_cols].mean().reset_index()
    curve = curve.sort_values(args.col_param).reset_index(drop=True)

    param = curve[args.col_param].values.astype(float)
    error = curve[args.col_error].values.astype(float)
    energy = curve[args.col_energy].values.astype(float)
    cond = (curve[args.col_cond].values.astype(float)
            if args.col_cond else np.full_like(energy, np.nan))
    lmin = (curve[args.col_lmin].values.astype(float)
            if args.col_lmin else np.full_like(energy, np.nan))

    observed_min, at_boundary = _argmin_location(param, error)
    print("=" * 72)
    print("OBSERVED")
    print("=" * 72)
    print(f"  grid: {len(param)} values of {args.col_param}, "
          f"range [{param.min():.4f}, {param.max():.4f}]")
    print(f"  observed error minimum at {args.col_param} = {observed_min:.4f}"
          + ("   *** AT GRID BOUNDARY -- interior-minimum claim does not hold here ***"
             if at_boundary else ""))

    # Confirm the two ingredient trends are actually monotone in this file.
    def monotone(v):
        d = np.diff(v)
        if np.all(d >= -1e-12):
            return "increasing"
        if np.all(d <= 1e-12):
            return "decreasing"
        return "NON-MONOTONE"
    print(f"  energy trend:      {monotone(energy)}")
    if args.col_cond:
        print(f"  conditioning trend: {monotone(cond)}")
    if args.col_lmin:
        print(f"  lambda_min trend:   {monotone(lmin)}")
    print("  (the two-trend story REQUIRES these to be opposing monotones; "
          "if either is non-monotone here, the premise needs revisiting)")

    # Null baseline: expected error of a uniformly random guess in range.
    rng = np.random.default_rng(0)
    random_guesses = rng.uniform(param.min(), param.max(), 20000)
    null_err = float(np.mean(np.abs(random_guesses - observed_min)))

    print()
    print("=" * 72)
    print("CANDIDATE FUNCTIONAL FORMS (ranked by |predicted - observed| minimum)")
    print("=" * 72)
    cands = build_candidates(energy, cond, lmin)
    rows = []
    for name, score in cands.items():
        if not np.all(np.isfinite(score)):
            rows.append({"candidate": name, "predicted_min": np.nan,
                         "abs_error": np.nan, "note": "non-finite"})
            continue
        loc, boundary = _argmin_location(param, score)
        rows.append({
            "candidate": name,
            "predicted_min": loc,
            "abs_error": abs(loc - observed_min),
            "note": "AT BOUNDARY" if boundary else "",
        })
    res = pd.DataFrame(rows).sort_values("abs_error", na_position="last")
    print(res.to_string(index=False))
    res.to_csv(os.path.join(args.out, "candidate_ranking.csv"), index=False)

    best = res.iloc[0]
    print()
    print(f"  null baseline (random guess): mean |error| = {null_err:.4f}")
    if np.isfinite(best["abs_error"]):
        if best["abs_error"] < 0.25 * null_err:
            verdict = "CLEARLY beats chance -- a real candidate mechanism"
        elif best["abs_error"] < null_err:
            verdict = "beats chance only modestly -- treat as suggestive, not settled"
        else:
            verdict = ("NO BETTER THAN CHANCE -- the two-trend story as "
                       "parameterised here does not explain the minimum")
        print(f"  best candidate error = {best['abs_error']:.4f}  ->  {verdict}")

    # Free-exponent diagnostic.
    print()
    print("=" * 72)
    print("FREE-EXPONENT FIT: score = energy^alpha * amplifier^beta")
    print("=" * 72)
    for amp_name, amp in [("conditioning", cond), ("1/lambda_min",
                                                   1.0 / np.maximum(lmin, 1e-30))]:
        if not np.all(np.isfinite(amp)):
            continue
        fit = free_exponent_fit(param, error, energy, amp, observed_min)
        if fit is None:
            print(f"  {amp_name}: fit failed")
            continue
        alpha, beta = fit.x
        resid = np.sqrt(max(fit.fun, 0))
        # A failed fit returns the boundary penalty (1e6 -> resid 1e3).
        # Reporting exponent 'interpretability' for a failed fit would be a
        # false positive, so the failure is checked FIRST and stated plainly.
        fit_failed = resid > 0.5 * (param.max() - param.min())
        print(f"  amplifier={amp_name}: alpha={alpha:.3f}, beta={beta:.3f}, "
              f"residual |loc error|={resid:.5f}")
        if fit_failed:
            print("     -> FIT FAILED (no exponent pair places the minimum in "
                  "the interior of the swept range). Exponent values above are "
                  "meaningless; do not interpret them.")
        elif 0.3 < abs(alpha) < 3 and 0.3 < abs(beta) < 3:
            print("     -> interpretable exponents; points at a derivable form")
        else:
            print("     -> exponents outside a physically interpretable range "
                  "-- evidence the two-trend story is incomplete")

    print()
    print("INTERPRETATION GUIDE (for the writeup):")
    print("  * A candidate that lands close AND has interpretable exponents")
    print("    is a mechanism worth deriving analytically to confirm.")
    print("  * lambda_min beating cond would be a real finding in itself:")
    print("    it would say amplification is set by the least-controllable")
    print("    direction, not by the spread of the spectrum.")
    print("  * Nothing beating chance is ALSO a result -- it would mean the")
    print("    two confirmed trends are necessary but not sufficient, and")
    print("    the honest paper statement is 'characterized, not explained'.")
    print(f"\nSaved to {args.out}/")


if __name__ == "__main__":
    main()
