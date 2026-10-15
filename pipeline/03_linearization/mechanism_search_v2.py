#!/usr/bin/env python3
"""
Path 1, Item 1 -- mechanistic account of the Path A1 interior minimum.
REDESIGNED for the actual structure of the real sweep data.

WHY THIS IS NOT THE FIRST VERSION
---------------------------------
The first version scored candidate functional forms by
|predicted argmin - observed argmin| on an averaged curve, with
parabolic refinement. That is the right approach for a finely sampled
parameter, and it validated cleanly on synthetic data with 40 grid
points.

The real sweep has FIVE w_EE values (3.0, 3.3402, 3.6805, 4.0207,
4.361). At that resolution the positional score degenerates: every
prediction and every observation collapses onto one of five points
spaced ~0.34 apart, so the ranking is dominated by grid resolution
rather than by mechanism. It would have produced a confident-looking
table that meant very little.

WHAT THIS VERSION DOES INSTEAD
------------------------------
The dataset's real statistical power is not in the w_EE grid -- it is
in the 238 SUBJECTS, each contributing an independent 5-point curve.
The project's headline Path A1 result is already stated per-subject
(88/238 = 37.0% of HCP subjects show an interior minimum). So the
mechanistic question is posed the same way:

    For each subject independently, does the candidate form pick out
    THE SAME w_EE as that subject's own error minimum?

Score = agreement rate across subjects. This turns a 5-point resolution
problem into a 238-sample classification problem, and it tests the
mechanism against exactly the between-subject variation the paper
already treats as signal.

Two agreement statistics are reported, because they answer different
questions:
  * exact-index agreement  -- does it pick the same grid point?
  * interior/edge agreement -- does it at least reproduce WHETHER the
    subject has an interior minimum?

    NOTE (added after later analysis): this docstring originally
    justified the interior/edge measure by calling "37% vs 96%" the
    paper's contrast between HCP and ADNI. **That contrast is
    confounded and is no longer the paper's claim.** It marginalises
    across target displacement, and at matched displacement the cohorts
    are similar (at ts=0.01/0.02 HCP is HIGHER than ADNI). The interior
    fraction also carries ~+/-0.08 run-to-run variability from random
    target-direction sampling. The paper's primary statistic is now the
    CRITICAL TARGET DISPLACEMENT, not the interior fraction. See
    `localize_critical_displacement.py` and this directory's README.
    The mechanism search below is unaffected -- it asks a different
    question (can any combining form predict per-subject argmin) and
    its answer was a clean null either way.

AMPLIFIER VARIABLE
------------------
The documented two-trend story used Gramian conditioning (~85 -> ~130).
That quantity is NOT in this sweep file. Available instead are
`stability_margin` and `rho_A`, which correlate at exactly -1.0 and are
therefore redundant -- one carries all the information.

Using 1/stability_margin as the amplifier is a SUBSTITUTION, not the
original hypothesis, and the paper must say so. It is arguably the
cleaner variable: it states distance-to-bifurcation directly rather
than through a derived matrix property, and it diverges in the same
place Gramian conditioning was blowing up. But it is a different claim
and should not be presented as the original one confirmed.

NUISANCE DIMENSIONS
-------------------
The sweep also varies T (5 levels), target_scale (5), and dir_seed (3).
Averaging over these could wash out the effect. By default this script
runs the search SEPARATELY at each (T, target_scale) cell and reports
whether the winning candidate is stable across cells -- a candidate
that only wins in one corner of the design is not a mechanism. Use
--marginalize to average instead, and --fix-T / --fix-target-scale to
pin a single cell.

USAGE
-----
    python3 mechanism_search_v2.py \
        --sweep /path/to/wc_linear_validity_sweep_real_coupling_HCP_full.csv \
        --out mechanism_v2_results
"""
import argparse
import os
import itertools

import numpy as np
import pandas as pd


def build_candidates(energy, amplifier):
    """Candidate combining forms. Each encodes a distinct mechanistic story.

    energy    : control energy (falls as w_EE rises)
    amplifier : 1 / stability_margin (rises as w_EE rises)

    An interior minimum requires one falling and one rising factor whose
    product turns over inside the range. Additive forms are included as
    controls: if an additive form wins, the 'competing trends' framing
    is wrong in kind, not just in exponent.
    """
    eps = 1e-30
    e = np.maximum(energy, eps)
    a = np.maximum(amplifier, eps)
    return {
        "energy * amp": e * a,
        "energy * amp^2": e * a ** 2,
        "energy^2 * amp": e ** 2 * a,
        "sqrt(energy) * amp": np.sqrt(e) * a,
        "energy * sqrt(amp)": e * np.sqrt(a),
        "energy * (1 + amp)": e * (1.0 + a),
        "energy^2 * amp^2": e ** 2 * a ** 2,
        # additive controls (normalised so the sum is meaningful)
        "energy + amp (norm)": e / np.nanmax(e) + a / np.nanmax(a),
        # energy alone / amplifier alone: monotone, cannot give an interior
        # minimum. Included so their failure confirms the test discriminates.
        "energy only (control)": e,
        "amp only (control)": a,
    }


def analyse_cell(sub_df, param_col, err_col, energy_col, amp_col):
    """Per-subject argmin agreement within one (T, target_scale) cell."""
    subjects = sub_df["subject_id"].unique()
    param_vals = np.sort(sub_df[param_col].unique())
    n_param = len(param_vals)

    cand_names = list(build_candidates(np.array([1.0]), np.array([1.0])).keys())
    exact_hits = {c: 0 for c in cand_names}
    interior_hits = {c: 0 for c in cand_names}
    n_used = 0
    obs_interior = 0
    obs_index_counts = np.zeros(n_param, dtype=int)

    for sid in subjects:
        d = sub_df[sub_df["subject_id"] == sid].groupby(param_col).mean(numeric_only=True)
        d = d.reindex(param_vals)
        if d[err_col].isna().any():
            continue
        err = d[err_col].values
        energy = d[energy_col].values
        amp = 1.0 / np.maximum(d[amp_col].values, 1e-30)

        obs_idx = int(np.argmin(err))
        obs_index_counts[obs_idx] += 1
        obs_is_interior = 0 < obs_idx < n_param - 1
        obs_interior += obs_is_interior
        n_used += 1

        for name, score in build_candidates(energy, amp).items():
            if not np.all(np.isfinite(score)):
                continue
            pred_idx = int(np.argmin(score))
            if pred_idx == obs_idx:
                exact_hits[name] += 1
            if (0 < pred_idx < n_param - 1) == obs_is_interior:
                interior_hits[name] += 1

    if n_used == 0:
        return None

    rows = []
    for name in cand_names:
        rows.append({
            "candidate": name,
            "exact_agreement": exact_hits[name] / n_used,
            "interior_agreement": interior_hits[name] / n_used,
        })
    return {
        "n_subjects": n_used,
        "observed_interior_fraction": obs_interior / n_used,
        "observed_index_distribution": obs_index_counts / n_used,
        "param_vals": param_vals,
        "results": pd.DataFrame(rows).sort_values("exact_agreement", ascending=False),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--param-col", default="w_EE")
    ap.add_argument("--err-col", default="rel_reachability_error")
    ap.add_argument("--energy-col", default="energy")
    ap.add_argument("--amp-col", default="stability_margin",
                    help="Amplifier is computed as 1/this. stability_margin "
                         "and rho_A are redundant (corr = -1.0 exactly).")
    ap.add_argument("--fix-T", type=int, default=None)
    ap.add_argument("--fix-target-scale", type=float, default=None)
    ap.add_argument("--marginalize", action="store_true",
                    help="Average over T/target_scale instead of running each "
                         "cell separately. Faster, but can wash out the effect.")
    ap.add_argument("--out", default="mechanism_v2_results")
    args = ap.parse_args()

    df = pd.read_csv(args.sweep)
    os.makedirs(args.out, exist_ok=True)

    if args.fix_T is not None:
        df = df[df["T"] == args.fix_T]
    if args.fix_target_scale is not None:
        df = df[np.isclose(df["target_scale"], args.fix_target_scale)]

    print("=" * 74)
    print("SETUP")
    print("=" * 74)
    print(f"  rows: {len(df)}, subjects: {df.subject_id.nunique()}")
    print(f"  {args.param_col}: {sorted(df[args.param_col].unique())}")
    print(f"  amplifier = 1 / {args.amp_col}")
    print(f"  NOTE: this substitutes for Gramian conditioning, which is not")
    print(f"        present in this file. State the substitution in the paper.")

    cells = []
    if args.marginalize or ("T" not in df.columns):
        cells = [("ALL", df)]
    else:
        for T_val, ts_val in itertools.product(sorted(df["T"].unique()),
                                               sorted(df["target_scale"].unique())):
            cell = df[(df["T"] == T_val) & np.isclose(df["target_scale"], ts_val)]
            if len(cell) > 0:
                cells.append((f"T={T_val},scale={ts_val}", cell))

    print(f"  analysing {len(cells)} cell(s)")
    print()

    all_rows = []
    winners = []
    for label, cell in cells:
        res = analyse_cell(cell, args.param_col, args.err_col,
                           args.energy_col, args.amp_col)
        if res is None:
            continue
        top = res["results"].iloc[0]
        winners.append(top["candidate"])
        for _, r in res["results"].iterrows():
            all_rows.append({"cell": label, **r.to_dict(),
                             "n_subjects": res["n_subjects"],
                             "observed_interior_fraction":
                                 res["observed_interior_fraction"]})

    out = pd.DataFrame(all_rows)
    out.to_csv(os.path.join(args.out, "per_cell_agreement.csv"), index=False)

    print("=" * 74)
    print("POOLED RANKING (mean agreement across cells)")
    print("=" * 74)
    pooled = (out.groupby("candidate")[["exact_agreement", "interior_agreement"]]
              .mean().sort_values("exact_agreement", ascending=False))
    print(pooled.to_string())
    pooled.to_csv(os.path.join(args.out, "pooled_ranking.csv"))

    n_param = df[args.param_col].nunique()

    # BASELINE CORRECTION -- read this before interpreting any score.
    #
    # The first version of this script compared agreement against uniform
    # chance (1/n_param = 0.200), which is the WRONG baseline and produces
    # a false positive.
    #
    # `energy` is monotone decreasing in w_EE, so its argmin is ALWAYS the
    # last grid point. It therefore scores exactly the fraction of subjects
    # whose minimum happens to sit at that edge -- for free, with no
    # mechanism, and while getting every interior-minimum subject wrong.
    # Those interior subjects are precisely the phenomenon under study.
    #
    # The correct baseline is the MAJORITY CLASS: what a constant predictor
    # achieves by always guessing the most common argmin index. A candidate
    # only demonstrates explanatory power by beating that.
    obs_dist = np.zeros(n_param)
    for _, cell in cells:
        r = analyse_cell(cell, args.param_col, args.err_col,
                         args.energy_col, args.amp_col)
        if r is not None:
            obs_dist += r["observed_index_distribution"] * r["n_subjects"]
    obs_dist = obs_dist / max(obs_dist.sum(), 1)
    majority_rate = float(obs_dist.max())
    majority_idx = int(np.argmax(obs_dist))

    print(f"\n  observed argmin distribution across grid points: "
          f"{np.round(obs_dist, 3).tolist()}")
    print(f"  MAJORITY-CLASS BASELINE = {majority_rate:.3f} "
          f"(constant predictor always guessing index {majority_idx})")
    print(f"  (uniform-chance would be {1.0/n_param:.3f}, but that is the "
          f"WRONG baseline here -- see comment in source)")

    best = pooled.index[0]
    best_score = pooled.iloc[0]["exact_agreement"]
    print(f"  best candidate: {best}  ({best_score:.3f})")

    if best_score <= majority_rate + 1e-9:
        print("  -> DOES NOT BEAT A CONSTANT PREDICTOR. No candidate form has")
        print("     any power to predict WHICH subjects show an interior")
        print("     minimum -- which is the actual question. This is a clean")
        print("     null. Honest paper statement: 'characterized, not")
        print("     explained'.")
    elif best_score < majority_rate + 0.05:
        print("  -> barely exceeds a constant predictor. Not a mechanism claim.")
    else:
        print(f"  -> beats the majority-class baseline by "
              f"{best_score - majority_rate:.3f}. Worth deriving analytically.")

    # Interior-subject-only score: the question that actually matters.
    print()
    print("  RESTRICTED TO INTERIOR-MINIMUM SUBJECTS (the phenomenon of")
    print("  interest): a form that only predicts edge minima explains")
    print("  nothing, so this column is the one to read.")
    print("  -> see interior_agreement in the ranking above; any form whose")
    print("     exact_agreement is driven entirely by edge subjects will")
    print("     show high interior_agreement only by coincidence.")

    # Stability of the winner across cells -- a mechanism should not depend
    # on which corner of the nuisance design you look at.
    if len(cells) > 1:
        from collections import Counter
        c = Counter(winners)
        print()
        print("  winner stability across cells:")
        for name, cnt in c.most_common():
            print(f"    {name:28s} wins in {cnt}/{len(winners)} cells")
        if c.most_common(1)[0][1] < 0.6 * len(winners):
            print("  -> WINNER IS NOT STABLE across the nuisance design.")
            print("     A form that only wins in some cells is not a mechanism.")

    print()
    print("  IDENTIFIABILITY LIMIT -- read before interpreting ties:")
    print("  argmin is invariant under any monotone transform, so forms")
    print("  related by a positive power are INDISTINGUISHABLE here:")
    print("  e.g. energy*amp^2 and sqrt(energy)*amp share an argmin exactly")
    print("  (the second is the square root of the first). Only the RATIO")
    print("  of exponents is identifiable, never their absolute scale.")
    print("  Ties of this kind are expected and are not a defect; report the")
    print("  identified ratio, not a single arbitrary member of the family.")
    print()
    print("  observed interior fraction (sanity check against the published")
    print(f"  37.0% for HCP): {out['observed_interior_fraction'].mean():.3f}")
    print(f"\nSaved to {args.out}/")


if __name__ == "__main__":
    main()
