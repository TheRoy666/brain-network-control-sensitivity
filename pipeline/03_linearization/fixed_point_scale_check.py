#!/usr/bin/env python3
"""
Is the HCP-vs-ADNI difference in critical target displacement REAL, or a
state-space scale artifact?

THE PROBLEM
-----------
In `wc_linear_validity_sweep_real_coupling.py` the target perturbation is

    target_perturb = target_scale * direction        # direction is a UNIT vector

so `target_scale` is an ABSOLUTE displacement in state space. It is never
normalised by the magnitude of the operating point ||x*||.

The Wilson-Cowan node parameters are identical across cohorts; only the
coupling matrix C differs. Different C gives different fixed points
(E*, I*). If HCP's fixed point sits at a systematically different
magnitude than ADNI's, then "displacement 0.023" means a different
RELATIVE perturbation in each cohort, and a difference in the critical
displacement could be entirely a units artifact.

Measured critical displacements (Phase 25):
    HCP              0.02934  [0.02862, 0.03020]
    ADNI-Schaefer400 0.02333  [0.02236, 0.02513]   <- CIs do NOT overlap with HCP
    ADNI-TianS3      0.02256  [0.02164, 0.04084]

HCP is ~26% higher than ADNI-Schaefer400.

IMPORTANT CONTEXT -- the error metric is ALREADY relative
---------------------------------------------------------
The sweep computes

    target_movement = ||delta_xT||          # == target_scale, direction is unit
    rel_err = ||realized_final - xT_absolute|| / target_movement

so `rel_reachability_error` is already normalised BY the displacement
magnitude. The concern is therefore NOT that larger displacements
mechanically give larger errors -- that is divided out. The concern is
narrower: whether a given absolute `target_scale` represents the same
RELATIVE perturbation of the operating point in each cohort.

Two facts push AGAINST the artifact hypothesis and should be weighed
before concluding either way:
  * Wilson-Cowan E* is a firing rate bounded in [0, 1], so the natural
    scale of the E subspace cannot differ much between cohorts.
  * `load_and_scale_real_coupling` row-sum-normalises C with an
    N-dependent factor, i.e. the coupling is DELIBERATELY scale-matched
    across cohorts and atlas sizes.
Both suggest ||x*|| should be similar across cohorts, which would mean
the observed difference is real. This script measures it rather than
assuming.

THE TEST
--------
Recompute each subject's fixed point at each w_EE and report ||x*||.

  * If HCP's ||x*|| is ~26% larger than ADNI's, the critical
    displacements are CONSISTENT in relative terms and the apparent
    cohort difference is a units artifact. Cohort-independence holds.
  * If ||x*|| is comparable across cohorts, the difference is REAL and
    should be reported as a genuine cohort effect.
  * Anything in between is reported as partial explanation, not a
    resolution.

This is the same class of check that saved the w_EE grid question, where
grids looked unequal in absolute terms but were correctly normalised to
each cohort's stability boundary. Absolute-vs-relative confusion has now
bitten this analysis twice; check it before claiming any cohort contrast.

RATIO TO COMPARE
----------------
The decisive quantity is critical_displacement / median||x*||. If that
ratio matches across cohorts, the underlying regime boundary is the
same and only the units differ.

USAGE
-----
    python3 fixed_point_scale_check.py \
        --sc HCP=/path/to/HCP_SC_matrices:_SC_SIFT2_410.csv \
        --sc ADNI-S400=/path/to/connectomes:_schaefer400_connectome.csv \
        --sc ADNI-TianS3=/path/to/connectomes:_tians3_connectome.csv \
        --out fixed_point_scale_check.csv

Each --sc is LABEL=DIRECTORY:SUFFIX. w_EE grids are taken from the
per-cohort defaults below (the same boundary-normalised grids the
sweeps used); override with --w-ee-values if needed.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

# Same import path convention as the sweep scripts.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from nctsens import simulation
except ImportError:
    sys.exit("Run this from the directory containing simulation.py "
             "(this directory), or install nctsens (pip install -e .).")

# Boundary-normalised w_EE grids actually used per cohort.
DEFAULT_GRIDS = {
    "HCP": [3.0, 3.3402, 3.6805, 4.0207, 4.361],
    "ADNI-S400": [3.0, 3.2422, 3.4845, 3.7268, 3.969],
    "ADNI-TianS3": [3.0, 3.1687, 3.3375, 3.5062, 3.675],
}

# Measured critical displacements, for the ratio comparison.
CRITICAL = {"HCP": 0.02934, "ADNI-S400": 0.02333, "ADNI-TianS3": 0.02256}


# Must match the sweep EXACTLY. Two errors were caught in the first
# version of this script, either of which would have produced fixed
# points from a different model than the sweep actually used:
#   (a) passing w_EE= directly instead of node_params=dict(w_EE=..., **BASE_PARAMS),
#       which would have silently used library defaults for w_IE, w_EI,
#       w_II, c_E, c_I, tau_E, tau_I;
#   (b) loading raw C instead of the log1p-transformed, row-sum-scaled
#       coupling the sweep builds.
BASE_PARAMS = dict(w_IE=4.0, w_EI=3.0, w_II=2.0, c_E=-2.0, c_I=-2.0,
                   tau_E=10.0, tau_I=20.0)


def load_and_scale_real_coupling(sc_path):
    """Verbatim from wc_linear_validity_sweep_real_coupling.py."""
    SC = np.loadtxt(sc_path, delimiter=",")
    N = SC.shape[0]
    np.fill_diagonal(SC, 0)
    SC_log = np.where(SC > 0, np.log1p(SC), 0)
    target_row_sum_mean = (0.075 / N) * 20 * (N - 1)
    scale_factor = target_row_sum_mean / SC_log.sum(axis=1).mean()
    return SC_log * scale_factor, N


def build_network(N, w_EE, C):
    params = dict(w_EE=w_EE, **BASE_PARAMS)
    return simulation.WilsonCowanNetwork(n_regions=N, C=C, node_params=params)


def fixed_point_norm(N, w_EE, C):
    """Return ||x*|| at the stable fixed point, or nan if none exists."""
    try:
        net = build_network(N, w_EE, C)
        E_star, I_star = net.find_fixed_point()
        J = net.jacobian_at(E_star, I_star)
        if np.linalg.eigvals(J).real.max() >= 0:
            return np.nan, np.nan, np.nan
        x_star = np.concatenate([E_star, I_star])
        return (float(np.linalg.norm(x_star)),
                float(np.linalg.norm(E_star)),
                float(np.linalg.norm(I_star)))
    except Exception:
        return np.nan, np.nan, np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc", action="append", required=True,
                    help="LABEL=DIRECTORY:SUFFIX ; repeat per cohort")
    ap.add_argument("--n-subjects", type=int, default=0,
                    help="Limit subjects per cohort (0 = all). A subsample "
                         "is fine: ||x*|| varies little across subjects "
                         "relative to any cross-cohort difference.")
    ap.add_argument("--w-ee-values", default=None,
                    help="Override the per-cohort default grid")
    ap.add_argument("--out", default="fixed_point_scale_check.csv")
    args = ap.parse_args()

    rows = []
    for spec in args.sc:
        if "=" not in spec or ":" not in spec:
            raise SystemExit(f"--sc needs LABEL=DIR:SUFFIX, got {spec!r}")
        label, rest = spec.split("=", 1)
        sc_dir, suffix = rest.rsplit(":", 1)
        grid = ([float(x) for x in args.w_ee_values.split(",")]
                if args.w_ee_values else DEFAULT_GRIDS.get(label))
        if grid is None:
            raise SystemExit(f"No default w_EE grid for {label!r}; "
                             f"pass --w-ee-values")

        files = sorted(f for f in os.listdir(sc_dir) if f.endswith(suffix))
        if args.n_subjects:
            files = files[:args.n_subjects]
        print(f"\n{label}: {len(files)} subjects, w_EE grid {grid}")

        for i, f in enumerate(files, 1):
            sid = f[:-len(suffix)]
            C, N = load_and_scale_real_coupling(os.path.join(sc_dir, f))
            for w in grid:
                n_x, n_E, n_I = fixed_point_norm(N, w, C)
                rows.append({"cohort": label, "subject_id": sid, "N": N,
                             "w_EE": w, "norm_x_star": n_x,
                             "norm_E_star": n_E, "norm_I_star": n_I})
            if i % 10 == 0 or i == len(files):
                print(f"  {i}/{len(files)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print()
    print("=" * 78)
    print("||x*|| BY COHORT  (median across subjects, per w_EE)")
    print("=" * 78)
    for label, g in df.groupby("cohort", sort=False):
        print(f"\n  {label}:")
        for w, gw in g.groupby("w_EE"):
            v = gw.norm_x_star.dropna()
            if len(v):
                print(f"    w_EE={w:<8.4f} median ||x*|| = {v.median():.5f}  "
                      f"IQR [{v.quantile(.25):.5f}, {v.quantile(.75):.5f}]  n={len(v)}")

    print()
    print("=" * 78)
    print("THE DECISIVE COMPARISON")
    print("=" * 78)
    med = df.groupby("cohort").norm_x_star.median()
    print(f"  {'cohort':<14}{'median ||x*||':>15}{'critical ts':>14}"
          f"{'ts / ||x*||':>14}")
    ratios = {}
    for label in med.index:
        if label not in CRITICAL:
            continue
        r = CRITICAL[label] / med[label]
        ratios[label] = r
        print(f"  {label:<14}{med[label]:>15.5f}{CRITICAL[label]:>14.5f}{r:>14.6f}")

    if len(ratios) >= 2:
        vals = np.array(list(ratios.values()))
        spread = (vals.max() - vals.min()) / vals.mean()
        print()
        print(f"  raw critical-displacement spread across cohorts: "
              f"{(max(CRITICAL.values())-min(CRITICAL.values()))/np.mean(list(CRITICAL.values())):.1%}")
        print(f"  SCALE-NORMALISED (ts/||x*||) spread:              {spread:.1%}")
        print()
        if spread < 0.10:
            print("  -> The cohort difference is largely a UNITS ARTIFACT.")
            print("     Normalised by state-space scale the thresholds agree.")
            print("     Report the NORMALISED threshold and state that")
            print("     target_scale is an absolute displacement.")
        elif spread < 0.20:
            print("  -> PARTIALLY explained by scale. Report both the raw")
            print("     difference and the normalisation; do not claim the")
            print("     confound is resolved either way.")
        else:
            print("  -> NOT explained by state-space scale. The cohort")
            print("     difference in critical displacement is REAL and")
            print("     should be reported as such.")

    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
