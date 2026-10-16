#!/usr/bin/env python3
"""
brain_maps.py — the per-region values drawn on the cortex in Figures 1f, 2g
and 2h, computed from the saved bootstrap effective-connectivity ensembles
({subject}_EC_boot.npy, shape (resamples, regions, regions)).

  drivers   For one subject per cohort, the fraction of the 200 resamples in
            which each region is among the five strongest drivers (Figures
            2g, 2h). The subject is chosen by rule: the one whose driver
            Kendall's tau is closest to the cohort median.
  senders   For every subject of a cohort, each region's out-strength minus
            in-strength of the bootstrap-mean EC estimate, averaged over
            subjects (Figure 1f). Positive: the region sends more than it
            receives.
  export    The .npz files written by the two commands above, as the tables
            deposited in results/02_ec_uncertainty/brain_maps/. Per-subject
            rows are not exported.
  selftest  No data: the driver score finds planted senders and not planted
            receivers.

CONVENTION. The estimator fits x(t+1) = A x(t), so A[i, j] is the influence
of region j on region i: column j holds j's outgoing connections, row i its
incoming ones. The sender value is signed (column sum minus row sum).

DRIVER DEFINITION. Average controllability as defined by Gu et al. (2015):
input at region j alone, sum_t ||A^t e_j||^2, after rescaling A to spectral
radius 0.9; equal to nctsens.physics.driver_average_controllability on the
rescaled matrix (tests/test_brain_maps.py). Rankings, Kendall's tau and
top-5 Jaccard are computed as in compare_ctrl_definitions.py.

CHECK. `drivers` recomputes the chosen subject's top-5 Jaccard and stops
unless it equals the value in the per-subject driver results
(ctrl_definition_comparison_<cohort>.csv, column jaccard_gu) to 1e-9: the
ensemble on disk and this code are the ones behind the published values.
With an even number of subjects the median can fall midway between two, and
the first in file order is taken. reported_values.py repeats the comparison
from the deposited files.

ROW ORDER. The 410-region (HCP) and 450-region (ADNI-450) connectomes put
the 50 Tian S3 subcortical regions first (ROW_BLOCKS below; for HCP the
label scheme is in pipeline/00_preprocessing/SCMatrixExtractor.py). Only
the cortical blocks are drawn on the surface.

Runs where the ensembles are; numpy, scipy and pandas only, so it does not
need the package installed.

    python3 brain_maps.py selftest
    python3 brain_maps.py drivers --ec-dir DIR --reference ctrl_definition_comparison_HCP.csv --label HCP --out maps/
    python3 brain_maps.py senders --ec-dir DIR --label HCP --out maps/
    python3 brain_maps.py export --maps maps/ --out results/02_ec_uncertainty/brain_maps/
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")

import argparse
import glob
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from scipy.stats import kendalltau

TOP_K = 5
TARGET_RHO = 0.9
COHORTS = ("HCP", "ADNI-400", "ADNI-450")

# (block, first row, end row); parcels are numbered from 1 within a block, in atlas label order
ROW_BLOCKS = {
    "HCP": (("Tian S3 subcortical", 0, 50), ("HCP-MMP right", 50, 230), ("HCP-MMP left", 230, 410)),
    "ADNI-400": (("Schaefer-400", 0, 400),),
    "ADNI-450": (("Tian S3 subcortical", 0, 50), ("Schaefer-400", 50, 450)),
}


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------
def normalise_matrix(A, target_rho=TARGET_RHO):
    """Rescale A so its spectral radius equals target_rho."""
    A = np.asarray(A, dtype=float)
    rho = float(np.max(np.abs(np.linalg.eigvals(A))))
    if rho < 1e-12:
        raise ValueError("matrix has near-zero spectral radius")
    return A * (target_rho / rho)


def driver_score(A):
    """Gu et al. (2015) average controllability: input at each region alone."""
    An = normalise_matrix(A)
    return np.diag(solve_discrete_lyapunov(An.T, np.eye(An.shape[0])))


def score_ensemble(EC_boot):
    """Per-resample driver scores, rankings, top-5 frequency, and the two stability metrics."""
    scores = np.array([driver_score(EC_boot[b]) for b in range(len(EC_boot))])
    ranks = np.argsort(-scores, axis=1)
    tops = [set(r[:TOP_K]) for r in ranks]
    pairs = list(combinations(range(len(scores)), 2))
    jac = float(np.mean([len(tops[i] & tops[j]) / len(tops[i] | tops[j]) for i, j in pairs]))
    tau = float(np.mean([kendalltau(scores[i], scores[j])[0] for i, j in pairs]))
    freq = np.zeros(scores.shape[1])
    for s in tops:
        freq[list(s)] += 1
    freq /= len(tops)
    shared = np.array([len(tops[i] & tops[j]) for i, j in pairs])
    return dict(scores=scores, freq=freq, jaccard=jac, tau=tau, shared=shared)


def out_minus_in(EC):
    """Out-strength minus in-strength per region; A[i, j] is j -> i."""
    return EC.sum(axis=0) - EC.sum(axis=1)


def ensemble_path(ec_dir, sid):
    for cand in (sid, f"sub-{sid}", sid.replace("sub-", "")):
        p = os.path.join(ec_dir, f"{cand}_EC_boot.npy")
        if os.path.exists(p):
            return p
    sys.exit(f"STOP: no {sid}_EC_boot.npy in {ec_dir}")


# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------
def cmd_drivers(a):
    ref = pd.read_csv(a.reference)
    if a.subject:
        row = ref[ref.subject_id.astype(str) == a.subject]
        if not len(row):
            sys.exit(f"STOP: {a.subject} is not in {a.reference}")
        row = row.iloc[0]; how = "given on the command line"
    else:
        if "error" in ref:
            ref = ref[ref.error.fillna("") == ""]
        ref = ref.dropna(subset=["tau_gu", "jaccard_gu"])
        med = ref.tau_gu.median()
        row = ref.iloc[(ref.tau_gu - med).abs().argsort().iloc[0]]
        how = f"closest to the cohort median driver tau ({med:.3f})"
    sid = str(row.subject_id)
    print(f"{a.label}: subject chosen by rule, {how}; its tau {row.tau_gu:.3f}")
    EC = np.load(ensemble_path(a.ec_dir, sid))
    N = EC.shape[1]
    r = score_ensemble(EC)
    print(f"  CHECK  top-5 Jaccard recomputed {r['jaccard']:.10f}   reference {row.jaccard_gu:.10f}")
    if abs(r["jaccard"] - row.jaccard_gu) > 1e-9:
        sys.exit("  STOP: does not match the reference. The ensemble on disk, or the cohort label, "
                 "is not the one behind the published values.")
    print(f"  CHECK  passed. Kendall tau {r['tau']:.4f} (reference {row.tau_gu:.4f}); "
          f"mean top-5 regions shared by a pair of resamples {r['shared'].mean():.2f} of {TOP_K}")
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, f"driver_map_{a.label}")
    np.savez(base + ".npz", freq=r["freq"], mean_score=r["scores"].mean(axis=0),
             shared=r["shared"], jaccard=r["jaccard"], tau=r["tau"],
             n_boot=len(EC), n_regions=N, rule=how)
    print(f"  wrote {base}.npz")


def cmd_senders(a):
    files = sorted(glob.glob(os.path.join(a.ec_dir, "*_EC_boot.npy")))
    if a.max_subjects:
        files = files[:a.max_subjects]
    if not files:
        sys.exit(f"STOP: no *_EC_boot.npy in {a.ec_dir}")
    vals = []
    for k, f in enumerate(files, 1):
        vals.append(out_minus_in(np.load(f, mmap_mode="r").mean(axis=0)))
        if k % 10 == 0 or k == len(files):
            print(f"  {k}/{len(files)}", flush=True)
    vals = np.array(vals); m = vals.mean(axis=0); N = len(m)
    consistent = np.mean(np.sign(vals) == np.sign(m), axis=0)
    print(f"{a.label}: {len(files)} subjects; out-minus-in range {m.min():.3g} to {m.max():.3g}; "
          f"median sign agreement across subjects {np.median(consistent):.2f}")
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, f"sender_receiver_map_{a.label}")
    np.savez(base + ".npz", mean=m, per_subject=vals, sign_agreement=consistent,
             n_subjects=len(files), n_regions=N)
    print(f"  wrote {base}.npz")


def regions(cohort, n):
    rows = []
    for block, s, e in ROW_BLOCKS[cohort]:
        rows += [(r, block, r - s + 1) for r in range(s, e)]
    if len(rows) != n:
        sys.exit(f"STOP: {cohort} map has {n} regions; the row blocks cover {len(rows)}")
    return pd.DataFrame(rows, columns=["row", "block", "parcel"])


def cmd_export(a):
    os.makedirs(a.out, exist_ok=True)
    summary = []
    for c in COHORTS:
        d = np.load(os.path.join(a.maps, f"driver_map_{c}.npz"), allow_pickle=False)
        t = regions(c, int(d["n_regions"]))
        t["top5_frequency"] = d["freq"]; t["mean_driver_score"] = d["mean_score"]
        t.to_csv(os.path.join(a.out, f"driver_map_{c}.csv"), index=False, lineterminator="\n")
        summary.append(dict(cohort=c, map="driver", n_regions=int(d["n_regions"]), n_resamples=int(d["n_boot"]),
                            n_subjects=1, driver_tau=float(d["tau"]), driver_jaccard=float(d["jaccard"]),
                            mean_top5_shared=float(d["shared"].mean()), selection=str(d["rule"])))
        s = np.load(os.path.join(a.maps, f"sender_receiver_map_{c}.npz"), allow_pickle=False)
        t = regions(c, int(s["n_regions"]))
        t["out_minus_in"] = s["mean"]; t["sign_agreement"] = s["sign_agreement"]
        t.to_csv(os.path.join(a.out, f"sender_receiver_map_{c}.csv"), index=False, lineterminator="\n")
        summary.append(dict(cohort=c, map="sender_receiver", n_regions=int(s["n_regions"]),
                            n_subjects=int(s["n_subjects"]), selection="all subjects"))
    summary = pd.DataFrame(summary).astype({"n_resamples": "Int64"})
    summary.to_csv(os.path.join(a.out, "brain_maps_summary.csv"), index=False, lineterminator="\n")
    print(f"wrote {len(COHORTS) * 2} maps and brain_maps_summary.csv to {a.out}")


def cmd_selftest(a):
    rng = np.random.default_rng(0); N = 60
    base = np.abs(rng.normal(0, 0.02, (N, N)))
    A = base.copy(); A[:, :5] += 0.6          # columns 0-4: regions 0-4 send to all
    top = set(np.argsort(-driver_score(A))[:TOP_K])
    ok1 = top == set(range(5))
    print(f"  planted senders 0-4 recovered as top drivers: {ok1}  {sorted(top)}")
    B = base.copy(); B[:5, :] += 0.6          # rows 0-4: regions 0-4 receive from all
    top_r = set(np.argsort(-driver_score(B))[:TOP_K])
    ok2 = not (top_r & set(range(5)))
    print(f"  planted receivers 0-4 not ranked as drivers: {ok2}  {sorted(top_r)}")
    ok3 = bool(np.all(out_minus_in(A)[:5] > 0) and np.all(out_minus_in(B)[:5] < 0))
    print(f"  out-minus-in positive for planted senders, negative for planted receivers: {ok3}")
    if not (ok1 and ok2 and ok3):
        sys.exit("SELFTEST FAILED")
    print("SELFTEST PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["drivers", "senders", "export", "selftest"])
    ap.add_argument("--ec-dir"); ap.add_argument("--reference")
    ap.add_argument("--subject", help="override the median-tau rule")
    ap.add_argument("--label", default="cohort", choices=COHORTS + ("cohort",))
    ap.add_argument("--maps", default="maps", help="export: directory holding the .npz files")
    ap.add_argument("--out", default="maps")
    ap.add_argument("--max-subjects", type=int, default=0)
    a = ap.parse_args()
    if a.command in ("drivers", "senders") and not a.ec_dir:
        sys.exit("--ec-dir is required")
    if a.command == "drivers" and not a.reference:
        sys.exit("--reference (ctrl_definition_comparison_<cohort>.csv) is required")
    {"drivers": cmd_drivers, "senders": cmd_senders, "export": cmd_export, "selftest": cmd_selftest}[a.command](a)


if __name__ == "__main__":
    main()
