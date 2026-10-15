#!/usr/bin/env python3
"""
Item 3, follow-up (3 of 3): does ADNI's collapsed rank-stability track
DATA QUALITY within the ADNI cohort itself?

WHY THIS TEST
-------------
The tau dichotomy is real and fully powered (HCP ~0.87, ADNI ~0.07,
non-overlapping). Its cause is not established, and three candidates
have been eliminated by direct experiment:

  * sample count      -- HCP truncated to T=200 gives tau 0.858
  * scan duration     -- same test
  * sampling rate     -- HCP at stride 4 (effective TR 2.88s, matched to
                         ADNI's ~3s) gives tau 0.857

ADNI stays at 0.07 through all of it. Three candidates remain: data
quality / SNR, population (young healthy vs elderly CN/MCI), and
connectome provenance.

This script attacks the first WITHOUT new compute, using a
within-cohort design that the cross-cohort comparison cannot provide.
If low tau is driven by data quality, then WITHIN ADNI the subjects
with worse motion / more censored frames / poorer QC should have
systematically lower tau. If tau is flat against every quality metric
inside ADNI, data quality is unlikely to be the driver and the question
narrows to population or connectome provenance.

WHY WITHIN-COHORT IS THE RIGHT DESIGN HERE
-------------------------------------------
Every cross-cohort comparison confounds quality with population, site,
scanner, pipeline and connectome construction simultaneously. Within
ADNI, population and pipeline are held approximately fixed and quality
varies subject to subject. It is the only lever available that isolates
one candidate at a time using data already on disk.

INTERPRETING A NULL HERE -- READ BEFORE CONCLUDING
---------------------------------------------------
A flat relationship inside ADNI is EVIDENCE AGAINST data quality but
not proof. Two real caveats:

  1. RESTRICTION OF RANGE. Every ADNI subject already passed QC, so the
     quality spread inside the cohort may be far narrower than the
     HCP-vs-ADNI gap. A flat within-cohort slope is compatible with a
     steep relationship that simply is not sampled here. The script
     reports the observed range of each metric so this can be judged
     rather than assumed.
  2. FLOOR EFFECT. ADNI's tau (~0.07) is already close to the
     chance-level floor for rank stability. A variable near its floor
     cannot show a gradient even if the underlying driver varies. The
     script reports how much tau actually varies within ADNI so this is
     visible.

Both caveats are reasons the test can fail to detect a real effect.
Neither is a reason to skip it: a POSITIVE result would be decisive and
is cheap to look for.

USAGE
-----
    # see what columns your QC files actually have first
    python3 adni_quality_vs_taustability.py --list-columns \
        --pathb <path_b_bootstrap_results.csv> \
        --qc <qc_postproc_report.csv>

    # then run with the mapping
    python3 adni_quality_vs_taustability.py \
        --pathb <path_b_bootstrap_results.csv> \
        --qc <qc_postproc_report.csv> \
        --qc-subject-col subject \
        --quality-cols fd_mean,n_censored \
        --meta <participants_qc.tsv> \
        --out adni_quality_vs_tau
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats


def normalise_id(x):
    """ADNI IDs appear as '000S0000', 'sub-000S0000', with/without
    separators depending on the file. Strip to a comparable core."""
    s = str(x).strip()
    for pre in ("sub-", "SUB-"):
        if s.startswith(pre):
            s = s[len(pre):]
    return s.replace("_", "").replace("-", "").upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pathb", required=True,
                    help="ADNI path_b_bootstrap_results.csv (has kendall_tau_mean)")
    ap.add_argument("--qc", default=None, help="QC/motion report csv or tsv")
    ap.add_argument("--meta", default=None, help="participants_qc.tsv")
    ap.add_argument("--conn-qc", default=None,
                    help="qc_connectomes report csv. Supplies CONNECTOME "
                         "PROVENANCE metrics (density, strength_mean, "
                         "strength_std, strength_max, n_empty_nodes) -- the "
                         "third candidate explanation, untestable from the "
                         "functional QC file alone because dwi_dice and "
                         "dwi_fd_mean are empty there.")
    ap.add_argument("--conn-atlas", default="schaefer400",
                    help="Which atlas rows to keep from --conn-qc")
    ap.add_argument("--control-cols", default=None,
                    help="Comma-separated CATEGORICAL columns to control for "
                         "via partial correlation, e.g. manufacturer. ADNI is "
                         "multi-site AND multi-scanner: without this, any "
                         "correlation may be a scanner effect. Verified real "
                         "confound -- the T=197/200 split is EXACTLY the "
                         "GE-vs-(Philips,Siemens) split, and GE's tSNR is ~5x "
                         "lower than the others.")
    ap.add_argument("--list-columns", action="store_true")
    ap.add_argument("--qc-subject-col", default=None)
    ap.add_argument("--meta-subject-col", default=None)
    ap.add_argument("--quality-cols", default=None,
                    help="Comma-separated quality metric column names")
    ap.add_argument("--out", default="adni_quality_vs_tau")
    args = ap.parse_args()

    def read_any(path):
        sep = "\t" if str(path).endswith((".tsv", ".txt")) else ","
        return pd.read_csv(path, sep=sep)

    pathb = read_any(args.pathb)

    if args.list_columns:
        print(f"--- {args.pathb} ---")
        for c in pathb.columns:
            print(f"  {c}  ({pathb[c].dtype})")
        for label, path in [("QC", args.qc), ("META", args.meta)]:
            if path:
                d = read_any(path)
                print(f"\n--- {path} ({label}) ---")
                for c in d.columns:
                    print(f"  {c}  ({d[c].dtype})")
                print(f"  [{len(d)} rows]")
        print("\nPick the subject-ID column and numeric quality columns, then")
        print("rerun with --qc-subject-col / --quality-cols.")
        return

    if not args.quality_cols:
        raise SystemExit("Specify --quality-cols (run --list-columns first).")

    os.makedirs(args.out, exist_ok=True)

    pathb = pathb[pathb["status"] == "ok"].copy()
    pathb["_sid"] = pathb["subject_id"].map(normalise_id)
    tau = pathb[["_sid", "kendall_tau_mean", "jaccard_topk_mean",
                 "n_timepoints"]].copy()

    print("=" * 74)
    print("STEP 1 — how much does tau actually vary within ADNI?")
    print("=" * 74)
    t = tau["kendall_tau_mean"]
    print(f"  n={len(t)}  mean={t.mean():.4f}  sd={t.std():.4f}")
    print(f"  range=[{t.min():.4f}, {t.max():.4f}]  IQR="
          f"[{t.quantile(.25):.4f}, {t.quantile(.75):.4f}]")
    print(f"  coefficient of variation = {t.std()/t.mean():.3f}")
    if t.std() < 0.02:
        print("  *** WARNING: tau is nearly constant within ADNI. A floor")
        print("      effect is likely and NO within-cohort predictor can be")
        print("      detected regardless of the true mechanism. Treat any")
        print("      null below as uninformative, not as evidence. ***")

    frames = []
    quality_cols = [c.strip() for c in args.quality_cols.split(",")]

    for path, subj_col, tag in [(args.qc, args.qc_subject_col, "qc"),
                                (args.meta, args.meta_subject_col, "meta")]:
        if not path:
            continue
        d = read_any(path)
        if subj_col is None:
            subj_col = d.columns[0]
            print(f"\n  (assuming '{subj_col}' is the subject column in {path})")
        d["_sid"] = d[subj_col].map(normalise_id)
        # Control columns MUST be merged in too. Previously only
        # quality_cols were kept, so --control-cols named a column that
        # never reached `merged` and STEP 3b silently did nothing -- a
        # no-op that looked like a clean run. Real bug, found when the
        # partial-correlation block failed to print at all.
        ctrl_wanted = ([c.strip() for c in args.control_cols.split(",")]
                       if args.control_cols else [])
        keep = ["_sid"] + [c for c in (quality_cols + ctrl_wanted)
                           if c in d.columns and c not in ("_sid",)]
        keep = list(dict.fromkeys(keep))
        if len(keep) > 1:
            frames.append(d[keep])

    # Connectome QC -- long format (one row per subject x atlas), so filter
    # to a single atlas before merging or subjects appear twice.
    if args.conn_qc:
        cq = read_any(args.conn_qc)
        if "atlas" in cq.columns:
            cq = cq[cq.atlas == args.conn_atlas]
        subj_col_c = "subject" if "subject" in cq.columns else cq.columns[0]
        cq["_sid"] = cq[subj_col_c].map(normalise_id)
        conn_metrics = [c for c in ["density", "strength_mean", "strength_std",
                                     "strength_max", "strength_min",
                                     "n_empty_nodes", "symmetry_err"]
                        if c in cq.columns]
        if conn_metrics:
            frames.append(cq[["_sid"] + conn_metrics])
            quality_cols.extend(conn_metrics)
            print(f"\n  connectome QC ({args.conn_atlas}): "
                  f"{len(cq)} rows, metrics {conn_metrics}")

    if not frames:
        raise SystemExit("None of --quality-cols found in the QC/meta files.")

    merged = tau
    for f in frames:
        merged = merged.merge(f, on="_sid", how="left", suffixes=("", "_dup"))

    present = [c for c in quality_cols if c in merged.columns]
    print()
    print("=" * 74)
    print("STEP 2 — merge coverage")
    print("=" * 74)
    print(f"  subjects with tau: {len(merged)}")
    for c in present:
        n_ok = merged[c].notna().sum()
        print(f"  {c}: {n_ok}/{len(merged)} matched")
        if n_ok < 0.8 * len(merged):
            print(f"    *** low match rate — check ID formats before trusting "
                  f"the correlation ***")

    merged.to_csv(os.path.join(args.out, "adni_tau_with_quality.csv"), index=False)

    print()
    print("=" * 74)
    print("STEP 3 — does quality predict tau WITHIN ADNI?")
    print("=" * 74)
    rows = []
    for c in present:
        # Skip non-numeric columns rather than crashing. Easy to include a
        # categorical by accident in a long --quality-cols list; a crash
        # after a successful merge wastes the run.
        if not pd.api.types.is_numeric_dtype(merged[c]):
            print(f"  {c}: non-numeric ({merged[c].dtype}) — skipped. "
                  f"If this is a confound, pass it to --control-cols instead.")
            continue
        sub = merged[["kendall_tau_mean", c]].dropna()
        if len(sub) < 10:
            print(f"  {c}: only {len(sub)} usable rows — skipped")
            continue
        if sub[c].nunique() <= 1:
            print(f"  {c}: constant ({sub[c].iloc[0]}) — no variance to "
                  f"correlate, skipped")
            continue
        rho, p = stats.spearmanr(sub[c], sub["kendall_tau_mean"])
        r, p_r = stats.pearsonr(sub[c], sub["kendall_tau_mean"])
        rng = (sub[c].min(), sub[c].max())
        rows.append({"metric": c, "n": len(sub), "spearman_rho": rho,
                     "spearman_p": p, "pearson_r": r, "pearson_p": p_r,
                     "metric_min": rng[0], "metric_max": rng[1],
                     "metric_cv": sub[c].std() / (abs(sub[c].mean()) + 1e-12)})
        print(f"  {c}: n={len(sub)}  Spearman rho={rho:+.4f} (p={p:.4f})  "
              f"Pearson r={r:+.4f} (p={p_r:.4f})")
        print(f"      metric range [{rng[0]:.4g}, {rng[1]:.4g}], "
              f"CV={sub[c].std()/(abs(sub[c].mean())+1e-12):.3f}")

    # --- Partial correlations controlling for categorical confounds ---
    if args.control_cols:
        requested = [c.strip() for c in args.control_cols.split(",")]
        ctrl = [c for c in requested if c in merged.columns]
        missing = [c for c in requested if c not in merged.columns]
        if missing:
            print()
            print(f"  *** --control-cols {missing} NOT FOUND after merge. ***")
            print(f"  Available: {sorted(merged.columns)}")
            print(f"  No control was applied for these. Do NOT read the")
            print(f"  uncontrolled correlations as if they were controlled.")
        if ctrl:
            print()
            print("=" * 74)
            print(f"STEP 3b — PARTIAL correlations controlling for {ctrl}")
            print("  ADNI is multi-site and multi-scanner. An uncontrolled")
            print("  correlation can be a scanner effect in disguise.")
            print("=" * 74)
            for c in present:
                if not pd.api.types.is_numeric_dtype(merged[c]):
                    continue
                sub = merged[["kendall_tau_mean", c] + ctrl].dropna()
                if len(sub) < 12:
                    continue
                # Controls may be CATEGORICAL (manufacturer) or CONTINUOUS
                # (fd_perc, or an effective-degrees-of-freedom variable).
                # Dummy-coding a continuous control yields one level per
                # subject, which makes every residual zero and silently
                # destroys the test. Split by dtype.
                _cat = [c2 for c2 in ctrl
                        if not pd.api.types.is_numeric_dtype(sub[c2])]
                _num = [c2 for c2 in ctrl
                        if pd.api.types.is_numeric_dtype(sub[c2])]
                _parts = []
                if _cat:
                    _parts.append(pd.get_dummies(sub[_cat].astype(str),
                                                 drop_first=True).astype(float))
                if _num:
                    _parts.append(sub[_num].astype(float))
                dm = (pd.concat(_parts, axis=1).values if _parts
                      else np.empty((len(sub), 0)))
                X = np.column_stack([np.ones(len(sub)), dm]) if dm.size \
                    else np.ones((len(sub), 1))
                def _resid(y):
                    b, *_ = np.linalg.lstsq(X, y, rcond=None)
                    return y - X @ b
                rho_p, p_p = stats.spearmanr(_resid(sub[c].values.astype(float)),
                                             _resid(sub["kendall_tau_mean"].values))
                raw = next((r for r in rows if r["metric"] == c), None)
                delta = ""
                if raw:
                    delta = f"  (uncontrolled was {raw['spearman_rho']:+.4f})"
                print(f"  {c}: partial rho={rho_p:+.4f}  p={p_p:.4f}{delta}")
                if raw:
                    raw["partial_rho"] = rho_p
                    raw["partial_p"] = p_p

    if rows:
        res = pd.DataFrame(rows)
        res.to_csv(os.path.join(args.out, "quality_vs_tau_correlations.csv"),
                   index=False)

        n_tests = len(res)
        bonf = 0.05 / n_tests
        sig = res[res.spearman_p < bonf]
        print()
        print(f"  Bonferroni threshold for {n_tests} tests: p < {bonf:.4f}")
        if len(sig) > 0:
            print("  *** SIGNIFICANT relationship(s) found: ***")
            for _, r in sig.iterrows():
                direction = "worse quality -> LOWER tau" if r.spearman_rho < 0 \
                    else "worse quality -> HIGHER tau (check metric sign!)"
                print(f"    {r['metric']}: rho={r.spearman_rho:+.4f} — {direction}")
            print()
            print("  *** CHECK THE SIGN BEFORE CONCLUDING ANYTHING. ***")
            print("  A POSITIVE rho with a motion metric means MORE motion ->")
            print("  HIGHER tau, which is the OPPOSITE of a data-quality")
            print("  story. It indicates tau is measuring REPRODUCIBILITY,")
            print("  not VALIDITY: a large, structured, temporally persistent")
            print("  artifact dominates the EC estimate and makes driver-node")
            print("  rankings reproducible -- reproducibly recovering the")
            print("  artifact. High tau can mean 'a consistent artifact is in")
            print("  charge', not 'this estimate is trustworthy'.")
            print("  Only a NEGATIVE rho with a motion/noise metric supports")
            print("  the straightforward data-quality explanation.")
        else:
            print("  No relationship survives correction.")
            print("  -> EVIDENCE AGAINST data quality as the driver, but read")
            print("     the two caveats in this script's docstring first:")
            print("     restriction of range (all subjects passed QC) and")
            print("     floor effect (tau already near its floor). Check the")
            print("     metric ranges printed above before concluding.")

    print(f"\nSaved to {args.out}/")


if __name__ == "__main__":
    main()
