#!/usr/bin/env python3
"""
Path 1, Item 3 -- Step 1 of the scan-duration threshold derivation.

Builds a LADDER of truncated HCP timeseries sets, one directory per
target T, so that run_path_b_bootstrap.py can be pointed at each in turn
to produce Kendall's tau as a function of scan length.

WHY THIS EXISTS
---------------
Path B currently has two points: HCP (T=4800, tau~0.87) and ADNI
(T~197-200, tau~0.07). Two points establish a contrast but cannot
produce a threshold. Truncating the SAME HCP subjects to a ladder of
T values isolates scan length as the manipulated variable -- same
subjects, same scanner, same preprocessing, same atlas -- and yields a
real tau(T) curve.

This also, as a free byproduct, directly tests the previously-untested
hypothesis that ADNI's tighter CV-reliability relationship is explained
by its shorter scans (see PROJECT_MAP.md, "Explicitly Out of Scope").

IMPORTANT SCIENTIFIC CAVEAT -- READ BEFORE INTERPRETING
-------------------------------------------------------
Truncating by TIMEPOINTS is not the same as matching DURATION, because
HCP and ADNI differ in TR. HCP-YA rfMRI is TR=0.72s; ADNI-3 rfMRI is
typically TR~3s. So HCP truncated to T=200 is ~2.4 minutes of data,
while ADNI at T=197 is ~10 minutes. Matched timepoints therefore means
matched STATISTICAL sample size, not matched acquisition time.

Three distinct axes are confounded in the raw HCP-vs-ADNI contrast:
  (1) number of timepoints  (statistical power for the regression)
  (2) acquisition duration  (how much slow physiological structure is sampled)
  (3) TR                    (sampling rate / autocorrelation structure)

This ladder cleanly isolates axis (1) within HCP. It does NOT resolve
(2) or (3). Report the threshold primarily in timepoints, secondarily
in minutes with the TR stated explicitly, and state this limitation
rather than letting a reader assume duration was matched.

TRUNCATION IS DETERMINISTIC: takes the FIRST T timepoints, not a random
window, matching this project's standing convention (the bootstrap
itself already uses a fixed seed).

USAGE
-----
    python3 build_truncation_ladder.py \
        --ts-dir /path/to/HCP_Native_Timeseries \
        --out-base /path/to/HCP_truncation_ladder \
        --t-values 200,400,800,1600,2400,3600 \
        --n-subjects 40

Produces:
    <out-base>/T00200/<sub>_truncated_timeseries.npy
    <out-base>/T00400/...
    ...

Then run run_path_b_bootstrap.py once per T directory (commands are
printed at the end of this script's output).
"""
import argparse
import os

import numpy as np


def detect_and_truncate(X, target_T, n_regions, stride=1):
    """Extract target_T timepoints along the TIME axis, optionally strided.

    stride=1 (default): the first target_T contiguous timepoints. This
    manipulates SAMPLE COUNT while holding sampling rate fixed.

    stride=k: every k-th timepoint, so target_T samples span k*target_T
    of the original series. This manipulates EFFECTIVE TR -- and hence
    temporal autocorrelation structure -- while holding sample count
    fixed. It is the complement of the stride=1 manipulation, and the
    two together separate two axes that are otherwise confounded in any
    cross-cohort comparison.

    Why this matters here: truncating HCP to T=200 at stride=1 did NOT
    reproduce ADNI's low rank-stability (HCP 0.858 vs ADNI 0.07 at the
    same T, same block length). Sample count is therefore ruled out as
    the driver. HCP is TR=0.72s and ADNI ~3s, so stride=4 gives an
    effective TR of 2.88s at matched T -- isolating sampling rate as the
    next candidate.

    Orientation is resolved by matching against the KNOWN region count,
    exactly as run_path_b_bootstrap.py resolves it against SC.shape[0] --
    NOT by a "more timepoints than regions" size heuristic.

    That distinction is not pedantic. The size heuristic fails precisely
    when a scan is shorter than the region count, which is exactly the
    too-short case this function must catch -- and is also the normal
    state of affairs for ADNI (T=197 < N=400). A size heuristic would
    silently mistake a 150-timepoint scan for a 150-region one and
    truncate the region axis, corrupting the data without raising
    anything. Caught in testing; fixed here.

    Returns (extracted_array, status_string), or (None, reason) if the
    subject genuinely cannot supply target_T timepoints at this stride.
    """
    if X.shape[0] == n_regions and X.shape[1] == n_regions:
        return None, (f"ambiguous: both axes equal n_regions={n_regions}; "
                      f"cannot resolve time axis")

    need = target_T * stride  # samples consumed from the original series

    if X.shape[1] == n_regions:
        # stored (T, N) -- time is axis 0
        T_avail = X.shape[0]
        if T_avail < need:
            return None, (f"too short: {T_avail} timepoints < {need} "
                          f"required for T={target_T} at stride={stride}")
        out = X[:need:stride, :]
        return out, f"(T,N) {X.shape} -> {out.shape} [stride={stride}]"

    if X.shape[0] == n_regions:
        # stored (N, T) -- time is axis 1
        T_avail = X.shape[1]
        if T_avail < need:
            return None, (f"too short: {T_avail} timepoints < {need} "
                          f"required for T={target_T} at stride={stride}")
        out = X[:, :need:stride]
        return out, f"(N,T) {X.shape} -> {out.shape} [stride={stride}]"

    return None, (f"neither axis matches n_regions={n_regions}: shape={X.shape}. "
                  f"Wrong --n-regions, or wrong atlas for this file.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts-dir", required=True,
                    help="Directory of full-length HCP timeseries .npy files")
    ap.add_argument("--out-base", required=True,
                    help="Base directory; one subdirectory per T value is created")
    ap.add_argument("--ts-suffix", default="_native_timeseries.npy")
    ap.add_argument("--out-suffix", default="_truncated_timeseries.npy")
    ap.add_argument("--t-values", default="200,400,800,1600,2400,3600",
                    help="Comma-separated target T values. The full-length "
                         "run (T=4800) already exists and is NOT regenerated.")
    ap.add_argument("--n-subjects", type=int, default=0,
                    help="Use only the first N subjects (0 = all). A subsample "
                         "is scientifically fine here: the tau(T) curve is a "
                         "within-subject manipulation, so precision comes from "
                         "the T ladder, not from cohort size.")
    ap.add_argument("--sc-dir", default=None,
                    help="Optional. If given, only subjects that ALSO have an "
                         "SC file are included -- avoids building truncated "
                         "timeseries for subjects the bootstrap will skip.")
    ap.add_argument("--sc-suffix", default="_SC_SIFT2_410.csv")
    ap.add_argument("--stride", type=int, default=1,
                    help="Take every k-th timepoint. stride=1 (default) "
                         "manipulates sample COUNT at fixed sampling rate. "
                         "stride>1 manipulates effective TR at fixed sample "
                         "count -- e.g. stride=4 on HCP (TR=0.72s) gives an "
                         "effective TR of 2.88s, matching ADNI's ~3s. Output "
                         "dirs are tagged so strided and non-strided runs "
                         "cannot collide.")
    ap.add_argument("--tr", type=float, default=0.72,
                    help="Original TR in seconds, used only to report the "
                         "effective TR under the chosen stride.")
    ap.add_argument("--n-regions", type=int, default=410,
                    help="Number of atlas regions. Used to resolve array "
                         "orientation unambiguously (see detect_and_truncate). "
                         "HCP here = 410. Must match the actual atlas.")
    args = ap.parse_args()

    t_values = sorted(int(t) for t in args.t_values.split(","))

    files = sorted(f for f in os.listdir(args.ts_dir) if f.endswith(args.ts_suffix))
    subjects = [f[:-len(args.ts_suffix)] for f in files]

    if args.sc_dir:
        sc_present = {f[:-len(args.sc_suffix)]
                      for f in os.listdir(args.sc_dir) if f.endswith(args.sc_suffix)}
        before = len(subjects)
        subjects = [s for s in subjects if s in sc_present]
        if before != len(subjects):
            print(f"Filtered to subjects with SC present: {before} -> {len(subjects)}")

    if args.n_subjects > 0:
        subjects = subjects[:args.n_subjects]

    print(f"Subjects: {len(subjects)}")
    print(f"T ladder: {t_values}")
    print(f"n_regions (for orientation resolution): {args.n_regions}")
    if args.stride > 1:
        print(f"STRIDE={args.stride}: effective TR = {args.tr*args.stride:.2f}s "
              f"(original {args.tr}s). Sample count held fixed; sampling rate "
              f"is the manipulated variable.")
    print()

    tag = "" if args.stride == 1 else f"_s{args.stride}"
    for target_T in t_values:
        out_dir = os.path.join(args.out_base, f"T{target_T:05d}{tag}")
        os.makedirs(out_dir, exist_ok=True)
        n_ok, n_skip = 0, 0
        for sub in subjects:
            src = os.path.join(args.ts_dir, sub + args.ts_suffix)
            try:
                X = np.load(src)
                X_trunc, status = detect_and_truncate(X, target_T, args.n_regions, args.stride)
                if X_trunc is None:
                    print(f"  [T={target_T}] SKIP {sub}: {status}")
                    n_skip += 1
                    continue
                np.save(os.path.join(out_dir, sub + args.out_suffix), X_trunc)
                n_ok += 1
            except Exception as e:
                print(f"  [T={target_T}] FAILED {sub}: {e}")
                n_skip += 1
        print(f"T={target_T:5d}: {n_ok} written, {n_skip} skipped -> {out_dir}")

    print()
    print("=" * 70)
    print("NEXT: run the bootstrap once per T directory.")
    print("Note --block-length auto is REQUIRED, not optional: it applies the")
    print("sqrt(T) rule per-directory, so each T gets the block length")
    print("appropriate to its own length (this is what makes T=200 here")
    print("methodologically comparable to ADNI's T~197 run, which used 14).")
    print("=" * 70)
    for target_T in t_values:
        out_dir = os.path.join(args.out_base, f"T{target_T:05d}{tag}")
        print(f"""
python3 run_path_b_bootstrap.py \\
    --ts-dir {out_dir} \\
    --sc-dir {args.sc_dir or '<SC_DIR>'} \\
    --ts-suffix {args.out_suffix} \\
    --sc-suffix {args.sc_suffix} \\
    --out-dir {out_dir}_bootstrap \\
    --n-boot 200 --block-length auto \\
    --n-workers 40 \\
    --timeout-per-subject 21600""")


if __name__ == "__main__":
    main()
