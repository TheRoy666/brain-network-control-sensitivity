# NeuroSim — Phase 0 Analysis Plan (Paths A & B)

Pre-specified before any sweep/bootstrap code is run, per Phase 0 checklist.
Purpose: lock the load-bearing design decisions so results can't be quietly
reframed after the fact.

---

## Path A — Validity regime of linear NCT

### Claim under test
"Linear control-energy predictions are trustworthy in regime X and misleading
in regime Y, where X/Y are defined by [operating-point stability margin,
spectral radius, horizon T, transition distance, and A's directionality]."

### A1 — Primary design (stable fixed-point regime)
1. Run `WilsonCowanNetwork` in a parameter regime with a stable fixed point
   x* = (E*, I*) (not `LIMIT_CYCLE_PARAMS`).
2. Solve for x* via root-finding on the WC ODE.
3. Derive A as the **exact Jacobian of the WC ODE at x***, discretized to
   match the TR. This A is *not* independently normalized — it is the
   linearization of the specific nonlinear system being tested against.
   `target_rho` becomes a diagnostic (distance from instability), not a
   free rescaling knob.
4. Derive the full optimal control trajectory u*[0..T-1] (extend
   `minimum_energy`, which currently returns only the first-step input):
   u*[k] = Bᵀ(Aᵀ)^(T-1-k) W_T⁻¹ δ
5. Zero-order-hold u*[k] over each TR interval, inject via a new control
   channel into the WC ODE (`_ode_network` currently has no input term —
   needs adding, analogous to single-node `P_ext`).
6. Integrate the driven nonlinear system. Compare:
   - realized end state vs. target x_T (reachability error)
   - realized energy (∫u²dt) vs. linear-predicted energy E*

### A2 — Secondary / stretch (limit-cycle regime)
Repeat using `LIMIT_CYCLE_PARAMS`, with target redefined as a phase point on
the limit cycle. Expect largest nonlinear error here — useful for discussion,
not the core claim (energy definition is murkier against a moving target).

### Sweep axes
- Distance of operating point from bifurcation
- Spectral radius of derived A (`target_rho` as diagnostic)
- Horizon T
- Transition distance ‖x_T − x0‖
- Operator choice: symmetric-A (FC/masked-SC) vs. directed-EC, applied to
  real HCP/ADNI cohorts (connects sim to real-data conclusions)

### Falsification
If linear error is flat across all axes (no regime dependence), or if
symmetric-vs-directed A never changes a conclusion on real data, Path A
degrades to a null methods note — still reportable, not the spine.

---

## Path B — Uncertainty propagation into control targets

### Claim under test
"Identified optimal control targets are [stable / unstable] under EC
estimation noise, with stability quantified per subject."

### Resampling design
- **Data:** ADNI BOLD, T ≈ 197–200 TRs (confirmed from
  `adni_results_schaefer400_public.csv`)
- **Method:** moving-block bootstrap (naive timepoint bootstrap invalid —
  BOLD is autocorrelated)
- **Block length:** 15 TRs (≈ √T rule of thumb)
- **B (resamples):** start at 200/subject; plot rank-stability metric vs. B
  to check convergence before deciding whether to scale to 500–1000
- **Compute check:** benchmark EC estimation × B on 2–3 subjects before
  launching the full batch; schedule alongside UNAM R740 jobs, not
  competing with them

### Metrics
- Per-subject distribution over control energy (CI from bootstrap)
- Driver-node rank-stability: top-k Jaccard overlap AND Kendall's τ across
  resamples (report both — they answer different questions)

### Falsification
If rankings are stable (high τ, high top-k overlap) — that's a real,
reportable, reassuring result, not a null. Only a genuinely inconclusive
or degenerate bootstrap (e.g., EC estimation fails to converge under
resampling) invalidates the path.

---

## Cross-cutting engineering tasks unlocked by this plan
- [ ] Add control-input channel to `WilsonCowanNetwork._ode_network`
- [ ] Extend `minimum_energy` (or add new function) to return full u*[0..T-1]
      trajectory, not just first-step input
- [ ] Add WC fixed-point solver + analytic Jacobian function
- [ ] Block-bootstrap wrapper around existing EC estimators in `connectivity.py`
- [ ] Rank-stability metric functions (Jaccard top-k, Kendall's τ)

---

## Addendum — Path C (UNAM-TLE) gating and QC findings

**Gate result:** `participants.tsv` has clinician-assigned `TLEside`
(Left/Right/Bi/Unknown) but **no Engel class, ILAE score, or surgical
outcome**. Path C is therefore lateralization-prediction only, not
outcome-prediction. State this explicitly in any writeup.

**QC-pass cohort (post functional-MRIQC screening):** 62 subjects — 34
healthy, 28 patients (17 Left-TLE, 11 Right-TLE). Clean binary split, no
Bi-TLE/Unknown-TLE survived QC.

**Verified:** all BOLD-based pass/fail/flagged decisions cross-checked
exactly against raw MRIQC output (fd_mean, fd_perc, tsnr) — no
discrepancies.

**Confound identified and resolved — acquisition batch effect:**
- 4 subjects (one further subject [excluded, motion], UNAM_042, UNAM_873,
  UNAM_741 — all Left-TLE) have 130-volume BOLD scans vs. the cohort
  standard 200 (260s vs. 400s acquisition).
- Sidecar check confirms: same scanner (Philips Achieva 3T), same
  protocol name (`Protocolo_Epilepsia`), but a distinct slice-timing
  scheme — i.e. an early/short protocol revision, not a scanner swap.
- Checked against 4 comparison subjects (2 Left-TLE: UNAM_034, UNAM_849; 2
  Right-TLE: UNAM_581, UNAM_597) — all show the standard/later protocol
  signature. **Confirmed contained to a small subset, not a scanner-level
  confound running through the whole TLEside comparison.**
- **Action:** add `early_protocol_batch` (True for UNAM_042/UNAM_873/UNAM_741) as
  a covariate in the Path C lateralization test, or run as a pre-specified
  sensitivity analysis excluding these 3 subjects. Verify
  `extract_timeseries_surface.py` correctly reads per-subject slice-timing
  from the sidecar rather than assuming one fixed scheme.

**Structural (T1w) QC — secondary, lower priority:** UNAM_523 (lowest
CNR/SNR in cohort) and UNAM_096 (largest qi_2 artifact-index outlier,
also worst fd_perc on BOLD) worth a visual check before FastSurfer,
given UNAM_096 is one of only 11 Right-TLE subjects and losing it would
shrink an already-thin minority class.

## Sign-off
This plan is written before sweep/bootstrap results exist. Any deviation
from it during analysis should be logged with a reason, not silently folded
into the writeup.
