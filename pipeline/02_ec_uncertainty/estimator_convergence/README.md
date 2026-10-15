# GraphNet EC convergence fix — verification scripts

These scripts document the discovery, diagnosis, and cohort-by-cohort
verification of a correctness bug in `nctsens.connectivity`'s (NeuroSim: `neurosim/connectivity.py`)
`graphnet_effective_connectivity`. They are kept because the *verification*
matters as much as the fix: no result was trusted again until it had been
checked directly against its pre-fix counterpart.

## The bug

The FISTA loop's convergence check compared `A_new` against `A` and, if
within tolerance, `break`'d **before** the line assigning `A = A_new`.
Whenever the loop converged — at any iteration — the function returned the
value from one step *before* the update that satisfied convergence,
discarding the only computation that had incorporated `lambda_graph`.

Consequences depended entirely on each cohort's data dimensionality:

| cohort | behaviour | impact |
|---|---|---|
| HCP-AUD (N=410, T=4800) | converged at iteration 0 | returned the ridge-only starting point |
| UNAM-TLE (N=68, T≈200) | converged at iteration 0 | same |
| ADNI (N=400/450, T=197) | never converged within 500 iterations | never hit the buggy branch, but the estimate was genuinely unconverged |

## The fix

Two changes, both in `graphnet_effective_connectivity`:

1. Reordered the loop so `A = A_new` happens **before** the tolerance check.
2. Raised `max_iter` from 500 to 3000 — ADNI's dimensionality genuinely
   needed the larger budget, confirmed by convergence-trajectory tracing
   showing clean monotonic decay rather than a stall.

## Outcome

Every cohort was rerun and every downstream metric compared old-vs-new.
**No reported finding changed.** HCP's group medians matched to 6 decimal
places; UNAM's node-asymmetry values — the input to the pre-registered
lateralization test — matched to a maximum absolute difference of 2.6e-06
across 62 subjects × 34 region pairs; ADNI's p-values and effect sizes were
unchanged.

That conclusion is *earned by direct verification*, not assumed.

## Scripts

| script | purpose |
|---|---|
| `check_all_cohorts_complete.py` | Per-cohort iteration-trace: at which iteration did each subject's loop break? Established the HCP/UNAM-vs-ADNI split. |
| `check_adni_convergence.py` | ADNI-specific: is the estimate still moving at iteration 500? |
| `check_adni_more_iters.py` | Extends ADNI to 3000 iterations to confirm genuine convergence rather than a stall. |
| `check_pathb_bootstrap_scope.py` | Does the iteration-0 pattern also hold on bootstrap-*resampled* data? (60 resamples, HCP + UNAM.) |
| `check_hcp_unam_convergence.py` | Convergence-trajectory and timing for HCP/UNAM under the corrected code. |
| `unam_full_cohort_drift.py` | Full 62-subject UNAM drift check across a 5000-iteration ladder. |
| `confirm_raw_rho_mechanism.py` | Closes a long-open puzzle: UNAM's `raw_rho` pinning at ~0.999995 is Ridge EC's characteristic spectral radius at N=68, not an artifact. |
| `diagnose_lambda_graph_sweep.py` | The diagnostic that surfaced the bug — `raw_rho` was bit-for-bit identical across the entire `lambda_graph` sweep. |

## A note on `diagnose_lambda_graph_sweep.py`

Its original version swept `lambda_graph` while naming the variable
`LAMBDA_RIDGE_VALUES` and labelling the output column `lambda_ridge`. The
*computation* was correct; only the labelling was wrong. Corrected here so
the saved results cannot be misread as testing the wrong parameter.
