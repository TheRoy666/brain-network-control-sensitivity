# 01 · Operator divergence

Compares minimum control energy under the functional-connectivity operator and under the directed effective-connectivity estimate, and asks how the gap depends on recording length and acquisition.

| order | script | does | output in `results/` |
|---|---|---|---|
| 1 | `run_hcp_aud_batch.py`, `run_adni_nctn_batch.py`, `run_unam_nctn_batch.py` | per-cohort energies at native length, FC/EC ratio, finite- versus infinite-horizon energy, cohort summaries | `05_clinical/` (the same runs feed the clinical contrasts) |
| 2 | `operator_divergence_both_directions.py` | the ratio for every subject in both transition directions, at spectral radii 0.80, 0.90 and 0.95, and for symmetrized EC with and without renormalization; one script for all cohorts | `01_operator_divergence/directions*.csv` |
| 3 | `build_truncation_ladder.py` | truncated and subsampled HCP timeseries | — |
| 4 | `operator_divergence_vs_length.py` | the ratio as HCP recordings shrink from 4800 to 600 timepoints, with states recomputed or held fixed, and at ADNI's sampling rate; ADNI at its own lengths | `01_operator_divergence/operator_divergence_*.csv` |
| 5 | `compare_frozen_vs_published.py` | summary of the held-fixed run against the recomputed one | printed |
| 6 | `manjunatha_hub_asymmetry_check.py` | rank correlation of each region's incoming and outgoing EC strength, per subject (eight subjects per cohort) | `01_operator_divergence/in_out_strength_*.csv` |

Shown in Figure 1 and Supplementary Table S1.

`reported_values.py` recomputes every number the paper reports for this stage from `results/` and prints it beside the printed value (`make values` runs all stages).
