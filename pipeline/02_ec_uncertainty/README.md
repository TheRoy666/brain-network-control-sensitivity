# 02 · Estimation uncertainty and driver rankings

Propagates the uncertainty of the effective-connectivity estimate into the ranking of regions by average controllability, under the driver definition (Gu et al., 2015) and the receiver definition (`docs/DEFINITIONS.md`).

| order | script | does | output in `results/02_ec_uncertainty/` |
|---|---|---|---|
| 1 | `run_ec_bootstrap_batch.py` | 200 moving-block bootstrap resamples per subject, EC re-estimated on each; ensembles saved | ensembles stay with the data |
| 2 | `compare_ctrl_definitions.py` | rescores the saved ensembles under both definitions: τ, top-5 Jaccard, and the agreement between definitions | `ctrl_definition_comparison_*.csv` |
| 3 | `twin_clustered_bootstrap.py` | HCP intervals with twin pairs as the resampling unit, for each ranking | `twin_clustered_ci_HCP_driver.csv`, `twin_clustered_ci_HCP_receiver.csv` |
| 4 | `graphnet_sensitivity.py`, then `compare_sweep_rerun.py` | nine settings of the ridge and structural-prior penalties, both definitions | `graphnet_sensitivity_*.csv` |
| 5 | `bootstrap_block_length.py` | block lengths 5 to 100, both definitions | `block_length_*.csv` |
| 6 | `sensitivity_vs_ntr_dual.py`, then `compare_ntr_rerun.py` | HCP truncated to ADNI's timepoints per region, at the published setting and across the penalty grid | `ntr_*.csv` |
| 7 | `hcp_motion_vs_tau.py`, `adni_quality_vs_taustability.py` | rank stability against head motion and data-quality measures | printed; Figure S1 is drawn from the files above |
| 8 | `frassle_stephan_cross_check.py` | connection strength against its variability across resamples | `frassle_stephan_*.csv` |
| 9 | `nctpy_definition_check.py` | nctpy 1.1.0 and 1.2.0 `ave_control` on planted senders and receivers, against both definitions (runs without data) | printed |
| 10 | `brain_maps.py` | from the saved ensembles: how often each region is a top-five driver across the 200 resamples, for the subject nearest the cohort median driver τ (checked against `ctrl_definition_comparison_*.csv`), and each region's out-strength minus in-strength averaged over subjects; `export` writes the deposited tables, without per-subject rows | `brain_maps/` |
| — | `compute_and_save_ec_matrices.py` | point-estimate EC matrices for inspection | — |

Shown in Figures 1f, 2, 3 and S1 and Supplementary Tables S2 and S3.

`reported_values.py` recomputes every number the paper reports for this stage from `results/` and prints it beside the printed value (`make values` runs all stages).

## `estimator_convergence/`

The checks behind the effective-connectivity estimator settings in Methods 4.5 (convergence test after the update, up to 3000 iterations). The GraphNet loop originally tested convergence before assigning the update and ran 500 iterations; these scripts found that, traced which cohorts it affected, and confirmed the corrected estimator before every analysis was rerun. The README in the folder gives the sequence and what each script established. Subject lists are read from `HCP_SUBJECTS`, `ADNI_SUBJECTS`, `UNAM_SUBJECTS` (space-separated) or `HCP_SUBJECT`, `ADNI_SUBJECT`, `UNAM_SUBJECT`.

