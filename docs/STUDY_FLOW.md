# Study flow: from question to figure

The paper asks three questions in sequence, then reports exploratory clinical contrasts. Each row below links a Results section to the scripts that produced it, the deposited files that hold the output and the display that shows it. Paths are relative to the repository root; `P/` is `pipeline/`, `R/` is `results/`.

```
00 data ──► 01 operator ──► 02 uncertainty ──► 03 linearization ──► 04 nonlinear control
                 │                                                        
                 └──────────────────────────────► 05 clinical (exploratory)
```

Each stage only needs the data and the stage outputs to its left.

## 00 · Data

| step | scripts | output |
|---|---|---|
| Anatomical reconstruction (FastSurfer), functional (fMRIPrep) and diffusion (QSIPrep) preprocessing for ADNI | `P/00_preprocessing/run_adni_pipeline.sh` | preprocessed images (not deposited) |
| Same for the epilepsy cohort, with surface-based functional processing and MRIQC | `run_unam_pipeline.sh`, `run_unam_mriqc.sh`, `extract_timeseries_surface.py` | regional timeseries (not deposited) |
| Regional timeseries (ADNI) and post-processing QC | `extract_timeseries.py`, `qc_postproc.py` | `R/cohort/adni_qc.csv` |
| Structural connectomes: ADNI (MRtrix3, SIFT2) and HCP | `build_connectomes.sh`, `qc_connectomes.py`, `SCMatrixExtractor.py` | connectomes (not deposited) |
| HCP head motion, mean framewise displacement per subject | — | `R/cohort/hcp_head_motion.csv` |

Cohorts and acquisition: **Table 1** (`tables/check_tables.py`); regions against timepoints per arm, and the three parcellations: **Figure S1**. Cohort numbers in the text: `P/00_preprocessing/reported_values.py`.

## 01 · Operator divergence

*Results: "Recording length and acquisition characteristics influence operator divergence"*

| claim | scripts (`P/01_operator_divergence/`) | results (`R/01_operator_divergence/`) | display |
|---|---|---|---|
| FC- and EC-based energies differ by median factors of 3 to 65 across cohorts; per-cohort native-length energies | `run_hcp_aud_batch.py`, `run_adni_nctn_batch.py`, `run_unam_nctn_batch.py` | `R/05_clinical/*teleportation*`, `*results*` | text |
| Both transition directions, spectral radii 0.80 and 0.95, and symmetrized EC with and without renormalization, recomputed with one script for all cohorts | `operator_divergence_both_directions.py` | `directions_*.csv`, `directions_ext_*.csv`, `directions_ext2_*.csv` | **Figure 1b–c**; Supplementary S8 |
| Truncating HCP from 4800 to 600 timepoints raises the ratio from 3.2 to 15.0, and to 9.3 with the states held fixed; subsampling to ADNI's sampling rate | `build_truncation_ladder.py`, `operator_divergence_vs_length.py`, `compare_frozen_vs_published.py` | `operator_divergence_vs_length_*.csv`, `operator_divergence_frozen_states_HCP.csv`, `operator_divergence_stride_HCP.csv` | **Figure 1d**, **Table S1** |
| Within ADNI, the ratio by scanner manufacturer | as above | `directions_ADNI-400.csv` with `R/05_clinical/adni_results_400.csv` and `R/cohort/adni_qc.csv` | **Figure 1e** |
| Finite-horizon energy exceeds the infinite-horizon estimate at short horizons | `run_hcp_aud_batch.py`, `run_adni_nctn_batch.py` | `R/05_clinical/adni_finite_vs_infinite_*.csv` (HCP not deposited) | text |
| Incoming and outgoing strength inversely related in HCP and epilepsy, not in ADNI (eight subjects per cohort; estimator before its convergence fix) | `manjunatha_hub_asymmetry_check.py` | `in_out_strength_*.csv` | text |
| On a simulated feedforward chain, whether a symmetric operator raises or lowers minimum energy depends on how it is built and scaled | `synthetic_chain_operator_check.py` | printed | text |

## 02 · Estimation uncertainty and driver rankings

*Results: "Directed controllability definitions can reverse the identity of candidate drivers" through "Bootstrap block length and ranking definition affect stability differently"*

| claim | scripts (`P/02_ec_uncertainty/`) | results (`R/02_ec_uncertainty/`) | display |
|---|---|---|---|
| Bootstrap EC ensembles: 200 moving-block resamples per subject, saved for rescoring | `run_ec_bootstrap_batch.py`, `compute_and_save_ec_matrices.py` | ensembles stay with the data | — |
| Receiver and driver rankings are largely opposite for directed operators | `compare_ctrl_definitions.py` | `ctrl_definition_comparison_*.csv` (`rho_package_vs_gu`) | **Figure 2e** |
| nctpy 1.1.0 behaves like the receiver ranking on directed operators; 1.2.0 like the driver ranking | `nctpy_definition_check.py` | printed | text, Methods |
| Driver rankings are stable in HCP (τ = 0.88; 3.3 of 5 top regions shared) and not in ADNI (τ = 0.33–0.37; 0.4 of 5), with no overlap between cohorts | `compare_ctrl_definitions.py` | `ctrl_definition_comparison_*.csv` (`tau_gu`, `jaccard_gu`) | **Figure 2c–d** |
| HCP interval allowing for twin clustering | `twin_clustered_bootstrap.py` | `twin_clustered_ci_HCP_driver.csv` (and `_receiver.csv`) | text |
| Truncating HCP to ADNI's timepoints per region reproduces the instability of its top regions | `sensitivity_vs_ntr_dual.py`, `compare_ntr_rerun.py` | `ntr_*.csv` | **Figure 2f**; S8 |
| Where the drivers are: how often each region is a top-five driver for the median-τ subject of each arm; where EC sends and receives: out-strength minus in-strength averaged over subjects | `brain_maps.py` | `brain_maps/*.csv` | **Figure 2g–h**, **Figure 1f** |
| Penalty weights barely move HCP and strongly move ADNI | `graphnet_sensitivity.py`, `compare_sweep_rerun.py` | `graphnet_sensitivity_*.csv` | **Figure 3b–c**, **Table S2** |
| Block length affects the two rankings differently | `bootstrap_block_length.py` | `block_length_*.csv` | **Figure 3d**, **Table S3** |
| Receiver-ranking stability rises with head motion in HCP and ADNI-400; driver-ranking stability does not | `hcp_motion_vs_tau.py`, `adni_quality_vs_taustability.py` | `ctrl_definition_comparison_*.csv` with `R/cohort/*` | **Figure S2** (Supplementary S5) |
| Coefficient of variation falls with connection strength, as in Frässle and Stephan (2022) | `frassle_stephan_cross_check.py` | `frassle_stephan_*.csv` | Supplementary Methods |
| Estimator settings: convergence tested after the update, up to 3000 iterations | `estimator_convergence/` | printed | Methods |

## 03 · Validity of the linearization

*Results: "Linearized control is accurate over much of the tested transition range…", "The transition is not explained by numerical error or non-normality", "The cohort difference in the transition persists under a degree-preserving null"*

| claim | scripts (`P/03_linearization/`) | results (`R/03_linearization/`) | display |
|---|---|---|---|
| Stability boundary per cohort and coupling grid | `real_coupling_boundary_recheck.py`, `check_subject_coupling.py`, `fixed_point_scale_check.py` | `fixed_point_scale_check.csv` | Supplementary Methods |
| Linearized control has low median endpoint error (3.4%); the interior error minimum disappears above a critical displacement that differs by cohort | `wc_linear_validity_sweep_real_coupling.py` (with `--target-scales` for the fine grid), `localize_critical_displacement.py` | `fine_sweep/wc_sweep_finets_*.csv.gz`; first sweep in `coarse_sweep/` | **Figure 4b–d** |
| Numerical error is a negligible share of the total | `linearization_vs_numerical_error.py` | `linearization_vs_numerical_error_*.csv` | text; S8 |
| No 10% breach anywhere; a 5% tolerance is crossed between 0.055 and 0.078, and the cohorts are 0.9% apart at mean coupling | `error_tolerance_threshold.py` | `fine_sweep/wc_sweep_finets_*.csv.gz` | S8 |
| Non-normality does not explain the transition (measured at the highest coupling) | `measure_non_normality.py` | `non_normality_*.csv` | text; S8 |
| The cohort difference persists under degree-preserving rewiring | `null_model_rewiring.py`, `wc_linear_validity_sweep_null_model.py`, `network_topology_characterization.py` | `fine_sweep/wc_null_finets_*`, `coarse_sweep/wc_null_*`, `network_topology_*.csv` | **Figure 4e**; Supplementary S2 |
| No combining form of energy and amplification beats a majority-class baseline | `mechanism_functional_form_search.py`, `mechanism_search_v2.py` | printed | Supplementary S2 |
| The transition does not appear in FitzHugh–Nagumo | `fhn_validity_sweep_toy.py`, `fhn_kappa_sensitivity_check.py` | printed | Discussion |
| Discretization step | `wc_timestep_discretization_check.py` | printed | Methods |
| Seeding: two runs agree statistically, not bit for bit | `localize_critical_displacement.py` on `coarse_sweep/` and `fine_sweep/` | — | Supplementary S6; `docs/REPRODUCIBILITY_NOTE.md` |

## 04 · Nonlinear optimal control

*Results: "Nonlinear optimization achieves lower endpoint error within the linear controller's energy budget"*

| claim | scripts (`P/04_nonlinear_control/`) | results (`R/04_nonlinear_control/`) | display |
|---|---|---|---|
| Adjoint gradient verified against finite differences in every model | `adjoint_gradient.py`, `adjoint_gradient_verification.py`, `adjoint_gradient_wc_verification.py` | printed by the scripts | Methods |
| Models: FitzHugh–Nagumo (nominal orbit), Stuart–Landau (closed form), Wilson–Cowan oscillatory | `fhn_simulation.py`, `fhn_generate_nominal_orbit.py`, `stuart_landau_model.py`, `wc_oscillatory_linearization_comparison.py` | `fhn_nominal_orbit_data/` (with the code) | — |
| Within the linear controller's energy, local nonlinear optimization reaches lower error in every condition tested | `constrained_dominance.py`, `constrained_dominance_sl.py`, `constrained_dominance_wc.py` | `constrained_dominance_*.csv` | **Figure 5d** |
| Secondary check: ten effort penalties, four initializations | `matched_cost_control_comparison*.py`, `fhn_broader_grid.py`, `wc_a2_broader_grid.py` | `matched_cost_*.csv`, `*_broader_grid.csv` | Supplementary S4, S8 |
| Worked example at the median error ratio | `fhn_example_trajectories.py` | `fhn_example.npz` | **Figure 5b, c, e** |
| Frozen and time-varying linearization disagree in direction between Wilson–Cowan and FitzHugh–Nagumo; Stuart–Landau as a third model, supercritical regimes one system up to rescaling | `fhn_phase_shift_comparison.py`, `sl_a2_grid.py`, `consolidate_sl_a2_replication.py` | printed | Supplementary S4; Discussion |

## 05 · Exploratory clinical contrasts

*Results: "Exploratory clinical contrasts"; Supplementary S1 and S3*

| contrast | scripts | results (`R/05_clinical/`) | display |
|---|---|---|---|
| Alcohol-use severity in HCP twins | `P/01_operator_divergence/run_hcp_aud_batch.py`, `P/05_clinical/run_hcp_aud_batch_real_reward.py`, `compare_real_reward_vs_original.py` | `hcp_aud_group_comparison.csv` (group level only; see DATA_ACCESS) | S1 |
| Cognitively normal versus MCI in ADNI, before and after controls-only site harmonization | `run_adni_nctn_batch.py`, `P/05_clinical/run_adni_harmonize.py` | `adni_*` | S1 |
| Seizure-focus lateralization and energy barrier in epilepsy | `run_unam_nctn_batch.py`, `P/05_clinical/tle_lateralization_test.py`, `energy_barrier_analysis.py`; checks in `lateralization_checks/` | `unam_*` | S1 |
| Minimum detectable effects | two-sample t-test power (statsmodels), in `P/05_clinical/reported_values.py` | — | S3 |

## Checking the numbers

| what | script | inputs |
|---|---|---|
| every number in the text, stage by stage | `P/<stage>/reported_values.py` (`make values`) | `results/` |
| Table 1, Tables S1–S3, cell by cell | `tables/check_tables.py` (`make tables`) | `results/`; Table 1 constants from the source datasets |

Each `reported_values.py` prints the recomputed value beside the printed one, and lists at the end the values that need data not deposited here, with the script that prints them. The figures were drawn from the same files and are not part of this repository.
