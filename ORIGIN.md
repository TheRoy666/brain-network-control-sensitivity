# Origin

This repository is the paper-specific extract of **NeuroSim**, the network-control toolkit built during Google Summer of Code 2026 for the International Neuroinformatics Coordinating Facility (INCF), project 39: <https://github.com/TheRoy666/NeuroSim>.

It keeps only what the paper uses, arranged in the order of the study rather than the order of development. Nothing here depends on the NeuroSim repository at run time: the package is vendored as `nctsens` so that both can be installed side by side.

**Reference point.** File status is given against NeuroSim commit `89face332fc9b3237c2065556e34d79ebf01fa05`, the public `main` branch when this repository was assembled. The analyses for the paper continued after that commit, so some files are later revisions and some do not exist upstream yet.

| status against `89face3` | meaning | files |
|---|---|---|
| identical | the source file matches NeuroSim at that commit (line endings aside) | 53 |
| later revision | the file is in NeuroSim; the version used for the paper is a later revision | 4 |
| not yet upstream | analysis code and plans from the NeuroSim project that are not yet in the public repository | 42 |
| paper only | analysis and checking scripts written for the paper outside the NeuroSim tree | 6 |

**Changes made in this repository**, listed per file below, are limited to: imports renamed from `neurosim` to `nctsens`; usage notes updated to this layout; machine-specific paths replaced by environment variables; subject identifiers removed from code, plans and embedded output; and the driver definition of average controllability added to `nctsens.physics` (the only change in behavior; the original definition remains the default so that earlier outputs reproduce exactly). The files listed under *New in this repository* were written for it.

Hashes are the first 12 hex digits of SHA-256 over the file with LF line endings: *upstream* at the reference commit where the file exists there, *here* in this repository.


## `src`

| file | NeuroSim path | vs `89face3` | upstream | here | changed here |
|---|---|---|---|---|---|
| `nctsens/physics.py` | `neurosim/physics.py` | identical | `8a6def41a657` | `882a4bb879e2` | docstrings name nctsens; `average_controllability` gains a `direction` argument and `driver_average_controllability` is added (driver definition, Gu et al., 2015); default behavior unchanged |
| `nctsens/simulation.py` | `neurosim/simulation.py` | identical | `5b5551ddb96b` | `22b379e50439` | docstrings name nctsens |
| `nctsens/connectivity.py` | `neurosim/connectivity.py` | identical | `2ae501d6f7f3` | `996be7af84d2` | docstrings name nctsens |
| `nctsens/loader.py` | `neurosim/loader.py` | identical | `66504236f12b` | `f1ef0a164788` | docstrings name nctsens |
| `nctsens/harmonize.py` | `neurosim/harmonize.py` | identical | `ec36d3cd0018` | `076d088b5905` | docstrings name nctsens |
| `nctsens/clinical.py` | `neurosim/clinical.py` | identical | `ea40ba7fedcb` | `f21264b8948e` | imports point to nctsens; docstrings name nctsens; passes `direction="receiver"` explicitly; only the node mean is used, which is the same under both definitions |

## `tests`

| file | NeuroSim path | vs `89face3` | upstream | here | changed here |
|---|---|---|---|---|---|
| `test_physics.py` | `tests/test_physics.py` | identical | `914d23e9837b` | `46f2f3408b78` | imports point to nctsens; docstrings name nctsens |
| `test_simulation.py` | `tests/test_simulation.py` | identical | `d2255f1ca3ae` | `021784de0e82` | imports point to nctsens; docstrings name nctsens |
| `test_connectivity.py` | `tests/test_connectivity.py` | identical | `debc4c592c3f` | `f045a2e4e4d8` | imports point to nctsens; docstrings name nctsens |
| `test_clinical.py` | `tests/original_project_tests/test_clinical.py` | identical | `d30e1f5ba34b` | `3249468a2ad3` | imports point to nctsens; docstrings name nctsens |
| `test_harmonize.py` | `tests/original_project_tests/test_harmonize.py` | identical | `a18a44df2ecd` | `0c3ff9d3f678` | imports point to nctsens; docstrings name nctsens |
| `test_loader.py` | `tests/original_project_tests/test_loader.py` | identical | `21f68cd1b5bb` | `fe9f464bb68a` | imports point to nctsens; docstrings name nctsens |

## `pipeline/00_preprocessing`

| file | NeuroSim path | vs `89face3` | upstream | here | changed here |
|---|---|---|---|---|---|
| `run_adni_pipeline.sh` | `scripts/run_adni_pipeline.sh` | identical | `47ca2cce8afa` | `1efd2835387e` | subject list read from $SUBJECT_LIST instead of hard-coded IDs |
| `build_connectomes.sh` | `scripts/build_connectomes.sh` | identical | `95570751499e` | `218202be9edd` | subject list read from $SUBJECT_LIST instead of hard-coded IDs; test subject read from $TEST_SUBJECT |
| `extract_timeseries.py` | `scripts/extract_timeseries.py` | identical | `10602c0e63af` | `10602c0e63af` | unchanged |
| `qc_postproc.py` | `scripts/qc_postproc.py` | identical | `2a33821a61b3` | `2a33821a61b3` | unchanged |
| `qc_connectomes.py` | `scripts/qc_connectomes.py` | identical | `c5db8c438441` | `c5db8c438441` | unchanged |
| `SCMatrixExtractor.py` | `scripts/SCMatrixExtractor.py` | later revision | `6cf45aff16ae` | `9b5075735991` | site paths read from $BNCS_ROOT; example identifier replaced by a placeholder |
| `run_unam_pipeline.sh` | `scripts/UNAM_TLE/run_unam_pipeline.sh` | identical | `7c57a15a8c01` | `7c57a15a8c01` | unchanged |
| `run_unam_mriqc.sh` | `scripts/UNAM_TLE/run_unam_mriqc.sh` | identical | `66966be36f14` | `66966be36f14` | unchanged |
| `extract_timeseries_surface.py` | `scripts/UNAM_TLE/extract_timeseries_surface.py` | identical | `bbfe26a03e0a` | `bbfe26a03e0a` | unchanged |

## `pipeline/01_operator_divergence`

| file | NeuroSim path | vs `89face3` | upstream | here | changed here |
|---|---|---|---|---|---|
| `run_hcp_aud_batch.py` | `scripts/run_hcp_aud_batch.py` | later revision | `14a31f662b75` | `e672a5902856` | imports point to nctsens |
| `run_adni_nctn_batch.py` | `scripts/run_adni_nctn_batch.py` | identical | `eafef1aff1b0` | `f060f73d543f` | imports point to nctsens |
| `run_unam_nctn_batch.py` | `scripts/UNAM_TLE/run_unam_nctn_batch.py` | identical | `ba5c117cabb2` | `10aa912d9f1f` | imports point to nctsens |
| `operator_divergence_vs_length.py` | `scripts/path1_upscale/operator_divergence_vs_length.py` | not yet upstream | — | `ea17dadf6501` | imports point to nctsens; docstrings name nctsens |
| `build_truncation_ladder.py` | `scripts/path1_upscale/build_truncation_ladder.py` | not yet upstream | — | `f45eb9b88745` | unchanged |
| `compare_frozen_vs_published.py` | `scripts/path1_upscale/compare_frozen_vs_published.py` | not yet upstream | — | `973bd5521e26` | unchanged |
| `operator_divergence_both_directions.py` | — | paper only | — | `9cd94bed5d29` | unchanged |
| `manjunatha_hub_asymmetry_check.py` | `scripts/manjunatha_hub_asymmetry_check.py` | identical | `d008b6a6500b` | `34f95bc64f3d` | in- and out-strength labels follow the paper's convention (row sums received); values unchanged |

## `pipeline/02_ec_uncertainty`

| file | NeuroSim path | vs `89face3` | upstream | here | changed here |
|---|---|---|---|---|---|
| `run_ec_bootstrap_batch.py` | `scripts/run_ec_bootstrap_batch.py` | identical | `f4971b8bdbf8` | `c5481e7d5d27` | imports point to nctsens |
| `compute_and_save_ec_matrices.py` | `scripts/compute_and_save_ec_matrices.py` | identical | `720069051247` | `338b7099f7f8` | imports point to nctsens |
| `frassle_stephan_cross_check.py` | `scripts/frassle_stephan_cross_check.py` | identical | `759005100f19` | `759005100f19` | unchanged |
| `compare_ctrl_definitions.py` | `scripts/compare_ctrl_definitions.py` | not yet upstream | — | `419ec1dcfb71` | imports point to nctsens; docstrings name nctsens |
| `graphnet_sensitivity.py` | `scripts/path1_upscale/graphnet_sensitivity.py` | not yet upstream | — | `295a6c291b50` | imports point to nctsens; docstrings name nctsens |
| `bootstrap_block_length.py` | `scripts/path1_upscale/bootstrap_block_length.py` | not yet upstream | — | `3a2628f84d83` | imports point to nctsens; docstrings name nctsens |
| `compare_sweep_rerun.py` | `scripts/path1_upscale/compare_sweep_rerun.py` | not yet upstream | — | `22c437faae12` | unchanged |
| `sensitivity_vs_ntr_dual.py` | `scripts/path1_upscale/sensitivity_vs_ntr_dual.py` | not yet upstream | — | `57e7f818b6fb` | docstrings name nctsens |
| `compare_ntr_rerun.py` | `scripts/path1_upscale/compare_ntr_rerun.py` | not yet upstream | — | `69de1e822634` | unchanged |
| `hcp_motion_vs_tau.py` | `scripts/path1_upscale/hcp_motion_vs_tau.py` | not yet upstream | — | `de53f481413c` | unchanged |
| `adni_quality_vs_taustability.py` | `scripts/path1_upscale/adni_quality_vs_taustability.py` | not yet upstream | — | `fb9c15f088fc` | example identifier in a comment replaced by a placeholder |
| `twin_clustered_bootstrap.py` | `scripts/path1_upscale/twin_clustered_bootstrap.py` | not yet upstream | — | `bda0f96fafd6` | unchanged |
| `estimator_convergence/README.md` | `scripts/graphnet_convergence_fix/README.md` | not yet upstream | — | `6ae14dfa1e84` | module path given for this repository |
| `estimator_convergence/check_all_cohorts_complete.py` | `scripts/graphnet_convergence_fix/check_all_cohorts_complete.py` | not yet upstream | — | `c96fc5bf8786` | imports point to nctsens; subject lists read from environment variables instead of hard-coded IDs; example paths name $BNCS_ROOT |
| `estimator_convergence/check_adni_convergence.py` | `scripts/graphnet_convergence_fix/check_adni_convergence.py` | not yet upstream | — | `2e6db90a5720` | imports point to nctsens; subject lists read from environment variables instead of hard-coded IDs |
| `estimator_convergence/check_adni_more_iters.py` | `scripts/graphnet_convergence_fix/check_adni_more_iters.py` | not yet upstream | — | `5ca6bf008ded` | imports point to nctsens; subject lists read from environment variables instead of hard-coded IDs |
| `estimator_convergence/check_pathb_bootstrap_scope.py` | `scripts/graphnet_convergence_fix/check_pathb_bootstrap_scope.py` | not yet upstream | — | `ceca20e966e7` | imports point to nctsens; subject lists read from environment variables instead of hard-coded IDs |
| `estimator_convergence/check_hcp_unam_convergence.py` | `scripts/graphnet_convergence_fix/check_hcp_unam_convergence.py` | not yet upstream | — | `86e9a058af9c` | imports point to nctsens; one subject per cohort read from $HCP_SUBJECT and $UNAM_SUBJECT instead of hard-coded IDs |
| `estimator_convergence/unam_full_cohort_drift.py` | `scripts/graphnet_convergence_fix/unam_full_cohort_drift.py` | not yet upstream | — | `c13623fb569f` | imports point to nctsens; subject identifier in text replaced by a description |
| `estimator_convergence/confirm_raw_rho_mechanism.py` | `scripts/graphnet_convergence_fix/confirm_raw_rho_mechanism.py` | not yet upstream | — | `0a68c4d69dba` | imports point to nctsens; subject lists read from environment variables instead of hard-coded IDs |
| `estimator_convergence/diagnose_lambda_graph_sweep.py` | `scripts/graphnet_convergence_fix/diagnose_lambda_graph_sweep.py` | not yet upstream | — | `7e0a2ab0f866` | imports point to nctsens |
| `estimator_convergence/compare_N_T_ratio.py` | `scripts/UNAM_TLE/compare_N_T_ratio.py` | identical | `1ae576e69d7a` | `1ae576e69d7a` | unchanged |
| `estimator_convergence/confirm_bug.py` | — | paper only | — | `21c008b4e4bb` | imports point to nctsens; subject read from $UNAM_SUBJECT; site paths read from $BNCS_ROOT |
| `estimator_convergence/check_iteration_count.py` | — | paper only | — | `e19baead3c2d` | imports point to nctsens |
| `estimator_convergence/scope_check.py` | — | paper only | — | `f2dedca820f5` | imports point to nctsens; subject read from $UNAM_SUBJECT; site paths read from $BNCS_ROOT |
| `estimator_convergence/scope_check_all_cohorts.py` | — | paper only | — | `501b654daa53` | imports point to nctsens; example identifier in a comment replaced by a placeholder |
| `estimator_convergence/hcp_convergence_confirmation.py` | — | paper only | — | `82990068df92` | imports point to nctsens; subject lists read from environment variables instead of hard-coded IDs; site paths read from $BNCS_ROOT |

## `pipeline/03_linearization`

| file | NeuroSim path | vs `89face3` | upstream | here | changed here |
|---|---|---|---|---|---|
| `wc_linear_validity_sweep_real_coupling.py` | `scripts/nonlinear_validity_regime/wc_linear_validity_sweep_real_coupling.py` | later revision | `2dc6f9f7947b` | `38edb05a90d2` | imports point to nctsens; example identifier in a comment replaced by a placeholder |
| `real_coupling_boundary_recheck.py` | `scripts/nonlinear_validity_regime/real_coupling_boundary_recheck.py` | identical | `34beb35608cd` | `cab1d4c29071` | imports point to nctsens |
| `check_subject_coupling.py` | `scripts/nonlinear_validity_regime/check_subject_coupling.py` | not yet upstream | — | `0b20390dc2ae` | docstrings name nctsens; subject identifier replaced by its study pseudonym |
| `fixed_point_scale_check.py` | `scripts/path1_upscale/fixed_point_scale_check.py` | not yet upstream | — | `350167c7f80b` | imports point to nctsens; docstrings name nctsens |
| `localize_critical_displacement.py` | `scripts/path1_upscale/localize_critical_displacement.py` | not yet upstream | — | `65bf808445b2` | unchanged |
| `linearization_vs_numerical_error.py` | `scripts/nonlinear_validity_regime/linearization_vs_numerical_error.py` | not yet upstream | — | `b2c48e1e6481` | imports point to nctsens; docstrings name nctsens |
| `measure_non_normality.py` | `scripts/nonlinear_validity_regime/measure_non_normality.py` | not yet upstream | — | `f269a040c446` | imports point to nctsens; docstrings name nctsens |
| `null_model_rewiring.py` | `scripts/nonlinear_validity_regime/null_model_rewiring.py` | identical | `4a5653a615cd` | `4a5653a615cd` | unchanged |
| `wc_linear_validity_sweep_null_model.py` | `scripts/nonlinear_validity_regime/wc_linear_validity_sweep_null_model.py` | later revision | `868a2551923d` | `ab05cb87ddab` | unchanged |
| `network_topology_characterization.py` | `scripts/nonlinear_validity_regime/network_topology_characterization.py` | identical | `f0f332cf555d` | `f0f332cf555d` | unchanged |
| `mechanism_functional_form_search.py` | `scripts/path1_upscale/mechanism_functional_form_search.py` | not yet upstream | — | `c6f2fe817523` | unchanged |
| `mechanism_search_v2.py` | `scripts/path1_upscale/mechanism_search_v2.py` | not yet upstream | — | `6d829f8379ae` | unchanged |
| `fhn_validity_sweep_toy.py` | `scripts/nonlinear_validity_regime/fhn_validity_sweep_toy.py` | identical | `69869012c60e` | `8778b6bcb056` | imports point to nctsens |
| `fhn_kappa_sensitivity_check.py` | `scripts/nonlinear_validity_regime/fhn_kappa_sensitivity_check.py` | identical | `7f1872231e9f` | `3303179bcda8` | imports point to nctsens |
| `wc_timestep_discretization_check.py` | `scripts/nonlinear_validity_regime/wc_timestep_discretization_check.py` | identical | `332b83d8ddc8` | `5f7d30caa1f8` | imports point to nctsens |

## `pipeline/04_nonlinear_control`

| file | NeuroSim path | vs `89face3` | upstream | here | changed here |
|---|---|---|---|---|---|
| `adjoint_gradient.py` | `scripts/nonlinear_validity_regime/adjoint_gradient.py` | identical | `c6166165f75f` | `c6166165f75f` | unchanged |
| `adjoint_gradient_verification.py` | `scripts/nonlinear_validity_regime/adjoint_gradient_verification.py` | identical | `ece115cdf82a` | `ece115cdf82a` | unchanged |
| `adjoint_gradient_wc_verification.py` | `scripts/nonlinear_validity_regime/adjoint_gradient_wc_verification.py` | identical | `b43ee8181866` | `b7fe7e24d601` | imports point to nctsens |
| `fhn_simulation.py` | `scripts/nonlinear_validity_regime/fhn_simulation.py` | identical | `e5659e4ae304` | `e5659e4ae304` | unchanged |
| `fhn_generate_nominal_orbit.py` | `scripts/nonlinear_validity_regime/fhn_generate_nominal_orbit.py` | identical | `809211b9e74e` | `809211b9e74e` | unchanged |
| `fhn_nominal_orbit_data/nominal_t.npy` | `scripts/nonlinear_validity_regime/fhn_nominal_orbit_data/nominal_t.npy` | identical | `a3dc6b54223c` | `a3dc6b54223c` | unchanged |
| `fhn_nominal_orbit_data/nominal_v.npy` | `scripts/nonlinear_validity_regime/fhn_nominal_orbit_data/nominal_v.npy` | identical | `41f81ee923aa` | `41f81ee923aa` | unchanged |
| `fhn_nominal_orbit_data/nominal_w.npy` | `scripts/nonlinear_validity_regime/fhn_nominal_orbit_data/nominal_w.npy` | identical | `69e31b53be74` | `69e31b53be74` | unchanged |
| `stuart_landau_model.py` | `scripts/path1_upscale/stuart_landau_model.py` | not yet upstream | — | `9a606abfa682` | unchanged |
| `wc_oscillatory_linearization_comparison.py` | `scripts/nonlinear_validity_regime/wc_oscillatory_linearization_comparison.py` | identical | `1d6a401f0495` | `7bf87cb9d775` | imports point to nctsens |
| `fhn_broader_grid.py` | `scripts/nonlinear_validity_regime/fhn_broader_grid.py` | identical | `48493c07c0a3` | `13b51132d1ac` | imports point to nctsens |
| `wc_a2_broader_grid.py` | `scripts/nonlinear_validity_regime/wc_a2_broader_grid.py` | not yet upstream | — | `20a20c41b65d` | imports point to nctsens |
| `matched_cost_control_comparison.py` | `scripts/nonlinear_validity_regime/matched_cost_control_comparison.py` | not yet upstream | — | `a20a2e819677` | imports point to nctsens; docstrings name nctsens |
| `matched_cost_control_comparison_sl.py` | `scripts/nonlinear_validity_regime/matched_cost_control_comparison_sl.py` | not yet upstream | — | `32f03f72d02d` | imports point to nctsens; docstrings name nctsens |
| `matched_cost_control_comparison_wc.py` | `scripts/nonlinear_validity_regime/matched_cost_control_comparison_wc.py` | not yet upstream | — | `32e0ed503879` | imports point to nctsens; docstrings name nctsens |
| `constrained_dominance.py` | `scripts/nonlinear_validity_regime/constrained_dominance.py` | not yet upstream | — | `d0bd1f25f95d` | imports point to nctsens; docstrings name nctsens |
| `constrained_dominance_sl.py` | `scripts/nonlinear_validity_regime/constrained_dominance_sl.py` | not yet upstream | — | `8e46c9966a53` | imports point to nctsens; docstrings name nctsens |
| `constrained_dominance_wc.py` | `scripts/nonlinear_validity_regime/constrained_dominance_wc.py` | not yet upstream | — | `780e3f881c6a` | imports point to nctsens; docstrings name nctsens |
| `fhn_example_trajectories.py` | `scripts/nonlinear_validity_regime/fhn_example_trajectories.py` | not yet upstream | — | `9c8e71422ce9` | imports point to nctsens; docstrings name nctsens |
| `fhn_phase_shift_comparison.py` | `scripts/nonlinear_validity_regime/fhn_phase_shift_comparison.py` | identical | `6b61ae244d51` | `629a5980ca0a` | imports point to nctsens |
| `sl_a2_grid.py` | `scripts/path1_upscale/sl_a2_grid.py` | not yet upstream | — | `66e2fbeaafaa` | imports point to nctsens; docstrings name nctsens |
| `consolidate_sl_a2_replication.py` | `scripts/path1_upscale/consolidate_sl_a2_replication.py` | not yet upstream | — | `37df4105056d` | unchanged |

## `pipeline/05_clinical`

| file | NeuroSim path | vs `89face3` | upstream | here | changed here |
|---|---|---|---|---|---|
| `run_hcp_aud_batch_real_reward.py` | `scripts/run_hcp_aud_batch_real_reward.py` | identical | `e92baa8fd2da` | `0530566baf84` | imports point to nctsens |
| `run_adni_harmonize.py` | `scripts/run_adni_harmonize.py` | identical | `fb7d2f566969` | `1adacee9a21f` | imports point to nctsens |
| `tle_lateralization_test.py` | `scripts/UNAM_TLE/tle_lateralization_test.py` | identical | `9d5a710d6c21` | `9d5a710d6c21` | unchanged |
| `energy_barrier_analysis.py` | `scripts/UNAM_TLE/energy_barrier_analysis.py` | identical | `e7a60709d6b3` | `e7a60709d6b3` | unchanged |
| `compare_real_reward_vs_original.py` | `scripts/compare_real_reward_vs_original.py` | identical | `66f58e690dd0` | `955f8c447bd0` | embedded per-subject batch output (restricted HCP data) replaced by two log files given on the command line |
| `lateralization_checks/verify_asymmetry_direction.py` | `scripts/UNAM_TLE/verify_asymmetry_direction.py` | identical | `3799e80f85cb` | `3799e80f85cb` | unchanged |
| `lateralization_checks/sanity_check_asymmetry_variation.py` | `scripts/UNAM_TLE/sanity_check_asymmetry_variation.py` | identical | `8404135dfffd` | `8404135dfffd` | unchanged |
| `lateralization_checks/check_outlier_conditioning.py` | `scripts/UNAM_TLE/check_outlier_conditioning.py` | identical | `4a8d1f50c9a1` | `4a8d1f50c9a1` | unchanged |

## `docs`

| file | NeuroSim path | vs `89face3` | upstream | here | changed here |
|---|---|---|---|---|---|
| `REPRODUCIBILITY_NOTE.md` | `scripts/nonlinear_validity_regime/REPRODUCIBILITY_NOTE.md` | not yet upstream | — | `37cbb037f197` | unchanged |
| `plans/analysis_plan.md` | `docs/phase0_analysis_plan.md` | not yet upstream | — | `754774ddc138` | renamed; UNAM subject identifiers replaced by study pseudonyms; text otherwise verbatim; identifier of an excluded subject removed |
| `plans/connectome_scale_rerun_plan.md` | `docs/stringent_rerun_scope.md` | not yet upstream | — | `6c26ad4dfd7f` | renamed |

## `LICENSE`

| file | NeuroSim path | vs `89face3` | upstream | here | changed here |
|---|---|---|---|---|---|
| `LICENSE` | `LICENSE` | identical | `18a9b41baf52` | `18a9b41baf52` | unchanged |

## New in this repository

| file | purpose |
|---|---|
| `src/nctsens/__init__.py` | package entry point under the new name |
| `tests/test_driver_definition.py` | driver and receiver definitions on planted senders and receivers |
| `pipeline/*/reported_values.py, pipeline/reported.py` | every number in the text recomputed from `results/` and compared with the paper |
| `pipeline/02_ec_uncertainty/nctpy_definition_check.py` | nctpy 1.1.0 and 1.2.0 `ave_control` on planted operators (Methods 4.5) |
| `pipeline/01_operator_divergence/synthetic_chain_operator_check.py` | directed against symmetric operators on a simulated feedforward chain |
| `pipeline/03_linearization/error_tolerance_threshold.py` | the error-tolerance threshold of Supplementary S8 |
| `pipeline/02_ec_uncertainty/brain_maps.py` | the per-region values drawn in Figures 1f, 2g and 2h; the computation run on the analysis servers (there as `render_brain_maps.py --no-render`), with an `export` step for the deposited tables |
| `tests/test_brain_maps.py` | the brain-map driver score, sign convention, row blocks and export |
| `tables/check_tables.py` | Table 1 and Tables S1-S3 from `results/`, with a cell-by-cell check against the paper |
| `tables/published_tables.json` | table values as printed in the paper |
| `README.md, ORIGIN.md, docs/*.md, pipeline/*/README.md` | documentation of the study flow |
| `pyproject.toml, requirements.txt, Makefile, CITATION.cff, NOTICE` | packaging, pinned versions, entry points, citation |

## Results

`results/` holds the deposited outputs of the analyses, copied from the analysis servers with subject identifiers replaced by study pseudonyms (`Subject_###` for HCP, `ADNI_###`, `UNAM_###`). The keys are held by the authors and are not part of this repository. Columns that are HCP restricted data (age, family structure, zygosity, alcohol-use phenotype) are not present in any subject-level file, and subject-level ADNI files carry diagnosis and scanner manufacturer only. Rows are sorted by pseudonym. `docs/STUDY_FLOW.md` lists which script produced each file.


## Not carried over

NeuroSim also contains material this paper does not use: the plotting module, the validation pipeline and notebooks, figure scripts, compute benchmarks and timing rechecks, the first synthetic-coupling prototypes of the linearization sweep (superseded by the real-coupling sweep), a scan-duration script whose premise was falsified, and the development notes. They remain in NeuroSim.

