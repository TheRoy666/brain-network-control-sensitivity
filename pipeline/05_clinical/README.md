# 05 · Exploratory clinical contrasts

Group contrasts of control energy, reported in Supplementary S1 as exploratory; none reached significance. Minimum detectable effects (Supplementary S3) come from `nctsens.clinical`.

| contrast | scripts | output in `results/05_clinical/` |
|---|---|---|
| alcohol-use severity in HCP twins, with a reward-network state definition as a sensitivity check | `../01_operator_divergence/run_hcp_aud_batch.py`, `run_hcp_aud_batch_real_reward.py` | `hcp_aud_group_comparison.csv` (group level only) |
| cognitively normal versus MCI, before and after controls-only harmonization across scanner manufacturers | `../01_operator_divergence/run_adni_nctn_batch.py`, `run_adni_harmonize.py` | `adni_*` |
| seizure-focus lateralization; energy barrier, healthy versus TLE | `../01_operator_divergence/run_unam_nctn_batch.py`, `tle_lateralization_test.py`, `energy_barrier_analysis.py` | `unam_*` |
| the reward-network sensitivity check, subject by subject | `compare_real_reward_vs_original.py` (reads the two batch logs, which stay with the data) | printed |

`lateralization_checks/` holds the checks made before the pre-registered lateralization result was read: the sign convention of the asymmetry score on the most extreme subjects (`verify_asymmetry_direction.py`), whether the spectral-radius anomaly of the epilepsy estimates reached the asymmetry scores (`sanity_check_asymmetry_variation.py`), and whether outlying scores track Gramian conditioning (`check_outlier_conditioning.py`).

`reported_values.py` recomputes every number the paper reports for this stage from `results/` and prints it beside the printed value (`make values` runs all stages).

