# 03 · Validity of the linearization

Places a Wilson–Cowan model on each subject's connectome, computes minimum-energy control from its linearization at the fixed point, applies that control to the nonlinear model and measures how far it lands from the target.

| order | script | does | output in `results/03_linearization/` |
|---|---|---|---|
| 1 | `real_coupling_boundary_recheck.py` | connectome loading and scaling; stability boundary per cohort | — |
| 2 | `wc_linear_validity_sweep_real_coupling.py` | the sweep over coupling, horizon, target displacement and direction (`--target-scales` for the 13-value grid) | `fine_sweep/wc_sweep_finets_*.csv.gz`; first five-value sweep in `coarse_sweep/` |
| 3 | `check_subject_coupling.py`, `fixed_point_scale_check.py` | the one missing (subject, coupling) cell; fixed-point scale by cohort | `fixed_point_scale_check.csv` |
| 4 | `localize_critical_displacement.py` | interior-minimum fraction by displacement; critical displacement with bootstrap interval | printed |
| 5 | `linearization_vs_numerical_error.py` | the same controls applied to the linear system, to separate numerical from linearization error | `linearization_vs_numerical_error_*.csv` |
| 5b | `error_tolerance_threshold.py` | the threshold defined by an error tolerance: displacement at which the share of subjects within 5% or 10% error falls through one half, at best, mean and worst coupling (Supplementary S8) | printed |
| 6 | `measure_non_normality.py` | non-normality of the Jacobian and coupling, against each subject's threshold | `non_normality_*.csv` |
| 7 | `null_model_rewiring.py`, `wc_linear_validity_sweep_null_model.py` | the sweep repeated on degree-preserving rewired connectomes | `fine_sweep/wc_null_finets_*`, `coarse_sweep/wc_null_*` |
| 8 | `network_topology_characterization.py` | hub concentration and rich-club structure of the connectomes | `network_topology_*.csv` |
| 9 | `mechanism_functional_form_search.py`, then `mechanism_search_v2.py` | candidate forms combining control energy and amplification, scored against each subject's own optimum and a majority-class baseline (Supplementary S2) | printed |
| 10 | `fhn_validity_sweep_toy.py`, `fhn_kappa_sensitivity_check.py` | the displacement sweep in FitzHugh–Nagumo, and its coupling-scale check: whether the transition replicates in a second model (Discussion) | printed |
| — | `wc_timestep_discretization_check.py` | calibration of the discretization step against the neural time constants (Methods 4.7) | printed |

Shown in Figure 4 and Supplementary S2.

`reported_values.py` recomputes every number the paper reports for this stage from `results/` and prints it beside the printed value (`make values` runs all stages).

The non-normality correlations in Results 2.9 are taken at the highest coupling, the operating point nearest the bifurcation, where transient amplification can occur; `measure_non_normality.py` prints the same test on each subject's median across couplings, which gives slightly different values (`reported_values.py` reproduces the reported ones).

`docs/REPRODUCIBILITY_NOTE.md` explains why the sweeps reproduce statistically but not bit for bit.

