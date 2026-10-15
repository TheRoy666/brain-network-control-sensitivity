# Definitions and fixed choices

The results depend on the choices below. Each is stated once here, with the module that implements it. The paper's Methods give the reasoning.

## Operator convention

All operators follow x(t+1) = A x(t). `A[i, j]` is therefore the influence of region *j* on region *i*: column *j* holds region *j*'s outgoing connections, row *i* its incoming ones.

## Operators

| operator | definition | code |
|---|---|---|
| functional (FC) | Pearson correlation of the linearly detrended regional timeseries; zero diagonal, negative values kept, no threshold | `nctsens.connectivity.functional_connectivity` |
| effective (EC) | first-order autoregressive fit with GraphNet regularization: ridge weight 1.0 plus a graph-Laplacian penalty from the subject's structural connectome, weight 1.0; lag 1; FISTA with tolerance 10⁻⁶ and at most 3000 iterations | `nctsens.connectivity.graphnet_effective_connectivity` |
| symmetrized EC | (A + Aᵀ)/2, renormalized; also without renormalization, with energy from the 10-step Gramian | `pipeline/01_operator_divergence/operator_divergence_both_directions.py` |

Every operator is rescaled to spectral radius 0.9 before use (`nctsens.physics.normalise_matrix`), except in the checks at 0.80 and 0.95.

## Control energy

Minimum energy over horizon T = 10 with B = I: E = (x_T − Aᵀx₀)ᵀ W_T⁺ (x_T − Aᵀx₀), with W_T = Σ_{t<T} Aᵗ(Aᵗ)ᵀ computed by doubling (`nctsens.physics.minimum_energy`, `compute_gramian_doubling`). Ill-conditioned Gramians are flagged, not dropped.

**States.** From each subject's own timeseries: the mean pattern over the fifth of timepoints with the highest variance across regions (high-variance state) and over the fifth with the lowest (quiescent state), each scaled to unit norm. They are a fixed probe, not physiological states. The FC/EC ratio is reported for high-variance → quiescent, with the reverse alongside.

## Average controllability: driver and receiver

| name | value at region *j* | ranks regions by | call |
|---|---|---|---|
| **driver** (Gu et al., 2015) | Σₜ ‖Aᵗ e_j‖², the diagonal of the Lyapunov solution for Aᵀ | how strongly an input at *j* spreads through the network | `average_controllability(A, direction="driver")` |
| receiver | Σₜ (Aᵗ Aᵗᵀ)_jj, the Gramian diagonal under input everywhere | how strongly *j* is reached by the network | `average_controllability(A, direction="receiver")` |

The two coincide when A is symmetric, so results on structural connectomes are unaffected; for a directed EC estimate they rank different regions. All driver rankings in the paper use the driver definition. The receiver definition is what NeuroSim's original function computed and what nctpy computed for directed matrices up to version 1.1.0; it remains the default of `average_controllability` so that NeuroSim outputs reproduce exactly, and calling it on a non-symmetric matrix without `direction` raises a warning. The node mean, the only quantity the clinical pipelines report, is identical under both definitions. `tests/test_driver_definition.py` checks both on planted senders and receivers.

## Rank stability

- **Resampling.** Moving-block bootstrap of the timeseries, re-estimating EC on each resample (`nctsens.connectivity.block_bootstrap_ec`). Block length follows the square-root rule per cohort: 69 timepoints in HCP, 14 in ADNI. 200 resamples per subject for the primary analysis; 50 in the sensitivity sweeps.
- **Agreement.** Kendall's τ across complete rankings (primary) and Jaccard overlap of the five top-ranked regions, ties broken by region index (`nctsens.connectivity.driver_node_rank_stability`). Chance Jaccard for 5 of 400–450 regions is about 0.0065.

## Linearization sweep

- **Model.** Wilson–Cowan, one excitatory and one inhibitory population per region; w_IE = 4.0, w_EI = 3.0, w_II = 2.0, c_E = c_I = −2.0, τ_E = 10, τ_I = 20; the excitatory self-coupling w_EE is swept (`nctsens.simulation.WilsonCowanNetwork`).
- **Connectome scaling.** log1p of streamline weights, rescaled to a mean row sum of (0.075 / N) × 20 × (N − 1), zero diagonal.
- **Grid.** Five couplings per cohort, set relative to each cohort's stability boundary (HCP 3.0–4.361; ADNI-400 3.0–3.969; ADNI-450 3.0–3.675); horizons 5, 20, 50, 100, 200; thirteen target displacements from 0.005 to 0.1; three random directions in the excitatory subspace.
- **Control.** Minimum-energy control from the Jacobian at the fixed point, discretized exactly (Van Loan), injected into the nonlinear system. Error = distance from target / requested displacement.
- **Interior minimum and critical displacement.** A subject has an interior minimum at a displacement if its direction- and horizon-averaged error curve over the five couplings is lowest strictly inside the grid. The critical displacement is where the proportion of subjects with an interior minimum falls through 0.5 (last descent, log-linear interpolation), with a 2000-resample subject bootstrap interval.
- **Seeding.** The deposited sweeps seeded target directions with Python's per-process string hash, so they are reproducible statistically, not bit for bit (two independent runs moved the interior-minimum fraction by up to 0.082 and the critical displacement by 3.0%). The code here seeds deterministically with CRC32.

## Nonlinear optimal control

Minimize half the squared terminal distance to the target subject to integrated control energy no greater than that of the time-varying linear solution, from zero initial control, with SLSQP, the analytic adjoint gradient and the analytic constraint gradient (iteration limit 500, tolerance 10⁻¹²). Solutions are re-integrated and their energy recomputed from the control sequence, identically for every method. Models: FitzHugh–Nagumo (nominal orbit integrated and spline-interpolated), Stuart–Landau (closed-form orbit and Jacobian) and Wilson–Cowan.
