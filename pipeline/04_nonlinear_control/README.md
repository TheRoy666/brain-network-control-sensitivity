# 04 · Nonlinear optimal control

In three oscillatory models, compares the linear controllers with a nonlinear controller that is given no more energy than the time-varying linear solution.

| model | module | notes |
|---|---|---|
| FitzHugh–Nagumo | `fhn_simulation.py`; orbit from `fhn_generate_nominal_orbit.py` (stored in `fhn_nominal_orbit_data/`) | nominal orbit integrated and spline-interpolated |
| Stuart–Landau | `stuart_landau_model.py` | orbit and Jacobian in closed form |
| Wilson–Cowan | `wc_oscillatory_linearization_comparison.py` | oscillatory regime of the network model |

| order | script | does | output in `results/04_nonlinear_control/` |
|---|---|---|---|
| 1 | `adjoint_gradient.py` | adjoint (costate) gradient of terminal error with respect to piecewise-constant control | — |
| 2 | `adjoint_gradient_verification.py`, `adjoint_gradient_wc_verification.py` | the gradient against finite differences | printed |
| 3 | `constrained_dominance.py`, `_sl.py`, `_wc.py` | **primary comparison**: minimize terminal error subject to the linear solution's energy, from zero control | `constrained_dominance_*.csv` |
| 4 | `matched_cost_control_comparison.py`, `_sl.py`, `_wc.py`; `fhn_broader_grid.py`, `wc_a2_broader_grid.py` | secondary check across ten effort penalties and four initializations | `matched_cost_*.csv`, `*_broader_grid.csv` |
| 5 | `fhn_example_trajectories.py` | trajectories and controls for the condition at the median error ratio | `fhn_example.npz` |
| 6 | `fhn_phase_shift_comparison.py` | frozen versus time-varying linearization in FitzHugh–Nagumo, the counterpart of the Wilson–Cowan comparison | printed |
| 7 | `sl_a2_grid.py`, then `consolidate_sl_a2_replication.py` | the Stuart–Landau grid across six regimes, and the check that the three supercritical regimes are one system up to rescaling (Supplementary S4; Discussion) | printed |

Shown in Figure 5 and Supplementary S4.

`reported_values.py` recomputes every number the paper reports for this stage from `results/` and prints it beside the printed value (`make values` runs all stages).

