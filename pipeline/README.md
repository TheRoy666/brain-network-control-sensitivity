# Pipeline

Stages run left to right; each needs only the data and the outputs of earlier stages.

| stage | question | needs |
|---|---|---|
| [`00_preprocessing`](00_preprocessing/) | regional timeseries and structural connectomes for each cohort | raw imaging data |
| [`01_operator_divergence`](01_operator_divergence/) | how far apart are FC- and EC-based control energies, and what drives the gap? | timeseries, connectomes |
| [`02_ec_uncertainty`](02_ec_uncertainty/) | do driver rankings survive resampling, and under which definition? | timeseries, connectomes |
| [`03_linearization`](03_linearization/) | where does linearized control fail on real connectomes? | connectomes |
| [`04_nonlinear_control`](04_nonlinear_control/) | what does a nonlinear controller achieve with the same energy? | nothing (model systems) |
| [`05_clinical`](05_clinical/) | exploratory group contrasts | outputs of stage 01 |

Install the package first (`pip install -e .` at the repository root); scripts import `nctsens`, and scripts in the same stage import each other, so run them from their own directory. Every deposited output is under `results/<stage>/`; `docs/STUDY_FLOW.md` maps them to the paper. Each stage's `reported_values.py` recomputes, from those outputs, every number the paper reports for that stage and compares it with the printed value; `reported.py` is the comparison they share.
