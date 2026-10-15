# Reproducibility note — validity-regime sweeps

**Read this before rerunning any sweep in this directory and comparing
against the results in `results/nonlinear_validity_regime/`.**

## The published results were produced with non-deterministic seeding

`wc_linear_validity_sweep_real_coupling.py` originally seeded the random
target-direction generator with Python's builtin `hash()` on a tuple
containing `subject_id`:

```python
hash((subject_id, int(w_EE*10), T, int(target_scale*1000), dir_seed))
```

`subject_id` comes from a filename slice (`f[:-len(sc_suffix)]`), so it
is a **string** at runtime for every cohort — including HCP, where the
stored CSV column later reads back as `int64` and looks numeric.

Python randomises string hashes per process (PEP 456, default since
3.3). **Every run therefore drew different target directions.** The
sweeps are statistically reproducible, not bit-reproducible.

`wc_linear_validity_sweep_null_model.py` imports `process_one_combo`
from the real-coupling module and inherits the same behaviour.

## This changes no conclusion

Target directions are random *by design* and averaged over `dir_seed`.
Different draws give statistically equivalent results. What was lost is
exact reproduction, not validity.

The variability was characterised directly by two independent
ADNI-Schaefer400 runs:

| quantity | run 1 | run 2 | difference |
|---|---|---|---|
| interior fraction (worst case) | 0.735 | 0.816 | 0.082 |
| **critical target displacement** | **0.02981** | **0.02893** | **3.0%** |

**The interior FRACTION is fragile** (±0.08 run-to-run, because
individual subjects' argmin flips). **The critical DISPLACEMENT is
robust** (3%), because it is set by a collapse spanning an eight-fold
change in the fraction, which a few subjects flipping cannot move.

This is why the analysis reports the critical displacement as the
primary statistic and treats the interior fraction as secondary.

## The current scripts are deterministic

Seeding now uses `zlib.crc32` over an explicit string key — stable
across processes, machines and Python versions.

**Consequence:** rerunning these scripts will *not* bit-reproduce the
CSVs in `results/`, because those were generated under the old seeding.
Results will be statistically equivalent. Expect agreement of the kind
shown in the table above: a few percent on the critical displacement,
up to ~0.08 on the interior fraction.

## Two further notes for anyone rerunning

1. **`--target-scales` is now an explicit flag on both scripts.**
   Previously the null-model sweep silently inherited `TARGET_SCALES`
   from the real-coupling module, so editing one file changed the other
   invisibly. Pass it explicitly on both; the null-model script warns
   if you do not.
2. **`--w-ee-values` is effectively required for non-HCP cohorts.**
   The default is the HCP grid. Each cohort's grid is normalised to its
   own stability boundary, so the grids differ in absolute w_EE while
   matching in `rho_A` to ~0.005 at every index. That is correct — do
   not "fix" it by forcing a common absolute grid.
