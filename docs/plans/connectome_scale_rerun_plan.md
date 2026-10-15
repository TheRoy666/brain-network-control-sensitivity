# Stringent server rerun — detailed scope

**Purpose:** everything run so far (A1 sweep, A2 three-way comparison) used
toy N=10 networks with random-uniform coupling, to verify the *methods*
work correctly. This document scopes what needs to change to turn those
verified methods into paper-grade evidence. Nothing here has been run yet —
this is the plan, written before execution, per the project's own
established discipline (Phase 0 pattern).

---

## 0. Gating question — fully resolved

**All required raw data confirmed available:**
- Structural connectivity (SC): streamline-count-weighted and
  SIFT2-weighted, per subject, for ADNI (**both atlas variants** —
  Schaefer400 cortex-only and Schaefer400+TianS3) and HCP-AUD.
- Native BOLD timeseries (`.npy`): confirmed for **all three cohorts**
  (ADNI — both atlas variants, HCP-AUD, and UNAM), matching the
  `_native_timeseries.npy` pattern referenced in the batch scripts.
- UNAM has no DTI/SC (expected, unchanged) — BOLD timeseries alone (with
  the shared ENIGMA template as structural prior) is what already
  produced the existing `unam_results.csv` EC estimates, so this isn't a
  new gap, just a confirmation of the existing setup.

**Net effect:** real per-subject directed EC can now be (re-)derived via
`graphnet_effective_connectivity` for all three cohorts — SC+BOLD for
ADNI/AUD, BOLD+shared-template for UNAM (as already done). Real SC-based
WC coupling is available for ADNI/AUD directly. The Manjunatha in/out-hub
check can now include all three cohorts, not just two — UNAM's estimates
carry the already-documented `raw_rho`-pinning caveat, which should be
noted alongside any UNAM result from this check, not treated as
disqualifying.

**Note on execution vs. planning:** this resolves *whether the data
exists and is accessible*, not whether the actual files are in front of
this chat session right now. Actual computation on these matrices/
timeseries still needs either the files uploaded here or the NBRC server
session — this section clears the planning dependency, not the execution
step itself.

---

## 1. Path A1 rerun

### 1a. Scale to real parcellation sizes
Current: N=10 toy. Target: match actual cohort parcellations —
- UNAM: N=68 (DK, shared template)
- ADNI-Schaefer400: N=400
- ADNI-TianS3: N to be confirmed (check whether this is Schaefer400+TianS3
  combined or TianS3 alone — ambiguous from file naming, resolve before
  committing compute budget)
- HCP-AUD: N to be confirmed (no N column in the public results CSV;
  check `run_hcp_aud_batch.py` / loader for the actual atlas used)

**State dimension is 2N** (E and I per region), so N=400 means an 800×800
Jacobian, `expm`, and `eig` per (seed, regime-point) evaluation. This is
qualitatively different compute cost from N=10 (20-dim) — **benchmark on
the actual server hardware before committing to any grid size.** Do not
assume the N=10 timings (0.04s/run) scale linearly; `expm`/`eig` are
superlinear (roughly O(n^3)) in matrix dimension, so 800³/20³ ≈ 64,000× more
expensive per linear-algebra call in the worst case, though real BLAS
implementations will do better than naive scaling — this needs an actual
measurement, not a guess.

### 1b. Real coupling matrices, not random-uniform
For each cohort, sample a representative subset of subjects (not
necessarily all — see 1e on statistical planning) and use their **actual
estimated EC matrices** (post-`normalise_matrix`, matching what
`finite_vs_infinite_comparison` etc. already operate on) as the WC
network's `C`. This directly connects the simulation-based validity map to
the real teleportation-ratio finding from the operator-divergence analysis.

### 1c. Parameter alignment with Muldoon et al. (2016)
Run the sweep **twice**: once with our own damped/w_EE-scan
parameterization (as already built, for the bifurcation-proximity story),
and once with Muldoon et al.'s exact published constants (`c1=16, c2=12,
c3=15, c4=3, a_e=1.3, a_i=2, θ_e=4, θ_i=3.7, τ=8ms`) for direct,
citable comparability. Note their parameterization uses a single time
constant `τ` for both E and I (unlike our separate `tau_E=10, tau_I=20`) —
translate carefully, don't just substitute blindly.

### 1d. Delays and noise — decide explicitly, don't default silently
Muldoon et al. include distance-based transmission delays (0.8–14.8ms,
derived from real inter-region distance / 10 m/s conduction velocity) and
additive noise (σ=0.00001). Our current network has neither.
- **Adding delays is a real engineering lift**: `simulate`/`simulate_controlled`
  use `solve_ivp`, which doesn't natively handle delay differential
  equations (DDEs). Would need a DDE-capable integrator (e.g. fixed-step
  integration with a history buffer/interpolator for the delayed terms) —
  scope this as its own sub-task with its own time estimate, not an
  afterthought.
- **Minimum acceptable fallback if delays aren't implemented**: explicitly
  justify the no-delay simplification in the limitations section, citing
  the specific delay range Muldoon et al. used, rather than silently
  omitting it.
- Noise is cheaper to add (already have the `sigma` pattern from
  `WilsonCowanNode`) — lower priority but should be included if time
  allows, since it's inexpensive and closes another gap vs. Muldoon.

### 1e. Statistical planning — how many subjects/seeds is "enough"
Before running: decide N subjects per cohort (all vs. a representative
sample) and N coupling-matrix draws for the synthetic/random-coupling arm,
with a stated justification (e.g. "N=20 real subjects per cohort gives a
CI of width X on the bifurcation-proximity correlation, based on the
effect size already observed in the toy sweep"), rather than picking a
number ad hoc after seeing how long the first few runs take.

### 1f. Real-data validity-map placement (new integrative analysis)
For each real subject's EC matrix, compute where it actually sits on the
validity-regime map: its `rho_A` after normalization, its distance from
whatever bifurcation boundary exists for its own dynamics, and its
Gramian conditioning — then check whether subjects with worse conditioning
(already shown to correlate with `teleport_ratio` in the real-data
analysis) also show larger *simulated* reachability error when their own
matrix is used as `C` in a WC network. This is the connective-tissue
result that ties the toy-network sweep directly to the real cross-cohort
finding — arguably the single most valuable addition the rerun can make.

---

## 2. Path A2 rerun

### 2a. Scale and real coupling
Same N/real-coupling considerations as 1a/1b apply to A2's oscillatory
regime (`LIMIT_CYCLE_PARAMS`) as well. Note: confirm real subjects'
post-normalization dynamics actually sit in a fixed-point regime (this
seems likely given `raw_rho`<1 or ≈1 but not oscillating in the cohorts
checked so far) — **if no real cohort's inferred dynamics are actually
oscillatory, A2's practical relevance needs careful framing**: it
characterizes a real theoretical failure mode of naive NCT practice, but
whether real fMRI-derived brain dynamics actually live in this regime is a
separate, nontrivial empirical question, likely out of scope to resolve
here. State this limitation explicitly rather than implying A2 describes
what's happening in the real cohorts we have.

### 2b. Adjoint gradient — the real engineering task for making the ceiling affordable
The nonlinear-optimal-control ceiling took ~2.75 min/case using
derivative-free Powell optimization over a reduced 5-segment control
parameterization. This does not scale to the multi-seed, multi-shift grid
needed for a real finding (only 1 of 2 planned cases finished in the
current session). **Priority engineering task: implement the adjoint
(costate) method** — standard Pontryagin's-principle machinery:
- Forward-simulate the state trajectory under a candidate control.
- Backward-integrate the costate equation `dλ/dt = -λᵀ(∂f/∂x) - (target-tracking term)`.
- Gradient of the cost w.r.t. `u(t)` falls out as `λᵀ(∂f/∂u)` at each
  time — using exactly the `input_jacobian_at`-style computation already
  built, evaluated along the trajectory.
- This turns each gradient evaluation from O(dimension) forward simulations
  (finite differences) into **one forward + one backward pass**, regardless
  of how many control parameters there are — the actual fix for the
  scaling problem, not just a smaller `maxiter`.

This is well-scoped, standard applied-math machinery, not a research risk —
but it's real implementation work and should be budgeted as its own task
before the broader A2 grid is run.

### 2c. Broader grid once 2b is done
With adjoint gradients making the ceiling cheap enough: rerun the
three-way comparison (naive / LTV / nonlinear ceiling) across the same
seed × shift × T grid density used for the cheap legs in this session (5
seeds × 3 shifts was tractable; extend to whatever the adjoint speedup
affords), plus the phase-dependent stability scan repeated at real N.

---

## 3. Engineering / compute considerations

**Real benchmark results (server, sequential, random coupling at real N):**

| cohort N | full single-run cost | full A1 grid (sequential) |
|---|---|---|
| UNAM (68) | 0.16s | 7.1 min |
| ADNI-Schaefer400 (400) | 2.50s | 78.9 min |
| HCP-AUD (410) | 4.45s | 174.2 min* |
| ADNI-Schaefer400+TianS3 (450) | 3.10s | 94.0 min |

*HCP-AUD's number is likely inflated by server contention noise, not a
real algorithmic cost — `compute_gramian_doubling` (pure matrix
multiplication, no conditioning-dependent branching) jumped 3x between
T=50 and T=100 only at this N, while N=400 stayed flat across the same T
range. Worth rerunning this one before trusting it precisely, though it
doesn't change the overall picture.

**Cost concentrates in two operations at real N:** `find_fixed_point`
(0.7-0.9s) and `minimum_energy_trajectory` (1.4-3.3s, dominated by `pinv`
on an ~800-900 dimensional matrix). `jacobian_at`, `input_jacobian_at`,
and `simulate_controlled` stay cheap (<0.2s) even at N=450 — if further
speedup is ever needed, focus there.

- **Parallelization is now a priority action, not just a nice-to-have.**
  The grid is embarrassingly parallel across the 8 seeds (each seed's
  fixed-point/Jacobian/discretization work is fully independent). At
  ~8-way parallelism, HCP-AUD's worst-case ~174 min becomes ~22 min.
  Implement a multiprocessing wrapper over the seed loop before running
  the real full grid — the sequential benchmark numbers above are the
  ceiling, not the expected real runtime.
- **A2 nonlinear ceiling still unbenchmarked at real N — do not attempt a
  naive real-N run.** Given how much the cheap legs already cost at real
  N, the Powell-based direct-shooting approach (2.75 min/case at toy
  N=10) is very likely impractical at N=400+. Prioritize the
  adjoint-gradient implementation (Section 2b) before touching A2 at
  real scale at all — don't burn server time on a naive attempt first.
- Numerical validation at real scale, not just toy scale (as before —
  rerun core correctness checks at N=400, not just N=3-10).

---

## 4. Pre-registration addendum

Before running anything in this scope, write a short pre-specified plan
(same discipline as the original Phase 0 doc and Path C's lateralization
test) stating: which cohorts/subjects, which N per arm, what the primary
claims being tested are (bifurcation-proximity correlation direction and
magnitude; naive vs. LTV vs. ceiling ordering), and what would count as
falsifying each one. This protects against quietly reframing results after
seeing them, consistent with how the rest of this project has operated.

---

## 6. Model robustness extension — FitzHugh-Nagumo as a second ground truth

**Rationale:** A1/A2's findings (bifurcation-proximity improving linear
accuracy; naive linearization failing in the oscillatory regime) are
currently demonstrated on Wilson-Cowan specifically. The obvious reviewer
question is whether this is a WC-specific quirk or a general property of
nonlinear neural-mass dynamics. FitzHugh-Nagumo (FN) is the standard
alternative model in exactly this space, with directly relevant prior
work already using it for genuine nonlinear optimal control on whole-brain
networks: Chouzouris, Roth, Cakan & Obermayer (2021), "Applications of
optimal nonlinear control to a whole-brain network of FitzHugh-Nagumo
oscillators," *Phys. Rev. E* 104:024213 — same research group as
Salfenmoser & Obermayer (2023), already cited in the novelty-positioning
doc.

**Scope:**
- Reimplement the A1 validity-regime sweep (same three axes: target
  distance, horizon, stability-margin/bifurcation-proximity) on an FN
  network, reusing the same `find_fixed_point`/`jacobian_at`/
  `input_jacobian_at`/`discretize_system` machinery pattern — FN's
  fixed-point structure and bifurcation behavior differ from WC's but the
  general linearization/control pipeline is model-agnostic by
  construction.
- Reimplement A2's three-way comparison (naive / LTV / nonlinear ceiling)
  on FN's oscillatory regime, citing Chouzouris et al. (2021) directly for
  the nonlinear-OCT reference approach.
- **This is a robustness check, not a new primary result** — frame it in
  the paper as "the same pattern holds under a second, independent
  neural-mass model," not as a separate finding requiring equal weight to
  the WC results.
- Sequence this *after* the WC stringent rerun (Sections 1-2) is complete
  and the adjoint-gradient method (2b) exists, since FN's A2 comparison
  would otherwise hit the same compute bottleneck.

## 7. Discussion framing — functional segregation/integration vocabulary

No rerun needed; a writing task. Average controllability (moving toward
nearby, easily-reachable states) and modal controllability (moving toward
distant, hard-to-reach states) map directly onto the field's standard
integration/segregation vocabulary — this is how Muldoon et al. (2016)
themselves frame their hub/non-hub and DMN/subcortical results. Use this
vocabulary explicitly in the Discussion when relating the operator-
divergence and validity-regime findings back to the broader NCT
literature; costs nothing, and reads as more fluent to reviewers already
thinking in these terms.

## 8. Explicitly considered and excluded from scope

For the record, so this isn't silently re-litigated later without context:
- **Hodgkin-Huxley / integrate-and-fire single-neuron models**: wrong
  scale for a region-level EC/SC-based NCT framework; would require an
  entirely different multi-scale bridging methodology, not a natural
  extension. At most, a motivating sentence in the Introduction on why
  WC/FN (mean-field reductions of this single-neuron detail) are the
  appropriate scale.
- **Dynamic field theory**: continuous spatial-field formulation,
  incompatible with the discrete N-region/N×N-connectivity-matrix
  approach used throughout this project. Would require reformulating the
  state space entirely, not extending it.
- **Higher-order sensory processing, attention, memory, consciousness**:
  cognitive/behavioral domains, not methods, and don't match the actual
  cohorts (AUD, MCI, TLE). Explicitly resisted as scope creep at the point
  of consolidating Phase A toward a paper.

- [ ] Confirmed data-access answer (Section 0)
- [ ] Benchmark numbers at real N, informing final grid size
- [ ] A1 rerun: own-parameterization + Muldoon-aligned-parameterization,
      both on real coupling matrices per cohort
- [ ] A1 real-data validity-map placement (Section 1f)
- [ ] Explicit delay/noise decision, documented either way
- [ ] A2 adjoint-gradient implementation, verified against the existing
      finite-difference/Powell result as a correctness check
- [ ] A2 broader grid using the adjoint method
- [ ] Updated novelty-positioning doc if any of the above changes the
      comparison against Muldoon et al. / Salfenmoser & Obermayer
- [ ] Pre-registration addendum written and dated before results are seen
- [ ] FitzHugh-Nagumo robustness rerun of A1 (Section 6), sequenced after
      the WC stringent rerun
- [ ] FitzHugh-Nagumo robustness rerun of A2, sequenced after the
      adjoint-gradient method (2b) exists
- [ ] Discussion section drafted using segregation/integration vocabulary
      (Section 7) — writing task, no rerun dependency
- [ ] Confirm HH/IF, DFT, and cognitive-domain extensions remain out of
      scope (Section 8) unless project scope deliberately changes
