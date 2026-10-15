#!/usr/bin/env python3
"""
synthetic_chain_operator_check.py — minimum control energy under directed and
symmetric operators on a simulated feedforward chain, whose true direction of
influence is known (Results 2.1).

Each chain, region 1 -> 2 -> ... -> n with weight 0.85 and noise 0.1, is
simulated for 8000 timepoints and processed as the cohorts are (Methods 4.5,
4.6): a directed autoregressive estimate (ridge weight 1; the chain has no
structural connectome, so the structural-prior term is absent), the functional
operator (Pearson correlation after detrending, zero diagonal), and the
symmetrized estimate (A + A')/2, each rescaled to spectral radius 0.9; input at
every region; horizon 10; states from the recording by the cohort rule (mean of
the lowest- and highest-variance fifths of timepoints, unit norm).

Printed for each chain length: the median over simulated recordings of the
energy ratio, symmetric or functional over directed, for both transitions and
for random transitions, and the share of recordings in which the ratio exceeds
one. Two sensitivity analyses follow: a three-step horizon, and symmetric
versions of the generating chain itself without rescaling.

    python3 pipeline/01_operator_divergence/synthetic_chain_operator_check.py [--n-seeds 50]
"""
import argparse

import numpy as np

from nctsens.connectivity import functional_connectivity, ridge_effective_connectivity, simulate_feedforward_network
from nctsens.physics import minimum_energy, normalise_matrix

LENGTHS = (3, 5, 10, 20)
HORIZON = 10


def states(X):
    """Cohort rule: quiescent and high-variance states, unit norm."""
    n = max(1, X.shape[1] // 5)
    idx = np.argsort(X.var(axis=0))
    q, h = X[:, idx[:n]].mean(axis=1), X[:, idx[-n:]].mean(axis=1)
    return q / np.linalg.norm(q), h / np.linalg.norm(h)


def ratio(other, directed, x0, xT, horizon):
    B = np.eye(directed.shape[0])
    return minimum_energy(other, B, x0, xT, horizon)[0] / minimum_energy(directed, B, x0, xT, horizon)[0]


def operators(n, seed):
    X, A = simulate_feedforward_network(n_nodes=n, n_timepoints=8000, causal_weight=0.85, noise_std=0.1, seed=seed)
    ec = ridge_effective_connectivity(X, alpha=1.0)
    ops = {"directed": normalise_matrix(ec, target_rho=0.9),
           "functional": normalise_matrix(functional_connectivity(X), target_rho=0.9),
           "symmetrized": normalise_matrix((ec + ec.T) / 2, target_rho=0.9)}
    return X, A, ops


def summary(values):
    v = np.asarray(values)
    return f"{np.median(v):5.2f} ({np.mean(v > 1):4.0%} > 1)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-seeds", type=int, default=50)
    ap.add_argument("--n-random", type=int, default=100, help="random transitions per recording")
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    for horizon, title in ((HORIZON, "cohort settings, horizon 10"), (3, "sensitivity: horizon 3")):
        print(f"\n{title}: energy ratio over the directed estimate, median over {a.n_seeds} recordings (share above one)")
        print(f"  {'regions':>7}  {'operator':11s}  {'high-var -> quiescent':>22}  {'quiescent -> high-var':>22}  {'random':>16}")
        for n in LENGTHS:
            r = {k: {"h2q": [], "q2h": [], "rand": []} for k in ("functional", "symmetrized")}
            for seed in range(a.n_seeds):
                X, _, ops = operators(n, seed)
                q, h = states(X)
                for k in r:
                    r[k]["h2q"].append(ratio(ops[k], ops["directed"], h, q, horizon))
                    r[k]["q2h"].append(ratio(ops[k], ops["directed"], q, h, horizon))
                    r[k]["rand"].append(np.median([ratio(ops[k], ops["directed"], *rng.standard_normal((2, n)), horizon)
                                                   for _ in range(a.n_random)]))
            for k in r:
                print(f"  {n:>7}  {k:11s}  {summary(r[k]['h2q']):>22}  {summary(r[k]['q2h']):>22}  {summary(r[k]['rand']):>16}")

    print("\nsensitivity: the generating chain against symmetric versions of it, no rescaling, random transitions")
    for n in (3, 10):
        A = np.zeros((n, n))
        A[np.arange(1, n), np.arange(n - 1)] = 0.85
        for name, S in (("(A + A')/2", (A + A.T) / 2), ("A + A'", A + A.T)):
            rho = float(np.max(np.abs(np.linalg.eigvals(S))))
            if rho >= 1:
                print(f"  {n:>2} regions  {name:10s} unstable (spectral radius {rho:.2f}); not computed")
                continue
            for horizon in (3, HORIZON):
                v = np.median([ratio(S, A, *rng.standard_normal((2, n)), horizon) for _ in range(500)])
                print(f"  {n:>2} regions  {name:10s} horizon {horizon:>2}  ratio {v:5.2f}")


if __name__ == "__main__":
    main()
