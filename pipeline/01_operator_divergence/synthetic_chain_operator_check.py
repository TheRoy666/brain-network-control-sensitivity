#!/usr/bin/env python3
"""
synthetic_chain_operator_check.py — does a symmetric operator raise or lower
minimum control energy on a feedforward chain whose true direction is known?

Two constructions are compared, because they answer different questions.

1. Operators estimated from simulated data, processed exactly as the cohorts
   are (Methods): a directed ridge effective-connectivity estimate, and either
   the functional connectivity of the same data or the symmetrized estimate,
   each normalized to spectral radius 0.9. This is the comparison the human
   results make.

2. The generating chain itself against symmetric versions of it with no
   rescaling: (A + A')/2, which halves each connection and adds its reverse,
   and A + A', which adds reverse connections at full weight.

Energy is the minimum control energy (nctsens.physics.minimum_energy, input
at every node) for random transitions between unit-variance states; the ratio
is symmetric over directed, and a ratio above one means the symmetric operator
needs more energy.

    python3 pipeline/01_operator_divergence/synthetic_chain_operator_check.py [--n-seeds 50]
"""
import argparse

import numpy as np

from nctsens.connectivity import functional_connectivity, ridge_effective_connectivity, simulate_feedforward_network
from nctsens.physics import minimum_energy, normalise_matrix


def median_ratio(directed, symmetric, horizon, rng, n_transitions):
    n = directed.shape[0]
    B = np.eye(n)
    r = []
    for _ in range(n_transitions):
        x0, xT = rng.standard_normal((2, n))
        r.append(minimum_energy(symmetric, B, x0, xT, horizon)[0] / minimum_energy(directed, B, x0, xT, horizon)[0])
    return float(np.median(r))


def chain(n, w):
    A = np.zeros((n, n))
    A[np.arange(1, n), np.arange(n - 1)] = w
    return A


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-seeds", type=int, default=50)
    ap.add_argument("--n-transitions", type=int, default=200)
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    print("1. Estimated operators, normalized to spectral radius 0.9 (the cohort pipeline)")
    print(f"   median over {a.n_seeds} simulated recordings of the per-recording median ratio; share of recordings above one")
    for n in (3, 10):
        for horizon in (3, 8):
            ratios = {"functional": [], "symmetrized": []}
            for seed in range(a.n_seeds):
                X, _ = simulate_feedforward_network(n_nodes=n, n_timepoints=8000, causal_weight=0.85, noise_std=0.1, seed=seed)
                ec = ridge_effective_connectivity(X, alpha=0.1)
                directed = normalise_matrix(ec, target_rho=0.9)
                ratios["functional"].append(median_ratio(directed, normalise_matrix(functional_connectivity(X), target_rho=0.9),
                                                         horizon, rng, a.n_transitions))
                ratios["symmetrized"].append(median_ratio(directed, normalise_matrix((ec + ec.T) / 2, target_rho=0.9),
                                                          horizon, rng, a.n_transitions))
            for k, v in ratios.items():
                v = np.asarray(v)
                print(f"   {n:2d} nodes, horizon {horizon}, {k:11s}  ratio {np.median(v):5.2f}  above one in {np.mean(v > 1):4.0%}")

    print("\n2. The generating chain against symmetric versions, no rescaling")
    for n in (3, 10):
        for w in (0.5, 0.85):
            A = chain(n, w)
            for name, S in (("(A + A')/2", (A + A.T) / 2), ("A + A'", A + A.T)):
                rho = float(np.max(np.abs(np.linalg.eigvals(S))))
                if rho >= 1:
                    print(f"   {n:2d} nodes, weight {w}, {name:10s} unstable (spectral radius {rho:.2f}); not computed")
                    continue
                for horizon in (3, 8):
                    r = median_ratio(A, S, horizon, rng, 500)
                    print(f"   {n:2d} nodes, weight {w}, {name:10s} horizon {horizon}  ratio {r:5.2f}")

    print("\nReading: whether a symmetric operator is cheaper depends on how it is built and scaled. Under the")
    print("cohort pipeline it is costlier in most settings, as in every human subject; it is cheaper when reverse")
    print("connections are added at full weight without rescaling (which can make the operator unstable), and in")
    print("the 10-node chain at the short horizon.")


if __name__ == "__main__":
    main()
