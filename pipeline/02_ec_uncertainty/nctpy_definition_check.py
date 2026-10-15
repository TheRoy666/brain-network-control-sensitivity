#!/usr/bin/env python3
"""
nctpy_definition_check.py — what nctpy's discrete-time ave_control ranks on a
directed operator, before and after version 1.2.0.

Methods 4.5 and Results 2.3: nctpy up to 1.1.0 computes, from the real Schur
decomposition A = U T U', the value at node i as sum_j U[i, j]^2 / (1 - T[j, j]^2).
That equals the Gu et al. (2015) definition when A is symmetric and not
otherwise. Version 1.2.0 computes the Gu et al. definition for directed
operators. This script applies both versions to planted operators, where
the answer is known, and compares their rankings with the two definitions in
nctsens.physics:

    driver   - input at one node, sum_t ||A^t e_j||^2   (Gu et al., 2015)
    receiver - diagonal of the Gramian with input at every node

Convention: x(t+1) = A x(t), so column j of A holds node j's outgoing influence.

    pip install --target /tmp/nctpy110 --no-deps nctpy==1.1.0
    pip install --target /tmp/nctpy120 --no-deps nctpy==1.2.0
    python3 pipeline/02_ec_uncertainty/nctpy_definition_check.py \\
        --nctpy-110 /tmp/nctpy110 --nctpy-120 /tmp/nctpy120

With --install the two versions are fetched into a temporary folder first.
"""
import argparse
import importlib
import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy.stats import spearmanr

from nctsens import physics

N, K = 40, 5


def planted(kind, seed):
    """Weak random background plus K planted senders (columns) or receivers (rows)."""
    rng = np.random.default_rng(seed)
    A = np.abs(rng.normal(scale=0.05, size=(N, N)))
    if kind == "senders":
        A[:, :K] += 1.0
    else:
        A[:K, :] += 1.0
    return physics.normalise_matrix(A, target_rho=0.9)


def load_nctpy(path):
    """Import nctpy.metrics from one site directory, isolated from any other copy."""
    for m in [m for m in sys.modules if m == "nctpy" or m.startswith("nctpy.")]:
        del sys.modules[m]
    sys.path.insert(0, path)
    try:
        mod = importlib.import_module("nctpy.metrics")
        ver = importlib.import_module("nctpy").__version__
    finally:
        sys.path.remove(path)
    return mod, ver


def top(v):
    return set(np.argsort(v)[::-1][:K])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nctpy-110")
    ap.add_argument("--nctpy-120")
    ap.add_argument("--install", action="store_true", help="pip-install both versions into a temporary folder")
    ap.add_argument("--n-seeds", type=int, default=20)
    a = ap.parse_args()
    if a.install:
        tmp = tempfile.mkdtemp(prefix="nctpy_")
        for v, attr in (("1.1.0", "nctpy_110"), ("1.2.0", "nctpy_120")):
            d = os.path.join(tmp, v)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", d, f"nctpy=={v}"], check=True)
            setattr(a, attr, d)
    if not (a.nctpy_110 and a.nctpy_120):
        ap.error("give --nctpy-110 and --nctpy-120, or --install")

    m110, v110 = load_nctpy(a.nctpy_110)
    ac110 = m110.ave_control
    m120, v120 = load_nctpy(a.nctpy_120)
    ac120 = m120.ave_control
    print(f"nctpy {v110} and {v120}; {a.n_seeds} planted operators of each kind, N = {N}, {K} planted nodes\n")

    rows = {k: [] for k in ("r110_receiver", "r110_driver", "r120_driver", "r120_receiver", "sym_110", "sym_120")}
    hits = {k: 0 for k in ("110_finds_senders", "110_finds_receivers", "120_finds_senders", "120_finds_receivers")}
    for kind in ("senders", "receivers"):
        for s in range(a.n_seeds):
            A = planted(kind, s)
            drv = physics.average_controllability(A, direction="driver")
            rec = physics.average_controllability(A, direction="receiver")
            n110, n120 = ac110(A, system="discrete"), ac120(A, system="discrete")
            rows["r110_receiver"].append(spearmanr(n110, rec)[0])
            rows["r110_driver"].append(spearmanr(n110, drv)[0])
            rows["r120_driver"].append(spearmanr(n120, drv)[0])
            rows["r120_receiver"].append(spearmanr(n120, rec)[0])
            planted_set = set(range(K))
            hits[f"110_finds_{kind}"] += top(n110) == planted_set
            hits[f"120_finds_{kind}"] += top(n120) == planted_set
    for s in range(a.n_seeds):
        rng = np.random.default_rng(100 + s)
        S = np.abs(rng.normal(size=(N, N)))
        S = physics.normalise_matrix(S + S.T, target_rho=0.9)
        g = physics.average_controllability(S, direction="driver")
        rows["sym_110"].append(np.max(np.abs(ac110(S, system="discrete") - g)) / np.max(g))
        rows["sym_120"].append(np.max(np.abs(ac120(S, system="discrete") - g)) / np.max(g))

    med = {k: float(np.median(v)) for k, v in rows.items()}
    print("Rank correlation (Spearman), median over all planted operators")
    print(f"  nctpy {v110} vs receiver definition   {med['r110_receiver']:+.3f}")
    print(f"  nctpy {v110} vs driver definition     {med['r110_driver']:+.3f}")
    print(f"  nctpy {v120} vs driver definition     {med['r120_driver']:+.3f}")
    print(f"  nctpy {v120} vs receiver definition   {med['r120_receiver']:+.3f}")
    print("\nTop five equal to the planted nodes (of the operators of that kind)")
    for k, v in hits.items():
        print(f"  nctpy {k.replace('_', ' ', 1).replace('_', ' ')}: {v}/{a.n_seeds}")
    print("\nSymmetric operators: largest relative difference from the driver definition")
    print(f"  nctpy {v110}: {max(rows['sym_110']):.1e}    nctpy {v120}: {max(rows['sym_120']):.1e}")
    print("\nReading: a version that behaves like the receiver ranking correlates near +1 with it and finds the")
    print("planted receivers; the driver definition finds the planted senders. Both definitions agree on")
    print("symmetric operators, so the difference only appears for directed (effective-connectivity) estimates.")


if __name__ == "__main__":
    main()
