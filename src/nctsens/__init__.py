"""
nctsens
=======
Analysis code for "brain-network-control-sensitivity": how network
control-theoretic conclusions depend on the operator, on estimation
uncertainty and on linearization.

This package is a frozen subset of NeuroSim (GSoC 2026, INCF project 39;
https://github.com/TheRoy666/NeuroSim), renamed so that it can be installed
next to NeuroSim without conflict. ORIGIN.md lists every file taken from
NeuroSim and what was changed. The one change in behavior is the
``direction`` argument of :func:`physics.average_controllability`, which
adds the driver definition used throughout the paper.

Modules
-------
physics       finite-horizon Gramians, minimum-energy control, controllability
connectivity  functional and effective connectivity (GraphNet), block bootstrap,
              driver-ranking stability
simulation    Wilson-Cowan network, discretization, Jacobians
loader        connectome and timeseries loading with orientation checks
harmonize     controls-only site harmonization
clinical      cohort pipelines, group contrasts and effect sizes
"""

__version__ = "1.0.0"
__license__ = "Apache-2.0"

from nctsens.physics import (
    normalise_matrix,
    compute_gramian_doubling,
    minimum_energy,
    minimum_energy_trajectory,
    average_controllability,
    driver_average_controllability,
    modal_controllability,
    finite_vs_infinite_comparison,
)
from nctsens.connectivity import (
    functional_connectivity,
    ridge_effective_connectivity,
    graphnet_effective_connectivity,
    graph_laplacian,
    block_bootstrap_ec,
    driver_node_rank_stability,
)
from nctsens.simulation import WilsonCowanNetwork

__all__ = [
    "normalise_matrix", "compute_gramian_doubling", "minimum_energy",
    "minimum_energy_trajectory", "average_controllability",
    "driver_average_controllability", "modal_controllability",
    "finite_vs_infinite_comparison", "functional_connectivity",
    "ridge_effective_connectivity", "graphnet_effective_connectivity",
    "graph_laplacian", "block_bootstrap_ec", "driver_node_rank_stability",
    "WilsonCowanNetwork",
]
