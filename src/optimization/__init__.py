"""Week 3 — Loss landscape topology & Edge of Stability.

Modules
-------
filter_normalization
    Li et al. (2018) filter-normalized random direction generation.
hessian_power_iteration
    Top eigenvalue / eigenvector of the loss Hessian via Hessian-vector
    products + Lanczos.
trajectory_capture
    Snapshot store for weight trajectories during training.
edge_of_stability
    Detector for the Edge-of-Stability regime where lambda_max >= 2/eta.
"""

from __future__ import annotations

from .edge_of_stability import (
    EdgeOfStabilityReport,
    detect_edge_of_stability,
)
from .filter_normalization import (
    generate_filter_normalized_direction,
    project_trajectory,
)
from .hessian_power_iteration import (
    hessian_top_eigenvalue,
    hessian_vector_product,
    lanczos_extrema,
)
from .trajectory_capture import (
    Snapshot,
    TrajectoryStore,
    compute_loss_grid,
)

__all__ = [
    "EdgeOfStabilityReport",
    "Snapshot",
    "TrajectoryStore",
    "compute_loss_grid",
    "detect_edge_of_stability",
    "generate_filter_normalized_direction",
    "hessian_top_eigenvalue",
    "hessian_vector_product",
    "lanczos_extrema",
    "project_trajectory",
]
