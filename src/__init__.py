"""TensorLens — first-principles LLM diagnostics.

A graduate-level toolkit for visualizing and mathematically analyzing the
internal mechanics of Large Language Models. All algorithms (SVD anisotropy
metrics, t-SNE / UMAP, Hessian power iteration, sparse autoencoders, roofline
modeling) are implemented from PyTorch / NumPy / SciPy primitives — no
opaque high-level wrappers.

Subpackages
-----------
geometry      : High-dimensional manifold geometry (Week 1).
mechanistic   : Residual stream hooks and sparse autoencoders (Weeks 2 & 5).
optimization  : Loss landscape topology and Edge of Stability (Week 3).
profiler      : HPC roofline analysis and kernel pipelines (Week 4).
utils         : Shared logging, seeding, and synthetic-tensor utilities.
"""

from __future__ import annotations

__version__: str = "0.1.0"
__author__: str = "TensorLens Contributors"
__license__: str = "MIT"

__all__: list[str] = ["__version__", "__author__", "__license__"]
