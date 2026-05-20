"""Week 1 — High-dimensional manifold geometry & representation collapse.

Modules
-------
anisotropy
    Empirical anisotropy index, SVD spectrum, effective rank.
manifold_projection
    From-scratch t-SNE (KL gradient descent) and UMAP (fuzzy-simplicial-set
    cross-entropy) implementations.
intrinsic_dimension
    TwoNN estimator (Facco et al., 2017) and the MLE estimator (Levina &
    Bickel, 2004).
"""

from __future__ import annotations

from .anisotropy import (
    AnisotropyReport,
    compute_anisotropy,
    cosine_similarity_distribution,
    explained_variance_gini,
    full_report,
    svd_effective_rank,
    svd_spectrum,
)
from .intrinsic_dimension import mle_intrinsic_dimension, twonn_intrinsic_dimension
from .manifold_projection import tsne_from_scratch, umap_from_scratch

__all__ = [
    "AnisotropyReport",
    "compute_anisotropy",
    "cosine_similarity_distribution",
    "explained_variance_gini",
    "full_report",
    "mle_intrinsic_dimension",
    "svd_effective_rank",
    "svd_spectrum",
    "tsne_from_scratch",
    "twonn_intrinsic_dimension",
    "umap_from_scratch",
]
