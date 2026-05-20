"""Tests for :mod:`src.geometry`."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.geometry import (
    compute_anisotropy,
    cosine_similarity_distribution,
    explained_variance_gini,
    mle_intrinsic_dimension,
    svd_effective_rank,
    svd_spectrum,
    tsne_from_scratch,
    twonn_intrinsic_dimension,
    umap_from_scratch,
)
from src.utils import GeometryError
from src.utils.synthetic import AnisotropicConfig, synth_anisotropic_hidden_states


def test_anisotropy_bounds_for_isotropic_gaussian() -> None:
    """A truly isotropic Gaussian point cloud should have anisotropy ≈ 0."""
    torch.manual_seed(0)
    x = torch.randn(500, 64)
    a = compute_anisotropy(x, sample_size=300)
    assert -0.1 < a < 0.1


def test_anisotropy_for_cone_collapsed_is_positive() -> None:
    cfg = AnisotropicConfig(n_tokens=400, dim=64, anisotropy=0.75, seed=0)
    h = synth_anisotropic_hidden_states(cfg)
    a = compute_anisotropy(h, sample_size=300)
    assert a > 0.2  # cone-collapsed clearly above zero


def test_anisotropy_rejects_invalid_shapes() -> None:
    with pytest.raises(GeometryError):
        compute_anisotropy(torch.randn(10))  # 1-D
    with pytest.raises(GeometryError):
        compute_anisotropy(torch.randn(1, 8))  # N < 2


def test_anisotropy_rejects_nan() -> None:
    x = torch.randn(50, 16)
    x[0, 0] = float("nan")
    with pytest.raises(GeometryError):
        compute_anisotropy(x)


def test_cosine_similarity_distribution_length() -> None:
    x = torch.randn(200, 16)
    cos = cosine_similarity_distribution(x, sample_pairs=500)
    assert cos.ndim == 1
    assert 0 < cos.size <= 500
    assert np.all(cos >= -1.0 - 1e-6)
    assert np.all(cos <= 1.0 + 1e-6)


def test_svd_spectrum_descending_and_real() -> None:
    x = torch.randn(80, 32)
    s = svd_spectrum(x)
    assert s.shape == (32,)
    assert np.all(s >= 0.0)
    assert np.all(np.diff(s) <= 1e-8)  # descending


def test_effective_rank_isotropic_close_to_dim() -> None:
    torch.manual_seed(1)
    x = torch.randn(2000, 16)
    r = svd_effective_rank(x)
    # For isotropic Gaussian, effective rank should be close to ambient dim
    assert 12.0 < r <= 16.0


def test_effective_rank_collapsed_low() -> None:
    cfg = AnisotropicConfig(n_tokens=600, dim=32, anisotropy=0.9, noise_scale=0.05, seed=0)
    h = synth_anisotropic_hidden_states(cfg)
    r = svd_effective_rank(h)
    assert r < 16.0


def test_gini_bounded() -> None:
    torch.manual_seed(0)
    x = torch.randn(200, 16)
    g = explained_variance_gini(x)
    assert 0.0 <= g < 1.0


def test_tsne_shape_and_dtype() -> None:
    torch.manual_seed(0)
    x = torch.randn(60, 16)
    y = tsne_from_scratch(x, n_components=2, perplexity=10.0, n_iter=80, seed=0)
    assert y.shape == (60, 2)
    assert torch.isfinite(y).all()


def test_tsne_validates_perplexity() -> None:
    x = torch.randn(20, 4)
    with pytest.raises(GeometryError):
        tsne_from_scratch(x, perplexity=100.0)
    with pytest.raises(GeometryError):
        tsne_from_scratch(x, n_components=4)


def test_umap_shape() -> None:
    torch.manual_seed(0)
    x = torch.randn(50, 10)
    y = umap_from_scratch(
        x, n_components=2, n_neighbors=8, n_epochs=20, seed=0
    )
    assert y.shape == (50, 2)
    assert torch.isfinite(y).all()


def test_twonn_on_5_sphere() -> None:
    """5-D unit sphere → intrinsic dim ≈ 5."""
    torch.manual_seed(0)
    x = torch.randn(300, 5)
    x = x / x.norm(dim=1, keepdim=True)
    d = twonn_intrinsic_dimension(x)
    # TwoNN underestimates on a sphere but should land in a sensible range
    assert 2.0 < d < 9.0


def test_mle_intrinsic_dimension() -> None:
    torch.manual_seed(0)
    x = torch.randn(300, 6)
    d = mle_intrinsic_dimension(x, k=8)
    assert 3.0 < d < 10.0


def test_anisotropy_subsample_matches_full() -> None:
    """Subsampled and full anisotropy should agree within tolerance."""
    torch.manual_seed(0)
    x = torch.randn(400, 32)
    a_full = compute_anisotropy(x)
    a_sub = compute_anisotropy(x, sample_size=200, seed=42)
    assert abs(a_full - a_sub) < 0.06
