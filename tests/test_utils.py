"""Tests for :mod:`src.utils`."""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch

from src.utils import (
    GeometryError,
    MechanisticError,
    OptimizationError,
    ProfilerError,
    TensorLensError,
    describe_device,
    get_logger,
    set_global_seed,
)
from src.utils.synthetic import (
    AnisotropicConfig,
    AttentionConfig,
    synth_anisotropic_hidden_states,
    synth_attention_matrix,
    synth_loss_landscape_weights,
    synth_polysemantic_activations,
    synth_residual_stream,
)


def test_exception_hierarchy() -> None:
    assert issubclass(GeometryError, TensorLensError)
    assert issubclass(MechanisticError, TensorLensError)
    assert issubclass(OptimizationError, TensorLensError)
    assert issubclass(ProfilerError, TensorLensError)


def test_get_logger_is_configured_once() -> None:
    log = get_logger("tensorlens.tests.utils")
    log2 = get_logger("tensorlens.tests.utils")
    assert log is log2
    assert len(log.handlers) == 1
    assert log.propagate is False
    assert isinstance(log, logging.Logger)


def test_set_global_seed_makes_torch_deterministic() -> None:
    set_global_seed(42)
    a = torch.randn(5)
    set_global_seed(42)
    b = torch.randn(5)
    assert torch.allclose(a, b)


def test_set_global_seed_validates() -> None:
    with pytest.raises(TensorLensError):
        set_global_seed(-1)


def test_describe_device_returns_string() -> None:
    s = describe_device("cpu")
    assert isinstance(s, str)
    assert "cpu" in s


def test_synth_anisotropic_shape_and_dtype() -> None:
    cfg = AnisotropicConfig(n_tokens=100, dim=16, anisotropy=0.5, n_clusters=2)
    x = synth_anisotropic_hidden_states(cfg)
    assert x.shape == (100, 16)
    assert x.dtype == torch.float32


def test_synth_anisotropic_rejects_invalid() -> None:
    with pytest.raises(GeometryError):
        synth_anisotropic_hidden_states(AnisotropicConfig(anisotropy=0.0))
    with pytest.raises(GeometryError):
        synth_anisotropic_hidden_states(AnisotropicConfig(anisotropy=1.0))


def test_synth_attention_is_row_stochastic_on_causal_rows() -> None:
    attn = synth_attention_matrix(AttentionConfig(seq_len=12, n_heads=3, seed=0))
    assert attn.shape == (3, 12, 12)
    # The first row (causal i=0) attends only to position 0 → row sum = 1
    assert torch.allclose(attn[0, 0].sum(), torch.tensor(1.0), atol=1e-5)
    # Every row sums to 1 (within causal prefix)
    sums = attn.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_synth_residual_stream_shape() -> None:
    t = synth_residual_stream(n_layers=3, n_heads=2, seq_len=8, d_model=16, seed=0)
    assert t.shape == (3, 2, 8, 16)


def test_synth_loss_landscape_shapes() -> None:
    w, l = synth_loss_landscape_weights(n_steps=20, n_params=4, seed=0)
    assert w.shape == (20, 4)
    assert l.shape == (20,)


def test_synth_polysemantic_requires_overcomplete() -> None:
    with pytest.raises(MechanisticError):
        synth_polysemantic_activations(n_samples=10, d_model=16, n_features=8)
    acts, codes = synth_polysemantic_activations(
        n_samples=50, d_model=8, n_features=32, sparsity=0.1, seed=0
    )
    assert acts.shape == (50, 8)
    assert codes.shape == (50, 32)
