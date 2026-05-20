"""Tests for :mod:`src.optimization`."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.optimization import (
    TrajectoryStore,
    compute_loss_grid,
    detect_edge_of_stability,
    generate_filter_normalized_direction,
    hessian_top_eigenvalue,
    hessian_vector_product,
    lanczos_extrema,
    project_trajectory,
)
from src.utils import OptimizationError


def _quadratic_model_and_loss() -> tuple[nn.Linear, callable, torch.Tensor]:
    """Build a deterministic quadratic loss for testing.

    Returns
    -------
    tuple
        ``(model, loss_fn, A)`` where the Hessian of the loss w.r.t.
        ``model.weight.flatten()`` is exactly ``2 * A^T A / N``.
    """
    torch.manual_seed(0)
    A = torch.randn(8, 4)
    y = torch.randn(8, 4)
    model = nn.Linear(4, 4, bias=False)

    def loss_fn() -> torch.Tensor:
        return ((model(A) - y) ** 2).mean()

    return model, loss_fn, A


def test_filter_normalized_direction_shapes_and_norms() -> None:
    model = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 4))
    direction = generate_filter_normalized_direction(model, seed=0)
    state = model.state_dict()
    assert set(direction.keys()) == set(state.keys())
    for name, d in direction.items():
        assert d.shape == state[name].shape
        if "weight" in name and d.dim() >= 2:
            # Filter norms should match parameter filter norms
            d_norms = d.view(d.shape[0], -1).norm(dim=1)
            p_norms = state[name].view(state[name].shape[0], -1).norm(dim=1)
            assert torch.allclose(d_norms, p_norms, atol=1e-4)
        if "bias" in name.lower():
            assert torch.all(d == 0)


def test_filter_normalized_direction_rejects_empty() -> None:
    with pytest.raises(OptimizationError):
        generate_filter_normalized_direction({}, seed=0)


def test_hvp_matches_explicit_for_small_quadratic() -> None:
    """HVP must equal H@v computed via explicit Hessian."""
    model, loss_fn, _ = _quadratic_model_and_loss()
    # Build explicit Hessian via autograd
    params = [p for p in model.parameters() if p.requires_grad]
    g = torch.autograd.grad(loss_fn(), params, create_graph=True)
    g_flat = torch.cat([gi.flatten() for gi in g])
    H = torch.zeros(g_flat.numel(), g_flat.numel())
    for i in range(g_flat.numel()):
        grad_i = torch.autograd.grad(g_flat[i], params, retain_graph=True)
        H[i] = torch.cat([gi.flatten() for gi in grad_i])

    # Now test HVP
    v_flat = torch.randn(g_flat.numel())
    # Reshape v into param-shaped list
    v = []
    pos = 0
    for p in params:
        n = p.numel()
        v.append(v_flat[pos : pos + n].view_as(p))
        pos += n
    Hv = hessian_vector_product(loss_fn, model, v)
    Hv_flat = torch.cat([h.flatten() for h in Hv])
    assert torch.allclose(Hv_flat, H @ v_flat, atol=1e-4)


def test_hessian_top_eigenvalue_converges() -> None:
    model, loss_fn, _ = _quadratic_model_and_loss()
    lam, _ = hessian_top_eigenvalue(loss_fn, model, n_iter=40, tol=1e-6, seed=0)
    # Quadratic with positive semidef Hessian; lam must be positive
    assert lam > 0.0


def test_lanczos_returns_pair_with_correct_order() -> None:
    model, loss_fn, _ = _quadratic_model_and_loss()
    lam_min, lam_max = lanczos_extrema(loss_fn, model, m=10, seed=0)
    assert lam_min <= lam_max
    # For convex quadratic, both should be non-negative (modulo numerical noise)
    assert lam_max > 0.0


def test_trajectory_store_capture_and_iterate() -> None:
    model = nn.Linear(4, 4)
    store = TrajectoryStore()
    for step in range(3):
        store.capture(model, step, metrics={"loss": float(step)})
        with torch.no_grad():
            model.weight.add_(0.01 * torch.randn_like(model.weight))
    assert len(store) == 3
    assert store.steps() == [0, 1, 2]
    tbl = store.metrics_table()
    assert tbl["loss"] == [0.0, 1.0, 2.0]


def test_project_trajectory_consistency() -> None:
    model = nn.Linear(4, 4, bias=False)
    anchor = {k: v.detach().clone() for k, v in model.state_dict().items()}
    dir_a = generate_filter_normalized_direction(model, seed=1)
    dir_b = generate_filter_normalized_direction(model, seed=2)
    # Build a snapshot exactly equal to anchor + alpha*dir_a + beta*dir_b
    alpha, beta = 0.7, -0.4
    snap = {k: anchor[k] + alpha * dir_a[k] + beta * dir_b[k] for k in anchor}
    coords = project_trajectory([snap], dir_a, dir_b, anchor)
    # Inner products should recover (alpha * |a|^2, beta * |b|^2)
    a_sq = sum(float((v * v).sum()) for v in dir_a.values())
    b_sq = sum(float((v * v).sum()) for v in dir_b.values())
    # Cross terms are not orthogonal, so adjust expected:
    a_dot_b = sum(float((dir_a[k] * dir_b[k]).sum()) for k in anchor)
    assert abs(float(coords[0, 0]) - (alpha * a_sq + beta * a_dot_b)) < 1e-3
    assert abs(float(coords[0, 1]) - (alpha * a_dot_b + beta * b_sq)) < 1e-3


def test_detect_edge_of_stability_positive() -> None:
    """Synthetic trajectory: λ_max climbs above 2/η and loss oscillates."""
    eta = 0.1
    n = 60
    lam = np.concatenate([np.linspace(2.0, 25.0, 30), np.full(30, 22.0)])
    base = np.linspace(2.0, 0.5, n)
    osc = 0.05 * np.sin(np.arange(n) * 2.5)
    losses = base + osc
    report = detect_edge_of_stability(losses, lam, learning_rate=eta, plateau_steps=3)
    assert report.is_eos
    assert report.onset_step >= 0


def test_detect_edge_of_stability_negative() -> None:
    """Smooth descent below threshold should NOT be flagged as EOS."""
    eta = 0.1
    lam = np.linspace(5.0, 8.0, 50)  # below 2/η = 20 throughout
    losses = np.linspace(2.0, 0.5, 50)
    report = detect_edge_of_stability(losses, lam, learning_rate=eta)
    assert not report.is_eos


def test_eos_validates_inputs() -> None:
    with pytest.raises(OptimizationError):
        detect_edge_of_stability([1, 2, 3], [1, 2], learning_rate=0.1)
    with pytest.raises(OptimizationError):
        detect_edge_of_stability([1, 2, 3], [1, 2, 3], learning_rate=0.0)


def test_compute_loss_grid_shape_and_anchor_restore() -> None:
    torch.manual_seed(0)
    model = nn.Linear(4, 1, bias=False)
    X = torch.randn(16, 4)
    Y = torch.randn(16, 1)
    anchor = {k: v.detach().clone() for k, v in model.state_dict().items()}
    dir_a = generate_filter_normalized_direction(model, seed=11)
    dir_b = generate_filter_normalized_direction(model, seed=22)
    def lf(m: nn.Module) -> float:
        return float(((m(X) - Y) ** 2).mean().detach())
    a, b, grid = compute_loss_grid(model, anchor, dir_a, dir_b,
                                    alpha_range=(-0.5, 0.5),
                                    beta_range=(-0.5, 0.5),
                                    n_alpha=5, n_beta=5, loss_fn=lf)
    assert grid.shape == (5, 5)
    assert a.shape == (5,)
    assert b.shape == (5,)
    # Anchor must be restored
    for k, v in anchor.items():
        assert torch.allclose(v, model.state_dict()[k])
