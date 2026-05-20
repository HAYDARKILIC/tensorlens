r"""Weight snapshot capture and loss-surface grid computation.

The :class:`TrajectoryStore` collects ``state_dict``-style snapshots at user
chosen training steps. :func:`compute_loss_grid` then evaluates the loss
along a 2-D mesh of perturbations:

.. math::

    \mathcal{L}(\alpha, \beta) \;=\;
    \mathcal{L}\bigl(\theta^* + \alpha \hat{\delta} + \beta \hat{\eta}\bigr)

with :math:`\hat{\delta}, \hat{\eta}` filter-normalized random directions.

Both modules are deliberately framework-agnostic: they manipulate
``state_dict``-style mappings rather than touching internal optimizer state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from ..utils import OptimizationError, get_logger

__all__ = ["Snapshot", "TrajectoryStore", "compute_loss_grid"]

_log = get_logger(__name__)


@dataclass(frozen=True)
class Snapshot:
    """A single weight snapshot.

    Attributes
    ----------
    step
        Training step at which the snapshot was taken.
    state_dict
        Parameter values (CPU tensors, detached).
    metrics
        Optional scalar metrics recorded at the same step (loss, grad-norm…).
    """

    step: int
    state_dict: dict[str, torch.Tensor]
    metrics: dict[str, float] = field(default_factory=dict)


class TrajectoryStore:
    """In-memory store of weight snapshots.

    Parameters
    ----------
    keep_on_cpu
        If True (default), tensors are moved to CPU before storage to keep
        VRAM low.
    """

    def __init__(self, *, keep_on_cpu: bool = True) -> None:
        self._snapshots: list[Snapshot] = []
        self._keep_on_cpu = keep_on_cpu

    def capture(
        self,
        model: nn.Module,
        step: int,
        *,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Capture a model snapshot.

        Parameters
        ----------
        model
            Module whose ``state_dict`` to clone.
        step
            Training step identifier.
        metrics
            Optional metric dict.
        """
        sd = {}
        for k, v in model.state_dict().items():
            t = v.detach().clone()
            if self._keep_on_cpu:
                t = t.cpu()
            sd[k] = t
        self._snapshots.append(Snapshot(step=step, state_dict=sd, metrics=metrics or {}))

    def __len__(self) -> int:
        return len(self._snapshots)

    def __getitem__(self, idx: int) -> Snapshot:
        return self._snapshots[idx]

    def steps(self) -> list[int]:
        """Return the list of captured step numbers."""
        return [s.step for s in self._snapshots]

    def state_dicts(self) -> list[dict[str, torch.Tensor]]:
        """Return all stored state dicts in capture order."""
        return [s.state_dict for s in self._snapshots]

    def metrics_table(self) -> dict[str, list[float]]:
        """Return metrics as a column-oriented table.

        Returns
        -------
        dict
            ``metric_name -> list[float]`` aligned with snapshot order.
        """
        all_keys: set[str] = set()
        for s in self._snapshots:
            all_keys.update(s.metrics.keys())
        out: dict[str, list[float]] = {k: [] for k in all_keys}
        for s in self._snapshots:
            for k in all_keys:
                out[k].append(float(s.metrics.get(k, float("nan"))))
        return out


def _apply_perturbation(
    state: dict[str, torch.Tensor],
    direction_a: dict[str, torch.Tensor],
    direction_b: dict[str, torch.Tensor],
    alpha: float,
    beta: float,
) -> dict[str, torch.Tensor]:
    """Return ``state + alpha * direction_a + beta * direction_b``."""
    out: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        delta = alpha * direction_a.get(k, torch.zeros_like(v))
        eta = beta * direction_b.get(k, torch.zeros_like(v))
        out[k] = v + delta.to(v.dtype).to(v.device) + eta.to(v.dtype).to(v.device)
    return out


def compute_loss_grid(
    model: nn.Module,
    anchor_state: dict[str, torch.Tensor],
    direction_a: dict[str, torch.Tensor],
    direction_b: dict[str, torch.Tensor],
    alpha_range: tuple[float, float] = (-1.0, 1.0),
    beta_range: tuple[float, float] = (-1.0, 1.0),
    n_alpha: int = 21,
    n_beta: int = 21,
    loss_fn: Callable[[nn.Module], float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Sweep the 2-D loss surface around an anchor state.

    Parameters
    ----------
    model
        Module to evaluate; its ``load_state_dict`` is used in-place.
    anchor_state
        Anchor :math:`\theta^*` (usually the final converged weights).
    direction_a, direction_b
        Filter-normalized directions.
    alpha_range, beta_range
        Span of perturbation coefficients.
    n_alpha, n_beta
        Grid resolution.
    loss_fn
        Callable ``model -> float`` that evaluates loss on a fixed reference
        batch. Required.

    Returns
    -------
    tuple
        ``(alphas, betas, losses)`` where ``alphas`` is shape ``(n_alpha,)``,
        ``betas`` is shape ``(n_beta,)``, and ``losses`` is
        ``(n_beta, n_alpha)`` arranged so that imshow / contour conventions
        place ``alphas`` along x and ``betas`` along y.

    Raises
    ------
    OptimizationError
        If ``loss_fn`` is not supplied.
    """
    if loss_fn is None:
        raise OptimizationError("loss_fn is required")
    alphas = np.linspace(alpha_range[0], alpha_range[1], n_alpha)
    betas = np.linspace(beta_range[0], beta_range[1], n_beta)
    losses = np.zeros((n_beta, n_alpha), dtype=np.float64)
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    try:
        for i, beta in enumerate(betas):
            for j, alpha in enumerate(alphas):
                perturbed = _apply_perturbation(
                    anchor_state, direction_a, direction_b, float(alpha), float(beta)
                )
                model.load_state_dict(perturbed, strict=False)
                losses[i, j] = float(loss_fn(model))
            _log.debug("Loss-grid row %d/%d complete", i + 1, n_beta)
    finally:
        model.load_state_dict(original, strict=False)
    return alphas, betas, losses
