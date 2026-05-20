r"""Filter-normalized random directions for loss-landscape visualization.

Following Li et al. (2018), a random direction :math:`\delta` is generated
with the same parameter structure as :math:`\theta^*`, then *each filter*
:math:`\delta_{i,j}` is rescaled by

.. math::

    \hat{\delta}_{i,j} \;=\; \frac{\delta_{i,j}}{\|\delta_{i,j}\|_F}
                              \cdot \|\theta_{i,j}\|_F

This removes scale-invariance artifacts (e.g. ReLU's :math:`\sigma(s x) =
\sigma(x)` for shifted activations) that would otherwise make the landscape
appear arbitrarily sharp or flat.

References
----------
* Li, H., Xu, Z., Taylor, G., Studer, C., Goldstein, T. (2018). "Visualizing
  the Loss Landscape of Neural Nets." NeurIPS.
"""

from __future__ import annotations

import torch
from torch import nn

from ..utils import OptimizationError, get_logger

__all__ = ["generate_filter_normalized_direction", "project_trajectory"]

_log = get_logger(__name__)


def _is_filter_param(name: str, p: torch.Tensor) -> bool:
    """Return True if ``p`` should be filter-normalized.

    Following the original paper, weight matrices of conv / linear layers are
    filter-normalized; biases, scalars, and 1-D normalization parameters are
    left zero (they would otherwise dominate small-norm directions).

    Parameters
    ----------
    name
        Parameter name (e.g. ``"layer.0.weight"``).
    p
        The tensor.

    Returns
    -------
    bool
        True for 2-D+ weight tensors.
    """
    if p.dim() < 2:
        return False
    if "bias" in name.lower():
        return False
    return True


def generate_filter_normalized_direction(
    reference: nn.Module | dict[str, torch.Tensor],
    *,
    seed: int = 0,
    device: torch.device | str = "cpu",
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    r"""Generate a filter-normalized random direction.

    Parameters
    ----------
    reference
        Either a ``torch.nn.Module`` (we use its ``state_dict``) or a state
        dict directly. The direction inherits the same parameter names,
        shapes, and dtypes.
    seed
        RNG seed.
    device
        Device to allocate the direction on.
    eps
        Numerical floor for filter-norm denominators.

    Returns
    -------
    dict
        Mapping ``name -> tensor`` with the same keys and shapes as the
        reference. Non-filter parameters are zero-tensors.

    Raises
    ------
    OptimizationError
        If reference is empty.
    """
    if isinstance(reference, nn.Module):
        state = {k: v.detach() for k, v in reference.state_dict().items()}
    else:
        state = {k: v.detach() for k, v in reference.items()}
    if not state:
        raise OptimizationError("reference has no parameters")
    g = torch.Generator(device="cpu").manual_seed(seed)
    direction: dict[str, torch.Tensor] = {}
    for name, p in state.items():
        if not _is_filter_param(name, p):
            direction[name] = torch.zeros_like(p, device=device)
            continue
        d = torch.randn(p.shape, generator=g, dtype=torch.float32).to(device)
        # Treat the first dim as the "filter" axis (output channels / rows).
        d_flat = d.view(p.shape[0], -1)
        p_flat = p.view(p.shape[0], -1).to(device).to(torch.float32)
        d_norms = d_flat.norm(dim=1, keepdim=True).clamp_min(eps)
        p_norms = p_flat.norm(dim=1, keepdim=True)
        d_flat = d_flat * (p_norms / d_norms)
        direction[name] = d_flat.view_as(p).to(p.dtype)
    _log.debug("Generated filter-normalized direction over %d params", len(direction))
    return direction


def project_trajectory(
    snapshots: list[dict[str, torch.Tensor]],
    direction_a: dict[str, torch.Tensor],
    direction_b: dict[str, torch.Tensor],
    anchor: dict[str, torch.Tensor],
) -> torch.Tensor:
    r"""Project a sequence of weight snapshots onto two directions.

    For each snapshot :math:`\theta_t`, computes

    .. math::

        \alpha_t = \langle \theta_t - \theta^*,\, \hat{\delta}\rangle,
        \qquad
        \beta_t = \langle \theta_t - \theta^*,\, \hat{\eta}\rangle

    where the inner product is the Euclidean dot product over all
    filter-normalized parameter tensors.

    Parameters
    ----------
    snapshots
        Sequence of state dicts.
    direction_a, direction_b
        Filter-normalized directions (from
        :func:`generate_filter_normalized_direction`).
    anchor
        Reference state dict (e.g. the final converged weights).

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(len(snapshots), 2)`` with columns
        :math:`(\alpha_t, \beta_t)`.

    Raises
    ------
    OptimizationError
        On key mismatch between snapshots / directions / anchor.
    """
    if not snapshots:
        raise OptimizationError("snapshots is empty")
    coords = torch.zeros(len(snapshots), 2)
    keys = set(anchor.keys())
    if set(direction_a.keys()) != keys or set(direction_b.keys()) != keys:
        raise OptimizationError("anchor / direction key sets disagree")
    for t, snap in enumerate(snapshots):
        if set(snap.keys()) != keys:
            raise OptimizationError(f"snapshot {t} key set disagrees with anchor")
        a_proj = 0.0
        b_proj = 0.0
        for name in keys:
            diff = (snap[name] - anchor[name]).to(torch.float32)
            a_proj += float((diff * direction_a[name].to(torch.float32)).sum().item())
            b_proj += float((diff * direction_b[name].to(torch.float32)).sum().item())
        coords[t, 0] = a_proj
        coords[t, 1] = b_proj
    return coords
