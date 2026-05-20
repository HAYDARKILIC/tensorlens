r"""Hessian top-eigenvalue estimation via Hessian-vector products.

We never materialize the full Hessian (it has :math:`|\theta|^2` entries and
is intractable). Instead, the **Hessian-vector product** :math:`Hv` is
computed efficiently by double backward through ``torch.autograd.grad``:

.. math::

    H v \;=\; \nabla_\theta \bigl(\nabla_\theta \mathcal{L}(\theta)^\top v\bigr)

The top eigenvalue :math:`\lambda_{\max}` is then obtained by **power
iteration**:

.. math::

    v_{t+1} = \frac{H v_t}{\|H v_t\|_2}, \qquad
    \lambda_{\max} \approx v_t^\top H v_t

For both top and bottom eigenvalues we use a small-`m` **Lanczos**
algorithm: building an :math:`m`-dimensional Krylov subspace yields a
tridiagonal matrix :math:`T_m` whose extreme eigenvalues approximate
:math:`\lambda_{\max}` and :math:`\lambda_{\min}`. We re-orthogonalize at
each step for numerical stability.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import nn

from ..utils import OptimizationError, get_logger

__all__ = [
    "hessian_vector_product",
    "hessian_top_eigenvalue",
    "lanczos_extrema",
]

_log = get_logger(__name__)


LossFn = Callable[[], torch.Tensor]


def _params_from(model: nn.Module) -> list[nn.Parameter]:
    """Return trainable parameters as a list (deterministic ordering)."""
    return [p for p in model.parameters() if p.requires_grad]


def _zero_like_params(params: list[nn.Parameter]) -> list[torch.Tensor]:
    """Allocate zero-tensors matching ``params``."""
    return [torch.zeros_like(p) for p in params]


def _flatten(vs: list[torch.Tensor]) -> torch.Tensor:
    """Flatten a list of tensors into one 1-D tensor."""
    return torch.cat([v.reshape(-1) for v in vs])


def _unflatten(flat: torch.Tensor, like: list[torch.Tensor]) -> list[torch.Tensor]:
    """Inverse of :func:`_flatten` using shapes from ``like``."""
    out: list[torch.Tensor] = []
    pos = 0
    for ref in like:
        numel = ref.numel()
        out.append(flat[pos : pos + numel].view_as(ref))
        pos += numel
    return out


def hessian_vector_product(
    loss_fn: LossFn,
    model: nn.Module,
    vector: list[torch.Tensor],
    *,
    create_graph: bool = False,
) -> list[torch.Tensor]:
    r"""Compute :math:`Hv` for the loss returned by ``loss_fn``.

    Parameters
    ----------
    loss_fn
        Zero-argument callable returning a scalar tensor representing the
        loss to differentiate. Must be differentiable through the model
        parameters.
    model
        Module whose parameters define :math:`\theta`.
    vector
        List of tensors with shapes matching the trainable parameters.
    create_graph
        If True, keep the graph for further differentiation (rare).

    Returns
    -------
    list of torch.Tensor
        :math:`Hv` as a parameter-shaped list.

    Raises
    ------
    OptimizationError
        On shape mismatch.
    """
    params = _params_from(model)
    if len(vector) != len(params):
        raise OptimizationError(
            f"vector length {len(vector)} disagrees with #params {len(params)}"
        )
    for v, p in zip(vector, params, strict=True):
        if v.shape != p.shape:
            raise OptimizationError(
                f"vector shape {tuple(v.shape)} mismatches param shape {tuple(p.shape)}"
            )

    model.zero_grad(set_to_none=True)
    loss = loss_fn()
    grad = torch.autograd.grad(
        loss, params, create_graph=True, retain_graph=True, allow_unused=False
    )
    dot = sum((g * v).sum() for g, v in zip(grad, vector, strict=True))
    hv = torch.autograd.grad(dot, params, retain_graph=create_graph, allow_unused=False)
    return list(hv)


def hessian_top_eigenvalue(
    loss_fn: LossFn,
    model: nn.Module,
    *,
    n_iter: int = 30,
    tol: float = 1e-4,
    seed: int = 0,
) -> tuple[float, list[torch.Tensor]]:
    r"""Power iteration for the top Hessian eigenvalue.

    Parameters
    ----------
    loss_fn
        Zero-argument loss closure.
    model
        Model providing parameters.
    n_iter
        Maximum power iterations.
    tol
        Convergence tolerance on the relative change of the eigenvalue.
    seed
        RNG seed for the initial vector.

    Returns
    -------
    tuple
        ``(lambda_max, eigenvector)`` where ``eigenvector`` is parameter-shaped.

    Raises
    ------
    OptimizationError
        If the model has no trainable parameters.
    """
    params = _params_from(model)
    if not params:
        raise OptimizationError("model has no trainable parameters")
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = [torch.randn(p.shape, generator=g).to(p.device).to(p.dtype) for p in params]
    # Normalize
    v_flat = _flatten(v)
    v = _unflatten(v_flat / v_flat.norm().clamp_min(1e-12), params)

    eig_prev = 0.0
    for it in range(n_iter):
        Hv = hessian_vector_product(loss_fn, model, v)
        Hv_flat = _flatten(Hv)
        norm = float(Hv_flat.norm().clamp_min(1e-12))
        eig = float((Hv_flat * _flatten(v)).sum())
        v = _unflatten(Hv_flat / norm, params)
        if it > 0 and abs(eig - eig_prev) < tol * max(abs(eig), 1.0):
            _log.debug("Power iteration converged at step %d: λ=%.4e", it, eig)
            return eig, v
        eig_prev = eig
    _log.debug("Power iteration finished without tol convergence: λ=%.4e", eig_prev)
    return eig_prev, v


def lanczos_extrema(
    loss_fn: LossFn,
    model: nn.Module,
    *,
    m: int = 20,
    seed: int = 0,
) -> tuple[float, float]:
    r"""Estimate the extreme Hessian eigenvalues via the Lanczos algorithm.

    Constructs an orthonormal Krylov basis :math:`\{q_1, \ldots, q_m\}` of
    :math:`\mathrm{span}(v, Hv, H^2 v, \ldots, H^{m-1}v)`. The projected
    operator :math:`T_m = Q^\top H Q` is tridiagonal; its extreme eigenvalues
    converge to :math:`\lambda_{\max}` and :math:`\lambda_{\min}` of
    :math:`H`.

    Parameters
    ----------
    loss_fn
        Zero-argument loss closure.
    model
        Model providing parameters.
    m
        Krylov subspace dimension (10–50 typical).
    seed
        RNG seed for the initial vector.

    Returns
    -------
    tuple of float
        ``(lambda_min, lambda_max)``.
    """
    params = _params_from(model)
    if not params:
        raise OptimizationError("model has no trainable parameters")
    if m < 2:
        raise OptimizationError("Krylov dimension m must be >= 2")
    g = torch.Generator(device="cpu").manual_seed(seed)
    q_flat = torch.randn(sum(p.numel() for p in params), generator=g)
    q_flat = q_flat / q_flat.norm().clamp_min(1e-12)

    alphas: list[float] = []
    betas: list[float] = []
    Q: list[torch.Tensor] = [q_flat.clone()]

    q_prev = torch.zeros_like(q_flat)
    beta_prev = 0.0

    for _j in range(m):
        v = _unflatten(Q[-1], params)
        Hv = hessian_vector_product(loss_fn, model, v)
        Hv_flat = _flatten(Hv).to(q_flat.dtype)
        alpha = float((Hv_flat * Q[-1]).sum())
        alphas.append(alpha)
        w = Hv_flat - alpha * Q[-1] - beta_prev * q_prev
        # Full re-orthogonalization for numerical stability
        for q in Q:
            w = w - float((w * q).sum()) * q
        beta = float(w.norm())
        if beta < 1e-10:
            _log.debug("Lanczos break at iter %d (beta=%.2e)", _j, beta)
            break
        q_prev = Q[-1]
        beta_prev = beta
        Q.append(w / beta)
        betas.append(beta)

    # Build tridiagonal and diagonalize
    n_eff = len(alphas)
    T = np.zeros((n_eff, n_eff))
    for i in range(n_eff):
        T[i, i] = alphas[i]
    for i in range(n_eff - 1):
        T[i, i + 1] = betas[i]
        T[i + 1, i] = betas[i]
    eigvals = np.linalg.eigvalsh(T)
    return float(eigvals[0]), float(eigvals[-1])
