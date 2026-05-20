"""Synthetic tensor fixtures that mathematically reflect real LLM phenomena.

These generators produce dummy hidden states, attention matrices, and weight
trajectories whose statistical properties replicate empirically-observed
behaviors in production transformers:

* :func:`synth_anisotropic_hidden_states` — cone-collapsed representations
  with a configurable anisotropy index :math:`\\mathcal{A} \\in (0, 1)`.
* :func:`synth_attention_matrix` — softmax-normalized attention with mixable
  induction-head, attention-sink, and uniform components.
* :func:`synth_residual_stream` — layered residual stream with per-head
  low-rank contributions.
* :func:`synth_loss_landscape_weights` — a sequence of weight snapshots
  tracing an Adam-like trajectory toward a saddle-adjacent basin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import GeometryError, MechanisticError, get_logger

__all__ = [
    "AnisotropicConfig",
    "AttentionConfig",
    "synth_anisotropic_hidden_states",
    "synth_attention_matrix",
    "synth_residual_stream",
    "synth_loss_landscape_weights",
    "synth_polysemantic_activations",
]

_log = get_logger(__name__)


@dataclass(frozen=True)
class AnisotropicConfig:
    """Configuration for synthesized anisotropic hidden states.

    Attributes
    ----------
    n_tokens
        Number of token vectors to generate (``N``).
    dim
        Embedding dimensionality (``d``); modern LLMs use ``d >= 4096``.
    anisotropy
        Target expected pairwise cosine similarity in ``(0, 1)``. Higher
        values mean tighter cone collapse.
    n_clusters
        Number of latent topic clusters; clusters share a common cone axis
        and differ only by within-cluster scatter.
    noise_scale
        Standard deviation of additive isotropic noise on top of the cone
        structure.
    seed
        RNG seed for full reproducibility.
    """

    n_tokens: int = 2048
    dim: int = 768
    anisotropy: float = 0.7
    n_clusters: int = 8
    noise_scale: float = 0.15
    seed: int = 0


@dataclass(frozen=True)
class AttentionConfig:
    """Configuration for synthesized attention matrices.

    Attributes
    ----------
    seq_len
        Sequence length ``T``.
    n_heads
        Number of attention heads ``H`` to synthesize.
    induction_weight, sink_weight, uniform_weight
        Convex mixture coefficients (non-negative, summing to one) controlling
        the proportion of induction, attention-sink, and uniform heads.
    sink_position
        Token index that attention-sink heads collapse onto (default 0).
    seed
        RNG seed.
    """

    seq_len: int = 64
    n_heads: int = 12
    induction_weight: float = 0.4
    sink_weight: float = 0.3
    uniform_weight: float = 0.3
    sink_position: int = 0
    seed: int = 0


def synth_anisotropic_hidden_states(cfg: AnisotropicConfig) -> torch.Tensor:
    r"""Generate cone-collapsed hidden states with a target anisotropy index.

    The construction concatenates a shared cone direction :math:`u \in
    \mathbb{R}^d` with cluster-specific scatter:

    .. math::

        h_i \;=\; \alpha\,u \;+\; \beta\,c_{k(i)} \;+\; \sigma\,\epsilon_i

    where :math:`\alpha` and :math:`\beta` are derived from the target
    anisotropy :math:`\mathcal{A}` through the closed-form relation
    :math:`\mathcal{A} \approx \alpha^2 / (\alpha^2 + \beta^2 + \sigma^2 d /
    \|h\|^2)`.

    Parameters
    ----------
    cfg
        Generator configuration.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(N, d)`` with dtype ``float32``.

    Raises
    ------
    GeometryError
        If ``cfg.anisotropy`` is not in ``(0, 1)`` or dimensions are
        non-positive.
    """
    if not 0.0 < cfg.anisotropy < 1.0:
        raise GeometryError(
            f"anisotropy must lie in (0, 1); got {cfg.anisotropy}"
        )
    if cfg.n_tokens <= 0 or cfg.dim <= 0 or cfg.n_clusters <= 0:
        raise GeometryError("n_tokens, dim, n_clusters must all be positive")

    g = torch.Generator().manual_seed(cfg.seed)

    # Solve for cone amplitude alpha given target anisotropy A:
    #   A = alpha^2 / (alpha^2 + 1)  ⇒  alpha = sqrt(A / (1 - A))
    alpha = float(np.sqrt(cfg.anisotropy / (1.0 - cfg.anisotropy)))

    cone_axis = torch.randn(cfg.dim, generator=g)
    cone_axis = cone_axis / cone_axis.norm()

    cluster_axes = torch.randn(cfg.n_clusters, cfg.dim, generator=g)
    cluster_axes = cluster_axes / cluster_axes.norm(dim=1, keepdim=True)

    cluster_assignments = torch.randint(
        0, cfg.n_clusters, (cfg.n_tokens,), generator=g
    )
    cluster_component = cluster_axes[cluster_assignments]

    noise = torch.randn(cfg.n_tokens, cfg.dim, generator=g) * cfg.noise_scale

    hidden = alpha * cone_axis.unsqueeze(0) + cluster_component + noise
    _log.debug(
        "Synthesized anisotropic hidden states: shape=%s, alpha=%.3f",
        tuple(hidden.shape),
        alpha,
    )
    return hidden.to(torch.float32)


def synth_attention_matrix(cfg: AttentionConfig) -> torch.Tensor:
    r"""Generate per-head attention matrices with mixed behavioral archetypes.

    Each head is drawn from one of three behavioral classes:

    * **Induction**: attention is concentrated on a 'previous-token' diagonal
      shifted by 1, emulating induction circuits that copy from
      ``t - offset``.
    * **Attention-sink**: a row collapses near-entirely onto ``sink_position``
      (typically the BOS token).
    * **Uniform**: a near-flat distribution over the sequence.

    Parameters
    ----------
    cfg
        Generator configuration.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(H, T, T)`` of row-stochastic attention matrices.

    Raises
    ------
    MechanisticError
        If mixture weights are not non-negative and finite.
    """
    weights = np.array(
        [cfg.induction_weight, cfg.sink_weight, cfg.uniform_weight], dtype=np.float64
    )
    if np.any(weights < 0) or not np.isfinite(weights).all():
        raise MechanisticError(f"mixture weights must be non-negative finite; got {weights}")
    total = weights.sum()
    if total <= 0.0:
        raise MechanisticError("at least one mixture weight must be positive")
    weights = weights / total

    g = torch.Generator().manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    archetype_ids = rng.choice(3, size=cfg.n_heads, p=weights)

    T = cfg.seq_len
    out = torch.zeros(cfg.n_heads, T, T)
    for h in range(cfg.n_heads):
        kind = int(archetype_ids[h])
        if kind == 0:  # induction
            logits = -10.0 * torch.ones(T, T)
            for t in range(T):
                if t - 1 >= 0:
                    logits[t, t - 1] = 5.0
            logits = logits + 0.5 * torch.randn(T, T, generator=g)
        elif kind == 1:  # attention sink
            logits = -2.0 * torch.ones(T, T) + 0.5 * torch.randn(T, T, generator=g)
            logits[:, cfg.sink_position] = 8.0
        else:  # uniform-ish
            logits = 0.1 * torch.randn(T, T, generator=g)
        # causal mask
        causal = torch.tril(torch.ones(T, T)).bool()
        logits = logits.masked_fill(~causal, float("-inf"))
        out[h] = torch.softmax(logits, dim=-1)
    _log.debug(
        "Synthesized attention: shape=%s, archetypes(induction,sink,uniform)=%s",
        tuple(out.shape),
        tuple(int((archetype_ids == k).sum()) for k in range(3)),
    )
    return out


def synth_residual_stream(
    n_layers: int = 6,
    n_heads: int = 8,
    seq_len: int = 32,
    d_model: int = 128,
    seed: int = 0,
) -> torch.Tensor:
    """Generate a layered residual stream with per-head low-rank writes.

    The returned tensor has shape ``(L, H, T, d_model)`` containing the
    per-layer, per-head contribution to the residual stream. The norms of
    these contributions decay geometrically across layers (typical of
    well-trained transformers) with small per-head heterogeneity.

    Parameters
    ----------
    n_layers, n_heads, seq_len, d_model
        Tensor shape parameters.
    seed
        RNG seed.

    Returns
    -------
    torch.Tensor
        Synthetic per-head residual-stream contributions.
    """
    if min(n_layers, n_heads, seq_len, d_model) <= 0:
        raise MechanisticError("all dimensions must be positive")
    g = torch.Generator().manual_seed(seed)
    contributions = torch.zeros(n_layers, n_heads, seq_len, d_model)
    for layer in range(n_layers):
        # Layer-wise norm decay
        layer_scale = 1.0 / (1.0 + 0.3 * layer)
        for head in range(n_heads):
            head_scale = 0.5 + torch.rand(1, generator=g).item()
            # Low-rank write: head projects through rank-2 subspace
            U = torch.randn(seq_len, 2, generator=g)
            V = torch.randn(2, d_model, generator=g)
            contributions[layer, head] = layer_scale * head_scale * (U @ V)
    return contributions


def synth_loss_landscape_weights(
    n_steps: int = 50,
    n_params: int = 256,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthesize a weight trajectory with an associated loss curve.

    Models an Adam-like descent toward a noisy basin with an Edge of Stability
    plateau in the final third of training, characterized by oscillating loss.

    Parameters
    ----------
    n_steps
        Number of optimization snapshots.
    n_params
        Dimensionality of the flattened weight vector.
    seed
        RNG seed.

    Returns
    -------
    tuple
        ``(weights, losses)`` where ``weights`` has shape ``(n_steps,
        n_params)`` and ``losses`` has shape ``(n_steps,)``.
    """
    g = torch.Generator().manual_seed(seed)
    target = torch.randn(n_params, generator=g) * 0.3
    weights = torch.zeros(n_steps, n_params)
    losses = torch.zeros(n_steps)
    w = torch.randn(n_params, generator=g) * 0.8
    lr = 0.05
    for t in range(n_steps):
        # Underlying quadratic loss + curvature increase near the end
        residual = w - target
        loss_smooth = 0.5 * (residual * residual).sum().item()
        # Edge-of-Stability oscillations in the final third
        if t > 2 * n_steps // 3:
            loss = loss_smooth + 0.08 * float(np.sin(t * 1.2))
            grad = residual + 0.6 * torch.sin(torch.tensor(t * 1.2)) * torch.randn(
                n_params, generator=g
            )
        else:
            loss = loss_smooth + 0.005 * float(np.random.RandomState(seed + t).randn())
            grad = residual + 0.05 * torch.randn(n_params, generator=g)
        w = w - lr * grad
        weights[t] = w
        losses[t] = loss
    return weights, losses


def synth_polysemantic_activations(
    n_samples: int = 4096,
    d_model: int = 64,
    n_features: int = 256,
    sparsity: float = 0.05,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthesize activations that exhibit feature superposition.

    We sample sparse latent feature codes :math:`z \\in \\mathbb{R}^{F}`
    (Bernoulli-gated standard normals at density ``sparsity``) and project
    them through a random Gaussian dictionary :math:`D \\in \\mathbb{R}^{d
    \\times F}` with :math:`F \\gg d`:

    .. math::

        x \\;=\\; D z \\;+\\; \\sigma\\, \\eta

    The returned ``(activations, true_codes)`` tuple lets a Sparse Autoencoder
    be evaluated on recovery of the planted feature codes.

    Parameters
    ----------
    n_samples
        Number of activation samples.
    d_model
        Activation dimensionality (small relative to ``n_features`` to force
        superposition).
    n_features
        Number of planted features ``F``; should exceed ``d_model``.
    sparsity
        Bernoulli activation probability of any single feature.
    seed
        RNG seed.

    Returns
    -------
    tuple
        ``(activations, codes)`` with shapes ``(n_samples, d_model)`` and
        ``(n_samples, n_features)`` respectively.
    """
    if n_features <= d_model:
        raise MechanisticError(
            f"n_features ({n_features}) must exceed d_model ({d_model}) "
            "for superposition to be non-trivial"
        )
    g = torch.Generator().manual_seed(seed)
    dictionary = torch.randn(d_model, n_features, generator=g)
    dictionary = dictionary / dictionary.norm(dim=0, keepdim=True)

    mask = (torch.rand(n_samples, n_features, generator=g) < sparsity).float()
    magnitudes = torch.relu(torch.randn(n_samples, n_features, generator=g))
    codes = mask * magnitudes

    activations = codes @ dictionary.T
    activations = activations + 0.02 * torch.randn(n_samples, d_model, generator=g)
    return activations, codes
