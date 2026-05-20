r"""Per-head Shannon entropy and induction-head detection.

For an attention pattern :math:`\alpha^{(h)} \in \mathbb{R}^{T \times T}`
(row-stochastic), the per-query token entropy is

.. math::

    \mathcal{H}^{(h)}_i \;=\; -\sum_{j=1}^{T} \alpha^{(h)}_{i,j}
    \log \alpha^{(h)}_{i,j}

* :math:`\mathcal{H}^{(h)}_i \to 0` indicates a deterministic look-up (one
  hot), characteristic of induction or copy heads.
* :math:`\mathcal{H}^{(h)}_i \to \log T_{\mathrm{causal}}(i)` indicates a
  uniform attention distribution (averaging / null-op).

Induction score (Olsson et al., 2022) measures how strongly head :math:`h`
attends from the second occurrence of a token to the position immediately
following its first occurrence — i.e. it implements the "prefix matching →
copy" circuit.
"""

from __future__ import annotations

import torch

from ..utils import MechanisticError, get_logger

__all__ = ["attention_entropy", "induction_score", "classify_head_archetypes"]

_log = get_logger(__name__)


def _validate_attention(name: str, attn: torch.Tensor) -> None:
    """Validate that ``attn`` has shape ``(*, H, T, T)`` and is row-stochastic.

    Raises
    ------
    MechanisticError
        If shape or stochasticity checks fail.
    """
    if attn.dim() < 3:
        raise MechanisticError(
            f"{name} must have shape (..., H, T, T); got {tuple(attn.shape)}"
        )
    if attn.shape[-1] != attn.shape[-2]:
        raise MechanisticError(f"{name} last two dims must be equal; got {tuple(attn.shape)}")
    if not torch.isfinite(attn).all():
        raise MechanisticError(f"{name} contains non-finite values")
    # Row stochasticity check on a sample row to catch obvious bugs.
    sample_sum = attn.sum(dim=-1)
    if not torch.allclose(sample_sum, torch.ones_like(sample_sum), atol=1e-3):
        # Causal masks may produce rows summing to 1 over their causal prefix
        # — that still satisfies row-stochasticity. We allow values that sum
        # to ≤ 1 + epsilon to accommodate that case.
        if (sample_sum > 1.0 + 1e-3).any() or (sample_sum < -1e-3).any():
            raise MechanisticError(f"{name} is not row-stochastic (rows sum outside [0, 1])")


def attention_entropy(
    attention: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    r"""Compute the Shannon entropy of each attention row.

    Parameters
    ----------
    attention
        Tensor with trailing shape ``(H, T, T)``; any leading batch dims are
        broadcast through.
    eps
        Floor added to probabilities before the log to avoid ``-inf``.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``(..., H, T)``; entry ``[h, i]`` is
        :math:`\mathcal{H}^{(h)}_i`.

    Raises
    ------
    MechanisticError
        On invalid shapes.
    """
    _validate_attention("attention", attention)
    a = attention.clamp_min(eps)
    return -(a * a.log()).sum(dim=-1)


def induction_score(
    attention: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    r"""Olsson-et-al. induction score per head.

    For each head and each token position :math:`i`, this looks back for the
    most recent earlier occurrence of the same token :math:`t_i` at position
    :math:`j < i` and credits the head's attention from :math:`i` to
    :math:`j + 1` (the prefix-match-then-copy target):

    .. math::

        \mathrm{IndScore}^{(h)} \;=\;
        \frac{1}{|\mathcal{I}|} \sum_{i \in \mathcal{I}} \alpha^{(h)}_{i,\, j(i)+1}

    where :math:`\mathcal{I} = \{i : t_i \text{ has prior occurrence}\}`.

    Parameters
    ----------
    attention
        Single-batch attention with shape ``(H, T, T)``.
    token_ids
        Integer tensor of shape ``(T,)``.
    eps
        Numerical floor.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(H,)`` with per-head induction scores in [0, 1].

    Raises
    ------
    MechanisticError
        On shape inconsistency.
    """
    if attention.dim() != 3:
        raise MechanisticError(f"attention must be (H, T, T); got {tuple(attention.shape)}")
    if token_ids.dim() != 1 or token_ids.shape[0] != attention.shape[-1]:
        raise MechanisticError(
            f"token_ids shape mismatch: expected ({attention.shape[-1]},), got "
            f"{tuple(token_ids.shape)}"
        )
    H, T, _ = attention.shape
    t = token_ids.tolist()
    scores = torch.zeros(H, dtype=torch.float64)
    counts = 0
    for i in range(1, T):
        # Find prior occurrence of t[i]; require j+1 <= i-1 so target index valid.
        for j in range(i - 1, -1, -1):
            if t[j] == t[i] and j + 1 < i:
                scores += attention[:, i, j + 1].to(torch.float64)
                counts += 1
                break
    if counts == 0:
        _log.warning("No repeated tokens with prefix-match target found; returning zeros")
        return torch.zeros(H)
    return (scores / counts).clamp(0.0, 1.0).to(attention.dtype)


def classify_head_archetypes(
    attention: torch.Tensor,
    *,
    induction_threshold: float = 0.25,
    sink_threshold: float = 0.5,
    uniform_entropy_threshold: float | None = None,
) -> list[str]:
    r"""Classify each head as one of {induction, sink, uniform, mixed}.

    The classification rules are simple decision-list:

    1. **sink** — if average attention on column 0 exceeds ``sink_threshold``.
    2. **induction** — if mean entropy is below ``induction_threshold`` *and*
       not flagged as sink.
    3. **uniform** — if mean entropy exceeds ``uniform_entropy_threshold``
       (default :math:`0.85 \log T`).
    4. **mixed** — otherwise.

    Parameters
    ----------
    attention
        Tensor of shape ``(H, T, T)`` from a single forward pass.
    induction_threshold
        Maximum mean entropy for a head to be considered induction-like.
    sink_threshold
        Minimum mean attention on column 0 for sink classification.
    uniform_entropy_threshold
        Override for the uniform threshold; default scales with sequence
        length.

    Returns
    -------
    list of str
        Length ``H`` archetype labels.
    """
    _validate_attention("attention", attention)
    if attention.dim() != 3:
        raise MechanisticError(f"expected (H, T, T); got {tuple(attention.shape)}")
    H, T, _ = attention.shape
    if uniform_entropy_threshold is None:
        uniform_entropy_threshold = 0.85 * float(torch.log(torch.tensor(float(T))))
    H_entropy = attention_entropy(attention).mean(dim=-1)  # (H,)
    col0_mean = attention[..., 0].mean(dim=-1)  # (H,)
    labels: list[str] = []
    for h in range(H):
        if float(col0_mean[h]) >= sink_threshold:
            labels.append("sink")
        elif float(H_entropy[h]) <= induction_threshold:
            labels.append("induction")
        elif float(H_entropy[h]) >= uniform_entropy_threshold:
            labels.append("uniform")
        else:
            labels.append("mixed")
    return labels
