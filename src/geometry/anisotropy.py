r"""Empirical anisotropy & spectral diagnostics for LLM hidden states.

Definitions
-----------

**Anisotropy index** (Ethayarajh, 2019):

.. math::

    \mathcal{A}(\mathcal{H}) \;=\; \frac{1}{N(N-1)} \sum_{i \neq j}
    \frac{\langle h_i, h_j \rangle}{\|h_i\|_2 \, \|h_j\|_2}

For an isotropic point cloud in :math:`\mathbb{R}^d`, :math:`\mathcal{A} \to
0`. Contextual transformer embeddings exhibit :math:`\mathcal{A} > 0.5` —
quantitative evidence of cone collapse.

**Effective rank** (Roy & Vetterli, 2007):

.. math::

    r_{\mathrm{eff}}(H) \;=\; \exp\!\Bigl(-\sum_{k=1}^{d} p_k \log p_k\Bigr),
    \qquad p_k = \frac{\sigma_k^2}{\sum_j \sigma_j^2}

where :math:`\sigma_k` are the singular values of the centered matrix
:math:`H - \bar{h}\mathbf{1}^\top`.

**Explained-variance Gini coefficient** measures concentration of singular
energy. For a balanced spectrum :math:`G \to 0`; for a degenerate rank-1
spectrum :math:`G \to 1`.

References
----------
* Ethayarajh, K. (2019). "How Contextual are Contextualized Word
  Representations?" EMNLP.
* Roy, O., Vetterli, M. (2007). "The effective rank: A measure of effective
  dimensionality." EUSIPCO.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..utils import GeometryError, get_logger

__all__ = [
    "AnisotropyReport",
    "compute_anisotropy",
    "cosine_similarity_distribution",
    "explained_variance_gini",
    "svd_effective_rank",
    "svd_spectrum",
]

_log = get_logger(__name__)


@dataclass(frozen=True)
class AnisotropyReport:
    """Full geometric report for a batch of hidden states.

    Attributes
    ----------
    anisotropy
        Mean pairwise cosine similarity.
    effective_rank
        Entropy-based effective rank of the centered spectrum.
    gini
        Gini concentration of squared singular values.
    singular_values
        Sorted singular values (descending).
    n_tokens, dim
        Shape diagnostics.
    """

    anisotropy: float
    effective_rank: float
    gini: float
    singular_values: np.ndarray
    n_tokens: int
    dim: int

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"AnisotropyReport(N={self.n_tokens}, d={self.dim}, "
            f"anisotropy={self.anisotropy:.4f}, "
            f"r_eff={self.effective_rank:.2f}, "
            f"gini={self.gini:.4f})"
        )


def _validate_2d(name: str, x: torch.Tensor) -> None:
    """Validate that ``x`` is a 2-D, finite tensor with positive shape.

    Parameters
    ----------
    name
        Argument name for error messages.
    x
        Tensor to validate.

    Raises
    ------
    GeometryError
        If validation fails.
    """
    if x.dim() != 2:
        raise GeometryError(f"{name} must be 2-D (N, d); got shape {tuple(x.shape)}")
    if x.shape[0] < 2 or x.shape[1] < 2:
        raise GeometryError(f"{name} requires N >= 2 and d >= 2; got {tuple(x.shape)}")
    if not torch.isfinite(x).all():
        raise GeometryError(f"{name} contains non-finite values (NaN/Inf)")


def compute_anisotropy(
    hidden_states: torch.Tensor,
    *,
    sample_size: int | None = None,
    seed: int = 0,
    eps: float = 1e-12,
) -> float:
    r"""Compute the mean pairwise cosine similarity of hidden states.

    Implements

    .. math::

        \mathcal{A}(\mathcal{H}) = \frac{1}{N(N-1)} \sum_{i \neq j}
        \frac{\langle h_i, h_j \rangle}{\|h_i\|_2 \|h_j\|_2}

    in an :math:`O(N^2)` matrix-multiply formulation. For very large ``N`` a
    uniform random subsample of size ``sample_size`` is used.

    Parameters
    ----------
    hidden_states
        Tensor of shape ``(N, d)``.
    sample_size
        Optional cap on the number of vectors sampled to keep the pairwise
        product feasible in memory. If ``None``, the full set is used.
    seed
        RNG seed for the subsample.
    eps
        Small constant added to the L2 norm denominator to avoid division by
        zero on near-zero vectors.

    Returns
    -------
    float
        Anisotropy value in ``[-1, 1]`` (typically ``> 0`` for transformer
        hidden states).

    Raises
    ------
    GeometryError
        If ``hidden_states`` is not 2-D or contains non-finite values.
    """
    _validate_2d("hidden_states", hidden_states)
    n = hidden_states.shape[0]
    h = hidden_states
    if sample_size is not None and sample_size < n:
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(n, generator=g)[:sample_size]
        h = hidden_states[idx]
        n = sample_size

    norms = h.norm(dim=1, keepdim=True).clamp_min(eps)
    h_unit = h / norms
    # Pairwise cosine; subtract diagonal (self-similarity = 1).
    gram = h_unit @ h_unit.T  # (n, n)
    sum_offdiag = gram.sum().item() - torch.diagonal(gram).sum().item()
    denom = float(n * (n - 1))
    aniso = sum_offdiag / denom
    _log.debug("Anisotropy over %d vectors: %.6f", n, aniso)
    return aniso


def cosine_similarity_distribution(
    hidden_states: torch.Tensor,
    *,
    sample_pairs: int = 50_000,
    seed: int = 0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Sample the empirical pairwise cosine similarity distribution.

    For visualization (histogram, KDE) and statistical testing the full
    :math:`N(N-1)/2` pairwise distribution is rarely needed; instead we draw
    ``sample_pairs`` uniform random pairs.

    Parameters
    ----------
    hidden_states
        Tensor of shape ``(N, d)``.
    sample_pairs
        Number of unordered pairs to sample.
    seed
        RNG seed.
    eps
        Small constant for the norm denominator.

    Returns
    -------
    numpy.ndarray
        1-D array of length ``sample_pairs`` containing cosine similarities.

    Raises
    ------
    GeometryError
        If shape is invalid or ``sample_pairs <= 0``.
    """
    _validate_2d("hidden_states", hidden_states)
    if sample_pairs <= 0:
        raise GeometryError(f"sample_pairs must be positive; got {sample_pairs}")
    n = hidden_states.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    i_idx = torch.randint(0, n, (sample_pairs,), generator=g)
    j_idx = torch.randint(0, n, (sample_pairs,), generator=g)
    # Reject self-pairs.
    mask = i_idx != j_idx
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]

    h = hidden_states
    a = h[i_idx]
    b = h[j_idx]
    cos = (a * b).sum(dim=1) / (
        a.norm(dim=1).clamp_min(eps) * b.norm(dim=1).clamp_min(eps)
    )
    return cos.detach().cpu().numpy()


def svd_spectrum(
    hidden_states: torch.Tensor,
    *,
    center: bool = True,
    full_matrices: bool = False,
) -> np.ndarray:
    """Return the singular values of (optionally centered) hidden states.

    Parameters
    ----------
    hidden_states
        Tensor of shape ``(N, d)``.
    center
        Whether to subtract the per-feature mean before decomposition. Should
        be True for anisotropy analysis: an off-center cone artificially
        inflates the leading singular value.
    full_matrices
        Forwarded to :func:`torch.linalg.svd`.

    Returns
    -------
    numpy.ndarray
        Sorted singular values (descending), length ``min(N, d)``.

    Raises
    ------
    GeometryError
        On invalid shape or non-finite input.
    """
    _validate_2d("hidden_states", hidden_states)
    x = hidden_states.detach().to(torch.float64)
    if center:
        x = x - x.mean(dim=0, keepdim=True)
    try:
        _, s, _ = torch.linalg.svd(x, full_matrices=full_matrices)
    except RuntimeError as exc:  # pragma: no cover - depends on backend
        raise GeometryError(f"SVD failed: {exc}") from exc
    return s.detach().cpu().numpy()


def svd_effective_rank(hidden_states: torch.Tensor, *, center: bool = True) -> float:
    r"""Entropy-based effective rank of the centered hidden states.

    .. math::

        r_{\mathrm{eff}} = \exp\!\Bigl(-\sum_k p_k \log p_k\Bigr),
        \qquad p_k = \sigma_k^2 / \sum_j \sigma_j^2

    Parameters
    ----------
    hidden_states
        Tensor of shape ``(N, d)``.
    center
        Center before decomposition (recommended).

    Returns
    -------
    float
        Effective rank in ``(0, min(N, d)]``.
    """
    s = svd_spectrum(hidden_states, center=center)
    energy = s**2
    total = energy.sum()
    if total <= 0.0:
        raise GeometryError("zero spectral energy; representations are identically zero")
    p = energy / total
    # Use only nonzero entries to avoid 0*log(0) NaN.
    p_pos = p[p > 0]
    entropy = float(-np.sum(p_pos * np.log(p_pos)))
    return float(np.exp(entropy))


def explained_variance_gini(hidden_states: torch.Tensor, *, center: bool = True) -> float:
    r"""Gini concentration of explained variance.

    For sorted :math:`p_1 \geq p_2 \geq \cdots \geq p_d` with :math:`\sum p_k
    = 1`, the Gini coefficient is

    .. math::

        G = \frac{2 \sum_{k=1}^{d} k\,p_k}{d \sum_{k=1}^{d} p_k} - \frac{d+1}{d}

    A uniform spectrum yields :math:`G = 0`; a rank-1 spectrum yields
    :math:`G \to (d - 1) / d \approx 1`.

    Parameters
    ----------
    hidden_states
        Tensor of shape ``(N, d)``.
    center
        Center the data before decomposition.

    Returns
    -------
    float
        Gini coefficient in ``[0, 1)``.
    """
    s = svd_spectrum(hidden_states, center=center)
    p = (s**2) / max(float((s**2).sum()), 1e-30)
    p_sorted = np.sort(p)[::-1]
    d = p_sorted.size
    k = np.arange(1, d + 1, dtype=np.float64)
    numerator = 2.0 * float(np.sum(k * p_sorted))
    denominator = d * float(np.sum(p_sorted))
    gini = numerator / denominator - (d + 1.0) / d
    return float(np.clip(gini, 0.0, 1.0))


def full_report(
    hidden_states: torch.Tensor,
    *,
    sample_size: int | None = 1024,
    seed: int = 0,
) -> AnisotropyReport:
    """Compute a full :class:`AnisotropyReport` in one call.

    Parameters
    ----------
    hidden_states
        Tensor of shape ``(N, d)``.
    sample_size
        Cap used for the anisotropy computation only; the SVD spectrum is
        computed on the full tensor.
    seed
        RNG seed for subsampling.

    Returns
    -------
    AnisotropyReport
        Aggregated diagnostics.
    """
    _validate_2d("hidden_states", hidden_states)
    aniso = compute_anisotropy(hidden_states, sample_size=sample_size, seed=seed)
    s = svd_spectrum(hidden_states, center=True)
    r_eff = svd_effective_rank(hidden_states, center=True)
    gini = explained_variance_gini(hidden_states, center=True)
    return AnisotropyReport(
        anisotropy=aniso,
        effective_rank=r_eff,
        gini=gini,
        singular_values=s,
        n_tokens=int(hidden_states.shape[0]),
        dim=int(hidden_states.shape[1]),
    )
