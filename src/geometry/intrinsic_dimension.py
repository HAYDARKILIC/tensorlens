r"""Intrinsic dimension estimators.

**TwoNN** (Facco et al., 2017): Given the ratio :math:`\mu_i = r_{i,2} /
r_{i,1}` of the second- to first-nearest-neighbor distances, the cumulative
distribution of :math:`\mu` follows :math:`F(\mu) = 1 - \mu^{-d}` under a
locally uniform density assumption. The intrinsic dimension :math:`d` is
estimated by a linear fit of :math:`-\log(1 - F(\mu))` vs. :math:`\log \mu`.

**MLE** (Levina & Bickel, 2004): For each point :math:`i`, given distances to
its :math:`k` nearest neighbors :math:`r_{i,j}`,

.. math::

    \hat{d}_i \;=\; \Bigl[\frac{1}{k-1} \sum_{j=1}^{k-1}
            \log \frac{r_{i,k}}{r_{i,j}}\Bigr]^{-1}

and the global estimate :math:`\hat{d} = \langle \hat{d}_i \rangle_i`.

References
----------
* Facco, E., d'Errico, M., Rodriguez, A., Laio, A. (2017). "Estimating the
  intrinsic dimension of datasets by a minimal neighborhood information."
  Scientific Reports.
* Levina, E., Bickel, P. J. (2004). "Maximum Likelihood Estimation of
  Intrinsic Dimension." NeurIPS.
"""

from __future__ import annotations

import numpy as np
import torch

from ..utils import GeometryError, get_logger

__all__ = ["twonn_intrinsic_dimension", "mle_intrinsic_dimension"]

_log = get_logger(__name__)


def _pairwise_distances(x: torch.Tensor) -> torch.Tensor:
    """Return pairwise Euclidean distances with zeroed diagonal."""
    sq = (x * x).sum(dim=1, keepdim=True)
    d2 = (sq + sq.T - 2.0 * (x @ x.T)).clamp_min(0.0)
    return d2.sqrt()


def twonn_intrinsic_dimension(
    x: torch.Tensor,
    *,
    discard_fraction: float = 0.1,
) -> float:
    r"""TwoNN intrinsic-dimension estimator.

    Parameters
    ----------
    x
        Tensor of shape ``(N, d)``.
    discard_fraction
        Tail fraction of the :math:`\mu` distribution to discard before
        linear fit to mitigate density inhomogeneity.

    Returns
    -------
    float
        Estimated intrinsic dimension.

    Raises
    ------
    GeometryError
        On invalid input.
    """
    if x.dim() != 2:
        raise GeometryError(f"x must be 2-D; got {tuple(x.shape)}")
    n = x.shape[0]
    if n < 8:
        raise GeometryError("TwoNN requires N >= 8")
    if not 0.0 <= discard_fraction < 0.5:
        raise GeometryError(f"discard_fraction must lie in [0, 0.5); got {discard_fraction}")

    D = _pairwise_distances(x.detach().to(torch.float64))
    # Set self-distances to +inf so they don't appear in topk
    D.fill_diagonal_(float("inf"))
    r1, _ = D.topk(2, dim=1, largest=False)
    r1_first = r1[:, 0]
    r1_second = r1[:, 1]
    # Filter degenerate zero-distance pairs
    valid = (r1_first > 0) & (r1_second > 0) & torch.isfinite(r1_first) & torch.isfinite(r1_second)
    mu = (r1_second[valid] / r1_first[valid]).cpu().numpy()
    mu_sorted = np.sort(mu)
    n_eff = mu_sorted.size
    # Empirical CDF (Hazen)
    F = (np.arange(1, n_eff + 1) - 0.5) / n_eff
    # Discard top tail
    keep = int(n_eff * (1.0 - discard_fraction))
    mu_fit = mu_sorted[:keep]
    F_fit = F[:keep]
    log_mu = np.log(mu_fit)
    log_one_minus_F = -np.log1p(-F_fit)
    # Least-squares slope through origin (since F(1) = 0)
    slope = float(np.sum(log_mu * log_one_minus_F) / np.sum(log_mu * log_mu + 1e-30))
    _log.debug("TwoNN: N_eff=%d, d_hat=%.3f", n_eff, slope)
    return slope


def mle_intrinsic_dimension(
    x: torch.Tensor,
    *,
    k: int = 10,
) -> float:
    r"""Maximum-likelihood intrinsic dimension (Levina & Bickel, 2004).

    Parameters
    ----------
    x
        Tensor of shape ``(N, d)``.
    k
        Number of nearest neighbors used in the local estimate
        (typically 5–20).

    Returns
    -------
    float
        Estimated intrinsic dimension.
    """
    if x.dim() != 2:
        raise GeometryError(f"x must be 2-D; got {tuple(x.shape)}")
    n = x.shape[0]
    if k < 2 or k >= n:
        raise GeometryError(f"k must lie in [2, {n - 1}]; got {k}")

    D = _pairwise_distances(x.detach().to(torch.float64))
    D.fill_diagonal_(float("inf"))
    knn, _ = D.topk(k, dim=1, largest=False)
    knn_np = knn.cpu().numpy()
    # Avoid log of zero
    r_k = knn_np[:, -1:].clip(min=1e-30)
    r_j = knn_np[:, :-1].clip(min=1e-30)
    ratios = np.log(r_k / r_j)  # (N, k-1)
    inv_d = ratios.mean(axis=1)
    valid = inv_d > 0
    d_local = 1.0 / inv_d[valid]
    return float(np.mean(d_local))
