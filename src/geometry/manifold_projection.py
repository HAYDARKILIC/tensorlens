r"""From-scratch nonlinear manifold projections.

This module implements t-SNE and a simplified UMAP entirely from PyTorch
primitives — no ``sklearn``, no ``umap-learn``. Both algorithms minimize a
divergence between a high-dimensional similarity matrix :math:`P` and a
low-dimensional similarity matrix :math:`Q`.

**t-SNE** (van der Maaten & Hinton, 2008):

.. math::

    p_{j|i} \;=\; \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}
                       {\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)},
    \qquad p_{ij} = \tfrac{p_{j|i} + p_{i|j}}{2N}

.. math::

    q_{ij} \;=\; \frac{(1 + \|y_i - y_j\|^2)^{-1}}
                      {\sum_{k \neq l}(1 + \|y_k - y_l\|^2)^{-1}}

.. math::

    \mathcal{L}_{\mathrm{tSNE}} = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}

Bandwidths :math:`\sigma_i` are calibrated per-point via binary search to
hit a target perplexity :math:`\mathrm{Perp}(P_i) = 2^{H(P_i)} = u`.

**UMAP** (McInnes et al., 2018) is implemented in its simplified form
optimizing the fuzzy-simplicial-set cross-entropy:

.. math::

    \mathcal{L}_{\mathrm{UMAP}} = \sum_{i \neq j} w_{ij} \log\frac{w_{ij}}{v_{ij}}
        + (1 - w_{ij}) \log\frac{1 - w_{ij}}{1 - v_{ij}}

with low-dimensional weights

.. math::

    v_{ij} = \bigl(1 + a\,\|y_i - y_j\|^{2b}\bigr)^{-1}

and high-dim weights derived from per-point local connectivity :math:`\rho_i`
and scale :math:`\sigma_i`.
"""

from __future__ import annotations

import numpy as np
import torch

from ..utils import GeometryError, get_logger

__all__ = ["tsne_from_scratch", "umap_from_scratch"]

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pairwise distances
# ---------------------------------------------------------------------------


def _pairwise_sq_dists(x: torch.Tensor) -> torch.Tensor:
    """Compute squared Euclidean pairwise distances.

    Parameters
    ----------
    x
        Tensor of shape ``(N, d)``.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(N, N)`` with diagonals set to zero.
    """
    sq_norms = (x * x).sum(dim=1, keepdim=True)
    d2 = sq_norms + sq_norms.T - 2.0 * (x @ x.T)
    return d2.clamp_min(0.0)


# ---------------------------------------------------------------------------
# Perplexity calibration for t-SNE
# ---------------------------------------------------------------------------


def _binary_search_sigma(
    sq_dists_i: torch.Tensor,
    target_perplexity: float,
    tol: float = 1e-5,
    max_iter: int = 50,
) -> torch.Tensor:
    r"""Binary-search per-point precision :math:`\beta_i = 1/(2\sigma_i^2)`.

    Solves :math:`H(P_i) = \log u` where :math:`u` is the target perplexity.

    Parameters
    ----------
    sq_dists_i
        Squared distances from point ``i`` to all other points, shape
        ``(N,)`` with ``sq_dists_i[i] = 0``.
    target_perplexity
        Desired effective number of neighbors.
    tol
        Convergence tolerance on entropy.
    max_iter
        Hard cap on bisection iterations.

    Returns
    -------
    torch.Tensor
        Length-``N`` conditional probability row :math:`p_{j|i}`.
    """
    log_target = float(np.log(target_perplexity))
    beta_lo = torch.tensor(1e-20)
    beta_hi = torch.tensor(1e20)
    beta = torch.tensor(1.0)
    p_row = torch.zeros_like(sq_dists_i)
    for _ in range(max_iter):
        logits = -sq_dists_i * beta
        # Numerical stabilization
        logits = logits - logits.max()
        unnorm = torch.exp(logits)
        unnorm[sq_dists_i == 0] = 0.0  # exclude self
        Z = unnorm.sum()
        if float(Z) <= 0.0:
            beta = beta * 0.5
            continue
        p_row = unnorm / Z
        # Shannon entropy of p_row using only nonzero entries
        nz = p_row > 0
        H = -float((p_row[nz] * torch.log(p_row[nz])).sum())
        diff = H - log_target
        if abs(diff) < tol:
            break
        if diff > 0:  # entropy too high → increase beta
            beta_lo = beta.clone()
            beta = (beta + beta_hi) / 2.0 if beta_hi.item() < 1e19 else beta * 2.0
        else:
            beta_hi = beta.clone()
            beta = (beta + beta_lo) / 2.0
    return p_row


def _compute_P_tsne(
    x: torch.Tensor, perplexity: float
) -> torch.Tensor:
    """Build the symmetric joint affinity matrix :math:`P` for t-SNE.

    Parameters
    ----------
    x
        Input data of shape ``(N, d)``.
    perplexity
        Effective neighborhood size; typical values 5–50.

    Returns
    -------
    torch.Tensor
        :math:`(N, N)` joint probability matrix with rows summing to 1.
    """
    n = x.shape[0]
    D2 = _pairwise_sq_dists(x)
    P = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        P[i] = _binary_search_sigma(D2[i].to(torch.float64), perplexity)
    # Symmetrize and normalize
    P = (P + P.T) / (2.0 * n)
    P = torch.clamp(P, min=1e-12)
    return P


# ---------------------------------------------------------------------------
# t-SNE
# ---------------------------------------------------------------------------


def tsne_from_scratch(
    x: torch.Tensor,
    *,
    n_components: int = 2,
    perplexity: float = 30.0,
    n_iter: int = 500,
    learning_rate: float = 200.0,
    early_exaggeration: float = 12.0,
    early_exaggeration_iters: int = 100,
    momentum_initial: float = 0.5,
    momentum_final: float = 0.8,
    seed: int = 0,
) -> torch.Tensor:
    """Pure-PyTorch t-SNE.

    Minimizes :math:`\\mathrm{KL}(P\\,\\|\\,Q)` via momentum gradient descent
    with early exaggeration. Implementation follows van der Maaten & Hinton
    (2008) with the now-standard improvements in van der Maaten (2014).

    Parameters
    ----------
    x
        Input data of shape ``(N, d)``.
    n_components
        Output embedding dimensionality (2 or 3).
    perplexity
        Effective neighborhood size.
    n_iter
        Total number of gradient descent steps.
    learning_rate
        Step size for the per-point gradient update.
    early_exaggeration
        Multiplier applied to :math:`P` during the early iterations to
        encourage cluster formation.
    early_exaggeration_iters
        Number of steps during which exaggeration is applied.
    momentum_initial, momentum_final
        Momentum coefficients before and after iteration 250.
    seed
        RNG seed for the initial embedding.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(N, n_components)`` — the low-dim embedding.

    Raises
    ------
    GeometryError
        On invalid input shape or parameters.
    """
    if x.dim() != 2:
        raise GeometryError(f"x must be 2-D; got {tuple(x.shape)}")
    n = x.shape[0]
    if n < 4:
        raise GeometryError("t-SNE requires N >= 4")
    if n_components not in (2, 3):
        raise GeometryError(f"n_components must be 2 or 3; got {n_components}")
    if perplexity <= 0 or perplexity >= n:
        raise GeometryError(
            f"perplexity must lie in (0, {n}); got {perplexity}"
        )

    _log.info("t-SNE: N=%d, d=%d → %d-D, perplexity=%.1f", n, x.shape[1], n_components, perplexity)
    P = _compute_P_tsne(x.detach(), perplexity).to(torch.float64)

    g = torch.Generator(device="cpu").manual_seed(seed)
    y = 1e-4 * torch.randn(n, n_components, generator=g, dtype=torch.float64)
    velocity = torch.zeros_like(y)
    gains = torch.ones_like(y)

    P_used = P * early_exaggeration
    for it in range(n_iter):
        if it == early_exaggeration_iters:
            P_used = P
        momentum = momentum_final if it >= 250 else momentum_initial

        # Q matrix (Student-t with 1 dof)
        D2_y = _pairwise_sq_dists(y)
        num = 1.0 / (1.0 + D2_y)
        num.fill_diagonal_(0.0)
        Z = num.sum()
        Q = (num / Z).clamp_min(1e-12)

        # Gradient (van der Maaten 2014, eqn. 5)
        # dC/dy_i = 4 * sum_j (P_ij - Q_ij) * num_ij * (y_i - y_j)
        PQ = (P_used - Q) * num  # (N, N)
        grad = 4.0 * (
            (PQ.sum(dim=1, keepdim=True) * y) - PQ @ y
        )

        # Adaptive gains (Jacobs heuristic): increase where grad direction
        # differs from velocity, decrease where consistent.
        gains = torch.where(
            (grad > 0) != (velocity > 0),
            gains + 0.2,
            gains * 0.8,
        ).clamp_min(0.01)

        velocity = momentum * velocity - learning_rate * gains * grad
        y = y + velocity
        # Re-center
        y = y - y.mean(dim=0, keepdim=True)

        if it % 50 == 0 or it == n_iter - 1:
            kl = float((P_used.clamp_min(1e-12) * (torch.log(P_used.clamp_min(1e-12)) - torch.log(Q))).sum())
            _log.debug("t-SNE iter %4d/%d  KL=%.4f", it, n_iter, kl)

    return y.to(torch.float32)


# ---------------------------------------------------------------------------
# UMAP (simplified)
# ---------------------------------------------------------------------------


def _find_ab(spread: float = 1.0, min_dist: float = 0.1) -> tuple[float, float]:
    """Calibrate UMAP's (a, b) parameters by least-squares fit.

    Approximates the curve

    .. math::

        \\phi(d) = \\begin{cases} 1, & d \\leq \\text{min\\_dist} \\\\
                                 e^{-(d - \\text{min\\_dist})/\\text{spread}}, & d > \\text{min\\_dist} \\end{cases}

    by :math:`(1 + a d^{2b})^{-1}` for non-negative ``d``.

    Parameters
    ----------
    spread
        Scale of the exponential tail.
    min_dist
        Distance below which target similarity saturates at 1.

    Returns
    -------
    tuple of float
        Calibrated ``(a, b)``.
    """
    xs = np.linspace(0.0, 3.0 * spread, 300)
    target = np.where(
        xs <= min_dist,
        1.0,
        np.exp(-(xs - min_dist) / spread),
    )

    def model(x: np.ndarray, a: float, b: float) -> np.ndarray:
        return 1.0 / (1.0 + a * x ** (2.0 * b))

    # Levenberg–Marquardt via SciPy if available, fallback to grid search.
    try:
        from scipy.optimize import curve_fit  # type: ignore[import-not-found]
        (a, b), _ = curve_fit(model, xs, target, p0=(1.0, 1.0), maxfev=10000)
        return float(a), float(b)
    except Exception:  # pragma: no cover - scipy optional fallback
        best = (1.0, 1.0)
        best_err = float("inf")
        for a in np.linspace(0.1, 3.0, 30):
            for b in np.linspace(0.5, 2.5, 21):
                err = float(np.mean((model(xs, a, b) - target) ** 2))
                if err < best_err:
                    best_err = err
                    best = (float(a), float(b))
        return best


def umap_from_scratch(
    x: torch.Tensor,
    *,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    spread: float = 1.0,
    n_epochs: int = 200,
    learning_rate: float = 1.0,
    negative_sample_rate: int = 5,
    seed: int = 0,
) -> torch.Tensor:
    """Pure-PyTorch simplified UMAP.

    Constructs a k-NN graph in input space, converts edge distances to fuzzy
    membership strengths through per-point local connectivity :math:`\\rho_i`
    and scale :math:`\\sigma_i`, then optimizes a low-dimensional embedding
    by stochastic edge-sampling with attractive and repulsive forces.

    Parameters
    ----------
    x
        Input data of shape ``(N, d)``.
    n_components
        Output dimensionality (2 or 3).
    n_neighbors
        Number of nearest neighbors used in graph construction.
    min_dist
        Effective minimum distance between embedded points.
    spread
        Effective scale of embedded clusters.
    n_epochs
        Number of stochastic optimization sweeps over the edge set.
    learning_rate
        Initial step size; linearly decayed to zero.
    negative_sample_rate
        Negative samples drawn per positive edge.
    seed
        RNG seed.

    Returns
    -------
    torch.Tensor
        Embedding of shape ``(N, n_components)``.

    Raises
    ------
    GeometryError
        On invalid parameters.
    """
    if x.dim() != 2:
        raise GeometryError(f"x must be 2-D; got {tuple(x.shape)}")
    n = x.shape[0]
    if n_neighbors < 2 or n_neighbors >= n:
        raise GeometryError(f"n_neighbors must lie in [2, {n - 1}]; got {n_neighbors}")
    if n_components not in (2, 3):
        raise GeometryError(f"n_components must be 2 or 3; got {n_components}")

    _log.info("UMAP: N=%d, d=%d → %d-D, k=%d", n, x.shape[1], n_components, n_neighbors)

    # 1) k-NN distances
    D2 = _pairwise_sq_dists(x.detach()).to(torch.float64)
    D = D2.clamp_min(0.0).sqrt()
    D.fill_diagonal_(float("inf"))
    knn_dist, knn_idx = D.topk(n_neighbors, dim=1, largest=False)

    # 2) Local connectivity rho_i and scale sigma_i via binary search
    rho = knn_dist[:, 0].clone()
    log_k = float(np.log2(n_neighbors))
    sigmas = torch.ones(n, dtype=torch.float64)
    for i in range(n):
        lo, hi, mid = 1e-20, 1e20, 1.0
        for _ in range(64):
            psum = float(
                torch.exp(-(knn_dist[i] - rho[i]).clamp_min(0.0) / mid).sum()
            )
            if abs(psum - log_k) < 1e-5:
                break
            if psum > log_k:
                hi = mid
                mid = (lo + mid) / 2.0
            else:
                lo = mid
                mid = (mid + hi) / 2.0 if hi < 1e19 else mid * 2.0
        sigmas[i] = mid

    # 3) Fuzzy weights w_ij
    w = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        diffs = (knn_dist[i] - rho[i]).clamp_min(0.0)
        w[i, knn_idx[i]] = torch.exp(-diffs / sigmas[i])
    # Probabilistic symmetrization: a OR b = a + b - a*b
    w = w + w.T - w * w.T

    # 4) Edge list with probabilities
    edges_i, edges_j = torch.nonzero(w > 0.0, as_tuple=True)
    edge_probs = w[edges_i, edges_j]
    n_edges = edges_i.numel()

    # 5) Initialize embedding via spectral / random
    g = torch.Generator(device="cpu").manual_seed(seed)
    y = 10.0 * torch.rand(n, n_components, generator=g, dtype=torch.float64) - 5.0

    # 6) (a, b) calibration
    a, b = _find_ab(spread=spread, min_dist=min_dist)
    _log.debug("UMAP curve params a=%.4f, b=%.4f", a, b)

    # 7) Stochastic optimization
    rng = np.random.default_rng(seed)
    edge_probs_np = edge_probs.cpu().numpy()
    edges_i_np = edges_i.cpu().numpy()
    edges_j_np = edges_j.cpu().numpy()
    eps_norm = 1e-3

    for epoch in range(n_epochs):
        alpha = learning_rate * (1.0 - epoch / max(n_epochs, 1))
        # Bernoulli sampling of positive edges by w_ij
        mask = rng.random(n_edges) < edge_probs_np
        active_i = edges_i_np[mask]
        active_j = edges_j_np[mask]

        for k in range(active_i.size):
            i = int(active_i[k])
            j = int(active_j[k])
            diff = y[i] - y[j]
            d2 = float((diff * diff).sum()) + eps_norm
            grad_coef = (-2.0 * a * b * d2 ** (b - 1.0)) / (1.0 + a * d2**b)
            grad = (grad_coef * diff).clamp(-4.0, 4.0)
            y[i] = y[i] + alpha * grad
            y[j] = y[j] - alpha * grad

            # Negative sampling
            for _ in range(negative_sample_rate):
                m = int(rng.integers(0, n))
                if m == i:
                    continue
                diffm = y[i] - y[m]
                dm2 = float((diffm * diffm).sum()) + eps_norm
                grad_coef = (2.0 * b) / ((eps_norm + dm2) * (1.0 + a * dm2**b))
                gradm = (grad_coef * diffm).clamp(-4.0, 4.0)
                y[i] = y[i] + alpha * gradm

        if epoch % 20 == 0 or epoch == n_epochs - 1:
            _log.debug("UMAP epoch %3d/%d  alpha=%.3f", epoch, n_epochs, alpha)

    return y.to(torch.float32)
