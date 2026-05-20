r"""Feature co-activation graphs.

Given an SAE code matrix :math:`Z \in \mathbb{R}^{N \times F}` (rows are
samples, columns are features), we model each feature :math:`k` by its
support :math:`S_k = \{i : z_{i,k} > \tau\}`. Two features are linked in the
co-activation graph with edge weight equal to the Jaccard similarity:

.. math::

    J(k, l) \;=\; \frac{|S_k \cap S_l|}{|S_k \cup S_l|}

Edges below a user-supplied threshold are pruned. The result is a
``networkx.Graph`` ready for force-directed layout and Plotly rendering.

Top-activating examples per feature are extracted for qualitative
inspection in :func:`top_activating_examples`.
"""

from __future__ import annotations

import networkx as nx
import torch

from ..utils import MechanisticError, get_logger

__all__ = ["build_feature_cooccurrence_graph", "top_activating_examples"]

_log = get_logger(__name__)


def build_feature_cooccurrence_graph(
    codes: torch.Tensor,
    *,
    activation_threshold: float = 0.0,
    edge_threshold: float = 0.1,
    keep_top_features: int | None = None,
) -> nx.Graph:
    """Construct a Jaccard co-activation graph over SAE features.

    Parameters
    ----------
    codes
        Tensor of shape ``(N, F)`` — SAE feature codes for ``N`` samples.
    activation_threshold
        Activations strictly greater than this are considered "fired".
    edge_threshold
        Minimum Jaccard similarity to include an edge.
    keep_top_features
        If set, restrict the graph to the ``k`` features with the highest
        activation density.

    Returns
    -------
    networkx.Graph
        Undirected graph; nodes are feature indices, edges carry attribute
        ``"weight"`` (Jaccard).

    Raises
    ------
    MechanisticError
        On invalid input shape.
    """
    if codes.dim() != 2:
        raise MechanisticError(f"codes must be (N, F); got {tuple(codes.shape)}")
    if not 0.0 <= edge_threshold <= 1.0:
        raise MechanisticError(f"edge_threshold must lie in [0, 1]; got {edge_threshold}")

    fires = (codes > activation_threshold).to(torch.float32)
    density = fires.mean(dim=0)
    keep_idx = torch.arange(codes.shape[1])
    if keep_top_features is not None:
        keep_top_features = min(keep_top_features, codes.shape[1])
        keep_idx = torch.topk(density, keep_top_features).indices.sort().values
    fires_sub = fires[:, keep_idx]  # (N, k)

    intersection = fires_sub.T @ fires_sub  # (k, k)
    sums = fires_sub.sum(dim=0)  # (k,)
    union = sums.unsqueeze(0) + sums.unsqueeze(1) - intersection
    union = union.clamp_min(1e-12)
    jaccard = (intersection / union).cpu().numpy()

    G = nx.Graph()
    for k, feat_id in enumerate(keep_idx.tolist()):
        G.add_node(int(feat_id), density=float(density[feat_id].item()))
    n_nodes = len(keep_idx)
    for a in range(n_nodes):
        for b in range(a + 1, n_nodes):
            w = float(jaccard[a, b])
            if w >= edge_threshold:
                G.add_edge(int(keep_idx[a].item()), int(keep_idx[b].item()), weight=w)
    _log.info(
        "Co-occurrence graph: |V|=%d, |E|=%d (edge_threshold=%.2f)",
        G.number_of_nodes(),
        G.number_of_edges(),
        edge_threshold,
    )
    return G


def top_activating_examples(
    codes: torch.Tensor,
    *,
    n_features: int | None = None,
    n_examples: int = 10,
) -> dict[int, torch.Tensor]:
    """Return the indices of the top-activating examples for each feature.

    Parameters
    ----------
    codes
        Tensor of shape ``(N, F)``.
    n_features
        If set, limit to the first ``n_features`` columns. Otherwise process
        all features.
    n_examples
        Number of top examples to return per feature.

    Returns
    -------
    dict
        Mapping ``feature_id -> tensor of sample indices`` (length ≤
        ``n_examples``).

    Raises
    ------
    MechanisticError
        On invalid shapes.
    """
    if codes.dim() != 2:
        raise MechanisticError(f"codes must be (N, F); got {tuple(codes.shape)}")
    if n_examples <= 0:
        raise MechanisticError("n_examples must be positive")
    n, F = codes.shape
    upper = F if n_features is None else min(n_features, F)
    out: dict[int, torch.Tensor] = {}
    for k in range(upper):
        col = codes[:, k]
        m = min(n_examples, n)
        top = torch.topk(col, m).indices
        out[k] = top.cpu()
    return out
