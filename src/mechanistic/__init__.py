"""Weeks 2 & 5 — Mechanistic interpretability.

Modules
-------
residual_stream_hooks
    Raw forward-hook capture engine for residual-stream contributions.
attention_entropy
    Per-head Shannon entropy and induction-head detection.
sparse_autoencoder
    PyTorch SAE for monosemantic feature recovery.
feature_graph
    Network graph of co-activating monosemantic features.
"""

from __future__ import annotations

from .attention_entropy import (
    attention_entropy,
    classify_head_archetypes,
    induction_score,
)
from .feature_graph import build_feature_cooccurrence_graph, top_activating_examples
from .residual_stream_hooks import (
    HookHandle,
    ResidualStreamCapture,
    StreamRecord,
    temporary_capture,
)
from .sparse_autoencoder import SparseAutoencoder, SAETrainingConfig, train_sae

__all__ = [
    "HookHandle",
    "ResidualStreamCapture",
    "SAETrainingConfig",
    "SparseAutoencoder",
    "StreamRecord",
    "attention_entropy",
    "build_feature_cooccurrence_graph",
    "classify_head_archetypes",
    "induction_score",
    "temporary_capture",
    "top_activating_examples",
    "train_sae",
]
