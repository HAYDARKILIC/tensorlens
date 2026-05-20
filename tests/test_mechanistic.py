"""Tests for :mod:`src.mechanistic`."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.mechanistic import (
    ResidualStreamCapture,
    SAETrainingConfig,
    SparseAutoencoder,
    attention_entropy,
    build_feature_cooccurrence_graph,
    classify_head_archetypes,
    induction_score,
    top_activating_examples,
    train_sae,
)
from src.utils import MechanisticError
from src.utils.synthetic import (
    AttentionConfig,
    synth_attention_matrix,
    synth_polysemantic_activations,
)


def test_attention_entropy_zero_for_one_hot() -> None:
    """A one-hot attention row has entropy = 0."""
    T = 8
    attn = torch.zeros(1, T, T)
    for i in range(T):
        attn[0, i, 0] = 1.0
    ent = attention_entropy(attn)
    assert ent.shape == (1, T)
    assert torch.allclose(ent, torch.zeros_like(ent), atol=1e-5)


def test_attention_entropy_log_T_for_uniform() -> None:
    T = 8
    causal_uniform = torch.zeros(1, T, T)
    for i in range(T):
        causal_uniform[0, i, : i + 1] = 1.0 / (i + 1)
    ent = attention_entropy(causal_uniform)
    # Last row: uniform over all T → entropy = log T
    assert abs(float(ent[0, T - 1]) - float(torch.log(torch.tensor(float(T))))) < 1e-4


def test_attention_entropy_rejects_non_stochastic() -> None:
    with pytest.raises(MechanisticError):
        attention_entropy(torch.randn(1, 4, 4))


def test_classify_head_archetypes_synthetic() -> None:
    cfg = AttentionConfig(seq_len=24, n_heads=12, induction_weight=0.4,
                          sink_weight=0.3, uniform_weight=0.3, seed=2)
    attn = synth_attention_matrix(cfg)
    labels = classify_head_archetypes(attn)
    assert len(labels) == 12
    valid = {"induction", "sink", "uniform", "mixed"}
    for lbl in labels:
        assert lbl in valid


def test_induction_score_in_unit_interval() -> None:
    cfg = AttentionConfig(seq_len=16, n_heads=4, seed=0)
    attn = synth_attention_matrix(cfg)
    # Ensure a repeat token in token_ids
    toks = torch.randint(0, 5, (16,))
    toks[12] = toks[3]
    scores = induction_score(attn, toks)
    assert scores.shape == (4,)
    assert torch.all(scores >= 0.0)
    assert torch.all(scores <= 1.0 + 1e-6)


def test_induction_score_rejects_shape_mismatch() -> None:
    attn = torch.softmax(torch.randn(4, 8, 8), dim=-1)
    with pytest.raises(MechanisticError):
        induction_score(attn, torch.randint(0, 4, (10,)))


def test_residual_stream_capture_smoke() -> None:
    """Minimal capture test: a tiny linear stack with named submodules."""

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([
                nn.ModuleDict({"heads": nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)]),
                                "mlp": nn.Linear(8, 8)})
                for _ in range(2)
            ])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for blk in self.blocks:
                for h in blk["heads"]:
                    x = x + h(x)
                x = x + blk["mlp"](x)
            return x

    def head_sel(_m: nn.Module, name: str):
        parts = name.split(".")
        if len(parts) == 4 and parts[0] == "blocks" and parts[2] == "heads":
            return int(parts[1]), int(parts[3])
        return None

    def mlp_sel(_m: nn.Module, name: str):
        parts = name.split(".")
        if len(parts) == 3 and parts[0] == "blocks" and parts[2] == "mlp":
            return int(parts[1])
        return None

    model = Tiny()
    with ResidualStreamCapture(model, attention_head_selector=head_sel,
                               mlp_selector=mlp_sel) as cap:
        _ = model(torch.randn(2, 8))
    # After the with block, hooks are removed but the record persists.
    assert len(cap.record.attention_writes) == 4
    assert len(cap.record.mlp_writes) == 2
    norms = cap.record.attention_write_norms()
    assert all(t.shape == (2,) for t in norms.values())


def test_residual_stream_requires_selector() -> None:
    with pytest.raises(MechanisticError):
        ResidualStreamCapture(nn.Linear(4, 4))


def test_sparse_autoencoder_forward_shape() -> None:
    sae = SparseAutoencoder(d_model=8, n_features=32)
    x = torch.randn(20, 8)
    recon, code = sae(x)
    assert recon.shape == x.shape
    assert code.shape == (20, 32)
    assert torch.all(code >= 0.0)  # ReLU output


def test_sparse_autoencoder_normalizes_decoder() -> None:
    sae = SparseAutoencoder(d_model=8, n_features=32)
    with torch.no_grad():
        sae.W_dec.mul_(5.0)  # blow up
    sae.normalize_decoder()
    norms = sae.W_dec.norm(dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_train_sae_decreases_loss() -> None:
    acts, _ = synth_polysemantic_activations(
        n_samples=600, d_model=8, n_features=32, sparsity=0.1, seed=0
    )
    sae = SparseAutoencoder(d_model=8, n_features=24)
    cfg = SAETrainingConfig(
        l1_coefficient=1e-3,
        learning_rate=5e-3,
        batch_size=128,
        n_epochs=3,
        resample_dead=False,
        seed=0,
    )
    res = train_sae(acts, sae, cfg)
    assert res.losses[-1] < res.losses[0]  # loss decreased
    assert res.feature_activation_density.shape == (24,)


def test_feature_cooccurrence_graph_structure() -> None:
    codes = torch.zeros(20, 5)
    # Plant features 0 and 1 always co-fire; feature 2 never fires.
    codes[:10, 0] = 1.0
    codes[:10, 1] = 1.0
    G = build_feature_cooccurrence_graph(codes, edge_threshold=0.5)
    assert G.has_edge(0, 1)
    assert G.nodes[2]["density"] == 0.0


def test_top_activating_examples_returns_indices() -> None:
    codes = torch.zeros(20, 4)
    codes[5, 0] = 5.0
    codes[3, 0] = 2.0
    out = top_activating_examples(codes, n_examples=2)
    assert out[0][0].item() == 5  # highest value at row 5
