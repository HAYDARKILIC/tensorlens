"""Coverage-targeted tests for edge cases.

Hits roofline rendering, residual-stream layer-selector path, SAE dead-feature
resampling, and other low-coverage branches.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # Headless rendering for CI

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.mechanistic import (
    ResidualStreamCapture,
    SAETrainingConfig,
    SparseAutoencoder,
    train_sae,
)
from src.mechanistic.residual_stream_hooks import _extract_tensor, temporary_capture
from src.optimization import generate_filter_normalized_direction
from src.profiler import (
    KernelTrace,
    RooflineModel,
    arithmetic_intensity,
    default_a100_hierarchy,
)
from src.utils import (
    GeometryError,
    MechanisticError,
    OptimizationError,
    ProfilerError,
    describe_device,
    get_logger,
)
from src.utils.synthetic import synth_polysemantic_activations


def test_roofline_render_returns_axes() -> None:
    rl = RooflineModel(peak_flops_per_sec=312e12, hierarchy=default_a100_hierarchy(),
                       device_name="A100")
    kernels = [
        KernelTrace("gemm", 0, 1e-3, 1e12, 1e8, 5e7, 1e7, 0),
        KernelTrace("memcpy", 0, 5e-4, 0, 1e9, 0, 0, 1),  # AI = 0
        KernelTrace("compute_bound", 0, 1e-3, 5e12, 1e7, 5e6, 1e6, 0),  # high AI
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax_out = rl.render(kernels, tier="dram", ax=ax)
    assert ax_out is ax
    plt.close(fig)


def test_roofline_render_no_kernels() -> None:
    rl = RooflineModel(peak_flops_per_sec=100e12, hierarchy=default_a100_hierarchy())
    ax = rl.render(None, tier="l2")
    assert ax is not None
    plt.close(ax.figure)


def test_roofline_evaluate_list() -> None:
    rl = RooflineModel(peak_flops_per_sec=100e12, hierarchy=default_a100_hierarchy())
    kernels = [
        KernelTrace("k1", 0, 1e-3, 1e12, 1e8, 0, 0, 0),
        KernelTrace("k2", 1e-3, 1e-3, 5e11, 5e7, 0, 0, 0),
    ]
    entries = rl.evaluate(kernels, tier="dram")
    assert len(entries) == 2
    assert entries[0].name == "k1"


def test_arithmetic_intensity_zero_bytes_zero_flops() -> None:
    k = KernelTrace("k", 0, 1e-3, 0, 0, 0, 0, 0)
    assert arithmetic_intensity(k, tier="dram") == 0.0


def test_arithmetic_intensity_l2_sram_zero() -> None:
    k = KernelTrace("k", 0, 1e-3, 1.0, 0, 0, 0, 0)
    assert arithmetic_intensity(k, tier="l2") == float("inf")
    assert arithmetic_intensity(k, tier="sram") == float("inf")


def test_residual_stream_layer_selector() -> None:
    """Exercise the layer_selector branch (not covered by other tests)."""

    class TwoLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.l0 = nn.Linear(4, 4)
            self.l1 = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.l1(self.l0(x))

    def layer_sel(_m: nn.Module, name: str):
        if name in ("l0", "l1"):
            return 0 if name == "l0" else 1
        return None

    m = TwoLayer()
    with temporary_capture(m, layer_selector=layer_sel) as cap:
        m(torch.randn(2, 4))
    assert set(cap.record.layer_outputs.keys()) == {0, 1}


def test_residual_stream_tuple_output_unwrapping() -> None:
    """Many HF modules return tuples — the extractor must take the first tensor."""

    class TupleOut(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor):
            return self.lin(x), None  # tuple

    def mlp_sel(_m: nn.Module, name: str):
        return 0 if name == "lin" else None

    m = TupleOut()
    with ResidualStreamCapture(m, mlp_selector=mlp_sel) as cap:
        m(torch.randn(2, 4))
    assert 0 in cap.record.mlp_writes


def test_residual_stream_extract_tensor_errors() -> None:
    import pytest
    with pytest.raises(MechanisticError):
        _extract_tensor("not a tensor")
    with pytest.raises(MechanisticError):
        _extract_tensor(())  # empty tuple


def test_residual_stream_no_selector_raises() -> None:
    import pytest
    with pytest.raises(MechanisticError):
        ResidualStreamCapture(nn.Linear(4, 4))


def test_residual_stream_double_install_raises() -> None:
    import pytest

    def sel(_m, name):
        return 0 if name == "" else None

    cap = ResidualStreamCapture(nn.Linear(4, 4), mlp_selector=sel)
    cap.install()
    with pytest.raises(MechanisticError):
        cap.install()
    cap.remove()


def test_residual_stream_reset_clears_record() -> None:
    def sel(_m, name):
        return 0 if name == "" else None

    m = nn.Linear(4, 4)
    cap = ResidualStreamCapture(m, mlp_selector=sel)
    cap.install()
    m(torch.randn(2, 4))
    assert len(cap.record.mlp_writes) == 1
    cap.reset()
    assert len(cap.record.mlp_writes) == 0
    cap.remove()


def test_sae_dead_feature_resample_path() -> None:
    """Force a tiny L1 so most features stay dead → triggers resample branch."""
    acts, _ = synth_polysemantic_activations(
        n_samples=400, d_model=8, n_features=40, sparsity=0.02, seed=0
    )
    sae = SparseAutoencoder(d_model=8, n_features=32)
    # Aggressive L1 → many dead features
    cfg = SAETrainingConfig(
        l1_coefficient=5e-1,
        learning_rate=5e-3,
        batch_size=64,
        n_epochs=2,
        dead_feature_window=2,
        dead_feature_threshold=0.05,
        resample_dead=True,
        seed=0,
    )
    res = train_sae(acts, sae, cfg)
    assert res.feature_activation_density.shape == (32,)


def test_sae_rejects_mismatched_shape() -> None:
    import pytest
    sae = SparseAutoencoder(d_model=8, n_features=16)
    with pytest.raises(MechanisticError):
        train_sae(torch.randn(10, 4), sae, SAETrainingConfig())


def test_sae_init_with_invalid_dims_raises() -> None:
    import pytest
    with pytest.raises(MechanisticError):
        SparseAutoencoder(d_model=0, n_features=16)


def test_describe_device_cpu_path() -> None:
    s = describe_device(None)
    assert isinstance(s, str)


def test_logger_idempotent_across_modules() -> None:
    log1 = get_logger("tensorlens.extra")
    log2 = get_logger("tensorlens.extra")
    assert log1 is log2


def test_filter_normalization_dict_input() -> None:
    """Accept a state-dict directly, not just a module."""
    state = {"layer.weight": torch.randn(8, 4), "layer.bias": torch.zeros(8)}
    direction = generate_filter_normalized_direction(state, seed=0)
    assert set(direction.keys()) == set(state.keys())
    assert torch.all(direction["layer.bias"] == 0)


def test_project_trajectory_empty_raises() -> None:
    import pytest
    state = {"w": torch.randn(4, 4)}
    with pytest.raises(OptimizationError):
        from src.optimization import project_trajectory
        project_trajectory([], state, state, anchor=state)


def test_project_trajectory_key_mismatch() -> None:
    import pytest
    from src.optimization import project_trajectory
    anchor = {"a": torch.zeros(4, 4)}
    dir_a = {"a": torch.zeros(4, 4)}
    dir_b = {"b": torch.zeros(4, 4)}  # mismatched key
    with pytest.raises(OptimizationError):
        project_trajectory([anchor], dir_a, dir_b, anchor)


def test_intrinsic_dimension_input_validation() -> None:
    import pytest
    from src.geometry import twonn_intrinsic_dimension, mle_intrinsic_dimension
    with pytest.raises(GeometryError):
        twonn_intrinsic_dimension(torch.randn(10))
    with pytest.raises(GeometryError):
        twonn_intrinsic_dimension(torch.randn(3, 4))  # N < 8
    with pytest.raises(GeometryError):
        mle_intrinsic_dimension(torch.randn(10), k=200)


def test_anisotropy_full_report_returns_struct() -> None:
    from src.geometry.anisotropy import full_report
    torch.manual_seed(0)
    x = torch.randn(80, 16)
    r = full_report(x, sample_size=50)
    assert r.dim == 16
    assert r.n_tokens == 80
    assert r.singular_values.shape == (16,)
    assert str(r).startswith("AnisotropyReport")


def test_attention_entropy_via_uniform() -> None:
    """Custom test: rows that are uniform should give log(T) entropy."""
    from src.mechanistic.attention_entropy import attention_entropy
    T = 16
    attn = torch.zeros(1, T, T)
    # Each row uniform over its causal prefix
    for i in range(T):
        attn[0, i, : i + 1] = 1.0 / (i + 1)
    ent = attention_entropy(attn)
    # Bottom row entropy ≈ log T
    assert abs(float(ent[0, T - 1]) - float(np.log(T))) < 1e-4


def test_kernel_event_duration() -> None:
    from src.profiler.kernel_gantt import KernelEvent
    ev = KernelEvent("k", stream_id=0, start_seconds=0.0, end_seconds=1.5, category="gemm")
    assert ev.duration_seconds == 1.5


def test_parse_nsight_json_alternative_keys() -> None:
    import json
    import tempfile
    from pathlib import Path
    from src.profiler import parse_nsight_json
    # The "data" key variant
    data = {"data": [{"name": "k", "start_seconds": 0, "duration_seconds": 1, "flops": 1, "bytes_dram": 1, "bytes_l2": 0, "bytes_sram": 0}]}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = Path(f.name)
    kernels = parse_nsight_json(path)
    assert len(kernels) == 1


def test_parse_nsight_json_malformed() -> None:
    import pytest
    import tempfile
    from pathlib import Path
    from src.profiler import parse_nsight_json
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("{not valid json")
        path = Path(f.name)
    with pytest.raises(ProfilerError):
        parse_nsight_json(path)


def test_parse_nsight_json_unknown_schema() -> None:
    import pytest
    import json
    import tempfile
    from pathlib import Path
    from src.profiler import parse_nsight_json
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"unknown_key": []}, f)
        path = Path(f.name)
    with pytest.raises(ProfilerError):
        parse_nsight_json(path)
