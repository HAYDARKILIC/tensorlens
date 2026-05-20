"""Tests for :mod:`src.profiler`."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import pytest

from src.profiler import (
    KernelTrace,
    RooflineModel,
    arithmetic_intensity,
    attainable_performance,
    build_gantt_segments,
    default_a100_hierarchy,
    default_h100_hierarchy,
    parse_nsight_json,
    parse_simple_csv,
)
from src.profiler.kernel_gantt import categorize_kernel
from src.utils import ProfilerError


def _make_kernel(name: str, flops: float, b_dram: float, b_l2: float = 0, b_sram: float = 0,
                 dur: float = 1e-3, start: float = 0.0, stream: int = 0) -> KernelTrace:
    return KernelTrace(
        name=name,
        start_seconds=start,
        duration_seconds=dur,
        flops=flops,
        bytes_dram=b_dram,
        bytes_l2=b_l2,
        bytes_sram=b_sram,
        stream_id=stream,
    )


def test_kernel_arithmetic_intensity() -> None:
    k = _make_kernel("gemm", flops=1000.0, b_dram=10.0)
    assert k.arithmetic_intensity_dram() == 100.0


def test_arithmetic_intensity_function_tiers() -> None:
    k = _make_kernel("k", flops=200.0, b_dram=20.0, b_l2=10.0, b_sram=5.0)
    assert arithmetic_intensity(k, tier="dram") == 10.0
    assert arithmetic_intensity(k, tier="l2") == 20.0
    assert arithmetic_intensity(k, tier="sram") == 40.0


def test_arithmetic_intensity_unknown_tier() -> None:
    k = _make_kernel("k", flops=1.0, b_dram=1.0)
    with pytest.raises(ProfilerError):
        arithmetic_intensity(k, tier="bogus")


def test_attainable_performance_picks_minimum() -> None:
    P_peak = 312e12
    beta = 2e12
    # Memory bound: I = 10 → β·I = 2e13 < P_peak (312e12) → wait, 2e13 = 20e12 < 312e12 yes
    assert attainable_performance(10.0, P_peak, beta) == 20e12
    # Compute bound: I = 1000 → β·I = 2e15 > P_peak → P_peak
    assert attainable_performance(1000.0, P_peak, beta) == P_peak


def test_memory_tier_ordering_in_default_hierarchies() -> None:
    """SRAM bandwidth > L2 bandwidth > HBM bandwidth > Host RAM bandwidth."""
    for hierarchy in (default_a100_hierarchy(), default_h100_hierarchy()):
        bw = {t.name: t.bandwidth_bytes_per_sec for t in hierarchy}
        sram = [v for k, v in bw.items() if "SRAM" in k][0]
        l2 = [v for k, v in bw.items() if "L2" in k][0]
        hbm = [v for k, v in bw.items() if "HBM" in k][0]
        host = [v for k, v in bw.items() if "Host" in k][0]
        assert sram > l2 > hbm > host


def test_roofline_model_ridge_points() -> None:
    rl = RooflineModel(peak_flops_per_sec=312e12, hierarchy=default_a100_hierarchy(),
                       device_name="A100")
    ridges = rl.ridge_points()
    assert len(ridges) == len(default_a100_hierarchy())
    # HBM ridge should land around 150 FLOP/B
    hbm_name = next(t.name for t in default_a100_hierarchy() if "HBM" in t.name)
    assert 100 < ridges[hbm_name] < 200


def test_roofline_evaluate_kernel() -> None:
    rl = RooflineModel(peak_flops_per_sec=312e12, hierarchy=default_a100_hierarchy(),
                       device_name="A100")
    k = _make_kernel("gemm_fp16", flops=2e12, b_dram=4e9, dur=10e-3)
    entry = rl.evaluate_kernel(k)
    assert entry.name == "gemm_fp16"
    assert entry.arithmetic_intensity == pytest.approx(500.0)
    assert entry.achieved_flops_per_sec == pytest.approx(2e14)


def test_roofline_rejects_zero_peak() -> None:
    with pytest.raises(ProfilerError):
        RooflineModel(peak_flops_per_sec=0.0, hierarchy=default_a100_hierarchy())


def test_roofline_rejects_empty_hierarchy() -> None:
    with pytest.raises(ProfilerError):
        RooflineModel(peak_flops_per_sec=312e12, hierarchy=[])


def test_categorize_kernel_patterns() -> None:
    assert categorize_kernel("gemm_fp16") == "gemm"
    assert categorize_kernel("cutlass_sm80_gemm") == "gemm"
    assert categorize_kernel("attention_softmax") == "attention"
    assert categorize_kernel("memcpy_d2h") == "memcpy"
    assert categorize_kernel("nccl_allreduce") == "reduce"
    assert categorize_kernel("elementwise_relu") == "elemwise"
    assert categorize_kernel("some_unknown_thing") == "other"


def test_gantt_segments_group_by_stream() -> None:
    kernels = [
        _make_kernel("gemm_0", 1e12, 1e9, start=0.0, dur=1e-3, stream=0),
        _make_kernel("gemm_1", 1e12, 1e9, start=2e-3, dur=1e-3, stream=0),
        _make_kernel("memcpy_d2h", 0, 1e9, start=0.5e-3, dur=1e-3, stream=1),
    ]
    gantt = build_gantt_segments(kernels)
    assert sorted(gantt.keys()) == [0, 1]
    assert len(gantt[0]) == 2
    # Sorted by start
    assert gantt[0][0].start_seconds < gantt[0][1].start_seconds


def test_gantt_segments_merge() -> None:
    kernels = [
        _make_kernel("gemm_0", 1e12, 1e9, start=0.0, dur=1e-3, stream=0),
        _make_kernel("gemm_1", 1e12, 1e9, start=1.0e-3, dur=1e-3, stream=0),  # contiguous, same cat
    ]
    gantt = build_gantt_segments(kernels, merge_below_seconds=1e-4)
    assert len(gantt[0]) == 1
    assert gantt[0][0].category == "gemm"


def test_gantt_rejects_negative_merge() -> None:
    with pytest.raises(ProfilerError):
        build_gantt_segments([], merge_below_seconds=-0.001)


def test_parse_simple_csv_roundtrip() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel_name", "start_ms", "duration_ms", "flops",
                    "bytes_dram", "bytes_l2", "bytes_sram", "stream_id"])
        w.writerow(["gemm_fp16", "0.0", "1.5", "1000000000", "100000", "50000", "25000", "0"])
        path = Path(f.name)
    kernels = parse_simple_csv(path)
    assert len(kernels) == 1
    assert kernels[0].name == "gemm_fp16"
    assert kernels[0].duration_seconds == pytest.approx(1.5e-3)
    assert kernels[0].flops == 1e9


def test_parse_simple_csv_missing_columns() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel_name", "duration_ms"])
        w.writerow(["k", "1.0"])
        path = Path(f.name)
    with pytest.raises(ProfilerError):
        parse_simple_csv(path)


def test_parse_nsight_json_minimal() -> None:
    data = {
        "kernels": [
            {
                "name": "gemm_kernel",
                "start_seconds": 0.0,
                "duration_seconds": 0.001,
                "flops": 1e9,
                "bytes_dram": 1e5,
                "bytes_l2": 5e4,
                "bytes_sram": 1e4,
                "stream_id": 0,
            }
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = Path(f.name)
    kernels = parse_nsight_json(path)
    assert len(kernels) == 1
    assert kernels[0].name == "gemm_kernel"


def test_parse_nsight_json_rejects_missing_file() -> None:
    with pytest.raises(ProfilerError):
        parse_nsight_json("/nonexistent/path.json")
