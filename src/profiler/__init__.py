"""Week 4 — HPC roofline analysis & kernel pipelines.

Modules
-------
log_parser
    Parse Nsight Compute / CUPTI / custom JSON kernel-trace exports.
roofline
    Build roofline plots from peak FLOP/s and memory bandwidth.
memory_hierarchy
    Predefined GPU memory hierarchy bandwidth/latency models.
kernel_gantt
    Render async kernel-pipeline Gantt charts from kernel traces.
"""

from __future__ import annotations

from .kernel_gantt import KernelEvent, build_gantt_segments
from .log_parser import (
    KernelTrace,
    parse_nsight_json,
    parse_simple_csv,
)
from .memory_hierarchy import (
    MemoryTier,
    default_a100_hierarchy,
    default_h100_hierarchy,
)
from .roofline import (
    RoofineEntry,
    RooflineModel,
    arithmetic_intensity,
    attainable_performance,
)

__all__ = [
    "KernelEvent",
    "KernelTrace",
    "MemoryTier",
    "RoofineEntry",
    "RooflineModel",
    "arithmetic_intensity",
    "attainable_performance",
    "build_gantt_segments",
    "default_a100_hierarchy",
    "default_h100_hierarchy",
    "parse_nsight_json",
    "parse_simple_csv",
]
