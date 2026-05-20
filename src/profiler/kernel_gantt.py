r"""Kernel Gantt-chart construction.

Given a kernel trace with per-kernel ``(start, duration, stream_id)`` tuples,
:func:`build_gantt_segments` returns matplotlib ``broken_barh``-ready data:

* One stream per row of the Y axis.
* Within a stream, kernels are non-overlapping intervals.

The optional categorical color scheme tags kernels as ``"gemm"`` (compute-
heavy), ``"memcpy"`` (data movement), ``"reduce"`` (collectives), or
``"other"``, enabling visual identification of stall structure in async
pipelines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..utils import ProfilerError, get_logger
from .log_parser import KernelTrace

__all__ = ["KernelEvent", "build_gantt_segments", "categorize_kernel"]

_log = get_logger(__name__)


@dataclass(frozen=True)
class KernelEvent:
    """One scheduled kernel on a stream's timeline.

    Attributes
    ----------
    name
        Kernel name.
    stream_id
        CUDA stream id.
    start_seconds, end_seconds
        Interval endpoints in seconds.
    category
        One of ``"gemm"``, ``"memcpy"``, ``"reduce"``, ``"elemwise"``,
        ``"attention"``, ``"other"``.
    """

    name: str
    stream_id: int
    start_seconds: float
    end_seconds: float
    category: str

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


_CATEGORY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("gemm", re.compile(r"(gemm|matmul|cutlass|hgemm|sgemm|cublas)", re.IGNORECASE)),
    ("attention", re.compile(r"(attn|attention|softmax|flash)", re.IGNORECASE)),
    ("memcpy", re.compile(r"(memcpy|memset|copy_d2d|copy_h2d|copy_d2h)", re.IGNORECASE)),
    ("reduce", re.compile(r"(reduce|allreduce|nccl|broadcast|allgather)", re.IGNORECASE)),
    ("elemwise", re.compile(r"(elementwise|relu|gelu|silu|add|mul|sub|div|bias)", re.IGNORECASE)),
]


def categorize_kernel(name: str) -> str:
    """Heuristically classify a kernel from its name.

    Parameters
    ----------
    name
        Kernel name string.

    Returns
    -------
    str
        Category label.
    """
    for cat, pat in _CATEGORY_PATTERNS:
        if pat.search(name):
            return cat
    return "other"


def build_gantt_segments(
    kernels: list[KernelTrace],
    *,
    merge_below_seconds: float = 0.0,
) -> dict[int, list[KernelEvent]]:
    """Group kernels by stream into per-stream timelines.

    Parameters
    ----------
    kernels
        List of :class:`KernelTrace` records.
    merge_below_seconds
        If positive, adjacent kernels of the same category on the same
        stream that are separated by less than this gap are merged into a
        single ``KernelEvent`` (the name becomes ``"<category>:N"``). Useful
        to declutter zoomed-out views.

    Returns
    -------
    dict
        Mapping ``stream_id -> list[KernelEvent]``, each list sorted by
        start time.

    Raises
    ------
    ProfilerError
        On invalid input.
    """
    if merge_below_seconds < 0:
        raise ProfilerError(f"merge_below_seconds must be non-negative; got {merge_below_seconds}")
    by_stream: dict[int, list[KernelEvent]] = {}
    for k in kernels:
        ev = KernelEvent(
            name=k.name,
            stream_id=k.stream_id,
            start_seconds=k.start_seconds,
            end_seconds=k.end_seconds,
            category=categorize_kernel(k.name),
        )
        by_stream.setdefault(k.stream_id, []).append(ev)

    # Sort each stream
    for stream_id in by_stream:
        by_stream[stream_id].sort(key=lambda e: e.start_seconds)

    if merge_below_seconds > 0:
        for stream_id, evs in by_stream.items():
            merged: list[KernelEvent] = []
            for ev in evs:
                if (
                    merged
                    and merged[-1].category == ev.category
                    and (ev.start_seconds - merged[-1].end_seconds) <= merge_below_seconds
                ):
                    last = merged[-1]
                    # Count merged events through name
                    if ":" in last.name and last.name.startswith(last.category + ":"):
                        try:
                            n = int(last.name.split(":")[1]) + 1
                        except ValueError:
                            n = 2
                    else:
                        n = 2
                    merged[-1] = KernelEvent(
                        name=f"{last.category}:{n}",
                        stream_id=last.stream_id,
                        start_seconds=last.start_seconds,
                        end_seconds=ev.end_seconds,
                        category=last.category,
                    )
                else:
                    merged.append(ev)
            by_stream[stream_id] = merged

    _log.info(
        "Gantt: %d streams, %d total events",
        len(by_stream),
        sum(len(v) for v in by_stream.values()),
    )
    return by_stream
