r"""Parse kernel-trace exports from Nsight Compute / CUPTI / custom logs.

Two input formats are supported:

* **Nsight Compute JSON** export — the schema produced by ``ncu --export
  json``. We extract per-kernel timing, FLOP counts (from the metric
  ``smsp__sass_thread_inst_executed_op_*``), and per-tier bytes transferred
  (``dram__bytes`` / ``lts__t_bytes`` / ``l1tex__t_bytes``).
* **Simple CSV** with columns ``kernel_name, start_ms, duration_ms, flops,
  bytes_dram, bytes_l2, bytes_sram, stream_id``.

The parsers normalize to a uniform :class:`KernelTrace` record so that
downstream roofline and Gantt-chart code is format-agnostic.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from ..utils import ProfilerError, get_logger

__all__ = ["KernelTrace", "parse_nsight_json", "parse_simple_csv"]

_log = get_logger(__name__)


@dataclass(frozen=True)
class KernelTrace:
    """Uniform per-kernel record across input formats.

    Attributes
    ----------
    name
        Kernel name (often demangled CUDA symbol).
    start_seconds
        Start timestamp in seconds.
    duration_seconds
        Wall-clock duration in seconds.
    flops
        Floating-point operations executed.
    bytes_dram, bytes_l2, bytes_sram
        Bytes transferred at each memory tier.
    stream_id
        CUDA stream identifier (0 for default).
    """

    name: str
    start_seconds: float
    duration_seconds: float
    flops: float
    bytes_dram: float
    bytes_l2: float
    bytes_sram: float
    stream_id: int = 0

    @property
    def end_seconds(self) -> float:
        """Compute end timestamp."""
        return self.start_seconds + self.duration_seconds

    @property
    def achieved_flops_per_sec(self) -> float:
        """Achieved performance in FLOP/s.

        Returns
        -------
        float
            ``flops / duration_seconds`` (0 if duration is zero).
        """
        if self.duration_seconds <= 0:
            return 0.0
        return self.flops / self.duration_seconds

    def arithmetic_intensity_dram(self) -> float:
        """Arithmetic intensity relative to DRAM traffic in FLOP/B."""
        if self.bytes_dram <= 0:
            return float("inf") if self.flops > 0 else 0.0
        return self.flops / self.bytes_dram


def parse_nsight_json(path: str | Path) -> list[KernelTrace]:
    """Parse an Nsight Compute JSON export.

    The Nsight schema is documented at NVIDIA developer site. This parser is
    permissive: missing metrics default to zero.

    Parameters
    ----------
    path
        Path to the JSON file.

    Returns
    -------
    list of KernelTrace
        Parsed kernel records.

    Raises
    ------
    ProfilerError
        On file-not-found, malformed JSON, or unsupported schema version.
    """
    path = Path(path)
    if not path.is_file():
        raise ProfilerError(f"Nsight JSON not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ProfilerError(f"Malformed Nsight JSON: {exc}") from exc

    # Nsight's export typically has a top-level "kernels" list; some exporters
    # wrap it under "report" or "data". We try several keys.
    kernels = None
    for key in ("kernels", "data", "report"):
        if isinstance(data, dict) and key in data:
            candidate = data[key]
            if isinstance(candidate, list):
                kernels = candidate
                break
    if kernels is None and isinstance(data, list):
        kernels = data
    if kernels is None:
        raise ProfilerError(
            "Unrecognized Nsight JSON schema; expected top-level 'kernels' list"
        )

    out: list[KernelTrace] = []
    for raw in kernels:
        if not isinstance(raw, dict):
            continue
        metrics = raw.get("metrics", {}) if isinstance(raw.get("metrics"), dict) else {}
        flops_add = float(metrics.get("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum", 0.0))
        flops_mul = float(metrics.get("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum", 0.0))
        flops_fma = float(metrics.get("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum", 0.0))
        flops = flops_add + flops_mul + 2.0 * flops_fma  # FMA counts as 2 ops
        if flops == 0 and "flops" in raw:
            flops = float(raw["flops"])

        bytes_dram = float(metrics.get("dram__bytes.sum", raw.get("bytes_dram", 0.0)))
        bytes_l2 = float(metrics.get("lts__t_bytes.sum", raw.get("bytes_l2", 0.0)))
        bytes_sram = float(metrics.get("l1tex__t_bytes.sum", raw.get("bytes_sram", 0.0)))
        out.append(
            KernelTrace(
                name=str(raw.get("name", raw.get("kernel_name", "<unknown>"))),
                start_seconds=float(raw.get("start_seconds", raw.get("start_ms", 0.0)) / 1000.0)
                if "start_ms" in raw
                else float(raw.get("start_seconds", 0.0)),
                duration_seconds=float(raw.get("duration_seconds", raw.get("duration_ms", 0.0)) / 1000.0)
                if "duration_ms" in raw
                else float(raw.get("duration_seconds", 0.0)),
                flops=flops,
                bytes_dram=bytes_dram,
                bytes_l2=bytes_l2,
                bytes_sram=bytes_sram,
                stream_id=int(raw.get("stream_id", 0)),
            )
        )
    _log.info("Parsed %d kernels from %s", len(out), path)
    return out


def parse_simple_csv(path: str | Path) -> list[KernelTrace]:
    """Parse a simple-format CSV kernel trace.

    Expected columns (with header row):

    ``kernel_name, start_ms, duration_ms, flops, bytes_dram, bytes_l2,
    bytes_sram, stream_id``

    Parameters
    ----------
    path
        Path to the CSV file.

    Returns
    -------
    list of KernelTrace
        Parsed kernel records.

    Raises
    ------
    ProfilerError
        On file-not-found or missing required columns.
    """
    path = Path(path)
    if not path.is_file():
        raise ProfilerError(f"CSV not found: {path}")
    required = {
        "kernel_name",
        "start_ms",
        "duration_ms",
        "flops",
        "bytes_dram",
        "bytes_l2",
        "bytes_sram",
        "stream_id",
    }
    out: list[KernelTrace] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            missing = required - set(reader.fieldnames or [])
            raise ProfilerError(f"CSV missing required columns: {sorted(missing)}")
        for row in reader:
            out.append(
                KernelTrace(
                    name=row["kernel_name"],
                    start_seconds=float(row["start_ms"]) / 1000.0,
                    duration_seconds=float(row["duration_ms"]) / 1000.0,
                    flops=float(row["flops"]),
                    bytes_dram=float(row["bytes_dram"]),
                    bytes_l2=float(row["bytes_l2"]),
                    bytes_sram=float(row["bytes_sram"]),
                    stream_id=int(row["stream_id"]),
                )
            )
    _log.info("Parsed %d kernels from %s", len(out), path)
    return out
