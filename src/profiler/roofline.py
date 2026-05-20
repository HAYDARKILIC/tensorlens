r"""Roofline performance model.

The roofline model (Williams, Waterman, Patterson, 2009) bounds attainable
performance of a workload with arithmetic intensity :math:`I` (FLOPs per
byte of off-chip traffic) by

.. math::

    P_{\mathrm{attainable}}(I) \;=\; \min\!\bigl( P_{\mathrm{peak}}, \beta \cdot I \bigr)

where :math:`P_{\mathrm{peak}}` is the device peak FLOP/s and :math:`\beta`
is the bandwidth of the slowest memory tier that the workload touches. The
**ridge point** :math:`I^* = P_{\mathrm{peak}} / \beta` separates the
memory-bound regime (:math:`I < I^*`) from the compute-bound regime
(:math:`I > I^*`).

A multi-tier extension considers each level of the memory hierarchy:

.. math::

    P(I) \;=\; \min\bigl( P_{\mathrm{peak}}, \min_k \beta_k I_k \bigr),
    \quad I_k = \mathrm{FLOPs} / B_k

References
----------
* Williams, S., Waterman, A., Patterson, D. (2009). "Roofline: an insightful
  visual performance model for multicore architectures." CACM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..utils import ProfilerError, get_logger
from .log_parser import KernelTrace
from .memory_hierarchy import MemoryTier

__all__ = [
    "RoofineEntry",
    "RooflineModel",
    "arithmetic_intensity",
    "attainable_performance",
]

_log = get_logger(__name__)


@dataclass(frozen=True)
class RoofineEntry:
    """A point on the roofline plot.

    Attributes
    ----------
    name
        Kernel name.
    arithmetic_intensity
        FLOPs / bytes (DRAM-relative by default).
    achieved_flops_per_sec
        Measured throughput.
    duration_seconds
        Kernel wall time.
    """

    name: str
    arithmetic_intensity: float
    achieved_flops_per_sec: float
    duration_seconds: float


def arithmetic_intensity(kernel: KernelTrace, *, tier: str = "dram") -> float:
    """Return the arithmetic intensity of a kernel at one memory tier.

    Parameters
    ----------
    kernel
        Kernel trace record.
    tier
        One of ``"dram"``, ``"l2"``, or ``"sram"``.

    Returns
    -------
    float
        FLOPs per byte transferred through that tier. Returns ``+inf`` if the
        tier saw zero bytes but the kernel did FLOP work; ``0`` if no work.

    Raises
    ------
    ProfilerError
        On unknown tier.
    """
    if tier == "dram":
        bytes_ = kernel.bytes_dram
    elif tier == "l2":
        bytes_ = kernel.bytes_l2
    elif tier == "sram":
        bytes_ = kernel.bytes_sram
    else:
        raise ProfilerError(f"Unknown tier '{tier}'; expected 'dram', 'l2', or 'sram'")
    if bytes_ <= 0:
        return float("inf") if kernel.flops > 0 else 0.0
    return kernel.flops / bytes_


def attainable_performance(
    intensity: float,
    peak_flops_per_sec: float,
    bandwidth_bytes_per_sec: float,
) -> float:
    """Return :math:`\\min(P_{\\mathrm{peak}}, \\beta \\cdot I)`.

    Parameters
    ----------
    intensity
        Arithmetic intensity in FLOP/B.
    peak_flops_per_sec
        Device peak FLOP/s.
    bandwidth_bytes_per_sec
        Memory bandwidth in bytes/s.

    Returns
    -------
    float
        Attainable performance in FLOP/s.
    """
    return float(min(peak_flops_per_sec, bandwidth_bytes_per_sec * intensity))


class RooflineModel:
    """Multi-tier roofline model.

    Parameters
    ----------
    peak_flops_per_sec
        Device peak floating-point throughput.
    hierarchy
        Ordered list of memory tiers (slow to fast or vice versa; order is
        irrelevant for ``min`` operations).
    device_name
        Human-readable device label.
    """

    def __init__(
        self,
        *,
        peak_flops_per_sec: float,
        hierarchy: list[MemoryTier],
        device_name: str = "device",
    ) -> None:
        if peak_flops_per_sec <= 0:
            raise ProfilerError("peak_flops_per_sec must be positive")
        if not hierarchy:
            raise ProfilerError("hierarchy cannot be empty")
        self.peak_flops_per_sec = peak_flops_per_sec
        self.hierarchy = hierarchy
        self.device_name = device_name

    def ridge_points(self) -> dict[str, float]:
        """Return ridge points :math:`I^* = P_{\\mathrm{peak}} / \\beta_k` per tier."""
        return {
            t.name: t.ridge_point_flops_per_byte(self.peak_flops_per_sec)
            for t in self.hierarchy
        }

    def evaluate_kernel(
        self,
        kernel: KernelTrace,
        *,
        tier: str = "dram",
    ) -> RoofineEntry:
        """Evaluate one kernel against this model.

        Parameters
        ----------
        kernel
            Kernel trace record.
        tier
            Memory tier to use for arithmetic intensity.

        Returns
        -------
        RoofineEntry
            Roofline plot entry.
        """
        ai = arithmetic_intensity(kernel, tier=tier)
        achieved = kernel.achieved_flops_per_sec
        return RoofineEntry(
            name=kernel.name,
            arithmetic_intensity=ai,
            achieved_flops_per_sec=achieved,
            duration_seconds=kernel.duration_seconds,
        )

    def evaluate(
        self,
        kernels: list[KernelTrace],
        *,
        tier: str = "dram",
    ) -> list[RoofineEntry]:
        """Evaluate a list of kernels."""
        return [self.evaluate_kernel(k, tier=tier) for k in kernels]

    def render(
        self,
        kernels: list[KernelTrace] | None = None,
        *,
        tier: str = "dram",
        ai_range: tuple[float, float] = (1e-2, 1e3),
        ax: Any = None,
    ) -> Any:
        """Render the roofline plot.

        Parameters
        ----------
        kernels
            Optional list of kernels to plot as scatter points.
        tier
            Memory tier for arithmetic intensity.
        ai_range
            X-axis (arithmetic intensity) span.
        ax
            Optional matplotlib axes; if ``None``, a new figure is created.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object for further styling.
        """
        # Imports are local to keep matplotlib optional at import time.
        import matplotlib.pyplot as plt  # noqa: PLC0415

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))
        ai_grid = np.logspace(np.log10(ai_range[0]), np.log10(ai_range[1]), 256)
        # Compute roof for each tier
        for tier_obj in self.hierarchy:
            roof = np.minimum(
                self.peak_flops_per_sec,
                tier_obj.bandwidth_bytes_per_sec * ai_grid,
            )
            ax.plot(ai_grid, roof, label=f"{tier_obj.name}")
        # Peak FLOPs ceiling (horizontal)
        ax.axhline(self.peak_flops_per_sec, linestyle="--", color="black", alpha=0.5)
        # Scatter kernels
        if kernels is not None:
            xs = []
            ys = []
            labels = []
            for k in kernels:
                ai = arithmetic_intensity(k, tier=tier)
                if not np.isfinite(ai) or ai <= 0:
                    continue
                xs.append(ai)
                ys.append(k.achieved_flops_per_sec)
                labels.append(k.name)
            ax.scatter(xs, ys, s=40, edgecolors="black")
            for x, y, l in zip(xs, ys, labels, strict=True):
                ax.annotate(l, (x, y), fontsize=8, alpha=0.7,
                            xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Arithmetic Intensity (FLOP / byte)")
        ax.set_ylabel("Performance (FLOP / s)")
        ax.set_title(f"Roofline — {self.device_name} (tier={tier})")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        return ax
