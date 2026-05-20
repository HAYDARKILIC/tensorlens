r"""GPU memory hierarchy models.

Each :class:`MemoryTier` describes one level of the GPU memory hierarchy
with capacity, peak bandwidth, and latency. These models drive the roofline
ceilings and inform whether a workload is memory-bound on one tier versus
compute-bound at the device level.

For a workload moving :math:`B` bytes through a tier with bandwidth
:math:`\beta_k` (B/s), the time spent in that tier is

.. math::

    t_k \;=\; \frac{B_k}{\beta_k}

and the device-wide attainable performance is bounded by

.. math::

    P_{\mathrm{attainable}}(I) \;=\; \min\!\Bigl( P_{\mathrm{peak}},
            \min_k \beta_k I_k\Bigr)

where :math:`I_k = \text{FLOPs} / B_k` is the per-tier arithmetic intensity.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "MemoryTier",
    "default_a100_hierarchy",
    "default_h100_hierarchy",
]


@dataclass(frozen=True)
class MemoryTier:
    """Specification of one memory tier.

    Attributes
    ----------
    name
        Human-readable identifier (e.g. ``"HBM3"``).
    capacity_bytes
        Tier capacity in bytes (per chip).
    bandwidth_bytes_per_sec
        Peak bandwidth in bytes / second.
    latency_seconds
        Approximate access latency in seconds.
    """

    name: str
    capacity_bytes: int
    bandwidth_bytes_per_sec: float
    latency_seconds: float

    def ridge_point_flops_per_byte(self, peak_flops_per_sec: float) -> float:
        """Return the FLOP/B ridge point at this tier.

        Parameters
        ----------
        peak_flops_per_sec
            Device peak FLOP/s.

        Returns
        -------
        float
            :math:`I^* = P_{\\mathrm{peak}} / \\beta_k`.
        """
        return float(peak_flops_per_sec) / max(self.bandwidth_bytes_per_sec, 1.0)


def default_a100_hierarchy() -> list[MemoryTier]:
    """Return a published A100 80 GB SXM memory hierarchy.

    Numbers are nominal and serve as illustrative defaults; real devices
    differ by SKU. Values from NVIDIA datasheets where available, otherwise
    standard public estimates.
    """
    return [
        MemoryTier(
            name="Register / L0",
            capacity_bytes=256 * 1024,
            bandwidth_bytes_per_sec=20e12,
            latency_seconds=1.0e-9,
        ),
        MemoryTier(
            name="SRAM (L1 / shared)",
            capacity_bytes=192 * 1024,
            bandwidth_bytes_per_sec=19e12,
            latency_seconds=30e-9,
        ),
        MemoryTier(
            name="L2 cache",
            capacity_bytes=40 * 1024 * 1024,
            bandwidth_bytes_per_sec=5e12,
            latency_seconds=200e-9,
        ),
        MemoryTier(
            name="HBM2e",
            capacity_bytes=80 * 1024**3,
            bandwidth_bytes_per_sec=2039e9,
            latency_seconds=500e-9,
        ),
        MemoryTier(
            name="Host DRAM (PCIe Gen4)",
            capacity_bytes=512 * 1024**3,
            bandwidth_bytes_per_sec=32e9,
            latency_seconds=10e-6,
        ),
    ]


def default_h100_hierarchy() -> list[MemoryTier]:
    """Return a published H100 80 GB SXM memory hierarchy.

    Nominal values; see NVIDIA H100 datasheet for exact figures.
    """
    return [
        MemoryTier(
            name="Register / L0",
            capacity_bytes=256 * 1024,
            bandwidth_bytes_per_sec=33e12,
            latency_seconds=1.0e-9,
        ),
        MemoryTier(
            name="SRAM (L1 / shared)",
            capacity_bytes=228 * 1024,
            bandwidth_bytes_per_sec=24e12,
            latency_seconds=30e-9,
        ),
        MemoryTier(
            name="L2 cache",
            capacity_bytes=50 * 1024 * 1024,
            bandwidth_bytes_per_sec=12e12,
            latency_seconds=200e-9,
        ),
        MemoryTier(
            name="HBM3",
            capacity_bytes=80 * 1024**3,
            bandwidth_bytes_per_sec=3350e9,
            latency_seconds=400e-9,
        ),
        MemoryTier(
            name="Host DRAM (PCIe Gen5)",
            capacity_bytes=1024 * 1024**3,
            bandwidth_bytes_per_sec=64e9,
            latency_seconds=10e-6,
        ),
    ]
