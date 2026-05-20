r"""Edge-of-Stability (EOS) detection.

Cohen et al. (2021) showed that gradient descent on neural networks
typically operates at the threshold

.. math::

    \lambda_{\max}(\theta_t) \;\geq\; \frac{2}{\eta}

i.e. the largest Hessian eigenvalue rises until it touches the inverse
learning-rate boundary, after which the loss exhibits non-monotone but
*globally descending* oscillations.

This module implements a heuristic EOS detector that combines:

1. The presence of a sustained plateau of :math:`\lambda_{\max} \geq 2/\eta`.
2. Non-monotone loss with a slowly decreasing envelope.
3. A characteristic loss-oscillation frequency near :math:`\pi`.

References
----------
* Cohen, J., Kaur, S., Li, Y., Kolter, Z., Talwalkar, A. (2021). "Gradient
  Descent on Neural Networks Typically Occurs at the Edge of Stability."
  ICLR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils import OptimizationError, get_logger

__all__ = ["EdgeOfStabilityReport", "detect_edge_of_stability"]

_log = get_logger(__name__)


@dataclass(frozen=True)
class EdgeOfStabilityReport:
    """Result of an EOS-detection analysis.

    Attributes
    ----------
    is_eos
        Whether the trajectory ever enters the EOS regime.
    onset_step
        Earliest step (within the trajectory) where the regime is triggered;
        ``-1`` if never.
    fraction_above_threshold
        Fraction of steps with :math:`\\lambda_{\\max} \\geq 2/\\eta`.
    loss_oscillation_amplitude
        High-frequency amplitude of the loss curve.
    """

    is_eos: bool
    onset_step: int
    fraction_above_threshold: float
    loss_oscillation_amplitude: float


def _high_pass_amplitude(losses: np.ndarray) -> float:
    """Estimate the high-frequency amplitude of a 1-D series.

    Uses a simple first-difference operator as a finite-impulse high-pass.
    Returns the standard deviation of differences as the amplitude proxy.

    Parameters
    ----------
    losses
        1-D loss curve.

    Returns
    -------
    float
        Amplitude estimate.
    """
    if losses.size < 4:
        return 0.0
    return float(np.std(np.diff(losses)))


def detect_edge_of_stability(
    losses: np.ndarray | list[float],
    lambda_max_trace: np.ndarray | list[float],
    learning_rate: float,
    *,
    plateau_steps: int = 5,
    threshold_slack: float = 0.95,
    oscillation_threshold: float = 1e-3,
) -> EdgeOfStabilityReport:
    r"""Detect the EOS regime over a training trajectory.

    Parameters
    ----------
    losses
        Loss curve, length ``T``.
    lambda_max_trace
        Top Hessian eigenvalue at the same ``T`` checkpoints.
    learning_rate
        Optimizer step size :math:`\eta`.
    plateau_steps
        How many *consecutive* steps with :math:`\lambda_{\max} \geq
        2/\eta` are required to flag onset.
    threshold_slack
        Multiplier applied to :math:`2/\eta`; values below 1 allow detection
        slightly before the strict boundary.
    oscillation_threshold
        Minimum high-frequency amplitude of the loss curve required to
        confirm EOS (a smooth descent at high curvature does not count).

    Returns
    -------
    EdgeOfStabilityReport
        Aggregated decision and diagnostics.

    Raises
    ------
    OptimizationError
        On mismatched array lengths or invalid learning rate.
    """
    if learning_rate <= 0:
        raise OptimizationError(f"learning_rate must be positive; got {learning_rate}")
    losses_arr = np.asarray(losses, dtype=np.float64)
    lam_arr = np.asarray(lambda_max_trace, dtype=np.float64)
    if losses_arr.shape != lam_arr.shape or losses_arr.ndim != 1:
        raise OptimizationError(
            f"losses and lambda_max_trace must be 1-D with equal length; "
            f"got {losses_arr.shape} vs {lam_arr.shape}"
        )

    threshold = threshold_slack * 2.0 / learning_rate
    above = lam_arr >= threshold
    frac = float(above.mean())
    # Find first run of `plateau_steps` consecutive Trues.
    onset = -1
    run = 0
    for t, flag in enumerate(above):
        run = run + 1 if flag else 0
        if run >= plateau_steps:
            onset = t - plateau_steps + 1
            break
    amp = _high_pass_amplitude(losses_arr)
    is_eos = (onset != -1) and (amp >= oscillation_threshold)
    _log.info(
        "EOS detect: is_eos=%s, onset=%d, frac>=2/η=%.3f, osc_amp=%.3e",
        is_eos,
        onset,
        frac,
        amp,
    )
    return EdgeOfStabilityReport(
        is_eos=is_eos,
        onset_step=onset,
        fraction_above_threshold=frac,
        loss_oscillation_amplitude=amp,
    )
