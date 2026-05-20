"""Shared utilities for TensorLens.

Provides:

* Structured logging via :func:`get_logger`.
* Deterministic seeding across ``random``, ``numpy``, and ``torch``.
* A custom exception hierarchy used throughout the library.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from typing import Final

import numpy as np
import torch

__all__ = [
    "TensorLensError",
    "GeometryError",
    "MechanisticError",
    "OptimizationError",
    "ProfilerError",
    "get_logger",
    "set_global_seed",
    "describe_device",
]

# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TensorLensError(Exception):
    """Base exception for all TensorLens-raised errors."""


class GeometryError(TensorLensError):
    """Raised by ``src.geometry`` for shape / numerical issues."""


class MechanisticError(TensorLensError):
    """Raised by ``src.mechanistic`` for hook / SAE issues."""


class OptimizationError(TensorLensError):
    """Raised by ``src.optimization`` for landscape / Hessian issues."""


class ProfilerError(TensorLensError):
    """Raised by ``src.profiler`` for log-parsing / roofline issues."""


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
)
_DATE_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%S"
_LOG_LEVEL_ENV: Final[str] = "TENSORLENS_LOG_LEVEL"


def _resolve_level() -> int:
    """Resolve log level from the ``TENSORLENS_LOG_LEVEL`` env var.

    Returns
    -------
    int
        A ``logging`` numeric level. Defaults to ``logging.INFO`` if the
        environment variable is missing or unparseable.
    """
    raw = os.environ.get(_LOG_LEVEL_ENV, "INFO").strip().upper()
    level = logging.getLevelName(raw)
    if isinstance(level, int):
        return level
    return logging.INFO


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger.

    The logger uses a single ``StreamHandler`` writing to ``stderr`` with a
    consistent format. Propagation is disabled so library output does not
    bubble into a host application's root logger.

    Parameters
    ----------
    name
        Logger name; usually ``__name__`` of the caller.

    Returns
    -------
    logging.Logger
        Configured logger instance, idempotent across calls.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(_resolve_level())
        logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def set_global_seed(seed: int, *, deterministic_cudnn: bool = True) -> None:
    """Seed ``random``, ``numpy``, and ``torch`` for reproducibility.

    Parameters
    ----------
    seed
        Non-negative integer seed.
    deterministic_cudnn
        If True (default), force cuDNN into deterministic mode. This may
        reduce throughput on GPU but yields bit-exact reproducibility.

    Raises
    ------
    TensorLensError
        If ``seed`` is negative or exceeds 2**32 - 1.
    """
    if not 0 <= seed <= 2**32 - 1:
        raise TensorLensError(f"seed must lie in [0, 2**32 - 1]; got {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    get_logger(__name__).debug("Global seed set to %d (cudnn_det=%s)", seed, deterministic_cudnn)


# ---------------------------------------------------------------------------
# Device description
# ---------------------------------------------------------------------------


def describe_device(device: torch.device | str | None = None) -> str:
    """Return a human-readable description of the active compute device.

    Parameters
    ----------
    device
        Torch device specifier. If ``None``, selects ``cuda`` when available,
        otherwise ``cpu``.

    Returns
    -------
    str
        Description string such as ``"cuda:0 (NVIDIA A100-SXM4-80GB, 80 GB)"``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        total_gb = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        return f"cuda:{idx} ({name}, {total_gb:.0f} GB)"
    return str(device)
