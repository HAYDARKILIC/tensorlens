r"""Raw PyTorch forward-hook capture for the residual stream.

In the Elhage et al. (2021) framework, the transformer is decomposed as

.. math::

    x_{\ell+1} \;=\; x_\ell \;+\; \sum_{h=1}^{H} A^{(\ell, h)}(x_\ell)
                                  \;+\; M^{(\ell)}(x_\ell)

where each attention head :math:`A^{(\ell, h)}` and each MLP block
:math:`M^{(\ell)}` writes a vector to the shared **residual stream**. This
module installs ``register_forward_hook`` callbacks on selected submodules
and records their outputs into a typed :class:`StreamRecord`.

The capture engine is deliberately library-agnostic: it operates on any
``torch.nn.Module`` and accepts user-supplied selector functions that
identify which submodules constitute "attention heads" vs. "MLPs". This
avoids hardwiring to a particular HuggingFace-style API.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from ..utils import MechanisticError, get_logger

__all__ = ["HookHandle", "ResidualStreamCapture", "StreamRecord"]

_log = get_logger(__name__)


@dataclass
class StreamRecord:
    """Captured residual-stream contributions from a single forward pass.

    Attributes
    ----------
    attention_writes
        Mapping ``(layer, head) -> tensor`` of shape ``(batch, seq, d_model)``
        for each attention head's per-token write.
    mlp_writes
        Mapping ``layer -> tensor`` of shape ``(batch, seq, d_model)`` for
        each MLP block's write.
    layer_outputs
        Mapping ``layer -> tensor`` of the running residual stream after
        each layer, shape ``(batch, seq, d_model)``.
    """

    attention_writes: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)
    mlp_writes: dict[int, torch.Tensor] = field(default_factory=dict)
    layer_outputs: dict[int, torch.Tensor] = field(default_factory=dict)

    def attention_write_norms(self) -> dict[tuple[int, int], torch.Tensor]:
        """Return L2 norms of each head's write, summed over the token axis.

        Behaves as follows depending on tensor rank (after the feature dim):

        * ``(..., seq, d_model)`` → returns ``(...,)`` (norm then sum over seq).
        * ``(..., d_model)``      → returns ``(...,)`` (norm over the feature
          dim only — there is no sequence axis to reduce).

        Returns
        -------
        dict
            Mapping ``(layer, head) -> tensor``.
        """
        out: dict[tuple[int, int], torch.Tensor] = {}
        for key, t in self.attention_writes.items():
            norms = t.norm(dim=-1)
            if norms.dim() >= 1:
                norms = norms.sum(dim=-1) if norms.dim() > 1 else norms
            out[key] = norms
        return out

    def mlp_write_norms(self) -> dict[int, torch.Tensor]:
        """Return L2 norms of each MLP write, summed over the token axis when present.

        Mirrors the rank-handling of :meth:`attention_write_norms`.
        """
        out: dict[int, torch.Tensor] = {}
        for layer, t in self.mlp_writes.items():
            norms = t.norm(dim=-1)
            if norms.dim() >= 1:
                norms = norms.sum(dim=-1) if norms.dim() > 1 else norms
            out[layer] = norms
        return out

    def summary(self) -> str:
        """Return a human-readable summary string."""
        n_att = len(self.attention_writes)
        n_mlp = len(self.mlp_writes)
        return f"StreamRecord(attention_writes={n_att}, mlp_writes={n_mlp})"


@dataclass
class HookHandle:
    """Container for a removable forward hook.

    Attributes
    ----------
    module
        Module the hook was attached to.
    handle
        The ``torch.utils.hooks.RemovableHandle`` returned by PyTorch.
    label
        Human-readable identifier (e.g. ``"layer.3.attn.head.5"``).
    """

    module: nn.Module
    handle: torch.utils.hooks.RemovableHandle
    label: str

    def remove(self) -> None:
        """Detach the hook."""
        self.handle.remove()


# Type aliases
AttentionSelector = Callable[[nn.Module, str], tuple[int, int] | None]
"""Returns ``(layer, head)`` if the module is a head output, else ``None``."""

MlpSelector = Callable[[nn.Module, str], int | None]
"""Returns ``layer`` if the module is an MLP output, else ``None``."""

LayerSelector = Callable[[nn.Module, str], int | None]
"""Returns ``layer`` if the module is a full transformer block, else ``None``."""


class ResidualStreamCapture:
    r"""Install hooks to capture per-head, per-MLP residual stream writes.

    Usage
    -----
    >>> capture = ResidualStreamCapture(
    ...     model,
    ...     attention_head_selector=my_head_selector,
    ...     mlp_selector=my_mlp_selector,
    ... )
    >>> with capture:
    ...     _ = model(input_ids)
    >>> record = capture.record  # StreamRecord

    Parameters
    ----------
    model
        Any ``torch.nn.Module`` to instrument.
    attention_head_selector
        Function ``(module, name) -> (layer, head) | None`` identifying head
        output modules. Returning ``None`` skips the module.
    mlp_selector
        Function ``(module, name) -> layer | None`` identifying MLP output
        modules.
    layer_selector
        Optional function identifying whole-layer output modules whose
        forward-output equals :math:`x_{\ell+1}`.
    keep_on_device
        If False (default), detach and move captured tensors to CPU to keep
        VRAM footprint low.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        attention_head_selector: AttentionSelector | None = None,
        mlp_selector: MlpSelector | None = None,
        layer_selector: LayerSelector | None = None,
        keep_on_device: bool = False,
    ) -> None:
        if attention_head_selector is None and mlp_selector is None and layer_selector is None:
            raise MechanisticError(
                "At least one selector (attention/MLP/layer) must be supplied"
            )
        self._model = model
        self._att_sel = attention_head_selector
        self._mlp_sel = mlp_selector
        self._lyr_sel = layer_selector
        self._keep_device = keep_on_device
        self._handles: list[HookHandle] = []
        self.record = StreamRecord()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def install(self) -> None:
        """Walk the module tree and attach hooks at matching submodules."""
        if self._handles:
            raise MechanisticError("Hooks already installed; call remove() first")
        for name, module in self._model.named_modules():
            self._maybe_attach_attention_hook(module, name)
            self._maybe_attach_mlp_hook(module, name)
            self._maybe_attach_layer_hook(module, name)
        _log.info("Installed %d hooks", len(self._handles))

    def remove(self) -> None:
        """Detach all installed hooks.

        The captured :attr:`record` is **preserved** so callers can use the
        common ``with capture: ... ; analyze(capture.record)`` pattern.
        Call :meth:`reset` to explicitly clear the record.
        """
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def reset(self) -> None:
        """Clear the captured record without touching the hooks."""
        self.record = StreamRecord()

    def __enter__(self) -> "ResidualStreamCapture":
        self.install()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.remove()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detach(self, t: torch.Tensor) -> torch.Tensor:
        out = t.detach()
        if not self._keep_device:
            out = out.cpu()
        return out

    def _maybe_attach_attention_hook(self, module: nn.Module, name: str) -> None:
        if self._att_sel is None:
            return
        result = self._att_sel(module, name)
        if result is None:
            return
        layer, head = result

        def _hook(_mod: nn.Module, _inp: tuple[Any, ...], out: Any, layer: int = layer, head: int = head) -> None:
            tensor = _extract_tensor(out)
            self.record.attention_writes[(layer, head)] = self._detach(tensor)

        h = module.register_forward_hook(_hook)
        self._handles.append(HookHandle(module=module, handle=h, label=f"attn[L{layer}.H{head}]"))

    def _maybe_attach_mlp_hook(self, module: nn.Module, name: str) -> None:
        if self._mlp_sel is None:
            return
        result = self._mlp_sel(module, name)
        if result is None:
            return
        layer = result

        def _hook(_mod: nn.Module, _inp: tuple[Any, ...], out: Any, layer: int = layer) -> None:
            tensor = _extract_tensor(out)
            self.record.mlp_writes[layer] = self._detach(tensor)

        h = module.register_forward_hook(_hook)
        self._handles.append(HookHandle(module=module, handle=h, label=f"mlp[L{layer}]"))

    def _maybe_attach_layer_hook(self, module: nn.Module, name: str) -> None:
        if self._lyr_sel is None:
            return
        result = self._lyr_sel(module, name)
        if result is None:
            return
        layer = result

        def _hook(_mod: nn.Module, _inp: tuple[Any, ...], out: Any, layer: int = layer) -> None:
            tensor = _extract_tensor(out)
            self.record.layer_outputs[layer] = self._detach(tensor)

        h = module.register_forward_hook(_hook)
        self._handles.append(HookHandle(module=module, handle=h, label=f"layer[L{layer}]"))


def _extract_tensor(out: Any) -> torch.Tensor:
    """Coerce a forward-output into a single Tensor.

    Modules return either a tensor or a tuple whose first element is a
    tensor. This helper isolates that idiom.

    Raises
    ------
    MechanisticError
        If the output is neither a tensor nor a tuple-with-tensor-first.
    """
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, tuple) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    raise MechanisticError(
        f"Cannot extract tensor from forward output of type {type(out).__name__}"
    )


@contextlib.contextmanager
def temporary_capture(
    model: nn.Module,
    **kwargs: Any,
) -> Iterator[ResidualStreamCapture]:
    """Context-manager wrapper around :class:`ResidualStreamCapture`.

    Parameters
    ----------
    model
        Module to instrument.
    **kwargs
        Forwarded to :class:`ResidualStreamCapture`.

    Yields
    ------
    ResidualStreamCapture
        The active capture object; access ``capture.record`` after the
        forward pass.
    """
    cap = ResidualStreamCapture(model, **kwargs)
    cap.install()
    try:
        yield cap
    finally:
        cap.remove()
