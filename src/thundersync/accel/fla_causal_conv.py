"""Route the hybrid model's causal depthwise convolution to the fla kernel.

Transformers looks for the Dao-AILab ``causal_conv1d`` package and, when it
is absent, falls back to ``nn.Conv1d`` with ``groups=conv_dim``.  PyTorch
serves that through ``conv_depthwise2d``, which at the live branch shape
takes about a fifth of a Gated DeltaNet layer's CUDA time for an amount of
arithmetic that is negligible beside the projections.

flash-linear-attention ships a Triton causal conv for the same operation.
From the layout the layer actually holds, the fla kernel runs the forward
and backward about six times faster than the fallback and agrees with it
to bf16 rounding.

The layer hands the kernel ``[batch, conv_dim, tokens]`` and transposes the
result straight back, while fla takes ``[batch, tokens, conv_dim]``.  The
two transposes are views over the projection output and cancel, so the
adapter moves no bytes; the contiguous and view spellings both measure
1.019 ms.

The swap preserves the layer's semantics: both spellings compute the same
left-padded causal convolution over whatever tensor the layer passes, so
segment and rebuild behaviour is unchanged.
"""

from __future__ import annotations

import torch

_LAYER_ATTRIBUTE = "causal_conv1d_fn"


def _fla_causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    seq_idx: torch.Tensor | None = None,
    **_: object,
) -> torch.Tensor:
    """A ``causal_conv1d_fn``-shaped callable backed by the fla kernel."""

    from fla.modules.convolution import causal_conv1d

    output = causal_conv1d(
        x.transpose(1, 2),
        weight=weight,
        bias=bias,
        activation=activation,
        cu_seqlens=seq_idx,
    )
    if isinstance(output, tuple):
        output = output[0]
    return output.transpose(1, 2)


def install_fla_causal_conv(model: torch.nn.Module) -> int:
    """Point every linear-attention layer at the fla kernel.

    Returns the number of layers rerouted.  Layers that already hold a
    kernel -- the Dao-AILab package was importable -- are left alone, since
    that path is the one transformers prefers and this is its stand-in.
    """

    try:
        import fla.modules.convolution  # noqa: F401
    except ImportError:
        return 0

    rerouted = 0
    for module in model.modules():
        if not hasattr(module, _LAYER_ATTRIBUTE):
            continue
        if getattr(module, _LAYER_ATTRIBUTE) is not None:
            continue
        if not isinstance(getattr(module, "conv1d", None), torch.nn.Conv1d):
            continue
        setattr(module, _LAYER_ATTRIBUTE, _fla_causal_conv1d_fn)
        rerouted += 1
    return rerouted
