"""FlashAttention 4 as a transformers attention implementation.

ThunderSync reaches FA4 through its own `_causal_sdpa`; a trainer that runs the
model's own attention would use the bundled FlashAttention-2 unless FA4 is
registered here.  Registering it lets both paths use the same kernel, so a
comparison between them measures scheduling rather than the attention kernel.

Registered under the name `fa4` so a model loads with
`attn_implementation="fa4"`.  Registration is a no-op when FA4 is not
importable, and the caller then falls back to whatever it asked for.
"""

from __future__ import annotations

from typing import Any

import torch

ATTENTION_NAME = "fa4"


def fa4_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Causal attention over one sequence, heads last on the way out.

    Takes ``[batch, heads, tokens, head_dim]`` as the interface passes it
    and returns ``[batch, tokens, heads, head_dim]``, which is the layout
    the caller reshapes.  FA4 carries grouped heads, so no expansion of
    key and value is needed.
    """

    from flash_attn.cute.interface import flash_attn_func

    if dropout:
        raise ValueError("attention dropout is not supported on this path")
    output = flash_attn_func(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        softmax_scale=scaling,
        causal=True,
    )
    if isinstance(output, tuple):
        output = output[0]
    return output, None


def register_fa4_attention() -> bool:
    """Register the implementation; return whether it is usable."""

    try:
        import flash_attn.cute.interface  # noqa: F401
    except Exception:
        return False
    from transformers.modeling_utils import AttentionInterface

    AttentionInterface.register(ATTENTION_NAME, fa4_attention_forward)
    return True
