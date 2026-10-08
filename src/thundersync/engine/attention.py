"""Attention over a segment forest, without materialising a mask.

A shared forward is only correct if a query in one branch cannot see another
branch. Building a `[P, P]` visibility mask is quadratic in memory and defeats
the purpose of the shared representation.

The way out is that the mask is not arbitrary. `plan.key_ranges[n]` says a query
in node `n` sees a few *contiguous* spans: every ancestor in full, then its own
span causally. Concatenate those spans and the visibility pattern becomes exactly
**bottom-right-aligned causal**: with `S` keys and `T` queries where the queries
are the last `T` keys, query `i` sees keys `0 .. S-T+i` -- all ancestors, plus its
own span up to itself. One dense kernel call per node, no mask tensor.

The alignment is the whole trick and it is easy to get wrong. PyTorch's
`is_causal=True` aligns **upper-left** when `kv_len > q_len`, which silently
gives each query the wrong window while still running fast.
`causal_lower_right` is the correct bias.

Rejected alternative, recorded because it looks better than it is: computing each
span separately and combining by the flash log-sum-exp identity avoids copying
ancestor K/V. It is exact in the forward, but `aten::_scaled_dot_product_flash_attention`
returns its `logsumexp` with **no `grad_fn`** -- so the merge weights are treated
as constants and the backward silently loses the `dw_i/dq` term. Same loss,
different gradient. The gather below costs some memory and is differentiable by
construction.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right

from thundersync.engine.plan import ExecutionPlan

_COMPILED_FLEX_ATTENTION = None
_ATTENTION_CHECKPOINT_MODE: ContextVar[Any] = ContextVar(
    "forest_attention_checkpoint_mode", default=None
)


@contextmanager
def attention_checkpoint_mode(mode: Any) -> Iterator[None]:
    """Apply a checkpoint dispatch mode only inside native attention calls."""
    token = _ATTENTION_CHECKPOINT_MODE.set(mode)
    try:
        yield
    finally:
        _ATTENTION_CHECKPOINT_MODE.reset(token)


def _forest_sdpa(query, key, value, **kwargs):
    # The caller constructs CausalBias before entering the dispatch mode;
    # constructing that metadata Tensor subclass inside a mode is unsupported.
    mode = _ATTENTION_CHECKPOINT_MODE.get()
    if mode is None:
        return F.scaled_dot_product_attention(query, key, value, **kwargs)
    with mode:
        return F.scaled_dot_product_attention(query, key, value, **kwargs)


def _physical_flex_attention_fn():
    """Compile FlexAttention once; shape-specific kernels cache after warmup."""
    global _COMPILED_FLEX_ATTENTION
    if _COMPILED_FLEX_ATTENTION is None:
        try:
            from torch.nn.attention.flex_attention import flex_attention
        except ImportError as exc:  # pragma: no cover - torch-version dependent
            raise RuntimeError(
                "physical forest attention requires torch FlexAttention"
            ) from exc
        _COMPILED_FLEX_ATTENTION = (
            torch.compile(flex_attention, dynamic=False)
            if hasattr(torch, "compile")
            else flex_attention
        )
    return _COMPILED_FLEX_ATTENTION


def physical_forest_mask_mod(plan: ExecutionPlan):
    """FlexAttention predicate for one DFS-packed physical prompt forest.

    A physical key is visible exactly when it precedes the query in DFS order
    and the query remains inside that key's contiguous subtree.  Logical RoPE
    positions remain in ``plan.position_ids``; physical order is used only for
    the ancestry relation.
    """
    subtree_end = plan.subtree_end

    def mask_mod(b, h, q_idx, kv_idx):  # noqa: ARG001 - FlexAttention API
        return (kv_idx <= q_idx) & (q_idx < subtree_end[kv_idx])

    return mask_mod


def physical_forest_block_mask(plan: ExecutionPlan):
    """Build the exact ancestry-causal BlockMask for the physical stream."""
    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError as exc:  # pragma: no cover - torch-version dependent
        raise RuntimeError(
            "physical forest attention requires torch FlexAttention"
        ) from exc
    p = plan.n_physical
    return create_block_mask(
        physical_forest_mask_mod(plan),
        B=None,
        H=None,
        Q_LEN=p,
        KV_LEN=p,
        device=plan.token_ids.device,
    )


def physical_forest_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: ExecutionPlan,
    *,
    scale: float | None = None,
    block_mask=None,
) -> torch.Tensor:
    """One FlexAttention call over the complete physical prompt forest.

    ``q`` is ``[P,H,D]`` and ``k``/``v`` are ``[P,Hkv,D]``.  The implementation
    currently expands grouped K/V heads before the kernel because FlexAttention
    has no stable public ``enable_gqa`` argument across the supported versions.
    This is a feature-gated dense-decoder reference backend; the existing
    node-wise SDPA path remains the default.
    """
    flex_attention = _physical_flex_attention_fn()
    h, hkv = q.shape[1], k.shape[1]
    if h % hkv:
        raise ValueError(f"query heads {h} are not divisible by KV heads {hkv}")
    if h != hkv:
        repeats = h // hkv
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
    if block_mask is None:
        if plan.physical_block_mask is None:
            plan.physical_block_mask = physical_forest_block_mask(plan)
        mask = plan.physical_block_mask
    else:
        mask = block_mask
    out = flex_attention(
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0),
        block_mask=mask,
        scale=scale,
    )
    return out[0].transpose(0, 1)


def _coalesce(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge adjacent spans. The common case collapses to a single slice.

    A node laid out immediately after its parent -- the first child on every
    DFS branch, which includes the whole trunk-to-first-branch path -- has
    ancestor and own spans that abut, so no gather is needed at all.
    """
    out: list[tuple[int, int]] = []
    for s, e in spans:
        if out and out[-1][1] == s:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def _keys(t: torch.Tensor, spans: list[tuple[int, int]]) -> torch.Tensor:
    """Ancestor + own K (or V) as one tensor, sliced when possible."""
    if len(spans) == 1:
        s, e = spans[0]
        return t[s:e]
    return torch.cat([t[s:e] for s, e in spans], dim=0)


def forest_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: ExecutionPlan,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Attention respecting the forest's visibility, on the physical stream.

    `q` is `[P, H, D]`, `k`/`v` are `[P, Hkv, D]`, all indexed by physical
    position. Returns `[P, H, D]`.

    Equivalent, position for position, to running each trajectory separately
    under a causal mask, checked against
    both a dense reference and an independent per-trajectory forward, in value
    and in gradient.
    """
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    gqa = q.shape[1] != k.shape[1]

    order = sorted(
        (n for n in range(plan.n_nodes)
         if int(plan.node_end[n]) > int(plan.node_start[n])),
        key=lambda n: int(plan.node_start[n]),
    )

    pieces: list[torch.Tensor] = []
    for n in order:
        s, e = int(plan.node_start[n]), int(plan.node_end[n])
        spans = _coalesce(plan.key_ranges[n])
        kn, vn = _keys(k, spans), _keys(v, spans)
        o = _forest_sdpa(
            q[s:e].transpose(0, 1).unsqueeze(0),
            kn.transpose(0, 1).unsqueeze(0),
            vn.transpose(0, 1).unsqueeze(0),
            attn_mask=causal_lower_right(e - s, kn.shape[0]),
            scale=scale,
            enable_gqa=gqa,
        )
        pieces.append(o[0].transpose(0, 1))

    # Nodes tile [0, P) and `order` is by start, so concatenation reproduces the
    # physical layout. Built rather than assigned in place: an in-place write
    # into a tensor that requires grad is a needless autograd hazard.
    return torch.cat(pieces, dim=0)


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: ExecutionPlan,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Dense `[P, P]`-masked attention. Correctness reference only.

    It exists so `forest_attention` can be checked against a direct reference
    and must never be used on a training path.
    """
    from thundersync.engine.plan import visibility_mask

    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    mask = visibility_mask(plan)
    h, hkv = q.shape[1], k.shape[1]
    if h != hkv:
        k = k.repeat_interleave(h // hkv, dim=1)
        v = v.repeat_interleave(h // hkv, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale
    scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    return torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), v.float()).to(q.dtype)
