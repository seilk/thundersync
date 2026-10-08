"""Streaming and close-time GRPO execution primitives.

The central invariant is that changing source release time does not change the
mathematical update.

Cross-chunk dependence flows through cached K/V and the boundary hidden state.
Both remain graph-attached until their consumers have executed. Chunked causal
attention retains per-chunk references and reconstructs transient ancestor
concatenations during backward, avoiding sum-of-prefixes graph retention.

`append_turns` can pack independent arrivals without changing their trajectory
graphs. With `stream_turns=False`, an append records tokens and
provenance at arrival and performs the reward-linear branch reconstruction only
when that trajectory source is legally admitted. Reconstruction is block
checkpointed, and group-level shared-prefix work waits for its actual group
dependencies. Deterministic source reduction occurs after execution and never
acts as an admission predicate.

Prompt boundary proxies must remain resident through group closure. Rebuilding
the prompt after branch gradients have been deposited would create different
proxy tensors and silently lose those gradients.
"""

from __future__ import annotations

import hashlib
import itertools
import collections
import json
import math
import os
import threading
import time
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right

from thundersync.accel.capability import kernel_runs
from thundersync.accel.fused_logprob import fused_logprob
from thundersync.accel.operator_profiling import (
    device_role,
    operator_profiling_enabled,
    operator_range,
)
from thundersync.engine.boundary_adjoint import (
    TRITON_MAX_DEGREE,
    reduce_boundary_adjoints,
)
from thundersync.engine.objective_ready import CanonicalGradientReducer
from thundersync.engine.step import assert_supported, base_model, hybrid_layer_kinds, unembedding


ForestBoundaryAdjointBackend = Literal["torch", "torch_reduce", "triton"]
BOUNDARY_ADJOINT_BACKENDS = frozenset(("torch", "torch_reduce", "triton"))
SharedPrefixVJPBackend = Literal[
    "sequential",
    "ready_frontier",
    "isolated_canonical",
]
SHARED_PREFIX_VJP_BACKENDS = frozenset(
    ("sequential", "ready_frontier", "isolated_canonical")
)
ParameterAdjointReduction = Literal["arrival", "canonical_source"]


@dataclass
class _Chunk:
    """One appended run of tokens: a group prompt, or one turn of one trajectory."""

    idx: int
    n_tokens: int
    parent: int | None  # chunk chain, root = the group's prompt


@dataclass
class _Seg:
    """One segment of a packed (possibly coalesced) forward: chunk `cid` of one
    trajectory, occupying `[start, start+length)` of the packed sequence dim
    and attending over `chain` = its own ancestor chunks plus itself."""

    cid: int
    chain: list[int]  # ancestors + own cid, chronological
    start: int
    length: int


class _StreamKV:
    """Per-layer, per-chunk K/V -- the retained graph tensors, not copies."""

    def __init__(self) -> None:
        self._store: dict[int, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}

    def put(self, layer: int, chunk: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self._store.setdefault(layer, {})[chunk] = (k, v)

    def get(self, layer: int, chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._store[layer][chunk]

    def replace(
        self, layer: int, chunk: int, k: torch.Tensor, v: torch.Tensor
    ) -> None:
        self._store[layer][chunk] = (k, v)

    def drop(self, chunks: list[int]) -> None:
        """Release the given chunks' K/V from every layer."""
        for per in self._store.values():
            for c in chunks:
                per.pop(c, None)

    def layers(self) -> list[int]:
        return sorted(self._store)

    def chunks(
        self, layer: int, chain: list[int]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Per-chunk K and V lists over the ancestor chain, chronological order.

        The stored graph tensors themselves, uncopied: `_ChunkedCausalSDPA`
        saves exactly these references for backward, so streaming retains each
        chunk's K/V once, not once per descendant event.
        """
        per = self._store[layer]
        ks, vs = zip(*(per[c] for c in chain), strict=True)
        return list(ks), list(vs)

    def gather(
        self, layer: int, chain: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """K/V over the ancestor chain in chronological order. [S, H, D] each.

        A differentiable `torch.cat`: whatever consumes the result makes
        autograd retain the concatenated copies. Used only by the concat
        reference attention (`StreamingRun(chunked_sdpa=False)`).
        """
        per = self._store[layer]
        ks, vs = zip(*(per[c] for c in chain), strict=True)
        return torch.cat(ks, dim=0), torch.cat(vs, dim=0)

    def bytes_resident(self) -> int:
        storages: dict[tuple[str, int], int] = {}
        for per in self._store.values():
            for k, v in per.values():
                for tensor in (k, v):
                    storage = tensor.untyped_storage()
                    storages[(str(tensor.device), storage.data_ptr())] = (
                        storage.nbytes()
                    )
        return sum(storages.values())

    def pairs(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Every retained (K, V) pair over all layers and chunks."""
        return [pair for per in self._store.values() for pair in per.values()]


# Bound while a chunk forward is in flight; the registered attention function
# selects it from the query device. Device-indexed module state remains visible
# to autograd engine threads and permits distinct teacher/student GPUs to
# execute concurrently. One active stream per device remains the contract.
_STREAMS: dict[torch.device, "StreamingRun"] = {}

# Every StreamingRun still alive, weakly. A run that outlives its step keeps
# everything it holds (K/V, ledgers, adjoint accumulators) resident;
# retained_device_bytes reports the count and the other live runs' K/V so
# retained memory can be attributed to its holder.
_LIVE_RUNS: "weakref.WeakSet[StreamingRun]" = weakref.WeakSet()


def live_runs() -> list["StreamingRun"]:
    """Diagnostic: the StreamingRun instances still referenced somewhere."""
    return list(_LIVE_RUNS)


def retained_bytes_by_attribute(
    owner: Any,
    device: torch.device,
    *,
    skip: tuple[str, ...] = (),
    depth_limit: int = 5,
) -> dict[str, int]:
    """Diagnostic: device bytes ``owner`` holds through its attributes, by name.

    Reflection over ``vars(owner)`` to a fixed depth, storage-deduplicated
    (views and aliases count once, in the first attribute that reaches
    them), following dicts, sequences and plain objects; modules, streaming
    runs, threads, locks and executors are not entered. A full-model-sized
    buffer that outlives a step is then named by the attribute holding it
    rather than inferred from arithmetic. Reflection, not a contract:
    attribute names are the owner's own.
    """

    seen: set[int] = set()
    totals: dict[str, int] = {}

    def visit(value: Any, key: str, depth: int) -> None:
        if depth > depth_limit:
            return
        if isinstance(value, torch.Tensor):
            if value.device != device:
                return
            storage = value.untyped_storage()
            pointer = storage.data_ptr()
            if pointer in seen or storage.nbytes() == 0:
                return
            seen.add(pointer)
            totals[key] = totals.get(key, 0) + storage.nbytes()
            if value.grad is not None:
                visit(value.grad, key + ".grad", depth)
            return
        if isinstance(value, (str, bytes, int, float, bool, type(None))):
            return
        if isinstance(value, (torch.nn.Module, StreamingRun)):
            return
        module = type(value).__module__ or ""
        if module.startswith(("threading", "concurrent", "queue", "_thread", "weakref")):
            return
        if isinstance(value, dict):
            for item in list(value.values()):
                visit(item, key, depth + 1)
        elif isinstance(value, (list, tuple, set, frozenset, collections.deque)):
            for item in list(value):
                visit(item, key, depth + 1)
        elif hasattr(value, "__dict__"):
            for item in list(vars(value).values()):
                visit(item, key, depth + 1)

    for name, value in list(vars(owner).items()):
        if name in skip:
            continue
        visit(value, name, 0)
    return {**dict(sorted(totals.items())), "accounted": sum(totals.values())}

def _device_long(values: list[int], device: torch.device) -> torch.Tensor:
    """Host integers as a long tensor on ``device``, without a host wait.

    ``torch.tensor(values, device=...)`` stages pageable memory and
    synchronizes the stream, so the host waits for every queued kernel
    before it can issue the next one. Pinned staging with a non-blocking
    copy keeps the device queue ahead of the host; the caching host
    allocator holds the staging block until the copy has run.
    """
    if device.type != "cuda":
        return torch.tensor(values, dtype=torch.long, device=device)
    staged = torch.tensor(values, dtype=torch.long, pin_memory=True)
    return staged.to(device, non_blocking=True)


# The close rebuild's plain transient estimate (`_plain_rebuild_fits`) counts
# the residual stream a few times over plus the MLP's gate and up
# projections per token per layer, which undercounts what the device retains
# by roughly a quarter, so the estimate is scaled to sit above the
# observation: choosing the plain path on an underestimate reaches CUDA out
# of memory. Turn appends, packs and replays read the turn memory rule below.
_PLAIN_REBUILD_ESTIMATE_MARGIN = 1.35
# `rebuild_logprobs`'s "use the run's own checkpoint storage" default.
_RUN_DEFAULT = object()
# Where the block rebuild's memory rule moves checkpoint inputs.
_HOST = torch.device("cpu")

# False forwards an unscored turn at arrival instead of deferring it to the
# next scored turn; a reference path kept for tests.
_DEFER_UNSCORED_TURNS = True

TURN_ACTIVATION_FACTOR = 1.15
_TURN_ESTIMATE_MARGIN = 1.10


def _allocator_block_bytes(size: int) -> int:
    """What the caching allocator counts for one allocation of ``size`` bytes.

    Allocations above 1 MiB come from 2 MiB-granular segments and keep a
    remainder of at most 1 MiB unsplit, so they count up to the next 2 MiB;
    smaller ones round up to 512 bytes.
    """

    granule = (2 << 20) if size > (1 << 20) else 512
    return -(-size // granule) * granule


def _require_cuda(device: torch.device) -> None:
    if torch.device(device).type != "cuda":
        raise ValueError(
            f"device memory accounting reads the CUDA caching allocator; got {device}"
        )


def device_memory_state(device: torch.device) -> dict[str, int]:
    """The allocator's counters and the memory a new allocation may use.

    CUDA devices only; any other device raises ValueError.

    ``available`` is the driver's free memory plus the bytes the caching
    allocator holds without an allocation in them, less those it cannot give
    back. When a request does not fit the driver's free memory, the allocator
    releases its wholly free cached segments (with expandable segments, the
    free pages of its segment) and retries, so those bytes serve a request
    of any size. A free block split off a segment that still holds a live
    allocation (``inactive_split_bytes``) can be neither released nor grown,
    so it serves only requests that fit inside it; it is not counted.
    """

    _require_cuda(device)
    stats = torch.cuda.memory_stats_as_nested_dict(device)
    free, _total = torch.cuda.mem_get_info(device)

    def read(key: str, field_name: str = "current") -> int:
        return int(stats.get(key, {}).get("all", {}).get(field_name, 0))

    releasable = max(
        0,
        read("reserved_bytes") - read("active_bytes") - read("inactive_split_bytes"),
    )
    return {
        "driver_free": int(free),
        "releasable_cached": releasable,
        "available": int(free) + releasable,
        "allocated": read("allocated_bytes"),
        "peak_allocated": read("allocated_bytes", "peak"),
        "peak_reserved": read("reserved_bytes", "peak"),
        "alloc_retries": int(stats.get("num_alloc_retries", 0)),
    }


def _reset_peak(device: torch.device) -> None:
    """Reset the allocator's peak counters so a memory window reads its own call."""

    _require_cuda(device)
    torch.cuda.reset_peak_memory_stats(device)


# `causal_lower_right` forces SDPA off its fastest backend. The visible region
# of a bottom-right causal block splits exactly into a fully visible prefix
# and a square causal tail, and the softmax over their union is recovered
# from the two log-sum-exp values, so both halves take the unmasked fast
# path. The split calls private ATen entry points whose availability is a
# runtime property; each split path checks it and falls back otherwise.
#
# Flash takes the split before cuDNN whenever it can take the shape. cuDNN's
# frontend builds an execution plan per call and its plan cache keys on the
# shape; a streamed turn changes the key length every call, so every call
# builds a plan on the host. Flash has no plan step.

# False makes the chunked attention op recompute the grouped flash split in
# backward instead of keeping its output and log-sum-exp; a reference path
# kept for tests.
_SAVED_STATE_FLASH_SPLIT = True

# False runs a plain turn pack's attention one segment at a time instead of
# through one `_PackedFlashSplit`; a reference path kept for tests.
_PACKED_FLASH_SPLIT = True

# How many attention calls each path served in this process, by name.
# Incremented through `count_attention_path`, which holds the lock; the
# Counter itself stays importable for callers outside this module.
_ATTENTION_PATH_COUNTS: collections.Counter[str] = collections.Counter()
_ATTENTION_PATH_COUNTS_LOCK = threading.Lock()


def count_attention_path(name: str, calls: int = 1) -> None:
    """Record ``calls`` attention calls served by path ``name``."""

    with _ATTENTION_PATH_COUNTS_LOCK:
        _ATTENTION_PATH_COUNTS[name] += calls


def attention_path_counts() -> dict[str, int]:
    """Diagnostic: attention calls served per path in this process, cumulative."""

    with _ATTENTION_PATH_COUNTS_LOCK:
        return dict(sorted(_ATTENTION_PATH_COUNTS.items()))


_CUDNN_SDPA = getattr(
    torch.ops.aten, "_scaled_dot_product_cudnn_attention", None
)
_CUDNN_SDPA_BACKWARD = getattr(
    torch.ops.aten, "_scaled_dot_product_cudnn_attention_backward", None
)
# The cuDNN SDPA backward rejects hidden dimensions above 128 ("Num
# hidden_dim should be less than or equal to 128"). Larger head dimensions
# take the flash split or the masked reference path instead.
CUDNN_SDPA_MAX_HEAD_DIM = 128

# False keeps the gated-delta layers' causal convolution on `F.conv1d`
# instead of the FLA kernel; a reference path kept for tests.
_LINEAR_ATTENTION_FLA_CONV = True

# False skips FA4 for the bottom-right causal attention whatever
# ``THUNDERSYNC_FA4`` says; a reference path kept for tests.
_SPLIT_BOTTOM_RIGHT_FA4 = True

# ``THUNDERSYNC_FA4`` selects the streaming attention kernel. ``require``
# fails the first dispatch (and every later one) while the FA4 build is not
# importable, ``off`` skips FA4 and takes the flash split, and ``auto`` --
# the value when the variable is unset -- takes FA4 when the import
# succeeds. `attention_path_counts` and `fa4_build()` report which kernel
# actually ran.
FA4_ENV = "THUNDERSYNC_FA4"
FA4_SETTINGS = ("require", "off", "auto")

_FA4_FUNC = None
_FA4_BUILD: str | None = None
_FA4_RESOLVED = False
# The setting the resolution read; None until a resolution completes.
_FA4_SETTING: str | None = None
_FA4_LOCK = threading.Lock()


def fa4_setting() -> str:
    """The ``THUNDERSYNC_FA4`` setting: ``require``, ``off`` or ``auto``."""

    value = os.environ.get(FA4_ENV, "auto").strip().lower() or "auto"
    if value not in FA4_SETTINGS:
        raise ValueError(
            f"{FA4_ENV} must be one of {', '.join(FA4_SETTINGS)}, not {value!r}"
        )
    return value


def _fa4_build_string(package: Any) -> str:
    """Name the installed FA4 build: ``<distribution> <version>``.

    A ``flash_attn`` installed as a namespace package has no
    ``__version__``; its distribution (``flash_attn_4``) still carries the
    version in its metadata, and ``flash_attn.cute`` carries one that falls
    back to ``0.0.0``. Resolved in that order; ``flash_attn unknown`` only
    when every source is silent.
    """
    import importlib.metadata as metadata

    version = getattr(package, "__version__", None)
    if isinstance(version, str) and version and version != "0.0.0":
        return f"flash_attn {version}"
    try:
        for distribution in metadata.packages_distributions().get("flash_attn", []):
            try:
                return f"{distribution} {metadata.version(distribution)}"
            except metadata.PackageNotFoundError:
                continue
    except (AttributeError, ValueError):
        pass
    try:
        from flash_attn import cute as cute_package
    except ImportError:
        return "flash_attn unknown"
    cute_version = getattr(cute_package, "__version__", None)
    if isinstance(cute_version, str) and cute_version and cute_version != "0.0.0":
        return f"flash_attn.cute {cute_version}"
    return "flash_attn unknown"


def _fa4_flash_attn_func():
    """The FA4 entry point, resolved once under ``THUNDERSYNC_FA4``, or None.

    Under ``require`` a failed import raises and leaves the resolution
    unmade, so the next dispatch raises again instead of falling to the
    flash split: a configuration that requires FA4 never runs another kernel.
    The resolution runs under a lock; resetting ``_FA4_RESOLVED`` makes the
    next call re-read the setting.
    """

    global _FA4_FUNC, _FA4_BUILD, _FA4_RESOLVED, _FA4_SETTING
    if _FA4_RESOLVED:
        return _FA4_FUNC
    with _FA4_LOCK:
        if _FA4_RESOLVED:
            return _FA4_FUNC
        setting = fa4_setting()
        if setting == "off":
            _FA4_FUNC = None
            _FA4_SETTING = setting
            _FA4_RESOLVED = True
            return None
        try:
            import flash_attn
            from flash_attn.cute.interface import flash_attn_func
        except Exception as error:
            if setting == "require":
                raise RuntimeError(
                    f"{FA4_ENV}=require but the FA4 build is not importable: {error!r}"
                ) from error
            _FA4_FUNC = None
        else:
            _FA4_FUNC = flash_attn_func
            _FA4_BUILD = _fa4_build_string(flash_attn)
        _FA4_SETTING = setting
        _FA4_RESOLVED = True
        return _FA4_FUNC


def fa4_build() -> str | None:
    """Diagnostic: the resolved FA4 build (``flash_attn <version>``), or None.

    Never raises: under ``require`` with no build it returns None and leaves
    the refusal to the dispatch.
    """

    try:
        _fa4_flash_attn_func()
    except RuntimeError:
        return None
    return _FA4_BUILD


def _fa4_bottom_right_causal(query, kt, vt, scale):
    """One FA4 call for the whole bottom-right branch attention, or None.

    Takes ``[1, H, T, D]`` as the rest of this module does and returns the
    same layout; FA4 wants heads last, and the transposes are views.  The
    prefix is implicit in the length difference, which is how the caller
    already derives it.
    """

    fn = _fa4_flash_attn_func()
    if fn is None:
        return None
    output = fn(
        query.transpose(1, 2),
        kt.transpose(1, 2),
        vt.transpose(1, 2),
        softmax_scale=scale,
        causal=True,
    )
    if isinstance(output, tuple):
        output = output[0]
    return output.transpose(1, 2)


def _fa4_bottom_right_is_available(query: torch.Tensor, prefix: int) -> bool:
    """FA4 carries grouped heads and head dimensions up to 256."""

    if not (
        _SPLIT_BOTTOM_RIGHT_FA4
        and prefix > 0
        and query.is_cuda
        and query.dtype in (torch.float16, torch.bfloat16)
        and _fa4_flash_attn_func() is not None
    ):
        return False
    # Under ``require`` a kernel that cannot run here must fail loudly at
    # the call, not fall through; every other setting asks the device. The
    # setting is the one the resolution read.
    if _FA4_SETTING == "require":
        return True
    return kernel_runs("fa4", query, query.shape[-1], _fa4_trial)


def trial_qkv(
    device: torch.device, dtype: torch.dtype, head_dim: int, *, heads: int, kv_heads: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A tiny bottom-right branch: 8 queries over a 16-token key window."""

    generator = torch.Generator(device=device).manual_seed(0)

    def make(shape):
        return torch.randn(shape, device=device, dtype=dtype, generator=generator).requires_grad_()

    return (
        make((1, heads, 8, head_dim)),
        make((1, kv_heads, 16, head_dim)),
        make((1, kv_heads, 16, head_dim)),
    )


def _fa4_trial(device: torch.device, dtype: torch.dtype, head_dim: int) -> None:
    query, key, value = trial_qkv(device, dtype, head_dim, heads=2, kv_heads=1)
    output = _fa4_bottom_right_causal(query, key, value, head_dim ** -0.5)
    torch.autograd.grad(output.float().sum(), (query, key, value))


def _fla_causal_conv_silu(channels_last, weight, bias):
    """``silu(causal_conv1d(x))`` from ``[1, D, T]``, returning ``[1, T, D]``.

    The two transposes are views over the projection output and cancel, so
    the layout adapter moves no bytes.  Returns None when the kernel is
    unavailable, leaving the caller on its fallback.
    """

    try:
        from fla.modules.convolution import causal_conv1d
    except ImportError:
        return None
    if not kernel_runs("fla_causal_conv", channels_last, weight.shape[-1], _fla_conv_trial):
        return None
    output = causal_conv1d(
        channels_last.transpose(1, 2), weight=weight, bias=bias,
        activation="silu",
    )
    return output[0] if isinstance(output, tuple) else output


def _fla_conv_trial(device: torch.device, dtype: torch.dtype, kernel_size: int) -> None:
    from fla.modules.convolution import causal_conv1d

    channels = 64
    x = torch.randn(1, 16, channels, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(channels, kernel_size, device=device, dtype=dtype, requires_grad=True)
    output = causal_conv1d(x, weight=weight, bias=None, activation="silu")
    output = output[0] if isinstance(output, tuple) else output
    torch.autograd.grad(output.float().sum(), (x, weight))


_FLASH_SDPA = getattr(
    torch.ops.aten, "_scaled_dot_product_flash_attention", None
)
_FLASH_SDPA_BACKWARD = getattr(
    torch.ops.aten, "_scaled_dot_product_flash_attention_backward", None
)


class _SplitBottomRightFlash(torch.autograd.Function):
    """The same split, on the flash kernel, for head dimensions cuDNN refuses.

    cuDNN's SDPA rejects hidden dimensions above 128, so without this split
    attention layers with a larger head dimension fall through to the
    masked math path, which materializes the mask and runs
    an unfused softmax.

    Flash takes no bias, so the bottom-right structure comes from the same
    decomposition the cuDNN split uses -- an unmasked prefix plus a causal
    square tail, merged by log-sum-exp -- and the backward runs each half's
    kernel against the merged output and log-sum-exp.  Flash has no grouped
    key/value path here, so the caller expands before calling; the expansion
    is a transient copy whose backward sums the gradient back over the group.
    """

    @staticmethod
    def forward(ctx, query, key, value, prefix, scale):
        head = _FLASH_SDPA(
            query, key[:, :, :prefix].contiguous(),
            value[:, :, :prefix].contiguous(),
            0.0, False, False, scale=scale,
        )
        tail = _FLASH_SDPA(
            query, key[:, :, prefix:].contiguous(),
            value[:, :, prefix:].contiguous(),
            0.0, True, False, scale=scale,
        )
        merged_lse = torch.logaddexp(head[1], tail[1])
        weight_head = torch.exp(head[1] - merged_lse).unsqueeze(-1)
        weight_tail = torch.exp(tail[1] - merged_lse).unsqueeze(-1)
        out = head[0] * weight_head.to(head[0].dtype)
        out = out + tail[0] * weight_tail.to(tail[0].dtype)
        ctx.save_for_backward(
            query, key, value, out, merged_lse,
            head[2], head[3], head[6], head[7],
            tail[2], tail[3], tail[6], tail[7],
        )
        ctx.split_meta = (prefix, scale, head[4], head[5], tail[4], tail[5])
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (
            query, key, value, out, merged_lse,
            head_cum_q, head_cum_k, head_seed, head_offset,
            tail_cum_q, tail_cum_k, tail_seed, tail_offset,
        ) = ctx.saved_tensors
        prefix, scale, head_max_q, head_max_k, tail_max_q, tail_max_k = (
            ctx.split_meta
        )
        grad_out = grad_out.contiguous()
        grad_key = torch.zeros_like(key)
        grad_value = torch.zeros_like(value)
        head_grads = _FLASH_SDPA_BACKWARD(
            grad_out, query, key[:, :, :prefix].contiguous(),
            value[:, :, :prefix].contiguous(), out, merged_lse,
            head_cum_q, head_cum_k, head_max_q, head_max_k,
            0.0, False, head_seed, head_offset, scale=scale,
        )
        grad_query = head_grads[0]
        grad_key[:, :, :prefix] = head_grads[1]
        grad_value[:, :, :prefix] = head_grads[2]
        del head_grads
        tail_grads = _FLASH_SDPA_BACKWARD(
            grad_out, query, key[:, :, prefix:].contiguous(),
            value[:, :, prefix:].contiguous(), out, merged_lse,
            tail_cum_q, tail_cum_k, tail_max_q, tail_max_k,
            0.0, True, tail_seed, tail_offset, scale=scale,
        )
        grad_query = grad_query + tail_grads[0]
        grad_key[:, :, prefix:] = tail_grads[1]
        grad_value[:, :, prefix:] = tail_grads[2]
        del tail_grads
        return grad_query, grad_key, grad_value, None, None


# False keeps grouped K/V on the expanded flash split instead of the grouped
# one; a reference path kept for tests.
_FLASH_SPLIT_GROUPED_KV = True


def _flash_split_halves(query, key, value, prefix, scale):
    """The unmasked prefix and the causal square tail, on views of K/V."""

    head = _FLASH_SDPA(
        query, key[:, :, :prefix], value[:, :, :prefix],
        0.0, False, False, scale=scale,
    )
    tail = _FLASH_SDPA(
        query, key[:, :, prefix:], value[:, :, prefix:],
        0.0, True, False, scale=scale,
    )
    return head, tail


def _flash_split_grouped_forward(query, key, value, prefix, scale):
    """The grouped split's merged output and what its backward reads.

    Returns ``(out, saved, meta)``: ``saved`` holds the merged log-sum-exp
    and each half's kernel state (cumulative lengths, RNG seed and offset),
    ``meta`` each half's maximum lengths. `_SplitBottomRightFlashGrouped`
    and the saved-state `_ChunkedCausalSDPA` both run this forward, and
    `_flash_split_grouped_backward` is the backward of either.
    """

    head, tail = _flash_split_halves(query, key, value, prefix, scale)
    merged_lse = torch.logaddexp(head[1], tail[1])
    weight_head = torch.exp(head[1] - merged_lse).unsqueeze(-1)
    weight_tail = torch.exp(tail[1] - merged_lse).unsqueeze(-1)
    out = head[0] * weight_head.to(head[0].dtype)
    out = out + tail[0] * weight_tail.to(tail[0].dtype)
    saved = (
        merged_lse,
        head[2], head[3], head[6], head[7],
        tail[2], tail[3], tail[6], tail[7],
    )
    return out, saved, (head[4], head[5], tail[4], tail[5])


def _flash_split_grouped_backward(grad_out, query, key, value, out, prefix, scale, saved, meta):
    """Each half's flash backward against the merged output and LSE.

    Writes each half's grouped dK/dV into the full gradient; returns
    ``(dQ, dK, dV)`` with dK/dV in ``key``'s ``[1, Hkv, S, D]`` layout.
    """

    (
        merged_lse,
        head_cum_q, head_cum_k, head_seed, head_offset,
        tail_cum_q, tail_cum_k, tail_seed, tail_offset,
    ) = saved
    head_max_q, head_max_k, tail_max_q, tail_max_k = meta
    grad_out = grad_out.contiguous()
    grad_key = torch.empty_like(key)
    grad_value = torch.empty_like(value)
    head_grads = _FLASH_SDPA_BACKWARD(
        grad_out, query, key[:, :, :prefix], value[:, :, :prefix],
        out, merged_lse, head_cum_q, head_cum_k, head_max_q, head_max_k,
        0.0, False, head_seed, head_offset, scale=scale,
    )
    grad_query = head_grads[0]
    grad_key[:, :, :prefix] = head_grads[1]
    grad_value[:, :, :prefix] = head_grads[2]
    del head_grads
    tail_grads = _FLASH_SDPA_BACKWARD(
        grad_out, query, key[:, :, prefix:], value[:, :, prefix:],
        out, merged_lse, tail_cum_q, tail_cum_k, tail_max_q, tail_max_k,
        0.0, True, tail_seed, tail_offset, scale=scale,
    )
    grad_query = grad_query + tail_grads[0]
    grad_key[:, :, prefix:] = tail_grads[1]
    grad_value[:, :, prefix:] = tail_grads[2]
    del tail_grads
    return grad_query, grad_key, grad_value


class _SplitBottomRightFlashGrouped(torch.autograd.Function):
    """`_SplitBottomRightFlash` on the physical grouped K/V, without copies.

    ``key`` and ``value`` are ``[1, Hkv, S, D]`` views of the store's
    ``[S, Hkv, D]`` layout. Each half is a view, the forward merges by
    log-sum-exp exactly as the expanded split does, and the backward
    writes each half's grouped dK/dV into the full gradient.
    """

    @staticmethod
    def forward(ctx, query, key, value, prefix, scale):
        out, saved, meta = _flash_split_grouped_forward(query, key, value, prefix, scale)
        ctx.save_for_backward(query, key, value, out, *saved)
        ctx.split_meta = (prefix, scale, meta)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        query, key, value, out, *saved = ctx.saved_tensors
        prefix, scale, meta = ctx.split_meta
        grad_query, grad_key, grad_value = _flash_split_grouped_backward(
            grad_out, query, key, value, out, prefix, scale, saved, meta
        )
        return grad_query, grad_key, grad_value, None, None


def _split_bottom_right_flash_grouped_primal(query, key, value, prefix, scale):
    """The grouped split's merge without autograd state, for the primal."""

    head, tail = _flash_split_halves(query, key, value, prefix, scale)
    merged_lse = torch.logaddexp(head[1], tail[1])
    out = head[0] * torch.exp(head[1] - merged_lse).unsqueeze(-1).to(head[0].dtype)
    return out + tail[0] * torch.exp(tail[1] - merged_lse).unsqueeze(-1).to(tail[0].dtype)


def _flash_split_grouped_trial(
    device: torch.device, dtype: torch.dtype, head_dim: int
) -> None:
    """Grouped heads in the store's layout, both directions, one query row
    and several, checked against the expanded split's values."""

    generator = torch.Generator(device=device).manual_seed(0)
    heads, kv_heads, scale = 4, 2, head_dim ** -0.5
    for tail, keys in ((8, 24), (1, 9)):
        query = torch.randn(
            1, tail, heads, head_dim, device=device, dtype=dtype, generator=generator
        ).transpose(1, 2).requires_grad_()
        key, value = (
            torch.randn(
                keys, kv_heads, head_dim, device=device, dtype=dtype, generator=generator
            ).requires_grad_()
            for _ in range(2)
        )
        kt, vt = key.transpose(0, 1).unsqueeze(0), value.transpose(0, 1).unsqueeze(0)
        prefix = keys - tail
        output = _SplitBottomRightFlashGrouped.apply(query, kt, vt, prefix, scale)
        torch.autograd.grad(output.float().sum(), (query, key, value))
        with torch.no_grad():
            primal = _split_bottom_right_flash_grouped_primal(query, kt, vt, prefix, scale)
            expanded = _split_bottom_right_flash_primal(
                query,
                kt.repeat_interleave(heads // kv_heads, dim=1),
                vt.repeat_interleave(heads // kv_heads, dim=1),
                prefix,
                scale,
            )
        if not torch.equal(primal, output.detach()) or not torch.allclose(
            primal.float(), expanded.float(), rtol=1e-2, atol=1e-2
        ):
            raise RuntimeError("the grouped flash split disagrees with the expanded split")


def _grouped_flash_split_prefix(
    query: torch.Tensor, ks: list[torch.Tensor]
) -> int | None:
    """The prefix length where the grouped flash split takes this call, else None.

    The split's own conditions: the flag, two or more query rows (see
    `_grouped_flash_split`), no FA4 (which keeps its precedence), the flash
    split's availability, whole head groups and the grouped trial on this
    device, dtype and head dimension.
    """

    if not _FLASH_SPLIT_GROUPED_KV:
        return None
    if query.shape[2] < 2:
        return None
    prefix = sum(chunk.shape[0] for chunk in ks) - query.shape[2]
    if _fa4_bottom_right_is_available(query, prefix):
        return None
    if not _split_bottom_right_flash_is_available(query, prefix):
        return None
    if query.shape[1] % ks[0].shape[1]:
        return None
    if not kernel_runs(
        "flash_split_grouped", query, query.shape[-1], _flash_split_grouped_trial
    ):
        return None
    return prefix


def grouped_key_value(
    ks: list[torch.Tensor], vs: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """The chunks' K/V as ``[1, Hkv, S, D]`` views of one ``[S, Hkv, D]`` tensor."""

    k = ks[0] if len(ks) == 1 else torch.cat(ks, dim=0)
    v = vs[0] if len(vs) == 1 else torch.cat(vs, dim=0)
    if k.stride(-1) != 1:
        k = k.contiguous()
    if v.stride(-1) != 1:
        v = v.contiguous()
    return k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0)


def _grouped_flash_split(
    query: torch.Tensor,
    ks: list[torch.Tensor],
    vs: list[torch.Tensor],
    scale: float,
) -> torch.Tensor | None:
    """The flash split on the physical grouped K/V, or None for `_causal_sdpa`.

    Taken exactly where the expanded flash split would be (FA4 keeps its
    precedence) and only where the grouped trial passes on this device,
    dtype and head dimension. Counted under the expanded split's names.

    A single query row stays on the expanded split: for one row the kernel
    folds each head group into the row dimension, which changes its split
    schedule and so the rounding of the forward. With two or more rows the
    grouped call runs the expanded call's schedule and its forward is the
    same to the bit.
    """

    prefix = _grouped_flash_split_prefix(query, ks)
    if prefix is None:
        return None
    kt, vt = grouped_key_value(ks, vs)
    if not torch.is_grad_enabled():
        count_attention_path("flash_split_primal")
        return _split_bottom_right_flash_grouped_primal(query, kt, vt, prefix, scale)
    count_attention_path("flash_split")
    return _SplitBottomRightFlashGrouped.apply(query, kt, vt, prefix, scale)


def _split_bottom_right_flash_is_available(
    query: torch.Tensor, prefix: int
) -> bool:
    """Flash carries the head dimensions cuDNN refuses, up to 256."""

    head_dim = query.shape[-1]
    return (
        prefix > 0
        and _FLASH_SDPA is not None
        and _FLASH_SDPA_BACKWARD is not None
        and query.is_cuda
        and query.dtype in (torch.float16, torch.bfloat16)
        and head_dim <= 256
        and kernel_runs("flash_split", query, head_dim, _flash_split_trial)
    )


def _flash_split_trial(device: torch.device, dtype: torch.dtype, head_dim: int) -> None:
    query, key, value = trial_qkv(device, dtype, head_dim, heads=2, kv_heads=2)
    output = _SplitBottomRightFlash.apply(query, key, value, 8, head_dim ** -0.5)
    torch.autograd.grad(output.float().sum(), (query, key, value))


class _SplitBottomRightCausal(torch.autograd.Function):
    """Bottom-right causal attention as an unmasked prefix plus a causal tail.

    ``out = sum_i exp(lse_i - lse) * out_i`` is the exact softmax over the
    union of the two key ranges.  The backward runs each half's kernel against
    the merged output and merged log-sum-exp, which is the same decomposition
    flash-style split-key backward uses, so the gradients are the gradients of
    the unsplit attention.
    """

    @staticmethod
    def forward(ctx, query, key, value, prefix, scale):
        head = _CUDNN_SDPA(
            query, key[:, :, :prefix], value[:, :, :prefix],
            None, True, 0.0, False, False, scale=scale,
        )
        tail = _CUDNN_SDPA(
            query, key[:, :, prefix:], value[:, :, prefix:],
            None, True, 0.0, True, False, scale=scale,
        )
        merged_lse = torch.logaddexp(head[1], tail[1])
        out = head[0] * torch.exp(head[1] - merged_lse).to(head[0].dtype)
        out = out + tail[0] * torch.exp(tail[1] - merged_lse).to(tail[0].dtype)
        ctx.save_for_backward(
            query, key, value, out, merged_lse,
            head[2], head[3], head[6], head[7],
            tail[2], tail[3], tail[6], tail[7],
        )
        ctx.split_meta = (prefix, scale, head[4], head[5], tail[4], tail[5])
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (
            query, key, value, out, merged_lse,
            head_cum_q, head_cum_k, head_seed, head_offset,
            tail_cum_q, tail_cum_k, tail_seed, tail_offset,
        ) = ctx.saved_tensors
        prefix, scale, head_max_q, head_max_k, tail_max_q, tail_max_k = (
            ctx.split_meta
        )
        grad_out = grad_out.contiguous()
        # cuDNN expands the bias to [batch, head, query, key]; a scalar zero
        # broadcasts without materializing the mask this split exists to avoid.
        zero_bias = torch.zeros((1, 1, 1, 1), device=query.device, dtype=query.dtype)
        # Write each half into the full key/value gradient and release it
        # before the other half runs.  Concatenating both halves would hold
        # two full key gradients at once, and the cuDNN call has no
        # headroom for that transient.
        grad_key = torch.empty_like(key)
        grad_value = torch.empty_like(value)
        head_grads = _CUDNN_SDPA_BACKWARD(
            grad_out, query, key[:, :, :prefix], value[:, :, :prefix],
            out, merged_lse, head_seed, head_offset, zero_bias,
            head_cum_q, head_cum_k, head_max_q, head_max_k,
            0.0, False, scale=scale,
        )
        grad_query = head_grads[0]
        grad_key[:, :, :prefix].copy_(head_grads[1])
        grad_value[:, :, :prefix].copy_(head_grads[2])
        del head_grads
        tail_grads = _CUDNN_SDPA_BACKWARD(
            grad_out, query, key[:, :, prefix:], value[:, :, prefix:],
            out, merged_lse, tail_seed, tail_offset, zero_bias,
            tail_cum_q, tail_cum_k, tail_max_q, tail_max_k,
            0.0, True, scale=scale,
        )
        grad_query = grad_query + tail_grads[0]
        grad_key[:, :, prefix:].copy_(tail_grads[1])
        grad_value[:, :, prefix:].copy_(tail_grads[2])
        del tail_grads
        return grad_query, grad_key, grad_value, None, None


def _split_bottom_right_is_available(
    query: torch.Tensor, prefix: int
) -> bool:
    return (
        prefix > 0
        and _CUDNN_SDPA is not None
        and _CUDNN_SDPA_BACKWARD is not None
        and query.is_cuda
        and query.dtype in (torch.float16, torch.bfloat16)
        and query.shape[-1] <= CUDNN_SDPA_MAX_HEAD_DIM
        and kernel_runs("cudnn_split", query, query.shape[-1], cudnn_split_trial)
    )


def cudnn_split_trial(device: torch.device, dtype: torch.dtype, head_dim: int) -> None:
    query, key, value = trial_qkv(device, dtype, head_dim, heads=2, kv_heads=1)
    output = _SplitBottomRightCausal.apply(query, key, value, 8, head_dim ** -0.5)
    torch.autograd.grad(output.float().sum(), (query, key, value))


def _split_bottom_right_flash_primal(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    prefix: int,
    scale: float,
) -> torch.Tensor:
    """The flash split's merge without autograd state, for the primal.

    ``key`` and ``value`` are already expanded to the query's heads.
    """

    head = _FLASH_SDPA(
        query, key[:, :, :prefix].contiguous(), value[:, :, :prefix].contiguous(),
        0.0, False, False, scale=scale,
    )
    tail = _FLASH_SDPA(
        query, key[:, :, prefix:].contiguous(), value[:, :, prefix:].contiguous(),
        0.0, True, False, scale=scale,
    )
    merged_lse = torch.logaddexp(head[1], tail[1])
    out = head[0] * torch.exp(head[1] - merged_lse).unsqueeze(-1).to(head[0].dtype)
    return out + tail[0] * torch.exp(tail[1] - merged_lse).unsqueeze(-1).to(tail[0].dtype)


def _split_bottom_right_primal(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    prefix: int,
    scale: float,
) -> torch.Tensor:
    """The split merge without autograd state, for the no-grad primal."""

    head = _CUDNN_SDPA(
        query, key[:, :, :prefix], value[:, :, :prefix],
        None, True, 0.0, False, False, scale=scale,
    )
    tail = _CUDNN_SDPA(
        query, key[:, :, prefix:], value[:, :, prefix:],
        None, True, 0.0, True, False, scale=scale,
    )
    merged_lse = torch.logaddexp(head[1], tail[1])
    out = head[0] * torch.exp(head[1] - merged_lse).to(head[0].dtype)
    return out + tail[0] * torch.exp(tail[1] - merged_lse).to(tail[0].dtype)


def _causal_sdpa(
    query: torch.Tensor,
    ks: list[torch.Tensor],
    vs: list[torch.Tensor],
    scale: float,
    *,
    backend: str | Callable[..., torch.Tensor] = "auto",
) -> torch.Tensor:
    """cat(per-chunk K/V) + SDPA + bottom-right causal bias, [1, H, T, D] out.

    The single definition both `_ChunkedCausalSDPA.forward` and its backward
    recompute call, so the backward differentiates exactly the computation the
    forward ran.
    """
    if callable(backend):
        return backend(query, ks, vs, scale)
    if backend not in ("auto", "cudnn"):
        raise ValueError(f"unknown source attention backend: {backend!r}")
    grouped = _grouped_flash_split(query, ks, vs, scale) if backend == "auto" else None
    if grouped is not None:
        return grouped
    k = torch.cat(ks, dim=0)  # [S, Hkv, D], chronological
    v = torch.cat(vs, dim=0)
    kt = k.transpose(0, 1).unsqueeze(0)
    vt = v.transpose(0, 1).unsqueeze(0)
    prefix = kt.shape[2] - query.shape[2]
    if backend == "cudnn" and prefix > 0:
        if not _split_bottom_right_is_available(query, prefix):
            raise RuntimeError("requested cuDNN source attention failed its device trial")
        if not torch.is_grad_enabled():
            count_attention_path("cudnn_split_primal")
            return _split_bottom_right_primal(query, kt, vt, prefix, scale)
        count_attention_path("cudnn_split")
        return _SplitBottomRightCausal.apply(query, kt, vt, prefix, scale)
    if _fa4_bottom_right_is_available(query, prefix):
        fused = _fa4_bottom_right_causal(query, kt, vt, scale)
        if fused is not None:
            count_attention_path("fa4")
            return fused
    if _split_bottom_right_flash_is_available(query, prefix):
        # Flash first: no per-call plan on the host.
        # It has no grouped key/value path, so expand first; the expanded
        # flash call still beats the grouped cuDNN one, and both are far
        # ahead of the masked math fallback. The expansion is a transient
        # copy whose backward sums the gradient back over each group.
        repeat = query.shape[1] // kt.shape[1]
        flash_k = kt if repeat == 1 else kt.repeat_interleave(repeat, dim=1)
        flash_v = vt if repeat == 1 else vt.repeat_interleave(repeat, dim=1)
        if not torch.is_grad_enabled():
            count_attention_path("flash_split_primal")
            return _split_bottom_right_flash_primal(
                query, flash_k, flash_v, prefix, scale
            )
        count_attention_path("flash_split")
        return _SplitBottomRightFlash.apply(
            query, flash_k, flash_v, prefix, scale
        )
    if _split_bottom_right_is_available(query, prefix):
        # cuDNN takes grouped key/value heads directly and skips the
        # expansion; it is the fallback for what flash refuses, since its
        # frontend builds a plan per call.
        if not torch.is_grad_enabled():
            count_attention_path("cudnn_split_primal")
            return _split_bottom_right_primal(query, kt, vt, prefix, scale)
        count_attention_path("cudnn_split")
        return _SplitBottomRightCausal.apply(query, kt, vt, prefix, scale)
    count_attention_path("masked")
    n_rep = query.shape[1] // kt.shape[1]
    if n_rep > 1:
        # The masked fallback needs the expansion: `enable_gqa=True` forces
        # SDPA's math backend here (fp32 + bottom-right bias), which
        # materializes the mask and pays an unfused softmax.  The expansion
        # is a transient copy, and its backward sums gradient over the
        # repeated heads -- the same GQA semantics, derived by autograd.
        kt = kt.repeat_interleave(n_rep, dim=1)
        vt = vt.repeat_interleave(n_rep, dim=1)
    return F.scaled_dot_product_attention(
        query,
        kt,
        vt,
        attn_mask=causal_lower_right(query.shape[2], k.shape[0]),
        scale=scale,
    )


def _ragged_causal_sdpa(
    queries: list[torch.Tensor],
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
    scale: float,
) -> list[torch.Tensor]:
    """Execute one variable-length, lower-right-causal GQA attention call.

    Every query is ``[1, Hq, T, D]`` and every key/value is ``[S, Hkv, D]``.
    PyTorch's CUDA flash-attention operator accepts a packed ``[sum(T), H, D]``
    representation plus cumulative sequence lengths.  This removes the Python
    segment loop and preserves native grouped-query attention, so K/V heads are
    read in their physical ``Hkv`` representation instead of being expanded to
    ``Hq`` before every attention call.

    ``is_causal=True`` on this variable-length operator is bottom-right aligned
    when ``S >= T``: each segment query attends to its complete ancestor prefix
    and the causal portion of its own newly appended block. CUDA validation
    compares values and gradients against the public lower-right-causal SDPA
    reference before this optional path is enabled.
    """
    if not queries:
        return []
    if not queries[0].is_cuda:
        raise ValueError("ragged cohort attention requires CUDA tensors")
    query_lengths = [query.shape[2] for query in queries]
    key_lengths = [key.shape[0] for key in keys]
    packed_query = torch.cat(
        [query[0].transpose(0, 1) for query in queries], dim=0
    )
    packed_key = torch.cat(keys, dim=0)
    packed_value = torch.cat(values, dim=0)
    device = packed_query.device
    cu_query = torch.tensor(
        [0, *itertools.accumulate(query_lengths)],
        dtype=torch.int32,
        device=device,
    )
    cu_key = torch.tensor(
        [0, *itertools.accumulate(key_lengths)],
        dtype=torch.int32,
        device=device,
    )
    output, *_ = torch.ops.aten._flash_attention_forward(
        packed_query,
        packed_key,
        packed_value,
        cu_query,
        cu_key,
        max(query_lengths),
        max(key_lengths),
        0.0,
        True,
        False,
        scale=scale,
        window_size_left=None,
        window_size_right=None,
        seqused_k=None,
        alibi_slopes=None,
    )
    pieces = output.split(query_lengths, dim=0)
    return [piece.transpose(0, 1).unsqueeze(0) for piece in pieces]


def _ragged_attention_runs(like: torch.Tensor, head_dim: int | None) -> bool:
    """Whether the variable-length flash operator runs this device and dtype."""

    return (
        head_dim is not None
        and like.is_cuda
        and like.dtype in (torch.float16, torch.bfloat16)
        and kernel_runs("ragged_flash", like, head_dim, _ragged_trial)
    )


def _ragged_trial(device: torch.device, dtype: torch.dtype, head_dim: int) -> None:
    generator = torch.Generator(device=device).manual_seed(0)
    queries = [
        torch.randn(1, 2, length, head_dim, device=device, dtype=dtype, generator=generator)
        for length in (4, 8)
    ]
    keys = [
        torch.randn(length, 1, head_dim, device=device, dtype=dtype, generator=generator)
        for length in (12, 16)
    ]
    _ragged_causal_sdpa(queries, keys, keys, head_dim ** -0.5)


class _RaggedAttentionIndependentVJP(torch.autograd.Function):
    """Adopt ragged-attention values and reproduce independent SDPA VJPs.

    The forward executes the cohort through one variable-length flash call.
    Its backward recomputes each segment with the established lower-right
    causal SDPA operation.  Parameter-bearing Q/K/V projections remain
    segment-local, and their cotangents therefore match the independent
    checkpoint path's BF16 reduction schedule.
    """

    @staticmethod
    def forward(ctx, scale, n_segments, *tensors):
        queries = list(tensors[:n_segments])
        keys = list(tensors[n_segments : 2 * n_segments])
        values = list(tensors[2 * n_segments :])
        ctx.scale = scale
        ctx.n_segments = n_segments
        ctx.save_for_backward(*tensors)
        return tuple(_ragged_causal_sdpa(queries, keys, values, scale))

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grad_outputs):
        n = ctx.n_segments
        saved = ctx.saved_tensors
        queries = [tensor.detach().requires_grad_(True) for tensor in saved[:n]]
        keys = [
            tensor.detach().requires_grad_(True) for tensor in saved[n : 2 * n]
        ]
        values = [
            tensor.detach().requires_grad_(True) for tensor in saved[2 * n :]
        ]
        with torch.enable_grad():
            outputs = [
                _causal_sdpa(query, [key], [value], ctx.scale)
                for query, key, value in zip(
                    queries, keys, values, strict=True
                )
            ]
        grads = torch.autograd.grad(
            outputs,
            [*queries, *keys, *values],
            grad_outputs,
        )
        return (None, None, *grads)


def _rotary_positions(rotary_emb: torch.nn.Module, pos: torch.Tensor) -> torch.Tensor:
    """Positions in the shape ``rotary_emb`` takes.

    Multi-section rotary embeddings read one row of positions per
    section, ``[3, B, T]``; their text model expands ``[B, T]`` text positions
    to that shape before the call, and so must a caller that runs the rotary
    embedding itself.
    """

    if getattr(rotary_emb, "mrope_section", None) is not None and pos.dim() == 2:
        return pos.unsqueeze(0).expand(3, *pos.shape)
    return pos


class _ChunkedCausalSDPA(torch.autograd.Function):
    """cat(ks) / cat(vs) + bottom-right-causal SDPA as ONE differentiable op.

    Why it exists: composed as `torch.cat` then `F.scaled_dot_product_attention`,
    autograd saves the CONCATENATED K/V for SDPA's backward, so every streamed
    event retains a fresh copy of its full ancestor prefix -- sum-of-prefixes
    retention, quadratic in turns per trajectory.
    This op saves only the query and the PER-CHUNK K/V references -- the same
    graph tensors `_StreamKV` already retains, and `save_for_backward` stores
    references, not copies -- so the concatenation and every SDPA intermediate
    are transient and the marginal retained cost of an event is its own chunk.

    Backward strategy: re-concatenate transiently, RE-RUN the SDPA forward on
    detached leaves under `enable_grad`, and obtain dQ/dK/dV from that
    recomputation with `torch.autograd.grad`; `cat`'s own backward splits dK/dV
    back into per-chunk gradients, which the outer graph then routes to each
    chunk's producer. Chosen over invoking
    `torch.ops.aten._scaled_dot_product_flash_attention(_backward)` explicitly
    because (a) the flash kernel does not run everywhere this module must --
    fp32 and CPU included, both exercised by the test gates -- while the
    recompute inherits SDPA's own backend dispatch, and (b) the flash op
    returns its logsumexp without a grad function; here every gradient term is
    derived by autograd from the same
    public composite op the forward ran, so a hand-dropped term is structurally
    impossible rather than carefully avoided. The price is one extra attention
    forward per chunk at backward time -- a small share of step FLOPs, and the
    recompute's memory is per-chunk-transient, which is the point.

    Where the forward takes the grouped flash split and its backward would
    re-run it (`_SAVED_STATE_FLASH_SPLIT`), the op keeps what that re-run
    would reproduce -- its merged output (the attention output the next
    projection saves anyway) and its merged log-sum-exp -- and its backward
    runs the split's flash backward on them directly, the backward the
    re-run would reach. The re-run's forward is deterministic, so the
    backward reads the same numbers; every other path keeps the recompute.
    ``keep_state`` False keeps it too (a pack, whose segments' outputs are
    concatenated rather than saved by the projection).
    """

    @staticmethod
    def forward(ctx, query, scale, n_chunks, keep_state, backend, *chunks):
        # grad recording is off inside Function.forward: the cat and SDPA
        # intermediates below die when this frame returns.
        ctx.scale = scale
        ctx.n_chunks = n_chunks
        ctx.backend = backend
        ks, vs = list(chunks[:n_chunks]), list(chunks[n_chunks:])
        prefix = (
            _grouped_flash_split_prefix(query, ks)
            if backend == "auto" and keep_state and _SAVED_STATE_FLASH_SPLIT
            else None
        )
        if prefix is None:
            ctx.split = None
            ctx.save_for_backward(query, *chunks)
            return _causal_sdpa(query, ks, vs, scale, backend=backend)
        kt, vt = grouped_key_value(ks, vs)
        out, saved, meta = _flash_split_grouped_forward(query, kt, vt, prefix, scale)
        # counted as `_grouped_flash_split` counts this forward, without grad
        count_attention_path("flash_split_primal")
        ctx.split = (prefix, meta, len(saved))
        ctx.save_for_backward(query, out, *saved, *chunks)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        n = ctx.n_chunks
        # inputs: (query, scale, n_chunks, keep_state, backend, *chunks)
        need = ctx.needs_input_grad
        if ctx.split is None:
            query, *chunks = ctx.saved_tensors
            q = query.detach().requires_grad_(True)
            ks = [c.detach().requires_grad_(True) for c in chunks[:n]]
            vs = [c.detach().requires_grad_(True) for c in chunks[n:]]
            with torch.enable_grad():
                out = _causal_sdpa(q, ks, vs, ctx.scale, backend=ctx.backend)
            grads = torch.autograd.grad(out, [q, *ks, *vs], grad_out)
            return (
                grads[0] if need[0] else None,
                None,
                None,
                None,
                None,
                *(g if need[5 + i] else None for i, g in enumerate(grads[1:])),
            )
        prefix, meta, n_saved = ctx.split
        query, out, *rest = ctx.saved_tensors
        saved, chunks = rest[:n_saved], rest[n_saved:]
        ks, vs = list(chunks[:n]), list(chunks[n:])
        kt, vt = grouped_key_value(ks, vs)
        count_attention_path("flash_split")
        grad_query, grad_key, grad_value = _flash_split_grouped_backward(
            grad_out, query, kt, vt, out, prefix, ctx.scale, saved, meta
        )
        # cat's backward: each chunk's rows of the [S, Hkv, D] gradient
        lengths = [chunk.shape[0] for chunk in ks]
        grad_keys = grad_key[0].transpose(0, 1).split(lengths, dim=0)
        grad_values = grad_value[0].transpose(0, 1).split(lengths, dim=0)
        return (
            grad_query if need[0] else None,
            None,
            None,
            None,
            None,
            *(
                g if need[5 + i] else None
                for i, g in enumerate((*grad_keys, *grad_values))
            ),
        )


# Set by the checkpointed executor around each layer call. Holds the ancestor
# K/V lists the current layer must attend over, and captures the fresh chunk's
# K/V so the executor can return them as checkpoint outputs. The device key
# preserves thread visibility while isolating concurrent teacher/student GPUs.
_CKPTS: dict[torch.device, dict] = {}


def checkpoint_segments(device: torch.device) -> list:
    """Diagnostic: the segments of the forward currently dispatched on ``device``."""

    return list(_CKPTS.get(device, {}).get("segments", []))


def stream_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    source_attention_backend: str | Callable[..., torch.Tensor] = "auto",
    **kwargs,
):
    """Attention for a streamed chunk: cache-extend, then bottom-right causal.

    The new chunk is strictly later than every cached key, so the visibility
    pattern over [ancestors..., own] is exactly bottom-right-aligned causal --
    one kernel call, no mask tensor. (`is_causal` would align UPPER-left when
    kv_len > q_len and silently give every query the wrong window; that trap is
    documented in attention.py and applies identically here.)
    """
    ckpt = _CKPTS.get(query.device)
    if ckpt is not None:
        # Checkpointed executor path: everything here is transient -- the layer
        # runs either as the packed no-grad primal or as a per-segment
        # recompute at backward. K/V are handed back through the context so the
        # executor can surface them as SEGMENT-NODE OUTPUTS (graph-attached
        # across chunks); stashing them into _StreamKV from here would capture
        # the no-grad pass and silently sever cross-chunk gradients, the exact
        # failure this design forbids.
        #
        # The stream may hold several coalesced segments concatenated along the
        # sequence dimension. The dense projections
        # that produced query/key/value already ran batched over the whole pack
        # -- that is coalescing's entire win; attention is the only per-segment
        # part. Each segment attends over its OWN ancestor chain plus itself,
        # bottom-right causal, and NEVER across segments.
        if query.shape[0] != 1:
            raise ValueError("streaming forwards one packed stream; batch dim must be 1")
        k_full = key[0].transpose(0, 1)  # [T, Hkv, D]
        v_full = value[0].transpose(0, 1)
        scale = scaling if scaling is not None else 1.0 / math.sqrt(query.shape[-1])
        queries, keys, values, captured = [], [], [], []
        for start, length, ks, vs in ckpt["segments"]:
            k = k_full.narrow(0, start, length)  # slicing keeps graph attachment
            v = v_full.narrow(0, start, length)
            queries.append(query.narrow(2, start, length))
            keys.append(torch.cat(ks + [k], dim=0))
            values.append(torch.cat(vs + [v], dim=0))
            captured.append((k, v))
        if (
            ckpt.get("ragged_attention", False)
            and source_attention_backend == "auto"
            and len(queries) > 1
            and _ragged_attention_runs(query, query.shape[-1])
        ):
            outs = _ragged_causal_sdpa(queries, keys, values, scale)
        else:
            outs = [
                _causal_sdpa(segment_query, [segment_key], [segment_value], scale,
                             backend=source_attention_backend)
                for segment_query, segment_key, segment_value in zip(
                    queries, keys, values, strict=True
                )
            ]
        ckpt["captured"] = captured
        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=2)
        return out.transpose(1, 2), None

    run = _STREAMS.get(query.device)
    if run is None or (run._active_chunk is None and run._active_segments is None):
        raise RuntimeError("stream attention invoked outside an append event")
    if query.shape[0] != 1:
        raise ValueError("streaming forwards one chunk at a time; batch dim must be 1")

    li = module.layer_idx
    if run._active_segments is not None:
        return _plain_packed_attention(run, li, query, key, value, scaling,
                                       backend=source_attention_backend)
    # [1, Hkv, T, D] -> [T, Hkv, D]: store the graph tensors themselves
    run._kv.put(li, run._active_chunk, key[0].transpose(0, 1), value[0].transpose(0, 1))
    scale = scaling if scaling is not None else 1.0 / math.sqrt(query.shape[-1])

    if run.chunked_sdpa:
        # One op over the per-chunk references: autograd retains each chunk's
        # K/V once (the _StreamKV tensors) instead of a concatenated prefix
        # copy per event.
        ks, vs = run._kv.chunks(li, run._active_chain)
        out = _ChunkedCausalSDPA.apply(query, scale, len(ks), True,
                                      source_attention_backend, *ks, *vs)
    else:
        # Concat reference path: retention grows as sum-of-prefixes. Kept as
        # the baseline the memory regression test measures the op above against.
        ks, vs = run._kv.gather(li, run._active_chain)
        out = F.scaled_dot_product_attention(
            query,
            ks.transpose(0, 1).unsqueeze(0),
            vs.transpose(0, 1).unsqueeze(0),
            attn_mask=causal_lower_right(query.shape[2], ks.shape[0]),
            scale=scale,
            enable_gqa=query.shape[1] != ks.shape[1],
        )
    return out.transpose(1, 2), None


def _plain_packed_attention(
    run: "StreamingRun",
    li: int,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float | None,
    *,
    backend: str | Callable[..., torch.Tensor] = "auto",
) -> tuple[torch.Tensor, None]:
    """Attention of a plain turn pack: each segment over its own chain.

    The segments are the pack's turns laid out back to back. The dense
    projections that produced query/key/value ran once over the whole pack;
    here each segment stores its own K/V under its chunk id and attends over
    its trajectory's ancestors plus itself. Where every segment takes the
    grouped flash split, one `_PackedFlashSplit` runs them all
    (`_PACKED_FLASH_SPLIT`): each segment's forward is the single-turn
    split's, and one backward serves the pack. Otherwise each segment runs
    `_ChunkedCausalSDPA`, the single-turn op; either way the pack retains
    per-chunk references only and every segment's attention computes what
    its single-turn append computes. Splitting (rather than narrowing)
    keeps the backward to one concatenation per tensor.
    """

    lengths = [length for _cid, _chain, length in run._active_segments]
    scale = scaling if scaling is not None else 1.0 / math.sqrt(query.shape[-1])
    own_keys = key[0].transpose(0, 1)
    own_values = value[0].transpose(0, 1)
    queries = query.split(lengths, dim=2)
    keys = own_keys.split(lengths, dim=0)
    values = own_values.split(lengths, dim=0)
    ancestors = []
    for (cid, chain, _length), segment_key, segment_value in zip(
        run._active_segments, keys, values, strict=True
    ):
        run._kv.put(li, cid, segment_key, segment_value)
        ancestors.append(run._kv.chunks(li, chain[:-1]) if len(chain) > 1 else ([], []))
    layout = _pack_layout(run, query, queries, ancestors, keys) if backend == "auto" else None
    if layout is not None:
        out = _PackedFlashSplit.apply(
            query,
            _last_dim_contiguous(own_keys),
            _last_dim_contiguous(own_values),
            scale,
            layout,
            *layout.inputs(ancestors),
        )
        return out.transpose(1, 2), None
    outs = []
    for segment_query, segment_key, segment_value, (ks, vs) in zip(
        queries, keys, values, ancestors, strict=True
    ):
        ks, vs = [*ks, segment_key], [*vs, segment_value]
        outs.append(
            _ChunkedCausalSDPA.apply(segment_query, scale, len(ks), False, backend, *ks, *vs)
        )
    return torch.cat(outs, dim=2).transpose(1, 2), None


def _last_dim_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.stride(-1) == 1 else tensor.contiguous()


def _device_int32(values: list[int], device: torch.device) -> torch.Tensor:
    """`_device_long` for the flash kernels' int32 cumulative lengths."""

    staged = torch.tensor(values, dtype=torch.int32, pin_memory=True)
    return staged.to(device, non_blocking=True)


@dataclass(eq=False)
class _PackLayout:
    """Where each segment of a plain pack sits in the packed attention.

    ``lengths`` and ``prefixes`` are each segment's query rows and ancestor
    tokens; ``counts`` its ancestor chunks, in segment order. The varlen
    backward reads the cumulative lengths, staged once per model call.
    The op's inputs hold the ancestor chunks grouped by segment, LAST
    segment first (`inputs`): a chunk several segments read (their shared
    prompt) receives one gradient per segment, which autograd sums in input
    order, and the per-segment ops this replaces delivered them in that
    order (the engine runs the most recently recorded node first).
    """

    lengths: list[int]
    prefixes: list[int]
    counts: list[int]
    cu_lengths: torch.Tensor
    cu_prefixes: torch.Tensor
    empty: torch.Tensor
    # False for the kernel trial, whose calls are not the run's attention
    counted: bool = True

    def inputs(
        self, ancestors: list[tuple[list[torch.Tensor], list[torch.Tensor]]]
    ) -> list[torch.Tensor]:
        ks = [chunk for ks, _vs in reversed(ancestors) for chunk in ks]
        vs = [chunk for _ks, vs in reversed(ancestors) for chunk in vs]
        return ks + vs

    def by_segment(self, chunks: tuple[torch.Tensor, ...]) -> list[list[torch.Tensor]]:
        """One half (K or V) of `inputs`, back in segment order."""

        grouped, offset = [], 0
        for count in reversed(self.counts):
            grouped.append(list(chunks[offset : offset + count]))
            offset += count
        return grouped[::-1]


def _pack_layout(
    run: "StreamingRun",
    query: torch.Tensor,
    queries: tuple[torch.Tensor, ...],
    ancestors: list[tuple[list[torch.Tensor], list[torch.Tensor]]],
    keys: tuple[torch.Tensor, ...],
) -> _PackLayout | None:
    """The pack's layout when `_PackedFlashSplit` takes it, else None.

    Every segment must take the grouped flash split its single-turn op would
    (`_grouped_flash_split_prefix`), the varlen flash backward must exist and
    the op's trial must pass on this device, dtype and head dimension. Built
    at the first layer; a model call's layers share their shapes.
    """

    shape_key = (query.shape[1], query.shape[-1], query.dtype, keys[0].shape[1])
    if shape_key in run._active_pack_layouts:
        return run._active_pack_layouts[shape_key]
    layout = None
    if (
        _PACKED_FLASH_SPLIT
        and _SAVED_STATE_FLASH_SPLIT
        and _FLASH_VARLEN_BACKWARD is not None
        and all(ks for ks, _vs in ancestors)
    ):
        prefixes = [
            _grouped_flash_split_prefix(segment_query, [*ks, segment_key])
            for segment_query, (ks, _vs), segment_key in zip(
                queries, ancestors, keys, strict=True
            )
        ]
        if all(prefix is not None for prefix in prefixes) and kernel_runs(
            "flash_split_packed", query, query.shape[-1], _packed_flash_split_trial
        ):
            lengths = [segment.shape[2] for segment in queries]
            device = query.device
            layout = _PackLayout(
                lengths=lengths,
                prefixes=prefixes,
                counts=[len(ks) for ks, _vs in ancestors],
                cu_lengths=_device_int32([0, *itertools.accumulate(lengths)], device),
                cu_prefixes=_device_int32([0, *itertools.accumulate(prefixes)], device),
                empty=torch.empty(0, device=device),
            )
    run._active_pack_layouts[shape_key] = layout
    return layout


_FLASH_VARLEN_BACKWARD = getattr(torch.ops.aten, "_flash_attention_backward", None)


class _PackedFlashSplit(torch.autograd.Function):
    """A plain pack's grouped flash split: per-segment forward, one backward.

    The forward runs each segment's split exactly as `_ChunkedCausalSDPA`
    does -- the flash kernel over the segment's ancestor K/V (unmasked) and
    over its own K/V (causal), the same calls on the same numbers -- and
    merges all segments' halves at once, the merge being elementwise. The
    ancestors of every segment are concatenated once for the pack rather
    than once per segment. The op keeps the packed merged output (the
    tensor the output projection saves, in its layout) and log-sum-exp.

    The backward runs the split's two halves as two variable-length flash
    backward calls over every segment at once, against the merged output
    and log-sum-exp, which is the backward each segment's saved-state op
    runs on its own: dK/dV per segment are that backward's, and dQ, which
    the kernel accumulates atomically, is too up to that accumulation's
    order. The ancestors are re-concatenated transiently, as the
    single-turn op does, so the pack retains per-chunk references only.

    Inputs: ``query [1, Hq, sum(T), D]``, the pack's own K/V ``[sum(T),
    Hkv, D]``, then the ancestor chunks in `_PackLayout.inputs` order.
    """

    @staticmethod
    def forward(ctx, query, key, value, scale, layout, *ancestors):
        total = sum(layout.counts)
        ancestor_keys = layout.by_segment(ancestors[:total])
        ancestor_values = layout.by_segment(ancestors[total:])
        prefix_keys = torch.cat([chunk for ks in ancestor_keys for chunk in ks], dim=0)
        prefix_values = torch.cat(
            [chunk for vs in ancestor_values for chunk in vs], dim=0
        )
        heads, tails = [], []
        start = prefix_start = 0
        for length, prefix in zip(layout.lengths, layout.prefixes, strict=True):
            segment_query = query.narrow(2, start, length)
            head = _FLASH_SDPA(
                segment_query,
                prefix_keys.narrow(0, prefix_start, prefix).transpose(0, 1).unsqueeze(0),
                prefix_values.narrow(0, prefix_start, prefix).transpose(0, 1).unsqueeze(0),
                0.0, False, False, scale=scale,
            )
            tail = _FLASH_SDPA(
                segment_query,
                key.narrow(0, start, length).transpose(0, 1).unsqueeze(0),
                value.narrow(0, start, length).transpose(0, 1).unsqueeze(0),
                0.0, True, False, scale=scale,
            )
            heads.append(head[:2])
            tails.append(tail[:2])
            start += length
            prefix_start += prefix
        del prefix_keys, prefix_values
        # [sum(T), Hq, D] outputs and [Hq, sum(T)] log-sum-exps
        head_out = torch.cat([out[0].transpose(0, 1) for out, _lse in heads], dim=0)
        tail_out = torch.cat([out[0].transpose(0, 1) for out, _lse in tails], dim=0)
        head_lse = torch.cat([lse[0] for _out, lse in heads], dim=1)
        tail_lse = torch.cat([lse[0] for _out, lse in tails], dim=1)
        del heads, tails
        merged_lse = torch.logaddexp(head_lse, tail_lse)
        weight_head = torch.exp(head_lse - merged_lse).transpose(0, 1).unsqueeze(-1)
        weight_tail = torch.exp(tail_lse - merged_lse).transpose(0, 1).unsqueeze(-1)
        out = head_out * weight_head.to(head_out.dtype)
        out = out + tail_out * weight_tail.to(tail_out.dtype)
        if layout.counted:
            # counted per segment, as each segment's single-turn op counts
            count_attention_path("flash_split_primal", len(layout.lengths))
        out = out.transpose(0, 1).unsqueeze(0)
        ctx.scale = scale
        ctx.layout = layout
        ctx.save_for_backward(query, key, value, out, merged_lse, *ancestors)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        query, key, value, out, merged_lse, *ancestors = ctx.saved_tensors
        layout = ctx.layout
        scale = ctx.scale
        total = sum(layout.counts)
        ancestor_keys = layout.by_segment(tuple(ancestors[:total]))
        ancestor_values = layout.by_segment(tuple(ancestors[total:]))
        prefix_keys = torch.cat([chunk for ks in ancestor_keys for chunk in ks], dim=0)
        prefix_values = torch.cat(
            [chunk for vs in ancestor_values for chunk in vs], dim=0
        )
        packed_query = _last_dim_contiguous(query[0].transpose(0, 1))
        packed_grad = grad_out[0].transpose(0, 1).contiguous()
        packed_out = out[0].transpose(0, 1)
        max_length = max(layout.lengths)
        if layout.counted:
            count_attention_path("flash_split", len(layout.lengths))
        head_grads = _FLASH_VARLEN_BACKWARD(
            packed_grad, packed_query, prefix_keys, prefix_values, packed_out,
            merged_lse, layout.cu_lengths, layout.cu_prefixes, max_length,
            max(layout.prefixes), 0.0, False, layout.empty, layout.empty,
            scale=scale,
        )
        del prefix_keys, prefix_values
        tail_grads = _FLASH_VARLEN_BACKWARD(
            packed_grad, packed_query, key, value, packed_out, merged_lse,
            layout.cu_lengths, layout.cu_lengths, max_length, max_length,
            0.0, True, layout.empty, layout.empty, scale=scale,
        )
        grad_query = (head_grads[0] + tail_grads[0]).transpose(0, 1).unsqueeze(0)
        # each ancestor chunk's rows of the packed prefix gradient, regrouped
        # into the input order
        chunk_lengths = [
            chunk.shape[0] for ks in ancestor_keys for chunk in ks
        ]
        grad_keys = list(head_grads[1].split(chunk_lengths, dim=0))
        grad_values = list(head_grads[2].split(chunk_lengths, dim=0))

        def input_order(grads: list[torch.Tensor]) -> list[torch.Tensor]:
            grouped, offset = [], 0
            for count in layout.counts:
                grouped.append(grads[offset : offset + count])
                offset += count
            return [grad for segment in reversed(grouped) for grad in segment]

        ancestor_grads = [*input_order(grad_keys), *input_order(grad_values)]
        need = ctx.needs_input_grad  # (query, key, value, scale, layout, *ancestors)
        return (
            grad_query if need[0] else None,
            tail_grads[1] if need[1] else None,
            tail_grads[2] if need[2] else None,
            None,
            None,
            *(
                grad if need[5 + index] else None
                for index, grad in enumerate(ancestor_grads)
            ),
        )


def _packed_flash_split_trial(
    device: torch.device, dtype: torch.dtype, head_dim: int
) -> None:
    """Two grouped-head segments sharing an ancestor chunk, both directions,
    checked against each segment's grouped split on its own.

    Calls the kernels' ops directly, never the dispatch, which would re-enter
    the capability lock this trial runs under.
    """

    generator = torch.Generator(device=device).manual_seed(0)
    heads, kv_heads, scale = 4, 2, head_dim ** -0.5
    lengths, own_chunks = [3, 5], [[7], [6, 4]]

    def make(*shape):
        return torch.randn(
            *shape, device=device, dtype=dtype, generator=generator
        ).requires_grad_()

    query = make(1, sum(lengths), heads, head_dim)
    key, value = make(sum(lengths), kv_heads, head_dim), make(sum(lengths), kv_heads, head_dim)
    shared_k, shared_v = make(9, kv_heads, head_dim), make(9, kv_heads, head_dim)
    chunk_ks = [[shared_k] + [make(n, kv_heads, head_dim) for n in own] for own in own_chunks]
    chunk_vs = [[shared_v] + [make(n, kv_heads, head_dim) for n in own] for own in own_chunks]
    prefixes = [sum(chunk.shape[0] for chunk in ks) for ks in chunk_ks]
    layout = _PackLayout(
        lengths=lengths,
        prefixes=prefixes,
        counts=[len(ks) for ks in chunk_ks],
        cu_lengths=torch.tensor([0, 3, 8], dtype=torch.int32, device=device),
        cu_prefixes=torch.tensor(
            [0, prefixes[0], sum(prefixes)], dtype=torch.int32, device=device
        ),
        empty=torch.empty(0, device=device),
        counted=False,
    )
    q = query.transpose(1, 2)
    packed = _PackedFlashSplit.apply(
        q, key, value, scale, layout, *layout.inputs(list(zip(chunk_ks, chunk_vs, strict=True)))
    )
    pieces = []
    for segment_query, segment_key, segment_value, ks, vs, prefix in zip(
        q.split(lengths, dim=2), key.split(lengths), value.split(lengths),
        chunk_ks, chunk_vs, prefixes, strict=True,
    ):
        kt, vt = grouped_key_value([*ks, segment_key], [*vs, segment_value])
        pieces.append(
            _SplitBottomRightFlashGrouped.apply(segment_query, kt, vt, prefix, scale)
        )
    reference = torch.cat(pieces, dim=2)
    leaves = [query, key, value, shared_k, shared_v]
    cotangent = torch.randn(
        packed.shape, device=device, dtype=dtype, generator=generator
    )
    got = torch.autograd.grad(packed, leaves, cotangent)
    want = torch.autograd.grad(reference, leaves, cotangent)
    if not torch.equal(packed.detach(), reference.detach()):
        raise RuntimeError("the packed flash split's forward differs from the per-segment split")
    for mine, theirs in zip(got, want, strict=True):
        if not torch.allclose(mine.float(), theirs.float(), rtol=1e-2, atol=1e-2):
            raise RuntimeError("the packed flash split's gradients differ from the per-segment split")


def _checkpoint_layer_index(layer) -> int | None:
    """The decoder layer's index, whichever mixing module carries it.

    A full-attention layer exposes `self_attn`; a Gated DeltaNet layer
    exposes `linear_attn`. Reading only the former tagged 48 of this
    family's 64 layers as `layer=None`, so their recompute and VJP time was
    unattributable in the profiling spans.
    """

    for attribute in ("self_attn", "linear_attn"):
        index = getattr(getattr(layer, attribute, None), "layer_idx", None)
        if index is not None:
            return index
    return None


def _linear_layer_fn(layer, h, meta, *past):
    """One packed chunk-layer of a LINEAR-ATTENTION (Gated DeltaNet) decoder
    layer: `(h, meta, past...) -> (h_out, state_0, tail_0, state_1, tail_1, ...)`.

    The hybrid generalisation of the boundary: a linear layer's cross-chunk
    dependence flows through EXACTLY two channels -- the recurrent state (one
    `[1, Hv, Dk, Dv]` tensor, fp32 as the kernels keep it) and the conv tail
    (the last `conv_kernel-1` pre-conv projected inputs, `[1, conv_dim, k-1]`)
    -- and both are gradient channels. They play precisely the role ancestor
    K/V play for an attention layer, so this function mirrors the attention
    path's shape: `meta` is one `(start, length, n_past)` triple per segment
    (`n_past` is 1 when the segment continues a parent chunk, 0 at a
    trajectory root), and `past` holds every segment's initial state in
    segment order, then every segment's conv tail. Each segment's FINAL state
    and tail are returned flat so the executor can surface them as graph
    outputs -- checkpoint-tracked I/O, so cross-chunk gradients survive
    recompute by construction, exactly as K/V do.

    The computation replicates the transformers decoder layer's forward for
    `block_type == "linear_attention"`, and is checked against the model
    called monolithically. Dense per-position ops -- the
    `in_proj_{qkv,z,b,a}` projections, gated norm, `out_proj`, layernorms and
    MLP -- run batched over the whole pack (coalescing's win); the two
    sequence-mixing ops run per segment: the depthwise causal conv over
    `[conv_tail, segment]` (the carried tail standing in for the monolithic
    zero left-pad), and the chunked gated-delta-rule kernel called with the
    segment's initial state. The fla kernel computes `dh0` and accepts `dht`
    (probed, not assumed); the torch fallback derives both by autograd. Per-
    segment kernel calls are serial on purpose: a linear kernel cannot batch
    segments with different initial states in one dense call, while the dense
    per-position operations are already batched. Segments are contiguous
    trajectory-ordered slices of the pack -- the executor asserts it -- which
    is the order-dependence invariant a recurrence needs.
    """
    n_total = sum(m[2] for m in meta)
    states, tails = past[:n_total], past[n_total:]
    mix = layer.linear_attn
    residual = h
    hs = layer.input_layernorm(h)

    mixed_qkv = mix.in_proj_qkv(hs).transpose(1, 2)  # [1, conv_dim, T]
    z = mix.in_proj_z(hs)
    b = mix.in_proj_b(hs)
    a = mix.in_proj_a(hs)
    beta = b.sigmoid()
    # fp32 as in the HF forward: in fp16 A might be -inf without the .float()
    g = -mix.A_log.float().exp() * F.softplus(a.float() + mix.dt_bias)

    kernel = mix.chunk_gated_delta_rule
    if not h.is_cuda:
        # the module binds the fla Triton kernel whenever fla is importable;
        # on CPU tensors it cannot run, and the model file always defines the
        # torch fallback alongside
        import sys as _sys

        kernel = getattr(
            _sys.modules[type(mix).__module__], "torch_chunk_gated_delta_rule", kernel
        )

    k_tail = mix.conv_kernel_size - 1
    rep = mix.num_v_heads // mix.num_k_heads
    outs, finals = [], []
    off = 0
    for start, length, n_past in meta:
        x = mixed_qkv.narrow(2, start, length)
        if n_past:
            tail = tails[off].to(x.dtype)
            conv_in = torch.cat([tail, x], dim=-1)
        else:
            tail = x.new_zeros(x.shape[0], x.shape[1], k_tail)
            conv_in = x
        qkv = None
        if _LINEAR_ATTENTION_FLA_CONV:
            # Both spellings left-pad by the kernel tail: the continuation
            # branch supplies the real left context and drops the outputs
            # that belong to it, the opening branch pads with zeros.
            fused = _fla_causal_conv_silu(
                conv_in, mix.conv1d.weight.squeeze(1), mix.conv1d.bias
            )
            if fused is not None:
                qkv = fused[:, k_tail:, :] if n_past else fused
        if qkv is None:
            conv_out = F.conv1d(
                conv_in, mix.conv1d.weight, bias=mix.conv1d.bias,
                padding=0 if n_past else k_tail, groups=conv_in.shape[1],
            )
            if not n_past:
                conv_out = conv_out[..., :length]
            qkv = F.silu(conv_out).transpose(1, 2)  # [1, L, conv_dim]
        new_tail = torch.cat([tail, x], dim=-1)[..., -k_tail:] if k_tail else tail
        query, key, value = torch.split(
            qkv, [mix.key_dim, mix.key_dim, mix.value_dim], dim=-1
        )
        query = query.reshape(1, length, -1, mix.head_k_dim)
        key = key.reshape(1, length, -1, mix.head_k_dim)
        value = value.reshape(1, length, -1, mix.head_v_dim)
        if rep > 1:
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)
        core, state = kernel(
            query,
            key,
            value,
            g=g.narrow(1, start, length),
            beta=beta.narrow(1, start, length),
            initial_state=states[off] if n_past else None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        outs.append(core)
        finals.append((state, new_tail))
        off += n_past

    core = outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)  # [1, T, Hv, Dv]
    core = core.reshape(-1, mix.head_v_dim)
    core = mix.norm(core, z.reshape(-1, mix.head_v_dim))
    core = core.reshape(1, h.shape[1], -1)
    h_out = residual + mix.out_proj(core)

    residual = h_out
    h_mlp = layer.mlp(layer.post_attention_layernorm(h_out))
    h_out = residual + h_mlp
    return (h_out, *(t for pair in finals for t in pair))


def _layer_fn(layer, h, cos, sin, meta, *past):
    """One packed chunk-layer: `(h, cos, sin, meta, past...) -> (h, k_0, v_0, ...)`.

    `meta` is one `(start, length, n_past)` triple per segment of the packed
    stream; `past` holds every segment's ancestor Ks in segment order, then
    every segment's ancestor Vs. Runs in BOTH executor phases -- the packed
    no-grad primal (all segments at once, batched dense ops) and the
    per-segment recompute at backward (a single segment, grad-enabled). The
    attention context is rebuilt from the arguments each time, so both
    executions see identical inputs; the captured per-segment K/V is returned
    flat so the executor can surface each as a graph output.

    Linear-attention layers of a hybrid stack dispatch to `_linear_layer_fn`,
    whose per-segment outputs are (recurrent state, conv tail) instead of
    (K, V) -- the same flat convention, so `_SegLayerCkpt` and both executor
    phases treat the two layer kinds identically.
    """
    if hasattr(layer, "linear_attn"):
        return _linear_layer_fn(layer, h, meta, *past)
    device = h.device
    prev = _CKPTS.get(device)
    n_total = sum(m[2] for m in meta)
    ks_flat, vs_flat = past[:n_total], past[n_total:]
    segs, off = [], 0
    for start, length, n_past in meta:
        segs.append(
            (start, length, list(ks_flat[off : off + n_past]), list(vs_flat[off : off + n_past]))
        )
        off += n_past
    _CKPTS[device] = {
        "segments": segs,
        "captured": None,
        # The no-grad primal remains byte-for-byte on the established path.
        # Ragged cohort attention is applied only by the checkpoint backward
        # recompute in `_RaggedQwen3LayerCkpt`.
        "ragged_attention": False,
    }
    try:
        out = layer(
            h,
            position_embeddings=(cos, sin),
            attention_mask=None,
            use_cache=False,
        )
        h_out = out[0] if isinstance(out, tuple) else out
        captured = _CKPTS[device]["captured"]
    finally:
        if prev is None:
            _CKPTS.pop(device, None)
        else:
            _CKPTS[device] = prev
    if captured is None:
        raise RuntimeError(
            "the layer ran without the stream attention capturing its K/V; "
            f"load the model with attn_implementation={STREAM_ATTENTION_NAME!r} "
            "after register_stream_attention()"
        )
    return (h_out, *(t for kv in captured for t in kv))


def _qwen3_ragged_layer_vjp_fn(
    layer,
    hidden_segments: tuple[torch.Tensor, ...],
    cos_segments: tuple[torch.Tensor, ...],
    sin_segments: tuple[torch.Tensor, ...],
    meta: tuple[tuple[int, int, int], ...],
    *past: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """A dense decoder layer with segment-local dense ops and cohort attention.

    This function is the recompute body for the feature-gated OPD cohort VJP.
    Every layernorm, projection, LoRA adapter, residual operation, and MLP is
    invoked once per segment with the same shape and operation order used by
    :class:`_SegLayerCkpt`.  Only attention is combined: the independently
    projected Q/K/V tensors enter one variable-length flash-attention call.

    The result is flattened as ``(h_0, k_0, v_0, h_1, k_1, v_1, ...)``.  Fresh
    K/V remain explicit checkpoint outputs and ancestor K/V remain explicit
    inputs, preserving the cross-block gradient channels.
    """
    if type(layer).__name__ != "Qwen3DecoderLayer":
        raise NotImplementedError(
            "ragged cohort VJP currently supports Qwen3DecoderLayer only"
        )
    n_segments = len(meta)
    if not (
        n_segments == len(hidden_segments)
        == len(cos_segments)
        == len(sin_segments)
    ):
        raise ValueError("ragged layer inputs have inconsistent segment counts")
    n_total = sum(item[2] for item in meta)
    if len(past) != 2 * n_total:
        raise ValueError("ragged layer received inconsistent ancestor K/V")
    ks_flat, vs_flat = past[:n_total], past[n_total:]

    # Use the model family's implementation so rotary arithmetic and any
    # registered kernel override remain identical to an independent
    # decoder-layer call.
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    attention = layer.self_attn
    queries: list[torch.Tensor] = []
    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []
    residuals: list[torch.Tensor] = []
    input_shapes: list[torch.Size] = []
    off = 0
    for hidden, cos, sin, (_start, length, n_past) in zip(
        hidden_segments,
        cos_segments,
        sin_segments,
        meta,
        strict=True,
    ):
        if hidden.shape[0] != 1 or hidden.shape[1] != length:
            raise ValueError("ragged layer hidden shape does not match segment metadata")
        residuals.append(hidden)
        input_shape = hidden.shape[:-1]
        input_shapes.append(input_shape)
        hidden_shape = (*input_shape, -1, attention.head_dim)
        normalized = layer.input_layernorm(hidden)
        query = attention.q_norm(
            attention.q_proj(normalized).view(hidden_shape)
        ).transpose(1, 2)
        key = attention.k_norm(
            attention.k_proj(normalized).view(hidden_shape)
        ).transpose(1, 2)
        value = attention.v_proj(normalized).view(hidden_shape).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        fresh_key = key[0].transpose(0, 1)
        fresh_value = value[0].transpose(0, 1)
        ancestor_keys = list(ks_flat[off : off + n_past])
        ancestor_values = list(vs_flat[off : off + n_past])
        off += n_past
        queries.append(query)
        keys.append(torch.cat([*ancestor_keys, fresh_key], dim=0))
        values.append(torch.cat([*ancestor_values, fresh_value], dim=0))
        captured.append((fresh_key, fresh_value))

    attended = _RaggedAttentionIndependentVJP.apply(
        attention.scaling,
        n_segments,
        *queries,
        *keys,
        *values,
    )
    outputs: list[torch.Tensor] = []
    for residual, input_shape, attention_output, (key, value) in zip(
        residuals,
        input_shapes,
        attended,
        captured,
        strict=True,
    ):
        attention_output = attention_output.transpose(1, 2)
        attention_output = attention_output.reshape(*input_shape, -1).contiguous()
        hidden = residual + attention.o_proj(attention_output)
        residual = hidden
        hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
        outputs.extend((hidden, key, value))
    return tuple(outputs)


class _SegLayerCkpt(torch.autograd.Function):
    """Checkpoint with a batched primal: one segment-layer of a packed stream.

    Forward adopts the segment's slice of the PACKED no-grad layer outputs
    (computed once, batched, outside this op) and returns contiguous clones;
    backward recomputes THIS SEGMENT ALONE through `_layer_fn` and
    differentiates that recomputation. Same economics as `torch.utils.checkpoint`
    -- inputs saved, activations rematerialised -- except the primal values come
    from the batched pass, which is coalescing's entire point: the
    dense kernels run once over all concurrent segments while the autograd
    graph stays PER SEGMENT.

    Per-segment graphs are load-bearing, not an implementation detail: the reward-linear
    backward closes trajectories one at a time with `retain_graph=False` backwards. Had
    the pack been recorded as one shared graph node, the first trajectory's
    backward would free that node's saved tensors and every window-mate's later
    backward would die on a spent graph. With one node per segment, a
    trajectory's backward traverses only its own segments' nodes -- coalescing
    changes the kernel schedule, never the graph topology the caller reasons
    about. (The clones serve the same isolation for storage: a trajectory's
    resident K/V shares no buffer with its window-mates, so per-trajectory
    freeing returns real bytes.)

    The layer's parameters are explicit inputs, so their gradients flow through
    the engine's normal routing, and `save_for_backward` gives them the
    version-counter guard: an optimizer step between a streamed forward and its
    deferred backward fails loudly (one optimizer step per batch), never silently.
    """

    @staticmethod
    def forward(
        ctx,
        layer,
        primal,
        cos,
        sin,
        n_past,
        input_storage_device,
        offload_stats,
        h,
        *rest,
    ):
        # rest = the segment's ancestor Ks, then Vs (n_past each), then params.
        # `primal` is a plain tuple, invisible to autograd on purpose: its
        # values are adopted, not differentiated through.
        ctx.layer = layer
        ctx.n_past = n_past
        inputs = (h, cos, sin, *rest[: 2 * n_past])
        params = rest[2 * n_past :]
        ctx.input_devices = tuple(value.device for value in inputs)
        ctx.input_count = len(inputs)
        ctx.offload_stats = offload_stats
        ctx.offloaded_bytes = 0
        ctx.offloaded_count = 0
        if input_storage_device is None:
            stored_inputs = inputs
        else:
            # Only the segment's OWN inputs move: the ancestor K/V are the
            # store's resident tensors (or their turn proxies), referenced
            # by every descendant, so copying them per segment would be
            # quadratic in the chain and would duplicate what stays on the
            # device anyway. For a prompt root (n_past 0) this is the whole
            # input set, as before.
            started = time.perf_counter()
            storage_device = torch.device(input_storage_device)
            own = inputs[:3]
            stored_own = tuple(
                value.detach().to(device=storage_device, copy=True)
                for value in own
            )
            stored_inputs = (*stored_own, *inputs[3:])
            ctx.offloaded_count = len(own)
            ctx.offloaded_bytes = sum(
                value.numel() * value.element_size() for value in own
            )
            if offload_stats is not None:
                offload_stats["write_s"] += time.perf_counter() - started
                offload_stats["write_bytes"] += ctx.offloaded_bytes
                offload_stats["tensor_count"] += len(own)
                offload_stats["current_bytes"] += ctx.offloaded_bytes
                offload_stats["peak_bytes"] = max(
                    offload_stats["peak_bytes"], offload_stats["current_bytes"]
                )
        ctx.save_for_backward(*stored_inputs, *params)
        h_out, k, v = primal
        return h_out.clone(), k.clone(), v.clone()

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gh, gk, gv):
        saved = ctx.saved_tensors
        stored_inputs = saved[: ctx.input_count]
        params = list(saved[ctx.input_count :])
        if ctx.offloaded_bytes:
            started = time.perf_counter()
            moved = ctx.offloaded_count
            inputs = [
                value.to(device=device)
                for value, device in zip(
                    stored_inputs[:moved], ctx.input_devices[:moved], strict=True
                )
            ]
            inputs.extend(stored_inputs[moved:])
            stats = ctx.offload_stats
            if stats is not None:
                stats["read_s"] += time.perf_counter() - started
                stats["read_bytes"] += ctx.offloaded_bytes
                stats["current_bytes"] -= ctx.offloaded_bytes
        else:
            inputs = list(stored_inputs)
        h, cos, sin, *past = inputs
        n = ctx.n_past
        if len(past) != 2 * n:
            raise RuntimeError("checkpoint input storage metadata is inconsistent")
        layer_index = _checkpoint_layer_index(ctx.layer)
        profile_fields = None
        if operator_profiling_enabled():
            profile_fields = {
                "role": device_role(h.device),
                "device": h.device,
                "layer": layer_index,
                "tokens": h.shape[1],
                "hidden": h.shape[2],
                "ancestor_chunks": n,
                "ancestor_tokens": sum(
                    tensor.shape[0] for tensor in past[:n]
                ),
                "dtype": h.dtype,
            }
        with operator_range(
            "segment_layer_checkpoint_backward", fields=profile_fields
        ):
            h_d = h.detach().requires_grad_(True)
            past_d = [t.detach().requires_grad_(True) for t in past]
            with operator_range(
                "checkpoint_layer_recompute", fields=profile_fields
            ):
                with torch.enable_grad():
                    out = _layer_fn(
                        ctx.layer,
                        h_d,
                        cos,
                        sin,
                        ((0, h.shape[1], n),),
                        *past_d,
                    )
            leaves = [h_d, *past_d, *params]
            need = [t.requires_grad for t in leaves]
            with operator_range("checkpoint_layer_vjp", fields=profile_fields):
                grads = torch.autograd.grad(
                    out,
                    [t for t, w in zip(leaves, need, strict=True) if w],
                    (gh, gk, gv),
                    allow_unused=True,
                )
            it = iter(grads)
            full = [next(it) if w else None for w in need]
        # inputs were: layer, primal, cos, sin, n_past, storage, stats,
        # h, *past, *params.
        return (None, None, None, None, None, None, None, *full)


class _PackedLayerCkpt(torch.autograd.Function):
    """Checkpoint one complete closure cohort as a packed layer operation.

    The no-grad primal already executes every ready segment together.  This
    operator adopts that packed primal as one autograd node and repeats the
    same packed layer operation once during backward.  Consequently, dense
    projection and MLP backward kernels retain the closure-cohort shape rather
    than executing once per segment as :class:`_SegLayerCkpt` requires.

    This graph topology has a strict ownership contract: every loss that can
    reach any output of the packed node must participate in the same autograd
    traversal.  It is therefore used only by the explicitly gated OPD
    multi-trajectory close, which sums all cohort losses before backward.
    Streaming GRPO and independently closed trajectories continue to use
    :class:`_SegLayerCkpt`.

    ``meta`` and ``primal`` are non-tensor inputs by design.  ``primal`` holds
    the packed no-grad values to adopt; ``meta`` reproduces the exact segment
    visibility during recomputation.  Ancestor K/V tensors and parameters are
    explicit inputs, preserving cross-segment gradients and parameter version
    checks.
    """

    @staticmethod
    def forward(
        ctx,
        ownership,
        layer,
        primal,
        meta,
        n_past,
        h,
        cos,
        sin,
        *rest,
    ):
        # rest = every segment's ancestor Ks, then every segment's ancestor
        # Vs (n_past each), then layer parameters.
        ctx.layer = layer
        ctx.ownership = ownership
        ctx.meta = meta
        ctx.n_past = n_past
        ctx.save_for_backward(h, cos, sin, *rest)
        return tuple(t.clone() for t in primal)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grad_outputs):
        if not ctx.ownership["validated"]:
            expected = ctx.ownership["expected_trajectory_ids"]
            seen = ctx.ownership["seen_trajectory_ids"]
            if seen != expected:
                missing = sorted(expected - seen)
                raise RuntimeError(
                    "closure-cohort VJP requires one joint backward over all "
                    f"scored cohort losses; missing trajectories {missing}"
                )
            ctx.ownership["validated"] = True
        h, cos, sin, *rest = ctx.saved_tensors
        n = ctx.n_past
        past, params = rest[: 2 * n], rest[2 * n :]
        h_d = h.detach().requires_grad_(True)
        past_d = [t.detach().requires_grad_(True) for t in past]
        with torch.enable_grad():
            out = _layer_fn(
                ctx.layer, h_d, cos, sin, ctx.meta, *past_d
            )

        # A packed output can be unused when a cohort member has no scored
        # descendant token.  Custom-function backward may materialize that
        # cotangent as either None or zeros depending on the PyTorch version;
        # normalize it here so the recompute VJP is version-independent.
        cotangents = tuple(
            torch.zeros_like(value) if grad is None else grad
            for value, grad in zip(out, grad_outputs, strict=True)
        )
        leaves = [h_d, *past_d, *params]
        need = [tensor.requires_grad for tensor in leaves]
        grads = torch.autograd.grad(
            out,
            [tensor for tensor, wanted in zip(leaves, need, strict=True) if wanted],
            cotangents,
            allow_unused=True,
        )
        iterator = iter(grads)
        full = [next(iterator) if wanted else None for wanted in need]
        h_grad, *past_and_param_grads = full
        # inputs: ownership, layer, primal, meta, n_past, h, cos, sin, ...
        return (
            None,
            None,
            None,
            None,
            None,
            h_grad,
            None,
            None,
            *past_and_param_grads,
        )


class _RaggedQwen3LayerCkpt(torch.autograd.Function):
    """Checkpoint a ready cohort while batching only its attention.

    The forward adopts the existing packed no-grad primal, preserving current
    logprob values.  Backward invokes every dense or LoRA module per segment
    and combines only the attention operation through
    :func:`_ragged_causal_sdpa`.  This removes the packed dense-GEMM reduction
    order that caused the BF16 parameter-update divergence of
    :class:`_PackedLayerCkpt` while retaining one cohort-width attention call.

    As with ``_PackedLayerCkpt``, every output must participate in one joint
    backward.  The operator is therefore restricted to an explicitly gated
    multi-trajectory OPD close.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, ownership, layer, primal, meta, n_past, *saved):
        n_segments = len(meta)
        ctx.layer = layer
        ctx.ownership = ownership
        ctx.meta = meta
        ctx.n_segments = n_segments
        ctx.n_past = n_past
        ctx.save_for_backward(*saved)
        hidden = primal[0]
        outputs: list[torch.Tensor] = []
        for index, (start, length, _count) in enumerate(meta):
            outputs.extend(
                (
                    hidden.narrow(1, start, length).clone(),
                    primal[1 + 2 * index].clone(),
                    primal[2 + 2 * index].clone(),
                )
            )
        return tuple(outputs)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grad_outputs):
        if not ctx.ownership["validated"]:
            expected = ctx.ownership["expected_trajectory_ids"]
            seen = ctx.ownership["seen_trajectory_ids"]
            if seen != expected:
                missing = sorted(expected - seen)
                raise RuntimeError(
                    "closure-cohort VJP requires one joint backward over all "
                    f"scored cohort losses; missing trajectories {missing}"
                )
            ctx.ownership["validated"] = True
        saved = list(ctx.saved_tensors)
        n_segments = ctx.n_segments
        n_past = ctx.n_past
        hidden = saved[:n_segments]
        cos = tuple(saved[n_segments : 2 * n_segments])
        sin = tuple(saved[2 * n_segments : 3 * n_segments])
        rest = saved[3 * n_segments :]
        past, params = rest[: 2 * n_past], rest[2 * n_past :]

        hidden_d = [tensor.detach().requires_grad_(True) for tensor in hidden]
        past_d = [tensor.detach().requires_grad_(True) for tensor in past]
        with torch.enable_grad():
            out = _qwen3_ragged_layer_vjp_fn(
                ctx.layer,
                tuple(hidden_d),
                cos,
                sin,
                ctx.meta,
                *past_d,
            )
        cotangents = tuple(
            torch.zeros_like(value) if grad is None else grad
            for value, grad in zip(out, grad_outputs, strict=True)
        )
        leaves = [*hidden_d, *past_d, *params]
        need = [tensor.requires_grad for tensor in leaves]
        grads = torch.autograd.grad(
            out,
            [tensor for tensor, wanted in zip(leaves, need, strict=True) if wanted],
            cotangents,
            allow_unused=True,
        )
        iterator = iter(grads)
        full = [next(iterator) if wanted else None for wanted in need]
        hidden_grads = full[:n_segments]
        past_and_param_grads = full[n_segments:]
        # Inputs: layer, primal, meta, n_past, h..., cos..., sin...,
        # ancestor K..., ancestor V..., params...
        return (
            None,
            None,
            None,
            None,
            None,
            *hidden_grads,
            *([None] * (2 * n_segments)),
            *past_and_param_grads,
        )


STREAM_ATTENTION_NAME = "thundersync_stream"


def register_stream_attention() -> None:
    from transformers.modeling_utils import AttentionInterface

    AttentionInterface.register(STREAM_ATTENTION_NAME, stream_attention_forward)


@dataclass
class _TurnNeed:
    """One turn of a model call, as the turn memory rule reads it.

    ``context`` is the number of ancestor tokens the turn attends (its
    trajectory position). ``new_ledger_tokens`` counts the ancestor tokens
    whose FP32 boundary ledger does not exist yet: the call's backward
    creates it, and the run holds it until the trajectory replays.
    """

    tokens: list[int]
    scored: int
    context: int
    new_ledger_tokens: int


@dataclass
class _TurnReplayBoundary:
    """Detached turn state whose descendant adjoint is replayed at close."""

    cid: int
    parent_chain: list[int]
    tokens: list[int]
    pos0: int
    proxy: list[torch.Tensor]
    adjoints: _Boundary = field(init=False, repr=False)

    def __post_init__(self) -> None:
        cid = self.cid
        # Retain only detached state here. Rebuilt graph outputs replace
        # `real` immediately before the one reverse-topological replay.
        self.adjoints = _Boundary(
            real=list(self.proxy),
            proxy=self.proxy,
            capture_source=lambda: ("turn_replay", cid),
        )


@dataclass
class _TrajState:
    chain: list[int]  # chunk ids: [prompt, turn1, turn2, ...]
    n_tokens: int  # trajectory-local position of the next token
    last_hidden: torch.Tensor  # [D], graph-attached; predicts the next chunk's first token
    logprobs: list[torch.Tensor] = field(default_factory=list)  # graph-attached, per chunk
    old_logprobs: list[torch.Tensor] = field(default_factory=list)  # detached twins
    turns: list[tuple[list[int], list[bool]]] = field(default_factory=list)
    turn_replay_boundaries: list[_TurnReplayBoundary] = field(default_factory=list)
    """(tokens, scored) per appended turn -- plain ints, kept so the reward-linear
    close can re-forward the whole branch in one packed pass
    (`rebuild_branch_logprobs`) instead of walking the per-event graph."""
    deferred_turns: list[tuple[list[int], list[bool]]] = field(default_factory=list)
    """Trailing unscored turns recorded in `turns` whose forward has not run
    (`_DEFER_UNSCORED_TURNS`)."""


@dataclass
class _GroupState:
    prompt_chain: list[int]
    """Chunks read by trajectory branches, in prompt order."""
    owned_prompt_chunks: list[int]
    """Prompt K/V entries released when this group closes."""
    prompt_len: int
    prompt_last_hidden: torch.Tensor  # shared by every trajectory's first turn


def _ledger_accumulate(
    ledgers: list[torch.Tensor | None], index: int, contribution: torch.Tensor
) -> None:
    """Add one adjoint contribution to its owned FP32 (FP64) ledger.

    The first contribution becomes the ledger as one widening copy; later
    ones are added by a mixed-dtype ``add_``, which widens each element
    exactly before the FP32 add, so the ledger holds the same bits as
    widening first and adding after, without the widened temporary.
    """

    accumulator = ledgers[index]
    if accumulator is None:
        dtype = (
            torch.float64 if contribution.dtype == torch.float64 else torch.float32
        )
        ledgers[index] = contribution.to(dtype=dtype, copy=True)
    else:
        accumulator.add_(contribution)


@dataclass
class _Boundary:
    """The graph cut at a shared forest node.

    Branch chunks attend to `proxy` tensors -- detached views of the prompt's
    K/V (same storage, fresh autograd identity, requires_grad). A trajectory's
    backward therefore stops AT the boundary instead of re-traversing the shared
    prompt subgraph; the gradients it deposits on the proxies are accumulated,
    and the `real` prompt subgraph is backwarded exactly once per group with the
    combined boundary gradient. This is what keeps CSE intact in the backward:
    without the cut, G per-trajectory backwards each traverse the prompt,
    costing ~2x logical instead of ~1x physical.
    """

    real: list[torch.Tensor]  # graph-attached prompt outputs, fixed order
    proxy: list[torch.Tensor]  # detached twins branch chunks actually read
    # Cross-task group boundaries contain graph-attached group-local suffix
    # outputs and leaf proxies belonging to shared nodes. The latter can use
    # the fused consumer-adjoint reduction. Empty means an ordinary boundary.
    shared_targets: list[torch.Tensor | None] = field(default_factory=list)
    capture_source: Callable[[], Any | None] | None = field(
        default=None,
        repr=False,
    )
    track_source_ownership: bool = field(default=False, repr=False)
    _adjoints: list[torch.Tensor | None] = field(init=False, repr=False)
    _contribution_counts: list[int] = field(init=False, repr=False)
    _pending_source: Any | None = field(default=None, init=False, repr=False)
    _lifetime_contribution_count: int = field(
        default=0,
        init=False,
        repr=False,
    )
    _drain_count: int = field(default=0, init=False, repr=False)
    _hook_handles: list[Any] = field(init=False, repr=False)
    _consumed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """Install one owned FP32 adjoint ledger per boundary tensor."""
        if len(self.real) != len(self.proxy):
            raise ValueError("boundary real and proxy tensors must align")
        if self.shared_targets and len(self.shared_targets) != len(self.real):
            raise ValueError("boundary shared targets must align with tensors")
        self._adjoints = [None] * len(self.proxy)
        self._contribution_counts = [0] * len(self.proxy)
        self._hook_handles = []
        for index, proxy in enumerate(self.proxy):
            if not proxy.is_leaf or not proxy.requires_grad:
                raise ValueError(
                    "boundary proxies must be gradient-requiring leaves"
                )
            self._hook_handles.append(
                proxy.register_post_accumulate_grad_hook(
                    self._capture_post_accumulate(index)
                )
            )

    def _capture_post_accumulate(
        self,
        index: int,
    ) -> Callable[[torch.Tensor], None]:
        capture_source = self.capture_source

        def capture(proxy: torch.Tensor) -> None:
            source = None if capture_source is None else capture_source()
            if source is None:
                return
            contribution = proxy.grad
            if contribution is None:
                raise RuntimeError(
                    "boundary post-accumulate hook received no gradient"
                )
            _ledger_accumulate(self._adjoints, index, contribution.detach())
            self._contribution_counts[index] += 1
            self._lifetime_contribution_count += 1
            if self.track_source_ownership:
                if self._pending_source is None:
                    self._pending_source = source
                elif self._pending_source != source:
                    raise RuntimeError(
                        "source-ready boundary received mixed sources before "
                        f"drain: {self._pending_source!r} and {source!r}"
                    )
            proxy.grad = None

        return capture

    def accumulate_explicit_adjoints(
        self,
        gradients: list[torch.Tensor | None],
        *,
        source: Any,
    ) -> None:
        """Add an explicit VJP result to the owned FP32 ledger."""
        if len(gradients) != len(self.proxy):
            raise ValueError("boundary gradients and proxies must align")
        has_contribution = any(gradient is not None for gradient in gradients)
        if self._consumed:
            if has_contribution:
                raise RuntimeError(
                    "a consumed boundary received an explicit adjoint"
                )
            return
        if self.has_native_adjoint():
            raise RuntimeError(
                "shared-forest boundary-adjoint ownership violation before "
                "explicit accumulation"
            )
        if self.track_source_ownership and has_contribution:
            if self._pending_source is None:
                self._pending_source = source
            elif self._pending_source != source:
                raise RuntimeError(
                    "source-ready boundary received mixed explicit sources "
                    f"before drain: {self._pending_source!r} and {source!r}"
                )
        for index, gradient in enumerate(gradients):
            if gradient is None:
                continue
            _ledger_accumulate(self._adjoints, index, gradient.detach())
            self._contribution_counts[index] += 1
            self._lifetime_contribution_count += 1

    def consume_adjoint_pairs(
        self,
        *,
        reject_native: bool = False,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Consume the boundary once while preserving FP32 accumulation."""
        if self._consumed:
            raise RuntimeError("boundary adjoints were consumed twice")
        if reject_native and self.has_native_adjoint():
            raise RuntimeError(
                "shared-forest boundary-adjoint ownership violation before "
                "boundary finalization"
            )
        outputs: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []
        for real, proxy, ledger in zip(
            self.real,
            self.proxy,
            self._adjoints,
            strict=True,
        ):
            native = proxy.grad
            if ledger is None and native is None:
                continue
            if ledger is None:
                reduced = native
            elif native is None:
                reduced = ledger
            else:
                # The mixed-dtype add widens the native term exactly.
                reduced = ledger.add(native.detach())
            assert reduced is not None
            outputs.append(real)
            gradients.append(reduced.to(dtype=real.dtype, device=real.device))
            proxy.grad = None
        self._consumed = True
        self._remove_hooks()
        return outputs, gradients

    def drain_adjoint_pairs(
        self,
        *,
        expected_source: Any,
        reject_native: bool = False,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Drain one source while retaining hooks and the primal graph."""
        if self._consumed:
            raise RuntimeError("a consumed boundary cannot be drained")
        if reject_native and self.has_native_adjoint():
            raise RuntimeError(
                "shared-forest boundary-adjoint ownership violation before "
                "source-ready drain"
            )
        if not self.track_source_ownership:
            raise RuntimeError(
                "source-ready drain requires source ownership tracking"
            )
        if (
            self._pending_source is not None
            and self._pending_source != expected_source
        ):
            raise RuntimeError(
                "source-ready boundary contains adjoints from a different "
                f"source: expected {expected_source!r}, observed "
                f"{self._pending_source!r}"
            )
        outputs: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []
        for real, proxy, ledger in zip(
            self.real,
            self.proxy,
            self._adjoints,
            strict=True,
        ):
            native = proxy.grad
            if ledger is None and native is None:
                continue
            if ledger is None:
                reduced = native
            elif native is None:
                reduced = ledger
            else:
                # The mixed-dtype add widens the native term exactly.
                reduced = ledger.add(native.detach())
            assert reduced is not None
            outputs.append(real)
            gradients.append(reduced.to(dtype=real.dtype, device=real.device))
        for index, proxy in enumerate(self.proxy):
            proxy.grad = None
            self._adjoints[index] = None
            self._contribution_counts[index] = 0
        self._pending_source = None
        self._drain_count += 1
        return outputs, gradients

    def adjoint_diagnostics(self) -> dict[str, Any]:
        active = [tensor for tensor in self._adjoints if tensor is not None]
        return {
            "accumulator_dtype": None if not active else str(active[0].dtype),
            "accumulator_tensor_count": len(active),
            "accumulator_bytes": sum(
                tensor.numel() * tensor.element_size() for tensor in active
            ),
            "contribution_count": sum(self._contribution_counts),
            "lifetime_contribution_count": self._lifetime_contribution_count,
            "drain_count": self._drain_count,
            "pending_source": self._pending_source,
            "max_contributions_per_tensor": max(
                self._contribution_counts,
                default=0,
            ),
        }

    def has_native_adjoint(self) -> bool:
        return any(proxy.grad is not None for proxy in self.proxy)

    def has_buffered_adjoint(self) -> bool:
        return (
            any(tensor is not None for tensor in self._adjoints)
            or self.has_native_adjoint()
            or self._pending_source is not None
        )

    def release(self, *, require_empty: bool = False) -> None:
        if require_empty and self.has_buffered_adjoint():
            raise RuntimeError(
                "source-ready boundary released with buffered adjoints"
            )
        self._remove_hooks()
        for proxy in self.proxy:
            proxy.grad = None
        self.real.clear()
        self.proxy.clear()
        self.shared_targets.clear()
        self._adjoints.clear()
        self._contribution_counts.clear()
        self._pending_source = None

    def _remove_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()


@dataclass
class _SharedPromptNode:
    """One physical radix edge and its reverse-mode dependencies.

    ``consumer_group_ids`` are the group-local alias chains that read this
    node directly. ``pending_child_cids`` records shared child VJPs that can
    still add an adjoint to this node. A node can execute after both dependency
    sets become empty, including while unrelated groups remain open.
    """

    cid: int
    parent_cid: int | None
    depth_start: int
    depth_end: int
    consumer_group_ids: tuple[int, ...]
    boundary: _Boundary
    child_cids: set[int] = field(default_factory=set)
    pending_group_ids: set[int] = field(default_factory=set)
    pending_child_cids: set[int] = field(default_factory=set)
    processed_group_ids: set[int] = field(default_factory=set)
    source_process_count: int = 0
    source_vjp_count: int = 0
    ancestor_shared_cids: tuple[int, ...] = ()
    finalized: bool = False
    finalization_index: int | None = None
    canonical_vjp_index: int | None = None


@dataclass(eq=False)
class _PromptPlanNode:
    """One radix edge planned before any prompt model execution."""

    group_ids: list[int]
    depth_start: int
    depth_end: int
    parent: _PromptPlanNode | None
    children: list[_PromptPlanNode] = field(default_factory=list)


class StreamingRun:
    """One batch's streamed execution. Events: open_group, append_turn (or
    append_turns for turns coalesced in one scheduling window), then the
    caller applies its own objective to `logprobs()` and calls backward.

    The objective boundary is identical to the batch trainer's: this class
    produces per-trajectory `log pi(token | context)` and nothing else. It never
    sees a reward. `old_logprobs()` returns the detached twins of the
    tensors the update backward flows through, so `old == new` holds
    bitwise at epoch 1.

    Attention dispatch and the checkpointed executor find the active run
    through module state keyed by device (`_STREAMS`, `_CKPTS`): at most one
    run may forward on a given device at a time. Runs on distinct devices
    (for example a student and a teacher) may execute concurrently.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        chunk: int = 1024,
        boundary_cut: bool = False,
        chunked_sdpa: bool = True,
        checkpoint: bool = False,
        stream_turns: bool = True,
        rebuild_block_tokens: int | None = 4096,
        rebuild_plain_reserve_bytes: int | None = None,
        prompt_checkpoint_storage_device: torch.device | str | None = None,
        rebuild_checkpoint_storage_device: torch.device | str | None = None,
        turn_boundary_rebuild: bool = False,
        boundary_adjoint_backend: ForestBoundaryAdjointBackend = "torch",
        boundary_adjoint_max_degree: int = 16,
        ragged_cohort_attention: bool = False,
        dependency_ready_shared_prefix_vjp: bool = False,
        shared_prefix_vjp_backend: SharedPrefixVJPBackend = "sequential",
        shared_prefix_vjp_timing: bool = False,
        shared_prefix_vjp_mode: str = "aggregate",
        explicit_adjoint_vjp: bool = False,
        parameter_adjoint_reduction: ParameterAdjointReduction = "arrival",
        parameter_adjoint_storage_device: torch.device | str | None = None,
    ) -> None:
        """`chunked_sdpa=False` selects the concat reference attention, whose
        retention grows as sum-of-prefixes (quadratic in turns). It is the
        baseline of the memory regression test and is not meant for training.

        `checkpoint=True` selects the per-layer checkpointed executor: each
        chunk-layer runs inside `torch.utils.checkpoint` with the fresh K/V as
        explicit checkpoint OUTPUTS and the ancestor K/V as explicit ARGS, so
        cross-chunk gradients survive recompute by construction. Retention
        drops from full activations to layer-boundary hiddens plus K/V. Cost:
        one recompute forward per chunk-layer
        during backward -- standard checkpointing economics.

        `stream_turns=True` (the default) forwards every turn as it arrives
        and retains the streamed per-turn graph; `logprobs()` reads it
        directly and the close-time rebuild is checked against it.
        `stream_turns=False` skips the per-turn forwards, whose outputs the
        close-time rebuild replaces: append_turn/append_turns become
        token+provenance bookkeeping on the CPU and the device runs only at
        open_group, the close-time branch rebuild, the group close, and the
        optimizer step. In this mode `logprobs()` requires `rebuild=True`
        and `old_logprobs()` is defined by the close pass at theta0
        (populated by the rebuild; the version guard forbids a step before
        it, so old == new holds by construction).

        `rebuild_block_tokens` selects the close-rebuild implementation (the
        memory budget selects an implementation, never a timing): an int
        rebuilds the branch in consecutive blocks of at most that many tokens
        (whole turns per block), each block one per-layer-checkpointed segment
        whose K/V feeds the next block -- close transient stays block-sized.
        `None` restores the plain whole-branch rebuild, which avoids a
        recompute pass but retains full-branch activations. Use it only when
        the prospective memory gate permits it.

        `prompt_checkpoint_storage_device` moves only the checkpoint inputs
        retained by independent prompt-root forwards to the named device.
        The layer parameters stay on the model device and retain their version
        guards. The reload runs inside the prompt-boundary VJP.

        `ragged_cohort_attention=True` enables the dense-decoder OPD cohort
        checkpoint whose backward preserves per-segment dense and LoRA module
        calls while executing cohort attention through one variable-length
        CUDA flash-attention operation. The default remains the established
        independent checkpoint topology.

        ``dependency_ready_shared_prefix_vjp=True`` executes a shared prompt
        node after every direct group consumer and shared child has completed.
        This moves exact shared-prefix work before the logical-batch barrier
        when the forest contains an independently completed subtree. The
        default finalizes shared nodes at the end of the batch.

        ``shared_prefix_vjp_backend="ready_frontier"`` submits all currently
        ready, dependency-independent nodes through one autograd call per
        reverse-topological frontier. ``"sequential"`` retains one autograd
        call per node. This setting changes execution granularity only; both
        backends consume the same node cotangents in child-before-parent order.
        ``shared_prefix_vjp_timing=True`` records CUDA events around each
        frontier; the diagnostics resolve device time after the caller's
        next synchronization. It is off by default.

        ``shared_prefix_vjp_backend="isolated_canonical"`` computes each
        shared node VJP with :func:`torch.autograd.grad`. Parameter-gradient
        contributions remain separate from global ``parameter.grad`` until
        the final safety gate, where both deferred and dependency-ready runs
        add them in one static child-before-parent order. Required parent
        proxy adjoints are produced at node completion and assembled in the
        same static order before the parent VJP. This backend isolates graph
        release time from parameter-gradient accumulation order.

        ``shared_prefix_vjp_mode="source_ready_serial"`` retains each
        physical node graph and executes one child-to-parent VJP per explicit
        objective source; it implies ``dependency_ready_shared_prefix_vjp``. ``explicit_adjoint_vjp=True`` obtains
        parameter and boundary cotangents through ``autograd.grad`` and writes
        them into the owned FP32 ledgers.

        ``parameter_adjoint_reduction="canonical_source"`` stages FP32
        contributions by VJP source and materializes them in a stable source
        order at logical-batch finalization.
        """
        assert_supported(model, allow_hybrid=True)
        if getattr(model, "is_gradient_checkpointing", False):
            raise NotImplementedError(
                "HF's gradient checkpointing severs cross-chunk K/V gradients "
                "(K/V stashed inside its no-grad region has no grad_fn); use "
                "StreamingRun(checkpoint=True), whose executor makes K/V "
                "checkpoint outputs instead"
            )
        self.stack = base_model(model)
        self.weight = unembedding(model)
        self.chunk = chunk
        self.boundary_cut = boundary_cut
        self.chunked_sdpa = chunked_sdpa
        self.checkpoint = checkpoint
        self.stream_turns = stream_turns
        self._prompt_checkpoint_storage_device = (
            None
            if prompt_checkpoint_storage_device is None
            else torch.device(prompt_checkpoint_storage_device)
        )
        # The close-time rebuild's per-layer checkpoint inputs (the block's
        # hidden state, cos, sin) go to this device between the branch's
        # forward and its backward. On the unstreamed path those inputs are
        # most of what a prepared source holds while it waits for its
        # reward, so with a device named here a held source keeps only its
        # K/V resident. The ancestor K/V are never
        # moved (see _SegLayerCkpt). None keeps the inputs on the device.
        self._rebuild_checkpoint_storage_device = (
            None
            if rebuild_checkpoint_storage_device is None
            else torch.device(rebuild_checkpoint_storage_device)
        )
        self._rebuild_checkpoint_offload_stats = {
            "write_s": 0.0,
            "read_s": 0.0,
            "write_bytes": 0,
            "read_bytes": 0,
            "tensor_count": 0,
            "current_bytes": 0,
            "peak_bytes": 0,
        }
        if turn_boundary_rebuild and not stream_turns:
            raise ValueError("turn-boundary rebuild requires stream_turns=True")
        if turn_boundary_rebuild and not boundary_cut:
            raise ValueError("turn-boundary rebuild requires boundary_cut=True")
        if turn_boundary_rebuild and model.training:
            raise ValueError(
                "turn-boundary rebuild requires model.eval() so replay is "
                "deterministic"
            )
        self.turn_boundary_rebuild = turn_boundary_rebuild
        if boundary_adjoint_backend not in BOUNDARY_ADJOINT_BACKENDS:
            raise ValueError(
                "boundary_adjoint_backend must be one of "
                f"{sorted(BOUNDARY_ADJOINT_BACKENDS)}, got "
                f"{boundary_adjoint_backend!r}"
            )
        if (
            not isinstance(boundary_adjoint_max_degree, int)
            or boundary_adjoint_max_degree < 1
        ):
            raise ValueError("boundary_adjoint_max_degree must be a positive int")
        if (
            boundary_adjoint_backend == "triton"
            and boundary_adjoint_max_degree > TRITON_MAX_DEGREE
        ):
            raise ValueError(
                "boundary_adjoint_max_degree exceeds the Triton kernel capacity "
                f"of {TRITON_MAX_DEGREE}"
            )
        self.boundary_adjoint_backend = boundary_adjoint_backend
        self.boundary_adjoint_max_degree = boundary_adjoint_max_degree
        if shared_prefix_vjp_mode not in {"aggregate", "source_ready_serial"}:
            raise ValueError(
                "shared_prefix_vjp_mode must be 'aggregate' or "
                "'source_ready_serial'"
            )
        if shared_prefix_vjp_mode == "source_ready_serial":
            dependency_ready_shared_prefix_vjp = True
            if boundary_adjoint_backend != "torch":
                raise ValueError(
                    "source_ready_serial requires boundary_adjoint_backend='torch'"
                )
            if shared_prefix_vjp_backend != "sequential":
                raise ValueError(
                    "source_ready_serial requires "
                    "shared_prefix_vjp_backend='sequential'"
                )
        if shared_prefix_vjp_backend not in SHARED_PREFIX_VJP_BACKENDS:
            raise ValueError(
                "shared_prefix_vjp_backend must be one of "
                f"{sorted(SHARED_PREFIX_VJP_BACKENDS)}, got "
                f"{shared_prefix_vjp_backend!r}"
            )
        self.dependency_ready_shared_prefix_vjp = bool(
            dependency_ready_shared_prefix_vjp
        )
        self.shared_prefix_vjp_backend = shared_prefix_vjp_backend
        self.shared_prefix_vjp_timing = bool(shared_prefix_vjp_timing)
        self.shared_prefix_vjp_mode = shared_prefix_vjp_mode
        self.explicit_adjoint_vjp = bool(explicit_adjoint_vjp)
        if parameter_adjoint_reduction not in {"arrival", "canonical_source"}:
            raise ValueError(
                "parameter_adjoint_reduction must be 'arrival' or "
                "'canonical_source'"
            )
        if (
            parameter_adjoint_reduction == "canonical_source"
            and shared_prefix_vjp_backend == "isolated_canonical"
        ):
            raise ValueError(
                "canonical source reduction and isolated_canonical cannot own "
                "the same parameter adjoints"
            )
        self.parameter_adjoint_reduction = parameter_adjoint_reduction
        self._parameter_adjoint_storage_device = (
            None
            if parameter_adjoint_storage_device is None
            else torch.device(parameter_adjoint_storage_device)
        )
        self._parameter_adjoint_offload_s = 0.0
        self._parameter_adjoint_materialize_s = 0.0
        self._manual_parameter_adjoint_call_count = 0
        self._manual_parameter_adjoint_peak_batch_bytes = 0
        self._parameter_adjoint_current_bytes = 0
        self._parameter_adjoint_peak_bytes = 0
        if (
            self.explicit_adjoint_vjp
            and shared_prefix_vjp_backend == "isolated_canonical"
        ):
            raise ValueError(
                "explicit_adjoint_vjp and isolated_canonical cannot own the "
                "same parameter adjoints"
            )
        self.ragged_cohort_attention = bool(ragged_cohort_attention)
        if self.ragged_cohort_attention:
            unsupported = sorted(
                {
                    type(layer).__name__
                    for layer in getattr(self.stack, "layers", ())
                    if type(layer).__name__ != "Qwen3DecoderLayer"
                }
            )
            if unsupported:
                raise NotImplementedError(
                    "ragged cohort attention currently supports "
                    f"Qwen3DecoderLayer layers only; found {unsupported}"
                )
        self._pending_shared_adjoints: dict[int, list[torch.Tensor]] = {}
        # Hybrid stacks (assert_supported's verified Gated DeltaNet family):
        # every forward runs through the per-layer executor, whatever the
        # checkpoint flag says -- a linear layer's boundary state must be
        # explicit graph I/O of its node, and only that executor makes it so.
        self._layer_kinds = hybrid_layer_kinds(model)
        if rebuild_block_tokens is not None and rebuild_block_tokens < 1:
            raise ValueError("rebuild_block_tokens must be >= 1 or None")
        self.rebuild_block_tokens = rebuild_block_tokens
        if (
            rebuild_plain_reserve_bytes is not None
            and rebuild_plain_reserve_bytes < 1
        ):
            raise ValueError(
                "rebuild_plain_reserve_bytes must be >= 1 or None"
            )
        self.rebuild_plain_reserve_bytes = rebuild_plain_reserve_bytes
        self._rebuild_path_counts = {"plain": 0, "blocked": 0}
        # The rung the last close rebuild ran on: "plain", "blocked" (block
        # rebuild, checkpoint inputs on the device),
        # "blocked_partial_host_checkpoints" (the leading blocks' inputs on
        # the host) or "blocked_host_checkpoints" (every input on the host).
        # And the branches, and branch blocks moved to the host, the block
        # rebuild's memory rule placed before rebuilding.
        self._last_rebuild_rung: str | None = None
        self._rebuild_rung_predictions = {
            "device_checkpoints": 0,
            "host_checkpoints": 0,
            "host_checkpoint_blocks": 0,
        }
        self._turn_path_counts = {
            "append": {"plain": 0, "checkpointed": 0},
            "replay": {"plain": 0, "checkpointed": 0},
            "unscored": {"deferred": 0, "dropped": 0},
        }
        # The turn memory rule's record: every turn model call's estimate
        # beside its measured peak, and the plain admissions it refused.
        self._memory_shape: dict[str, int] | None = None
        self._memory_gradient_bytes: list[tuple[torch.Tensor, int]] | None = None
        self._memory_window: dict[str, Any] | None = None
        self._memory_windows: list[dict[str, Any]] = []
        self._memory_refusals: collections.Counter[str] = collections.Counter()
        self._memory_min_available: int | None = None
        self._memory_retries_at_start: int | None = None
        if checkpoint or rebuild_block_tokens is not None or self._layer_kinds:
            # the block rebuild drives the checkpointed executor even on runs
            # whose streamed forwards are plain, so it needs the same access
            for attr in ("embed_tokens", "layers", "norm", "rotary_emb"):
                if not hasattr(self.stack, attr):
                    raise NotImplementedError(
                        f"checkpointed streaming drives the stack manually and "
                        f"needs `{attr}`; this model family does not expose it"
                    )
        self._kv = _StreamKV()
        _LIVE_RUNS.add(self)
        self._chunks: list[_Chunk] = []
        self._groups: dict[int, _GroupState] = {}
        self._trajs: dict[int, _TrajState] = {}
        self._boundaries: dict[int, _Boundary] = {}
        # Shared prompt nodes are appended parent-before-child. Finalization
        # walks this list in reverse so every child adjoint reaches its parent
        # proxy before the parent node is backwarded.
        self._shared_prompt_nodes: list[_SharedPromptNode] = []
        self._shared_prompt_nodes_by_cid: dict[int, _SharedPromptNode] = {}
        self._shared_prompt_nodes_by_group: dict[int, list[_SharedPromptNode]] = {}
        self._closed_shared_prompt_groups: set[int] = set()
        self._shared_prompt_finalization_events: list[dict[str, Any]] = []
        self._shared_prompt_source_vjp_events: list[dict[str, Any]] = []
        self._source_ready_sources_by_group: dict[int, list[Any]] = {}
        self._source_ready_source_keys_by_group: dict[int, set[Any]] = {}
        self._active_boundary_adjoint_source: Any | None = None
        run_ref = weakref.ref(self)

        def boundary_capture_source() -> Any | None:
            run = run_ref()
            return None if run is None else run._active_boundary_adjoint_source

        self._boundary_capture_source = boundary_capture_source
        self._parameter_adjoint_capture_enabled = False
        self._parameter_adjoint_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        self._parameter_adjoint_index = {
            id(parameter): index
            for index, parameter in enumerate(self._parameter_adjoint_parameters)
        }
        self._parameter_adjoints: list[torch.Tensor | None] = [
            None
        ] * len(self._parameter_adjoint_parameters)
        self._parameter_adjoint_contribution_counts = [
            0
        ] * len(self._parameter_adjoint_parameters)
        self._canonical_parameter_adjoints = (
            CanonicalGradientReducer(
                len(self._parameter_adjoint_parameters),
                storage_device=parameter_adjoint_storage_device,
            )
            if parameter_adjoint_reduction == "canonical_source"
            else None
        )
        self._parameter_adjoint_hook_handles: list[Any] = []
        self._parameter_adjoint_hook_finalizer: weakref.finalize | None = None
        self._parameter_adjoints_materialized = False
        self._adjoint_capture_failed = False
        self._adjoint_capture_failure_source: Any | None = None
        self._shared_prompt_vjp_calls = 0
        self._shared_prompt_vjp_frontier_widths: list[int] = []
        self._shared_prompt_vjp_cuda_intervals: list[
            tuple[torch.cuda.Event, torch.cuda.Event]
        ] = []
        self._shared_prompt_trainable_parameters: tuple[
            torch.nn.Parameter, ...
        ] = (
            tuple(
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            if shared_prefix_vjp_backend == "isolated_canonical"
            else ()
        )
        self._isolated_shared_parameter_gradients: dict[
            int, tuple[torch.Tensor | None, ...]
        ] = {}
        self._isolated_ancestor_proxy_adjoints: dict[
            int, dict[int, tuple[torch.Tensor | None, ...]]
        ] = {}
        self._isolated_parameter_gradients_accumulated = False
        self._isolated_parameter_gradient_bytes_current = 0
        self._isolated_parameter_gradient_bytes_peak = 0
        self._isolated_ancestor_adjoint_bytes_current = 0
        self._isolated_ancestor_adjoint_bytes_peak = 0
        self._logical_prompt_tokens = 0
        self._physical_prompt_tokens = 0
        self._prompt_topology_parts: list[dict[str, Any]] = []
        self._global_lcp_tokens = 0
        self._traj_group: dict[int, int] = {}
        self._active_chunk: int | None = None
        self._active_chain: list[int] | None = None
        # (cid, chain, length) per segment while a plain turn pack forwards.
        self._active_segments: list[tuple[int, list[int], int]] | None = None
        # The pack's attention layout, built at its first layer and read by
        # every other layer of the same model call (`_plain_packed_attention`).
        self._active_pack_layouts: dict[tuple, _PackLayout | None] = {}
        if (
            self._canonical_parameter_adjoints is not None
            or self._parameter_adjoint_storage_device is not None
        ):
            self._enable_parameter_adjoint_capture()

    # ------------------------------------------------------------------ events

    def open_group(self, group_id: int, prompt_tokens: list[int]) -> None:
        """The prompt is known before generation starts; forward it once, now."""
        if group_id in self._groups:
            raise ValueError(f"group {group_id} already open")
        cid = self._new_chunk(len(prompt_tokens), parent=None)
        hidden = self._forward(cid, [cid], prompt_tokens, pos0=0)
        last_hidden = hidden[-1]

        if self.boundary_cut:
            # Cut the graph at the shared node: branches read detached twins
            # (same storage), so per-trajectory backwards stop here instead of
            # re-traversing the prompt subgraph G times.
            real: list[torch.Tensor] = []
            proxy: list[torch.Tensor] = []
            for layer in self._kv.layers():
                k, v = self._kv.get(layer, cid)
                pk = k.detach().requires_grad_(True)
                pv = v.detach().requires_grad_(True)
                self._kv.replace(layer, cid, pk, pv)
                real.extend((k, v))
                proxy.extend((pk, pv))
            ph = last_hidden.detach().requires_grad_(True)
            real.append(last_hidden)
            proxy.append(ph)
            self._boundaries[group_id] = _Boundary(
                real=real,
                proxy=proxy,
                capture_source=self._boundary_capture_source,
                track_source_ownership=(
                    self.shared_prefix_vjp_mode == "source_ready_serial"
                ),
            )
            last_hidden = ph

        self._groups[group_id] = _GroupState(
            prompt_chain=[cid],
            owned_prompt_chunks=[cid],
            prompt_len=len(prompt_tokens),
            prompt_last_hidden=last_hidden,
        )
        self._logical_prompt_tokens += len(prompt_tokens)
        self._physical_prompt_tokens += len(prompt_tokens)
        self._prompt_topology_parts.append(
            {
                "groups": [group_id],
                "offset": 0,
                "tokens": list(prompt_tokens),
            }
        )

    def open_groups_shared_prefix(
        self,
        prompts: dict[int, list[int]],
        *,
        internal_nodes: bool = True,
        coalesce_nodes: bool = False,
        max_coalesced_nodes: int | None = None,
    ) -> int:
        """Open exact cross-task prompts as a radix-compressed forest.

        Every maximal prefix shared by two or more groups is forwarded once,
        including internal clusters when the whole batch has no common token.
        A boundary cut is installed at each shared physical node. Group closure
        deposits branch adjoints into those proxies; finalization walks shared
        nodes in reverse construction order, so child adjoints reach a parent
        before that parent is backwarded once.

        Requires ``boundary_cut=True``. Returns the number of tokens common
        to every supplied prompt; internal cluster sharing may occur even
        when this value is zero. ``internal_nodes=False`` shares only the
        batch-wide LCP. ``coalesce_nodes=True`` packs independent nodes at
        the same tree depth into one checkpointed primal while retaining one
        graph per node; sequential node execution is the numerical reference.
        """
        if not prompts:
            raise ValueError("open_groups_shared_prefix needs at least one prompt")
        if self._groups or self._shared_prompt_nodes:
            raise RuntimeError("shared prompt groups must open on a fresh run")
        if len(set(prompts)) != len(prompts):
            raise ValueError("duplicate group ids")
        if any(not tokens for tokens in prompts.values()):
            raise ValueError("shared prompt groups require non-empty prompts")
        if max_coalesced_nodes is not None and max_coalesced_nodes < 1:
            raise ValueError("max_coalesced_nodes must be positive or None")

        ordered = sorted(prompts)
        values = [prompts[gid] for gid in ordered]
        global_common = min(len(tokens) for tokens in values)
        first = values[0]
        for offset in range(global_common):
            if any(tokens[offset] != first[offset] for tokens in values[1:]):
                global_common = offset
                break

        if not self.boundary_cut:
            raise ValueError(
                "cross-task shared prefixes require boundary_cut=True"
            )

        if self.shared_prefix_vjp_backend != "isolated_canonical":
            self._enable_parameter_adjoint_capture()

        chains: dict[int, list[int]] = {}
        owned: dict[int, list[int]] = {gid: [] for gid in ordered}
        last_hidden: dict[int, torch.Tensor] = {}
        physical_prompt_tokens = 0
        shared_target_by_id: dict[int, torch.Tensor] = {}

        def cut_shared_node(
            cid: int,
            hidden: torch.Tensor,
            *,
            parent_cid: int | None,
            depth_start: int,
            depth_end: int,
            consumer_group_ids: list[int],
        ) -> torch.Tensor:
            real: list[torch.Tensor] = []
            proxy: list[torch.Tensor] = []
            for layer in self._kv.layers():
                k, v = self._kv.get(layer, cid)
                pk = k.detach().requires_grad_(True)
                pv = v.detach().requires_grad_(True)
                self._kv.replace(layer, cid, pk, pv)
                real.extend((k, v))
                proxy.extend((pk, pv))
            real_last = hidden[-1]
            proxy_last = real_last.detach().requires_grad_(True)
            real.append(real_last)
            proxy.append(proxy_last)
            self._shared_prompt_nodes.append(
                _SharedPromptNode(
                    cid=cid,
                    parent_cid=parent_cid,
                    depth_start=depth_start,
                    depth_end=depth_end,
                    consumer_group_ids=tuple(sorted(consumer_group_ids)),
                    boundary=_Boundary(
                        real=real,
                        proxy=proxy,
                        capture_source=self._boundary_capture_source,
                        track_source_ownership=(
                            self.shared_prefix_vjp_mode
                            == "source_ready_serial"
                        ),
                    ),
                )
            )
            shared_target_by_id.update((id(tensor), tensor) for tensor in proxy)
            return proxy_last

        def plan_node(
            group_ids: list[int],
            offset: int,
            parent: _PromptPlanNode | None,
        ) -> _PromptPlanNode:
            """Build a deterministic compressed radix subtree without tensors."""
            if len(group_ids) == 1:
                gid = group_ids[0]
                end = len(prompts[gid])
                if end == offset:
                    raise AssertionError("empty singleton prompt suffix")
            else:
                end = min(len(prompts[gid]) for gid in group_ids)
                anchor = prompts[group_ids[0]]
                for position in range(offset, end):
                    if any(
                        prompts[gid][position] != anchor[position]
                        for gid in group_ids[1:]
                    ):
                        end = position
                        break
                if end == offset:
                    raise AssertionError("radix partition has no common token")

            node = _PromptPlanNode(
                group_ids=list(group_ids),
                depth_start=offset,
                depth_end=end,
                parent=parent,
            )
            remaining = [gid for gid in group_ids if len(prompts[gid]) > end]
            if not internal_nodes:
                node.children = [
                    plan_node([gid], end, node) for gid in remaining
                ]
                return node
            partitions: dict[int, list[int]] = {}
            for gid in remaining:
                partitions.setdefault(prompts[gid][end], []).append(gid)
            node.children = [
                plan_node(sorted(partitions[token]), end, node)
                for token in sorted(partitions)
            ]
            return node

        if internal_nodes:
            top_partitions: dict[int, list[int]] = {}
            for gid in ordered:
                top_partitions.setdefault(prompts[gid][0], []).append(gid)
            roots = [
                plan_node(sorted(top_partitions[token]), 0, None)
                for token in sorted(top_partitions)
            ]
        elif global_common:
            roots = [plan_node(ordered, 0, None)]
        else:
            roots = [plan_node([gid], 0, None) for gid in ordered]

        node_chain: dict[_PromptPlanNode, list[int]] = {}
        frontier = roots
        while frontier:
            next_frontier: list[_PromptPlanNode] = []
            capacity = max_coalesced_nodes or len(frontier)
            for pack_start in range(0, len(frontier), capacity):
                node_pack = frontier[pack_start : pack_start + capacity]
                cids: list[int] = []
                segments: list[_Seg] = []
                packed_tokens: list[int] = []
                packed_positions: list[int] = []
                start = 0
                for node in node_pack:
                    parent_chain = (
                        node_chain[node.parent]
                        if node.parent is not None
                        else []
                    )
                    length = node.depth_end - node.depth_start
                    cid = self._new_chunk(
                        length,
                        parent=parent_chain[-1] if parent_chain else None,
                    )
                    chain = parent_chain + [cid]
                    node_chain[node] = chain
                    cids.append(cid)
                    segments.append(
                        _Seg(cid=cid, chain=chain, start=start, length=length)
                    )
                    anchor = prompts[node.group_ids[0]]
                    packed_tokens.extend(
                        anchor[node.depth_start : node.depth_end]
                    )
                    packed_positions.extend(
                        range(node.depth_start, node.depth_end)
                    )
                    start += length

                can_pack = (
                    self.checkpoint
                    or bool(self._layer_kinds)
                    or not torch.is_grad_enabled()
                )
                if coalesce_nodes and len(node_pack) > 1 and can_pack:
                    hiddens = self._forward_packed(
                        segments,
                        packed_tokens,
                        packed_positions,
                        force_checkpoint=(
                            not self.checkpoint and not self._layer_kinds
                        ),
                    )
                else:
                    hiddens = []
                    for node, cid in zip(node_pack, cids, strict=True):
                        anchor = prompts[node.group_ids[0]]
                        hiddens.append(
                            self._forward(
                                cid,
                                node_chain[node],
                                anchor[node.depth_start : node.depth_end],
                                pos0=node.depth_start,
                            )
                        )

                for node, cid, hidden in zip(
                    node_pack, cids, hiddens, strict=True
                ):
                    physical_prompt_tokens += node.depth_end - node.depth_start
                    anchor = prompts[node.group_ids[0]]
                    self._prompt_topology_parts.append(
                        {
                            "groups": list(node.group_ids),
                            "offset": node.depth_start,
                            "tokens": list(
                                anchor[node.depth_start : node.depth_end]
                            ),
                        }
                    )
                    chain = node_chain[node]
                    if len(node.group_ids) > 1:
                        terminal_hidden = cut_shared_node(
                            cid,
                            hidden,
                            parent_cid=(chain[-2] if len(chain) > 1 else None),
                            depth_start=node.depth_start,
                            depth_end=node.depth_end,
                            consumer_group_ids=node.group_ids,
                        )
                    else:
                        gid = node.group_ids[0]
                        owned[gid].append(cid)
                        terminal_hidden = hidden[-1]

                    for gid in node.group_ids:
                        if len(prompts[gid]) == node.depth_end:
                            chains[gid] = list(chain)
                            last_hidden[gid] = terminal_hidden
                    next_frontier.extend(node.children)
            frontier = next_frontier

        self._initialize_shared_prompt_dependencies()

        for gid in ordered:
            real_chain = chains[gid]
            prompt_last_real = last_hidden[gid]
            # A group-specific proxy chain gives each task an independent
            # adjoint accumulator while proxies remain detached views of the
            # same physical prompt state. No K/V values are copied.
            alias_chain: list[int] = []
            previous: int | None = None
            for real_cid in real_chain:
                alias = self._new_chunk(
                    self._chunks[real_cid].n_tokens, parent=previous
                )
                for layer in self._kv.layers():
                    k, v = self._kv.get(layer, real_cid)
                    self._kv.put(
                        layer,
                        alias,
                        k.detach().requires_grad_(True),
                        v.detach().requires_grad_(True),
                    )
                alias_chain.append(alias)
                previous = alias

            group_real: list[torch.Tensor] = []
            group_proxy: list[torch.Tensor] = []
            group_shared_targets: list[torch.Tensor | None] = []
            for layer in self._kv.layers():
                for real_cid, alias_cid in zip(real_chain, alias_chain, strict=True):
                    rk, rv = self._kv.get(layer, real_cid)
                    pk, pv = self._kv.get(layer, alias_cid)
                    group_real.extend((rk, rv))
                    group_proxy.extend((pk, pv))
                    group_shared_targets.extend(
                        (
                            shared_target_by_id.get(id(rk)),
                            shared_target_by_id.get(id(rv)),
                        )
                    )
            prompt_last_proxy = prompt_last_real.detach().requires_grad_(True)
            group_real.append(prompt_last_real)
            group_proxy.append(prompt_last_proxy)
            group_shared_targets.append(
                shared_target_by_id.get(id(prompt_last_real))
            )
            self._boundaries[gid] = _Boundary(
                real=group_real,
                proxy=group_proxy,
                shared_targets=group_shared_targets,
                capture_source=self._boundary_capture_source,
                track_source_ownership=(
                    self.shared_prefix_vjp_mode == "source_ready_serial"
                ),
            )
            self._groups[gid] = _GroupState(
                prompt_chain=alias_chain,
                owned_prompt_chunks=owned[gid] + alias_chain,
                prompt_len=len(prompts[gid]),
                prompt_last_hidden=prompt_last_proxy,
            )
        self._logical_prompt_tokens = sum(len(prompt) for prompt in prompts.values())
        self._physical_prompt_tokens = physical_prompt_tokens
        self._global_lcp_tokens = global_common
        return global_common

    def append_turn(
        self,
        group_id: int,
        traj_id: int,
        tokens: list[int],
        scored: list[bool],
    ) -> None:
        """One emitted turn: forward at arrival, project only scored positions.

        `scored` is provenance, supplied by whoever produced the tokens: True on
        action tokens, False on observation tokens. Arbitrary interleavings
        within the turn are fine -- the projection is driven by the flags alone.
        """
        self.append_turns([(group_id, traj_id, tokens, scored)])

    def append_turns(
        self,
        items: list[tuple[int, int, list[int], list[bool]]],
        *,
        joint_backward: bool = False,
    ) -> None:
        """Coalesced event: turns of SEVERAL trajectories, one model call.

        `items` holds `(group_id, traj_id, tokens, scored)` for turns that
        arrived in the same scheduling window, across trajectories and groups.
        Forwarding them one at a time repeats small dense-kernel launches.
        Packing amortizes embedding, projection, and MLP launch overhead.
        So the items' tokens are CONCATENATED along the sequence dim (no
        padding, in keeping with this repo's no-rectangles rule), each segment
        keeping its own trajectory-local position ids, and forwarded as one
        packed stream: one embed, batched dense ops, per-segment attention over
        each trajectory's own ancestor chain -- never across segments.

        A trajectory may appear at most once per call: its turn k+1 is
        predicted from turn k's LAST hidden state, which does not exist until
        turn k's forward has run. Violations raise ValueError before any state
        is touched.

        ``joint_backward=True`` is the caller's promise that every scored
        logprob of these turns is differentiated in ONE autograd traversal
        before anything else backwards through them. Several turns then
        forward on the plain executor as one pack whose graph is shared
        across the turns; the caller sizes the pack with
        `joint_turn_pack_width` so it fits the plain executor's memory rule.
        It requires `joint_turn_pack_ready`.
        """
        if not items:
            raise ValueError("append_turns needs at least one item")
        if joint_backward and len(items) > 1 and not self.joint_turn_pack_ready:
            raise ValueError(
                "a jointly backwarded turn pack needs turn-boundary replay, "
                "streamed turns, chunked attention, a non-hybrid stack and "
                "arrival-order parameter adjoints (joint_turn_pack_ready)"
            )
        seen: set[int] = set()
        for group_id, traj_id, tokens, scored in items:
            if len(tokens) != len(scored):
                raise ValueError("tokens and scored must align")
            if not tokens:
                raise ValueError("an empty turn is not an event")
            if traj_id in seen:
                raise ValueError(
                    f"trajectory {traj_id} appears twice in one append_turns "
                    "call; its turn k+1 is predicted from turn k's last hidden "
                    "state, which does not exist until turn k has been forwarded"
                )
            seen.add(traj_id)
            if group_id not in self._groups:
                raise KeyError(group_id)
            if (
                traj_id in self._traj_group
                and self._traj_group[traj_id] != group_id
            ):
                raise ValueError(
                    f"trajectory {traj_id} belongs to group "
                    f"{self._traj_group[traj_id]}, received group {group_id}"
                )

        if not self.stream_turns:
            # The per-turn forward would be consumed by nothing -- old
            # logprobs are defined by the close-time rebuild at theta0 -- so
            # an append is pure token + provenance bookkeeping. No chunk is allocated, no K/V stored,
            # no CUDA work issued.
            for group_id, traj_id, tokens, scored in items:
                g = self._groups[group_id]
                traj = self._trajs.get(traj_id)
                if traj is None:
                    traj = _TrajState(
                        chain=list(g.prompt_chain),
                        n_tokens=g.prompt_len,
                        last_hidden=g.prompt_last_hidden,
                    )
                    self._trajs[traj_id] = traj
                    self._traj_group[traj_id] = group_id
                traj.n_tokens += len(tokens)
                traj.turns.append((list(tokens), list(scored)))
            return

        joint_pack = joint_backward and len(items) > 1
        if len(items) > 1 and not self.checkpoint and not joint_pack:
            # The plain executor retains the full activation graph, so a packed
            # forward would record graph nodes SHARED across trajectories; the
            # first per-trajectory backward through such a node would
            # free its saved tensors and corrupt every window-mate's pending
            # backward. The checkpointed executor coalesces with per-segment
            # nodes (_SegLayerCkpt); here the plain executor keeps exact
            # sequential semantics instead: one model call per turn.
            for item in items:
                self.append_turns([item])
            return

        if (
            _DEFER_UNSCORED_TURNS
            and self.turn_boundary_rebuild
            and len(items) == 1
            and not any(items[0][3])
        ):
            group_id, traj_id, tokens, scored = items[0]
            traj = self._traj_state(group_id, traj_id)
            traj.turns.append((list(tokens), list(scored)))
            traj.deferred_turns.append((list(tokens), list(scored)))
            self._turn_path_counts["unscored"]["deferred"] += 1
            return
        for _group_id, traj_id, _tokens, _scored in items:
            self.forward_deferred_turns(traj_id)
        self._forward_turns(items, joint_pack=joint_pack)

    @property
    def joint_turn_pack_ready(self) -> bool:
        """Whether several ready turns may run as one plain, jointly
        backwarded pack on this run.

        The pack's K/V and last hidden states must be handed to detached
        proxies when it returns (turn-boundary replay), so its losses are
        the only consumers of its shared graph. Parameter adjoints must
        accumulate in arrival order: a canonical per-source reduction, a
        source-ready shared-prefix drain or an explicit per-source VJP
        attributes each backward to one trajectory, and a pack's backward
        belongs to several.
        """

        return (
            self.stream_turns
            and self.turn_boundary_rebuild
            and self.chunked_sdpa
            and not self._layer_kinds
            and self._canonical_parameter_adjoints is None
            and self.shared_prefix_vjp_mode != "source_ready_serial"
            and not self.explicit_adjoint_vjp
        )

    def joint_turn_pack_width(
        self, items: list[tuple[int, int, list[int], list[bool]]]
    ) -> int:
        """How many leading turns of ``items`` one plain joint pack carries.

        The pack retains every layer's activations for all its tokens until
        its backward, so it is admitted by the rule a single plain turn
        reads (`_plain_turn_fits`) applied to the pack: the widest prefix
        whose estimated peak (`_turn_peak_estimate`) fits the memory
        available now with `rebuild_plain_reserve_bytes` spare. The device
        is read once for every width. A run without the checkpoint flag
        retains full activations by construction and packs every turn. 1
        means no pack: the first turn appends on its own under the
        single-turn rule.
        """

        if len(items) < 2 or not self.joint_turn_pack_ready:
            return 1
        if not self.checkpoint:
            return len(items)
        if self.rebuild_plain_reserve_bytes is None:
            return 1
        device = self.weight.device
        memory = device_memory_state(device) if device.type == "cuda" else None
        needs = [
            self._turn_need(group_id, traj_id, tokens, scored)
            for group_id, traj_id, tokens, scored in items
        ]
        for width in range(len(items), 1, -1):
            if self._plain_turn_fits(needs[:width], memory=memory):
                if width < len(items):
                    self._memory_refusals["pack_narrowed"] += 1
                return width
        self._memory_refusals["pack_refused"] += 1
        return 1

    @property
    def joint_turn_replay_ready(self) -> bool:
        """Whether the residual replays of several closed trajectories may
        run as one plain, jointly backwarded pack on this run.

        The conditions of `joint_turn_pack_ready`: the replay pack's graph is shared across its
        trajectories and its one backward belongs to all of them, so
        parameter adjoints must accumulate in arrival order.
        """

        return (
            self.stream_turns
            and self.turn_boundary_rebuild
            and self.chunked_sdpa
            and not self._layer_kinds
            and self._canonical_parameter_adjoints is None
            and self.shared_prefix_vjp_mode != "source_ready_serial"
            and not self.explicit_adjoint_vjp
        )

    def _turn_replay_pack(
        self, needs: list[_TurnNeed], memory: dict[str, int] | None
    ) -> list[int]:
        """Which of the ready replays ``needs`` one model call carries.

        ``needs`` holds one ready turn per cohort trajectory, in the cohort's
        order. Several turns run as one plain joint pack when the turn memory
        rule admits them together: each turn in order joins the pack if the
        pack's estimated peak with it (`_turn_peak_estimate`, replay) plus the
        reserve still fits the memory available now; a turn that does not fit
        waits for a later pack. Fewer than two admitted turns is no pack: the
        first turn replays on its own under the single-turn rule. A run
        without the checkpoint flag retains full activations by construction
        and packs every turn. Returns indices into ``needs``.
        """

        if len(needs) < 2 or not self.joint_turn_replay_ready:
            return [0]
        if not self.checkpoint:
            return list(range(len(needs)))
        if self.rebuild_plain_reserve_bytes is None:
            return [0]
        chosen: list[int] = []
        for index, need in enumerate(needs):
            candidate = [needs[i] for i in chosen] + [need]
            if self._plain_turn_fits(candidate, replay=True, memory=memory):
                chosen.append(index)
        if len(chosen) > 1:
            if len(chosen) < len(needs) and memory is not None:
                self._memory_refusals["replay_pack_narrowed"] += 1
            return chosen
        if memory is not None:
            self._memory_refusals["replay_pack_refused"] += 1
        return [0]

    def _traj_state(self, group_id: int, traj_id: int) -> _TrajState:
        """The trajectory's state, opened at its group's prompt on first use."""
        traj = self._trajs.get(traj_id)
        if traj is None:
            g = self._groups[group_id]
            traj = _TrajState(
                chain=list(g.prompt_chain),
                n_tokens=g.prompt_len,
                last_hidden=g.prompt_last_hidden,
            )
            self._trajs[traj_id] = traj
            self._traj_group[traj_id] = group_id
        return traj

    def forward_deferred_turns(self, traj_id: int) -> None:
        """Forward a trajectory's deferred unscored turns, oldest first.

        The trajectory's next arrival does this before its own forward, so
        each runs at the positions, ancestor state and parameters it would
        have had on arrival, as the single-turn call it arrived as.
        """
        traj = self._trajs.get(traj_id)
        if traj is None:
            return
        group_id = self._traj_group[traj_id]
        while traj.deferred_turns:
            tokens, scored = traj.deferred_turns[0]
            self._forward_turns([(group_id, traj_id, tokens, scored)], record=False)
            traj.deferred_turns.pop(0)
            # Nothing backwards through a deferred forward: its graph is
            # released when the call returns.
            self.close_memory_window()

    def _forward_turns(
        self,
        items: list[tuple[int, int, list[int], list[bool]]],
        *,
        record: bool = True,
        joint_pack: bool = False,
    ) -> None:
        """One model call over the turns; ``record=False`` for a deferred turn
        already in the trajectory's `turns`. ``joint_pack`` forwards several
        turns on the plain executor as one shared graph (the caller backwards
        them jointly and sized the pack to the plain memory rule).

        The call opens a memory window (`close_memory_window`) that records
        its estimated and measured peak; the caller closes it after the
        call's backward, or the next window does."""
        needs = [
            self._turn_need(group_id, traj_id, tokens, scored)
            for group_id, traj_id, tokens, scored in items
        ]
        memory = (
            device_memory_state(self.weight.device)
            if self.weight.device.type == "cuda"
            else None
        )
        if len(items) == 1:
            plain = self._plain_turn_append(needs, memory)
        else:
            plain = joint_pack
        if self.turn_boundary_rebuild:
            # Only here does the caller close the window after the call's
            # own backward (the reward-linear backward runs per trajectory).
            self._open_memory_window(
                "deferred" if not record else "pack" if len(items) > 1 else "append",
                plain,
                needs,
                memory,
            )
        # A plain pack's K/V and hidden states are views of the packed
        # tensors; their proxies get storage of their own so one
        # trajectory's release returns its bytes whatever its window-mates
        # still hold.
        shared_storage = plain and len(items) > 1
        segments: list[_Seg] = []
        packed_tokens: list[int] = []
        packed_pos: list[int] = []
        states = []
        start = 0
        for group_id, traj_id, tokens, scored in items:
            traj = self._traj_state(group_id, traj_id)
            cid = self._new_chunk(len(tokens), parent=traj.chain[-1])
            segments.append(
                _Seg(cid=cid, chain=traj.chain + [cid], start=start, length=len(tokens))
            )
            packed_tokens.extend(tokens)
            packed_pos.extend(range(traj.n_tokens, traj.n_tokens + len(tokens)))
            states.append((traj, tokens, scored))
            start += len(tokens)

        hiddens = self._forward_packed(
            segments,
            packed_tokens,
            packed_pos,
            plain=plain,
            joint_backward=shared_storage,
        )
        executor = self._executor_name(plain)
        self._turn_path_counts["append"][executor] += len(segments)

        for seg, hidden, (traj, tokens, scored) in zip(segments, hiddens, states, strict=True):
            pos0 = traj.n_tokens
            # Predictors: position j of the chunk is predicted by j-1's hidden
            # state; the first position by the previous chunk's last hidden
            # state, which is the cross-chunk boundary channel described in the
            # module docstring.
            scored_positions = [j for j, s in enumerate(scored) if s]
            if scored_positions:
                idx = _device_long(scored_positions, hidden.device)
                tgt = _device_long(
                    [tokens[j] for j in scored_positions], hidden.device
                )
                src = torch.cat([traj.last_hidden.unsqueeze(0), hidden[:-1]], dim=0)
                lp = fused_logprob(
                    src.index_select(0, idx), self.weight, tgt,
                    chunk=self.chunk,
                )
                traj.logprobs.append(lp)
                traj.old_logprobs.append(lp.detach())

            if self.turn_boundary_rebuild:
                proxies: list[torch.Tensor] = []
                for layer in self._kv.layers():
                    k, v = self._kv.get(layer, seg.cid)
                    k, v = k.detach(), v.detach()
                    if shared_storage:
                        k, v = k.clone(), v.clone()
                    pk = k.requires_grad_(True)
                    pv = v.requires_grad_(True)
                    self._kv.replace(layer, seg.cid, pk, pv)
                    proxies.extend((pk, pv))
                last = hidden[-1].detach()
                last_proxy = (
                    last.clone() if shared_storage else last
                ).requires_grad_(True)
                proxies.append(last_proxy)
                traj.turn_replay_boundaries.append(
                    _TurnReplayBoundary(
                        cid=seg.cid,
                        parent_chain=list(seg.chain[:-1]),
                        tokens=list(tokens),
                        pos0=pos0,
                        proxy=proxies,
                    )
                )
            else:
                last_proxy = hidden[-1]

            traj.chain = seg.chain
            traj.n_tokens += seg.length
            traj.last_hidden = last_proxy
            if record:
                traj.turns.append((list(tokens), list(scored)))

    # ----------------------------------------------------------------- results

    def logprobs(self, traj_id: int, *, rebuild: bool = False) -> torch.Tensor:
        """Graph-attached scored logprobs, in the trajectory's own token order.

        ``rebuild=True`` returns the same scored logprobs recomputed through
        one packed whole-branch forward -- a FRESH graph at close-time
        granularity instead of the retained per-event streamed graph (see
        `_rebuild_branch_logprobs`, the traj-bwd granularity lever). Values
        match the streamed ones to kernel-schedule numerics; the detached
        old-logprobs are the streamed values either way.
        """
        if rebuild:
            return self.rebuild_logprobs([traj_id])[traj_id]
        if not self.stream_turns:
            raise RuntimeError(
                "no streamed per-turn graph exists with "
                "stream_turns=False; the branch fires at close -- use "
                "logprobs(traj_id, rebuild=True)"
            )
        t = self._trajs[traj_id]
        if not t.logprobs:
            return torch.zeros(0, device=self.weight.device)
        return torch.cat(t.logprobs)

    def old_logprobs(self, traj_id: int) -> torch.Tensor:
        """The detached twins of `logprobs()` -- pi_old in trainer numerics, free.

        With stream_turns=False these are defined by the
        close-time rebuild at theta0; before the trajectory's close they do not
        exist, and asking for them is an ordering bug that fails loudly."""
        t = self._trajs[traj_id]
        if not t.old_logprobs:
            if not self.stream_turns and any(any(sc) for _, sc in t.turns):
                raise RuntimeError(
                    f"trajectory {traj_id}: old-logprobs are defined by the "
                    "close-time rebuild at theta0 (stream_turns=False); close the "
                    "trajectory first"
                )
            return torch.zeros(0, device=self.weight.device)
        return torch.cat(t.old_logprobs)

    def consume_latest_turn_logprobs(
        self, traj_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Remove and return the most recent turn's current and old scores."""
        t = self._trajs[traj_id]
        if not t.turns:
            raise RuntimeError(f"trajectory {traj_id} has no appended turn")
        scored_count = sum(t.turns[-1][1])
        if scored_count == 0:
            empty = torch.zeros(0, device=self.weight.device)
            return empty, empty.detach()
        if not t.logprobs or not t.old_logprobs:
            raise RuntimeError(
                f"trajectory {traj_id}: latest turn scores were already consumed"
            )
        current = t.logprobs.pop()
        old = t.old_logprobs.pop()
        if current.numel() != scored_count:
            raise RuntimeError(
                f"trajectory {traj_id}: latest turn has {scored_count} scored "
                f"tokens but {current.numel()} log probabilities"
            )
        return current, old

    @property
    def rebuild_checkpoint_storage_device(self) -> torch.device | None:
        """Where the block executor keeps checkpoint inputs by default."""

        return self._rebuild_checkpoint_storage_device

    @property
    def last_rebuild_rung(self) -> str | None:
        """Where the last close rebuild ran: "plain", "blocked",
        "blocked_partial_host_checkpoints" or "blocked_host_checkpoints"
        (None before the first)."""

        return self._last_rebuild_rung

    @property
    def rebuild_path_counts(self) -> dict[str, int]:
        """Branches rebuilt through each implementation, since construction.

        A residual turn replay is a one-turn rebuild; ``blocked`` counts the
        ones the checkpointed executor ran.
        """

        return dict(self._rebuild_path_counts)

    @property
    def turn_path_counts(self) -> dict[str, dict[str, int]]:
        """Diagnostic: streamed turn appends and residual turn replays, per executor."""

        return {
            phase: dict(counts) for phase, counts in self._turn_path_counts.items()
        }

    def _executor_name(self, plain: bool) -> str:
        """The executor `_forward_packed(plain=plain)` runs on this run."""

        if self._layer_kinds or (self.checkpoint and not plain):
            return "checkpointed"
        return "plain"

    def _plain_turn_append(
        self, needs: list[_TurnNeed], memory: dict[str, int] | None
    ) -> bool:
        """Whether one streamed turn may take the plain executor.

        Under turn-boundary replay the append swaps the turn's K/V and last
        hidden state for detached proxies before it returns, so the only
        consumer of its graph is the turn's own scored logprobs, which the
        objective differentiates at once; checkpointing would recompute a
        graph that is about to be consumed. Admission is the turn memory
        rule (`_plain_turn_fits`), read when the turn arrives.
        """

        if not (
            self.turn_boundary_rebuild
            and self.checkpoint
            and not self._layer_kinds
        ):
            return False
        if self._plain_turn_fits(needs, memory=memory):
            return True
        if memory is not None and self.rebuild_plain_reserve_bytes is not None:
            self._memory_refusals["append"] += 1
        return False

    def _turn_need(
        self, group_id: int, traj_id: int, tokens: list[int], scored: list[bool]
    ) -> _TurnNeed:
        """What appending ``tokens`` to the trajectory asks of the memory rule.

        Deferred unscored turns are forwarded first in the same call, so
        they add to the context, and their boundaries, like every other
        ancestor boundary no descendant has backwarded into yet, receive
        their FP32 ledger in this call's backward. A boundary is reached by
        every descendant's backward (attention reads all ancestor K/V), so
        once one boundary holds a ledger every older one does.
        """

        group = self._groups[group_id]
        traj = self._trajs.get(traj_id)
        if traj is None:
            context, deferred, boundaries = group.prompt_len, 0, []
            chain = group.prompt_chain
        else:
            deferred = sum(len(turn) for turn, _flags in traj.deferred_turns)
            context = traj.n_tokens + deferred
            boundaries = traj.turn_replay_boundaries
            chain = traj.chain
        new_ledger = deferred
        for boundary in reversed(boundaries):
            if boundary.adjoints.has_buffered_adjoint():
                break
            new_ledger += len(boundary.tokens)
        else:
            cut = self._boundaries.get(group_id)
            if cut is not None and not cut.has_buffered_adjoint():
                new_ledger += group.prompt_len
            for node in self._shared_prompt_nodes:
                if (
                    not node.finalized
                    and node.cid in chain
                    and not node.boundary.has_buffered_adjoint()
                ):
                    new_ledger += node.depth_end - node.depth_start
        return _TurnNeed(
            tokens=tokens,
            scored=sum(scored),
            context=context,
            new_ledger_tokens=new_ledger,
        )

    def _turn_memory_shape(self) -> dict[str, int]:
        """Bytes per token (or per call) of each term of the memory rule.

        Derived once from the model's configuration and dtype. A decoder
        layer's plain forward keeps, per token, for its backward: the FP32
        input and the normalized output of each RMSNorm (the two residual
        norms and the per-head query/key norms), the rotated query the
        attention op saves, the attention output the output projection
        saves (the attention op keeps the same tensor, with its FP32
        log-sum-exp per head), and the MLP's gate, up, activation and
        product. The new
        turn's K/V are kept by the run itself. The checkpointed executor
        keeps each layer's input hidden state and recomputes one layer at a
        time. The attention backward re-concatenates a segment's ancestor
        K/V and, on the expanded flash split, repeats it to the query heads,
        each with its gradient; a plain pack's (`_PackedFlashSplit`)
        concatenates every segment's at once, with their gradients and the
        per-query-head gradients the kernel sums over each head group,
        beside the pack's output gradient and query gradients (FP32 while
        the kernel accumulates them). Low-precision norms, the output head and
        every boundary ledger accumulate in FP32. An output head tied to
        the input embedding shares its weight, whose gradient the head's
        backward leaves pending until the embedding's backward, the last to
        run, adds its own (``tied_head_gradient``).
        """

        if self._memory_shape is not None:
            return self._memory_shape
        config = self.stack.config
        layers = int(config.num_hidden_layers)
        hidden = int(config.hidden_size)
        heads = int(config.num_attention_heads)
        kv_heads = int(getattr(config, "num_key_value_heads", None) or heads)
        head_dim = int(getattr(config, "head_dim", None) or hidden // heads)
        intermediate = int(getattr(config, "intermediate_size", None) or 4 * hidden)
        query, kv = heads * head_dim, kv_heads * head_dim
        element = self.weight.element_size()
        accumulate = max(element, 4)
        layer = (
            (accumulate + element) * (2 * hidden + query + kv)
            + element * (2 * query + 4 * intermediate)
            + accumulate * heads
        )
        vocab, width = self.weight.shape
        embed = getattr(getattr(self.stack, "embed_tokens", None), "weight", None)
        tied = isinstance(embed, torch.Tensor) and (
            embed is self.weight
            or (
                embed.data_ptr() == self.weight.data_ptr()
                and embed.shape == self.weight.shape
            )
        )
        self._memory_shape = {
            "plain_activation": int(layers * layer * TURN_ACTIVATION_FACTOR),
            "checkpoint_activation": int(
                (layers * element * hidden + layer) * TURN_ACTIVATION_FACTOR
            ),
            "kv": layers * 2 * kv * element,
            "hidden": hidden * element,
            "ledger": layers * 2 * kv * accumulate,
            "context": kv * element * (4 + 4 * max(1, heads // kv_heads)),
            "packed_context": kv * element * (4 + 2 * max(1, heads // kv_heads)),
            "packed_query": query * (4 * element + accumulate),
            "head_gradient": (
                vocab * width * element if self.weight.requires_grad else 0
            ),
            "embedding_gradient": (
                embed.numel() * embed.element_size()
                if isinstance(embed, torch.Tensor) and embed.requires_grad
                else 0
            ),
            "tied_head_gradient": (
                vocab * width * element
                if tied and self.weight.requires_grad
                else 0
            ),
            # the final norm saves its FP32 input, its normalized output and
            # the per-row inverse root for the backward
            "final_norm": (2 * hidden + 1) * accumulate,
            # a checkpointed segment keeps its rotary cos/sin and positions
            "rotary": 2 * head_dim * element + 8,
        }
        return self._memory_shape

    def _head_backward_bytes(self, scored: int) -> int:
        """The output head's backward peak for one turn's scored rows.

        `fused_logprob` rematerializes one block of at most ``chunk`` rows'
        logits at a time: the FP32 logits, their softmax, its product with
        the upstream gradient and the low-precision cast coexist while the
        coefficients form, and all but the product stay alive through the
        weight VJP. The weight gradient accumulates in FP32 across blocks
        (a later block's contribution beside the running sum) and is cast
        once on return, with the last block still alive.
        """

        vocab, width = self.weight.shape
        element = self.weight.element_size()
        accumulate = max(element, 4)
        block = min(scored, self.chunk) * vocab
        coefficient = block * (3 * accumulate + element)
        held = block * (2 * accumulate + element)
        moments = [coefficient]
        if self.weight.requires_grad:
            weight_gradient = vocab * width * accumulate
            moments.append(weight_gradient + vocab * width * element + held)
            if scored > self.chunk:
                moments.append(2 * weight_gradient + held)
                moments.append(weight_gradient + coefficient)
        return max(moments) + scored * width * (element + accumulate)

    def _head_forward_bytes(self, scored: int) -> int:
        """The output head's forward peak for one branch's scored rows.

        `fused_logprob` forms one block of at most ``chunk`` rows' logits at
        a time in the accumulation precision; its logsumexp takes a second
        block of the same shape (the shifted logits' exponent) beside it.
        Each block is one allocation, which the caching allocator rounds up
        to its large-allocation granularity. Beside them sit the per-row
        output, maximum, log-sum-exp and picked logit.
        """

        vocab, _width = self.weight.shape
        accumulate = max(self.weight.element_size(), 4)
        block = min(scored, self.chunk) * vocab * accumulate
        return 2 * _allocator_block_bytes(block) + 4 * scored * accumulate

    def _unmaterialized_gradient_bytes(self) -> int:
        """Gradient storage the next backward creates on this device.

        After the optimizer's ``zero_grad(set_to_none=True)`` the first
        backward of a batch materializes every trainable parameter's
        gradient. Under parameter-adjoint capture each gradient is moved
        into an owned FP32 accumulator as it lands: the largest parameter's
        gradient and its FP32 copy are transient, and a device-resident
        accumulator is created once per parameter (per source under the
        canonical reduction).
        """

        parameters = self._parameter_adjoint_parameters
        if not self._parameter_adjoint_capture_enabled:
            if self._memory_gradient_bytes is None:
                device = self.weight.device
                self._memory_gradient_bytes = [
                    (parameter, parameter.numel() * parameter.element_size())
                    for parameter in parameters
                    if parameter.device == device
                ]
            return sum(
                size
                for parameter, size in self._memory_gradient_bytes
                if parameter.grad is None
            )
        largest = max(
            (
                parameter.numel() * (parameter.element_size() + max(parameter.element_size(), 4))
                for parameter in parameters
            ),
            default=0,
        )
        if self._parameter_adjoint_storage_device is not None:
            return largest
        if self._canonical_parameter_adjoints is not None:
            return largest + sum(
                parameter.numel() * max(parameter.element_size(), 4)
                for parameter in parameters
            )
        return largest + sum(
            parameter.numel() * max(parameter.element_size(), 4)
            for parameter, held in zip(parameters, self._parameter_adjoints, strict=True)
            if held is None
        )

    def _turn_peak_estimate(
        self,
        executor: str,
        needs: list[_TurnNeed],
        *,
        replay: bool = False,
        memory: dict[str, int] | None = None,
    ) -> int:
        """Upper bound on a turn call's allocation peak above its start.

        The call covers the forward of ``needs`` (one model call on
        ``executor``) and, when any turn is scored or the call is a
        ``replay``, the backward the caller runs next. The terms, each from
        `_turn_memory_shape`:

        * retained: the turns' K/V and last hidden states, which the run
          keeps (an append pack on the plain executor clones them into
          storage of their own; a replay pack backwards them in place), and
          a replay's incoming adjoints cast to the activation dtype;
        * the attention recompute of the longest segment (ancestors plus
          its own tokens), one segment at a time, or, when larger, a plain
          pack's attention backward over every segment at once;
        * the larger of the two moments the backward can peak at: at its
          start every layer's saved activations are alive beside the output
          head's workspace (`_head_backward_bytes` of the largest scored
          turn, plus the head gradient the pack's earlier turns left
          pending); at its end the activations are gone and what the
          backward created remains: parameter gradients that did not exist
          (`_unmaterialized_gradient_bytes`), the new FP32 boundary ledgers
          and the embedding's dense gradient, beside the head gradient a
          scored call leaves pending on a head tied to the embedding
          (`_tied_head_gradient_pending`).

        A call with no scored turn and no replay has no backward: its peak
        is the forward's. The sum carries `_TURN_ESTIMATE_MARGIN`. The
        gradient term is read once per decision and kept in ``memory``.
        """

        shape = self._turn_memory_shape()
        tokens = sum(len(need.tokens) for need in needs)
        activations = tokens * shape[
            "plain_activation" if executor == "plain" else "checkpoint_activation"
        ]
        cloned = executor == "plain" and len(needs) > 1 and not replay
        retained = tokens * shape["kv"] * (2 if cloned else 1) + len(needs) * shape["hidden"]
        scored = [] if replay else [need.scored for need in needs if need.scored]
        if not replay and not scored:
            return int((retained + activations) * _TURN_ESTIMATE_MARGIN)
        head = 0
        if scored:
            head = max(self._head_backward_bytes(count) for count in scored)
            if len(scored) > 1:
                head += shape["head_gradient"]
        if replay:
            retained += tokens * shape["kv"]
        contexts = [need.context + len(need.tokens) for need in needs]
        context = max(contexts) * shape["context"]
        if executor == "plain" and len(needs) > 1:
            context = max(
                context,
                sum(contexts) * shape["packed_context"] + tokens * shape["packed_query"],
            )
        if memory is None:
            gradients = self._unmaterialized_gradient_bytes()
        else:
            if "unmaterialized_gradients" not in memory:
                memory["unmaterialized_gradients"] = self._unmaterialized_gradient_bytes()
            gradients = memory["unmaterialized_gradients"]
        created = (
            gradients
            + sum(need.new_ledger_tokens for need in needs) * shape["ledger"]
            + shape["embedding_gradient"]
        )
        if scored and self._tied_head_gradient_pending():
            created += shape["tied_head_gradient"]
        peak = retained + context + max(activations + head, created)
        return int(peak * _TURN_ESTIMATE_MARGIN)

    def _tied_head_gradient_pending(self) -> bool:
        """Whether a scored call's head gradient is held beside what its
        backward creates.

        A head tied to the input embedding reads the embedding's weight, so
        autograd sums the two contributions to that weight before it
        accumulates them, and the embedding's backward runs last: the head's
        weight gradient waits for it beside the embedding's dense gradient
        and the ledgers. Where the weight's gradient does not exist yet (no
        parameter-adjoint capture), the pending sum becomes that gradient
        and `_unmaterialized_gradient_bytes` already counts it.
        """

        if not self._turn_memory_shape()["tied_head_gradient"]:
            return False
        return self._parameter_adjoint_capture_enabled or self.weight.grad is not None

    def _plain_turn_fits(
        self,
        needs: list[_TurnNeed],
        *,
        replay: bool = False,
        memory: dict[str, int] | None = None,
    ) -> bool:
        """Whether a turn call's plain peak fits with the reserve spare.

        Selection is a memory decision and never a timing one: both
        executors compute the same values and gradients. The call takes the
        plain executor only when its estimated peak (`_turn_peak_estimate`)
        plus `rebuild_plain_reserve_bytes` fits the memory available now
        (`device_memory_state`: the driver's free memory and the cached
        memory the allocator can release). ``memory`` is a state already
        read for this decision.
        """

        reserve = self.rebuild_plain_reserve_bytes
        if reserve is None or not needs:
            return False
        device = self.weight.device
        if device.type != "cuda":
            return False
        if memory is None:
            memory = device_memory_state(device)
        available = memory["available"]
        if self._memory_min_available is None or available < self._memory_min_available:
            self._memory_min_available = available
        estimate = self._turn_peak_estimate(
            "plain", needs, replay=replay, memory=memory
        )
        return estimate + reserve <= available

    def _open_memory_window(
        self,
        phase: str,
        plain: bool,
        needs: list[_TurnNeed],
        memory: dict[str, int] | None,
    ) -> None:
        """Start measuring one turn model call against its estimate.

        The allocator's peak counter is reset, so the call's peak above its
        start is read when the window closes. Any open window closes first.
        """

        self.close_memory_window()
        if memory is None:
            return
        executor = self._executor_name(plain)
        if self._memory_retries_at_start is None:
            self._memory_retries_at_start = memory["alloc_retries"]
        self._memory_window = {
            "phase": phase,
            "executor": executor,
            "turns": len(needs),
            "tokens": sum(len(need.tokens) for need in needs),
            "scored": sum(need.scored for need in needs),
            "context": max(need.context for need in needs),
            "new_ledger_tokens": sum(need.new_ledger_tokens for need in needs),
            "estimate": self._turn_peak_estimate(
                executor,
                needs,
                replay=phase in ("replay", "replay_pack"),
                memory=memory,
            ),
            "unmaterialized_gradients": memory.get("unmaterialized_gradients", 0),
            "available": memory["available"],
            "allocated": memory["allocated"],
        }
        _reset_peak(self.weight.device)

    def close_memory_window(self) -> None:
        """Record the open turn window's measured peak; no-op without one.

        A turn's caller closes it after the turn's backward. No device
        synchronization: the allocator's counters are host bookkeeping.
        """

        window = self._memory_window
        if window is None:
            return
        self._memory_window = None
        stats = torch.cuda.memory_stats_as_nested_dict(self.weight.device)
        peak = int(stats.get("allocated_bytes", {}).get("all", {}).get("peak", 0))
        window["measured"] = max(0, peak - window["allocated"])
        self._memory_windows.append(window)

    @property
    def turn_memory_windows(self) -> list[dict[str, Any]]:
        """Every closed turn memory window of this run, in call order."""

        self.close_memory_window()
        return list(self._memory_windows)

    @property
    def turn_memory_report(self) -> dict[str, Any]:
        """The turn memory rule's decisions and its estimates against peaks.

        A public diagnostic: nothing in the library reads it.

        ``windows`` holds, per phase (append, pack, deferred, replay,
        replay_pack) and executor, the calls, the largest estimate and
        measured peak above the call's start, the largest
        measured-to-estimate ratio and the calls whose peak exceeded their
        estimate; a validated rule shows none. ``refused`` counts plain
        admissions the rule turned down (``pack_narrowed``: a pack sized
        below its ready set; ``replay_pack_narrowed``: a replay pack
        without every ready turn of its cohort).
        ``allocator_retries`` counts allocations that had to release cached
        memory first since the run's first window.
        """

        self.close_memory_window()
        windows: dict[str, dict[str, Any]] = {}
        for window in self._memory_windows:
            record = windows.setdefault(
                f"{window['phase']}/{window['executor']}",
                {
                    "calls": 0,
                    "max_estimate_bytes": 0,
                    "max_measured_bytes": 0,
                    "max_measured_over_estimate": 0.0,
                    "exceeded": 0,
                },
            )
            record["calls"] += 1
            record["max_estimate_bytes"] = max(
                record["max_estimate_bytes"], window["estimate"]
            )
            record["max_measured_bytes"] = max(
                record["max_measured_bytes"], window["measured"]
            )
            if window["estimate"]:
                record["max_measured_over_estimate"] = max(
                    record["max_measured_over_estimate"],
                    round(window["measured"] / window["estimate"], 4),
                )
            record["exceeded"] += int(window["measured"] > window["estimate"])
        retries = None
        if self._memory_retries_at_start is not None:
            stats = torch.cuda.memory_stats_as_nested_dict(self.weight.device)
            retries = int(stats.get("num_alloc_retries", 0)) - self._memory_retries_at_start
        return {
            "reserve_bytes": self.rebuild_plain_reserve_bytes,
            "activation_factor": TURN_ACTIVATION_FACTOR,
            "estimate_margin": _TURN_ESTIMATE_MARGIN,
            "refused": dict(sorted(self._memory_refusals.items())),
            "min_available_bytes": self._memory_min_available,
            "allocator_retries": retries,
            "windows": dict(sorted(windows.items())),
        }

    def _plain_rebuild(
        self, branches: list[dict[str, Any]], created: list[int]
    ) -> bool:
        """The plain whole-branch forward; False when it ran out of memory.

        One non-checkpointed forward per branch, so the backward pays no
        recompute pass. Each branch is its own forward, so no graph node is
        shared across branches and a per-branch backward cannot free a
        window-mate's saved tensors. Only the block path (a set
        ``rebuild_block_tokens``) can take a failed close over.
        """

        attempted: list[int] = []
        try:
            for b in branches:
                tokens = b["tokens"]
                cid = self._new_chunk(
                    len(tokens), parent=b["g"].prompt_chain[-1]
                )
                attempted.append(cid)
                b["hiddens"].append(
                    self._forward(
                        cid, b["g"].prompt_chain + [cid], tokens,
                        pos0=b["g"].prompt_len, plain=True,
                    )
                )
        except torch.OutOfMemoryError:
            if self.rebuild_block_tokens is None:
                created.extend(attempted)
                raise
            failed = True
        else:
            failed = False
        if failed:
            # Release the attempt before the block path allocates: its K/V
            # and chunk records (the tail of the chunk list) and its hiddens.
            for b in branches:
                b["hiddens"] = []
            self._kv.drop(attempted)
            for cid in sorted(attempted, reverse=True):
                if self._chunks and self._chunks[-1].idx == cid:
                    self._chunks.pop()
            self._memory_refusals["plain_rebuild_out_of_memory"] += 1
            return False
        created.extend(attempted)
        self._rebuild_path_counts["plain"] += len(branches)
        return True

    def _plain_rebuild_transient_bytes(self, tokens: int) -> int:
        """Activation bytes the plain whole-branch forward retains.

        The checkpointed executor retains layer-boundary hiddens plus K/V; the
        plain executor retains every layer's intermediates instead. Per token
        per layer that is the residual stream a few times over plus the MLP's
        gate and up projections. The raw count undercounts what the device
        retains, so it is scaled by `_PLAIN_REBUILD_ESTIMATE_MARGIN` to sit
        above the observation. Selecting the plain path on an underestimate
        reaches CUDA out of memory, so the estimate errs high by construction.
        """

        config = self.stack.config
        hidden = config.hidden_size
        intermediate = config.intermediate_size
        layers = config.num_hidden_layers
        element = self.weight.element_size()
        per_token_per_layer = (4 * hidden + 3 * intermediate) * element
        raw = tokens * layers * per_token_per_layer
        return int(raw * _PLAIN_REBUILD_ESTIMATE_MARGIN)

    def _plain_rebuild_fits(self, branches: list[dict[str, Any]]) -> bool:
        """Whether this rebuild's branches fit the declared plain headroom.

        The close-time branch rebuild's rule (`rebuild_logprobs`); turn
        appends, packs and turn replays read `_plain_turn_fits`, whose
        estimate is calibrated on measured peaks.

        Selection is a memory decision and never a timing one: the two
        implementations compute the same values and the same gradients, and
        the block path exists so the close-time transient stays bounded. A
        rebuild takes the plain path only when its estimated transient fits in
        the device memory available at that moment with
        `rebuild_plain_reserve_bytes` still spare. Available memory is read at
        each rebuild rather than assumed, because the streamed source service
        holds a varying number of model-sized sources in HBM while it drains
        them. It is the turn rule's reading (`device_memory_state`): the
        driver's free memory plus the cached memory the allocator can release.
        The driver's free memory alone falls toward zero in a long-lived
        process whose allocator keeps its peak pool reserved, which sent every
        live close to the recomputing block path. Counting that pool gives up
        the slack it silently provided, so the requirement also carries the
        gradient the close's backward writes beside the retained activations:
        one full set of trainable-parameter gradients, derived from the model.
        A plain forward that still runs out of memory falls back to the block
        path for that close (`rebuild_logprobs`).

        The plain path issues one model call per branch, so no graph node is
        shared across branches and a per-branch backward cannot free a
        window-mate's saved tensors; the hazard that forces the packed
        executor to checkpoint does not arise here. The branches' graphs do
        coexist until each is backwarded, so the check is against their summed
        transient.
        """

        reserve = self.rebuild_plain_reserve_bytes
        if reserve is None:
            return False
        tokens = sum(len(b["tokens"]) for b in branches)
        if not tokens:
            return False
        device = self.weight.device
        if device.type != "cuda":
            return False
        available = device_memory_state(device)["available"]
        needed = (
            self._plain_rebuild_transient_bytes(tokens)
            + self._trainable_gradient_bytes()
        )
        return needed + reserve <= available

    def _trainable_gradient_bytes(self) -> int:
        """Bytes of one full set of trainable-parameter gradients."""

        return sum(
            parameter.numel() * parameter.element_size()
            for parameter in self._parameter_adjoint_parameters
        )

    def _largest_trainable_gradient_bytes(self) -> int:
        """Bytes of the largest single trainable-parameter gradient."""

        return max(
            (
                parameter.numel() * parameter.element_size()
                for parameter in self._parameter_adjoint_parameters
            ),
            default=0,
        )

    def _block_rebuild_peak_bytes(
        self, branches: list[dict[str, Any]], *, host_blocks: int = 0
    ) -> tuple[int, int]:
        """Device bytes a block rebuild of ``branches`` adds, at its two peaks.

        Returns ``(rebuild, backward)``: what the pack's block rounds hold once
        every branch is re-forwarded, and the most the pack holds above the
        rebuild's start while its branches backward one at a time, in pack
        order (the reward-linear per-branch
        backwards). Counted from the model's shapes
        (`_turn_memory_shape`) and the branches' token counts, as an upper
        bound: every term is held at once, although the tied head's pending
        gradient and the K/V gradients peak apart:

        * kept per branch until its backward: each layer's checkpoint input
          (the block's hidden state), except in its first ``host_blocks``
          blocks, whose inputs wait on the host; the K/V of
          every block but the last (later blocks' attention saves them), the
          final hidden states twice (the per-block outputs and their
          concatenation for the head), what the final norm saves, each
          segment's rotary cos/sin and positions, and the head's scored rows;
        * a branch's backward: one full set of parameter gradients, the K/V
          gradients of its earlier blocks and of the prompt's proxies (all
          alive once its last block has backwarded), one block's layer
          recompute and that block's attention over the whole context, and
          the output head's gradient held until the tied input embedding's
          backward adds its own -- or, when larger, the output head's
          backward workspace, which comes first;
        * a later branch of a pack backwards beside the previous branch's
          gradient, whose copy off the device may still run;
        * the rebuild itself: one block round's layer transient beside what
          the branches keep, or, when larger, a branch's output-head forward
          (`_head_forward_bytes`, beside the predictor rows it selects from),
          which runs once every round has finished and every branch's kept
          bytes are alive.
        """

        shape = self._turn_memory_shape()
        layers = int(self.stack.config.num_hidden_layers)
        hidden = shape["hidden"]
        inputs = layers * hidden
        layer_recompute = shape["plain_activation"] // max(1, layers)
        gradient = self._trainable_gradient_bytes()
        kept: list[int] = []
        moments: list[int] = []
        round_tokens = 0
        head_forward = 0
        for b in branches:
            tokens = len(b["tokens"])
            if not tokens:
                continue
            scored = sum(1 for flag in b["scored"] if flag)
            last_start, last_end = b["block_ranges"][-1]
            earlier = tokens - (last_end - last_start)
            block = max(end - start for start, end in b["block_ranges"])
            prompt = b["g"].prompt_len
            on_host = sum(
                end - start for start, end in b["block_ranges"][:host_blocks]
            )
            round_tokens += block
            if scored:
                head_forward = max(
                    head_forward,
                    self._head_forward_bytes(scored) + tokens * (hidden + 8),
                )
            kept.append(
                (tokens - on_host) * inputs
                + tokens * (2 * hidden + shape["final_norm"] + shape["rotary"])
                + earlier * shape["kv"]
                + scored * hidden
            )
            late = (
                gradient
                + (prompt + earlier) * shape["kv"]
                + block * layer_recompute
                + (prompt + tokens) * shape["context"]
                + (shape["tied_head_gradient"] if scored else 0)
            )
            moments.append(max(self._head_backward_bytes(scored), late))
        if not kept:
            return 0, 0
        rebuild = sum(kept) + max(round_tokens * layer_recompute, head_forward)
        backward = max(
            sum(kept[index:]) + moment + (gradient if index else 0)
            for index, moment in enumerate(moments)
        )
        return rebuild, max(rebuild, backward)

    def _block_rebuild_host_blocks(
        self,
        branches: list[dict[str, Any]],
        *,
        releasing_bytes: Callable[[], int] | None = None,
    ) -> int:
        """How many leading blocks' checkpoint inputs a block rebuild keeps
        on the host (0: every input stays on the device).

        The block rebuild's rung decision, made before it rebuilds. Every
        placement computes the same logprobs and gradients; a block whose
        inputs wait on the host copies each layer's input off the device
        after its forward and back for its recompute, so the rule moves the
        fewest blocks that let the pack's backward fit, starting with the
        first, whose inputs are held longest. Beside the estimate
        (`_block_rebuild_peak_bytes`, an upper bound) the largest parameter's
        gradient stays spare: the biggest single allocation the backward
        makes, which the pool must place while the backward's own frees lie
        in pieces. It is read against the memory the allocator can hand out
        now (`device_memory_state`); the backward's peak may also count
        ``releasing_bytes()``, device memory the caller frees before the
        pack's first backward (earlier sources whose host copies it waits
        for), which the rebuild itself still runs beside. It is read after
        the device state, so memory freed in between is not counted twice.
        When nothing fits, every block's inputs go to the host. Off a CUDA
        device every rebuild keeps its inputs where they are.
        """

        device = self.weight.device
        if device.type != "cuda":
            return 0
        rounds = max(
            (len(b["block_ranges"]) for b in branches if b["tokens"]),
            default=0,
        )
        spare = self._largest_trainable_gradient_bytes()
        available = device_memory_state(device)["available"]
        releasing = (
            0 if releasing_bytes is None else max(0, int(releasing_bytes()))
        )
        for host_blocks in range(rounds + 1):
            rebuild, backward = self._block_rebuild_peak_bytes(
                branches, host_blocks=host_blocks
            )
            if (
                rebuild + spare <= available
                and backward + spare <= available + releasing
            ):
                return host_blocks
        return rounds

    def _branch_layout(
        self, t: _TrajState
    ) -> tuple[list[int], list[bool], list[tuple[int, int]]]:
        """Flatten one branch once and represent rebuild blocks as ranges.

        Every turn is already token IDs and score provenance, so the memory
        budget applies to token ranges, including turns longer than the
        budget. Block boundaries are a kernel-granularity decision only:
        attention is causal over the whole branch, positions remain
        continuous, and score masks are sliced at the same offsets, so the
        cuts do not change values or gradients.
        """
        tokens = [token for turn, _score in t.turns for token in turn]
        scored = [flag for _turn, score in t.turns for flag in score]
        if not tokens:
            return tokens, scored, []
        budget = self.rebuild_block_tokens
        if budget is None:
            return tokens, scored, [(0, len(tokens))]
        ranges = [
            (start, min(start + budget, len(tokens)))
            for start in range(0, len(tokens), budget)
        ]
        return tokens, scored, ranges

    def rebuild_logprobs(
        self,
        traj_ids: list[int],
        *,
        joint_backward: bool = False,
        closure_cohort_vjp: bool | None = None,
        executor: str = "auto",
        checkpoint_input_storage_device: Any = _RUN_DEFAULT,
        releasing_bytes: Callable[[], int] | None = None,
    ) -> dict[int, torch.Tensor]:
        """Scored logprobs for one or SEVERAL branches, re-forwarded fresh.

        The reward-linear close-time path (the per-trajectory backward granularity
        lever; single-branch
        callers reach it through `logprobs(tid, rebuild=True)`). The streamed
        per-event graph is correct but pays one backward walk per segment,
        including repeated small projection, attention, and head kernels.

        So the close does not walk the streamed graph (with
        stream_turns=False no streamed graph exists). It re-forwards each branch over the
        group's prompt chunk: the branch's turns are consecutive in trajectory
        order, so bottom-right causal attention over [prompt, branch]
        reproduces the streamed per-turn visibility EXACTLY. The predictor
        rows are [prompt_last_hidden, branch_hidden[:-1]] -- the flattening of
        the per-turn boundary-channel construction -- and the gated head runs
        once per branch over its scored positions (per-branch head calls keep
        the branch graphs disjoint, which the per-branch G1/G2 backwards
        require).

        ``executor="blocked"`` skips the plain executor whatever memory is
        free, and ``checkpoint_input_storage_device`` overrides, for this call,
        where the block executor keeps its per-layer checkpoint inputs (the
        run's `rebuild_checkpoint_storage_device` otherwise): the ladder a
        caller climbs when a close's backward runs out of device memory. Both
        change where values live, never the values.

        A caller that backwards the branches one at a time (not
        ``joint_backward``) may pass ``releasing_bytes``, a callable giving
        the device bytes it frees before the first backward (not before the
        rebuild): without a storage override, on a run whose inputs stay on
        the device, the block rebuild then asks its memory rule
        (`_block_rebuild_host_blocks`) how many leading blocks' inputs must
        wait on the host for the pack's backward to fit, instead of finding
        out in the backward. None keeps the run's placement.
        `last_rebuild_rung` names the rung the call ran on.

        IMPLEMENTATION, selected by the memory budget (never the timing):
        with `rebuild_block_tokens` set (the default), the branch re-forwards
        as consecutive BLOCKS of whole turns, each block one segment through
        the per-layer checkpointed executor -- its fresh K/V an explicit graph
        output feeding the next block's attention, so the close-time transient
        is block-sized instead of branch-sized. With
        `rebuild_block_tokens=None` the plain whole-branch
        forward runs instead: no recompute pass, full-branch transience.

        SEVERAL branches (coalesced closes, RewardLinearBackward.close_trajectories): block
        round j packs the j-th block of every branch still alive into ONE
        multi-segment call -- the _Seg packed-executor machinery, so the dense
        primal is batched across branches while each branch's graph stays its
        own (per-segment `_SegLayerCkpt` nodes). Branches of different groups
        attend their own group's proxies through their own chains. A branch
        appears once per round (block k+1 needs block k's K/V), which the
        round structure guarantees by construction.

        ``joint_backward=True`` also packs the checkpoint RECOMPUTE and VJP by
        recording one `_PackedLayerCkpt` per block-round layer.  This option is
        exact only when every returned branch loss is reduced into one
        autograd traversal.  OPD's gated closure-cohort path satisfies that
        contract.  Callers that backward trajectories independently, including
        the GRPO reward-linear backward, must
        retain the default per-segment graph topology.

        Gradient contract: for reward-affine objectives the gradient
        of `sum(w * logprob)` is a function of the parameters only -- it does
        not read the logprob VALUES -- so differentiating the rebuild equals
        differentiating the streamed graph up to kernel-schedule numerics
        (same tolerance class as checkpoint recompute; gated by the retained
        equality tests).

        Old-logprobs: in streamed mode the detached old twins are untouched
        (they remain the streamed values). With stream_turns=False THIS
        pass defines them -- the version guard
        forbids an optimizer step while the group is open, so the rebuild runs
        at theta0 and `old == new` holds by construction; the detached rebuild
        values are stored as the trajectory's old-logprobs at first rebuild.

        The prompt is attended through its boundary proxies when the cut is
        armed, so per-trajectory closes deposit boundary gradients instead of
        re-traversing the shared prompt subgraph. Everything the rebuild
        allocates is transient: block K/V and chunk records are dropped before
        returning (the graph keeps its own references until backward frees
        them), and nothing prefix-quadratic is retained.
        """
        if executor not in ("auto", "blocked"):
            raise ValueError("executor must be 'auto' or 'blocked'")
        storage_device = (
            self._rebuild_checkpoint_storage_device
            if checkpoint_input_storage_device is _RUN_DEFAULT
            else (
                None
                if checkpoint_input_storage_device is None
                else torch.device(checkpoint_input_storage_device)
            )
        )
        if closure_cohort_vjp is not None:
            if joint_backward and not closure_cohort_vjp:
                raise ValueError(
                    "joint_backward and closure_cohort_vjp disagree"
                )
            joint_backward = bool(closure_cohort_vjp)
        if joint_backward and self.rebuild_block_tokens is None:
            raise ValueError(
                "closure cohort VJP requires rebuild_block_tokens to be set"
            )
        branches: list[dict] = []
        out: dict[int, torch.Tensor] = {}
        for tid in traj_ids:
            if tid in out or any(b["tid"] == tid for b in branches):
                raise ValueError(f"trajectory {tid} appears twice in one rebuild")
            t = self._trajs[tid]
            g = self._groups[self._traj_group[tid]]
            tokens, scored, block_ranges = self._branch_layout(t)
            if not block_ranges:
                out[tid] = torch.zeros(0, device=self.weight.device)
                continue
            branches.append(
                {"tid": tid, "t": t, "g": g, "tokens": tokens,
                 "scored": scored, "block_ranges": block_ranges,
                 "chain": list(g.prompt_chain), "pos": g.prompt_len,
                 "hiddens": []}
            )

        use_joint_backward = joint_backward and len(branches) > 1
        cohort_ownership: dict[str, Any] | None = None
        if use_joint_backward:
            cohort_ownership = {
                "expected_trajectory_ids": set(),
                "seen_trajectory_ids": set(),
                "validated": False,
            }

        created: list[int] = []
        try:
            use_plain = executor == "auto" and (
                self.rebuild_block_tokens is None
                or self._plain_rebuild_fits(branches)
            )
            if use_plain and not self._plain_rebuild(branches, created):
                # The plain forward ran out of memory and released what it
                # held; the block path computes the same logprobs.
                use_plain = False
            host_blocks = 0
            if use_plain:
                self._last_rebuild_rung = "plain"
            else:
                if (
                    branches
                    and releasing_bytes is not None
                    and checkpoint_input_storage_device is _RUN_DEFAULT
                    and storage_device is None
                    and not use_joint_backward
                ):
                    # The rung is chosen before the rebuild: the inputs of
                    # the leading blocks the backward could not run beside
                    # wait on the host.
                    host_blocks = self._block_rebuild_host_blocks(
                        branches, releasing_bytes=releasing_bytes
                    )
                    predictions = self._rebuild_rung_predictions
                    if host_blocks:
                        predictions["host_checkpoints"] += len(branches)
                        predictions["host_checkpoint_blocks"] += sum(
                            min(host_blocks, len(b["block_ranges"]))
                            for b in branches
                        )
                    else:
                        predictions["device_checkpoints"] += len(branches)
                rounds = max(
                    (len(b["block_ranges"]) for b in branches), default=0
                )
                if storage_device is not None and storage_device.type == "cpu":
                    self._last_rebuild_rung = "blocked_host_checkpoints"
                elif host_blocks >= rounds and host_blocks:
                    self._last_rebuild_rung = "blocked_host_checkpoints"
                elif host_blocks:
                    self._last_rebuild_rung = "blocked_partial_host_checkpoints"
                else:
                    self._last_rebuild_rung = "blocked"
                self._rebuild_path_counts["blocked"] += len(branches)
                depth = 0
                while True:
                    alive = [
                        b for b in branches if depth < len(b["block_ranges"])
                    ]
                    if not alive:
                        break
                    segments: list[_Seg] = []
                    packed_tokens: list[int] = []
                    packed_pos: list[int] = []
                    start = 0
                    for b in alive:
                        block_start, block_end = b["block_ranges"][depth]
                        tokens = b["tokens"][block_start:block_end]
                        cid = self._new_chunk(len(tokens), parent=b["chain"][-1])
                        created.append(cid)
                        segments.append(
                            _Seg(cid=cid, chain=b["chain"] + [cid],
                                 start=start, length=len(tokens))
                        )
                        packed_tokens.extend(tokens)
                        packed_pos.extend(range(b["pos"], b["pos"] + len(tokens)))
                        start += len(tokens)
                    with operator_range(
                        "suffix_checkpoint_forward",
                        role=device_role(self.weight.device),
                        device=self.weight.device,
                        block_round=depth,
                        trajectories=",".join(str(b["tid"]) for b in alive),
                        segment_count=len(segments),
                        tokens=len(packed_tokens),
                        max_segment_tokens=max(seg.length for seg in segments),
                        joint_backward=joint_backward,
                    ):
                        hiddens = self._forward_packed(
                            segments,
                            packed_tokens,
                            packed_pos,
                            force_checkpoint=True,
                            joint_backward=use_joint_backward,
                            cohort_ownership=cohort_ownership,
                            checkpoint_input_storage_device=(
                                _HOST if depth < host_blocks else storage_device
                            ),
                            checkpoint_offload_stats=(
                                self._rebuild_checkpoint_offload_stats
                            ),
                        )
                    for b, seg, h in zip(alive, segments, hiddens, strict=True):
                        b["chain"] = seg.chain
                        b["pos"] += seg.length
                        b["hiddens"].append(h)
                    depth += 1
        finally:
            # the rebuild is transient: release its K/V and chunk records (the
            # created cids are the tail of self._chunks -- nothing else
            # allocates during a rebuild)
            self._kv.drop(created)
            for cid in sorted(created, reverse=True):
                if self._chunks and self._chunks[-1].idx == cid:
                    self._chunks.pop()

        for b in branches:
            t, g = b["t"], b["g"]
            hidden = (
                b["hiddens"][0] if len(b["hiddens"]) == 1
                else torch.cat(b["hiddens"], dim=0)
            )
            tokens = b["tokens"]
            scored = b["scored"]
            dev = hidden.device
            idx = torch.tensor([j for j, s in enumerate(scored) if s], device=dev)
            if not idx.numel():
                out[b["tid"]] = torch.zeros(0, device=self.weight.device)
                continue
            src = torch.cat(
                [g.prompt_last_hidden.unsqueeze(0), hidden[:-1]], dim=0
            )
            tgt = torch.tensor(tokens, device=dev)
            with operator_range(
                "suffix_logprob_head_forward",
                role=device_role(self.weight.device),
                device=self.weight.device,
                trajectory=b["tid"],
                suffix_tokens=len(tokens),
                scored_tokens=idx.numel(),
                chunk=self.chunk,
            ):
                lp = fused_logprob(
                    src.index_select(0, idx),
                    self.weight,
                    tgt.index_select(0, idx),
                    chunk=self.chunk,
                )
            if cohort_ownership is not None:
                trajectory_id = b["tid"]
                cohort_ownership["expected_trajectory_ids"].add(trajectory_id)

                def record_cohort_ownership(
                    gradient: torch.Tensor,
                    *,
                    tid: int = trajectory_id,
                    ownership: dict[str, Any] = cohort_ownership,
                ) -> torch.Tensor:
                    ownership["seen_trajectory_ids"].add(tid)
                    return gradient

                lp.register_hook(record_cohort_ownership)
            if not self.stream_turns and not t.old_logprobs:
                # stream_turns=False: the close pass at theta0 defines pi_old
                t.old_logprobs.append(lp.detach())
            out[b["tid"]] = lp
        return out

    def kv_bytes_resident(self) -> int:
        """The paradigm's named cost, measured not estimated."""
        return self._kv.bytes_resident()

    def retained_device_bytes(self) -> dict[str, int]:
        """Bytes this run keeps resident on the model device, by holder.

        Storage-deduplicated (views and aliases count once, in the first
        category that reaches them) over the run's own containers: the K/V
        store and the gradients accumulated on its leaves, each open
        trajectory's graph-attached state, the turn-replay proxies and their
        gradients (aliases of the store after the turn-boundary swap, so
        they add bytes only for chunks the store has dropped), the boundary
        ledgers of the group and shared-prompt cuts, the group prompt
        hiddens, the parameter adjoint accumulators, the parameters'
        optimizer-visible gradients, and the parameters themselves.

        ``allocated`` is the allocator's count on the device; ``unaccounted``
        is what the run holds by no reference of its own -- an in-flight
        backward's autograd graph, kernel workspaces, allocator slack -- and
        is the number to read when a trajectory's footprint exceeds what the
        categories explain. Cheap: a dictionary walk, no device
        synchronization; call it between events, not inside a backward.
        """

        device = self.weight.device
        seen: set[int] = set()
        totals: dict[str, int] = {}

        def count(category: str, value: Any) -> None:
            if not isinstance(value, torch.Tensor) or value.device != device:
                return
            storage = value.untyped_storage()
            key = storage.data_ptr()
            if key in seen or storage.nbytes() == 0:
                return
            seen.add(key)
            totals[category] = totals.get(category, 0) + storage.nbytes()

        kv_grad_own_bytes = 0
        for key, value in self._kv.pairs():
            count("kv", key)
            count("kv", value)
            count("kv_grad", key.grad)
            count("kv_grad", value.grad)
            for grad in (key.grad, value.grad):
                if isinstance(grad, torch.Tensor) and grad.device == device:
                    kv_grad_own_bytes += grad.numel() * grad.element_size()
        for traj in self._trajs.values():
            count("trajectory_state", traj.last_hidden)
            for tensor in traj.logprobs:
                count("trajectory_state", tensor)
            for tensor in traj.old_logprobs:
                count("trajectory_state", tensor)
            for boundary in traj.turn_replay_boundaries:
                for proxy in boundary.proxy:
                    count("turn_replay_proxy", proxy)
                    count("turn_replay_grad", proxy.grad)
                for gradient in boundary.adjoints._adjoints:
                    count("turn_replay_grad", gradient)
        cuts = list(self._boundaries.values())
        cuts.extend(node.boundary for node in self._shared_prompt_nodes)
        for boundary in cuts:
            for tensor in boundary.real:
                count("boundary_real", tensor)
            for tensor in boundary.proxy:
                count("boundary_proxy", tensor)
                count("boundary_proxy_grad", tensor.grad)
            for tensor in boundary._adjoints:
                count("boundary_ledger", tensor)
        for group in self._groups.values():
            count("group_prompt_hidden", group.prompt_last_hidden)
        for tensor in self._parameter_adjoints:
            count("parameter_adjoints", tensor)
        for parameter in self._parameter_adjoint_parameters:
            count("parameter_grads", parameter.grad)
        for parameter in self._parameter_adjoint_parameters:
            count("parameters", parameter)
        others = [run for run in _LIVE_RUNS if run is not self]
        for run in others:
            for key, value in run._kv.pairs():
                count("other_live_runs_kv", key)
                count("other_live_runs_kv", value)
                count("other_live_runs_kv", key.grad)
                count("other_live_runs_kv", value.grad)
            for tensor in run._parameter_adjoints:
                count("other_live_runs_parameter_adjoints", tensor)
        accounted = sum(totals.values())
        allocated = (
            int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0
        )
        return {
            **{name: totals.get(name, 0) for name in (
                "kv", "kv_grad", "trajectory_state", "turn_replay_proxy",
                "turn_replay_grad", "boundary_real", "boundary_proxy",
                "boundary_proxy_grad", "boundary_ledger", "group_prompt_hidden",
                "parameter_adjoints", "parameter_grads", "parameters",
                "other_live_runs_kv", "other_live_runs_parameter_adjoints",
            )},
            "live_runs": len(others) + 1,
            # The leaves' own gradient bytes, undeduplicated: kv_grad above
            # this is storage a gradient view pins beyond the leaf it serves.
            "kv_grad_own_bytes": kv_grad_own_bytes,
            "accounted": accounted,
            "allocated": allocated,
            "unaccounted": max(0, allocated - accounted) if allocated else 0,
            "open_trajectories": sum(
                1 for traj in self._trajs.values() if traj.turn_replay_boundaries or traj.logprobs
            ),
            "kv_chunks": sum(len(per) for per in self._kv._store.values()) // max(1, len(self._kv._store)),
        }

    def prompt_token_accounting(self) -> dict[str, int | float | str]:
        """Group-local reference work and executed prompt-forest work."""
        group_local = self._logical_prompt_tokens
        physical = self._physical_prompt_tokens
        encoded = json.dumps(
            self._prompt_topology_parts,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return {
            "group_prompt_tokens_without_cross_task_sharing": group_local,
            "physical_prompt_forward_tokens": physical,
            "cross_task_saved_prompt_forward_tokens": group_local - physical,
            "prompt_forward_reduction": group_local / max(physical, 1),
            "global_lcp_tokens": self._global_lcp_tokens,
            "shared_node_count": sum(
                len(part["groups"]) > 1
                for part in self._prompt_topology_parts
            ),
            "root_count": sum(
                part["offset"] == 0 for part in self._prompt_topology_parts
            ),
            "maximum_shared_depth": max(
                (
                    part["offset"] + len(part["tokens"])
                    for part in self._prompt_topology_parts
                    if len(part["groups"]) > 1
                ),
                default=0,
            ),
            "topology_sha256": hashlib.sha256(encoded).hexdigest(),
        }

    def rebuild_checkpoint_offload_diagnostics(self) -> dict[str, Any]:
        return {
            "storage_device": (
                None
                if self._rebuild_checkpoint_storage_device is None
                else str(self._rebuild_checkpoint_storage_device)
            ),
            **self._rebuild_checkpoint_offload_stats,
            "rung_predictions": dict(self._rebuild_rung_predictions),
        }

    def consume_boundary_adjoint_pairs(
        self,
        group_id: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        self._assert_adjoint_capture_valid(stage="group boundary consumption")
        return self.boundary(group_id).consume_adjoint_pairs(
            reject_native=self._parameter_adjoint_capture_enabled,
        )

    def drain_shared_prefix_source(
        self,
        group_id: int,
        *,
        source: Any,
    ) -> int:
        """Drain one explicit OPD or GRPO source through its forest path."""
        if self.shared_prefix_vjp_mode != "source_ready_serial":
            raise RuntimeError(
                "drain_shared_prefix_source requires source_ready_serial mode"
            )
        if source is None:
            raise ValueError("source-ready drain requires an explicit source")
        try:
            hash(source)
        except TypeError as error:
            raise TypeError(
                "source-ready drain requires a hashable source identifier"
            ) from error
        self._assert_adjoint_capture_valid(
            stage="source-ready group-boundary drain"
        )
        if group_id not in self._groups:
            raise ValueError(f"group {group_id} is not open")
        source_keys = self._source_ready_source_keys_by_group.setdefault(
            group_id, set()
        )
        if source in source_keys:
            raise RuntimeError(
                f"source-ready group {group_id} drained source {source!r} twice"
            )
        outputs, gradients = self.boundary(group_id).drain_adjoint_pairs(
            expected_source=source,
            reject_native=True,
        )
        if outputs:
            self.backward_with_boundary_adjoint_capture(
                outputs,
                grad_tensors=gradients,
                source=source,
                retain_graph=True,
            )
        processed = self._finalize_source_ready_shared_prompt_source(
            group_id,
            source=source,
        )
        self._source_ready_sources_by_group.setdefault(group_id, []).append(source)
        source_keys.add(source)
        return processed

    def backward_with_boundary_adjoint_capture(
        self,
        tensors: torch.Tensor | list[torch.Tensor],
        *,
        grad_tensors: list[torch.Tensor] | None = None,
        source: Any,
        retain_graph: bool = False,
    ) -> None:
        """Execute one VJP while the FP32 boundary ledgers own cut adjoints."""
        self._assert_adjoint_capture_valid(stage="captured backward")
        if source is None:
            raise ValueError("boundary adjoint source must be explicit")
        if self._active_boundary_adjoint_source is not None:
            raise RuntimeError(
                "nested boundary adjoint capture is unsupported: active "
                f"source {self._active_boundary_adjoint_source!r}, new "
                f"source {source!r}"
            )
        if self._parameter_adjoints_materialized:
            raise RuntimeError(
                "captured backward after parameter adjoint finalization"
            )
        if self._parameter_adjoint_capture_enabled:
            self._assert_empty_owned_parameter_grads(stage="captured backward")
            native_boundaries = [
                ("group", gid)
                for gid, boundary in self._boundaries.items()
                if boundary.has_native_adjoint()
            ]
            native_boundaries.extend(
                ("shared", node.cid)
                for node in self._shared_prompt_nodes
                if not node.finalized and node.boundary.has_native_adjoint()
            )
            native_boundaries.extend(
                ("turn", boundary.cid)
                for traj in self._trajs.values()
                for boundary in traj.turn_replay_boundaries
                if boundary.adjoints.has_native_adjoint()
            )
            if native_boundaries:
                raise RuntimeError(
                    "shared-forest boundary-adjoint ownership violation before "
                    f"captured backward: {native_boundaries[:8]}"
                )
        self._active_boundary_adjoint_source = source
        try:
            if self._parameter_adjoint_capture_enabled and self.explicit_adjoint_vjp:
                boundaries = [
                    boundary
                    for _, boundary in sorted(self._boundaries.items())
                    if boundary.proxy
                ]
                boundaries.extend(
                    node.boundary
                    for node in self._shared_prompt_nodes
                    if not node.finalized and node.boundary.proxy
                )
                boundaries.extend(
                    boundary.adjoints
                    for traj in self._trajs.values()
                    for boundary in traj.turn_replay_boundaries
                    if not boundary.adjoints._consumed
                )
                boundary_inputs = [
                    proxy for boundary in boundaries for proxy in boundary.proxy
                ]
                inputs = [*self._parameter_adjoint_parameters, *boundary_inputs]
                gradients = torch.autograd.grad(
                    tensors,
                    inputs,
                    grad_outputs=grad_tensors,
                    retain_graph=retain_graph,
                    allow_unused=True,
                )
                parameter_count = len(self._parameter_adjoint_parameters)
                for index, gradient in enumerate(gradients[:parameter_count]):
                    if gradient is not None:
                        self._add_parameter_adjoint(
                            index, gradient, source=source
                        )
                cursor = parameter_count
                for boundary in boundaries:
                    end = cursor + len(boundary.proxy)
                    boundary.accumulate_explicit_adjoints(
                        list(gradients[cursor:end]),
                        source=source,
                    )
                    cursor = end
            else:
                torch.autograd.backward(
                    tensors,
                    grad_tensors=grad_tensors,
                    retain_graph=retain_graph,
                )
        except BaseException:
            self._mark_adjoint_capture_failed(source)
            raise
        finally:
            self._active_boundary_adjoint_source = None

    def _enable_parameter_adjoint_capture(self) -> None:
        self._assert_adjoint_capture_valid(stage="shared-forest initialization")
        if self._parameter_adjoint_capture_enabled:
            return
        if self._parameter_adjoints_materialized:
            raise RuntimeError("parameter adjoints were already materialized")
        self._assert_empty_owned_parameter_grads(stage="shared-forest initialization")
        run_ref = weakref.ref(self)

        def make_hook(index: int) -> Callable[[torch.Tensor], None]:
            def capture(parameter: torch.Tensor) -> None:
                run = run_ref()
                if run is not None:
                    run._capture_parameter_adjoint_value(index, parameter)

            return capture

        self._parameter_adjoint_hook_handles.extend(
            parameter.register_post_accumulate_grad_hook(make_hook(index))
            for index, parameter in enumerate(self._parameter_adjoint_parameters)
        )
        self._parameter_adjoint_hook_finalizer = weakref.finalize(
            self,
            self._remove_parameter_adjoint_hook_handles,
            self._parameter_adjoint_hook_handles,
        )
        self._parameter_adjoint_capture_enabled = True

    def _assert_empty_owned_parameter_grads(self, *, stage: str) -> None:
        occupied = [
            index
            for index, parameter in enumerate(self._parameter_adjoint_parameters)
            if parameter.grad is not None
        ]
        if occupied:
            raise RuntimeError(
                "shared-forest parameter-gradient ownership violation during "
                f"{stage}: trainable parameter indices {occupied[:8]} already "
                "have optimizer-visible gradients"
            )

    def _assert_adjoint_capture_valid(self, *, stage: str) -> None:
        if self._adjoint_capture_failed:
            raise RuntimeError(
                "StreamingRun adjoint state is invalid after a failed backward "
                f"at source {self._adjoint_capture_failure_source!r}; discard "
                f"the run before {stage}"
            )

    def _mark_adjoint_capture_failed(self, source: Any) -> None:
        self._adjoint_capture_failed = True
        if self._adjoint_capture_failure_source is None:
            self._adjoint_capture_failure_source = source

    def _capture_parameter_adjoint_value(
        self,
        index: int,
        parameter: torch.Tensor,
    ) -> None:
        if (
            self._active_boundary_adjoint_source is None
            or not self._parameter_adjoint_capture_enabled
        ):
            return
        contribution = parameter.grad
        if contribution is None:
            raise RuntimeError("parameter post-accumulate hook received no gradient")
        self._add_parameter_adjoint(
            index,
            contribution,
            source=self._active_boundary_adjoint_source,
        )
        parameter.grad = None

    @staticmethod
    def _remove_parameter_adjoint_hook_handles(handles: list[Any]) -> None:
        for handle in tuple(handles):
            handle.remove()
        handles.clear()

    def _release_parameter_adjoint_hooks(self) -> None:
        finalizer = self._parameter_adjoint_hook_finalizer
        if finalizer is not None and finalizer.alive:
            finalizer.detach()
        self._parameter_adjoint_hook_finalizer = None
        self._remove_parameter_adjoint_hook_handles(
            self._parameter_adjoint_hook_handles
        )

    def _add_parameter_adjoint(
        self,
        index: int,
        contribution: torch.Tensor,
        *,
        source: Any,
    ) -> None:
        if contribution.is_sparse:
            raise NotImplementedError(
                "streaming FP32 parameter accumulation requires dense gradients"
            )
        if self._canonical_parameter_adjoints is not None:
            self._canonical_parameter_adjoints.add_parameter(
                source,
                index,
                contribution,
            )
            self._parameter_adjoint_contribution_counts[index] += 1
            return
        dtype = torch.float64 if contribution.dtype == torch.float64 else torch.float32
        started = time.perf_counter()
        if self._parameter_adjoint_storage_device is None:
            value = contribution.detach().to(dtype=dtype)
        else:
            value = contribution.detach().to(
                device=self._parameter_adjoint_storage_device,
                dtype=dtype,
                copy=True,
            )
        accumulator = self._parameter_adjoints[index]
        if accumulator is None:
            self._parameter_adjoints[index] = (
                value.clone()
                if self._parameter_adjoint_storage_device is None
                else value
            )
            self._parameter_adjoint_current_bytes += (
                value.numel() * value.element_size()
            )
        else:
            accumulator.add_(value)
        self._parameter_adjoint_offload_s += time.perf_counter() - started
        self._parameter_adjoint_contribution_counts[index] += 1
        self._parameter_adjoint_peak_bytes = max(
            self._parameter_adjoint_peak_bytes,
            self._parameter_adjoint_current_bytes,
        )

    def accumulate_parameter_adjoints(
        self,
        parameters: list[torch.nn.Parameter],
        gradients: list[torch.Tensor | None],
        *,
        source: Any,
    ) -> None:
        if source is None:
            raise ValueError("parameter adjoint source must be explicit")
        self._assert_adjoint_capture_valid(
            stage="manual parameter-adjoint accumulation"
        )
        capture_enabled = self._parameter_adjoint_capture_enabled
        try:
            if len(parameters) != len(gradients):
                raise ValueError("parameters and gradients must align")
            validated: list[tuple[torch.nn.Parameter, torch.Tensor, int]] = []
            for parameter, gradient in zip(parameters, gradients, strict=True):
                if gradient is None:
                    continue
                index = self._parameter_adjoint_index.get(id(parameter))
                if index is None:
                    raise ValueError(
                        "parameter does not belong to this StreamingRun"
                    )
                if (
                    gradient.shape != parameter.shape
                    or gradient.device != parameter.device
                ):
                    raise ValueError(
                        "manual parameter adjoint differs from parameter"
                    )
                validated.append((parameter, gradient, index))
        except BaseException:
            if capture_enabled:
                self._mark_adjoint_capture_failed(source)
            raise
        self._manual_parameter_adjoint_call_count += 1
        self._manual_parameter_adjoint_peak_batch_bytes = max(
            self._manual_parameter_adjoint_peak_batch_bytes,
            sum(
                gradient.numel() * gradient.element_size()
                for _, gradient, _ in validated
            ),
        )
        if not self._parameter_adjoint_capture_enabled:
            for parameter, gradient, _ in validated:
                value = gradient.to(dtype=parameter.dtype)
                if parameter.grad is None:
                    parameter.grad = value
                else:
                    parameter.grad.add_(value)
            return
        self._assert_empty_owned_parameter_grads(
            stage="manual parameter-adjoint accumulation"
        )
        for _, gradient, index in validated:
            self._add_parameter_adjoint(index, gradient, source=source)

    def accumulate_parameter_adjoint(
        self,
        parameter: torch.nn.Parameter,
        gradient: torch.Tensor,
        *,
        source: Any,
    ) -> None:
        """Consume one ready parameter cotangent without retaining its peers."""

        self.accumulate_parameter_adjoints(
            [parameter],
            [gradient],
            source=source,
        )

    def parameter_adjoint_diagnostics(self) -> dict[str, Any]:
        if self._canonical_parameter_adjoints is not None:
            canonical = self._canonical_parameter_adjoints.diagnostics()
            return {
                "materialized": self._parameter_adjoints_materialized,
                "explicit_adjoint_vjp": self.explicit_adjoint_vjp,
                "capture_failed": self._adjoint_capture_failed,
                "failure_source": self._adjoint_capture_failure_source,
                "reduction": "canonical_source",
                "accumulator_dtype": "torch.float32_or_float64",
                "accumulator_tensor_count": canonical["tensor_count"],
                "accumulator_bytes": canonical["bytes"],
                "contribution_count": canonical["contribution_count"],
                "source_count": canonical["source_count"],
                "peak_source_count": canonical["peak_source_count"],
                "storage_device": canonical["storage_device"],
                "peak_accumulator_bytes": canonical["peak_bytes"],
                "byte_accounting_scope": canonical["byte_accounting_scope"],
                "transient_cast_clone_bytes_included": canonical[
                    "transient_cast_clone_bytes_included"
                ],
                "canonical_fold_s": canonical["canonical_fold_s"],
                "canonical_materialization_s": canonical[
                    "materialization_s"
                ],
                "canonical_fold_materialization_s": canonical[
                    "canonical_fold_materialization_s"
                ],
                "materialize_s": self._parameter_adjoint_materialize_s,
                "manual_accumulation_calls": (
                    self._manual_parameter_adjoint_call_count
                ),
                "manual_peak_batch_bytes": (
                    self._manual_parameter_adjoint_peak_batch_bytes
                ),
            }
        active = [tensor for tensor in self._parameter_adjoints if tensor is not None]
        return {
            "materialized": self._parameter_adjoints_materialized,
            "explicit_adjoint_vjp": self.explicit_adjoint_vjp,
            "capture_failed": self._adjoint_capture_failed,
            "failure_source": self._adjoint_capture_failure_source,
            "reduction": "arrival",
            "accumulator_dtype": None if not active else str(active[0].dtype),
            "accumulator_tensor_count": len(active),
            "accumulator_bytes": sum(
                tensor.numel() * tensor.element_size() for tensor in active
            ),
            "contribution_count": sum(self._parameter_adjoint_contribution_counts),
            "storage_device": (
                None
                if self._parameter_adjoint_storage_device is None
                else str(self._parameter_adjoint_storage_device)
            ),
            "peak_accumulator_bytes": self._parameter_adjoint_peak_bytes,
            "offload_s": self._parameter_adjoint_offload_s,
            "materialize_s": self._parameter_adjoint_materialize_s,
            "manual_accumulation_calls": self._manual_parameter_adjoint_call_count,
            "manual_peak_batch_bytes": (
                self._manual_parameter_adjoint_peak_batch_bytes
            ),
        }

    def _materialize_parameter_adjoints(self) -> None:
        self._assert_adjoint_capture_valid(stage="parameter-adjoint materialization")
        if self._parameter_adjoints_materialized:
            return
        if self._parameter_adjoint_capture_enabled:
            self._assert_empty_owned_parameter_grads(
                stage="logical-batch finalization"
            )
        materialize_started = time.perf_counter()
        if self._canonical_parameter_adjoints is None:
            ledgers = self._parameter_adjoints
        else:
            ledgers = self._canonical_parameter_adjoints.finalize(
                output_devices=[
                    parameter.device
                    for parameter in self._parameter_adjoint_parameters
                ],
                output_dtypes=[
                    parameter.dtype
                    for parameter in self._parameter_adjoint_parameters
                ],
            )
        for parameter, ledger in zip(
            self._parameter_adjoint_parameters, ledgers, strict=True
        ):
            if ledger is not None:
                parameter.grad = ledger.to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
        self._parameter_adjoint_materialize_s += (
            time.perf_counter() - materialize_started
        )
        self._release_parameter_adjoint_hooks()
        self._parameter_adjoints.clear()
        self._parameter_adjoint_current_bytes = 0
        self._parameter_adjoints_materialized = True

    def boundary(self, group_id: int) -> _Boundary:
        if group_id not in self._boundaries:
            raise ValueError(
                f"group {group_id} has no boundary; construct StreamingRun with "
                "boundary_cut=True for the reward-linear backward"
            )
        return self._boundaries[group_id]

    def finalize_group_boundary(
        self,
        group_id: int,
        gradients: list[torch.Tensor | None] | None = None,
        *,
        source: Any | None = None,
        retain_graph: bool = False,
    ) -> None:
        """Propagate one group's adjoints into its retained prompt forest.

        The reference backend applies autograd to every active boundary output.
        With the feature-gated Triton backend, direct adjoints whose output is
        a shared-node leaf are queued for fused reduction. Group-local suffix
        outputs still run through autograd immediately, and their indirect
        adjoints accumulate on the same shared leaves. Shared nodes are later
        finalized in reverse topological order.
        """
        boundary = self.boundary(group_id)
        if gradients is None:
            gradients = [proxy.grad for proxy in boundary.proxy]
        if len(gradients) != len(boundary.real):
            raise ValueError(
                f"group {group_id}: received {len(gradients)} boundary adjoints "
                f"for {len(boundary.real)} outputs"
            )
        targets = boundary.shared_targets or [None] * len(boundary.real)
        if len(targets) != len(boundary.real):
            raise RuntimeError(
                f"group {group_id}: shared-boundary metadata is misaligned"
            )

        outputs: list[torch.Tensor] = []
        active_gradients: list[torch.Tensor] = []
        for real, gradient, target in zip(boundary.real, gradients, targets, strict=True):
            if gradient is None:
                continue
            converted = gradient.detach().to(dtype=real.dtype, device=real.device)
            if self.boundary_adjoint_backend != "torch" and target is not None:
                self._queue_shared_adjoint(target, converted)
            else:
                outputs.append(real)
                active_gradients.append(converted)
        if outputs:
            with operator_range(
                "group_boundary_vjp",
                role=device_role(outputs[0].device),
                device=outputs[0].device,
                group=group_id,
                output_tensors=len(outputs),
                output_elements=sum(output.numel() for output in outputs),
            ):
                if source is None:
                    torch.autograd.backward(
                        outputs,
                        grad_tensors=active_gradients,
                        retain_graph=retain_graph,
                    )
                else:
                    self.backward_with_boundary_adjoint_capture(
                        outputs,
                        grad_tensors=active_gradients,
                        source=source,
                        retain_graph=retain_graph,
                    )

    def _queue_shared_adjoint(
        self,
        target: torch.Tensor,
        gradient: torch.Tensor,
    ) -> None:
        pending = self._pending_shared_adjoints.setdefault(id(target), [])
        pending.append(gradient)
        if len(pending) >= self.boundary_adjoint_max_degree:
            self._flush_shared_adjoints(target)

    def _flush_shared_adjoints(self, target: torch.Tensor) -> None:
        pending = self._pending_shared_adjoints.pop(id(target), None)
        if not pending:
            return
        reduced = reduce_boundary_adjoints(
            pending,
            existing=target.grad,
            backend=(
                "triton"
                if self.boundary_adjoint_backend == "triton"
                else "torch"
            ),
        )
        if reduced is None:
            raise AssertionError("a non-empty adjoint reduction returned None")
        target.grad = reduced

    def group_of(self, traj_id: int) -> int:
        return self._traj_group[traj_id]

    def open_group_ids(self) -> list[int]:
        """Groups whose retained state has not been freed by `free_group`."""
        return sorted(self._groups)

    def trajectories_of(self, group_id: int) -> list[int]:
        return [t for t, g in self._traj_group.items() if g == group_id]

    def shared_prefix_vjp_diagnostics(self) -> dict[str, Any]:
        """Return scalar dependency state and completed VJP events.

        Graph tensors are excluded, so the diagnostic remains valid after a
        node has released its retained prompt state. Event order is sufficient
        to verify child-before-parent execution and whether a VJP moved before
        the logical-batch barrier.
        """
        nodes = [
            {
                "cid": node.cid,
                "parent_cid": node.parent_cid,
                "depth_start": node.depth_start,
                "depth_end": node.depth_end,
                "consumer_group_ids": list(node.consumer_group_ids),
                "pending_group_ids": sorted(node.pending_group_ids),
                "child_cids": sorted(node.child_cids),
                "ancestor_shared_cids": list(node.ancestor_shared_cids),
                "pending_child_cids": sorted(node.pending_child_cids),
                "processed_group_ids": sorted(node.processed_group_ids),
                "source_process_count": node.source_process_count,
                "expected_source_process_count": sum(
                    len(self._source_ready_sources_by_group.get(gid, ()))
                    for gid in node.consumer_group_ids
                ),
                "source_vjp_count": node.source_vjp_count,
                "finalized": node.finalized,
                "finalization_index": node.finalization_index,
                "canonical_vjp_index": node.canonical_vjp_index,
            }
            for node in self._shared_prompt_nodes
        ]
        return {
            "shared_prefix_vjp_mode": self.shared_prefix_vjp_mode,
            "incremental_enabled": self.dependency_ready_shared_prefix_vjp,
            "dependency_ready_enabled": (
                self.dependency_ready_shared_prefix_vjp
            ),
            "backend": self.shared_prefix_vjp_backend,
            "total_nodes": len(nodes),
            "finalized_nodes": sum(node["finalized"] for node in nodes),
            "pending_nodes": sum(not node["finalized"] for node in nodes),
            "group_consumer_edges_total": sum(
                len(node["consumer_group_ids"]) for node in nodes
            ),
            "group_consumer_edges_pending": sum(
                len(node["pending_group_ids"]) for node in nodes
            ),
            "child_edges_total": sum(len(node["child_cids"]) for node in nodes),
            "child_edges_pending": sum(
                len(node["pending_child_cids"]) for node in nodes
            ),
            "closed_group_ids": sorted(self._closed_shared_prompt_groups),
            "open_group_ids": self.open_group_ids(),
            "autograd_call_count": self._shared_prompt_vjp_calls,
            "source_process_count": sum(
                node["source_process_count"] for node in nodes
            ),
            "group_process_count": sum(
                len(node["processed_group_ids"]) for node in nodes
            ),
            "source_process_accounting_applicable": (
                self.shared_prefix_vjp_mode == "source_ready_serial"
            ),
            "expected_source_process_count": (
                sum(node["expected_source_process_count"] for node in nodes)
                if self.shared_prefix_vjp_mode == "source_ready_serial"
                else None
            ),
            "source_vjp_count": sum(node["source_vjp_count"] for node in nodes),
            "source_ready_sources_by_group": {
                gid: list(sources)
                for gid, sources in sorted(self._source_ready_sources_by_group.items())
            },
            "frontier_widths": list(self._shared_prompt_vjp_frontier_widths),
            "isolated_parameter_gradient_bytes_current": (
                self._isolated_parameter_gradient_bytes_current
            ),
            "isolated_parameter_gradient_bytes_peak": (
                self._isolated_parameter_gradient_bytes_peak
            ),
            "isolated_parameter_gradient_nodes_staged": len(
                self._isolated_shared_parameter_gradients
            ),
            "isolated_ancestor_adjoint_targets_staged": len(
                self._isolated_ancestor_proxy_adjoints
            ),
            "isolated_ancestor_adjoint_sources_staged": sum(
                len(by_source)
                for by_source in self._isolated_ancestor_proxy_adjoints.values()
            ),
            "isolated_ancestor_adjoint_bytes_current": (
                self._isolated_ancestor_adjoint_bytes_current
            ),
            "isolated_ancestor_adjoint_bytes_peak": (
                self._isolated_ancestor_adjoint_bytes_peak
            ),
            "isolated_parameter_gradients_accumulated": (
                self._isolated_parameter_gradients_accumulated
            ),
            "nodes": nodes,
            "events": self._shared_prompt_vjp_event_diagnostics(),
            "source_vjp_events": [
                dict(event) for event in self._shared_prompt_source_vjp_events
            ],
        }

    def shared_prefix_finalization_diagnostics(self) -> dict[str, Any]:
        """Compatibility diagnostic containing the complete VJP state."""
        return self.shared_prefix_vjp_diagnostics()

    def _shared_prompt_vjp_event_diagnostics(self) -> list[dict[str, Any]]:
        """Resolve CUDA-event durations after the caller's synchronization."""
        events = []
        for event in self._shared_prompt_finalization_events:
            record = dict(event)
            interval_index = record.pop("vjp_cuda_interval_index", None)
            if interval_index is None:
                record["vjp_device_duration_s"] = None
            else:
                begun, finished = self._shared_prompt_vjp_cuda_intervals[
                    interval_index
                ]
                record["vjp_device_duration_s"] = (
                    begun.elapsed_time(finished) / 1000.0
                    if finished.query()
                    else None
                )
            events.append(record)
        return events

    # ---------------------------------------------------------------- freeing

    def finalize_trajectories_turn_boundaries(self, traj_ids: list[int]) -> None:
        """Replay a closure cohort's descendant state adjoints, packed.

        Each turn's direct OPD loss may backward when the turn arrives. Future
        turns attend detached state proxies, whose adjoints remain until the
        trajectory closes. Each trajectory here rebuilds its turns one at a
        time, last to first, under the unchanged parameters and applies only
        those descendant state adjoints: the resulting parameter and
        parent-state gradients complete the exact chain rule without
        retaining per-turn activation graphs.

        ``traj_ids`` are trajectories whose content is complete, handed over
        together because they closed together. Across the cohort, the turns
        ready at the same point (the latest
        unreplayed turn of each member) run as one plain joint pack when the
        turn memory rule admits them (`_turn_replay_pack`): one forward over the pack's turns and one
        backward with every member's incoming state adjoints, whose parameter
        and ancestor adjoints are the sums of the per-turn replays. What each
        turn's replay computes is unchanged; only where the per-turn
        parameter gradients are summed changes, which arrival-order
        accumulation already leaves order-dependent at its rounding.
        """
        if not self.turn_boundary_rebuild:
            raise RuntimeError("turn-boundary replay is disabled for this run")
        if len(set(traj_ids)) != len(traj_ids):
            raise ValueError("a trajectory appears twice in one replay cohort")
        cohort = [(traj_id, self._trajs[traj_id]) for traj_id in traj_ids]
        for traj_id, traj in cohort:
            if traj.logprobs or traj.old_logprobs:
                raise RuntimeError(
                    f"trajectory {traj_id}: consume every turn loss before replay"
                )
        for _traj_id, traj in cohort:
            # Deferred turns trail the trajectory (every other arrival
            # forwards them first) and have no scored token: nothing reads
            # their state.
            self._turn_path_counts["unscored"]["dropped"] += len(traj.deferred_turns)
            traj.deferred_turns.clear()
        while True:
            ready: list[tuple[int, _TrajState, _TurnReplayBoundary]] = []
            for traj_id, traj in cohort:
                boundary = self._next_turn_replay(traj)
                if boundary is not None:
                    ready.append((traj_id, traj, boundary))
            if not ready:
                break
            needs = [
                _TurnNeed(
                    tokens=boundary.tokens,
                    scored=0,
                    context=boundary.pos0,
                    new_ledger_tokens=0,
                )
                for _traj_id, _traj, boundary in ready
            ]
            memory = (
                device_memory_state(self.weight.device)
                if self.weight.device.type == "cuda"
                else None
            )
            chosen = self._turn_replay_pack(needs, memory)
            if len(chosen) > 1:
                plain = True
            else:
                # This graph is consumed immediately, so retain this turn's
                # dense activations when the turn memory rule admits it.
                plain = self._plain_turn_fits(
                    [needs[chosen[0]]], replay=True, memory=memory
                )
                if (
                    not plain
                    and memory is not None
                    and self.rebuild_plain_reserve_bytes is not None
                ):
                    self._memory_refusals["replay"] += 1
            self._replay_turns(
                [ready[index] for index in chosen],
                [needs[index] for index in chosen],
                plain=plain,
                memory=memory,
            )
        for traj_id, traj in cohort:
            group = self._groups[self._traj_group[traj_id]]
            traj.chain = list(group.prompt_chain)
            traj.last_hidden = group.prompt_last_hidden

    def _next_turn_replay(self, traj: _TrajState) -> _TurnReplayBoundary | None:
        """The trajectory's latest turn that holds a descendant adjoint.

        Later turns with none are released on the way: no descendant read
        their state (or none is left to), so there is nothing to replay.
        A turn's adjoint is complete once every later turn of its trajectory
        has replayed, which is when it becomes the latest.
        """
        while traj.turn_replay_boundaries:
            boundary = traj.turn_replay_boundaries[-1]
            if boundary.adjoints.has_buffered_adjoint():
                return boundary
            self._kv.drop([boundary.cid])
            boundary.adjoints.release()
            traj.turn_replay_boundaries.pop()
        return None

    def _replay_turns(
        self,
        ready: list[tuple[int, _TrajState, _TurnReplayBoundary]],
        needs: list[_TurnNeed],
        *,
        plain: bool,
        memory: dict[str, int] | None,
    ) -> None:
        """Rebuild the turns under the unchanged parameters; one backward.

        ``ready`` holds at most one turn per trajectory, each its
        trajectory's latest unreplayed turn. One turn runs on the executor
        ``plain`` selects; several run as one plain joint pack whose graph
        the single backward below consumes. The rebuilt K/V and last hidden
        state of each turn receive the adjoints its descendants deposited on
        its proxies; the backward carries them to the parameters and to the
        proxies of each turn's ancestors.
        """
        self._open_memory_window(
            "replay_pack" if len(ready) > 1 else "replay", plain, needs, memory
        )
        for _traj_id, _traj, boundary in ready:
            self._kv.drop([boundary.cid])
        if len(ready) == 1:
            boundary = ready[0][2]
            hiddens = [
                self._forward(
                    boundary.cid,
                    boundary.parent_chain + [boundary.cid],
                    boundary.tokens,
                    pos0=boundary.pos0,
                    plain=plain,
                )
            ]
        else:
            if not plain:
                raise ValueError("a replay pack runs on the plain executor")
            segments: list[_Seg] = []
            packed_tokens: list[int] = []
            packed_pos: list[int] = []
            for _traj_id, _traj, boundary in ready:
                segments.append(
                    _Seg(
                        cid=boundary.cid,
                        chain=boundary.parent_chain + [boundary.cid],
                        start=len(packed_tokens),
                        length=len(boundary.tokens),
                    )
                )
                packed_tokens.extend(boundary.tokens)
                packed_pos.extend(
                    range(boundary.pos0, boundary.pos0 + len(boundary.tokens))
                )
            hiddens = self._forward_packed(
                segments,
                packed_tokens,
                packed_pos,
                plain=True,
                joint_backward=True,
            )
        executor = self._executor_name(plain)
        self._turn_path_counts["replay"][executor] += len(ready)
        self._rebuild_path_counts[
            "plain" if executor == "plain" else "blocked"
        ] += len(ready)
        outputs: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []
        for (_traj_id, _traj, boundary), hidden in zip(ready, hiddens, strict=True):
            real: list[torch.Tensor] = []
            for layer in self._kv.layers():
                k, v = self._kv.get(layer, boundary.cid)
                real.extend((k, v))
            real.append(hidden[-1])
            boundary.adjoints.real = real
            turn_outputs, turn_gradients = boundary.adjoints.consume_adjoint_pairs()
            outputs.extend(turn_outputs)
            gradients.extend(turn_gradients)
        if len(ready) == 1:
            source: Any = ("trajectory_turn_replay", ready[0][0], ready[0][2].cid)
        else:
            source = (
                "trajectory_turn_replay_pack",
                tuple((traj_id, boundary.cid) for traj_id, _traj, boundary in ready),
            )
        self.backward_with_boundary_adjoint_capture(
            outputs,
            grad_tensors=gradients,
            source=source,
        )
        for _traj_id, traj, boundary in ready:
            self._kv.drop([boundary.cid])
            boundary.adjoints.release()
            traj.turn_replay_boundaries.pop()
        self.close_memory_window()

    def free_trajectory(self, traj_id: int) -> None:
        """Release a trajectory's retained state: its branch K/V and its graph.

        The detached old-logprobs survive (they are values, not graph). The
        prompt chunk is shared and is NOT freed here -- `free_group` owns it.
        """
        traj = self._trajs[traj_id]
        if traj.turn_replay_boundaries:
            raise RuntimeError(
                f"trajectory {traj_id}: finalize turn adjoints before freeing state"
            )
        prompt_chain = self._groups[self._traj_group[traj_id]].prompt_chain
        prompt_chunks = set(prompt_chain)
        self._kv.drop([c for c in traj.chain if c not in prompt_chunks])
        traj.logprobs.clear()
        traj.deferred_turns.clear()
        traj.last_hidden = torch.zeros(0)  # drop the graph reference
        traj.chain = list(prompt_chain)

    def free_group(self, group_id: int) -> None:
        """Release one group and advance shared-node dependency counts."""
        for tid in self.trajectories_of(group_id):
            traj = self._trajs[tid]
            if traj.logprobs:
                raise RuntimeError(
                    f"free_group({group_id}): trajectory {tid} still holds its "
                    "graph; close it first"
                )
            if traj.turn_replay_boundaries:
                raise RuntimeError(
                    f"free_group({group_id}): trajectory {tid} has unreplayed turn adjoints"
                )
        if self.shared_prefix_vjp_mode == "source_ready_serial":
            boundary = self.boundary(group_id)
            if boundary.has_buffered_adjoint():
                raise RuntimeError(
                    f"source-ready group {group_id} closed before its group "
                    "boundary adjoints were drained"
                )
            if not self._source_ready_sources_by_group.get(group_id):
                raise RuntimeError(
                    f"source-ready group {group_id} closed before any explicit "
                    "source drain"
                )
            for node in self._shared_prompt_nodes_by_group.get(group_id, []):
                if group_id in node.processed_group_ids:
                    raise RuntimeError(
                        f"source-ready group {group_id} completed node "
                        f"{node.cid} twice"
                    )
                node.processed_group_ids.add(group_id)
        try:
            g = self._groups.pop(group_id)
            self._kv.drop(g.owned_prompt_chunks)
            boundary = self._boundaries.pop(group_id, None)
            if boundary is not None:
                boundary.release(
                    require_empty=(
                        self.shared_prefix_vjp_mode == "source_ready_serial"
                    )
                )
            self._mark_shared_prompt_group_closed(group_id)
            if self.shared_prefix_vjp_mode == "source_ready_serial":
                self._release_source_ready_shared_prompt_group(group_id)
            elif self.dependency_ready_shared_prefix_vjp:
                self._finalize_ready_shared_prompt_nodes(
                    trigger_group_id=group_id
                )
        except BaseException:
            if self.shared_prefix_vjp_mode == "source_ready_serial":
                self._mark_adjoint_capture_failed(
                    ("source_ready_group_close", group_id)
                )
            raise

    def finalize_shared_prefixes(self) -> int:
        """Complete and verify shared prompt finalization before the update."""
        self._assert_adjoint_capture_valid(stage="shared-prefix finalization")
        if self._groups:
            raise RuntimeError(
                "cannot finalize shared prefixes with open groups "
                f"{sorted(self._groups)}"
            )
        before = len(self._shared_prompt_finalization_events)
        if self.shared_prefix_vjp_mode == "aggregate":
            self._finalize_ready_shared_prompt_nodes(trigger_group_id=None)
        pending = [node for node in self._shared_prompt_nodes if not node.finalized]
        if pending:
            details = [
                {
                    "cid": node.cid,
                    "pending_groups": sorted(node.pending_group_ids),
                    "pending_children": sorted(node.pending_child_cids),
                }
                for node in pending
            ]
            raise RuntimeError(
                "shared prefix dependency accounting did not reach zero: "
                f"{details}"
            )
        if self._pending_shared_adjoints:
            raise RuntimeError("unconsumed shared-node adjoints after finalization")
        if self._isolated_ancestor_proxy_adjoints:
            raise RuntimeError(
                "unconsumed isolated descendant-to-ancestor adjoints after "
                "finalization"
            )
        self._accumulate_isolated_shared_prompt_parameter_gradients()
        self._materialize_parameter_adjoints()
        return len(self._shared_prompt_finalization_events) - before

    def _initialize_shared_prompt_dependencies(self) -> None:
        """Build group-consumer and shared-child edges once per forest."""
        self._shared_prompt_nodes_by_cid = {
            node.cid: node for node in self._shared_prompt_nodes
        }
        self._shared_prompt_nodes_by_group.clear()
        for node in self._shared_prompt_nodes:
            node.pending_group_ids = set(node.consumer_group_ids)
            node.child_cids.clear()
            node.pending_child_cids.clear()
            node.processed_group_ids.clear()
            node.source_process_count = 0
            node.source_vjp_count = 0
            ancestor_cids: list[int] = []
            ancestor_cid = node.parent_cid
            while ancestor_cid is not None:
                ancestor = self._shared_prompt_nodes_by_cid.get(ancestor_cid)
                if ancestor is None:
                    break
                ancestor_cids.append(ancestor.cid)
                ancestor_cid = ancestor.parent_cid
            node.ancestor_shared_cids = tuple(reversed(ancestor_cids))
            node.finalized = False
            node.finalization_index = None
            node.canonical_vjp_index = None
            for group_id in node.consumer_group_ids:
                self._shared_prompt_nodes_by_group.setdefault(
                    group_id, []
                ).append(node)
        for node in self._shared_prompt_nodes:
            parent = self._shared_prompt_nodes_by_cid.get(node.parent_cid)
            if parent is not None:
                parent.child_cids.add(node.cid)
                parent.pending_child_cids.add(node.cid)
        self._assign_canonical_shared_prompt_vjp_indices()

    def _assign_canonical_shared_prompt_vjp_indices(self) -> None:
        """Record the established deferred-sequential node execution order."""
        pending_children = {
            node.cid: set(node.child_cids) for node in self._shared_prompt_nodes
        }
        completed: set[int] = set()
        order: list[_SharedPromptNode] = []
        while len(completed) < len(self._shared_prompt_nodes):
            ready = [
                node
                for node in reversed(self._shared_prompt_nodes)
                if node.cid not in completed and not pending_children[node.cid]
            ]
            if not ready:
                raise RuntimeError("shared prompt forest contains a dependency cycle")
            for node in ready:
                completed.add(node.cid)
                order.append(node)
                if node.parent_cid in pending_children:
                    pending_children[node.parent_cid].remove(node.cid)
        for index, node in enumerate(order):
            node.canonical_vjp_index = index

    def _mark_shared_prompt_group_closed(self, group_id: int) -> None:
        """Remove the direct-consumer edges owned by one closed group."""
        if group_id in self._closed_shared_prompt_groups:
            raise RuntimeError(f"shared prompt group {group_id} was released twice")
        self._closed_shared_prompt_groups.add(group_id)
        for node in self._shared_prompt_nodes_by_group.get(group_id, []):
            if node.finalized:
                raise RuntimeError(
                    f"shared prompt node {node.cid} finalized before consumer "
                    f"group {group_id} closed"
                )
            if group_id not in node.pending_group_ids:
                raise RuntimeError(
                    f"shared prompt node {node.cid} lost consumer group "
                    f"{group_id} before group closure"
                )
            node.pending_group_ids.remove(group_id)

    def _finalize_source_ready_shared_prompt_source(
        self,
        group_id: int,
        *,
        source: Any,
    ) -> int:
        """Execute one source VJP along one child-to-parent forest path."""
        self._assert_adjoint_capture_valid(
            stage="source-ready shared-prefix finalization"
        )
        path = self._shared_prompt_nodes_by_group.get(group_id, [])
        processed = 0
        processed_cids: set[int] = set()
        try:
            for node in reversed(path):
                if node.finalized:
                    raise RuntimeError(
                        f"shared prompt node {node.cid} was released before "
                        f"source group {group_id}"
                    )
                source_children = [
                    self._shared_prompt_nodes_by_cid[child_cid]
                    for child_cid in node.child_cids
                    if group_id
                    in self._shared_prompt_nodes_by_cid[
                        child_cid
                    ].consumer_group_ids
                ]
                if len(source_children) > 1:
                    raise RuntimeError(
                        f"source group {group_id} has multiple child paths "
                        f"below shared node {node.cid}"
                    )
                unprocessed = [
                    child.cid
                    for child in source_children
                    if child.cid not in processed_cids
                ]
                if unprocessed:
                    raise RuntimeError(
                        f"shared node {node.cid} reached before children "
                        f"{unprocessed}"
                    )
                adjoint_diagnostics = node.boundary.adjoint_diagnostics()
                outputs, gradients = node.boundary.drain_adjoint_pairs(
                    expected_source=source,
                    reject_native=True,
                )
                if outputs:
                    self.backward_with_boundary_adjoint_capture(
                        outputs,
                        grad_tensors=gradients,
                        source=source,
                        retain_graph=True,
                    )
                    node.source_vjp_count += 1
                    self._shared_prompt_vjp_calls += 1
                node.source_process_count += 1
                processed_cids.add(node.cid)
                self._shared_prompt_source_vjp_events.append(
                    {
                        "source_vjp_index": len(
                            self._shared_prompt_source_vjp_events
                        ),
                        "cid": node.cid,
                        "parent_cid": node.parent_cid,
                        "group_id": group_id,
                        "readiness": (
                            "source_backward_and_source_child_complete"
                        ),
                        "vjp_invoked": bool(outputs),
                        "source_process_count": node.source_process_count,
                        "retained_graph_for_later_sources": True,
                        "gradient_tensor_count": len(outputs),
                        "adjoint_accumulator_dtype": adjoint_diagnostics[
                            "accumulator_dtype"
                        ],
                        "adjoint_accumulator_tensor_count": adjoint_diagnostics[
                            "accumulator_tensor_count"
                        ],
                        "adjoint_accumulator_bytes": adjoint_diagnostics[
                            "accumulator_bytes"
                        ],
                        "adjoint_contribution_count": adjoint_diagnostics[
                            "contribution_count"
                        ],
                        "open_group_ids": self.open_group_ids(),
                    }
                )
                processed += 1
        except BaseException:
            self._mark_adjoint_capture_failed(source)
            raise
        return processed

    def _release_source_ready_shared_prompt_group(self, group_id: int) -> int:
        """Release nodes after all direct consumers and children close."""
        finalized = 0
        for node in reversed(self._shared_prompt_nodes):
            if node.finalized or node.pending_group_ids or node.pending_child_cids:
                continue
            if node.processed_group_ids != set(node.consumer_group_ids):
                raise RuntimeError(
                    f"shared prompt node {node.cid} has incomplete source "
                    "coverage at release"
                )
            if node.pending_child_cids:
                raise RuntimeError(
                    f"shared prompt node {node.cid} retained child edges at "
                    "release"
                )
            if node.boundary.has_buffered_adjoint():
                raise RuntimeError(
                    f"shared prompt node {node.cid} retained an adjoint at "
                    "release"
                )
            self._kv.drop([node.cid])
            node.boundary.release(require_empty=True)
            node.finalized = True
            node.finalization_index = len(self._shared_prompt_finalization_events)
            parent = self._shared_prompt_nodes_by_cid.get(node.parent_cid)
            if parent is not None:
                if node.cid not in parent.pending_child_cids:
                    raise RuntimeError(
                        f"shared prompt parent {parent.cid} lost child edge "
                        f"{node.cid}"
                    )
                parent.pending_child_cids.remove(node.cid)
            self._shared_prompt_finalization_events.append(
                {
                    "finalization_index": node.finalization_index,
                    "cid": node.cid,
                    "parent_cid": node.parent_cid,
                    "depth_start": node.depth_start,
                    "depth_end": node.depth_end,
                    "consumer_group_ids": list(node.consumer_group_ids),
                    "child_cids": sorted(node.child_cids),
                    "trigger_group_id": group_id,
                    "open_group_ids": self.open_group_ids(),
                    "source_vjp_count": node.source_vjp_count,
                    "source_process_count": node.source_process_count,
                    "group_process_count": len(node.processed_group_ids),
                    "release_condition": "all_group_consumers_closed",
                    "vjp_cuda_interval_index": None,
                }
            )
            finalized += 1
        return finalized

    @staticmethod
    def _shared_prompt_node_ready(node: _SharedPromptNode) -> bool:
        return (
            not node.finalized
            and not node.pending_group_ids
            and not node.pending_child_cids
        )

    def _shared_prompt_node_cotangents(
        self, node: _SharedPromptNode
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if self._parameter_adjoint_capture_enabled:
            reduced_direct: list[torch.Tensor | None] = []
            for proxy in node.boundary.proxy:
                pending = self._pending_shared_adjoints.pop(id(proxy), None)
                native = proxy.grad
                if pending:
                    reduced = reduce_boundary_adjoints(
                        pending,
                        existing=native,
                        backend=(
                            "triton"
                            if self.boundary_adjoint_backend == "triton"
                            else "torch"
                        ),
                    )
                else:
                    reduced = native
                reduced_direct.append(reduced)
                proxy.grad = None
            if any(value is not None for value in reduced_direct):
                node.boundary.accumulate_explicit_adjoints(
                    reduced_direct,
                    source=("boundary_reduction", node.cid),
                )
            return node.boundary.consume_adjoint_pairs(reject_native=True)
        self._assemble_isolated_parent_proxy_adjoints(node)
        outputs: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []
        for real, proxy in zip(node.boundary.real, node.boundary.proxy, strict=True):
            self._flush_shared_adjoints(proxy)
            if proxy.grad is not None:
                outputs.append(real)
                gradients.append(proxy.grad.to(real.dtype))
        return outputs, gradients

    def _assemble_isolated_parent_proxy_adjoints(
        self, node: _SharedPromptNode
    ) -> None:
        """Add descendant cotangents to a node's proxies in canonical order."""
        by_child = self._isolated_ancestor_proxy_adjoints.pop(node.cid, None)
        if not by_child:
            return
        released_bytes = sum(
            contribution.numel() * contribution.element_size()
            for contributions in by_child.values()
            for contribution in contributions
            if contribution is not None
        )
        self._isolated_ancestor_adjoint_bytes_current -= released_bytes
        if self._isolated_ancestor_adjoint_bytes_current < 0:
            raise RuntimeError("isolated ancestor-adjoint byte count underflow")
        for child_index in sorted(by_child):
            contributions = by_child[child_index]
            if len(contributions) != len(node.boundary.proxy):
                raise RuntimeError(
                    f"shared node {node.cid}: child cotangent arity differs "
                    "from the parent boundary"
                )
            for proxy, contribution in zip(
                node.boundary.proxy, contributions, strict=True
            ):
                if contribution is None:
                    continue
                value = contribution.to(dtype=proxy.dtype, device=proxy.device)
                if proxy.grad is None:
                    proxy.grad = value
                else:
                    proxy.grad.add_(value)

    def _execute_isolated_shared_prompt_node_vjp(
        self,
        node: _SharedPromptNode,
        outputs: list[torch.Tensor],
        gradients: list[torch.Tensor],
    ) -> bool:
        """Compute one node VJP without changing global parameter gradients."""
        canonical_index = node.canonical_vjp_index
        if canonical_index is None:
            raise RuntimeError(f"shared node {node.cid} lacks a canonical index")
        if canonical_index in self._isolated_shared_parameter_gradients:
            raise RuntimeError(
                f"shared node {node.cid} parameter contribution was staged twice"
            )

        ancestors = [
            self._shared_prompt_nodes_by_cid[cid]
            for cid in node.ancestor_shared_cids
        ]
        ancestor_proxy_counts = [
            len(ancestor.boundary.proxy) for ancestor in ancestors
        ]
        ancestor_proxies = tuple(
            proxy for ancestor in ancestors for proxy in ancestor.boundary.proxy
        )
        inputs: tuple[torch.Tensor, ...] = (
            *self._shared_prompt_trainable_parameters,
            *ancestor_proxies,
        )
        if outputs:
            returned = torch.autograd.grad(
                outputs,
                inputs,
                grad_outputs=gradients,
                allow_unused=True,
            )
        else:
            returned = tuple(None for _ in inputs)

        parameter_count = len(self._shared_prompt_trainable_parameters)
        raw_parameter_gradients = returned[:parameter_count]
        parameter_gradients = tuple(
            None
            if gradient is None
            else gradient.detach().to(dtype=torch.float32)
            for gradient in raw_parameter_gradients
        )
        self._isolated_shared_parameter_gradients[canonical_index] = (
            parameter_gradients
        )
        staged_bytes = sum(
            gradient.numel() * gradient.element_size()
            for gradient in parameter_gradients
            if gradient is not None
        )
        self._isolated_parameter_gradient_bytes_current += staged_bytes
        self._isolated_parameter_gradient_bytes_peak = max(
            self._isolated_parameter_gradient_bytes_peak,
            self._isolated_parameter_gradient_bytes_current,
        )

        ancestor_gradients = tuple(
            None if gradient is None else gradient.detach()
            for gradient in returned[parameter_count:]
        )
        if len(ancestor_gradients) != len(ancestor_proxies):
            raise RuntimeError(
                f"shared node {node.cid}: ancestor cotangent arity mismatch"
            )
        offset = 0
        for ancestor, count in zip(
            ancestors, ancestor_proxy_counts, strict=True
        ):
            contributions = ancestor_gradients[offset : offset + count]
            offset += count
            if not any(contribution is not None for contribution in contributions):
                continue
            by_child = self._isolated_ancestor_proxy_adjoints.setdefault(
                ancestor.cid, {}
            )
            if canonical_index in by_child:
                raise RuntimeError(
                    f"shared node {node.cid}: ancestor cotangent was staged twice"
                )
            by_child[canonical_index] = contributions
            staged_bytes = sum(
                contribution.numel() * contribution.element_size()
                for contribution in contributions
                if contribution is not None
            )
            self._isolated_ancestor_adjoint_bytes_current += staged_bytes
            self._isolated_ancestor_adjoint_bytes_peak = max(
                self._isolated_ancestor_adjoint_bytes_peak,
                self._isolated_ancestor_adjoint_bytes_current,
            )
        return bool(outputs)

    def _accumulate_isolated_shared_prompt_parameter_gradients(self) -> None:
        """Apply staged node gradients once in deferred-sequential order."""
        if self.shared_prefix_vjp_backend != "isolated_canonical":
            return
        if self._isolated_parameter_gradients_accumulated:
            if self._isolated_shared_parameter_gradients:
                raise RuntimeError(
                    "isolated parameter contributions appeared after accumulation"
                )
            return
        expected = set(range(len(self._shared_prompt_nodes)))
        actual = set(self._isolated_shared_parameter_gradients)
        if actual != expected:
            raise RuntimeError(
                "isolated parameter contribution indices differ from the "
                f"canonical forest order: expected={sorted(expected)}, "
                f"actual={sorted(actual)}"
            )
        with torch.no_grad():
            for canonical_index in sorted(expected):
                gradients = self._isolated_shared_parameter_gradients[
                    canonical_index
                ]
                for parameter, gradient in zip(
                    self._shared_prompt_trainable_parameters,
                    gradients,
                    strict=True,
                ):
                    if gradient is None:
                        continue
                    value = gradient.to(
                        dtype=parameter.dtype, device=parameter.device
                    )
                    if parameter.grad is None:
                        parameter.grad = value
                    else:
                        parameter.grad.add_(value)
        self._isolated_shared_parameter_gradients.clear()
        self._isolated_parameter_gradient_bytes_current = 0
        self._isolated_parameter_gradients_accumulated = True

    def _commit_shared_prompt_node_finalization(
        self,
        node: _SharedPromptNode,
        *,
        trigger_group_id: int | None,
        frontier_index: int,
        frontier_width: int,
        gradient_tensor_count: int,
        adjoint_diagnostics: dict[str, Any],
        vjp_started_monotonic_s: float,
        vjp_finished_monotonic_s: float,
        vjp_cuda_interval_index: int | None,
    ) -> None:
        if node.pending_child_cids:
            raise RuntimeError(
                f"shared prompt node {node.cid} reached final release with "
                f"pending children {sorted(node.pending_child_cids)}"
            )
        self._kv.drop([node.cid])
        # Drop graph-bearing references at the exact dependency transition.
        node.boundary.release()
        node.finalized = True
        node.finalization_index = len(self._shared_prompt_finalization_events)
        parent = self._shared_prompt_nodes_by_cid.get(node.parent_cid)
        if parent is not None:
            if node.cid not in parent.pending_child_cids:
                raise RuntimeError(
                    f"shared prompt parent {parent.cid} lost child edge "
                    f"{node.cid} before child finalization"
                )
            parent.pending_child_cids.remove(node.cid)
        self._shared_prompt_finalization_events.append(
            {
                "finalization_index": node.finalization_index,
                "cid": node.cid,
                "parent_cid": node.parent_cid,
                "depth_start": node.depth_start,
                "depth_end": node.depth_end,
                "consumer_group_ids": list(node.consumer_group_ids),
                "child_cids": sorted(node.child_cids),
                "trigger_group_id": trigger_group_id,
                "open_group_ids": self.open_group_ids(),
                "frontier_index": frontier_index,
                "frontier_width": frontier_width,
                "gradient_tensor_count": gradient_tensor_count,
                "adjoint_accumulator_dtype": adjoint_diagnostics[
                    "accumulator_dtype"
                ],
                "adjoint_accumulator_tensor_count": adjoint_diagnostics[
                    "accumulator_tensor_count"
                ],
                "adjoint_accumulator_bytes": adjoint_diagnostics[
                    "accumulator_bytes"
                ],
                "adjoint_contribution_count": adjoint_diagnostics[
                    "contribution_count"
                ],
                "vjp_started_monotonic_s": vjp_started_monotonic_s,
                "vjp_finished_monotonic_s": vjp_finished_monotonic_s,
                "vjp_duration_s": (
                    vjp_finished_monotonic_s - vjp_started_monotonic_s
                ),
                "vjp_timing_uses_cuda_events": (
                    vjp_cuda_interval_index is not None
                ),
                "vjp_cuda_interval_index": vjp_cuda_interval_index,
            }
        )

    def _finalize_ready_shared_prompt_nodes(
        self, *, trigger_group_id: int | None
    ) -> int:
        """Execute every ready VJP in child-before-parent frontiers.

        Each frontier is an antichain: no node in it is an ancestor of another.
        The optimized backend submits all active outputs in that frontier to one
        autograd traversal. The subsequent dependency update can expose the
        parent frontier. This preserves the exact reverse-topological relation.
        """
        finalized = 0
        frontier_index = 0
        while True:
            ready = [
                node
                for node in reversed(self._shared_prompt_nodes)
                if self._shared_prompt_node_ready(node)
            ]
            if not ready:
                break
            if self.shared_prefix_vjp_backend in (
                "sequential",
                "isolated_canonical",
            ):
                frontiers = [[node] for node in ready]
            else:
                frontiers = [ready]
            for frontier in frontiers:
                outputs: list[torch.Tensor] = []
                gradients: list[torch.Tensor] = []
                tensor_counts: dict[int, int] = {}
                adjoint_diagnostics_by_cid: dict[int, dict[str, Any]] = {}
                for node in frontier:
                    adjoint_diagnostics_by_cid[node.cid] = (
                        node.boundary.adjoint_diagnostics()
                    )
                    node_outputs, node_gradients = (
                        self._shared_prompt_node_cotangents(node)
                    )
                    tensor_counts[node.cid] = len(node_outputs)
                    outputs.extend(node_outputs)
                    gradients.extend(node_gradients)
                cuda_interval_index = None
                start_event = finish_event = None
                vjp_cuda_stream = None
                if (
                    self.shared_prefix_vjp_timing
                    and self.weight.device.type == "cuda"
                ):
                    start_event = torch.cuda.Event(enable_timing=True)
                    finish_event = torch.cuda.Event(enable_timing=True)
                    vjp_cuda_stream = torch.cuda.current_stream(
                        self.weight.device
                    )
                    start_event.record(vjp_cuda_stream)
                vjp_started = time.perf_counter()
                with operator_range(
                    "shared_prompt_forest_vjp",
                    role=device_role(self.weight.device),
                    device=self.weight.device,
                    backend=self.shared_prefix_vjp_backend,
                    frontier=frontier_index,
                    frontier_width=len(frontier),
                    nodes=",".join(str(node.cid) for node in frontier),
                    tokens=sum(
                        node.depth_end - node.depth_start for node in frontier
                    ),
                    maximum_ancestor_nodes=max(
                        (len(node.ancestor_shared_cids) for node in frontier),
                        default=0,
                    ),
                    output_tensors=len(outputs),
                ):
                    if self.shared_prefix_vjp_backend == "isolated_canonical":
                        executed = self._execute_isolated_shared_prompt_node_vjp(
                            frontier[0], outputs, gradients
                        )
                        if executed:
                            self._shared_prompt_vjp_calls += 1
                    elif outputs:
                        if self._parameter_adjoint_capture_enabled:
                            self.backward_with_boundary_adjoint_capture(
                                outputs,
                                grad_tensors=gradients,
                                source=(
                                    "shared_prompt_frontier",
                                    frontier_index,
                                    tuple(node.cid for node in frontier),
                                ),
                            )
                        else:
                            torch.autograd.backward(
                                outputs,
                                grad_tensors=gradients,
                            )
                        self._shared_prompt_vjp_calls += 1
                if finish_event is not None:
                    assert start_event is not None
                    assert vjp_cuda_stream is not None
                    finish_event.record(vjp_cuda_stream)
                    cuda_interval_index = len(
                        self._shared_prompt_vjp_cuda_intervals
                    )
                    self._shared_prompt_vjp_cuda_intervals.append(
                        (start_event, finish_event)
                    )
                vjp_finished = time.perf_counter()
                self._shared_prompt_vjp_frontier_widths.append(len(frontier))
                for node in frontier:
                    self._commit_shared_prompt_node_finalization(
                        node,
                        trigger_group_id=trigger_group_id,
                        frontier_index=frontier_index,
                        frontier_width=len(frontier),
                        gradient_tensor_count=tensor_counts[node.cid],
                        adjoint_diagnostics=adjoint_diagnostics_by_cid[node.cid],
                        vjp_started_monotonic_s=vjp_started,
                        vjp_finished_monotonic_s=vjp_finished,
                        vjp_cuda_interval_index=cuda_interval_index,
                    )
                    finalized += 1
                frontier_index += 1
            # The sequential backend visits a reverse-topological list in one
            # pass; its parent dependencies may have become ready within that
            # pass. Looping once more is inexpensive and validates completion.
        return finalized

    # ---------------------------------------------------------------- internal

    def _new_chunk(self, n_tokens: int, parent: int | None) -> int:
        cid = len(self._chunks)
        self._chunks.append(_Chunk(idx=cid, n_tokens=n_tokens, parent=parent))
        return cid

    def _forward(
        self, cid: int, chain: list[int], tokens: list[int], pos0: int,
        *, plain: bool = False,
    ) -> torch.Tensor:
        """Run one chunk through the training graph. Returns [T, D] hidden states.

        ``plain=True`` forces the non-checkpointed executor even on a
        checkpointed run: used by `rebuild_branch_logprobs`, whose graph is
        transient (built, backwarded and freed within one close), so per-layer
        checkpointing would only add a redundant recompute pass.
        """
        seg = _Seg(cid=cid, chain=chain, start=0, length=len(tokens))
        return self._forward_packed(
            [seg], tokens, list(range(pos0, pos0 + len(tokens))), plain=plain
        )[0]

    def _forward_packed(
        self, segments: list[_Seg], tokens: list[int], positions: list[int],
        plain: bool = False, force_checkpoint: bool = False,
        joint_backward: bool = False,
        cohort_ownership: dict[str, Any] | None = None,
        checkpoint_input_storage_device: torch.device | None = None,
        checkpoint_offload_stats: dict[str, Any] | None = None,
    ) -> list[torch.Tensor]:
        """One model call over the packed segments; per-segment [T_i, D] out.

        ``force_checkpoint=True`` selects the per-layer checkpointed executor
        regardless of the run's own `checkpoint` flag: the block rebuild uses
        it so its transient stays block-sized even on runs whose streamed
        forwards retain full activations."""
        dev = self.weight.device
        ids = _device_long(tokens, dev).unsqueeze(0)
        pos = _device_long(positions, dev).unsqueeze(0)

        if self._layer_kinds:
            # Hybrid stack: ALWAYS the per-layer executor, `plain` included.
            # The plain executor calls the model's own forward, whose linear
            # layers would start every chunk from a zero state -- silently a
            # different model. The per-layer executor threads (state, conv
            # tail) explicitly; its extra recompute pass on plain rebuilds is
            # the disclosed cost of exactness on this family.
            return self._forward_checkpointed(
                segments,
                ids,
                pos,
                joint_backward=joint_backward,
                cohort_ownership=cohort_ownership,
                checkpoint_input_storage_device=checkpoint_input_storage_device,
                checkpoint_offload_stats=checkpoint_offload_stats,
            )

        if force_checkpoint or (self.checkpoint and not plain):
            return self._forward_checkpointed(
                segments,
                ids,
                pos,
                joint_backward=joint_backward,
                cohort_ownership=cohort_ownership,
                checkpoint_input_storage_device=checkpoint_input_storage_device,
                checkpoint_offload_stats=checkpoint_offload_stats,
            )

        if cohort_ownership is not None:
            raise ValueError(
                "cohort ownership requires the checkpointed joint VJP"
            )

        if len(segments) != 1 and not joint_backward:
            # Several segments on the plain executor record ONE graph across
            # their trajectories, so a per-trajectory backward through it
            # would free its window-mates' saved tensors. Only a turn pack
            # whose losses are backwarded in one traversal may take it.
            raise RuntimeError(
                "the plain executor forwards several segments only as a "
                "jointly backwarded pack"
            )
        prev = _STREAMS.get(dev)
        _STREAMS[dev] = self
        if len(segments) == 1:
            seg = segments[0]
            self._active_chunk, self._active_chain = seg.cid, seg.chain
        else:
            offset = 0
            for seg in segments:
                if seg.start != offset:
                    raise AssertionError(
                        "packed segments must be contiguous in trajectory order"
                    )
                offset += seg.length
            self._active_segments = [
                (seg.cid, seg.chain, seg.length) for seg in segments
            ]
        try:
            out = self.stack(
                input_ids=ids,
                position_ids=pos,
                attention_mask=None,
                use_cache=False,
            )
        finally:
            self._active_chunk, self._active_chain = None, None
            self._active_segments = None
            self._active_pack_layouts = {}
            if prev is None:
                _STREAMS.pop(dev, None)
            else:
                _STREAMS[dev] = prev
        hidden = out.last_hidden_state[0]
        if len(segments) == 1:
            return [hidden]
        return list(hidden.split([seg.length for seg in segments], dim=0))

    def _forward_checkpointed(
        self,
        segments: list[_Seg],
        ids: torch.Tensor,
        pos: torch.Tensor,
        *,
        joint_backward: bool = False,
        cohort_ownership: dict[str, Any] | None = None,
        checkpoint_input_storage_device: torch.device | None = None,
        checkpoint_offload_stats: dict[str, Any] | None = None,
    ) -> list[torch.Tensor]:
        """The packed per-layer checkpointed executor.

        Drives embed -> layers -> norm manually. Per layer, two phases.  The
        default records one graph per segment:

        * PRIMAL: the whole packed stream runs through the layer ONCE under
          no-grad -- one batched call for the dense ops, which is coalescing's
          entire win; attention loops over segments via the device-local
          checkpoint context. The primal produces values only, never graph.
        * GRAPH: for each segment, one `_SegLayerCkpt` node adopts the
          segment's slice of the primal outputs and will RECOMPUTE that segment
          alone at backward. Its outputs `(hidden, k, v)` are graph-attached
          (fresh K/V is a node OUTPUT, never a stash from the no-grad region)
          and the ancestor K/V are node INPUTS, so both directions of the
          cross-chunk gradient path survive recompute, with the checkpoint's
          primal batched across segments.
          One node per segment keeps different trajectories' graphs disjoint,
          which the reward-linear per-trajectory `retain_graph=False`
          backwards require
          (see _SegLayerCkpt).

        The embedding gather and the final norm are per-token and cheap, so
        they run per segment WITH grad -- their nodes are naturally
        per-segment. What stays resident per segment-layer: the boundary hidden
        state and the K/V (the S1 checkpoint arithmetic), cloned contiguous so
        window-mates share no storage and per-trajectory freeing returns real
        bytes.

        Recompute correctness note: the recompute re-runs `_layer_fn`, which
        re-establishes the attention context from the node's own saved
        arguments -- nothing is read from state that could have moved between
        forward and backward. Parameters ARE read live, which is exactly why the
        version guard (the reward-linear backward) forbids optimizer steps while groups are open.

        With ``joint_backward=True``, the graph phase adopts the same packed
        layer output through one `_PackedLayerCkpt`.  Its backward repeats the
        layer once for the complete closure cohort.  This topology requires a
        single joint backward over every cohort loss; see
        :meth:`rebuild_logprobs`.
        """
        # A one-segment block round uses the established independent checkpoint.
        # Cohort ownership applies only to a real pack.
        joint_backward = joint_backward and len(segments) > 1
        if joint_backward and cohort_ownership is None:
            raise ValueError("joint backward requires cohort ownership")
        stack = self.stack
        # Order invariant, asserted rather than trusted: the pack lays each
        # segment's tokens out contiguously, in trajectory order, segments
        # back to back. Attention only needs the slices to be disjoint;
        # a linear layer's recurrence needs exactly this layout.
        off = 0
        for s in segments:
            if s.start != off:
                raise AssertionError(
                    "packed segments must be contiguous in trajectory order"
                )
            off += s.length
        h_seg = [
            stack.embed_tokens(ids.narrow(1, s.start, s.length)) for s in segments
        ]
        with torch.no_grad():
            # the packed primal input is the same lookup the per-segment nodes
            # just did; reuse their values instead of embedding twice
            h_primal = (
                h_seg[0].detach()
                if len(segments) == 1
                else torch.cat([h.detach() for h in h_seg], dim=1)
            )
            cos, sin = stack.rotary_emb(h_primal, _rotary_positions(stack.rotary_emb, pos))
        h_joint = None
        if joint_backward:
            h_joint = (
                h_seg[0]
                if len(segments) == 1
                else torch.cat(h_seg, dim=1)
            )
        cos_seg = [cos.narrow(1, s.start, s.length) for s in segments]
        sin_seg = [sin.narrow(1, s.start, s.length) for s in segments]

        for li, layer in enumerate(stack.layers):
            linear = hasattr(layer, "linear_attn")
            meta, per_seg_past = [], []
            for seg in segments:
                if linear:
                    # A linear layer's boundary is its parent chunk's final
                    # (recurrent state, conv tail) -- the state already
                    # summarises the whole prefix, so the chain collapses to
                    # one hop. Stored in the same per-(layer, chunk) store as
                    # K/V; a trajectory root starts from the zero state, which
                    # is exactly the monolithic model's start.
                    parent = seg.chain[-2] if len(seg.chain) > 1 else None
                    if parent is None:
                        ks, vs = [], []
                    else:
                        state, tail = self._kv.get(li, parent)
                        ks, vs = [state], [tail]
                else:
                    ancestors = seg.chain[:-1]
                    ks, vs = self._kv.chunks(li, ancestors) if ancestors else ([], [])
                meta.append((seg.start, seg.length, len(ks)))
                per_seg_past.append((ks, vs))
            flat_ks = [k for ks, _ in per_seg_past for k in ks]
            flat_vs = [v for _, vs in per_seg_past for v in vs]

            with torch.no_grad():
                out = _layer_fn(
                    layer, h_primal, cos, sin, tuple(meta), *flat_ks, *flat_vs
                )
            h_primal = out[0]

            params = list(layer.parameters())
            if joint_backward:
                assert h_joint is not None
                if (
                    self.ragged_cohort_attention
                    and len(segments) > 1
                    and _ragged_attention_runs(
                        h_primal, getattr(getattr(layer, "self_attn", None), "head_dim", None)
                    )
                ):
                    adopted = _RaggedQwen3LayerCkpt.apply(
                        cohort_ownership,
                        layer,
                        tuple(out),
                        tuple(meta),
                        len(flat_ks),
                        *h_seg,
                        *cos_seg,
                        *sin_seg,
                        *flat_ks,
                        *flat_vs,
                        *params,
                    )
                    for i, seg in enumerate(segments):
                        h_i = adopted[3 * i]
                        k_i = adopted[3 * i + 1]
                        v_i = adopted[3 * i + 2]
                        self._kv.put(li, seg.cid, k_i, v_i)
                        h_seg[i] = h_i
                    continue
                adopted = _PackedLayerCkpt.apply(
                    cohort_ownership,
                    layer,
                    tuple(out),
                    tuple(meta),
                    len(flat_ks),
                    h_joint,
                    cos,
                    sin,
                    *flat_ks,
                    *flat_vs,
                    *params,
                )
                h_joint = adopted[0]
                for i, seg in enumerate(segments):
                    self._kv.put(
                        li, seg.cid, adopted[1 + 2 * i], adopted[2 + 2 * i]
                    )
                    h_seg[i] = h_joint.narrow(1, seg.start, seg.length)
                continue

            for i, seg in enumerate(segments):
                ks, vs = per_seg_past[i]
                primal = (
                    out[0].narrow(1, seg.start, seg.length),
                    out[1 + 2 * i],
                    out[2 + 2 * i],
                )
                h_i, k_i, v_i = _SegLayerCkpt.apply(
                    layer,
                    primal,
                    cos_seg[i],
                    sin_seg[i],
                    len(ks),
                    checkpoint_input_storage_device,
                    checkpoint_offload_stats,
                    h_seg[i],
                    *ks, *vs, *params,
                )
                self._kv.put(li, seg.cid, k_i, v_i)
                h_seg[i] = h_i

        return [stack.norm(h)[0] for h in h_seg]
