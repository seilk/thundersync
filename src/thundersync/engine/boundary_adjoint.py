"""Reduction of consumer adjoints at a differentiable forest boundary.

A shared segment has several consumers. Reverse-mode differentiation produces
one adjoint for each consumer and the segment's backward needs their sum:

    parent_adjoint = existing_parent_adjoint + sum(consumer_adjoints)

This operation is the objective-independent boundary between the segment
forest scheduler and model autograd. GRPO and OPD determine the consumer
adjoints differently, while both require this same reduction before a shared
segment is finalized in reverse topological order.

The PyTorch implementation is the numerical reference and remains the default.
The Triton implementation is an opt-in prototype. One program reads up to 16
separate consumer buffers and fuses their reduction with accumulation into an
existing parent adjoint. The fixed pointer capacity bounds compilation and
register use; the scheduler flushes a partial reduction at that degree.

Gradient semantics: this function is used while executing a first-order
backward, where boundary adjoints are detached tensors. If a caller supplies a
tensor that requires gradients, the PyTorch implementation is selected so the
derivative is explicit: the upstream derivative is copied to every active
consumer adjoint and to ``existing``. The Triton path does not implement
higher-order differentiation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch

try:  # Optional on CPU development and test hosts.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on installations without Triton
    triton = None
    tl = None


BoundaryAdjointBackend = Literal["torch", "triton"]
SUPPORTED_BACKENDS = frozenset(("torch", "triton"))
_TRITON_DTYPES = frozenset((torch.float16, torch.bfloat16, torch.float32))
TRITON_MAX_DEGREE = 16


if triton is not None:

    @triton.jit
    def _boundary_adjoint_reduce_kernel(
        adjoint_0_ptr,
        adjoint_1_ptr,
        adjoint_2_ptr,
        adjoint_3_ptr,
        adjoint_4_ptr,
        adjoint_5_ptr,
        adjoint_6_ptr,
        adjoint_7_ptr,
        adjoint_8_ptr,
        adjoint_9_ptr,
        adjoint_10_ptr,
        adjoint_11_ptr,
        adjoint_12_ptr,
        adjoint_13_ptr,
        adjoint_14_ptr,
        adjoint_15_ptr,
        existing_ptr,
        output_ptr,
        n_elements,
        DEGREE: tl.constexpr,
        HAS_EXISTING: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        if HAS_EXISTING:
            total = tl.load(existing_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
        else:
            total = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        if DEGREE > 0:
            total += tl.load(adjoint_0_ptr + offsets, mask=mask, other=0.0)
        if DEGREE > 1:
            total += tl.load(adjoint_1_ptr + offsets, mask=mask, other=0.0)
        if DEGREE > 2:
            total += tl.load(adjoint_2_ptr + offsets, mask=mask, other=0.0)
        if DEGREE > 3:
            total += tl.load(adjoint_3_ptr + offsets, mask=mask, other=0.0)
        if DEGREE > 4:
            total += tl.load(adjoint_4_ptr + offsets, mask=mask, other=0.0)
        if DEGREE > 5:
            total += tl.load(adjoint_5_ptr + offsets, mask=mask, other=0.0)
        if DEGREE > 6:
            total += tl.load(adjoint_6_ptr + offsets, mask=mask, other=0.0)
        if DEGREE > 7:
            total += tl.load(adjoint_7_ptr + offsets, mask=mask, other=0.0)
        if DEGREE > 8:
            total += tl.load(adjoint_8_ptr + offsets, mask=mask, other=0.0)
        if DEGREE > 9:
            total += tl.load(adjoint_9_ptr + offsets, mask=mask, other=0.0)
        if DEGREE > 10:
            total += tl.load(adjoint_10_ptr + offsets, mask=mask, other=0.0)
        if DEGREE > 11:
            total += tl.load(adjoint_11_ptr + offsets, mask=mask, other=0.0)
        if DEGREE > 12:
            total += tl.load(adjoint_12_ptr + offsets, mask=mask, other=0.0)
        if DEGREE > 13:
            total += tl.load(adjoint_13_ptr + offsets, mask=mask, other=0.0)
        if DEGREE > 14:
            total += tl.load(adjoint_14_ptr + offsets, mask=mask, other=0.0)
        if DEGREE > 15:
            total += tl.load(adjoint_15_ptr + offsets, mask=mask, other=0.0)
        tl.store(output_ptr + offsets, total, mask=mask)


def triton_boundary_adjoint_available() -> bool:
    """Whether this process can execute the custom CUDA kernel."""
    return triton is not None and torch.cuda.is_available()


def _validate_inputs(
    adjoints: Sequence[torch.Tensor | None],
    existing: torch.Tensor | None,
) -> list[torch.Tensor]:
    active = [adjoint for adjoint in adjoints if adjoint is not None]
    reference = active[0] if active else existing
    if reference is None:
        return active
    if not reference.is_floating_point():
        raise TypeError(
            "boundary adjoints must be floating-point tensors, got "
            f"{reference.dtype}"
        )
    for name, tensor in [
        *(('adjoint', adjoint) for adjoint in active),
        *((('existing', existing),) if existing is not None else ()),
    ]:
        if tensor.shape != reference.shape:
            raise ValueError(
                f"{name} shape {tuple(tensor.shape)} does not match "
                f"{tuple(reference.shape)}"
            )
        if tensor.dtype != reference.dtype:
            raise ValueError(
                f"{name} dtype {tensor.dtype} does not match {reference.dtype}"
            )
        if tensor.device != reference.device:
            raise ValueError(
                f"{name} device {tensor.device} does not match {reference.device}"
            )
    return active


def _torch_reduce(
    active: Sequence[torch.Tensor],
    existing: torch.Tensor | None,
) -> torch.Tensor | None:
    if any(tensor.requires_grad for tensor in active) or (
        existing is not None and existing.requires_grad
    ):
        reference = active[0] if active else existing
        if reference is None:
            return None
        result = torch.zeros_like(reference)
        if existing is not None:
            result = result + existing
        for adjoint in active:
            result = result + adjoint
        return result
    if existing is not None:
        result = existing.clone()
    elif active:
        result = torch.zeros_like(active[0])
    else:
        return None
    for adjoint in active:
        result.add_(adjoint)
    return result


def _can_use_triton(
    active: Sequence[torch.Tensor],
    existing: torch.Tensor | None,
) -> bool:
    reference = active[0] if active else existing
    return bool(
        active
        and reference is not None
        and reference.is_cuda
        and reference.dtype in _TRITON_DTYPES
        and len(active) <= TRITON_MAX_DEGREE
        and all(tensor.is_contiguous() for tensor in active)
        and triton_boundary_adjoint_available()
        and not any(tensor.requires_grad for tensor in active)
        and (existing is None or not existing.requires_grad)
    )


def _triton_reduce(
    active: Sequence[torch.Tensor],
    existing: torch.Tensor | None,
) -> torch.Tensor:
    existing_contiguous = None if existing is None else existing.contiguous()
    output = torch.empty_like(active[0], memory_format=torch.contiguous_format)
    n_elements = output.numel()
    if n_elements == 0:
        return output
    block_size = 256
    grid = (triton.cdiv(n_elements, block_size),)
    # Unused pointers are compile-time eliminated by ``DEGREE``. Supplying a
    # valid pointer keeps the launch signature uniform across degrees.
    pointers = [*active, *([active[0]] * (TRITON_MAX_DEGREE - len(active)))]
    _boundary_adjoint_reduce_kernel[grid](
        *pointers,
        active[0] if existing_contiguous is None else existing_contiguous,
        output,
        n_elements,
        DEGREE=len(active),
        HAS_EXISTING=existing_contiguous is not None,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return output


def reduce_boundary_adjoints(
    adjoints: Sequence[torch.Tensor | None],
    *,
    existing: torch.Tensor | None = None,
    backend: BoundaryAdjointBackend = "torch",
) -> torch.Tensor | None:
    """Reduce consumer adjoints and optionally accumulate a parent adjoint.

    ``None`` entries represent missing adjoints and are ignored. If every
    entry and ``existing`` are ``None``, the result is ``None``. Inputs are not
    mutated. Requesting ``backend="triton"`` selects the custom kernel on a
    supported CUDA tensor and otherwise uses the PyTorch reference. The
    fallback permits one feature-gated integration to run on CPU, unsupported
    dtypes, and installations where Triton is absent.
    """
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"boundary adjoint backend must be one of {sorted(SUPPORTED_BACKENDS)}, "
            f"got {backend!r}"
        )
    active = _validate_inputs(adjoints, existing)
    if backend == "triton" and _can_use_triton(active, existing):
        return _triton_reduce(active, existing)
    return _torch_reduce(active, existing)
