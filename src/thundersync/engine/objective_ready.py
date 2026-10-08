"""Objective-ready execution primitives with schedule-independent reduction.

Readiness controls *when* a source contribution may be computed.  It must not
silently control the floating-point order used to construct the logical-batch
gradient.  The reducers in this module stage source-local contributions and
materialize them in a canonical logical order.

The implementation is deliberately independent of OPD and GRPO model code.
Both objectives produce the same ABI: a logical source identifier and one
optional tensor per trainable parameter.  Full-model runners may choose GPU or
CPU staging; that placement is part of their declared performance contract.
"""

from __future__ import annotations

import time
from typing import Sequence

import torch

CanonicalSource = int | str | tuple["CanonicalSource", ...]


def canonical_source_key(source: CanonicalSource) -> tuple:
    """Return a total, type-stable ordering key for logical source IDs.

    Python does not order heterogeneous tuples (for example ``("group", 1)``
    and ``(2, "turn")``).  Explicit type tags make the reduction order stable
    across processes and fail closed for opaque runtime objects.
    """

    if isinstance(source, bool):
        raise TypeError("boolean values are not valid canonical source IDs")
    if isinstance(source, int):
        return (0, source)
    if isinstance(source, str):
        return (1, source)
    if isinstance(source, tuple):
        return (2, tuple(canonical_source_key(item) for item in source))
    raise TypeError(
        "canonical source IDs must contain only ints, strings, and tuples; "
        f"got {type(source).__name__}"
    )


def _accumulator_dtype(tensor: torch.Tensor) -> torch.dtype:
    return torch.float64 if tensor.dtype == torch.float64 else torch.float32


class CanonicalGradientReducer:
    """Stage parameter-gradient contributions and reduce by logical source.

    Repeated contributions to one source are allowed because an OPD trajectory
    can emit several direct-turn and causal-replay contributions.  Their
    within-source order must be intrinsic to that source (turn order followed
    by reverse causal replay), while inter-source arrival may be arbitrary.
    Each source's FP32 (FP64) bucket lives on ``storage_device`` (the
    contribution's own device when None); ``finalize`` folds the buckets in
    canonical source order.
    """

    def __init__(
        self,
        parameter_count: int,
        *,
        storage_device: torch.device | str | None = None,
    ) -> None:
        if parameter_count < 0:
            raise ValueError("parameter_count must be non-negative")
        self.parameter_count = parameter_count
        self.storage_device = (
            None if storage_device is None else torch.device(storage_device)
        )
        self._by_source: dict[CanonicalSource, list[torch.Tensor | None]] = {}
        self._shapes: list[torch.Size | None] = [None] * parameter_count
        self._devices: list[torch.device | None] = [None] * parameter_count
        self._dtypes: list[torch.dtype | None] = [None] * parameter_count
        self._contribution_count = 0
        self._current_bytes = 0
        self._peak_bytes = 0
        self._peak_source_count = 0
        self._canonical_fold_s = 0.0
        self._materialization_s = 0.0
        self._finalized = False

    def add(
        self,
        source: CanonicalSource,
        gradients: Sequence[torch.Tensor | None],
    ) -> None:
        """Add one source-local contribution without exposing it to AdamW."""

        if self._finalized:
            raise RuntimeError("cannot add after canonical reduction finalization")
        canonical_source_key(source)
        try:
            hash(source)
        except TypeError as exc:
            raise TypeError("canonical source ID must be hashable") from exc
        if len(gradients) != self.parameter_count:
            raise ValueError(
                f"received {len(gradients)} gradients for "
                f"{self.parameter_count} parameters"
            )

        for index, gradient in enumerate(gradients):
            if gradient is None:
                continue
            self.add_parameter(source, index, gradient)

    def add_parameter(
        self,
        source: CanonicalSource,
        parameter_index: int,
        gradient: torch.Tensor,
    ) -> None:
        """Add one parameter tensor without constructing a sparse Python list."""

        if self._finalized:
            raise RuntimeError("cannot add after canonical reduction finalization")
        canonical_source_key(source)
        try:
            hash(source)
        except TypeError as exc:
            raise TypeError("canonical source ID must be hashable") from exc
        if parameter_index < 0 or parameter_index >= self.parameter_count:
            raise IndexError(
                f"parameter index {parameter_index} is outside "
                f"[0, {self.parameter_count})"
            )
        if gradient.is_sparse:
            raise NotImplementedError(
                "canonical parameter reduction requires dense gradients"
            )
        shape = self._shapes[parameter_index]
        if shape is None:
            self._shapes[parameter_index] = gradient.shape
            self._devices[parameter_index] = gradient.device
            self._dtypes[parameter_index] = gradient.dtype
        elif shape != gradient.shape:
            raise ValueError(
                f"parameter {parameter_index}: gradient shape {gradient.shape} "
                f"does not match established shape {shape}"
            )
        new_source = source not in self._by_source
        bucket = self._by_source.setdefault(source, [None] * self.parameter_count)
        if new_source:
            self._peak_source_count = max(
                self._peak_source_count, len(self._by_source)
            )
        value = gradient.detach().to(
            device=self.storage_device or gradient.device,
            dtype=_accumulator_dtype(gradient),
        )
        if bucket[parameter_index] is None:
            bucket[parameter_index] = value.clone()
            self._current_bytes += value.numel() * value.element_size()
            self._peak_bytes = max(self._peak_bytes, self._current_bytes)
        else:
            if bucket[parameter_index].device != value.device:
                raise RuntimeError(
                    f"canonical source {source!r} parameter {parameter_index} "
                    "received contributions on different devices"
                )
            bucket[parameter_index].add_(value)
        self._contribution_count += 1

    def _fold(self) -> list[torch.Tensor | None]:
        result: list[torch.Tensor | None] = [None] * self.parameter_count
        for source in sorted(self._by_source, key=canonical_source_key):
            for index, contribution in enumerate(self._by_source[source]):
                if contribution is None:
                    continue
                if result[index] is None:
                    result[index] = contribution.clone()
                else:
                    result[index].add_(contribution)
        return result

    def finalize(
        self,
        *,
        output_devices: Sequence[torch.device | str | None] | None = None,
        output_dtypes: Sequence[torch.dtype | None] | None = None,
    ) -> list[torch.Tensor | None]:
        """Reduce staged sources in canonical order and consume the ledger."""

        if self._finalized:
            raise RuntimeError("canonical reduction was already finalized")
        if output_devices is not None and len(output_devices) != self.parameter_count:
            raise ValueError("output_devices must align with parameters")
        if output_dtypes is not None and len(output_dtypes) != self.parameter_count:
            raise ValueError("output_dtypes must align with parameters")

        fold_started = time.perf_counter()
        result = self._fold()
        self._canonical_fold_s += time.perf_counter() - fold_started

        materialize_started = time.perf_counter()
        for index, value in enumerate(result):
            if value is None:
                continue
            device = (
                self._devices[index]
                if output_devices is None or output_devices[index] is None
                else torch.device(output_devices[index])
            )
            dtype = (
                self._dtypes[index]
                if output_dtypes is None or output_dtypes[index] is None
                else output_dtypes[index]
            )
            assert device is not None and dtype is not None
            result[index] = value.to(device=device, dtype=dtype)
        self._materialization_s += time.perf_counter() - materialize_started

        self._by_source.clear()
        self._current_bytes = 0
        self._finalized = True
        return result

    def diagnostics(self) -> dict[str, int | float | bool | str | None]:
        tensors = [
            tensor
            for gradients in self._by_source.values()
            for tensor in gradients
            if tensor is not None
        ]
        return {
            "finalized": self._finalized,
            "source_count": len(self._by_source),
            "peak_source_count": self._peak_source_count,
            "contribution_count": self._contribution_count,
            "tensor_count": len(tensors),
            "bytes": self._current_bytes,
            "peak_bytes": self._peak_bytes,
            "byte_accounting_scope": "resident_canonical_ledger_only",
            "transient_cast_clone_bytes_included": False,
            "canonical_fold_s": self._canonical_fold_s,
            "materialization_s": self._materialization_s,
            "canonical_fold_materialization_s": (
                self._canonical_fold_s + self._materialization_s
            ),
            "storage_device": (
                None if self.storage_device is None else str(self.storage_device)
            ),
        }
