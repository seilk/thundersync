"""ZeRO-1 optimizer-state sharding for the replicated data-parallel trainer.

Scope: shard OPTIMIZER STATE only. Parameters and gradients stay fully
replicated on every rank, exactly as in the data-parallel ranks -- the
gradient all-reduce keeps every rank's grads bitwise identical, the global
clip is computed over ALL parameters on every rank, and each rank then runs
`optimizer.step()` over ONLY its contiguous shard. The `optimizer_factory`
is invoked with just that shard, so the optimizer's per-parameter state
(AdamW's exp_avg and exp_avg_sq, in each parameter's dtype) is materialized
for roughly `1/world_size` of the model per rank. That is the memory win
this module exists for: a full AdamW state replica can exceed one device's
memory beside the parameter replica while a shard of it fits. Nothing else
about the trainers' step semantics changes.

Canonical parameter ordering -- the cross-rank contract
-------------------------------------------------------
The partition is a function of parameter ORDER, so every rank must construct
this class with the identical ordered sequence of ``(name, parameter)``
pairs. Derive it from ``model.named_parameters()`` (module registration
order, deterministic for identical architectures), filtered to trainable
parameters, materialized ONCE as a list. Never derive the order from a
``set`` or from any dict whose insertion order differs across ranks. When
``world_size > 1`` the constructor exchanges a partition digest (names,
shapes, dtypes, shard assignment) via all-gather and fails closed on any
mismatch, so silent partition drift between ranks is impossible.

Step semantics (must match the unsharded trainers exactly)
----------------------------------------------------------
1. Per-parameter ``all_reduce(ReduceOp.AVG)`` over grads, as in
   `thundersync.grpo.data_parallel`. After this every rank holds the
   identical averaged gradient for EVERY parameter.
2. ``torch.nn.utils.clip_grad_norm_`` over ALL parameters with
   ``foreach=False`` -- because grads are fully replicated, the local
   all-parameter norm IS the global norm; no cross-rank norm reduction is
   needed for correctness. An all-gather agreement gate asserts the norm is
   bitwise identical on every rank and fails closed on replica drift.
3. Rank-local ``optimizer.step()`` on the owned shard only. AdamW is a
   per-parameter map, so the shard-local update is bitwise identical to the
   corresponding slice of an unsharded step given identical grads.
4. Broadcast each shard's updated parameter data from its owner rank, so
   the replicas re-synchronize before the next batch streams.

``world_size == 1`` short-circuits every collective and runs the plain
``clip_grad_norm_`` + ``optimizer.step()`` pair, numerically identical to
the single-GPU trainers.

Fail-closed on: a parameter losing ``requires_grad`` after construction, a
missing gradient (unless ``missing_grad_policy="zero"`` explicitly opts into
zero materialization, the mathematically correct contribution of an
untouched parameter -- see the grad-materialization note in
`thundersync.grpo.data_parallel`), an optimizer
factory that captures parameters outside the owned shard, and cross-rank
partition drift.
"""

from __future__ import annotations

import hashlib
from itertools import groupby
import time
from typing import Callable, Sequence

import torch
import torch.distributed as dist

OptimizerFactory = Callable[[list[torch.nn.Parameter]], torch.optim.Optimizer]

_MISSING_GRAD_POLICIES = ("error", "zero")


def partition_contiguous_by_numel(
    numels: Sequence[int], world_size: int
) -> tuple[tuple[int, int], ...]:
    """Split indices ``[0, len(numels))`` into ``world_size`` contiguous,
    non-empty, numel-balanced half-open spans.

    Deterministic in (numels, world_size) alone: each boundary is placed at
    the cumulative-numel point closest to the ideal ``s/world_size`` fraction
    of the total (ties resolved to the smaller index), subject to every shard
    keeping at least one parameter. Every rank computing this over the
    canonical parameter order derives the identical partition.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if any(n <= 0 for n in numels):
        raise ValueError("every parameter must have positive numel")
    count = len(numels)
    if count < world_size:
        raise ValueError(
            f"cannot partition {count} parameters into {world_size} non-empty shards"
        )
    if world_size == 1:
        return ((0, count),)

    cumulative: list[int] = []
    running = 0
    for numel in numels:
        running += numel
        cumulative.append(running)
    total = running

    bounds = [0]
    for shard in range(1, world_size):
        target = total * shard / world_size
        low = bounds[-1] + 1  # the previous shard keeps >= 1 parameter
        high = count - (world_size - shard)  # each remaining shard keeps >= 1
        best = low
        best_error = abs(cumulative[low - 1] - target)
        for candidate in range(low + 1, high + 1):
            error = abs(cumulative[candidate - 1] - target)
            if error < best_error:
                best = candidate
                best_error = error
        bounds.append(best)
    bounds.append(count)
    return tuple((bounds[s], bounds[s + 1]) for s in range(world_size))


class Collectives:
    """Injectable collective backend: the seam that makes multi-rank math
    testable in one CPU process.

    The default is :class:`TorchDistributedCollectives`; tests inject a fake
    that runs several rank instances over shared buffers. Every method
    mutates its tensor argument in place and must behave like the
    corresponding ``torch.distributed`` collective: identical results on
    every rank, call sequences structurally identical across ranks.
    """

    def all_reduce_average_(self, tensor: torch.Tensor) -> None:
        """In-place average of ``tensor`` across all ranks."""
        raise NotImplementedError

    def broadcast_(self, tensor: torch.Tensor, source_rank: int) -> None:
        """In-place overwrite of ``tensor`` with ``source_rank``'s copy."""
        raise NotImplementedError

    def all_reduce_averages_(self, tensors: Sequence[torch.Tensor]) -> None:
        for tensor in tensors:
            self.all_reduce_average_(tensor)

    def broadcasts_(
        self, tensors: Sequence[torch.Tensor], source_ranks: Sequence[int]
    ) -> None:
        for tensor, source_rank in zip(tensors, source_ranks, strict=True):
            self.broadcast_(tensor, source_rank)

    def all_gather_(
        self, outputs: list[torch.Tensor], tensor: torch.Tensor
    ) -> None:
        """Fill ``outputs[r]`` with rank r's ``tensor`` for every rank."""
        raise NotImplementedError


class TorchDistributedCollectives(Collectives):
    """Default backend over ``torch.distributed`` (optionally a subgroup)."""

    def __init__(self, process_group: object | None = None) -> None:
        if not dist.is_initialized():
            raise RuntimeError(
                "TorchDistributedCollectives requires an initialized process "
                "group; initialize torch.distributed before constructing a "
                "sharded step, or inject a Collectives backend"
            )
        self._group = process_group

    def all_reduce_average_(self, tensor: torch.Tensor) -> None:
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG, group=self._group)

    def broadcast_(self, tensor: torch.Tensor, source_rank: int) -> None:
        dist.broadcast(tensor, src=source_rank, group=self._group)

    def all_reduce_averages_(self, tensors: Sequence[torch.Tensor]) -> None:
        if tensors[0].device.type != "cuda":
            return super().all_reduce_averages_(tensors)
        # Coalesced AVG requires one dtype. Contiguous dtype runs retain the
        # original per-parameter order without packing or allocating tensors.
        for _dtype, batch in groupby(tensors, key=lambda tensor: tensor.dtype):
            batch = tuple(batch)
            if len(batch) == 1:
                self.all_reduce_average_(batch[0])
                continue
            with dist._coalescing_manager(
                group=self._group, async_ops=True
            ) as manager:
                for tensor in batch:
                    dist.all_reduce(
                        tensor, op=dist.ReduceOp.AVG,
                        group=self._group, async_op=True,
                    )
            manager.wait()

    def broadcasts_(
        self, tensors: Sequence[torch.Tensor], source_ranks: Sequence[int]
    ) -> None:
        if tensors[0].device.type != "cuda":
            return super().broadcasts_(tensors, source_ranks)
        with dist._coalescing_manager(
            group=self._group, device=tensors[0].device, async_ops=True
        ) as manager:
            for tensor, source_rank in zip(tensors, source_ranks, strict=True):
                dist.broadcast(
                    tensor, src=source_rank, group=self._group, async_op=True
                )
        manager.wait()

    def all_gather_(
        self, outputs: list[torch.Tensor], tensor: torch.Tensor
    ) -> None:
        dist.all_gather(outputs, tensor, group=self._group)


def _partition_digest(
    names: Sequence[str],
    parameters: Sequence[torch.nn.Parameter],
    spans: Sequence[tuple[int, int]],
) -> int:
    """int64 digest of (shard assignment, name, shape, dtype) per parameter.

    Ranks exchanging equal digests have provably identical canonical orders
    and partitions; any divergence (renamed module, dropped adapter, dtype
    drift) fails closed at construction instead of corrupting the broadcast.
    """
    hasher = hashlib.sha256()
    for shard_index, (start, end) in enumerate(spans):
        for index in range(start, end):
            parameter = parameters[index]
            hasher.update(
                f"{shard_index}|{names[index]}|{tuple(parameter.shape)}|"
                f"{parameter.dtype}\n".encode()
            )
    return int.from_bytes(hasher.digest()[:8], "big", signed=True)


class ShardedOptimizerStep:
    """ZeRO-1 style optimizer step: replicated params/grads, sharded state.

    Constructed with the canonical ordered ``(name, parameter)`` sequence
    (see the module docstring for the ordering contract), an
    ``optimizer_factory`` invoked with ONLY this rank's owned shard, the
    world topology, and the frozen ``grad_clip``. ``step()`` performs the
    four-phase update documented at module level and returns a stats dict.
    """

    def __init__(
        self,
        named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
        optimizer_factory: OptimizerFactory,
        *,
        world_size: int,
        rank: int,
        collectives: Collectives | None = None,
        grad_clip: float | None = None,
        missing_grad_policy: str = "error",
    ) -> None:
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} outside [0, {world_size})")
        if grad_clip is not None and grad_clip <= 0:
            raise ValueError("grad_clip must be positive when enabled")
        if missing_grad_policy not in _MISSING_GRAD_POLICIES:
            raise ValueError(
                f"missing_grad_policy must be one of {_MISSING_GRAD_POLICIES}, "
                f"got {missing_grad_policy!r}"
            )

        pairs = list(named_parameters)
        if not pairs:
            raise ValueError("named_parameters is empty")
        names = [name for name, _ in pairs]
        parameters = [parameter for _, parameter in pairs]
        if len(set(names)) != len(names):
            raise ValueError("duplicate parameter names in canonical order")
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise ValueError(
                "aliased parameter objects in canonical order; tied weights "
                "must appear exactly once (named_parameters() deduplicates)"
            )
        for name, parameter in pairs:
            if not parameter.requires_grad:
                raise ValueError(
                    f"parameter {name!r} has requires_grad=False; pass only "
                    "trainable parameters in the canonical order"
                )
        devices = {parameter.device for parameter in parameters}
        if len(devices) != 1:
            raise ValueError(
                f"parameters span multiple devices {sorted(map(str, devices))}; "
                "one replica per rank must live on one device"
            )

        self._names = tuple(names)
        self._parameters = parameters
        self._world_size = world_size
        self._rank = rank
        self._grad_clip = grad_clip
        self._missing_grad_policy = missing_grad_policy
        self._device = parameters[0].device
        self._steps_done = 0

        self._spans = partition_contiguous_by_numel(
            [parameter.numel() for parameter in parameters], world_size
        )
        start, end = self._spans[rank]
        self._owned_parameters = parameters[start:end]
        self._owned_names = tuple(names[start:end])
        # owner rank per parameter index, for the re-sync broadcast
        self._owner_ranks = [
            shard_index
            for shard_index, (span_start, span_end) in enumerate(self._spans)
            for _ in range(span_start, span_end)
        ]

        self._optimizer = optimizer_factory(list(self._owned_parameters))
        # The memory win is that state exists ONLY for the owned shard; an
        # optimizer capturing anything else silently reintroduces the full
        # replica, so verify the factory honored the contract.
        optimizer_ids = {
            id(parameter)
            for group in self._optimizer.param_groups
            for parameter in group["params"]
        }
        owned_ids = {id(parameter) for parameter in self._owned_parameters}
        if optimizer_ids != owned_ids:
            raise ValueError(
                "optimizer_factory must build an optimizer over exactly the "
                "owned shard; it captured "
                f"{len(optimizer_ids - owned_ids)} foreign and missed "
                f"{len(owned_ids - optimizer_ids)} owned parameters"
            )

        if world_size > 1:
            self._collectives: Collectives | None = (
                collectives
                if collectives is not None
                else TorchDistributedCollectives()
            )
            self._exchange_partition_digest()
        else:
            # world_size == 1 is the plain single-GPU path: zero collectives.
            self._collectives = None

    # ------------------------------------------------------------- inspection

    def retained_device_bytes(self) -> dict[str, int]:
        """Device bytes this shard holds, by attribute: the replicated
        parameters under ``_parameters``, the owned shard's optimizer state
        under ``_optimizer``, and whatever else the step keeps resident, so
        bytes that neither the run nor the backward references have a
        named holder."""

        from thundersync.engine.streaming import retained_bytes_by_attribute

        return retained_bytes_by_attribute(self, self._device)

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def partition_spans(self) -> tuple[tuple[int, int], ...]:
        """Half-open canonical-order index span per rank.

        Public introspection of the partition; the step itself does not
        read it.
        """
        return self._spans

    @property
    def owned_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return tuple(self._owned_parameters)

    @property
    def owned_parameter_names(self) -> tuple[str, ...]:
        """Names of the parameters whose optimizer state this rank owns.

        Public introspection of the partition; the step itself does not
        read it.
        """
        return self._owned_names

    @property
    def optimizer_steps_completed(self) -> int:
        return self._steps_done

    # ------------------------------------------------------------------- step

    def step(self) -> dict[str, object]:
        """Average grads, clip globally, step the owned shard, re-sync.

        Returns a stats dict (this rank's view): ``global_grad_norm``,
        ``gradient_was_clipped``, owned-shard shape/bytes, ``broadcast_bytes``
        (total parameter bytes moved through the re-sync broadcasts), and
        per-phase wall splits ``allreduce_s`` / ``clip_s`` / ``opt_step_s`` /
        ``broadcast_s`` / ``step_total_s``.
        """
        self._validate_step_preconditions()

        self._synchronize_device()
        t0 = time.perf_counter()
        if self._world_size > 1:
            # Per-parameter AVG, as in thundersync.grpo.data_parallel. After
            # this, grads are identical on every rank.
            self._require_collectives().all_reduce_averages_(
                [parameter.grad for parameter in self._parameters]
            )
        self._synchronize_device()
        t1 = time.perf_counter()

        # Grads are fully replicated, so the local all-parameter norm IS the
        # global norm; foreach=False matches the unsharded trainers bitwise.
        global_norm = torch.nn.utils.clip_grad_norm_(
            self._parameters,
            self._grad_clip if self._grad_clip is not None else float("inf"),
            foreach=False,
        )
        if self._world_size > 1:
            self._assert_norm_agreement(global_norm)
        self._synchronize_device()
        t2 = time.perf_counter()

        # Shard-local update: AdamW is a per-parameter map, so this slice of
        # the update is bitwise identical to the unsharded step's.
        self._optimizer.step()
        self._synchronize_device()
        t3 = time.perf_counter()

        broadcast_bytes = 0
        if self._world_size > 1:
            collectives = self._require_collectives()
            with torch.no_grad():
                collectives.broadcasts_(
                    [parameter.data for parameter in self._parameters],
                    self._owner_ranks,
                )
                broadcast_bytes = sum(
                    parameter.numel() * parameter.element_size()
                    for parameter in self._parameters
                )
        for parameter in self._parameters:
            parameter.grad = None
        self._synchronize_device()
        t4 = time.perf_counter()

        global_norm_value = float(global_norm.detach())
        self._steps_done += 1
        owned_numel = sum(
            parameter.numel() for parameter in self._owned_parameters
        )
        return {
            "rank": self._rank,
            "world_size": self._world_size,
            "n_params": len(self._parameters),
            "global_grad_norm": global_norm_value,
            "grad_clip": self._grad_clip,
            "gradient_was_clipped": (
                self._grad_clip is not None
                and global_norm_value > self._grad_clip
            ),
            "owned_param_count": len(self._owned_parameters),
            "owned_param_numel": owned_numel,
            "owned_param_bytes": sum(
                parameter.numel() * parameter.element_size()
                for parameter in self._owned_parameters
            ),
            "broadcast_bytes": broadcast_bytes,
            "allreduce_collectives": (
                len(self._parameters) if self._world_size > 1 else 0
            ),
            "allreduce_s": t1 - t0,
            "clip_s": t2 - t1,
            "opt_step_s": t3 - t2,
            "broadcast_s": t4 - t3,
            "step_total_s": t4 - t0,
        }

    # --------------------------------------------------------------- internal

    def _validate_step_preconditions(self) -> None:
        """Fail closed BEFORE any collective, so a bad rank raises loudly on
        its own instead of deadlocking the world inside a reduce."""
        missing: list[str] = []
        for name, parameter in zip(self._names, self._parameters, strict=True):
            if not parameter.requires_grad:
                raise RuntimeError(
                    f"parameter {name!r} lost requires_grad after construction;"
                    " the partition and optimizer shard are frozen at"
                    " construction time"
                )
            if parameter.grad is None:
                missing.append(name)
        if not missing:
            return
        if self._missing_grad_policy == "error":
            raise RuntimeError(
                f"{len(missing)} parameters have no gradient (first: "
                f"{missing[0]!r}); pass missing_grad_policy='zero' only if a "
                "zero contribution is the declared semantics"
            )
        # "zero": the mathematically correct contribution of an untouched
        # parameter, and required for structurally identical collectives.
        with torch.no_grad():
            for parameter in self._parameters:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)

    def _exchange_partition_digest(self) -> None:
        """One all-gather at construction: every rank must derive the same
        canonical order and partition, or the shard broadcast would silently
        desynchronize the replicas. Fails closed on any mismatch."""
        collectives = self._require_collectives()
        digest = _partition_digest(self._names, self._parameters, self._spans)
        local = torch.tensor([digest], dtype=torch.int64, device=self._device)
        gathered = [torch.zeros_like(local) for _ in range(self._world_size)]
        collectives.all_gather_(gathered, local)
        values = [int(tensor.item()) for tensor in gathered]
        if any(value != digest for value in values):
            raise RuntimeError(
                f"partition drift across ranks: rank {self._rank} digest "
                f"{digest} vs gathered {values}; every rank must construct "
                "ShardedOptimizerStep from the identical canonical "
                "named-parameter order"
            )

    def _assert_norm_agreement(self, global_norm: torch.Tensor) -> None:
        """Replica-agreement gate: after the AVG all-reduce every rank must
        compute the bitwise-identical global norm; drift means the replicas
        (or their grads) diverged, and stepping would compound it."""
        collectives = self._require_collectives()
        if not bool(torch.isfinite(global_norm)):
            raise RuntimeError(
                f"non-finite global grad norm {float(global_norm)}; refusing "
                "the sharded step"
            )
        local = global_norm.detach().reshape(1).clone()
        gathered = [torch.zeros_like(local) for _ in range(self._world_size)]
        collectives.all_gather_(gathered, local)
        for remote_rank, remote in enumerate(gathered):
            if not torch.equal(remote, local):
                raise RuntimeError(
                    "global grad norm disagrees across ranks: rank "
                    f"{self._rank} computed {float(local)} but rank "
                    f"{remote_rank} computed {float(remote)}; replicas have "
                    "diverged"
                )

    def _require_collectives(self) -> Collectives:
        """The cross-rank collectives; a multi-rank step without them raises."""
        if self._collectives is None:
            raise RuntimeError(
                f"world_size={self._world_size} step has no collectives; "
                "construct ShardedOptimizerStep with world_size > 1 to use them"
            )
        return self._collectives

    def _synchronize_device(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
