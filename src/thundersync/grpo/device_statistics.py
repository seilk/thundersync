"""Device homes for the open-group statistic pairs, and how many fit.

With device-resident sources, the executor can fold each trajectory's gradient into its
group's Sigma-g / Sigma-rg on the device that produced it: an AXPY at device
bandwidth instead of a device-to-host copy and a host fold. The statistics
then hold two model-sized tensors per open group on the device, and every
device memory rule of the step (the close rebuild's plain and block rules
read the allocator before each rebuild) has to see them. Allocated lazily by
a reducer thread at a group's first fold, they appear in the middle of
another branch's rebuild, after its rule has already been read.

``DeviceStatisticPool`` allocates one home pair per open group when it is
built, so the statistics are resident before the first rebuild of the step,
and hands a home to each group that opens. A group that finds every home
owned gets none and folds on the host instead (the executor routes its
sources there). The executor builds the pool at the step's first group
registration, which precedes the step's rebuilds (an executor that never
opens a group never holds the homes), and drops it when the step is safe to
take, so the optimizer step runs with that memory back in the allocator's
cache.

``device_statistic_slots`` is the rule that sizes the pool from the device
and the model when the configuration leaves the choice to it (statistic
device "auto"): the open groups whose
pairs fit beside the parameters, their accumulated gradient, the optimizer
state, the sources the residency gate lets wait on the device, and a reserve
for the step's transients. No device number lives here; each input is read
from the device, derived from the model, or named by the configuration.
"""

from __future__ import annotations

import threading
from typing import Any

import torch

from thundersync.grpo.pinned_arena import StatisticSlot, tensor_offsets


class DeviceStatisticPool:
    """Preallocated device homes for open groups' parameter statistic pairs.

    One slot holds every trainable parameter's Sigma-g (``unit``) and
    Sigma-rg (``reward``) at ``tensor_offsets`` of their byte sizes in the
    statistic dtype, the layout ``PinnedStatisticPool`` gives host homes.
    ``acquire`` never waits: with every slot owned it returns None. Every
    slot is allocated at construction. ``release`` takes an event after the
    group's last read of its home, which the next ``acquire`` of that slot
    completes before handing it out.
    """

    def __init__(
        self,
        element_counts: list[int],
        dtype: torch.dtype,
        *,
        slots: int,
        device: torch.device | str,
    ) -> None:
        if isinstance(slots, bool) or not isinstance(slots, int) or slots < 1:
            raise ValueError("a device statistic pool needs at least one slot")
        resolved = torch.device(device)
        if resolved.type != "cuda":
            raise ValueError("device statistic homes live on a CUDA device")
        if resolved.index is None:
            resolved = torch.device("cuda", torch.cuda.current_device())
        if any(count < 0 for count in element_counts):
            raise ValueError("element counts must not be negative")
        self.element_counts = [int(count) for count in element_counts]
        self.dtype = dtype
        self.device = resolved
        element_size = torch.empty((), dtype=dtype).element_size()
        self._sizes = [count * element_size for count in self.element_counts]
        self._offsets, total = tensor_offsets(self._sizes)
        self._capacity = max(1, total)
        self._lock = threading.Lock()
        self._closed = False
        self._acquires = 0
        self._misses = 0
        self._busy = 0
        self._peak_busy = 0
        self._slots: list[StatisticSlot] = []
        for index in range(slots):
            unit = torch.empty(self._capacity, dtype=torch.uint8, device=resolved)
            reward = torch.empty(self._capacity, dtype=torch.uint8, device=resolved)
            self._slots.append(
                StatisticSlot(
                    index=index,
                    unit=unit,
                    reward=reward,
                    storages=(unit, reward),
                )
            )

    @property
    def slot_bytes(self) -> int:
        return 2 * self._capacity

    @property
    def slots(self) -> int:
        return len(self._slots)

    def acquire(self, owner: Any) -> StatisticSlot | None:
        with self._lock:
            if self._closed:
                raise RuntimeError("the device statistic pool is closed")
            slot = next(
                (candidate for candidate in self._slots if candidate.owner is None),
                None,
            )
            if slot is None:
                self._misses += 1
                return None
            slot.owner = owner
            self._acquires += 1
            self._busy += 1
            self._peak_busy = max(self._peak_busy, self._busy)
            event = slot.reload_event
            slot.reload_event = None
        if event is not None:
            event.synchronize()
        return slot

    def fits(self, index: int, value: torch.Tensor) -> bool:
        """Does parameter ``index``'s home hold a statistic shaped as ``value``?"""

        return (
            0 <= index < len(self.element_counts)
            and value.numel() == self.element_counts[index]
        )

    def view(
        self,
        slot: StatisticSlot,
        index: int,
        shape: torch.Size,
        *,
        statistic: str,
    ) -> torch.Tensor:
        """Parameter ``index``'s ``unit`` or ``reward`` home, typed and shaped."""

        buffer = {"unit": slot.unit, "reward": slot.reward}[statistic]
        start = self._offsets[index]
        return (
            buffer[start : start + self._sizes[index]]
            .view(self.dtype)
            .view(shape)
        )

    def release(self, slot: StatisticSlot, *, reload_event: Any = None) -> None:
        with self._lock:
            if self._closed:
                slot.owner = None
                return
            if slot.owner is None:
                raise RuntimeError("device statistic slot released twice")
            slot.owner = None
            slot.reload_event = reload_event
            self._busy -= 1

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "device": str(self.device),
                "slots": len(self._slots),
                "slot_bytes": self.slot_bytes,
                "allocated_bytes": 0 if self._closed else len(self._slots) * self.slot_bytes,
                "busy_slots": self._busy,
                "peak_busy_slots": self._peak_busy,
                "acquires": self._acquires,
                "misses": self._misses,
                "dtype": str(self.dtype),
                "closed": self._closed,
            }

    def close(self) -> None:
        """Drop every home; the memory returns to the allocator's cache."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            slots = list(self._slots)
            self._busy = 0
        for slot in slots:
            if slot.reload_event is not None:
                slot.reload_event.synchronize()
                slot.reload_event = None
            slot.owner = None
            empty = torch.empty(0, dtype=torch.uint8, device=self.device)
            slot.unit = empty
            slot.reward = empty
            slot.storages = (empty, empty)


def device_statistic_slots(
    *,
    device_total_bytes: int,
    parameter_bytes: int,
    gradient_bytes: int,
    optimizer_state_bytes: int,
    source_bytes: int,
    device_held_sources: int,
    statistic_pair_bytes: int,
    reserve_bytes: int,
    open_groups: int,
) -> int:
    """How many open groups' statistic pairs the device holds, at most all.

    The device keeps, beside the statistics: the parameters, one accumulated
    gradient, the optimizer state this rank owns, ``device_held_sources``
    unfolded sources (what the residency gate admits at once) and
    ``reserve_bytes`` for the step's transients -- the close rebuild's
    activations, the backward's own gradient, allocator slack and the CUDA
    context. What remains holds ``floor(remaining / pair)`` groups'
    statistics; the other open groups fold on the host.
    """

    values = {
        "device_total_bytes": device_total_bytes,
        "parameter_bytes": parameter_bytes,
        "gradient_bytes": gradient_bytes,
        "optimizer_state_bytes": optimizer_state_bytes,
        "source_bytes": source_bytes,
        "device_held_sources": device_held_sources,
        "reserve_bytes": reserve_bytes,
        "open_groups": open_groups,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if isinstance(statistic_pair_bytes, bool) or not isinstance(
        statistic_pair_bytes, int
    ) or statistic_pair_bytes < 1:
        raise ValueError("statistic_pair_bytes must be a positive integer")
    remaining = (
        device_total_bytes
        - parameter_bytes
        - gradient_bytes
        - optimizer_state_bytes
        - device_held_sources * source_bytes
        - reserve_bytes
    )
    if remaining <= 0:
        return 0
    return min(open_groups, remaining // statistic_pair_bytes)
