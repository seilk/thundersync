"""Exact-size page-locked host slabs for per-trajectory sources.

A pinned host allocation made through ``torch.empty(..., pin_memory=True)``
reaches the NVIDIA driver's ``os_lock_user_pages``, which pins the whole
range with one ``pin_user_pages(FOLL_LONGTERM)`` call. Kernels whose
call first ``kcalloc``s one pointer per page, physically contiguous, need
2 MiB (order 9) for a 1 GiB range, 128 KiB (order 5) for 64 MiB. Once
pinned, unmovable pages fragment the zone, such a costly-order allocation
fails, the driver reports the pin as an invalid address, and CUDA surfaces
``cudaErrorInvalidValue``. torch's caching host allocator also rounds every
request to the next power of two and never returns a block, so per-copy
pins hold more than the sources' bytes and keep asking the kernel for
costly-order arrays.

A slab is one pageable, 2 MiB-aligned buffer registered with
``cudaHostRegister`` in pieces of at most ``_PIN_PIECE_PAGES`` pages, so no
pin needs more than an order-3 array (the page allocator's last
non-costly order, which it retries instead of failing). A device-to-host
copy may not straddle two registrations, so ``copy_into`` splits each copy
at piece boundaries. Slabs are reused across trajectories and steps; the
arena bounds their total bytes and count and makes an acquirer wait for a
released slab instead of pinning more.

``PinnedStatisticPool`` gives an open group's host statistic pair a reused
home: its Sigma-rg in registered pieces, so the close's reload of the
combined statistic is an asynchronous DMA instead of a pageable staging
copy, and its Sigma-g in a pageable buffer kept across groups, so no group
faults fresh pages in at its first fold or unmaps them at its close.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

# pin_user_pages(FOLL_LONGTERM) allocates 8 bytes per page contiguously;
# 4096 pages keep that array at 32 KiB, PAGE_ALLOC_COSTLY_ORDER (3).
_PIN_PIECE_PAGES = 4096
_SLAB_ALIGN_BYTES = 2 << 20
_TENSOR_ALIGN_BYTES = 512
# Room a slab keeps above its first request: a group with a longer prompt
# carries larger boundary sources.
_SLAB_HEADROOM_DIVISOR = 32
_WAIT_POLL_S = 0.05
_DEFAULT_WAIT_TIMEOUT_S = 1800.0
_CUDA_HOST_REGISTER_DEFAULT = 0


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def _gib(value: float) -> float:
    return round(value / (1 << 30), 3)


def default_piece_bytes() -> int:
    return _PIN_PIECE_PAGES * os.sysconf("SC_PAGE_SIZE")


def tensor_offsets(sizes: list[int]) -> tuple[list[int], int]:
    """Byte offset of each tensor packed into one slab, and the total."""

    offsets = []
    cursor = 0
    for size in sizes:
        cursor = _round_up(cursor, _TENSOR_ALIGN_BYTES)
        offsets.append(cursor)
        cursor += size
    return offsets, cursor


def _aligned_host_buffer(capacity: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A pageable byte buffer of ``capacity`` starting on a 2 MiB boundary."""

    storage = torch.empty(capacity + _SLAB_ALIGN_BYTES, dtype=torch.uint8)
    offset = (-storage.data_ptr()) % _SLAB_ALIGN_BYTES
    return storage, storage[offset : offset + capacity]


def registration_device() -> int | None:
    """The CUDA device current on the calling thread, once CUDA is up.

    An arena or pool takes it at construction -- the rank's device, set on
    the trainer's main thread -- and registers every piece under it.
    """

    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return None
    return torch.cuda.current_device()


def _on_device(device: int | None) -> Any:
    """Make ``device`` current for a registration, whatever the thread.

    A thread starts on device 0. ``cudaHostRegister`` there opens a context
    on device 0 -- another rank's GPU when several ranks share a host -- and
    pins the pages for that context only, not for the rank's.
    """

    if device is None:
        return contextlib.nullcontext()
    return torch.cuda.device(device)


def _register_pieces(
    base: int, capacity: int, piece_bytes: int, device: int | None = None
) -> list[int]:
    """``cudaHostRegister`` a range in pieces on ``device``; all or nothing."""

    cudart = torch.cuda.cudart()
    pieces: list[int] = []
    with _on_device(device):
        try:
            for start in range(0, capacity, piece_bytes):
                size = min(piece_bytes, capacity - start)
                status = cudart.cudaHostRegister(
                    base + start, size, _CUDA_HOST_REGISTER_DEFAULT
                )
                if int(status) != 0:
                    raise RuntimeError(
                        f"cudaHostRegister of a {size}-byte host piece "
                        f"failed with {status}"
                    )
                pieces.append(base + start)
        except BaseException:
            _unregister_pieces(pieces)
            raise
    return pieces


def _unregister_pieces(pieces: list[int], device: int | None = None) -> None:
    cudart = torch.cuda.cudart()
    with _on_device(device):
        for pointer in pieces:
            cudart.cudaHostUnregister(pointer)


def read_mem_available_bytes(path: str = "/proc/meminfo") -> int:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError(f"{path} does not report MemAvailable")


def host_ranks_from_env() -> int:
    """Trainer ranks sharing this host: torchrun's LOCAL_WORLD_SIZE."""

    value = os.environ.get("LOCAL_WORLD_SIZE", "1")
    ranks = int(value)
    if ranks < 1:
        raise ValueError(f"LOCAL_WORLD_SIZE must be positive, got {value}")
    return ranks


def host_pinned_budget(
    *,
    host_ranks: int,
    host_fraction: float,
    reserved_bytes: int,
    available_bytes: int | None = None,
) -> tuple[int, dict[str, Any]]:
    """This rank's pinned-source bytes: its share of the host minus reserve.

    ``host_fraction`` of MemAvailable is split evenly over the ``host_ranks``
    trainer ranks on the host; ``reserved_bytes`` is what the rank must keep
    for the rest of what it holds on the host (its group statistics).
    """

    if host_ranks < 1:
        raise ValueError("host_ranks must be positive")
    if not 0.0 < host_fraction <= 1.0:
        raise ValueError("host_fraction must be in (0, 1]")
    if reserved_bytes < 0:
        raise ValueError("reserved_bytes must not be negative")
    available = (
        read_mem_available_bytes() if available_bytes is None else available_bytes
    )
    share = int(available * host_fraction) // host_ranks
    budget = max(0, share - reserved_bytes)
    evidence = {
        "mem_available_bytes": available,
        "host_fraction": host_fraction,
        "host_ranks": host_ranks,
        "rank_share_bytes": share,
        "reserved_bytes": reserved_bytes,
        "budget_bytes": budget,
    }
    return budget, evidence


@dataclass
class PinnedSlab:
    index: int
    capacity: int
    buffer: torch.Tensor
    storage: torch.Tensor
    pieces: list[int] = field(default_factory=list)
    owner: Any = None
    copy_event: Any = None


class PinnedSourceArena:
    """Bounded, reusable page-locked slabs, one per unfolded source.

    ``budget_bytes`` caps the registered bytes and ``max_slabs`` (optional)
    the slab count. ``acquire`` hands out the smallest free slab that holds
    the request, registers a new one while both caps allow (replacing a
    free slab that is too small if only that fits the budget), and
    otherwise waits for a ``release``. A wait that cannot end -- no busy
    slab belongs to a source whose fold can still run, as the caller's
    ``release_possible`` reports -- raises instead of hanging, as does a
    request larger than the whole budget.
    """

    def __init__(
        self,
        *,
        budget_bytes: int,
        max_slabs: int | None = None,
        piece_bytes: int | None = None,
        wait_timeout_s: float = _DEFAULT_WAIT_TIMEOUT_S,
        evidence: dict[str, Any] | None = None,
        statistics: PinnedStatisticPool | None = None,
    ) -> None:
        page = os.sysconf("SC_PAGE_SIZE")
        piece = default_piece_bytes() if piece_bytes is None else piece_bytes
        if piece < page or piece % page:
            raise ValueError("piece_bytes must be a positive multiple of the page size")
        if piece > default_piece_bytes():
            raise ValueError(
                f"piece_bytes above {_PIN_PIECE_PAGES} pages needs a "
                "costly-order kernel allocation per pin"
            )
        if budget_bytes < 1:
            raise ValueError("budget_bytes must be positive")
        if max_slabs is not None and max_slabs < 1:
            raise ValueError("max_slabs must be positive when set")
        if wait_timeout_s <= 0.0:
            raise ValueError("wait_timeout_s must be positive")
        self._piece_bytes = piece
        self._budget_bytes = budget_bytes
        self._max_slabs = max_slabs
        self._wait_timeout_s = wait_timeout_s
        self._evidence = dict(evidence or {})
        self._condition = threading.Condition()
        self._slabs: list[PinnedSlab] = []
        self._registered_bytes = 0
        self._pending_slabs = 0
        self._next_index = 0
        self._closed = False
        self._acquires = 0
        self._waits = 0
        self._wait_s = 0.0
        self._max_wait_s = 0.0
        self._busy = 0
        self._peak_busy = 0
        self._peak_registered_bytes = 0
        self._register_s = 0.0
        self._registered_pieces = 0
        self._replaced_slabs = 0
        # The rank's reused statistic homes, closed with the arena. Their
        # bytes are the group statistics ``for_host`` already reserves.
        self.statistics = statistics
        # Every slab registers under the device current at construction,
        # whichever thread acquires it.
        self._device = registration_device()

    @classmethod
    def for_host(
        cls,
        source_bytes: int,
        *,
        host_ranks: int,
        host_fraction: float,
        reserved_bytes: int,
        min_slabs: int,
        max_slabs: int | None,
        available_bytes: int | None = None,
        piece_bytes: int | None = None,
        statistics: PinnedStatisticPool | None = None,
    ) -> PinnedSourceArena:
        """Size the arena from the host, failing closed below ``min_slabs``."""

        if source_bytes < 1:
            raise ValueError("source_bytes must be positive")
        if min_slabs < 1:
            raise ValueError("min_slabs must be positive")
        budget, evidence = host_pinned_budget(
            host_ranks=host_ranks,
            host_fraction=host_fraction,
            reserved_bytes=reserved_bytes,
            available_bytes=available_bytes,
        )
        piece = default_piece_bytes() if piece_bytes is None else piece_bytes
        slab_bytes = cls.slab_capacity(source_bytes, piece)
        fits = budget // slab_bytes
        if fits < min_slabs:
            raise RuntimeError(
                f"pinned sources: this rank's host budget holds {fits} "
                f"source slab(s), the configuration needs {min_slabs}. "
                f"MemAvailable {_gib(evidence['mem_available_bytes'])} GiB x "
                f"{host_fraction} / {host_ranks} trainer rank(s) on this host "
                f"= {_gib(evidence['rank_share_bytes'])} GiB, minus "
                f"{_gib(reserved_bytes)} GiB of group statistics = "
                f"{_gib(budget)} GiB; one slab is {_gib(slab_bytes)} GiB. "
                "Free host memory, run fewer ranks on this host, or lower "
                "max_held_sources / prepare_pack_max."
            )
        slabs = fits if max_slabs is None else min(fits, max_slabs)
        evidence = {
            **evidence,
            "source_bytes": source_bytes,
            "slab_bytes": slab_bytes,
            "min_slabs": min_slabs,
            "max_slabs_requested": max_slabs,
            "slabs_fit": fits,
        }
        return cls(
            budget_bytes=budget,
            max_slabs=slabs,
            piece_bytes=piece,
            evidence=evidence,
            statistics=statistics,
        )

    @staticmethod
    def slab_capacity(nbytes: int, piece_bytes: int | None = None) -> int:
        piece = default_piece_bytes() if piece_bytes is None else piece_bytes
        return _round_up(
            max(1, nbytes + nbytes // _SLAB_HEADROOM_DIVISOR), piece
        )

    @property
    def piece_bytes(self) -> int:
        return self._piece_bytes

    def acquire(
        self,
        nbytes: int,
        *,
        owner: Any,
        release_possible: Callable[[], bool],
    ) -> PinnedSlab:
        """A slab of at least ``nbytes`` for ``owner``, waiting if needed.

        ``release_possible`` is called without the arena lock held; it must
        report whether some busy slab can still be released without this
        caller returning (its source's fold can run).
        """

        if nbytes < 1:
            raise ValueError("a pinned slab request must be positive")
        capacity = self.slab_capacity(nbytes, self._piece_bytes)
        started = time.perf_counter()
        waited = False
        while True:
            with self._condition:
                if self._closed:
                    raise RuntimeError("the pinned source arena is closed")
                slab = self._take_free_locked(nbytes, owner)
                if slab is not None:
                    self._count_acquire_locked(started, waited)
                    return slab
                plan = self._reserve_locked(capacity)
                if plan is None:
                    if self._busy == 0 and self._pending_slabs == 0:
                        raise RuntimeError(
                            f"pinned sources: a {_gib(nbytes)} GiB source "
                            f"does not fit this rank's "
                            f"{_gib(self._budget_bytes)} GiB pinned budget "
                            f"({len(self._slabs)} slab(s), "
                            f"{_gib(self._registered_bytes)} GiB registered)"
                        )
                    waited = True
                    self._condition.wait(timeout=_WAIT_POLL_S)
                    if self._progress_locked(nbytes, capacity):
                        continue
                    if time.perf_counter() - started > self._wait_timeout_s:
                        raise RuntimeError(
                            "pinned sources: no slab was released within "
                            f"{self._wait_timeout_s:.0f} s ({self._busy} of "
                            f"{len(self._slabs)} busy)"
                        )
            if plan is not None:
                return self._register(capacity, owner, plan, started, waited)
            if release_possible():
                continue
            with self._condition:
                if self._progress_locked(nbytes, capacity):
                    continue
                busy = self._busy
            raise RuntimeError(
                "pinned sources: every busy slab holds a source that "
                "cannot fold until its reward is bound, and the caller "
                f"waiting for a slab is the one that binds rewards ({busy} "
                f"busy of {self._max_slabs} slab(s), "
                f"{_gib(self._budget_bytes)} GiB budget). Raise the host "
                "budget or keep max_held_sources and prepare_pack_max "
                "at or below the arena's slab count."
            )

    def try_acquire(self, nbytes: int, *, owner: Any) -> PinnedSlab | None:
        """A slab of at least ``nbytes`` for ``owner`` if one is free now.

        As ``acquire``, including registering a new slab while both caps
        allow, but None instead of waiting for a release.
        """

        if nbytes < 1:
            raise ValueError("a pinned slab request must be positive")
        capacity = self.slab_capacity(nbytes, self._piece_bytes)
        started = time.perf_counter()
        with self._condition:
            if self._closed:
                raise RuntimeError("the pinned source arena is closed")
            slab = self._take_free_locked(nbytes, owner)
            if slab is not None:
                self._count_acquire_locked(started, False)
                return slab
            plan = self._reserve_locked(capacity)
            if plan is None:
                return None
        return self._register(capacity, owner, plan, started, False)

    def release(self, slab: PinnedSlab) -> None:
        """Return a slab; any copy still writing into it completes first."""

        event = slab.copy_event
        if event is not None:
            event.synchronize()
        with self._condition:
            if self._closed:
                slab.owner = None
                return
            if slab.owner is None:
                raise RuntimeError("pinned slab released twice")
            slab.owner = None
            slab.copy_event = None
            self._busy -= 1
            self._condition.notify_all()

    def copy_into(
        self,
        slab: PinnedSlab,
        offset: int,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Enqueue ``value``'s device-to-host copy on the current stream.

        Returns the slab view the copy lands in, shaped and typed as
        ``value``. Each ``copy_`` stays inside one registered piece.
        """

        if value.device.type != "cuda":
            raise RuntimeError("pinned slab copies take CUDA sources")
        if not value.is_contiguous():
            raise RuntimeError("pinned slab copies take contiguous sources")
        nbytes = value.numel() * value.element_size()
        if offset < 0 or offset % _TENSOR_ALIGN_BYTES or offset + nbytes > slab.capacity:
            raise RuntimeError("pinned slab copy is outside the slab")
        destination = slab.buffer[offset : offset + nbytes]
        if nbytes:
            source = value.reshape(-1).view(torch.uint8)
            piece = self._piece_bytes
            start = offset
            end = offset + nbytes
            while start < end:
                stop = min(end, (start // piece + 1) * piece)
                destination[start - offset : stop - offset].copy_(
                    source[start - offset : stop - offset], non_blocking=True
                )
                start = stop
        return destination.view(value.dtype).view(value.shape)

    def close(self) -> None:
        """Unregister every slab. Pending copies complete first."""

        with self._condition:
            if self._closed:
                return
            self._closed = True
            slabs = list(self._slabs)
            self._slabs = []
            self._registered_bytes = 0
            self._busy = 0
            self._condition.notify_all()
        for slab in slabs:
            if slab.copy_event is not None:
                slab.copy_event.synchronize()
            self._unregister(slab)
        if self.statistics is not None:
            self.statistics.close()

    @property
    def slab_limit(self) -> int | None:
        """The slab-count cap, or None when only the byte budget bounds it."""

        with self._condition:
            return self._max_slabs

    def extend_slab_limit(self, count: int) -> int:
        """Raise the slab cap by ``count``, failing closed if the host budget
        ``for_host`` measured holds fewer. Slabs still register lazily, at
        the first acquire that needs one. Returns the new cap."""

        if count < 1:
            raise ValueError("the slab cap extension must be positive")
        with self._condition:
            if self._max_slabs is None:
                raise RuntimeError(
                    "the arena has no slab cap to extend: its byte budget "
                    "alone bounds it"
                )
            extended = self._max_slabs + count
            fits = self._evidence.get("slabs_fit")
            if fits is not None and extended > int(fits):
                raise RuntimeError(
                    f"pinned sources: {extended} slabs requested, this "
                    f"rank's host budget holds {fits} "
                    f"({_gib(self._budget_bytes)} GiB); free host memory or "
                    "lower known_reserve_slabs"
                )
            self._max_slabs = extended
            self._evidence["max_slabs_extended_by"] = (
                int(self._evidence.get("max_slabs_extended_by", 0)) + count
            )
            return extended

    def report(self) -> dict[str, Any]:
        with self._condition:
            return {
                "piece_bytes": self._piece_bytes,
                "budget_bytes": self._budget_bytes,
                "max_slabs": self._max_slabs,
                "slabs": len(self._slabs),
                "slab_capacities": [slab.capacity for slab in self._slabs],
                "registered_bytes": self._registered_bytes,
                "peak_registered_bytes": self._peak_registered_bytes,
                "busy_slabs": self._busy,
                "peak_busy_slabs": self._peak_busy,
                "acquires": self._acquires,
                "waits": self._waits,
                "wait_s": self._wait_s,
                "max_wait_s": self._max_wait_s,
                "register_s": self._register_s,
                "registered_pieces": self._registered_pieces,
                "replaced_slabs": self._replaced_slabs,
                "host_budget": dict(self._evidence),
                "statistics": (
                    None if self.statistics is None else self.statistics.report()
                ),
            }

    def _take_free_locked(self, nbytes: int, owner: Any) -> PinnedSlab | None:
        fitting = [
            slab
            for slab in self._slabs
            if slab.owner is None and slab.capacity >= nbytes
        ]
        if not fitting:
            return None
        slab = min(fitting, key=lambda candidate: candidate.capacity)
        slab.owner = owner
        slab.copy_event = None
        return slab

    def _victims_locked(self, capacity: int) -> list[PinnedSlab] | None:
        """Free slabs to replace, largest first, so ``capacity`` fits both caps."""

        victims: list[PinnedSlab] = []
        registered = self._registered_bytes
        count = len(self._slabs) + self._pending_slabs
        free = sorted(
            (slab for slab in self._slabs if slab.owner is None),
            key=lambda slab: slab.capacity,
            reverse=True,
        )
        while not (
            (self._max_slabs is None or count < self._max_slabs)
            and registered + capacity <= self._budget_bytes
        ):
            if not free:
                return None
            victim = free.pop(0)
            victims.append(victim)
            registered -= victim.capacity
            count -= 1
        return victims

    def _reserve_locked(self, capacity: int) -> list[PinnedSlab] | None:
        """Reserve budget for a new slab; returns the free slabs it replaces."""

        victims = self._victims_locked(capacity)
        if victims is None:
            return None
        for victim in victims:
            self._slabs.remove(victim)
            self._registered_bytes -= victim.capacity
        self._registered_bytes += capacity
        self._pending_slabs += 1
        self._peak_registered_bytes = max(
            self._peak_registered_bytes, self._registered_bytes
        )
        return victims

    def _progress_locked(self, nbytes: int, capacity: int) -> bool:
        if any(
            slab.owner is None and slab.capacity >= nbytes for slab in self._slabs
        ):
            return True
        return bool(self._pending_slabs) or (
            self._victims_locked(capacity) is not None
        )

    def _count_acquire_locked(self, started: float, waited: bool) -> None:
        self._acquires += 1
        self._busy += 1
        self._peak_busy = max(self._peak_busy, self._busy)
        if waited:
            elapsed = time.perf_counter() - started
            self._waits += 1
            self._wait_s += elapsed
            self._max_wait_s = max(self._max_wait_s, elapsed)

    def _register(
        self,
        capacity: int,
        owner: Any,
        victims: list[PinnedSlab],
        started: float,
        waited: bool,
    ) -> PinnedSlab:
        try:
            for victim in victims:
                self._unregister(victim)
            slab = self._pin(capacity)
        except BaseException:
            with self._condition:
                self._registered_bytes -= capacity
                self._pending_slabs -= 1
                self._condition.notify_all()
            raise
        with self._condition:
            self._pending_slabs -= 1
            slab.index = self._next_index
            self._next_index += 1
            slab.owner = owner
            self._slabs.append(slab)
            self._replaced_slabs += len(victims)
            self._count_acquire_locked(started, waited)
        return slab

    def _pin(self, capacity: int) -> PinnedSlab:
        started = time.perf_counter()
        storage, buffer = _aligned_host_buffer(capacity)
        pieces = _register_pieces(
            buffer.data_ptr(), capacity, self._piece_bytes, self._device
        )
        with self._condition:
            self._register_s += time.perf_counter() - started
            self._registered_pieces += len(pieces)
        return PinnedSlab(
            index=-1,
            capacity=capacity,
            buffer=buffer,
            storage=storage,
            pieces=pieces,
        )

    def _unregister(self, slab: PinnedSlab) -> None:
        _unregister_pieces(slab.pieces, self._device)
        slab.pieces = []


@dataclass
class StatisticSlot:
    index: int
    unit: torch.Tensor
    reward: torch.Tensor
    storages: tuple[torch.Tensor, torch.Tensor]
    pieces: list[int] = field(default_factory=list)
    owner: Any = None
    reload_event: Any = None


class PinnedStatisticPool:
    """Reused host homes for open groups' parameter statistic pairs.

    One slot holds every trainable parameter's Sigma-g (``unit``, pageable)
    and Sigma-rg (``reward``, registered in pieces) at ``tensor_offsets`` of
    their byte sizes in the statistic dtype. ``acquire`` never waits: with
    every slot owned it returns None and the group allocates its statistics
    as before. A slot is built on its first acquire and kept until
    ``close``; ``release`` takes the event after the last reload copy, which
    the next ``acquire`` of that slot completes before handing it out.
    """

    def __init__(
        self,
        element_counts: list[int],
        dtype: torch.dtype,
        *,
        slots: int,
        piece_bytes: int | None = None,
    ) -> None:
        if slots < 1:
            raise ValueError("a statistic pool needs at least one slot")
        page = os.sysconf("SC_PAGE_SIZE")
        piece = default_piece_bytes() if piece_bytes is None else piece_bytes
        if piece < page or piece % page or piece > default_piece_bytes():
            raise ValueError(
                "piece_bytes must be a page multiple of at most "
                f"{_PIN_PIECE_PAGES} pages"
            )
        if any(count < 0 for count in element_counts):
            raise ValueError("element counts must not be negative")
        self.element_counts = [int(count) for count in element_counts]
        self.dtype = dtype
        element_size = torch.empty((), dtype=dtype).element_size()
        self._sizes = [count * element_size for count in self.element_counts]
        self._offsets, total = tensor_offsets(self._sizes)
        self._capacity = _round_up(max(1, total), piece)
        self._piece_bytes = piece
        self._max_slots = slots
        self._lock = threading.Lock()
        self._slots: list[StatisticSlot] = []
        self._building = 0
        self._next_index = 0
        self._closed = False
        self._acquires = 0
        self._misses = 0
        self._busy = 0
        self._peak_busy = 0
        self._build_s = 0.0
        # Every slot registers under the device current at construction,
        # whichever thread builds it.
        self._device = registration_device()

    @property
    def slot_bytes(self) -> int:
        return 2 * self._capacity

    def acquire(self, owner: Any) -> StatisticSlot | None:
        with self._lock:
            if self._closed:
                raise RuntimeError("the statistic pool is closed")
            slot = next(
                (candidate for candidate in self._slots if candidate.owner is None),
                None,
            )
            if slot is None:
                if len(self._slots) + self._building >= self._max_slots:
                    self._misses += 1
                    return None
                self._building += 1
            else:
                slot.owner = owner
                self._count_locked()
        if slot is None:
            try:
                slot = self._build(owner)
            finally:
                with self._lock:
                    self._building -= 1
            with self._lock:
                self._slots.append(slot)
                self._count_locked()
        event = slot.reload_event
        if event is not None:
            event.synchronize()
            slot.reload_event = None
        return slot

    def _count_locked(self) -> None:
        self._acquires += 1
        self._busy += 1
        self._peak_busy = max(self._peak_busy, self._busy)

    def _build(self, owner: Any) -> StatisticSlot:
        started = time.perf_counter()
        unit_storage, unit = _aligned_host_buffer(self._capacity)
        reward_storage, reward = _aligned_host_buffer(self._capacity)
        pieces = _register_pieces(
            reward.data_ptr(), self._capacity, self._piece_bytes, self._device
        )
        with self._lock:
            index = self._next_index
            self._next_index += 1
            self._build_s += time.perf_counter() - started
        return StatisticSlot(
            index=index,
            unit=unit,
            reward=reward,
            storages=(unit_storage, reward_storage),
            pieces=pieces,
            owner=owner,
        )

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

    def reload_reward(
        self,
        slot: StatisticSlot,
        index: int,
        shape: torch.Size,
        device: torch.device,
    ) -> torch.Tensor:
        """Enqueue parameter ``index``'s Sigma-rg host-to-device on the current stream.

        Each copy stays inside one registered piece, so the transfer is a
        DMA from pinned memory that does not block the calling thread.
        """

        nbytes = self._sizes[index]
        result = torch.empty(shape, dtype=self.dtype, device=device)
        if nbytes:
            destination = result.view(-1).view(torch.uint8)
            offset = self._offsets[index]
            start, end = offset, offset + nbytes
            while start < end:
                stop = min(end, (start // self._piece_bytes + 1) * self._piece_bytes)
                destination[start - offset : stop - offset].copy_(
                    slot.reward[start:stop], non_blocking=True
                )
                start = stop
        return result

    def release(self, slot: StatisticSlot, *, reload_event: Any = None) -> None:
        with self._lock:
            if self._closed:
                slot.owner = None
                return
            if slot.owner is None:
                raise RuntimeError("statistic slot released twice")
            slot.owner = None
            slot.reload_event = reload_event
            self._busy -= 1

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "max_slots": self._max_slots,
                "slots": len(self._slots),
                "slot_bytes": self.slot_bytes,
                "registered_bytes": len(self._slots) * self._capacity,
                "busy_slots": self._busy,
                "peak_busy_slots": self._peak_busy,
                "acquires": self._acquires,
                "misses": self._misses,
                "build_s": self._build_s,
                "dtype": str(self.dtype),
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            slots = list(self._slots)
            self._slots = []
            self._busy = 0
        for slot in slots:
            if slot.reload_event is not None:
                slot.reload_event.synchronize()
            _unregister_pieces(slot.pieces, self._device)
            slot.pieces = []
