"""Source admission: when a trajectory's gradient may enter the batch.

Guarantees: content-ready admission hands every legally objective-ready
source to the trainer the moment its content is complete, with no
ordering imposed beyond the residency gates the caller configured, and
the batch still closes on the complete logical batch. It reports what it
held and why it waited.

Requires: a trainer implementing EarlyTrainer, and gates whose in-flight
count comes from the trainer's own pending-reduction accounting rather
than from an admission-side tally.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

__all__ = [
    "ContentReadyAdmission",
    "EarlyTrainer",
    "TrainerCall",
]


# The scheduler carries a close-time coefficient per trajectory without
# reading it: the reward for GRPO, and None for on-policy distillation,
# whose teacher log-probabilities supervise each turn as it arrives, so
# there is nothing left to bind when the trajectory closes.
# Objectives that require a coefficient refuse None rather than defaulting
# it, because a missing reward is a wiring error and a zero reward is a
# legitimate value.
CloseCoefficient = float | None


class TrainerCall(Protocol):
    def __call__(
        self,
        phase: str,
        action: Callable[[], None],
        **metadata: int | list[int],
    ) -> None:
        ...


class EarlyTrainer(Protocol):
    def prepare_trajectories(self, trajectory_ids: list[int]) -> None:
        ...

    def bind_rewards(self, binds: list[tuple[int, CloseCoefficient]]) -> None:
        ...

    def close_trajectories(
        self, closes: list[tuple[int, CloseCoefficient]]
    ) -> None:
        ...

    def close_group(self, group_id: int) -> None:
        ...

    def group_reduction_complete(self, group_id: int) -> bool:
        ...


class ContentReadyAdmission:
    """Early-training admission: prepare at completion, bind at verdict.

    ``add_completion`` batches content-ready branch backwards: completions
    that land while the trainer is busy share one packed rebuild at the
    next flush.
    ``bind_verdict`` attaches the reward scalar immediately -- binding is a
    lock-and-notify, not a backward -- so the reducer starts folding without
    waiting for another event.

    Group closes never block ingestion: a fully bound group is closed only
    when ``group_reduction_complete`` reports the reducer already drained it
    (``close_ready_groups``, non-blocking sweep). ``drain_group_closes`` at
    the batch boundary closes the remainder; any wait there is remaining
    objective work, not an ingestion stall. Queue growth is bounded by the
    registered workload: at most ``registered_trajectories`` completions and
    ``registered_groups`` deferred closes can exist per policy step, and one
    trajectory can enter each stage exactly once (the trainer raises on any
    repeat), so no unbounded accumulation is possible.
    """

    def __init__(
        self,
        trainer: EarlyTrainer,
        *,
        group_size: int,
        trainer_call: TrainerCall,
        max_prepare_pack: int | None = None,
        max_held_sources: int | None = None,
        in_flight_sources: Callable[[], int] | None = None,
        memory_pressure: Callable[[], bool] | None = None,
    ) -> None:
        if max_prepare_pack is not None and max_prepare_pack < 1:
            raise ValueError("max_prepare_pack must be positive when set")
        if max_held_sources is not None and max_held_sources < 1:
            raise ValueError("max_held_sources must be positive when set")
        if max_held_sources is not None and in_flight_sources is None:
            raise ValueError(
                "max_held_sources needs in_flight_sources: the budget must "
                "release on fold completion, not on any admission-side guess"
            )
        self._trainer = trainer
        self._group_size = group_size
        self._trainer_call = trainer_call
        self._max_prepare_pack = max_prepare_pack
        self._max_held_sources = max_held_sources
        self._in_flight_sources = in_flight_sources
        self._memory_pressure = memory_pressure
        self._pending_prepares: list[int] = []
        self._pending_prepare_group_list: list[int] = []
        self._ready_close_queue: list[tuple[int, int, CloseCoefficient]] = []
        self._bound_counts: dict[int, int] = {}
        self._pending_group_closes: list[int] = []

    def add_completion(self, group_id: int, trajectory_id: int) -> None:
        """Record one content-ready trajectory without waiting for another."""

        self._pending_prepares.append(trajectory_id)
        self._pending_prepare_group_list.append(group_id)

    def _admission_open(self, incoming: int, *, gate_memory: bool = True) -> bool:
        """May ``incoming`` more sources be produced right now?

        The budget compares against the trainer's own count of the sources
        it holds (``in_flight_sources``) -- never an admission-side guess, so
        releasing on anything the trainer has not confirmed (a queue
        hand-off) cannot let a verdict burst overrun the container cap. ``memory_pressure`` reads
        the cap's own gauge as a backstop, but only for prepares
        (``gate_memory``): ready closes are what RELEASE memory (their fold
        frees the source, the finished group frees its statistic pair), so
        gating them on the gauge deadlocks exactly when relief is needed.
        """

        if (
            gate_memory
            and self._memory_pressure is not None
            and self._memory_pressure()
        ):
            return False
        if self._max_held_sources is None:
            return True
        if self._in_flight_sources is None:
            raise RuntimeError("max_held_sources is set without in_flight_sources")
        return (
            self._in_flight_sources() + incoming <= self._max_held_sources
        )

    def _prepare_pack(
        self, pack: list[int], group_ids: list[int], **labels: str
    ) -> None:
        self._trainer_call(
            "trajectory_backward",
            lambda: self._trainer.prepare_trajectories(pack),
            trajectory_count=len(pack),
            trajectory_ids=list(pack),
            group_ids=group_ids,
            **labels,
        )

    def flush_prepares(self, *, force: bool = False) -> None:
        """Run the pending content-ready branch backwards as bounded packs.

        Without ``max_prepare_pack`` every pending completion shares one
        packed rebuild. The bound splits the flush into completion-order
        chunks so one pack's checkpoint working set stays inside HBM on
        long-context cells; prepare order (and therefore nothing about
        bind/fold order) changes.

        ``max_held_sources`` is the source-residency budget: a chunk is
        admitted only while differentiated-but-unfolded sources stay under
        it. Deferred completions keep only their (cheap) pending state;
        their backward runs from the ready-close queue once their verdict
        lands, still inside the convoy window the budgeted admissions mask.
        ``force`` ignores the gates for the batch boundary.
        """

        if not self._pending_prepares:
            return
        prepares = self._pending_prepares
        prepare_groups = self._pending_prepare_group_list
        self._pending_prepares = []
        self._pending_prepare_group_list = []
        pack_size = self._max_prepare_pack or len(prepares)
        for start in range(0, len(prepares), pack_size):
            pack = prepares[start : start + pack_size]
            if not force and not self._admission_open(len(pack)):
                # Budget exhausted: everything from here back to pending, in
                # completion order. Later sweeps resume when folds free it.
                self._pending_prepares = prepares[start:]
                self._pending_prepare_group_list = prepare_groups[start:]
                return
            self._prepare_pack(
                pack,
                sorted(set(prepare_groups[start : start + pack_size])),
            )

    def bind_verdict(
        self, group_id: int, trajectory_id: int, reward: CloseCoefficient
    ) -> None:
        """Attach one landed close coefficient; bind order is readiness order.

        The coefficient is carried, never read: this layer decides WHEN a
        source is admitted, and what the objective does with the scalar is
        the objective's. An objective without one passes ``None``.
        """

        if trajectory_id in self._pending_prepares:
            if self._max_held_sources is None:
                # The verdict outran the flush boundary: its completion is
                # still in the prepare batch. Binding requires a prepared
                # source, so flush first; the pack is what had landed.
                self.flush_prepares()
            else:
                # Budget-deferred completion: never a trainer call here.
                # Executing at verdict arrival is an unbounded admission
                # funnel: a verdict burst produces sources faster than folds
                # free them, up to the container's memory cap. The close
                # (backward with the reward known, fold immediately after)
                # drains from the sweep under the same budget instead.
                self._defer_to_ready_close(group_id, trajectory_id, reward)
                return
        self._trainer_call(
            "reward_bind",
            lambda: self._trainer.bind_rewards([(trajectory_id, reward)]),
            trajectory_id=trajectory_id,
            group_id=group_id,
        )
        self._record_bound(group_id)

    def _defer_to_ready_close(
        self, group_id: int, trajectory_id: int, reward: CloseCoefficient
    ) -> None:
        """A pending completion's verdict landed: its backward runs as a close."""

        index = self._pending_prepares.index(trajectory_id)
        del self._pending_prepares[index]
        del self._pending_prepare_group_list[index]
        self._ready_close_queue.append((group_id, trajectory_id, reward))

    def _record_bound(self, group_id: int) -> None:
        bound = self._bound_counts.get(group_id, 0) + 1
        self._bound_counts[group_id] = bound
        if bound == self._group_size:
            self._pending_group_closes.append(group_id)

    def drain_ready_closes(self) -> None:
        """Execute queued verdict-time backwards under the residency gates.

        Each close is prepare+bind with the reward already known, so its
        source folds (and frees) as soon as the reducer reaches it. Chunks
        respect ``max_prepare_pack``; the loop stops the moment the budget
        or the memory gauge says stop and resumes on a later sweep, once
        folds have freed room.
        """

        self._drain_closes(self._ready_close_queue, gated=True)

    def _drain_closes(
        self,
        queue: list[tuple[int, int, CloseCoefficient]],
        *,
        gated: bool,
        **labels: str,
    ) -> None:
        """Run ``queue``'s closes in order, in packs, consuming it.

        ``gated`` stops at the residency budget (never at the memory gauge,
        see `_admission_open`) and leaves the rest queued.
        """

        pack_size = self._max_prepare_pack or max(1, len(queue))
        while queue:
            chunk = queue[:pack_size]
            if gated and not self._admission_open(len(chunk), gate_memory=False):
                return
            del queue[: len(chunk)]
            closes = [
                (trajectory_id, reward) for _, trajectory_id, reward in chunk
            ]
            self._trainer_call(
                "trajectory_backward",
                lambda closes=closes: self._trainer.close_trajectories(
                    closes
                ),
                trajectory_count=len(closes),
                trajectory_ids=[trajectory_id for trajectory_id, _ in closes],
                group_ids=sorted({group_id for group_id, _, _ in chunk}),
                **labels,
            )
            for chunk_group, _, _ in chunk:
                self._record_bound(chunk_group)

    @property
    def ready_close_backlog(self) -> int:
        return len(self._ready_close_queue)

    @property
    def deferred_work_reason(self) -> str | None:
        """Which queue a sweep would act on first, or None if there is none.

        Every queue here was filled by an event that already landed and is
        held back only by a gate that releases on its own (a fold frees the
        budget, a reducer drains a group). The loop reads this to decide
        whether waiting for the next event is safe or is parking work that
        is already runnable.

        The precedence is the sweep's own order: ready closes drain first
        because their folds free the budget the prepares then reuse, then
        prepares, then group closes. Naming the first actionable queue is
        what keeps a recorded stall attributable -- waiting on a group
        close is the reducer still folding, which is a different stall from
        a source-budget block and must not be read as one.
        """

        if self._ready_close_queue:
            return "ready_close"
        if self._pending_prepares:
            return "prepare"
        if self._pending_group_closes:
            return "group_close"
        return None

    def close_ready_groups(self) -> None:
        """Close fully bound groups whose reduction already drained.

        Non-blocking: a group whose reducer is still folding stays deferred,
        so event ingestion never waits on ``close_group``'s reducer barrier.
        """

        still_pending = []
        for group_id in self._pending_group_closes:
            if self._trainer.group_reduction_complete(group_id):
                self._trainer_call(
                    "group_close",
                    lambda gid=group_id: self._trainer.close_group(gid),
                    group_id=group_id,
                )
            else:
                still_pending.append(group_id)
        self._pending_group_closes = still_pending

    def drain_group_closes(self) -> None:
        """Close every remaining fully bound group at the batch boundary."""

        if self._pending_prepares:
            raise RuntimeError(
                "content-ready admission drained with unflushed preparations"
            )
        if self._ready_close_queue:
            raise RuntimeError(
                "content-ready admission drained with queued ready closes"
            )
        for group_id in self._pending_group_closes:
            self._trainer_call(
                "group_close",
                lambda gid=group_id: self._trainer.close_group(gid),
                group_id=group_id,
            )
        self._pending_group_closes = []
