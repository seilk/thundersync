"""GRPO admission: a trajectory whose group is fully bound runs first.

A runnable trajectory backward has one of three classes when the sweep
picks it:

* ``final``: every reward of its group has landed. A trainer built with
  ``direct_when_group_bound`` runs it on the direct path, ``A_i * S_i``
  straight into the gradients with no source, so the held-source budget
  never gates it.
* ``known``: its own reward landed and its group's have not all: a ready
  close, whose source folds as soon as it lands.
* ``content``: no reward yet: a prepare, whose source waits for the verdict.

The sweep serves final, then known, then content work, each in completion
order. Every class is runnable from the event that made it so and only the
base admission's residency gates hold any of it back, so no runnable work
waits for a later event; against ``ContentReadyAdmission`` the order of
service and the backward's path change, never a release time. Each
backward's interval records its class as ``job_class``.

Two options change how already-runnable work executes, never when it is
released:

* ``final_forest``: once every reward of the rank's step has landed (every
  remaining trajectory is final), the sweep hands the whole final backlog
  to the trainer's ``close_final_forest`` in one call, which runs it as the
  barrier objective's forest micro-batches (``job_class`` final_forest).
  Before that point final work runs one trajectory at a time, as it lands.
* ``known_reserve_slabs``: the trainer's slab arena holds that many slabs
  beyond the content budget, and a known close is gated on the arena -- a
  slab that no source awaiting its reward holds -- instead of on the held
  count, so it never waits behind content work. Content prepares keep the
  held budget and the memory gauge.

With ``events_waiting`` (a non-blocking "has an event landed that the loop
has not ingested?"), the final and known drains stop between packs when it
says so, and that sweep runs no content prepare: the loop ingests the
landed events at once and the next sweep resumes with every class read
afresh -- a known close whose group's last verdict has just landed runs as
final work, a completion whose verdict has landed is no longer prepared as
content, and the final forest sees the step's last verdict as soon as it
lands instead of after the drain that was running.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from thundersync.scheduling.admission import (
    CloseCoefficient,
    ContentReadyAdmission,
    EarlyTrainer,
    TrainerCall,
)

__all__ = ["FinalFirstAdmission", "GroupLedgerTrainer"]


class GroupLedgerTrainer(EarlyTrainer, Protocol):
    def record_rewards(self, binds: list[tuple[int, CloseCoefficient]]) -> None:
        ...

    def registered_groups_fully_bound(self) -> bool:
        ...

    def close_final_forest(
        self, closes: list[tuple[int, CloseCoefficient]]
    ) -> None:
        ...

    @property
    def known_slab_capacity(self) -> int | None:
        ...


class FinalFirstAdmission(ContentReadyAdmission):
    """Content-ready admission that serves final, known, then content work.

    A verdict for a completion still waiting to be prepared enters the
    trainer's reward ledger and makes the completion a close (known, or
    final once its group's last verdict has landed); it never forces the
    other pending completions through a prepare, as the base admission does
    without a held-source budget.
    """

    def __init__(
        self,
        trainer: GroupLedgerTrainer,
        *,
        group_size: int,
        trainer_call: TrainerCall,
        final_forest: bool = False,
        known_reserve_slabs: int = 0,
        events_waiting: Callable[[], bool] | None = None,
        **gates: object,
    ) -> None:
        super().__init__(
            trainer, group_size=group_size, trainer_call=trainer_call, **gates
        )
        if known_reserve_slabs < 0:
            raise ValueError("known_reserve_slabs must not be negative")
        if known_reserve_slabs and self._max_held_sources is None:
            raise ValueError(
                "known_reserve_slabs reserves slabs beside the held-source "
                "budget, which is unset"
            )
        if known_reserve_slabs and trainer.known_slab_capacity is None:
            raise ValueError(
                "known_reserve_slabs gates known closes on the trainer's slab "
                "arena, which it does not have"
            )
        self._ledger_trainer = trainer
        self._verdict_counts: dict[int, int] = {}
        self._final_forest = bool(final_forest)
        self._known_reserve_slabs = int(known_reserve_slabs)
        self._events_waiting = events_waiting
        # This sweep's drain stopped for landed events: its prepares wait
        # for the next sweep, after the events are ingested.
        self._yielded = False

    def bind_verdict(
        self, group_id: int, trajectory_id: int, reward: CloseCoefficient
    ) -> None:
        self._verdict_counts[group_id] = self._verdict_counts.get(group_id, 0) + 1
        if trajectory_id not in self._pending_prepares:
            super().bind_verdict(group_id, trajectory_id, reward)
            return
        self._ledger_trainer.record_rewards([(trajectory_id, reward)])
        self._defer_to_ready_close(group_id, trajectory_id, reward)

    def drain_ready_closes(self) -> None:
        """Final closes at once, ungated; then known closes under the budget.

        With ``final_forest``, a backlog that is all final because every
        reward of the step has landed runs as one forest call instead.
        """

        self._yielded = False
        if self._final_forest_ready():
            backlog = self._ready_close_queue
            self._ready_close_queue = []
            self._run_final_forest(backlog)
            return
        final = [
            item for item in self._ready_close_queue if self._group_final(item[0])
        ]
        if final:
            self._ready_close_queue = [
                item
                for item in self._ready_close_queue
                if not self._group_final(item[0])
            ]
            if self._events_waiting is None:
                self._drain_closes(final, gated=False, job_class="final")
            else:
                rest = self._drain_yielding(final, job_class="final")
                if rest:
                    self._ready_close_queue = rest + self._ready_close_queue
                    return
            if self._ready_close_queue and self._should_yield():
                return
        if self._known_reserve_slabs:
            self._drain_known_on_arena()
            return
        self._drain_closes(self._ready_close_queue, gated=True, job_class="known")

    def _run_final_forest(
        self, backlog: list[tuple[int, int, CloseCoefficient]]
    ) -> None:
        """One forest call over ``backlog``, every member bound after it."""

        closes = [(trajectory_id, reward) for _, trajectory_id, reward in backlog]
        self._trainer_call(
            "trajectory_backward",
            lambda: self._ledger_trainer.close_final_forest(closes),
            trajectory_count=len(closes),
            trajectory_ids=[trajectory_id for trajectory_id, _ in closes],
            group_ids=sorted({group_id for group_id, _, _ in backlog}),
            job_class="final_forest",
        )
        for group_id, _, _ in backlog:
            self._record_bound(group_id)

    def _final_forest_ready(self) -> bool:
        """Is the whole backlog final work with no reward left to land?"""

        return (
            self._final_forest
            and bool(self._ready_close_queue)
            and not self._pending_prepares
            and self._ledger_trainer.registered_groups_fully_bound()
            and all(self._group_final(group_id) for group_id, _, _ in self._ready_close_queue)
        )

    def _drain_known_on_arena(self) -> None:
        """Known closes in packs that slabs no unbound source holds can take.

        Such a slab is free, or holds a source whose reward is bound and so
        folds without this caller; a pack never waits on a content verdict.
        """

        queue = self._ready_close_queue
        pack_size = self._max_prepare_pack or max(1, len(queue))
        while queue:
            capacity = self._ledger_trainer.known_slab_capacity
            if capacity is None:
                raise RuntimeError(
                    "known_reserve_slabs gates known closes on the trainer's "
                    "slab arena, which it no longer reports"
                )
            if capacity < 1:
                return
            chunk = queue[: min(pack_size, capacity)]
            del queue[: len(chunk)]
            self._drain_closes(chunk, gated=False, job_class="known")
            if queue and self._should_yield():
                return

    def _drain_yielding(
        self, queue: list[tuple[int, int, CloseCoefficient]], **labels: str
    ) -> list[tuple[int, int, CloseCoefficient]]:
        """Ungated closes pack by pack until events wait; returns the rest."""

        pack_size = self._max_prepare_pack or max(1, len(queue))
        queue = list(queue)
        while queue:
            chunk = queue[:pack_size]
            del queue[:pack_size]
            self._drain_closes(chunk, gated=False, **labels)
            if queue and self._should_yield():
                return queue
        return []

    def _should_yield(self) -> bool:
        if self._events_waiting is not None and self._events_waiting():
            self._yielded = True
            return True
        return False

    def flush_prepares(self, *, force: bool = False) -> None:
        """Ingest landed events before starting more content work."""

        if self._yielded and not force:
            return
        super().flush_prepares(force=force)

    def _admission_open(self, incoming: int, *, gate_memory: bool = True) -> bool:
        """Recheck landed events at each content pack's residency gate."""

        if gate_memory and (self._yielded or self._should_yield()):
            return False
        return super()._admission_open(incoming, gate_memory=gate_memory)

    def _prepare_pack(
        self, pack: list[int], group_ids: list[int], **labels: str
    ) -> None:
        super()._prepare_pack(pack, group_ids, job_class="content", **labels)

    def _group_final(self, group_id: int) -> bool:
        return self._verdict_counts.get(group_id, 0) == self._group_size
