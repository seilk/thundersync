"""Trajectory balance: an idle rank claims a peer's final work.

Groups stay where their worker placed them. A trajectory whose group has
every reward bound (``final``) has a known advantage ``A_i``, so what is left
of it is ``A_i * S_i``, which needs nothing of its group but the scalar. Such
a trajectory, while not started, is offered through the store
(``thundersync.scheduling.work_claims``) with what a thief needs to run it: the
group's prompt tokens, the trajectory's turns and their score masks, and
``A_i`` as the owner's float64 bits. Whoever claims it first runs it.

* The owner claims each final trajectory right before it starts it, one
  pack (direct path) or one forest micro-batch (final forest) at a time, in
  its queue order, and cedes the ones a peer won: they join their group as
  final-forest members (no boundary adjoint), unrun here.
* An idle rank claims one batch of peers' open final work after ingesting
  landed events and draining its own work and ready group closes. It need
  not wait for its own final verdict. It takes the largest prices first,
  while what it takes stays at most what it leaves the owner (the owner's
  unfinished offered price, its running micro-batch included). At its own
  step boundary it publishes done and continues claiming until every peer
  is done. Every claimed batch uses the final-forest executor.

Exactness: each trajectory is differentiated once, on one rank, with its
owner's ``A_i``; the thief's forest carries its prompt term, so the owner's
group close leaves it out, as for any final-forest member. Group counts per
rank are unchanged, so every rank applies the same gradient scale and the
all-reduce sums the parts into the update. The claim never waits for or
changes a release: only work already runnable on the owner moves, and the
owner never waits for a thief.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from thundersync.grpo.admission import FinalFirstAdmission, GroupLedgerTrainer
from thundersync.scheduling.admission import CloseCoefficient, TrainerCall
from thundersync.scheduling.work_claims import Offer, StoreWorkClaims

__all__ = [
    "ClaimTrainer",
    "ClaimingFinalFirstAdmission",
    "decode_final_work",
    "encode_final_work",
]

PAYLOAD_SCHEMA = "thundersync-final-work-v1"


def encode_final_work(
    *,
    group_id: int,
    trajectory_id: int,
    advantage: float,
    prompt_tokens: Sequence[int],
    turns: Sequence[tuple[Sequence[int], Sequence[bool]]],
) -> bytes:
    """One final trajectory's payload; ``advantage`` travels as float64 bits."""

    return json.dumps(
        {
            "schema": PAYLOAD_SCHEMA,
            "group_id": int(group_id),
            "trajectory_id": int(trajectory_id),
            "advantage": float(advantage).hex(),
            "prompt_tokens": [int(token) for token in prompt_tokens],
            "turns": [
                [[int(token) for token in tokens], "".join("1" if s else "0" for s in scored)]
                for tokens, scored in turns
            ],
        },
        separators=(",", ":"),
    ).encode()


def decode_final_work(
    payload: bytes,
) -> tuple[int, int, float, list[int], list[tuple[list[int], list[bool]]]]:
    """(group id, trajectory id, advantage, prompt tokens, turns)."""

    record = json.loads(payload)
    if record.get("schema") != PAYLOAD_SCHEMA:
        raise ValueError(f"final work payload schema {record.get('schema')!r} differs")
    turns = [
        ([int(token) for token in tokens], [flag == "1" for flag in scored])
        for tokens, scored in record["turns"]
    ]
    return (
        int(record["group_id"]),
        int(record["trajectory_id"]),
        float.fromhex(record["advantage"]),
        [int(token) for token in record["prompt_tokens"]],
        turns,
    )


class ClaimTrainer(GroupLedgerTrainer, Protocol):
    def set_statistics_branch_observer(
        self, observer: Callable[[int], None] | None
    ) -> None:
        ...

    def final_work(self, trajectory_id: int) -> tuple[int, int, bytes]:
        """(tokens, price, payload) of a final trajectory not yet started."""
        ...

    def cede_final_closes(self, closes: list[tuple[int, CloseCoefficient]]) -> None:
        ...

    def run_claimed_work(self, payloads: list[bytes]) -> None:
        ...

    def record_work_claims(self, report: dict[str, Any]) -> None:
        ...

    @property
    def final_forest_max_tokens(self) -> int:
        ...


class ClaimingFinalFirstAdmission(FinalFirstAdmission):
    """Final-first admission whose final work any rank may claim.

    Needs the final forest: the owner's forest micro-batches are the unit it
    claims post-rollout, and a thief runs claimed work through the same
    executor. ``claims`` is the rank's store view; the admission begins its
    next step. ``poll_s`` is the idle thief's wait between looks at the
    store; ``timeout_s`` bounds the wait for peers to finish their step
    (zero disables the deadline, as for the rollout progress timeout).
    """

    def __init__(
        self,
        trainer: ClaimTrainer,
        *,
        group_size: int,
        trainer_call: TrainerCall,
        claims: StoreWorkClaims,
        poll_s: float = 0.02,
        timeout_s: float = 3600.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        received_verdicts: Callable[[], list[dict[str, Any]]] | None = None,
        **options: Any,
    ) -> None:
        if not options.get("final_forest"):
            raise ValueError("claiming final work needs the final forest")
        super().__init__(
            trainer, group_size=group_size, trainer_call=trainer_call, **options
        )
        if poll_s <= 0 or timeout_s < 0:
            raise ValueError("poll_s must be positive and timeout_s nonnegative")
        self._claim_trainer = trainer
        self._claims = claims
        self._poll_s = float(poll_s)
        self._timeout_s = float(timeout_s)
        self._sleep = sleep
        self._clock = clock
        # trajectory id -> (tokens, price) of this rank's offered final work
        self._sizes: dict[int, tuple[int, int]] = {}
        self._idle_polls = 0
        self._idle_s = 0.0
        self._received_verdicts = received_verdicts
        self._early_verdicts: dict[tuple[int, int], float] = {}
        self._unbound_groups: dict[int, int] = {}
        self._prepared_unbound: set[int] = set()
        claims.begin_step((claims.step or 0) + 1)
        if received_verdicts is not None:
            trainer.set_statistics_branch_observer(self._bind_pending_verdicts)

    def add_completion(self, group_id: int, trajectory_id: int) -> None:
        super().add_completion(group_id, trajectory_id)
        self._unbound_groups[trajectory_id] = group_id

    def _prepare_pack(
        self, pack: list[int], group_ids: list[int], **labels: str
    ) -> None:
        # The trainer marks the entire admitted pack prepared before its
        # first branch observer. Later, unadmitted packs are not eligible.
        self._prepared_unbound.update(pack)
        super()._prepare_pack(pack, group_ids, **labels)

    def bind_verdict(
        self, group_id: int, trajectory_id: int, reward: CloseCoefficient
    ) -> None:
        key = (group_id, trajectory_id)
        if key in self._early_verdicts:
            if reward != self._early_verdicts[key]:
                raise ValueError("a received reward changed before event ingestion")
            del self._early_verdicts[key]
            return
        super().bind_verdict(group_id, trajectory_id, reward)
        self._unbound_groups.pop(trajectory_id, None)
        self._prepared_unbound.discard(trajectory_id)

    def _bind_pending_verdicts(self, after_trajectory_id: int) -> None:
        """Bind final groups' pending verdicts between statistic branches.

        This runs on the training thread, outside autograd, and never starts
        a backward. Prepared sources may bind rewards here, including those
        in the current pack. Only completed, unstarted work is offered.
        Canonical ingestion later checks the reward without binding it twice.
        """

        if self._received_verdicts is None:
            raise RuntimeError(
                "binding pending verdicts needs a received_verdicts source"
            )
        changed = False
        bind = super().bind_verdict
        pending: dict[tuple[int, int], dict[str, Any]] = {}
        for event in self._received_verdicts():
            if event["policy_version"] != self._claims.step - 1:
                continue
            trajectory_id = int(event["trajectory_id"])
            if (trajectory_id not in self._pending_prepares
                    and trajectory_id not in self._prepared_unbound):
                continue
            group_id = int(event["group_id"])
            if self._unbound_groups[trajectory_id] != group_id:
                raise ValueError("received verdict group differs from completed work")
            pending[group_id, trajectory_id] = event
        counts = Counter(group_id for group_id, _ in pending)
        final_groups = {
            group_id for group_id, count in counts.items()
            if self._verdict_counts.get(group_id, 0) + count == self._group_size
        }
        for (group_id, trajectory_id), event in pending.items():
            if group_id not in final_groups:
                continue
            reward = float(event["training_reward"])
            self._trainer_call(
                "reward_bind",
                lambda gid=group_id, tid=trajectory_id, value=reward: bind(
                    gid, tid, value
                ),
                trajectory_id=trajectory_id,
                group_id=group_id,
                early=True,
                after_trajectory_id=after_trajectory_id,
                trainer_received_at=event.get("_trainer_received_at"),
            )
            self._early_verdicts[group_id, trajectory_id] = reward
            self._unbound_groups.pop(trajectory_id)
            self._prepared_unbound.discard(trajectory_id)
            changed = True
        if changed:
            self._offer_final_work()

    # ------------------------------------------------------------ owner

    def drain_ready_closes(self) -> None:
        self._offer_final_work()
        super().drain_ready_closes()

    @property
    def deferred_work_reason(self) -> str | None:
        reason = super().deferred_work_reason
        if reason is not None:
            return reason
        return None if self._claims.all_peers_done() else "peer_work"

    def close_ready_groups(self) -> None:
        """Serve local work and landed events before one non-blocking claim."""

        super().close_ready_groups()
        if super().deferred_work_reason is None and not self._should_yield():
            self._claim_peer_batch()

    def _offer_final_work(self) -> None:
        """Offer every final close not yet offered, in queue order."""

        for group_id, trajectory_id, _reward in self._ready_close_queue:
            if not self._group_final(group_id) or self._claims.is_offered(trajectory_id):
                continue
            tokens, price, payload = self._claim_trainer.final_work(trajectory_id)
            self._sizes[trajectory_id] = (tokens, price)
            self._claims.offer(trajectory_id, price, payload)

    def _claim(
        self, chunk: list[tuple[int, int, CloseCoefficient]]
    ) -> list[tuple[int, int, CloseCoefficient]]:
        """Claim a chunk before starting it; cede what a peer won."""

        won_ids, lost_ids = self._claims.claim_own([item[1] for item in chunk])
        lost = set(lost_ids)
        if lost:
            ceded = [item for item in chunk if item[1] in lost]
            self._claim_trainer.cede_final_closes(
                [(trajectory_id, reward) for _, trajectory_id, reward in ceded]
            )
            for group_id, _, _ in ceded:
                self._record_bound(group_id)
        won = set(won_ids)
        return [item for item in chunk if item[1] in won]

    def _run_final_forest(
        self, backlog: list[tuple[int, int, CloseCoefficient]]
    ) -> None:
        """The forest backlog one micro-batch at a time, each claimed first."""

        cap = self._claim_trainer.final_forest_max_tokens
        queue = list(backlog)
        while queue:
            chunk, tokens = [], 0
            while queue:
                size = self._sizes.get(queue[0][1], (0, 0))[0]
                if chunk and tokens + size > cap:
                    break
                chunk.append(queue.pop(0))
                tokens += size
            won = self._claim(chunk)
            if won:
                super()._run_final_forest(won)
                self._claims.finish_own([item[1] for item in won])
            if queue and self._should_yield():
                self._ready_close_queue = queue + self._ready_close_queue
                return

    def _drain_closes(
        self,
        queue: list[tuple[int, int, CloseCoefficient]],
        *,
        gated: bool,
        **labels: str,
    ) -> None:
        if labels.get("job_class") != "final":
            super()._drain_closes(queue, gated=gated, **labels)
            return
        pack_size = self._max_prepare_pack or max(1, len(queue))
        while queue:
            chunk = queue[:pack_size]
            del queue[: len(chunk)]
            won = self._claim(chunk)
            if won:
                completed_ids = [item[1] for item in won]
                super()._drain_closes(won, gated=gated, **labels)
                self._claims.finish_own(completed_ids)

    # ------------------------------------------------------------ step end

    def drain_group_closes(self) -> None:
        """Close this rank's groups, publish done, then claim peers' work."""

        super().drain_group_closes()
        self._claims.publish_done()
        self._claim_peer_work()
        self._claim_trainer.record_work_claims(
            {
                **self._claims.report(),
                "idle_polls": self._idle_polls,
                "idle_s": self._idle_s,
            }
        )

    def _claim_peer_work(self) -> None:
        deadline = self._clock() + self._timeout_s if self._timeout_s else None
        while not self._claims.all_peers_done():
            if not self._claim_peer_batch():
                if deadline is not None and self._clock() > deadline:
                    raise RuntimeError(
                        "work claims: peers did not finish their step within "
                        f"{self._timeout_s:.0f} s"
                    )
                self._idle_polls += 1
                started = self._clock()
                self._sleep(self._poll_s)
                self._idle_s += self._clock() - started

    def _claim_peer_batch(self) -> bool:
        """Attempt one available forest batch without waiting for a peer."""

        batch = self._steal_batch()
        if not batch:
            return False
        payloads, owners = [], []
        for offer in batch:
            payload = self._claims.steal(offer)
            if payload is not None:
                payloads.append(payload)
                owners.append(offer)
        if payloads:
            self._trainer_call(
                "trajectory_backward",
                lambda payloads=payloads: self._claim_trainer.run_claimed_work(payloads),
                trajectory_count=len(owners),
                trajectory_ids=[offer.item for offer in owners],
                owner_ranks=sorted({offer.owner for offer in owners}),
                job_class="claimed",
            )
        return True

    def _steal_batch(self) -> list[Offer]:
        """The offers to claim next from the peer with the most work left.

        Largest price first; an offer is taken only while what this rank
        takes stays at most what the owner keeps, and the batch stays within
        one forest micro-batch's price.
        """

        best: tuple[int, int] | None = None
        for peer in self._claims.peers:
            if self._claims.peer_done(peer):
                continue
            load = self._claims.peer_load(peer)
            if best is None or load > best[1]:
                best = (peer, load)
        if best is None or best[1] <= 0:
            return []
        peer, load = best
        cap = self._claim_trainer.final_forest_max_tokens
        taken, batch = 0, []
        for offer in sorted(
            self._claims.open_offers(peer), key=lambda item: (-item.price, item.item)
        ):
            if batch and taken + offer.price > cap:
                continue
            if 2 * (taken + offer.price) > load:
                continue
            batch.append(offer)
            taken += offer.price
        return batch
