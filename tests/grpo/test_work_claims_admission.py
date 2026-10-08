"""Work claims (no torch): the store protocol and the claiming admission.

* an offered item runs exactly once: whoever claims it first through the
  store's compare-and-set runs it, the owner skips (cedes) what it lost;
* the owner never waits for a thief, a thief stops when every peer is done,
  and every step's keys are its own;
* a thief takes the largest open offers while what it takes stays at most
  what it leaves the owner, one forest micro-batch per claim;
* two ranks run their steps concurrently (threads over one store): every
  trajectory is differentiated once, on one rank, the idle rank takes part of
  the busy rank's final work, and both reach the step's end;
* the payload carries the advantage's float64 bits.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

from thundersync.grpo.claims import (  # noqa: E402
    ClaimingFinalFirstAdmission,
    decode_final_work,
    encode_final_work,
)
from thundersync.grpo.work_price import attention_weight, lpt_assign, trajectory_price  # noqa: E402
from thundersync.scheduling.work_claims import StoreWorkClaims, WorkClaimError  # noqa: E402


class MemoryStore:
    """A thread-safe store with c10d's compare_set semantics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, bytes] = {}

    @staticmethod
    def _bytes(value) -> bytes:
        return value if isinstance(value, bytes) else str(value).encode()

    def set(self, key, value) -> None:
        with self._lock:
            self._data[key] = self._bytes(value)

    def get(self, key) -> bytes:
        with self._lock:
            return self._data[key]

    def add(self, key, amount) -> int:
        with self._lock:
            value = int(self._data.get(key, b"0")) + int(amount)
            self._data[key] = str(value).encode()
            return value

    def compare_set(self, key, expected, desired) -> bytes:
        expected, desired = self._bytes(expected), self._bytes(desired)
        with self._lock:
            current = self._data.get(key)
            if (current is None and expected == b"") or current == expected:
                self._data[key] = desired
                return desired
            return current if current is not None else expected

    def check(self, keys) -> bool:
        with self._lock:
            return all(key in self._data for key in keys)

    def delete_key(self, key) -> bool:
        with self._lock:
            return self._data.pop(key, None) is not None

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._data)


def _pair(store=None):
    store = store or MemoryStore()
    owner = StoreWorkClaims(store, rank=0, world_size=2)
    thief = StoreWorkClaims(store, rank=1, world_size=2)
    owner.begin_step(1)
    thief.begin_step(1)
    return store, owner, thief


# ---- the store protocol ---------------------------------------------------


def test_an_offer_runs_once_whoever_claims_first() -> None:
    _, owner, thief = _pair()
    for item, price in ((10, 5), (11, 7), (12, 3)):
        owner.offer(item, price, f"payload-{item}".encode())
    assert thief.peer_load(0) == 15
    offers = {offer.item: offer for offer in thief.open_offers(0)}
    assert set(offers) == {10, 11, 12}
    assert thief.steal(offers[11]) == b"payload-11"
    assert thief.peer_load(0) == 8
    won, lost = owner.claim_own([10, 11, 12, 99])
    assert won == [10, 12, 99] and lost == [11]
    # the owner's own win: a later steal of it loses
    assert thief.steal(offers[10]) is None
    owner.finish_own([10, 12])
    assert thief.peer_load(0) == 0
    assert not thief.peer_done(0)
    owner.publish_done()
    assert thief.all_peers_done()
    assert owner.report()["ceded"] == 1 and thief.report()["stolen"] == 1


def test_done_needs_every_offer_resolved_and_steps_do_not_share_keys() -> None:
    store, owner, thief = _pair()
    owner.offer(1, 4, b"x")
    with pytest.raises(WorkClaimError, match="unresolved"):
        owner.publish_done()
    owner.claim_own([1])
    owner.finish_own([1])
    owner.publish_done()
    with pytest.raises(WorkClaimError, match="done offers nothing"):
        owner.offer(2, 1, b"y")
    thief.publish_done()
    owner.begin_step(2)
    thief.begin_step(2)
    assert not thief.peer_done(0)
    assert thief.open_offers(0) == []
    assert thief.peer_load(0) == 0
    # the same item id in a new step is a new claim
    owner.offer(1, 4, b"z")
    assert thief.steal(thief.open_offers(0)[0]) == b"z"
    with pytest.raises(WorkClaimError, match="without publishing done"):
        owner.begin_step(3)


def test_concurrent_claimants_get_exactly_one_winner_per_item() -> None:
    store = MemoryStore()
    ranks = 8
    claims = [StoreWorkClaims(store, rank=rank, world_size=ranks) for rank in range(ranks)]
    for claim in claims:
        claim.begin_step(1)
    items = list(range(400))
    wins: list[list[int]] = [[] for _ in range(ranks)]
    barrier = threading.Barrier(ranks)

    def race(rank: int) -> None:
        barrier.wait()
        for item in items:
            if claims[rank]._try_claim(item):
                wins[rank].append(item)

    threads = [threading.Thread(target=race, args=(rank,)) for rank in range(ranks)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    flat = [item for won in wins for item in won]
    assert sorted(flat) == items


# ---- prices and placement ---------------------------------------------------


def test_prices_count_turn_tokens_and_their_attention() -> None:
    assert trajectory_price(100, 0, 0.5) == 0
    assert trajectory_price(0, 10, 0.0) == 10
    # (p + n)^2 - p^2 pairs at the weight
    assert trajectory_price(10, 10, 0.01) == 10 + 3
    config = type("C", (), {"num_hidden_layers": 4, "num_attention_heads": 2, "hidden_size": 16})()
    assert attention_weight(config, 1024) == 4 * 16 / 1024


def test_lpt_is_deterministic_and_balances() -> None:
    prices = [(item, price) for item, price in enumerate((9, 1, 8, 2, 7, 3, 6, 4, 5, 5))]
    placement = lpt_assign(prices, 2)
    assert placement == lpt_assign(list(reversed(prices)), 2)
    loads = [sum(price for item, price in prices if placement[item] == rank) for rank in (0, 1)]
    assert abs(loads[0] - loads[1]) <= 1
    with pytest.raises(ValueError, match="twice"):
        lpt_assign([(1, 2), (1, 3)], 2)


def test_the_payload_carries_the_advantage_bits() -> None:
    advantage = (0.1 + 0.2) / 3.0
    payload = encode_final_work(
        group_id=3,
        trajectory_id=10000007,
        advantage=advantage,
        prompt_tokens=[1, 2, 3],
        turns=[([4, 5], [True, False]), ([6], [True])],
    )
    group_id, trajectory_id, decoded, prompt, turns = decode_final_work(payload)
    assert (group_id, trajectory_id, prompt) == (3, 10000007, [1, 2, 3])
    assert decoded == advantage and decoded.hex() == advantage.hex()
    assert turns == [([4, 5], [True, False]), ([6], [True])]


# ---- the claiming admission ---------------------------------------------------


class ClaimTrainer:
    """Final work only: every trajectory's group is fully bound."""

    def __init__(self, groups: dict[int, dict[int, int]], *, seconds_per_token: float, cap: int) -> None:
        # group -> trajectory -> tokens
        self.groups = groups
        self.tokens = {item: size for members in groups.values() for item, size in members.items()}
        self.verdicts: set[int] = set()
        self.seconds_per_token = seconds_per_token
        self.cap = cap
        self.ran_own: list[int] = []
        self.ran_claimed: list[int] = []
        self.ceded: list[int] = []
        self.closed_groups: list[int] = []
        self.claims_report: dict | None = None
        self.calls: list[tuple[str, tuple[int, ...]]] = []

    def _work(self, items) -> None:
        time.sleep(self.seconds_per_token * sum(self.tokens.get(item, 0) for item in items))

    def record_rewards(self, binds) -> None:
        self.verdicts.update(item for item, _ in binds)

    def registered_groups_fully_bound(self) -> bool:
        return self.verdicts >= set(self.tokens)

    def close_final_forest(self, closes) -> None:
        items = [item for item, _ in closes]
        assert 0 < sum(self.tokens[item] for item in items) <= self.cap or len(items) == 1
        self.calls.append(("forest", tuple(items)))
        self._work(items)
        self.ran_own.extend(items)

    def close_trajectories(self, closes) -> None:
        items = [item for item, _ in closes]
        self.calls.append(("close", tuple(items)))
        self._work(items)
        self.ran_own.extend(items)

    def prepare_trajectories(self, items) -> None:
        raise AssertionError("final work is never prepared")

    def bind_rewards(self, binds) -> None:
        raise AssertionError("final work is never bound")

    def close_group(self, group_id: int) -> None:
        self.closed_groups.append(group_id)

    def group_reduction_complete(self, group_id: int) -> bool:
        return True

    @property
    def known_slab_capacity(self):
        return None

    @property
    def final_forest_max_tokens(self) -> int:
        return self.cap

    def final_work(self, item: int):
        group_id = next(group for group, members in self.groups.items() if item in members)
        size = self.tokens[item]
        payload = encode_final_work(
            group_id=group_id, trajectory_id=item, advantage=0.5, prompt_tokens=[1],
            turns=[([2] * size, [True] * size)],
        )
        return self.tokens[item], self.tokens[item], payload

    def cede_final_closes(self, closes) -> None:
        self.ceded.extend(item for item, _ in closes)

    def run_claimed_work(self, payloads) -> None:
        decoded = [decode_final_work(payload) for payload in payloads]
        items = [entry[1] for entry in decoded]
        self.calls.append(("claimed", tuple(items)))
        time.sleep(self.seconds_per_token * sum(len(entry[4][0][0]) for entry in decoded))
        self.ran_claimed.extend(items)

    def record_work_claims(self, report) -> None:
        self.claims_report = report


class Recorder:
    def __init__(self) -> None:
        self.intervals: list[dict] = []
        self._lock = threading.Lock()

    def __call__(self, phase, action, **metadata):
        result = action()
        with self._lock:
            self.intervals.append({"phase": phase, **metadata})
        return result


def _run_rank(admission, trainer) -> None:
    """The driver's order: ingest completions and verdicts, sweep, then the
    batch boundary (forced flush, ready-close backlog, group closes)."""

    for group_id, members in trainer.groups.items():
        for item in members:
            admission.add_completion(group_id, item)
    for group_id, members in trainer.groups.items():
        for item in members:
            admission.bind_verdict(group_id, item, float(item % 2))
    admission.drain_ready_closes()
    admission.flush_prepares(force=True)
    while admission.ready_close_backlog:
        admission.drain_ready_closes()
        admission.close_ready_groups()
    admission.drain_group_closes()


def _claiming(trainer, recorder, claims, group_size):
    return ClaimingFinalFirstAdmission(
        trainer,
        group_size=group_size,
        trainer_call=recorder,
        claims=claims,
        final_forest=True,
        max_prepare_pack=2,
        poll_s=0.001,
        timeout_s=30.0,
    )


@pytest.mark.parametrize("steps", [1, 3])
def test_two_ranks_run_every_trajectory_once_and_the_idle_rank_takes_part(steps) -> None:
    store = MemoryStore()
    claims = [StoreWorkClaims(store, rank=rank, world_size=2) for rank in (0, 1)]
    for step in range(steps):
        # rank 0: two groups of eight, sizes 100..800; rank 1: one small group
        busy = ClaimTrainer(
            {0: {step * 1000 + i: 100 * (i % 8 + 1) for i in range(8)},
             2: {step * 1000 + 100 + i: 100 * (i % 8 + 1) for i in range(8)}},
            seconds_per_token=2e-5,
            cap=1000,
        )
        idle = ClaimTrainer({1: {step * 1000 + 500 + i: 50 for i in range(4)}}, seconds_per_token=2e-5, cap=1000)
        recorders = [Recorder(), Recorder()]
        admissions = [
            _claiming(busy, recorders[0], claims[0], 8),
            _claiming(idle, recorders[1], claims[1], 4),
        ]
        errors: list[BaseException] = []

        def run(rank, trainer):
            try:
                _run_rank(admissions[rank], trainer)
            except BaseException as error:  # surfaced below
                errors.append(error)

        threads = [threading.Thread(target=run, args=(0, busy)), threading.Thread(target=run, args=(1, idle))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        assert not any(thread.is_alive() for thread in threads), "a rank did not reach its step's end"
        assert not errors, errors
        ran = busy.ran_own + busy.ran_claimed + idle.ran_own + idle.ran_claimed
        everything = set(busy.tokens) | set(idle.tokens)
        assert sorted(ran) == sorted(everything)
        # the busy rank's work moved, and the owner ceded exactly what moved
        assert idle.ran_claimed and sorted(idle.ran_claimed) == sorted(busy.ceded)
        assert not busy.ran_claimed and not idle.ceded
        # every group of each rank closed once, ceded members included
        assert sorted(busy.closed_groups) == [0, 2] and idle.closed_groups == [1]
        # each claim takes at most half of what is left, so the split stays near even
        moved = sum(busy.tokens[item] for item in idle.ran_claimed)
        assert moved <= 2 * sum(busy.tokens.values()) / 3
        claimed = [entry for entry in recorders[1].intervals if entry.get("job_class") == "claimed"]
        assert claimed and all(entry["owner_ranks"] == [0] for entry in claimed)
        assert busy.claims_report["ceded"] == len(busy.ceded)
        assert idle.claims_report["stolen"] == len(idle.ran_claimed)
        # one claimed batch is within one forest micro-batch's price
        for kind, items in idle.calls:
            if kind == "claimed":
                assert sum(busy.tokens[item] for item in items) <= busy.cap


def test_the_owner_alone_runs_everything_when_no_peer_claims() -> None:
    store = MemoryStore()
    owner_claims = StoreWorkClaims(store, rank=0, world_size=2)
    peer = StoreWorkClaims(store, rank=1, world_size=2)
    peer.begin_step(1)
    peer.publish_done()  # a peer that finished and claims nothing
    trainer = ClaimTrainer({0: {i: 100 for i in range(4)}}, seconds_per_token=0.0, cap=250)
    recorder = Recorder()
    _run_rank(_claiming(trainer, recorder, owner_claims, 4), trainer)
    assert sorted(trainer.ran_own) == [0, 1, 2, 3] and not trainer.ceded
    # one claim, one forest micro-batch at a time, within the cap
    forests = [items for kind, items in trainer.calls if kind == "forest"]
    assert forests == [(0, 1), (2, 3)]
    assert trainer.claims_report["own_claims"] == 4 and trainer.claims_report["offered"] == 4


def test_direct_final_closes_retire_load_before_the_next_group() -> None:
    store = MemoryStore()
    owner = StoreWorkClaims(store, rank=0, world_size=2)
    peer = StoreWorkClaims(store, rank=1, world_size=2)
    peer.begin_step(1)
    trainer = ClaimTrainer(
        {0: {10: 100, 11: 100}, 1: {12: 100, 13: 100}},
        seconds_per_token=0.0,
        cap=400,
    )
    admission = _claiming(trainer, Recorder(), owner, 2)
    for group, members in trainer.groups.items():
        for item in members:
            admission.add_completion(group, item)
    for item in trainer.groups[0]:
        admission.bind_verdict(0, item, float(item % 2))
    # The second group still lacks verdicts, so the first uses direct closes.
    admission.drain_ready_closes()
    assert trainer.calls == [("close", (10, 11))]
    assert peer.peer_load(0) == 0
    for item in trainer.groups[1]:
        admission.bind_verdict(1, item, float(item % 2))
    admission.drain_ready_closes()
    assert trainer.calls[-1] == ("forest", (12, 13))
    assert peer.peer_load(0) == 0


def test_claiming_needs_the_final_forest() -> None:
    trainer = ClaimTrainer({0: {1: 1}}, seconds_per_token=0.0, cap=1)
    with pytest.raises(ValueError, match="final forest"):
        ClaimingFinalFirstAdmission(
            trainer,
            group_size=1,
            trainer_call=Recorder(),
            claims=StoreWorkClaims(MemoryStore(), rank=0, world_size=1),
        )


def test_a_thief_times_out_when_a_peer_never_finishes() -> None:
    store = MemoryStore()
    thief_claims = StoreWorkClaims(store, rank=1, world_size=2)
    trainer = ClaimTrainer({1: {7: 10}}, seconds_per_token=0.0, cap=100)
    admission = ClaimingFinalFirstAdmission(
        trainer,
        group_size=1,
        trainer_call=Recorder(),
        claims=thief_claims,
        final_forest=True,
        poll_s=0.001,
        timeout_s=0.05,
    )
    with pytest.raises(RuntimeError, match="did not finish"):
        _run_rank(admission, trainer)


def test_zero_timeout_disables_the_peer_deadline() -> None:
    store = MemoryStore()
    owner = StoreWorkClaims(store, rank=0, world_size=2)
    peer = StoreWorkClaims(store, rank=1, world_size=2)
    peer.begin_step(1)
    trainer = ClaimTrainer({0: {1: 10}}, seconds_per_token=0.0, cap=100)
    waited = []

    def finish_peer(delay):
        waited.append(delay)
        peer.publish_done()

    admission = ClaimingFinalFirstAdmission(
        trainer,
        group_size=1,
        trainer_call=Recorder(),
        claims=owner,
        final_forest=True,
        timeout_s=0.0,
        sleep=finish_peer,
    )
    _run_rank(admission, trainer)
    assert waited and trainer.ran_own == [1]
    assert trainer.claims_report["idle_polls"] == 1
