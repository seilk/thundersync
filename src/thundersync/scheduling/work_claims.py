"""Work claims between ranks through a key-value store with compare-and-set.

A rank (the owner) offers work items it has not started; any rank may claim
an offered item, the owner included, and the store's atomic ``compare_set``
on the item's claim key decides who runs it: the first claimant writes its
rank into the key, every later one reads that rank back and loses. So an
item runs exactly once, whoever asks first, and the owner never waits for a
thief -- it claims each item before it starts it and skips the ones it lost.

Per step and rank the store holds:

* ``offers``: how many items the owner has offered, and per index the item's
  id and price (the owner's own cost estimate in integer units);
* ``payload/<item>``: what a thief needs to run the item, written before the
  item's index entry, so a listed item's payload is always present;
* ``load``: the price of the owner's offered work it has not finished and no
  thief has claimed; the owner adds an item's price when it offers it and
  subtracts it when it finishes the item, a thief subtracts it when it
  claims. Thieves read it to stop before they take more than is left;
* ``done``: set once the owner has no unclaimed offer and will offer no more
  in the step. A thief stops when every peer is done, and only then: an
  owner offers only before it is done, so no offer can appear later.

Claims of one step live under that step's keys, so nothing carries over.
Row-agnostic: an item is an integer id, a price and opaque bytes; what the
work is and how a thief runs it belongs to the caller.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

__all__ = ["ClaimStore", "Offer", "StoreWorkClaims", "WorkClaimError"]


class WorkClaimError(RuntimeError):
    pass


class ClaimStore(Protocol):
    """The subset of ``torch.distributed.Store`` the claims use."""

    def set(self, key: str, value: bytes | str) -> None: ...

    def get(self, key: str) -> bytes: ...

    def add(self, key: str, amount: int) -> int: ...

    def compare_set(
        self, key: str, expected_value: bytes | str, desired_value: bytes | str
    ) -> bytes: ...

    def check(self, keys: list[str]) -> bool: ...

    def delete_key(self, key: str) -> bool: ...


@dataclass(frozen=True)
class Offer:
    owner: int
    item: int
    price: int


class StoreWorkClaims:
    """One rank's view of the claims of the current step."""

    def __init__(
        self,
        store: ClaimStore,
        *,
        rank: int,
        world_size: int,
        namespace: str = "work-claims",
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is not in a world of {world_size}")
        if not namespace or "/" in namespace:
            raise ValueError("the namespace must be one nonempty key segment")
        self._store = store
        self.rank = int(rank)
        self.world_size = int(world_size)
        self._namespace = namespace
        self._clock = clock
        self._step: int | None = None
        self._offered: dict[int, int] = {}
        self._offer_count = 0
        self._own_claims: set[int] = set()
        self._lost: set[int] = set()
        self._done = False
        # Per peer: how many of its index entries this rank has read, and
        # the offers among them it has not seen resolved.
        self._read: dict[int, int] = {}
        self._open: dict[int, dict[int, Offer]] = {}
        self._counters: dict[str, float] = {}

    # ------------------------------------------------------------ keys

    def _key(self, *parts: object) -> str:
        if self._step is None:
            raise WorkClaimError("no step has begun")
        return "/".join((self._namespace, str(self._step), *map(str, parts)))

    def _claim_key(self, item: int) -> str:
        return self._key("claim", item)

    # ------------------------------------------------------------ step

    def begin_step(self, step: int) -> None:
        """Start a step's claims; the previous step must have ended done."""

        if self._step is not None and not self._done:
            raise WorkClaimError(f"step {self._step} ended without publishing done")
        self._step = int(step)
        self._offered = {}
        self._offer_count = 0
        self._own_claims = set()
        self._lost = set()
        self._done = False
        self._read = {peer: 0 for peer in self.peers}
        self._open = {peer: {} for peer in self.peers}
        self._counters = dict.fromkeys(
            (
                "offered",
                "offered_price",
                "own_claims",
                "ceded",
                "ceded_price",
                "stolen",
                "stolen_price",
                "claim_attempts",
                "claim_s",
                "payload_bytes_out",
                "payload_bytes_in",
            ),
            0,
        )

    @property
    def step(self) -> int | None:
        return self._step

    @property
    def peers(self) -> list[int]:
        return [rank for rank in range(self.world_size) if rank != self.rank]

    @property
    def done(self) -> bool:
        return self._done

    # ------------------------------------------------------------ owner

    def offer(self, item: int, price: int, payload: bytes) -> None:
        """Make an unstarted item claimable; its payload goes first."""

        if self._done:
            raise WorkClaimError("an owner that is done offers nothing more")
        if item in self._offered:
            raise WorkClaimError(f"item {item} is already offered")
        if price < 0:
            raise ValueError("an offer's price must not be negative")
        self._store.set(self._key(self.rank, "payload", item), payload)
        self._store.set(
            self._key(self.rank, "offer", self._offer_count),
            json.dumps({"item": int(item), "price": int(price)}),
        )
        self._offer_count += 1
        self._store.add(self._key(self.rank, "offers"), 1)
        self._store.add(self._key(self.rank, "load"), int(price))
        self._offered[item] = int(price)
        self._counters["offered"] += 1
        self._counters["offered_price"] += int(price)
        self._counters["payload_bytes_out"] += len(payload)

    def is_offered(self, item: int) -> bool:
        return item in self._offered

    def claim_own(self, items: Sequence[int]) -> tuple[list[int], list[int]]:
        """Claim offered items before starting them: (won, lost to a peer).

        An item this rank never offered is its own without a claim.
        """

        won: list[int] = []
        lost: list[int] = []
        for item in items:
            if item not in self._offered:
                won.append(item)
                continue
            if item in self._own_claims:
                raise WorkClaimError(f"item {item} is already claimed by its owner")
            if self._try_claim(item):
                self._own_claims.add(item)
                self._counters["own_claims"] += 1
                won.append(item)
                self._store.delete_key(self._key(self.rank, "payload", item))
            else:
                self._lost.add(item)
                self._counters["ceded"] += 1
                self._counters["ceded_price"] += self._offered[item]
                lost.append(item)
        return won, lost

    def finish_own(self, items: Sequence[int]) -> None:
        """The owner finished these items (claimed by it): their price leaves
        the load."""

        price = 0
        for item in items:
            if item not in self._offered:
                continue
            if item not in self._own_claims:
                raise WorkClaimError(f"item {item} was not claimed by its owner")
            price += self._offered[item]
        if price:
            self._store.add(self._key(self.rank, "load"), -price)

    def publish_done(self) -> None:
        """No unclaimed offer is left and none will follow this step."""

        unresolved = set(self._offered) - self._own_claims - self._lost
        if unresolved:
            raise WorkClaimError(
                f"items {sorted(unresolved)} are offered and unresolved; claim "
                "them before publishing done"
            )
        self._store.set(self._key(self.rank, "done"), "1")
        self._done = True

    # ------------------------------------------------------------ thief

    def peer_done(self, peer: int) -> bool:
        return bool(self._store.check([self._key(peer, "done")]))

    def all_peers_done(self) -> bool:
        return all(self.peer_done(peer) for peer in self.peers)

    def peer_load(self, peer: int) -> int:
        return int(self._store.add(self._key(peer, "load"), 0))

    def open_offers(self, peer: int) -> list[Offer]:
        """The peer's unclaimed offers, oldest first; later claims still race."""

        listed = int(self._store.add(self._key(peer, "offers"), 0))
        open_offers = self._open[peer]
        for index in range(self._read[peer], listed):
            entry = json.loads(self._store.get(self._key(peer, "offer", index)))
            offer = Offer(owner=peer, item=int(entry["item"]), price=int(entry["price"]))
            open_offers[offer.item] = offer
        self._read[peer] = listed
        # Claimed work remains in the owner's load until it finishes, but
        # cannot consume the budget of a batch this rank may still steal.
        for item in list(open_offers):
            if self._store.check([self._claim_key(item)]):
                del open_offers[item]
        return list(open_offers.values())

    def steal(self, offer: Offer) -> bytes | None:
        """Claim a peer's offer; its payload when won, None when lost."""

        self._open[offer.owner].pop(offer.item, None)
        if not self._try_claim(offer.item):
            return None
        self._store.add(self._key(offer.owner, "load"), -offer.price)
        payload_key = self._key(offer.owner, "payload", offer.item)
        payload = self._store.get(payload_key)
        self._store.delete_key(payload_key)
        self._counters["stolen"] += 1
        self._counters["stolen_price"] += offer.price
        self._counters["payload_bytes_in"] += len(payload)
        return payload

    # ------------------------------------------------------------ shared

    def _try_claim(self, item: int) -> bool:
        started = self._clock()
        token = str(self.rank).encode()
        holder = self._store.compare_set(self._claim_key(item), "", token)
        self._counters["claim_attempts"] += 1
        self._counters["claim_s"] += self._clock() - started
        if isinstance(holder, str):
            holder = holder.encode()
        return bytes(holder) == token

    def report(self) -> dict[str, Any]:
        return {
            "step": self._step,
            "rank": self.rank,
            "world_size": self.world_size,
            "done": self._done,
            **self._counters,
        }
