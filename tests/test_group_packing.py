"""The batch-ready GRPO objective's micro-batch packing. No torch."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from thundersync.grpo.packing import pack_groups


@dataclass(frozen=True)
class Item:
    group_id: int
    traj_id: int
    n_tokens: int


def _group(group_id: int, lengths: list[int]) -> list[Item]:
    return [Item(group_id, group_id * 100 + index, length) for index, length in enumerate(lengths)]


def _ids(microbatches: list[list[Item]]) -> list[list[int]]:
    return [[item.traj_id for item in microbatch] for microbatch in microbatches]


def test_complete_groups_pack_under_the_budget_in_group_order() -> None:
    items = _group(1, [3, 3]) + _group(0, [2, 2]) + _group(2, [4, 4])
    assert _ids(pack_groups(items, max_tokens_per_microbatch=10)) == [[0, 1, 100, 101], [200, 201]]


def test_an_oversized_group_stays_whole_by_default() -> None:
    items = _group(0, [6, 6, 6]) + _group(1, [1])
    assert _ids(pack_groups(items, max_tokens_per_microbatch=10)) == [[0, 1, 2], [100]]


def test_an_oversized_group_splits_into_runs_that_fit() -> None:
    items = _group(0, [1]) + _group(1, [6, 3, 6, 6]) + _group(2, [2, 2])
    packed = pack_groups(items, max_tokens_per_microbatch=10, split_oversized_groups=True)
    # group 0 flushes before the split; each run of group 1 fits; group 2 packs afresh
    assert _ids(packed) == [[0], [100, 101], [102], [103], [200, 201]]
    assert all(sum(item.n_tokens for item in run) <= 10 for run in packed[1:4])


def test_a_trajectory_over_the_budget_is_its_own_micro_batch() -> None:
    items = _group(0, [4, 25, 4])
    packed = pack_groups(items, max_tokens_per_microbatch=10, split_oversized_groups=True)
    assert _ids(packed) == [[0], [1], [2]]


@pytest.mark.parametrize("split", [False, True])
def test_every_trajectory_lands_once_in_canonical_order(split: bool) -> None:
    items = _group(3, [5, 9, 2]) + _group(1, [7, 7]) + _group(2, [1, 1, 1, 1])
    packed = pack_groups(items, max_tokens_per_microbatch=12, split_oversized_groups=split)
    flat = [item.traj_id for microbatch in packed for item in microbatch]
    assert flat == sorted(item.traj_id for item in items)


def test_the_budget_must_be_positive() -> None:
    with pytest.raises(ValueError):
        pack_groups(_group(0, [1]), max_tokens_per_microbatch=0)
