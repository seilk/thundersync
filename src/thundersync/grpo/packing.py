"""How the batch-ready GRPO objective cuts its trajectories into micro-batches.

The objective is linear in its trajectories: ``sum_i A_i * S_i`` with every
advantage ``A_i`` fixed from the whole batch's rewards before any forward.
A micro-batch boundary therefore changes memory and the order bf16 values
are summed in, never the objective. Two rules place the boundaries:

* complete groups (the default) -- groups in canonical group-ID order, each
  kept whole, packed greedily under a logical-token budget; a group larger
  than the budget is one micro-batch of its own at any size. Each group's
  prompt is shared by all of its trajectories in one forest.
* ``split_oversized_groups`` -- the same, except that a group larger than
  the budget is cut into consecutive runs of its trajectories (trajectory-ID
  order), each within the budget; a single trajectory larger than the
  budget is a micro-batch of its own. Each run carries the group's prompt,
  so the prompt is forwarded once per run instead of once per group. This
  bounds the device memory a long group can take on a smaller device.

Nothing here imports torch; items only need ``group_id``, ``traj_id`` and
``n_tokens``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar


class Packable(Protocol):
    @property
    def group_id(self) -> int: ...

    @property
    def traj_id(self) -> int: ...

    @property
    def n_tokens(self) -> int: ...


T = TypeVar("T", bound=Packable)


def pack_groups(
    trajectories: Sequence[T],
    *,
    max_tokens_per_microbatch: int,
    split_oversized_groups: bool = False,
) -> list[list[T]]:
    """Micro-batches of ``trajectories`` in canonical group, then trajectory, order."""

    if max_tokens_per_microbatch < 1:
        raise ValueError("GRPO microbatch token budget must be positive")
    by_group: dict[int, list[T]] = {}
    for trajectory in trajectories:
        by_group.setdefault(trajectory.group_id, []).append(trajectory)
    microbatches: list[list[T]] = []
    current: list[T] = []
    used = 0
    for group_id, members in sorted(by_group.items()):
        members = sorted(members, key=lambda item: item.traj_id)
        if any(item.group_id != group_id for item in members):
            raise AssertionError("GRPO group pack changed logical membership")
        group_tokens = sum(item.n_tokens for item in members)
        if split_oversized_groups and group_tokens > max_tokens_per_microbatch:
            if current:
                microbatches.append(current)
                current, used = [], 0
            run: list[T] = []
            run_tokens = 0
            for item in members:
                if run and run_tokens + item.n_tokens > max_tokens_per_microbatch:
                    microbatches.append(run)
                    run, run_tokens = [], 0
                run.append(item)
                run_tokens += item.n_tokens
            microbatches.append(run)
            continue
        if current and used + group_tokens > max_tokens_per_microbatch:
            microbatches.append(current)
            current, used = [], 0
        current.extend(members)
        used += group_tokens
    if current:
        microbatches.append(current)
    return microbatches


__all__ = ["Packable", "pack_groups"]
