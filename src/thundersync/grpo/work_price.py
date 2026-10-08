"""Trajectory work prices and longest-processing-time placement.

A price is a trajectory's training work in token units, from its sizes and
the model's shape only: one unit per turn token for the linear layers, plus
``attention_weight`` per token pair its turns attend to. Balance
rules read it: the claims in ``thundersync.grpo.claims`` use it to stop a thief
before it takes more than it leaves, and placement helpers use it to assign
trajectories to ranks. Nothing here imports torch.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

__all__ = ["attention_weight", "lpt_assign", "trajectory_price"]


def trajectory_price(
    prompt_tokens: int, turn_tokens: int, attention_weight: float
) -> int:
    """Turn tokens, plus the attention its turns pay over the whole sequence."""

    if prompt_tokens < 0 or turn_tokens < 0 or attention_weight < 0:
        raise ValueError("a price takes non-negative sizes")
    total = prompt_tokens + turn_tokens
    pairs = total * total - prompt_tokens * prompt_tokens
    return int(math.ceil(turn_tokens + attention_weight * pairs))


def attention_weight(config: Any, parameter_count: int) -> float:
    """Attention work per token pair over linear work per token, from the
    model's shape: ``layers * attention width / parameters``."""

    layers = int(config.num_hidden_layers)
    heads = int(config.num_attention_heads)
    head_dim = int(getattr(config, "head_dim", None) or config.hidden_size // heads)
    if parameter_count < 1:
        raise ValueError("the model has no parameters")
    return layers * heads * head_dim / float(parameter_count)


def lpt_assign(prices: Sequence[tuple[int, int]], ranks: int) -> dict[int, int]:
    """Longest processing time first: item id -> rank.

    ``prices`` are (item id, price) pairs. Items go in decreasing price
    (ties by item id) to the rank with the least assigned price so far
    (ties by rank), so every rank that reads the same prices computes the
    same placement.
    """

    if ranks < 1:
        raise ValueError("placement needs at least one rank")
    items = [int(item) for item, _ in prices]
    if len(items) != len(set(items)):
        raise ValueError("an item appears twice in the placement")
    loads = [0] * ranks
    placement: dict[int, int] = {}
    for item, price in sorted(prices, key=lambda pair: (-int(pair[1]), int(pair[0]))):
        if price < 0:
            raise ValueError("a price must not be negative")
        rank = min(range(ranks), key=lambda index: (loads[index], index))
        placement[int(item)] = rank
        loads[rank] += int(price)
    return placement
