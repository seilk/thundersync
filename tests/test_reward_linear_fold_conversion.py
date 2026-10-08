"""Bitwise equality of the converted host fold against direct accumulation.

The converted path (window-wise dtype conversion + same-dtype AXPY) must
produce statistics bitwise identical to the direct mixed-dtype fold it
replaces, across: zero-reward G1 elision, mid-sequence reward-statistic
creation, and tensors larger than the conversion window.
"""

from __future__ import annotations

import threading

import torch

from thundersync.grpo.reward_linear import RewardLinearBackward  # noqa: E402


def _bare_backward(window_bytes: int) -> RewardLinearBackward:
    bw = RewardLinearBackward.__new__(RewardLinearBackward)
    bw._accounting_lock = threading.Lock()
    bw._streaming_fold_buffers = {}
    bw._streaming_fold_buffer_reserved_bytes = 0
    bw._source_offload_window_bytes = window_bytes
    bw._current_statistic_bytes = 0
    bw._peak_statistic_bytes = 0
    bw._current_source_bytes = 10**12
    bw._streaming_fold_conversion_bytes = 0
    return bw


def _direct_reference(
    sources: list[torch.Tensor], rewards: list[float]
) -> tuple[torch.Tensor, torch.Tensor | None]:
    unit: torch.Tensor | None = None
    reward_sum: torch.Tensor | None = None
    for value, reward in zip(sources, rewards, strict=True):
        if unit is None:
            unit = value.to(torch.float32)
            reward_sum = None if reward == 0.0 else unit * reward
            continue
        unit.add_(value)
        if reward != 0.0:
            if reward_sum is None:
                reward_sum = value.to(torch.float32).mul_(reward)
            else:
                reward_sum.add_(value, alpha=reward)
    assert unit is not None
    return unit, reward_sum


def _fold_all(
    bw: RewardLinearBackward,
    sources: list[torch.Tensor],
    rewards: list[float],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    unit_sum: list[torch.Tensor | None] = [None]
    reward_sum: list[torch.Tensor | None] = [None]
    for value, reward in zip(sources, rewards, strict=True):
        bw._fold_one_statistic(
            value.clone(),
            index=0,
            reward_sum=reward_sum,
            unit_sum=unit_sum,
            reward=reward,
            keep_source_dtype=False,
        )
    assert unit_sum[0] is not None
    return unit_sum[0], reward_sum[0]


def _case(shape: tuple[int, ...], rewards: list[float], window_bytes: int) -> None:
    generator = torch.Generator().manual_seed(4207)
    sources = [
        torch.randn(shape, generator=generator, dtype=torch.float32).to(
            torch.bfloat16
        )
        for _ in rewards
    ]
    expected_unit, expected_reward = _direct_reference(sources, rewards)
    unit, reward = _fold_all(_bare_backward(window_bytes), sources, rewards)
    assert torch.equal(unit, expected_unit)
    if expected_reward is None:
        assert reward is None
    else:
        assert reward is not None
        assert torch.equal(reward, expected_reward)


def test_converted_fold_matches_direct_fold_bitwise():
    _case((37, 53), rewards=[0.0, 1.25, -0.5, 2.0], window_bytes=256)


def test_zero_reward_elision_and_late_reward_creation_match():
    _case((129,), rewards=[0.0, 0.0, 3.5], window_bytes=64)


def test_single_window_and_multi_window_agree_with_each_other():
    generator = torch.Generator().manual_seed(97)
    sources = [
        torch.randn(511, generator=generator, dtype=torch.float32).to(
            torch.bfloat16
        )
        for _ in range(3)
    ]
    rewards = [1.0, -2.5, 0.75]
    small_unit, small_reward = _fold_all(_bare_backward(32), sources, rewards)
    large_unit, large_reward = _fold_all(_bare_backward(1 << 20), sources, rewards)
    assert torch.equal(small_unit, large_unit)
    assert small_reward is not None and large_reward is not None
    assert torch.equal(small_reward, large_reward)
