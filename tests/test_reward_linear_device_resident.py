"""Device-resident source gates: fold H_i where it was produced.

``device_resident`` keeps each trajectory's gradient source on the parameter
device and folds it there; ``source_offload_device`` names only the pinned
spill destination used above the sampled allocator watermark. The mode must be
an exact drop-in for ``blocking`` at the objective level: same dual
(materialized AND reward-bound) reduction gate, same recorded bind-order folds,
same G1 elision, same residency accounting -- only the memory where sources
wait and where statistics accumulate changes.

Guarded failure modes:

* a fold arithmetic or order change relative to the CPU-offload reference;
* sources silently copied to host on the hot path (the cost this mode removes);
* spill above the watermark losing or double-folding a source;
* mixed residency after a spill episode corrupting a statistic pair;
* residency counters not returning to zero after group close.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tests.test_reward_linear import EPS, build_model, make_batch, reward_linear_grads  # noqa: E402
from tests.test_reward_linear_early_training import (  # noqa: E402
    _completion_order,
    early_reward_linear_grads,
    legacy_deferred_close_grads,
)

from thundersync.grpo.reward_linear import RewardLinearBackward  # noqa: E402
from thundersync.engine.streaming import (  # noqa: E402
    STREAM_ATTENTION_NAME,
    StreamingRun,
)


def test_device_resident_close_path_equals_blocking_exactly():
    trajs, groups, events, rewards = make_batch(n_groups=2, n_traj=3, seed=511)
    reference_model = build_model(STREAM_ATTENTION_NAME, seed=512)
    candidate_model = build_model(STREAM_ATTENTION_NAME, seed=512)

    reference, _run, _bw = reward_linear_grads(
        reference_model, trajs, groups, events, rewards
    )
    candidate, _candidate_run, candidate_bw = reward_linear_grads(
        candidate_model,
        trajs,
        groups,
        events,
        rewards,
        source_offload_mode="device_resident",
    )

    for name in reference:
        if reference[name] is None:
            assert candidate[name] is None, name
            continue
        assert torch.equal(reference[name], candidate[name]), name
    report = candidate_bw.source_offload_report
    assert report["source_offload_mode"] == "device_resident"
    assert report["source_queue_storage_device"] == (
        "parameter_device_until_watermark_spill"
    )
    assert report["current_source_bytes"] == 0
    assert report["current_statistic_bytes"] == 0
    assert report["device_resident_spilled_source_count"] == 0
    assert report["device_resident_spilled_source_bytes"] == 0


def test_device_resident_early_split_out_of_order_matches_legacy():
    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=4, seed=521)
    _last_event, completion = _completion_order(events)
    reward_order = list(reversed(completion))
    legacy_model = build_model(STREAM_ATTENTION_NAME, seed=522)
    candidate_model = build_model(STREAM_ATTENTION_NAME, seed=522)

    legacy, _run, _legacy_bw = legacy_deferred_close_grads(
        legacy_model, trajs, groups, events, rewards, close_order=reward_order
    )
    candidate, _candidate_run, candidate_bw = early_reward_linear_grads(
        candidate_model,
        trajs,
        groups,
        events,
        rewards,
        bind_order=reward_order,
        source_offload_mode="device_resident",
    )

    for name in legacy:
        if legacy[name] is None:
            assert candidate[name] is None, name
            continue
        assert torch.equal(legacy[name], candidate[name]), name
    report = candidate_bw.source_offload_report
    assert report["source_reduction_order_by_group"] == {0: reward_order}
    assert report["source_execution_order_by_group"] == {0: reward_order}
    assert report["source_preparation_order_by_group"] == {0: completion}
    assert report["prepared_awaiting_reward_current"] == 0
    assert report["current_source_bytes"] == 0


def test_device_resident_zero_reward_preserves_g1_elision():
    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=3, seed=531)
    zero_rewards = {t_id: 0.0 for t_id in rewards}
    reference_model = build_model(STREAM_ATTENTION_NAME, seed=532)
    candidate_model = build_model(STREAM_ATTENTION_NAME, seed=532)

    reference, _run, _bw = reward_linear_grads(
        reference_model, trajs, groups, events, zero_rewards
    )
    candidate, _candidate_run, candidate_bw = early_reward_linear_grads(
        candidate_model,
        trajs,
        groups,
        events,
        zero_rewards,
        source_offload_mode="device_resident",
    )

    for name in reference:
        if reference[name] is None:
            assert candidate[name] is None, name
            continue
        assert torch.equal(reference[name], candidate[name]), name
    report = candidate_bw.source_offload_report
    assert report["current_statistic_bytes"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_device_resident_keeps_sources_on_device_and_matches_blocking():
    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=4, seed=541)
    reference_model = build_model(STREAM_ATTENTION_NAME, seed=542).cuda()
    candidate_model = build_model(STREAM_ATTENTION_NAME, seed=542).cuda()

    reference, _run, _bw = reward_linear_grads(
        reference_model, trajs, groups, events, rewards
    )
    candidate, _candidate_run, candidate_bw = reward_linear_grads(
        candidate_model,
        trajs,
        groups,
        events,
        rewards,
        source_offload_mode="device_resident",
    )

    for name in reference:
        if reference[name] is None:
            assert candidate[name] is None, name
            continue
        assert torch.equal(reference[name], candidate[name]), name
    report = candidate_bw.source_offload_report
    # Below the watermark nothing crosses PCIe: no blocking copies, no pinned
    # transfers, no spills, and the fp32 statistics accumulate on the
    # parameter device so group close reloads nothing.
    assert report["source_offload_s"] == 0.0
    assert report["peak_source_offload_inflight_bytes"] == 0
    assert report["device_resident_spilled_source_count"] == 0
    assert report["reloaded_source_bytes"] == 0
    assert report["current_source_bytes"] == 0
    assert report["current_statistic_bytes"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_device_resident_pressure_spills_and_preserves_gradient():
    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=3, seed=551)
    reference_model = build_model(STREAM_ATTENTION_NAME, seed=552).cuda()
    candidate_model = build_model(STREAM_ATTENTION_NAME, seed=552).cuda()

    reference, _run, _bw = reward_linear_grads(
        reference_model, trajs, groups, events, rewards
    )
    candidate, _candidate_run, candidate_bw = reward_linear_grads(
        candidate_model,
        trajs,
        groups,
        events,
        rewards,
        source_offload_mode="device_resident",
        source_offload_hbm_threshold=1e-9,
    )

    for name in reference:
        if reference[name] is None:
            assert candidate[name] is None, name
            continue
        assert torch.equal(reference[name], candidate[name]), name
    report = candidate_bw.source_offload_report
    assert report["device_resident_spilled_source_count"] > 0
    assert report["device_resident_spilled_source_bytes"] > 0
    assert report["source_offload_inflight_bytes"] == 0
    assert report["current_source_bytes"] == 0
    assert report["current_statistic_bytes"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_device_resident_mixed_residency_after_spill_episode():
    """A spill episode mid-group must fold into the already-resident pair."""

    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=3, seed=561)
    reference_model = build_model(STREAM_ATTENTION_NAME, seed=562).cuda()
    candidate_model = build_model(STREAM_ATTENTION_NAME, seed=562).cuda()

    reference, _run, _bw = reward_linear_grads(
        reference_model, trajs, groups, events, rewards
    )

    run = StreamingRun(candidate_model, boundary_cut=True)
    backward = RewardLinearBackward(
        run,
        candidate_model,
        eps=EPS,
        source_offload_device="cpu",
        source_offload_mode="device_resident",
    )
    group_id = next(iter(groups))
    trajectory_ids = sorted(
        trajectory.traj_id
        for trajectory in trajs
        if trajectory.group_id == group_id
    )
    run.open_group(group_id, groups[group_id])
    backward.register_group_trajectories(group_id, trajectory_ids)
    last_event, _completion = _completion_order(events)

    # Pressure flips on after the first trajectory: later sources spill to
    # pinned CPU while the statistics already live on CUDA.
    pressure_samples = {"count": 0}

    def pressure_after_first(_sources):
        pressure_samples["count"] += 1
        return 0.0 if pressure_samples["count"] <= 1 else 1.0

    backward._hbm_pressure_fraction = pressure_after_first

    for index, (gid, t_id, tokens, scored) in enumerate(events):
        run.append_turn(gid, t_id, tokens, scored)
        if last_event[t_id] == index:
            backward.close_trajectory(t_id, rewards[t_id])
    backward.close_group(group_id)
    backward.assert_safe_to_step()

    candidate = {
        name: (
            parameter.grad.clone() if parameter.grad is not None else None
        )
        for name, parameter in candidate_model.named_parameters()
    }
    for name in reference:
        if reference[name] is None:
            assert candidate[name] is None, name
            continue
        assert torch.equal(reference[name], candidate[name]), name
    report = backward.source_offload_report
    assert report["device_resident_spilled_source_count"] > 0
    assert report["current_source_bytes"] == 0
    assert report["current_statistic_bytes"] == 0
