"""Reward-linear backward: the streamed backward must equal the barrier backward.

The four properties that make the reward-linear backward safe to run, each with the failure mode it
guards against:

* gradient equality with the monolithic barrier backward -- the whole point;
* the shared prompt subgraph is backwarded ONCE per group -- without the
  boundary cut this silently costs ~2x logical instead of ~1x physical;
* memory actually returns per trajectory -- the residency claim, at the
  allocator's level of truth, not the bookkeeper's;
* an optimizer step with open groups fails LOUDLY -- the retained graphs
  reference in-place-mutated parameters and would otherwise be silently wrong
  or crash later with an unhelpful autograd error.
"""

from __future__ import annotations

import threading
import time

import pytest
import torch

from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from thundersync.engine.segments import Segment, Trajectory, build_forest
from thundersync.engine.plan import compile_plan# noqa: E402
from thundersync.grpo.reward_linear import (  # noqa: E402
    RewardLinearBackward,
    _GroupAccum,
    _PendingCpuTransfer,
    _StreamingTensorOffload,
)
from thundersync.engine.step import (  # noqa: E402
    ATTENTION_NAME,
    forest_logprobs,
    register_attention,
    split_by_trajectory,
)
from thundersync.engine.streaming import (  # noqa: E402
    STREAM_ATTENTION_NAME,
    StreamingRun,
    register_stream_attention,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
register_attention()
register_stream_attention()
EPS = 1e-6


def build_model(impl, vocab=512, seed=0):
    torch.manual_seed(seed)
    cfg = AutoConfig.for_model(
        "qwen3",
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=vocab,
        max_position_embeddings=8192,
        tie_word_embeddings=False,
    )
    return AutoModelForCausalLM.from_config(
        cfg, attn_implementation=impl, dtype=torch.float32
    ).to(DEV).eval()


def make_batch(n_groups=2, n_traj=4, prompt_len=24, vocab=512, seed=1):
    gen = torch.Generator().manual_seed(seed)
    r = lambda n: torch.randint(0, vocab, (n,), generator=gen).tolist()  # noqa: E731
    trajs, groups, events, tid = [], {}, [], 0
    for gid in range(n_groups):
        groups[gid] = r(prompt_len)
        per = []
        for i in range(n_traj):
            turns = [
                (r(4 + k + i), r(6 + i)) for k in range(1 + (i + gid) % 3)
            ]
            segs = [Segment.prompt(groups[gid])]
            for a, o in turns:
                segs.append(Segment.action(a))
                segs.append(Segment.observation(o))
            trajs.append(Trajectory(segs, group_id=gid, traj_id=tid))
            per.append((tid, turns))
            tid += 1
        depth = 0
        while any(depth < len(t) for _, t in per):
            for t_id, turns in per:
                if depth < len(turns):
                    a, o = turns[depth]
                    events.append((gid, t_id, a + o, [True] * len(a) + [False] * len(o)))
            depth += 1
    gen2 = torch.Generator().manual_seed(seed + 99)
    rewards = {t.traj_id: float(torch.rand(1, generator=gen2)) for t in trajs}
    return trajs, groups, events, rewards


def reward_linear_grads(
    model,
    trajs,
    groups,
    events,
    rewards,
    *,
    source_offload_device="cpu",
    source_offload_mode="blocking",
    source_offload_window_bytes=256 * 1024 * 1024,
    source_offload_hbm_threshold=0.65,
    source_reduction_worker_count=2,
    source_offload_worker_count=2,
    source_offload_pinned_windows=2,
    statistic_device=None,
    gradient_accumulation_storage_device=None,
):
    """Stream, close trajectories at their own completion, close groups."""
    run = StreamingRun(
        model,
        boundary_cut=True,
        parameter_adjoint_storage_device=(
            gradient_accumulation_storage_device
        ),
    )
    bw = RewardLinearBackward(
        run,
        model,
        eps=EPS,
        source_offload_device=source_offload_device,
        source_offload_mode=source_offload_mode,
        source_offload_window_bytes=source_offload_window_bytes,
        source_offload_hbm_threshold=source_offload_hbm_threshold,
        source_reduction_worker_count=source_reduction_worker_count,
        source_offload_worker_count=source_offload_worker_count,
        source_offload_pinned_windows=source_offload_pinned_windows,
        statistic_device=statistic_device,
    )
    for gid, prompt in groups.items():
        run.open_group(gid, prompt)
        bw.register_group_trajectories(
            gid,
            sorted(
                trajectory.traj_id
                for trajectory in trajs
                if trajectory.group_id == gid
            ),
        )
    last_event = {}
    for i, (gid, t_id, *_rest) in enumerate(events):
        last_event[t_id] = i
    for i, (gid, t_id, tokens, scored) in enumerate(events):
        run.append_turn(gid, t_id, tokens, scored)
        if last_event[t_id] == i:  # trajectory finished: its reward lands NOW
            bw.close_trajectory(t_id, rewards[t_id])
    for t in trajs:
        pass
    for gid in groups:
        bw.close_group(gid)
    bw.assert_safe_to_step()
    return {n: (p.grad.clone() if p.grad is not None else None)
            for n, p in model.named_parameters()}, run, bw


def test_cpu_source_offload_preserves_reward_linear_gradient_and_releases_sources():
    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=4)
    reference_model = build_model(STREAM_ATTENTION_NAME, seed=91)
    offload_model = build_model(STREAM_ATTENTION_NAME, seed=91)

    reference, _run, _bw = reward_linear_grads(
        reference_model, trajs, groups, events, rewards
    )
    offloaded, _offload_run, offload_bw = reward_linear_grads(
        offload_model,
        trajs,
        groups,
        events,
        rewards,
        source_offload_device="cpu",
        gradient_accumulation_storage_device="cpu",
    )

    for name in reference:
        assert torch.equal(reference[name], offloaded[name]), name
    report = offload_bw.source_offload_report
    assert report["storage_device"] == "cpu"
    assert report["peak_source_bytes"] > 0
    assert report["current_source_bytes"] == 0
    assert report["stored_source_bytes"] >= report["peak_source_bytes"]
    gradient_report = _offload_run.parameter_adjoint_diagnostics()
    assert gradient_report["storage_device"] == "cpu"
    assert gradient_report["materialized"] is True
    assert gradient_report["peak_accumulator_bytes"] > 0


def test_group_close_accumulates_one_parameter_cotangent_at_a_time():
    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=4, seed=911)
    model = build_model(STREAM_ATTENTION_NAME, seed=912)

    _gradients, run, _backward = reward_linear_grads(
        model,
        trajs,
        groups,
        events,
        rewards,
        source_offload_device="cpu",
    )

    report = run.parameter_adjoint_diagnostics()
    largest_parameter_gradient = max(
        parameter.numel() * torch.empty((), dtype=torch.float32).element_size()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert report["manual_accumulation_calls"] > 1
    assert 0 < report["manual_peak_batch_bytes"] <= largest_parameter_gradient


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pinned_nonblocking_source_offload_preserves_gradient_and_drains_events():
    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=4, seed=92)
    reference_model = build_model(STREAM_ATTENTION_NAME, seed=191)
    candidate_model = build_model(STREAM_ATTENTION_NAME, seed=191)

    reference, _run, _bw = reward_linear_grads(
        reference_model,
        trajs,
        groups,
        events,
        rewards,
        source_offload_device="cpu",
    )
    candidate, _candidate_run, candidate_bw = reward_linear_grads(
        candidate_model,
        trajs,
        groups,
        events,
        rewards,
        source_offload_device="cpu",
        source_offload_mode="pinned_nonblocking",
    )

    for name in reference:
        assert torch.equal(reference[name], candidate[name]), name
    report = candidate_bw.source_offload_report
    assert report["source_offload_mode"] == "pinned_nonblocking"
    assert report["source_offload_inflight_bytes"] == 0
    assert report["peak_source_offload_inflight_bytes"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pinned_streaming_preserves_gradient_and_bounds_pinned_window():
    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=4, seed=192)
    reference_model = build_model(STREAM_ATTENTION_NAME, seed=291).cuda()
    candidate_model = build_model(STREAM_ATTENTION_NAME, seed=291).cuda()

    reference, _run, _bw = reward_linear_grads(
        reference_model,
        trajs,
        groups,
        events,
        rewards,
        source_offload_device="cpu",
        source_offload_mode="pinned_nonblocking",
    )
    candidate, _candidate_run, candidate_bw = reward_linear_grads(
        candidate_model,
        trajs,
        groups,
        events,
        rewards,
        source_offload_device="cpu",
        source_offload_mode="pinned_streaming",
        source_offload_window_bytes=1024,
        source_reduction_worker_count=2,
        source_offload_worker_count=3,
    )

    for name in reference:
        assert torch.equal(reference[name], candidate[name]), name
    report = candidate_bw.source_offload_report
    assert report["source_offload_mode"] == "pinned_streaming"
    assert report["source_queue_storage_device"] == (
        "parameter_device_until_streamed_d2h"
    )
    assert report["source_offload_window_bytes"] == 1024
    assert report["source_offload_inflight_bytes"] == 0
    # Two pinned windows per worker: one chunk copies while the previous
    # chunk folds.  Residency stays declared and bounded at twice the window.
    assert 0 < report["peak_source_offload_inflight_bytes"] <= 2048
    assert 0 < report["streaming_pinned_buffer_reserved_bytes"] <= 10240
    assert report["source_reduction_worker_count"] == 2
    assert report["source_offload_worker_count"] == 3
    assert report["streaming_pinned_buffer_capacity_bytes"] == 10240
    assert report["current_pageable_stage_bytes"] == 0
    assert report["current_source_bytes"] == 0
    assert report["current_statistic_bytes"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pinned_streaming_hbm_pressure_stages_without_changing_gradient():
    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=2, seed=193)
    reference_model = build_model(STREAM_ATTENTION_NAME, seed=292).cuda()
    candidate_model = build_model(STREAM_ATTENTION_NAME, seed=292).cuda()
    reference, _run, _bw = reward_linear_grads(
        reference_model,
        trajs,
        groups,
        events,
        rewards,
        source_offload_device="cpu",
        source_offload_mode="pinned_nonblocking",
    )
    candidate, _candidate_run, candidate_bw = reward_linear_grads(
        candidate_model,
        trajs,
        groups,
        events,
        rewards,
        source_offload_device="cpu",
        source_offload_mode="pinned_streaming",
        source_offload_window_bytes=1024,
        source_offload_hbm_threshold=1e-9,
    )
    for name in reference:
        assert torch.equal(reference[name], candidate[name]), name
    report = candidate_bw.source_offload_report
    assert report["hbm_staged_source_count"] == 2
    assert report["hbm_staged_source_bytes"] > 0
    assert report["hbm_stage_s"] > 0
    assert report["peak_pageable_stage_bytes"] == 0
    assert report["current_pageable_stage_bytes"] == 0
    assert report["source_offload_inflight_bytes"] == 0
    assert report["current_source_bytes"] == 0
    assert report["current_statistic_bytes"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pinned_streaming_all_unscored_trajectory_closes_without_deadlock():
    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=2, seed=196)
    unscored_events = [
        (group_id, trajectory_id, tokens, [False] * len(tokens))
        for group_id, trajectory_id, tokens, _scored in events
    ]
    model = build_model(STREAM_ATTENTION_NAME, seed=295).cuda()

    gradients, _run, backward = reward_linear_grads(
        model,
        trajs,
        groups,
        unscored_events,
        rewards,
        source_offload_device="cpu",
        source_offload_mode="pinned_streaming",
        source_offload_window_bytes=1024,
    )

    assert all(value is None for value in gradients.values())
    report = backward.source_offload_report
    expected_order = sorted(rewards)
    assert report["source_execution_order_by_group"] == {0: expected_order}
    assert report["source_reduction_order_by_group"] == {0: expected_order}
    assert report["current_source_bytes"] == 0
    assert report["current_statistic_bytes"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pinned_streaming_preserves_existing_optimizer_grad_buffers():
    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=2, seed=194)
    model = build_model(STREAM_ATTENTION_NAME, seed=293).cuda()
    run = StreamingRun(model, boundary_cut=True)
    backward = RewardLinearBackward(
        run,
        model,
        source_offload_device="cpu",
        source_offload_mode="pinned_streaming",
        source_offload_window_bytes=1024,
        source_offload_hbm_threshold=0.999,
    )
    group_id = next(iter(groups))
    trajectory_ids = sorted(
        trajectory.traj_id for trajectory in trajs
        if trajectory.group_id == group_id
    )
    run.open_group(group_id, groups[group_id])
    backward.register_group_trajectories(group_id, trajectory_ids)
    first_trajectory = trajectory_ids[0]
    for event_group, trajectory_id, tokens, scored in events:
        run.append_turn(event_group, trajectory_id, tokens, scored)

    prior = []
    for parameter in backward.params:
        parameter.grad = torch.randn_like(parameter)
        prior.append((parameter.grad, parameter.grad.clone()))
    backward._hbm_pressure_fraction = lambda _sources: 0.0
    backward.close_trajectory(first_trajectory, rewards[first_trajectory])
    backward.wait_for_trajectory_reduction(group_id, first_trajectory)
    for parameter, (gradient_object, gradient_value) in zip(
        backward.params, prior
    ):
        assert parameter.grad is gradient_object
        assert torch.equal(parameter.grad, gradient_value)
    backward.abort()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pinned_streaming_no_stage_waits_nondefault_producer_stream():
    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=2, seed=195)
    reference_model = build_model(STREAM_ATTENTION_NAME, seed=294).cuda()
    candidate_model = build_model(STREAM_ATTENTION_NAME, seed=294).cuda()
    reference, _run, _backward = reward_linear_grads(
        reference_model,
        trajs,
        groups,
        events,
        rewards,
        source_offload_device="cpu",
        source_offload_mode="pinned_nonblocking",
    )

    run = StreamingRun(candidate_model, boundary_cut=True)
    backward = RewardLinearBackward(
        run,
        candidate_model,
        source_offload_device="cpu",
        source_offload_mode="pinned_streaming",
        source_offload_window_bytes=1024,
        source_offload_hbm_threshold=0.999,
    )
    for group_id, prompt in groups.items():
        run.open_group(group_id, prompt)
        backward.register_group_trajectories(
            group_id,
            sorted(
                trajectory.traj_id for trajectory in trajs
                if trajectory.group_id == group_id
            ),
        )
    last_event = {}
    for index, (_group_id, trajectory_id, *_rest) in enumerate(events):
        last_event[trajectory_id] = index
    producer_stream = torch.cuda.Stream()
    backward._hbm_pressure_fraction = lambda _sources: 0.0
    for index, (group_id, trajectory_id, tokens, scored) in enumerate(events):
        with torch.cuda.stream(producer_stream):
            run.append_turn(group_id, trajectory_id, tokens, scored)
            if last_event[trajectory_id] == index:
                backward.close_trajectory(
                    trajectory_id,
                    rewards[trajectory_id],
                )
    for group_id in groups:
        backward.close_group(group_id)
    backward.assert_safe_to_step()
    candidate = {
        name: (parameter.grad.clone() if parameter.grad is not None else None)
        for name, parameter in candidate_model.named_parameters()
    }
    for name in reference:
        assert torch.equal(reference[name], candidate[name]), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pinned_streaming_slow_group_does_not_block_other_ready_source():
    trajs, groups, events, rewards = make_batch(n_groups=2, n_traj=2, seed=292)
    model = build_model(STREAM_ATTENTION_NAME, seed=391).cuda()
    run = StreamingRun(model, boundary_cut=True)
    backward = RewardLinearBackward(
        run,
        model,
        source_offload_device="cpu",
        source_offload_mode="pinned_streaming",
        source_offload_window_bytes=1024,
        source_reduction_worker_count=2,
    )
    by_group = {
        group_id: sorted(
            trajectory.traj_id
            for trajectory in trajs
            if trajectory.group_id == group_id
        )
        for group_id in groups
    }
    for group_id, prompt in groups.items():
        run.open_group(group_id, prompt)
        backward.register_group_trajectories(group_id, by_group[group_id])
    for group_id, trajectory_id, tokens, scored in events:
        run.append_turn(group_id, trajectory_id, tokens, scored)

    first_fold_started = threading.Event()
    release_first_fold = threading.Event()
    original_fold = backward._fold_streaming_statistics
    block_lock = threading.Lock()
    blocked = False

    def blocked_fold(*args, **kwargs):
        nonlocal blocked
        should_block = False
        with block_lock:
            if not blocked:
                blocked = True
                should_block = True
        if should_block:
            first_fold_started.set()
            assert release_first_fold.wait(timeout=5.0)
        return original_fold(*args, **kwargs)

    backward._fold_streaming_statistics = blocked_fold
    first_group, second_group = sorted(groups)
    first_trajectory = by_group[first_group][0]
    second_trajectory = by_group[second_group][0]
    backward.close_trajectory(first_trajectory, rewards[first_trajectory])
    assert first_fold_started.wait(timeout=5.0)
    backward.close_trajectory(second_trajectory, rewards[second_trajectory])
    assert backward.source_offload_report["source_execution_order_by_group"] == {
        first_group: [first_trajectory],
        second_group: [second_trajectory],
    }

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        reduced = backward.source_offload_report[
            "source_reduction_order_by_group"
        ]
        if reduced.get(second_group) == [second_trajectory]:
            break
        time.sleep(0.01)
    assert reduced.get(second_group) == [second_trajectory]

    release_first_fold.set()
    for group_id in sorted(groups):
        for trajectory_id in by_group[group_id][1:]:
            backward.close_trajectory(trajectory_id, rewards[trajectory_id])
        backward.close_group(group_id)
    backward.assert_safe_to_step()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pinned_streaming_parallel_tensor_fold_preserves_readiness_order():
    """One trajectory's tensors may fold in parallel without source reordering.

    Pinned streaming is a device-to-host path; with CUDA hidden the node
    CPU pass at 7ed615c failed here while CPU-only torch on a machine
    without a driver passes, so the test states the device it needs.
    """

    model = torch.nn.Linear(1, 1, bias=False)

    class Run:
        boundary_cut = True

    backward = RewardLinearBackward(
        Run(),
        model,
        source_offload_device="cpu",
        source_offload_mode="pinned_streaming",
        source_offload_window_bytes=8,
        source_offload_hbm_threshold=0.5,
        source_reduction_worker_count=1,
        source_offload_worker_count=2,
    )
    group = _GroupAccum()
    group.parameter_sources[0] = [torch.tensor([1.0]), torch.tensor([2.0])]
    group.boundary_sources[0] = []
    group.parameter_centered_sum = [None, None]
    group.parameter_source_mean = [None, None]
    group.rewards_by_trajectory[0] = 1.0
    group.differentiated.add(0)
    backward._groups[0] = group
    backward._expected_trajectory_order_by_group[0] = (0,)
    backward._source_execution_order_by_group[0] = [0]
    backward._current_source_bytes = 8
    backward._stored_source_bytes = 8

    first_started = threading.Event()
    release_first = threading.Event()
    second_finished = threading.Event()

    original_fold = backward._fold_streaming_statistics

    def fold_out_of_order(sources, **kwargs):
        value = next((value for value in sources if value is not None), None)
        if value is None:
            return original_fold(sources, **kwargs)
        if float(value.item()) == 1.0:
            first_started.set()
            assert release_first.wait(timeout=5.0)
        else:
            second_finished.set()
        return original_fold(sources, **kwargs)

    backward._hbm_pressure_fraction = lambda _sources: 1.0
    backward._fold_streaming_statistics = fold_out_of_order
    with backward._reducer_condition:
        queued_at = time.perf_counter()
        key = (0, 0)
        backward._source_queued_at[key] = queued_at
        backward._streaming_pending_tensors[key] = 2
        backward._streaming_capture_complete.add(key)
        destination = group.parameter_sources[0]
        for destination_index in (0, 1):
            backward._offload_queue.append(
                _StreamingTensorOffload(
                    group_id=0,
                    trajectory_id=0,
                    destination=destination,
                    destination_index=destination_index,
                    source_tensor=destination[destination_index],
                    source_ready_event=None,
                    reward_sum=group.parameter_centered_sum,
                    unit_sum=group.parameter_source_mean,
                    reward=1.0,
                    keep_source_dtype=False,
                )
            )
        backward._reducer_condition.notify_all()

    try:
        assert first_started.wait(timeout=5.0)
        assert second_finished.wait(timeout=5.0)
        time.sleep(0.05)
        report = backward.source_offload_report
        assert report["source_offload_active_trajectories_peak"] == 1
        assert report[
            "source_offload_active_trajectories_peak_by_group"
        ] == {0: 1}
        assert report["hbm_stage_active_trajectories_peak"] == 1
        assert report[
            "hbm_stage_active_trajectories_peak_by_group"
        ] == {0: 1}
        assert report["hbm_stage_active_tensor_jobs_peak"] == 2
        assert report[
            "hbm_stage_active_tensor_jobs_peak_by_group"
        ] == {0: 2}
        assert report["source_reduction_order_by_group"] == {}

        release_first.set()
        backward.wait_for_trajectory_reduction(0, 0)
        report = backward.source_offload_report
        assert report["source_execution_order_by_group"] == {0: [0]}
        assert report["source_reduction_order_by_group"] == {0: [0]}
        assert report["source_offload_active_trajectories"] == 0
        assert report["source_offload_active_trajectories_by_group"] == {}
        assert report["hbm_stage_active_trajectories"] == 0
        assert report["hbm_stage_active_trajectories_by_group"] == {}
        assert report["hbm_stage_active_tensor_jobs"] == 0
        assert report["hbm_stage_active_tensor_jobs_by_group"] == {}
        assert report["current_source_bytes"] == 0
    finally:
        release_first.set()
        backward._shutdown_workers()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pinned_streaming_accounts_source_before_fast_worker_release():
    """A worker may finish immediately after publication without underflow."""

    model = torch.nn.Linear(1, 1, bias=False).cuda()

    class Run:
        boundary_cut = True

    backward = RewardLinearBackward(
        Run(),
        model,
        source_offload_device="cpu",
        source_offload_mode="pinned_streaming",
        source_offload_window_bytes=8,
        source_offload_hbm_threshold=0.5,
        source_reduction_worker_count=1,
        source_offload_worker_count=1,
    )
    source = torch.tensor([3.0], device="cuda")
    group = _GroupAccum(
        parameter_sources={0: [None]},
        boundary_sources={0: []},
        rewards_by_trajectory={0: 1.0},
        parameter_centered_sum=[None],
        parameter_source_mean=[None],
        differentiated={0},
    )
    backward._groups[0] = group
    backward._expected_trajectory_order_by_group[0] = (0,)
    backward._source_execution_order_by_group[0] = [0]
    backward._source_queued_at[(0, 0)] = time.perf_counter()
    backward._hbm_pressure_fraction = lambda _sources: 1.0

    try:
        backward._register_streaming_source_tensor(
            group_id=0,
            trajectory_id=0,
            destination=group.parameter_sources[0],
            destination_index=0,
            gradient=source,
            reward_sum=group.parameter_centered_sum,
            unit_sum=group.parameter_source_mean,
            reward=1.0,
            keep_source_dtype=False,
        )
        with backward._reducer_condition:
            backward._streaming_capture_complete.add((0, 0))
            backward._mark_streaming_source_ready_locked((0, 0))
            backward._reducer_condition.notify_all()
        backward.wait_for_trajectory_reduction(0, 0)
        report = backward.source_offload_report
        assert report["current_source_bytes"] == 0
        assert report["peak_source_bytes"] == source.numel() * source.element_size()
    finally:
        backward._shutdown_workers()


def test_offload_completion_reduces_without_waiting_for_trajectory_prefix():
    class Event:
        synchronized = False

        def synchronize(self):
            self.synchronized = True

    model = torch.nn.Linear(1, 1)

    class Run:
        boundary_cut = True

    run = Run()
    backward = RewardLinearBackward(run, model)
    event = Event()
    transfer = _PendingCpuTransfer(
        cpu_tensor=torch.ones(1),
        source_tensor=torch.ones(1),
        event=event,
        bytes=4,
    )
    parameter_sources = [transfer]
    boundary_sources = []
    group = _GroupAccum()
    group.parameter_sources[1] = parameter_sources
    group.boundary_sources[1] = boundary_sources
    group.rewards_by_trajectory[1] = 1.0
    group.differentiated.add(1)
    # The legacy close path binds the reward before any offload job exists;
    # reduction eligibility now requires the bound reward, so the fixture
    # must model that state to exercise only the no-prefix-wait property.
    group.closed.add(1)
    backward._groups[0] = group
    backward._expected_trajectory_order_by_group[0] = (0, 1)
    backward._source_execution_order_by_group[0] = [1]
    backward._source_offload_inflight_bytes = 4
    backward._current_source_bytes = 4
    backward._stored_source_bytes = 4
    backward._source_queued_at[(0, 1)] = time.perf_counter()
    with backward._reducer_condition:
        backward._offload_queue.append(
            (0, 1, parameter_sources, boundary_sources, None)
        )
        backward._reducer_condition.notify_all()

    failures = []

    def wait_for_reduction():
        try:
            backward.wait_for_trajectory_reduction(0, 1)
        except BaseException as error:
            failures.append(error)

    waiter = threading.Thread(target=wait_for_reduction, daemon=True)
    waiter.start()
    waiter.join(timeout=2.0)
    assert not waiter.is_alive()
    assert failures == []

    assert event.synchronized
    assert transfer.source_tensor is None
    assert backward.source_offload_report["source_offload_inflight_bytes"] == 0
    assert group.reduced == {1}
    assert backward.source_offload_report["current_source_bytes"] == 0
    backward._shutdown_workers()


def test_zero_reward_source_defers_g1_without_changing_group_formula():
    model = torch.nn.Linear(4, 1, bias=False)

    class Run:
        boundary_cut = True

    run = Run()
    backward = RewardLinearBackward(run, model)
    reward_sum: list[torch.Tensor | None] = []
    unit_sum: list[torch.Tensor | None] = []
    zero_reward_source = torch.tensor([1.0, -2.0, 3.0, -4.0])
    nonzero_reward_source = torch.tensor([-5.0, 6.0, -7.0, 8.0])

    try:
        first = backward._store_sources([zero_reward_source.clone()])
        backward._fold_in_memory_statistics(
            first,
            reward_sum=reward_sum,
            unit_sum=unit_sum,
            reward=0.0,
            keep_source_dtype=False,
        )

        statistic_bytes = zero_reward_source.numel() * torch.float32.itemsize
        assert reward_sum == [None]
        assert torch.equal(unit_sum[0], zero_reward_source)
        assert backward.source_offload_report["current_source_bytes"] == 0
        assert (
            backward.source_offload_report["current_statistic_bytes"]
            == statistic_bytes
        )
        assert (
            backward.source_offload_report["peak_statistic_bytes"]
            == statistic_bytes
        )

        second = backward._store_sources([nonzero_reward_source.clone()])
        backward._fold_in_memory_statistics(
            second,
            reward_sum=reward_sum,
            unit_sum=unit_sum,
            reward=3.0,
            keep_source_dtype=False,
        )

        assert torch.equal(
            reward_sum[0],
            nonzero_reward_source * 3.0,
        )
        assert torch.equal(
            unit_sum[0],
            zero_reward_source + nonzero_reward_source,
        )
        mean = 1.5
        std = 1.5 + EPS
        combined = (reward_sum[0] - mean * unit_sum[0]) / std
        expected = (
            -zero_reward_source + nonzero_reward_source
        ) * (1.5 / std)
        assert torch.equal(combined, expected)
        assert (
            backward.source_offload_report["current_statistic_bytes"]
            == 2 * statistic_bytes
        )

        backward._release_statistic_pair(reward_sum, unit_sum, 0)
        assert backward.source_offload_report["current_statistic_bytes"] == 0
    finally:
        backward._shutdown_workers()


def test_k0_statistics_preserve_gradient_with_readiness_order_reduction():
    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=4, seed=93)
    trajectory_ids = sorted(trajectory.traj_id for trajectory in trajs)

    def execute(close_order, reduction_mode):
        model = build_model(STREAM_ATTENTION_NAME, seed=95)
        run = StreamingRun(
            model,
            boundary_cut=True,
            parameter_adjoint_storage_device="cpu",
        )
        backward = RewardLinearBackward(
            run,
            model,
            eps=EPS,
            source_offload_device="cpu",
            source_reduction_mode=reduction_mode,
            source_reduction_pack_size=1,
        )
        run.open_group(0, groups[0])
        backward.register_group_trajectories(0, trajectory_ids)
        for group_id, trajectory_id, tokens, scored in events:
            run.append_turn(group_id, trajectory_id, tokens, scored)
        for trajectory_id in close_order:
            backward.close_trajectory(trajectory_id, rewards[trajectory_id])
        backward.close_group(0)
        backward.assert_safe_to_step()
        return {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }, backward.source_offload_report

    reference, reference_report = execute(
        trajectory_ids,
        "unrestricted_statistics",
    )
    unrestricted, unrestricted_report = execute(
        reversed(trajectory_ids),
        "unrestricted_statistics",
    )
    unrestricted_logical, unrestricted_logical_report = execute(
        trajectory_ids,
        "unrestricted_statistics",
    )

    assert (
        reference.keys() == unrestricted.keys()
        == unrestricted_logical.keys()
    )
    for name in reference:
        torch.testing.assert_close(
            reference[name],
            unrestricted[name],
            rtol=2e-3,
            atol=1e-5,
            msg=lambda message, parameter_name=name: (
                f"{parameter_name}: {message}"
            ),
        )
        assert torch.equal(reference[name], unrestricted_logical[name]), name
    assert unrestricted_report["source_execution_order_by_group"] == {
        0: list(reversed(trajectory_ids))
    }
    assert unrestricted_report["source_reduction_order_by_group"] == {
        0: list(reversed(trajectory_ids))
    }
    assert unrestricted_report["deferred_ready_trajectories_peak"] == 0
    assert 1 <= unrestricted_report[
        "pending_reduction_trajectories_peak"
    ] <= 4
    assert unrestricted_report["peak_source_bytes"] <= (
        len(trajectory_ids) * reference_report["peak_source_bytes"]
    )
    assert unrestricted_report["statistic_representation"] == (
        "readiness_order_reward_and_unit_sums_in_memory"
    )
    assert unrestricted_report["peak_spill_bytes"] == 0
    assert unrestricted_report["current_spill_bytes"] == 0
    assert 1 <= unrestricted_logical_report[
        "pending_reduction_trajectories_peak"
    ] <= len(trajectory_ids)
    assert unrestricted_report["current_source_bytes"] == 0

    difference_sq = sum(
        (reference[name].double() - unrestricted[name].double()).square().sum()
        for name in reference
    )
    reference_sq = sum(
        reference[name].double().square().sum() for name in reference
    )
    dot = sum(
        (reference[name].double() * unrestricted[name].double()).sum()
        for name in reference
    )
    candidate_sq = sum(
        unrestricted[name].double().square().sum() for name in reference
    )
    relative_l2 = float((difference_sq / reference_sq).sqrt())
    cosine = float(dot / (reference_sq * candidate_sq).sqrt())
    assert relative_l2 <= 1e-5
    assert cosine >= 0.999999999


def test_unrestricted_statistics_admit_ready_sources_without_prefix_wait():
    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=4, seed=97)
    trajectory_ids = sorted(trajectory.traj_id for trajectory in trajs)
    reverse_order = list(reversed(trajectory_ids))
    model = build_model(STREAM_ATTENTION_NAME, seed=99)
    run = StreamingRun(
        model,
        boundary_cut=True,
        parameter_adjoint_storage_device="cpu",
    )
    backward = RewardLinearBackward(
        run,
        model,
        eps=EPS,
        source_offload_device="cpu",
        source_reduction_mode="unrestricted_statistics",
        source_reduction_pack_size=1,
    )
    run.open_group(0, groups[0])
    backward.register_group_trajectories(0, trajectory_ids)
    for group_id, trajectory_id, tokens, scored in events:
        run.append_turn(group_id, trajectory_id, tokens, scored)

    for admitted_count, trajectory_id in enumerate(reverse_order, start=1):
        backward.close_trajectory(trajectory_id, rewards[trajectory_id])
        report = backward.source_offload_report
        assert report["source_execution_order_by_group"] == {
            0: reverse_order[:admitted_count]
        }
        assert report["deferred_ready_trajectories_peak"] == 0
        assert report["current_source_bytes"] >= 0
        assert report["current_spill_bytes"] == 0

    backward.close_group(0)
    backward.assert_safe_to_step()
    report = backward.source_offload_report
    assert report["source_reduction_mode"] == "unrestricted_statistics"
    assert report["source_reduction_worker_count"] == 2
    assert report["source_reduction_order_by_group"] == {0: reverse_order}
    assert 1 <= report["pending_reduction_trajectories_peak"] <= len(
        trajectory_ids
    )
    assert report["statistic_representation"] == (
        "readiness_order_reward_and_unit_sums_in_memory"
    )
    assert report["peak_source_bytes"] > 0
    assert report["peak_statistic_bytes"] > 0
    assert report["current_statistic_bytes"] == 0
    assert report["pending_reduction_trajectories_current"] == 0
    assert report["reducer_queue_s"] >= 0
    assert report["reducer_queue_max_s"] >= 0
    assert report["reducer_fold_s"] > 0
    assert report["reducer_executing_source_bytes"] == 0
    assert report["reducer_executing_source_bytes_peak"] > 0
    assert report["peak_spill_bytes"] == 0
    assert report["current_spill_bytes"] == 0


def test_unrestricted_statistics_rejects_synchronous_spill(tmp_path):
    model = build_model(STREAM_ATTENTION_NAME, seed=100)
    run = StreamingRun(model, boundary_cut=True)
    with pytest.raises(ValueError, match="synchronous source spill"):
        RewardLinearBackward(
            run,
            model,
            source_offload_device="cpu",
            source_reduction_mode="unrestricted_statistics",
            source_spill_directory=tmp_path / "spill",
        )


def test_unrestricted_statistics_rejects_a_hidden_pack_barrier():
    model = build_model(STREAM_ATTENTION_NAME, seed=101)
    run = StreamingRun(model, boundary_cut=True)
    with pytest.raises(ValueError, match="one-source admission"):
        RewardLinearBackward(
            run,
            model,
            source_reduction_mode="unrestricted_statistics",
            source_reduction_pack_size=2,
        )


def test_unrestricted_reducer_failure_aborts_a_partially_open_group():
    _trajs, groups, events, rewards = make_batch(
        n_groups=1, n_traj=2, seed=102
    )
    trajectory_ids = sorted({trajectory_id for _, trajectory_id, _, _ in events})
    trajectory_id = trajectory_ids[0]
    model = build_model(STREAM_ATTENTION_NAME, seed=104)
    run = StreamingRun(
        model,
        boundary_cut=True,
        parameter_adjoint_storage_device="cpu",
    )
    backward = RewardLinearBackward(
        run,
        model,
        source_offload_device="cpu",
        source_reduction_mode="unrestricted_statistics",
    )
    run.open_group(0, groups[0])
    backward.register_group_trajectories(0, trajectory_ids)
    for group_id, item_id, tokens, scored in events:
        run.append_turn(group_id, item_id, tokens, scored)

    def fail_fold(*_args, **_kwargs):
        raise OSError("reducer failed")

    backward._fold_in_memory_statistics = fail_fold
    backward.close_trajectory(trajectory_id, rewards[trajectory_id])
    with pytest.raises(RuntimeError, match="readiness reduction worker failed"):
        backward.wait_for_trajectory_reduction(0, trajectory_id)
    with pytest.raises(RuntimeError, match="readiness reduction worker failed"):
        backward.abort()
    assert backward._offload_threads == []
    assert backward._reducer_threads == []
    assert backward._groups == {}
    assert backward.open_groups == []
    assert backward._reducer_failure.__traceback__ is None


def test_unrestricted_offload_failure_stops_every_worker():
    trajs, groups, events, rewards = make_batch(
        n_groups=1, n_traj=1, seed=103
    )
    trajectory_id = trajs[0].traj_id
    model = build_model(STREAM_ATTENTION_NAME, seed=105)
    run = StreamingRun(
        model,
        boundary_cut=True,
        parameter_adjoint_storage_device="cpu",
    )
    backward = RewardLinearBackward(
        run,
        model,
        source_offload_device="cpu",
        source_reduction_mode="unrestricted_statistics",
    )
    run.open_group(0, groups[0])
    backward.register_group_trajectories(0, [trajectory_id])
    for group_id, item_id, tokens, scored in events:
        run.append_turn(group_id, item_id, tokens, scored)

    def fail_materialize(_sources):
        raise OSError("offload failed")

    backward._materialize_sources = fail_materialize
    backward.close_trajectory(trajectory_id, rewards[trajectory_id])
    with pytest.raises(RuntimeError, match="source offload worker failed"):
        backward.close_group(0)
    with pytest.raises(RuntimeError, match="source offload worker failed"):
        backward.abort()
    assert backward._offload_threads == []
    assert backward._reducer_threads == []
    assert backward._groups == {}
    assert backward.open_groups == []
    assert backward._offload_failure.__traceback__ is None


def test_unrestricted_group_close_rejects_registered_membership_mismatch():
    trajs, groups, events, rewards = make_batch(
        n_groups=1, n_traj=1, seed=110
    )
    trajectory_id = trajs[0].traj_id
    model = build_model(STREAM_ATTENTION_NAME, seed=112)
    run = StreamingRun(
        model,
        boundary_cut=True,
        parameter_adjoint_storage_device="cpu",
    )
    backward = RewardLinearBackward(
        run,
        model,
        source_offload_device="cpu",
        source_reduction_mode="unrestricted_statistics",
    )
    run.open_group(0, groups[0])
    backward.register_group_trajectories(0, [trajectory_id, trajectory_id + 1])
    for group_id, item_id, tokens, scored in events:
        run.append_turn(group_id, item_id, tokens, scored)
    backward.close_trajectory(trajectory_id, rewards[trajectory_id])

    with pytest.raises(RuntimeError, match="registered trajectory membership"):
        backward.close_group(0)
    backward._shutdown_workers()


def test_unrestricted_multigroup_ingress_continues_while_reducer_is_busy():
    trajs, groups, events, rewards = make_batch(
        n_groups=2, n_traj=2, seed=114
    )
    trajectory_ids = sorted(trajectory.traj_id for trajectory in trajs)
    by_group = {
        group_id: sorted(
            trajectory.traj_id
            for trajectory in trajs
            if trajectory.group_id == group_id
        )
        for group_id in groups
    }
    model = build_model(STREAM_ATTENTION_NAME, seed=116)
    run = StreamingRun(
        model,
        boundary_cut=True,
        parameter_adjoint_storage_device="cpu",
    )
    backward = RewardLinearBackward(
        run,
        model,
        source_offload_device="cpu",
        source_reduction_mode="unrestricted_statistics",
    )
    for group_id, prompt in groups.items():
        run.open_group(group_id, prompt)
        backward.register_group_trajectories(group_id, by_group[group_id])
    for group_id, item_id, tokens, scored in events:
        run.append_turn(group_id, item_id, tokens, scored)

    fold_started = threading.Event()
    release_fold = threading.Event()
    original_fold = backward._fold_in_memory_statistics
    blocked_once = False

    def blocked_fold(*args, **kwargs):
        nonlocal blocked_once
        if not blocked_once:
            blocked_once = True
            fold_started.set()
            assert release_fold.wait(timeout=5.0)
        return original_fold(*args, **kwargs)

    backward._fold_in_memory_statistics = blocked_fold
    backward.close_trajectory(trajectory_ids[0], rewards[trajectory_ids[0]])
    assert fold_started.wait(timeout=5.0)
    for trajectory_id in trajectory_ids[1:]:
        backward.close_trajectory(trajectory_id, rewards[trajectory_id])

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        report = backward.source_offload_report
        if report["source_reduction_order_by_group"].get(1):
            break
        time.sleep(0.01)
    assert sum(
        len(values)
        for values in report["source_execution_order_by_group"].values()
    ) == len(trajectory_ids)
    assert report["source_reduction_order_by_group"].get(1)
    assert report["source_reduction_worker_count"] == 2

    release_fold.set()
    for group_id in sorted(groups):
        backward.close_group(group_id)
    backward.assert_safe_to_step()


@pytest.mark.parametrize("worker_count", [False, 0, 65])
def test_source_reduction_worker_count_must_be_bounded(worker_count):
    model = build_model(STREAM_ATTENTION_NAME, seed=120)
    run = StreamingRun(model, boundary_cut=True)
    with pytest.raises(ValueError, match=r"integer in \[1, 64\]"):
        RewardLinearBackward(
            run,
            model,
            source_reduction_worker_count=worker_count,
        )


@pytest.mark.parametrize("worker_count", [False, 0, 65])
def test_source_offload_worker_count_must_be_bounded(worker_count):
    model = build_model(STREAM_ATTENTION_NAME, seed=121)
    run = StreamingRun(model, boundary_cut=True)
    with pytest.raises(ValueError, match=r"integer in \[1, 64\]"):
        RewardLinearBackward(
            run,
            model,
            source_offload_worker_count=worker_count,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA BF16")
def test_matched_readiness_order_is_reproducible_with_bfloat16_sources():
    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=4, seed=103)
    trajectory_ids = sorted(trajectory.traj_id for trajectory in trajs)

    def execute(close_order):
        model = build_model(STREAM_ATTENTION_NAME, seed=105).to(torch.bfloat16)
        run = StreamingRun(
            model,
            boundary_cut=True,
            parameter_adjoint_storage_device="cpu",
        )
        backward = RewardLinearBackward(
            run,
            model,
            eps=EPS,
            source_offload_device="cpu",
            source_reduction_mode="unrestricted_statistics",
            source_reduction_pack_size=1,
        )
        run.open_group(0, groups[0])
        backward.register_group_trajectories(0, trajectory_ids)
        for group_id, trajectory_id, tokens, scored in events:
            run.append_turn(group_id, trajectory_id, tokens, scored)
        for trajectory_id in close_order:
            backward.close_trajectory(trajectory_id, rewards[trajectory_id])
        backward.close_group(0)
        backward.assert_safe_to_step()
        return (
            [
                parameter.grad.detach().clone()
                for parameter in model.parameters()
                if parameter.grad is not None
            ],
            backward.source_offload_report,
        )

    first, first_report = execute(trajectory_ids)
    second, second_report = execute(trajectory_ids)
    assert first_report["source_reduction_order_by_group"] == {0: trajectory_ids}
    assert second_report["source_reduction_order_by_group"] == {0: trajectory_ids}
    assert len(first) == len(second)
    for expected, actual in zip(first, second):
        assert torch.equal(expected, actual)


def barrier_grads(model, trajs, rewards):
    """The monolithic reference: same advantage formula, one backward."""
    plan = compile_plan(build_forest(trajs), device=DEV)
    per = split_by_trajectory(plan, forest_logprobs(model, plan))
    loss = 0.0
    by_group: dict[int, list] = {}
    for t in trajs:
        by_group.setdefault(t.group_id, []).append(t.traj_id)
    for gid, tids in by_group.items():
        r = torch.tensor([rewards[t] for t in tids], dtype=torch.float64)
        mean, std = float(r.mean()), float(r.std(unbiased=False)) + EPS
        for t in tids:
            adv = (rewards[t] - mean) / std
            loss = loss + adv * per[t].sum()
    loss.backward()
    return {n: (p.grad.clone() if p.grad is not None else None)
            for n, p in model.named_parameters()}


# ----------------------------------------------------------------------


def test_streamed_backward_equals_barrier_backward():
    sm = build_model(STREAM_ATTENTION_NAME, seed=3)
    gm = build_model(ATTENTION_NAME, seed=3)
    gm.load_state_dict(sm.state_dict())
    trajs, groups, events, rewards = make_batch(seed=5)

    got, _, _ = reward_linear_grads(sm, trajs, groups, events, rewards)
    want = barrier_grads(gm, trajs, rewards)

    checked = 0
    for name, g in got.items():
        w = want[name]
        if g is None and w is None:
            continue
        assert g is not None and w is not None, f"{name}: one side missing grad"
        # fp32 reorder noise: the streamed path sums per-trajectory grads into
        # buffers, the barrier path sums inside one backward -- different
        # association, same tolerance the repo's other gradient tests use.
        torch.testing.assert_close(
            g, w, rtol=2e-3, atol=1e-5, msg=lambda m, n=name: f"{n}: {m}"
        )
        checked += 1
    assert checked > 5


def test_reward_arrival_order_changes_only_fp32_association_noise():
    """Readiness-order reduction stays within the frozen numerical gate."""

    _trajs, groups, events, rewards = make_batch(
        n_groups=1,
        n_traj=4,
        seed=6,
    )
    trajectory_ids = sorted({trajectory_id for _, trajectory_id, _, _ in events})

    def execute(close_order):
        model = build_model(STREAM_ATTENTION_NAME, seed=8)
        run = StreamingRun(model, boundary_cut=True)
        backward = RewardLinearBackward(run, model, eps=EPS)
        run.open_group(0, groups[0])
        backward.register_group_trajectories(0, trajectory_ids)
        for group_id, trajectory_id, tokens, scored in events:
            run.append_turn(group_id, trajectory_id, tokens, scored)
        for trajectory_id in close_order:
            backward.close_trajectory(trajectory_id, rewards[trajectory_id])
        backward.close_group(0)
        return {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }

    canonical = execute(trajectory_ids)
    reversed_arrival = execute(reversed(trajectory_ids))
    assert canonical.keys() == reversed_arrival.keys()
    for name in canonical:
        torch.testing.assert_close(
            canonical[name],
            reversed_arrival[name],
            rtol=2e-3,
            atol=1e-5,
            msg=lambda message, parameter_name=name: (
                f"{parameter_name}: {message}"
            ),
        )


def test_prompt_subgraph_backwards_once_per_group():
    """The boundary cut's purpose, observed directly via a gradient hook."""
    sm = build_model(STREAM_ATTENTION_NAME, seed=7)
    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=5, seed=11)

    run = StreamingRun(sm, boundary_cut=True)
    bw = RewardLinearBackward(run, sm)
    run.open_group(0, groups[0])
    trajectory_ids = sorted(
        {trajectory_id for _, trajectory_id, _, _ in events}
    )
    bw.register_group_trajectories(0, trajectory_ids)

    calls = []
    run.boundary(0).real[0].register_hook(lambda g: calls.append(1))

    last = {}
    for i, (gid, t_id, *_r) in enumerate(events):
        last[t_id] = i
    for i, (gid, t_id, tokens, scored) in enumerate(events):
        run.append_turn(gid, t_id, tokens, scored)
        if last[t_id] == i:
            bw.close_trajectory(t_id, rewards[t_id])
            assert not calls, (
                "prompt subgraph received gradient during a per-trajectory "
                "close; the boundary cut is not cutting"
            )
    bw.close_group(0)
    assert len(calls) == 1, f"prompt backward ran {len(calls)} times, want 1"


def test_memory_returns_per_trajectory():
    sm = build_model(STREAM_ATTENTION_NAME, seed=13)
    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=4, seed=17)
    run = StreamingRun(sm, boundary_cut=True)
    bw = RewardLinearBackward(run, sm)
    run.open_group(0, groups[0])
    order = sorted({t for _, t, _, _ in events})
    bw.register_group_trajectories(0, order)
    for gid, t_id, tokens, scored in events:
        run.append_turn(gid, t_id, tokens, scored)

    before = run.kv_bytes_resident()
    bw.close_trajectory(order[0], rewards[order[0]])
    after_one = run.kv_bytes_resident()
    assert after_one < before, "closing a trajectory did not release its K/V"

    for t in order[1:]:
        bw.close_trajectory(t, rewards[t])
    bw.close_group(0)
    assert run.kv_bytes_resident() == 0, "group close left K/V resident"


def test_optimizer_step_with_open_groups_fails_loudly():
    sm = build_model(STREAM_ATTENTION_NAME, seed=19)
    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=2, seed=23)
    run = StreamingRun(sm, boundary_cut=True)
    bw = RewardLinearBackward(run, sm)
    run.open_group(0, groups[0])
    for gid, t_id, tokens, scored in events[:2]:
        run.append_turn(gid, t_id, tokens, scored)

    with pytest.raises(RuntimeError, match="open groups"):
        bw.assert_safe_to_step()

    # simulate the corruption the guard exists for: step in place, then close
    opt = torch.optim.SGD(sm.parameters(), lr=1e-3)
    for p in sm.parameters():
        p.grad = torch.zeros_like(p)
    opt.step()
    first = sorted({t for _, t, _, _ in events})[0]
    with pytest.raises(RuntimeError, match="in place"):
        bw.close_trajectory(first, rewards[first])


def test_double_close_is_an_error():
    sm = build_model(STREAM_ATTENTION_NAME, seed=29)
    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=2, seed=31)
    run = StreamingRun(sm, boundary_cut=True)
    bw = RewardLinearBackward(run, sm)
    run.open_group(0, groups[0])
    for gid, t_id, tokens, scored in events:
        run.append_turn(gid, t_id, tokens, scored)
    tids = sorted({t for _, t, _, _ in events})
    bw.close_trajectory(tids[0], 1.0)
    with pytest.raises(RuntimeError, match="already closed"):
        bw.close_trajectory(tids[0], 1.0)
    with pytest.raises(RuntimeError, match="not closed"):
        bw.close_group(0)


def test_boundary_cut_required():
    sm = build_model(STREAM_ATTENTION_NAME, seed=37)
    run = StreamingRun(sm)  # no cut
    with pytest.raises(ValueError, match="boundary_cut"):
        RewardLinearBackward(run, sm)


def test_pinned_nonblocking_source_offload_requires_cpu_destination():
    sm = build_model(STREAM_ATTENTION_NAME, seed=39)
    run = StreamingRun(sm, boundary_cut=True)
    with pytest.raises(ValueError, match="requires CPU destination"):
        RewardLinearBackward(
            run,
            sm,
            source_offload_device="none",
            source_offload_mode="pinned_nonblocking",
        )
    with pytest.raises(ValueError, match="must be blocking"):
        RewardLinearBackward(
            run,
            sm,
            source_offload_device="cpu",
            source_offload_mode="unsupported",
        )
    with pytest.raises(ValueError, match="positive"):
        RewardLinearBackward(
            run,
            sm,
            source_offload_device="cpu",
            source_reduction_pack_size=True,
        )
    with pytest.raises(ValueError, match="8-byte-aligned integer"):
        RewardLinearBackward(
            run,
            sm,
            source_offload_device="cpu",
            source_offload_mode="pinned_streaming",
            source_offload_window_bytes=0,
        )
    with pytest.raises(ValueError, match="8-byte-aligned integer"):
        RewardLinearBackward(
            run,
            sm,
            source_offload_device="cpu",
            source_offload_mode="pinned_streaming",
            source_offload_window_bytes=True,
        )
    with pytest.raises(ValueError, match="8-byte-aligned integer"):
        RewardLinearBackward(
            run,
            sm,
            source_offload_device="cpu",
            source_offload_mode="pinned_streaming",
            source_offload_window_bytes=2 * 1024 * 1024 * 1024,
        )
    with pytest.raises(ValueError, match="hbm_threshold must be in"):
        RewardLinearBackward(
            run,
            sm,
            source_offload_device="cpu",
            source_offload_mode="pinned_streaming",
            source_offload_hbm_threshold=True,
        )


def test_source_ready_grpo_drains_group_barriers_and_zero_group() -> None:
    """GRPO exposes one exact source after its group statistics are ready."""
    _trajs, groups, events, rewards = make_batch(
        n_groups=2,
        n_traj=2,
        prompt_len=16,
        seed=41,
    )
    common = list(groups[0][:8])
    groups = {
        group_id: common + list(prompt[8:])
        for group_id, prompt in groups.items()
    }
    group_zero_trajectories = sorted(
        trajectory_id
        for group_id, trajectory_id, _tokens, _scored in events
        if group_id == 0
    )
    for trajectory_id in group_zero_trajectories:
        rewards[trajectory_id] = 0.5

    def execute(*, source_ready: bool) -> tuple[dict[str, torch.Tensor], dict]:
        model = build_model(STREAM_ATTENTION_NAME, seed=43)
        run = StreamingRun(
            model,
            boundary_cut=True,
            shared_prefix_vjp_mode=(
                "source_ready_serial" if source_ready else "aggregate"
            ),
        )
        objective = RewardLinearBackward(run, model, eps=EPS)
        run.open_groups_shared_prefix(groups)
        for group_id in sorted(groups):
            objective.register_group_trajectories(
                group_id,
                sorted(
                    {
                        trajectory_id
                        for event_group_id, trajectory_id, _tokens, _scored in events
                        if event_group_id == group_id
                    }
                ),
            )
        last_event = {
            trajectory_id: index
            for index, (
                _group_id,
                trajectory_id,
                _tokens,
                _scored,
            ) in enumerate(events)
        }
        for index, (group_id, trajectory_id, tokens, scored) in enumerate(
            events
        ):
            run.append_turn(group_id, trajectory_id, tokens, scored)
            if last_event[trajectory_id] == index:
                objective.close_trajectory(
                    trajectory_id,
                    rewards[trajectory_id],
                )
        for group_id in sorted(groups):
            objective.close_group(group_id)
        objective.assert_safe_to_step()
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        return gradients, run.shared_prefix_finalization_diagnostics()

    aggregate_gradients, aggregate_diagnostics = execute(source_ready=False)
    source_gradients, source_diagnostics = execute(source_ready=True)
    assert aggregate_gradients.keys() == source_gradients.keys()
    for name in aggregate_gradients:
        torch.testing.assert_close(
            source_gradients[name],
            aggregate_gradients[name],
            rtol=2e-5,
            atol=2e-6,
            msg=lambda message, parameter_name=name: (
                f"{parameter_name}: {message}"
            ),
        )

    assert aggregate_diagnostics[
        "source_process_accounting_applicable"
    ] is False
    assert source_diagnostics["source_process_count"] == 2
    assert source_diagnostics["expected_source_process_count"] == 2
    assert source_diagnostics["source_vjp_count"] == 2
    assert source_diagnostics["group_process_count"] == 2
    assert source_diagnostics["source_ready_sources_by_group"] == {
        0: [("group", 0)],
        1: [("group", 1)],
    }
    assert len(source_diagnostics["source_vjp_events"]) == 2
    assert sum(
        event["vjp_invoked"]
        for event in source_diagnostics["source_vjp_events"]
    ) == 2
    assert source_diagnostics["pending_nodes"] == 0


def test_branch_phase_counters_split_rebuild_from_backward():
    """Trainer-side phase accounting must separate rebuild from backward.

    Trainer MFU is reported as one aggregate forward/backward span.
    Without this split a low measured Trainer MFU cannot be attributed to the
    branch re-forward, the branch backward, or host source service.
    """

    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=4, seed=4242)
    model = build_model(STREAM_ATTENTION_NAME, seed=4242)

    _gradients, _run, backward = reward_linear_grads(
        model, trajs, groups, events, rewards, source_offload_device="cpu"
    )

    report = backward.source_offload_report
    assert 1 <= report["branch_rebuild_count"] <= 4
    assert 1 <= report["branch_backward_count"] <= 4
    assert report["branch_rebuild_s"] > 0.0
    assert report["branch_backward_s"] > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_streaming_chunk_counters_split_copy_wait_from_fold():
    """Streamed source service must separate D2H wait from host fold.

    ``hbm_stage_s`` currently reports one span covering submit, transfer
    completion, and the FP32 statistic fold.  A capacity decision needs the
    transfer share and the host-arithmetic share as separate evidence.
    """

    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=4, seed=4243)
    model = build_model(STREAM_ATTENTION_NAME, seed=4243).cuda()

    _gradients, _run, backward = reward_linear_grads(
        model,
        trajs,
        groups,
        events,
        rewards,
        source_offload_device="cpu",
        source_offload_mode="pinned_streaming",
        source_offload_window_bytes=4096,
        source_offload_hbm_threshold=0.01,
    )

    report = backward.source_offload_report
    assert report["streaming_chunk_count"] > 0
    assert report["streaming_chunk_copy_wait_s"] > 0.0
    assert report["streaming_chunk_fold_s"] > 0.0
    assert (
        report["streaming_chunk_copy_wait_s"] <= report["source_offload_wait_s"]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_streaming_chunk_pipeline_overlaps_copy_with_fold():
    """One streamed chunk must copy while the previous chunk folds.

    A single pinned window per worker serializes transfer and host arithmetic:
    the next chunk's D2H cannot start until the previous chunk's FP32 fold
    releases the window.  The worker therefore needs a two-window pipeline.
    """

    window_bytes = 4096
    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=2, seed=4244)
    model = build_model(STREAM_ATTENTION_NAME, seed=4244).cuda()

    _gradients, _run, backward = reward_linear_grads(
        model,
        trajs,
        groups,
        events,
        rewards,
        source_offload_device="cpu",
        source_offload_mode="pinned_streaming",
        source_offload_window_bytes=window_bytes,
        source_offload_hbm_threshold=0.01,
        source_offload_worker_count=1,
        source_reduction_worker_count=1,
    )

    report = backward.source_offload_report
    assert report["streaming_chunk_count"] > 2
    assert report["peak_source_offload_inflight_bytes"] >= 2 * window_bytes
    assert report["streaming_pinned_buffer_reserved_bytes"] >= 2 * window_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_streaming_pinned_windows_are_reserved_before_the_first_source():
    """Pinned windows must not be allocated inside a measured branch.

    ``cudaHostAlloc`` of a 256 MiB window costs order 100 ms.  Allocating it
    lazily during the first fold charges that cost to the branch backward and
    to every timed source-service window.
    """

    model = torch.nn.Linear(4, 4, bias=False).cuda()

    class Run:
        boundary_cut = True

    window_bytes = 8192
    backward = RewardLinearBackward(
        Run(),
        model,
        source_offload_device="cpu",
        source_offload_mode="pinned_streaming",
        source_offload_window_bytes=window_bytes,
        source_offload_hbm_threshold=0.5,
        source_reduction_worker_count=2,
        source_offload_worker_count=3,
    )
    try:
        report = backward.source_offload_report
        deadline = time.perf_counter() + 5.0
        while (
            report["streaming_pinned_buffer_reserved_bytes"]
            < report["streaming_pinned_buffer_capacity_bytes"]
            and time.perf_counter() < deadline
        ):
            time.sleep(0.01)
            report = backward.source_offload_report
        assert (
            report["streaming_pinned_buffer_reserved_bytes"]
            == report["streaming_pinned_buffer_capacity_bytes"]
        )
    finally:
        backward._shutdown_workers()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_streaming_fold_converts_the_chunk_once_per_statistic_pair():
    """The fold must not accumulate a bfloat16 chunk into fp32 statistics.

    ATen's mixed-dtype accumulate is unvectorized.  Converting the chunk once
    and accumulating in one dtype measured 13.2 ms against 63.9 ms for a 58 MB
    chunk with two statistics on one host CPU.  bfloat16 to float32 is
    exact, so the accumulated statistic is unchanged.
    """

    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=4, seed=4245)
    reference_model = (
        build_model(STREAM_ATTENTION_NAME, seed=4245).cuda().to(torch.bfloat16)
    )
    candidate_model = (
        build_model(STREAM_ATTENTION_NAME, seed=4245).cuda().to(torch.bfloat16)
    )

    reference, _reference_run, _reference_bw = reward_linear_grads(
        reference_model, trajs, groups, events, rewards,
        source_offload_device="cpu", source_offload_mode="pinned_nonblocking",
    )
    candidate, _candidate_run, candidate_bw = reward_linear_grads(
        candidate_model, trajs, groups, events, rewards,
        source_offload_device="cpu", source_offload_mode="pinned_streaming",
        source_offload_window_bytes=4096, source_offload_hbm_threshold=0.01,
    )

    for name in reference:
        torch.testing.assert_close(
            candidate[name], reference[name], rtol=2e-2, atol=1e-3, msg=name
        )
    report = candidate_bw.source_offload_report
    assert report["streaming_fold_conversion_bytes"] > 0
    assert report["streaming_fold_buffer_reserved_bytes"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_group_close_combines_statistics_on_their_own_device():
    """The group AXPY must not move both statistics across the bus.

    Needs the parameters on a device other than the statistics' host: the
    combine counts only statistics whose device differs from the
    parameter's, so on CPU-only torch there is nothing to combine off
    device and the byte count is legitimately zero (the node CPU pass at
    7ed615c, CUDA hidden, failed here on exactly that).

    ``close_group`` evaluates ``(G1 - mean * G2) / std``.  Reloading both
    model-sized fp32 statistics to the parameter device moves twice the bytes
    the result needs.  Combining on the statistic's own device and reloading
    only the result halves the transfer, and the arithmetic is unchanged.
    """

    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=4, seed=4246)
    reference_model = build_model(STREAM_ATTENTION_NAME, seed=4246)
    candidate_model = build_model(STREAM_ATTENTION_NAME, seed=4246)

    reference, _reference_run, _reference_bw = reward_linear_grads(
        reference_model, trajs, groups, events, rewards
    )
    candidate, _candidate_run, candidate_bw = reward_linear_grads(
        candidate_model, trajs, groups, events, rewards,
        source_offload_device="cpu",
    )

    for name in reference:
        torch.testing.assert_close(
            candidate[name], reference[name], rtol=2e-3, atol=1e-5, msg=name
        )
    report = candidate_bw.source_offload_report
    assert report["group_close_host_combined_bytes"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compiled_pointwise_modules_preserve_the_reward_linear_gradient():
    """Compiling the MLP and norms must not change the learner gradient.

    The attention path stays eager, so the streaming executor is unchanged.
    Measured on one GPU, the compiled decoder MLP is 7.92 ms
    against 15.66 ms for forward plus backward at the branch shape and the
    norm 0.32 ms against 1.34 ms.
    """

    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=2, seed=4248)
    reference_model = build_model(STREAM_ATTENTION_NAME, seed=4248).cuda()
    candidate_model = build_model(STREAM_ATTENTION_NAME, seed=4248).cuda()
    for layer in candidate_model.model.layers:
        for name in ("mlp", "input_layernorm", "post_attention_layernorm"):
            module = getattr(layer, name, None)
            if module is not None:
                module.forward = torch.compile(module.forward, dynamic=True)
    assert set(candidate_model.state_dict()) == set(
        reference_model.state_dict()
    ), "compiling the bound forward must not rename parameters"
    try:
        candidate_model.model.layers[0].input_layernorm(
            torch.zeros(
                1, 2, candidate_model.config.hidden_size,
                device="cuda", dtype=torch.bfloat16,
            )
        )
    except (ImportError, RuntimeError, OSError) as error:
        # Inductor writes and executes generated modules from its cache
        # directory; a `noexec` cache mount makes compilation unavailable.
        pytest.skip(f"torch.compile is unavailable here: {error}")

    reference, _reference_run, _reference_bw = reward_linear_grads(
        reference_model, trajs, groups, events, rewards,
        source_offload_device="cpu",
    )
    candidate, _candidate_run, _candidate_bw = reward_linear_grads(
        candidate_model, trajs, groups, events, rewards,
        source_offload_device="cpu",
    )

    checked = 0
    for name in reference:
        want, got = reference[name], candidate[name]
        if want is None and got is None:
            continue
        torch.testing.assert_close(got, want, rtol=2e-2, atol=1e-4, msg=name)
        checked += 1
    assert checked > 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_branch_phases_report_device_time_beside_host_time():
    """Phase counters must separate device work from host wall time.

    `branch_rebuild_s`, `branch_backward_s` and `group_close_boundary_s` are
    host wall times around calls that synchronize, so on a busy queue they
    absorb device work issued earlier.  CUDA-event spans over the same code
    give the device side of the same phases.
    """

    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=4, seed=4250)
    model = build_model(STREAM_ATTENTION_NAME, seed=4250).cuda()

    _gradients, _run, backward = reward_linear_grads(
        model, trajs, groups, events, rewards, source_offload_device="cpu"
    )

    report = backward.source_offload_report
    for name in (
        "branch_rebuild_device_s",
        "branch_backward_device_s",
        "group_close_boundary_device_s",
    ):
        assert name in report, name
        assert report[name] >= 0.0, name
    assert report["branch_rebuild_device_s"] > 0.0
    assert report["branch_backward_device_s"] > 0.0


def _reward_linear_grads_batched_closes(model, trajs, groups, events, rewards):
    """Close every trajectory that landed in the same arrival batch at once.

    ``close_trajectories`` takes several already-ready closes.  Nothing waits
    to fill the batch: the caller hands over exactly the rewards that had
    landed by the time the trainer was free, which is what the clustered
    straggler window produces.
    """

    run = StreamingRun(model, boundary_cut=True)
    bw = RewardLinearBackward(run, model, eps=EPS)
    for gid, prompt in groups.items():
        run.open_group(gid, prompt)
        bw.register_group_trajectories(
            gid,
            sorted(
                trajectory.traj_id
                for trajectory in trajs
                if trajectory.group_id == gid
            ),
        )
    last_event = {}
    for i, (gid, t_id, *_rest) in enumerate(events):
        last_event[t_id] = i
    landed: list[tuple[int, float]] = []
    for i, (gid, t_id, tokens, scored) in enumerate(events):
        run.append_turn(gid, t_id, tokens, scored)
        if last_event[t_id] == i:
            landed.append((t_id, rewards[t_id]))
    bw.close_trajectories(landed)
    for gid in groups:
        bw.close_group(gid)
    bw.assert_safe_to_step()
    return {
        n: (p.grad.clone() if p.grad is not None else None)
        for n, p in model.named_parameters()
    }, run, bw


def test_batched_closes_pack_one_rebuild_and_keep_the_reward_linear_gradient():
    """Several already-ready closes must share ONE branch rebuild.

    ``close_trajectories`` documents that with several closes "each block
    round packs all branches into one primal".  A per-close rebuild pays the
    block-checkpointed re-forward k times over k small branches, which is the
    dominant branch cost measured on the reference site.  The backward stays per branch --
    G1 needs ``r_i * grad(S_i)`` and G2 needs ``grad(S_i)`` separately -- so
    packing changes the kernel schedule and never the graph walk count.
    """

    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=4, seed=771)
    serial_model = build_model(STREAM_ATTENTION_NAME, seed=771)
    packed_model = build_model(STREAM_ATTENTION_NAME, seed=771)

    serial, _serial_run, serial_bw = reward_linear_grads(
        serial_model, trajs, groups, events, rewards
    )
    packed, _packed_run, packed_bw = _reward_linear_grads_batched_closes(
        packed_model, trajs, groups, events, rewards
    )

    serial_report = serial_bw.source_offload_report
    packed_report = packed_bw.source_offload_report
    assert serial_report["branch_rebuild_count"] == 4
    assert packed_report["branch_rebuild_count"] == 1
    assert packed_report["branch_backward_count"] == 4

    for name, reference in serial.items():
        candidate = packed[name]
        if reference is None:
            assert candidate is None
            continue
        assert candidate is not None
        torch.testing.assert_close(candidate, reference, rtol=2e-4, atol=2e-5)


def test_report_publishes_the_widest_rebuild_pack_actually_executed():
    """``source_reduction_pack_size`` alone cannot describe a packed run.

    That field is a declared admission invariant validated in the constructor;
    nothing reads it at runtime.  A report carrying only ``pack_size: 1``
    therefore reads as one-source admission even when several already-landed
    closes shared one branch rebuild.  Publish the widest pack the trainer
    actually executed so the two are separable in the record.
    """

    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=4, seed=772)

    _serial, _serial_run, serial_bw = reward_linear_grads(
        build_model(STREAM_ATTENTION_NAME, seed=772),
        trajs,
        groups,
        events,
        rewards,
    )
    _packed, _packed_run, packed_bw = _reward_linear_grads_batched_closes(
        build_model(STREAM_ATTENTION_NAME, seed=772),
        trajs,
        groups,
        events,
        rewards,
    )

    assert serial_bw.source_offload_report["source_reduction_pack_size"] == 1
    assert packed_bw.source_offload_report["source_reduction_pack_size"] == 1
    assert serial_bw.source_offload_report["executed_rebuild_pack_max"] == 1
    assert packed_bw.source_offload_report["executed_rebuild_pack_max"] == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_streaming_pinned_window_depth_is_configurable():
    """Pipeline depth must be settable above the two-window minimum.

    Two windows let one chunk's D2H overlap the previous chunk's fold, which
    keeps exactly one transfer outstanding per worker.  When the host path is
    contended -- two trainer ranks staging model-sized sources through it --
    one outstanding transfer is not enough to keep the copy engine fed, the
    drain falls behind, and the sources stay resident in HBM.
    """

    model = torch.nn.Linear(4, 4, bias=False).cuda()

    class Run:
        boundary_cut = True

    window_bytes = 8192
    backward = RewardLinearBackward(
        Run(),
        model,
        source_offload_device="cpu",
        source_offload_mode="pinned_streaming",
        source_offload_window_bytes=window_bytes,
        source_offload_hbm_threshold=0.5,
        source_reduction_worker_count=2,
        source_offload_worker_count=3,
        source_offload_pinned_windows=5,
    )
    try:
        report = backward.source_offload_report
        assert report["source_offload_pinned_windows"] == 5
        assert (
            report["streaming_pinned_buffer_capacity_bytes"]
            == window_bytes * 5 * (3 + 2)
        )
    finally:
        backward._shutdown_workers()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deeper_streaming_pipeline_preserves_the_reward_linear_gradient():
    """Extra windows change the transfer schedule and nothing else.

    Chunks still fold in offset order into disjoint ranges of the two
    statistics, so a deeper pipeline cannot reorder the arithmetic.
    """

    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=4, seed=881)
    shallow_model = build_model(STREAM_ATTENTION_NAME, seed=881).cuda()
    deep_model = build_model(STREAM_ATTENTION_NAME, seed=881).cuda()

    shallow, _shallow_run, _shallow_bw = reward_linear_grads(
        shallow_model, trajs, groups, events, rewards,
        source_offload_device="cpu",
        source_offload_mode="pinned_streaming",
        source_offload_window_bytes=4096,
        source_offload_hbm_threshold=0.01,
    )
    deep, _deep_run, deep_bw = reward_linear_grads(
        deep_model, trajs, groups, events, rewards,
        source_offload_device="cpu",
        source_offload_mode="pinned_streaming",
        source_offload_window_bytes=4096,
        source_offload_hbm_threshold=0.01,
        source_offload_pinned_windows=6,
    )

    assert deep_bw.source_offload_report["source_offload_pinned_windows"] == 6
    assert deep_bw.source_offload_report["streaming_chunk_count"] > 0

    for name, reference in shallow.items():
        candidate = deep[name]
        if reference is None:
            assert candidate is None, name
            continue
        assert candidate is not None, name
        torch.testing.assert_close(candidate, reference, rtol=2e-4, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_device_resident_statistics_preserve_the_reward_linear_gradient():
    """G1/G2 on the parameter device must change nothing the gradient sees.

    Host-resident statistics force every branch's model-sized source through
    one host path: the registered run moves 3.97 TB per rank per step, two
    ranks contend for the path (branch device time 1.83x), and the sources
    occupy 77.3 GB of HBM while they drain.  Folding into device-resident
    fp32 statistics in the same execution order removes the transport; the
    arithmetic and its order are unchanged.
    """

    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=4, seed=995)
    host_model = build_model(STREAM_ATTENTION_NAME, seed=995).cuda()
    device_model = build_model(STREAM_ATTENTION_NAME, seed=995).cuda()

    host, _host_run, _host_bw = reward_linear_grads(
        host_model, trajs, groups, events, rewards,
        source_offload_device="cpu",
        source_offload_mode="pinned_streaming",
        source_offload_window_bytes=4096,
        source_offload_hbm_threshold=0.01,
    )
    device, _device_run, device_bw = reward_linear_grads(
        device_model, trajs, groups, events, rewards,
        source_offload_device="cpu",
        source_offload_mode="pinned_streaming",
        source_offload_window_bytes=4096,
        source_offload_hbm_threshold=0.01,
        statistic_device="cuda",
    )

    report = device_bw.source_offload_report
    assert report["statistic_device"] == "cuda"

    for name, reference in host.items():
        candidate = device[name]
        if reference is None:
            assert candidate is None, name
            continue
        assert candidate is not None, name
        torch.testing.assert_close(candidate, reference, rtol=2e-4, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_device_resident_statistics_stage_no_bytes_through_the_host():
    """With device statistics the host staging path must stay untouched."""

    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=4, seed=997)
    model = build_model(STREAM_ATTENTION_NAME, seed=997).cuda()

    _grads, _run, backward = reward_linear_grads(
        model, trajs, groups, events, rewards,
        source_offload_device="cpu",
        source_offload_mode="pinned_streaming",
        source_offload_window_bytes=4096,
        source_offload_hbm_threshold=0.01,
        statistic_device="cuda",
    )

    report = backward.source_offload_report
    assert report["streaming_chunk_count"] == 0
    assert report["streaming_chunk_copy_wait_s"] == 0.0
    assert report["streaming_chunk_fold_s"] == 0.0
    assert report["streaming_pinned_buffer_reserved_bytes"] == 0
    assert report["stored_source_bytes"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_device_resident_statistics_fold_in_execution_order():
    """Reversed closes must reduce in their own arrival order, bit-stably.

    The determinism contract is per close order: one order always produces
    one bit pattern, and the recorded reduction order equals the execution
    order.  Device residence changes where the fold runs and nothing else.
    """

    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=4, seed=999)
    trajectory_ids = sorted(trajectory.traj_id for trajectory in trajs)

    def execute(close_order):
        model = build_model(STREAM_ATTENTION_NAME, seed=999).cuda()
        run = StreamingRun(model, boundary_cut=True)
        backward = RewardLinearBackward(
            run,
            model,
            eps=EPS,
            source_offload_device="cpu",
            source_offload_mode="pinned_streaming",
            source_offload_window_bytes=4096,
            source_offload_hbm_threshold=0.01,
            statistic_device="cuda",
        )
        run.open_group(0, groups[0])
        backward.register_group_trajectories(0, trajectory_ids)
        for group_id, trajectory_id, tokens, scored in events:
            run.append_turn(group_id, trajectory_id, tokens, scored)
        for trajectory_id in close_order:
            backward.close_trajectory(trajectory_id, rewards[trajectory_id])
        backward.close_group(0)
        backward.assert_safe_to_step()
        return {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }, backward.source_offload_report

    reversed_ids = list(reversed(trajectory_ids))
    first, first_report = execute(reversed_ids)
    second, _second_report = execute(reversed_ids)

    assert first_report["source_reduction_order_by_group"] == {0: reversed_ids}
    for name in first:
        assert torch.equal(first[name], second[name]), name




def test_shutdown_joins_every_worker_before_reporting_one_that_did_not_stop():
    """A worker that outlives its join must not leave the others running."""

    class StuckThread:
        name = "stuck-offload"
        joined = False

        def join(self, timeout=None):
            self.joined = True

        def is_alive(self):
            return True

    model = torch.nn.Linear(1, 1)

    class Run:
        boundary_cut = True

    backward = RewardLinearBackward(Run(), model)
    real_offload = backward._offload_threads
    reducers = list(backward._reducer_threads)
    stuck = StuckThread()
    backward._offload_threads = [stuck, *real_offload]

    with pytest.raises(RuntimeError, match="stuck-offload"):
        backward._shutdown_workers()

    assert stuck.joined
    assert all(not thread.is_alive() for thread in [*real_offload, *reducers])
    assert backward._offload_threads == [stuck]
    assert backward._reducer_threads == []


def _one_group_gradients(single_call: bool) -> list[torch.Tensor]:
    model = build_model(STREAM_ATTENTION_NAME, vocab=64, seed=5)
    run = StreamingRun(model, boundary_cut=True)
    objective = RewardLinearBackward(run, model, eps=EPS, source_offload_device="cpu")
    if single_call:
        objective.open_group(0, [1, 2, 3], [0, 1])
    else:
        run.open_group(0, [1, 2, 3])
        objective.register_group_trajectories(0, [0, 1])
    run.append_turn(0, 0, [4, 5], [True, True])
    objective.close_trajectory(0, 0.0)
    run.append_turn(0, 1, [6, 7, 8], [True, True, True])
    objective.close_trajectory(1, 1.0)
    objective.close_group(0)
    objective.assert_safe_to_step()
    return [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]


def test_open_group_equals_run_open_group_plus_registration():
    single, paired = _one_group_gradients(True), _one_group_gradients(False)
    assert single and len(single) == len(paired)
    assert all(torch.equal(a, b) for a, b in zip(single, paired, strict=True))


def test_open_group_refuses_bad_trajectory_ids_before_opening_the_group():
    model = build_model(STREAM_ATTENTION_NAME, vocab=64, seed=5)
    run = StreamingRun(model, boundary_cut=True)
    objective = RewardLinearBackward(run, model, source_offload_device="cpu")
    for trajectory_ids in ([], [0, 0]):
        with pytest.raises(ValueError, match="nonempty and unique"):
            objective.open_group(0, [1, 2, 3], trajectory_ids)
    assert not run.open_group_ids()
