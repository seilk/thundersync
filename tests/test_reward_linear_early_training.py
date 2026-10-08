"""Early-training gates: prepare at content-readiness, bind rewards at verdict.

The split must be an exact decomposition of the legacy close path. A branch
backward never reads the reward -- only the readiness-order fold consumes the
scalar -- so `close_trajectories == prepare_trajectories + bind_rewards` must
hold bitwise: same source tensors, same fold code, same per-group fold order.

Guarded failure modes:

* a fold that starts before the reward is bound (peeking at the prebound
  grade would break the measurement semantics);
* a reduction order that follows preparation order instead of the recorded
  objective-readiness (reward) order;
* unbounded retention: prepared-but-unbound sources must be tracked and must
  return to zero;
* an optimizer step between prepare and bind silently invalidating sources;
* pinned_streaming composing with early preparation (its chunk folds consume
  the reward scalar during the backward, so the combination is illegal).
"""

from __future__ import annotations

import contextlib
import sys
import threading
from pathlib import Path

import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tests.test_reward_linear import EPS, build_model, make_batch, reward_linear_grads  # noqa: E402

from thundersync.grpo.reward_linear import RewardLinearBackward  # noqa: E402
from thundersync.engine.streaming import (  # noqa: E402
    STREAM_ATTENTION_NAME,
    StreamingRun,
)


def _setup(model, trajs, groups, **kwargs):
    run = StreamingRun(model, boundary_cut=True)
    bw = RewardLinearBackward(
        run,
        model,
        eps=EPS,
        source_offload_device="cpu",
        **kwargs,
    )
    for gid, prompt in groups.items():
        run.open_group(gid, prompt)
        bw.register_group_trajectories(
            gid,
            sorted(t.traj_id for t in trajs if t.group_id == gid),
        )
    return run, bw


def _completion_order(events):
    last_event = {}
    for index, (_gid, t_id, *_rest) in enumerate(events):
        last_event[t_id] = index
    order = []
    for index, (_gid, t_id, *_rest) in enumerate(events):
        if last_event[t_id] == index:
            order.append(t_id)
    return last_event, order


def early_reward_linear_grads(
    model,
    trajs,
    groups,
    events,
    rewards,
    *,
    bind_order=None,
    close_groups=True,
    **kwargs,
):
    """Prepare each branch at its own completion; bind rewards afterwards."""

    run, bw = _setup(model, trajs, groups, **kwargs)
    last_event, completion = _completion_order(events)
    for index, (gid, t_id, tokens, scored) in enumerate(events):
        run.append_turn(gid, t_id, tokens, scored)
        if last_event[t_id] == index:
            bw.prepare_trajectory(t_id)
    for t_id in bind_order if bind_order is not None else completion:
        bw.bind_reward(t_id, rewards[t_id])
    if close_groups:
        for gid in groups:
            bw.close_group(gid)
        bw.assert_safe_to_step()
    grads = {
        name: (parameter.grad.clone() if parameter.grad is not None else None)
        for name, parameter in model.named_parameters()
    }
    return grads, run, bw


def legacy_deferred_close_grads(
    model, trajs, groups, events, rewards, *, close_order
):
    """Legacy path with every close deferred to a caller-chosen reward order."""

    run, bw = _setup(model, trajs, groups)
    for gid, t_id, tokens, scored in events:
        run.append_turn(gid, t_id, tokens, scored)
    for t_id in close_order:
        bw.close_trajectory(t_id, rewards[t_id])
    for gid in groups:
        bw.close_group(gid)
    bw.assert_safe_to_step()
    grads = {
        name: (parameter.grad.clone() if parameter.grad is not None else None)
        for name, parameter in model.named_parameters()
    }
    return grads, run, bw


def test_prepare_then_bind_equals_close_path_exactly():
    trajs, groups, events, rewards = make_batch(n_groups=2, n_traj=3, seed=311)
    legacy_model = build_model(STREAM_ATTENTION_NAME, seed=312)
    early_model = build_model(STREAM_ATTENTION_NAME, seed=312)

    legacy, _run, _bw = reward_linear_grads(legacy_model, trajs, groups, events, rewards)
    early, _early_run, early_bw = early_reward_linear_grads(
        early_model, trajs, groups, events, rewards
    )

    for name in legacy:
        if legacy[name] is None:
            assert early[name] is None, name
            continue
        assert torch.equal(legacy[name], early[name]), name
    report = early_bw.source_offload_report
    assert report["current_source_bytes"] == 0
    assert report["prepared_awaiting_reward_current"] == 0


def test_out_of_order_binding_reduces_in_reward_order():
    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=4, seed=321)
    _last_event, completion = _completion_order(events)
    reward_order = list(reversed(completion))
    legacy_model = build_model(STREAM_ATTENTION_NAME, seed=322)
    early_model = build_model(STREAM_ATTENTION_NAME, seed=322)

    legacy, _run, legacy_bw = legacy_deferred_close_grads(
        legacy_model, trajs, groups, events, rewards, close_order=reward_order
    )
    early, _early_run, early_bw = early_reward_linear_grads(
        early_model, trajs, groups, events, rewards, bind_order=reward_order
    )

    for name in legacy:
        if legacy[name] is None:
            assert early[name] is None, name
            continue
        assert torch.equal(legacy[name], early[name]), name
    early_report = early_bw.source_offload_report
    assert early_report["source_reduction_order_by_group"] == {0: reward_order}
    assert early_report["source_execution_order_by_group"] == {0: reward_order}
    assert early_report["source_preparation_order_by_group"] == {0: completion}
    legacy_report = legacy_bw.source_offload_report
    assert legacy_report["source_reduction_order_by_group"] == {0: reward_order}


def test_prepare_computes_sources_without_folding_before_reward():
    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=3, seed=331)
    model = build_model(STREAM_ATTENTION_NAME, seed=332)
    run, bw = _setup(model, trajs, groups)
    last_event, completion = _completion_order(events)
    for index, (gid, t_id, tokens, scored) in enumerate(events):
        run.append_turn(gid, t_id, tokens, scored)
        if last_event[t_id] == index:
            bw.prepare_trajectory(t_id)

    report = bw.source_offload_report
    assert report["prepared_awaiting_reward_current"] == len(completion)
    assert report["prepared_awaiting_reward_peak"] == len(completion)
    # Branch backwards ran (sources stored) but nothing folded: no statistic
    # bytes exist and no reduction order was recorded.
    assert report["current_source_bytes"] > 0
    assert report["current_statistic_bytes"] == 0
    assert report["source_reduction_order_by_group"] == {}
    assert report["source_execution_order_by_group"] == {}
    assert report["source_preparation_order_by_group"] == {0: completion}

    for t_id in completion:
        bw.bind_reward(t_id, rewards[t_id])
    for gid in groups:
        bw.close_group(gid)
    bw.assert_safe_to_step()
    report = bw.source_offload_report
    assert report["prepared_awaiting_reward_current"] == 0
    assert report["current_source_bytes"] == 0
    assert report["current_statistic_bytes"] == 0


def test_verifier_straggler_holds_only_its_own_group():
    trajs, groups, events, rewards = make_batch(n_groups=2, n_traj=3, seed=341)
    model = build_model(STREAM_ATTENTION_NAME, seed=342)
    run, bw = _setup(model, trajs, groups)
    last_event, completion = _completion_order(events)
    for index, (gid, t_id, tokens, scored) in enumerate(events):
        run.append_turn(gid, t_id, tokens, scored)
        if last_event[t_id] == index:
            bw.prepare_trajectory(t_id)

    group_of = {t.traj_id: t.group_id for t in trajs}
    group_zero = [t_id for t_id in completion if group_of[t_id] == 0]
    group_one = [t_id for t_id in completion if group_of[t_id] == 1]
    straggler = group_one[-1]

    for t_id in group_zero:
        bw.bind_reward(t_id, rewards[t_id])
    for t_id in group_one[:-1]:
        bw.bind_reward(t_id, rewards[t_id])

    # Group 0 closes while group 1's verifier straggler is still unbound.
    bw.close_group(0)
    with pytest.raises(RuntimeError, match="not closed"):
        bw.close_group(1)

    bw.bind_reward(straggler, rewards[straggler])
    bw.close_group(1)
    bw.assert_safe_to_step()


def test_bind_without_prepare_raises():
    trajs, groups, events, _rewards = make_batch(n_groups=1, n_traj=2, seed=351)
    model = build_model(STREAM_ATTENTION_NAME, seed=352)
    run, bw = _setup(model, trajs, groups)
    for gid, t_id, tokens, scored in events:
        run.append_turn(gid, t_id, tokens, scored)
    with pytest.raises(RuntimeError, match="not prepared"):
        bw.bind_reward(trajs[0].traj_id, 1.0)
    bw.abort()


def test_double_prepare_and_double_bind_raise():
    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=2, seed=361)
    model = build_model(STREAM_ATTENTION_NAME, seed=362)
    run, bw = _setup(model, trajs, groups)
    for gid, t_id, tokens, scored in events:
        run.append_turn(gid, t_id, tokens, scored)
    first = trajs[0].traj_id
    bw.prepare_trajectory(first)
    with pytest.raises(RuntimeError, match="already prepared"):
        bw.prepare_trajectory(first)
    bw.bind_reward(first, rewards[first])
    with pytest.raises(RuntimeError, match="already closed"):
        bw.bind_reward(first, rewards[first])
    # The legacy close path on a prepared trajectory is a double preparation.
    with pytest.raises(RuntimeError, match="already closed"):
        bw.close_trajectory(first, rewards[first])
    bw.abort()


def test_zero_reward_binding_preserves_g1_elision():
    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=3, seed=371)
    zero_rewards = {t_id: 0.0 for t_id in rewards}
    legacy_model = build_model(STREAM_ATTENTION_NAME, seed=372)
    early_model = build_model(STREAM_ATTENTION_NAME, seed=372)

    legacy, _run, _bw = reward_linear_grads(
        legacy_model, trajs, groups, events, zero_rewards
    )
    early, _early_run, early_bw = early_reward_linear_grads(
        early_model, trajs, groups, events, zero_rewards
    )

    for name in legacy:
        if legacy[name] is None:
            assert early[name] is None, name
            continue
        assert torch.equal(legacy[name], early[name]), name
    report = early_bw.source_offload_report
    assert report["current_statistic_bytes"] == 0


def test_optimizer_step_between_prepare_and_bind_raises():
    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=2, seed=381)
    model = build_model(STREAM_ATTENTION_NAME, seed=382)
    run, bw = _setup(model, trajs, groups)
    last_event, completion = _completion_order(events)
    for index, (gid, t_id, tokens, scored) in enumerate(events):
        run.append_turn(gid, t_id, tokens, scored)
        if last_event[t_id] == index:
            bw.prepare_trajectory(t_id)

    with torch.no_grad():
        next(model.parameters()).add_(1.0)

    with pytest.raises(RuntimeError, match="modified in place"):
        bw.bind_reward(completion[0], rewards[completion[0]])
    with contextlib.suppress(BaseException):
        bw.abort()


def test_prepare_rejects_pinned_streaming():
    trajs, groups, events, _rewards = make_batch(n_groups=1, n_traj=2, seed=391)
    model = build_model(STREAM_ATTENTION_NAME, seed=392)
    run, bw = _setup(model, trajs, groups, source_offload_mode="pinned_streaming")
    try:
        for gid, t_id, tokens, scored in events:
            run.append_turn(gid, t_id, tokens, scored)
        with pytest.raises(RuntimeError, match="pinned_streaming"):
            bw.prepare_trajectory(trajs[0].traj_id)
    finally:
        with contextlib.suppress(BaseException):
            bw.abort()


def test_group_close_before_last_bind_raises_then_succeeds():
    trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=4, seed=401)
    model = build_model(STREAM_ATTENTION_NAME, seed=402)
    run, bw = _setup(model, trajs, groups)
    last_event, completion = _completion_order(events)
    for index, (gid, t_id, tokens, scored) in enumerate(events):
        run.append_turn(gid, t_id, tokens, scored)
        if last_event[t_id] == index:
            bw.prepare_trajectory(t_id)
    for t_id in completion[:-1]:
        bw.bind_reward(t_id, rewards[t_id])
    with pytest.raises(RuntimeError, match="not closed"):
        bw.close_group(0)

    # The race in the persistent loop: the last bind lands on another thread
    # while group close begins. Close must observe the completed bind and the
    # reducer barrier, never a partial fold.
    binder = threading.Thread(
        target=bw.bind_reward, args=(completion[-1], rewards[completion[-1]])
    )
    binder.start()
    binder.join()
    bw.close_group(0)
    bw.assert_safe_to_step()


def test_concurrent_binds_from_two_threads_reduce_every_source_once():
    trajs, groups, events, rewards = make_batch(n_groups=2, n_traj=4, seed=411)
    model = build_model(STREAM_ATTENTION_NAME, seed=412)
    run, bw = _setup(model, trajs, groups)
    last_event, completion = _completion_order(events)
    for index, (gid, t_id, tokens, scored) in enumerate(events):
        run.append_turn(gid, t_id, tokens, scored)
        if last_event[t_id] == index:
            bw.prepare_trajectory(t_id)

    group_of = {t.traj_id: t.group_id for t in trajs}
    lanes = (
        [t_id for t_id in completion if group_of[t_id] == 0],
        [t_id for t_id in completion if group_of[t_id] == 1],
    )

    def bind_lane(lane):
        for t_id in lane:
            bw.bind_reward(t_id, rewards[t_id])

    threads = [threading.Thread(target=bind_lane, args=(lane,)) for lane in lanes]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for gid in groups:
        bw.close_group(gid)
    bw.assert_safe_to_step()
    report = bw.source_offload_report
    assert report["prepared_awaiting_reward_current"] == 0
    assert report["current_source_bytes"] == 0
    assert sorted(report["source_reduction_order_by_group"][0]) == sorted(lanes[0])
    assert sorted(report["source_reduction_order_by_group"][1]) == sorted(lanes[1])
