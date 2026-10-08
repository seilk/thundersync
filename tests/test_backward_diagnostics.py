"""The run's and the backward's diagnostics are callable and serializable.

Runs wherever torch runs (the node's CPU pass): a tiny model, one group
streamed through the plain (run, backward) pair the DP worker adapts, then
every probe the GRPO trainer's step statistics read -- called the way the
trainer calls it and dumped as JSON, since the step record is written that
way. A misplaced @property that turns one of these into a dict fails the
first step.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402
import torch  # noqa: E402

from tests.test_reward_linear import EPS, build_model, make_batch  # noqa: E402

from thundersync.grpo.reward_linear import RewardLinearBackward  # noqa: E402
from thundersync.engine.streaming import STREAM_ATTENTION_NAME, StreamingRun  # noqa: E402


def _stream_one_group(model, groups, events, rewards):
    run = StreamingRun(model, boundary_cut=True)
    backward = RewardLinearBackward(run, model, eps=EPS, source_offload_mode="blocking")
    gid = sorted(groups)[0]
    group_events = [item for item in events if item[0] == gid]
    trajectories = sorted({tid for _, tid, _, _ in group_events})
    backward.register_group_trajectories(gid, trajectories)
    run.open_group(gid, groups[gid])
    last = {tid: index for index, (_, tid, *_rest) in enumerate(group_events)}
    for index, (_, tid, tokens, scored) in enumerate(group_events):
        run.append_turn(gid, tid, tokens, scored)
        if last[tid] == index:
            backward.close_trajectory(tid, rewards[tid])
    backward.close_group(gid)
    return run, backward


def test_every_probe_the_step_reads_is_callable_and_json_serializable():
    torch.manual_seed(3)
    model = build_model(STREAM_ATTENTION_NAME, seed=3)
    # test_reward_linear.make_batch returns the reward map as a fourth value; these
    # tests draw their own rewards so the map is dropped, not unpacked into
    # three names (the node CPU pass at 7ad8b8b failed here on exactly that).
    trajs, groups, events, _rewards = make_batch(seed=3)
    rewards = {t.traj_id: float(index % 2) for index, t in enumerate(trajs)}
    run, backward = _stream_one_group(model, groups, events, rewards)
    try:
        assert callable(run.retained_device_bytes)
        assert callable(backward.retained_device_bytes)
        assert callable(run.rebuild_checkpoint_offload_diagnostics)
        assert callable(run.parameter_adjoint_diagnostics)
        report = backward.source_offload_report  # a property, read not called
        assert isinstance(report, dict)
        stats = {
            "source_offload_report": report,
            "rebuild_path_counts": run.rebuild_path_counts,
            "rebuild_checkpoint_offload": run.rebuild_checkpoint_offload_diagnostics(),
            "gradient_accumulation_report": run.parameter_adjoint_diagnostics(),
            "run_retained_bytes": run.retained_device_bytes(),
            "backward_retained_bytes": backward.retained_device_bytes(),
        }
        encoded = json.dumps(stats, sort_keys=True)
        assert '"accounted"' in encoded and '"live_runs"' in encoded
        assert stats["run_retained_bytes"]["live_runs"] >= 1
        assert stats["rebuild_checkpoint_offload"]["storage_device"] is None
    finally:
        backward.abort()


_GROUP_ACCUM_FIELDS = (
    "parameter_sources",
    "boundary_sources",
    "rewards_by_trajectory",
    "token_weights_by_trajectory",
    "parameter_centered_sum",
    "parameter_source_mean",
    "boundary_centered_sum",
    "boundary_source_mean",
    "statistic_count",
    "closed",
    "differentiated",
    "reduced",
    "prepared",
    "materialized",
    "other",
)


def _group_kind_bytes(report):
    return {
        kind: value
        for kind, value in report["_groups_by_kind"].items()
        if kind not in ("open_groups", "group_tensor_count")
    }


@pytest.mark.parametrize("statistic_device", [None] + (["cuda"] if torch.cuda.is_available() else []))
def test_the_group_accumulator_bytes_are_broken_down_by_field(statistic_device):
    """`_groups` is what the streaming trainer's in-step device peak is made of. The breakdown names the accumulator
    field each storage came from and shares the walk's one storage ledger, so
    the buckets sum to the flat `_groups` entry and leave it unchanged."""
    torch.manual_seed(3)
    model = build_model(STREAM_ATTENTION_NAME, seed=3)
    # test_reward_linear.make_batch returns the reward map as a fourth value; these
    # tests draw their own rewards so the map is dropped, not unpacked into
    # three names (the node CPU pass at 7ad8b8b failed here on exactly that).
    trajs, groups, events, _rewards = make_batch(seed=3)
    rewards = {t.traj_id: float(index % 2) for index, t in enumerate(trajs)}
    run = StreamingRun(model, boundary_cut=True)
    backward = RewardLinearBackward(
        run, model, eps=EPS, source_offload_mode="blocking",
        statistic_device=statistic_device,
    )
    try:
        gid = sorted(groups)[0]
        group_events = [item for item in events if item[0] == gid]
        trajectories = sorted({tid for _, tid, _, _ in group_events})
        backward.register_group_trajectories(gid, trajectories)
        run.open_group(gid, groups[gid])
        last = {tid: index for index, (_, tid, *_rest) in enumerate(group_events)}
        for index, (_, tid, tokens, scored) in enumerate(group_events):
            run.append_turn(gid, tid, tokens, scored)
            if last[tid] == index:
                backward.close_trajectory(tid, rewards[tid])
        # the folds are done, so the accumulator's statistics are resident and
        # the open group's bytes are a settled number rather than a race
        for tid in trajectories:
            backward.wait_for_trajectory_reduction(gid, tid)

        report = backward.retained_device_bytes()
        json.dumps(report, sort_keys=True)
        kinds = _group_kind_bytes(report)
        statistics = [t for t in backward._groups[gid].parameter_source_mean if t is not None]
        assert statistics, "the open group must hold reduced statistics"
        # The report counts only the model device. Default CPU offload must
        # not appear as GPU residency; explicitly resident statistics must.
        resident = statistics[0].device == next(model.parameters()).device
        assert bool(kinds) == resident
        for kind in kinds:
            assert kind.removesuffix(".grad") in _GROUP_ACCUM_FIELDS, kind
        assert sum(kinds.values()) == report.get("_groups", 0) + report.get(
            "_groups.grad", 0
        )
        assert report["_groups_by_kind"]["open_groups"] == 1
        # every counted storage is non-empty, so the tensor count and the
        # bucket bytes are non-zero together
        assert (report["_groups_by_kind"]["group_tensor_count"] > 0) == resident
        assert (sum(kinds.values()) > 0) == resident
        # additive: `accounted` still sums exactly the flat byte entries
        assert report["accounted"] == sum(
            value for key, value in report.items()
            if key != "accounted" and isinstance(value, int)
        )

        # the close pops the accumulator, so the group holds nothing
        backward.close_group(gid)
        closed = backward.retained_device_bytes()
        assert closed["_groups_by_kind"]["open_groups"] == 0
        assert closed["_groups_by_kind"]["group_tensor_count"] == 0
    finally:
        backward.abort()


def test_the_attribute_walker_names_holders_and_deduplicates_storage():
    from types import SimpleNamespace

    from thundersync.engine.streaming import retained_bytes_by_attribute

    base = torch.zeros(1024, dtype=torch.uint8)
    owner = SimpleNamespace(
        buffer=base,
        alias=base[:16],  # a view of the same storage: counted once, under `buffer`
        nested={"inner": [torch.zeros(256, dtype=torch.uint8)]},
        scalar=3,
        text="x",
    )
    report = retained_bytes_by_attribute(owner, base.device)
    assert report["buffer"] == 1024
    assert "alias" not in report
    assert report["nested"] == 256
    assert report["accounted"] == 1280
    assert retained_bytes_by_attribute(owner, base.device, skip=("buffer",))["alias"] == 1024
