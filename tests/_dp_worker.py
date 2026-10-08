"""torchrun worker for tests/test_dp.py -- NOT a pytest file (no test_ prefix).

Launched as `torchrun --nproc-per-node=2 tests/_dp_worker.py --mode M --out J`.
Rank 0 writes a JSON verdict to --out; any rank raising makes torchrun exit
nonzero. The pytest wrapper asserts on both.

Modes:
  equality   the DP gate: 2 ranks x 1 group each through DataParallelThunderSync.step()
             must equal a single-process StreamingRun+RewardLinearBackward run
             over BOTH groups with grads divided by 2 (the mean-over-groups
             semantics dp.py documents), same SGD step, same seeds, fp32.
  unequal    rank 0 streams two groups while rank 1 streams one; global group
             normalization must still equal a single-process three-group run.
  broadcast  ranks build models from DIFFERENT seeds; init_from_env must make
             them identical (state hashes compared across ranks).
  guard      dp.step() with open groups must raise on the offending rank,
             locally, before any collective; after closing properly the same
             step must succeed and leave the replicas identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_reward_linear import EPS, build_model, make_batch  # noqa: E402

from thundersync.grpo.data_parallel import DataParallelThunderSync  # noqa: E402
from thundersync.grpo.reward_linear import RewardLinearBackward  # noqa: E402
from thundersync.engine.streaming import StreamingRun  # noqa: E402

LR = 0.5  # SGD: updated-param delta = LR * grad, so comparing deltas at the
# repo's gradient tolerances compares the gradients themselves
GRAD_CLIP = 0.05
RTOL, ATOL = 2e-3, 1e-5
# The pytest wrapper selects the source-transfer implementation under test;
# both the DP ranks and the single-process reference use the same mode, so
# each gate isolates the DP mechanics from the offload implementation.
SOURCE_OFFLOAD_MODE = os.environ.get(
    "THUNDERSYNC_DP_WORKER_SOURCE_OFFLOAD_MODE", "blocking"
)


def group_events(events, gid):
    return [(g, t, tok, sc) for (g, t, tok, sc) in events if g == gid]


def stream_group(dp_or_pair, gid, events, rewards):
    """Replay one group's event stream; close trajectories at their last event."""
    dp_or_pair.register_group_trajectories(
        gid,
        sorted({trajectory_id for _, trajectory_id, _, _ in events}),
    )
    last = {}
    for i, (_g, t_id, *_r) in enumerate(events):
        last[t_id] = i
    for i, (_g, t_id, tokens, scored) in enumerate(events):
        dp_or_pair.append_turn(gid, t_id, tokens, scored)
        if last[t_id] == i:
            dp_or_pair.close_trajectory(t_id, rewards[t_id])
    dp_or_pair.close_group(gid)


class _Plain:
    """Adapter giving (StreamingRun, RewardLinearBackward) the dp event API."""

    def __init__(self, model):
        self.run = StreamingRun(model, boundary_cut=True)
        self.bw = RewardLinearBackward(
            self.run,
            model,
            eps=EPS,
            source_offload_mode=SOURCE_OFFLOAD_MODE,
        )

    def open_group(self, gid, prompt):
        self.run.open_group(gid, prompt)

    def append_turn(self, gid, tid, tokens, scored):
        self.run.append_turn(gid, tid, tokens, scored)

    def register_group_trajectories(self, gid, trajectory_ids):
        self.bw.register_group_trajectories(gid, trajectory_ids)

    def close_trajectory(self, tid, reward):
        self.bw.close_trajectory(tid, reward)

    def close_group(self, gid):
        self.bw.close_group(gid)


def state_hash(model) -> str:
    h = hashlib.sha256()
    for name, t in model.state_dict().items():
        h.update(name.encode())
        h.update(t.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def gather_hashes(model) -> list[str]:
    out: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(out, state_hash(model))
    return out  # type: ignore[return-value]


# ------------------------------------------------------------------- modes


def mode_equality(rank: int) -> dict:
    # same seed on every rank: the broadcast is semantically a no-op here
    # (broadcast correctness has its own mode); what this gate proves is that
    # DP over {rank r streams group r} equals one process streaming both.
    model = build_model("thundersync_stream", seed=3)
    init_state = {k: v.clone() for k, v in model.state_dict().items()}
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    dp = DataParallelThunderSync.init_from_env(
        model,
        opt,
        grad_clip=GRAD_CLIP,
        source_offload_mode=SOURCE_OFFLOAD_MODE,
    )

    # deterministic synthetic batch, identical on every rank; 2 groups total
    trajs, groups, events, rewards = make_batch(n_groups=2, n_traj=4, seed=5)
    gid = rank  # rank r streams group r -- data parallelism at group granularity
    dp.open_group(gid, groups[gid])
    stream_group(dp, gid, group_events(events, gid), rewards)
    assert dp.local_groups_closed == 1
    stats = dp.step()

    hashes = gather_hashes(model)
    result = {"stats": stats, "replicas_identical": len(set(hashes)) == 1}
    if rank != 0:
        return result

    # single-process reference on rank 0's GPU: same kernels, same device.
    ref = build_model("thundersync_stream", seed=3)
    for (n1, p1), (n2, p2) in zip(init_state.items(), ref.state_dict().items()):
        assert n1 == n2 and torch.equal(p1, p2), f"init mismatch at {n1}"
    plain = _Plain(ref)
    for g in sorted(groups):
        plain.open_group(g, groups[g])
    for g in sorted(groups):
        stream_group(plain, g, group_events(events, g), rewards)
    with torch.no_grad():
        for p in ref.parameters():
            if p.grad is not None:
                p.grad.div_(len(groups))  # sum over N groups -> mean, = AVG of
                # global mean-over-groups normalization (dp.py)
    torch.nn.utils.clip_grad_norm_(
        ref.parameters(),
        GRAD_CLIP,
        foreach=False,
    )
    ref_opt = torch.optim.SGD(ref.parameters(), lr=LR)
    ref_opt.step()

    worst_param, worst_delta, checked = 0.0, 0.0, 0
    dp_state = model.state_dict()
    for name, ref_p in ref.state_dict().items():
        dp_p = dp_state[name]
        torch.testing.assert_close(
            dp_p, ref_p, rtol=RTOL, atol=ATOL, msg=lambda m, n=name: f"{n}: {m}"
        )
        d_dp = dp_p - init_state[name]
        d_ref = ref_p - init_state[name]
        # the meaningful gate: the UPDATE (lr * averaged grad) at gradient
        # tolerances -- params alone would pass trivially at rtol 2e-3
        torch.testing.assert_close(
            d_dp, d_ref, rtol=RTOL, atol=ATOL,
            msg=lambda m, n=name: f"update of {n}: {m}",
        )
        if d_ref.numel():
            worst_param = max(worst_param, float((dp_p - ref_p).abs().max()))
            worst_delta = max(worst_delta, float((d_dp - d_ref).abs().max()))
        checked += 1
    assert checked > 5
    result.update(
        n_params_checked=checked,
        max_abs_param_diff=worst_param,
        max_abs_update_diff=worst_delta,
        n_groups_total=len(groups),
    )
    return result


def mode_unequal(rank: int) -> dict:
    model = build_model("thundersync_stream", seed=29)
    init_state = {key: value.clone() for key, value in model.state_dict().items()}
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    dp = DataParallelThunderSync.init_from_env(
        model,
        opt,
        grad_clip=GRAD_CLIP,
        source_offload_mode=SOURCE_OFFLOAD_MODE,
    )
    if dist.get_world_size() != 2:
        raise RuntimeError("unequal DP gate requires exactly two ranks")

    _trajs, groups, events, rewards = make_batch(n_groups=3, n_traj=6, seed=31)
    local_group_ids = (0, 1) if rank == 0 else (2,)
    for group_id in local_group_ids:
        dp.open_group(group_id, groups[group_id])
        stream_group(dp, group_id, group_events(events, group_id), rewards)
    stats = dp.step()

    hashes = gather_hashes(model)
    result = {"stats": stats, "replicas_identical": len(set(hashes)) == 1}
    if rank != 0:
        return result

    ref = build_model("thundersync_stream", seed=29)
    plain = _Plain(ref)
    for group_id in sorted(groups):
        plain.open_group(group_id, groups[group_id])
    for group_id in sorted(groups):
        stream_group(plain, group_id, group_events(events, group_id), rewards)
    with torch.no_grad():
        for parameter in ref.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(len(groups))
    torch.nn.utils.clip_grad_norm_(ref.parameters(), GRAD_CLIP, foreach=False)
    ref_opt = torch.optim.SGD(ref.parameters(), lr=LR)
    ref_opt.step()

    worst_delta = 0.0
    for name, ref_parameter in ref.state_dict().items():
        dp_parameter = model.state_dict()[name]
        dp_delta = dp_parameter - init_state[name]
        ref_delta = ref_parameter - init_state[name]
        torch.testing.assert_close(
            dp_delta,
            ref_delta,
            rtol=RTOL,
            atol=ATOL,
            msg=lambda message, parameter_name=name: (
                f"unequal update of {parameter_name}: {message}"
            ),
        )
        if ref_delta.numel():
            worst_delta = max(
                worst_delta, float((dp_delta - ref_delta).abs().max())
            )
    result.update(
        max_abs_update_diff=worst_delta,
        n_groups_total=len(groups),
    )
    return result


def mode_broadcast(rank: int) -> dict:
    model = build_model("thundersync_stream", seed=100 + rank)  # deliberately different
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    dp = DataParallelThunderSync.init_from_env(
        model,
        opt,
        source_offload_mode=SOURCE_OFFLOAD_MODE,
    )
    hashes = gather_hashes(model)
    assert len(set(hashes)) == 1, f"broadcast left replicas different: {hashes}"
    # and they are rank 0's weights, not some accident: rebuild seed-100 locally
    ref0 = build_model("thundersync_stream", seed=100)
    assert state_hash(ref0) == hashes[0], "replicas are not rank 0's init"
    del dp
    return {"hash": hashes[0], "world": len(hashes)}


def mode_guard(rank: int) -> dict:
    model = build_model("thundersync_stream", seed=19)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    dp = DataParallelThunderSync.init_from_env(
        model,
        opt,
        source_offload_mode=SOURCE_OFFLOAD_MODE,
    )
    _trajs, groups, events, rewards = make_batch(n_groups=1, n_traj=2, seed=23)

    dp.open_group(0, groups[0])
    ev = group_events(events, 0)
    dp.register_group_trajectories(
        0,
        sorted({trajectory_id for _, trajectory_id, _, _ in ev}),
    )
    for g, t, tokens, scored in ev[:2]:
        dp.append_turn(g, t, tokens, scored)

    raised = False
    try:
        dp.step()  # open group: must raise BEFORE any collective (no deadlock)
    except RuntimeError as e:
        raised = "open groups" in str(e)
    assert raised, "step() with open groups did not raise"

    # the guard did not poison anything: finish the batch properly and step
    last = {}
    for i, (_g, t_id, *_r) in enumerate(ev):
        last[t_id] = i
    for i, (_g, t_id, tokens, scored) in enumerate(ev):
        if i >= 2:
            dp.append_turn(0, t_id, tokens, scored)
        if last[t_id] == i:
            dp.close_trajectory(t_id, rewards[t_id])
    dp.close_group(0)
    stats = dp.step()
    hashes = gather_hashes(model)
    assert len(set(hashes)) == 1, "replicas diverged after recovery step"
    return {"raised": True, "recovered": True, "stats": stats}


def mode_fresh_batches(rank: int) -> dict:
    model = build_model("thundersync_stream", seed=41)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    dp = DataParallelThunderSync.init_from_env(
        model,
        opt,
        grad_clip=GRAD_CLIP,
        source_offload_mode=SOURCE_OFFLOAD_MODE,
    )
    runs = [dp.run]
    closed_after_steps = []
    for batch_seed in (43, 47):
        _trajs, groups, events, rewards = make_batch(
            n_groups=2,
            n_traj=4,
            seed=batch_seed,
        )
        group_id = rank
        dp.open_group(group_id, groups[group_id])
        stream_group(
            dp,
            group_id,
            group_events(events, group_id),
            rewards,
        )
        dp.step()
        runs.append(dp.run)
        closed_after_steps.append(dp.local_groups_closed)
    hashes = gather_hashes(model)
    dp.shutdown()
    return {
        "replicas_identical": len(set(hashes)) == 1,
        "run_was_rearmed": all(
            current is not previous
            for previous, current in zip(runs, runs[1:])
        ),
        "steps": 2,
        "local_groups_closed_after_each_step": closed_after_steps,
        "shutdown": True,
    }


MODES = {
    "equality": mode_equality,
    "unequal": mode_unequal,
    "broadcast": mode_broadcast,
    "guard": mode_guard,
    "fresh_batches": mode_fresh_batches,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=sorted(MODES))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.manual_seed(0)

    result = MODES[args.mode](int(os.environ["RANK"]))

    dist.barrier()  # every rank's assertions passed before the verdict lands
    if int(os.environ["RANK"]) == 0:
        with open(args.out, "w") as f:
            json.dump({"mode": args.mode, "ok": True, "result": result}, f, indent=2)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
