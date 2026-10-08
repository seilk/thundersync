"""Gates for coalesced turn arrivals: many concurrent turns, one call.

`StreamingRun.append_turns` packs turns of DIFFERENT trajectories (and groups)
that arrived in the same scheduling window into one forward: tokens
concatenated along the sequence dim, per-segment position ids, batched dense
ops, per-segment attention over each trajectory's own ancestor chain. The
invariant is the usual one, sharpened: coalescing changes the kernel schedule
and NOTHING else -- same values and parameter gradients as the sequential
per-turn path AND as the independent per-trajectory reference, and the reward-linear backward's
per-trajectory backwards still work because the recorded graph stays
per-segment (streaming._SegLayerCkpt).

The adversarial mix throughout: 2 groups x 3 trajectories with different turn
counts and different turn lengths, arriving round-robin, so every window mixes
groups, trajectories, and segment lengths in one `append_turns` call.
"""

from __future__ import annotations


import pytest
import torch

from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from thundersync.engine.segments import Segment, Trajectory, build_forest
from thundersync.engine.plan import compile_plan# noqa: E402
from tests.reference_logprobs import reference_logprobs  # noqa: E402
from thundersync.grpo.reward_linear import RewardLinearBackward  # noqa: E402
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


def build_model(impl, vocab=512, seed=0, layers=2, hidden=128):
    torch.manual_seed(seed)
    cfg = AutoConfig.for_model(
        "qwen3",
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=layers,
        num_attention_heads=hidden // 32,
        num_key_value_heads=max(hidden // 64, 1),
        head_dim=32,
        vocab_size=vocab,
        max_position_embeddings=8192,
        tie_word_embeddings=False,
    )
    return AutoModelForCausalLM.from_config(
        cfg, attn_implementation=impl, dtype=torch.float32
    ).to(DEV).eval()


def make_windows(n_groups=2, n_traj=3, prompt_len=24, vocab=512, seed=1):
    """Trajectories plus their arrivals grouped into scheduling windows.

    Window d holds turn d of every trajectory that has one -- ACROSS groups --
    so a single `append_turns` call mixes groups, trajectories, and lengths.
    """
    gen = torch.Generator().manual_seed(seed)
    r = lambda n: torch.randint(0, vocab, (n,), generator=gen).tolist()  # noqa: E731
    trajs, groups, per, tid = [], {}, [], 0
    for gid in range(n_groups):
        groups[gid] = r(prompt_len)
        for i in range(n_traj):
            turns = [(r(4 + k + i), r(6 + i)) for k in range(1 + (i + gid) % 3)]
            segs = [Segment.prompt(groups[gid])]
            for a, o in turns:
                segs.append(Segment.action(a))
                segs.append(Segment.observation(o))
            trajs.append(Trajectory(segs, group_id=gid, traj_id=tid))
            per.append((gid, tid, turns))
            tid += 1
    windows, depth = [], 0
    while True:
        w = [
            (gid, t_id, a + o, [True] * len(a) + [False] * len(o))
            for gid, t_id, turns in per
            if depth < len(turns)
            for a, o in [turns[depth]]
        ]
        if not w:
            break
        windows.append(w)
        depth += 1
    return trajs, groups, windows


def total_loss(per_traj):
    loss = 0.0
    for tid in sorted(per_traj):
        lp = per_traj[tid]
        g = torch.Generator(device=DEV).manual_seed(1000 + tid)
        loss = loss + (torch.randn(lp.shape, generator=g, device=DEV) * lp).sum()
    return loss


def assert_grads_equal(model_a, model_b, ctx, rtol=2e-3, atol=2e-5):
    checked = 0
    b = dict(model_b.named_parameters())
    for name, p in dict(model_a.named_parameters()).items():
        if p.grad is None and b[name].grad is None:
            continue
        assert p.grad is not None and b[name].grad is not None, f"{ctx}: {name}"
        torch.testing.assert_close(
            p.grad, b[name].grad, rtol=rtol, atol=atol,
            msg=lambda m, n=name: f"{ctx}: {n}: {m}",
        )
        checked += 1
    assert checked > 5, ctx


# ----------------------------------------------------------------------


@pytest.mark.parametrize("checkpoint", [True, False])
def test_coalesced_equals_sequential_and_reference(checkpoint):
    """(a) The reorder invariant, extended across concurrent trajectories."""
    sm = build_model(STREAM_ATTENTION_NAME, seed=3)
    qm = build_model(STREAM_ATTENTION_NAME, seed=3)
    qm.load_state_dict(sm.state_dict())
    rm = build_model("sdpa", seed=3)
    rm.load_state_dict(sm.state_dict())
    trajs, groups, windows = make_windows(seed=5)

    # the mix must be genuinely adversarial or this gate proves nothing:
    # every window holds segments of BOTH groups and >=2 trajectories, the
    # first all 6, with differing segment lengths inside one call
    assert len(windows) >= 3
    assert len(windows[0]) == 2 * 3
    for w in windows:
        assert len(w) >= 2
        assert len({g for g, *_ in w}) == 2
    assert len({len(t) for _, _, t, _ in windows[0]}) > 1

    coal = StreamingRun(sm, checkpoint=checkpoint)
    for gid, p in groups.items():
        coal.open_group(gid, p)
    for w in windows:
        coal.append_turns(w)

    seq = StreamingRun(qm, checkpoint=checkpoint)
    for gid, p in groups.items():
        seq.open_group(gid, p)
    for w in windows:
        for gid, t_id, toks, sc in w:
            seq.append_turn(gid, t_id, toks, sc)

    want = reference_logprobs(rm, trajs)
    for t in trajs:
        got = coal.logprobs(t.traj_id)
        torch.testing.assert_close(
            got, seq.logprobs(t.traj_id), rtol=1e-4, atol=1e-5,
            msg=lambda m, i=t.traj_id: f"traj {i} vs sequential: {m}",
        )
        torch.testing.assert_close(
            got, want[t.traj_id], rtol=1e-4, atol=1e-5,
            msg=lambda m, i=t.traj_id: f"traj {i} vs reference: {m}",
        )
        assert torch.equal(got.detach(), coal.old_logprobs(t.traj_id)), (
            f"traj {t.traj_id}: old != new bitwise under coalescing (D1)"
        )

    total_loss({t.traj_id: coal.logprobs(t.traj_id) for t in trajs}).backward()
    total_loss({t.traj_id: seq.logprobs(t.traj_id) for t in trajs}).backward()
    total_loss(want).backward()
    assert_grads_equal(sm, qm, "coalesced vs sequential")
    assert_grads_equal(sm, rm, "coalesced vs reference")


def test_same_trajectory_twice_in_one_call_raises():
    """(b) Turn k+1 needs turn k's last hidden state; packing both is illegal."""
    sm = build_model(STREAM_ATTENTION_NAME, seed=7)
    run = StreamingRun(sm, checkpoint=True)
    run.open_group(0, [1, 2, 3, 4])
    with pytest.raises(ValueError, match="twice"):
        run.append_turns([(0, 5, [7, 8], [True, True]), (0, 5, [9], [True])])
    # rejected BEFORE any state was committed
    assert 5 not in run._trajs
    assert len(run._chunks) == 1  # only the prompt chunk exists


@pytest.mark.parametrize("checkpoint", [True, False])
def test_reward_linear_equality_with_coalesced_arrivals(checkpoint):
    """(c) Streamed per-trajectory backwards over coalesced arrivals == barrier.

    The killer this guards: had coalescing recorded graph nodes shared across
    trajectories, the first close_trajectory (retain_graph=False) would spend
    its window-mates' graphs and the later closes would crash or corrupt.
    Different groups' segments must also keep reading their own boundary
    proxies inside one packed call.
    """
    sm = build_model(STREAM_ATTENTION_NAME, seed=23)
    gm = build_model(ATTENTION_NAME, seed=23)
    gm.load_state_dict(sm.state_dict())
    trajs, groups, windows = make_windows(seed=29)
    gen = torch.Generator().manual_seed(31)
    rewards = {t.traj_id: float(torch.rand(1, generator=gen)) for t in trajs}

    run = StreamingRun(sm, boundary_cut=True, checkpoint=checkpoint)
    bw = RewardLinearBackward(run, sm, eps=EPS)
    for gid, p in groups.items():
        run.open_group(gid, p)
        bw.register_group_trajectories(
            gid,
            sorted(
                trajectory.traj_id
                for trajectory in trajs
                if trajectory.group_id == gid
            ),
        )
    last = {}
    for i, w in enumerate(windows):
        for _, t_id, *_rest in w:
            last[t_id] = i
    for i, w in enumerate(windows):
        run.append_turns(w)
        for _, t_id, *_rest in w:
            if last[t_id] == i:  # trajectory finished: its reward lands NOW
                bw.close_trajectory(t_id, rewards[t_id])
    for gid in groups:
        bw.close_group(gid)
    assert run.kv_bytes_resident() == 0, "coalesced close left K/V resident"

    plan = compile_plan(build_forest(trajs), device=DEV)
    per = split_by_trajectory(plan, forest_logprobs(gm, plan))
    loss = 0.0
    by_group: dict[int, list[int]] = {}
    for t in trajs:
        by_group.setdefault(t.group_id, []).append(t.traj_id)
    for gid, tids in by_group.items():
        r = torch.tensor([rewards[t] for t in tids], dtype=torch.float64)
        mean, std = float(r.mean()), float(r.std(unbiased=False)) + EPS
        for t in tids:
            loss = loss + ((rewards[t] - mean) / std) * per[t].sum()
    loss.backward()

    assert_grads_equal(sm, gm, "reward-linear coalesced vs barrier", atol=1e-5)
