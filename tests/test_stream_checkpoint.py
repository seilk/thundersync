"""Gates for the checkpointed streaming executor.

The executor changes HOW activations are held, and must change nothing else.
So: the full F1 battery re-run with checkpoint=True (values, gradients,
cross-chunk paths, the reward-linear backward equality), plus the one property that justifies the
executor's existence -- retained memory collapsing toward the checkpoint
arithmetic instead of full activations.

The dangerous failure mode is the one round 1 proved for HF's checkpointing:
K/V captured in a no-grad region loses its graph and later chunks silently stop
sending gradient to earlier chunks. Every gradient test here dies on that.
"""

from __future__ import annotations


import pytest
import torch

from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from thundersync.engine.segments import Segment, Trajectory# noqa: E402
from tests.reference_logprobs import reference_logprobs  # noqa: E402
from thundersync.grpo.reward_linear import RewardLinearBackward  # noqa: E402
from thundersync.engine.streaming import (  # noqa: E402
    STREAM_ATTENTION_NAME,
    StreamingRun,
    register_stream_attention,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
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


def make_batch(n_groups=2, n_traj=3, prompt_len=24, vocab=512, seed=1):
    gen = torch.Generator().manual_seed(seed)
    r = lambda n: torch.randint(0, vocab, (n,), generator=gen).tolist()  # noqa: E731
    trajs, groups, events, tid = [], {}, [], 0
    for gid in range(n_groups):
        groups[gid] = r(prompt_len)
        per = []
        for i in range(n_traj):
            turns = [(r(4 + k + i), r(6 + i)) for k in range(1 + (i + gid) % 3)]
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
                    events.append(
                        (gid, t_id, a + o, [True] * len(a) + [False] * len(o))
                    )
            depth += 1
    return trajs, groups, events


def stream(model, groups, events, **kw):
    run = StreamingRun(model, checkpoint=True, **kw)
    for gid, p in groups.items():
        run.open_group(gid, p)
    for gid, tid, toks, sc in events:
        run.append_turn(gid, tid, toks, sc)
    return run


def total_loss(per_traj, seed=7):
    loss = 0.0
    for tid in sorted(per_traj):
        lp = per_traj[tid]
        g = torch.Generator(device=DEV).manual_seed(1000 + tid)
        loss = loss + (torch.randn(lp.shape, generator=g, device=DEV) * lp).sum()
    return loss


# ----------------------------------------------------------------------


def test_checkpointed_values_match_reference():
    sm = build_model(STREAM_ATTENTION_NAME, seed=3)
    rm = build_model("sdpa", seed=3)
    rm.load_state_dict(sm.state_dict())
    trajs, groups, events = make_batch(seed=5)
    run = stream(sm, groups, events)
    want = reference_logprobs(rm, trajs)
    for t in trajs:
        torch.testing.assert_close(
            run.logprobs(t.traj_id), want[t.traj_id], rtol=1e-4, atol=1e-5
        )


def test_checkpointed_gradients_match_reference():
    """Recompute must preserve BOTH cross-chunk channels."""
    sm = build_model(STREAM_ATTENTION_NAME, seed=11)
    rm = build_model("sdpa", seed=11)
    rm.load_state_dict(sm.state_dict())
    trajs, groups, events = make_batch(seed=13)

    run = stream(sm, groups, events)
    total_loss({t.traj_id: run.logprobs(t.traj_id) for t in trajs}).backward()
    total_loss(reference_logprobs(rm, trajs)).backward()

    checked = 0
    for name, p in dict(sm.named_parameters()).items():
        q = dict(rm.named_parameters())[name]
        if p.grad is None and q.grad is None:
            continue
        assert p.grad is not None and q.grad is not None, name
        torch.testing.assert_close(
            p.grad, q.grad, rtol=2e-3, atol=2e-5,
            msg=lambda m, n=name: f"{n}: {m}",
        )
        checked += 1
    assert checked > 5


def test_last_chunk_loss_reaches_first_chunk_parameters():
    """The sever detector, under recompute: kill-shot for stashed no-grad K/V."""
    sm = build_model(STREAM_ATTENTION_NAME, seed=17)
    rm = build_model("sdpa", seed=17)
    rm.load_state_dict(sm.state_dict())
    trajs, groups, events = make_batch(n_groups=1, n_traj=1, seed=19)
    run = stream(sm, groups, events)
    tid = trajs[0].traj_id

    run.logprobs(tid)[-1].backward()
    ref = reference_logprobs(rm, trajs)
    ref[tid][-1].backward()

    for name, p in dict(sm.named_parameters()).items():
        q = dict(rm.named_parameters())[name]
        if q.grad is None:
            continue
        torch.testing.assert_close(
            p.grad, q.grad, rtol=2e-3, atol=2e-5,
            msg=lambda m, n=name: f"{n}: cross-chunk path broken under recompute: {m}",
        )


def test_reward_linear_equality_with_checkpointing():
    """The full streamed backward on top of the checkpointed executor."""
    from thundersync.engine.segments import build_forest
    from thundersync.engine.plan import compile_plan
    from thundersync.engine.step import (
        ATTENTION_NAME,
        forest_logprobs,
        register_attention,
        split_by_trajectory,
    )

    register_attention()
    sm = build_model(STREAM_ATTENTION_NAME, seed=23)
    gm = build_model(ATTENTION_NAME, seed=23)
    gm.load_state_dict(sm.state_dict())
    trajs, groups, events = make_batch(seed=29)
    gen = torch.Generator().manual_seed(31)
    rewards = {t.traj_id: float(torch.rand(1, generator=gen)) for t in trajs}

    run = StreamingRun(sm, checkpoint=True, boundary_cut=True)
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
    for i, (gid, t_id, *_r) in enumerate(events):
        last[t_id] = i
    for i, (gid, t_id, toks, sc) in enumerate(events):
        run.append_turn(gid, t_id, toks, sc)
        if last[t_id] == i:
            bw.close_trajectory(t_id, rewards[t_id])
    for gid in groups:
        bw.close_group(gid)

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

    checked = 0
    for name, p in dict(sm.named_parameters()).items():
        q = dict(gm.named_parameters())[name]
        if p.grad is None and q.grad is None:
            continue
        torch.testing.assert_close(
            p.grad, q.grad, rtol=2e-3, atol=1e-5,
            msg=lambda m, n=name: f"{n}: {m}",
        )
        checked += 1
    assert checked > 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="allocator-level test")
def test_checkpointing_collapses_retention():
    """The executor's reason to exist, at the allocator's level of truth."""
    def peak(checkpoint):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        m = build_model(STREAM_ATTENTION_NAME, seed=37, layers=4, hidden=256)
        base = torch.cuda.memory_allocated()
        trajs, groups, events = make_batch(
            n_groups=1, n_traj=2, prompt_len=512, seed=41
        )
        # lengthen turns so activations dominate weights at this tiny scale
        gen = torch.Generator().manual_seed(43)
        big = [
            (g, t, torch.randint(0, 512, (384,), generator=gen).tolist(),
             [True] * 192 + [False] * 192)
            for g, t, _, _ in events
        ]
        run = StreamingRun(m, checkpoint=checkpoint)
        for gid, p in groups.items():
            run.open_group(gid, p)
        for gid, tid, toks, sc in big:
            run.append_turn(gid, tid, toks, sc)
        torch.cuda.synchronize()
        resident = torch.cuda.memory_allocated() - base
        del m, run
        return resident

    full = peak(False)
    ckpt = peak(True)
    assert ckpt < 0.5 * full, (
        f"checkpointed retention {ckpt/2**20:.1f} MiB is not materially below "
        f"full retention {full/2**20:.1f} MiB"
    )


# ----------------------------------------------------------------------
# Memory-gated rebuild implementation


def _rebuild_run(model, groups, events, **kwargs):
    run = StreamingRun(
        model, boundary_cut=True, checkpoint=True, stream_turns=False, **kwargs
    )
    for gid, prompt in groups.items():
        run.open_group(gid, prompt)
    for gid, tid, tokens, scored in events:
        run.append_turn(gid, tid, tokens, scored)
    return run


def _rebuild_grads(model, trajs, groups, events, **kwargs):
    run = _rebuild_run(model, groups, events, **kwargs)
    ids = [t.traj_id for t in trajs]
    for tid in ids:
        run.free_trajectory(tid)
    logprobs = run.rebuild_logprobs(ids)
    total_loss(logprobs).backward()
    grads = {
        n: (p.grad.clone() if p.grad is not None else None)
        for n, p in model.named_parameters()
    }
    return grads, run


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_plain_rebuild_headroom_selects_the_non_recomputing_path():
    """A branch that fits free device memory must skip the recompute forward.

    The block-checkpointed rebuild pays one recompute forward per chunk-layer
    during backward.  The MFU numerator credits one forward and two backwards,
    so that recompute is executed work the metric refuses to credit.  When the
    branch's activation transient fits free device memory with the declared
    reserve still spare, the plain whole-branch forward is the same arithmetic
    without the recompute.
    """

    trajs, groups, events = make_batch(seed=101)
    model = build_model(STREAM_ATTENTION_NAME, seed=101)

    _grads, run = _rebuild_grads(
        model,
        trajs,
        groups,
        events,
        rebuild_block_tokens=8,
        rebuild_plain_reserve_bytes=1 << 10,
    )

    assert run.rebuild_path_counts["plain"] == len(trajs)
    assert run.rebuild_path_counts["blocked"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_plain_rebuild_headroom_falls_back_when_the_branch_does_not_fit():
    """A reserve larger than the device must keep the block rebuild."""

    trajs, groups, events = make_batch(seed=103)
    model = build_model(STREAM_ATTENTION_NAME, seed=103)

    _grads, run = _rebuild_grads(
        model,
        trajs,
        groups,
        events,
        rebuild_block_tokens=8,
        rebuild_plain_reserve_bytes=1 << 50,
    )

    assert run.rebuild_path_counts["plain"] == 0
    assert run.rebuild_path_counts["blocked"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_plain_rebuild_headroom_preserves_the_rebuild_gradient():
    """Selecting by memory must change nothing the gradient can see."""

    trajs, groups, events = make_batch(seed=107)
    blocked_model = build_model(STREAM_ATTENTION_NAME, seed=107)
    plain_model = build_model(STREAM_ATTENTION_NAME, seed=107)

    blocked, blocked_run = _rebuild_grads(
        blocked_model, trajs, groups, events, rebuild_block_tokens=8
    )
    plain, plain_run = _rebuild_grads(
        plain_model,
        trajs,
        groups,
        events,
        rebuild_block_tokens=8,
        rebuild_plain_reserve_bytes=1 << 10,
    )

    assert blocked_run.rebuild_path_counts["plain"] == 0
    assert plain_run.rebuild_path_counts["blocked"] == 0

    checked = 0
    for name, reference in blocked.items():
        candidate = plain[name]
        if reference is None:
            assert candidate is None, name
            continue
        assert candidate is not None, name
        torch.testing.assert_close(
            candidate, reference, rtol=2e-3, atol=2e-5,
            msg=lambda m, n=name: f"{n}: {m}",
        )
        checked += 1
    assert checked > 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_split_bottom_right_causal_declines_head_dims_above_cudnn_limit():
    """Head dimension 256 must route to the masked fallback, not cuDNN.

    The split path's backward uses the cuDNN SDPA backward, which rejects
    hidden dimensions above 128. The qwen3_5 hybrid family's full-attention
    layers run head_dim 256, so the availability gate must decline and the
    masked reference path must produce the gradient.
    """

    from thundersync.engine import streaming as streaming_module

    torch.manual_seed(31)
    head_dim = 256
    query = torch.randn(
        1, 4, 6, head_dim, device="cuda", dtype=torch.bfloat16,
        requires_grad=True,
    )
    keys = [
        torch.randn(5, 2, head_dim, device="cuda", dtype=torch.bfloat16,
                    requires_grad=True),
        torch.randn(6, 2, head_dim, device="cuda", dtype=torch.bfloat16,
                    requires_grad=True),
    ]
    values = [
        torch.randn(5, 2, head_dim, device="cuda", dtype=torch.bfloat16,
                    requires_grad=True),
        torch.randn(6, 2, head_dim, device="cuda", dtype=torch.bfloat16,
                    requires_grad=True),
    ]

    assert not streaming_module._split_bottom_right_is_available(
        query, prefix=5
    )

    out = streaming_module._causal_sdpa(query, keys, values, scale=0.0625)
    out.sum().backward()
    assert query.grad is not None
    assert all(k.grad is not None for k in keys)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_split_bottom_right_flash_matches_the_masked_reference_at_head_dim_256():
    """Head dimension 256 must reach a fused kernel, not the math fallback.

    cuDNN's SDPA rejects hidden dimensions above 128, so the qwen3_5 hybrid
    family's full-attention layers fell back to the masked math path, which
    measures 2.8% of peak against 14.8% for the flash backend on the same
    shapes -- attention is under a tenth of the branch's FLOPs but was
    taking about a third of its time. The flash split must produce the
    masked reference's values and gradients.
    """

    from thundersync.engine import streaming as streaming_module

    if streaming_module._FLASH_SDPA is None:
        pytest.skip("flash SDPA op unavailable")

    torch.manual_seed(53)
    head_dim, heads, prefix, tail = 256, 4, 40, 24
    scale = head_dim ** -0.5

    def build():
        q = torch.randn(1, heads, tail, head_dim, device="cuda",
                        dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(1, heads, prefix + tail, head_dim, device="cuda",
                        dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(1, heads, prefix + tail, head_dim, device="cuda",
                        dtype=torch.bfloat16, requires_grad=True)
        return q, k, v

    torch.manual_seed(53)
    q1, k1, v1 = build()
    torch.manual_seed(53)
    q2, k2, v2 = build()

    bias = torch.zeros(tail, prefix + tail, device="cuda", dtype=torch.bfloat16)
    row = torch.arange(tail, device="cuda").unsqueeze(1) + prefix
    col = torch.arange(prefix + tail, device="cuda").unsqueeze(0)
    bias.masked_fill_(col > row, float("-inf"))
    reference = torch.nn.functional.scaled_dot_product_attention(
        q1, k1, v1, attn_mask=bias, scale=scale
    )
    reference.sum().backward()

    candidate = streaming_module._SplitBottomRightFlash.apply(
        q2, k2, v2, prefix, scale
    )
    candidate.sum().backward()

    torch.testing.assert_close(candidate, reference, rtol=2e-2, atol=2e-2)
    for a, b in ((q2, q1), (k2, k1), (v2, v1)):
        torch.testing.assert_close(a.grad, b.grad, rtol=5e-2, atol=5e-2)


def test_rebuild_checkpoint_offload_preserves_the_rebuild_gradient():
    """Parking the rebuild's checkpoint inputs on the host changes nothing
    the gradient can see, and moves only the block's own inputs."""

    trajs, groups, events = make_batch(seed=109)
    resident_model = build_model(STREAM_ATTENTION_NAME, seed=109)
    offloaded_model = build_model(STREAM_ATTENTION_NAME, seed=109)

    resident, resident_run = _rebuild_grads(
        resident_model, trajs, groups, events, rebuild_block_tokens=8
    )
    offloaded, offloaded_run = _rebuild_grads(
        offloaded_model,
        trajs,
        groups,
        events,
        rebuild_block_tokens=8,
        rebuild_checkpoint_storage_device="cpu",
    )

    resident_stats = resident_run.rebuild_checkpoint_offload_diagnostics()
    offloaded_stats = offloaded_run.rebuild_checkpoint_offload_diagnostics()
    assert resident_stats["storage_device"] is None
    assert resident_stats["write_bytes"] == 0
    assert offloaded_stats["storage_device"] == "cpu"
    assert offloaded_stats["write_bytes"] > 0
    assert offloaded_stats["read_bytes"] == offloaded_stats["write_bytes"]
    assert offloaded_stats["current_bytes"] == 0
    # three tensors per segment per layer (the block's hidden state, cos, sin):
    # the ancestor K/V are referenced, never copied
    assert offloaded_stats["tensor_count"] % 3 == 0

    checked = 0
    for name, reference in resident.items():
        candidate = offloaded[name]
        if reference is None:
            assert candidate is None, name
            continue
        assert candidate is not None, name
        torch.testing.assert_close(
            candidate, reference, rtol=2e-3, atol=2e-5,
            msg=lambda m, n=name: f"{n}: {m}",
        )
        checked += 1
    assert checked > 5
