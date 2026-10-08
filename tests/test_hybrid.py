"""Hybrid (Gated DeltaNet + full attention) gates: the full battery.

Hybrid stacks interleave linear-attention (Gated DeltaNet) layers with
full-attention ones. A linear layer's cross-chunk dependence is its recurrent
state plus the depthwise-conv tail, and both are gradient channels; the
streaming executor carries both explicitly (streaming._linear_layer_fn). These
tests hold that construction to an independent reference -- the HF model
itself, called monolithically per trajectory, without thundersync:

* (a) streamed / rebuilt values AND parameter gradients == the per-trajectory
  monolithic reference;
* (b) the state-boundary gradient channel: a loss on ONLY the last turn must
  reproduce the reference gradients, including in prompt-region rows, via the
  recurrent-state path through the boundary proxies -- the hybrid sever
  detector;
* (c) the reward-linear backward's update == the barrier update;
* (d) coalesced closes == serial closes == barrier;
* (e) prompt state forked to G branches: gradients sum correctly into the
  shared prompt (multiplicity);
* refusals: the batch/forest path still refuses hybrids loudly, and unverified
  recurrent families refuse everywhere.

Kernel note: the fla Triton kernel's gradients carry an internal noise floor
of about 1e-3 relative (tf32/bf16 intermediates). That is a property of the
kernel, not of the streaming order, so tests (a)-(e) run the HF torch-fallback
kernel on both sides, and a separate test runs fla on both sides: values
tight, gradients at the kernel's own measured floor, plus a severed-channel
detector the noise cannot mask. Training never mixes the two kernels.
"""

from __future__ import annotations


import pytest
import torch

from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from thundersync.engine.segments import Segment, Trajectory, build_forest
from thundersync.engine.plan import compile_plan# noqa: E402
from tests.reference_logprobs import reference_logprobs  # noqa: E402
from thundersync.grpo.reward_linear import RewardLinearBackward  # noqa: E402
from thundersync.opd.objectives import OPDStream, OPDTurnStream  # noqa: E402
from thundersync.engine.step import assert_supported, forest_logprobs  # noqa: E402
from thundersync.engine.streaming import (  # noqa: E402
    STREAM_ATTENTION_NAME,
    StreamingRun,
    register_stream_attention,
)

# fla's Triton kernels (bound by the HF model whenever fla is importable) and
# its FusedRMSNormGated need a GPU; the hybrid family is CUDA-only in practice.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="hybrid gates need CUDA (fla kernels)"
)

DEV = "cuda"
register_stream_attention()
EPS = 1e-6


def build_hybrid(impl, vocab=512, seed=0, torch_kernel=True):
    """Tiny hybrid: linear, linear, full, linear -- built through
    transformers' Qwen3_5ForCausalLM.

    `torch_kernel=True` rebinds each layer's `chunk_gated_delta_rule` instance
    attribute to the HF torch fallback -- the same assignment the HF module
    itself performs when fla is absent, on both the streamed and the reference
    model, so equality gates compare one kernel against itself.
    """
    torch.manual_seed(seed)
    cfg = AutoConfig.for_model(
        "qwen3_5_text",
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
        ],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_value_head_dim=32,
        vocab_size=vocab,
        max_position_embeddings=8192,
        tie_word_embeddings=False,
    )
    model = (
        AutoModelForCausalLM.from_config(cfg, attn_implementation=impl, dtype=torch.float32)
        .to(DEV)
        .eval()
    )
    if torch_kernel:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            torch_chunk_gated_delta_rule,
        )

        for layer in model.model.layers:
            if hasattr(layer, "linear_attn"):
                layer.linear_attn.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
    return model


def make_batch(n_groups=2, n_traj=3, prompt_len=24, vocab=512, seed=1):
    """Trajectories + round-robin arrival events + rewards. Turn lengths vary
    across trajectories AND include a 1-token action turn (the seq_len==1 edge
    the HF cached path special-cases; the executor must not)."""
    gen = torch.Generator().manual_seed(seed)
    r = lambda n: torch.randint(0, vocab, (n,), generator=gen).tolist()  # noqa: E731
    trajs, groups, events, tid = [], {}, [], 0
    for gid in range(n_groups):
        groups[gid] = r(prompt_len)
        per = []
        for i in range(n_traj):
            turns = []
            for k in range(1 + (i + gid) % 3):
                a_len = 1 if (i == 0 and k == 0) else 4 + k + i
                turns.append((r(a_len), r(6 + i) if k % 2 == 0 else []))
            segs = [Segment.prompt(groups[gid])]
            for a, o in turns:
                segs.append(Segment.action(a))
                if o:
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
    gen2 = torch.Generator().manual_seed(seed + 99)
    rewards = {t.traj_id: float(torch.rand(1, generator=gen2)) for t in trajs}
    return trajs, groups, events, rewards


def reference_barrier_grads(model, trajs, rewards):
    """The independent monolithic reference: per-trajectory HF forwards (no
    thundersync machinery -- prompt recomputed per trajectory), the documented GRPO
    advantage formula, ONE backward."""
    per = reference_logprobs(model, trajs)
    by_group: dict[int, list] = {}
    for t in trajs:
        by_group.setdefault(t.group_id, []).append(t.traj_id)
    loss = 0.0
    for gid, tids in by_group.items():
        r = torch.tensor([rewards[t] for t in tids], dtype=torch.float64)
        mean, std = float(r.mean()), float(r.std(unbiased=False)) + EPS
        for t in tids:
            loss = loss + ((rewards[t] - mean) / std) * per[t].sum()
    loss.backward()
    return {
        n: (p.grad.clone() if p.grad is not None else None)
        for n, p in model.named_parameters()
    }


def named_grads(model):
    return {
        n: (p.grad.clone() if p.grad is not None else None)
        for n, p in model.named_parameters()
    }


def assert_named_grads_equal(got, want, ctx, rtol=2e-3, atol=3e-5):
    checked = 0
    for name, g in got.items():
        w = want[name]
        if g is None and w is None:
            continue
        assert g is not None and w is not None, f"{ctx}: {name}: one side None"
        torch.testing.assert_close(
            g, w, rtol=rtol, atol=atol, msg=lambda m, n=name: f"{ctx}: {n}: {m}"
        )
        checked += 1
    assert checked > 5, ctx


def streaming_run(model, groups, events, *, block_tokens=8):
    run = StreamingRun(
        model,
        boundary_cut=True,
        checkpoint=True,
        stream_turns=False,
        rebuild_block_tokens=block_tokens,
    )
    bw = RewardLinearBackward(run, model, eps=EPS)
    trajectories_by_group: dict[int, set[int]] = {}
    for gid, t_id, _toks, _sc in events:
        trajectories_by_group.setdefault(gid, set()).add(t_id)
    for gid, p in groups.items():
        run.open_group(gid, p)
        bw.register_group_trajectories(
            gid, sorted(trajectories_by_group[gid])
        )
    for gid, t_id, toks, sc in events:
        run.append_turn(gid, t_id, toks, sc)
    return run, bw


# ---------------------------------------------------------------- (a) values+grads


def test_streamed_and_rebuilt_equal_monolithic_reference():
    """Gate (a): streamed values, rebuilt values and parameter gradients all
    equal the HF model called per trajectory. boundary_cut off, so gradients
    flow end to end without any the reward-linear backward machinery in the comparison."""
    sm = build_hybrid(STREAM_ATTENTION_NAME, seed=3)
    rm = build_hybrid("sdpa", seed=3)
    rm.load_state_dict(sm.state_dict())
    trajs, groups, events, _ = make_batch(seed=5)

    run = StreamingRun(sm, checkpoint=True, rebuild_block_tokens=8)
    for gid, p in groups.items():
        run.open_group(gid, p)
    for gid, t_id, toks, sc in events:
        run.append_turn(gid, t_id, toks, sc)

    want = reference_logprobs(rm, trajs)
    loss_s = 0.0
    for t in trajs:
        lp_s = run.logprobs(t.traj_id)
        lp_r = run.logprobs(t.traj_id, rebuild=True)
        torch.testing.assert_close(
            lp_s, want[t.traj_id], rtol=1e-4, atol=5e-5,
            msg=lambda m, i=t.traj_id: f"traj {i} streamed values: {m}",
        )
        torch.testing.assert_close(
            lp_r, want[t.traj_id], rtol=1e-4, atol=5e-5,
            msg=lambda m, i=t.traj_id: f"traj {i} rebuilt values: {m}",
        )
        loss_s = loss_s + lp_s.sum()
    loss_s.backward()

    sum(want[t.traj_id].sum() for t in trajs).backward()
    assert_named_grads_equal(
        named_grads(sm), named_grads(rm), "streamed vs reference"
    )


# ----------------------------------------------------- (b) the sever detector


def test_last_turn_loss_reaches_prompt_through_state_boundary():
    """Gate (b): loss on ONLY the last turn of each trajectory must produce
    the reference gradients -- including in the prompt region, reachable only
    through the linear layers' recurrent-state/conv-tail (and the attention
    layers' K/V) boundary proxies. A severed state channel fails this loudly:
    the prompt-token embedding rows lose their gradient mass."""
    sm = build_hybrid(STREAM_ATTENTION_NAME, seed=7)
    rm = build_hybrid("sdpa", seed=7)
    rm.load_state_dict(sm.state_dict())
    trajs, groups, events, _ = make_batch(n_groups=1, n_traj=2, seed=11)

    run = StreamingRun(
        sm, boundary_cut=True, checkpoint=True, stream_turns=False,
        rebuild_block_tokens=8,
    )
    run.open_group(0, groups[0])
    for gid, t_id, toks, sc in events:
        run.append_turn(gid, t_id, toks, sc)

    params = [p for p in sm.parameters() if p.requires_grad]
    boundary = run.boundary(0)
    # per-trajectory closes, loss restricted to the LAST turn's scored tokens
    accum_p = [torch.zeros_like(p) for p in params]
    accum_b = [torch.zeros_like(x) for x in boundary.proxy]
    last_counts = {}
    for t in trajs:
        n_last = sum(run._trajs[t.traj_id].turns[-1][1])
        last_counts[t.traj_id] = n_last
        lp = run.logprobs(t.traj_id, rebuild=True)
        grads = torch.autograd.grad(
            lp[-n_last:].sum(), [*params, *boundary.proxy], allow_unused=True
        )
        for buf, g in zip(accum_p, grads[: len(params)]):
            if g is not None:
                buf.add_(g)
        for buf, g in zip(accum_b, grads[len(params):]):
            if g is not None:
                buf.add_(g)
    # one prompt backward with the combined boundary gradient (the reward-linear backward's shape)
    torch.autograd.backward(boundary.real, grad_tensors=accum_b)
    got = {}
    it = iter(accum_p)
    for n, p in sm.named_parameters():
        if p.requires_grad:
            extra = p.grad if p.grad is not None else 0.0
            got[n] = next(it) + extra
        else:
            got[n] = None

    # reference: same last-turn-only loss on per-trajectory monolithic calls
    want_lp = reference_logprobs(rm, trajs)
    loss = sum(
        want_lp[t.traj_id][-last_counts[t.traj_id]:].sum() for t in trajs
    )
    loss.backward()
    want = named_grads(rm)
    assert_named_grads_equal(got, want, "last-turn loss vs reference")

    prompt_rows = torch.tensor(sorted(set(groups[0])), device=DEV)
    mass = got["model.embed_tokens.weight"][prompt_rows].abs().sum().item()
    ref_mass = want["model.embed_tokens.weight"][prompt_rows].abs().sum().item()
    assert ref_mass > 0, "degenerate reference: prompt got no gradient at all"
    assert mass > 0.5 * ref_mass, (
        f"prompt embedding gradient mass {mass:.3e} vs reference {ref_mass:.3e}: "
        "the state boundary channel is severed"
    )


# ------------------------------------------------------- (c) the reward-linear backward vs barrier


def test_clustered_opd_prompt_forest_preserves_hybrid_state_gradients():
    """Nested prompt nodes carry recurrent state and convolution-tail adjoints."""
    _trajs, _original_groups, events, _rewards = make_batch(
        n_groups=4, n_traj=1, prompt_len=8, seed=503
    )
    groups = {
        0: [1, 2, 3, 4],
        1: [1, 2, 3, 5],
        2: [1, 2, 6, 7],
        3: [1, 2, 6, 8],
    }
    last = {
        tid: index
        for index, (_gid, tid, _tokens, _scored) in enumerate(events)
    }
    scored = {}
    for _gid, tid, _tokens, mask in events:
        scored[tid] = scored.get(tid, 0) + sum(mask)

    def execute(*, forest: bool):
        model = build_hybrid(STREAM_ATTENTION_NAME, seed=509)
        run = StreamingRun(
            model,
            boundary_cut=True,
            checkpoint=True,
            stream_turns=False,
            rebuild_block_tokens=8,
        )
        objective = OPDStream(run, model)
        if forest:
            assert objective.open_groups_shared_prefix(groups) == 2
            assert run.prompt_token_accounting()[
                "physical_prompt_forward_tokens"
            ] == 8
        else:
            for gid, prompt in groups.items():
                objective.open_group(gid, prompt)
        for index, (gid, tid, tokens, mask) in enumerate(events):
            objective.append_turn(gid, tid, tokens, mask)
            if last[tid] == index:
                objective.close_trajectory(
                    tid,
                    teacher_logprobs=torch.zeros(scored[tid], device=DEV),
                )
        for gid in groups:
            objective.close_group(gid)
        objective.assert_safe_to_step()
        assert run.kv_bytes_resident() == 0
        return named_grads(model)

    independent = execute(forest=False)
    forest = execute(forest=True)
    assert_named_grads_equal(
        forest, independent, "clustered hybrid OPD prompt forest",
        rtol=3e-3, atol=4e-5,
    )


def test_turn_arrival_opd_replays_hybrid_state_adjoints():
    """Turn-local OPD plus reverse replay preserves hybrid-model gradients."""
    _trajs, _original_groups, events, _rewards = make_batch(
        n_groups=2, n_traj=2, prompt_len=8, seed=521
    )
    groups = {
        0: [1, 2, 3, 4],
        1: [1, 2, 3, 5],
    }
    last = {
        tid: index
        for index, (_gid, tid, _tokens, _scored) in enumerate(events)
    }
    teacher_by_event = [
        torch.zeros(sum(scored), device=DEV)
        for _gid, _tid, _tokens, scored in events
    ]
    teacher_by_trajectory: dict[int, list[torch.Tensor]] = {}
    for score, (_gid, tid, _tokens, _scored) in zip(
        teacher_by_event, events
    ):
        teacher_by_trajectory.setdefault(tid, []).append(score)

    def execute(*, turn_arrival: bool):
        model = build_hybrid(STREAM_ATTENTION_NAME, seed=523)
        run = StreamingRun(
            model,
            boundary_cut=True,
            checkpoint=True,
            stream_turns=True,
            rebuild_block_tokens=8,
            turn_boundary_rebuild=turn_arrival,
        )
        objective = (
            OPDTurnStream(run, model)
            if turn_arrival
            else OPDStream(run, model)
        )
        objective.open_groups_shared_prefix(groups)
        for index, (gid, tid, tokens, scored) in enumerate(events):
            if turn_arrival:
                objective.append_scored_turn(
                    gid,
                    tid,
                    tokens,
                    scored,
                    teacher_logprobs=teacher_by_event[index],
                )
                if last[tid] == index:
                    objective.close_trajectory(tid)
            else:
                objective.append_turn(gid, tid, tokens, scored)
                if last[tid] == index:
                    objective.close_trajectory(
                        tid,
                        teacher_logprobs=torch.cat(
                            teacher_by_trajectory[tid]
                        ),
                    )
        for gid in groups:
            objective.close_group(gid)
        objective.assert_safe_to_step()
        assert run.kv_bytes_resident() == 0
        return objective.loss_value, named_grads(model)

    close_value, close_gradients = execute(turn_arrival=False)
    turn_value, turn_gradients = execute(turn_arrival=True)
    assert turn_value == pytest.approx(close_value, rel=3e-4, abs=3e-5)
    assert_named_grads_equal(
        turn_gradients,
        close_gradients,
        "turn-arrival hybrid OPD",
        rtol=4e-3,
        atol=5e-5,
    )


def test_reward_linear_equals_barrier():
    sm = build_hybrid(STREAM_ATTENTION_NAME, seed=13)
    rm = build_hybrid("sdpa", seed=13)
    rm.load_state_dict(sm.state_dict())
    trajs, groups, events, rewards = make_batch(seed=17)

    run, bw = streaming_run(sm, groups, events, block_tokens=8)
    for tid in sorted({t for _, t, _, _ in events}):
        bw.close_trajectory(tid, rewards[tid])
    for gid in groups:
        bw.close_group(gid)
    assert run.kv_bytes_resident() == 0, "group close left K/V or state resident"

    want = reference_barrier_grads(rm, trajs, rewards)
    assert_named_grads_equal(named_grads(sm), want, "reward-linear vs barrier")


# --------------------------------------------------- (d) coalesced == serial


def test_coalesced_closes_equal_serial_and_barrier():
    sm = build_hybrid(STREAM_ATTENTION_NAME, seed=19)
    qm = build_hybrid(STREAM_ATTENTION_NAME, seed=19)
    qm.load_state_dict(sm.state_dict())
    rm = build_hybrid("sdpa", seed=19)
    rm.load_state_dict(sm.state_dict())
    trajs, groups, events, rewards = make_batch(seed=23)
    tids = sorted({t for _, t, _, _ in events})
    assert len(tids) >= 4

    runc, bwc = streaming_run(sm, groups, events, block_tokens=8)
    mid = len(tids) // 2 + 1
    batches = [tids[:mid], tids[mid:]]
    assert len({runc.group_of(t) for t in batches[0]}) == 2, (
        "the coalesced batch must mix groups or the gate proves nothing"
    )
    for batch in batches:
        bwc.close_trajectories([(t, rewards[t]) for t in batch])
    for gid in groups:
        bwc.close_group(gid)

    runs, bws = streaming_run(qm, groups, events, block_tokens=8)
    for t in tids:
        bws.close_trajectory(t, rewards[t])
    for gid in groups:
        bws.close_group(gid)

    got = named_grads(sm)
    assert_named_grads_equal(got, named_grads(qm), "coalesced vs serial")
    assert_named_grads_equal(
        got, reference_barrier_grads(rm, trajs, rewards), "coalesced vs barrier"
    )


# --------------------------------------------- (e) fork multiplicity into prompt


def test_prompt_state_forked_to_branches_sums_gradients():
    """G branches all start their linear layers from the SAME prompt state
    proxy; autograd must sum the branch gradients into it, so the combined
    update equals the reference where the prompt contributes through every
    branch. Single group, unit weights, so the multiplicity is isolated from
    the advantage formula."""
    G = 3
    sm = build_hybrid(STREAM_ATTENTION_NAME, seed=29)
    rm = build_hybrid("sdpa", seed=29)
    rm.load_state_dict(sm.state_dict())

    gen = torch.Generator().manual_seed(31)
    r = lambda n: torch.randint(0, 512, (n,), generator=gen).tolist()  # noqa: E731
    prompt = r(20)
    trajs, turns = [], {}
    for i in range(G):
        turns[i] = [(r(3 + 2 * i), [True] * (3 + 2 * i))]
        segs = [Segment.prompt(prompt), Segment.action(turns[i][0][0])]
        trajs.append(Trajectory(segs, group_id=0, traj_id=i))

    run = StreamingRun(
        sm, boundary_cut=True, checkpoint=True, stream_turns=False,
        rebuild_block_tokens=8,
    )
    run.open_group(0, prompt)
    for i in range(G):
        for toks, sc in turns[i]:
            run.append_turn(0, i, toks, sc)

    params = [p for p in sm.parameters() if p.requires_grad]
    boundary = run.boundary(0)
    accum_p = [torch.zeros_like(p) for p in params]
    accum_b = [torch.zeros_like(x) for x in boundary.proxy]
    for i in range(G):
        lp = run.logprobs(i, rebuild=True)
        grads = torch.autograd.grad(
            lp.sum(), [*params, *boundary.proxy], allow_unused=True
        )
        for buf, g in zip(accum_p, grads[: len(params)]):
            if g is not None:
                buf.add_(g)
        for buf, g in zip(accum_b, grads[len(params):]):
            if g is not None:
                buf.add_(g)
    torch.autograd.backward(boundary.real, grad_tensors=accum_b)
    got = {}
    it = iter(accum_p)
    for n, p in sm.named_parameters():
        got[n] = next(it) + (p.grad if p.grad is not None else 0.0)

    want_lp = reference_logprobs(rm, trajs)
    sum(want_lp[i].sum() for i in range(G)).backward()
    assert_named_grads_equal(
        got, named_grads(rm), f"prompt state forked to {G} branches"
    )


# ------------------------------------------------------------ fla kernel test


def test_fla_kernel_values_tight_and_no_severed_channel():
    """The fla Triton kernel is the training path. Values must match
    the monolithic reference tightly. Gradients are gated against a MEASURED
    per-tensor noise floor: the discrepancy between the fla and torch kernels
    on the SAME monolithic computation (single calls, no thundersync, no chaining).
    The streamed update may not deviate from the fla monolithic reference by
    more than that intrinsic kernel noise allows -- decomposed on this exact
    batch, the streaming residual is consistently SMALLER than the kernel's
    own single-call noise (chaining adds nothing), while a severed boundary
    channel adds signal-sized error no noise floor can absorb. The remaining
    fla-vs-torch g-channel noise (dt_bias/A_log/in_proj_a under GRPO's
    advantage cancellation, ~1e-2 absolute L2 here) is a property of the
    kernel a lab already accepts by running fla, disclosed, not of thundersync's
    reordering -- gates (a)-(e) prove the reordering exact on a kernel whose
    backward is bit-stable."""
    sm = build_hybrid(STREAM_ATTENTION_NAME, seed=37, torch_kernel=False)
    rm = build_hybrid("sdpa", seed=37, torch_kernel=False)
    rm.load_state_dict(sm.state_dict())
    rt = build_hybrid("sdpa", seed=37, torch_kernel=True)
    rt.load_state_dict(sm.state_dict())
    trajs, groups, events, rewards = make_batch(seed=41)

    run, bw = streaming_run(sm, groups, events, block_tokens=8)
    for tid in sorted({t for _, t, _, _ in events}):
        bw.close_trajectory(tid, rewards[tid])
    for gid in groups:
        bw.close_group(gid)

    # values: the old-logprobs the close pass defined, vs the reference
    want_lp = reference_logprobs(rm, trajs)
    for t in trajs:
        old = run.old_logprobs(t.traj_id)
        torch.testing.assert_close(
            old, want_lp[t.traj_id], rtol=2e-4, atol=1e-4,
            msg=lambda m, i=t.traj_id: f"traj {i} fla values: {m}",
        )

    want = reference_barrier_grads(rm, trajs, rewards)   # fla, monolithic
    noise = reference_barrier_grads(rt, trajs, rewards)  # torch, monolithic
    got = named_grads(sm)
    checked = 0
    for name, g in got.items():
        w = want[name]
        if g is None and w is None:
            continue
        assert g is not None and w is not None, f"{name}: one side None"
        denom = w.norm().item()
        if denom < 1e-7:
            continue
        floor = (w - noise[name]).norm().item()  # the kernel's own noise
        err = (g - w).norm().item()
        assert err <= max(5e-2 * denom, 2.0 * floor + 1e-7), (
            f"{name}: streamed-vs-monolithic gradient error {err:.3e} exceeds "
            f"both 5% of |grad|={denom:.3e} and 2x the kernel's own "
            f"single-call noise {floor:.3e}; a boundary channel is likely "
            "severed"
        )
        checked += 1
    assert checked > 5

    # the direct sever detector under fla: prompt embedding rows keep the
    # reference's gradient mass
    prompt_rows = torch.tensor(
        sorted(set(groups[0]) | set(groups[1])), device=DEV
    )
    mass = got["model.embed_tokens.weight"][prompt_rows].abs().sum().item()
    ref_mass = want["model.embed_tokens.weight"][prompt_rows].abs().sum().item()
    assert ref_mass > 0
    assert abs(mass - ref_mass) / ref_mass < 5e-2


def test_fla_kernel_state_carry_matches_monolithic_at_kernel_level():
    """Chained chunk_gated_delta_rule calls with carried state must equal ONE
    monolithic fla call -- values and input gradients -- within the kernel's
    own single-call noise. Guards the exact property Phase 1 relies on: the
    kernel's dh0/dht channels are complete."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    torch.manual_seed(43)
    B, T, H, K, V = 1, 96, 2, 32, 32
    q = torch.randn(B, T, H, K, device=DEV, requires_grad=True)
    k = torch.randn(B, T, H, K, device=DEV, requires_grad=True)
    v = torch.randn(B, T, H, V, device=DEV, requires_grad=True)
    g = (-torch.rand(B, T, H, device=DEV)).requires_grad_(True)
    beta = torch.rand(B, T, H, device=DEV, requires_grad=True)

    def run(splits):
        outs, state = [], None
        for s, e in splits:
            o, state = chunk_gated_delta_rule(
                q[:, s:e], k[:, s:e], v[:, s:e], g=g[:, s:e], beta=beta[:, s:e],
                initial_state=state, output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            outs.append(o)
        out = torch.cat(outs, dim=1)
        return out, torch.autograd.grad(out.sum(), [q, k, v, g, beta])

    out_m, g_m = run([(0, T)])
    out_c, g_c = run([(0, 24), (24, 28), (28, 29), (29, 60), (60, T)])
    torch.testing.assert_close(out_c, out_m, rtol=1e-3, atol=1e-3)
    for name, a, b in zip("qkv g beta".split(), g_c, g_m):
        rel = (a - b).norm().item() / (b.norm().item() + 1e-9)
        assert rel < 5e-3, f"d{name}: chained-vs-mono rel L2 {rel:.3e}"


# ---------------------------------------------------------------- refusals


def test_batch_forest_path_still_refuses_hybrid():
    model = build_hybrid("sdpa", seed=47)
    trajs, *_ = make_batch(n_groups=1, n_traj=2, seed=49)
    plan = compile_plan(build_forest(trajs), device=DEV)
    with pytest.raises(NotImplementedError, match="batch/forest"):
        forest_logprobs(model, plan)
    with pytest.raises(NotImplementedError, match="batch/forest"):
        assert_supported(model, recheck=True)


def test_unverified_recurrent_family_refuses_everywhere():
    class MambaMixer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

    class Cfg:
        layer_types = None

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = Cfg()
            self.mixer = MambaMixer()

    with pytest.raises(NotImplementedError, match="recurrent state"):
        assert_supported(FakeModel(), recheck=True, allow_hybrid=True)


def test_declared_hybrid_with_unverified_module_refuses():
    """A stack that declares linear_attention but is not the verified Gated
    DeltaNet layout must refuse, not guess."""
    model = build_hybrid("sdpa", seed=53)
    # sabotage one layer: remove an attribute the executor drives
    del model.model.layers[0].linear_attn.in_proj_qkv
    with pytest.raises(NotImplementedError, match="lacks"):
        assert_supported(model, recheck=True, allow_hybrid=True)


def test_lora_publication_handle_does_not_enter_the_module_graph():
    """The publication handle must not make the module graph cyclic.

    Attaching the PEFT wrapper as a plain attribute of the inner model
    registers it as a submodule -- nn.Module.__setattr__ intercepts Module
    values -- and the wrapper contains the inner model, so state_dict
    recurses forever. The loader must bypass module registration; the handle
    stays reachable for the merged-checkpoint publication and invisible to
    state_dict, named_modules, and parameters.
    """

    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(
        0, str(_Path(__file__).resolve().parents[1] / "src")
    )
    from thundersync.policy_model import apply_lora, attach_publication_handle

    model = build_hybrid(STREAM_ATTENTION_NAME, seed=41)
    reference_keys = set(model.state_dict().keys())
    peft_model, inner = apply_lora(model, rank=4, alpha=8)
    attach_publication_handle(inner, peft_model)

    assert getattr(inner, "_thundersync_peft_model") is peft_model
    assert "_thundersync_peft_model" not in dict(inner.named_children())
    keys = set(inner.state_dict().keys())
    assert not any("_thundersync_peft_model" in key for key in keys)

    # Publication is the run's terminal act: merge_and_unload folds the
    # adapters into plain Linear layers and restores exactly the original
    # key set, which is what the rollout engine loads.
    published = peft_model.merge_and_unload()
    assert set(published.state_dict().keys()) == reference_keys


def test_checkpoint_spans_identify_linear_attention_layers():
    """DeltaNet layers must not vanish from the profiling spans.

    The span's layer index is read from `self_attn.layer_idx`, which a Gated
    DeltaNet layer does not have -- it carries `linear_attn` -- so 48 of this
    family's 64 layers tagged themselves `layer=None`. Their share of the
    recompute and VJP time was therefore unattributable, which is exactly the
    blind spot that hid the attention path's cost until it was measured.
    """

    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
    from thundersync.engine import streaming as streaming_module

    model = build_hybrid(STREAM_ATTENTION_NAME, seed=61)
    kinds = list(model.config.layer_types)
    assert "linear_attention" in kinds and "full_attention" in kinds

    for index, layer in enumerate(model.model.layers):
        resolved = streaming_module._checkpoint_layer_index(layer)
        assert resolved == index, (
            f"layer {index} ({kinds[index]}) resolved to {resolved}"
        )
