"""F1: the streamed execution is the monolithic execution, reordered.

This is the falsification gate for design decision D1 (docs/STREAMING.md).
Three independent comparisons, because they fail differently:

* against the per-trajectory reference (no sharing, no streaming, full softmax)
  -- catches everything, including errors the two fast paths might share;
* gradients, not just values -- the cross-chunk K/V and boundary-hidden channels
  are invisible to a value check (detaching either yields correct losses and
  wrong gradients, the failure mode this project has hit twice);
* old == new as BITWISE tensor equality -- D1's claim is identity, not
  closeness, so the test uses torch.equal, not assert_close.

Arrival order is adversarial on purpose: turns arrive round-robin across
trajectories and groups, never one trajectory at a time.
"""

from __future__ import annotations


import math

import pytest
import torch
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from thundersync.engine.segments import Segment, Trajectory, build_forest
from thundersync.engine.plan import compile_plan# noqa: E402
from tests.reference_logprobs import reference_logprobs  # noqa: E402
from thundersync.engine.step import forest_logprobs, register_attention, split_by_trajectory  # noqa: E402
from thundersync.engine.step import ATTENTION_NAME  # noqa: E402
from thundersync.engine.streaming import (  # noqa: E402
    STREAM_ATTENTION_NAME,
    StreamingRun,
    register_stream_attention,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
register_attention()
register_stream_attention()


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


def synced_models(seed=0):
    s = build_model(STREAM_ATTENTION_NAME, seed=seed)
    r = build_model("sdpa", seed=seed)
    r.load_state_dict(s.state_dict())
    g = build_model(ATTENTION_NAME, seed=seed)
    g.load_state_dict(s.state_dict())
    return s, r, g


def make_batch(n_groups=2, n_traj=3, prompt_len=24, vocab=512, seed=1):
    """Trajectories plus their event stream in adversarial round-robin order."""
    gen = torch.Generator().manual_seed(seed)
    r = lambda n: torch.randint(0, vocab, (n,), generator=gen).tolist()  # noqa: E731

    trajs, groups, events = [], {}, []
    tid = 0
    for gid in range(n_groups):
        groups[gid] = r(prompt_len)
        turn_lists = []
        for i in range(n_traj):
            n_turns = 1 + (i + gid) % 3
            turns = []
            for k in range(n_turns):
                a, o = r(4 + k + i), r(6 + i)
                turns.append((a, o))
            segs = [Segment.prompt(groups[gid])]
            for a, o in turns:
                segs.append(Segment.action(a))
                segs.append(Segment.observation(o))
            trajs.append(Trajectory(segs, group_id=gid, traj_id=tid))
            turn_lists.append((tid, turns))
            tid += 1
        # round-robin arrival within the group
        depth = 0
        while any(depth < len(t) for _, t in turn_lists):
            for t_id, turns in turn_lists:
                if depth < len(turns):
                    a, o = turns[depth]
                    events.append((gid, t_id, a + o, [True] * len(a) + [False] * len(o)))
            depth += 1
    return trajs, groups, events


def run_stream(model, groups, events):
    run = StreamingRun(model)
    for gid, prompt in groups.items():
        run.open_group(gid, prompt)
    for gid, tid, tokens, scored in events:
        run.append_turn(gid, tid, tokens, scored)
    return run


def total_loss(per_traj, seed=7):
    loss = 0.0
    for tid in sorted(per_traj):
        lp = per_traj[tid]
        g = torch.Generator(device=DEV).manual_seed(1000 + tid)
        loss = loss + (torch.randn(lp.shape, generator=g, device=DEV) * lp).sum()
    return loss


# ----------------------------------------------------------------------


def test_streamed_values_match_the_independent_reference():
    sm, rm, _ = synced_models(seed=3)
    trajs, groups, events = make_batch(seed=5)
    run = run_stream(sm, groups, events)
    want = reference_logprobs(rm, trajs)
    for t in trajs:
        got = run.logprobs(t.traj_id)
        assert got.shape == want[t.traj_id].shape, f"traj {t.traj_id} scored count"
        torch.testing.assert_close(got, want[t.traj_id], rtol=1e-4, atol=1e-5)


def test_streamed_gradients_match_the_reference():
    """The cross-chunk channels carry gradient or this fails."""
    sm, rm, _ = synced_models(seed=11)
    trajs, groups, events = make_batch(seed=13)

    run = run_stream(sm, groups, events)
    total_loss({t.traj_id: run.logprobs(t.traj_id) for t in trajs}).backward()
    total_loss(reference_logprobs(rm, trajs)).backward()

    a, b = dict(sm.named_parameters()), dict(rm.named_parameters())
    checked = 0
    for name, p in a.items():
        if p.grad is None and b[name].grad is None:
            continue
        assert p.grad is not None and b[name].grad is not None, name
        torch.testing.assert_close(
            p.grad, b[name].grad, rtol=2e-3, atol=2e-5,
            msg=lambda m, n=name: f"{n}: {m}",
        )
        checked += 1
    assert checked > 5


def test_early_turns_receive_gradient_through_later_turns():
    """Sever-the-KV-path detector, direct form: a loss ONLY on the last turn
    must still produce gradient at parameters via the first turn's K/V."""
    sm, rm, _ = synced_models(seed=17)
    trajs, groups, events = make_batch(n_groups=1, n_traj=1, seed=19)
    run = run_stream(sm, groups, events)
    tid = trajs[0].traj_id

    # loss reads only the final scored token
    run.logprobs(tid)[-1].backward()
    ref = reference_logprobs(rm, trajs)
    ref[tid][-1].backward()

    for name, p in dict(sm.named_parameters()).items():
        q = dict(rm.named_parameters())[name]
        if q.grad is None:
            continue
        torch.testing.assert_close(
            p.grad, q.grad, rtol=2e-3, atol=2e-5,
            msg=lambda m, n=name: f"{n}: cross-chunk gradient path broken: {m}",
        )


def test_old_equals_new_by_identity_not_tolerance():
    """D1's actual claim: bitwise equality, because they are the same tensors."""
    sm, _, _ = synced_models(seed=23)
    trajs, groups, events = make_batch(seed=29)
    run = run_stream(sm, groups, events)
    for t in trajs:
        new = run.logprobs(t.traj_id).detach()
        old = run.old_logprobs(t.traj_id)
        assert torch.equal(new, old), f"traj {t.traj_id}: old != new bitwise"


def test_streamed_matches_monolithic_forest():
    """The reorder invariant against the batch trainer itself."""
    sm, _, gm = synced_models(seed=31)
    trajs, groups, events = make_batch(seed=37)
    run = run_stream(sm, groups, events)
    plan = compile_plan(build_forest(trajs), device=DEV)
    mono = split_by_trajectory(plan, forest_logprobs(gm, plan))
    for t in trajs:
        torch.testing.assert_close(
            run.logprobs(t.traj_id), mono[t.traj_id], rtol=1e-4, atol=1e-5
        )


def test_prompt_is_forwarded_once_per_group():
    """CSE composes with streaming: KV residency counts the prompt once."""
    sm, _, _ = synced_models(seed=41)
    trajs, groups, events = make_batch(n_groups=1, n_traj=4, prompt_len=64, seed=43)
    run = run_stream(sm, groups, events)
    # chunk 0 is the group prompt; no other chunk may contain prompt tokens
    prompt_chunks = [c for c in run._chunks if c.parent is None]
    assert len(prompt_chunks) == 1
    total = sum(c.n_tokens for c in run._chunks)
    logical = sum(t.n_tokens for t in trajs)
    assert total < logical, "streaming re-forwarded the shared prompt"


def test_checkpointed_model_is_refused():
    """The known-unsound configuration must be an error, not a wrong gradient."""
    sm = build_model(STREAM_ATTENTION_NAME, seed=47)
    sm.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    sm.train()
    with pytest.raises(NotImplementedError, match="checkpoint"):
        StreamingRun(sm)


def test_trajectory_id_cannot_change_groups():
    model = build_model(STREAM_ATTENTION_NAME, seed=49)
    run = StreamingRun(model)
    run.open_group(0, [1, 2, 3])
    run.open_group(1, [4, 5, 6])
    run.append_turn(0, 17, [7, 8], [True, False])
    with pytest.raises(ValueError, match="belongs to group 0"):
        run.append_turn(1, 17, [9, 10], [True, False])


def test_turn_boundary_rebuild_requires_deterministic_eval_mode():
    model = build_model(STREAM_ATTENTION_NAME, seed=51).train()
    with pytest.raises(ValueError, match=r"requires model\.eval\(\)"):
        StreamingRun(
            model,
            boundary_cut=True,
            checkpoint=True,
            turn_boundary_rebuild=True,
        )


def test_deep_chain_last_chunk_loss_reaches_first_chunk_kv():
    """Dropped-dQ/dK/dV detector for the chunked SDPA op.

    A chain of prompt + 4 turn chunks; the loss reads ONLY the last chunk's
    final scored token. Its gradient must flow through the custom op's per-chunk
    dK/dV all the way to the FIRST chunks' K/V -- observed directly with hooks
    on the stored K/V graph tensors -- and every parameter gradient must equal
    the independent reference. A backward that dropped any of dQ/dK/dV (the
    missing-gradient logsumexp trap) would
    keep the loss value bitwise and fail both checks."""
    sm, rm, _ = synced_models(seed=53)
    gen = torch.Generator().manual_seed(59)
    r = lambda n: torch.randint(0, 512, (n,), generator=gen).tolist()  # noqa: E731

    prompt = r(16)
    turns = [(r(5 + k), r(4)) for k in range(4)]
    segs = [Segment.prompt(prompt)]
    for a, o in turns:
        segs.append(Segment.action(a))
        segs.append(Segment.observation(o))
    traj = Trajectory(segs, group_id=0, traj_id=0)

    run = StreamingRun(sm)
    run.open_group(0, prompt)
    for a, o in turns:
        run.append_turn(0, 0, a + o, [True] * len(a) + [False] * len(o))
    chain = run._trajs[0].chain
    assert len(chain) >= 4, "test must exercise a multi-chunk (>=3) chain"

    # watch gradient ARRIVE at the earliest chunks' stored K/V graph tensors
    hits: dict[tuple[int, int], float] = {}
    for cid in chain[:2]:  # the prompt chunk and the first turn chunk
        for layer in run._kv.layers():
            k, v = run._kv.get(layer, cid)
            for j, t in enumerate((k, v)):
                t.register_hook(
                    lambda g, key=(cid, 2 * layer + j): hits.__setitem__(
                        key, float(g.abs().sum())
                    )
                )

    run.logprobs(0)[-1].backward()
    ref = reference_logprobs(rm, [traj])
    ref[0][-1].backward()

    n_watched = 2 * len(run._kv.layers()) * 2
    assert len(hits) == n_watched, (
        f"gradient reached only {len(hits)}/{n_watched} watched K/V tensors: "
        "the chunked SDPA backward dropped per-chunk dK/dV terms"
    )
    assert all(h > 0 for h in hits.values()), f"zero K/V gradient in {hits}"

    checked = 0
    for name, p in dict(sm.named_parameters()).items():
        q = dict(rm.named_parameters())[name]
        if q.grad is None:
            continue
        torch.testing.assert_close(
            p.grad, q.grad, rtol=2e-3, atol=2e-5,
            msg=lambda m, n=name: f"{n}: chunked SDPA gradient wrong: {m}",
        )
        checked += 1
    assert checked > 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_split_bottom_right_causal_matches_the_masked_reference():
    """The split fast path must reproduce the bottom-right causal attention.

    `causal_lower_right` as an SDPA mask forces a slower backend.  The split
    computes the same softmax from an unmasked prefix and a square causal
    tail, merged by log-sum-exp, and its backward runs each half against the
    merged output and log-sum-exp.
    """

    from torch.nn.attention.bias import causal_lower_right

    from thundersync.engine import streaming as streaming_module

    if (
        streaming_module._CUDNN_SDPA is None
        or streaming_module._CUDNN_SDPA_BACKWARD is None
    ):
        pytest.skip("cuDNN attention entry points are unavailable")

    torch.manual_seed(0)
    heads, dim, q_len, kv_len = 8, 64, 256, 700
    scale = 1.0 / math.sqrt(dim)
    query = torch.randn(
        1, heads, q_len, dim, device="cuda", dtype=torch.bfloat16,
        requires_grad=True,
    )
    key = torch.randn(
        1, heads, kv_len, dim, device="cuda", dtype=torch.bfloat16,
        requires_grad=True,
    )
    value = torch.randn(
        1, heads, kv_len, dim, device="cuda", dtype=torch.bfloat16,
        requires_grad=True,
    )

    reference = F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=causal_lower_right(q_len, kv_len), scale=scale,
    )
    candidate = streaming_module._SplitBottomRightCausal.apply(
        query, key, value, kv_len - q_len, scale
    )
    torch.testing.assert_close(candidate, reference, rtol=2e-2, atol=1e-2)

    want = torch.autograd.grad(
        reference.float().pow(2).sum(), [query, key, value], retain_graph=True
    )
    got = torch.autograd.grad(
        candidate.float().pow(2).sum(), [query, key, value]
    )
    for name, expected, actual in zip("qkv", want, got, strict=True):
        relative = (
            (expected - actual).float().norm() / expected.float().norm()
        ).item()
        assert relative < 1e-2, f"{name}: relative gradient error {relative}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_split_bottom_right_primal_matches_the_masked_reference():
    """The no-grad primal split must reproduce the masked attention.

    The primal runs under ``torch.no_grad`` and keeps nothing for backward, so
    the split costs only the two half outputs while they merge.  Its result
    still has to equal the bottom-right causal attention the graph path uses.
    """

    from torch.nn.attention.bias import causal_lower_right

    from thundersync.engine import streaming as streaming_module

    if (
        streaming_module._CUDNN_SDPA is None
        or streaming_module._CUDNN_SDPA_BACKWARD is None
    ):
        pytest.skip("cuDNN attention entry points are unavailable")

    torch.manual_seed(1)
    heads, dim, q_len, kv_len = 8, 64, 256, 700
    scale = 1.0 / math.sqrt(dim)
    query = torch.randn(1, heads, q_len, dim, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, heads, kv_len, dim, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(1, heads, kv_len, dim, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        reference = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=causal_lower_right(q_len, kv_len), scale=scale,
        )
        candidate = streaming_module._split_bottom_right_primal(
            query, key, value, kv_len - q_len, scale
        )
    torch.testing.assert_close(candidate, reference, rtol=2e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_split_bottom_right_causal_accepts_grouped_key_value_heads():
    """The split must take grouped key/value heads without expanding them.

    Expanding eight key/value heads to forty materializes five times the
    transient and its backward sums the gradient back down.  cuDNN takes the
    grouped shape directly: 4.14 ms against 4.38 ms for forward plus backward
    at the branch shape.
    """

    from thundersync.engine import streaming as streaming_module

    if (
        streaming_module._CUDNN_SDPA is None
        or streaming_module._CUDNN_SDPA_BACKWARD is None
    ):
        pytest.skip("cuDNN attention entry points are unavailable")

    torch.manual_seed(2)
    heads, kv_heads, dim, q_len, kv_len = 8, 2, 64, 256, 700
    scale = 1.0 / math.sqrt(dim)
    query = torch.randn(
        1, heads, q_len, dim, device="cuda", dtype=torch.bfloat16,
        requires_grad=True,
    )
    key = torch.randn(
        1, kv_heads, kv_len, dim, device="cuda", dtype=torch.bfloat16,
        requires_grad=True,
    )
    value = torch.randn(
        1, kv_heads, kv_len, dim, device="cuda", dtype=torch.bfloat16,
        requires_grad=True,
    )
    repeats = heads // kv_heads
    expanded = streaming_module._SplitBottomRightCausal.apply(
        query,
        key.repeat_interleave(repeats, dim=1),
        value.repeat_interleave(repeats, dim=1),
        kv_len - q_len,
        scale,
    )
    grouped = streaming_module._SplitBottomRightCausal.apply(
        query, key, value, kv_len - q_len, scale
    )
    torch.testing.assert_close(grouped, expanded, rtol=2e-2, atol=1e-2)

    want = torch.autograd.grad(
        expanded.float().pow(2).sum(), [query, key, value], retain_graph=True
    )
    got = torch.autograd.grad(
        grouped.float().pow(2).sum(), [query, key, value]
    )
    for name, expected, actual in zip("qkv", want, got, strict=True):
        relative = (
            (expected - actual).float().norm() / expected.float().norm()
        ).item()
        assert relative < 1e-2, f"{name}: relative gradient error {relative}"
