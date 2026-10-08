"""Gates for the close-time whole-branch rebuild (traj-bwd granularity lever).

the reward-linear backward's close no longer walks the streamed per-event graph: it re-forwards
the trajectory's branch as ONE packed chunk over the prompt boundary proxies
and differentiates that (streaming._rebuild_branch_logprobs, reached through
`logprobs(tid, rebuild=True)`). The invariant is the repo's usual one,
sharpened for this lever:

    the rebuild is the streamed computation, re-executed at a coarser kernel
    granularity -- same values, same gradients, nothing retained afterwards.

the reward-linear backward equality against the barrier backward is already gated by
tests/test_reward_linear.py (which now exercises the rebuild path end to end); this
file gates the rebuild AGAINST THE STREAMED GRAPH directly, so a regression
points at the lever and not at the reward-linear backward.
"""

from __future__ import annotations


import torch

from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from thundersync.engine.streaming import (  # noqa: E402
    STREAM_ATTENTION_NAME,
    StreamingRun,
    register_stream_attention,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
register_stream_attention()


def build_model(vocab=512, seed=0, layers=2, hidden=128):
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
        cfg, attn_implementation=STREAM_ATTENTION_NAME, dtype=torch.float32
    ).to(DEV).eval()


def make_events(n_traj=3, prompt_len=24, vocab=512, seed=1):
    gen = torch.Generator().manual_seed(seed)
    r = lambda n: torch.randint(0, vocab, (n,), generator=gen).tolist()  # noqa: E731
    prompt = r(prompt_len)
    events = []
    for tid in range(n_traj):
        for k in range(2 + tid):
            a, o = r(5 + k + tid), r(7 + tid)
            events.append((0, tid, a + o, [True] * len(a) + [False] * len(o)))
    return prompt, events


def streamed(run, prompt, events):
    run.open_group(0, prompt)
    for gid, tid, toks, sc in events:
        run.append_turn(gid, tid, toks, sc)


# ----------------------------------------------------------------------


def test_rebuild_matches_streamed_values_and_gradients():
    """Same logprob values and the same gradients as walking the streamed
    graph -- for every trajectory, on both executors, with the boundary cut
    armed (the default configuration)."""
    for checkpoint in (False, True):
        model = build_model(seed=3)
        run = StreamingRun(model, boundary_cut=True, checkpoint=checkpoint)
        prompt, events = make_events(seed=5)
        streamed(run, prompt, events)

        params = [p for p in model.parameters() if p.requires_grad]
        proxy = run.boundary(0).proxy
        for tid in sorted({t for _, t, _, _ in events}):
            lp_s = run.logprobs(tid)
            lp_r = run.logprobs(tid, rebuild=True)
            torch.testing.assert_close(
                lp_r, lp_s, rtol=1e-4, atol=1e-5,
                msg=lambda m, i=tid: f"traj {i} ckpt={checkpoint} values: {m}",
            )
            assert torch.equal(lp_s.detach(), run.old_logprobs(tid)), (
                "old-logprobs must stay the STREAMED values (D1)"
            )
            g_s = torch.autograd.grad(
                lp_s.sum(), [*params, *proxy], retain_graph=True,
                allow_unused=True,
            )
            g_r = torch.autograd.grad(
                run.logprobs(tid, rebuild=True).sum(), [*params, *proxy],
                allow_unused=True,
            )
            checked = 0
            for a, b in zip(g_s, g_r):
                if a is None and b is None:
                    continue
                assert a is not None and b is not None
                torch.testing.assert_close(
                    a, b, rtol=2e-3, atol=2e-5,
                    msg=lambda m, i=tid: f"traj {i} ckpt={checkpoint} grads: {m}",
                )
                checked += 1
            assert checked > 5


def test_rebuild_is_transient():
    """The rebuild retains nothing: no K/V, no chunk records, and it works
    after the streamed graph was already freed (the close-time ordering)."""
    model = build_model(seed=7)
    run = StreamingRun(model, boundary_cut=True, checkpoint=True)
    prompt, events = make_events(seed=11)
    streamed(run, prompt, events)

    before_bytes = run.kv_bytes_resident()
    before_chunks = len(run._chunks)
    lp = run.logprobs(0, rebuild=True)
    assert lp.numel() > 0
    assert run.kv_bytes_resident() == before_bytes, "rebuild leaked K/V"
    assert len(run._chunks) == before_chunks, "rebuild leaked a chunk record"

    # the close-time ordering: streamed graph freed FIRST, then the rebuild
    ref = run.logprobs(0, rebuild=True)
    run.free_trajectory(0)
    lp2 = run.logprobs(0, rebuild=True)
    torch.testing.assert_close(lp2, ref, rtol=1e-5, atol=1e-6)
    grads = torch.autograd.grad(
        lp2.sum(), run.boundary(0).proxy, allow_unused=True
    )
    assert any(g is not None for g in grads), (
        "boundary proxies received no gradient from the post-free rebuild"
    )
