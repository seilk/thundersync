"""The chunked attention op backwards the flash split from its saved state.

On the grouped flash split `_ChunkedCausalSDPA` keeps the merged output and
the merged log-sum-exp and runs the split's flash backward on them, instead
of re-running the split forward at backward to rebuild the same numbers.
The forward is the same arithmetic on the same numbers and matches bit for
bit, and so do dK/dV. dQ is accumulated atomically by the kernel, so two
runs of the recompute already differ in a few elements by a rounding step;
the saved-state dQ stays within that floor. Other paths (one query row, a
pack) keep the recompute. The saved output is the storage the output
projection keeps for its own backward, so a streamed turn retains only the
log-sum-exp more.
"""

from __future__ import annotations

import importlib.util

import pytest


def _torch_is_real() -> bool:
    try:
        spec = importlib.util.find_spec("torch")
    except ValueError:
        return False
    return spec is not None and spec.origin is not None


if not _torch_is_real():
    pytest.skip("needs a real torch", allow_module_level=True)
import torch  # noqa: E402

from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from thundersync.engine import streaming# noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or streaming._FLASH_SDPA is None,
    reason="the flash split needs CUDA",
)

HEAD_DIM = 128
GROUPINGS = [(32, 8), (40, 8)]
# (query rows, key/value chunk lengths): a turn over a prompt and earlier
# turns, a short turn, a prompt-only prefix
SHAPES = [
    (157, [492, 157, 157, 157]),
    (42, [300, 42]),
    (64, [640, 64]),
    (2, [16, 2]),
]


def _inputs(tail, chunks, seed, heads, kv_heads):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    query = torch.randn(1, tail, heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16,
                        generator=generator).transpose(1, 2)
    keys = [torch.randn(n, kv_heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16,
                        generator=generator) for n in chunks]
    values = [torch.randn(n, kv_heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16,
                          generator=generator) for n in chunks]
    cotangent = torch.randn(1, tail, heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16,
                            generator=generator).transpose(1, 2)
    return query, keys, values, cotangent


def _run(monkeypatch, saved, tail, chunks, seed, heads, kv_heads, keep_state=True):
    monkeypatch.setattr(streaming, "_SAVED_STATE_FLASH_SPLIT", saved)
    query, keys, values, cotangent = _inputs(tail, chunks, seed, heads, kv_heads)
    leaves = [query.detach().requires_grad_()]
    leaves += [k.detach().requires_grad_() for k in keys]
    leaves += [v.detach().requires_grad_() for v in values]
    n = len(keys)
    backwards = []
    original = streaming._flash_split_grouped_backward

    def counted(*args):
        # the measured call only, not the grouped trial's own small heads
        if args[1].shape[1] == heads:
            backwards.append(args[1].shape)
        return original(*args)

    monkeypatch.setattr(streaming, "_flash_split_grouped_backward", counted)
    before = dict(streaming.attention_path_counts())
    out = streaming._ChunkedCausalSDPA.apply(
        leaves[0], HEAD_DIM ** -0.5, n, keep_state, "auto", *leaves[1:]
    )
    grads = torch.autograd.grad(out, leaves, cotangent)
    after = streaming.attention_path_counts()
    monkeypatch.setattr(streaming, "_flash_split_grouped_backward", original)
    counts = {name: after.get(name, 0) - before.get(name, 0) for name in after}
    counts = {name: count for name, count in counts.items() if count}
    return out.detach(), grads, counts, len(backwards)


def _within_the_dq_floor(got, want, again):
    floor = float((again.float() - want.float()).norm())
    base = float(want.float().norm())
    difference = float((got.float() - want.float()).norm())
    assert difference <= max(4.0 * floor, 1e-4 * base), (difference, floor, base)


@pytest.mark.parametrize("heads,kv_heads", GROUPINGS)
@pytest.mark.parametrize("tail,chunks", SHAPES)
def test_the_saved_state_backward_is_the_recompute(monkeypatch, tail, chunks, heads, kv_heads):
    seed = len(chunks) + tail
    shape = dict(heads=heads, kv_heads=kv_heads)
    recompute = _run(monkeypatch, False, tail, chunks, seed, **shape)
    again = _run(monkeypatch, False, tail, chunks, seed, **shape)
    saved = _run(monkeypatch, True, tail, chunks, seed, **shape)
    # the saved state is backwarded once, without the forward re-run
    assert saved[3] == 1
    assert recompute[3] == 1  # the re-run reaches the same backward
    assert saved[2] == recompute[2] == {"flash_split": 1, "flash_split_primal": 1}
    assert torch.equal(saved[0], recompute[0])
    _within_the_dq_floor(saved[1][0], recompute[1][0], again[1][0])
    for got, want in zip(saved[1][1:], recompute[1][1:], strict=True):
        assert torch.equal(got, want)


def test_one_query_row_and_a_pack_keep_the_recompute(monkeypatch):
    shape = dict(heads=32, kv_heads=8)
    # one query row takes the expanded split, whose backward re-runs
    single = _run(monkeypatch, True, 1, [492, 124, 1], 5, **shape)
    assert single[3] == 0
    reference = _run(monkeypatch, False, 1, [492, 124, 1], 5, **shape)
    assert torch.equal(single[0], reference[0])
    # a pack segment does not keep its output
    segment = _run(monkeypatch, True, 157, [492, 157], 6, keep_state=False, **shape)
    recompute = _run(monkeypatch, False, 157, [492, 157], 6, **shape)
    again = _run(monkeypatch, False, 157, [492, 157], 6, **shape)
    assert segment[3] == 1  # through the recompute's own split
    assert torch.equal(segment[0], recompute[0])
    _within_the_dq_floor(segment[1][0], recompute[1][0], again[1][0])
    for got, want in zip(segment[1][1:], recompute[1][1:], strict=True):
        assert torch.equal(got, want)


def _student(seed=0):
    streaming.register_stream_attention()
    torch.manual_seed(seed)
    config = AutoConfig.for_model(
        "qwen3", hidden_size=512, intermediate_size=1024, num_hidden_layers=3,
        num_attention_heads=8, num_key_value_heads=2, head_dim=64, vocab_size=512,
        max_position_embeddings=8192, tie_word_embeddings=False,
    )
    return AutoModelForCausalLM.from_config(
        config, attn_implementation=streaming.STREAM_ATTENTION_NAME, dtype=torch.bfloat16,
    ).cuda().eval()


def _stream(model, turns):
    run = streaming.StreamingRun(model)
    generator = torch.Generator().manual_seed(3)
    run.open_group(0, torch.randint(0, 512, (96,), generator=generator).tolist())
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    for turn in range(turns):
        tokens = torch.randint(0, 512, (48,), generator=generator).tolist()
        run.append_turn(0, 0, tokens, [False] * 8 + [True] * 40)
    torch.cuda.synchronize()
    retained = torch.cuda.memory_allocated() - base
    loss = run.logprobs(0).float().sum()
    loss.backward()
    return run, retained, loss.detach()


def test_a_streamed_turn_retains_only_the_log_sum_exp_more(monkeypatch):
    model = _student()
    turns, layers, heads = 6, 3, 8
    monkeypatch.setattr(streaming, "_SAVED_STATE_FLASH_SPLIT", False)
    _run_, recompute, loss_recompute = _stream(model, turns)
    del _run_
    monkeypatch.setattr(streaming, "_SAVED_STATE_FLASH_SPLIT", True)
    before = dict(streaming.attention_path_counts())
    _run_, saved, loss_saved = _stream(model, turns)
    del _run_
    after = streaming.attention_path_counts()
    assert after.get("flash_split", 0) - before.get("flash_split", 0) == turns * layers
    # the same forward: bitwise-equal scores
    assert torch.equal(loss_saved, loss_recompute)
    # per layer and turn: an FP32 LSE row per head and token, and the
    # allocator's granularity for each kernel's small state tensors; a
    # saved output of its own would add a query-sized tensor
    lse = 48 * heads * 4
    allowance = turns * layers * (lse + 8 * 512)
    query_sized = 48 * heads * 64 * 2
    assert saved - recompute <= allowance, (saved, recompute)
    assert allowance < turns * layers * query_sized
