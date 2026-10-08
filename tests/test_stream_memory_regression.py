"""Regression gate: streamed attention must not retain each full prefix.

Cat-then-SDPA makes autograd save concatenated ancestor K/V per event, so
retention grows as sum-of-prefixes. `_ChunkedCausalSDPA` saves only per-chunk
references, so the
marginal cost of an event must be FLAT in turn index and the peak materially
lower. Both claims are asserted here at the allocator's level of truth
(`torch.cuda.memory_allocated` / `max_memory_allocated`), streaming the same
1-group, 1-trajectory, many-turn batch through both paths:

* new-path peak < 75% of the legacy concat path's peak at this shape;
* least-squares slope of allocated-vs-tokens over the second half of the turns,
  divided by the first half's: legacy > 1.1 (the growth is present at this
  shape, so the comparison means something), new < 1.1 (flat).

CUDA-only: the claim is about the CUDA caching allocator's bookkeeping; CPU has
no equivalent exact counter.
"""

from __future__ import annotations

import gc

import pytest
import torch

from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from thundersync.engine.streaming import (  # noqa: E402
    STREAM_ATTENTION_NAME,
    StreamingRun,
    register_stream_attention,
)

register_stream_attention()

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="allocator-level gate needs CUDA"
)

VOCAB = 512
PROMPT_LEN = 96
TURN_LEN = 96
TURNS = 16  # >= 12, and deep enough that the prefix term dominates


def build_model(seed=0):
    torch.manual_seed(seed)
    cfg = AutoConfig.for_model(
        "qwen3",
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,  # no GQA: K/V per token = hidden, prefix term large
        head_dim=32,
        vocab_size=VOCAB,
        max_position_embeddings=8192,
        tie_word_embeddings=False,
    )
    return AutoModelForCausalLM.from_config(
        cfg, attn_implementation=STREAM_ATTENTION_NAME, dtype=torch.float32
    ).cuda().eval()


def _tokens(gen, n):
    return torch.randint(0, VOCAB, (n,), generator=gen).tolist()


def _stream(model, *, chunked: bool, turns: int = TURNS):
    """One streamed group; returns (peak_bytes_over_base, per-event allocated
    series over base). Ends with a backward so (a) the retained graph is proven
    usable at this shape on both paths and (b) everything is released before
    the next measurement."""
    gen = torch.Generator().manual_seed(7)  # identical workload for both paths
    run = StreamingRun(model, chunked_sdpa=chunked)
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()

    run.open_group(0, _tokens(gen, PROMPT_LEN))
    series = []
    for _ in range(turns):
        toks = _tokens(gen, TURN_LEN)
        scored = [True] * (TURN_LEN // 2) + [False] * (TURN_LEN - TURN_LEN // 2)
        run.append_turn(0, 0, toks, scored)
        torch.cuda.synchronize()
        series.append(torch.cuda.memory_allocated() - base)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base

    run.logprobs(0).sum().backward()
    del run
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    return peak, series


def _lsq_slope(ys):
    xs = range(len(ys))
    n = len(ys)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx


def _half_slope_ratio(series):
    """Second-half slope of allocated-vs-event over first-half slope. Turns are
    equal-length, so the slope IS the marginal allocated bytes per event; a
    ratio near 1 means the marginal cost does not grow with turn index."""
    n = len(series)
    return _lsq_slope(series[n // 2 :]) / _lsq_slope(series[: n // 2])


def test_streamed_retention_is_linear_not_sum_of_prefixes():
    model = build_model()

    # warm both code paths at small scale: kernels, cublas workspaces, autograd
    # metadata -- so neither measured run pays first-touch costs
    for warm_chunked in (False, True):
        _stream(model, chunked=warm_chunked, turns=2)

    peak_old, series_old = _stream(model, chunked=False)
    peak_new, series_new = _stream(model, chunked=True)

    ratio_old = _half_slope_ratio(series_old)
    ratio_new = _half_slope_ratio(series_new)
    print(
        f"\npeak over base: concat {peak_old / 2**20:.1f} MiB -> "
        f"chunked {peak_new / 2**20:.1f} MiB "
        f"({100 * (1 - peak_new / peak_old):.1f}% lower); "
        f"marginal-slope ratio (2nd half / 1st half): "
        f"concat {ratio_old:.3f}, chunked {ratio_new:.3f}"
    )

    # the disease must be visible in the baseline at this shape, or the
    # comparison proves nothing
    assert ratio_old > 1.1, (
        f"legacy concat path's marginal cost did not grow (ratio {ratio_old:.3f}); "
        "shape too small to exercise the sum-of-prefixes retention"
    )
    # the cure: flat marginal cost, materially lower peak
    assert ratio_new < 1.1, (
        f"chunked path's marginal allocated bytes still grow with turn index "
        f"(ratio {ratio_new:.3f}): the prefix is being retained again"
    )
    assert peak_new < 0.75 * peak_old, (
        f"chunked path peak {peak_new} not >25% below concat peak {peak_old}"
    )
