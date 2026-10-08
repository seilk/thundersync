"""The entropy output must obey the same law as the logprob it rides with.

`fused_logprob_entropy` adds a second per-position scalar to the chunked
kernel. The claims to verify are the same three as for the logprob alone:
identical values, identical gradients (for every combination of which output
the loss reads), and peak memory that still does not scale with N.
"""

from __future__ import annotations


import pytest
import torch

from thundersync.accel.fused_logprob import (  # noqa: E402
    _accum_dtype,
    fused_logprob_entropy,
)

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def naive_logprob_entropy(hidden, weight, target):
    """Reference path: materialise the full [N, V] block, let autograd do backward."""
    acc = _accum_dtype(hidden.dtype)
    logits = torch.nn.functional.linear(hidden.to(acc), weight.to(acc))
    logp = torch.log_softmax(logits, dim=-1)
    lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    ent = -(logp.exp() * logp).sum(-1)
    return lp, ent


# ----------------------------------------------------------------------
# value equivalence
# ----------------------------------------------------------------------


@pytest.mark.parametrize("chunk", [1, 7, 64, 4096])
def test_matches_naive_values(chunk):
    N, D, V = 129, 64, 501
    h = torch.randn(N, D, dtype=torch.float32, device=DEV)
    w = torch.randn(V, D, dtype=torch.float32, device=DEV) * 0.05
    t = torch.randint(0, V, (N,), device=DEV)

    lp, ent = fused_logprob_entropy(h, w, t, chunk=chunk)
    want_lp, want_ent = naive_logprob_entropy(h, w, t)
    torch.testing.assert_close(lp, want_lp, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(ent, want_ent, rtol=1e-5, atol=1e-6)


def test_entropy_is_within_its_analytic_bounds():
    """0 <= H <= log V, whatever the logits."""
    N, D, V = 200, 32, 311
    h = torch.randn(N, D, device=DEV) * 3.0  # sharpen some rows
    w = torch.randn(V, D, device=DEV) * 0.2
    t = torch.randint(0, V, (N,), device=DEV)
    _, ent = fused_logprob_entropy(h, w, t, chunk=64)
    assert float(ent.min()) >= 0.0
    assert float(ent.max()) <= float(torch.log(torch.tensor(float(V)))) + 1e-5


def test_bf16_input_accumulates_and_returns_fp32():
    N, D, V = 64, 32, 203
    h = torch.randn(N, D, dtype=torch.bfloat16, device=DEV)
    w = torch.randn(V, D, dtype=torch.bfloat16, device=DEV) * 0.05
    t = torch.randint(0, V, (N,), device=DEV)
    lp, ent = fused_logprob_entropy(h, w, t, chunk=16)
    assert lp.dtype == torch.float32 and ent.dtype == torch.float32
    want_lp, want_ent = naive_logprob_entropy(h, w, t)
    torch.testing.assert_close(lp, want_lp, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(ent, want_ent, rtol=1e-2, atol=1e-2)


# ----------------------------------------------------------------------
# gradient equivalence
# ----------------------------------------------------------------------


def test_matches_naive_gradients_reading_both_outputs():
    """A loss with independent coefficients on logprob and entropy."""
    N, D, V = 96, 48, 307
    t = torch.randint(0, V, (N,), device=DEV)
    c_lp = torch.randn(N, device=DEV)
    c_ent = torch.randn(N, device=DEV)

    h0 = torch.randn(N, D, dtype=torch.float32, device=DEV)
    w0 = torch.randn(V, D, dtype=torch.float32, device=DEV) * 0.05

    ha = h0.clone().requires_grad_(True)
    wa = w0.clone().requires_grad_(True)
    lp, ent = fused_logprob_entropy(ha, wa, t, chunk=16)
    (c_lp * lp + c_ent * ent).sum().backward()

    hb = h0.clone().requires_grad_(True)
    wb = w0.clone().requires_grad_(True)
    lp2, ent2 = naive_logprob_entropy(hb, wb, t)
    (c_lp * lp2 + c_ent * ent2).sum().backward()

    torch.testing.assert_close(ha.grad, hb.grad, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(wa.grad, wb.grad, rtol=1e-4, atol=1e-6)


def _gradcheck_inputs():
    N, D, V = 11, 6, 23
    h = torch.randn(N, D, dtype=torch.float64, device=DEV, requires_grad=True)
    w = (torch.randn(V, D, dtype=torch.float64, device=DEV) * 0.1).requires_grad_(True)
    t = torch.randint(0, V, (N,), device=DEV)
    return h, w, t


def test_gradcheck_logprob_only():
    """Loss reads only the logprob output; entropy grad arrives as zeros."""
    h, w, t = _gradcheck_inputs()
    assert torch.autograd.gradcheck(
        lambda a, b: fused_logprob_entropy(a, b, t, chunk=4)[0],
        (h, w),
        eps=1e-6,
        atol=1e-8,
    )


def test_gradcheck_entropy_only():
    """Loss reads only the entropy output; this isolates the dH/dlogits term."""
    h, w, t = _gradcheck_inputs()
    assert torch.autograd.gradcheck(
        lambda a, b: fused_logprob_entropy(a, b, t, chunk=4)[1],
        (h, w),
        eps=1e-6,
        atol=1e-8,
    )


def test_gradcheck_both_outputs_combined():
    """Both upstream grads nonzero at once, with unequal weights."""
    h, w, t = _gradcheck_inputs()

    def f(a, b):
        lp, ent = fused_logprob_entropy(a, b, t, chunk=4)
        return lp + 0.7 * ent

    assert torch.autograd.gradcheck(f, (h, w), eps=1e-6, atol=1e-8)


def test_frozen_weight_gets_no_gradient():
    """With a frozen unembedding, the backward must not allocate a weight grad."""
    N, D, V = 32, 16, 97
    h = torch.randn(N, D, device=DEV, requires_grad=True)
    w = torch.randn(V, D, device=DEV) * 0.05  # requires_grad=False
    t = torch.randint(0, V, (N,), device=DEV)
    lp, ent = fused_logprob_entropy(h, w, t, chunk=8)
    (lp.sum() + ent.sum()).backward()
    assert h.grad is not None
    assert w.grad is None


# ----------------------------------------------------------------------
# memory
# ----------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_chunking_bounds_peak_memory():
    """Entropy support must not reintroduce O(N*V) memory."""
    D, V = 512, 20_000
    w = torch.randn(V, D, dtype=torch.bfloat16, device="cuda") * 0.02

    def peak(n, chunk):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        h = torch.randn(n, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        t = torch.randint(0, V, (n,), device="cuda")
        lp, ent = fused_logprob_entropy(h, w, t, chunk=chunk)
        (lp.sum() + ent.sum()).backward()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()

    small, large = peak(4096, 256), peak(16384, 256)
    growth = large / small
    print(
        f"\npeak memory: N=4096 -> {small / 2**20:.1f} MiB, "
        f"N=16384 -> {large / 2**20:.1f} MiB, growth {growth:.2f}x"
    )
    assert growth < 3.0, (
        f"4x more positions grew peak memory {growth:.1f}x; chunking is not bounding it"
    )
