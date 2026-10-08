"""The acceleration must be invisible: same values, same gradients, less memory.

Every mechanism here claims to remove work that is *provably unread*. That is
only true if the output is indistinguishable from the naive path -- in value and
in gradient. A memory saving that perturbs the update is not an optimisation, it
is a different algorithm.

So each test compares against the reference implementation rather than against
an expected constant.
"""

from __future__ import annotations


import pytest
import torch

from thundersync.accel.fused_logprob import (  # noqa: E402
    fused_logprob,
    gathered_logprobs,
    naive_logprobs,
)

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------
# fused logprob: value and gradient equivalence
# ----------------------------------------------------------------------


@pytest.mark.parametrize("chunk", [1, 7, 64, 4096])
def test_fused_matches_naive_values(chunk):
    N, D, V = 129, 64, 501
    h = torch.randn(N, D, dtype=torch.float32, device=DEV)
    w = torch.randn(V, D, dtype=torch.float32, device=DEV) * 0.05
    ids = torch.randint(0, V, (N,), device=DEV)
    mask = torch.rand(N, device=DEV) < 0.3

    got = gathered_logprobs(h, w, ids, mask, chunk=chunk)
    want = naive_logprobs(h, w, ids, mask)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


def test_fused_matches_naive_gradients():
    """The update must be identical, not just the loss value."""
    N, D, V = 96, 48, 307
    ids = torch.randint(0, V, (N,), device=DEV)
    mask = torch.rand(N, device=DEV) < 0.4
    coeff = torch.randn(N, device=DEV)

    h0 = torch.randn(N, D, dtype=torch.float32, device=DEV)
    w0 = torch.randn(V, D, dtype=torch.float32, device=DEV) * 0.05

    ha = h0.clone().requires_grad_(True)
    wa = w0.clone().requires_grad_(True)
    (coeff * gathered_logprobs(ha, wa, ids, mask, chunk=16)).sum().backward()

    hb = h0.clone().requires_grad_(True)
    wb = w0.clone().requires_grad_(True)
    (coeff * naive_logprobs(hb, wb, ids, mask)).sum().backward()

    torch.testing.assert_close(ha.grad, hb.grad, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(wa.grad, wb.grad, rtol=1e-4, atol=1e-6)


def test_gradcheck_against_autograd():
    """Independent check of the hand-written backward, in double precision."""
    N, D, V = 11, 6, 23
    h = torch.randn(N, D, dtype=torch.float64, device=DEV, requires_grad=True)
    w = torch.randn(V, D, dtype=torch.float64, device=DEV, requires_grad=True) * 0.1
    w = w.detach().requires_grad_(True)
    t = torch.randint(0, V, (N,), device=DEV)
    assert torch.autograd.gradcheck(
        lambda a, b: fused_logprob(a, b, t, chunk=4), (h, w), eps=1e-6, atol=1e-8
    )


def test_frozen_weight_gets_no_gradient():
    """With a frozen unembedding, the backward must not allocate a weight grad."""
    N, D, V = 32, 16, 97
    h = torch.randn(N, D, device=DEV, requires_grad=True)
    w = torch.randn(V, D, device=DEV) * 0.05  # requires_grad=False
    t = torch.randint(0, V, (N,), device=DEV)
    fused_logprob(h, w, t, chunk=8).sum().backward()
    assert h.grad is not None
    assert w.grad is None


def test_no_scored_positions_is_not_an_error():
    N, D, V = 16, 8, 41
    h = torch.randn(N, D, device=DEV)
    w = torch.randn(V, D, device=DEV)
    ids = torch.randint(0, V, (N,), device=DEV)
    out = gathered_logprobs(h, w, ids, torch.zeros(N, dtype=torch.bool, device=DEV))
    assert out.shape == (N,) and float(out.abs().sum()) == 0.0


def test_position_zero_is_never_scored():
    """Token 0 has no predecessor; scoring it would read a garbage hidden state."""
    N, D, V = 8, 4, 17
    h = torch.randn(N, D, device=DEV)
    w = torch.randn(V, D, device=DEV)
    ids = torch.randint(0, V, (N,), device=DEV)
    mask = torch.ones(N, dtype=torch.bool, device=DEV)
    assert float(gathered_logprobs(h, w, ids, mask)[0]) == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_chunking_bounds_peak_memory():
    """The whole point: peak logit memory must not scale with N."""
    D, V = 512, 20_000
    w = torch.randn(V, D, dtype=torch.bfloat16, device="cuda") * 0.02

    def peak(n, chunk):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        h = torch.randn(n, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        t = torch.randint(0, V, (n,), device="cuda")
        fused_logprob(h, w, t, chunk=chunk).sum().backward()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()

    small, large = peak(4096, 256), peak(16384, 256)
    growth = large / small
    assert growth < 3.0, (
        f"4x more positions grew peak memory {growth:.1f}x; chunking is not bounding it"
    )


def test_operator_profiling_reads_its_environment_lazily_and_refuses_invalid_modes(
    monkeypatch,
):
    from thundersync.accel import operator_profiling

    monkeypatch.setattr(operator_profiling, "_mode", None)
    monkeypatch.setenv(operator_profiling.PROFILE_RANGES_ENV, "sometimes")
    with pytest.raises(ValueError, match=operator_profiling.PROFILE_RANGES_ENV):
        operator_profiling.operator_profiling_mode()

    monkeypatch.setenv(operator_profiling.PROFILE_RANGES_ENV, " NVTX ")
    assert operator_profiling.operator_profiling_mode() == "nvtx"
    monkeypatch.setattr(operator_profiling, "_mode", None)
    monkeypatch.delenv(operator_profiling.PROFILE_RANGES_ENV)
    assert not operator_profiling.operator_profiling_enabled()
