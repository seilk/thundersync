"""Gates for serving the branch attention with one FlashAttention 4 call."""

from __future__ import annotations


import pytest
from tests.device_kernels import require_fa4
import torch

from thundersync.engine import streaming# noqa: E402


def _branch_inputs(prefix: int, tail: int, seed: int = 5, *,
                   heads_q: int = 24, heads_kv: int = 4, head_dim: int = 256):
    """A branch's queries against the shared prefix plus its own tail."""

    device = torch.device("cuda")
    torch.manual_seed(seed)
    query = torch.randn(
        1, heads_q, tail, head_dim, device=device, dtype=torch.bfloat16
    )
    key = torch.randn(
        prefix + tail, heads_kv, head_dim, device=device, dtype=torch.bfloat16
    )
    value = torch.randn(
        prefix + tail, heads_kv, head_dim, device=device, dtype=torch.bfloat16
    )
    return query, key, value, head_dim ** -0.5


def _run(use_fa4: bool, query, key, value, scale):
    previous = streaming._SPLIT_BOTTOM_RIGHT_FA4
    streaming._SPLIT_BOTTOM_RIGHT_FA4 = use_fa4
    try:
        query = query.detach().requires_grad_(True)
        key = key.detach().requires_grad_(True)
        value = value.detach().requires_grad_(True)
        out = streaming._causal_sdpa(query, [key], [value], scale)
        out.float().pow(2).sum().backward()
        return out, query.grad, key.grad, value.grad
    finally:
        streaming._SPLIT_BOTTOM_RIGHT_FA4 = previous


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FA4 needs CUDA")
@pytest.mark.parametrize("prefix,tail", [(3000, 9288), (2000, 4000)])
def test_fa4_matches_the_split_bottom_right_path(prefix, tail):
    """One FA4 call must compute what the two-call split computes.

    The split path expands key and value six-fold for a kernel with no
    grouped path, runs it twice, and merges by log-sum-exp.  FA4 defines
    ``causal=True`` as bottom-right when the lengths differ and takes
    grouped heads directly.  Measured forward plus backward at
    prefix 3000 with tail 9288: 22.995 ms for the split, 7.023 ms for FA4.
    """

    require_fa4(head_dim=128)
    inputs = _branch_inputs(prefix, tail)
    want = _run(False, *inputs)
    got = _run(True, *inputs)

    assert not torch.equal(want[0], got[0]), (
        "the two paths agreed bit-for-bit, so the FA4 branch was not taken "
        "and this gate proves nothing"
    )
    labels = ("output", "grad_query", "grad_key", "grad_value")
    for label, expected, actual in zip(labels, want, got):
        scale = expected.float().abs().max()
        drift = (expected.float() - actual.float()).abs().max() / scale
        assert drift < 3e-2, f"{label} drifted {drift:.3e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FA4 needs CUDA")
@pytest.mark.parametrize("prefix,tail", [(1024, 160), (15872, 512)])
def test_qwen3_14b_ragged_values_and_vjps_match_independent_attention(prefix, tail):
    """Exercise a 40/8 GQA geometry, including a 16K context."""
    pytest.importorskip("flash_attn.cute.interface")
    inputs = _branch_inputs(prefix, tail, heads_q=40, heads_kv=8, head_dim=128)
    expected = _run(False, *inputs)
    query, key, value, scale = inputs
    query = query.detach().requires_grad_(True)
    key = key.detach().requires_grad_(True)
    value = value.detach().requires_grad_(True)
    out, = streaming._RaggedAttentionIndependentVJP.apply(scale, 1, query, key, value)
    out.float().square().sum().backward()
    for name, actual, reference in zip(
        ("output", "grad_query", "grad_key", "grad_value"),
        (out, query.grad, key.grad, value.grad), expected,
    ):
        relative_l2 = (actual.float() - reference.float()).norm() / reference.float().norm()
        assert relative_l2 < 3e-2, f"{name} relative L2 {relative_l2.item():.4g}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FA4 needs CUDA")
def test_the_split_path_stands_in_when_fa4_is_absent():
    """Availability must be a runtime question, not an import-time one."""

    resolved, func = streaming._FA4_RESOLVED, streaming._FA4_FUNC
    streaming._FA4_RESOLVED, streaming._FA4_FUNC = True, None
    try:
        query, key, value, scale = _branch_inputs(512, 512)
        assert not streaming._fa4_bottom_right_is_available(query, 512)
        out = streaming._causal_sdpa(query, [key], [value], scale)
        assert out.shape == query.shape
    finally:
        streaming._FA4_RESOLVED, streaming._FA4_FUNC = resolved, func
