"""FlashAttention 4 registered as a transformers attention implementation."""

from __future__ import annotations


import pytest
from tests.device_kernels import require_fa4
import torch

from thundersync.accel.fa4_attention import (  # noqa: E402
    ATTENTION_NAME,
    fa4_attention_forward,
    register_fa4_attention,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FA4 needs CUDA")
def test_the_shared_kernel_matches_the_reference_attention():
    """A baseline left on a slower kernel would manufacture a win.

    ThunderSync reaches FA4
    through its own dispatch; a baseline runs the model's own attention,
    so it needs this registration to be on the same kernel. Forward plus
    backward at the branch shape measures 21.471 ms on the bundled
    kernel and 9.240 ms on FA4, which is larger than the scheduling
    effect being claimed.
    """

    require_fa4(head_dim=256)
    device = torch.device("cuda")
    torch.manual_seed(23)
    tokens, heads_q, heads_kv, head_dim = 512, 24, 4, 256
    scale = head_dim ** -0.5
    query = torch.randn(
        1, heads_q, tokens, head_dim, device=device, dtype=torch.bfloat16
    )
    key = torch.randn(
        1, heads_kv, tokens, head_dim, device=device, dtype=torch.bfloat16
    )
    value = torch.randn(
        1, heads_kv, tokens, head_dim, device=device, dtype=torch.bfloat16
    )

    expanded_key = key.repeat_interleave(heads_q // heads_kv, dim=1)
    expanded_value = value.repeat_interleave(heads_q // heads_kv, dim=1)
    want = torch.nn.functional.scaled_dot_product_attention(
        query, expanded_key, expanded_value, is_causal=True, scale=scale
    ).transpose(1, 2)

    got, weights = fa4_attention_forward(
        torch.nn.Module(), query, key, value, None, scaling=scale
    )
    assert weights is None
    assert got.shape == want.shape, "the caller reshapes heads-last output"
    drift = (want.float() - got.float()).abs().max() / want.float().abs().max()
    assert drift < 3e-2


def test_registration_reports_whether_the_kernel_is_usable():
    """A silent failure would leave the baseline on another kernel while
    the report claimed the shared one."""

    usable = register_fa4_attention()
    if not usable:
        pytest.skip("FA4 is not importable here")
    from transformers.modeling_utils import AttentionInterface

    assert ATTENTION_NAME in AttentionInterface._global_mapping


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FA4 needs CUDA")
def test_attention_dropout_is_refused_rather_than_ignored():
    """Silently dropping dropout would change the objective without saying so."""

    pytest.importorskip("flash_attn.cute.interface")
    device = torch.device("cuda")
    shape = (1, 4, 64, 256)
    tensor = torch.randn(*shape, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="dropout"):
        fa4_attention_forward(
            torch.nn.Module(), tensor, tensor, tensor, None,
            scaling=0.0625, dropout=0.1,
        )
