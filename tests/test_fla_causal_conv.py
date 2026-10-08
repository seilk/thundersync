"""Gates for routing the hybrid causal convolution to the fla kernel."""

from __future__ import annotations


import pytest
import torch
import torch.nn.functional as F

from thundersync.accel.fla_causal_conv import (  # noqa: E402
    _fla_causal_conv1d_fn,
    install_fla_causal_conv,
)


class _LinearAttentionStub(torch.nn.Module):
    """The two attributes the installer keys on."""

    def __init__(self, conv_dim: int = 32, kernel: int = 4) -> None:
        super().__init__()
        self.causal_conv1d_fn = None
        self.conv1d = torch.nn.Conv1d(
            conv_dim, conv_dim, kernel_size=kernel,
            groups=conv_dim, padding=kernel - 1,
        )


def test_installer_reroutes_only_unclaimed_linear_attention_layers():
    """A layer that already holds a kernel keeps it; a full-attention layer
    has no such attribute and must be skipped."""

    model = torch.nn.ModuleDict(
        {
            "linear_0": _LinearAttentionStub(),
            "linear_1": _LinearAttentionStub(),
            "full_0": torch.nn.Linear(8, 8),
        }
    )
    claimed = _LinearAttentionStub()
    sentinel = object()
    claimed.causal_conv1d_fn = sentinel
    model["linear_2"] = claimed

    pytest.importorskip("fla.modules.convolution")
    assert install_fla_causal_conv(model) == 2
    assert model["linear_0"].causal_conv1d_fn is not None
    assert model["linear_1"].causal_conv1d_fn is not None
    assert model["linear_2"].causal_conv1d_fn is sentinel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fla needs CUDA")
def test_fla_kernel_matches_the_torch_fallback_within_bf16_rounding():
    """The reroute changes the kernel, not the operation.

    On the reference site at the live branch shape the fallback runs the forward and
    backward in 6.147 ms and this path in 1.019 ms, a 6.03x reduction,
    agreeing to 3.6e-3 relative -- bf16 rounding at this magnitude.
    """

    pytest.importorskip("fla.modules.convolution")
    device = torch.device("cuda")
    torch.manual_seed(3)
    conv_dim, kernel, tokens = 256, 4, 512
    conv = torch.nn.Conv1d(
        conv_dim, conv_dim, kernel_size=kernel, groups=conv_dim,
        padding=kernel - 1,
    ).to(device, torch.bfloat16)
    source = torch.randn(
        1, tokens, conv_dim, device=device, dtype=torch.bfloat16
    )

    want_x = source.detach().requires_grad_(True)
    want = F.silu(conv(want_x.transpose(1, 2))[:, :, :tokens]).transpose(1, 2)

    got_x = source.detach().requires_grad_(True)
    got = _fla_causal_conv1d_fn(
        got_x.transpose(1, 2),
        conv.weight.squeeze(1),
        conv.bias,
        activation="silu",
    ).transpose(1, 2)

    scale = want.float().abs().max()
    assert (want.float() - got.float()).abs().max() / scale < 1e-2

    upstream = torch.randn_like(want)
    want.backward(upstream)
    got.backward(upstream)
    grad_scale = want_x.grad.float().abs().max()
    assert (
        want_x.grad.float() - got_x.grad.float()
    ).abs().max() / grad_scale < 1e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fla needs CUDA")
def test_streaming_linear_layer_matches_between_conv_backends():
    """ThunderSync's own linear-attention path must agree across the two kernels.

    ``_linear_layer_fn`` calls the convolution directly rather than through
    the transformers layer attribute, so routing the transformers attribute
    alone leaves the trainer on the fallback.
    This covers both segment branches, including the continuation that
    supplies real left context and drops the outputs belonging to it.
    """

    pytest.importorskip("fla.modules.convolution")
    transformers = pytest.importorskip("transformers")
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    from thundersync.engine import streaming

    config = transformers.AutoConfig.for_model(
        "qwen3_5",
        hidden_size=256, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=64, intermediate_size=512,
        vocab_size=128, linear_conv_kernel_dim=4,
    )
    config = getattr(config, "text_config", config)
    device = torch.device("cuda")
    torch.manual_seed(19)
    index = list(config.layer_types).index("linear_attention")
    layer = Qwen3_5DecoderLayer(config, layer_idx=index).to(
        device, torch.bfloat16
    )

    half = 48
    opening = torch.randn(
        1, half, config.hidden_size, device=device, dtype=torch.bfloat16
    )
    continuing = torch.randn(
        1, half, config.hidden_size, device=device, dtype=torch.bfloat16
    )

    def run(use_fla: bool):
        previous = streaming._LINEAR_ATTENTION_FLA_CONV
        streaming._LINEAR_ATTENTION_FLA_CONV = use_fla
        try:
            first = streaming._linear_layer_fn(
                layer, opening, ((0, half, 0),)
            )
            second = streaming._linear_layer_fn(
                layer, continuing, ((0, half, 1),), *first[1:]
            )
            return first[0], second[0]
        finally:
            streaming._LINEAR_ATTENTION_FLA_CONV = previous

    want_open, want_cont = run(False)
    got_open, got_cont = run(True)

    assert not torch.equal(want_open, got_open), (
        "the two kernels produced bit-identical output, so the fla branch "
        "was not taken and this gate proves nothing"
    )
    for want, got, label in (
        (want_open, got_open, "opening"),
        (want_cont, got_cont, "continuation"),
    ):
        scale = want.float().abs().max()
        drift = (want.float() - got.float()).abs().max() / scale
        assert drift < 5e-2, f"{label} segment drifted {drift:.3e}"
