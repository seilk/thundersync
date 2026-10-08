"""Learned RMSNorm scaling must not round before radial-gradient cancellation."""

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from thundersync.accel.rms_norm import use_stable_qwen3_rms_norm


def test_replacement_preserves_weights_epsilon_mode_and_state_keys():
    model = torch.nn.Sequential(Qwen3RMSNorm(16, eps=1e-6), torch.nn.LayerNorm(16))
    model.eval()
    parameters = dict(model.named_parameters())
    state = {name: value.clone() for name, value in model.state_dict().items()}
    assert use_stable_qwen3_rms_norm(model) == 1
    assert model[0].variance_epsilon == 1e-6
    assert not model[0].training
    assert isinstance(model[1], torch.nn.LayerNorm)
    assert use_stable_qwen3_rms_norm(model) == 0
    assert dict(model.named_parameters()).keys() == parameters.keys()
    for name, value in model.named_parameters():
        assert value is parameters[name]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("record_graph", [False, True])
def test_stable_vjp_preserves_primal_bytes(dtype, record_graph):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(71)
    model = torch.nn.Sequential(Qwen3RMSNorm(128, eps=1e-6)).to(device, dtype)
    with torch.no_grad():
        model[0].weight.copy_(torch.linspace(0.4, 3.0, 128, device=device))
    x = torch.randn(64, 128, device=device, dtype=dtype)
    with torch.set_grad_enabled(record_graph):
        expected = model(x)
        use_stable_qwen3_rms_norm(model)
        torch.testing.assert_close(model(x), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the stable norm VJP is CUDA")
def test_nearly_radial_vjp_matches_fp32_with_learned_bf16_scales():
    torch.manual_seed(71)
    model = torch.nn.Sequential(Qwen3RMSNorm(128, eps=1e-6)).to("cuda", torch.bfloat16)
    with torch.no_grad():
        model[0].weight.copy_(torch.linspace(0.4, 3.0, 128, device="cuda"))
    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = model[0].weight
    upstream = (x.detach().float() / weight.detach().float()).to(torch.bfloat16)
    reference_x = x.detach().float().requires_grad_()
    normalized = reference_x * torch.rsqrt(reference_x.square().mean(-1, keepdim=True) + 1e-6)
    reference = normalized * weight.detach().float()
    expected = torch.autograd.grad(reference, reference_x, upstream.float())[0]
    use_stable_qwen3_rms_norm(model)
    actual = torch.autograd.grad(model(x), x, upstream)[0]
    relative = (actual.float() - expected).norm() / expected.norm()
    assert float(relative) < 0.01
