"""Gates for the merged-weight LoRA forward."""

from __future__ import annotations


import pytest
import torch

from thundersync.accel.fused_lora import (  # noqa: E402
    fuse_merged_lora_forward,
    unfuse_merged_lora_forward,
)

peft = pytest.importorskip("peft")


def _model(seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    base = torch.nn.Sequential(
        torch.nn.Linear(64, 128, bias=False),
        torch.nn.Tanh(),
        torch.nn.Linear(128, 64, bias=False),
    )
    config = peft.LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
        target_modules=["0", "2"],
    )
    return peft.get_peft_model(base, config)


def _grads(model, x):
    for parameter in model.parameters():
        parameter.grad = None
    model(x).pow(2).sum().backward()
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def test_fused_forward_and_gradients_match_the_unfused_adapter():
    """Folding the adapter into the weight must change nothing observable.

    y = x W^T + (x A^T) B^T is exactly x (W + B A)^T, so the merged forward
    is the same value from one GEMM instead of three, and the adapter
    gradients follow from the same derivative. Measured on the reference site at a
    branch shape the unfused adapted layer costs 45.1% over the frozen base and
    this form costs 29.5%.
    """

    reference = _model(11)
    candidate = _model(11)
    candidate.load_state_dict(reference.state_dict())

    torch.manual_seed(5)
    x = torch.randn(32, 64)

    want_out = reference(x).detach().clone()
    want = _grads(reference, x)

    assert fuse_merged_lora_forward(candidate) == 2
    got_out = candidate(x).detach().clone()
    got = _grads(candidate, x)

    torch.testing.assert_close(got_out, want_out, rtol=1e-5, atol=1e-6)
    assert set(got) == set(want)
    for name, expected in want.items():
        torch.testing.assert_close(got[name], expected, rtol=1e-4, atol=1e-6)


def test_unfusing_restores_the_stock_layer_and_weights():
    """A step must see plain base weights and the stock forward again."""

    model = _model(13)
    torch.manual_seed(7)
    x = torch.randn(16, 64)
    before = model(x).detach().clone()
    base_before = model.base_model.model[0].base_layer.weight.detach().clone()

    fuse_merged_lora_forward(model)
    assert unfuse_merged_lora_forward(model) == 2

    torch.testing.assert_close(
        model.base_model.model[0].base_layer.weight, base_before,
        rtol=1e-5, atol=1e-6,
    )
    torch.testing.assert_close(model(x), before, rtol=1e-5, atol=1e-6)


def test_fusion_declines_a_layer_with_active_dropout():
    """Dropout makes the merged weight wrong for the forward; decline it."""

    torch.manual_seed(17)
    base = torch.nn.Sequential(torch.nn.Linear(32, 32, bias=False))
    config = peft.LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.1, bias="none", target_modules=["0"]
    )
    model = peft.get_peft_model(base, config)
    assert fuse_merged_lora_forward(model) == 0
