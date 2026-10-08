"""apply_lora: deterministic adapters, the caller's RNG left alone, and the
target modules chosen per model family."""

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from thundersync.policy_model import (
    HYBRID_LORA_TARGETS,
    STANDARD_LORA_TARGETS,
    apply_lora,
    lora_target_modules,
)

pytest.importorskip("peft")


def _tiny_qwen3() -> torch.nn.Module:
    config = AutoConfig.for_model(
        "qwen3", hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        vocab_size=64, max_position_embeddings=128, tie_word_embeddings=False,
    )
    return AutoModelForCausalLM.from_config(config, dtype=torch.float32)


def _tiny_qwen3_5_hybrid() -> torch.nn.Module:
    config = AutoConfig.for_model(
        "qwen3_5_text", hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_value_head_dim=8,
        vocab_size=64, max_position_embeddings=128, tie_word_embeddings=False,
    )
    return AutoModelForCausalLM.from_config(config, dtype=torch.float32)


def _adapter_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if "lora_" in name
    }


def test_apply_lora_leaves_the_caller_rng_unchanged():
    model = _tiny_qwen3()
    torch.manual_seed(1234)
    seed_before = torch.initial_seed()
    state_before = torch.get_rng_state()
    expected_next = torch.rand(4)
    torch.set_rng_state(state_before)

    apply_lora(model, rank=4, alpha=8)

    assert torch.initial_seed() == seed_before
    assert torch.equal(torch.rand(4), expected_next)


def test_apply_lora_initializes_adapters_independently_of_the_caller_seed():
    torch.manual_seed(1)
    first = _adapter_state(apply_lora(_tiny_qwen3(), rank=4, alpha=8)[1])
    torch.manual_seed(2)
    second = _adapter_state(apply_lora(_tiny_qwen3(), rank=4, alpha=8)[1])

    assert first and first.keys() == second.keys()
    assert all(torch.equal(first[name], second[name]) for name in first)


def test_dense_model_gets_the_standard_targets():
    assert lora_target_modules(_tiny_qwen3()) == STANDARD_LORA_TARGETS


def test_hybrid_model_gets_the_hybrid_targets_and_adapts():
    model = _tiny_qwen3_5_hybrid()
    assert lora_target_modules(model) == HYBRID_LORA_TARGETS

    _, inner = apply_lora(model, rank=4, alpha=8)

    adapted = {
        name.split(".lora_A", 1)[0].rsplit(".", 1)[-1]
        for name, _ in inner.named_parameters()
        if ".lora_A" in name
    }
    assert {"in_proj_qkv", "out_proj", "q_proj", "down_proj"} <= adapted


def test_model_without_a_decoder_stack_is_refused():
    with pytest.raises(NotImplementedError, match="model.model.layers"):
        lora_target_modules(torch.nn.Linear(4, 4))


def test_adapters_are_fp32_and_the_base_keeps_its_dtype():
    model = _tiny_qwen3().to(torch.bfloat16)
    peft_model, _ = apply_lora(model, rank=4, alpha=8)

    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    frozen = [p for p in peft_model.parameters() if not p.requires_grad]
    assert trainable and all(p.dtype == torch.float32 for p in trainable)
    assert frozen and all(p.dtype == torch.bfloat16 for p in frozen)


def test_fused_lora_backward_returns_adapter_dtype_gradients():
    from thundersync.accel.fused_lora import FusedLoRALinear

    torch.manual_seed(0)
    x = torch.randn(5, 16, dtype=torch.bfloat16)
    weight = torch.randn(8, 16, dtype=torch.bfloat16)
    lora_a = torch.randn(4, 16, dtype=torch.float32, requires_grad=True)
    lora_b = torch.randn(8, 4, dtype=torch.float32, requires_grad=True)
    grad_out = torch.randn(5, 8, dtype=torch.bfloat16)

    FusedLoRALinear.apply(x, weight, lora_a, lora_b, 2.0).backward(grad_out)

    reference_a = ((grad_out.float() @ lora_b.detach()).t() @ x.float()) * 2.0
    reference_b = (grad_out.float().t() @ (x.float() @ lora_a.detach().t())) * 2.0
    assert lora_a.grad.dtype == torch.float32 and lora_b.grad.dtype == torch.float32
    torch.testing.assert_close(lora_a.grad, reference_a, rtol=3e-2, atol=3e-1)
    torch.testing.assert_close(lora_b.grad, reference_b, rtol=3e-2, atol=3e-1)
