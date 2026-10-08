"""load_hybrid_causal_lm: key remapping, the documented exclusions, and
coverage in both directions, on a tiny local snapshot (CPU)."""

import json

import pytest
import torch

pytest.importorskip("safetensors")
from safetensors.torch import save_file  # noqa: E402
from transformers import AutoConfig  # noqa: E402

from thundersync.policy_model import load_hybrid_causal_lm  # noqa: E402

CPU = torch.device("cpu")


def _text_config():
    return AutoConfig.for_model(
        "qwen3_5_text", hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_value_head_dim=8,
        vocab_size=64, max_position_embeddings=128, tie_word_embeddings=False,
    )


def _write_snapshot(directory, extra=None):
    """A ForConditionalGeneration-layout snapshot of a tiny text model."""

    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    config = _text_config()
    torch.manual_seed(3)
    model = Qwen3_5ForCausalLM(config).to(torch.bfloat16)
    state = {}
    for key, value in model.state_dict().items():
        name = (
            "model.language_model." + key[len("model."):] if key.startswith("model.") else key
        )
        state[name] = value.detach().clone().contiguous()
    state["model.visual.patch_embed.weight"] = torch.zeros(2, 2, dtype=torch.bfloat16)
    state["mtp.fc.weight"] = torch.zeros(2, 2, dtype=torch.bfloat16)
    state.update(extra or {})
    save_file(state, str(directory / "model-00001-of-00001.safetensors"))
    (directory / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "text_config": config.to_dict()})
    )
    return model


def test_a_hybrid_snapshot_loads_its_text_model_and_skips_vision_and_mtp(tmp_path):
    reference = _write_snapshot(tmp_path)
    model = load_hybrid_causal_lm(tmp_path, device=CPU, attn_implementation="eager")
    expected = reference.state_dict()
    loaded = model.state_dict()
    assert loaded.keys() == expected.keys()
    assert all(torch.equal(loaded[key], expected[key]) for key in expected)


def test_a_remapped_key_the_model_lacks_is_refused(tmp_path):
    _write_snapshot(
        tmp_path, {"model.language_model.bogus.weight": torch.zeros(2, dtype=torch.bfloat16)}
    )
    with pytest.raises(RuntimeError, match="model.bogus.weight"):
        load_hybrid_causal_lm(tmp_path, device=CPU, attn_implementation="eager")


def test_a_key_outside_every_known_prefix_is_refused(tmp_path):
    _write_snapshot(tmp_path, {"vision_head.weight": torch.zeros(2, dtype=torch.bfloat16)})
    with pytest.raises(RuntimeError, match="vision_head.weight"):
        load_hybrid_causal_lm(tmp_path, device=CPU, attn_implementation="eager")


def test_a_missing_parameter_is_refused(tmp_path):
    _write_snapshot(tmp_path)
    shard = tmp_path / "model-00001-of-00001.safetensors"
    from safetensors.torch import load_file

    state = load_file(str(shard))
    del state["lm_head.weight"]
    save_file(state, str(shard))
    with pytest.raises(RuntimeError, match="missing"):
        load_hybrid_causal_lm(tmp_path, device=CPU, attn_implementation="eager")


def test_a_hub_id_is_refused_with_a_clear_error():
    with pytest.raises(ValueError, match="local snapshot directory"):
        load_hybrid_causal_lm("org/hybrid-model", device=CPU, attn_implementation="eager")


class _Reached(Exception):
    pass


def _policy_config(model_path):
    from pathlib import Path

    from thundersync.config import PolicyModelConfig

    return PolicyModelConfig(
        model_path=Path(model_path), disable_fla_causal_conv=True, lora_rank=0,
        lora_alpha=None, fused_lora=False, compile_pointwise_modules=False,
        learning_rate=1e-5, optimizer_state_sharding=False, grad_clip=1.0,
        deterministic_algorithms=False,
    )


def _patch_loaders(monkeypatch, hub_model_type):
    import transformers

    from thundersync import policy_model

    calls = []

    class _Auto:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append((args, kwargs))
            raise _Reached

    class _HubConfig:
        model_type = hub_model_type

    monkeypatch.setattr(policy_model, "AutoModelForCausalLM", _Auto)
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", staticmethod(lambda *_a, **_k: _HubConfig())
    )
    return calls


def test_a_hub_id_is_resolved_by_transformers(monkeypatch):
    from thundersync.policy_model import load_policy_model

    calls = _patch_loaders(monkeypatch, "qwen3")
    with pytest.raises(_Reached):
        load_policy_model(_policy_config("org/name"), CPU, attn_implementation="eager")
    (args, kwargs), = calls
    assert args == ("org/name",) and kwargs["local_files_only"] is False


def test_a_local_snapshot_loads_without_network_access(monkeypatch, tmp_path):
    from thundersync.policy_model import load_policy_model

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    calls = _patch_loaders(monkeypatch, "qwen3")
    with pytest.raises(_Reached):
        load_policy_model(_policy_config(tmp_path), CPU, attn_implementation="eager")
    (args, kwargs), = calls
    assert args == (str(tmp_path),) and kwargs["local_files_only"] is True


def test_a_hybrid_hub_id_is_refused(monkeypatch):
    from thundersync.policy_model import load_policy_model

    calls = _patch_loaders(monkeypatch, "qwen3_5")
    with pytest.raises(ValueError, match="local snapshot directory"):
        load_policy_model(_policy_config("org/hybrid"), CPU, attn_implementation="eager")
    assert not calls
