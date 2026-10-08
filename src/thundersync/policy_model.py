"""The policy model a trainer rank loads, its LoRA adapters, and its state digests."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from thundersync.config import PolicyModelConfig
from thundersync.accel.fla_causal_conv import install_fla_causal_conv
from thundersync.accel.fused_lora import fuse_merged_lora_forward

__all__ = [
    "HYBRID_EXCLUDED_PREFIXES",
    "HYBRID_LORA_TARGETS",
    "HYBRID_SNAPSHOT_MODEL_TYPE",
    "STANDARD_LORA_TARGETS",
    "apply_lora",
    "attach_publication_handle",
    "compile_pointwise_modules",
    "load_hybrid_causal_lm",
    "load_policy_model",
    "lora_target_modules",
    "require_native_bf16",
]


POINTWISE_COMPILE_MODULE_NAMES = (
    "mlp",
    "input_layernorm",
    "post_attention_layernorm",
)


def compile_pointwise_modules(model: torch.nn.Module) -> int:
    """Compile each decoder layer's MLP and norms, leaving attention eager.

    The streaming executor owns the attention path, so only the pointwise
    modules are compiled.  The bound forward is replaced rather than the
    module, because wrapping the module renames every parameter in
    ``state_dict`` and would change the checkpoint format.

    Inductor writes and executes generated modules from its cache directory,
    so ``TORCHINDUCTOR_CACHE_DIR`` must not resolve to a ``noexec`` mount.
    """

    compiled = 0
    for layer in _decoder_layers(model):
        for name in POINTWISE_COMPILE_MODULE_NAMES:
            module = getattr(layer, name, None)
            if module is None:
                continue
            module.forward = torch.compile(module.forward, dynamic=True)
            compiled += 1
    return compiled


def require_native_bf16(device: torch.device) -> None:
    """Refuse a device without native bf16 arithmetic.

    The trainer holds weights, activations and optimizer inputs in bf16.
    ``including_emulation=False`` asks whether the device computes bf16
    itself; an emulated path would run, but slowly and without the numerics
    the measurements assume, so it is refused rather than taken silently.
    """

    if device.type != "cuda":
        return
    with torch.cuda.device(device):
        if torch.cuda.is_bf16_supported(including_emulation=False):
            return
        major, minor = torch.cuda.get_device_capability(device)
        name = torch.cuda.get_device_name(device)
    raise RuntimeError(
        f"the trainer runs in bf16 and {name} (compute capability "
        f"{major}.{minor}) at {device} has no native bf16 support"
    )


# Checkpoint key prefixes of a hybrid vision-language snapshot that the text
# model does not load: the vision tower and the multi-token-prediction head.
HYBRID_EXCLUDED_PREFIXES = ("model.visual.", "mtp.")
# The config.json model_type of a snapshot that load_hybrid_causal_lm loads.
HYBRID_SNAPSHOT_MODEL_TYPE = "qwen3_5"


def load_hybrid_causal_lm(
    snapshot: Path,
    *,
    device: torch.device,
    attn_implementation: str,
) -> torch.nn.Module:
    """Text model plus lm_head from a hybrid vision-language snapshot.

    ``snapshot`` must be a local snapshot directory; a Hub model ID raises
    ``ValueError`` (download the snapshot first, e.g. with
    ``huggingface_hub.snapshot_download``).

    The checkpoint ships the ForConditionalGeneration layout
    (``model.language_model.*`` / ``model.visual.*`` / ``mtp.*`` /
    ``lm_head.*``); ``Qwen3_5ForCausalLM`` wants ``model.*``, so keys are
    remapped on load. The vision tower and the MTP head are excluded -- the
    learner trains the language model the rollout engine serves. Coverage is
    verified both ways: every entry of the constructed model's state dict
    must be filled from the checkpoint, and every checkpoint key outside the
    two excluded prefixes must land in it; either failure raises
    ``RuntimeError``.
    """

    from safetensors.torch import load_file
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5TextConfig,
    )
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForCausalLM,
        Qwen3_5TextRotaryEmbedding,
    )

    if not Path(snapshot).is_dir():
        raise ValueError(
            f"the hybrid snapshot loader reads a local snapshot directory; "
            f"{str(snapshot)!r} is not one (download the snapshot first)"
        )
    require_native_bf16(device)
    with open(Path(snapshot) / "config.json", encoding="utf-8") as handle:
        raw = json.load(handle)
    text_config = Qwen3_5TextConfig(**raw["text_config"])
    text_config._attn_implementation = attn_implementation

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            model = Qwen3_5ForCausalLM(text_config)
    finally:
        torch.set_default_dtype(previous_dtype)
    model.to_empty(device=device)
    # Non-persistent buffers (rotary inv_freq) are garbage after to_empty;
    # rebuild the rotary module for real, fp32 as its own constructor makes it.
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(
        config=text_config, device=device
    )

    wanted = set(model.state_dict().keys())
    loaded: set[str] = set()
    unexpected: set[str] = set()
    prefix = "model.language_model."
    for shard in sorted(glob.glob(str(Path(snapshot) / "model*.safetensors"))):
        shard_state = load_file(shard, device=str(device))
        renamed = {}
        for key, value in shard_state.items():
            if key.startswith(prefix):
                renamed["model." + key[len(prefix):]] = value
            elif key == "lm_head.weight":
                renamed[key] = value
            elif not key.startswith(HYBRID_EXCLUDED_PREFIXES):
                unexpected.add(key)
        if renamed:
            result = model.load_state_dict(renamed, strict=False)
            unexpected.update(result.unexpected_keys)
            loaded |= set(renamed) & wanted
        del shard_state, renamed
        torch.cuda.empty_cache()
    missing = wanted - loaded
    if missing:
        raise RuntimeError(
            f"checkpoint coverage: {len(missing)} parameters missing, "
            f"e.g. {sorted(missing)[:5]}"
        )
    if unexpected:
        raise RuntimeError(
            f"checkpoint coverage: {len(unexpected)} checkpoint keys match no "
            f"parameter, e.g. {sorted(unexpected)[:5]}"
        )
    model.config.use_cache = False
    return model


# LoRA targets for a hybrid decoder: full-attention q/k/v/o, the MLP
# projections, and the Gated DeltaNet projections (all nn.Linear;
# conv1d, A_log, and dt_bias stay frozen).
HYBRID_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
]
# LoRA targets for a dense decoder: attention q/k/v/o and the MLP projections.
STANDARD_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
HYBRID_MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_text"})
# Seed of the generator state LoRA adapters are initialized from.
LORA_INIT_SEED = 820_042


def _decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """The decoder layer stack at ``model.model.layers``, or a refusal."""

    layers = getattr(getattr(model, "model", None), "layers", None)
    if not isinstance(layers, torch.nn.ModuleList):
        raise NotImplementedError(
            f"{type(model).__name__} has no decoder layer stack at "
            "model.model.layers; only causal LMs with that layout are supported"
        )
    return layers


def lora_target_modules(model: torch.nn.Module) -> list[str]:
    """The linear modules ``apply_lora`` wraps in ``model``.

    A hybrid model (``HYBRID_MODEL_TYPES``) gets ``HYBRID_LORA_TARGETS``.
    Any other model gets ``STANDARD_LORA_TARGETS`` and must expose every one
    of them in every decoder layer; a model that does not is refused rather
    than adapted partially.
    """

    model_type = getattr(getattr(model, "config", None), "model_type", None)
    layers = _decoder_layers(model)
    if model_type in HYBRID_MODEL_TYPES:
        return list(HYBRID_LORA_TARGETS)
    for index, layer in enumerate(layers):
        present = {
            name.rsplit(".", 1)[-1]
            for name, module in layer.named_modules()
            if isinstance(module, torch.nn.Linear)
        }
        missing = [name for name in STANDARD_LORA_TARGETS if name not in present]
        if missing:
            raise NotImplementedError(
                f"model_type {model_type!r} is not a supported LoRA family: "
                f"decoder layer {index} lacks the linear modules {missing}"
            )
    return list(STANDARD_LORA_TARGETS)


def apply_lora(
    model: torch.nn.Module,
    *,
    rank: int,
    alpha: int,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Wrap the model with LoRA adapters; return (peft model, inner model).

    The streaming machinery drives the CausalLM directly, so it receives the
    inner module -- the adapters are injected in place and the frozen base
    keeps requires_grad False, which is exactly the trainable-parameter
    filter RewardLinearBackward applies to its statistics.

    Adapter initialization is identical across runs and ranks: the LoRA
    A-matrices are drawn from the CPU generator and the generators of
    the model's CUDA devices, each seeded with ``LORA_INIT_SEED``. Those
    generators are restored afterwards, so the caller's RNG state is the
    same after the call as before it. Adapter parameters are FP32; the base
    weights keep the dtype they were loaded in.
    """

    from peft import LoraConfig, get_peft_model

    target_modules = lora_target_modules(model)
    cuda_devices = sorted(
        {
            parameter.device.index
            for parameter in model.parameters()
            if parameter.device.type == "cuda" and parameter.device.index is not None
        }
    )
    with torch.random.fork_rng(devices=cuda_devices):
        torch.random.default_generator.manual_seed(LORA_INIT_SEED)
        for index in cuda_devices:
            with torch.cuda.device(index):
                torch.cuda.manual_seed(LORA_INIT_SEED)
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.0,
            bias="none",
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(model, lora_config)
    # The frozen base keeps its dtype; the trainable adapters are held in
    # FP32, so their gradients and optimizer state are FP32 as well.
    for parameter in peft_model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    return peft_model, peft_model.base_model.model


def attach_publication_handle(
    model: torch.nn.Module, peft_model: torch.nn.Module
) -> None:
    """Keep the PEFT wrapper reachable without entering the module graph.

    Publication needs the wrapper -- a merged checkpoint is what the rollout
    engine can load without adapter support -- but a plain attribute
    assignment registers the wrapper as a submodule and the wrapper contains
    the model, so state_dict recurses forever. Writing the instance dict
    directly keeps the handle out of _modules.
    """

    object.__setattr__(model, "_thundersync_peft_model", peft_model)


def _snapshot_model_type(path: Path) -> str | None:
    """The ``model_type`` of a local snapshot's config.json, or of a Hub ID's."""

    if not Path(path).is_dir():
        from transformers import AutoConfig

        return getattr(AutoConfig.from_pretrained(str(path)), "model_type", None)
    config_path = Path(path) / "config.json"
    if not config_path.is_file():
        return None
    with open(config_path, encoding="utf-8") as handle:
        return json.load(handle).get("model_type")


def load_policy_model(
    config: PolicyModelConfig,
    device: torch.device,
    *,
    attn_implementation: str,
) -> torch.nn.Module:
    """Load the policy model a trainer rank trains, in training mode, with
    its adapter (if any) attached and its standard kernels installed.

    ``config.model_path`` is a local snapshot directory, loaded without
    network access, or a Hub model ID, which transformers resolves (and
    downloads when it is not cached). A hybrid vision-language snapshot
    (``HYBRID_SNAPSHOT_MODEL_TYPE``) must be a local directory; its Hub ID
    raises ``ValueError``.
    """

    require_native_bf16(device)
    if _snapshot_model_type(config.model_path) == HYBRID_SNAPSHOT_MODEL_TYPE:
        model = load_hybrid_causal_lm(
            config.model_path,
            device=device,
            attn_implementation=attn_implementation,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(config.model_path),
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            local_files_only=Path(config.model_path).is_dir(),
        ).to(device)
    model.config.use_cache = False
    # A hybrid's linear-attention layers otherwise run their causal
    # depthwise convolution through nn.Conv1d; the FLA kernel is the
    # architecture's standard one. Models without such layers are unchanged.
    if not config.disable_fla_causal_conv:
        install_fla_causal_conv(model)
    model.train()
    peft_model: torch.nn.Module | None = None
    if config.lora_rank > 0:
        peft_model = apply_lora(
            model,
            rank=config.lora_rank,
            # The flag exists with a None default; resolve the unset case
            # against the rank.
            alpha=int(config.lora_alpha or 2 * config.lora_rank),
        )[0]
        model = peft_model.base_model.model
        model.train()
        attach_publication_handle(model, peft_model)
        if config.fused_lora:
            fuse_merged_lora_forward(peft_model)
    else:
        model.requires_grad_(True)
    if config.compile_pointwise_modules:
        compile_pointwise_modules(model)
    return model
