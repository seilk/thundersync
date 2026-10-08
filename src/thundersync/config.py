"""Configuration shared by the trainers: the policy model a trainer rank
loads, and the error every configuration check raises."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["ConfigError", "PolicyModelConfig"]


class ConfigError(ValueError):
    """A configuration a trainer refuses to accept."""


@dataclass(frozen=True, slots=True)
class PolicyModelConfig:
    """The policy model a trainer rank loads and the optimizer it builds.

    Every field is a property of the trained model, not of an objective.
    No field has a default: each trainer's entry point supplies its own, so
    one objective's default never becomes another's.
    """

    # A local snapshot directory, loaded without network access, or a
    # Hugging Face Hub model ID (e.g. "org/name"), which transformers
    # resolves and downloads when it is not cached. A hybrid
    # vision-language snapshot must be a local directory
    # (thundersync.policy_model.load_hybrid_causal_lm).
    model_path: Path
    disable_fla_causal_conv: bool
    lora_rank: int
    lora_alpha: int | None
    fused_lora: bool
    compile_pointwise_modules: bool
    learning_rate: float
    optimizer_state_sharding: bool
    grad_clip: float
    # torch's deterministic algorithms for the whole trainer process, applied
    # through thundersync.determinism.apply_trainer_determinism before the
    # model loads (thundersync.grpo.runtime.load_trainer does this). When
    # true on CUDA, cuBLAS requires CUBLAS_WORKSPACE_CONFIG (":4096:8" or
    # ":16:8") in the environment before the first CUDA call.
    deterministic_algorithms: bool

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ConfigError("learning_rate must be positive")
        if self.lora_rank < 0:
            raise ConfigError("lora_rank must not be negative")
        if not isinstance(self.deterministic_algorithms, bool):
            raise ConfigError("deterministic_algorithms must be true or false")
