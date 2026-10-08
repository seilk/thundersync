"""Streaming backward execution for GRPO and on-policy distillation.

The entry points below are importable from the top level. Each is loaded
from its submodule on first access, so ``import thundersync`` imports
neither torch nor transformers.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

# Public name -> the submodule that defines it.
_EXPORTS = {
    "ConfigError": "thundersync.config",
    "DataParallelThunderSync": "thundersync.grpo.data_parallel",
    "GrpoTrainerConfig": "thundersync.grpo.config",
    "OPDTurnStream": "thundersync.opd.objectives",
    "PolicyModelConfig": "thundersync.config",
    "RewardLinearBackward": "thundersync.grpo.reward_linear",
    "STREAM_ATTENTION_NAME": "thundersync.engine.streaming",
    "StreamingRun": "thundersync.engine.streaming",
    "StreamingTrainerConfig": "thundersync.grpo.config",
    "apply_trainer_determinism": "thundersync.determinism",
    "load_trainer": "thundersync.grpo.runtime",
    "register_stream_attention": "thundersync.engine.streaming",
}

__all__ = [
    "ConfigError",
    "DataParallelThunderSync",
    "GrpoTrainerConfig",
    "OPDTurnStream",
    "PolicyModelConfig",
    "RewardLinearBackward",
    "STREAM_ATTENTION_NAME",
    "StreamingRun",
    "StreamingTrainerConfig",
    "__version__",
    "apply_trainer_determinism",
    "load_trainer",
    "register_stream_attention",
]

if TYPE_CHECKING:
    from thundersync.config import ConfigError, PolicyModelConfig
    from thundersync.determinism import apply_trainer_determinism
    from thundersync.engine.streaming import (
        STREAM_ATTENTION_NAME,
        StreamingRun,
        register_stream_attention,
    )
    from thundersync.grpo.config import GrpoTrainerConfig, StreamingTrainerConfig
    from thundersync.grpo.data_parallel import DataParallelThunderSync
    from thundersync.grpo.reward_linear import RewardLinearBackward
    from thundersync.grpo.runtime import load_trainer
    from thundersync.opd.objectives import OPDTurnStream


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
