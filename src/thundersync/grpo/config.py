"""Typed GRPO trainer configuration.

A ``GrpoTrainerConfig`` that constructs names every value the GRPO trainer
builder reads: the policy model, the advantage-stabilization epsilon and
the streaming executor's residency. ``from_namespace`` reads the three
values the builder requires by name and every other value with the
builder's default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from thundersync.config import ConfigError, PolicyModelConfig

__all__ = [
    "DEFAULT_GRPO_EPSILON",
    "DEFAULT_PINNED_HOST_FRACTION",
    "DEFAULT_SOURCE_OFFLOAD_HBM_THRESHOLD",
    "GrpoTrainerConfig",
    "StreamingTrainerConfig",
    "SOURCE_OFFLOAD_MODES",
    "SOURCE_REDUCTION_MODES",
    "STATISTIC_DTYPES",
]

# The advantage-stabilization epsilon in the group-normalized advantage
# A_i = (r_i - mean(r)) / (std(r) + eps); RewardLinearBackward and
# DataParallelThunderSync default to it too.
DEFAULT_GRPO_EPSILON = 1e-6
# Share of the host's MemAvailable that a host's trainer ranks may split
# between their pinned source slabs and their host group statistics; the
# rest is left to the rollout engines and everything else on the host. A
# starting point, not a measured optimum: lower it when other processes on
# the host need more memory.
DEFAULT_PINNED_HOST_FRACTION = 0.8
# Device-memory fraction above which blocking sources are offloaded to the
# host. A starting point; tune it to the device's memory and the workload.
DEFAULT_SOURCE_OFFLOAD_HBM_THRESHOLD = 0.65
# The source-transfer implementations and the fold mode the streaming executor
# accepts (thundersync.grpo.reward_linear), and the parameter group-statistic
# dtypes the trainer builder accepts.
SOURCE_OFFLOAD_MODES = (
    "blocking",
    "pinned_nonblocking",
    "pinned_streaming",
    "device_resident",
)
SOURCE_REDUCTION_MODES = ("unrestricted_statistics",)
STATISTIC_DTYPES = ("fp32", "source")


def _optional_path(value: Any) -> Path | None:
    return None if value is None else Path(value)


@dataclass(frozen=True, slots=True)
class StreamingTrainerConfig:
    """The streaming executor: its close rebuild, where sources and
    group statistics live, and the direct path."""

    rebuild_block_tokens: int
    rebuild_plain_reserve_gib: float
    gradient_accumulation_offload_device: str | None
    # Where the close-time rebuild parks each block's checkpoint inputs
    # between the branch's forward and its backward; None keeps them on
    # the device. A prepared source waiting for its reward holds exactly
    # those inputs, so its per-token cost grows with the model's hidden
    # size and depth.
    rebuild_checkpoint_storage_device: str | None
    # Every Nth source event the trainer logs its retained-bytes sample as
    # one JSON line at INFO; 0 = never.
    retained_bytes_log_every: int
    source_offload_device: str | None
    source_offload_mode: str
    source_offload_window_mib: int
    source_offload_hbm_threshold: float
    source_offload_pinned_windows: int
    source_reduction_mode: str
    source_reduction_pack_size: int
    source_reduction_worker_count: int
    source_offload_worker_count: int
    source_spill_directory: Path | None
    # Where the open groups' parameter statistics live: None on the host
    # (or, with device_resident sources, where each group's first source
    # folds); a CUDA device; or "auto", device_resident only, which keeps
    # on the device as many open groups' pairs as fit beside the step's
    # resident state and device_statistic_reserve_gib, and folds the
    # rest on the host (thundersync.grpo.device_statistics).
    statistic_device: str | None
    statistic_dtype: str
    # pinned_nonblocking sources wait in slabs of a per-rank arena sized
    # from the host (0 keeps the per-copy caching-allocator path). The slab
    # count is bounded by the residency gates, which must fit in it.
    source_pinned_host_fraction: float = DEFAULT_PINNED_HOST_FRACTION
    source_pinned_min_slabs: int = 1
    source_pinned_max_slabs: int | None = None
    # Groups a rank holds open at once: each keeps a statistic pair on the
    # host, which the pinned budget leaves room for.
    statistic_open_groups: int = 0
    # statistic_device "auto": device memory kept beside the statistics for
    # the step's transients (close rebuild activations, the backward's own
    # gradient, allocator slack, the CUDA context).
    device_statistic_reserve_gib: float = 0.0
    # A branch backward that starts after its group's last verdict runs
    # A_i * S_i straight into the parameters' gradients (no source, no
    # statistics). Off is the statistics path for every trajectory.
    direct_when_group_bound: bool = False
    # Source rebuilds may use a different checkpoint block size from the
    # shared scoring head and final forest. Zero inherits the shared size.
    source_rebuild_block_tokens: int = 0
    source_attention_backend: str = "auto"
    source_attention_query_chunk: int = 0
    source_attention_kv_bucket: int = 0
    release_source_gradients: bool = False
    combine_known_reward_sources: bool = False

    def __post_init__(self) -> None:
        if self.rebuild_block_tokens < 1:
            raise ConfigError("rebuild_block_tokens must be positive")
        if self.source_offload_mode not in SOURCE_OFFLOAD_MODES:
            raise ConfigError(
                "source_offload_mode must be one of " + ", ".join(SOURCE_OFFLOAD_MODES)
            )
        if self.source_reduction_mode not in SOURCE_REDUCTION_MODES:
            raise ConfigError(
                "source_reduction_mode must be one of " + ", ".join(SOURCE_REDUCTION_MODES)
            )
        if self.retained_bytes_log_every < 0:
            raise ConfigError("retained_bytes_log_every must not be negative")
        if self.statistic_dtype not in STATISTIC_DTYPES:
            raise ConfigError(
                "statistic_dtype must be one of " + ", ".join(STATISTIC_DTYPES)
            )
        if not 0.0 <= self.source_pinned_host_fraction <= 1.0:
            raise ConfigError("source_pinned_host_fraction must be in [0, 1]")
        if self.source_pinned_min_slabs < 1:
            raise ConfigError("source_pinned_min_slabs must be positive")
        if (
            self.source_pinned_max_slabs is not None
            and self.source_pinned_max_slabs < self.source_pinned_min_slabs
        ):
            raise ConfigError(
                "source_pinned_max_slabs must not be below source_pinned_min_slabs"
            )
        if self.statistic_open_groups < 0:
            raise ConfigError("statistic_open_groups must not be negative")
        if self.statistic_device == "auto" and self.source_offload_mode != "device_resident":
            raise ConfigError(
                "statistic_device auto places device-resident sources' "
                "statistics; it requires source_offload_mode device_resident"
            )
        if self.device_statistic_reserve_gib < 0.0:
            raise ConfigError("device_statistic_reserve_gib must not be negative")
        if not isinstance(self.direct_when_group_bound, bool):
            raise ConfigError("direct_when_group_bound must be true or false")
        if self.source_rebuild_block_tokens < 0:
            raise ConfigError("source_rebuild_block_tokens must not be negative")
        if self.source_attention_backend not in ("auto", "cudnn", "cudnn_graph"):
            raise ConfigError("source_attention_backend must be auto, cudnn or cudnn_graph")
        for name in ("source_attention_query_chunk", "source_attention_kv_bucket"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ConfigError(f"{name} must be a non-negative integer")
        if self.source_attention_kv_bucket and not self.source_attention_query_chunk:
            raise ConfigError("source_attention_kv_bucket requires source_attention_query_chunk")
        if (
            self.source_attention_query_chunk or self.source_attention_kv_bucket
        ) and self.source_attention_backend != "cudnn_graph":
            raise ConfigError("source attention chunk and bucket require cudnn_graph")
        for name in ("release_source_gradients", "combine_known_reward_sources"):
            if not isinstance(getattr(self, name), bool):
                raise ConfigError(f"{name} must be true or false")
        if self.release_source_gradients and self.source_offload_mode != "pinned_nonblocking":
            raise ConfigError("release_source_gradients requires pinned_nonblocking sources")


@dataclass(frozen=True, slots=True)
class GrpoTrainerConfig:
    """Everything the GRPO trainer builder reads about one run."""

    policy: PolicyModelConfig
    streaming: StreamingTrainerConfig
    # The advantage-stabilization epsilon: A_i = (r_i - mean) / (std + eps).
    grpo_epsilon: float = DEFAULT_GRPO_EPSILON

    @classmethod
    def from_namespace(cls, args: Any) -> GrpoTrainerConfig:
        """Translate a GRPO entry point's parsed arguments.

        The model snapshot, the learning rate and the rebuild block size
        are required, and a missing one raises ``ConfigError`` naming it.
        ``grpo_epsilon`` defaults to ``DEFAULT_GRPO_EPSILON``. Every other
        value is read with the
        builder's default, which is GRPO's: a trainer deterministic unless
        the entry point turns it off, every trajectory on the statistics
        path unless it turns the direct path on.

        The flags validated as bools (``deterministic_algorithms``,
        ``direct_when_group_bound``, ``release_source_gradients``,
        ``combine_known_reward_sources``) pass through uncoerced, so a value
        such as the string ``"off"`` is refused rather than read as true.
        """

        def required(name: str) -> Any:
            if not hasattr(args, name):
                raise ConfigError(
                    f"trainer build configuration requires {name}, which "
                    "the entry point does not define"
                )
            return getattr(args, name)

        def probe(name: str, default: Any) -> Any:
            return getattr(args, name, default)

        def device(name: str) -> str | None:
            # The argparse spelling of "no device" is the string; the
            # builder takes the absence.
            value = probe(name, None)
            return None if value == "none" else value

        held_sources = int(probe("max_held_sources", 0) or 0)
        prepare_pack = int(probe("prepare_pack_max", 0) or 0)
        policy = PolicyModelConfig(
            model_path=Path(required("model_path")),
            disable_fla_causal_conv=bool(probe("disable_fla_causal_conv", False)),
            lora_rank=int(probe("lora_rank", 0) or 0),
            lora_alpha=probe("lora_alpha", None),
            fused_lora=bool(probe("fused_lora", False)),
            compile_pointwise_modules=bool(probe("compile_pointwise_modules", False)),
            learning_rate=float(required("learning_rate")),
            optimizer_state_sharding=bool(probe("optimizer_state_sharding", False)),
            grad_clip=float(probe("grad_clip", 1.0)),
            deterministic_algorithms=probe("deterministic_algorithms", True),
        )
        streaming = StreamingTrainerConfig(
            rebuild_block_tokens=int(required("rebuild_block_tokens")),
            rebuild_plain_reserve_gib=float(probe("rebuild_plain_reserve_gib", 0.0)),
            rebuild_checkpoint_storage_device=device("rebuild_checkpoint_storage_device"),
            gradient_accumulation_offload_device=device(
                "gradient_accumulation_offload_device"
            ),
            # Unlike the two device selections above, the source offload
            # target keeps its spelling: RewardLinearBackward treats None and
            # "none" alike as no offload and refuses any destination but the
            # CPU.
            source_offload_device=probe("source_offload_device", None),
            source_offload_mode=str(probe("source_offload_mode", "blocking")),
            source_offload_window_mib=int(probe("source_offload_window_mib", 256)),
            source_offload_hbm_threshold=float(
                probe("source_offload_hbm_threshold", DEFAULT_SOURCE_OFFLOAD_HBM_THRESHOLD)
            ),
            source_offload_pinned_windows=int(probe("source_offload_pinned_windows", 2)),
            source_reduction_mode=str(
                probe("source_reduction_mode", "unrestricted_statistics")
            ),
            source_reduction_pack_size=int(probe("source_reduction_pack_size", 1)),
            source_reduction_worker_count=int(probe("source_reduction_worker_count", 2)),
            source_offload_worker_count=int(probe("source_offload_worker_count", 2)),
            source_spill_directory=_optional_path(probe("source_spill_directory", None)),
            statistic_device=device("statistic_device"),
            statistic_dtype=str(probe("statistic_dtype", "fp32")),
            retained_bytes_log_every=int(probe("retained_bytes_log_every", 0)),
            source_pinned_host_fraction=float(
                probe("pinned_host_fraction", DEFAULT_PINNED_HOST_FRACTION)
            ),
            # The residency gates admit at most max(held, pack) unfolded
            # sources, so that many slabs keep a gated admission from ever
            # waiting on the arena; without a held gate, memory bounds it.
            source_pinned_min_slabs=max(1, held_sources, prepare_pack),
            source_pinned_max_slabs=(
                max(held_sources, prepare_pack) if held_sources > 0 else None
            ),
            # The groups one rank holds open; when unset, the per-worker
            # admission limit stands in for it.
            statistic_open_groups=int(
                probe("statistic_open_groups", 0) or probe("max_concurrent_groups", 0) or 0
            ),
            device_statistic_reserve_gib=float(
                probe("device_statistic_reserve_gib", 0.0) or 0.0
            ),
            direct_when_group_bound=probe("direct_when_group_bound", False),
            source_rebuild_block_tokens=int(probe("source_rebuild_block_tokens", 0)),
            source_attention_backend=str(probe("source_attention_backend", "auto")),
            source_attention_query_chunk=int(probe("source_attention_query_chunk", 0)),
            source_attention_kv_bucket=int(probe("source_attention_kv_bucket", 0)),
            release_source_gradients=probe("release_source_gradients", False),
            combine_known_reward_sources=probe("combine_known_reward_sources", False),
        )
        return cls(
            policy=policy,
            streaming=streaming,
            grpo_epsilon=float(probe("grpo_epsilon", DEFAULT_GRPO_EPSILON)),
        )
