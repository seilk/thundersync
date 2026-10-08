"""The GRPO trainer builder: the policy model, its maximizing optimizer, and
the data-parallel rank with its source and statistic residency."""

from __future__ import annotations

import json
from typing import Any

import torch

from thundersync.engine.streaming import STREAM_ATTENTION_NAME, StreamingRun, stream_attention_forward
from thundersync.grpo.config import GrpoTrainerConfig, StreamingTrainerConfig
from thundersync.grpo.data_parallel import DataParallelThunderSync
from thundersync.grpo.device_statistics import device_statistic_slots
from thundersync.grpo.pinned_arena import (
    PinnedSourceArena,
    PinnedStatisticPool,
    host_ranks_from_env,
)
from thundersync.config import ConfigError
from thundersync.determinism import apply_trainer_determinism
from thundersync.policy_model import load_policy_model

__all__ = ["load_trainer", "pinned_source_arena", "statistic_placement"]

_CUDNN_STREAM_ATTENTION_NAME = "grpo_cudnn_stream"


class _GraphSourceStreamingRun(StreamingRun):
    def _branch_layout(self, trajectory):
        tokens, scored, ranges = super()._branch_layout(trajectory)
        # Future observations have no causal path to any scored token. Keep
        # the immutable trajectory and every target; trim only its source
        # rebuild's physical suffix, including the last target's own token.
        end = max((index + 1 for index, flag in enumerate(scored) if flag), default=len(tokens))
        return tokens[:end], scored[:end], [(start, min(stop, end)) for start, stop in ranges if start < end]

    def set_source_gradient_window(self, window_bytes: int | None) -> None:
        """Bind the source executor's bounded in-backward parameter copies."""
        self._source_gradient_window_bytes = window_bytes

    def _trainable_gradient_bytes(self) -> int:
        full = super()._trainable_gradient_bytes()
        window = getattr(self, "_source_gradient_window_bytes", None)
        if window is None:
            return full
        # A previous source's last copy, the producing gradient and its
        # contiguous staging value may coexist beside bounded copy windows.
        # Captured endpoints are scalar views, rather than a whole source.
        return min(full, 3 * self._largest_trainable_gradient_bytes() + 2 * window)

    def _plain_rebuild_fits(self, branches: list[dict[str, Any]]) -> bool:
        if getattr(self, "_source_gradient_window_bytes", None) is None:
            return super()._plain_rebuild_fits(branches)
        from thundersync.engine.streaming import device_memory_state

        tokens = sum(len(branch["tokens"]) for branch in branches)
        if not tokens or self.rebuild_plain_reserve_bytes is None or self.weight.device.type != "cuda":
            return False
        head = max((self._head_backward_bytes(sum(branch["scored"]))
                    for branch in branches), default=0)
        head += self._largest_trainable_gradient_bytes() + self._source_gradient_window_bytes
        gradient = max(self._trainable_gradient_bytes(), head)
        needed = self._plain_rebuild_transient_bytes(tokens) + gradient
        return needed + self.rebuild_plain_reserve_bytes <= device_memory_state(self.weight.device)["available"]

    def _turn_memory_shape(self) -> dict[str, int]:
        if self._memory_shape is not None:
            return self._memory_shape
        shape = super()._turn_memory_shape()
        from thundersync.accel.rms_norm import StableQwen3RMSNorm
        from thundersync.engine.streaming import TURN_ACTIVATION_FACTOR

        layers = getattr(self.stack, "layers", ())
        norms = [getattr(self.stack, "norm", None)]
        for layer in layers:
            attention = getattr(layer, "self_attn", None)
            norms.extend((getattr(layer, "input_layernorm", None),
                          getattr(layer, "post_attention_layernorm", None),
                          getattr(attention, "q_norm", None),
                          getattr(attention, "k_norm", None)))
        if not layers or not all(isinstance(norm, StableQwen3RMSNorm) for norm in norms):
            return shape
        if self.weight.dtype not in (torch.bfloat16, torch.float16):
            return shape
        config = self.stack.config
        hidden, heads = int(config.hidden_size), int(config.num_attention_heads)
        kv_heads = int(config.num_key_value_heads or heads)
        head_dim = int(getattr(config, "head_dim", None) or hidden // heads)
        intermediate = int(config.intermediate_size)
        element, accumulate = self.weight.element_size(), 4
        query, kv = heads * head_dim, kv_heads * head_dim
        # The stable VJP saves the native input and one FP32 inverse root per
        # row, instead of the original FP32 input and normalized output.
        layer_bytes = (element * (2 * hidden + query + kv)
                       + accumulate * (2 + heads + kv_heads)
                       + element * (2 * query + 4 * intermediate)
                       + accumulate * heads)
        shape["plain_activation"] = int(len(layers) * layer_bytes * TURN_ACTIVATION_FACTOR)
        shape["checkpoint_activation"] = int(
            (len(layers) * element * hidden + layer_bytes) * TURN_ACTIVATION_FACTOR
        )
        shape["final_norm"] = element * hidden + accumulate
        return shape

    def _plain_rebuild_transient_bytes(self, tokens: int) -> int:
        """Count the saved norms and projections of the graph source path.

        The common turn estimate includes FP32 RMSNorm inputs and per-head
        query/key norms. The close-time residual/MLP estimate omits them.
        Include the branch's retained K/V and final output beside those
        activations; the inherited fit check separately counts gradients.
        """
        shape = self._turn_memory_shape()
        per_token = (shape["plain_activation"] + shape["kv"]
                     + 2 * shape["hidden"] + shape["final_norm"] + shape["rotary"])
        return tokens * per_token


def _cudnn_stream_attention_forward(module, query, key, value, *args: Any, **kwargs: Any) -> Any:
    try:
        return stream_attention_forward(module, query, key, value, *args,
                                        source_attention_backend=getattr(
                                            module, "_thundersync_source_attention", "cudnn"
                                        ), **kwargs)
    except Exception as error:
        from thundersync.engine.streaming import checkpoint_segments

        record = {
            "layer": module.layer_idx,
            "dtype": str(query.dtype),
            "qkv_shapes": [list(tensor.shape) for tensor in (query, key, value)],
            "qkv_strides": [list(tensor.stride()) for tensor in (query, key, value)],
            "segments": [{"start": start, "query_tokens": length,
                          "prefix_tokens": sum(chunk.shape[0] for chunk in keys)}
                         for start, length, keys, _values in checkpoint_segments(query.device)],
        }
        if query.is_cuda:
            free, total = torch.cuda.mem_get_info(query.device)
            memory = torch.cuda.memory_stats(query.device)
            record["device_memory"] = {
                "free": free, "total": total,
                **{name: memory[name] for name in ("allocated_bytes.all.current",
                                                  "reserved_bytes.all.current",
                                                  "active_bytes.all.current")},
            }
        error.add_note("GRPO source attention: " + json.dumps(record))
        raise


def load_trainer(
    config: GrpoTrainerConfig,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, DataParallelThunderSync]:
    """Build the model, its optimizer and the streaming trainer.

    Every value comes from the typed build config; the builder reads no
    parsed arguments and supplies no default of its own. Before the model
    loads, ``config.policy.deterministic_algorithms`` is applied to the
    whole process through ``apply_trainer_determinism``.
    """

    apply_trainer_determinism(config.policy.deterministic_algorithms)

    attention_name = STREAM_ATTENTION_NAME
    source_run = (_GraphSourceStreamingRun
                  if config.streaming.source_attention_backend == "cudnn_graph" else StreamingRun)
    if config.streaming.source_attention_backend in ("cudnn", "cudnn_graph"):
        from transformers.modeling_utils import AttentionInterface

        AttentionInterface.register(_CUDNN_STREAM_ATTENTION_NAME, _cudnn_stream_attention_forward)
        attention_name = _CUDNN_STREAM_ATTENTION_NAME
    model = load_policy_model(config.policy, device, attn_implementation=attention_name)
    if config.streaming.source_attention_backend == "cudnn":
        from thundersync.accel.capability import kernel_probe_records, kernel_runs
        from thundersync.engine.streaming import CUDNN_SDPA_MAX_HEAD_DIM, cudnn_split_trial

        head_dim = getattr(model.config, "head_dim", None) or (
            model.config.hidden_size // model.config.num_attention_heads
        )
        sample = next(model.parameters())
        if head_dim > CUDNN_SDPA_MAX_HEAD_DIM or not kernel_runs(
            "cudnn_split", sample, head_dim, cudnn_split_trial
        ):
            raise ConfigError(f"cuDNN source attention failed its device trial: {kernel_probe_records()}")
    elif config.streaming.source_attention_backend == "cudnn_graph":
        from thundersync.accel.capability import kernel_probe_records, kernel_runs
        from thundersync.engine.streaming import trial_qkv
        from thundersync.grpo.attention import source_graph_attention

        head_dim = getattr(model.config, "head_dim", None) or (
            model.config.hidden_size // model.config.num_attention_heads
        )
        options = dict(query_chunk=config.streaming.source_attention_query_chunk,
                       kv_bucket=config.streaming.source_attention_kv_bucket)

        def trial(trial_device, dtype, dimension):
            attention = source_graph_attention(**options)
            query, key, value = trial_qkv(trial_device, dtype, dimension, heads=2, kv_heads=1)
            output = attention(query, [key[0].transpose(0, 1)], [value[0].transpose(0, 1)],
                               dimension ** -0.5)
            torch.autograd.grad(output.float().sum(), (query, key, value))

        trial_name = f"cudnn_graph_q{options['query_chunk']}_kv{options['kv_bucket']}"
        if not kernel_runs(trial_name, next(model.parameters()), head_dim, trial):
            raise ConfigError(f"cuDNN graph source attention failed its device trial: {kernel_probe_records()}")
        attention = source_graph_attention(**options)
        for module in model.modules():
            if hasattr(module, "layer_idx"):
                module._thundersync_source_attention = attention
        from thundersync.accel.rms_norm import use_stable_qwen3_rms_norm
        from thundersync.grpo.rms_norm import use_compiled_source_rms_norm

        # _GraphSourceStreamingRun takes its reduced memory estimate only
        # when every norm is stable, and the compiled source norm below
        # replaces only stable norms.
        use_stable_qwen3_rms_norm(model)
        use_compiled_source_rms_norm(model)

    def _adamw(parameters: Any) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            parameters,
            lr=config.policy.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            maximize=True,
        )

    sharded_step = None
    if config.policy.optimizer_state_sharding:
        # ZeRO-1: each rank owns only its optimizer-state shard. The full
        # AdamW is never constructed; the
        # exposed optimizer is the rank-local shard optimizer, so state
        # hashes are per-rank evidence under sharding.
        import torch.distributed as dist

        from thundersync.engine.sharded_step import ShardedOptimizerStep

        if not dist.is_initialized():
            raise RuntimeError(
                "optimizer state sharding requires an initialized process group"
            )
        sharded_step = ShardedOptimizerStep(
            [
                (name, parameter)
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            ],
            _adamw,
            world_size=dist.get_world_size(),
            rank=dist.get_rank(),
            grad_clip=config.policy.grad_clip,
            missing_grad_policy="zero",
        )
        optimizer = sharded_step.optimizer
    else:
        optimizer = _adamw([p for p in model.parameters() if p.requires_grad])
    statistic_device, device_statistic_slots = statistic_placement(
        config.streaming,
        model,
        owned_parameters=[
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ],
        optimizer_state_tensors=_ADAMW_STATE_TENSORS,
    )
    trainer = DataParallelThunderSync.init_from_env(
        model,
        optimizer,
        source_pinned_arena=_pinned_source_arena(config.streaming, model),
        sharded_step=sharded_step,
        run_factory=lambda owned_model: source_run(
            owned_model,
            boundary_cut=True,
            checkpoint=True,
            stream_turns=False,
            rebuild_block_tokens=(
                config.streaming.source_rebuild_block_tokens or config.streaming.rebuild_block_tokens
            ),
            rebuild_plain_reserve_bytes=(
                None
                if config.streaming.rebuild_plain_reserve_gib <= 0
                else int(config.streaming.rebuild_plain_reserve_gib * (1 << 30))
            ),
            rebuild_checkpoint_storage_device=(
                config.streaming.rebuild_checkpoint_storage_device
            ),
            parameter_adjoint_storage_device=(
                config.streaming.gradient_accumulation_offload_device
            ),
        ),
        eps=config.grpo_epsilon,
        grad_clip=config.policy.grad_clip,
        source_offload_device=config.streaming.source_offload_device,
        source_offload_mode=config.streaming.source_offload_mode,
        source_offload_window_bytes=(
            config.streaming.source_offload_window_mib * 1024 * 1024
        ),
        source_offload_hbm_threshold=config.streaming.source_offload_hbm_threshold,
        source_reduction_mode=config.streaming.source_reduction_mode,
        source_reduction_pack_size=config.streaming.source_reduction_pack_size,
        source_reduction_worker_count=config.streaming.source_reduction_worker_count,
        source_offload_worker_count=config.streaming.source_offload_worker_count,
        source_offload_pinned_windows=config.streaming.source_offload_pinned_windows,
        source_spill_directory=config.streaming.source_spill_directory,
        # Host-resident statistics fold at host bandwidth; device-resident
        # statistics hold each open group's pair on the device and fold at
        # device bandwidth instead.
        statistic_device=statistic_device,
        device_statistic_slots=device_statistic_slots,
        statistic_keep_source_dtype=(config.streaming.statistic_dtype == "source"),
        retained_bytes_log_every=config.streaming.retained_bytes_log_every,
        direct_when_group_bound=config.streaming.direct_when_group_bound,
        source_release_during_backward=config.streaming.release_source_gradients,
        combine_known_reward_sources=config.streaming.combine_known_reward_sources,
    )
    return model, optimizer, trainer


# AdamW (amsgrad off) keeps exp_avg and exp_avg_sq per parameter, each in the
# parameter's dtype.
_ADAMW_STATE_TENSORS = 2


def statistic_placement(
    config: StreamingTrainerConfig,
    model: torch.nn.Module,
    *,
    owned_parameters: list[torch.nn.Parameter],
    optimizer_state_tensors: int,
) -> tuple[str | None, int | None]:
    """The statistic device and device statistic homes the executor gets.

    Host statistics, and device statistics of a streamed (pinned_streaming)
    fold, keep the configured device and no homes. With device-resident
    sources a CUDA statistic device gets one home per open group; "auto"
    gets as many as `device_statistic_slots` finds room for on this rank's
    device, read from the device and derived from the model, the optimizer
    shard this rank owns and the configured residency gate and reserve; the
    other groups fold on the host.
    """

    if config.statistic_device is None:
        return None, None
    if config.source_offload_mode != "device_resident":
        return config.statistic_device, None
    if config.statistic_open_groups < 1:
        raise ConfigError(
            "device statistic homes are sized by the rank's open groups, "
            "which the entry point did not set"
        )
    if config.statistic_device != "auto":
        return config.statistic_device, config.statistic_open_groups
    parameters = list(model.parameters())
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    device = trainable[0].device
    if device.type != "cuda":
        raise ConfigError("statistic_device auto needs a CUDA parameter device")

    def nbytes(values: list[torch.Tensor]) -> int:
        return sum(value.numel() * value.element_size() for value in values)

    source_bytes = nbytes(trainable)
    statistic_bytes = (
        source_bytes
        if config.statistic_dtype == "source"
        else sum(parameter.numel() * 4 for parameter in trainable)
    )
    slots = device_statistic_slots(
        device_total_bytes=int(torch.cuda.get_device_properties(device).total_memory),
        parameter_bytes=nbytes(parameters) + nbytes(list(model.buffers())),
        gradient_bytes=source_bytes,
        optimizer_state_bytes=optimizer_state_tensors * nbytes(owned_parameters),
        source_bytes=source_bytes,
        # The residency gates admit at most this many unfolded sources.
        device_held_sources=config.source_pinned_min_slabs,
        statistic_pair_bytes=2 * statistic_bytes,
        reserve_bytes=int(config.device_statistic_reserve_gib * (1 << 30)),
        open_groups=config.statistic_open_groups,
    )
    return "cuda", slots


def _pinned_source_arena(
    config: StreamingTrainerConfig, model: torch.nn.Module
) -> Any:
    """The rank's pinned source slabs, or None when sources are not pinned
    through an arena (another offload mode, or a host fraction of 0)."""

    if (
        config.source_offload_mode != "pinned_nonblocking"
        or config.source_pinned_host_fraction <= 0.0
    ):
        return None
    return pinned_source_arena(
        model,
        host_fraction=config.source_pinned_host_fraction,
        statistic_dtype=config.statistic_dtype,
        statistic_device=config.statistic_device,
        statistic_open_groups=config.statistic_open_groups,
        min_slabs=config.source_pinned_min_slabs,
        max_slabs=config.source_pinned_max_slabs,
    )


def pinned_source_arena(
    model: torch.nn.Module,
    *,
    host_fraction: float,
    statistic_dtype: str,
    statistic_device: Any,
    statistic_open_groups: int,
    min_slabs: int,
    max_slabs: int | None,
    host_ranks: int | None = None,
) -> Any:
    """A rank's pinned source slabs, sized from this host's memory.

    The rank's share is MemAvailable over the trainer ranks on this host
    (``host_ranks``, else those torchrun placed here), less the host
    statistic pairs of its open groups. Those pairs get reused homes (one
    slot per open group) carried by the arena. ``statistic_dtype`` is
    ``"source"`` or ``"fp32"``, as ``StreamingTrainerConfig.statistic_dtype``
    names it.
    """

    trainable = [p for p in model.parameters() if p.requires_grad]
    source_bytes = sum(p.numel() * p.element_size() for p in trainable)
    statistic_bytes = (
        source_bytes
        if statistic_dtype == "source"
        else sum(p.numel() * 4 for p in trainable)
    )
    reserved = (
        0
        if statistic_device is not None
        else 2 * statistic_open_groups * statistic_bytes
    )
    statistic_dtypes = {
        p.dtype if statistic_dtype == "source" else torch.float32
        for p in trainable
    }
    statistics = (
        None
        if statistic_device is not None
        or statistic_open_groups < 1
        or len(statistic_dtypes) != 1
        else PinnedStatisticPool(
            [p.numel() for p in trainable],
            next(iter(statistic_dtypes)),
            slots=statistic_open_groups,
        )
    )
    return PinnedSourceArena.for_host(
        source_bytes,
        host_ranks=host_ranks or host_ranks_from_env(),
        host_fraction=host_fraction,
        reserved_bytes=reserved,
        min_slabs=min_slabs,
        max_slabs=max_slabs,
        statistics=statistics,
    )
