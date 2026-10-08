"""The reward-linear streamed backward.

For the advantage-weighted policy-gradient component of objectives affine in
rewards given the group statistics, the gradient decomposes exactly:

    sum_i A_i * grad(S_i)  where  A_i = (r_i - mean) / (std + eps)
                        =  (G1 - mean * G2) / (std + eps)

    G2 = sum_i grad(S_i)          valid when trajectory content is immutable
    G1 = sum_i r_i * grad(S_i)    valid when the content-bound reward lands

so the group barrier gates a **parameter-vector AXPY with scalar coefficients**,
not a backward. Retained equality tests validate the decomposition.

Additive KL, entropy, reference-model, and other auxiliary gradient sources are
outside this G1/G2 identity. They need a separate executor and proven release
event, or an explicitly detached reward-shaping contract. The executor
differentiates each legally ready source immediately, in actual readiness
order, and folds completed transfers into G1/G2 statistics in that same
readiness order. It never retains a source merely to wait for an earlier
trajectory index. This removes both canonical head-of-line retention and
synchronous spill I/O from the objective-ready critical path.

The boundary cut (streaming._Boundary) is what makes the cost right. A
trajectory's `grad(S_i)` is computed only through its OWN branch subgraph,
stopping at the shared prompt's detached proxies; the prompt subgraph is
backwarded once per group with the combined boundary gradient. Without the cut,
G backwards each re-traverse the prompt: ~2x logical tokens of backward work
instead of ~1x physical, silently undoing the forest's CSE.

Cost accounting, honest: one branch backward per trajectory (streamed, hidden
behind other trajectories' generation) and one prompt backward per group.
The reference retains one native-dtype source per closed trajectory;
objective-ready statistics retain two fp32 parameter statistics plus only
sources whose transfer or readiness-order fold is in flight. The transient close peak,
allocator fragmentation, statistics traffic, and open-group count remain
empirical memory gates. Total
backward FLOPs ~= the barrier backward's. Since
the traj-bwd granularity lever, the
branch backward executes as a branch re-forward + backward
(streaming.rebuild_logprobs) rather than a walk of the per-event graph: same
FLOPs and the same gradient function (reward-affine objectives never read the
logprob values) at block-checkpointed kernel granularity, with a block-sized
close transient instead of the plain rebuild's branch-sized one.

Closes that queue while the trainer is busy coalesce: `close_trajectories`
packs the queued branches' rebuilds into ONE multi-segment primal per block
round (the same _Seg machinery as coalesced turn arrivals), then runs one
backward PER branch -- the branch graphs are disjoint by construction, and a
single backward over any weighted sum cannot produce G1 and G2 both, so
per-branch traversal is the minimum (the rejected reward-window batching
lever: 2k walks for k branches, strictly worse; kept binding).
The opt-in known-reward cohort path is an exception: already-ready members
of one group with the same landed reward r need only H = grad(sum_i S_i),
because their contributions are G2 += H and G1 += r H. It waits for no
additional member or verdict and reports the physical aggregate separately
from its logical members.

Safety rails, all loud:
* an optimizer step while any group is open corrupts the retained graphs; the
  version guard detects it at the next close_* and raises with the parameter
  name, instead of letting autograd fail later or -- worse -- succeed wrongly;
* `close_group` before all its trajectories closed raises;
* a second `close_trajectory` for the same trajectory raises.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import time
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

from thundersync.accel.operator_profiling import operator_range
from thundersync.grpo.close_profiler import ClosePackProfiler
from thundersync.grpo.config import DEFAULT_GRPO_EPSILON, DEFAULT_SOURCE_OFFLOAD_HBM_THRESHOLD
from thundersync.grpo.device_statistics import DeviceStatisticPool
from thundersync.grpo.final_forest import backward_final_forest, forest_trajectory
from thundersync.grpo.pinned_arena import PinnedSlab, PinnedSourceArena, tensor_offsets
from thundersync.engine.streaming import StreamingRun

logger = logging.getLogger(__name__)

_MAX_SOURCE_REDUCTION_WORKERS = 64
# Seconds shutdown waits for each offload or reducer worker to stop.
_WORKER_JOIN_TIMEOUT_S = 30.0
# THUNDERSYNC_GRPO_CLOSE_TRACE=1 records every executed close pack of a step
# (members, rung, device spans of the rebuild, of each branch's autograd walk
# and of its source store) in the source report as ``close_trace``. Timing
# only; off by default, and nothing extra is recorded when off.
_CLOSE_TRACE_ENV = "THUNDERSYNC_GRPO_CLOSE_TRACE"
_CLOSE_TRACE_MAX_PACKS = 512


def _close_trace_enabled() -> bool:
    return os.environ.get(_CLOSE_TRACE_ENV, "").strip() not in ("", "0")


def _trajectory_order(trajectory_ids: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    """A group's trajectory ids in reduction order; refuses empty or repeated ids."""
    order = tuple(sorted(trajectory_ids))
    if not order or len(order) != len(set(order)):
        raise ValueError("group trajectory order must be nonempty and unique")
    return order


def _worker_failure(error: BaseException) -> RuntimeError:
    """Detach worker diagnostics from frames that may own full-model tensors."""
    rendered = "".join(traceback.format_exception(error)).strip()
    return RuntimeError(rendered)


@dataclass
class _HostFoldTask:
    run: Callable[[], None]


@dataclass
class _HostCombineTask:
    group_id: int


@dataclass
class _PendingCpuTransfer:
    """A CUDA-to-pinned-CPU copy whose completion is owned by the reducer.

    Keeping the detached CUDA source alive until the event completes is
    required for correctness: the admission thread must be free to continue
    objective-ready work, while the reducer may consume the CPU value only
    after the nonblocking copy has finished.
    """

    cpu_tensor: torch.Tensor
    source_tensor: torch.Tensor | None
    event: Any | None
    bytes: int

    def materialize(self) -> torch.Tensor:
        if self.event is not None:
            self.event.synchronize()
        # Drop the CUDA reference as soon as the event is complete. The pinned
        # destination is the only value needed by the CPU reducer.
        self.source_tensor = None
        return self.cpu_tensor


_SourceValue = torch.Tensor | _PendingCpuTransfer | None


@dataclass
class _StreamingTensorOffload:
    """One gradient tensor handed off while its source VJP is still running."""

    group_id: int
    trajectory_id: int
    destination: list[_SourceValue]
    destination_index: int
    source_tensor: torch.Tensor
    source_ready_event: Any
    reward_sum: list[torch.Tensor | None]
    unit_sum: list[torch.Tensor | None]
    reward: float
    keep_source_dtype: bool


_MIN_SOURCE_OFFLOAD_WINDOW_BYTES = 8
# Each streaming worker owns this many pinned windows by default.  Two is the
# minimum that lets one chunk's D2H overlap the previous chunk's host fold,
# which keeps exactly one transfer outstanding per worker.  A contended host
# path needs more outstanding transfers to keep the copy engine fed, so the
# depth is settable per run through `source_offload_pinned_windows`.
_STREAMING_PINNED_WINDOWS = 2
_MAX_STREAMING_PINNED_WINDOWS = 16
_MAX_SOURCE_OFFLOAD_WINDOW_BYTES = 1024 * 1024 * 1024


@dataclass
class _GroupAccum:
    # Objective-ready execution owns the G2 accumulator and materializes G1 on
    # the first nonzero reward. Sources remain here only while their transfer
    # or readiness-order fold is in flight; trajectory index order is never a
    # reduction predicate.
    parameter_sources: dict[int, list[_SourceValue]] = field(
        default_factory=dict
    )
    boundary_sources: dict[int, list[_SourceValue]] = field(
        default_factory=dict
    )
    rewards_by_trajectory: dict[int, float] = field(default_factory=dict)
    token_weights_by_trajectory: dict[int, torch.Tensor | None] = field(
        default_factory=dict
    )
    # Declared binary host statistics use this parameter pair for P and N;
    # the boundary pair always retains the original weighted/unit sums.
    parameter_centered_sum: list[torch.Tensor | None] = field(default_factory=list)
    parameter_source_mean: list[torch.Tensor | None] = field(default_factory=list)
    boundary_centered_sum: list[torch.Tensor | None] = field(default_factory=list)
    boundary_source_mean: list[torch.Tensor | None] = field(default_factory=list)
    statistic_count: int = 0
    closed: set[int] = field(default_factory=set)
    differentiated: set[int] = field(default_factory=set)
    reduced: set[int] = field(default_factory=set)
    # Early-training split: `prepared` holds trajectories whose content-ready
    # branch backward ran; `materialized` holds trajectories whose offloaded
    # source is host-resident. A source becomes reduction-eligible only when
    # it is BOTH materialized and reward-bound (`closed`).
    prepared: set[int] = field(default_factory=set)
    materialized: set[int] = field(default_factory=set)
    # Every landed reward of the group, whether or not its trajectory has
    # been differentiated yet.
    verdicts: dict[int, float] = field(default_factory=dict)
    # Trajectories differentiated after the group's last verdict: their
    # A_i S_i went straight into the parameter gradients and the prompt
    # boundary proxies' gradients, with the moments in ``direct_moments``.
    direct: set[int] = field(default_factory=set)
    direct_moments: tuple[float, float] | None = None
    # The subset of ``direct`` the final forest differentiated
    # (`close_final_forest`): their prompt term came from the forest's own
    # prompt forward, so they leave nothing in the boundary proxies.
    forest: set[int] = field(default_factory=set)
    # The reducer that folds a group's last source also applies
    # G1 -= mean * G2 to the host statistics before the close, which then
    # finds them combined. ``finalizing`` holds the close's barrier until
    # that combine has finished.
    finalizing: bool = False
    combined_mean: float | None = None
    combined_indices: set[int] = field(default_factory=set)
    # The last fold of a group with host statistics folds and combines
    # parameter by parameter and publishes each one here, so a close already
    # waiting reloads it while later parameters still fold.
    pipelined: bool = False
    ready_parameters: set[int] = field(default_factory=set)
    # The reused host home of this group's parameter statistics, when the
    # arena's statistic pool had a free slot at the group's first fold.
    statistic_slot: Any = None
    statistic_slot_tried: bool = False

    @property
    def rewards(self) -> list[float]:
        """Compatibility view with a schedule-independent logical order."""

        return [
            self.rewards_by_trajectory[trajectory_id]
            for trajectory_id in sorted(self.rewards_by_trajectory)
        ]


class RewardLinearBackward:
    """Streams per-trajectory backwards; the group barrier becomes an axpy.

    Loss-agnostic boundary: this class sees per-token logprobs, a per-trajectory
    scalar reward, and optional reward-independent token weights (e.g. 1/|resp|
    for length-normalized GRPO variants). It never sees the environment, and the
    advantage formula is the one documented here -- population std with eps,
    A_i = (r_i - mean) / (std + eps) -- and equivalence tests compare against
    exactly this formula.
    """

    def __init__(
        self,
        run: StreamingRun,
        model: torch.nn.Module,
        *,
        eps: float = DEFAULT_GRPO_EPSILON,
        source_offload_device: str | torch.device | None = "cpu",
        source_offload_mode: str = "blocking",
        source_reduction_mode: str = "unrestricted_statistics",
        source_reduction_pack_size: int = 1,
        source_offload_pinned_windows: int = _STREAMING_PINNED_WINDOWS,
        statistic_device: str | torch.device | None = None,
        device_statistic_slots: int | None = None,
        statistic_keep_source_dtype: bool = False,
        source_reduction_worker_count: int = 2,
        source_offload_worker_count: int = 2,
        source_offload_window_bytes: int = 256 * 1024 * 1024,
        source_offload_hbm_threshold: float = DEFAULT_SOURCE_OFFLOAD_HBM_THRESHOLD,
        source_spill_directory: str | Path | None = None,
        source_pinned_arena: PinnedSourceArena | None = None,
        defer_group_close: bool = False,
        source_copy_during_backward: bool = False,
        source_release_during_backward: bool = False,
        combine_known_reward_sources: bool = False,
        direct_when_group_bound: bool = False,
    ) -> None:
        if not run.boundary_cut:
            raise ValueError(
                "RewardLinearBackward requires StreamingRun(boundary_cut=True); without the "
                "cut each trajectory backward re-traverses the shared prompt"
            )
        self.run = run
        self.eps = eps
        if source_offload_device in {None, "none"}:
            self.source_offload_device: torch.device | None = None
        else:
            device = torch.device(source_offload_device)
            if device.type != "cpu":
                raise ValueError("reward-linear source offload currently supports CPU only")
            self.source_offload_device = device
        if source_offload_mode not in {
            "blocking",
            "pinned_nonblocking",
            "pinned_streaming",
            "device_resident",
        }:
            raise ValueError(
                "source_offload_mode must be blocking, pinned_nonblocking, "
                "pinned_streaming, or device_resident"
            )
        if (
            source_offload_mode
            in {"pinned_nonblocking", "pinned_streaming", "device_resident"}
            and self.source_offload_device != torch.device("cpu")
        ):
            raise ValueError(
                "pinned source offload (and the device_resident spill "
                "target) requires CPU destination"
            )
        self.source_offload_mode = source_offload_mode
        if source_reduction_mode != "unrestricted_statistics":
            raise ValueError(
                "source_reduction_mode must be unrestricted_statistics"
            )
        self.source_reduction_mode = source_reduction_mode
        if isinstance(source_reduction_pack_size, bool) or not isinstance(
            source_reduction_pack_size, int
        ) or (
            source_reduction_pack_size < 1
        ):
            raise ValueError("source_reduction_pack_size must be positive")
        if source_reduction_pack_size != 1:
            raise ValueError(
                "unrestricted_statistics requires one-source admission"
            )
        if self.source_offload_device != torch.device("cpu"):
            raise ValueError(
                "unrestricted_statistics requires CPU source offload"
            )
        if source_spill_directory is not None:
            raise ValueError(
                "synchronous source spill is not supported on the training path"
            )
        self.source_spill_directory: Path | None = None
        self.source_reduction_pack_size = source_reduction_pack_size
        if (
            isinstance(source_offload_pinned_windows, bool)
            or not isinstance(source_offload_pinned_windows, int)
            or not 2 <= source_offload_pinned_windows
            <= _MAX_STREAMING_PINNED_WINDOWS
        ):
            raise ValueError(
                "source_offload_pinned_windows must be in "
                f"[2, {_MAX_STREAMING_PINNED_WINDOWS}]"
            )
        self._streaming_pinned_windows = source_offload_pinned_windows
        if statistic_device is None:
            self._statistic_device: torch.device | None = None
        else:
            resolved_statistic_device = torch.device(statistic_device)
            if resolved_statistic_device.type != "cuda":
                raise ValueError(
                    "statistic_device supports CUDA devices; host-resident "
                    "statistics are the default"
                )
            self._statistic_device = resolved_statistic_device
        # None keeps the statistics where each group's first folded source
        # lives. A count places that many open groups' parameter statistic
        # pairs in device homes this executor allocates at its first group
        # registration; a group that finds every home owned folds on the host.
        if device_statistic_slots is not None:
            if (
                isinstance(device_statistic_slots, bool)
                or not isinstance(device_statistic_slots, int)
                or device_statistic_slots < 0
            ):
                raise ValueError(
                    "device_statistic_slots must be a non-negative integer"
                )
            if self._statistic_device is None:
                raise ValueError(
                    "device_statistic_slots places statistic homes on "
                    "statistic_device, which is unset"
                )
            if source_offload_mode != "device_resident":
                raise ValueError(
                    "device statistic homes fold device-resident sources; "
                    "they require source_offload_mode device_resident"
                )
        self._device_statistic_slots = device_statistic_slots
        if (
            isinstance(source_reduction_worker_count, bool)
            or not isinstance(source_reduction_worker_count, int)
            or not 1
            <= source_reduction_worker_count
            <= _MAX_SOURCE_REDUCTION_WORKERS
        ):
            raise ValueError(
                "source_reduction_worker_count must be an integer in [1, 64]"
            )
        self._source_reduction_worker_count = source_reduction_worker_count
        if (
            isinstance(source_offload_worker_count, bool)
            or not isinstance(source_offload_worker_count, int)
            or not 1
            <= source_offload_worker_count
            <= _MAX_SOURCE_REDUCTION_WORKERS
        ):
            raise ValueError(
                "source_offload_worker_count must be an integer in [1, 64]"
            )
        self._source_offload_worker_count = source_offload_worker_count
        if (
            isinstance(source_offload_window_bytes, bool)
            or not isinstance(source_offload_window_bytes, int)
            or source_offload_window_bytes < _MIN_SOURCE_OFFLOAD_WINDOW_BYTES
            or source_offload_window_bytes > _MAX_SOURCE_OFFLOAD_WINDOW_BYTES
            or source_offload_window_bytes % _MIN_SOURCE_OFFLOAD_WINDOW_BYTES != 0
        ):
            raise ValueError(
                "source_offload_window_bytes must be an 8-byte-aligned integer "
                "between 8 bytes and 1 GiB"
            )
        self._source_offload_window_bytes = source_offload_window_bytes
        # Tile the existing host additions without allocating another buffer.
        self._host_statistic_tile_bytes = max(
            256, (source_offload_window_bytes // 64) // 256 * 256
        )
        self._host_statistic_tiled_parameters = 0
        self._host_statistic_tiled_elements = 0
        self._host_statistic_tile_count = 0
        if (
            isinstance(source_offload_hbm_threshold, bool)
            or not isinstance(source_offload_hbm_threshold, (int, float))
            or not 0.0 < source_offload_hbm_threshold < 1.0
        ):
            raise ValueError("source_offload_hbm_threshold must be in (0, 1)")
        self._source_offload_hbm_threshold = float(source_offload_hbm_threshold)
        if (
            source_pinned_arena is not None
            and source_offload_mode != "pinned_nonblocking"
        ):
            raise ValueError(
                "source_pinned_arena holds pinned_nonblocking sources only"
            )
        # Each unfolded source's parameter and boundary gradients live in
        # one arena slab from admission to fold; the arena outlives this
        # executor so slabs are registered once per rank, not per step.
        self._pinned_arena = source_pinned_arena
        self._source_slabs: dict[tuple[int, int], PinnedSlab] = {}
        self._source_pinned_acquire_s = 0.0
        # Off by default: on, the update's bits vary run to run (the
        # embedding's gradient, by O(1)).
        # With an arena, each gradient's copy into its slab starts inside the
        # branch backward, from a hook on the leaf, as soon as autograd has
        # produced it (`_differentiate_into_slab`), when the arena has a slab
        # free as the backward starts; False, or a busy arena, copies the
        # whole source after the backward returns. The hooks capture the
        # gradients before the optional endpoint sink releases them. They tell the outer
        # walk from a checkpoint's nested one by the autograd graph task id;
        # a torch without that query copies after the backward.
        self._source_copy_during_backward = bool(
            source_copy_during_backward
        ) and callable(getattr(torch._C, "_current_graph_task_id", None))
        if source_release_during_backward and (
            not self._source_copy_during_backward or self._pinned_arena is None
        ):
            raise ValueError("source_release_during_backward requires the in-backward slab copy")
        self._source_release_during_backward = bool(source_release_during_backward)
        if combine_known_reward_sources and self.source_offload_mode == "pinned_streaming":
            raise ValueError("known-reward cohorts require prepared full sources")
        self._combine_known_reward_sources = bool(combine_known_reward_sources)
        self._known_reward_cohorts: list[dict[str, Any]] = []
        self._source_gradients_released_in_backward = 0
        self._max_returned_parameter_gradient_storage_bytes = 0
        self._max_source_copy_pending_bytes = 0
        self._source_copy_stream_joins = 0
        self._backward_copy_streams: dict[torch.device, Any] = {}
        self._source_copies_in_backward = 0
        self._source_copies_after_backward = 0
        # Backwards that found every slab busy and copied after the backward.
        self._source_slab_busy_before_backward = 0
        self.params = [p for p in model.parameters() if p.requires_grad]
        self._device_statistic_pool: DeviceStatisticPool | None = None
        self._device_statistic_spec: tuple[list[int], torch.dtype, torch.device] | None = None
        self._device_statistic_pool_lock = threading.Lock()
        if device_statistic_slots:
            statistic_dtypes = {
                (p.dtype if statistic_keep_source_dtype else torch.float32)
                for p in self.params
            }
            parameter_devices = {p.device for p in self.params}
            if len(statistic_dtypes) != 1 or len(parameter_devices) != 1:
                raise ValueError(
                    "device statistic homes need one statistic dtype and one "
                    "parameter device"
                )
            (parameter_device,) = parameter_devices
            if self._statistic_device is None:
                raise RuntimeError("device statistic homes need a statistic device")
            if parameter_device.type != "cuda" or (
                self._statistic_device.index is not None
                and self._statistic_device != parameter_device
            ):
                raise ValueError(
                    "device statistic homes live on the parameters' device, "
                    "where the group close reads them"
                )
            # Allocated at the step's first group registration
            # (_device_statistic_homes), before any of its rebuilds: an
            # executor that never opens a group never holds the homes.
            self._device_statistic_spec = (
                [p.numel() for p in self.params],
                next(iter(statistic_dtypes)),
                parameter_device,
            )
        # Groups whose statistics fold on the host because every device home
        # was owned when their first source arrived.
        self._host_routed_group_count = 0
        self._device_source_drain_s = 0.0
        # Device bytes of each device-resident source until it folds or
        # spills, keyed by (group, trajectory).
        self._device_source_bytes: dict[tuple[int, int], int] = {}
        self._statistic_pool = (
            None
            if source_pinned_arena is None
            else source_pinned_arena.statistics
        )
        if self._statistic_pool is not None and source_pinned_arena is not None:
            pool = self._statistic_pool
            statistic_dtypes = {
                (p.dtype if statistic_keep_source_dtype else torch.float32)
                for p in self.params
            }
            if (
                statistic_device is not None
                or pool.element_counts != [p.numel() for p in self.params]
                or statistic_dtypes != {pool.dtype}
            ):
                raise ValueError(
                    "the arena's statistic pool must be laid out for this "
                    "model's trainable parameters in the statistic dtype, "
                    "with host statistics"
                )
        # Version guard: an optimizer step edits parameters in place, which
        # invalidates every retained graph. Record versions now; check at every
        # close. Failing loudly here is a design guarantee.
        self._versions = [p._version for p in self.params]
        self._groups: dict[int, _GroupAccum] = {}
        # Parameter group statistics accumulate in the source dtype when set
        # (each group's Sigma-g / Sigma-rg pair is 2x full-model, so fp32
        # pairs for several open groups of a large model can exceed host
        # memory); the
        # boundary statistics already fold in source dtype. Same-dtype AXPY
        # is also the fast host fold.
        self._statistic_keep_source_dtype = bool(statistic_keep_source_dtype)
        self._binary_rewards = False
        self._binary_host_statistics = False
        self._binary_statistic_parameters = 0
        self._binary_statistic_adds = 0
        self._current_source_bytes = 0
        self._peak_source_bytes = 0
        self._stored_source_bytes = 0
        self._reloaded_source_bytes = 0
        self._source_offload_s = 0.0
        self._source_offload_submit_s = 0.0
        self._source_offload_wait_s = 0.0
        self._source_offload_inflight_bytes = 0
        self._peak_source_offload_inflight_bytes = 0
        self._source_contiguous_copy_bytes = 0
        self._max_single_source_contiguous_copy_bytes = 0
        self._offload_streams: dict[
            tuple[torch.device, int], torch.cuda.Stream
        ] = {}
        self._streaming_pinned_buffers: dict[int, list[torch.Tensor]] = {}
        self._streaming_fold_buffers: dict[int, torch.Tensor] = {}
        self._streaming_fold_buffer_reserved_bytes = 0
        self._streaming_fold_conversion_bytes = 0
        self._group_close_host_combined_bytes = 0
        self._group_close_host_combine_s = 0.0
        self._group_close_adjoint_s = 0.0
        self._group_close_release_s = 0.0
        self._group_close_barrier_s = 0.0
        self._group_close_boundary_s = 0.0
        self._group_close_free_s = 0.0
        self._group_close_parameter_s = 0.0
        self._group_finalized_in_reducer_count = 0
        self._sealed_statistics_combine_count = 0
        self._group_last_fold_s = 0.0
        self._group_close_parameter_wait_s = 0.0
        self._pooled_reload_bytes = 0
        self._statistic_slots: dict[int, Any] = {}
        self._source_copy_drain_s = 0.0
        self._source_copy_drain_released = 0
        # Group closes recorded while the group's last source still folded
        # (``defer_group_close``), run in this order at later safe points.
        self._defer_group_close = bool(defer_group_close)
        self._requested_group_closes: deque[int] = deque()
        self._group_close_deferred_count = 0
        # A close whose group has every reward bound runs the direct path
        # (`_backward_direct`): no source, no statistics. Streaming offload
        # folds the reward inside its own backward and keeps the one path.
        if direct_when_group_bound and source_offload_mode == "pinned_streaming":
            raise ValueError(
                "the direct path needs prepare/bind sources; pinned_streaming "
                "folds the reward inside the branch backward"
            )
        self._direct_when_group_bound = bool(direct_when_group_bound)
        self._model = model
        self._final_forest_calls = 0
        self._final_forest_trajectory_count = 0
        self._final_forest_logical_tokens = 0
        self._final_forest_microbatches = 0
        self._final_forest_out_of_memory_retries = 0
        self._final_forest_s = 0.0
        self._final_forest_budgets: list[int] = []
        self._final_forest_peak_allocated_bytes = 0
        # Cache accounting includes this rank's local and claimed forests.
        self._forest_attention_cache_saved = 0
        self._forest_attention_cache_replayed = 0
        self._forest_attention_cache_declined = 0
        self._forest_attention_cache_budget_bytes = 0
        self._forest_attention_cache_charged_bytes = 0
        # Final work handed to or taken from another rank (work claims).
        self._ceded_final_count = 0
        self._claimed_final_calls = 0
        self._claimed_final_count = 0
        self._claimed_final_logical_tokens = 0
        self._claimed_final_microbatches = 0
        self._claimed_final_out_of_memory_retries = 0
        self._claimed_final_s = 0.0
        self._direct_backward_count = 0
        self._direct_backward_s = 0.0
        self._direct_group_count = 0
        self._uniform_reward_group_count = 0
        # Branches whose backward ran out of device memory and were rebuilt
        # on the named rung of the close's memory ladder.
        self._backward_out_of_memory_retries: dict[str, int] = {}
        # Branches whose close-time rebuild itself ran out of device memory
        # and were rebuilt, after the drain, on the named rung.
        self._rebuild_out_of_memory_retries: dict[str, int] = {}
        # Close packs whose first rebuild attempt ran out of device memory,
        # whether or not a lower rung then succeeded.
        self._rebuild_out_of_memory_first_attempts = 0
        self._worker_cuda_devices: dict[str, int] = {}
        self._group_close_boundary_statistics_s = 0.0
        self._streaming_pinned_buffer_reserved_bytes = 0
        self._hbm_staged_source_count = 0
        self._hbm_staged_source_bytes = 0
        self._hbm_stage_s = 0.0
        self._device_resident_spilled_source_count = 0
        self._device_resident_spilled_source_bytes = 0
        # Trainer-side and source-service phase attribution.  ``hbm_stage_s``
        # alone cannot separate transfer completion from host arithmetic, and
        # the aggregate forward/backward span cannot separate the branch
        # re-forward from the branch backward; these timers split both.
        self._branch_rebuild_s = 0.0
        self._branch_rebuild_count = 0
        self._executed_rebuild_pack_max = 0
        self._branch_backward_s = 0.0
        self._branch_backward_count = 0
        self._statistics_branch_observer: Callable[[int], None] | None = None
        self._streaming_chunk_copy_wait_s = 0.0
        self._streaming_chunk_fold_s = 0.0
        self._streaming_chunk_count = 0
        # CUDA-event spans over the same phases as the host timers.
        # The host timers wrap calls that synchronize, so on a busy
        # queue they absorb device work issued earlier; these do not.
        self._branch_rebuild_events: list[tuple[Any, Any]] = []
        self._branch_backward_events: list[tuple[Any, Any]] = []
        self._group_close_boundary_events: list[tuple[Any, Any]] = []
        # Per-pack close records (THUNDERSYNC_GRPO_CLOSE_TRACE); None when off.
        self._close_trace: list[dict[str, Any]] | None = (
            [] if _close_trace_enabled() else None
        )
        # A profiler window around selected close packs
        # (THUNDERSYNC_GRPO_PROFILE_DIR); None when off.
        self._close_profiler = ClosePackProfiler.from_env()
        self._diagnostic_rank = 0
        self._diagnostic_step = 1
        self._max_observed_allocator_reserved_fraction = 0.0
        self._current_pageable_stage_bytes = 0
        self._peak_pageable_stage_bytes = 0
        self._offload_queue: deque[
            tuple[int, int, list[_SourceValue], list[_SourceValue], Any | None]
            | _StreamingTensorOffload
        ] = deque()
        self._streaming_pending_tensors: dict[tuple[int, int], int] = {}
        self._streaming_capture_complete: set[tuple[int, int]] = set()
        self._streaming_staged_bytes_by_trajectory: dict[
            tuple[int, int], int
        ] = {}
        self._reduction_ready_queue: deque[
            tuple[int, int, Any | None]
        ] = deque()
        self._offload_pending_trajectories_peak = 0
        self._source_reload_s = 0.0
        self._source_spill_write_s = 0.0
        self._source_spill_read_s = 0.0
        self._current_spill_bytes = 0
        self._peak_spill_bytes = 0
        self._stored_spill_bytes = 0
        self._reloaded_spill_bytes = 0
        self._current_statistic_bytes = 0
        self._peak_statistic_bytes = 0
        self._deferred_ready_trajectories_peak = 0
        self._pending_reduction_trajectories_peak = 0
        self._reducer_queue_s = 0.0
        self._reducer_queue_max_s = 0.0
        self._reducer_fold_s = 0.0
        self._reducer_executing_source_bytes = 0
        self._reducer_executing_source_bytes_peak = 0
        self._source_queued_at: dict[tuple[int, int], float] = {}
        self._source_execution_order_by_group: dict[int, list[int]] = {}
        self._completed_source_headers = 0
        self._source_reduction_order_by_group: dict[int, list[int]] = {}
        self._source_preparation_order_by_group: dict[int, list[int]] = {}
        self._prepared_awaiting_reward_peak = 0
        self._expected_trajectory_order_by_group: dict[int, tuple[int, ...]] = {}
        # Groups whose close has completed (their accumulators are gone).
        self._closed_group_ids: set[int] = set()
        self._accounting_lock = threading.Lock()
        self._reducer_condition = threading.Condition()
        self._offload_failure: BaseException | None = None
        self._offload_stop = False
        self._active_offload_jobs = 0
        self._active_offload_jobs_peak = 0
        self._active_offload_jobs_by_group: dict[int, int] = {}
        self._active_offload_jobs_peak_by_group: dict[int, int] = {}
        self._active_offload_tensor_jobs_by_trajectory: dict[
            tuple[int, int], int
        ] = {}
        self._active_hbm_stage_jobs = 0
        self._active_hbm_stage_jobs_peak = 0
        self._active_hbm_stage_jobs_by_group: dict[int, int] = {}
        self._active_hbm_stage_jobs_peak_by_group: dict[int, int] = {}
        self._active_hbm_stage_tensor_jobs_by_trajectory: dict[
            tuple[int, int], int
        ] = {}
        self._active_hbm_stage_tensor_jobs = 0
        self._active_hbm_stage_tensor_jobs_peak = 0
        self._active_hbm_stage_tensor_jobs_by_group: dict[int, int] = {}
        self._active_hbm_stage_tensor_jobs_peak_by_group: dict[int, int] = {}
        offload_worker_count = (
            self._source_offload_worker_count
            if self.source_offload_mode == "pinned_streaming"
            else 1
        )
        self._offload_threads = [
            threading.Thread(
                target=self._offload_loop,
                name=f"thundersync-source-offload-{index}",
                daemon=True,
            )
            for index in range(offload_worker_count)
        ]
        # Reduction order is meaningful only within a reward group.  Multiple
        # workers let independent groups release source payloads concurrently;
        # `_groups_reducing` preserves source order within each group; its
        # owner can share independent host parameters with these same workers.
        self._groups_reducing: set[int] = set()
        self._reducer_failure: BaseException | None = None
        self._reducer_stop = False
        self._host_fold_tasks: deque[_HostFoldTask] = deque()
        self._host_parameter_dispatches = 0
        self._host_parameter_tasks = 0
        self._reducer_threads = [
            threading.Thread(
                target=self._reducer_loop,
                name=f"thundersync-readiness-reducer-{index}",
                daemon=True,
            )
            for index in range(self._source_reduction_worker_count)
        ]
        for offload_thread in self._offload_threads:
            offload_thread.start()
        for reducer_thread in self._reducer_threads:
            reducer_thread.start()

    def _store_sources(
        self,
        gradients: list[torch.Tensor | None],
    ) -> list[_SourceValue]:
        stored = []
        for gradient in gradients:
            if gradient is None:
                stored.append(None)
                continue
            size_bytes = gradient.numel() * gradient.element_size()
            value = gradient.detach()
            if self.source_offload_device is not None and value.device != (
                self.source_offload_device
            ):
                if self.source_offload_mode == "device_resident":
                    # The source stays on its producing device and folds
                    # there.  ``source_offload_device`` names only the pinned
                    # spill destination the offload worker uses above the
                    # sampled allocator watermark; admission never copies.
                    pass
                elif self.source_offload_mode == "pinned_streaming":
                    if value.device.type != "cuda":
                        raise RuntimeError(
                            "pinned_streaming requires CUDA gradient sources"
                        )
                    # Admission retains the already-produced CUDA source and
                    # returns immediately. A group-local reducer allocates one
                    # pinned tensor window at a time, copies it, folds it, and
                    # releases the window before advancing to the next tensor.
                elif (
                    self.source_offload_mode == "pinned_nonblocking"
                    and value.device.type == "cuda"
                ):
                    value = self._submit_pinned_cpu_copy(value, size_bytes)
                else:
                    started = time.perf_counter()
                    value = value.to(self.source_offload_device)
                    with self._accounting_lock:
                        self._source_offload_s += time.perf_counter() - started
            stored.append(value)
            with self._accounting_lock:
                self._current_source_bytes += size_bytes
                self._stored_source_bytes += size_bytes
                self._peak_source_bytes = max(
                    self._peak_source_bytes,
                    self._current_source_bytes,
                )
        return stored

    def _register_streaming_source_tensor(
        self,
        *,
        group_id: int,
        trajectory_id: int,
        destination: list[_SourceValue],
        destination_index: int,
        gradient: torch.Tensor,
        reward_sum: list[torch.Tensor | None],
        unit_sum: list[torch.Tensor | None],
        reward: float,
        keep_source_dtype: bool,
    ) -> None:
        """Hand one completed leaf gradient to D2H workers during autograd."""

        value = gradient.detach()
        if value.device.type != "cuda":
            raise RuntimeError(
                "pinned_streaming requires CUDA gradient sources"
            )
        size_bytes = value.numel() * value.element_size()
        ready_event = torch.cuda.Event()
        ready_event.record(torch.cuda.current_stream(value.device))
        key = (group_id, trajectory_id)
        with self._reducer_condition:
            if destination[destination_index] is not None:
                raise RuntimeError(
                    "streaming source hook produced the same tensor twice"
                )
            destination[destination_index] = value
            self._streaming_pending_tensors[key] = (
                self._streaming_pending_tensors.get(key, 0) + 1
            )
            # Account before publishing the job.  A fast background fold may
            # otherwise release the tensor before the producer records its
            # residency, creating an underflow and an understated peak.
            with self._accounting_lock:
                self._current_source_bytes += size_bytes
                self._stored_source_bytes += size_bytes
                self._peak_source_bytes = max(
                    self._peak_source_bytes,
                    self._current_source_bytes,
                )
            self._offload_queue.append(
                _StreamingTensorOffload(
                    group_id=group_id,
                    trajectory_id=trajectory_id,
                    destination=destination,
                    destination_index=destination_index,
                    source_tensor=value,
                    source_ready_event=ready_event,
                    reward_sum=reward_sum,
                    unit_sum=unit_sum,
                    reward=reward,
                    keep_source_dtype=keep_source_dtype,
                )
            )
            self._offload_pending_trajectories_peak = max(
                self._offload_pending_trajectories_peak,
                len(self._streaming_pending_tensors),
            )
            self._reducer_condition.notify_all()

    def _differentiate_streaming_source(
        self,
        *,
        score: torch.Tensor,
        group_id: int,
        trajectory_id: int,
        boundary_proxy: list[torch.Tensor],
    ) -> tuple[list[_SourceValue], list[_SourceValue]]:
        """Stream leaf gradients out while autograd is producing the source.

        A post-accumulate hook clears each leaf's ``.grad`` immediately after
        handing the immutable tensor to a D2H worker.  This bounds HBM by the
        gradient tensors whose copies are genuinely in flight instead of
        retaining a full-model source until ``torch.autograd.grad`` returns.
        """

        inputs = [*self.params, *boundary_proxy]
        if any(not value.is_leaf for value in inputs):
            raise RuntimeError(
                "pinned_streaming source inputs must be autograd leaves"
            )
        # Completed groups may already own optimizer-gradient buffers.  Save
        # those references, isolate this source in empty leaf slots, then
        # restore them after the source VJP.  The streamed source must never be
        # accumulated into or erase the eventual optimizer update.
        prior_gradients = [value.grad for value in inputs]
        for value in inputs:
            value.grad = None
        parameter_sources: list[_SourceValue] = [None] * len(self.params)
        boundary_sources: list[_SourceValue] = [None] * len(boundary_proxy)
        acc = self._groups[group_id]
        with self._reducer_condition:
            if not acc.parameter_centered_sum:
                acc.parameter_centered_sum.extend([None] * len(self.params))
                acc.parameter_source_mean.extend([None] * len(self.params))
            if not acc.boundary_centered_sum:
                acc.boundary_centered_sum.extend([None] * len(boundary_proxy))
                acc.boundary_source_mean.extend([None] * len(boundary_proxy))
            if (
                len(acc.parameter_centered_sum) != len(self.params)
                or len(acc.parameter_source_mean) != len(self.params)
                or len(acc.boundary_centered_sum) != len(boundary_proxy)
                or len(acc.boundary_source_mean) != len(boundary_proxy)
            ):
                raise RuntimeError("streaming statistic layout changed")
        handles = []

        def install(
            leaf: torch.Tensor,
            destination: list[_SourceValue],
            destination_index: int,
            reward_sum: list[torch.Tensor | None],
            unit_sum: list[torch.Tensor | None],
            keep_source_dtype: bool,
        ) -> None:
            def capture(completed_leaf: torch.Tensor) -> None:
                gradient = completed_leaf.grad
                if gradient is None:
                    return
                completed_leaf.grad = None
                self._register_streaming_source_tensor(
                    group_id=group_id,
                    trajectory_id=trajectory_id,
                    destination=destination,
                    destination_index=destination_index,
                    gradient=gradient,
                    reward_sum=reward_sum,
                    unit_sum=unit_sum,
                    reward=acc.rewards_by_trajectory[trajectory_id],
                    keep_source_dtype=keep_source_dtype,
                )

            handles.append(leaf.register_post_accumulate_grad_hook(capture))

        for index, parameter in enumerate(self.params):
            install(
                parameter,
                parameter_sources,
                index,
                acc.parameter_centered_sum,
                acc.parameter_source_mean,
                False,
            )
        for index, proxy in enumerate(boundary_proxy):
            install(
                proxy,
                boundary_sources,
                index,
                acc.boundary_centered_sum,
                acc.boundary_source_mean,
                True,
            )
        try:
            torch.autograd.backward(
                score,
                inputs=inputs,
                retain_graph=False,
            )
        finally:
            for handle in handles:
                handle.remove()
            for value, prior_gradient in zip(inputs, prior_gradients, strict=True):
                value.grad = prior_gradient
        return parameter_sources, boundary_sources

    def _submit_pinned_cpu_copy(
        self,
        value: torch.Tensor,
        size_bytes: int,
        *,
        source_ready_event: Any | None = None,
    ) -> _PendingCpuTransfer:
        """Submit a CUDA→pinned-CPU copy without blocking admission."""

        if self.source_offload_device != torch.device("cpu"):
            raise RuntimeError("pinned source copies require a CPU destination")
        started = time.perf_counter()
        destination = torch.empty_like(
            value,
            device=self.source_offload_device,
            pin_memory=True,
        )
        stream_key = (value.device, threading.get_ident())
        copy_stream = self._offload_streams.get(stream_key)
        if copy_stream is None:
            copy_stream = torch.cuda.Stream(device=value.device)
            self._offload_streams[stream_key] = copy_stream
        if source_ready_event is None:
            current_stream = torch.cuda.current_stream(value.device)
            copy_stream.wait_stream(current_stream)
        else:
            copy_stream.wait_event(source_ready_event)
        with torch.cuda.stream(copy_stream):
            value.record_stream(copy_stream)
            destination.copy_(value, non_blocking=True)
            event = torch.cuda.Event()
            event.record(copy_stream)
        submit_s = time.perf_counter() - started
        with self._accounting_lock:
            self._source_offload_submit_s += submit_s
            self._source_offload_inflight_bytes += size_bytes
            self._peak_source_offload_inflight_bytes = max(
                self._peak_source_offload_inflight_bytes,
                self._source_offload_inflight_bytes,
            )
        return _PendingCpuTransfer(
            cpu_tensor=destination,
            source_tensor=value,
            event=event,
            bytes=size_bytes,
        )

    def _store_sources_in_slab(
        self,
        key: tuple[int, int],
        parameter_gradients: list[torch.Tensor | None],
        boundary_gradients: list[torch.Tensor | None],
    ) -> tuple[list[_SourceValue], list[_SourceValue]]:
        """Copy one trajectory's gradients into a single arena slab.

        The bytes that land are the gradients' own, so the fold reads the
        same values the per-copy pinned path gave it; only the host memory
        they wait in differs.
        """

        if self._pinned_arena is None:
            raise RuntimeError("pinned-slab source copies need a pinned source arena")
        gradients = [*parameter_gradients, *boundary_gradients]
        values: list[torch.Tensor | None] = []
        for gradient in gradients:
            if gradient is None:
                values.append(None)
                continue
            if gradient.device.type != "cuda":
                raise RuntimeError("pinned slab sources must be CUDA gradients")
            value = gradient.detach()
            values.append(value if value.is_contiguous() else value.contiguous())
        present = [value for value in values if value is not None]
        stored: list[_SourceValue] = [None] * len(values)
        if not present:
            return stored[: len(parameter_gradients)], stored[len(parameter_gradients) :]
        sizes = [value.numel() * value.element_size() for value in present]
        offsets, total = tensor_offsets(sizes)
        acquire_started = time.perf_counter()
        slab = self._pinned_arena.acquire(
            max(1, total),
            owner=key,
            release_possible=self._slab_release_possible,
        )
        with self._reducer_condition:
            self._source_slabs[key] = slab
        with self._accounting_lock:
            self._source_pinned_acquire_s += time.perf_counter() - acquire_started
        started = time.perf_counter()
        device = present[0].device
        stream_key = (device, threading.get_ident())
        copy_stream = self._offload_streams.get(stream_key)
        if copy_stream is None:
            copy_stream = torch.cuda.Stream(device=device)
            self._offload_streams[stream_key] = copy_stream
        copy_stream.wait_stream(torch.cuda.current_stream(device))
        destinations = []
        with torch.cuda.stream(copy_stream):
            for value, offset in zip(present, offsets, strict=True):
                value.record_stream(copy_stream)
                destinations.append(
                    self._pinned_arena.copy_into(slab, offset, value)
                )
            event = torch.cuda.Event()
            event.record(copy_stream)
        slab.copy_event = event
        cursor = 0
        for index, value in enumerate(values):
            if value is None:
                continue
            stored[index] = _PendingCpuTransfer(
                cpu_tensor=destinations[cursor],
                source_tensor=value,
                event=event,
                bytes=sizes[cursor],
            )
            cursor += 1
        submit_s = time.perf_counter() - started
        stored_bytes = sum(sizes)
        with self._accounting_lock:
            self._source_offload_submit_s += submit_s
            self._source_offload_inflight_bytes += stored_bytes
            self._peak_source_offload_inflight_bytes = max(
                self._peak_source_offload_inflight_bytes,
                self._source_offload_inflight_bytes,
            )
            self._current_source_bytes += stored_bytes
            self._stored_source_bytes += stored_bytes
            self._peak_source_bytes = max(
                self._peak_source_bytes,
                self._current_source_bytes,
            )
        return stored[: len(parameter_gradients)], stored[len(parameter_gradients) :]

    def _backward_copy_stream(self, device: torch.device) -> Any:
        stream = self._backward_copy_streams.get(device)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._backward_copy_streams[device] = stream
        return stream

    def _take_free_slab(
        self,
        key: tuple[int, int],
        boundary_proxy: list[torch.Tensor],
    ) -> tuple[PinnedSlab, list[int], list[int]] | None:
        """A slab for this branch's source if the arena has one free now.

        With its per-input byte sizes and offsets, in input order. A busy
        arena frees a slab only when an earlier source folds; waiting for
        that before the backward would stall the backward behind the fold,
        so the caller then copies after the backward, as without the hooks.
        """

        if self._pinned_arena is None:
            raise RuntimeError("pinned-slab source copies need a pinned source arena")
        inputs = [*self.params, *boundary_proxy]
        sizes = [value.numel() * value.element_size() for value in inputs]
        offsets, total = tensor_offsets(sizes)
        started = time.perf_counter()
        slab = self._pinned_arena.try_acquire(max(1, total), owner=key)
        with self._accounting_lock:
            self._source_pinned_acquire_s += time.perf_counter() - started
        if slab is None:
            with self._accounting_lock:
                self._source_slab_busy_before_backward += 1
            return None
        with self._reducer_condition:
            self._source_slabs[key] = slab
        return slab, sizes, offsets

    def _differentiate_into_slab(
        self,
        score: torch.Tensor,
        key: tuple[int, int],
        boundary_proxy: list[torch.Tensor],
        taken: tuple[PinnedSlab, list[int], list[int]],
        *,
        raise_out_of_memory: bool,
    ) -> tuple[list[_SourceValue], list[_SourceValue]] | None:
        """One branch's source, each gradient copied out as autograd makes it.

        The source is ``torch.autograd.grad`` over the parameters and the
        boundary proxies, as in the copy-after path. A hook on each input
        leaf sees the gradient autograd is about to return (the same tensor;
        a leaf used twice, like a tied embedding, fires once with the summed
        gradient) and enqueues its copy into the source's slab on a copy
        stream, behind the gradient's own producing work. The copies so run
        during the rest of the backward instead of after it, and the next
        pack's drain waits only for the last ones. The bytes that land are the
        gradients' own, in the same slab offsets for every source. A gradient
        no hook copied (or one whose tensor differs from the hooked one) is
        copied after the backward. ``taken`` is the slab `_take_free_slab`
        found free. Returns None, with the slab released and nothing kept,
        when the backward ran out of device memory and
        ``raise_out_of_memory`` is false.

        With source_release_during_backward, captured parameter endpoints
        return scalar zero views to grad(): the source values live in the
        slab. A bounded queue owns their device storage until the producing
        stream joins the copy event. Boundary proxies keep the ordinary
        record_stream capture path.
        """

        if self._pinned_arena is None:
            raise RuntimeError("pinned-slab source copies need a pinned source arena")
        inputs = [*self.params, *boundary_proxy]
        slab, sizes, offsets = taken
        device = score.device
        copy_stream = self._backward_copy_stream(device)
        hooked: list[torch.Tensor | None] = [None] * len(inputs)
        destinations: list[torch.Tensor | None] = [None] * len(inputs)
        released = [False] * len(inputs)
        endpoint_zeros: dict[torch.dtype, torch.Tensor] = {}
        pending_copies: deque[tuple[Any, int, torch.Tensor, Any]] = deque()
        pending_copy_bytes = [0]
        stream_joins = 0
        submit = [0.0]
        # A checkpointed block's recompute differentiates its own subgraph
        # with a nested `torch.autograd.grad`, in which a parameter's hook
        # fires with that block's partial gradient. Only the outer walk's
        # call carries what is returned: its graph task is the one in which
        # the score's own hook runs, first.
        outer_task: list[int | None] = [None]

        def mark_outer(_gradient: torch.Tensor) -> None:
            outer_task[0] = torch._C._current_graph_task_id()

        def retire_copy() -> None:
            nonlocal stream_joins
            copied_event, copied_bytes, _value, producing_stream = pending_copies[0]
            if not copied_event.query():
                producing_stream.wait_event(copied_event)
                stream_joins += 1
            # The queue keeps the gradient alive until the producing stream
            # has joined its copy, so that stream can safely reuse its storage.
            pending_copies.popleft()
            pending_copy_bytes[0] -= copied_bytes

        def install(index: int, leaf: torch.Tensor) -> Any:
            def hook(gradient: torch.Tensor) -> torch.Tensor | None:
                if (
                    outer_task[0] is None
                    or torch._C._current_graph_task_id() != outer_task[0]
                ):
                    return None
                started = time.perf_counter()
                value = gradient.detach()
                if (
                    value.device == device
                    and value.dtype == leaf.dtype
                    and value.shape == leaf.shape
                    and (value.is_contiguous() or (
                        self._source_release_during_backward and index < len(self.params)
                    ))
                ):
                    if not value.is_contiguous():
                        value = value.contiguous()
                    if self._source_release_during_backward and index < len(self.params):
                        # Bound the owned copy sources. One tensor larger
                        # than the window may run alone. Retiring a source
                        # joins its copy on the producing stream.
                        while pending_copies and pending_copies[0][0].query():
                            retire_copy()
                        while pending_copies and (
                            pending_copy_bytes[0] + sizes[index] > self._source_offload_window_bytes
                        ):
                            retire_copy()
                    ready = torch.cuda.Event()
                    ready.record(torch.cuda.current_stream(device))
                    copy_stream.wait_event(ready)
                    with torch.cuda.stream(copy_stream):
                        if not self._source_release_during_backward or index >= len(self.params):
                            value.record_stream(copy_stream)
                        destinations[index] = self._pinned_arena.copy_into(
                            slab, offsets[index], value
                        )
                    if self._source_release_during_backward and index < len(self.params):
                        copied_event = torch.cuda.Event()
                        copied_event.record(copy_stream)
                        pending_copies.append((
                            copied_event, sizes[index], value,
                            torch.cuda.current_stream(device),
                        ))
                        pending_copy_bytes[0] += sizes[index]
                        self._max_source_copy_pending_bytes = max(
                            self._max_source_copy_pending_bytes, pending_copy_bytes[0]
                        )
                        # The leaf's complete source is already in flight to
                        # its slab. Capture a scalar view at the autograd
                        # endpoint instead of retaining another full source
                        # until grad() returns. The queue owns the real
                        # source until its producing stream joins the copy.
                        zero = endpoint_zeros.get(value.dtype)
                        if zero is None:
                            zero = torch.zeros((), dtype=value.dtype, device=device)
                            endpoint_zeros[value.dtype] = zero
                        released[index] = True
                        submit[0] += time.perf_counter() - started
                        return zero.expand_as(value)
                    hooked[index] = value
                submit[0] += time.perf_counter() - started
                return None

            return leaf.register_hook(hook)

        handles = [score.register_hook(mark_outer)]
        handles.extend(install(index, leaf) for index, leaf in enumerate(inputs))
        gradients = None
        try:
            gradients = torch.autograd.grad(
                score,
                inputs,
                retain_graph=False,
                allow_unused=True,
            )
        except torch.OutOfMemoryError:
            if raise_out_of_memory:
                self._abandon_slab_copies(key, copy_stream, hooked)
                raise
        except BaseException:
            self._abandon_slab_copies(key, copy_stream, hooked)
            raise
        finally:
            for handle in handles:
                handle.remove()
            while pending_copies:
                retire_copy()
            with self._accounting_lock:
                self._source_copy_stream_joins += stream_joins
        if gradients is None:
            self._abandon_slab_copies(key, copy_stream, hooked)
            return None
        returned_storage_bytes = sum({
            gradient.untyped_storage().data_ptr(): gradient.untyped_storage().nbytes()
            for gradient in gradients[:len(self.params)] if gradient is not None
        }.values())
        with self._accounting_lock:
            self._max_returned_parameter_gradient_storage_bytes = max(
                self._max_returned_parameter_gradient_storage_bytes, returned_storage_bytes
            )
        started = time.perf_counter()
        # The gradients no hook copied, made contiguous here on the
        # backward's own stream: a `contiguous()` enqueued on the copy stream
        # would read the original gradient there, behind the copies already
        # queued, after this frame drops the original's last reference --
        # and the allocator hands a block freed without a recorded copy-stream
        # use back to the backward's stream at once.
        late_values: list[tuple[int, torch.Tensor]] = []
        for index, gradient in enumerate(gradients):
            if gradient is None:
                hooked[index] = None
                destinations[index] = None
                continue
            if released[index]:
                continue
            copied = hooked[index]
            if copied is not None and (
                copied.data_ptr() == gradient.data_ptr()
                and copied.shape == gradient.shape
                and copied.dtype == gradient.dtype
            ):
                continue
            value = gradient.detach()
            if value.numel() * value.element_size() != sizes[index]:
                raise RuntimeError(
                    "a source gradient's size differs from its input's"
                )
            late_values.append(
                (index, value if value.is_contiguous() else value.contiguous())
            )
        late = len(late_values)
        copy_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(copy_stream):
            for index, value in late_values:
                value.record_stream(copy_stream)
                destinations[index] = self._pinned_arena.copy_into(
                    slab, offsets[index], value
                )
                hooked[index] = value
            event = torch.cuda.Event()
            event.record(copy_stream)
        late_values = []
        slab.copy_event = event
        gradients = None
        stored: list[_SourceValue] = [None] * len(inputs)
        stored_bytes = 0
        for index, value in enumerate(hooked):
            if destinations[index] is None:
                continue
            stored[index] = _PendingCpuTransfer(
                cpu_tensor=destinations[index],
                source_tensor=value,
                event=event,
                bytes=sizes[index],
            )
            stored_bytes += sizes[index]
        submit_s = submit[0] + time.perf_counter() - started
        with self._accounting_lock:
            self._source_copies_in_backward += sum(
                value is not None for value in stored
            ) - late
            self._source_copies_after_backward += late
            self._source_gradients_released_in_backward += sum(released)
            self._source_offload_submit_s += submit_s
            self._source_offload_inflight_bytes += stored_bytes
            self._peak_source_offload_inflight_bytes = max(
                self._peak_source_offload_inflight_bytes,
                self._source_offload_inflight_bytes,
            )
            self._current_source_bytes += stored_bytes
            self._stored_source_bytes += stored_bytes
            self._peak_source_bytes = max(
                self._peak_source_bytes,
                self._current_source_bytes,
            )
        parameter_count = len(self.params)
        return stored[:parameter_count], stored[parameter_count:]

    def _abandon_slab_copies(
        self,
        key: tuple[int, int],
        copy_stream: Any,
        hooked: list[torch.Tensor | None],
    ) -> None:
        """Release a slab whose source will not be kept, once its copies land."""

        event = torch.cuda.Event()
        event.record(copy_stream)
        event.synchronize()
        for index in range(len(hooked)):
            hooked[index] = None
        with self._reducer_condition:
            self._release_source_slab_locked(key)

    @staticmethod
    def _current_stream_event() -> Any:
        if not torch.cuda.is_available():
            return None
        event = torch.cuda.Event()
        event.record()
        return event

    def _device_statistic_homes(self) -> DeviceStatisticPool | None:
        """The step's device homes, allocated on first use.

        The step's first group registration builds them, so they are
        resident before any of its close rebuilds reads the device.
        """

        if self._device_statistic_spec is None:
            return None
        with self._device_statistic_pool_lock:
            if self._device_statistic_pool is None:
                element_counts, dtype, device = self._device_statistic_spec
                if not self._device_statistic_slots:
                    raise RuntimeError("device statistic homes need device_statistic_slots")
                pool = DeviceStatisticPool(
                    element_counts,
                    dtype,
                    slots=self._device_statistic_slots,
                    device=device,
                )
                self._device_statistic_pool = pool
                self._statistic_pool = pool
            return self._device_statistic_pool

    def _group_statistics_on_device(self, group_id: int) -> bool:
        """Does this group's parameter statistic pair fold on the device?

        Without device homes the statistics stay where the group's first
        folded source lives. With them, the group takes a home when its
        first source reaches the offload worker (the one thread that asks),
        before that source's fold side is chosen; a group that finds every
        home owned folds on the host for its whole life.
        """

        if self._device_statistic_slots is None:
            return True
        pool = self._device_statistic_homes()
        with self._reducer_condition:
            accumulator = self._groups.get(group_id)
            if accumulator is None:
                raise RuntimeError(
                    f"group {group_id} vanished during source offload"
                )
            if accumulator.statistic_slot_tried:
                return accumulator.statistic_slot is not None
            accumulator.statistic_slot_tried = True
        slot = None if pool is None else pool.acquire((id(self), group_id))
        with self._reducer_condition:
            accumulator.statistic_slot = slot
            if slot is not None:
                self._statistic_slots[group_id] = slot
            else:
                self._host_routed_group_count += 1
        return slot is not None

    def _bound_device_source_bytes(self) -> int:
        """Device bytes of reward-bound sources still on the device.

        Each folds without any further admission, so a backward may count
        its memory as free once `_drain_folding_device_sources` returned.
        """

        with self._reducer_condition:
            return sum(
                size
                for (group_id, trajectory_id), size in (
                    self._device_source_bytes.items()
                )
                if (accumulator := self._groups.get(group_id)) is not None
                and trajectory_id in accumulator.closed
            )

    def _drain_folding_device_sources(self) -> None:
        """Let every reward-bound device-resident source fold before a backward.

        The device counterpart of `_drain_earlier_source_copies`: a bound
        source on the device frees its memory when its fold has been
        enqueued, which needs only the reducer. A source still awaiting its
        reward stays; the memory rules read it as held.
        """

        if not self._device_source_bytes:
            return
        started = time.perf_counter()
        waited = False
        with self._reducer_condition:
            while any(
                (accumulator := self._groups.get(group_id)) is not None
                and trajectory_id in accumulator.closed
                for group_id, trajectory_id in self._device_source_bytes
            ):
                self._raise_worker_failure()
                waited = True
                self._reducer_condition.wait()
            self._raise_worker_failure()
        if waited:
            with self._accounting_lock:
                self._device_source_drain_s += time.perf_counter() - started

    def _release_statistic_slots(self) -> int:
        """Return every held statistic home; a later taker waits for copies."""

        with self._reducer_condition:
            slots = list(self._statistic_slots.values())
            self._statistic_slots.clear()
            for accumulator in self._groups.values():
                accumulator.statistic_slot = None
        if slots:
            if self._statistic_pool is None:
                raise RuntimeError("a statistic slot is held without a device statistic pool")
            event = self._current_stream_event()
            for slot in slots:
                self._statistic_pool.release(slot, reload_event=event)
        return len(slots)

    def _drain_earlier_source_copies(self) -> None:
        """Let every earlier source leave the device before a backward.

        The residency budget admits a pack while earlier bound sources still
        fold, so an earlier source's full-model device gradient may still be
        copying out when the pack's first backward starts. Waiting for the
        copy is not enough: the device tensor is freed only when its last
        reference goes, and the offload worker drops that reference on its
        own schedule. So this waits for every pending copy and drops the
        device reference itself; the backward then starts with no earlier
        source on the device (the rebuild still overlaps the copies).
        """

        with self._reducer_condition:
            transfers = [
                value
                for group in self._groups.values()
                for sources in (
                    *group.parameter_sources.values(),
                    *group.boundary_sources.values(),
                )
                for value in sources
                if isinstance(value, _PendingCpuTransfer)
                and value.source_tensor is not None
            ]
        if not transfers:
            return
        started = time.perf_counter()
        for transfer in transfers:
            if transfer.event is not None:
                transfer.event.synchronize()
            # The copy has landed; the pinned destination is all the
            # reducer reads. materialize() drops the same reference.
            transfer.source_tensor = None
        with self._accounting_lock:
            self._source_copy_drain_s += time.perf_counter() - started
            self._source_copy_drain_released += len(transfers)

    def _slab_release_possible(self) -> bool:
        """Can some held slab be freed without the admitting caller?

        A slab frees when its source folds, and a source folds only once its
        reward is bound; a held source whose reward is unbound waits on the
        caller that is itself waiting for a slab.
        """

        with self._reducer_condition:
            self._raise_worker_failure()
            if not self._source_slabs:
                return True
            for group_id, trajectory_id in self._source_slabs:
                accumulator = self._groups.get(group_id)
                if accumulator is not None and trajectory_id in accumulator.closed:
                    return True
            return False

    def _release_source_slab_locked(self, key: tuple[int, int]) -> None:
        slab = self._source_slabs.pop(key, None)
        if slab is not None:
            if self._pinned_arena is None:
                raise RuntimeError("pinned-slab source copies need a pinned source arena")
            self._pinned_arena.release(slab)

    def _release_all_source_slabs(self) -> int:
        with self._reducer_condition:
            keys = list(self._source_slabs)
            for key in keys:
                self._release_source_slab_locked(key)
        return len(keys)

    def _streaming_pinned_buffer(self, slot: int) -> torch.Tensor:
        """Return one of this worker's fixed-size pinned staging windows.

        Each worker owns ``source_offload_pinned_windows`` windows so several
        chunks' D2H can be outstanding while an earlier chunk's fold still
        reads its own window.  Windows are allocated on first use of each slot.
        """

        if not 0 <= slot < self._streaming_pinned_windows:
            raise RuntimeError("streaming pinned window slot is out of range")
        thread_id = threading.get_ident()
        buffers = self._streaming_pinned_buffers.get(thread_id)
        if buffers is None:
            buffers = []
            self._streaming_pinned_buffers[thread_id] = buffers
        while len(buffers) <= slot:
            destination = torch.empty(
                self._source_offload_window_bytes,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            )
            buffers.append(destination)
            with self._accounting_lock:
                self._streaming_pinned_buffer_reserved_bytes += (
                    destination.numel()
                )
        return buffers[slot]

    def _parameter_device(self) -> torch.device | None:
        return self.params[0].device if self.params else None

    @staticmethod
    def _device_span_start(device: torch.device | None) -> Any:
        if device is None or device.type != "cuda":
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    @staticmethod
    def _device_span_end(start: Any, sink: list) -> None:
        if start is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        sink.append((start, end))

    @staticmethod
    def _device_span_total_s(spans: list) -> float:
        if not spans:
            return 0.0
        torch.cuda.synchronize()
        return sum(start.elapsed_time(end) for start, end in spans) / 1000.0

    def _streaming_fold_buffer(
        self, elements: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """Return this worker's reusable statistic-dtype conversion buffer.

        ATen accumulates a bfloat16 source into an fp32 statistic through an
        unvectorized mixed-dtype kernel.  Converting the chunk once and then
        accumulating in one dtype is several times faster.  The conversion is
        exact, so the accumulated statistic is unchanged.
        """

        thread_id = threading.get_ident()
        buffer = self._streaming_fold_buffers.get(thread_id)
        if buffer is None or buffer.numel() < elements or buffer.dtype != dtype:
            capacity = max(
                elements,
                self._source_offload_window_bytes
                // max(1, torch.empty(0, dtype=dtype).element_size()),
            )
            buffer = torch.empty(capacity, dtype=dtype)
            self._streaming_fold_buffers[thread_id] = buffer
            with self._accounting_lock:
                self._streaming_fold_buffer_reserved_bytes += (
                    buffer.numel() * buffer.element_size()
                )
        return buffer[:elements]

    def _submit_streaming_chunk(
        self,
        value: torch.Tensor,
        *,
        source_ready_event: Any | None,
        slot: int = 0,
    ) -> _PendingCpuTransfer:
        """Copy one flat CUDA chunk through a reducer-owned pinned window."""

        if value.device.type != "cuda" or value.ndim != 1:
            raise RuntimeError("streaming source chunk must be a flat CUDA tensor")
        size_bytes = value.numel() * value.element_size()
        if size_bytes > self._source_offload_window_bytes:
            raise RuntimeError("streaming source chunk exceeds its pinned window")
        raw_destination = self._streaming_pinned_buffer(slot)
        destination = raw_destination[:size_bytes].view(value.dtype)
        started = time.perf_counter()
        stream_key = (value.device, threading.get_ident())
        copy_stream = self._offload_streams.get(stream_key)
        if copy_stream is None:
            copy_stream = torch.cuda.Stream(device=value.device)
            self._offload_streams[stream_key] = copy_stream
        if source_ready_event is None:
            copy_stream.wait_stream(torch.cuda.current_stream(value.device))
        else:
            copy_stream.wait_event(source_ready_event)
        with torch.cuda.stream(copy_stream):
            value.record_stream(copy_stream)
            destination.copy_(value, non_blocking=True)
            event = torch.cuda.Event()
            event.record(copy_stream)
        submit_s = time.perf_counter() - started
        with self._accounting_lock:
            self._source_offload_submit_s += submit_s
            self._source_offload_inflight_bytes += size_bytes
            self._peak_source_offload_inflight_bytes = max(
                self._peak_source_offload_inflight_bytes,
                self._source_offload_inflight_bytes,
            )
        return _PendingCpuTransfer(
            cpu_tensor=destination,
            source_tensor=value,
            event=event,
            bytes=size_bytes,
        )

    def _hbm_pressure_fraction(
        self,
        sources: list[_SourceValue],
    ) -> float:
        device = next(
            (
                value.device
                for value in sources
                if isinstance(value, torch.Tensor)
                and value.device.type == "cuda"
            ),
            None,
        )
        if device is None:
            return 0.0
        capacity = torch.cuda.get_device_properties(device).total_memory
        fraction = torch.cuda.memory_reserved(device) / capacity
        with self._accounting_lock:
            self._max_observed_allocator_reserved_fraction = max(
                self._max_observed_allocator_reserved_fraction,
                fraction,
            )
        return fraction

    def _materialize_sources(self, sources: list[_SourceValue]) -> None:
        """Wait for pending D2H events in the completion worker, never admission."""

        for index, value in enumerate(sources):
            if not isinstance(value, _PendingCpuTransfer):
                continue
            started = time.perf_counter()
            materialized = value.materialize()
            waited_s = time.perf_counter() - started
            with self._accounting_lock:
                self._source_offload_wait_s += waited_s
                self._source_offload_inflight_bytes -= value.bytes
                if self._source_offload_inflight_bytes < 0:
                    raise RuntimeError("source offload residency accounting underflow")
            sources[index] = materialized

    def _spill_device_resident_sources(
        self,
        sources: list[_SourceValue],
    ) -> None:
        """Move one trajectory's device-resident sources to pinned CPU.

        Runs only in the offload worker, only above the sampled allocator
        watermark, after the producing stream's event has completed.  The
        readiness-order fold later consumes the spilled copy unchanged, so
        admission timing, fold order, and arithmetic are untouched -- only
        the memory where this trajectory's sources wait differs.
        """

        spilled_count = 0
        spilled_bytes = 0
        for index, value in enumerate(sources):
            if not isinstance(value, torch.Tensor):
                continue
            if value.device.type != "cuda":
                continue
            size_bytes = value.numel() * value.element_size()
            sources[index] = self._submit_pinned_cpu_copy(value, size_bytes)
            spilled_count += 1
            spilled_bytes += size_bytes
        if spilled_count:
            with self._accounting_lock:
                self._device_resident_spilled_source_count += spilled_count
                self._device_resident_spilled_source_bytes += spilled_bytes

    @staticmethod
    def _source_bytes(
        sources: dict[int, list[_SourceValue]],
    ) -> int:
        return sum(
            (
                value.bytes
                if isinstance(value, _PendingCpuTransfer)
                else value.numel() * value.element_size()
            )
            for gradients in sources.values()
            for value in gradients
            if value is not None
        )

    def _reload_combined(
        self,
        slot: Any,
        index: int,
        value: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """A group's combined parameter statistic, on ``device``.

        From a pooled home the copy is an asynchronous DMA from registered
        memory; otherwise it is the pageable reload. The bytes are the same.
        """

        pool = self._statistic_pool
        if (
            slot is None
            or pool is None
            or device.type != "cuda"
            or value.dtype != pool.dtype
            or not pool.fits(index, value)
            or value.data_ptr()
            != pool.view(slot, index, value.shape, statistic="reward").data_ptr()
        ):
            return self._reload_source(value, device)
        started = time.perf_counter()
        result = pool.reload_reward(slot, index, value.shape, device)
        with self._accounting_lock:
            self._source_reload_s += time.perf_counter() - started
            self._reloaded_source_bytes += value.numel() * value.element_size()
            self._pooled_reload_bytes += value.numel() * value.element_size()
        return result

    def _reload_source(
        self,
        value: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if value.device == device:
            return value
        size_bytes = value.numel() * value.element_size()
        started = time.perf_counter()
        result = value.to(device)
        with self._accounting_lock:
            self._source_reload_s += time.perf_counter() - started
            self._reloaded_source_bytes += size_bytes
        return result

    def retained_device_bytes(self) -> dict[str, Any]:
        """Device bytes this backward holds, by the attribute that holds them.

        The run's own probe (``StreamingRun.retained_device_bytes``) covers
        the parameters, the K/V and the ledgers; what it reports as
        unaccounted is held elsewhere or by nothing. This walks this object's
        attributes -- source queues, staging, fold accumulators, whatever a
        mode keeps resident -- to a fixed depth, storage-deduplicated, and
        attributes the bytes to the top-level attribute name, so a
        full-model-sized buffer that outlives the step is named rather than
        inferred from arithmetic. The run and the model are skipped (the run
        counts them); threads, locks and executors are skipped by module.
        Reflection, not a contract: attribute names are this class's own.

        ``_groups`` -- the per-group accumulators held until their group
        closes -- is the largest such holder on the streaming path, and the flat
        entry does not say what kind of tensor those bytes are.
        ``_groups_by_kind`` breaks it down by the
        ``_GroupAccum`` field that reaches each storage:
        ``parameter_sources`` and ``boundary_sources`` are the per-trajectory
        adjoints awaiting their fold (device-resident on the streaming path, and the
        still-referenced CUDA side of an in-flight pinned copy on the
        offloading configurations); ``parameter_centered_sum``,
        ``parameter_source_mean``, ``boundary_centered_sum`` and
        ``boundary_source_mean`` are the running group statistics;
        ``token_weights_by_trajectory`` is the caller's per-token weighting;
        ``other`` is whatever the ``_groups`` dictionary reaches outside an
        accumulator field. The buckets share the one storage ledger with the
        flat entries, so they sum to ``_groups`` plus any ``_groups.grad``.
        ``open_groups`` (accumulators whose group has not closed) and
        ``group_tensor_count`` (distinct storages the buckets counted) are
        counts, not bytes.
        """

        device = self.run.weight.device
        seen: set[int] = set()
        totals: dict[str, int] = {}
        group_totals: dict[str, int] = {}
        group_tensors = 0

        def visit(value: Any, key: str, depth: int, kind: str | None) -> None:
            nonlocal group_tensors
            if depth > 5:
                return
            if isinstance(value, torch.Tensor):
                if value.device != device:
                    return
                storage = value.untyped_storage()
                pointer = storage.data_ptr()
                if pointer in seen or storage.nbytes() == 0:
                    return
                seen.add(pointer)
                totals[key] = totals.get(key, 0) + storage.nbytes()
                if kind is not None:
                    group_totals[kind] = (
                        group_totals.get(kind, 0) + storage.nbytes()
                    )
                    group_tensors += 1
                if value.grad is not None:
                    visit(
                        value.grad,
                        key + ".grad",
                        depth,
                        None if kind is None else kind + ".grad",
                    )
                return
            if isinstance(value, (str, bytes, int, float, bool, type(None))):
                return
            if isinstance(value, (torch.nn.Module, StreamingRun)):
                return
            module = type(value).__module__ or ""
            if module.startswith(("threading", "concurrent", "queue", "_thread", "weakref")):
                return
            if kind is not None and isinstance(value, _GroupAccum):
                # The one place the walk keeps a field name: inside the
                # `_groups` subtree the accumulator's own field says which
                # kind of tensor the bytes are.
                for name, item in list(vars(value).items()):
                    visit(item, key, depth + 1, name)
            elif isinstance(value, dict):
                for item in list(value.values()):
                    visit(item, key, depth + 1, kind)
            elif isinstance(value, (list, tuple, set, frozenset, deque)):
                for item in list(value):
                    visit(item, key, depth + 1, kind)
            elif hasattr(value, "__dict__"):
                for item in list(vars(value).values()):
                    visit(item, key, depth + 1, kind)

        self._reducer_condition.acquire()
        self._accounting_lock.acquire()
        try:
            for name, value in list(vars(self).items()):
                if name in ("run", "model"):
                    continue
                visit(value, name, 0, "other" if name == "_groups" else None)
            open_groups = len(self._groups)
        finally:
            self._accounting_lock.release()
            self._reducer_condition.release()
        return {
            **dict(sorted(totals.items())),
            "accounted": sum(totals.values()),
            "_groups_by_kind": {
                **dict(sorted(group_totals.items())),
                "open_groups": open_groups,
                "group_tensor_count": group_tensors,
            },
        }

    @property
    def source_offload_report(self) -> dict[str, Any]:
        # Admission, offload, and reduction mutate different portions of this
        # evidence concurrently.  Acquire locks in the single declared order
        # used by reporting so callers never observe a mixed-time snapshot.
        self._reducer_condition.acquire()
        self._accounting_lock.acquire()
        try:
            report = {
            "storage_device": (
                str(self.source_offload_device)
                if self.source_offload_device is not None
                else "parameter_device"
            ),
            "source_queue_storage_device": (
                "parameter_device_until_streamed_d2h"
                if self.source_offload_mode == "pinned_streaming"
                else "parameter_device_until_watermark_spill"
                if self.source_offload_mode == "device_resident"
                else (
                    str(self.source_offload_device)
                    if self.source_offload_device is not None
                    else "parameter_device"
                )
            ),
            "current_source_bytes": self._current_source_bytes,
            "peak_source_bytes": self._peak_source_bytes,
            "stored_source_bytes": self._stored_source_bytes,
            "reloaded_source_bytes": self._reloaded_source_bytes,
            "source_offload_mode": self.source_offload_mode,
            "source_offload_s": self._source_offload_s,
            "source_offload_submit_s": self._source_offload_submit_s,
            "source_offload_wait_s": self._source_offload_wait_s,
            "source_offload_inflight_bytes": self._source_offload_inflight_bytes,
            "peak_source_offload_inflight_bytes": (
                self._peak_source_offload_inflight_bytes
            ),
            "source_offload_window_bytes": self._source_offload_window_bytes,
            "binary_rewards_declared": self._binary_rewards,
            "binary_host_statistics": self._binary_host_statistics,
            "binary_statistic_parameters": self._binary_statistic_parameters,
            "binary_statistic_adds": self._binary_statistic_adds,
            "host_statistic_tile_bytes": self._host_statistic_tile_bytes,
            "host_statistic_tiled_parameters": (
                self._host_statistic_tiled_parameters
            ),
            "host_statistic_tiled_elements": self._host_statistic_tiled_elements,
            "host_statistic_tile_count": self._host_statistic_tile_count,
            "host_parameter_dispatches": self._host_parameter_dispatches,
            "host_parameter_tasks": self._host_parameter_tasks,
            "source_offload_hbm_threshold": (
                self._source_offload_hbm_threshold
            ),
            "source_offload_hbm_trigger_semantics": (
                "sampled_allocator_reserved_fraction_soft_trigger_not_cap"
            ),
            "streaming_pinned_buffer_reserved_bytes": (
                self._streaming_pinned_buffer_reserved_bytes
            ),
            "source_pinned_acquire_s": self._source_pinned_acquire_s,
            "source_pinned_slabs_held": len(self._source_slabs),
            "pinned_arena": (
                None
                if self._pinned_arena is None
                else self._pinned_arena.report()
            ),
            "streaming_pinned_buffer_capacity_bytes": (
                (
                    self._source_offload_worker_count
                    + self._source_reduction_worker_count
                )
                * self._streaming_pinned_windows
                * self._source_offload_window_bytes
                if self.source_offload_mode == "pinned_streaming"
                else 0
            ),
            "hbm_staged_source_count": self._hbm_staged_source_count,
            "hbm_staged_source_bytes": self._hbm_staged_source_bytes,
            "hbm_stage_s": self._hbm_stage_s,
            "device_resident_spilled_source_count": (
                self._device_resident_spilled_source_count
            ),
            "device_resident_spilled_source_bytes": (
                self._device_resident_spilled_source_bytes
            ),
            "branch_rebuild_s": self._branch_rebuild_s,
            "branch_rebuild_count": self._branch_rebuild_count,
            "branch_backward_s": self._branch_backward_s,
            "branch_backward_count": self._branch_backward_count,
            "direct_backward_count": self._direct_backward_count,
            "direct_backward_s": self._direct_backward_s,
            "direct_group_count": self._direct_group_count,
            "uniform_reward_group_count": self._uniform_reward_group_count,
            "final_forest_calls": self._final_forest_calls,
            "final_forest_trajectory_count": self._final_forest_trajectory_count,
            "final_forest_logical_tokens": self._final_forest_logical_tokens,
            "final_forest_microbatches": self._final_forest_microbatches,
            "final_forest_out_of_memory_retries": (
                self._final_forest_out_of_memory_retries
            ),
            "final_forest_s": self._final_forest_s,
            "final_forest_budgets": list(self._final_forest_budgets),
            "final_forest_peak_allocated_bytes": (
                self._final_forest_peak_allocated_bytes
            ),
            "forest_attention_cache_saved": self._forest_attention_cache_saved,
            "forest_attention_cache_replayed": self._forest_attention_cache_replayed,
            "forest_attention_cache_declined": self._forest_attention_cache_declined,
            "forest_attention_cache_budget_bytes": (
                self._forest_attention_cache_budget_bytes
            ),
            "forest_attention_cache_charged_bytes": (
                self._forest_attention_cache_charged_bytes
            ),
            "ceded_final_count": self._ceded_final_count,
            "claimed_final_calls": self._claimed_final_calls,
            "claimed_final_count": self._claimed_final_count,
            "claimed_final_logical_tokens": self._claimed_final_logical_tokens,
            "claimed_final_microbatches": self._claimed_final_microbatches,
            "claimed_final_out_of_memory_retries": (
                self._claimed_final_out_of_memory_retries
            ),
            "claimed_final_s": self._claimed_final_s,
            "streaming_chunk_copy_wait_s": self._streaming_chunk_copy_wait_s,
            "streaming_chunk_fold_s": self._streaming_chunk_fold_s,
            "streaming_chunk_count": self._streaming_chunk_count,
            "branch_rebuild_device_s": self._device_span_total_s(
                self._branch_rebuild_events
            ),
            "branch_backward_device_s": self._device_span_total_s(
                self._branch_backward_events
            ),
            "group_close_boundary_device_s": self._device_span_total_s(
                self._group_close_boundary_events
            ),
            "streaming_fold_conversion_bytes": (
                self._streaming_fold_conversion_bytes
            ),
            "streaming_fold_buffer_reserved_bytes": (
                self._streaming_fold_buffer_reserved_bytes
            ),
            "group_close_host_combined_bytes": (
                self._group_close_host_combined_bytes
            ),
            "group_close_host_combine_s": self._group_close_host_combine_s,
            "group_close_adjoint_s": self._group_close_adjoint_s,
            "group_close_release_s": self._group_close_release_s,
            "group_close_barrier_s": self._group_close_barrier_s,
            "group_close_boundary_s": self._group_close_boundary_s,
            "group_close_free_s": self._group_close_free_s,
            "group_close_parameter_s": self._group_close_parameter_s,
            "group_finalized_in_reducer_count": (
                self._group_finalized_in_reducer_count
            ),
            "sealed_statistics_combine_count": self._sealed_statistics_combine_count,
            "group_last_fold_s": self._group_last_fold_s,
            "group_close_parameter_wait_s": self._group_close_parameter_wait_s,
            "pooled_reload_bytes": self._pooled_reload_bytes,
            "source_copy_drain_s": self._source_copy_drain_s,
            "source_copy_drain_released": self._source_copy_drain_released,
            "source_copy_during_backward": (
                self._source_copy_during_backward
                and self._pinned_arena is not None
            ),
            "source_release_during_backward": self._source_release_during_backward,
            "combine_known_reward_sources": self._combine_known_reward_sources,
            "known_reward_cohorts": list(self._known_reward_cohorts),
            "source_gradients_released_in_backward": self._source_gradients_released_in_backward,
            "max_returned_parameter_gradient_storage_bytes": (
                self._max_returned_parameter_gradient_storage_bytes
            ),
            "max_source_copy_pending_bytes": self._max_source_copy_pending_bytes,
            "source_copy_stream_joins": self._source_copy_stream_joins,
            "source_copies_in_backward": self._source_copies_in_backward,
            "source_copies_after_backward": self._source_copies_after_backward,
            "source_slab_busy_before_backward": (
                self._source_slab_busy_before_backward
            ),
            "group_close_deferred_count": self._group_close_deferred_count,
            "backward_out_of_memory_retries": dict(
                self._backward_out_of_memory_retries
            ),
            "rebuild_out_of_memory_retries": dict(
                self._rebuild_out_of_memory_retries
            ),
            "rebuild_out_of_memory_first_attempts": (
                self._rebuild_out_of_memory_first_attempts
            ),
            "group_close_boundary_statistics_s": (
                self._group_close_boundary_statistics_s
            ),
            "max_observed_allocator_reserved_fraction": (
                self._max_observed_allocator_reserved_fraction
            ),
            "current_pageable_stage_bytes": self._current_pageable_stage_bytes,
            "peak_pageable_stage_bytes": self._peak_pageable_stage_bytes,
            "source_contiguous_copy_bytes": self._source_contiguous_copy_bytes,
            "max_single_source_contiguous_copy_bytes": (
                self._max_single_source_contiguous_copy_bytes
            ),
            "source_offload_pending_trajectories": (
                len(self._streaming_pending_tensors)
                if self.source_offload_mode == "pinned_streaming"
                else len(self._offload_queue)
            ),
            "completed_source_headers": self._completed_source_headers,
            "source_offload_pending_trajectories_peak": (
                self._offload_pending_trajectories_peak
            ),
            "source_offload_active_trajectories": self._active_offload_jobs,
            "source_offload_active_trajectories_peak": (
                self._active_offload_jobs_peak
            ),
            "source_offload_active_trajectories_by_group": {
                group_id: active
                for group_id, active in sorted(
                    self._active_offload_jobs_by_group.items()
                )
                if active > 0
            },
            "source_offload_active_trajectories_peak_by_group": {
                group_id: peak
                for group_id, peak in sorted(
                    self._active_offload_jobs_peak_by_group.items()
                )
            },
            "hbm_stage_active_trajectories": self._active_hbm_stage_jobs,
            "hbm_stage_active_trajectories_peak": (
                self._active_hbm_stage_jobs_peak
            ),
            "hbm_stage_active_trajectories_by_group": {
                group_id: active
                for group_id, active in sorted(
                    self._active_hbm_stage_jobs_by_group.items()
                )
                if active > 0
            },
            "hbm_stage_active_trajectories_peak_by_group": {
                group_id: peak
                for group_id, peak in sorted(
                    self._active_hbm_stage_jobs_peak_by_group.items()
                )
            },
            "hbm_stage_active_tensor_jobs": (
                self._active_hbm_stage_tensor_jobs
            ),
            "hbm_stage_active_tensor_jobs_peak": (
                self._active_hbm_stage_tensor_jobs_peak
            ),
            "hbm_stage_active_tensor_jobs_by_group": {
                group_id: active
                for group_id, active in sorted(
                    self._active_hbm_stage_tensor_jobs_by_group.items()
                )
                if active > 0
            },
            "hbm_stage_active_tensor_jobs_peak_by_group": {
                group_id: peak
                for group_id, peak in sorted(
                    self._active_hbm_stage_tensor_jobs_peak_by_group.items()
                )
            },
            "source_reload_s": self._source_reload_s,
            "source_spill_directory": (
                str(self.source_spill_directory)
                if self.source_spill_directory is not None
                else None
            ),
            "source_spill_write_s": self._source_spill_write_s,
            "source_spill_read_s": self._source_spill_read_s,
            "current_spill_bytes": self._current_spill_bytes,
            "peak_spill_bytes": self._peak_spill_bytes,
            "stored_spill_bytes": self._stored_spill_bytes,
            "reloaded_spill_bytes": self._reloaded_spill_bytes,
            "source_reduction_mode": self.source_reduction_mode,
            "source_reduction_pack_size": self.source_reduction_pack_size,
            "source_offload_pinned_windows": self._streaming_pinned_windows,
            "statistic_device": (
                None
                if self._statistic_device is None
                else str(self._statistic_device)
            ),
            "device_statistic_slots": self._device_statistic_slots,
            "device_statistic_pool": (
                None
                if self._device_statistic_pool is None
                else self._device_statistic_pool.report()
            ),
            "statistic_host_routed_group_count": (
                self._host_routed_group_count
            ),
            "device_source_drain_s": self._device_source_drain_s,
            "device_resident_source_bytes_current": sum(
                self._device_source_bytes.values()
            ),
            "executed_rebuild_pack_max": self._executed_rebuild_pack_max,
            "source_reduction_worker_count": (
                self._source_reduction_worker_count
            ),
            "source_offload_worker_count": (
                self._source_offload_worker_count
                if self.source_offload_mode == "pinned_streaming"
                else 1
            ),
            "statistic_representation": (
                "readiness_order_reward_and_unit_sums_in_memory"
            ),
            "current_statistic_bytes": self._current_statistic_bytes,
            "peak_statistic_bytes": self._peak_statistic_bytes,
            "deferred_ready_trajectories_peak": (
                self._deferred_ready_trajectories_peak
            ),
            "pending_reduction_trajectories_peak": (
                self._pending_reduction_trajectories_peak
            ),
            "pending_reduction_trajectories_current": sum(
                len(group.differentiated - group.reduced)
                for group in self._groups.values()
            ),
            "reducer_queue_s": self._reducer_queue_s,
            "reducer_queue_max_s": self._reducer_queue_max_s,
            "reducer_fold_s": self._reducer_fold_s,
            "reducer_executing_source_bytes": (
                self._reducer_executing_source_bytes
            ),
            "reducer_executing_source_bytes_peak": (
                self._reducer_executing_source_bytes_peak
            ),
            "source_execution_order_by_group": {
                group_id: list(trajectory_ids)
                for group_id, trajectory_ids in sorted(
                    self._source_execution_order_by_group.items()
                )
            },
            "source_reduction_order_by_group": {
                group_id: list(trajectory_ids)
                for group_id, trajectory_ids in sorted(
                    self._source_reduction_order_by_group.items()
                )
            },
            "source_preparation_order_by_group": {
                group_id: list(trajectory_ids)
                for group_id, trajectory_ids in sorted(
                    self._source_preparation_order_by_group.items()
                )
            },
            "prepared_awaiting_reward_current": sum(
                len(group.prepared - group.closed)
                for group in self._groups.values()
            ),
            "prepared_awaiting_reward_peak": (
                self._prepared_awaiting_reward_peak
            ),
            }
            if self._close_trace is not None:
                report["close_trace"] = self._resolved_close_trace()
        finally:
            self._accounting_lock.release()
            self._reducer_condition.release()
        return report

    def _resolved_close_trace(self) -> list[dict[str, Any]]:
        """The close trace with its CUDA-event spans read as milliseconds."""

        synchronized = False

        def span_ms(span: tuple[Any, Any] | None) -> float | None:
            nonlocal synchronized
            if span is None:
                return None
            if not synchronized:
                torch.cuda.synchronize()
                synchronized = True
            return span[0].elapsed_time(span[1])

        def public(record: dict[str, Any]) -> dict[str, Any]:
            return {k: v for k, v in record.items() if not k.startswith("_")}

        resolved = []
        for pack in self._close_trace or []:
            record = public(pack)
            record["rebuild_device_ms"] = span_ms(pack.get("_rebuild_span"))
            record["branches"] = [
                {
                    **public(branch),
                    "grad_device_ms": span_ms(branch.get("_grad_span")),
                    "backward_device_ms": span_ms(
                        branch.get("_backward_span")
                    ),
                }
                for branch in pack["branches"]
            ]
            resolved.append(record)
        return resolved

    def _close_trace_pack(
        self, trajectory_ids: list[int]
    ) -> dict[str, Any] | None:
        """Open one pack's close-trace record, or None when tracing is off."""

        if self._close_trace is None:
            return None
        device = self._parameter_device()
        on_cuda = device is not None and device.type == "cuda"
        checkpoint = self.run.rebuild_checkpoint_offload_diagnostics()
        return {
            "trajectory_ids": list(trajectory_ids),
            "branch_tokens": self._branch_token_counts(trajectory_ids),
            "started_at": time.perf_counter(),
            "allocated_bytes": (
                torch.cuda.memory_allocated(device) if on_cuda else None
            ),
            "reserved_bytes": (
                torch.cuda.memory_reserved(device) if on_cuda else None
            ),
            "rungs": [],
            "out_of_memory_retries": [],
            "branches": [],
            "_checkpoint_read": (
                checkpoint.get("read_bytes", 0),
                checkpoint.get("read_s", 0.0),
            ),
        }

    def _branch_token_counts(self, trajectory_ids: list[int]) -> list[int | None]:
        """Each branch's rebuilt token count (diagnostics only)."""

        states = getattr(self.run, "_trajs", {})
        counts: list[int | None] = []
        for trajectory_id in trajectory_ids:
            state = states.get(trajectory_id)
            counts.append(
                None
                if state is None
                else sum(len(turn) for turn, _score in state.turns)
            )
        return counts

    def set_diagnostic_scope(self, *, rank: int, step: int) -> None:
        """The rank and step the diagnostic profiler window selects on."""

        self._diagnostic_rank = int(rank)
        self._diagnostic_step = int(step)

    def _close_profile_start(self, trajectory_ids: list[int]) -> bool:
        profiler = self._close_profiler
        if profiler is None:
            return False
        tokens = [count or 0 for count in self._branch_token_counts(trajectory_ids)]
        if not profiler.selects(
            rank=self._diagnostic_rank,
            step=self._diagnostic_step,
            branch_tokens=tokens,
        ):
            return False
        profiler.start(
            rank=self._diagnostic_rank,
            step=self._diagnostic_step,
            trajectory_ids=trajectory_ids,
            branch_tokens=tokens,
        )
        return True

    def _close_trace_finish(self, trace: dict[str, Any] | None) -> None:
        if trace is None:
            return
        read_bytes, read_s = trace.pop("_checkpoint_read")
        checkpoint = self.run.rebuild_checkpoint_offload_diagnostics()
        trace["checkpoint_read_bytes"] = (
            checkpoint.get("read_bytes", 0) - read_bytes
        )
        trace["checkpoint_read_s"] = checkpoint.get("read_s", 0.0) - read_s
        trace["host_s"] = time.perf_counter() - trace["started_at"]
        with self._accounting_lock:
            if self._close_trace is None:
                raise RuntimeError("close tracing is not enabled")
            if len(self._close_trace) < _CLOSE_TRACE_MAX_PACKS:
                self._close_trace.append(trace)

    def configure_binary_rewards(self) -> None:
        """Declare a binary objective before admitting any group.

        Host source-dtype parameter statistics keep P=sum(r*g) and
        N=sum((r-1)*g). Their numerator is lerp(P, N, mean), equal to
        sum((r-mean)*g). Each binary source adds to only one statistic.
        The boundary and other residency/precision paths retain G1/G2.
        No reward value is inspected before its ordinary binding event.
        """
        with self._reducer_condition:
            if self._groups or getattr(self, "_expected_trajectory_order_by_group", {}):
                raise RuntimeError("binary rewards must be declared before group registration")
            self._binary_rewards = True
            self._binary_host_statistics = (
                self._statistic_keep_source_dtype
                and self.source_offload_mode in {"blocking", "pinned_nonblocking"}
                and self._statistic_device is None
            )

    def set_statistics_branch_observer(
        self, observer: Callable[[int], None] | None
    ) -> None:
        """Observe a completed statistic branch on its training thread."""
        self._statistics_branch_observer = observer

    def open_group(
        self,
        group_id: int,
        prompt_tokens: list[int],
        trajectory_ids: list[int] | tuple[int, ...],
    ) -> None:
        """Open a group on the run and register its trajectories.

        Equivalent to ``run.open_group(group_id, prompt_tokens)`` followed by
        ``register_group_trajectories(group_id, trajectory_ids)``. The
        trajectory ids are validated before the run opens the group, so a
        refused id list leaves no group open.
        """

        _trajectory_order(trajectory_ids)
        self.run.open_group(group_id, prompt_tokens)
        self.register_group_trajectories(group_id, trajectory_ids)

    def register_group_trajectories(
        self,
        group_id: int,
        trajectory_ids: list[int] | tuple[int, ...],
    ) -> None:
        """Declare a group's trajectory ids; their sorted order is the
        group's reduction order and must not change once registered."""
        order = _trajectory_order(trajectory_ids)
        self._device_statistic_homes()
        with self._reducer_condition:
            existing = self._expected_trajectory_order_by_group.get(group_id)
            if existing is not None and existing != order:
                raise RuntimeError(f"group {group_id}: trajectory order changed")
            self._expected_trajectory_order_by_group[group_id] = order
            self._reducer_condition.notify()

    # ------------------------------------------------------------------ events

    def close_trajectory(
        self,
        traj_id: int,
        reward: float,
        *,
        token_weights: torch.Tensor | None = None,
    ) -> None:
        """The trajectory's reward landed: run its branch backward now."""
        self.close_trajectories(
            [(traj_id, reward)],
            token_weights=None if token_weights is None
            else {traj_id: token_weights},
        )

    def close_trajectories(
        self,
        closes: list[tuple[int, float]],
        *,
        token_weights: dict[int, torch.Tensor] | None = None,
    ) -> None:
        """One or SEVERAL landed rewards: branch backwards now, rebuild shared.

        Frees each trajectory's retained K/V and graph up front (the
        residency win: memory returns per trajectory, not per batch), then
        computes every grad(S_i) through ONE block-checkpointed branch rebuild
        (streaming.rebuild_logprobs) -- with several closes, each block round
        packs all branches into one primal, so rewards that queued while the
        trainer was busy (the clustered-straggler window) pay batched dense
        kernels instead of k small rebuilds. The default backward stays PER branch:
        graphs are disjoint by construction, G1 needs r_i*grad(S_i) and G2
        needs grad(S_i) separately, and one traversal per branch is exactly
        what serial closes pay -- so coalescing changes the kernel schedule,
        never the graph walk count. Coalescing is expected to match
        serial closes. Opt-in known-reward cohorts instead execute one
        aggregate for already-ready members with the same group and reward.

        For non-streaming offload this call IS the composition
        ``prepare_trajectories`` (content-ready physical work) followed by
        ``bind_rewards`` (objective coupling); routing the legacy path through
        the same two calls keeps the early-training split bitwise equal to
        this path by construction. ``pinned_streaming`` keeps the one-shot
        body because its per-chunk folds consume the reward during autograd.
        """
        if not closes:
            raise ValueError("close_trajectories needs at least one close")
        if self.source_offload_mode != "pinned_streaming":
            direct = self._direct_closes(closes)
            if direct:
                self._close_with_direct(closes, direct, token_weights or {})
                return
            self.prepare_trajectories(
                [traj_id for traj_id, _ in closes],
                token_weights=token_weights,
            )
            self.bind_rewards(closes)
            return
        self._check_versions()
        token_weights = token_weights or {}
        with self._reducer_condition:
            self._record_verdicts_locked(closes)
            seen: set[int] = set()
            for traj_id, _ in closes:
                gid = self.run.group_of(traj_id)
                acc = self._groups.setdefault(gid, _GroupAccum())
                if traj_id in acc.closed or traj_id in seen:
                    raise RuntimeError(f"trajectory {traj_id} already closed")
                seen.add(traj_id)
            for traj_id, reward in closes:
                acc = self._groups[self.run.group_of(traj_id)]
                acc.closed.add(traj_id)
                acc.rewards_by_trajectory[traj_id] = float(reward)
                acc.token_weights_by_trajectory[traj_id] = token_weights.get(
                    traj_id
                )

        # Each close crossed its declared objective-readiness event. Execute
        # them immediately, in observed arrival order, as ONE pack: the pack is
        # exactly what had already landed when the trainer became free, so no
        # ready source waits for another to arrive. Packing shares one
        # block-checkpointed branch rebuild across the arrivals; the backward
        # stays per branch.
        self._execute_statistics_pack([traj_id for traj_id, _ in closes])

    def prepare_trajectory(
        self,
        traj_id: int,
        *,
        token_weights: torch.Tensor | None = None,
    ) -> None:
        """The trajectory content is immutable: run its branch backward now.

        The reward is neither required nor read; the produced source is
        retained (offloaded) until ``bind_reward`` attaches the scalar.
        """
        self.prepare_trajectories(
            [traj_id],
            token_weights=None if token_weights is None
            else {traj_id: token_weights},
        )

    def prepare_trajectories(
        self,
        trajectory_ids: list[int],
        *,
        token_weights: dict[int, torch.Tensor] | None = None,
    ) -> None:
        """Content-ready admission: differentiate and offload without rewards.

        Legal because the branch backward never reads an unlanded reward -- G1's AXPY
        and G2's fold consume the SAME source tensor, and only the deferred
        readiness-order fold (released by ``bind_rewards``) touches the
        scalar. The prebound grade is never peeked, and reduction eligibility
        requires the real reward event. Known-reward cohorts may combine only
        members whose own verdicts are already in the landed ledger.
        """
        self._check_versions()
        if not trajectory_ids:
            raise ValueError(
                "prepare_trajectories needs at least one trajectory"
            )
        if self.source_offload_mode == "pinned_streaming":
            raise RuntimeError(
                "early source preparation is incompatible with "
                "pinned_streaming: its chunk folds consume the reward scalar "
                "during the branch backward; use blocking or "
                "pinned_nonblocking offload, or close at the reward event"
            )
        token_weights = token_weights or {}
        with self._reducer_condition:
            self._raise_worker_failure()
            self._check_unadmitted_locked(trajectory_ids)
            for traj_id in trajectory_ids:
                acc = self._groups[self.run.group_of(traj_id)]
                acc.prepared.add(traj_id)
                acc.token_weights_by_trajectory[traj_id] = token_weights.get(
                    traj_id
                )
            self._prepared_awaiting_reward_peak = max(
                self._prepared_awaiting_reward_peak,
                sum(
                    len(group.prepared - group.closed)
                    for group in self._groups.values()
                ),
            )
        self._execute_statistics_pack(trajectory_ids)

    def bind_reward(self, traj_id: int, reward: float) -> None:
        """The trajectory's reward landed: attach the scalar to its source."""
        self.bind_rewards([(traj_id, reward)])

    def bind_rewards(self, binds: list[tuple[int, float]]) -> None:
        """Attach landed rewards to prepared sources.

        The recording order here IS the group's objective-readiness order:
        the reducer folds each group's sources in exactly this recorded
        order, never in preparation order. A source becomes
        reduction-eligible only when its offload has materialized AND its
        reward is bound; whichever side finishes second publishes the job,
        so no source is folded twice and none is folded early.
        """
        self._check_versions()
        if not binds:
            raise ValueError("bind_rewards needs at least one bind")
        with self._reducer_condition:
            self._raise_worker_failure()
            seen: set[int] = set()
            for traj_id, _ in binds:
                gid = self.run.group_of(traj_id)
                acc = self._groups.get(gid)
                if acc is None or traj_id not in acc.prepared:
                    raise RuntimeError(
                        f"trajectory {traj_id} was not prepared before "
                        "reward binding"
                    )
                if traj_id in acc.closed or traj_id in seen:
                    raise RuntimeError(f"trajectory {traj_id} already closed")
                seen.add(traj_id)
            self._record_verdicts_locked(binds)
            for traj_id, reward in binds:
                gid = self.run.group_of(traj_id)
                acc = self._groups[gid]
                acc.closed.add(traj_id)
                acc.rewards_by_trajectory[traj_id] = float(reward)
                self._source_execution_order_by_group.setdefault(
                    gid, []
                ).append(traj_id)
                self._source_queued_at[(gid, traj_id)] = time.perf_counter()
                if traj_id in acc.materialized:
                    self._reduction_ready_queue.append((gid, traj_id, None))
            self._reducer_condition.notify_all()

    def record_rewards(self, binds: list[tuple[int, float]]) -> None:
        """Landed rewards enter their groups' ledgers; nothing else happens.

        The ledger decides only which path a later close of the group's
        trajectories takes (``direct_when_group_bound``): no work starts or
        waits on it, and no source is bound by it.
        """

        with self._reducer_condition:
            self._raise_worker_failure()
            self._record_verdicts_locked(binds)

    def group_fully_bound(self, group_id: int) -> bool:
        """Has every registered trajectory of the group a recorded reward?"""

        with self._reducer_condition:
            return self._group_fully_bound_locked(group_id)

    def _record_verdicts_locked(self, binds: list[tuple[int, float]]) -> None:
        if getattr(self, "_binary_rewards", False) and any(
            float(reward) not in (0.0, 1.0) for _, reward in binds
        ):
            raise ValueError("declared binary rewards must be zero or one")
        for traj_id, reward in binds:
            acc = self._groups.setdefault(
                self.run.group_of(traj_id), _GroupAccum()
            )
            value = float(reward)
            recorded = acc.verdicts.setdefault(traj_id, value)
            if recorded != value:
                raise RuntimeError(
                    f"trajectory {traj_id}: reward {value} differs from its "
                    f"recorded verdict {recorded}"
                )

    def registered_groups_fully_bound(self) -> bool:
        """Has every trajectory of every registered group a recorded reward?

        Then every trajectory not yet differentiated is final work: its
        group's advantages are known.
        """

        with self._reducer_condition:
            self._raise_worker_failure()
            return bool(self._expected_trajectory_order_by_group) and all(
                group_id in self._closed_group_ids
                or self._group_fully_bound_locked(group_id)
                for group_id in self._expected_trajectory_order_by_group
            )

    @property
    def known_slab_capacity(self) -> int | None:
        """Arena slabs that no source awaiting its reward holds, or None.

        A close whose reward is known folds as soon as its source lands, so
        it can take any slab not held by such a source without waiting on
        an event this caller has to deliver. None without an arena.
        """

        if self._pinned_arena is None:
            return None
        limit = self._pinned_arena.slab_limit
        with self._reducer_condition:
            awaiting = sum(
                len(group.differentiated - group.closed)
                for group in self._groups.values()
            )
        return limit - awaiting

    def close_final_forest(
        self,
        closes: list[tuple[int, float]],
        *,
        prompt_tokens: dict[int, list[int] | tuple[int, ...]],
        max_tokens_per_microbatch: int,
        chunk: int,
    ) -> dict[str, Any]:
        """Final closes as forest micro-batches into the parameters' gradients.

        Every close's group must have every reward recorded, so ``A_i`` is
        the group's float64 advantage the close computes, and none of the
        trajectories may have been differentiated. Each is freed in the
        streaming run, rebuilt from the prompt and its recorded turns in the
        barrier objective's forest (``thundersync.grpo.final_forest``) and
        backwarded with its group's other final closes. It joins its group
        as a direct member with no boundary adjoint: the forest's prompt
        forward carries its prompt term, and the group's close still
        backwards the held and direct members' prompt term once.
        """

        if not closes:
            raise ValueError("close_final_forest needs at least one close")
        self._check_final_forest_mode()
        self._check_versions()
        trajectory_ids = [traj_id for traj_id, _ in closes]
        advantages, trajectories = self._enter_final_closes(closes, prompt_tokens)
        report = self._backward_forest(
            trajectories,
            advantages,
            max_tokens_per_microbatch=max_tokens_per_microbatch,
            chunk=chunk,
        )
        self._settle_final_closes(trajectory_ids)
        with self._accounting_lock:
            self._final_forest_calls += 1
            self._final_forest_trajectory_count += report.trajectories
            self._final_forest_logical_tokens += report.logical_tokens
            self._final_forest_microbatches += report.microbatches
            self._final_forest_out_of_memory_retries += (
                report.out_of_memory_retries
            )
            self._final_forest_s += report.wall_s
            self._final_forest_budgets.extend(report.budgets)
            self._final_forest_peak_allocated_bytes = max(
                self._final_forest_peak_allocated_bytes,
                report.peak_allocated_bytes,
            )
        return report.as_record()

    def cede_final_closes(self, closes: list[tuple[int, float]]) -> None:
        """Final closes another rank claimed: they join their groups unrun.

        Each is a final close the claims protocol gave to a peer, which runs
        ``A_i * S_i`` -- prompt term included -- in its own forest
        (`backward_claimed_final`) into its own gradients; the all-reduce
        sums the two. Here the trajectory is entered exactly as a final-forest
        member (closed, direct, no boundary adjoint), freed, and counted as
        differentiated and reduced, so its group's close runs once its other
        members are done and the evidence covers it on this, its owner's,
        rank.
        """

        if not closes:
            raise ValueError("cede_final_closes needs at least one close")
        self._check_final_forest_mode()
        self._enter_final_closes(closes, None)
        self._settle_final_closes([traj_id for traj_id, _ in closes])
        with self._accounting_lock:
            self._ceded_final_count += len(closes)

    def final_advantage(self, traj_id: int) -> float:
        """The float64 advantage of a trajectory whose group is fully bound,
        as the final forest and the direct path compute it."""

        with self._reducer_condition:
            self._raise_worker_failure()
            group_id = self.run.group_of(traj_id)
            if not self._group_fully_bound_locked(group_id):
                raise RuntimeError(
                    f"trajectory {traj_id}: its group {group_id} has rewards outstanding"
                )
            acc = self._groups[group_id]
            mean, std = self._reward_moments(
                [
                    acc.verdicts[member]
                    for member in self._expected_trajectory_order_by_group[group_id]
                ]
            )
            return (float(acc.verdicts[traj_id]) - mean) / std

    def backward_claimed_final(
        self,
        trajectories: list[Any],
        advantages: dict[int, float],
        *,
        max_tokens_per_microbatch: int,
        chunk: int,
    ) -> dict[str, Any]:
        """Another rank's final work, claimed by this one, into the gradients.

        ``trajectories`` are complete forest trajectories (prompt, then
        turns) and ``advantages`` their owner's float64 ``A_i``; nothing of
        their groups lives here, so no statistic, boundary or group state is
        touched. The forest accumulates ``sum_i A_i grad S_i`` -- the direct
        path's term with the prompt's part from the forest's own prompt
        forward -- into the parameters' gradients beside this rank's own.
        """

        if not trajectories:
            raise ValueError("backward_claimed_final needs at least one trajectory")
        self._check_final_forest_mode()
        self._check_versions()
        report = self._backward_forest(
            trajectories,
            advantages,
            max_tokens_per_microbatch=max_tokens_per_microbatch,
            chunk=chunk,
        )
        with self._accounting_lock:
            self._claimed_final_calls += 1
            self._claimed_final_count += report.trajectories
            self._claimed_final_logical_tokens += report.logical_tokens
            self._claimed_final_microbatches += report.microbatches
            self._claimed_final_out_of_memory_retries += report.out_of_memory_retries
            self._claimed_final_s += report.wall_s
        return report.as_record()

    def _check_final_forest_mode(self) -> None:
        if self.source_offload_mode == "pinned_streaming":
            raise RuntimeError(
                "the final forest needs prepare/bind sources; pinned_streaming "
                "folds the reward inside the branch backward"
            )
        if self.run._parameter_adjoint_capture_enabled:
            raise RuntimeError(
                "the final forest accumulates into the parameters' gradients, "
                "which this run's parameter-adjoint capture owns"
            )
        if self.run.shared_prefix_vjp_mode == "source_ready_serial":
            raise RuntimeError(
                "the final forest leaves no boundary source for a "
                "source-ready shared prefix"
            )

    def _enter_final_closes(
        self,
        closes: list[tuple[int, float]],
        prompt_tokens: dict[int, list[int] | tuple[int, ...]] | None,
    ) -> tuple[dict[int, float], list[Any]]:
        """Enter final closes as direct members with no boundary adjoint and
        free them in the run; their advantages and, with ``prompt_tokens``,
        their forest trajectories."""

        trajectory_ids = [traj_id for traj_id, _ in closes]
        advantages: dict[int, float] = {}
        with self._reducer_condition:
            self._raise_worker_failure()
            self._check_unadmitted_locked(trajectory_ids)
            self._record_verdicts_locked(closes)
            for traj_id, reward in closes:
                group_id = self.run.group_of(traj_id)
                if prompt_tokens is not None and group_id not in prompt_tokens:
                    raise RuntimeError(f"group {group_id}: no prompt tokens")
                if not self._group_fully_bound_locked(group_id):
                    raise RuntimeError(
                        f"trajectory {traj_id}: its group {group_id} has rewards "
                        "outstanding; only final work runs in the forest"
                    )
            for traj_id, reward in closes:
                group_id = self.run.group_of(traj_id)
                acc = self._groups[group_id]
                moments = self._reward_moments(
                    [
                        acc.verdicts[member]
                        for member in self._expected_trajectory_order_by_group[
                            group_id
                        ]
                    ]
                )
                if acc.direct_moments is None:
                    acc.direct_moments = moments
                    self._direct_group_count += 1
                elif acc.direct_moments != moments:
                    raise RuntimeError(f"group {group_id}: direct moments changed")
                mean, std = moments
                advantages[traj_id] = (float(reward) - mean) / std
                acc.token_weights_by_trajectory[traj_id] = None
                acc.closed.add(traj_id)
                acc.rewards_by_trajectory[traj_id] = float(reward)
                acc.direct.add(traj_id)
                acc.forest.add(traj_id)
                self._source_execution_order_by_group.setdefault(
                    group_id, []
                ).append(traj_id)
            self._reducer_condition.notify_all()
        trajectories = []
        for traj_id in trajectory_ids:
            group_id = self.run.group_of(traj_id)
            turns = [
                (list(tokens), list(scored))
                for tokens, scored in self.run._trajs[traj_id].turns
            ]
            self.run.free_trajectory(traj_id)
            if prompt_tokens is not None:
                trajectories.append(
                    forest_trajectory(group_id, traj_id, prompt_tokens[group_id], turns)
                )
        return advantages, trajectories

    def _backward_forest(
        self,
        trajectories: list[Any],
        advantages: dict[int, float],
        *,
        max_tokens_per_microbatch: int,
        chunk: int,
    ) -> Any:
        # Earlier sources leave the device before the forest's first walk,
        # as before any branch backward.
        self._drain_earlier_source_copies()
        self._drain_folding_device_sources()
        report = backward_final_forest(
            self._model,
            trajectories,
            advantages,
            cap_tokens=max_tokens_per_microbatch,
            chunk=chunk,
        )
        with self._accounting_lock:
            self._forest_attention_cache_saved += report.attention_cache_saved
            self._forest_attention_cache_replayed += report.attention_cache_replayed
            self._forest_attention_cache_declined += report.attention_cache_declined
            self._forest_attention_cache_budget_bytes = max(
                self._forest_attention_cache_budget_bytes,
                report.attention_cache_budget_bytes,
            )
            self._forest_attention_cache_charged_bytes = max(
                self._forest_attention_cache_charged_bytes,
                report.attention_cache_charged_bytes,
            )
        return report

    def _settle_final_closes(self, trajectory_ids: list[int]) -> None:
        with self._reducer_condition:
            for traj_id in trajectory_ids:
                group_id = self.run.group_of(traj_id)
                acc = self._groups[group_id]
                acc.differentiated.add(traj_id)
                acc.reduced.add(traj_id)
                for order in (
                    self._source_preparation_order_by_group,
                    self._source_reduction_order_by_group,
                ):
                    order.setdefault(group_id, []).append(traj_id)
            self._reducer_condition.notify_all()

    def _group_fully_bound_locked(self, group_id: int) -> bool:
        acc = self._groups.get(group_id)
        expected = self._expected_trajectory_order_by_group.get(group_id)
        return (
            acc is not None
            and expected is not None
            and all(traj_id in acc.verdicts for traj_id in expected)
        )

    def _check_unadmitted_locked(self, trajectory_ids: list[int]) -> None:
        seen: set[int] = set()
        for traj_id in trajectory_ids:
            acc = self._groups.setdefault(
                self.run.group_of(traj_id), _GroupAccum()
            )
            if traj_id in acc.closed:
                raise RuntimeError(f"trajectory {traj_id} already closed")
            if traj_id in acc.prepared or traj_id in seen:
                raise RuntimeError(f"trajectory {traj_id} already prepared")
            seen.add(traj_id)

    def _direct_closes(self, closes: list[tuple[int, float]]) -> set[int]:
        """The closes whose group has every reward bound once these land."""

        with self._reducer_condition:
            self._raise_worker_failure()
            self._record_verdicts_locked(closes)
            if not self._direct_when_group_bound:
                return set()
            return {
                traj_id
                for traj_id, _ in closes
                if self._group_fully_bound_locked(self.run.group_of(traj_id))
            }

    def _close_with_direct(
        self,
        closes: list[tuple[int, float]],
        direct: set[int],
        token_weights: dict[int, torch.Tensor],
    ) -> None:
        """One pack in which the fully bound groups' closes run direct.

        A direct trajectory backwards ``A_i * S_i`` into the parameters' and
        its group's boundary proxies' gradients (`_backward_direct`); the
        others are prepared and bound as ``close_trajectories`` always does.
        The group's close adds the statistics of its held members and the
        proxies' direct sum, which by linearity is the update the statistics
        of every member give.

        Each direct trajectory is rebuilt and backwarded alone, in the pack's
        order; consecutive held ones share a rebuild. A pack's branch graphs
        coexist until each is backwarded, so the rebuild's memory rule reads
        the pack's summed tokens and sends a pack to the recomputing block
        path where each member alone takes the plain one; a direct branch
        holds no source whose copy a later branch could overlap, so it gains
        nothing from sharing.
        """

        self._check_versions()
        held = [(traj_id, reward) for traj_id, reward in closes if traj_id not in direct]
        with self._reducer_condition:
            self._raise_worker_failure()
            self._check_unadmitted_locked([traj_id for traj_id, _ in closes])
            for traj_id, reward in closes:
                group_id = self.run.group_of(traj_id)
                acc = self._groups[group_id]
                acc.token_weights_by_trajectory[traj_id] = token_weights.get(
                    traj_id
                )
                if traj_id not in direct:
                    acc.prepared.add(traj_id)
                    continue
                moments = self._reward_moments(
                    [
                        acc.verdicts[member]
                        for member in self._expected_trajectory_order_by_group[
                            group_id
                        ]
                    ]
                )
                if acc.direct_moments is None:
                    acc.direct_moments = moments
                    self._direct_group_count += 1
                elif acc.direct_moments != moments:
                    raise RuntimeError(
                        f"group {group_id}: direct moments changed"
                    )
                acc.closed.add(traj_id)
                acc.rewards_by_trajectory[traj_id] = float(reward)
                acc.direct.add(traj_id)
                self._source_execution_order_by_group.setdefault(
                    group_id, []
                ).append(traj_id)
        run: list[int] = []
        for traj_id, _ in closes:
            if traj_id not in direct:
                run.append(traj_id)
                continue
            if run:
                self._execute_statistics_pack(run)
                run = []
            self._execute_statistics_pack([traj_id])
        if run:
            self._execute_statistics_pack(run)
        if held:
            self.bind_rewards(held)

    def wait_for_trajectory_reduction(
        self,
        group_id: int,
        trajectory_id: int,
    ) -> None:
        """Wait for one already-admitted source to leave the host queue.

        This is an explicit diagnostic/replay control. Objective-ready live
        execution never calls it: admission remains unrestricted and the
        reducer continues independently of subsequent ready sources.
        """

        with self._reducer_condition:
            acc = self._groups.get(group_id)
            if acc is None or trajectory_id not in acc.differentiated:
                raise RuntimeError(
                    f"group {group_id}: trajectory {trajectory_id} was not "
                    "admitted before reduction wait"
                )
            while trajectory_id not in acc.reduced:
                self._raise_worker_failure()
                self._reducer_condition.wait()
            self._raise_worker_failure()

    def _combine_statistics_off_device(
        self,
        reward_sums: list[_SourceValue],
        unit_sums: list[_SourceValue],
        mean: float,
    ) -> set[int]:
        """Evaluate ``G1 -= mean * G2`` where the statistics already live.

        Both statistics are model-sized fp32.  Reloading both to the parameter
        device moves twice the bytes the result needs, and the statistic
        device is idle at group closure.  The AXPY is independent per
        parameter, so it runs across the offload worker entitlement; the
        arithmetic and its per-parameter order are unchanged.
        """

        candidates = [
            index
            for index, (reward_sum, unit_sum) in enumerate(
                zip(reward_sums, unit_sums, strict=True)
            )
            if isinstance(reward_sum, torch.Tensor)
            and isinstance(unit_sum, torch.Tensor)
            and reward_sum.device == unit_sum.device
            and index < len(self.params)
            and reward_sum.device != self.params[index].device
        ]
        if not candidates:
            return set()
        started = time.perf_counter()

        def combine(index: int) -> int:
            reward_sum = reward_sums[index]
            self._combine_parameter_statistics(reward_sum, unit_sums[index], mean)
            return reward_sum.numel() * reward_sum.element_size()

        if len(candidates) == 1:
            combined_bytes = combine(candidates[0])
        else:
            with ThreadPoolExecutor(
                max_workers=min(
                    self._source_offload_worker_count, len(candidates)
                )
            ) as pool:
                combined_bytes = sum(pool.map(combine, candidates))
        with self._accounting_lock:
            self._group_close_host_combined_bytes += combined_bytes
            self._group_close_host_combine_s += time.perf_counter() - started
        return set(candidates)

    def _combine_parameter_statistics(
        self, first: torch.Tensor, second: torch.Tensor, mean: float
    ) -> torch.Tensor:
        if getattr(self, "_binary_host_statistics", False):
            return torch.lerp(first, second, mean, out=first)
        return first.add_(second, alpha=-mean)

    def group_reduction_complete(self, group_id: int) -> bool:
        """Non-blocking: may ``close_group`` proceed without waiting?

        True only when every registered trajectory of the group is
        reward-bound and its source already folded. Callers use this to
        defer group closes off the event-ingestion path; a False result is
        never an error, it means "not yet".
        """

        with self._reducer_condition:
            self._raise_worker_failure()
            acc = self._groups.get(group_id)
            if acc is None:
                return False
            expected = self._expected_trajectory_order_by_group.get(group_id)
            if expected is None or set(expected) != acc.closed:
                return False
            return acc.reduced == acc.closed and not acc.finalizing

    def _group_moments(self, acc: _GroupAccum) -> tuple[float, float]:
        """Mean and population std (+ eps) of a closed group's rewards."""

        return self._reward_moments(
            [acc.rewards_by_trajectory[traj_id] for traj_id in sorted(acc.closed)]
        )

    def _reward_moments(self, rewards: list[float]) -> tuple[float, float]:
        """Mean and population std (+ eps), in float64, in trajectory order."""

        r = torch.tensor(rewards, dtype=torch.float64)
        return float(r.mean()), float(r.std(unbiased=False)) + self.eps

    def _finalize_group_statistics(self, acc: _GroupAccum) -> None:
        """Apply the close's host combine as soon as the last source folded.

        Every reward of the group is bound once its last source folds, so
        the mean is final; the combine is the close's own per-parameter
        ``G1.add_(G2, alpha=-mean)`` over the same host tensors, only run
        by the reducer instead of the thread that calls ``close_group``.
        """

        mean, _std = self._group_moments(acc)
        acc.combined_indices = self._combine_statistics_off_device(
            acc.parameter_centered_sum,
            acc.parameter_source_mean,
            mean,
        )
        acc.combined_mean = mean
        with self._accounting_lock:
            self._group_finalized_in_reducer_count += 1

    def _uniform_advantage(
        self, acc: _GroupAccum, mean: float, std: float
    ) -> float | None:
        """The one advantage of a group whose rewards are all equal, else None.

        Then sum_i A_i g_i is ``A * G2``, with ``A = (r - mean) / std`` the
        float64 value the barrier objective multiplies every member's score
        by (the forest objective): exactly 0 when the float64 mean is the reward,
        tiny otherwise. ``(G1 - mean G2) / std`` would instead divide the
        statistics' rounding residual by ``eps``.
        """

        rewards = {acc.rewards_by_trajectory[traj_id] for traj_id in acc.closed}
        if len(rewards) != 1:
            return None
        with self._accounting_lock:
            self._uniform_reward_group_count += 1
        return (rewards.pop() - mean) / std

    def _boundary_adjoints(
        self,
        acc: _GroupAccum,
        boundary: Any,
        mean: float,
        std: float,
        uniform_advantage: float | None,
    ) -> list[torch.Tensor | None]:
        """The prompt boundary's adjoint: held members' statistics plus direct.

        The direct members' ``A_i b_i`` sit in the proxies' gradients, where
        their backwards accumulated them.
        """

        b1, b2 = acc.boundary_centered_sum, acc.boundary_source_mean
        gradients: list[torch.Tensor | None] = [None] * len(boundary.real)
        for index, real in enumerate(boundary.real):
            reward_sum = b1[index] if index < len(b1) else None
            unit_sum = b2[index] if index < len(b2) else None
            if uniform_advantage is not None and unit_sum is not None:
                gradients[index] = torch.mul(
                    self._reload_source(unit_sum, real.device).to(torch.float32),
                    uniform_advantage,
                ).to(real.dtype)
            elif reward_sum is not None:
                reward_sum = self._reload_source(reward_sum, real.device)
                unit_sum = self._reload_source(unit_sum, real.device)
                combined = reward_sum.add_(unit_sum, alpha=-mean)
                gradients[index] = combined.div_(std).to(real.dtype)
            if unit_sum is not None:
                self._release_statistic_pair(b1, b2, index)
            if not acc.direct:
                continue
            proxy = boundary.proxy[index]
            direct = proxy.grad
            proxy.grad = None
            if direct is None:
                continue
            direct = direct.detach().to(real.dtype)
            gradients[index] = (
                direct if gradients[index] is None else gradients[index].add_(direct)
            )
        return gradients

    def close_group(self, group_id: int) -> None:
        """All rewards in: combine the buffers, backward the prompt once.

        With ``defer_group_close``, a group whose last source has not folded
        yet is recorded and the call returns: the close runs at a later safe
        point -- the end of a later prepare or close call, whose backwards
        are then already on the device, another ``close_group``, or
        `assert_safe_to_step`, which waits -- once
        the reducer has folded and combined that group. Requested closes run
        in request order, so the parameter gradients accumulate in the same
        order as closing at once, and none runs between a pack's rebuild and
        its backwards, whose memory rules read the device before rebuilding.
        """
        self._check_versions()
        acc = self._groups.get(group_id)
        if acc is None:
            raise RuntimeError(f"group {group_id}: no closed trajectories")
        open_trajs = [
            t for t in self.run.trajectories_of(group_id) if t not in acc.closed
        ]
        if open_trajs:
            raise RuntimeError(
                f"group {group_id}: trajectories {open_trajs} not closed"
            )
        with self._reducer_condition:
            expected_order = self._expected_trajectory_order_by_group.get(
                group_id
            )
            if expected_order is None or set(expected_order) != acc.closed:
                raise RuntimeError(
                    f"group {group_id}: registered trajectory membership "
                    "differs from closed trajectories"
                )
            if group_id in self._requested_group_closes:
                raise RuntimeError(f"group {group_id}: close already requested")
            if self._defer_group_close:
                self._requested_group_closes.append(group_id)
        if self._defer_group_close:
            self._close_requested_groups(wait=False)
            with self._reducer_condition:
                deferred = group_id in self._requested_group_closes
            if deferred:
                with self._accounting_lock:
                    self._group_close_deferred_count += 1
            return
        self._complete_group_close(group_id, acc)

    def _close_requested_groups(self, *, wait: bool) -> None:
        """Run requested closes in request order; ``wait=False`` stops at
        the first whose last source is still folding."""

        while True:
            with self._reducer_condition:
                self._raise_worker_failure()
                if not self._requested_group_closes:
                    return
                group_id = self._requested_group_closes[0]
                acc = self._groups[group_id]
                if not wait and (acc.reduced != acc.closed or acc.finalizing):
                    return
                self._requested_group_closes.popleft()
            self._complete_group_close(group_id, acc)

    def _complete_group_close(self, group_id: int, acc: _GroupAccum) -> None:
        with self._reducer_condition:
            barrier_started = time.perf_counter()
            # A pipelined last fold hands parameters over one by one (the
            # loop below waits for each); otherwise the whole reduction first.
            while not acc.pipelined and (
                acc.reduced != acc.closed or acc.finalizing
            ):
                self._raise_worker_failure()
                self._reducer_condition.wait()
            self._group_close_barrier_s += time.perf_counter() - barrier_started
            self._raise_worker_failure()
            pipelined = acc.pipelined

        mean, std = self._group_moments(acc)
        if acc.direct_moments is not None and acc.direct_moments != (mean, std):
            raise RuntimeError(
                f"group {group_id}: the direct path's moments differ from "
                "the close's"
            )
        uniform_advantage = self._uniform_advantage(acc, mean, std)

        # Objective-ready execution admits and folds sources in legal readiness
        # order before closure. A replay of the same recorded readiness order
        # reproduces the same arithmetic order.
        g1, g2 = acc.parameter_centered_sum, acc.parameter_source_mean

        # Evaluate (G1 - mean*G2) / std. The primary buffer is consumed in place.
        parameter_source = ("reward_linear_group", group_id)
        if acc.combined_mean is not None:
            if acc.combined_mean != mean:
                raise RuntimeError(
                    f"group {group_id}: the reducer combined with a mean "
                    "that differs from the close's"
                )
            combined_off_device = acc.combined_indices
        else:
            combined_off_device = self._combine_statistics_off_device(
                g1, g2, mean
            )
        # A pooled device home is reused by the next group, while the
        # combined value may become the parameter's gradient itself; the
        # close writes it out of place there (the same kernel, so the same
        # values) instead of into the home.
        device_home = (
            acc.statistic_slot is not None
            and self._device_statistic_pool is not None
        )
        parameter_started = time.perf_counter()
        for index in range(len(g1)):
            if pipelined:
                self._wait_parameter_ready(acc, index)
            reward_sum, unit_sum = g1[index], g2[index]
            if uniform_advantage is not None:
                # A uniform declared binary group has exactly zero advantage,
                # so its alternative second statistic is never used as G2.
                if unit_sum is not None and uniform_advantage != 0.0:
                    unit_value = self._reload_source(
                        unit_sum, self.params[index].device
                    ).to(torch.float32)
                    self.run.accumulate_parameter_adjoint(
                        self.params[index],
                        torch.mul(unit_value, uniform_advantage),
                        source=parameter_source,
                    )
                self._release_statistic_pair(
                    acc.parameter_centered_sum,
                    acc.parameter_source_mean,
                    index,
                )
                continue
            if reward_sum is None and unit_sum is not None:
                if mean != 0.0:
                    unit_value = self._reload_source(
                        unit_sum,
                        self.params[index].device,
                    ).to(torch.float32)
                    combined = (
                        torch.mul(unit_value, -mean)
                        if device_home
                        else unit_value.mul_(-mean)
                    ).div_(std)
                    self.run.accumulate_parameter_adjoint(
                        self.params[index],
                        combined,
                        source=parameter_source,
                    )
                self._release_statistic_pair(
                    acc.parameter_centered_sum,
                    acc.parameter_source_mean,
                    index,
                )
                continue
            if reward_sum is not None:
                # Evaluate the AXPY where the statistics already live.  Both
                # statistics are model-sized fp32; reloading both moves twice
                # the bytes the result needs, and the statistic device is idle
                # at group closure.  The arithmetic and its order are the same.
                if index in combined_off_device:
                    combined = self._reload_combined(
                        acc.statistic_slot,
                        index,
                        reward_sum,
                        self.params[index].device,
                    ).div_(std)
                    adjoint_started = time.perf_counter()
                    self.run.accumulate_parameter_adjoint(
                        self.params[index],
                        combined,
                        source=parameter_source,
                    )
                    release_started = time.perf_counter()
                    self._release_statistic_pair(
                        acc.parameter_centered_sum,
                        acc.parameter_source_mean,
                        index,
                    )
                    finished = time.perf_counter()
                    with self._accounting_lock:
                        self._group_close_adjoint_s += (
                            release_started - adjoint_started
                        )
                        self._group_close_release_s += finished - release_started
                    continue
                reward_sum = self._reload_source(
                    reward_sum,
                    self.params[index].device,
                )
                unit_sum = self._reload_source(
                    unit_sum,
                    self.params[index].device,
                )
                combined = (
                    torch.add(reward_sum, unit_sum, alpha=-mean)
                    if device_home
                    else self._combine_parameter_statistics(reward_sum, unit_sum, mean)
                ).div_(std)
                self.run.accumulate_parameter_adjoint(
                    self.params[index],
                    combined,
                    source=parameter_source,
                )
                self._release_statistic_pair(
                    acc.parameter_centered_sum,
                    acc.parameter_source_mean,
                    index,
                )
        if acc.statistic_slot is not None:
            if self._statistic_pool is None:
                raise RuntimeError("a statistic slot is held without a device statistic pool")
            # The next group that takes this home first completes every
            # reload copy enqueued above.
            with self._reducer_condition:
                self._statistic_slots.pop(group_id, None)
            self._statistic_pool.release(
                acc.statistic_slot, reload_event=self._current_stream_event()
            )
            acc.statistic_slot = None
        with self._reducer_condition:
            # The boundary statistics fold after the parameters'.
            while acc.reduced != acc.closed or acc.finalizing:
                self._raise_worker_failure()
                self._reducer_condition.wait()
            self._raise_worker_failure()
            self._groups.pop(group_id)
            self._closed_group_ids.add(group_id)
        if acc.differentiated != acc.closed:
            raise RuntimeError(
                f"group {group_id}: ready trajectories were not differentiated"
            )
        if acc.reduced != acc.closed:
            raise RuntimeError(
                f"group {group_id}: differentiated sources were not reduced"
            )
        if acc.statistic_count != len(acc.closed - acc.direct):
            raise RuntimeError(
                f"group {group_id}: statistic count differs"
            )
        group_source_bytes = self._source_bytes(
            acc.parameter_sources
        ) + self._source_bytes(acc.boundary_sources)
        boundary_statistics_started = time.perf_counter()
        # boundary: one backward through the shared prompt subgraph, total.
        boundary = self.run.boundary(group_id)
        boundary_gradients = self._boundary_adjoints(
            acc, boundary, mean, std, uniform_advantage
        )
        source = ("group", group_id)
        boundary_started = time.perf_counter()
        with self._accounting_lock:
            self._group_close_parameter_s += (
                boundary_statistics_started - parameter_started
            )
            self._group_close_boundary_statistics_s += (
                boundary_started - boundary_statistics_started
            )
        boundary_device = self._device_span_start(self._parameter_device())
        with operator_range("group_close_boundary", fields={"group": group_id}):
            self.run.finalize_group_boundary(
                group_id,
                boundary_gradients,
                source=source,
            )
            if self.run.shared_prefix_vjp_mode == "source_ready_serial":
                self.run.drain_shared_prefix_source(group_id, source=source)
        self._device_span_end(
            boundary_device, self._group_close_boundary_events
        )
        free_started = time.perf_counter()

        self.run.free_group(group_id)
        with self._accounting_lock:
            self._group_close_boundary_s += free_started - boundary_started
            self._group_close_free_s += time.perf_counter() - free_started
        with self._accounting_lock:
            self._current_source_bytes -= group_source_bytes
            if self._current_source_bytes < 0:
                raise RuntimeError("reward-linear source residency accounting underflow")

    @property
    def open_groups(self) -> list[int]:
        """Groups whose retained graphs still exist -- open in the RUN, not
        merely groups that have started closing. A group with zero closed
        trajectories is the most dangerous case, not an exempt one."""
        return sorted(self.run.open_group_ids())

    @property
    def pending_reduction_source_count(self) -> int:
        """Differentiated-but-unfolded trajectory sources currently resident.

        Each one is a full-model gradient waiting (in HBM or pinned host
        memory, by offload mode) for its fold; admission budgets gate on
        this count because the fold is the moment the memory actually
        frees.
        """

        with self._reducer_condition:
            return sum(
                len(group.differentiated - group.reduced)
                for group in self._groups.values()
            )

    @property
    def held_source_count(self) -> int:
        """Unfolded sources the residency budget must count.

        With a slab arena, a source whose reward is bound folds and frees its
        slab without any further admission, and an admission that finds no
        free slab waits in ``acquire`` for that fold; what can hold memory
        indefinitely is a source still awaiting its reward. Without an arena
        nothing else bounds residency, so every unfolded source counts.
        """

        with self._reducer_condition:
            if self._pinned_arena is None:
                return sum(
                    len(group.differentiated - group.reduced)
                    for group in self._groups.values()
                )
            return sum(
                len(group.differentiated - group.closed)
                for group in self._groups.values()
            )

    def assert_safe_to_step(self) -> None:
        """Call before optimizer.step(): stepping with open groups corrupts them."""
        self._statistics_branch_observer = None
        self._close_requested_groups(wait=True)
        if self.open_groups:
            raise RuntimeError(
                f"optimizer step with open groups {self.open_groups}: their "
                "retained graphs reference current parameter values and would "
                "be silently invalidated"
            )
        self._shutdown_workers()
        with self._accounting_lock:
            if self._source_offload_inflight_bytes != 0:
                raise RuntimeError(
                    "source offload events remained pending after reducer shutdown"
                )
        leaked = self._release_all_source_slabs()
        if leaked:
            raise RuntimeError(
                f"{leaked} pinned source slab(s) were still held after every "
                "group closed"
            )
        leaked = self._release_statistic_slots()
        if leaked:
            raise RuntimeError(
                f"{leaked} statistic home(s) were still held after every "
                "group closed"
            )
        if self._device_statistic_pool is not None:
            # The step runs with the homes' memory back in the allocator's
            # cache; the next batch's executor allocates its own.
            self._device_statistic_pool.close()
        self.run.finalize_shared_prefixes()

    def abort(self) -> None:
        """Stop background work and release retained graphs after a failed run."""
        self._statistics_branch_observer = None
        failure: BaseException | None = None
        try:
            self._shutdown_workers()
        except BaseException as error:
            failure = error
        try:
            self._release_all_source_slabs()
        except BaseException as error:
            failure = failure or error
        try:
            self._release_statistic_slots()
        except BaseException as error:
            failure = failure or error
        with self._reducer_condition:
            self._requested_group_closes.clear()
            self._offload_queue.clear()
            self._reduction_ready_queue.clear()
            self._device_source_bytes.clear()
            self._streaming_pending_tensors.clear()
            self._streaming_capture_complete.clear()
            self._streaming_staged_bytes_by_trajectory.clear()
            self._active_offload_tensor_jobs_by_trajectory.clear()
            self._active_hbm_stage_tensor_jobs_by_trajectory.clear()
            self._active_hbm_stage_tensor_jobs = 0
            self._active_hbm_stage_tensor_jobs_by_group.clear()
            for accumulator in self._groups.values():
                accumulator.parameter_sources.clear()
                accumulator.boundary_sources.clear()
                accumulator.parameter_centered_sum.clear()
                accumulator.parameter_source_mean.clear()
                accumulator.boundary_centered_sum.clear()
                accumulator.boundary_source_mean.clear()
            self._groups.clear()
            self._expected_trajectory_order_by_group.clear()
        cleanup_failure: BaseException | None = None
        for group_id in list(self.run.open_group_ids()):
            try:
                for trajectory_id in self.run.trajectories_of(group_id):
                    self.run.free_trajectory(trajectory_id)
                self.run.free_group(group_id)
            except BaseException as error:
                cleanup_failure = cleanup_failure or error
        self._streaming_pinned_buffers.clear()
        self._streaming_fold_buffers.clear()
        self._offload_streams.clear()
        if self._device_statistic_pool is not None:
            self._device_statistic_pool.close()
        with self._accounting_lock:
            self._current_source_bytes = 0
            self._current_statistic_bytes = 0
            self._current_pageable_stage_bytes = 0
            self._source_offload_inflight_bytes = 0
            self._reducer_executing_source_bytes = 0
        if failure is not None:
            raise failure
        if cleanup_failure is not None:
            raise cleanup_failure

    # ---------------------------------------------------------------- internal

    def _differentiate_source_pack(
        self,
        trajectory_ids: list[int],
        *,
        publish_completed: bool = False,
    ) -> list[
        tuple[
            int,
            list[torch.Tensor | None],
            list[torch.Tensor | None],
            Any | None,
        ]
    ]:
        """Execute objective-ready branch backwards in the supplied order."""

        if self._close_profiler is None or not trajectory_ids:
            return self._differentiate_pack(trajectory_ids, publish_completed=publish_completed)
        if not self._close_profile_start(trajectory_ids):
            return self._differentiate_pack(trajectory_ids, publish_completed=publish_completed)
        try:
            return self._differentiate_pack(trajectory_ids, publish_completed=publish_completed)
        finally:
            self._close_profiler.stop()

    def _differentiate_pack(
        self,
        trajectory_ids: list[int],
        *,
        publish_completed: bool = False,
    ) -> list[
        tuple[
            int,
            list[torch.Tensor | None],
            list[torch.Tensor | None],
            Any | None,
        ]
    ]:
        if not trajectory_ids:
            return []
        cohort = self._known_reward_cohort(trajectory_ids)
        drained = False
        for trajectory_id in trajectory_ids:
            # Drop the retained event graph before rebuilding this branch.  The
            # detached tokens and old-logprobs needed by the rebuild survive.
            self.run.free_trajectory(trajectory_id)
        trace = self._close_trace_pack(trajectory_ids)
        rebuild_started = time.perf_counter()
        rebuild_device = self._device_span_start(self._parameter_device())
        # Earlier sources still copying off the device leave it before this
        # pack's first backward (the drain below), so the block rebuild's
        # memory rule may count them free for the backward, not the rebuild.
        logprobs, drained = (
            self._rebuild_pack(trajectory_ids, joint_backward=True)
            if cohort else self._rebuild_pack(trajectory_ids)
        )
        if cohort:
            self._aggregate_cohort_logprobs(logprobs, trajectory_ids)
            record = {"group_id": self.run.group_of(trajectory_ids[0]),
                      "representative": trajectory_ids[0], "members": list(trajectory_ids),
                      "reward": cohort[0], "representation": "aggregate source plus logical members"}
            with self._accounting_lock:
                self._known_reward_cohorts.append(record)
            if trace is not None:
                trace["known_reward_cohort"] = record
        self._device_span_end(rebuild_device, self._branch_rebuild_events)
        # The plain rebuild's memory rule allows one full gradient beside the
        # pack's activations, so in a plain pack each branch's backward waits
        # for the previous branch's source to leave the device; a block pack
        # keeps its branches' copies overlapping the next backward.
        rung = self.run.last_rebuild_rung or "blocked"
        plain_pack = len(trajectory_ids) > 1 and rung == "plain"
        if trace is not None:
            trace["rebuild_host_s"] = time.perf_counter() - rebuild_started
            trace["_rebuild_span"] = (
                self._branch_rebuild_events[-1]
                if rebuild_device is not None
                else None
            )
            trace["rungs"].append(rung)
        with self._accounting_lock:
            self._branch_rebuild_s += time.perf_counter() - rebuild_started
            self._branch_rebuild_count += 1
            self._executed_rebuild_pack_max = max(
                self._executed_rebuild_pack_max, len(trajectory_ids)
            )

        results = []
        direct_pending: dict[int, list[bool]] = {}
        position = 0
        while position < len(trajectory_ids):
            trajectory_id = trajectory_ids[position]
            group_id = self.run.group_of(trajectory_id)
            acc = self._groups[group_id]
            direct = trajectory_id in acc.direct
            if self.source_offload_mode == "pinned_streaming":
                # Publish readiness order before the empty-score branch and
                # before autograd hooks can enqueue tensor work.  A trajectory
                # with no scored tokens still has to advance the ordered
                # reducer instead of waiting forever at group closure.
                with self._reducer_condition:
                    self._source_execution_order_by_group.setdefault(
                        group_id, []
                    ).append(trajectory_id)
                    self._source_preparation_order_by_group.setdefault(
                        group_id, []
                    ).append(trajectory_id)
            lp = logprobs[trajectory_id]
            boundary = self.run.boundary(group_id)
            if not lp.numel():
                parameter_sources = [None for _ in self.params]
                boundary_sources = [None for _ in boundary.proxy]
            else:
                weight = acc.token_weights_by_trajectory[trajectory_id]
                score = (lp * weight).sum() if weight is not None else lp.sum()
                drain_started = time.perf_counter()
                if not drained or plain_pack:
                    # The pack's rebuild overlapped earlier packs' copies;
                    # its backward writes a full gradient only once they
                    # have left the device.
                    self._drain_earlier_source_copies()
                    self._drain_folding_device_sources()
                    drained = True
                backward_started = time.perf_counter()
                backward_device = self._device_span_start(score.device)
                grad_end = None
                grad_host_s = None
                if self.source_offload_mode == "pinned_streaming":
                    parameter_sources, boundary_sources = (
                        self._differentiate_streaming_source(
                            score=score,
                            group_id=group_id,
                            trajectory_id=trajectory_id,
                            boundary_proxy=list(boundary.proxy),
                        )
                    )
                else:
                    gradients = None
                    slab_sources = None
                    next_rung = self._backward_fallback_rung(rung)
                    taken = (
                        self._take_free_slab(
                            (group_id, trajectory_id), list(boundary.proxy)
                        )
                        if self._pinned_arena is not None
                        and self._source_copy_during_backward
                        and not direct
                        else None
                    )
                    if direct:
                        gradients = (
                            ()
                            if self._backward_direct(
                                score,
                                acc,
                                trajectory_id,
                                list(boundary.proxy),
                                direct_pending.setdefault(trajectory_id, []),
                                raise_out_of_memory=next_rung is None,
                            )
                            else None
                        )
                    elif taken is not None:
                        # The source's copy into its slab starts inside
                        # this backward; the gradients are the same.
                        slab_sources = self._differentiate_into_slab(
                            score,
                            (group_id, trajectory_id),
                            list(boundary.proxy),
                            taken,
                            raise_out_of_memory=next_rung is None,
                        )
                        gradients = None if slab_sources is None else ()
                    else:
                        try:
                            gradients = torch.autograd.grad(
                                score,
                                [*self.params, *boundary.proxy],
                                retain_graph=False,
                                allow_unused=True,
                            )
                        except torch.OutOfMemoryError:
                            if next_rung is None:
                                raise
                    if trace is not None and gradients is not None:
                        grad_host_s = time.perf_counter() - backward_started
                        if backward_device is not None:
                            grad_end = torch.cuda.Event(enable_timing=True)
                            grad_end.record()
                    if gradients is None:
                        # Out of device memory in this close's backward.
                        # Release this and the pack's later branches' graphs
                        # (the exception, and with it the failed walk, is
                        # gone here), rebuild them one rung down the ladder
                        # and backward this branch again. The autograd walk
                        # has no side effect, so the source is the one the
                        # lower rung computes.
                        score = lp = None
                        remaining = trajectory_ids[position:]
                        for remaining_id in remaining:
                            logprobs[remaining_id] = None
                        rebuilt, rung = (
                            self._rebuild_on_rung(remaining, next_rung, joint_backward=True)
                            if cohort else self._rebuild_on_rung(remaining, next_rung)
                        )
                        if cohort:
                            self._aggregate_cohort_logprobs(rebuilt, remaining)
                        logprobs.update(rebuilt)
                        if trace is not None:
                            trace["out_of_memory_retries"].append(
                                trajectory_id
                            )
                            trace["rungs"].append(rung)
                        plain_pack = False
                        # Retry with every earlier source off the device.
                        drained = False
                        continue
                    parameter_count = len(self.params)
                    if direct:
                        parameter_sources = boundary_sources = []
                    elif slab_sources is not None:
                        parameter_sources, boundary_sources = slab_sources
                    elif self._pinned_arena is not None:
                        parameter_sources, boundary_sources = (
                            self._store_sources_in_slab(
                                (group_id, trajectory_id),
                                list(gradients[:parameter_count]),
                                list(gradients[parameter_count:]),
                            )
                        )
                    else:
                        parameter_sources = self._store_sources(
                            list(gradients[:parameter_count])
                        )
                        boundary_sources = self._store_sources(
                            list(gradients[parameter_count:])
                        )
                    # The stored transfers hold the device tensors until
                    # their copies land; this frame must not hold them past
                    # the next branch's backward.
                    gradients = None
                self._device_span_end(
                    backward_device, self._branch_backward_events
                )
                if trace is not None:
                    backward_span = (
                        self._branch_backward_events[-1]
                        if backward_device is not None
                        else None
                    )
                    trace["branches"].append(
                        {
                            "trajectory_id": trajectory_id,
                            "path": "direct" if direct else "statistics",
                            "scored_tokens": int(lp.numel()),
                            "drain_s": backward_started - drain_started,
                            "started_at": backward_started,
                            "backward_host_s": (
                                time.perf_counter() - backward_started
                            ),
                            "grad_host_s": grad_host_s,
                            "_backward_span": backward_span,
                            "_grad_span": (
                                None
                                if grad_end is None
                                else (backward_device, grad_end)
                            ),
                        }
                    )
                with self._accounting_lock:
                    self._branch_backward_s += (
                        time.perf_counter() - backward_started
                    )
                    self._branch_backward_count += 1
            if direct:
                with self._reducer_condition:
                    acc.differentiated.add(trajectory_id)
                    acc.reduced.add(trajectory_id)
                    for order in (
                        self._source_preparation_order_by_group,
                        self._source_reduction_order_by_group,
                    ):
                        order.setdefault(group_id, []).append(trajectory_id)
                    self._reducer_condition.notify_all()
                position += 1
                continue
            source_ready_event = None
            if self.source_offload_mode == "device_resident":
                device = self._parameter_device()
                if device is not None and device.type == "cuda":
                    # Sources stay on this thread's current stream.  The
                    # offload worker completes this event before marking the
                    # trajectory materialized, so a reducer on any stream may
                    # consume the tensors afterwards.
                    source_ready_event = torch.cuda.Event()
                    source_ready_event.record(
                        torch.cuda.current_stream(device)
                    )
            with self._reducer_condition:
                acc.differentiated.add(trajectory_id)
                if self.source_offload_mode != "pinned_streaming":
                    # The objective-readiness (reduction) order is recorded at
                    # reward binding, not here: differentiation may legally
                    # run at content readiness, before the reward exists.
                    self._source_preparation_order_by_group.setdefault(
                        group_id, []
                    ).append(trajectory_id)
            source = (trajectory_id, parameter_sources, boundary_sources, source_ready_event)
            if publish_completed:
                self._publish_statistics_source(*source)
                self._completed_source_headers += 1
            else:
                results.append(source)
            position += 1
            if self._statistics_branch_observer is not None:
                self._statistics_branch_observer(trajectory_id)
        self._close_trace_finish(trace)
        return results

    def _known_reward_cohort(self, trajectory_ids: list[int]) -> tuple[float] | None:
        """A common landed reward for an already-ready pack, never a future one."""
        if not self._combine_known_reward_sources or len(trajectory_ids) < 2:
            return None
        group_ids = {self.run.group_of(t) for t in trajectory_ids}
        if len(group_ids) != 1:
            return None
        with self._reducer_condition:
            acc = self._groups[next(iter(group_ids))]
            if any(t in acc.direct or t not in acc.verdicts
                   or acc.token_weights_by_trajectory.get(t) is not None for t in trajectory_ids):
                return None
            rewards = {acc.verdicts[t] for t in trajectory_ids}
            return (next(iter(rewards)),) if len(rewards) == 1 else None

    @staticmethod
    def _aggregate_cohort_logprobs(logprobs, trajectory_ids) -> None:
        # The representative carries the complete aggregate. Empty entries
        # cover the other logical members for the existing closure/reducer
        # bookkeeping; they do not claim an individual zero gradient.
        pieces = [logprobs[t] for t in trajectory_ids]
        logprobs[trajectory_ids[0]] = torch.cat(pieces)
        for trajectory_id in trajectory_ids[1:]:
            logprobs[trajectory_id] = pieces[0].new_empty(0)

    def _backward_direct(
        self,
        score: torch.Tensor,
        acc: _GroupAccum,
        trajectory_id: int,
        boundary_proxy: list[torch.Tensor],
        pending: list[bool],
        *,
        raise_out_of_memory: bool,
    ) -> bool:
        """Backward ``A_i * S_i`` into the parameters' and proxies' gradients.

        ``A_i`` is the group's advantage from ``direct_moments``, the moments
        the close computes. Autograd sums a leaf's contributions before its
        one accumulation into ``.grad``, so a walk that runs out of device
        memory leaves each input either fully accumulated or untouched:
        ``pending`` marks the untouched ones, and the retry (one rung down)
        walks for those alone. Returns False after such an out-of-memory
        walk, unless ``raise_out_of_memory``.
        """

        if self.run._parameter_adjoint_capture_enabled:
            raise RuntimeError(
                "the direct path accumulates into the parameters' gradients, "
                "which this run's parameter-adjoint capture owns"
            )
        if acc.direct_moments is None:
            raise RuntimeError(f"trajectory {trajectory_id}'s group has no direct-path moments")
        mean, std = acc.direct_moments
        advantage = (acc.rewards_by_trajectory[trajectory_id] - mean) / std
        inputs = [*self.params, *boundary_proxy]
        if not pending:
            pending.extend([True] * len(inputs))
        targets = [index for index, todo in enumerate(pending) if todo]
        if not targets:
            return True

        def install(index: int) -> Any:
            def accumulated(_leaf: torch.Tensor) -> None:
                if not pending[index]:
                    raise RuntimeError(
                        "a direct-path input accumulated twice in one walk"
                    )
                pending[index] = False

            return inputs[index].register_post_accumulate_grad_hook(accumulated)

        handles = [install(index) for index in targets]
        started = time.perf_counter()
        try:
            torch.autograd.backward(
                score,
                grad_tensors=torch.full_like(score, advantage),
                inputs=[inputs[index] for index in targets],
            )
        except torch.OutOfMemoryError:
            if raise_out_of_memory:
                raise
            return False
        finally:
            for handle in handles:
                handle.remove()
        with self._accounting_lock:
            self._direct_backward_count += 1
            self._direct_backward_s += time.perf_counter() - started
        return True

    def _rebuild_pack(
        self, trajectory_ids: list[int], *, joint_backward: bool = False
    ) -> tuple[dict[int, torch.Tensor], bool]:
        """The pack's close-time rebuild, and whether earlier sources drained.

        The rebuild runs beside earlier sources whose copies off the device
        have not landed. When it runs out of device memory anyway (the block
        rule's estimate leaves the output head's forward blocks to the
        largest gradient's spare), its partial graph is released with the
        exception, every earlier source leaves the device, and the pack is
        rebuilt on the block path, its rule reading the memory now free; if
        that runs out too, with every checkpoint input on the host. Every
        rung computes the same logprobs, so the update is the one the lower
        rung gives. At the bottom of the ladder the error surfaces.
        """

        configure_window = getattr(self.run, "set_source_gradient_window", None)
        if configure_window is not None:
            configure_window(None)
            if self._source_release_during_backward and not any(
                trajectory_id in self._groups[self.run.group_of(trajectory_id)].direct
                for trajectory_id in trajectory_ids
            ):
                arena = self._pinned_arena.report()
                free = arena["slabs"] - arena["busy_slabs"]
                capacities = arena["slab_capacities"]
                parameter_sizes = [parameter.numel() * parameter.element_size() for parameter in self.params]
                sizes = [tensor_offsets([*parameter_sizes, *(
                    value.numel() * value.element_size()
                    for value in self.run.boundary(self.run.group_of(trajectory_id)).proxy
                )])[1] for trajectory_id in trajectory_ids]
                # Only the foreground executor acquires these slabs; workers
                # release them. Do not assume a copy window on a direct walk
                # or when a busy arena can force copy-after-backward.
                if free >= len(trajectory_ids) and min(capacities, default=0) >= max(sizes, default=0):
                    configure_window(self._source_offload_window_bytes)
        options = {"joint_backward": True} if joint_backward else {}
        try:
            return (
                self.run.rebuild_logprobs(
                    trajectory_ids,
                    releasing_bytes=self._device_bytes_awaiting_copy,
                    **options,
                ),
                False,
            )
        except torch.OutOfMemoryError as error:
            with self._accounting_lock:
                self._rebuild_out_of_memory_first_attempts += 1
            logger.debug(
                "close rebuild of %d trajectories ran out of device memory; "
                "retrying on the blocked rung: %s",
                len(trajectory_ids),
                str(error),
            )
        rung: str | None = "blocked"
        while True:
            # The failed rebuild's graph went with the exception above.
            self._drain_earlier_source_copies()
            self._drain_folding_device_sources()
            try:
                if rung == "blocked":
                    logprobs = self.run.rebuild_logprobs(
                        trajectory_ids,
                        executor="blocked",
                        releasing_bytes=self._device_bytes_awaiting_copy,
                        **options,
                    )
                else:
                    logprobs = self.run.rebuild_logprobs(
                        trajectory_ids,
                        executor="blocked",
                        checkpoint_input_storage_device="cpu",
                        **options,
                    )
                landed = self.run.last_rebuild_rung or rung
                with self._accounting_lock:
                    self._rebuild_out_of_memory_retries[landed] = (
                        self._rebuild_out_of_memory_retries.get(landed, 0)
                        + len(trajectory_ids)
                    )
                return logprobs, True
            except torch.OutOfMemoryError:
                rung = self._rebuild_fallback_rung(rung)
                if rung is None:
                    raise

    def _rebuild_fallback_rung(self, rung: str) -> str | None:
        """The rung below a close-time rebuild that ran out of memory."""

        if rung == "blocked":
            storage = self.run.rebuild_checkpoint_storage_device
            if storage is None or storage.type != "cpu":
                return "blocked_host_checkpoints"
        return None

    def _backward_fallback_rung(self, rung: str) -> str | None:
        """The next rung down a close's memory ladder, or None at the bottom.

        plain -> block rebuild with device checkpoint inputs (or the
        leading blocks' on the host, as the rebuild's memory rule places
        them) -> block rebuild with every checkpoint input on the host.
        Every rung computes the same source;
        they differ in the device memory the backward needs. Streaming
        sources consume the reward inside their backward, so they never
        retry.
        """

        if self.source_offload_mode == "pinned_streaming":
            return None
        if rung == "plain":
            return "blocked"
        if rung == "blocked_partial_host_checkpoints":
            return "blocked_host_checkpoints"
        if rung == "blocked":
            storage = self.run.rebuild_checkpoint_storage_device
            if storage is None or storage.type != "cpu":
                return "blocked_host_checkpoints"
        return None

    def _rebuild_on_rung(
        self, trajectory_ids: list[int], rung: str, *, joint_backward: bool = False
    ) -> tuple[dict[int, torch.Tensor], str]:
        """Rebuild branches whose backward ran out of memory, one rung down.

        Returns the logprobs and the rung they were rebuilt on: a block
        rebuild's memory rule may place the inputs on the host at once.
        """

        started = time.perf_counter()
        options = {"joint_backward": True} if joint_backward else {}
        if rung == "blocked":
            logprobs = self.run.rebuild_logprobs(
                trajectory_ids,
                executor="blocked",
                releasing_bytes=self._device_bytes_awaiting_copy,
                **options,
            )
        elif rung == "blocked_host_checkpoints":
            logprobs = self.run.rebuild_logprobs(
                trajectory_ids,
                executor="blocked",
                checkpoint_input_storage_device="cpu",
                **options,
            )
        else:
            raise ValueError(f"unknown rebuild rung {rung!r}")
        landed = self.run.last_rebuild_rung or rung
        with self._accounting_lock:
            self._branch_rebuild_s += time.perf_counter() - started
            self._backward_out_of_memory_retries[landed] = (
                self._backward_out_of_memory_retries.get(landed, 0)
                + len(trajectory_ids)
            )
        return logprobs, landed

    def _device_bytes_awaiting_copy(self) -> int:
        """Device bytes of sources whose copy to the host has not landed,
        and of reward-bound sources whose device fold has not been enqueued.

        Exactly what `_drain_earlier_source_copies` and
        `_drain_folding_device_sources` release before a pack's first
        backward.
        """

        with self._reducer_condition:
            copying = sum(
                value.bytes
                for group in self._groups.values()
                for sources in (
                    *group.parameter_sources.values(),
                    *group.boundary_sources.values(),
                )
                for value in sources
                if isinstance(value, _PendingCpuTransfer)
                and value.source_tensor is not None
            )
        return copying + self._bound_device_source_bytes()

    def _publish_statistics_source(
        self,
        trajectory_id: int,
        parameter_sources: list[_SourceValue],
        boundary_sources: list[_SourceValue],
        source_ready_event: Any | None,
    ) -> None:
        """Let the copy drain observe one successfully completed branch."""

        group_id = self.run.group_of(trajectory_id)
        acc = self._groups[group_id]
        # The reward is deliberately not read here: content-ready
        # preparation must stay reward-free, and the fold reads the
        # scalar from the accumulator at reduction time.
        with self._reducer_condition:
            self._raise_worker_failure()
            acc.parameter_sources[trajectory_id] = parameter_sources
            acc.boundary_sources[trajectory_id] = boundary_sources
            self._source_queued_at[(group_id, trajectory_id)] = (
                time.perf_counter()
            )
            if self.source_offload_mode == "pinned_streaming":
                key = (group_id, trajectory_id)
                self._streaming_capture_complete.add(key)
                self._mark_streaming_source_ready_locked(key)
            else:
                if self.source_offload_mode == "device_resident":
                    device_bytes = sum(
                        value.numel() * value.element_size()
                        for value in (*parameter_sources, *boundary_sources)
                        if isinstance(value, torch.Tensor)
                        and value.device.type == "cuda"
                    )
                    if device_bytes:
                        self._device_source_bytes[
                            (group_id, trajectory_id)
                        ] = device_bytes
                self._offload_queue.append(
                    (
                        group_id,
                        trajectory_id,
                        parameter_sources,
                        boundary_sources,
                        source_ready_event,
                    )
                )
                self._offload_pending_trajectories_peak = max(
                    self._offload_pending_trajectories_peak,
                    len(self._offload_queue),
                )
            self._pending_reduction_trajectories_peak = max(
                self._pending_reduction_trajectories_peak,
                sum(
                    len(group.differentiated - group.reduced)
                    for group in self._groups.values()
                ),
            )
            self._reducer_condition.notify_all()

    def _execute_statistics_pack(
        self,
        trajectory_ids: list[int],
    ) -> None:
        for (
            trajectory_id,
            parameter_sources,
            boundary_sources,
            source_ready_event,
        ) in (
            self._differentiate_source_pack(
                trajectory_ids,
                publish_completed=self.source_offload_mode == "pinned_nonblocking",
            )
        ):
            self._publish_statistics_source(
                trajectory_id, parameter_sources, boundary_sources, source_ready_event
            )
        # The pack's backwards are on the device: a requested close whose
        # reduction has finished runs now, behind them and before the next
        # pack's memory rules read the device.
        self._close_requested_groups(wait=False)

    def _next_reduction_job(
        self,
    ) -> tuple[
        int,
        int,
        float,
        float,
        list[_SourceValue],
        list[_SourceValue],
        Any | None,
    ] | _HostFoldTask | _HostCombineTask | None:
        if self._host_fold_tasks:
            return self._host_fold_tasks.popleft()

        def is_next_in_group(group_id: int, trajectory_id: int) -> bool:
            if group_id in self._groups_reducing:
                return False
            acc = self._groups.get(group_id)
            execution_order = self._source_execution_order_by_group.get(group_id)
            if acc is None or execution_order is None:
                return False
            held_order = [t for t in execution_order if t not in acc.direct]
            next_index = len(acc.reduced - acc.direct)
            return (
                next_index < len(held_order)
                and held_order[next_index] == trajectory_id
            )

        selected_index = next(
            (
                index
                for index, (group_id, trajectory_id, _ready_event) in enumerate(
                    self._reduction_ready_queue
                )
                if is_next_in_group(group_id, trajectory_id)
            ),
            None,
        )
        if selected_index is None:
            return self._next_sealed_statistics_job()
        group_id, trajectory_id, source_ready_event = (
            self._reduction_ready_queue[selected_index]
        )
        del self._reduction_ready_queue[selected_index]
        self._groups_reducing.add(group_id)
        acc = self._groups.get(group_id)
        order = self._expected_trajectory_order_by_group.get(group_id)
        if acc is None or order is None or trajectory_id not in order:
            raise RuntimeError(
                f"group {group_id}: reward-linear reduction requires registered "
                "trajectory membership"
            )
        if (
            trajectory_id not in acc.parameter_sources
            or trajectory_id not in acc.boundary_sources
        ):
            raise RuntimeError(
                f"group {group_id}: ready source {trajectory_id} is missing"
            )
        return (
            group_id,
            trajectory_id,
            acc.rewards_by_trajectory[trajectory_id],
            self._source_queued_at.pop((group_id, trajectory_id)),
            acc.parameter_sources.pop(trajectory_id),
            acc.boundary_sources.pop(trajectory_id),
            source_ready_event,
        )

    def _next_sealed_statistics_job(self) -> _HostCombineTask | None:
        """Reserve sealed host statistics while final forest work is in flight.

        Called under the reducer condition, after source jobs have priority.
        Forest members write parameter gradients, never these host statistics.
        The close still waits for both the forest and this combine to finish.
        """

        if (
            self.source_offload_mode not in {"blocking", "pinned_nonblocking"}
            or self._device_statistic_pool is not None
        ):
            return None
        for group_id, acc in self._groups.items():
            expected = self._expected_trajectory_order_by_group.get(group_id)
            if (
                expected is None
                or set(expected) != acc.closed
                or not acc.forest - acc.reduced
                or not acc.closed - acc.direct <= acc.reduced
                or group_id in self._groups_reducing
                or acc.finalizing
                or acc.combined_mean is not None
            ):
                continue
            if not any(
                isinstance(first, torch.Tensor)
                and isinstance(second, torch.Tensor)
                and first.device.type == second.device.type == "cpu"
                and first.device != self.params[index].device
                for index, (first, second) in enumerate(zip(
                    acc.parameter_centered_sum,
                    acc.parameter_source_mean,
                    strict=True,
                ))
            ):
                continue
            acc.finalizing = True
            self._groups_reducing.add(group_id)
            return _HostCombineTask(group_id)
        return None

    def _mark_streaming_source_ready_locked(
        self,
        key: tuple[int, int],
    ) -> None:
        if key not in self._streaming_capture_complete:
            return
        if self._streaming_pending_tensors.get(key, 0) != 0:
            return
        self._streaming_capture_complete.remove(key)
        self._streaming_pending_tensors.pop(key, None)
        staged_bytes = self._streaming_staged_bytes_by_trajectory.pop(key, 0)
        if staged_bytes:
            with self._accounting_lock:
                self._hbm_staged_source_count += 1
                self._hbm_staged_source_bytes += staged_bytes
        self._reduction_ready_queue.append((*key, None))

    def _next_offload_job(
        self,
    ) -> (
        tuple[int, int, list[_SourceValue], list[_SourceValue], Any | None]
        | _StreamingTensorOffload
        | None
    ):
        if not self._offload_queue:
            return None
        selected_index = 0
        if self.source_offload_mode == "pinned_streaming":
            selected_index = next(
                (
                    index
                    for index, candidate in enumerate(self._offload_queue)
                    if isinstance(candidate, _StreamingTensorOffload)
                    and candidate.group_id not in self._groups_reducing
                    and (
                        execution_order := self._source_execution_order_by_group.get(
                            candidate.group_id
                        )
                    )
                    and (
                        accumulator := self._groups.get(candidate.group_id)
                    )
                    is not None
                    and len(accumulator.reduced) < len(execution_order)
                    and execution_order[len(accumulator.reduced)]
                    == candidate.trajectory_id
                ),
                -1,
            )
            if selected_index < 0:
                return None
        job = self._offload_queue[selected_index]
        del self._offload_queue[selected_index]
        if isinstance(job, _StreamingTensorOffload):
            key = (job.group_id, job.trajectory_id)
        else:
            key = (job[0], job[1])
        active_jobs = self._active_offload_tensor_jobs_by_trajectory.get(key, 0)
        self._active_offload_tensor_jobs_by_trajectory[key] = active_jobs + 1
        if active_jobs == 0:
            group_id = key[0]
            self._active_offload_jobs += 1
            self._active_offload_jobs_peak = max(
                self._active_offload_jobs_peak,
                self._active_offload_jobs,
            )
            active_for_group = (
                self._active_offload_jobs_by_group.get(group_id, 0) + 1
            )
            self._active_offload_jobs_by_group[group_id] = active_for_group
            self._active_offload_jobs_peak_by_group[group_id] = max(
                self._active_offload_jobs_peak_by_group.get(group_id, 0),
                active_for_group,
            )
        return job

    def _finish_active_offload_locked(self, key: tuple[int, int]) -> None:
        active_jobs = self._active_offload_tensor_jobs_by_trajectory.get(key, 0)
        if active_jobs < 1:
            raise RuntimeError("streaming offload worker accounting underflow")
        if active_jobs > 1:
            self._active_offload_tensor_jobs_by_trajectory[key] = active_jobs - 1
            return
        del self._active_offload_tensor_jobs_by_trajectory[key]
        group_id = key[0]
        self._active_offload_jobs -= 1
        if self._active_offload_jobs < 0:
            raise RuntimeError("streaming offload worker accounting underflow")
        active_for_group = self._active_offload_jobs_by_group.get(group_id, 0) - 1
        if active_for_group < 0:
            raise RuntimeError("streaming group offload worker accounting underflow")
        if active_for_group == 0:
            self._active_offload_jobs_by_group.pop(group_id, None)
        else:
            self._active_offload_jobs_by_group[group_id] = active_for_group

    def _start_hbm_stage(self, key: tuple[int, int]) -> None:
        with self._accounting_lock:
            group_id = key[0]
            self._active_hbm_stage_tensor_jobs += 1
            self._active_hbm_stage_tensor_jobs_peak = max(
                self._active_hbm_stage_tensor_jobs_peak,
                self._active_hbm_stage_tensor_jobs,
            )
            tensor_jobs_for_group = (
                self._active_hbm_stage_tensor_jobs_by_group.get(group_id, 0)
                + 1
            )
            self._active_hbm_stage_tensor_jobs_by_group[group_id] = (
                tensor_jobs_for_group
            )
            self._active_hbm_stage_tensor_jobs_peak_by_group[group_id] = max(
                self._active_hbm_stage_tensor_jobs_peak_by_group.get(group_id, 0),
                tensor_jobs_for_group,
            )
            active_jobs = self._active_hbm_stage_tensor_jobs_by_trajectory.get(
                key, 0
            )
            self._active_hbm_stage_tensor_jobs_by_trajectory[key] = active_jobs + 1
            if active_jobs != 0:
                return
            self._active_hbm_stage_jobs += 1
            self._active_hbm_stage_jobs_peak = max(
                self._active_hbm_stage_jobs_peak,
                self._active_hbm_stage_jobs,
            )
            active_for_group = (
                self._active_hbm_stage_jobs_by_group.get(group_id, 0) + 1
            )
            self._active_hbm_stage_jobs_by_group[group_id] = active_for_group
            self._active_hbm_stage_jobs_peak_by_group[group_id] = max(
                self._active_hbm_stage_jobs_peak_by_group.get(group_id, 0),
                active_for_group,
            )

    def _finish_hbm_stage(self, key: tuple[int, int]) -> None:
        with self._accounting_lock:
            group_id = key[0]
            self._active_hbm_stage_tensor_jobs -= 1
            if self._active_hbm_stage_tensor_jobs < 0:
                raise RuntimeError("streaming HBM tensor-stage accounting underflow")
            tensor_jobs_for_group = (
                self._active_hbm_stage_tensor_jobs_by_group.get(group_id, 0)
                - 1
            )
            if tensor_jobs_for_group < 0:
                raise RuntimeError(
                    "streaming group HBM tensor-stage accounting underflow"
                )
            if tensor_jobs_for_group == 0:
                self._active_hbm_stage_tensor_jobs_by_group.pop(group_id, None)
            else:
                self._active_hbm_stage_tensor_jobs_by_group[group_id] = (
                    tensor_jobs_for_group
                )
            active_jobs = self._active_hbm_stage_tensor_jobs_by_trajectory.get(
                key, 0
            )
            if active_jobs < 1:
                raise RuntimeError("streaming HBM-stage accounting underflow")
            if active_jobs > 1:
                self._active_hbm_stage_tensor_jobs_by_trajectory[key] = (
                    active_jobs - 1
                )
                return
            del self._active_hbm_stage_tensor_jobs_by_trajectory[key]
            self._active_hbm_stage_jobs -= 1
            if self._active_hbm_stage_jobs < 0:
                raise RuntimeError("streaming HBM-stage accounting underflow")
            active_for_group = (
                self._active_hbm_stage_jobs_by_group.get(group_id, 0) - 1
            )
            if active_for_group < 0:
                raise RuntimeError("streaming group HBM-stage accounting underflow")
            if active_for_group == 0:
                self._active_hbm_stage_jobs_by_group.pop(group_id, None)
            else:
                self._active_hbm_stage_jobs_by_group[group_id] = active_for_group

    def _bind_worker_device(self) -> None:
        """Make this worker thread's current CUDA device the parameters' one.

        The CUDA runtime keeps a current device per host thread, starting
        at 0. A worker that touches the runtime -- an event wait, a host
        registration -- would otherwise open a context on device 0, which
        is another rank's GPU when several ranks share a host, and a host
        registration made there is pinned for that context only.
        """

        device = self._parameter_device()
        if device is None or device.type != "cuda":
            return
        torch.cuda.set_device(device)
        with self._accounting_lock:
            self._worker_cuda_devices[threading.current_thread().name] = (
                torch.cuda.current_device()
            )

    def _offload_loop(self) -> None:
        """Release completed CUDA copies independently of reduction order."""

        self._bind_worker_device()
        # Reserve this worker's pinned windows before any source arrives.
        # ``cudaHostAlloc`` of a full window costs order 100 ms and would
        # otherwise be charged to the first measured branch backward.
        if self.source_offload_mode == "pinned_streaming":
            if self._statistic_device is None:
                for slot in range(self._streaming_pinned_windows):
                    self._streaming_pinned_buffer(slot)

        try:
            while True:
                with self._reducer_condition:
                    if self._offload_stop:
                        return
                    job = self._next_offload_job()
                    while job is None and not self._offload_stop:
                        self._reducer_condition.wait()
                        job = self._next_offload_job()
                    if job is None:
                        return
                if isinstance(job, _StreamingTensorOffload):
                    group_id = job.group_id
                    trajectory_id = job.trajectory_id
                    key = (group_id, trajectory_id)
                    pressure = self._hbm_pressure_fraction([job.source_tensor])
                    if pressure >= self._source_offload_hbm_threshold:
                        started = time.perf_counter()
                        self._start_hbm_stage(key)
                        try:
                            # Fold this tensor directly through the worker's
                            # fixed pinned window.  A full pageable source copy
                            # would retain Q model-sized sources until group
                            # reduction and can exceed the host memory cgroup.
                            # Only the next objective-readiness trajectory in
                            # a group is eligible here, so per-parameter
                            # arithmetic order remains unchanged.
                            isolated_source: list[_SourceValue] = [
                                None
                            ] * len(job.destination)
                            isolated_source[job.destination_index] = (
                                job.source_tensor
                            )
                            self._fold_streaming_statistics(
                                isolated_source,
                                reward_sum=job.reward_sum,
                                unit_sum=job.unit_sum,
                                reward=job.reward,
                                keep_source_dtype=job.keep_source_dtype,
                                source_ready_event=job.source_ready_event,
                            )
                            job.destination[job.destination_index] = None
                        finally:
                            self._finish_hbm_stage(key)
                        tensor_bytes = (
                            job.source_tensor.numel()
                            * job.source_tensor.element_size()
                        )
                        with self._accounting_lock:
                            self._hbm_stage_s += time.perf_counter() - started
                        with self._reducer_condition:
                            self._streaming_staged_bytes_by_trajectory[key] = (
                                self._streaming_staged_bytes_by_trajectory.get(
                                    key, 0
                                )
                                + tensor_bytes
                            )
                    else:
                        # The source remains on CUDA, but reduction may run on
                        # a different worker/default stream.  Complete the
                        # producing autograd stream event in this background
                        # worker before making the trajectory reduction-ready.
                        # Admission never waits here.
                        job.source_ready_event.synchronize()
                    with self._reducer_condition:
                        pending = self._streaming_pending_tensors.get(key, 0) - 1
                        if pending < 0:
                            raise RuntimeError(
                                "streaming streaming tensor accounting underflow"
                            )
                        self._streaming_pending_tensors[key] = pending
                        self._finish_active_offload_locked(key)
                        self._mark_streaming_source_ready_locked(key)
                        self._reducer_condition.notify_all()
                else:
                    (
                        group_id,
                        trajectory_id,
                        parameter,
                        boundary,
                        source_ready_event,
                    ) = job
                    if self.source_offload_mode == "device_resident":
                        # Complete the producing stream's event in this
                        # background worker so a reducer on any stream may
                        # consume the device-resident tensors.  Admission
                        # never waits here.
                        if source_ready_event is not None:
                            source_ready_event.synchronize()
                            source_ready_event = None
                        # A group without a device home folds on the host,
                        # so its sources leave the device here.
                        on_device = self._group_statistics_on_device(group_id)
                        pressure = self._hbm_pressure_fraction(parameter)
                        spilled = (
                            not on_device
                            or pressure >= self._source_offload_hbm_threshold
                        )
                        if spilled:
                            # From here the copy drain owns the source's
                            # device memory, not the fold drain.
                            with self._reducer_condition:
                                self._device_source_bytes.pop(
                                    (group_id, trajectory_id), None
                                )
                            self._spill_device_resident_sources(parameter)
                            self._spill_device_resident_sources(boundary)
                    self._materialize_sources(parameter)
                    self._materialize_sources(boundary)
                    key = (group_id, trajectory_id)
                    with self._reducer_condition:
                        self._finish_active_offload_locked(key)
                        accumulator = self._groups.get(group_id)
                        if accumulator is None:
                            raise RuntimeError(
                                f"group {group_id} vanished during source "
                                "offload"
                            )
                        accumulator.materialized.add(trajectory_id)
                        # Reduction eligibility is the conjunction of a
                        # host-resident source and a bound reward; whichever
                        # side completes second publishes the job.
                        if trajectory_id in accumulator.closed:
                            self._reduction_ready_queue.append(
                                (group_id, trajectory_id, source_ready_event)
                            )
                        self._reducer_condition.notify_all()
        except BaseException as error:
            failure = _worker_failure(error)
            with self._reducer_condition:
                self._offload_failure = failure
                self._reducer_condition.notify_all()

    def _reducer_loop(self) -> None:
        self._bind_worker_device()
        # Reserve this worker's pinned windows before any source arrives.
        # ``cudaHostAlloc`` of a full window costs order 100 ms and would
        # otherwise be charged to the first measured branch backward.
        if self.source_offload_mode == "pinned_streaming":
            if self._statistic_device is None:
                for slot in range(self._streaming_pinned_windows):
                    self._streaming_pinned_buffer(slot)

        try:
            while True:
                with self._reducer_condition:
                    if self._reducer_stop:
                        return
                    job = self._next_reduction_job()
                    while job is None and not self._reducer_stop:
                        self._reducer_condition.wait()
                        job = self._next_reduction_job()
                    if job is None:
                        return
                if isinstance(job, _HostFoldTask):
                    job.run()
                    continue
                if isinstance(job, _HostCombineTask):
                    acc = self._groups[job.group_id]
                    try:
                        self._finalize_group_statistics(acc)
                        with self._accounting_lock:
                            self._sealed_statistics_combine_count += 1
                    except BaseException as error:
                        with self._reducer_condition:
                            self._reducer_failure = _worker_failure(error)
                        raise
                    finally:
                        with self._reducer_condition:
                            acc.finalizing = False
                            self._groups_reducing.remove(job.group_id)
                            self._reducer_condition.notify_all()
                    continue
                (
                    group_id,
                    trajectory_id,
                    reward,
                    queued_at,
                    parameter,
                    boundary,
                    source_ready_event,
                ) = job
                acc = self._groups[group_id]
                started = time.perf_counter()
                queue_s = started - queued_at
                executing_bytes = self._gradient_list_bytes(
                    parameter
                ) + self._gradient_list_bytes(boundary)
                with self._reducer_condition:
                    self._reducer_queue_s += queue_s
                    self._reducer_queue_max_s = max(
                        self._reducer_queue_max_s,
                        queue_s,
                    )
                    self._reducer_executing_source_bytes += executing_bytes
                    self._reducer_executing_source_bytes_peak = max(
                        self._reducer_executing_source_bytes_peak,
                        self._reducer_executing_source_bytes,
                    )
                with operator_range(
                    "reducer_fold",
                    fields={"group": group_id, "mode": self.source_offload_mode},
                ):
                    if self.source_offload_mode == "pinned_streaming":
                        self._fold_streaming_statistics(
                            parameter,
                            reward_sum=acc.parameter_centered_sum,
                            unit_sum=acc.parameter_source_mean,
                            reward=reward,
                            keep_source_dtype=self._statistic_keep_source_dtype,
                            source_ready_event=source_ready_event,
                        )
                        self._fold_streaming_statistics(
                            boundary,
                            reward_sum=acc.boundary_centered_sum,
                            unit_sum=acc.boundary_source_mean,
                            reward=reward,
                            keep_source_dtype=True,
                            source_ready_event=source_ready_event,
                        )
                    else:
                        # A device home is decided where the source's fold
                        # side is (_group_statistics_on_device); a host home
                        # at the group's first fold.
                        if (
                            self._statistic_pool is not None
                            and self._device_statistic_pool is None
                            and not acc.statistic_slot_tried
                        ):
                            acc.statistic_slot_tried = True
                            acc.statistic_slot = self._statistic_pool.acquire(
                                (id(self), group_id)
                            )
                            if acc.statistic_slot is not None:
                                with self._reducer_condition:
                                    self._statistic_slots[group_id] = (
                                        acc.statistic_slot
                                    )
                        last_fold_mean = self._start_last_fold(group_id, acc)
                        if last_fold_mean is not None:
                            self._fold_and_publish_parameters(
                                acc, parameter, reward=reward, mean=last_fold_mean
                            )
                        else:
                            self._fold_in_memory_statistics(
                                parameter,
                                reward_sum=acc.parameter_centered_sum,
                                unit_sum=acc.parameter_source_mean,
                                reward=reward,
                                keep_source_dtype=self._statistic_keep_source_dtype,
                                slot=acc.statistic_slot,
                                binary=getattr(self, "_binary_host_statistics", False),
                            )
                        self._fold_in_memory_statistics(
                            boundary,
                            reward_sum=acc.boundary_centered_sum,
                            unit_sum=acc.boundary_source_mean,
                            reward=reward,
                            keep_source_dtype=True,
                        )
                fold_s = time.perf_counter() - started
                with self._reducer_condition:
                    # Free the slab before the source counts as reduced, so
                    # a residency gate reading the reduced count never
                    # admits a source the arena has no slab for.
                    self._release_source_slab_locked((group_id, trajectory_id))
                    # The fold is enqueued and the reducer holds no source
                    # tensor: a device-resident source's memory is free.
                    self._device_source_bytes.pop((group_id, trajectory_id), None)
                    expected = self._expected_trajectory_order_by_group.get(
                        group_id
                    )
                    # reduced <= closed <= expected, so the last registered
                    # source to fold is the group's last source.
                    finalize = (
                        expected is not None
                        and len(acc.reduced) + 1 == len(expected)
                    )
                    if finalize:
                        acc.finalizing = True
                    self._reducer_fold_s += fold_s
                    self._reducer_executing_source_bytes -= executing_bytes
                    if self._reducer_executing_source_bytes < 0:
                        raise RuntimeError(
                            "streaming reducer residency accounting underflow"
                        )
                    self._groups_reducing.remove(group_id)
                    acc.reduced.add(trajectory_id)
                    acc.statistic_count += 1
                    self._source_reduction_order_by_group.setdefault(
                        group_id, []
                    ).append(trajectory_id)
                    self._reducer_condition.notify_all()
                if finalize:
                    try:
                        if not acc.pipelined:
                            self._finalize_group_statistics(acc)
                    finally:
                        with self._reducer_condition:
                            acc.finalizing = False
                            self._reducer_condition.notify_all()
        except BaseException as error:
            failure = _worker_failure(error)
            with self._reducer_condition:
                self._reducer_failure = failure
                self._reducer_condition.notify_all()

    def _raise_reducer_failure(self) -> None:
        if self._reducer_failure is not None:
            raise RuntimeError("streaming readiness reduction worker failed") from (
                self._reducer_failure
            )

    def _raise_worker_failure(self) -> None:
        if self._offload_failure is not None:
            raise RuntimeError("streaming source offload worker failed") from (
                self._offload_failure
            )
        self._raise_reducer_failure()

    @staticmethod
    def _join_workers(threads: list[threading.Thread]) -> list[threading.Thread]:
        """Join every thread with a timeout; return those still alive."""

        for thread in threads:
            thread.join(timeout=_WORKER_JOIN_TIMEOUT_S)
        return [thread for thread in threads if thread.is_alive()]

    def _shutdown_workers(self) -> None:
        """Stop and join the offload and reducer workers.

        Every worker is joined, and a worker failure's cleanup runs, even
        when an earlier worker does not stop in time. Workers still alive
        stay listed, so a later call joins them again; their names are
        raised together at the end, chained from any worker failure. A
        worker still alive may hold pinned or device buffers, so those are
        released only once every worker has stopped.
        """

        failure = self._offload_failure or self._reducer_failure
        stuck: list[threading.Thread] = []
        if self._offload_threads:
            with self._reducer_condition:
                self._offload_stop = True
                self._reducer_stop = True
                self._reducer_condition.notify_all()
            self._offload_threads = self._join_workers(self._offload_threads)
            stuck.extend(self._offload_threads)
        if self._reducer_threads:
            with self._reducer_condition:
                self._reducer_stop = True
                self._reducer_condition.notify_all()
            self._reducer_threads = self._join_workers(self._reducer_threads)
            stuck.extend(self._reducer_threads)
        failure = failure or self._offload_failure or self._reducer_failure
        if failure is not None:
            self._release_failed_worker_state(release_buffers=not stuck)
        if stuck:
            names = ", ".join(thread.name for thread in stuck)
            raise RuntimeError(
                f"streaming workers did not stop within "
                f"{_WORKER_JOIN_TIMEOUT_S:g} s: {names}"
            ) from failure
        if failure is not None:
            if self._offload_failure is not None:
                raise RuntimeError("streaming source offload worker failed") from failure
            raise RuntimeError("streaming readiness reduction worker failed") from failure

    def _release_failed_worker_state(self, *, release_buffers: bool) -> None:
        """Release a failed executor's queued sources and staging resources.

        No caller can safely reuse a failed executor, so every queued source
        is dropped before the original worker error surfaces, and with
        ``release_buffers`` every per-thread CUDA/pinned staging resource.
        """

        with self._reducer_condition:
            self._offload_queue.clear()
            self._reduction_ready_queue.clear()
            self._streaming_pending_tensors.clear()
            self._streaming_capture_complete.clear()
            self._streaming_staged_bytes_by_trajectory.clear()
            self._active_offload_jobs = 0
            self._active_offload_jobs_by_group.clear()
            self._active_offload_tensor_jobs_by_trajectory.clear()
            self._active_hbm_stage_jobs = 0
            self._active_hbm_stage_jobs_by_group.clear()
            self._active_hbm_stage_tensor_jobs_by_trajectory.clear()
            self._active_hbm_stage_tensor_jobs = 0
            self._active_hbm_stage_tensor_jobs_by_group.clear()
            self._groups_reducing.clear()
            self._source_queued_at.clear()
            for accumulator in self._groups.values():
                accumulator.parameter_sources.clear()
                accumulator.boundary_sources.clear()
                accumulator.parameter_centered_sum.clear()
                accumulator.parameter_source_mean.clear()
                accumulator.boundary_centered_sum.clear()
                accumulator.boundary_source_mean.clear()
        if not release_buffers:
            return
        self._release_all_source_slabs()
        self._release_statistic_slots()
        with self._reducer_condition:
            self._device_source_bytes.clear()
        if self._device_statistic_pool is not None:
            self._device_statistic_pool.close()
        self._streaming_pinned_buffers.clear()
        self._streaming_fold_buffers.clear()
        self._offload_streams.clear()

    def _fold_in_memory_statistics(
        self,
        sources: list[_SourceValue],
        *,
        reward_sum: list[torch.Tensor | None],
        unit_sum: list[torch.Tensor | None],
        reward: float,
        keep_source_dtype: bool,
        slot: Any = None,
        binary: bool = False,
    ) -> None:
        if not reward_sum:
            reward_sum.extend([None] * len(sources))
            unit_sum.extend([None] * len(sources))

        def fold(index: int) -> int:
            value = sources[index]
            if value is None:
                return 0
            if isinstance(value, _PendingCpuTransfer):
                raise RuntimeError(
                    "ordered reducer received an unmaterialized source transfer"
                )
            self._fold_one_statistic(
                value,
                index=index,
                reward_sum=reward_sum,
                unit_sum=unit_sum,
                reward=reward,
                keep_source_dtype=keep_source_dtype,
                slot=slot,
                binary=binary,
            )
            sources[index] = None
            return 0

        if self._can_share_host_fold(sources, unit_sum, reward_sum):
            self._share_host_parameters(fold, len(sources))
        else:
            for index in range(len(sources)):
                fold(index)

    def _can_share_host_fold(
        self,
        sources: list[_SourceValue],
        unit_sum: list[torch.Tensor | None],
        reward_sum: list[torch.Tensor | None],
    ) -> bool:
        return (
            threading.current_thread() in self._reducer_threads
            and self._source_reduction_worker_count > 1
            and len(sources) == len(self.params)
            and len(sources) > 1
            and self._device_statistic_pool is None
            and any(isinstance(value, torch.Tensor) for value in sources)
            and all(
                value is None
                or isinstance(value, torch.Tensor) and value.device.type == "cpu"
                for values in (sources, unit_sum, reward_sum)
                for value in values
            )
        )

    def _share_host_parameters(
        self, function: Callable[[int], int], count: int
    ) -> list[int]:
        """Share independent parameters while retaining the group's reservation.

        Waiting owners also serve queued helpers, so every reducer can own a
        group without exhausting the pool. No new thread or tensor is created.
        All helpers finish before the owner advances to the group's next source.
        """

        remaining = deque(range(count))
        pending = min(self._source_reduction_worker_count, count)
        errors: list[BaseException] = []
        results = [0] * count

        def part() -> None:
            nonlocal pending
            try:
                while True:
                    with self._reducer_condition:
                        if not remaining or errors:
                            break
                        index = remaining.popleft()
                    results[index] = function(index)
            except BaseException as error:
                with self._reducer_condition:
                    errors.append(error)
            finally:
                with self._reducer_condition:
                    pending -= 1
                    self._reducer_condition.notify_all()

        with self._reducer_condition:
            self._host_fold_tasks.extend(
                _HostFoldTask(part) for _ in range(1, pending)
            )
            self._reducer_condition.notify_all()
        part()
        while True:
            with self._reducer_condition:
                if not pending:
                    break
                if not self._host_fold_tasks:
                    self._reducer_condition.wait()
                    continue
                job = self._host_fold_tasks.popleft()
            job.run()
        if errors:
            raise errors[0]
        with self._accounting_lock:
            self._host_parameter_dispatches += 1
            self._host_parameter_tasks += count
        return results

    def _start_last_fold(self, group_id: int, acc: _GroupAccum) -> float | None:
        """Mark a group's last host fold as pipelined; its reward mean, or None.

        The group's last source to fold completes its statistics, so every
        reward is bound and the mean is final. Host statistics only: the
        device homes (device_resident) combine on the device at the close.
        """

        if (
            self.source_offload_mode not in {"blocking", "pinned_nonblocking"}
            or self._device_statistic_pool is not None
        ):
            return None
        with self._reducer_condition:
            expected = self._expected_trajectory_order_by_group.get(group_id)
            if expected is None or len(acc.reduced) + 1 != len(expected):
                return None
            mean, _std = self._group_moments(acc)
            if not acc.parameter_centered_sum:
                # A one-trajectory group's last fold is also its first; the
                # close sizes its parameter loop from these lists.
                acc.parameter_centered_sum.extend([None] * len(self.params))
                acc.parameter_source_mean.extend([None] * len(self.params))
            acc.combined_mean = mean
            acc.finalizing = True
            acc.pipelined = True
            self._reducer_condition.notify_all()
        return mean

    def _fold_and_publish_parameters(
        self,
        acc: _GroupAccum,
        sources: list[_SourceValue],
        *,
        reward: float,
        mean: float,
    ) -> None:
        """Fold a group's last source and combine each parameter at once.

        Per parameter, the fold (``G2 += g``, ``G1 += r g``) and then the
        close's combine (``G1 -= mean G2``): the same elementwise operations
        in the same order on the same host tensors as folding every parameter
        and then combining them, so the values are the same. Parameters run
        across the worker entitlement the combine already uses, and each is
        published (``acc.ready_parameters``) when done, so a close waiting on
        the group reloads it to the device while later ones still fold.
        """

        reward_sum = acc.parameter_centered_sum
        unit_sum = acc.parameter_source_mean
        if not reward_sum:
            reward_sum.extend([None] * len(sources))
            unit_sum.extend([None] * len(sources))
        if any(isinstance(value, _PendingCpuTransfer) for value in sources):
            raise RuntimeError(
                "ordered reducer received an unmaterialized source transfer"
            )
        started = time.perf_counter()

        def fold_and_combine(index: int) -> int:
            value = sources[index]
            if value is not None:
                self._fold_one_statistic(
                    value,
                    index=index,
                    reward_sum=reward_sum,
                    unit_sum=unit_sum,
                    reward=reward,
                    keep_source_dtype=self._statistic_keep_source_dtype,
                    slot=acc.statistic_slot,
                    binary=getattr(self, "_binary_host_statistics", False),
                )
                sources[index] = None
            first, second = reward_sum[index], unit_sum[index]
            combined = (
                isinstance(first, torch.Tensor)
                and isinstance(second, torch.Tensor)
                and first.device == second.device
                and index < len(self.params)
                and first.device != self.params[index].device
            )
            if combined:
                self._combine_parameter_statistics(first, second, mean)
            with self._reducer_condition:
                if combined:
                    acc.combined_indices.add(index)
                acc.ready_parameters.add(index)
                self._reducer_condition.notify_all()
            return first.numel() * first.element_size() if combined else 0

        workers = min(self._source_offload_worker_count, len(sources))
        if self._can_share_host_fold(sources, unit_sum, reward_sum):
            combined_bytes = sum(
                self._share_host_parameters(fold_and_combine, len(sources))
            )
        elif workers <= 1:
            combined_bytes = sum(map(fold_and_combine, range(len(sources))))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                combined_bytes = sum(
                    pool.map(fold_and_combine, range(len(sources)))
                )
        with self._accounting_lock:
            self._group_close_host_combined_bytes += combined_bytes
            self._group_last_fold_s += time.perf_counter() - started
            self._group_finalized_in_reducer_count += 1

    def _wait_parameter_ready(self, acc: _GroupAccum, index: int) -> None:
        with self._reducer_condition:
            if index in acc.ready_parameters:
                return
            started = time.perf_counter()
            while index not in acc.ready_parameters and (
                acc.reduced != acc.closed or acc.finalizing
            ):
                self._raise_worker_failure()
                self._reducer_condition.wait()
            self._raise_worker_failure()
            self._group_close_parameter_wait_s += time.perf_counter() - started

    def _fold_streaming_statistics(
        self,
        sources: list[_SourceValue],
        *,
        reward_sum: list[torch.Tensor | None],
        unit_sum: list[torch.Tensor | None],
        reward: float,
        keep_source_dtype: bool,
        source_ready_event: Any | None,
    ) -> None:
        """Copy and consume one source tensor at a time in a reducer thread."""

        if not reward_sum:
            reward_sum.extend([None] * len(sources))
            unit_sum.extend([None] * len(sources))
        for index, value in enumerate(sources):
            if value is None:
                continue
            if isinstance(value, _PendingCpuTransfer):
                raise RuntimeError(
                    "streaming reducer received an unexpected pending transfer"
                )
            fold_in_place = value.device.type == "cpu" or (
                self._statistic_device is not None
                and value.device.type == "cuda"
                and (
                    self._statistic_device.index is None
                    or value.device == self._statistic_device
                )
            )
            if fold_in_place:
                if value.device.type == "cuda":
                    # The fold runs on this worker's current stream (the
                    # device default), ordered after the producer through the
                    # readiness event. Stream ordering is the correctness
                    # contract; a dedicated fold stream is a later
                    # measurement question, not a correctness one.
                    if source_ready_event is not None:
                        torch.cuda.current_stream(value.device).wait_event(
                            source_ready_event
                        )
                self._fold_one_statistic(
                    value,
                    index=index,
                    reward_sum=reward_sum,
                    unit_sum=unit_sum,
                    reward=reward,
                    keep_source_dtype=keep_source_dtype,
                )
                sources[index] = None
                continue
            if value.device.type != "cuda":
                raise RuntimeError(
                    "pinned_streaming reducer requires CUDA source tensors"
                )
            tensor_bytes = value.numel() * value.element_size()
            if not value.is_contiguous():
                reducer_stream = torch.cuda.current_stream(value.device)
                if source_ready_event is not None:
                    reducer_stream.wait_event(source_ready_event)
                value = value.contiguous()
                sources[index] = value
                source_ready_event = torch.cuda.Event()
                source_ready_event.record(reducer_stream)
                with self._accounting_lock:
                    self._source_contiguous_copy_bytes += tensor_bytes
                    self._max_single_source_contiguous_copy_bytes = max(
                        self._max_single_source_contiguous_copy_bytes,
                        tensor_bytes,
                    )
            statistic_dtype = value.dtype if keep_source_dtype else torch.float32
            unit_was_empty = unit_sum[index] is None
            if unit_was_empty:
                unit_sum[index] = torch.empty_like(
                    value,
                    dtype=statistic_dtype,
                    device="cpu",
                    memory_format=torch.contiguous_format,
                )
            reward_was_empty = reward_sum[index] is None
            reward_is_needed = (
                not (unit_was_empty and not keep_source_dtype and reward == 0.0)
                if unit_was_empty
                else reward != 0.0
            )
            if reward_was_empty and reward_is_needed:
                reward_sum[index] = torch.empty_like(
                    unit_sum[index],
                    device="cpu",
                    memory_format=torch.contiguous_format,
                )
            if unit_was_empty or (reward_was_empty and reward_is_needed):
                added_statistic_bytes = (
                    unit_sum[index].numel() * unit_sum[index].element_size()
                    if unit_was_empty
                    else 0
                ) + (
                    reward_sum[index].numel()
                    * reward_sum[index].element_size()
                    if reward_was_empty and reward_is_needed
                    else 0
                )
                with self._accounting_lock:
                    self._current_statistic_bytes += added_statistic_bytes
                    self._peak_statistic_bytes = max(
                        self._peak_statistic_bytes,
                        self._current_statistic_bytes,
                    )

            source_flat = value.view(-1)
            unit_flat = unit_sum[index].view(-1)
            reward_flat = (
                None
                if reward_sum[index] is None
                else reward_sum[index].view(-1)
            )
            chunk_elements = max(
                1,
                self._source_offload_window_bytes // value.element_size(),
            )
            # Two-window pipeline: issue the next chunk's D2H while the
            # previous chunk's FP32 fold still reads its own pinned window.
            # A single window would serialize transfer behind host arithmetic
            # without changing readiness order or arithmetic order.
            windows = self._streaming_pinned_windows
            pending: list[tuple[_PendingCpuTransfer, int, int]] = []
            slot = 0
            for offset in range(0, source_flat.numel(), chunk_elements):
                end = min(offset + chunk_elements, source_flat.numel())
                transfer = self._submit_streaming_chunk(
                    source_flat[offset:end],
                    source_ready_event=source_ready_event,
                    slot=slot,
                )
                slot = (slot + 1) % windows
                pending.append((transfer, offset, end))
                # Fold only once every window is spoken for, so `windows - 1`
                # transfers stay outstanding. Folding is FIFO, so chunks reach
                # the statistics in offset order whatever the depth.
                if len(pending) == windows:
                    self._fold_streaming_chunk(
                        pending.pop(0),
                        unit_flat=unit_flat,
                        reward_flat=reward_flat,
                        reward=reward,
                        unit_was_empty=unit_was_empty,
                        reward_was_empty=reward_was_empty,
                    )
            while pending:
                self._fold_streaming_chunk(
                    pending.pop(0),
                    unit_flat=unit_flat,
                    reward_flat=reward_flat,
                    reward=reward,
                    unit_was_empty=unit_was_empty,
                    reward_was_empty=reward_was_empty,
                )
            sources[index] = None
            with self._accounting_lock:
                self._current_source_bytes -= tensor_bytes
                if self._current_source_bytes < 0:
                    raise RuntimeError(
                        "reward-linear source residency accounting underflow"
                    )

    def _fold_streaming_chunk(
        self,
        pending: tuple[_PendingCpuTransfer, int, int],
        *,
        unit_flat: torch.Tensor,
        reward_flat: torch.Tensor | None,
        reward: float,
        unit_was_empty: bool,
        reward_was_empty: bool,
    ) -> None:
        """Complete one staged chunk and fold it into its fp32 statistics."""

        transfer, offset, end = pending
        started = time.perf_counter()
        cpu_chunk = transfer.materialize()
        waited_s = time.perf_counter() - started
        with self._accounting_lock:
            self._source_offload_wait_s += waited_s
            self._streaming_chunk_copy_wait_s += waited_s
            self._streaming_chunk_count += 1

        fold_started = time.perf_counter()
        unit_chunk = unit_flat[offset:end]
        source_chunk = cpu_chunk
        if unit_chunk.dtype != cpu_chunk.dtype:
            source_chunk = self._streaming_fold_buffer(
                cpu_chunk.numel(), unit_chunk.dtype
            )
            source_chunk.copy_(cpu_chunk)
            converted = cpu_chunk.numel() * cpu_chunk.element_size()
        else:
            converted = 0
        if unit_was_empty:
            unit_chunk.copy_(source_chunk)
        else:
            unit_chunk.add_(source_chunk)
        if reward_flat is not None and reward != 0.0:
            reward_chunk = reward_flat[offset:end]
            if reward_was_empty:
                reward_chunk.copy_(source_chunk).mul_(reward)
            else:
                reward_chunk.add_(source_chunk, alpha=reward)
        elif reward_flat is not None and unit_was_empty:
            reward_flat[offset:end].zero_()
        fold_s = time.perf_counter() - fold_started

        with self._accounting_lock:
            self._streaming_chunk_fold_s += fold_s
            self._streaming_fold_conversion_bytes += converted
            self._source_offload_inflight_bytes -= transfer.bytes
            if self._source_offload_inflight_bytes < 0:
                raise RuntimeError(
                    "source offload residency accounting underflow"
                )

    def _fold_converted_accumulate(
        self,
        value: torch.Tensor,
        *,
        index: int,
        reward_sum: list[torch.Tensor | None],
        unit_sum: list[torch.Tensor | None],
        reward: float,
        slot: Any = None,
    ) -> None:
        """Accumulate a CPU source into differently-typed statistics.

        Windows of the source convert once into the per-thread fold buffer
        (bounded by the offload window, so the buffer never grows to the
        full tensor), then fold with same-dtype AXPYs. Values are bitwise
        identical to the direct mixed-dtype accumulation: both cast the
        source element to the statistic dtype before the add.
        """

        target = unit_sum[index]
        if target is None:
            raise RuntimeError(f"the unit statistic of parameter {index} is not materialized")
        reward_target = reward_sum[index]
        created_reward = False
        if reward != 0.0 and reward_target is None:
            # Same delayed-G1 semantics as the direct path: the statistic
            # materializes from this source alone, so every window writes
            # its slice below and no uninitialized element survives.
            reward_target = self._statistic_home(
                slot, index, target, target.dtype, "reward"
            )
            if reward_target is None:
                reward_target = torch.empty_like(target)
            reward_sum[index] = reward_target
            created_reward = True
            with self._accounting_lock:
                self._current_statistic_bytes += (
                    reward_target.numel() * reward_target.element_size()
                )
                self._peak_statistic_bytes = max(
                    self._peak_statistic_bytes,
                    self._current_statistic_bytes,
                )
        flat_value = value.reshape(-1)
        # view (not reshape) for the write targets: a non-contiguous
        # statistic must fail loudly, never silently fold into a copy.
        flat_unit = target.view(-1)
        flat_reward = (
            None
            if reward_target is None or reward == 0.0
            else reward_target.view(-1)
        )
        window = max(
            1,
            self._source_offload_window_bytes // target.element_size(),
        )
        total = flat_value.numel()
        for start in range(0, total, window):
            end = min(start + window, total)
            converted = self._streaming_fold_buffer(end - start, target.dtype)
            converted.copy_(flat_value[start:end])
            flat_unit[start:end].add_(converted)
            if flat_reward is not None:
                if created_reward:
                    flat_reward[start:end].copy_(converted).mul_(reward)
                else:
                    flat_reward[start:end].add_(converted, alpha=reward)
        with self._accounting_lock:
            self._streaming_fold_conversion_bytes += (
                total * flat_value.element_size()
            )

    def _statistic_home(
        self,
        slot: Any,
        index: int,
        value: torch.Tensor,
        dtype: torch.dtype,
        statistic: str,
    ) -> torch.Tensor | None:
        """Parameter ``index``'s pooled ``unit``/``reward`` home, if it has one.

        A host home takes host values only; a device home takes a value
        from either side (a spilled source copies into it).
        """

        if slot is None:
            return None
        pool = self._statistic_pool
        if (
            pool is None
            or (
                pool is not self._device_statistic_pool
                and value.device.type != "cpu"
            )
            or dtype != pool.dtype
            or not pool.fits(index, value)
        ):
            return None
        return pool.view(slot, index, value.shape, statistic=statistic)

    def _add_host_statistic_pair(
        self,
        value: torch.Tensor,
        unit: torch.Tensor,
        weighted: torch.Tensor | None,
        reward: float,
    ) -> bool:
        """Tile unit-reward additions when both host moments already exist."""

        if (
            # Nonunit BF16 alpha can round differently in ATen's scalar
            # and vector tails. Keep its original whole-tensor call.
            reward != 1.0
            or weighted is None
            or not all(
                tensor.device.type == "cpu"
                and tensor.is_contiguous()
                and tensor.dtype == value.dtype
                and tensor.shape == value.shape
                for tensor in (value, unit, weighted)
            )
            or value.numel() * value.element_size()
            <= self._host_statistic_tile_bytes
        ):
            return False
        source, first, second = (
            tensor.view(-1) for tensor in (value, unit, weighted)
        )
        width = self._host_statistic_tile_bytes // value.element_size()
        tiles = 0
        # Every element retains the same two ATen calls and source order.
        # Views share the existing storage; no conversion or sum is introduced.
        for start in range(0, source.numel(), width):
            end = min(start + width, source.numel())
            first[start:end].add_(source[start:end])
            second[start:end].add_(source[start:end], alpha=reward)
            tiles += 1
        with self._accounting_lock:
            self._host_statistic_tiled_parameters += 1
            self._host_statistic_tiled_elements += value.numel()
            self._host_statistic_tile_count += tiles
        return True

    def _fold_one_statistic(
        self,
        value: torch.Tensor,
        *,
        index: int,
        reward_sum: list[torch.Tensor | None],
        unit_sum: list[torch.Tensor | None],
        reward: float,
        keep_source_dtype: bool,
        slot: Any = None,
        binary: bool = False,
    ) -> None:
        tensor_bytes = value.numel() * value.element_size()
        existing = unit_sum[index]
        if existing is not None and value.device != existing.device:
            # Mixed residency after a device_resident spill episode: the
            # statistic pair's first-touch device wins, so a spilled source
            # reloads (or a resident source stages) to that device before
            # folding.  Arithmetic and fold order are unchanged.
            value = self._reload_source(value, existing.device)
        if binary:
            self._fold_binary_statistic(
                value, index=index, reward_sum=reward_sum, unit_sum=unit_sum,
                reward=reward, slot=slot,
            )
        elif unit_sum[index] is None:
            statistic_dtype = value.dtype if keep_source_dtype else torch.float32
            home = self._statistic_home(slot, index, value, statistic_dtype, "unit")
            if home is not None:
                # The same values the clone (or the exact fp32 widening)
                # would hold, written into the group's reused home.
                unit_sum[index] = home.copy_(value)
            else:
                owned = value if keep_source_dtype else value.to(torch.float32)
                unit_sum[index] = owned.clone() if owned is value else owned
            # r_i H_i is exactly zero when r_i is zero.  Delay G1 instead
            # of allocating and writing a full FP32 parameter vector.  G2
            # is still folded immediately, so legal admission and
            # readiness-order execution are unchanged.
            if not keep_source_dtype and reward == 0.0:
                reward_sum[index] = None
            else:
                reward_home = self._statistic_home(
                    slot, index, value, statistic_dtype, "reward"
                )
                reward_sum[index] = (
                    unit_sum[index] * reward
                    if reward_home is None
                    else torch.mul(unit_sum[index], reward, out=reward_home)
                )
            with self._accounting_lock:
                self._current_statistic_bytes += (
                    unit_sum[index].numel() * unit_sum[index].element_size()
                    + (
                        0
                        if reward_sum[index] is None
                        else reward_sum[index].numel()
                        * reward_sum[index].element_size()
                    )
                )
                self._peak_statistic_bytes = max(
                    self._peak_statistic_bytes,
                    self._current_statistic_bytes,
                )
        elif (
            value.device.type == "cpu"
            and value.dtype != unit_sum[index].dtype
        ):
            # ATen accumulates a bfloat16 source into an fp32 statistic
            # through an unvectorized mixed-dtype kernel, an order of
            # magnitude slower than a same-dtype AXPY. Convert each window
            # once through the per-thread fold buffer, then accumulate in
            # the statistic dtype. The conversion is exact and the
            # per-element accumulation order is unchanged.
            self._fold_converted_accumulate(
                value,
                index=index,
                reward_sum=reward_sum,
                unit_sum=unit_sum,
                reward=reward,
                slot=slot,
            )
        elif self._add_host_statistic_pair(
            value, unit_sum[index], reward_sum[index], reward
        ):
            pass
        else:
            unit_sum[index].add_(value)
            if reward != 0.0:
                if reward_sum[index] is None:
                    reward_home = self._statistic_home(
                        slot, index, value, torch.float32, "reward"
                    )
                    reward_sum[index] = (
                        # FP32 widening is a no-op: own the statistic instead
                        # of retaining and modifying a reusable source slab.
                        value.to(torch.float32).mul(reward)
                        if reward_home is None
                        else reward_home.copy_(value).mul_(reward)
                    )
                    with self._accounting_lock:
                        self._current_statistic_bytes += (
                            reward_sum[index].numel()
                            * reward_sum[index].element_size()
                        )
                        self._peak_statistic_bytes = max(
                            self._peak_statistic_bytes,
                            self._current_statistic_bytes,
                        )
                else:
                    reward_sum[index].add_(value, alpha=reward)
        with self._accounting_lock:
            self._current_source_bytes -= tensor_bytes
            if self._current_source_bytes < 0:
                raise RuntimeError(
                    "reward-linear source residency accounting underflow"
                )

    def _fold_binary_statistic(
        self, value: torch.Tensor, *, index: int,
        reward_sum: list[torch.Tensor | None],
        unit_sum: list[torch.Tensor | None], reward: float, slot: Any,
    ) -> None:
        if reward_sum[index] is None:
            # Keep the original two homes and their source dtype. Even the
            # zero half is initialized, so partial/direct groups need no
            # missing-statistic special case at their close.
            for values, coefficient, name in (
                (reward_sum, reward, "reward"), (unit_sum, reward - 1.0, "unit")
            ):
                home = self._statistic_home(slot, index, value, value.dtype, name)
                values[index] = (value * coefficient if home is None
                                 else torch.mul(value, coefficient, out=home))
            with self._accounting_lock:
                self._current_statistic_bytes += 2 * value.numel() * value.element_size()
                self._peak_statistic_bytes = max(
                    self._peak_statistic_bytes, self._current_statistic_bytes
                )
        else:
            target = reward_sum[index] if reward == 1.0 else unit_sum[index]
            target.add_(value, alpha=1.0 if reward == 1.0 else -1.0)
            with self._accounting_lock:
                self._binary_statistic_adds += 1
        with self._accounting_lock:
            self._binary_statistic_parameters += 1

    @staticmethod
    def _gradient_list_bytes(
        values: list[_SourceValue],
    ) -> int:
        return sum(
            (
                value.bytes
                if isinstance(value, _PendingCpuTransfer)
                else value.numel() * value.element_size()
            )
            for value in values
            if value is not None
        )

    def _release_statistic_pair(
        self,
        reward_sum: list[torch.Tensor | None],
        unit_sum: list[torch.Tensor | None],
        index: int,
    ) -> None:
        values = (reward_sum[index], unit_sum[index])
        released_bytes = sum(
            value.numel() * value.element_size()
            for value in values
            if value is not None
        )
        with self._accounting_lock:
            self._current_statistic_bytes -= released_bytes
            if self._current_statistic_bytes < 0:
                raise RuntimeError(
                    "reward-linear statistic residency accounting underflow"
                )
        reward_sum[index] = None
        unit_sum[index] = None

    def _check_versions(self) -> None:
        for p, v in zip(self.params, self._versions, strict=True):
            if p._version != v:
                raise RuntimeError(
                    "a parameter was modified in place (optimizer step?) while "
                    "streamed groups were open; the retained graphs are invalid. "
                    "Close all groups before stepping -- see assert_safe_to_step()."
                )
