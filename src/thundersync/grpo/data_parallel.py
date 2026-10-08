"""Data-parallel thundersync: replicated streaming trainers, one averaged update.

Multi-GPU scope: **replication, not sharding**. The reward-linear backward's `autograd.grad`
bypasses reduce-scatter
hooks (silently unreduced gradients), and the version guard's `p._version`
contract inverts under shard-mutation + re-gather. Under replication neither
hazard exists: every rank holds a full model replica, runs the unmodified
single-GPU streaming machinery (StreamingRun + RewardLinearBackward), and the
ONLY cross-rank communication is one gradient all-reduce at step time. The
per-rank semantics -- retained graphs, boundary cuts, the version guard, Mode
C's per-trajectory `autograd.grad` backwards -- hold unchanged, because
nothing between two optimizer steps ever touches another rank.

Why this is correct data parallelism for GRPO: groups are independent until
the optimizer step. A group's advantages are computed from ITS OWN rewards
(mean/std within the group), so no cross-rank statistic exists anywhere in the
objective -- rank r can stream, close, and accumulate its groups knowing
nothing of rank s. The batch is sharded at GROUP granularity: each rank
streams its own groups' events against its own replica.

Scale normalization -- the documented contract
----------------------------------------------

The global objective is defined as

    L_global = (1 / N) * sum over ALL N groups of L_group(g)

i.e. the MEAN over groups of the per-group GRPO loss (each `L_group` is
`sum_i A_i * S_i` for that group). The reward-linear `close_group` accumulates
each group's gradient into `param.grad` as a SUM, so after a rank closes its
G_r local groups it holds `sum_{g local} grad L_group(g)`. `step()` therefore

    1. scales every local group-sum by ``world_size / N``, then
    2. all-reduces with ReduceOp.AVG.

The composition equals `(1/N) sum_all grad L_group(g)`, including when ranks
own unequal numbers of complete groups.  For equal counts, the implementation
retains the original divide-by-local-count path exactly.  A single-process run
over all N groups whose accumulated grads are divided by N produces the same
mathematical update.

Grad materialization: a parameter untouched by any local group (a frozen
adapter target, an unused head) has `grad=None` on some ranks. Collectives
must be structurally identical across ranks, so before the reduce every
trainable parameter with `grad is None` gets an explicit zero grad -- every
rank materializes ALL of them, so the reduce list is the same length, order,
and shapes everywhere. Zeros are the mathematically correct contribution of
an untouched parameter under the mean-over-groups objective.

Lifecycle: `step()` finalizes one generation batch (one optimizer step per
batch is the reward-linear backward's contract) and then re-arms -- a FRESH StreamingRun and
RewardLinearBackward are built from the factory, so the version guard records
the post-step parameter versions and the next batch's events stream against a
clean run. `assert_safe_to_step()` runs BEFORE the barrier, so a rank with
open groups raises locally and loudly without deadlocking the others inside
a collective.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
import torch.distributed as dist

from thundersync.grpo.config import DEFAULT_GRPO_EPSILON, DEFAULT_SOURCE_OFFLOAD_HBM_THRESHOLD
from thundersync.grpo.gradient_average import per_parameter_all_reduce_average
from thundersync.grpo.work_price import attention_weight, trajectory_price
from thundersync.grpo.claims import decode_final_work, encode_final_work
from thundersync.grpo.final_forest import forest_trajectory
from thundersync.grpo.reward_linear import RewardLinearBackward
from thundersync.grpo.pinned_arena import PinnedSourceArena
from thundersync.accel.capability import kernel_probe_records
from thundersync.engine.streaming import StreamingRun, attention_path_counts

logger = logging.getLogger(__name__)

RunFactory = Callable[[torch.nn.Module], StreamingRun]
GradientObserver = Callable[[str, list[torch.nn.Parameter]], None]


def _default_run_factory(model: torch.nn.Module) -> StreamingRun:
    # boundary_cut=True is a requirement of the reward-linear backward, not a preference:
    # RewardLinearBackward refuses a run without the cut.
    return StreamingRun(model, boundary_cut=True)


class DataParallelThunderSync:
    """One rank of a replicated data-parallel thundersync trainer.

    Wraps (model, optimizer, StreamingRun factory, RewardLinearBackward) and
    mirrors the single-GPU event API for the rank's LOCAL groups:

        open_group / append_turn / append_turns    -> StreamingRun
        close_trajectory / close_group             -> RewardLinearBackward
        step()                                     -> the one collective point

    Construct via `DataParallelThunderSync.init_from_env(...)` under torchrun; the
    constructor itself assumes the process group already exists (so tests can
    compose it differently if they ever need to).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        run_factory: RunFactory | None = None,
        eps: float = DEFAULT_GRPO_EPSILON,
        grad_clip: float | None = 1.0,
        source_offload_device: str | torch.device | None = "cpu",
        source_offload_mode: str = "blocking",
        source_offload_window_bytes: int = 256 * 1024 * 1024,
        source_offload_hbm_threshold: float = DEFAULT_SOURCE_OFFLOAD_HBM_THRESHOLD,
        source_reduction_mode: str = "unrestricted_statistics",
        source_reduction_pack_size: int = 1,
        source_reduction_worker_count: int = 2,
        source_offload_worker_count: int = 2,
        source_offload_pinned_windows: int = 2,
        source_spill_directory: str | Path | None = None,
        statistic_device: str | torch.device | None = None,
        device_statistic_slots: int | None = None,
        statistic_keep_source_dtype: bool = False,
        sharded_step: Any | None = None,
        retained_bytes_log_every: int = 0,
        source_pinned_arena: PinnedSourceArena | None = None,
        direct_when_group_bound: bool = False,
        source_release_during_backward: bool = False,
        combine_known_reward_sources: bool = False,
    ) -> None:
        if not dist.is_initialized():
            raise RuntimeError(
                "DataParallelThunderSync needs an initialized process group; use "
                "DataParallelThunderSync.init_from_env() under torchrun"
            )
        self.model = model
        self.optimizer = optimizer
        # ZeRO-1 delegate: when set, allreduce+clip+optimizer step run inside
        # it, with only this rank's optimizer-state shard resident, and
        # self.optimizer must be its shard optimizer.
        self._sharded_step = sharded_step
        self.run_factory = run_factory or _default_run_factory
        self.eps = eps
        if grad_clip is not None and grad_clip <= 0:
            raise ValueError("grad_clip must be positive when enabled")
        self.grad_clip = grad_clip
        self.source_offload_device = source_offload_device
        self.source_offload_mode = source_offload_mode
        self.source_offload_window_bytes = source_offload_window_bytes
        self.source_offload_hbm_threshold = source_offload_hbm_threshold
        self.source_reduction_mode = source_reduction_mode
        self.source_reduction_pack_size = source_reduction_pack_size
        self.source_reduction_worker_count = source_reduction_worker_count
        self.source_offload_worker_count = source_offload_worker_count
        self.source_offload_pinned_windows = source_offload_pinned_windows
        self.source_spill_directory = source_spill_directory
        self.statistic_device = statistic_device
        self.device_statistic_slots = device_statistic_slots
        self.statistic_keep_source_dtype = statistic_keep_source_dtype
        self.direct_when_group_bound = direct_when_group_bound
        self.source_release_during_backward = source_release_during_backward
        self.combine_known_reward_sources = combine_known_reward_sources
        self._binary_rewards = False
        # Owned by the trainer: every step's backward reuses its slabs, and
        # shutdown unregisters them.
        self.source_pinned_arena = source_pinned_arena
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = next(model.parameters()).device
        if self.device.type != "cuda":
            raise ValueError(
                f"replica must live on a cuda device, got {self.device}; "
                "NCCL collectives cannot reduce CPU grads"
            )
        self._trainable = [p for p in model.parameters() if p.requires_grad]
        self._steps_done = 0
        self._shutdown = False
        # In-step memory sampling: every Nth source event (prepare, bind,
        # close) logs one JSON line at INFO with the allocator's counters and
        # what the run, the backward and this trainer hold, so a failure
        # inside a step still leaves the holder breakdown. 0 = never.
        if retained_bytes_log_every < 0:
            raise ValueError("retained_bytes_log_every must be >= 0")
        self._retained_bytes_log_every = int(retained_bytes_log_every)
        self._retained_events = 0
        # Post-rollout final work as forest micro-batches
        # (`configure_final_forest`); None runs it on the direct path.
        self._final_forest: tuple[int, int] | None = None
        self._known_reserve_slabs = 0
        # Price model of a forest trajectory (thundersync.grpo.claims), from the
        # model's shape on first use.
        self._price_attention_weight: float | None = None
        self._new_batch()

    # ------------------------------------------------------------ construction

    @classmethod
    def init_from_env(
        cls,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        run_factory: RunFactory | None = None,
        eps: float = DEFAULT_GRPO_EPSILON,
        grad_clip: float | None = 1.0,
        source_offload_device: str | torch.device | None = "cpu",
        source_offload_mode: str = "blocking",
        source_offload_window_bytes: int = 256 * 1024 * 1024,
        source_offload_hbm_threshold: float = DEFAULT_SOURCE_OFFLOAD_HBM_THRESHOLD,
        source_reduction_mode: str = "unrestricted_statistics",
        source_reduction_pack_size: int = 1,
        source_reduction_worker_count: int = 2,
        source_offload_worker_count: int = 2,
        source_offload_pinned_windows: int = 2,
        source_spill_directory: str | Path | None = None,
        statistic_device: str | torch.device | None = None,
        device_statistic_slots: int | None = None,
        statistic_keep_source_dtype: bool = False,
        sharded_step: Any | None = None,
        retained_bytes_log_every: int = 0,
        source_pinned_arena: PinnedSourceArena | None = None,
        direct_when_group_bound: bool = False,
        source_release_during_backward: bool = False,
        combine_known_reward_sources: bool = False,
    ) -> "DataParallelThunderSync":
        """torchrun entry point: init NCCL, pin the device, sync the replicas.

        Reads LOCAL_RANK (torchrun's contract), pins this process to
        `cuda:LOCAL_RANK`, initializes the nccl process group if the caller
        has not, and **broadcasts rank 0's parameters and buffers** so every
        replica starts bitwise identical -- replicated DP is only correct if
        the replicas never diverge, and identical init + identical averaged
        grads + identical deterministic optimizers is the whole induction.
        The model must already be on `cuda:LOCAL_RANK` (or is moved there).
        """
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        if not dist.is_initialized():
            dist.init_process_group("nccl", device_id=device)
        if next(model.parameters()).device != device:
            model.to(device)

        # Rank 0's weights are THE weights. Broadcast params and buffers in
        # module order (identical architecture => identical order on every
        # rank). In-place writes under no_grad: this happens before any
        # RewardLinearBackward exists, so the version bumps are unobserved.
        with torch.no_grad():
            for t in model.state_dict().values():
                if isinstance(t, torch.Tensor) and t.numel():
                    dist.broadcast(t, src=0)
        return cls(
            model,
            optimizer,
            run_factory=run_factory,
            eps=eps,
            grad_clip=grad_clip,
            source_offload_device=source_offload_device,
            source_offload_mode=source_offload_mode,
            source_offload_window_bytes=source_offload_window_bytes,
            source_offload_hbm_threshold=source_offload_hbm_threshold,
            source_reduction_mode=source_reduction_mode,
            source_reduction_pack_size=source_reduction_pack_size,
            source_offload_pinned_windows=source_offload_pinned_windows,
            source_reduction_worker_count=source_reduction_worker_count,
            source_offload_worker_count=source_offload_worker_count,
            source_spill_directory=source_spill_directory,
            statistic_device=statistic_device,
            device_statistic_slots=device_statistic_slots,
            statistic_keep_source_dtype=statistic_keep_source_dtype,
            sharded_step=sharded_step,
            retained_bytes_log_every=retained_bytes_log_every,
            source_pinned_arena=source_pinned_arena,
            direct_when_group_bound=direct_when_group_bound,
            source_release_during_backward=source_release_during_backward,
            combine_known_reward_sources=combine_known_reward_sources,
        )

    # ------------------------------------------------------------ diagnostics

    def retained_device_bytes(self) -> dict[str, Any]:
        """What this trainer and its optimizer shard hold on the device.

        The run and the backward report themselves; this covers the rest:
        the trainer's own attributes (the model, the run, the backward and
        the shard are skipped -- they are reported separately) and the
        sharded step's, where the replicated parameters and the owned
        shard's optimizer state live.
        """

        from thundersync.engine.streaming import retained_bytes_by_attribute

        report: dict[str, Any] = {
            "trainer": retained_bytes_by_attribute(
                self, self.device, skip=("model", "run", "bw", "_sharded_step", "optimizer")
            ),
        }
        shard = self._sharded_step
        if shard is not None and hasattr(shard, "retained_device_bytes"):
            report["sharded_step"] = shard.retained_device_bytes()
        else:
            report["optimizer_state"] = retained_bytes_by_attribute(
                self.optimizer, self.device, skip=("param_groups",)
            )
        return report

    def _sample_retained(self, event: str) -> None:
        every = self._retained_bytes_log_every
        if not every:
            return
        self._retained_events += 1
        if self._retained_events % every:
            return
        device = self.device
        line = {
            "t_wall": time.time(),
            "t_mono": time.perf_counter(),
            "event": event,
            "events": self._retained_events,
            "allocated": int(torch.cuda.memory_allocated(device)),
            "reserved": int(torch.cuda.memory_reserved(device)),
            "max_allocated": int(torch.cuda.max_memory_allocated(device)),
            "run": self.run.retained_device_bytes(),
            "backward": self.bw.retained_device_bytes(),
            **self.retained_device_bytes(),
        }
        logger.info("[dp-trainer] retained %s", json.dumps(line, sort_keys=True))

    @contextlib.contextmanager
    def _abort_on_failure(self) -> Iterator[None]:
        """Abort the backward when the wrapped call raises, then re-raise.

        The original exception propagates unchanged. If the abort itself
        fails, that failure is logged and attached to the original
        exception as a note rather than replacing it.
        """

        try:
            yield
        except BaseException as error:
            try:
                self.bw.abort()
            except BaseException as abort_error:
                note = (
                    "aborting the reward-linear backward also failed: "
                    f"{type(abort_error).__name__}: {abort_error}"
                )
                logger.error(note, exc_info=abort_error)
                error.add_note(note)
            raise

    # ------------------------------------------------------------ event mirror

    def open_group(self, group_id: int, prompt_tokens: list[int]) -> None:
        """Open a LOCAL group: ids are rank-local, never coordinated."""
        self.run.open_group(group_id, prompt_tokens)
        self._prompt_tokens[group_id] = tuple(prompt_tokens)

    # ------------------------------------------------------- final work

    def configure_final_forest(
        self, *, max_tokens_per_microbatch: int, chunk: int
    ) -> None:
        """Run post-rollout final work as forest micro-batches.

        ``max_tokens_per_microbatch`` caps a micro-batch's logical tokens
        (the device's free memory may lower it per call); ``chunk`` is the
        fused log-prob's chunk, as the barrier objective takes it.
        """

        if max_tokens_per_microbatch < 1 or chunk < 1:
            raise ValueError("the final forest's token cap and chunk must be positive")
        if not self.direct_when_group_bound:
            raise ValueError(
                "the final forest runs final work, which only the direct "
                "path's admission (direct_when_group_bound) classifies"
            )
        self._final_forest = (int(max_tokens_per_microbatch), int(chunk))

    @property
    def final_forest_enabled(self) -> bool:
        return self._final_forest is not None

    def reserve_known_slabs(self, count: int) -> int:
        """Add ``count`` arena slabs that sources awaiting a reward never take.

        The content-ready budget keeps its own cap, so a verdict-known close
        always finds a slab no unbound source holds. Fails closed when the
        host budget the arena measured holds fewer slabs. Returns the cap.

        This is the trainer side of ``FinalFirstAdmission``'s
        ``known_reserve_slabs`` option (thundersync.grpo.admission), which
        gates known closes on ``known_slab_capacity``; reserve the same
        count here so those closes have slabs beyond the content budget.
        """

        if count < 1:
            raise ValueError("count must be positive")
        if self._known_reserve_slabs:
            raise RuntimeError("known-close slabs are already reserved")
        if self.source_pinned_arena is None:
            raise ValueError("known-close slabs reserve arena slabs; there is no arena")
        limit = self.source_pinned_arena.extend_slab_limit(count)
        self._known_reserve_slabs = int(count)
        return limit

    @property
    def known_slab_capacity(self) -> int | None:
        """Arena slabs no source awaiting its reward holds (None: no arena)."""
        return self.bw.known_slab_capacity

    def registered_groups_fully_bound(self) -> bool:
        """Has every trajectory of every local group a recorded reward?"""
        return self.bw.registered_groups_fully_bound()

    def configure_binary_rewards(self) -> None:
        self.bw.configure_binary_rewards()
        self._binary_rewards = True

    def set_statistics_branch_observer(
        self, observer: Callable[[int], None] | None
    ) -> None:
        self.bw.set_statistics_branch_observer(observer)

    def close_final_forest(self, items: list[tuple[int, float | None]]) -> None:
        """Run final closes as forest micro-batches into the gradients."""
        if self._final_forest is None:
            raise RuntimeError("the final forest is not configured")
        for trajectory_id, reward in items:
            if reward is None:
                raise ValueError(
                    f"trajectory {trajectory_id} closed without a reward"
                )
        max_tokens, chunk = self._final_forest
        with self._abort_on_failure():
            self.bw.close_final_forest(
                items,
                prompt_tokens=self._prompt_tokens,
                max_tokens_per_microbatch=max_tokens,
                chunk=chunk,
            )
            self._sample_retained("close_final_forest")

    @property
    def final_forest_max_tokens(self) -> int:
        if self._final_forest is None:
            raise RuntimeError("the final forest is not configured")
        return self._final_forest[0]

    # ------------------------------------------------------- work claims

    def final_work(self, trajectory_id: int) -> tuple[int, int, bytes]:
        """(logical tokens, price, payload) of a final trajectory not started.

        The payload carries what another rank needs to run ``A_i * S_i``:
        the group's prompt, the trajectory's turns and score masks, and
        ``A_i`` as this rank's float64 (``thundersync.grpo.claims``).
        """

        group_id = self.run.group_of(trajectory_id)
        prompt = self._prompt_tokens[group_id]
        turns = [
            (list(tokens), list(scored))
            for tokens, scored in self.run._trajs[trajectory_id].turns
        ]
        turn_tokens = sum(len(tokens) for tokens, _ in turns)
        payload = encode_final_work(
            group_id=group_id,
            trajectory_id=trajectory_id,
            advantage=self.bw.final_advantage(trajectory_id),
            prompt_tokens=prompt,
            turns=turns,
        )
        price = trajectory_price(len(prompt), turn_tokens, self._attention_weight())
        return len(prompt) + turn_tokens, price, payload

    def _attention_weight(self) -> float:
        if self._price_attention_weight is None:
            self._price_attention_weight = attention_weight(
                self.model.config, sum(p.numel() for p in self.model.parameters())
            )
        return self._price_attention_weight

    def cede_final_closes(self, items: list[tuple[int, float | None]]) -> None:
        """Final closes a peer claimed: entered in their groups, never run here."""
        for trajectory_id, reward in items:
            if reward is None:
                raise ValueError(f"trajectory {trajectory_id} ceded without a reward")
        with self._abort_on_failure():
            self.bw.cede_final_closes(items)

    def run_claimed_work(self, payloads: list[bytes]) -> None:
        """Peers' final trajectories this rank claimed, as one forest call."""

        if self._final_forest is None:
            raise RuntimeError("claimed work runs on the final forest, which is not configured")
        trajectories, advantages = [], {}
        for payload in payloads:
            group_id, trajectory_id, advantage, prompt, turns = decode_final_work(payload)
            if trajectory_id in advantages:
                raise RuntimeError(f"trajectory {trajectory_id} claimed twice")
            advantages[trajectory_id] = advantage
            trajectories.append(forest_trajectory(group_id, trajectory_id, prompt, turns))
        max_tokens, chunk = self._final_forest
        with self._abort_on_failure():
            self.bw.backward_claimed_final(
                trajectories,
                advantages,
                max_tokens_per_microbatch=max_tokens,
                chunk=chunk,
            )
            self._sample_retained("run_claimed_work")

    def record_work_claims(self, report: dict[str, Any]) -> None:
        """The step's claims report, carried into the step's statistics."""
        self._work_claims_report = dict(report)

    def register_group_trajectories(
        self,
        group_id: int,
        trajectory_ids: list[int] | tuple[int, ...],
    ) -> None:
        self.bw.register_group_trajectories(group_id, trajectory_ids)

    def append_turn(
        self, group_id: int, traj_id: int, tokens: list[int], scored: list[bool]
    ) -> None:
        self.run.append_turn(group_id, traj_id, tokens, scored)

    def append_turns(
        self, items: list[tuple[int, int, list[int], list[bool]]]
    ) -> None:
        self.run.append_turns(items)

    def close_trajectory(
        self,
        traj_id: int,
        reward: float,
        *,
        token_weights: torch.Tensor | None = None,
    ) -> None:
        with self._abort_on_failure():
            self.bw.close_trajectory(traj_id, reward, token_weights=token_weights)
            self._sample_retained("close_trajectory")

    def close_trajectories(self, items: list[tuple[int, float | None]]) -> None:
        """Close objective-ready trajectories through one packed rebuild."""
        for trajectory_id, reward in items:
            if reward is None:
                raise ValueError(
                    f"trajectory {trajectory_id} closed without a reward; "
                    "this objective is reward-gated, so a missing coefficient "
                    "is a wiring error rather than a zero reward"
                )
        with self._abort_on_failure():
            self.bw.close_trajectories(items)
            self._sample_retained("close_trajectories")

    @property
    def pending_reduction_source_count(self) -> int:
        """Differentiated-but-unfolded sources resident on this rank."""
        return self.bw.pending_reduction_source_count

    @property
    def held_source_count(self) -> int:
        """Unfolded sources the residency budget counts on this rank."""
        return self.bw.held_source_count

    def prepare_trajectories(self, trajectory_ids: list[int]) -> None:
        """Run content-ready branch backwards through one packed rebuild."""
        with self._abort_on_failure():
            self.bw.prepare_trajectories(trajectory_ids)
            self._sample_retained("prepare_trajectories")

    def bind_rewards(self, items: list[tuple[int, float | None]]) -> None:
        """Attach landed rewards to prepared sources in readiness order.

        The whole batch is checked BEFORE any of it reaches the reducer.
        The reward-linear backward commits ``acc.closed`` for a trajectory before it converts
        that trajectory's reward, so refusing mid-batch would leave the
        earlier binds committed and the failing one recorded as closed with
        no reward -- a torn accumulator behind an exception.
        """
        for trajectory_id, reward in items:
            if reward is None:
                raise ValueError(
                    f"trajectory {trajectory_id} bound without a reward; "
                    "this objective is reward-gated, so a missing coefficient "
                    "is a wiring error rather than a zero reward"
                )
        with self._abort_on_failure():
            self.bw.bind_rewards(items)
            self._sample_retained("bind_rewards")

    def group_reduction_complete(self, group_id: int) -> bool:
        """Non-blocking readiness check for a deferred group close."""
        return self.bw.group_reduction_complete(group_id)

    def record_rewards(self, items: list[tuple[int, float | None]]) -> None:
        """Enter landed rewards in their groups' ledgers; nothing runs."""
        for trajectory_id, reward in items:
            if reward is None:
                raise ValueError(
                    f"trajectory {trajectory_id} recorded without a reward"
                )
        self.bw.record_rewards(items)

    def group_fully_bound(self, group_id: int) -> bool:
        """Has every trajectory of the local group a recorded reward?"""
        return self.bw.group_fully_bound(group_id)

    def wait_for_trajectory_reduction(
        self,
        group_id: int,
        trajectory_id: int,
    ) -> None:
        with self._abort_on_failure():
            self.bw.wait_for_trajectory_reduction(group_id, trajectory_id)

    def close_group(self, group_id: int) -> None:
        with self._abort_on_failure():
            self.bw.close_group(group_id)
        self._local_groups_closed += 1

    @property
    def local_groups_closed(self) -> int:
        """Groups this rank has closed since the last step (= G_r at step time)."""
        return self._local_groups_closed

    @property
    def optimizer_steps_completed(self) -> int:
        return self._steps_done

    # ------------------------------------------------------------------- step

    def step(
        self,
        *,
        gradient_observer: GradientObserver | None = None,
    ) -> dict:
        """The one collective point: average grads across ranks, step everywhere.

        Order and rationale:
          1. `assert_safe_to_step()` LOCALLY, before any collective -- a rank
             with open groups fails loudly on its own instead of deadlocking
             the world inside a barrier.
          2. barrier -- every rank's streamed backwards are complete.
          3. all-gather group counts, then scale each local group-sum so the
             subsequent rank average equals the global mean over groups.
          4. every trainable param with `grad is None` gets a zero grad, so
             the reduce list is structurally identical on every rank.
          5. all_reduce(ReduceOp.AVG) over all grads: the declared scaling
             makes this the mean over all N groups.
          6. compute the global norm and apply the frozen common clip after the
             all-reduce; an optional correctness observer may snapshot the
             reduced gradient immediately before and after this operation.
          7. optimizer.step() on every rank -- identical grads, identical
             optimizer state, identical result; no weight broadcast needed.
          8. zero_grad and re-arm a fresh (run, bw) for the next batch.

        Returns a stats dict with per-phase wall timings (this rank's view).
        """
        self.bw.assert_safe_to_step()

        stats: dict = {
            "rank": self.rank,
            "world_size": self.world_size,
            "n_local_groups": self._local_groups_closed,
        }

        torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        dist.barrier()
        torch.cuda.synchronize(self.device)
        t1 = time.perf_counter()
        stats["barrier_s"] = t1 - t0

        counts = torch.zeros(self.world_size, dtype=torch.long, device=self.device)
        counts[self.rank] = self._local_groups_closed
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        counts_l = counts.tolist()
        n_local = counts_l[self.rank]
        n_global = sum(counts_l)
        if n_global < 1 or any(value < 1 for value in counts_l):
            raise RuntimeError(f"invalid groups across ranks: {counts_l}")

        with torch.no_grad():
            if len(set(counts_l)) == 1:
                # Preserve the original equal-cardinality arithmetic exactly.
                scale = 1.0 / n_local
                for p in self._trainable:
                    if p.grad is not None:
                        if n_local > 1:
                            p.grad.div_(n_local)
            else:
                scale = self.world_size / n_global
                for p in self._trainable:
                    if p.grad is not None:
                        p.grad.mul_(scale)
            # structural identity of the reduce list across ranks: every rank
            # materializes a zero grad for EVERY trainable param that has none
            for p in self._trainable:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)

        torch.cuda.synchronize(self.device)
        t2 = time.perf_counter()
        stats["n_global_groups"] = n_global
        stats["group_counts_by_rank"] = counts_l
        stats["local_gradient_scale"] = scale
        stats["grad_bytes"] = sum(
            p.grad.numel() * p.grad.element_size() for p in self._trainable
        )
        stats["n_grad_tensors"] = len(self._trainable)
        if self._sharded_step is not None:
            if gradient_observer is not None:
                raise RuntimeError(
                    "gradient observers are not supported under the sharded "
                    "optimizer step"
                )
            shard_stats = self._sharded_step.step()
            torch.cuda.synchronize(self.device)
            t4 = time.perf_counter()
            stats["allreduce_s"] = shard_stats["allreduce_s"]
            stats["global_grad_norm"] = shard_stats["global_grad_norm"]
            stats["grad_clip"] = shard_stats["grad_clip"]
            stats["gradient_was_clipped"] = shard_stats["gradient_was_clipped"]
            stats["opt_step_s"] = shard_stats["opt_step_s"]
            stats["sharded_step"] = shard_stats
        else:
            allreduce_report = per_parameter_all_reduce_average(
                self._trainable
            )
            torch.cuda.synchronize(self.device)
            t3 = time.perf_counter()
            stats["allreduce_s"] = t3 - t2
            stats.update(allreduce_report)

            if gradient_observer is not None:
                gradient_observer("raw_reduced", self._trainable)
            global_norm = torch.nn.utils.clip_grad_norm_(
                self._trainable,
                self.grad_clip if self.grad_clip is not None else float("inf"),
                foreach=False,
            )
            torch.cuda.synchronize(self.device)
            global_norm_value = float(global_norm.detach())
            stats["global_grad_norm"] = global_norm_value
            stats["grad_clip"] = self.grad_clip
            stats["gradient_was_clipped"] = (
                self.grad_clip is not None and global_norm_value > self.grad_clip
            )
            if gradient_observer is not None:
                gradient_observer("clipped", self._trainable)

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(self.device)
            t4 = time.perf_counter()
            stats["opt_step_s"] = t4 - t3
        stats["step_total_s"] = t4 - t0
        stats["source_offload_report"] = self.bw.source_offload_report
        if self._work_claims_report is not None:
            stats["work_claims"] = self._work_claims_report
        stats["rebuild_path_counts"] = self.run.rebuild_path_counts
        stats["rebuild_checkpoint_offload"] = (
            self.run.rebuild_checkpoint_offload_diagnostics()
        )
        stats["gradient_accumulation_report"] = (
            self.run.parameter_adjoint_diagnostics()
        )
        # What the run still holds once the update is applied, by holder,
        # and what is left once the next batch's run has replaced it: the
        # second sample's live_runs and other_live_runs_* say whether the
        # spent run survived its replacement (a reference cycle between the
        # run, its boundaries' capture source and the backward keeps it for
        # the cyclic collector, not for refcounting), which is the shape a
        # step-over-step floor rise takes.
        stats["run_retained_bytes"] = self.run.retained_device_bytes()
        # What the backward itself still holds, by attribute.
        stats["backward_retained_bytes"] = self.bw.retained_device_bytes()
        stats["trainer_retained_bytes"] = self.retained_device_bytes()
        stats["attention_path_counts"] = attention_path_counts()
        stats["kernel_probes"] = kernel_probe_records()
        # The caching allocator's cumulative counters: a retry frees cached
        # blocks and synchronizes the device, and device alloc/free count
        # the segment (or expandable page) maps behind the step's work.
        allocator = torch.cuda.memory_stats(self.device)
        stats["cuda_allocator"] = {
            key: allocator.get(key)
            for key in (
                "num_alloc_retries",
                "num_ooms",
                "num_device_alloc",
                "num_device_free",
                "num_sync_all_streams",
            )
        }

        self._steps_done += 1
        self._new_batch()
        stats["run_retained_bytes_after_replacement"] = (
            self.run.retained_device_bytes()
        )
        return stats

    def shutdown(self) -> None:
        if self._shutdown:
            return
        try:
            self.bw.abort()
        finally:
            if self.source_pinned_arena is not None:
                self.source_pinned_arena.close()
            self._shutdown = True

    # --------------------------------------------------------------- internal

    def _new_batch(self) -> None:
        """Fresh run + backward for the next batch: the version guard re-arms
        against the post-step parameter versions, and the streaming state
        (chunk ids, trajectory registry) starts clean."""
        self.run = self.run_factory(self.model)
        self.bw = RewardLinearBackward(
            self.run,
            self.model,
            eps=self.eps,
            source_offload_device=self.source_offload_device,
            source_offload_mode=self.source_offload_mode,
            source_offload_window_bytes=self.source_offload_window_bytes,
            source_offload_hbm_threshold=self.source_offload_hbm_threshold,
            source_reduction_mode=self.source_reduction_mode,
            source_reduction_pack_size=self.source_reduction_pack_size,
            source_reduction_worker_count=self.source_reduction_worker_count,
            source_offload_worker_count=self.source_offload_worker_count,
            source_offload_pinned_windows=self.source_offload_pinned_windows,
            source_spill_directory=self.source_spill_directory,
            statistic_device=self.statistic_device,
            device_statistic_slots=self.device_statistic_slots,
            statistic_keep_source_dtype=self.statistic_keep_source_dtype,
            source_pinned_arena=self.source_pinned_arena,
            direct_when_group_bound=self.direct_when_group_bound,
            source_copy_during_backward=self.source_release_during_backward,
            source_release_during_backward=self.source_release_during_backward,
            combine_known_reward_sources=self.combine_known_reward_sources,
        )
        if self._binary_rewards:
            self.bw.configure_binary_rewards()
        self.bw.set_diagnostic_scope(rank=self.rank, step=self._steps_done + 1)
        self._local_groups_closed = 0
        self._prompt_tokens: dict[int, tuple[int, ...]] = {}
        self._work_claims_report: dict[str, Any] | None = None
