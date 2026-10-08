"""The OPD streaming objectives over the shared streaming trainer.

On-policy distillation reads the policy only through per-token logprobs of
the emitted tokens and a frozen teacher's logprobs on the same context, so
its coefficient is known once a trajectory (``OPDStream``) or a turn
(``OPDTurnStream``, the streaming objective) is complete: there is no reward and
hence no group barrier, and token-mean normalization is finalized once the
batch is closed.

Both adapters share the streaming event surface (`open_group` /
`append_turn` / `append_turns`, forwarded to the `StreamingRun` unchanged)
and the per-group boundary discipline: with `boundary_cut=True` a
trajectory's backward stops at the shared prompt's detached proxies, and
`close_group` backwards the prompt subgraph exactly once with the
accumulated boundary gradient. Without the cut, the first per-trajectory
`retain_graph=False` backward would free the shared prompt subgraph under
the group's remaining trajectories, so the cut is a requirement here, not an
optimisation.

Each adapter accumulates into ``.grad`` the gradient of the *loss* its
docstring writes down (descent direction); the caller owns the step.

One instance serves one generation batch, like the `StreamingRun` it wraps.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import torch

from thundersync.accel.operator_profiling import device_role, operator_range
from thundersync.engine.streaming import STREAM_ATTENTION_NAME, StreamingRun, register_stream_attention


class _StreamAdapter:
    """The shared event surface: forward events to the run, then hook.

    The adapters read the run ONLY through its public API -- events in,
    per-token logprobs out, plus the boundary/freeing calls. They never touch
    the forest, the plan, or the K/V store; that boundary is enforced by tests, not merely documented.
    """

    def __init__(self, run: StreamingRun) -> None:
        self.run = run

    def open_group(self, group_id: int, prompt_tokens: list[int]) -> None:
        self.run.open_group(group_id, prompt_tokens)
        self._on_open(group_id, list(prompt_tokens))

    def open_groups_shared_prefix(
        self,
        prompts: dict[int, list[int]],
        *,
        internal_nodes: bool = True,
        coalesce_nodes: bool = False,
        max_coalesced_nodes: int | None = None,
    ) -> int:
        """Open a compatible cross-task prompt domain on the streaming run."""
        shared = self.run.open_groups_shared_prefix(
            prompts,
            internal_nodes=internal_nodes,
            coalesce_nodes=coalesce_nodes,
            max_coalesced_nodes=max_coalesced_nodes,
        )
        for group_id, prompt_tokens in sorted(prompts.items()):
            self._on_open(group_id, list(prompt_tokens))
        return shared

    def append_turn(
        self, group_id: int, traj_id: int, tokens: list[int], scored: list[bool]
    ) -> None:
        self.append_turns([(group_id, traj_id, tokens, scored)])

    def append_turns(
        self, items: list[tuple[int, int, list[int], list[bool]]]
    ) -> None:
        self.run.append_turns(items)
        self._on_turns(items)

    # hooks, run AFTER the run accepted the event (so a raised event records nothing)
    def _on_open(self, group_id: int, prompt_tokens: list[int]) -> None:
        pass

    def _on_turns(
        self, items: list[tuple[int, int, list[int], list[bool]]]
    ) -> None:
        pass


class _ImmediateBackward(_StreamAdapter):
    """Per-trajectory immediate backward, for a coefficient that is a
    constant at close time (OPD's detached teacher).

    No G1/G2 buffers: the scaled loss backwards straight into ``param.grad``
    and the boundary proxies' ``.grad``; `close_group` backwards the shared
    prompt subgraph once with the accumulated proxy gradients and frees the
    group. Safety rails match the GRPO reward-linear backward's: the version guard turns an optimizer
    step over open groups into a loud error at the next close, never a silent
    corruption.
    """

    supports_source_ready_serial = False

    def __init__(self, run: StreamingRun, model: torch.nn.Module) -> None:
        if not run.boundary_cut:
            raise ValueError(
                f"{type(self).__name__} requires StreamingRun(boundary_cut=True): "
                "per-trajectory retain_graph=False backwards through an uncut "
                "shared prompt would free it under the group's other trajectories"
            )
        if (
            run.shared_prefix_vjp_mode == "source_ready_serial"
            and not self.supports_source_ready_serial
        ):
            raise ValueError(
                f"{type(self).__name__} does not support "
                "shared_prefix_vjp_mode='source_ready_serial'"
            )
        super().__init__(run)
        self.params = [p for p in model.parameters() if p.requires_grad]
        if any(parameter.grad is not None for parameter in self.params):
            raise ValueError(
                f"{type(self).__name__} requires an empty parameter-gradient "
                "accumulator at construction"
            )
        self._versions = [p._version for p in self.params]
        self._closed: dict[int, set[int]] = {}

    def append_turns(
        self, items: list[tuple[int, int, list[int], list[bool]]]
    ) -> None:
        for group_id, traj_id, _tokens, _scored in items:
            if traj_id in self._closed.get(group_id, set()):
                raise RuntimeError(f"trajectory {traj_id} already closed")
        super().append_turns(items)

    def _ensure_trajectory_can_close(self, traj_id: int) -> int:
        """Reject duplicate closure before any rebuild or state mutation."""
        self._check_versions()
        gid = self.run.group_of(traj_id)
        if traj_id in self._closed.get(gid, set()):
            raise RuntimeError(f"trajectory {traj_id} already closed")
        return gid

    def _register_close(self, traj_id: int) -> int:
        gid = self._ensure_trajectory_can_close(traj_id)
        closed = self._closed.setdefault(gid, set())
        closed.add(traj_id)
        return gid

    def close_group(self, group_id: int) -> None:
        """Every trajectory closed: backward the prompt once, free the group."""
        self._check_versions()
        closed = self._closed.pop(group_id, None)
        if closed is None:
            raise RuntimeError(f"group {group_id}: no closed trajectories")
        open_trajs = [
            t for t in self.run.trajectories_of(group_id) if t not in closed
        ]
        if open_trajs:
            self._closed[group_id] = closed
            raise RuntimeError(
                f"group {group_id}: trajectories {open_trajs} not closed"
            )
        if self.run.shared_prefix_vjp_mode != "source_ready_serial":
            outputs, gradients = self.run.consume_boundary_adjoint_pairs(group_id)
            if outputs:
                self.run.backward_with_boundary_adjoint_capture(
                    outputs,
                    grad_tensors=gradients,
                    source=("group", group_id),
                )
        self.run.free_group(group_id)
        self._on_group_closed(group_id)

    def _on_group_closed(self, group_id: int) -> None:
        pass

    @property
    def open_groups(self) -> list[int]:
        return sorted(self.run.open_group_ids())

    def assert_safe_to_step(self) -> None:
        self._check_versions()
        if self.open_groups:
            raise RuntimeError(
                f"optimizer step with open groups {self.open_groups}: their "
                "retained graphs reference current parameter values and would "
                "be silently invalidated"
            )
        self.run.finalize_shared_prefixes()

    def _check_versions(self) -> None:
        for p, v in zip(self.params, self._versions, strict=True):
            if p._version != v:
                raise RuntimeError(
                    "a parameter was modified in place (optimizer step?) while "
                    "streamed groups were open; the retained graphs are invalid. "
                    "Close all groups before stepping -- see assert_safe_to_step()."
                )


class OPDStream(_ImmediateBackward):
    """SkyRL-style sampled reverse-KL on the streaming schedule.

    Let ``b`` be the behavior policy that sampled token ``y``, ``p`` the
    current student, and ``q`` the frozen teacher.  The SkyRL OPD recipe
    constructs ``advantage = log q(y) - log b(y)`` and applies its importance-
    sampling policy loss.  This class writes the equivalent loss numerator as

        L_num = sum_t w_t * exp(log p_t - log b_t)
                          * stopgrad(log b_t - log q_t).

    At the on-policy point ``p == b``, its gradient is

        sum_t w_t * (log b_t - log q_t) * grad(log p_t),

    the score-function estimator of
    ``grad KL(p(.|s) || q(.|s))`` at each sampled context ``s``.  The context
    occupancy is treated as fixed, as in SkyRL's token-level recipe; this is
    therefore a conditional-KL semi-gradient rather than the gradient of a
    full autoregressive trajectory KL.  The omitted ``+1`` score term has zero
    expectation within a context and is a constant baseline.  Keeping the
    importance ratio in the computation graph is required: directly
    differentiating ``log p - detached(log q)`` produces a teacher-independent
    gradient.

    ``StreamingRun.logprobs`` contains scored policy tokens only, so prompt and
    observation tokens are excluded before this objective.  ``token_weights``
    can apply an additional non-negative mask or weighting.  The default
    ``normalization="token_mean"`` divides the accumulated batch gradient by
    the sum of these weights (or by the number of scored tokens).  Because the
    final token count is unknown at the first completion, backwards execute on
    the unnormalized numerator immediately and :meth:`assert_safe_to_step`
    applies the one deferred scalar division after every group is closed.

    OPD has no reward dependency: `close_trajectory` fires the trajectory's
    backward at completion through the retained graph with
    ``retain_graph=False`` and frees its K/V and activations. `close_group`
    only marks that the shared prompt's last reader finished, so the prompt
    subgraph can take its single backward and be freed.

    Teacher logprobs, either way:
    * pass ``teacher=`` a frozen model at construction -- it is streamed under
      ``torch.no_grad`` in a second, plain (non-checkpoint, no-cut)
      `StreamingRun`, fed by the same mirrored events, its per-token logprobs
      read at close; or
    * pass ``teacher_logprobs=`` per trajectory at `close_trajectory` (already
      computed elsewhere, e.g. offline).
    """

    supports_source_ready_serial = True

    def __init__(
        self,
        run: StreamingRun,
        model: torch.nn.Module,
        *,
        teacher: torch.nn.Module | None = None,
        normalization: str = "token_mean",
        joint_backward: bool = False,
        closure_cohort_vjp: bool | None = None,
        ragged_cohort_attention: bool = False,
        concurrent_teacher_replay: bool = False,
    ) -> None:
        super().__init__(run, model)
        if normalization not in ("token_mean", "sum"):
            raise ValueError(
                "OPD normalization must be 'token_mean' or 'sum', got "
                f"{normalization!r}"
            )
        self.normalization = normalization
        if closure_cohort_vjp is not None:
            if joint_backward and not closure_cohort_vjp:
                raise ValueError(
                    "joint_backward and closure_cohort_vjp disagree"
                )
            joint_backward = bool(closure_cohort_vjp)
        self.joint_backward = bool(joint_backward)
        self.closure_cohort_vjp = self.joint_backward
        if self.closure_cohort_vjp and run.rebuild_block_tokens is None:
            raise ValueError(
                "closure cohort VJP requires rebuild_block_tokens to be set"
            )
        if ragged_cohort_attention and not self.joint_backward:
            raise ValueError(
                "ragged cohort attention requires joint_backward=True"
            )
        if ragged_cohort_attention != run.ragged_cohort_attention:
            raise ValueError(
                "OPDStream and StreamingRun must select the same ragged "
                "cohort-attention setting"
            )
        self.ragged_cohort_attention = ragged_cohort_attention
        self.concurrent_teacher_replay = bool(concurrent_teacher_replay)
        self._normalization_mass = 0.0
        self._loss_numerator = 0.0
        self._normalization_finalized = False
        self._concurrent_replay_lock = threading.Lock()
        self._concurrent_replay_error: BaseException | None = None
        self._teacher_executor: ThreadPoolExecutor | None = None
        self._teacher_run: StreamingRun | None = None
        if teacher is not None:
            register_stream_attention()
            teacher.config._attn_implementation = STREAM_ATTENTION_NAME
            # Mirror the student's firing mode.  In the
            # ``stream_turns=False`` path both models retain their prompt once,
            # record arriving turns on the CPU, and rebuild only the completed
            # branch at close.  The teacher is frozen, so its rebuild remains
            # under ``no_grad`` and does not need a boundary cut.
            self._teacher_run = StreamingRun(
                teacher,
                boundary_cut=True,
                checkpoint=(
                    run.checkpoint
                    and run.stream_turns
                    and self.concurrent_teacher_replay
                ),
                stream_turns=run.stream_turns,
                rebuild_block_tokens=run.rebuild_block_tokens,
            )
        if self.concurrent_teacher_replay:
            if self._teacher_run is None:
                raise ValueError(
                    "concurrent teacher replay requires teacher= at construction"
                )
            student_device = self.run.weight.device
            teacher_device = self._teacher_run.weight.device
            if (
                student_device.type != "cuda"
                or teacher_device.type != "cuda"
                or student_device == teacher_device
            ):
                raise ValueError(
                    "concurrent teacher replay requires student and teacher on "
                    "distinct CUDA devices"
                )
            if teacher.training or any(
                parameter.requires_grad for parameter in teacher.parameters()
            ):
                raise ValueError(
                    "concurrent teacher replay requires a frozen teacher in "
                    "evaluation mode"
                )
            active_dropout = [
                name
                for name, module in model.named_modules()
                if isinstance(module, torch.nn.Dropout)
                and module.training
                and module.p > 0
            ]
            if active_dropout:
                raise ValueError(
                    "concurrent teacher replay requires deterministic student "
                    "replay; active dropout modules: "
                    f"{active_dropout[:4]}"
                )
            self._teacher_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="opd-teacher-replay"
            )

    def _raise_if_concurrent_replay_failed(self) -> None:
        if self._concurrent_replay_error is not None:
            raise RuntimeError(
                "concurrent student-teacher replay failed; discard this OPD "
                "step and its gradients"
            ) from self._concurrent_replay_error

    def _poison_concurrent_replay(self, error: BaseException) -> None:
        if self.concurrent_teacher_replay:
            self._concurrent_replay_error = error

    def shutdown_concurrent_replay(self) -> None:
        """Release the optional teacher worker after one OPD step completes."""
        if self._teacher_executor is not None:
            self._teacher_executor.shutdown(wait=True)
            self._teacher_executor = None

    def _concurrent_student_teacher(
        self,
        student_fn: Callable[[], Any],
        teacher_fn: Callable[[], Any],
    ) -> tuple[Any, Any]:
        """Execute one student pass and one frozen-teacher pass concurrently.

        The two ``StreamingRun`` instances use distinct CUDA devices. Their
        attention/checkpoint contexts are device-indexed, so each worker owns
        one run and one device. The student pass remains on the caller thread;
        only the frozen, no-grad teacher pass uses the temporary worker.

        Both passes are joined before returning. Any failure invalidates the
        objective because a CUDA/autograd failure may leave partially created
        state or gradients; subsequent closure and optimizer finalization are
        rejected.
        """
        self._raise_if_concurrent_replay_failed()
        if not self._concurrent_replay_lock.acquire(blocking=False):
            raise RuntimeError(
                "concurrent replay already active for this OPDStream"
            )
        if self._teacher_run is None:
            raise RuntimeError("this OPD stream has no teacher run")
        student_device = self.run.weight.device
        teacher_device = self._teacher_run.weight.device

        def run_teacher() -> Any:
            # CUDA's current-device state is thread-local. Use a restoring
            # context rather than a permanent set_device so this worker cannot
            # leak device state into later executor submissions.
            with torch.cuda.device(teacher_device), torch.no_grad():
                result = teacher_fn()
                # CUDA launch is asynchronous. Complete this device's pass
                # inside its owner thread so errors are observed and the
                # returned tensor is ready before another thread consumes it.
                torch.cuda.synchronize(teacher_device)
            return result

        student_result = teacher_result = None
        student_error: BaseException | None = None
        teacher_error: BaseException | None = None
        try:
            if self._teacher_executor is None:
                raise RuntimeError("this OPD stream has no teacher executor")
            teacher_future = self._teacher_executor.submit(run_teacher)
            try:
                # The caller can itself be a queue worker whose current CUDA
                # device has never been initialized. Tensor-dispatched PyTorch
                # operations usually select the correct device, while custom
                # attention/checkpoint kernels may inspect the thread-local
                # device. Bind and restore it explicitly for the full student
                # graph construction.
                with torch.cuda.device(student_device), torch.enable_grad():
                    student_result = student_fn()
                    torch.cuda.synchronize(student_device)
            except BaseException as exc:
                student_error = exc
            try:
                teacher_result = teacher_future.result()
            except BaseException as exc:
                teacher_error = exc
            error = student_error if student_error is not None else teacher_error
            if error is not None:
                self._concurrent_replay_error = error
                raise RuntimeError(
                    "concurrent student-teacher replay failed; discard this OPD "
                    "step and its gradients"
                ) from error
            return student_result, teacher_result
        finally:
            self._concurrent_replay_lock.release()

    # ------------------------------------------------------------- event hooks

    def _on_open(self, group_id: int, prompt_tokens: list[int]) -> None:
        self._raise_if_concurrent_replay_failed()
        if self._teacher_run is not None:
            with torch.no_grad():
                self._teacher_run.open_group(group_id, prompt_tokens)

    def open_group(self, group_id: int, prompt_tokens: list[int]) -> None:
        """Open one student/teacher prompt, optionally on separate GPUs."""
        self._raise_if_concurrent_replay_failed()
        if self._teacher_run is None or not self.concurrent_teacher_replay:
            return super().open_group(group_id, prompt_tokens)

        def open_student() -> None:
            self.run.open_group(group_id, prompt_tokens)

        def open_teacher() -> None:
            if self._teacher_run is None:
                raise RuntimeError("this OPD stream has no teacher run")
            self._teacher_run.open_group(group_id, prompt_tokens)

        self._concurrent_student_teacher(open_student, open_teacher)

    def open_groups_shared_prefix(
        self,
        prompts: dict[int, list[int]],
        *,
        internal_nodes: bool = True,
        coalesce_nodes: bool = False,
        max_coalesced_nodes: int | None = None,
    ) -> int:
        """Open the same exact prompt domain for student and frozen teacher."""
        self._raise_if_concurrent_replay_failed()

        def open_student() -> int:
            return self.run.open_groups_shared_prefix(
                prompts,
                internal_nodes=internal_nodes,
                coalesce_nodes=coalesce_nodes,
                max_coalesced_nodes=max_coalesced_nodes,
            )

        def open_teacher() -> int:
            if self._teacher_run is None:
                raise RuntimeError("this OPD stream has no teacher run")
            return self._teacher_run.open_groups_shared_prefix(
                prompts,
                internal_nodes=internal_nodes,
                coalesce_nodes=coalesce_nodes,
                max_coalesced_nodes=max_coalesced_nodes,
            )

        if self._teacher_run is not None and self.concurrent_teacher_replay:
            shared, teacher_shared = self._concurrent_student_teacher(
                open_student, open_teacher
            )
        else:
            shared = open_student()
            if self._teacher_run is not None:
                with torch.no_grad():
                    teacher_shared = open_teacher()
            else:
                teacher_shared = None
        if self._teacher_run is not None:
            if teacher_shared is None:
                raise RuntimeError("the teacher run did not open the shared prefix")
            if teacher_shared != shared:
                raise RuntimeError(
                    "student and teacher derived different shared prompt lengths"
                )
            student_topology = self.run.prompt_token_accounting()
            teacher_topology = self._teacher_run.prompt_token_accounting()
            if student_topology != teacher_topology:
                raise RuntimeError(
                    "student and teacher derived different prompt forests"
                )
        return shared

    def _on_turns(
        self, items: list[tuple[int, int, list[int], list[bool]]]
    ) -> None:
        self._raise_if_concurrent_replay_failed()
        if self._teacher_run is not None:
            with torch.no_grad():
                self._teacher_run.append_turns(items)

    def append_turns(
        self, items: list[tuple[int, int, list[int], list[bool]]]
    ) -> None:
        """Append one OPD block, optionally scoring both models concurrently.

        In streamed-block mode the student retains one checkpoint graph per
        block while the frozen teacher retains only its no-grad state and
        scores.  Distinct-device execution preserves the existing fixed-policy
        objective: both passes consume the same immutable token range before
        the method returns, and the parameter-version guard remains active
        until the logical update batch closes.
        """
        if not (
            self.concurrent_teacher_replay
            and self.run.stream_turns
            and self._teacher_run is not None
        ):
            return super().append_turns(items)
        self._raise_if_concurrent_replay_failed()
        self._check_versions()
        for group_id, traj_id, _tokens, _scored in items:
            if traj_id in self._closed.get(group_id, set()):
                raise RuntimeError(f"trajectory {traj_id} already closed")

        def append_student() -> None:
            self.run.append_turns(items)

        def append_teacher() -> None:
            if self._teacher_run is None:
                raise RuntimeError("this OPD stream has no teacher run")
            self._teacher_run.append_turns(items)

        self._concurrent_student_teacher(append_student, append_teacher)

    # ------------------------------------------------------------------ close

    def close_trajectory(
        self,
        traj_id: int,
        *,
        teacher_logprobs: torch.Tensor | None = None,
        token_weights: torch.Tensor | None = None,
    ) -> float | None:
        """The trajectory completed; backward its OPD numerator immediately.

        No reward exists in this objective, so nothing later than completion
        gates the backward. Returns the trajectory's unnormalized numerator
        contribution (None if nothing was scored). After all groups close,
        :attr:`loss_value` reports the token-mean or summed batch value.
        """
        if self._normalization_finalized:
            raise RuntimeError("cannot close a trajectory after OPD finalization")
        self._raise_if_concurrent_replay_failed()
        group_id = self._ensure_trajectory_can_close(traj_id)
        # thundersync-final does not retain a per-turn student graph.  At completion it
        # rebuilds the branch once against the retained shared prompt, which
        # defines both the graph-attached current logprobs and their detached
        # behavior-policy twins at the same parameter version.
        if (
            teacher_logprobs is None
            and self._teacher_run is not None
            and self.concurrent_teacher_replay
            and not self.run.stream_turns
        ):
            lp, teacher_lp = self._concurrent_student_teacher(
                lambda: self.run.logprobs(traj_id, rebuild=True),
                lambda: self._teacher_run.logprobs(traj_id, rebuild=True),
            )
        else:
            lp = self.run.logprobs(traj_id, rebuild=not self.run.stream_turns)
            teacher_lp = None
        try:
            old = self.run.old_logprobs(traj_id)
            if old.shape != lp.shape or not torch.equal(lp.detach(), old):
                raise RuntimeError(
                    "OPD requires on-policy behavior logprobs from the same "
                    "streamed forward; current and behavior logprobs differ"
                )
            if teacher_logprobs is None:
                if self._teacher_run is None:
                    raise ValueError(
                        "no teacher: construct OPDStream(teacher=model) or pass "
                        "teacher_logprobs to close_trajectory"
                    )
                if teacher_lp is None:
                    with torch.no_grad():
                        teacher_lp = self._teacher_run.logprobs(
                            traj_id,
                            rebuild=not self._teacher_run.stream_turns,
                        )
            else:
                teacher_lp = teacher_logprobs
            teacher_lp = teacher_lp.detach().to(device=lp.device, dtype=lp.dtype)
            if teacher_lp.shape != lp.shape:
                raise ValueError(
                    f"teacher logprobs have shape {tuple(teacher_lp.shape)}, "
                    f"scored student logprobs have {tuple(lp.shape)}"
                )
            if not bool(torch.isfinite(teacher_lp).all()):
                raise ValueError("teacher logprobs must be finite")

            if token_weights is None:
                weights = torch.ones_like(lp)
            else:
                if token_weights.requires_grad:
                    raise ValueError("OPD token_weights must be detached")
                weights = token_weights.detach().to(
                    device=lp.device, dtype=lp.dtype
                )
                if weights.shape != lp.shape:
                    raise ValueError(
                        f"token_weights have shape {tuple(weights.shape)}, "
                        f"scored student logprobs have {tuple(lp.shape)}"
                    )
                if not bool(torch.isfinite(weights).all()):
                    raise ValueError("OPD token_weights must be finite")
                if bool((weights < 0).any()):
                    raise ValueError("OPD token_weights must be non-negative")
        except BaseException as exc:
            # The rebuild has already materialized graph state and
            # pi_old. A corrected retry would reuse partially changed state.
            self._poison_concurrent_replay(exc)
            raise

        value = None
        if lp.numel():
            ratio = torch.exp(lp - old)
            reverse_kl_score = (old - teacher_lp).detach()
            loss = (weights * ratio * reverse_kl_score).sum()
            value = float(loss.detach())
            try:
                # retain_graph=False: the branch graph is released here.
                with operator_range(
                    "opd_suffix_backward",
                    role=device_role(lp.device),
                    device=lp.device,
                    trajectories=traj_id,
                    trajectory_count=1,
                    scored_tokens=lp.numel(),
                ):
                    self.run.backward_with_boundary_adjoint_capture(
                        loss,
                        source=("trajectory", traj_id),
                    )
            except BaseException as exc:
                self._poison_concurrent_replay(exc)
                raise
        if self.run.shared_prefix_vjp_mode == "source_ready_serial":
            self.run.drain_shared_prefix_source(
                group_id,
                source=("trajectory", traj_id),
            )
        # Commit closure and accounting only after a successful backward. A
        # CUDA/autograd failure can still leave partial gradients; concurrent
        # mode records that failure and rejects optimizer finalization.
        self._register_close(traj_id)
        if value is not None:
            self._loss_numerator += value
        self._normalization_mass += float(
            weights.detach().to(dtype=torch.float64).sum().cpu()
        )
        self.run.free_trajectory(traj_id)
        if self._teacher_run is not None:
            self._teacher_run.free_trajectory(traj_id)
        return value

    def close_trajectories(
        self,
        traj_ids: list[int],
        *,
        teacher_logprobs: dict[int, torch.Tensor] | None = None,
        token_weights: dict[int, torch.Tensor] | None = None,
    ) -> dict[int, float | None]:
        """Close queued branches through one packed rebuild.

        This is the OPD counterpart of the GRPO multi-trajectory closure.
        Branches that became ready while the trainer was occupied share each
        packed rebuild round. Their OPD numerators remain independent and add
        before one backward, while batch normalization is still deferred until
        every group closes.  When ``joint_backward`` was enabled at
        construction, each layer's checkpoint recompute and VJP also execute
        once for the complete ready cohort.  That graph is safe here because
        this method reduces every cohort loss into the single backward below.
        """
        if self._normalization_finalized:
            raise RuntimeError("cannot close trajectories after OPD finalization")
        if not traj_ids:
            raise ValueError("close_trajectories needs at least one trajectory")
        if len(set(traj_ids)) != len(traj_ids):
            raise ValueError("a trajectory appears twice in one OPD close batch")
        if self.run.stream_turns:
            raise ValueError(
                "packed OPD closure requires StreamingRun(stream_turns=False)"
            )
        if teacher_logprobs is not None:
            missing = sorted(set(traj_ids) - set(teacher_logprobs))
            if missing:
                raise ValueError(
                    f"teacher_logprobs missing trajectories {missing}"
                )
        token_weights = token_weights or {}
        unknown_weights = sorted(set(token_weights) - set(traj_ids))
        if unknown_weights:
            raise ValueError(
                f"token_weights contains trajectories outside this close "
                f"batch: {unknown_weights}"
            )
        for weights in token_weights.values():
            if weights.requires_grad:
                raise ValueError("OPD token_weights must be detached")
            if not bool(torch.isfinite(weights).all()):
                raise ValueError("OPD token_weights must be finite")
            if bool((weights < 0).any()):
                raise ValueError("OPD token_weights must be non-negative")
        if len(traj_ids) == 1:
            tid = traj_ids[0]
            value = self.close_trajectory(
                tid,
                teacher_logprobs=(
                    None if teacher_logprobs is None else teacher_logprobs[tid]
                ),
                token_weights=(
                    None if token_weights is None else token_weights.get(tid)
                ),
            )
            return {tid: value}
        if teacher_logprobs is None and self._teacher_run is None:
            raise ValueError(
                "no teacher: construct OPDStream(teacher=model) or pass a "
                "teacher_logprobs mapping"
            )
        for tid in traj_ids:
            self._ensure_trajectory_can_close(tid)
        self._raise_if_concurrent_replay_failed()
        source_ready_group_id: int | None = None
        if self.run.shared_prefix_vjp_mode == "source_ready_serial":
            group_ids = {self.run.group_of(tid) for tid in traj_ids}
            if len(group_ids) != 1:
                raise ValueError(
                    "source-ready serial mode cannot drain one trajectory "
                    "cohort spanning multiple groups"
                )
            source_ready_group_id = next(iter(group_ids))
        source = ("trajectory_cohort", tuple(sorted(traj_ids)))

        def rebuild_student() -> dict[int, torch.Tensor]:
            return self.run.rebuild_logprobs(
                traj_ids,
                closure_cohort_vjp=self.closure_cohort_vjp,
            )

        def rebuild_teacher() -> dict[int, torch.Tensor]:
            if self._teacher_run is None:
                raise RuntimeError("this OPD stream has no teacher run")
            return self._teacher_run.rebuild_logprobs(traj_ids)

        if (
            teacher_logprobs is None
            and self.concurrent_teacher_replay
        ):
            current_by_tid, teacher_by_tid = self._concurrent_student_teacher(
                rebuild_student, rebuild_teacher
            )
        else:
            current_by_tid = rebuild_student()
            if teacher_logprobs is None:
                if self._teacher_run is None:
                    raise RuntimeError("this OPD stream has no teacher run")
                with torch.no_grad():
                    teacher_by_tid = rebuild_teacher()
            else:
                teacher_by_tid = teacher_logprobs

        losses: list[torch.Tensor] = []
        values: dict[int, float | None] = {}
        masses: dict[int, float] = {}
        try:
            for tid in traj_ids:
                lp = current_by_tid[tid]
                old = self.run.old_logprobs(tid)
                if old.shape != lp.shape or not torch.equal(lp.detach(), old):
                    raise RuntimeError(
                        "OPD requires on-policy behavior logprobs from the same "
                        "packed rebuild"
                    )
                teacher_lp = teacher_by_tid[tid].detach().to(
                    device=lp.device, dtype=lp.dtype
                )
                if teacher_lp.shape != lp.shape:
                    raise ValueError(
                        f"trajectory {tid}: teacher logprobs have shape "
                        f"{tuple(teacher_lp.shape)}, scored student logprobs have "
                        f"{tuple(lp.shape)}"
                    )
                if not bool(torch.isfinite(teacher_lp).all()):
                    raise ValueError("teacher logprobs must be finite")
                supplied_weights = token_weights.get(tid)
                weights = (
                    torch.ones_like(lp)
                    if supplied_weights is None
                    else supplied_weights.detach().to(
                        device=lp.device, dtype=lp.dtype
                    )
                )
                if weights.shape != lp.shape:
                    raise ValueError(
                        f"trajectory {tid}: token_weights have shape "
                        f"{tuple(weights.shape)}, scored student logprobs have "
                        f"{tuple(lp.shape)}"
                    )
                masses[tid] = float(
                    weights.detach().to(dtype=torch.float64).sum().cpu()
                )
                if not lp.numel():
                    values[tid] = None
                    continue
                ratio = torch.exp(lp - old)
                reverse_kl_score = (old - teacher_lp).detach()
                loss = (weights * ratio * reverse_kl_score).sum()
                value = float(loss.detach())
                values[tid] = value
                losses.append(loss)
        except BaseException as exc:
            self._poison_concurrent_replay(exc)
            raise

        # All caller-supplied tensors have now been validated. Backward precedes
        # the closure-state commit, so a failed traversal cannot make the
        # trajectory appear successfully closed.
        if losses:
            try:
                with operator_range(
                    "opd_suffix_backward",
                    role=device_role(losses[0].device),
                    device=losses[0].device,
                    trajectories=",".join(str(tid) for tid in traj_ids),
                    trajectory_count=len(traj_ids),
                    scored_tokens=sum(
                        current_by_tid[tid].numel() for tid in traj_ids
                    ),
                ):
                    self.run.backward_with_boundary_adjoint_capture(
                        torch.stack(losses).sum(),
                        source=source,
                    )
            except BaseException as exc:
                self._poison_concurrent_replay(exc)
                raise
        if source_ready_group_id is not None:
            self.run.drain_shared_prefix_source(
                source_ready_group_id,
                source=source,
            )
        for tid in traj_ids:
            self._register_close(tid)
            self.run.free_trajectory(tid)
            if self._teacher_run is not None:
                self._teacher_run.free_trajectory(tid)
            value = values[tid]
            if value is not None:
                self._loss_numerator += value
            self._normalization_mass += masses[tid]
        return values

    def _on_group_closed(self, group_id: int) -> None:
        if self._teacher_run is not None:
            self._teacher_run.free_group(group_id)

    def assert_safe_to_step(self) -> None:
        """Verify closure and finalize the declared batch normalization.

        Token-mean scaling is deferred so trajectory backwards can still fire
        before the final batch token count is known.  One OPDStream instance
        therefore owns the parameter-gradient accumulator for one generation
        batch; callers must not mix unrelated gradients before this method.
        """
        self._raise_if_concurrent_replay_failed()
        super().assert_safe_to_step()
        if self._teacher_run is not None:
            self._teacher_run.finalize_shared_prefixes()
        if self._normalization_finalized:
            return
        divisor = self.normalization_divisor
        if divisor != 1.0:
            for parameter in self.params:
                if parameter.grad is not None:
                    parameter.grad.div_(divisor)
        self._normalization_finalized = True
        self.shutdown_concurrent_replay()

    @property
    def normalization_divisor(self) -> float:
        """Final scalar divisor; SkyRL's token mean clamps an empty mask to 1."""
        if self.normalization == "sum":
            return 1.0
        return max(self._normalization_mass, 1.0)

    @property
    def loss_value(self) -> float:
        """Normalized batch loss after :meth:`assert_safe_to_step`."""
        if not self._normalization_finalized:
            raise RuntimeError("OPD loss_value requires assert_safe_to_step first")
        return self._loss_numerator / self.normalization_divisor


class OPDTurnStream(_ImmediateBackward):
    """Exact turn-arrival OPD with deferred state-adjoint replay.

    OPD supervision for a scored token depends only on its realized causal
    context and the frozen teacher, so the turn-local loss can backward as soon
    as the teacher score arrives. ``StreamingRun(turn_boundary_rebuild=True)``
    cuts each turn's outgoing state. Later turns accumulate adjoints on those
    proxies; trajectory closure rebuilds the turns in reverse and propagates
    the deferred state adjoints exactly. Parameters remain fixed until the
    declared update batch closes and one global token divisor is applied.
    """

    def __init__(
        self,
        run: StreamingRun,
        model: torch.nn.Module,
        *,
        normalization: str = "token_mean",
    ) -> None:
        if not run.turn_boundary_rebuild:
            raise ValueError(
                "OPDTurnStream requires StreamingRun("
                "turn_boundary_rebuild=True)"
            )
        if normalization not in ("token_mean", "sum"):
            raise ValueError(
                "OPD normalization must be 'token_mean' or 'sum', got "
                f"{normalization!r}"
            )
        super().__init__(run, model)
        self.normalization = normalization
        self._normalization_mass = 0.0
        self._loss_numerator = 0.0
        self._loss_numerator_device: torch.Tensor | None = None
        self._normalization_finalized = False
        # One device flag per device: every accepted device-resident teacher
        # score was finite. Read once, where the step already waits.
        self._teacher_finite_device: dict[torch.device, torch.Tensor] = {}

    def open_groups_shared_prefix(
        self,
        prompts: dict[int, list[int]],
        *,
        internal_nodes: bool = True,
        coalesce_nodes: bool = False,
        max_coalesced_nodes: int | None = None,
    ) -> int:
        return self.run.open_groups_shared_prefix(
            prompts,
            internal_nodes=internal_nodes,
            coalesce_nodes=coalesce_nodes,
            max_coalesced_nodes=max_coalesced_nodes,
        )

    def append_scored_turn(
        self,
        group_id: int,
        traj_id: int,
        tokens: list[int],
        scored: list[bool],
        *,
        teacher_logprobs: torch.Tensor,
        token_weights: torch.Tensor | None = None,
        return_loss: bool = True,
    ) -> float | None:
        """Differentiate one ready turn; optionally read its loss back to the host."""
        return self.append_scored_turns(
            [
                (
                    group_id,
                    traj_id,
                    tokens,
                    scored,
                    teacher_logprobs,
                    token_weights,
                )
            ],
            return_loss=return_loss,
        )[0]

    def append_scored_turns(
        self,
        items: list[
            tuple[
                int,
                int,
                list[int],
                list[bool],
                torch.Tensor,
                torch.Tensor | None,
            ]
        ],
        *,
        return_loss: bool = True,
    ) -> list[float | None]:
        """Backward ready turns; defer host loss reads when return_loss is false."""
        if not items:
            raise ValueError("append_scored_turns needs at least one item")
        if self._normalization_finalized:
            raise RuntimeError("cannot append a turn after OPD finalization")
        self._check_versions()
        seen: set[int] = set()
        deferred_finiteness: list[torch.Tensor] = []
        for (
            group_id,
            traj_id,
            _tokens,
            scored,
            teacher_logprobs,
            token_weights,
        ) in items:
            if traj_id in seen:
                raise ValueError(
                    f"trajectory {traj_id} appears twice in one scored-turn pack"
                )
            seen.add(traj_id)
            if traj_id in self._closed.get(group_id, set()):
                raise RuntimeError(f"trajectory {traj_id} already closed")
            expected_shape = (sum(scored),)
            if tuple(teacher_logprobs.shape) != expected_shape:
                raise ValueError(
                    "teacher logprobs have shape "
                    f"{tuple(teacher_logprobs.shape)}, expected {expected_shape} "
                    "from the turn score mask"
                )
            if teacher_logprobs.device.type == "cpu":
                if not bool(torch.isfinite(teacher_logprobs).all()):
                    raise ValueError("teacher logprobs must be finite")
            else:
                # Reading a device value here would make the host wait for
                # every queued kernel on each turn. The check is folded into
                # a device flag and refused in assert_safe_to_step, before
                # the divisor and the optimizer.
                deferred_finiteness.append(torch.isfinite(teacher_logprobs).all())
            if token_weights is not None:
                if token_weights.requires_grad:
                    raise ValueError("OPD token_weights must be detached")
                if tuple(token_weights.shape) != expected_shape:
                    raise ValueError(
                        "token_weights have shape "
                        f"{tuple(token_weights.shape)}, expected "
                        f"{expected_shape} from the turn score mask"
                    )
                if not bool(torch.isfinite(token_weights).all()):
                    raise ValueError("OPD token_weights must be finite")
                if bool((token_weights < 0).any()):
                    raise ValueError("OPD token_weights must be non-negative")
        for finite in deferred_finiteness:
            flag = self._teacher_finite_device.get(finite.device)
            self._teacher_finite_device[finite.device] = (
                finite if flag is None else flag & finite
            )

        turns = [
            (group_id, traj_id, tokens, scored)
            for group_id, traj_id, tokens, scored, _teacher, _weights in items
        ]
        if len(items) > 1 and self.run.joint_turn_pack_ready:
            return self._append_turn_packs(items, turns, return_loss=return_loss)
        self.run.append_turns(turns)
        values: list[float | None] = []
        for _group_id, traj_id, _tokens, _scored, teacher_logprobs, token_weights in items:
            turn = self._turn_loss(traj_id, teacher_logprobs, token_weights)
            if turn is None:
                values.append(None)
                continue
            loss, lp, weights = turn
            self.run.backward_with_boundary_adjoint_capture(
                loss,
                source=("trajectory_turn", traj_id),
            )
            values.append(self._record_turn_loss(loss, lp, weights, return_loss))
        # The call's memory window spans its forward and every backward.
        self.run.close_memory_window()
        return values

    def _append_turn_packs(
        self,
        items: list[
            tuple[
                int,
                int,
                list[int],
                list[bool],
                torch.Tensor,
                torch.Tensor | None,
            ]
        ],
        turns: list[tuple[int, int, list[int], list[bool]]],
        *,
        return_loss: bool,
    ) -> list[float | None]:
        """Ready turns of distinct trajectories as jointly backwarded packs.

        A turn with no scored token has no loss; it appends on its own so
        its forward stays deferred to its trajectory's next turn. The
        scored turns run as packs sized by the run's plain memory rule
        (`joint_turn_pack_width`): one forward over the pack and ONE
        backward over the pack's losses, whose parameter and state
        adjoints are the sums of the per-turn ones. Each turn's loss,
        coefficients and normalization mass are computed exactly as on
        its own; only where the per-turn gradients are summed changes,
        which arrival-order accumulation already leaves order-dependent
        at its rounding. A pack of one is the single-turn append.
        """

        values: list[float | None] = [None] * len(items)
        scored: list[int] = []
        for index, turn in enumerate(turns):
            if any(turn[3]):
                scored.append(index)
            else:
                self.run.append_turns([turn])
                _group_id, traj_id, _tokens, _scored, teacher, weights = items[index]
                self._turn_loss(traj_id, teacher, weights)
        start = 0
        while start < len(scored):
            width = self.run.joint_turn_pack_width(
                [turns[index] for index in scored[start:]]
            )
            pack = scored[start : start + width]
            start += width
            self.run.append_turns(
                [turns[index] for index in pack], joint_backward=len(pack) > 1
            )
            losses: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
            for index in pack:
                _group_id, traj_id, _tokens, _scored, teacher, weights = items[index]
                turn = self._turn_loss(traj_id, teacher, weights)
                if turn is not None:
                    losses.append((index, *turn))
            if not losses:
                self.run.close_memory_window()
                continue
            if len(losses) == 1:
                source: Any = ("trajectory_turn", items[losses[0][0]][1])
            else:
                source = (
                    "trajectory_turn_pack",
                    tuple(items[index][1] for index, *_rest in losses),
                )
            self.run.backward_with_boundary_adjoint_capture(
                [loss for _index, loss, _lp, _weights in losses],
                source=source,
            )
            # The pack's memory window spans its forward and its backward.
            self.run.close_memory_window()
            for index, loss, lp, weights in losses:
                values[index] = self._record_turn_loss(loss, lp, weights, return_loss)
        return values

    def _turn_loss(
        self,
        traj_id: int,
        teacher_logprobs: torch.Tensor,
        token_weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
        """The latest appended turn's loss numerator, or None if unscored."""

        lp, old = self.run.consume_latest_turn_logprobs(traj_id)
        shared_reference = (
            old.dtype == lp.dtype and old.device == lp.device and lp.is_set_to(old)
        )
        if old.shape != lp.shape or (
            not shared_reference and not torch.equal(lp.detach(), old)
        ):
            raise RuntimeError(
                "OPD requires on-policy behavior logprobs from the same turn"
            )
        teacher_lp = teacher_logprobs.detach().to(
            device=lp.device, dtype=lp.dtype
        )
        if teacher_lp.shape != lp.shape:
            raise RuntimeError(
                "validated teacher logprobs changed shape during forward"
            )
        weights = (
            None
            if token_weights is None
            else token_weights.detach().to(device=lp.device, dtype=lp.dtype)
        )
        if not lp.numel():
            return None
        torch._assert_async(torch.isfinite(lp).all(), "OPD learner logprobs must be finite")
        ratio = torch.exp(lp - old)
        reverse_kl_score = (old - teacher_lp).detach()
        loss = (
            (ratio * reverse_kl_score).sum()
            if weights is None
            else (weights * ratio * reverse_kl_score).sum()
        )
        return loss, lp, weights

    def _record_turn_loss(
        self,
        loss: torch.Tensor,
        lp: torch.Tensor,
        weights: torch.Tensor | None,
        return_loss: bool,
    ) -> float | None:
        """Fold a backwarded turn loss into the batch numerator and mass."""

        # Reporting must not hold a ready action ahead of its backward.
        detached_loss = loss.detach().to(dtype=torch.float64)
        if self._loss_numerator_device is None:
            self._loss_numerator_device = detached_loss.clone()
        else:
            self._loss_numerator_device.add_(detached_loss)
        self._normalization_mass += (
            lp.numel() if weights is None
            else float(weights.detach().to(dtype=torch.float64).sum().cpu())
        )
        return float(loss.detach()) if return_loss else None

    def close_trajectory(self, traj_id: int) -> None:
        """Replay descendant state adjoints and release the completed branch."""
        self.close_trajectories([traj_id])

    def close_trajectories(self, traj_ids: list[int]) -> None:
        """Replay a closure cohort's state adjoints and release its branches.

        ``traj_ids`` completed together; the run replays their turns as
        cohort packs (`StreamingRun.finalize_trajectories_turn_boundaries`),
        each trajectory last turn first. Every trajectory is checked before
        any state changes.
        """
        if self._normalization_finalized:
            raise RuntimeError("cannot close a trajectory after OPD finalization")
        if len(set(traj_ids)) != len(traj_ids):
            raise ValueError("a trajectory appears twice in one closure cohort")
        for traj_id in traj_ids:
            self._ensure_trajectory_can_close(traj_id)
        for traj_id in traj_ids:
            self._register_close(traj_id)
        self.run.finalize_trajectories_turn_boundaries(list(traj_ids))
        for traj_id in traj_ids:
            self.run.free_trajectory(traj_id)

    def assert_safe_to_step(self) -> None:
        """Finalize shared prompts and apply the one batch token divisor.

        A device-resident teacher score that was not finite refuses the
        step here, before the divisor, the loss read and the optimizer.
        """
        if not self._normalization_finalized:
            for flag in self._teacher_finite_device.values():
                if not bool(flag):
                    raise ValueError("teacher logprobs must be finite")
        super().assert_safe_to_step()
        if self._normalization_finalized:
            return
        divisor = self.normalization_divisor
        if divisor != 1.0:
            for parameter in self.params:
                if parameter.grad is not None:
                    parameter.grad.div_(divisor)
        if self._loss_numerator_device is not None:
            self._loss_numerator = float(self._loss_numerator_device)
            self._loss_numerator_device = None
        self._normalization_finalized = True

    @property
    def normalization_divisor(self) -> float:
        if self.normalization == "sum":
            return 1.0
        return max(self._normalization_mass, 1.0)

    @property
    def loss_value(self) -> float:
        if not self._normalization_finalized:
            raise RuntimeError("OPD loss_value requires assert_safe_to_step first")
        return self._loss_numerator / self.normalization_divisor
