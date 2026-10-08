"""The post-rollout final work, run as forest micro-batches.

Once every reward of a rank's step has landed, each trajectory not
yet differentiated has a known advantage ``A_i``, so what is left of the
update is ``sum_i A_i grad S_i`` over those trajectories: the batch forest
objective on a subset of each group. This module runs that sum as forest
micro-batches -- complete groups
(runs of an oversized one) packed under a logical-token budget, each group's
prompt shared by its members in one prefix forest, activation checkpointed
-- and lets autograd accumulate it into the parameters' ``.grad``, where
the direct backwards and group closes accumulate theirs. The prompt term
of these members comes from the forest's own prompt forward; the group's
retained prompt in the streaming run carries only its other members'
adjoints, so the two parts sum to the group's update by linearity.

Memory: a micro-batch's budget is the smaller of the configured cap and
what the device has free beside the executor's resident state, at a per-token cost
derived from the model's shape (``forest_token_budget``). A micro-batch
whose forward runs out of device memory is cut in half and run again; one
whose backward runs out part way is run again for the parameters the walk
had not reached (autograd sums a leaf's contributions before its one
accumulation, so each parameter has either all of the micro-batch or none
of it), so every parameter receives every trajectory exactly once.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

from thundersync.engine.attention import attention_checkpoint_mode
from thundersync.engine.plan import compile_plan
from thundersync.engine.segments import Segment, SegmentKind, Trajectory, build_forest
from thundersync.engine.step import (
    ATTENTION_NAME,
    forest_context,
    forest_logprobs,
    register_attention,
    split_by_trajectory,
)
from thundersync.grpo.packing import pack_groups

__all__ = [
    "FinalForestReport",
    "backward_final_forest",
    "forest_token_budget",
    "forest_trajectory",
]

# Device bytes the budget leaves free beside the forest's own estimate:
# the allocator's rounding and the plan's index tensors.
_RESERVE_BYTES = 2 << 30
# The smallest budget the rule hands out; a trajectory longer than the
# budget is a micro-batch of its own either way.
_MIN_TOKENS = 1024


def forest_trajectory(
    group_id: int,
    trajectory_id: int,
    prompt_tokens: Sequence[int],
    turns: Sequence[tuple[Sequence[int], Sequence[bool]]],
) -> Trajectory:
    """One trajectory as the forest objective builds it: prompt, then turns."""

    segments = [Segment.prompt(tuple(prompt_tokens))]
    for tokens, scored in turns:
        if not tokens or len(tokens) != len(scored):
            raise ValueError(
                f"trajectory {trajectory_id}: a turn's tokens and score mask "
                "must be nonempty and aligned"
            )
        start = 0
        for index in range(1, len(tokens) + 1):
            if index == len(tokens) or bool(scored[index]) != bool(scored[start]):
                kind = SegmentKind.ACTION if scored[start] else SegmentKind.OBSERVATION
                segments.append(
                    Segment(tuple(tokens[start:index]), kind, bool(scored[start]))
                )
                start = index
    return Trajectory(segments, group_id=group_id, traj_id=trajectory_id)


def _per_token_bytes(model: torch.nn.Module) -> tuple[int, int]:
    """(bytes per logical token, fixed bytes) of one checkpointed forest step.

    Per token: every layer's checkpointed input, one layer's recomputed
    activations and their gradients in the parameters' dtype. Fixed: the
    fused log-prob's chunked logits and their gradient in fp32.
    """

    config = model.config
    hidden = int(config.hidden_size)
    layers = int(config.num_hidden_layers)
    intermediate = int(getattr(config, "intermediate_size", 4 * hidden))
    vocab = int(config.vocab_size)
    element = next(model.parameters()).element_size()
    per_token = element * (hidden * (layers + 24) + intermediate * 8)
    return per_token, vocab * 4 * 3


def forest_token_budget(
    model: torch.nn.Module,
    *,
    cap_tokens: int,
    chunk: int,
    reserve_bytes: int = _RESERVE_BYTES,
) -> tuple[int, dict[str, int]]:
    """The micro-batch budget that fits beside what the device holds now."""

    if cap_tokens < 1:
        raise ValueError("the forest token cap must be positive")
    device = next(model.parameters()).device
    per_token, fixed_per_row = _per_token_bytes(model)
    fixed = fixed_per_row * chunk
    if device.type != "cuda":
        return cap_tokens, {"per_token_bytes": per_token, "fixed_bytes": fixed}
    driver_free, _total = torch.cuda.mem_get_info(device)
    cached_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    free = int(driver_free) + max(0, int(cached_free))
    fits = (free - reserve_bytes - fixed) // per_token
    budget = int(min(cap_tokens, max(_MIN_TOKENS, fits)))
    return budget, {
        "free_bytes": free,
        "per_token_bytes": per_token,
        "fixed_bytes": fixed,
        "fits_tokens": int(fits),
    }


@dataclass
class FinalForestReport:
    trajectories: int = 0
    logical_tokens: int = 0
    microbatches: int = 0
    out_of_memory_retries: int = 0
    budgets: list[int] = field(default_factory=list)
    wall_s: float = 0.0
    peak_allocated_bytes: int = 0
    objective_value: float = 0.0
    attention_cache_budget_bytes: int = 0
    attention_cache_charged_bytes: int = 0
    attention_cache_saved: int = 0
    attention_cache_replayed: int = 0
    attention_cache_declined: int = 0

    def as_record(self) -> dict[str, Any]:
        return {
            "trajectories": self.trajectories,
            "logical_tokens": self.logical_tokens,
            "microbatches": self.microbatches,
            "out_of_memory_retries": self.out_of_memory_retries,
            "budgets": list(self.budgets),
            "wall_s": self.wall_s,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "objective_value": self.objective_value,
            "attention_cache_budget_bytes": self.attention_cache_budget_bytes,
            "attention_cache_charged_bytes": self.attention_cache_charged_bytes,
            "attention_cache_saved": self.attention_cache_saved,
            "attention_cache_replayed": self.attention_cache_replayed,
            "attention_cache_declined": self.attention_cache_declined,
        }


def _attention_cache_budget(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    free, _ = torch.cuda.mem_get_info(device)
    cached = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    return (int(free) + max(0, int(cached))) // 4


class _AttentionCheckpointCache:
    """Retain bounded native attention outputs, with original recompute fallback."""

    def __init__(self, report: FinalForestReport):
        self.report = report

    def context(self):
        decisions = deque()
        report = self.report

        def policy(ctx, op, *args, **kwargs):
            if op not in (
                torch.ops.aten._scaled_dot_product_flash_attention.default,
                torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default,
            ):
                return CheckpointPolicy.MUST_RECOMPUTE
            if ctx.is_recompute:
                expected_op, saved = decisions.popleft()
                if op is not expected_op:
                    raise RuntimeError("forest attention recompute order changed")
                report.attention_cache_replayed += int(saved)
                return (
                    CheckpointPolicy.MUST_SAVE
                    if saved else CheckpointPolicy.MUST_RECOMPUTE
                )
            query = args[0]
            dropout = args[3] if len(args) > 3 else kwargs.get("dropout_p", 0.0)
            debug = False
            if op is torch.ops.aten._scaled_dot_product_flash_attention.default:
                debug = (
                    args[5] if len(args) > 5
                    else kwargs.get("return_debug_mask", False)
                )
            # Q-shaped output, FP32 log-sum-exp, and a conservative bound
            # for scalar/batch metadata. Debug attention masks are not cached.
            size = (
                query.numel() * query.element_size()
                + query.shape[0] * query.shape[1] * query.shape[2] * 4
                + (2 * query.shape[0] + 4) * 8
            )
            saved = (
                dropout == 0 and not debug
                and report.attention_cache_charged_bytes + size
                <= report.attention_cache_budget_bytes
            )
            decisions.append((op, saved))
            if saved:
                report.attention_cache_charged_bytes += size
                report.attention_cache_saved += 1
            else:
                report.attention_cache_declined += 1
            return (
                CheckpointPolicy.MUST_SAVE
                if saved else CheckpointPolicy.MUST_RECOMPUTE
            )

        forward, recompute = create_selective_checkpoint_contexts(policy)
        return attention_checkpoint_mode(forward), attention_checkpoint_mode(recompute)


class _ForwardOutOfMemory(Exception):
    pass


def _microbatch_backward(
    model: torch.nn.Module,
    microbatch: list[Trajectory],
    advantages: dict[int, float],
    targets: list[torch.nn.Parameter],
    *,
    chunk: int,
    report: FinalForestReport,
) -> list[torch.nn.Parameter]:
    """Backward one micro-batch into ``targets``; the targets it did not reach.

    Raises ``_ForwardOutOfMemory`` when the forward runs out of device
    memory, before anything accumulated.
    """

    device = targets[0].device
    host_plan = compile_plan(build_forest(microbatch), device="cpu")
    plan = host_plan.to(device)
    # Node-wise attention reads these scalars on the host in every layer.
    # Keep their original geometry beside the device token/score tensors.
    plan.node_start, plan.node_end = host_plan.node_start, host_plan.node_end
    reached = [False] * len(targets)
    with forest_context(plan):
        try:
            per_trajectory = split_by_trajectory(
                plan, forest_logprobs(model, plan, chunk=chunk)
            )
        except torch.OutOfMemoryError as error:
            raise _ForwardOutOfMemory from error
        objective = None
        for trajectory_id in sorted(per_trajectory):
            term = per_trajectory[trajectory_id].sum() * advantages[trajectory_id]
            objective = term if objective is None else objective + term
        if objective is None or not objective.requires_grad:
            return []
        report.objective_value += float(objective.detach())

        def install(index: int) -> Any:
            def accumulated(_leaf: torch.Tensor) -> None:
                if reached[index]:
                    raise RuntimeError(
                        "a final-forest input accumulated twice in one walk"
                    )
                reached[index] = True

            return targets[index].register_post_accumulate_grad_hook(accumulated)

        handles = [install(index) for index in range(len(targets))]
        try:
            torch.autograd.backward(objective, inputs=targets)
        except torch.OutOfMemoryError:
            del objective, per_trajectory
            return [
                parameter
                for parameter, done in zip(targets, reached, strict=True)
                if not done
            ]
        finally:
            for handle in handles:
                handle.remove()
    return []


def _run(
    model: torch.nn.Module,
    trajectories: list[Trajectory],
    advantages: dict[int, float],
    targets: list[torch.nn.Parameter],
    *,
    budget: int,
    chunk: int,
    report: FinalForestReport,
) -> None:
    for microbatch in pack_groups(
        trajectories, max_tokens_per_microbatch=budget, split_oversized_groups=True
    ):
        report.budgets.append(budget)
        report.microbatches += 1
        try:
            remaining = _microbatch_backward(
                model, microbatch, advantages, targets, chunk=chunk, report=report
            )
        except _ForwardOutOfMemory:
            remaining = targets
        if not remaining:
            continue
        report.out_of_memory_retries += 1
        if len(microbatch) > 1:
            tokens = sum(item.n_tokens for item in microbatch)
            _run(
                model,
                microbatch,
                advantages,
                remaining,
                budget=max(1, min(budget, tokens) // 2),
                chunk=chunk,
                report=report,
            )
            continue
        # One trajectory cannot be cut: run it once more alone, after the
        # allocator has returned the failed walk's cached blocks.
        torch.cuda.empty_cache()
        try:
            remaining = _microbatch_backward(
                model, microbatch, advantages, remaining, chunk=chunk, report=report
            )
        except _ForwardOutOfMemory:
            pass
        if remaining:
            raise torch.OutOfMemoryError(
                f"final forest: trajectory {microbatch[0].traj_id} "
                f"({microbatch[0].n_tokens} tokens) does not fit the device"
            )


def backward_final_forest(
    model: torch.nn.Module,
    trajectories: list[Trajectory],
    advantages: dict[int, float],
    *,
    cap_tokens: int,
    chunk: int,
) -> FinalForestReport:
    """Accumulate ``sum_i A_i grad S_i`` over ``trajectories`` into ``.grad``."""

    if not trajectories:
        raise ValueError("the final forest needs at least one trajectory")
    missing = [item.traj_id for item in trajectories if item.traj_id not in advantages]
    if missing:
        raise ValueError(f"trajectories {missing} have no advantage")
    targets = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not targets:
        raise ValueError("the final forest needs trainable parameters")
    report = FinalForestReport(
        trajectories=len(trajectories),
        logical_tokens=sum(item.n_tokens for item in trajectories),
    )
    device = targets[0].device
    started = time.perf_counter()
    budget, _evidence = forest_token_budget(model, cap_tokens=cap_tokens, chunk=chunk)
    register_attention()
    previous_attention = model.config._attn_implementation
    checkpointing_was_enabled = bool(getattr(model, "is_gradient_checkpointing", False))
    previous_functions = []
    missing = object()
    model.config._attn_implementation = ATTENTION_NAME
    try:
        if not checkpointing_was_enabled:
            options: dict[str, Any] = {"use_reentrant": False}
            report.attention_cache_budget_bytes = _attention_cache_budget(device)
            if report.attention_cache_budget_bytes:
                previous_functions = [
                    (
                        module,
                        module.__dict__.get("_gradient_checkpointing_func", missing),
                    )
                    for module in model.modules()
                    if hasattr(module, "gradient_checkpointing")
                ]
                options["context_fn"] = _AttentionCheckpointCache(report).context
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=options
            )
        _run(
            model,
            trajectories,
            advantages,
            targets,
            budget=budget,
            chunk=chunk,
            report=report,
        )
    finally:
        if not checkpointing_was_enabled:
            model.gradient_checkpointing_disable()
            disable_input_grads = getattr(model, "disable_input_require_grads", None)
            if disable_input_grads is not None and hasattr(model, "_require_grads_hook"):
                disable_input_grads()
            for module, previous in previous_functions:
                if previous is missing:
                    module.__dict__.pop("_gradient_checkpointing_func", None)
                else:
                    module._gradient_checkpointing_func = previous
        model.config._attn_implementation = previous_attention
    if device.type == "cuda":
        # The step's peak so far (the step loop resets it at step entry).
        report.peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
    report.wall_s = time.perf_counter() - started
    return report
