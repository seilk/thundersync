"""The training step: one forward over the forest, projecting only what is read.

This is where the composition happens.
The execution plan combines three properties at the projection boundary:

    sharing   the physical stream holds each shared token once
    packing   there is no rectangle at any point, so no pad is ever projected
    gating    `plan.score_src` gathers *before* the unembedding, so a logit is
              formed only where the objective will read one

The gather is the whole point:

    src = hidden[plan.score_src]        # [S, D]  -- S is the scored count
    lp  = fused_logprob(src, W, tgt)    # never [P, V], never [B, T, V]

verl's fused kernel is called as `forward(hidden_states=hidden_states, input_ids=rolled_labels)`
-- the *full* sequence, with no mask parameter, because that function is the
patched model `forward` and the loss mask lives downstream in the objective. It
computes a logprob everywhere and multiplies by zero afterwards. Here the mask is
a field of the plan, so zero-coefficient positions are excluded before the
projection rather than computed and discarded.

Loss-agnostic by construction: this returns per-token `log pi(token | context)`
for the caller's own tokens and nothing else. It never sees a reward, an
advantage, a ratio or a KL coefficient. GRPO, PPO, on-policy distillation and
anything else of the form `sum_i sum_t c[i,t] g_t(theta)` consume it unchanged.

Exact under sharing: several logical tokens may resolve to one physical slot, and
`lp[plan.logical_to_slot]` re-expands them. Autograd sums the shared slot's
gradient over its logical uses on the way back. No coefficient is ever folded by
hand -- which matters because folding a full per-token weight is exactly what
breaks gradients for objectives whose coefficients depend on theta (the PPO/GRPO
importance ratio), a bug that leaves the loss value correct.
"""

from __future__ import annotations

import contextlib
import weakref
from dataclasses import dataclass

import torch

from thundersync.accel.fused_logprob import fused_logprob
from thundersync.engine.attention import forest_attention, physical_forest_attention
from thundersync.engine.plan import ExecutionPlan

# Device-indexed module state rather than a `ContextVar`. Activation
# checkpointing may recompute on an autograd engine thread, where a ContextVar
# set on the caller thread is unavailable. Selecting by the query device keeps
# that thread visibility and also permits one teacher plan and one student plan
# to execute concurrently on distinct GPUs in the same OPD process. One active
# plan per device remains the executor contract.
_PLANS: dict[torch.device, ExecutionPlan] = {}

ATTENTION_NAME = "thundersync_forest"
PHYSICAL_ATTENTION_NAME = "thundersync_physical_forest"


def forest_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs,
):
    """Attention implementation for the HF `AttentionInterface` registry.

    Registered rather than monkey-patched: `transformers` exposes this as a
    supported extension point, so the model's own `forward` is untouched and
    stays compatible with whatever else the model does. The plan travels by
    context variable instead of by keyword, which is what lets it reach *both*
    the attention and the unembedding -- the coupling verl's two patches cannot
    have, since one of them replaces the function the other needs.

    `attention_mask` is ignored on purpose: the plan is the mask, exactly, and a
    dense one would be the quadratic tensor this design exists to avoid.
    """
    plan = _PLANS.get(query.device)
    if plan is None:
        raise RuntimeError(
            "forest attention was invoked outside `forest_context`; "
            "the model would silently attend across trajectories"
        )
    if query.shape[0] != 1:
        raise ValueError(
            f"the forest is already the batch; expected batch dim 1, got {query.shape[0]}"
        )
    out = forest_attention(
        query[0].transpose(0, 1),
        key[0].transpose(0, 1),
        value[0].transpose(0, 1),
        plan,
        scale=scaling,
    )
    return out.unsqueeze(0), None  # [B, T, H, D], as the registry expects


def physical_forest_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs,
):
    """Feature-gated one-kernel attention over the complete physical forest."""
    plan = _PLANS.get(query.device)
    if plan is None:
        raise RuntimeError(
            "physical forest attention was invoked outside `forest_context`"
        )
    if query.shape[0] != 1:
        raise ValueError(
            f"the physical forest is the batch; expected batch 1, got {query.shape[0]}"
        )
    out = physical_forest_attention(
        query[0].transpose(0, 1),
        key[0].transpose(0, 1),
        value[0].transpose(0, 1),
        plan,
        scale=scaling,
    )
    return out.unsqueeze(0), None


def register_attention() -> None:
    """Register the default and feature-gated physical forest backends."""
    from transformers.modeling_utils import AttentionInterface

    AttentionInterface.register(ATTENTION_NAME, forest_attention_forward)
    AttentionInterface.register(
        PHYSICAL_ATTENTION_NAME, physical_forest_attention_forward
    )


@contextlib.contextmanager
def forest_context(plan: ExecutionPlan):
    """Bind a plan for the duration of a forward -- or a whole step.

    With activation checkpointing on, wrap **forward, loss and backward**::

        with forest_context(plan):
            lp   = forest_logprobs(model, plan)
            loss = my_objective(lp, ...)      # framework's own
            loss.backward()

    The recompute happens inside `backward()`, so a context that ended after the
    forward would leave it unbound. Nesting is safe: `forest_logprobs` binds the
    same plan internally and restores whatever was bound before.
    """
    device = plan.token_ids.device
    previous = _PLANS.get(device)
    _PLANS[device] = plan
    try:
        yield
    finally:
        if previous is None:
            _PLANS.pop(device, None)
        else:
            _PLANS[device] = previous


@dataclass
class StepStats:
    """What was computed versus what a rectangle-and-mask trainer would have."""

    physical_tokens: int
    logical_tokens: int
    padded_tokens: int
    projected_positions: int
    logical_scored: int

    @property
    def token_compression(self) -> float:
        """Forward tokens saved: padding removal and prefix sharing together."""
        return self.padded_tokens / max(self.physical_tokens, 1)

    @property
    def projection_reduction(self) -> float:
        """Unembedding rows saved: what the loss mask is worth at the kernel."""
        return self.padded_tokens / max(self.projected_positions, 1)

    def __str__(self) -> str:
        return (
            f"tokens {self.padded_tokens:,} -> {self.physical_tokens:,} "
            f"({self.token_compression:.2f}x)   "
            f"projected {self.padded_tokens:,} -> {self.projected_positions:,} "
            f"({self.projection_reduction:.2f}x)"
        )


# Layer families that carry state along the sequence outside the attention
# interface. A recurrent layer handed the packed physical stream would run its
# state straight from one branch into the next: no error, no NaN, just a
# different model. Refusing is the only safe behaviour unless the executor
# forks that state explicitly, and this is checked rather than documented.
_RECURRENT_MARKERS = (
    "deltanet",
    "gateddelta",
    "mamba",
    "linearattention",
    "linear_attention",
    "retention",
    "rwkv",
    "ssm",
)

# The supported hybrid layer: a Gated DeltaNet with split projections. The
# streaming executor (streaming._linear_layer_fn) drives exactly these
# attributes. A module whose class matches a recurrent marker but not this
# shape -- Mamba, RWKV, retention, or a fused-projection Gated DeltaNet --
# is refused.
_VERIFIED_LINEAR_CLASS_MARKERS = ("gateddeltanet",)
_VERIFIED_LINEAR_ATTRS = (
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "conv1d",
    "conv_kernel_size",
    "norm",
    "out_proj",
    "A_log",
    "dt_bias",
    "key_dim",
    "value_dim",
    "head_k_dim",
    "head_v_dim",
    "num_k_heads",
    "num_v_heads",
    "chunk_gated_delta_rule",
)


_CHECKED: weakref.WeakKeyDictionary[torch.nn.Module, set[bool]] = (
    weakref.WeakKeyDictionary()
)


def _mark_supported(model: torch.nn.Module, allow_hybrid: bool) -> None:
    modes = _CHECKED.get(model)
    if modes is None:
        modes = set()
        _CHECKED[model] = modes
    modes.add(allow_hybrid)


def hybrid_layer_kinds(model: torch.nn.Module) -> tuple[str, ...] | None:
    """`config.layer_types` if the stack declares mixed layer kinds, else None."""
    kinds = getattr(model.config, "layer_types", None)
    if kinds and any(k != "full_attention" for k in kinds):
        return tuple(kinds)
    return None


def _verify_hybrid(model: torch.nn.Module, kinds: tuple[str, ...]) -> None:
    """Verify a hybrid stack is the supported family, or refuse loudly."""
    bad = sorted(set(kinds) - {"full_attention", "linear_attention"})
    if bad:
        raise NotImplementedError(
            f"model declares layer types {bad}; only full_attention + "
            "linear_attention (Gated DeltaNet) hybrids are verified"
        )
    layers = getattr(base_model(model), "layers", None)
    if layers is None or len(layers) != len(kinds):
        raise NotImplementedError(
            "hybrid support needs the decoder stack's `layers` list aligned "
            "with config.layer_types; this model family does not expose it"
        )
    exempt: set[int] = set()
    for li, (kind, layer) in enumerate(zip(kinds, layers, strict=True)):
        if kind != "linear_attention":
            continue
        mix = getattr(layer, "linear_attn", None)
        cls = type(mix).__name__.lower() if mix is not None else "<missing>"
        if mix is None or not any(m in cls for m in _VERIFIED_LINEAR_CLASS_MARKERS):
            raise NotImplementedError(
                f"layer {li} is declared linear_attention but its module "
                f"({cls}) is not a verified Gated DeltaNet; refusing"
            )
        missing = [a for a in _VERIFIED_LINEAR_ATTRS if not hasattr(mix, a)]
        if missing:
            raise NotImplementedError(
                f"layer {li}: Gated DeltaNet module lacks {missing}; the "
                "streaming executor drives these explicitly (split-projection "
                "layout) and will not guess at another layout"
            )
        exempt.add(id(mix))
        exempt.update(id(sub) for sub in mix.modules())
    for name, mod in model.named_modules():
        cls = type(mod).__name__.lower()
        if any(m in cls for m in _RECURRENT_MARKERS) and id(mod) not in exempt:
            raise NotImplementedError(
                f"module {name} ({type(mod).__name__}) carries recurrent state "
                "outside both the attention interface and the verified "
                "linear-attention layers; see assert_supported"
            )


def assert_supported(
    model: torch.nn.Module, *, recheck: bool = False, allow_hybrid: bool = False
) -> None:
    """Refuse models whose layers bypass the attention interface.

    Hybrid stacks that interleave Gated DeltaNet layers with full-attention
    ones split by caller:

    * `allow_hybrid=False` (the batch/forest path, `forest_logprobs`): refused.
      Forest attention is an attention mask; it says nothing about a recurrence,
      which would run its state straight across branch boundaries -- silently
      wrong, not slow.
    * `allow_hybrid=True` (the streaming executor): layer_types of
      full_attention + linear_attention where every linear layer is a
      split-projection Gated DeltaNet are accepted;
      the executor forks and carries (recurrent state, conv tail) explicitly as
      checkpoint-tracked graph I/O. Anything else still refuses loudly.

    Memoised per (model instance, mode): this runs on every training step and
    walking every submodule of a large model each time is pure overhead once
    the answer is known.
    """
    checked_modes = _CHECKED.get(model)
    if (
        not recheck
        and checked_modes is not None
        and allow_hybrid in checked_modes
    ):
        return
    kinds = hybrid_layer_kinds(model)
    if kinds is not None:
        if not allow_hybrid:
            bad = sorted({k for k in kinds if k != "full_attention"})
            raise NotImplementedError(
                f"model has non-attention layer types {bad}; the batch/forest "
                "path cannot fork recurrent state across branches. The "
                "streaming executor (StreamingRun) supports the verified "
                "Gated DeltaNet hybrid family."
            )
        _verify_hybrid(model, kinds)
        _mark_supported(model, allow_hybrid)
        return
    for name, mod in model.named_modules():
        cls = type(mod).__name__.lower()
        if any(m in cls for m in _RECURRENT_MARKERS):
            raise NotImplementedError(
                f"module {name} ({type(mod).__name__}) carries recurrent state "
                "outside the attention interface; see assert_supported"
            )
    _mark_supported(model, allow_hybrid)


def base_model(model: torch.nn.Module) -> torch.nn.Module:
    """The decoder stack, without the unembedding.

    Calling the full `*ForCausalLM` would compute the `[P, V]` logits this design
    exists to avoid, so the head is reached only through the fused path below.
    """
    for attr in ("model", "transformer", "gpt_neox"):
        inner = getattr(model, attr, None)
        if isinstance(inner, torch.nn.Module) and not hasattr(inner, "lm_head"):
            return inner
    return model


def unembedding(model: torch.nn.Module) -> torch.Tensor:
    for attr in ("lm_head", "output", "embed_out"):
        mod = getattr(model, attr, None)
        if isinstance(mod, torch.nn.Linear):
            return mod.weight
    if hasattr(model, "get_output_embeddings"):
        emb = model.get_output_embeddings()
        if emb is not None:
            return emb.weight
    raise AttributeError("could not locate the unembedding; pass `weight=` explicitly")


def forest_logprobs(
    model: torch.nn.Module,
    plan: ExecutionPlan,
    *,
    weight: torch.Tensor | None = None,
    chunk: int = 1024,
    hidden_fn=None,
    return_stats: bool = False,
    gate: bool = True,
):
    """Per-token `log pi(token | context)` for every scored token, in logical order.

    Returns `[L]` where `L = sum over trajectories of their scored token count`,
    ordered trajectory by trajectory and, within a trajectory, in its own token
    order. `split_by_trajectory` recovers the per-rollout view a framework's
    objective expects.

    `gate=False` projects every physical position and selects afterwards -- what
    a framework must do, since its kernel is never handed the mask. It exists so
    the gating term can be measured on its own rather than asserted; combined
    with an unshared forest (`unshared`), the two switches give the full 2x2 and
    show whether the composition is worth more than its parts.
    """
    w = weight if weight is not None else unembedding(model)
    assert_supported(model)

    if hidden_fn is None:
        stack = base_model(model)

        def hidden_fn(ids, pos):
            out = stack(
                input_ids=ids.unsqueeze(0),
                position_ids=pos.unsqueeze(0),
                attention_mask=None,
                use_cache=False,
            )
            return out.last_hidden_state[0]

    with forest_context(plan):
        hidden = hidden_fn(plan.token_ids, plan.position_ids)  # [P, D]

    if gate:
        # The mask reaching the kernel. Everything after this line is O(scored),
        # not O(physical) and certainly not O(padded).
        src = hidden.index_select(0, plan.score_src)             # [S, D]
        lp = fused_logprob(src, w, plan.score_tgt, chunk=chunk)  # [S]
        projected = plan.n_scored
    else:
        # Project everywhere, select afterwards: what a framework must do, since
        # its kernel is handed the whole sequence and no mask. Rolled targets so
        # position j scores token j+1, as in verl's `torch.roll(labels, -1)`.
        targets = torch.roll(plan.token_ids, shifts=-1, dims=0)
        lp_all = fused_logprob(hidden, w, targets, chunk=chunk)  # [P]
        lp = lp_all.index_select(0, plan.score_src)
        projected = plan.n_physical

        # A positional projection cannot serve a shared forest. The last token of
        # a shared node predicts a *different* token in each branch, but it holds
        # one physical slot, so the roll can only be right for one of them. This
        # is the same wall verl hits: `PrefixGrouper.split_output` has to project
        # the branch side separately for exactly this reason. Charged honestly --
        # the extra projections count toward this path's cost.
        nxt = plan.token_ids.index_select(
            0, (plan.score_src + 1).clamp(max=plan.n_physical - 1)
        )
        boundary = (nxt != plan.score_tgt).nonzero(as_tuple=True)[0]
        if boundary.numel():
            fix = fused_logprob(
                hidden.index_select(0, plan.score_src.index_select(0, boundary)),
                w,
                plan.score_tgt.index_select(0, boundary),
                chunk=chunk,
            )
            lp = lp.index_copy(0, boundary, fix)
            projected += int(boundary.numel())

    logical = lp.index_select(0, plan.logical_to_slot)           # [L]

    if not return_stats:
        return logical
    f = plan.forest
    return logical, StepStats(
        physical_tokens=plan.n_physical,
        logical_tokens=f.logical_tokens,
        padded_tokens=f.padded_tokens,
        projected_positions=projected,
        logical_scored=plan.n_logical_scored,
    )


def split_by_trajectory(
    plan: ExecutionPlan, logical: torch.Tensor
) -> dict[int, torch.Tensor]:
    """Regroup `[L]` logprobs by trajectory, in each trajectory's own order.

    The handoff point to a framework's objective: from here on the caller sees
    exactly the per-rollout, per-token logprobs it would have computed itself,
    and nothing about the forest.
    """
    out: dict[int, torch.Tensor] = {}
    traj = plan.logical_traj
    for tid in traj.unique(sorted=True).tolist():
        out[int(tid)] = logical[traj == tid]
    return out
