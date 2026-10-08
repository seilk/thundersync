"""Reference GRPO log-probability paths used by correctness tests.

The two implementations answer different questions.

`reference_logprobs` -- ground truth. Each trajectory alone, plain causal
attention, plain `log_softmax` over the full vocabulary. Slow and obviously
correct; nothing here is shared, packed, gated or fused, so if the forest path
agrees with it then the entire construction is exact.

`baseline_logprobs` -- a padded `[B, T_max]` rectangle with the prompt repeated
per rollout, a fused linear-to-logprob projection, and the loss mask applied
afterward. It uses the same fused kernel as the streaming implementation so
correctness tests isolate the schedule and representation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from thundersync.accel.fused_logprob import fused_logprob
from thundersync.engine.segments import Trajectory


@dataclass
class BaselineStats:
    sequences: int
    padded_tokens: int
    real_tokens: int
    projected_positions: int

    @property
    def padding_waste(self) -> float:
        return 1 - self.real_tokens / max(self.padded_tokens, 1)


def _pad(trajectories: Sequence[Trajectory], device, pad_id: int = 0):
    lengths = [t.n_tokens for t in trajectories]
    tmax = max(lengths)
    ids = torch.full((len(trajectories), tmax), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros(len(trajectories), tmax, dtype=torch.long, device=device)
    scored = torch.zeros(len(trajectories), tmax, dtype=torch.bool, device=device)
    for i, t in enumerate(trajectories):
        n = t.n_tokens
        ids[i, :n] = torch.tensor(t.tokens, device=device)
        attn[i, :n] = 1
        scored[i, :n] = torch.tensor(t.scored_mask, device=device)
    return ids, attn, scored, lengths


def baseline_logprobs(
    model: torch.nn.Module,
    trajectories: Sequence[Trajectory],
    *,
    weight: torch.Tensor | None = None,
    chunk: int = 1024,
    return_stats: bool = False,
):
    """`[B, T_max]` rectangle, projection over everything, mask applied after.

    Mirrors the standard path's two structural choices exactly: the prompt is
    stored once per rollout, and the unembedding sees the whole sequence because
    the loss mask is not available where the projection happens.
    """
    from thundersync.engine.step import base_model, unembedding

    dev = next(model.parameters()).device
    w = weight if weight is not None else unembedding(model)
    ids, attn, scored, lengths = _pad(trajectories, dev)

    stack = base_model(model)
    hidden = stack(
        input_ids=ids, attention_mask=attn, use_cache=False
    ).last_hidden_state  # [B, T, D]

    b, t, d = hidden.shape
    # The projection a framework performs: every position of the rectangle. The
    # kernel is memory-efficient, but it is never told which positions matter.
    #
    # Labels are rolled left, mirroring `torch.roll(labels, shifts=-1, dims=-1)`
    # in verl/models/transformers/dense_common.py:118, so `lp_all[i, j]` is
    # log pi(token j+1 | context through j). The wrap-around at j = T-1 is
    # meaningless there and here; neither reads it.
    flat = hidden.reshape(b * t, d)
    targets = torch.roll(ids, shifts=-1, dims=-1).reshape(b * t)
    lp_all = fused_logprob(flat, w, targets, chunk=chunk).reshape(b, t)

    out: dict[int, torch.Tensor] = {}
    for i, traj in enumerate(trajectories):
        # token k is predicted from position k-1
        sel = scored[i].clone()
        sel[0] = False
        idx = sel.nonzero(as_tuple=True)[0]
        out[traj.traj_id] = lp_all[i].index_select(0, idx - 1)

    if not return_stats:
        return out
    return out, BaselineStats(
        sequences=b,
        padded_tokens=b * t,
        real_tokens=sum(lengths),
        projected_positions=b * t,
    )


def reference_logprobs(
    model: torch.nn.Module,
    trajectories: Sequence[Trajectory],
    *,
    weight: torch.Tensor | None = None,
):
    """One trajectory at a time, full `[T, V]` log-softmax. Ground truth only.

    Deliberately the slowest possible implementation: no sharing, no packing, no
    gating, no fusion, batch size one. Its only job is to be beyond suspicion.
    """
    from thundersync.engine.step import base_model, unembedding

    dev = next(model.parameters()).device
    w = weight if weight is not None else unembedding(model)
    stack = base_model(model)

    out: dict[int, torch.Tensor] = {}
    for traj in trajectories:
        ids = torch.tensor(traj.tokens, device=dev).unsqueeze(0)
        hidden = stack(input_ids=ids, use_cache=False).last_hidden_state[0]
        logits = torch.nn.functional.linear(hidden, w).float()
        lp = torch.log_softmax(logits, dim=-1)
        picked = []
        for k, sc in enumerate(traj.scored_mask):
            if sc and k > 0:
                picked.append(lp[k - 1, traj.tokens[k]])
        out[traj.traj_id] = (
            torch.stack(picked) if picked else torch.zeros(0, device=dev)
        )
    return out
