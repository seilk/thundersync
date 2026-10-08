"""Fused, chunked linear -> log-softmax -> gather, with a hand-written backward.

This is the trainer's single largest activation and its most wasteful one. A
standard implementation materialises logits for every position:

    logits   = hidden @ W.T           # [N, V]
    logprobs = log_softmax(logits)    # [N, V]  again
    picked   = logprobs.gather(target)

At V = 248,320 and N = 16k tokens that is 16 GiB per tensor, twice, and only the
gathered `[N]` vector is ever read. Agentic trajectories also contain many
environment-authored positions with zero objective coefficient.

Two structural facts make that removable *exactly*:

1. A position whose coefficient is zero contributes nothing to the loss, so its
   logits are never read. It still needs its forward hidden state (later tokens
   attend to its K/V) but not its logits. Gate before the projection, not after.
2. `logprob = logit[target] - logsumexp(logit)` needs only two scalars per
   position. The `[C, V]` block that produces them is pure intermediate, so it
   can be produced a chunk at a time and discarded, and recomputed in backward.

Peak logit memory becomes `O(chunk * V)` instead of `O(N * V)` -- with chunk=1024
that is ~0.5 GiB regardless of sequence length. That headroom is the point: it is
what lets the trainer drop activation checkpointing, which is worth more than the
FLOPs this saves.

Low-precision projections retain their FP32 matrix-product output through
log-softmax. Chunk-local weight gradients are accumulated in FP32 and cast
once on return. FP32/FP64 inputs retain their precision. Accelerator tests
compare the resulting values and gradients with an independent dense reference.
"""

from __future__ import annotations

import torch

from .operator_profiling import device_role, operator_range

# bf16/fp16 must accumulate in fp32 for a numerically sound log-softmax, but
# fp32/fp64 inputs must be left alone -- an unconditional `.float()` silently
# downcasts double precision, which is invisible in normal training and breaks
# gradcheck outright.
_ACCUM = {torch.bfloat16: torch.float32, torch.float16: torch.float32}


def _accum_dtype(dt: torch.dtype) -> torch.dtype:
    return _ACCUM.get(dt, dt)


def _matmul_accum(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Keep low-precision GEMM accumulators instead of rounding before softmax."""
    acc = _accum_dtype(left.dtype)
    # Incoming tensor dtypes define this operation, not an outer AMP context.
    with torch.autocast(device_type=left.device.type, enabled=False):
        if left.is_cuda and acc != left.dtype and left.dtype == right.dtype:
            return torch.mm(left, right, out_dtype=acc)
        return torch.mm(left.to(acc), right.to(acc))


class _FusedLinearLogprob(torch.autograd.Function):
    """logprob of `target` under `hidden @ weight.T`, chunked over positions.

    Saves only `[N, D]` hidden and `[V, D]` weight; the `[C, V]` logit block is
    rematerialised in backward. That trade is strongly favourable because V is
    two orders of magnitude larger than D.
    """

    @staticmethod
    def forward(ctx, hidden, weight, target, chunk, needs_weight_grad):
        n = hidden.shape[0]
        acc = _accum_dtype(hidden.dtype)
        with operator_range(
            "fused_logprob_forward",
            role=device_role(hidden.device),
            device=hidden.device,
            tokens=n,
            hidden=hidden.shape[1],
            vocab=weight.shape[0],
            chunk=chunk,
            chunks=(n + chunk - 1) // chunk,
            dtype=hidden.dtype,
        ):
            out = torch.empty(n, dtype=acc, device=hidden.device)
            for i in range(0, n, chunk):
                h = hidden[i : i + chunk]
                rows = h.shape[0]
                with operator_range(
                    "logprob_head_projection_forward",
                    role=device_role(hidden.device),
                    device=hidden.device,
                    rows=rows,
                    hidden=hidden.shape[1],
                    vocab=weight.shape[0],
                ):
                    logits = _matmul_accum(h, weight.t())
                with operator_range(
                    "logprob_head_reduction_forward",
                    role=device_role(hidden.device),
                    device=hidden.device,
                    rows=rows,
                    vocab=weight.shape[0],
                ):
                    logits.sub_(logits.amax(dim=-1, keepdim=True))
                    lse = torch.logsumexp(logits, dim=-1)
                    picked = logits.gather(
                        -1, target[i : i + chunk].unsqueeze(-1)
                    ).squeeze(-1)
                out[i : i + chunk] = picked - lse
        ctx.save_for_backward(hidden, weight, target)
        ctx.chunk = chunk
        ctx.needs_weight_grad = needs_weight_grad
        return out

    @staticmethod
    def backward(ctx, grad_out):
        hidden, weight, target = ctx.saved_tensors
        chunk = ctx.chunk
        n = hidden.shape[0]

        with operator_range(
            "fused_logprob_backward",
            role=device_role(hidden.device),
            device=hidden.device,
            tokens=n,
            hidden=hidden.shape[1],
            vocab=weight.shape[0],
            chunk=chunk,
            chunks=(n + chunk - 1) // chunk,
            dtype=hidden.dtype,
            weight_grad=ctx.needs_weight_grad,
        ):
            grad_hidden = torch.zeros_like(hidden)
            grad_weight = (
                torch.zeros_like(weight) if ctx.needs_weight_grad and n == 0 else None
            )
            for i in range(0, n, chunk):
                h = hidden[i : i + chunk]
                t = target[i : i + chunk]
                g = grad_out[i : i + chunk].unsqueeze(-1)
                rows = h.shape[0]

                # rematerialise this block only
                with operator_range(
                    "logprob_head_projection_recompute",
                    role=device_role(hidden.device),
                    device=hidden.device,
                    rows=rows,
                    hidden=hidden.shape[1],
                    vocab=weight.shape[0],
                ):
                    logits = _matmul_accum(h, weight.t())
                with operator_range(
                    "logprob_head_coefficient_backward",
                    role=device_role(hidden.device),
                    device=hidden.device,
                    rows=rows,
                    vocab=weight.shape[0],
                ):
                    probs = torch.softmax(logits, dim=-1)
                    # d/dlogit [ logit[target] - logsumexp(logit) ] =
                    # onehot(target) - softmax
                    probs.neg_()
                    probs.scatter_add_(-1, t.unsqueeze(-1), torch.ones_like(g))
                    dlogits = (probs * g).to(hidden.dtype)

                with operator_range(
                    "logprob_head_hidden_vjp",
                    role=device_role(hidden.device),
                    device=hidden.device,
                    rows=rows,
                    hidden=hidden.shape[1],
                    vocab=weight.shape[0],
                ):
                    grad_hidden[i : i + chunk] = _matmul_accum(dlogits, weight).to(hidden.dtype)
                if ctx.needs_weight_grad:
                    with operator_range(
                        "logprob_head_weight_vjp",
                        role=device_role(hidden.device),
                        device=hidden.device,
                        rows=rows,
                        hidden=hidden.shape[1],
                        vocab=weight.shape[0],
                    ):
                        contribution = _matmul_accum(dlogits.t(), h)
                        if grad_weight is None:
                            grad_weight = contribution
                        else:
                            grad_weight.add_(contribution)
                        del contribution

        if grad_weight is not None:
            grad_weight = grad_weight.to(weight.dtype)
        return grad_hidden, grad_weight, None, None, None


def fused_logprob(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    *,
    chunk: int = 1024,
) -> torch.Tensor:
    """Per-position log pi(target | context), without materialising `[N, V]`.

    ``hidden`` is ``[N, D]`` -- already restricted to the positions that matter.
    ``weight`` is the unembedding ``[V, D]``. ``target`` is ``[N]``.
    """
    if hidden.dim() != 2:
        raise ValueError(f"hidden must be [N, D], got {tuple(hidden.shape)}")
    if target.shape[0] != hidden.shape[0]:
        raise ValueError("target must have one entry per hidden row")
    if weight.shape[1] != hidden.shape[1]:
        raise ValueError(
            f"weight {tuple(weight.shape)} does not match hidden dim {hidden.shape[1]}"
        )
    return _FusedLinearLogprob.apply(
        hidden, weight, target, chunk, weight.requires_grad
    )


class _FusedLinearLogprobEntropy(torch.autograd.Function):
    """(logprob, entropy) of the chunk-local distribution, chunked over positions.

    Forward, per position with logits ``l`` and ``p = softmax(l)``:

        logprob = l[target] - lse(l)
        H       = -sum_j p_j log p_j  =  lse(l) - sum_j p_j l_j

    Backward, hand-derived. The logprob term is the classic one:

        d(logprob)/dl_j = onehot(target)_j - p_j

    For the entropy, with dp_k/dl_j = p_k (delta_kj - p_j):

        dH/dl_j = -sum_k dp_k/dl_j (log p_k + 1)
                = -p_j (log p_j + 1) + p_j sum_k p_k (log p_k + 1)
                = -p_j (log p_j + H)

    (the +1 terms cancel because sum_k p_k = 1, and sum_k p_k log p_k = -H).
    So with upstream grads g_lp and g_H the logit gradient of a block is

        dlogits = -p * (g_lp + g_H * (log p + H)) + g_lp * onehot(target)

    Only ``[C, V]`` blocks are ever live; the per-position scalars ``lse`` and
    ``H`` are saved from forward so backward rematerialises just the logits.
    """

    @staticmethod
    def forward(ctx, hidden, weight, target, chunk, needs_weight_grad):
        n = hidden.shape[0]
        acc = _accum_dtype(hidden.dtype)
        out_lp = torch.empty(n, dtype=acc, device=hidden.device)
        out_ent = torch.empty(n, dtype=acc, device=hidden.device)
        for i in range(0, n, chunk):
            h = hidden[i : i + chunk]
            logits = _matmul_accum(h, weight.t())
            logits.sub_(logits.amax(dim=-1, keepdim=True))
            lse = torch.logsumexp(logits, dim=-1)
            picked = logits.gather(-1, target[i : i + chunk].unsqueeze(-1)).squeeze(-1)
            out_lp[i : i + chunk] = picked - lse
            probs = torch.softmax(logits, dim=-1)
            out_ent[i : i + chunk] = lse - (probs * logits).sum(-1)
        ctx.save_for_backward(hidden, weight, target, out_ent)
        ctx.chunk = chunk
        ctx.needs_weight_grad = needs_weight_grad
        return out_lp, out_ent

    @staticmethod
    def backward(ctx, grad_lp, grad_ent):
        hidden, weight, target, ent = ctx.saved_tensors
        chunk = ctx.chunk
        n = hidden.shape[0]
        acc = _accum_dtype(hidden.dtype)

        grad_hidden = torch.zeros_like(hidden)
        grad_weight = torch.zeros_like(weight) if ctx.needs_weight_grad and n == 0 else None

        for i in range(0, n, chunk):
            h = hidden[i : i + chunk]
            t = target[i : i + chunk].unsqueeze(-1)
            g_lp = grad_lp[i : i + chunk].unsqueeze(-1)
            g_ent = grad_ent[i : i + chunk].unsqueeze(-1)

            # rematerialise this block only
            logits = _matmul_accum(h, weight.t())
            logits.sub_(logits.amax(dim=-1, keepdim=True))
            logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
            probs = logp.exp()
            hh = ent[i : i + chunk].unsqueeze(-1)

            # dlogits = -p*(g_lp + g_ent*(log p + H)) + g_lp*onehot(target)
            dlogits = probs.mul_(-(g_lp + g_ent * (logp + hh)))
            dlogits.scatter_add_(-1, t, g_lp.to(acc))
            dlogits = dlogits.to(hidden.dtype)

            grad_hidden[i : i + chunk] = _matmul_accum(dlogits, weight).to(hidden.dtype)
            if ctx.needs_weight_grad:
                contribution = _matmul_accum(dlogits.t(), h)
                if grad_weight is None:
                    grad_weight = contribution
                else:
                    grad_weight.add_(contribution)
                del contribution

        if grad_weight is not None:
            grad_weight = grad_weight.to(weight.dtype)
        return grad_hidden, grad_weight, None, None, None


def fused_logprob_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    *,
    chunk: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-position (log pi(target | context), entropy), without ``[N, V]``.

    Same contract as :func:`fused_logprob`, plus a second ``[N]`` output holding
    the entropy of each position's full-vocabulary distribution. The entropy is
    a per-position scalar for the same reason the logprob is, so it rides the
    same chunked schedule at no extra peak memory.

    A public kernel for entropy monitoring or an entropy term; the shipped
    objectives call only :func:`fused_logprob`.
    """
    if hidden.dim() != 2:
        raise ValueError(f"hidden must be [N, D], got {tuple(hidden.shape)}")
    if target.shape[0] != hidden.shape[0]:
        raise ValueError("target must have one entry per hidden row")
    if weight.shape[1] != hidden.shape[1]:
        raise ValueError(
            f"weight {tuple(weight.shape)} does not match hidden dim {hidden.shape[1]}"
        )
    return _FusedLinearLogprobEntropy.apply(
        hidden, weight, target, chunk, weight.requires_grad
    )


def gathered_logprobs(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    token_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    chunk: int = 1024,
) -> torch.Tensor:
    """Logprobs laid back out over the full sequence, zero where not needed.

    This is the shape a framework expects: ``[N]`` aligned with its tokens, so it
    can apply its own coefficients without knowing anything was skipped. The
    saving is invisible from the outside, which is the point.

    ``loss_mask`` marks positions whose logprob the objective will actually read.
    Position ``k`` is predicted from position ``k-1``, so the *hidden state* used
    is the predecessor's.
    """
    n = token_ids.shape[0]
    idx = loss_mask.nonzero(as_tuple=True)[0]
    idx = idx[idx > 0]  # position 0 has no predecessor
    if idx.numel() == 0:
        return torch.zeros(n, dtype=_accum_dtype(hidden.dtype), device=hidden.device)

    lp = fused_logprob(hidden[idx - 1], weight, token_ids[idx], chunk=chunk)
    out = torch.zeros(n, dtype=lp.dtype, device=lp.device)
    return out.index_put((idx,), lp)


def naive_logprobs(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    token_ids: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Reference path: materialise everything. Used only to verify the fused one."""
    acc = _accum_dtype(hidden.dtype)
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        logits = torch.nn.functional.linear(hidden.to(acc), weight.to(acc))
    lp = torch.log_softmax(logits, dim=-1)
    n = token_ids.shape[0]
    out = torch.zeros(n, dtype=acc, device=hidden.device)
    idx = loss_mask.nonzero(as_tuple=True)[0]
    idx = idx[idx > 0]
    if idx.numel() == 0:
        return out
    picked = lp[idx - 1].gather(-1, token_ids[idx].unsqueeze(-1)).squeeze(-1)
    return out.index_put((idx,), picked)
