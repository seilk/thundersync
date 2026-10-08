"""Use the native RMSNorm VJP without an intermediate BF16 affine-gradient cast."""

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm


class FP32RMSNormVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, eps):
        # Match the replaced norm's primal operation order and both casts.
        x = hidden.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        rstd = torch.rsqrt(variance + eps)
        primal = weight * (x * rstd).to(hidden.dtype)
        ctx.save_for_backward(hidden, weight, rstd)
        ctx.eps = eps
        return primal

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        hidden, weight, rstd = ctx.saved_tensors
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            if hidden.is_cuda:
                # Reuse the original forward's statistic. Only the VJP's
                # cancellation and reductions move to FP32, on the pinned
                # runtime's native kernel; no extra normalization forward.
                dx, dw = torch.ops.aten._fused_rms_norm_backward(
                    gradient.float().contiguous(), hidden.float(), list(weight.shape),
                    rstd, weight.float(), list(ctx.needs_input_grad[:2]),
                )
            else:
                # The private fused backward has no CPU dispatch.
                with torch.enable_grad():
                    x = hidden.detach().float().requires_grad_()
                    w = weight.detach().float().requires_grad_()
                    output = torch.nn.functional.rms_norm(x, tuple(weight.shape), w, ctx.eps)
                    dx, dw = torch.autograd.grad(output, (x, w), gradient.float())
        return (
            dx.to(hidden.dtype) if ctx.needs_input_grad[0] else None,
            dw.to(weight.dtype) if ctx.needs_input_grad[1] else None,
            None,
        )


class StableQwen3RMSNorm(Qwen3RMSNorm):
    """Keep the original primal bytes; cancel the radial VJP component in FP32."""

    def forward(self, hidden):
        if hidden.dtype not in (torch.bfloat16, torch.float16) or not torch.is_grad_enabled():
            return super().forward(hidden)
        return FP32RMSNormVJP.apply(hidden, self.weight, self.variance_epsilon)


def use_stable_qwen3_rms_norm(model: torch.nn.Module) -> int:
    """Replace only their low-precision VJP; preserve values, parameters and keys."""
    replaced = 0
    for name, module in list(model.named_modules()):
        if (not name or not isinstance(module, Qwen3RMSNorm)
                or isinstance(module, StableQwen3RMSNorm)):
            continue
        # Allocate no duplicate weight storage. Optimizers and publication
        # keep exactly the original Parameter and state-dict identity.
        with torch.device("meta"):
            norm = StableQwen3RMSNorm(module.weight.numel(), eps=module.variance_epsilon)
        norm.weight = module.weight
        norm.train(module.training)
        model.set_submodule(name, norm)
        replaced += 1
    return replaced
