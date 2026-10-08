"""Optional GRPO source RMSNorm with the native FP32 VJP arithmetic."""

import torch

from thundersync.accel.capability import kernel_runs
from thundersync.accel.rms_norm import FP32RMSNormVJP, StableQwen3RMSNorm


_COMPILED_PRIMAL = None


def _primal(hidden, weight, eps):
    x = hidden.float()
    rstd = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return weight * (x * rstd).to(hidden.dtype), rstd


class _CompiledForwardVJP(FP32RMSNormVJP):
    @staticmethod
    def forward(ctx, hidden, weight, eps):
        global _COMPILED_PRIMAL
        if _COMPILED_PRIMAL is None:
            _COMPILED_PRIMAL = torch.compile(
                _primal, dynamic=True, fullgraph=True,
                options={"emulate_precision_casts": True,
                         "compile_threads": torch.get_num_threads()},
            )
        output, rstd = _COMPILED_PRIMAL(hidden, weight, eps)
        ctx.save_for_backward(hidden, weight, rstd)
        ctx.eps = eps
        return output


class _CompiledSourceRMSNorm(StableQwen3RMSNorm):
    """Compile the forward reduction and retain the native FP32 VJP."""

    def forward(self, hidden):
        if hidden.is_cuda and hidden.dtype == torch.bfloat16:
            return _CompiledForwardVJP.apply(hidden, self.weight, self.variance_epsilon)
        return super().forward(hidden)


def use_compiled_source_rms_norm(model: torch.nn.Module) -> int:
    """Try the optional forward on this device; keep every original Parameter."""
    replaced = 0
    for name, module in list(model.named_modules()):
        if (not name or not isinstance(module, StableQwen3RMSNorm)
                or isinstance(module, _CompiledSourceRMSNorm)
                or module.weight.dtype != torch.bfloat16):
            continue

        def trial(device, dtype, width, eps=module.variance_epsilon):
            generator = torch.Generator(device=device).manual_seed(71)
            weight = torch.linspace(.4, 3., width, device=device, dtype=dtype).requires_grad_()
            for shape in ((1, 3, width), (1, 3, 2, width)):
                hidden = torch.randn(shape, device=device, dtype=dtype,
                                     generator=generator, requires_grad=True)
                output = _CompiledForwardVJP.apply(hidden, weight, eps)
                gradients = torch.autograd.grad(output.float().sum(), (hidden, weight))
                if not all(bool(value.isfinite().all()) for value in (output, *gradients)):
                    raise RuntimeError("compiled source RMSNorm produced nonfinite values")

        if not kernel_runs("grpo_source_rms_norm_forward", module.weight,
                           module.weight.numel(), trial):
            continue
        norm_type = _CompiledSourceRMSNorm

        def native_trial(device, dtype, width):
            from thundersync.grpo.rms_norm_native import native_source_rms_trial

            native_source_rms_trial(device, dtype, width)

        if module.weight.is_cuda and kernel_runs("grpo_source_rms_norm_native_vjp", module.weight,
                       module.weight.numel(), native_trial):
            from thundersync.grpo.rms_norm_native import NativeSourceRMSNorm

            norm_type = NativeSourceRMSNorm
        with torch.device("meta"):
            norm = norm_type(module.weight.numel(), eps=module.variance_epsilon)
        norm.weight = module.weight
        norm.train(module.training)
        model.set_submodule(name, norm)
        replaced += 1
    return replaced
