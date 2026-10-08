"""Optional BF16 loads with the native FP32 RMSNorm accumulation order."""

import torch
import triton
import triton.language as tl


@triton.jit
def _partial_weight(X, DY, RS, OUT, M, N: tl.constexpr,
                    GROUPS, COLS: tl.constexpr):
    group = tl.program_id(0)
    col = tl.program_id(1) * COLS + tl.arange(0, COLS)
    acc = tl.full((COLS,), 0, tl.float32)
    for base in range(group * 32, M, GROUPS * 32):
        for i in tl.static_range(32):
            row = base + i
            mask = (row < M) & (col < N)
            x = tl.load(X + row * N + col, mask, 0).to(tl.float32)
            dy = tl.load(DY + row * N + col, mask, 0).to(tl.float32)
            rs = tl.load(RS + row, row < M, 0)
            product = dy * x
            acc = tl.fma(product, rs, acc)
    tl.store(OUT + group * N + col, acc, col < N)


@triton.jit
def _full_weight(X, DY, RS, OUT, M, N: tl.constexpr,
                 BY: tl.constexpr, COLS: tl.constexpr, LOG_BY: tl.constexpr):
    ty = tl.arange(0, BY)
    col = tl.program_id(0) * COLS + tl.arange(0, COLS)
    acc = tl.full((BY, COLS), 0, tl.float32)
    for base in range(0, M, BY * 8):
        for i in tl.static_range(8):
            row = base + ty * 8 + i
            mask = (row[:, None] < M) & (col[None, :] < N)
            x = tl.load(X + row[:, None] * N + col[None, :], mask, 0).to(tl.float32)
            dy = tl.load(DY + row[:, None] * N + col[None, :], mask, 0).to(tl.float32)
            rs = tl.load(RS + row, row < M, 0)
            product = dy * x
            acc = tl.fma(product, rs[:, None], acc)
    for bit in tl.static_range(LOG_BY):
        delta = BY >> (bit + 1)
        indices = tl.broadcast_to((ty ^ delta)[:, None], (BY, COLS))
        acc = acc + tl.gather(acc, indices, axis=0)
    value = tl.sum(tl.where(ty[:, None] == 0, acc, 0), axis=0)
    tl.store(OUT + col, value, col < N)


def native_source_rms_backward(gradient, hidden, weight, rstd, needs=(True, True)):
    gradient = gradient.contiguous()
    dx = dw = None
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        if needs[0]:
            dx, _ = torch.ops.aten._fused_rms_norm_backward(
                gradient, hidden, list(weight.shape), rstd, weight, [True, False])
        if needs[1]:
            width = weight.numel()
            rows = hidden.numel() // width
            sm_count = torch.cuda.get_device_properties(hidden.device).multi_processor_count
            if rows > 64 * 1024 and width // 32 < sm_count // 2:
                groups = min(32768 // triton.cdiv(width, 32), triton.cdiv(rows, 32))
                parts = torch.empty((groups, width), device=hidden.device, dtype=torch.float32)
                _partial_weight[(groups, triton.cdiv(width, 128))](
                    hidden, gradient, rstd, parts, rows, width, groups, 128,
                    num_warps=4, enable_fp_fusion=False)
                dw = parts.sum(dim=0).to(weight.dtype)
            else:
                by = 1 if rows < 64 else 8 if rows < 128 else 16 if rows < 256 else 32
                result = torch.empty((width,), device=hidden.device, dtype=torch.float32)
                _full_weight[(triton.cdiv(width, 32),)](
                    hidden, gradient, rstd, result, rows, width, by, 32, by.bit_length() - 1,
                    num_warps=4, enable_fp_fusion=False)
                dw = result.to(weight.dtype)
    return dx, dw


from thundersync.accel.rms_norm import FP32RMSNormVJP
from thundersync.grpo.rms_norm import _CompiledForwardVJP, _CompiledSourceRMSNorm


class _NativeSourceVJP(_CompiledForwardVJP):
    @staticmethod
    def forward(ctx, hidden, weight, eps):
        ctx.native_source_norm = hidden.is_contiguous()
        return _CompiledForwardVJP.forward(ctx, hidden, weight, eps)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        if not ctx.native_source_norm:
            return FP32RMSNormVJP.backward(ctx, gradient)
        hidden, weight, rstd = ctx.saved_tensors
        dx, dw = native_source_rms_backward(gradient, hidden, weight, rstd, ctx.needs_input_grad[:2])
        return dx, dw, None


class NativeSourceRMSNorm(_CompiledSourceRMSNorm):
    def forward(self, hidden):
        if hidden.is_cuda and hidden.dtype == torch.bfloat16:
            return _NativeSourceVJP.apply(hidden, self.weight, self.variance_epsilon)
        return super().forward(hidden)


def native_source_rms_trial(device, dtype, width):
    generator = torch.Generator(device=device).manual_seed(710)
    weight = torch.linspace(.4, 3., width, device=device, dtype=dtype)
    for rows in (3, 129, 65537):
        hidden = torch.randn((rows, width), device=device, dtype=dtype, generator=generator)
        gradient = torch.randn(hidden.shape, device=device, dtype=dtype, generator=generator)
        rstd = torch.rsqrt(hidden.float().square().mean(-1, keepdim=True) + 1e-6)
        dx, dw = torch.ops.aten._fused_rms_norm_backward(
            gradient.float(), hidden.float(), [width], rstd, weight.float(), [True, True])
        observed = native_source_rms_backward(gradient, hidden, weight, rstd)
        if not (torch.equal(observed[0], dx.to(dtype)) and torch.equal(observed[1], dw.to(dtype))):
            raise RuntimeError("native source RMSNorm differs from the device's FP32 VJP")
