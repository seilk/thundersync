"""LoRA linear layers evaluated against a merged effective weight.

A PEFT LoRA linear computes ``y = x W^T + (x A^T) B^T`` as three separate
GEMMs plus an add.  The base is frozen, so ``W_eff = W + B A`` is constant
between optimizer steps and the same value comes out of one GEMM.  At the
branch shape the unfused adapted layer costs close to half again the frozen
base and this fused form under a third, with gradients matching the
unfused path to fp32 rounding.

The forward's adapter contribution is folded into the weight; the adapter
gradients come from rank-sized GEMMs off the saved activations:

    dx = g W_eff        dA = (g B)^T x        dB = g^T (x A^T)

which is the exact derivative of the unfused expression, so the update is
unchanged.  The caller merges the adapters into the base weight in place --
no second copy of the model -- and must unmerge before the optimizer step,
since the step moves A and B and leaves the merged weight stale.
"""

from __future__ import annotations

import torch


class FusedLoRALinear(torch.autograd.Function):
    """``x @ W_eff.T`` with adapter gradients, no adapter GEMM in forward."""

    @staticmethod
    def forward(ctx, x, weight, lora_a, lora_b, scaling):
        ctx.save_for_backward(x, weight, lora_a, lora_b)
        ctx.scaling = scaling
        return x @ weight.t()

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, lora_a, lora_b = ctx.saved_tensors
        scaling = ctx.scaling
        grad_x = grad_out @ weight
        grad_a = grad_b = None
        # The adapters may be held in a wider dtype than the activations;
        # the products run in the activation dtype and the gradients are
        # returned in each adapter's own dtype.
        a = lora_a.to(x.dtype)
        b = lora_b.to(x.dtype)
        if ctx.needs_input_grad[2]:
            grad_a = ((grad_out @ b).transpose(-2, -1) @ x * scaling).to(lora_a.dtype)
        if ctx.needs_input_grad[3]:
            grad_b = (grad_out.transpose(-2, -1) @ (x @ a.t()) * scaling).to(lora_b.dtype)
        return grad_x, None, grad_a, grad_b, None


def _lora_parts(module: torch.nn.Module, adapter: str):
    """The active adapter's A, B, and scaling, or None when not applicable."""

    lora_a = getattr(module, "lora_A", None)
    lora_b = getattr(module, "lora_B", None)
    if lora_a is None or lora_b is None:
        return None
    if adapter not in lora_a or adapter not in lora_b:
        return None
    dropout_dict = getattr(module, "lora_dropout", None)
    dropout = dropout_dict[adapter] if dropout_dict is not None and adapter in dropout_dict else None
    if dropout is not None and getattr(dropout, "p", 0.0):
        # Dropout would make the merged weight wrong for the forward.
        return None
    scaling = float(getattr(module, "scaling", {}).get(adapter, 1.0))
    return lora_a[adapter].weight, lora_b[adapter].weight, scaling


def fuse_merged_lora_forward(peft_model: torch.nn.Module) -> int:
    """Merge adapters in place and route the layers through the fused path.

    Returns the number of layers converted.  The merge is exact at
    initialization, where PEFT zeroes ``B`` and the effective weight equals
    the base weight; afterwards it is the same sum the unfused forward would
    have computed.  Call :func:`unfuse_merged_lora_forward` before the
    optimizer step.
    """

    adapter = getattr(peft_model, "active_adapter", "default")
    if isinstance(adapter, (list, tuple)):
        if len(adapter) != 1:
            raise ValueError("fused LoRA supports one active adapter")
        adapter = adapter[0]
    converted = 0
    for module in peft_model.modules():
        parts = _lora_parts(module, adapter)
        if parts is None:
            continue
        base = getattr(module, "base_layer", None)
        if base is None or not isinstance(base, torch.nn.Linear):
            continue
        if getattr(module, "_thundersync_fused", False):
            continue
        module.merge(adapter_names=[adapter])
        lora_a, lora_b, scaling = parts

        def forward(x, _module=module, _a=lora_a, _b=lora_b, _s=scaling):
            return FusedLoRALinear.apply(
                x, _module.base_layer.weight, _a, _b, _s
            )

        module._thundersync_unfused_forward = module.forward
        module.forward = forward
        module._thundersync_fused = True
        converted += 1
    return converted


def unfuse_merged_lora_forward(peft_model: torch.nn.Module) -> int:
    """Restore the stock forward and unmerge, so a step sees plain weights."""

    adapter = getattr(peft_model, "active_adapter", "default")
    if isinstance(adapter, (list, tuple)):
        adapter = adapter[0]
    restored = 0
    for module in peft_model.modules():
        if not getattr(module, "_thundersync_fused", False):
            continue
        module.forward = module._thundersync_unfused_forward
        del module._thundersync_unfused_forward
        module._thundersync_fused = False
        module.unmerge()
        restored += 1
    return restored
