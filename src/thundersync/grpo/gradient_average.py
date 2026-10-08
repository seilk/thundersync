"""The data-parallel gradient average the GRPO trainer runs at step time."""

from __future__ import annotations

import torch
import torch.distributed as dist


def per_parameter_all_reduce_average(
    parameters: list[torch.nn.Parameter],
) -> dict[str, int | str]:
    """Average gradients without changing per-parameter collective ordering."""

    handles = [
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.AVG, async_op=True)
        for parameter in parameters
    ]
    for handle in handles:
        handle.wait()
    return {
        "allreduce_mode": "per_parameter_async",
        "allreduce_collectives": len(handles),
        "allreduce_bucket_cap_bytes": 0,
        "allreduce_peak_bucket_bytes": 0,
    }
