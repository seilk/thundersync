"""Which optional kernels a device runs, decided by running them once.

An import that succeeds says nothing about whether the kernel supports the
device's architecture, dtype or head dimension. Each optional kernel is
therefore tried once per ``(kernel, device, dtype, shape key)`` on a tiny
input, forward and backward, and the outcome is cached. A kernel that fails
its trial is skipped and the caller takes its next path; the trial never
wraps the real call, so a real shape bug still raises there.

No architecture list decides anything here. Every outcome is recorded with
the device's name and compute capability so a run's evidence says which
kernels it used and why.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

import torch

__all__ = [
    "device_identity",
    "kernel_probe_records",
    "kernel_runs",
    "reset_kernel_probes",
]


@dataclass(frozen=True, slots=True)
class _Outcome:
    runs: bool
    error: str | None


_LOCK = threading.Lock()
_OUTCOMES: dict[tuple[str, int, str, int], _Outcome] = {}


def device_identity(device: torch.device | int | None = None) -> dict[str, object]:
    """Name, compute capability and memory of a CUDA device, or ``{}``."""

    if not torch.cuda.is_available():
        return {}
    if isinstance(device, int):
        index = device
    elif device is not None and device.index is not None:
        index = device.index
    else:
        index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "index": index,
        "name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_mib": properties.total_memory // (1 << 20),
    }


def kernel_runs(
    kernel: str,
    like: torch.Tensor,
    shape_key: int,
    trial: Callable[[torch.device, torch.dtype, int], None],
) -> bool:
    """Whether ``trial`` completes on ``like``'s device and dtype.

    ``trial(device, dtype, shape_key)`` must build its own small inputs and
    run the kernel's forward and backward. It runs at most once per key, with
    gradients enabled and the device synchronized, so an asynchronous launch
    failure is caught here instead of at a later, unrelated call.
    """

    if not like.is_cuda:
        return False
    key = (kernel, like.device.index, str(like.dtype), shape_key)
    outcome = _OUTCOMES.get(key)
    if outcome is not None:
        return outcome.runs
    with _LOCK:
        outcome = _OUTCOMES.get(key)
        if outcome is None:
            outcome = _try(trial, like.device, like.dtype, shape_key)
            _OUTCOMES[key] = outcome
    return outcome.runs


def _try(
    trial: Callable[[torch.device, torch.dtype, int], None],
    device: torch.device,
    dtype: torch.dtype,
    shape_key: int,
) -> _Outcome:
    try:
        with torch.cuda.device(device), torch.enable_grad():
            trial(device, dtype, shape_key)
            torch.cuda.synchronize(device)
    except Exception as error:  # noqa: BLE001 - any failure means "does not run here"
        return _Outcome(False, f"{type(error).__name__}: {str(error)[:300]}")
    return _Outcome(True, None)


def kernel_probe_records() -> list[dict[str, object]]:
    """Every trial this process ran, with the device it ran on."""

    identities: dict[int, dict[str, object]] = {}
    records = []
    for (kernel, index, dtype, shape_key), outcome in sorted(_OUTCOMES.items()):
        if index not in identities:
            identities[index] = device_identity(index)
        records.append(
            {
                "kernel": kernel,
                "device": identities[index],
                "dtype": dtype,
                "shape_key": shape_key,
                "runs": outcome.runs,
                "error": outcome.error,
            }
        )
    return records


def reset_kernel_probes() -> None:
    """Forget every outcome; tests use it to re-probe."""

    with _LOCK:
        _OUTCOMES.clear()
