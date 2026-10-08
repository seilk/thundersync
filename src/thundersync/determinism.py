"""The trainer's determinism setting, applied once per process.

``apply_trainer_determinism(True)`` turns torch's deterministic algorithms
on without the NaN fill of every fresh allocation (a debugging aid that
changes no result), turns cuDNN's deterministic selection on and its
autotuning off. ``apply_trainer_determinism(False)`` turns the deterministic
algorithms and cuDNN's deterministic selection off and leaves autotuning
off: every pack has a new length, so autotuning would re-run per shape and
add run-to-run variation. ``trainer_determinism()`` reads the effective
state back.

Seeds, TF32 and every other numeric setting stay with the caller. A kernel
outside torch that takes its own ``deterministic`` argument (FA4's
backward) is not governed by this switch. Call it before the model loads.
"""

from __future__ import annotations

__all__ = ["apply_trainer_determinism", "trainer_determinism"]


def apply_trainer_determinism(enabled: bool) -> dict[str, bool]:
    """Set the process's determinism and return what is now in effect."""

    if not isinstance(enabled, bool):
        raise TypeError(f"the determinism setting must be a bool, not {enabled!r}")
    import torch
    import torch.utils.deterministic

    torch.use_deterministic_algorithms(enabled)
    torch.utils.deterministic.fill_uninitialized_memory = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = enabled
    return trainer_determinism()


def trainer_determinism() -> dict[str, bool]:
    """The determinism this process runs under, as torch reports it."""

    import torch
    import torch.utils.deterministic

    return {
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "fill_uninitialized_memory": bool(
            torch.utils.deterministic.fill_uninitialized_memory
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }
