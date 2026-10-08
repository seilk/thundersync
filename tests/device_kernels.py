"""Skip a kernel test on a device the kernel does not support.

An importable build says nothing about the device. These helpers run the
same one-time trial the runtime dispatch runs and skip with the device's
compute capability when the trial fails, so a kernel-specific test fails
only where the kernel is supposed to work.
"""

from __future__ import annotations

import pytest


def require_fa4(head_dim: int = 128) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("flash_attn.cute.interface")
    from thundersync.accel.capability import device_identity, kernel_runs
    from thundersync.engine import streaming

    like = torch.empty(1, device="cuda", dtype=torch.bfloat16)
    if not kernel_runs("fa4", like, head_dim, streaming._fa4_trial):
        identity = device_identity(like.device)
        pytest.skip(
            f"FA4 does not run on {identity.get('name')} "
            f"(compute capability {identity.get('compute_capability')}) at head_dim {head_dim}"
        )
