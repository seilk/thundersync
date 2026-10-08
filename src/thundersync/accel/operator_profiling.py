"""Disabled-by-default operator ranges for profiling.

The training implementation imports :func:`operator_range` at selected phase
boundaries.  With profiling disabled, the context manager performs no PyTorch
or CUDA calls.  A profiler enables Kineto ranges, NVTX ranges, or both, either
with :func:`configure_operator_profiling` or through the
``THUNDERSYNC_OPERATOR_PROFILE_RANGES`` environment variable, which is read
on first use and must name one of the same modes.

Range names use a stable, machine-readable format::

    thundersync::<category>|field=value|field=value

The fields are diagnostics only.  They are derived from tensor metadata and do
not participate in the training computation.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from typing import Iterator, Mapping

import torch


_VALID_MODES = frozenset(("off", "kineto", "nvtx", "both"))
PROFILE_RANGES_ENV = "THUNDERSYNC_OPERATOR_PROFILE_RANGES"
# None until the first use reads the environment or a caller configures it.
_mode: str | None = None
_device_roles: dict[str, str] = {}


def _normalized_mode(mode: str, *, source: str) -> str:
    normalized = mode.strip().lower()
    if normalized not in _VALID_MODES:
        raise ValueError(
            f"{source} must be one of {sorted(_VALID_MODES)}, got {mode!r}"
        )
    return normalized


def _resolved_mode() -> str:
    """The active mode, read from the environment on first use."""

    global _mode
    if _mode is None:
        _mode = _normalized_mode(
            os.environ.get(PROFILE_RANGES_ENV, "off"), source=PROFILE_RANGES_ENV
        )
    return _mode


def configure_operator_profiling(
    mode: str,
    *,
    device_roles: Mapping[str, str] | None = None,
) -> None:
    """Configure range emission for the current process.

    Configuration is process-global so worker threads used by two-device OPD
    replay observe the same setting.  NVTX range stacks remain thread-local.
    """

    normalized = _normalized_mode(mode, source="operator profiling mode")
    global _mode, _device_roles
    _mode = normalized
    _device_roles = {
        str(torch.device(device)): str(role)
        for device, role in (device_roles or {}).items()
    }


def operator_profiling_mode() -> str:
    """Return the active profiling mode."""

    return _resolved_mode()


def operator_profiling_enabled() -> bool:
    """Return whether the current process emits operator ranges."""

    return _resolved_mode() != "off"


def device_role(device: torch.device | str) -> str:
    """Return the configured semantic role for a logical device."""

    canonical = str(torch.device(device))
    return _device_roles.get(canonical, canonical)


def _format_value(value: object) -> str:
    text = str(value)
    return text.replace("|", "/").replace("\n", " ")


def range_name(category: str, **fields: object) -> str:
    """Construct the stable name used by Kineto and NVTX."""

    parts = [f"thundersync::{category}"]
    parts.extend(
        f"{key}={_format_value(value)}"
        for key, value in fields.items()
        if value is not None
    )
    return "|".join(parts)


class _DisabledRange:
    def __enter__(self) -> None:
        return None

    def __exit__(self, _exc_type, _exc_value, _traceback) -> bool:
        return False


_DISABLED_RANGE = _DisabledRange()


@contextmanager
def _enabled_operator_range(
    mode: str,
    category: str,
    fields: Mapping[str, object],
) -> Iterator[None]:
    name = range_name(category, **fields)
    use_kineto = mode in ("kineto", "both")
    use_nvtx = mode in ("nvtx", "both")
    record = torch.profiler.record_function(name) if use_kineto else None
    pushed = False
    if record is not None:
        record.__enter__()
    try:
        if use_nvtx:
            torch.cuda.nvtx.range_push(name)
            pushed = True
        yield
    finally:
        if pushed:
            torch.cuda.nvtx.range_pop()
        if record is not None:
            record.__exit__(None, None, None)


def operator_range(
    category: str,
    *,
    fields: Mapping[str, object] | None = None,
    **named_fields: object,
):
    """Return a disabled singleton or an active profiling context manager."""

    mode = _mode
    if mode is None:
        mode = _resolved_mode()
    if mode == "off":
        return _DISABLED_RANGE
    combined = dict(fields or ())
    combined.update(named_fields)
    return _enabled_operator_range(mode, category, combined)
