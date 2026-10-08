"""A per-sandbox budget on what policy commands write to the scratch tree.

A sandbox's worktree and ``/tmp`` are host directories under the sandbox
scratch root, which a site may put on a RAM-backed filesystem; a policy
command such as ``yes > /tmp/f`` then fills host memory at disk-cache
speed. ``ScratchGuard`` bounds how far one trajectory may grow its
sandbox's scratch:

* usage is the allocated bytes (``st_blocks``) of every file and directory
  under the sandbox's scratch directories, each inode counted once and
  symlinks not followed, so a sparse file costs only what it occupies.
  Files that vanish or cannot be read while the tree is walked are not
  counted.
* the budget is on growth over a baseline: the usage when the guard is
  made, before the trajectory's first tool call.
* ``run_guarded`` checks the budget before a policy command (an
  over-budget sandbox does not start it), lets ``run_bounded`` measure
  it while the command runs, and measures it after the command returns.
  At a breach the command is killed (with anything still running in the
  sandbox), and files the command created or changed are removed, largest
  first, until the growth is back within the budget. The observation keeps
  the command's own text and ends in a note naming the breach and the
  removal. A sandbox that cannot be brought back within its budget is
  marked so that a pool discards it instead of reusing it.

The measurement walks the tree, so the watchdog interval grows with the
cost of a walk: at least ``MIN_CHECK_INTERVAL_S``, else ``CHECK_COST_FACTOR``
times the last walk.
"""

from __future__ import annotations

import logging
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .bounded_output import CommandOutput, command_output

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_SCRATCH_SHARE",
    "MINIMUM_WRITE_BUDGET_BYTES",
    "OVER_BUDGET_ATTRIBUTE",
    "Reclaim",
    "ScratchGuard",
    "allocated_bytes",
    "default_write_budget",
    "ensure_private_dir",
    "over_scratch_budget",
    "run_guarded",
    "scratch_paths",
]

# The share of the scratch filesystem the policy sandboxes' growth may take
# together; the rest holds the sandboxes' baselines, the verifiers and
# whatever else the site keeps there.
DEFAULT_SCRATCH_SHARE = 0.5
# No default budget below this: a normal build or install writes this much.
MINIMUM_WRITE_BUDGET_BYTES = 256 << 20
MIN_CHECK_INTERVAL_S = 0.5
CHECK_COST_FACTOR = 10
# ctime comes from the kernel's coarse clock, which may trail time_ns().
_CTIME_SLACK_NS = 1_000_000_000
# The attribute that marks a sandbox a pool must discard.
OVER_BUDGET_ATTRIBUTE = "_thundersync_scratch_over_budget"


def ensure_private_dir(path: Path) -> None:
    """Create ``path`` mode 0700, or accept an existing directory this user owns.

    The default scratch roots have predictable names under ``/tmp``; another
    user who creates one first, or plants a symlink there, would otherwise
    receive every sandbox's working tree.
    """

    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        os.mkdir(path, 0o700)
    except FileExistsError:
        pass
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RuntimeError(f"scratch directory {path} is not a plain directory")
    if info.st_uid != os.getuid():
        raise RuntimeError(f"scratch directory {path} is owned by uid {info.st_uid}, not this user")
    if info.st_mode & stat.S_IWOTH:
        raise RuntimeError(f"scratch directory {path} is writable by other users")


def _walk(paths: Iterable[Path | str], visit: Callable[[str, os.stat_result], None]) -> None:
    pending = [os.fspath(path) for path in paths]
    while pending:
        directory = pending.pop()
        try:
            entries = os.scandir(directory)
        except OSError:
            continue
        with entries:
            for entry in entries:
                try:
                    info = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                visit(entry.path, info)
                if stat.S_ISDIR(info.st_mode):
                    pending.append(entry.path)


def _allocated(info: os.stat_result) -> int:
    blocks = getattr(info, "st_blocks", None)
    return info.st_size if blocks is None else blocks * 512


def allocated_bytes(paths: Iterable[Path | str]) -> int:
    """Allocated bytes under ``paths``; each inode once, symlinks not followed."""
    total = 0
    linked: set[tuple[int, int]] = set()

    def visit(_path: str, info: os.stat_result) -> None:
        nonlocal total
        if info.st_nlink > 1 and not stat.S_ISDIR(info.st_mode):
            key = (info.st_dev, info.st_ino)
            if key in linked:
                return
            linked.add(key)
        total += _allocated(info)

    _walk(paths, visit)
    return total


def scratch_paths(sandbox: Any) -> list[Path]:
    """The host directories a sandbox writes, or none it declares."""
    mounts = getattr(sandbox, "scratch_mounts", None)
    return [] if mounts is None else list(mounts())


def default_write_budget(
    scratch_dir: Path | str,
    sandboxes: int,
    *,
    share: float = DEFAULT_SCRATCH_SHARE,
) -> int | None:
    """``share`` of the scratch filesystem split across ``sandboxes``.

    At least ``MINIMUM_WRITE_BUDGET_BYTES``; None when the filesystem's size
    is unknown or ``share`` is 0.
    """
    if share <= 0:
        return None
    if share > 1:
        raise ValueError(f"a scratch share is at most 1, not {share}")
    if sandboxes <= 0:
        raise ValueError(f"the sandbox count is positive, not {sandboxes}")
    directory = Path(scratch_dir)
    while not directory.exists() and directory != directory.parent:
        directory = directory.parent
    try:
        info = os.statvfs(directory)
    except OSError:
        return None
    size = info.f_blocks * info.f_frsize
    if size <= 0:
        return None
    return max(int(size * share) // sandboxes, MINIMUM_WRITE_BUDGET_BYTES)


@dataclass(frozen=True)
class Reclaim:
    """What a breach's cleanup removed and where it left the sandbox."""

    removed_files: int
    removed_bytes: int
    growth_bytes: int
    within_budget: bool


class ScratchGuard:
    """One trajectory's write budget on one sandbox's scratch directories."""

    def __init__(
        self,
        paths: Iterable[Path | str],
        budget_bytes: int,
        *,
        measure: Callable[[Iterable[Path]], int] = allocated_bytes,
    ) -> None:
        if budget_bytes <= 0:
            raise ValueError(f"a write budget is positive, not {budget_bytes}")
        self.paths = tuple(Path(path) for path in paths)
        self.budget_bytes = budget_bytes
        self._measure = measure
        self._walk_s = 0.0
        self.started_ns = time.time_ns()
        self.baseline_bytes = self.usage()

    def usage(self) -> int:
        started = time.perf_counter()
        usage = self._measure(self.paths)
        self._walk_s = time.perf_counter() - started
        return usage

    @property
    def check_interval_s(self) -> float:
        return max(MIN_CHECK_INTERVAL_S, CHECK_COST_FACTOR * self._walk_s)

    def over_budget(self) -> int | None:
        """The growth over the baseline when it exceeds the budget, else None."""
        growth = self.usage() - self.baseline_bytes
        return growth if growth > self.budget_bytes else None

    def reclaim(
        self,
        since_ns: int,
        *,
        remove_in_sandbox: Callable[[list[str]], None] | None = None,
        mounts: Mapping[Path, str] | None = None,
    ) -> Reclaim:
        """Remove files changed since ``since_ns``, largest first, to the budget.

        A file the host may not remove is removed inside the sandbox
        (``remove_in_sandbox`` with its in-sandbox path through ``mounts``).
        """
        changed: list[tuple[int, str]] = []
        since = since_ns - _CTIME_SLACK_NS

        def visit(path: str, info: os.stat_result) -> None:
            if stat.S_ISREG(info.st_mode) and info.st_ctime_ns >= since:
                changed.append((_allocated(info), path))

        _walk(self.paths, visit)
        changed.sort(reverse=True)
        # Largest first, only as many as bring the growth within the budget.
        growth = self.usage() - self.baseline_bytes
        planned: list[tuple[int, str]] = []
        for size, path in changed:
            if growth <= self.budget_bytes:
                break
            planned.append((size, path))
            growth -= size
        removed_files = removed_bytes = 0
        refused: list[tuple[int, str]] = []
        for size, path in planned:
            try:
                os.unlink(path)
            except FileNotFoundError:
                continue
            except OSError:
                refused.append((size, path))
                continue
            removed_files += 1
            removed_bytes += size
        inside = {
            path: mapped
            for _size, path in refused
            if (mapped := _in_sandbox(Path(path), mounts or {})) is not None
        }
        if inside and remove_in_sandbox is not None:
            try:
                remove_in_sandbox(list(inside.values()))
            except Exception:  # the measurement below decides
                logger.warning("in-sandbox scratch removal failed", exc_info=True)
            for size, path in refused:
                if path in inside and not os.path.lexists(path):
                    removed_files += 1
                    removed_bytes += size
        growth = self.usage() - self.baseline_bytes
        return Reclaim(
            removed_files=removed_files,
            removed_bytes=removed_bytes,
            growth_bytes=growth,
            within_budget=growth <= self.budget_bytes,
        )


def _in_sandbox(path: Path, mounts: Mapping[Path, str]) -> str | None:
    for host, inside in mounts.items():
        try:
            relative = path.relative_to(host)
        except ValueError:
            continue
        return str(Path(inside) / relative)
    return None


def _breach(
    sandbox: Any, guard: ScratchGuard, growth: int, since_ns: int
) -> tuple[dict[str, Any], str]:
    mounts = getattr(sandbox, "scratch_mounts", None)
    cleaned = guard.reclaim(
        since_ns,
        remove_in_sandbox=getattr(sandbox, "remove_in_sandbox", None),
        mounts=None if mounts is None else mounts(),
    )
    if not cleaned.within_budget:
        setattr(sandbox, OVER_BUDGET_ATTRIBUTE, True)
    event = {
        "growth_bytes": growth,
        "budget_bytes": guard.budget_bytes,
        "removed_files": cleaned.removed_files,
        "removed_bytes": cleaned.removed_bytes,
        "growth_after_bytes": cleaned.growth_bytes,
        "within_budget": cleaned.within_budget,
    }
    files = "file" if cleaned.removed_files == 1 else "files"
    note = (
        f"<{cleaned.removed_files} {files} the command created or changed "
        f"({cleaned.removed_bytes} bytes) removed"
    )
    if not cleaned.within_budget:
        note += (
            f"; the sandbox's files are still {cleaned.growth_bytes} bytes "
            "over their starting size"
        )
    return event, note + ">"


def run_guarded(
    sandbox: Any,
    command: str,
    guard: ScratchGuard | None,
    **kwargs: Any,
) -> CommandOutput:
    """A policy command's output under ``guard``; see the module docstring.

    Without a guard it is ``command_output``. With one, an output whose
    sandbox stayed within the budget is exactly that output.
    """
    if guard is None:
        return command_output(sandbox, command, **kwargs)
    growth = guard.over_budget()
    if growth is not None:
        event, note = _breach(sandbox, guard, growth, guard.started_ns)
        event["refused"] = True
        refused = CommandOutput(()).with_scratch(
            "<command not run: the sandbox's files grew by "
            f"{growth} bytes, past the {guard.budget_bytes} byte write budget>\n{note}",
            event,
            refused=True,
        )
        _report(sandbox, command, event)
        return refused
    started_ns = time.time_ns()
    output = command_output(sandbox, command, write_guard=guard, **kwargs)
    if output.killed == "write_budget":
        # The sandbox killed the command and whatever it left running.
        growth = guard.usage() - guard.baseline_bytes
        lead = "\n"
    else:
        growth = guard.over_budget()
        if growth is None:
            return output
        lead = (
            "<the sandbox's files grew by "
            f"{growth} bytes, past the {guard.budget_bytes} byte write budget>\n"
        )
        # A descendant that left the command's process group may still write.
        stop = getattr(sandbox, "stop_commands", None)
        if stop is not None:
            stop()
    event, note = _breach(sandbox, guard, growth, started_ns)
    event["refused"] = False
    _report(sandbox, command, event)
    return output.with_scratch(lead + note, event)


def over_scratch_budget(sandbox: Any, sealed_bytes: int | None, budget_bytes: int | None) -> bool:
    """Whether a sandbox may not be reused: marked, or grown past the budget.

    ``sealed_bytes`` is its scratch usage when it was sealed; a sandbox
    whose leftovers (``/tmp`` is not reset between uses) grew it past the
    budget is not reused.
    """
    if getattr(sandbox, OVER_BUDGET_ATTRIBUTE, False):
        return True
    if sealed_bytes is None or budget_bytes is None:
        return False
    return allocated_bytes(scratch_paths(sandbox)) - sealed_bytes > budget_bytes


def _report(sandbox: Any, command: str, event: Mapping[str, Any]) -> None:
    logger.warning(
        "sandbox %s write budget: %s command=%r",
        getattr(sandbox, "name", "?"),
        " ".join(f"{key}={value}" for key, value in event.items()),
        command[:200],
    )
