"""Reusable per-task sandbox pool behind a trusted pristine gate.

A container create copies a full image tree (udocker) or starts a
container over a copied worktree (docker), then seals the bug state with
several in-container commands; a destroy deletes what the create made.
The pool pays that once per container and afterwards reuses it: a release resets the sealed bug state in-container
and then verifies the tree ON THE HOST against the trusted bug-base object
store captured at setup (engine._capture_trusted_bug_tree). The gate fails
closed: a container whose tree does not reproduce the sealed state -- a
mangled in-container Git, a stray tracked edit, an untracked leftover --
is closed and discarded, and the next acquire pays full creation again.

``report`` carries where the time went: seconds blocked on the member cap,
spent creating (with the backend's per-phase ``start_timings`` summed when
the sandbox records them) and spent discarding.

Role soundness:

* verifier containers execute trusted verification commands only; reuse is
  unconditionally sound under the pristine gate.
* policy containers are exposed to policy commands. The gate restores the
  repository tree exactly, but non-repository container state (installed
  packages, files outside the worktree) is NOT restored, so a reused policy
  container is not equivalent to a fresh one.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from typing import Any, Callable

from .bounded_output import run_bounded
from .engine import (
    HOST_GIT_CONFIG,
    HOST_GIT_ENV,
    Task,
    _capture_bug_base,
    _capture_trusted_bug_tree,
    new_sandbox,
)

logger = logging.getLogger(__name__)

_RESET_COMMAND = (
    "GIT_NO_REPLACE_OBJECTS=1 git checkout -q -- . && "
    "GIT_NO_REPLACE_OBJECTS=1 git clean -fdxq && "
    "echo THUNDERSYNC_POOL_RESET_OK"
)
_GIT_TIMEOUT_S = 120
# Any leftover path fails the gate, so the listing is read no further than
# this; a policy can create files without bound.
_LEFTOVER_LISTING_BYTES = 1 << 20


class SandboxPoolError(RuntimeError):
    pass


def default_pool_setup(sandbox: Any, task: Task) -> None:
    """Bring a fresh container to the sealed bug state with a trusted base.

    The trusted host snapshot is mandatory for pool members: the pristine
    gate is only as strong as the object store it compares against, so it
    is captured even for a task without a bug patch.
    """

    sandbox.start()
    if task.bug_patch:
        sandbox.apply_bug_patch(task.bug_patch)
        sandbox.assert_bug_not_revertible()
        _capture_bug_base(sandbox, task, trusted_host_snapshot=True)
    else:
        _capture_trusted_bug_tree(sandbox, "HEAD")


def trusted_tree_pristine(sandbox: Any) -> bool:
    """Host-side check: does the worktree reproduce the sealed bug tree?

    Authoritative by construction -- it reads the trusted bare object store
    captured before any use, never the container's own (possibly mangled)
    Git state. Any failure to prove pristineness returns False.
    """

    git_dir = getattr(sandbox, "_thundersync_trusted_git_dir", None)
    tree = getattr(sandbox, "_thundersync_trusted_tree", None)
    workdir = getattr(sandbox, "host_workdir", None)
    if git_dir is None or tree is None or workdir is None:
        return False
    base = [
        "git",
        *HOST_GIT_CONFIG,
        f"--git-dir={git_dir}",
        f"--work-tree={workdir}",
    ]
    # The same runtime-state exclusions as trusted policy-patch capture: the
    # container repo's own .git and generated Python caches are runtime
    # state, not policy source changes, and the capture contract already
    # treats them that way.
    runtime_state_excludes = [
        ":(exclude).git",
        ":(exclude,glob).git/**",
        ":(exclude,glob)**/__pycache__/**",
        ":(exclude,glob)**/*.pyc",
        ":(exclude,glob)**/*.pyo",
        ":(exclude,glob)**/*.egg-info/**",
        ":(exclude,glob)**/.pytest_cache/**",
    ]
    try:
        # Policy-patch capture legitimately mutates the trusted store's
        # INDEX during a trajectory; the gate compares against the
        # immutable tree object, so restore the index first to keep the
        # verdict independent of per-use index state.
        restored = subprocess.run(
            ["git", *HOST_GIT_CONFIG, f"--git-dir={git_dir}", "read-tree", tree],
            capture_output=True,
            env={**os.environ, **HOST_GIT_ENV},
            timeout=_GIT_TIMEOUT_S,
        )
        if restored.returncode != 0:
            return False
        # Porcelain diff, not diff-index: the freshly read index carries no
        # stat cache, and diff-index would report every file as modified.
        tracked = subprocess.run(
            [*base, "diff", "--quiet", tree, "--"],
            capture_output=True,
            env={**os.environ, **HOST_GIT_ENV},
            timeout=_GIT_TIMEOUT_S,
        )
        if tracked.returncode != 0:
            return False
        others = run_bounded(
            [*base, "ls-files", "--others", "--", ".", *runtime_state_excludes],
            env={**os.environ, **HOST_GIT_ENV},
            timeout=_GIT_TIMEOUT_S,
            keep_bytes=_LEFTOVER_LISTING_BYTES,
            kill_bytes=_LEFTOVER_LISTING_BYTES,
        )
        if others.timed_out or others.killed is not None or others.returncode != 0:
            return False
        if others.stdout.raw is None:
            return False
        leftover = [
            line
            for line in others.stdout.raw.decode(errors="replace").splitlines()
            if line
        ]
        return not leftover
    except (OSError, subprocess.SubprocessError):
        return False


class SandboxPool:
    """Per-task free list of sealed-state containers.

    ``acquire`` hands out any pristine idle container for the task, or
    creates one through the backend's normal path. ``release`` resets
    and gates it; only a proven-pristine container returns to the free
    list. Thread-safe; container operations run outside the pool lock.
    """

    def __init__(
        self,
        factory: Callable[[str, str], Any],
        *,
        name_prefix: str,
        max_uses: int = 0,
        max_members: int = 0,
        setup: Callable[[Any, Task], None] = default_pool_setup,
        reusable: Callable[[Any], bool] | None = None,
    ) -> None:
        if max_uses < 0:
            raise ValueError("max_uses must be zero (unlimited) or positive")
        if max_members < 0:
            raise ValueError("max_members must be zero (unlimited) or positive")
        self._factory = factory
        self._name_prefix = name_prefix
        self._max_uses = max_uses
        self._max_members = max_members
        self._setup = setup
        # A caller's own reuse condition beside the pristine gate (the
        # rollout worker's scratch write budget).
        self._reusable = reusable
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._idle: dict[str, list[Any]] = {}
        self._uses: dict[int, int] = {}
        self._members = 0
        self._sequence = 0
        self._closed = False
        self.created = 0
        self.reused = 0
        self.discarded = 0
        self.evicted = 0
        self.pristine_failures = 0
        self._seconds = {"wait": 0.0, "create": 0.0, "create_max": 0.0, "discard": 0.0}
        self._start_phase_seconds: dict[str, float] = {}

    def acquire(self, task: Task) -> Any:
        """Reuse, create, evict-then-create, or wait — bounded by the cap.

        Containers cost gigabytes each; without a member cap the task-keyed
        idle lists grow with task spread until the store fills the disk.
        At the cap an idle member
        of another task is evicted to make room; with every member busy the
        acquire waits for a release.
        """

        while True:
            evict = None
            with self._lock:
                if self._closed:
                    raise SandboxPoolError("sandbox pool is closed")
                free = self._idle.get(task.instance_id)
                if free:
                    sandbox = free.pop()
                    self.reused += 1
                    return sandbox
                if self._max_members == 0 or self._members < self._max_members:
                    self._members += 1
                    self._sequence += 1
                    sequence = self._sequence
                    break
                for members in self._idle.values():
                    if members:
                        evict = members.pop(0)
                        break
                if evict is None:
                    blocked = time.perf_counter()
                    self._condition.wait(timeout=60.0)
                    self._seconds["wait"] += time.perf_counter() - blocked
                    continue
                self.evicted += 1
                self._uses.pop(id(evict), None)
            self._discard(evict)
        name = f"{self._name_prefix}-{sequence}"
        sandbox = None
        started = time.perf_counter()
        try:
            sandbox = new_sandbox(self._factory, task, name)
            self._setup(sandbox, task)
        except BaseException:
            if sandbox is not None:
                self._discard(sandbox, count=False)
            else:
                with self._condition:
                    self._members -= 1
                    self._condition.notify_all()
            raise
        elapsed = time.perf_counter() - started
        phases = getattr(sandbox, "start_timings", None)
        with self._lock:
            self.created += 1
            self._uses[id(sandbox)] = 0
            self._seconds["create"] += elapsed
            self._seconds["create_max"] = max(self._seconds["create_max"], elapsed)
            if isinstance(phases, dict):
                for phase, seconds in phases.items():
                    self._start_phase_seconds[phase] = (
                        self._start_phase_seconds.get(phase, 0.0) + float(seconds)
                    )
        return sandbox

    def release(self, task: Task, sandbox: Any) -> None:
        """Reset, gate, and return the container -- or discard it.

        The in-container reset is best-effort (a policy may have mangled
        the container Git); the host-side trusted gate alone decides.
        """

        # A reused member must never carry per-use grading state forward:
        # grade_isolated consults these attributes BEFORE capturing a fresh
        # policy patch, so a stale cache would grade the previous
        # trajectory's diff.
        for attribute in (
            "_thundersync_policy_patch_cache",
            "_thundersync_policy_patch_rejection",
            "_thundersync_policy_patch_capture_error",
        ):
            if hasattr(sandbox, attribute):
                delattr(sandbox, attribute)
        try:
            output = sandbox.exec(_RESET_COMMAND)
            reset_ok = "THUNDERSYNC_POOL_RESET_OK" in output
        except Exception:
            logger.warning(
                "sandbox pool %s: reset failed for %s",
                self._name_prefix,
                task.instance_id,
                exc_info=True,
            )
            reset_ok = False
        pristine = reset_ok and trusted_tree_pristine(sandbox)
        reusable = True
        if pristine and self._reusable is not None:
            try:
                reusable = bool(self._reusable(sandbox))
            except Exception:
                logger.warning(
                    "sandbox pool %s: reuse check failed for %s",
                    self._name_prefix,
                    task.instance_id,
                    exc_info=True,
                )
                reusable = False
        with self._condition:
            uses = self._uses.get(id(sandbox), 0) + 1
            self._uses[id(sandbox)] = uses
            expired = self._max_uses > 0 and uses >= self._max_uses
            keep = pristine and reusable and not expired and not self._closed
            if keep:
                self._idle.setdefault(task.instance_id, []).append(sandbox)
                # A capped acquire may be waiting to evict an idle member.
                self._condition.notify_all()
            else:
                self._uses.pop(id(sandbox), None)
                if not pristine:
                    self.pristine_failures += 1
        if not keep:
            # A discard forces the next acquire back through full creation;
            # every discard is logged with its cause.
            healthy = reset_ok and pristine and reusable
            logger.log(
                logging.INFO if healthy else logging.WARNING,
                "sandbox pool %s: discarding member for %s: reset_ok=%s "
                "pristine=%s reusable=%s uses=%s expired=%s closed=%s",
                self._name_prefix,
                task.instance_id,
                reset_ok,
                pristine,
                reusable,
                uses,
                expired,
                self._closed,
            )
            self._discard(sandbox)

    def close(self) -> None:
        """Close every idle container. Busy containers must be released first."""

        with self._lock:
            self._closed = True
            idle = [
                sandbox
                for free in self._idle.values()
                for sandbox in free
            ]
            self._idle.clear()
            self._uses.clear()
        for sandbox in idle:
            self._discard(sandbox)

    def retire(self) -> list[Any]:
        """Close the pool without touching its containers; return the idle ones.

        The shutdown form: the caller disposes of the returned members in
        bulk instead of one close per member -- under udocker by pruning the
        store as a whole, under docker in one batch
        (`docker_sandbox.retire_sandboxes`). Busy containers must have been
        released first, as for `close`.
        """

        with self._lock:
            self._closed = True
            retired = [sandbox for free in self._idle.values() for sandbox in free]
            self._idle.clear()
            self._uses.clear()
        return retired

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "created": self.created,
                "reused": self.reused,
                "discarded": self.discarded,
                "evicted": self.evicted,
                "members": self._members,
                "pristine_failures": self.pristine_failures,
                "idle": sum(len(free) for free in self._idle.values()),
                "wait_s": round(self._seconds["wait"], 3),
                "create_s": round(self._seconds["create"], 3),
                "create_max_s": round(self._seconds["create_max"], 3),
                "discard_s": round(self._seconds["discard"], 3),
                "start_phase_s": {
                    phase: round(seconds, 3)
                    for phase, seconds in sorted(self._start_phase_seconds.items())
                },
            }

    def _discard(self, sandbox: Any, *, count: bool = True) -> None:
        started = time.perf_counter()
        trusted_root = getattr(sandbox, "_thundersync_trusted_root", None)
        try:
            sandbox.close()
        except Exception:
            logger.warning(
                "sandbox pool %s: closing a discarded member failed",
                self._name_prefix,
                exc_info=True,
            )
        if trusted_root is not None:
            shutil.rmtree(trusted_root, ignore_errors=True)
        elapsed = time.perf_counter() - started
        with self._condition:
            # The slot frees whether or not the discard is counted; a capped
            # acquire may be waiting on it.
            self._members -= 1
            if count:
                self.discarded += 1
            self._seconds["discard"] += elapsed
            self._condition.notify_all()
