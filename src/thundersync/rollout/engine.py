"""Batched rollout engine over vLLM's offline API.

Deliberately not the OpenAI server: the prefix tree needs *exact* generated
token ids, and the HTTP API returns token strings whose re-encoding can shift
boundaries and manufacture divergence that never happened. The offline engine
hands back ``token_ids`` directly.

Two rollout schedules are implemented, selected by ``rollout_group(schedule=...)``.

``schedule="lockstep"``: every live rollout generates
one action, then all tools run, then the next action begins.

The whole lockstep batch waits on the slowest sandbox in each round, so its
environment wall time includes harness barrier cost.

``schedule="async"``: one worker thread per rollout, each owning its own turn
loop. Three barriers present in the lockstep path are removed, none of which the
dependency structure requires.

* Round barrier. A rollout issues its next generation request as soon as ITS OWN
  sandbox returns, without waiting for siblings. Generation stays batched: the
  workers submit to :class:`_GenerationBroker`, which coalesces whatever requests
  are outstanding into one ``generate`` call.
* Group barrier on grading. Each trajectory is graded inside its own worker at
  its own finish, so ``on_verdict`` fires per trajectory rather than once the
  slowest sibling has finished.
* Sequential groups. :meth:`BatchedRolloutEngine.rollout_groups` runs several
  groups at once against a shared broker, so their rollouts batch together.

What the schedule changes is WHEN work runs. What is sampled is unchanged:
every generation request carries its own seed (:func:`request_seed`) and vLLM
seeds each sequence independently, so the returned ids are a function of
``(prompt_token_ids, seed)`` alone and do not depend on which siblings shared
the call. Token-exactness across the two schedules is asserted by the test
suite.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import queue
import shlex
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from secrets import token_hex
from typing import Callable, Sequence

from .bounded_output import command_output, completed_process, run_bounded
from .agent import (
    CONTEXT_TEMPLATE,
    SYSTEM_PROMPT,
    TASK_TEMPLATE,
    Step,
    Task,
    Trajectory,
    build_repo_context,
    parse_action,
    truncate_observation_tokens,
)
from .docker_sandbox import DockerSandbox

logger = logging.getLogger(__name__)


@dataclass
class GenResult:
    token_ids: list[int]
    logprobs: list[float]
    finish_reason: str
    text: str
    ready_at: float | None = None


@dataclass(frozen=True)
class CommittedBlock:
    """One immutable causal block exposed at its runtime readiness point.

    Action tokens become immutable when generation returns. Observation tokens
    become immutable when the environment call returns and its text has been
    tokenized. ``committed_at`` is captured before callback-lock acquisition,
    so it records source readiness even when callback delivery is delayed.
    ``on_block`` callbacks receive these records before the combined
    ``on_step`` callback. A frozen OPD teacher can process action tokens while
    the corresponding tool executes, without a token-count boundary.

    ``terminal_reason_at_commit`` records termination that is known when an
    action block becomes immutable. ``submit`` is known from the parsed action;
    ``max_steps`` is known from the configured loop bound. Termination decided
    after the action commit, such as ``context_limit`` or ``generation_limit``,
    remains absent here.
    """

    step_index: int
    kind: str
    token_ids: tuple[int, ...]
    scored: tuple[bool, ...]
    committed_at: float
    terminal_reason_at_commit: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("action", "observation"):
            raise ValueError(f"unknown committed block kind {self.kind!r}")
        if not self.token_ids:
            raise ValueError("a committed block must contain at least one token")
        if len(self.token_ids) != len(self.scored):
            raise ValueError("committed block tokens and score mask must align")
        if self.kind == "observation" and any(self.scored):
            raise ValueError(
                "observation blocks cannot contain scored policy tokens"
            )
        if self.terminal_reason_at_commit not in (None, "submit", "max_steps"):
            raise ValueError(
                "committed block terminal reason must be submit, max_steps, "
                "or None"
            )
        if self.kind != "action" and self.terminal_reason_at_commit is not None:
            raise ValueError(
                "only action blocks can declare a terminal reason at commit"
            )
        if not math.isfinite(self.committed_at) or self.committed_at < 0.0:
            raise ValueError("committed block timestamp must be finite and non-negative")


@dataclass
class BatchTimings:
    generation_s: float = 0.0
    generation_batch_sizes: list[int] = field(default_factory=list)
    """Batch size of each physical generation call on the lockstep path."""
    shared_generation_s: float = 0.0
    """Physical vLLM wall time of the shared async broker.

    ``rollout_groups(schedule="async")`` copies the same batch-level value to
    every group's timing record. Read it once per call; do not sum it over
    groups.
    """
    shared_generation_batch_sizes: list[int] = field(default_factory=list)
    """Batch size of each physical generation call issued by the async broker."""
    shared_generation_prompt_tokens: int = 0
    shared_generation_completion_tokens: int = 0
    tool_s: float = 0.0
    """Wall-clock the batch spent blocked on sandboxes, GPU idle.

    Under ``schedule="lockstep"`` the phases do not overlap, so this is
    wall-clock. Under ``schedule="async"`` it is the SUM over rollouts of each
    rollout's own sandbox time, and those intervals overlap in real time; use
    ``rollout_wall_s`` for the makespan and never add these two together.
    """
    verify_s: float = 0.0
    """Grading. Also GPU-idle, and also counted against whole-loop MFU.

    Sums over rollouts under ``schedule="async"``, where grading overlaps the
    turn loops of slower siblings.
    """
    graded_at: float = 0.0
    """time.perf_counter() at the moment the last verdict landed -- the instant
    a streaming trainer's TTW clock starts. 0.0 when scoring was skipped."""
    schedule: str = "lockstep"
    """Which path produced these numbers. Recorded so a timing artifact can
    never be read without it."""
    rollout_wall_s: float = 0.0
    """Makespan of the turn-loop phase: perf_counter across the whole loop.

    Under async this includes the grading that ran inside the workers, because
    grading a finished trajectory overlaps a sibling still taking turns.
    """
    verdict_at: list[float] = field(default_factory=list)
    """Per-rollout perf_counter at which that rollout's verdict landed.

    Indexed by rollout_id, 0.0 where scoring was skipped. ``max(verdict_at)``
    equals ``graded_at``; the spread between min and max is what the group
    grading barrier costs the earliest finisher.
    """
    trajectory_completed_at: list[float] = field(default_factory=list)
    """Per-rollout timestamp after its final emitted turn and before grading.

    This is the earliest legal close event for reward-independent objectives
    such as OPD.  It is separate from ``verdict_at`` because SWE verification
    may take seconds and OPD does not consume its reward.
    """
    n_steps: int = 0
    """Lockstep: rounds executed. Async: the largest turn count of any rollout,
    since rounds do not exist on that path."""
    tool_calls: int = 0
    model_rows: int = 0
    """Physical model row-steps actually executed."""
    logical_rows: int = 0
    """Logical row-steps the trainer believes it received.

    The ratio ``logical_rows / model_rows`` is the physical-work identity L/U --
    measured, not inferred from token coincidence.
    """

    @property
    def compression(self) -> float:
        return self.logical_rows / max(self.model_rows, 1)


# The observation returned for a reply that contains no parsable action.
PARSE_ERROR_OBSERVATION = "No bash block found. Reply with one ```bash ... ``` block."


def parse_test_ids(fail_to_pass) -> list[str]:
    """Test IDs from a sequence or a JSON-encoded list; raises ``ValueError``
    on a string that is not a JSON list."""
    if isinstance(fail_to_pass, str):
        try:
            decoded = json.loads(fail_to_pass)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"test IDs must be a sequence or a JSON list, got {fail_to_pass[:200]!r}"
            ) from exc
        if not isinstance(decoded, list):
            raise ValueError(
                f"test IDs must decode to a JSON list, got {type(decoded).__name__}"
            )
        return [str(test_id) for test_id in decoded]
    return list(fail_to_pass or [])


@dataclass
class Verdict:
    """Why a rollout scored what it scored. Never reduce this to a float silently."""

    reward: float
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    """Requested but absent from the report -- a collection error, not a pass."""
    regressed: list[str] = field(default_factory=list)
    """PASS_TO_PASS tests broken by the edit. Any of these voids the fix."""
    n_pass_to_pass: int = 0
    reason: str = ""

    @property
    def conclusive(self) -> bool:
        """False when the harness could not determine the outcome at all."""
        return not self.missing and self.reason not in ("timeout", "no_report")


def _reward_and_reason(value) -> tuple[float, str]:
    if isinstance(value, Verdict):
        return value.reward, value.reason
    return float(value), ""


class PolicyPatchRejected(RuntimeError):
    """The policy worktree cannot be represented as a bounded repository patch."""


# A verifier report is parsed whole. One larger than this is read no
# further and counts as absent, which is inconclusive, as an unparsable
# report always was.
REPORT_KEEP_BYTES = 64 << 20


# Git's own messages may accompany a patch at the limit.
_PATCH_STDERR_ALLOWANCE = 1 << 20


def _read_report(sandbox, path: str) -> str | None:
    """The report file's text, or None when it is larger than the parser takes."""
    output = command_output(
        sandbox, f"cat {path} 2>/dev/null", keep_bytes=REPORT_KEEP_BYTES
    )
    if not output.complete or output.killed is not None:
        logger.warning(
            "verifier report %s not read whole: stdout_bytes=%s killed=%s",
            path,
            output.stdout_bytes,
            output.killed,
        )
        return None
    return output.text


def new_sandbox(factory: Callable[[str, str], object], task: Task, name: str):
    """``factory(task.image_name, name)``, carrying the task's network need.

    A task with ``requires_network`` sets ``requires_network = True`` on the
    sandbox before it starts; factories keep the ``(image, name)`` signature.
    """
    sandbox = factory(task.image_name, name)
    if task.requires_network:
        sandbox.requires_network = True
    return sandbox


def _parse_junit(xml: str | None) -> ET.Element | None:
    """The JUnit report's root element, or None for an absent or malformed report.

    A document type declaration or entity declaration is malformed here:
    pytest never writes one, and refusing it keeps entity expansion out of
    the parser's reach.
    """
    if xml is None or "<testsuite" not in xml:
        return None
    if "<!DOCTYPE" in xml or "<!ENTITY" in xml:
        return None
    try:
        return ET.fromstring(xml)
    except ET.ParseError:
        return None


# Host Git runs against trees the task image or the policy wrote. No
# configuration may run a program there: no system or global config (and so
# no filter drivers from it), no fsmonitor, no hooks.
HOST_GIT_ENV = {
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": os.devnull,
}
HOST_GIT_CONFIG = ("-c", "core.fsmonitor=false", "-c", f"core.hooksPath={os.devnull}")


def _capture_trusted_bug_tree(sandbox, bug_base: str) -> None:
    """Snapshot the sealed source tree outside the policy-visible worktree.

    The snapshot is a bare Git object store containing only the baseline tree
    and its blobs.  Policy commands may replace ``/testbed``, delete ``.git``,
    or create new history; later patch capture reads this independent object
    store and the current host-backed worktree.
    """
    workdir = getattr(sandbox, "host_workdir", None)
    if workdir is None:
        raise RuntimeError("trusted patch capture requires a host-backed sandbox")
    cwd = Path(workdir)
    root = Path(
        tempfile.mkdtemp(prefix=f".thundersync-trusted-{cwd.name}-", dir=cwd.parent)
    )
    git_dir = root / "baseline.git"
    env = {**os.environ, **HOST_GIT_ENV}
    try:
        initialized = subprocess.run(
            ["git", "init", "--bare", "--quiet", str(git_dir)],
            env=env,
            capture_output=True,
            timeout=120,
        )
        if initialized.returncode:
            raise RuntimeError(
                "could not initialize trusted Git store: "
                f"{initialized.stderr.decode(errors='replace')[:300]}"
            )
        tracked = subprocess.run(
            [
                "git",
                *HOST_GIT_CONFIG,
                "-C",
                str(cwd),
                "ls-tree",
                "-r",
                "--name-only",
                "-z",
                bug_base,
            ],
            env=env,
            capture_output=True,
            timeout=120,
        )
        if tracked.returncode:
            raise RuntimeError(
                "could not enumerate sealed source tree: "
                f"{tracked.stderr.decode(errors='replace')[:300]}"
            )
        added = subprocess.run(
            [
                "git",
                f"--git-dir={git_dir}",
                f"--work-tree={cwd}",
                "add",
                "-f",
                "--pathspec-from-file=-",
                "--pathspec-file-nul",
            ],
            input=tracked.stdout,
            env=env,
            capture_output=True,
            timeout=120,
        )
        if added.returncode:
            raise RuntimeError(
                "could not snapshot sealed source files: "
                f"{added.stderr.decode(errors='replace')[:300]}"
            )
        written = subprocess.run(
            ["git", f"--git-dir={git_dir}", "write-tree"],
            env=env,
            capture_output=True,
            timeout=120,
        )
        tree = written.stdout.decode().strip()
        if written.returncode or len(tree) != 40:
            raise RuntimeError(
                "could not write trusted source tree: "
                f"{written.stderr.decode(errors='replace')[:300]}"
            )
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise
    old_root = getattr(sandbox, "_thundersync_trusted_root", None)
    if old_root is not None:
        shutil.rmtree(old_root, ignore_errors=True)
    sandbox._thundersync_trusted_root = root
    sandbox._thundersync_trusted_git_dir = git_dir
    sandbox._thundersync_trusted_tree = tree


def _capture_bug_base(
    sandbox, task: Task, *, trusted_host_snapshot: bool = False
) -> str | None:
    """Capture the sealed buggy commit before the policy can change history."""
    if not task.bug_patch:
        return None
    value = sandbox.exec(
        "GIT_NO_REPLACE_OBJECTS=1 git rev-parse HEAD"
    ).strip().splitlines()[0]
    if len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise RuntimeError(
            f"{task.instance_id}: could not capture the sealed bug commit: {value!r}"
        )
    sandbox._thundersync_bug_base = value
    if trusted_host_snapshot:
        _capture_trusted_bug_tree(sandbox, value)
    return value


def _source_diff(sandbox, bug_base: str | None) -> str:
    """Return changes from the sealed bug state, including committed fixes."""
    if getattr(sandbox, "_thundersync_trusted_git_dir", None) is not None:
        try:
            patch = _capture_policy_patch(sandbox, max_bytes=8_000_000)
            sandbox._thundersync_policy_patch_cache = patch
        except PolicyPatchRejected as exc:
            sandbox._thundersync_policy_patch_rejection = str(exc)
            return f"policy patch rejected: {exc}"
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            sandbox._thundersync_policy_patch_capture_error = str(exc)
            return f"policy patch capture failed: {exc}"
        if not patch:
            return ""
        summary = subprocess.run(
            ["git", "apply", "--stat"],
            input=patch,
            capture_output=True,
            timeout=120,
        )
        return summary.stdout.decode(errors="replace").strip() or (
            f"repository patch: {len(patch)} bytes"
        )

    target = bug_base or "HEAD"
    tracked = sandbox.exec(
        f"GIT_NO_REPLACE_OBJECTS=1 git diff {target} --stat -- . "
        "2>/dev/null | tail -20"
    ).strip()
    untracked = sandbox.exec(
        "GIT_NO_REPLACE_OBJECTS=1 git ls-files --others --exclude-standard "
        "-- . 2>/dev/null | head -20"
    ).strip()
    if untracked:
        tracked = f"{tracked}\nuntracked files:\n{untracked}".strip()
    return tracked


def _is_oracle_controlled(path: str, exact_test_files: set[str]) -> bool:
    """Whether grading must restore this repository path from the bug commit."""
    clean = path.removeprefix("./")
    item = PurePosixPath(clean)
    directories = {part.lower() for part in item.parts[:-1]}
    name = item.name.lower()
    return (
        clean in exact_test_files
        or bool(directories & {"test", "tests"})
        or name == "conftest.py"
        or (name.startswith("test_") and name.endswith(".py"))
        or name.endswith("_test.py")
        or name.endswith("_test.go")
        or name
        in {
            "pytest.ini",
            "tox.ini",
            "setup.cfg",
            "pyproject.toml",
            "sitecustomize.py",
        }
    )


def _exec_idempotent_with_marker(
    sandbox,
    command: str,
    *,
    marker: str = "THUNDERSYNC_ORACLE_RESTORE_OK",
    attempts: int = 3,
) -> str:
    """Retry a trusted idempotent command when its completion marker is absent.

    Each udocker exec is a fresh ``udocker run``, a new PRoot process, and
    under verifier concurrency that startup can fail before the command begins.
    Oracle restoration and tree enumeration are idempotent, so retrying only
    those trusted operations preserves fail-closed grading without replaying a
    generated policy command.
    """
    output = ""
    for _attempt in range(attempts):
        output = sandbox.exec(command)
        if marker in output:
            return output
    return output


def _capture_policy_patch(sandbox, *, max_bytes: int) -> bytes:
    """Capture committed, uncommitted, and untracked work against the bug state.

    The docker and udocker sandboxes expose their worktree as a host directory
    (``host_workdir``). Running the host's Git binary on it keeps policy
    changes to the container environment out of patch capture. The diff is
    taken against the trusted baseline snapshot (``_capture_trusted_bug_tree``),
    never through the policy's own
    ``.git``, so committed policy work shows up as changes against the sealed
    bug state.  Intent-to-add entries include new files in the binary patch.
    """
    workdir = getattr(sandbox, "host_workdir", None)
    if workdir is None:
        raise RuntimeError("isolated grading requires a host-backed sandbox")
    cwd = Path(workdir)
    env = {**os.environ, **HOST_GIT_ENV}
    trusted_git_dir = getattr(sandbox, "_thundersync_trusted_git_dir", None)
    trusted_tree = getattr(sandbox, "_thundersync_trusted_tree", None)
    # The external trusted Git store treats the policy repository's ``.git``
    # directory, generated Python caches, and package-install metadata as
    # ordinary untracked files. They are runtime state rather than policy source
    # changes. Including them can make a valid source patch fail to apply when
    # the fresh verifier generated the same paths during startup.
    trusted_pathspec = [
        ".",
        ":(exclude).git",
        ":(exclude,glob).git/**",
        ":(exclude,glob)**/__pycache__/**",
        ":(exclude,glob)**/*.pyc",
        ":(exclude,glob)**/*.pyo",
        ":(exclude,glob)**/*.egg-info/**",
        ":(exclude,glob)**/.pytest_cache/**",
    ]
    if trusted_git_dir is None or trusted_tree is None:
        # Host Git must never read the policy-written ``.git``: its config
        # (core.fsmonitor, filter drivers) would run on the host.
        raise RuntimeError("patch capture requires the trusted baseline snapshot")
    git = [
        "git",
        *HOST_GIT_CONFIG,
        f"--git-dir={Path(trusted_git_dir)}",
        f"--work-tree={cwd}",
    ]
    preparation = (
        (
            [
                "git",
                *HOST_GIT_CONFIG,
                f"--git-dir={Path(trusted_git_dir)}",
                "cat-file",
                "-e",
                f"{trusted_tree}^{{tree}}",
            ],
            False,
        ),
        (
            [
                "git",
                *HOST_GIT_CONFIG,
                f"--git-dir={Path(trusted_git_dir)}",
                "read-tree",
                trusted_tree,
            ],
            False,
        ),
        ([*git, "add", "-N", "-f", "--", *trusted_pathspec], True),
    )
    diff_target = trusted_tree
    for command, policy_dependent in preparation:
        result = subprocess.run(
            command, cwd=cwd, env=env, capture_output=True, timeout=120
        )
        if result.returncode:
            exception = PolicyPatchRejected if policy_dependent else RuntimeError
            raise exception(
                f"could not capture policy patch ({' '.join(command)}): "
                f"{result.stderr.decode(errors='replace')[:300]}"
            )
    # Read no further than the limit: the worktree is policy-written, and
    # its diff has no size of its own.
    run = run_bounded(
        [
            *git,
            "--no-pager",
            "diff",
            "--binary",
            "--full-index",
            "--no-ext-diff",
            "--no-textconv",
            diff_target,
            "--",
            *trusted_pathspec,
        ],
        cwd=cwd,
        env=env,
        timeout=120,
        keep_bytes=max_bytes + 1,
        kill_bytes=max_bytes + _PATCH_STDERR_ALLOWANCE,
    )
    completed_process(run, timeout=120)
    if run.killed is not None:
        raise PolicyPatchRejected(
            f"policy patch is over {run.stdout.nbytes} bytes; limit is {max_bytes}"
        )
    if run.returncode:
        stderr = run.stderr.raw if run.stderr.raw is not None else b""
        raise PolicyPatchRejected(
            "could not capture policy patch: "
            f"{stderr.decode(errors='replace')[:300]}"
        )
    if run.stdout.nbytes > max_bytes:
        raise PolicyPatchRejected(
            f"policy patch is {run.stdout.nbytes} bytes; limit is {max_bytes}"
        )
    return run.stdout.raw


def grade_isolated(
    sandbox,
    task: Task,
    *,
    sandbox_factory: Callable[[str, str], object],
    verifier_name: str,
    check_regressions: bool = True,
    max_patch_bytes: int = 8_000_000,
    verifier_pool=None,
) -> Verdict:
    """Grade a repository patch inside a fresh or pooled sandbox instance.

    Policy commands can modify more than the repository.  The fresh verifier
    starts from the task image, receives the sealed bug patch, then receives
    only the repository diff captured by the host Git binary.  Test restoration
    in :func:`grade` remains a second independent control.

    With ``verifier_pool`` the verifier comes from a SandboxPool whose
    members were brought to the sealed bug state at creation and whose
    every return passes the trusted host-side pristine gate, so the
    per-verification container create/destroy leaves the verification
    path. Verifier containers only ever run
    trusted commands; the policy patch still arrives exclusively as the
    host-captured repository diff.
    """
    captured = capture_isolated_patch(sandbox, max_patch_bytes=max_patch_bytes)
    if isinstance(captured, Verdict):
        return captured
    return grade_captured_patch(
        captured,
        task,
        sandbox_factory=sandbox_factory,
        verifier_name=verifier_name,
        check_regressions=check_regressions,
        verifier_pool=verifier_pool,
    )


def capture_isolated_patch(
    sandbox, *, max_patch_bytes: int = 8_000_000
) -> bytes | Verdict:
    """The half of :func:`grade_isolated` that reads the policy sandbox.

    Returns the host-captured repository patch, or the zero verdict that
    ends grading without a verifier. Only reads the per-use capture state
    :func:`_source_diff` leaves on the sandbox; never writes it.
    """
    bug_base = getattr(sandbox, "_thundersync_bug_base", None)
    if not bug_base:
        return Verdict(0.0, reason="missing_bug_base")
    try:
        rejection = getattr(sandbox, "_thundersync_policy_patch_rejection", None)
        capture_error = getattr(sandbox, "_thundersync_policy_patch_capture_error", None)
        cached_patch = getattr(sandbox, "_thundersync_policy_patch_cache", None)
        if rejection is not None:
            return Verdict(0.0, reason=f"policy_patch_rejected:{rejection}")
        if capture_error is not None:
            return Verdict(0.0, reason=f"policy_patch_capture_failed:{capture_error}")
        return (
            cached_patch
            if cached_patch is not None
            else _capture_policy_patch(sandbox, max_bytes=max_patch_bytes)
        )
    except PolicyPatchRejected as exc:
        return Verdict(0.0, reason=f"policy_patch_rejected:{exc}")
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        return Verdict(0.0, reason=f"policy_patch_capture_failed:{exc}")


def grade_captured_patch(
    policy_patch: bytes,
    task: Task,
    *,
    sandbox_factory: Callable[[str, str], object],
    verifier_name: str,
    check_regressions: bool = True,
    verifier_pool=None,
) -> Verdict:
    """The verifier half of :func:`grade_isolated`.

    Reads nothing from the policy sandbox: the captured bytes are its only
    input from the policy, so it may run while the policy keeps acting.
    """
    if verifier_pool is not None:
        verifier = verifier_pool.acquire(task)
    else:
        verifier = new_sandbox(sandbox_factory, task, verifier_name)
    patch_path: Path | None = None
    try:
        if verifier_pool is None:
            verifier.start()
            if task.bug_patch:
                verifier.apply_bug_patch(task.bug_patch)
                verifier.assert_bug_not_revertible()
            _capture_bug_base(verifier, task)
        if policy_patch:
            host_workdir = getattr(verifier, "host_workdir", None)
            if host_workdir is None:
                return Verdict(0.0, reason="verifier_not_host_backed")
            patch_path = Path(host_workdir) / ".thundersync_policy.patch"
            patch_path.write_bytes(policy_patch)
            applied = verifier.exec(
                "GIT_NO_REPLACE_OBJECTS=1 git apply --binary --whitespace=nowarn "
                ".thundersync_policy.patch 2>/dev/null "
                "&& rm -f .thundersync_policy.patch "
                "&& echo THUNDERSYNC_POLICY_PATCH_OK"
            )
            patch_path.unlink(missing_ok=True)
            patch_path = None
            if "THUNDERSYNC_POLICY_PATCH_OK" not in applied:
                return Verdict(0.0, reason="policy_patch_rejected:apply_failed")
        return grade(
            verifier,
            task,
            restore_tests=True,
            check_regressions=check_regressions,
        )
    finally:
        if patch_path is not None:
            patch_path.unlink(missing_ok=True)
        if verifier_pool is not None:
            verifier_pool.release(task, verifier)
        else:
            verifier.close()


def _selector_batches(
    selectors: list[str], *, max_items: int = 500, max_quoted_chars: int = 96_000
) -> list[list[str]]:
    """Batch selectors below both pytest-count and shell-argument bounds."""
    batches: list[list[str]] = []
    batch: list[str] = []
    quoted_chars = 0
    for selector in selectors:
        selector_chars = len(shlex.quote(selector)) + 1
        if batch and (
            len(batch) >= max_items
            or quoted_chars + selector_chars > max_quoted_chars
        ):
            batches.append(batch)
            batch = []
            quoted_chars = 0
        batch.append(selector)
        quoted_chars += selector_chars
    if batch:
        batches.append(batch)
    return batches


def _run_pytest_junit(
    sandbox: DockerSandbox, node_ids: list[str], *, chunk: int = 40
) -> tuple[dict[str, str], str]:
    """Run tests and parse pytest's JUnit XML. Returns (id -> outcome, error reason).

    Outcomes come from the structured report, never from console text: a
    passing test whose output contains the word "Error" is still a pass.
    Each call writes its reports under a fresh random name in the sandbox's
    ``/tmp``, so no file left there beforehand is read as a report.
    Trailing-``::`` selectors identify custom-collected files in repositories
    such as Pygments.  Direct collection otherwise lets pytest's Python plugin
    claim ``.py`` example files and report their sample syntax as collection
    errors.  These selectors run with the Python collector disabled; the
    repository's nested collector remains active.  Return code and complete
    JUnit contents determine the outcome of each bounded selector batch.
    """
    outcomes: dict[str, str] = {}
    report_id = token_hex(8)
    custom_file_selectors = [t for t in node_ids if t.endswith("::")]
    plain_file_selectors = [t for t in node_ids if "::" not in t]
    node_selectors = [
        t for t in node_ids if t not in custom_file_selectors + plain_file_selectors
    ]

    file_batches = [
        (batch, "-p no:python")
        for batch in _selector_batches(custom_file_selectors, max_items=chunk)
    ] + [([selector], "") for selector in plain_file_selectors]
    for i, (batch, collector_option) in enumerate(file_batches):
        quoted = " ".join(shlex.quote(selector) for selector in batch)
        xml_path = f"/tmp/thundersync_file_report_{report_id}_{i}.xml"
        output = sandbox.exec(
            f"rm -f {xml_path}; "
            f"python -m pytest --junit-xml={xml_path} -q --no-header "
            f"-p no:cacheprovider {collector_option} --tb=no {quoted} "
            ">/dev/null 2>&1; "
            "rc=$?; printf 'THUNDERSYNC_PYTEST_RC=%s\\n' \"$rc\""
        )
        marker = "THUNDERSYNC_PYTEST_RC="
        return_codes = [
            line.removeprefix(marker)
            for line in output.splitlines()
            if line.startswith(marker)
        ]
        if len(return_codes) != 1 or not return_codes[0].isdigit():
            reason = "timeout" if "timed out" in output.lower() else "no_report"
            return outcomes, reason
        return_code = int(return_codes[0])
        root = _parse_junit(_read_report(sandbox, xml_path))
        if root is None:
            return outcomes, "no_report"
        cases = list(root.iter("testcase"))
        if len(cases) < len(batch):
            for selector in batch:
                outcomes[selector] = "failed"
            continue
        bad = return_code != 0 or any(
            case.find(tag) is not None
            for case in cases
            for tag in ("failure", "error", "skipped")
        )
        for selector in batch:
            outcomes[selector] = "failed" if bad else "passed"

    for i, batch in enumerate(
        _selector_batches(node_selectors, max_items=chunk)
    ):
        quoted = " ".join(shlex.quote(t) for t in batch)
        xml_path = f"/tmp/thundersync_report_{report_id}_{i}.xml"
        # The marker proves pytest returned at all; the removal keeps a
        # batch that writes no report from reading a stale file.
        output = sandbox.exec(
            f"rm -f {xml_path}; "
            f"python -m pytest --junit-xml={xml_path} -q --no-header "
            f"-p no:cacheprovider --tb=no {quoted} >/dev/null 2>&1; "
            "rc=$?; printf 'THUNDERSYNC_PYTEST_RC=%s\\n' \"$rc\""
        )
        if "THUNDERSYNC_PYTEST_RC=" not in output:
            reason = "timeout" if "timed out" in output.lower() else "no_report"
            return outcomes, reason
        root = _parse_junit(_read_report(sandbox, xml_path))
        if root is None:
            return outcomes, "no_report"
        for case in root.iter("testcase"):
            cls = case.get("classname", "") or ""
            name = case.get("name", "") or ""
            bad = any(
                case.find(tag) is not None for tag in ("failure", "error", "skipped")
            )
            outcomes[f"{cls}::{name}"] = "failed" if bad else "passed"
            outcomes[name] = "failed" if bad else "passed"
    return outcomes, ""


def _run_go_test_json(
    sandbox: DockerSandbox, test_names: list[str]
) -> tuple[dict[str, str], str]:
    """Run the Go repository suite once and parse terminal JSON test events.

    For ``Task.test_framework == "go"`` the selectors are test names rather
    than file-qualified node IDs, as in SWE-smith's Go tasks.  The whole
    module runs once under ``go test -json ./...`` and the named outcomes are
    graded.  A nonzero suite exit is expected when a target test still fails,
    so report presence and per-test terminal events determine the verdict.
    """
    if not test_names:
        return {}, ""
    report_path = f"/tmp/thundersync_go_report_{token_hex(8)}.jsonl"
    output = sandbox.exec(
        f"rm -f {report_path}; "
        "if [ -x /usr/local/go/bin/go ]; then "
        "export PATH=/go/bin:/usr/local/go/bin:$PATH "
        "GOROOT=/usr/local/go GOPATH=/go GOTOOLCHAIN=local GOTELEMETRY=off; "
        "fi; "
        f"go test -json -count=1 ./... >{report_path} 2>&1; "
        "rc=$?; printf 'THUNDERSYNC_GO_TEST_RC=%s\\n' \"$rc\""
    )
    marker = "THUNDERSYNC_GO_TEST_RC="
    return_codes = [
        line.removeprefix(marker)
        for line in output.splitlines()
        if line.startswith(marker)
    ]
    if len(return_codes) != 1 or not return_codes[0].isdigit():
        reason = "timeout" if "timed out" in output.lower() else "no_report"
        return {}, reason

    report = _read_report(sandbox, report_path)
    if report is None or not report.strip():
        return {}, "no_report"
    terminal: dict[str, list[str]] = {}
    parsed_record = False
    for line in report.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return {}, "no_report"
        if not isinstance(event, dict):
            return {}, "no_report"
        parsed_record = True
        name = event.get("Test")
        action = event.get("Action")
        if isinstance(name, str) and action in {"pass", "fail", "skip"}:
            terminal.setdefault(name, []).append(action)
    if not parsed_record:
        return {}, "no_report"

    requested = set(test_names)
    outcomes = {
        name: "passed" if actions and all(x == "pass" for x in actions) else "failed"
        for name, actions in terminal.items()
        if name in requested
    }
    return outcomes, ""


def _pytest_selector_candidates(selector: str) -> list[str]:
    """JUnit keys that can represent one pytest selector.

    Parameter values may contain ``::`` themselves.  Structural separators are
    recognized only outside square brackets, so a selector such as
    ``test_func[dict::builder]`` retains its complete parameterized name.
    """
    parts: list[str] = []
    start = 0
    depth = 0
    index = 0
    while index < len(selector):
        character = selector[index]
        if character == "[":
            depth += 1
        elif character == "]" and depth:
            depth -= 1
        elif selector[index : index + 2] == "::" and depth == 0:
            parts.append(selector[start:index])
            start = index + 2
            index += 1
        index += 1
    parts.append(selector[start:])
    if len(parts) < 2:
        return [selector]
    path, *nodes = parts
    module = path.removeprefix("./")
    if module.endswith(".py"):
        module = module[:-3]
    module = module.replace("/", ".")
    name = nodes[-1]
    classname = ".".join([module, *nodes[:-1]])
    candidates = [selector, f"{classname}::{name}", name]
    if len(nodes) > 1:
        candidates.append("::".join(nodes))
    return list(dict.fromkeys(candidates))


def grade(
    sandbox: DockerSandbox,
    task: Task,
    *,
    restore_tests: bool = True,
    check_regressions: bool = True,
) -> Verdict:
    """The standard SWE oracle: FAIL_TO_PASS all pass **and** PASS_TO_PASS all pass.

    Checking only FAIL_TO_PASS is the common shortcut and it is wrong. SWE-smith
    ships a few target tests against hundreds of regression tests; grading on the targets alone
    scores a "fix" that repairs the target while deleting a guard elsewhere as a
    success.

    Four properties are required for the reward to be trustworthy:

    * **no false negatives** -- correct work must never be scored as failure, so
      the outcome comes from pytest's structured report rather than from matching
      words in console text;
    * **no false positives** -- a fix that breaks PASS_TO_PASS is not a fix;
    * **no silent truncation** -- every requested test is run, in batches;
    * **no ambiguous pass** -- a test missing from the report (collection error,
      deleted file) is not a pass, and the verdict says so.

    ``restore_tests`` reverts the graded test files first: editing the assertions
    is a shorter path to reward than fixing the code, and an RL loop will find it.
    """
    tests = parse_test_ids(task.fail_to_pass)
    if not tests:
        return Verdict(0.0, reason="no_fail_to_pass")

    if restore_tests:
        baseline = getattr(sandbox, "_thundersync_bug_base", None) or "HEAD"
        p2p_all = parse_test_ids(task.pass_to_pass) if check_regressions else []
        files = sorted(
            {t.split("::")[0] for t in list(tests) + list(p2p_all) if t.endswith(".py") or "::" in t}
        )
        for i in range(0, len(files), 60):  # keep the command line bounded
            quoted_files = " ".join(shlex.quote(f) for f in files[i : i + 60])
            restored = _exec_idempotent_with_marker(
                sandbox,
                f"git checkout {baseline} -- {quoted_files} 2>/dev/null "
                "&& echo THUNDERSYNC_ORACLE_RESTORE_OK"
            )
            if "THUNDERSYNC_ORACLE_RESTORE_OK" not in restored:
                return Verdict(0.0, reason="oracle_restore_failed")

        # A policy can commit a test helper or a new conftest.py, so restoring
        # only the named test is insufficient.  Enumerate both trees outside a
        # shell pipeline, remove additions in oracle-controlled locations, and
        # restore the complete baseline test surface.  Each repository
        # operation has an explicit success marker.
        baseline_listing = _exec_idempotent_with_marker(
            sandbox,
            f"git cat-file -e {baseline}^{{commit}} && "
            f"git ls-tree -r --name-only {baseline} && "
            "echo THUNDERSYNC_ORACLE_RESTORE_OK"
        )
        current_listing = _exec_idempotent_with_marker(
            sandbox,
            "find . -type f -not -path './.git/*' -print "
            "&& echo THUNDERSYNC_ORACLE_RESTORE_OK"
        )
        if (
            "THUNDERSYNC_ORACLE_RESTORE_OK" not in baseline_listing
            or "THUNDERSYNC_ORACLE_RESTORE_OK" not in current_listing
        ):
            return Verdict(0.0, reason="oracle_restore_failed")
        exact_test_files = set(files)
        baseline_oracle_files = sorted(
            path
            for path in baseline_listing.splitlines()
            if _is_oracle_controlled(path, exact_test_files)
        )
        current_oracle_files = sorted(
            path.removeprefix("./")
            for path in current_listing.splitlines()
            if _is_oracle_controlled(path, exact_test_files)
        )
        added_oracle_files = sorted(
            set(current_oracle_files) - set(baseline_oracle_files)
        )
        for paths, operation in (
            (added_oracle_files, "rm -f --"),
            (baseline_oracle_files, f"git checkout {baseline} --"),
        ):
            for i in range(0, len(paths), 60):
                quoted_paths = " ".join(shlex.quote(path) for path in paths[i : i + 60])
                restored = _exec_idempotent_with_marker(
                    sandbox,
                    f"{operation} {quoted_paths} 2>/dev/null "
                    "&& echo THUNDERSYNC_ORACLE_RESTORE_OK"
                )
                if "THUNDERSYNC_ORACLE_RESTORE_OK" not in restored:
                    return Verdict(0.0, reason="oracle_restore_failed")

    p2p = parse_test_ids(task.pass_to_pass) if check_regressions else []
    framework = task.test_framework
    if framework == "go":
        outcomes, err = _run_go_test_json(
            sandbox, list(dict.fromkeys([*tests, *p2p]))
        )
    else:
        outcomes, err = _run_pytest_junit(sandbox, tests)
    if err:
        return Verdict(0.0, reason=err)

    passed, failed, missing = [], [], []
    for t in tests:
        key = next(
            (candidate for candidate in _pytest_selector_candidates(t) if candidate in outcomes),
            None,
        )
        if key is None:
            missing.append(t)
        elif outcomes[key] == "passed":
            passed.append(t)
        else:
            failed.append(t)

    if failed or missing:
        return Verdict(
            reward=0.0,
            passed=passed,
            failed=failed,
            missing=missing,
            reason="missing_tests" if missing else "failing_tests",
        )

    # target tests pass; now the half that was missing -- did the edit break
    # anything else? Only run this when it can change the answer.
    regressed: list[str] = []
    if p2p:
        if framework == "go":
            p2p_out, p2p_err = outcomes, ""
        else:
            p2p_out, p2p_err = _run_pytest_junit(sandbox, p2p)
        if p2p_err:
            return Verdict(
                reward=0.0, passed=passed, n_pass_to_pass=len(p2p), reason=p2p_err
            )
        for t in p2p:
            key = next(
                (
                    candidate
                    for candidate in _pytest_selector_candidates(t)
                    if candidate in p2p_out
                ),
                None,
            )
            if key is None or p2p_out[key] != "passed":
                regressed.append(t)

    ok = not regressed
    return Verdict(
        reward=1.0 if ok else 0.0,
        passed=passed,
        failed=failed,
        missing=missing,
        regressed=regressed,
        n_pass_to_pass=len(p2p),
        reason="" if ok else "regressions",
    )


def run_tests(
    sandbox: DockerSandbox,
    task: Task,
    *,
    restore_tests: bool = True,
    check_regressions: bool = True,
) -> float:
    """Binary reward. Thin wrapper over :func:`grade`, which carries the detail.

    Binary by design: a group with no reward spread is then exactly a group
    whose group-relative advantages are all zero.
    """
    return grade(
        sandbox, task, restore_tests=restore_tests, check_regressions=check_regressions
    ).reward


_STOP = object()

COUPLED_ROLLOUT_SLOT = 0
"""Rollout slot whose seed a coupled-prefix step samples with, so the shared
action is the one that rollout draws at that turn when uncoupled."""


def request_seed(base_seed: int, group_id: int, rollout_id: int, turn: int) -> int:
    """Sampling seed of one generation request, a non-negative 63-bit integer.

    A stable hash (BLAKE2b) of the four signed 64-bit integers: distinct
    ``(group_id, rollout_id, turn)`` get distinct seeds except with negligible
    probability, the value does not depend on the schedule, and it is the
    same in every process and Python version.
    """
    packed = struct.pack("<4q", base_seed, group_id, rollout_id, turn)
    digest = hashlib.blake2b(packed, digest_size=8).digest()
    return int.from_bytes(digest, "little") >> 1


def _check_prompt_cap(max_prompt_tokens: int) -> None:
    if (
        isinstance(max_prompt_tokens, bool)
        or not isinstance(max_prompt_tokens, int)
        or max_prompt_tokens < 1
    ):
        raise ValueError(
            f"max_prompt_tokens must be a positive integer, not {max_prompt_tokens!r}"
        )


def _check_prompt_length(
    task: Task, prompt_ids: Sequence[int], max_prompt_tokens: int
) -> None:
    """Refuse an initial task prompt longer than ``max_prompt_tokens``."""
    if len(prompt_ids) > max_prompt_tokens:
        raise ValueError(
            f"{task.instance_id}: the initial prompt is {len(prompt_ids)} tokens, "
            f"over max_prompt_tokens={max_prompt_tokens}"
        )


def _limited_call(semaphore: threading.Semaphore | None, action, *args, **kwargs):
    if semaphore is None:
        return action(*args, **kwargs)
    with semaphore:
        return action(*args, **kwargs)


@dataclass
class _GenRequest:
    """One rollout's generation request, waiting on its own event."""

    prompt: list[int]
    seed: int | None
    max_tokens: int
    temperature: float
    done: threading.Event
    result: GenResult | None = None
    error: BaseException | None = None


class _GenerationBroker:
    """Serves per-rollout generation requests out of batched engine calls.

    Removing the round barrier must not serialize generation one sequence at a
    time, which would replace a barrier with a throughput collapse. Worker
    threads submit their own prompt and block on their own event; a dispatcher
    thread drains whatever requests are already outstanding and issues ONE
    ``engine.generate`` call for them. A rollout whose sandbox returned early
    therefore rides the next call rather than waiting for a round to close.

    Batch composition is a scheduling decision and not a sampling one. Every
    request carries its own seed and vLLM seeds each sequence independently, so
    the ids a request receives are a function of ``(prompt_token_ids, seed)``
    alone; which siblings share the call changes when a request is served and
    not what it returns. That property is what allows the async path to claim
    token-exactness against lockstep, and it is asserted directly by the test suite.

    ``linger_s`` is the coalescing window: after the first request arrives the
    dispatcher waits up to this long for siblings before calling the engine.
    It trades a bounded per-turn latency for batch size.
    """

    def __init__(
        self,
        engine,
        *,
        max_batch: int = 64,
        linger_s: float = 0.002,
        request_timeout_s: float = 300.0,
    ):
        self._engine = engine
        self._max_batch = max_batch
        self._linger_s = linger_s
        self._request_timeout_s = request_timeout_s
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._loop, name="thundersync-genbroker", daemon=True
        )
        self.batch_sizes: list[int] = []
        """Size of every ``generate`` call actually issued. Diagnostic only:
        it reports how much batching survived the removal of the barrier."""
        self.generation_s = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    # -- lifecycle ----------------------------------------------------

    def start(self) -> "_GenerationBroker":
        self._thread.start()
        return self

    def close(self) -> None:
        self._q.put(_STOP)
        self._thread.join(timeout=30.0)
        if self._thread.is_alive():
            raise RuntimeError("generation broker did not stop within 30 seconds")

    def __enter__(self) -> "_GenerationBroker":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    # -- worker side --------------------------------------------------

    def submit(
        self,
        prompt: Sequence[int],
        *,
        seed: int | None,
        max_tokens: int,
        temperature: float,
    ) -> GenResult:
        """Block this worker until its own completion is ready."""
        req = _GenRequest(
            prompt=list(prompt),
            seed=seed,
            max_tokens=max_tokens,
            temperature=temperature,
            done=threading.Event(),
        )
        self._q.put(req)
        deadline = time.monotonic() + self._request_timeout_s
        while not req.done.wait(
            timeout=min(1.0, max(0.0, deadline - time.monotonic()))
        ):
            if not self._thread.is_alive():
                raise RuntimeError(
                    "generation broker stopped before completing a request"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "generation broker request exceeded "
                    f"{self._request_timeout_s:g} seconds"
                )
        if req.error is not None:
            raise req.error
        if req.result is None:
            raise RuntimeError("generation broker completed a request without a result")
        return req.result

    # -- dispatcher side ----------------------------------------------

    def _collect(self) -> tuple[list[_GenRequest], bool]:
        first = self._q.get()
        if first is _STOP:
            return [], True
        batch = [first]
        deadline = time.perf_counter() + self._linger_s
        stop = False
        while len(batch) < self._max_batch:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                nxt = self._q.get(timeout=remaining)
            except queue.Empty:
                break
            if nxt is _STOP:
                stop = True
                break
            batch.append(nxt)
        return batch, stop

    def _loop(self) -> None:
        stop = False
        while not stop:
            batch, stop = self._collect()
            if not batch:
                continue
            # generate() takes scalar max_tokens/temperature, so a mixed batch
            # is split on those; seeds stay per request either way.
            groups: dict[tuple, list[_GenRequest]] = {}
            for r in batch:
                groups.setdefault(
                    (r.max_tokens, r.temperature, r.seed is None), []
                ).append(r)
            for reqs in groups.values():
                self._serve(reqs)

    def _serve(self, reqs: list[_GenRequest]) -> None:
        try:
            started = time.perf_counter()
            outs = self._engine.generate(
                [r.prompt for r in reqs],
                max_tokens=reqs[0].max_tokens,
                temperature=reqs[0].temperature,
                seeds=None if reqs[0].seed is None else [r.seed for r in reqs],
            )
            self.generation_s += time.perf_counter() - started
            if len(outs) != len(reqs):
                raise RuntimeError(
                    f"generate returned {len(outs)} results for {len(reqs)} "
                    "prompts; the broker cannot attribute completions"
                )
            self.batch_sizes.append(len(reqs))
            self.prompt_tokens += sum(len(r.prompt) for r in reqs)
            self.completion_tokens += sum(len(o.token_ids) for o in outs)
            ready_at = time.perf_counter()
            for r, o in zip(reqs, outs, strict=True):
                o.ready_at = ready_at
                r.result = o
        except BaseException as exc:  # noqa: BLE001 -- re-raised in every worker
            for r in reqs:
                r.error = exc
        finally:
            for r in reqs:
                r.done.set()


class BatchedRolloutEngine:
    def __init__(
        self,
        model_path: str | Path,
        *,
        tensor_parallel_size: int,
        max_model_len: int = 131072,
        gpu_memory_utilization: float = 0.65,
        enable_prefix_caching: bool = True,
        seed: int = 0,
        enable_lora: bool = False,
        max_lora_rank: int = 32,
        enable_chunked_prefill: bool | None = True,
        max_num_seqs: int | None = 16,
        max_num_batched_tokens: int | None = None,
    ):
        # How many devices one engine spans is a site decision the caller
        # passes; the engine assumes none.
        if isinstance(tensor_parallel_size, bool) or not isinstance(
            tensor_parallel_size, int
        ) or tensor_parallel_size < 1:
            raise ValueError(
                f"tensor_parallel_size must be a positive integer, not {tensor_parallel_size!r}"
            )
        from vllm import LLM  # imported lazily; heavy

        kwargs = {}
        if enable_lora:
            kwargs.update(enable_lora=True, max_lora_rank=max_lora_rank, max_loras=1)
        if max_num_seqs is not None:
            kwargs["max_num_seqs"] = max_num_seqs
        if max_num_batched_tokens is not None:
            kwargs["max_num_batched_tokens"] = max_num_batched_tokens
        if enable_chunked_prefill is not None:
            kwargs["enable_chunked_prefill"] = enable_chunked_prefill
        self.llm = LLM(
            model=str(model_path),
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
            seed=seed,
            disable_log_stats=True,
            **kwargs,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.max_model_len = max_model_len
        # Set to a vllm.lora.request.LoRARequest to sample from base + adapter.
        # Swapped between steps by the RL loop; vLLM loads the new adapter on
        # first use (hot swap, no engine restart).
        self.lora_request = None

    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    # ------------------------------------------------------------------

    def generate(
        self,
        prompts: Sequence[Sequence[int]],
        *,
        max_tokens: int | Sequence[int] = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seeds: Sequence[int] | None = None,
    ) -> list[GenResult]:
        """Sample one completion per prompt.

        ``max_tokens`` is one cap for every prompt, or one cap per prompt.
        """
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        caps = (
            [int(cap) for cap in max_tokens]
            if isinstance(max_tokens, Sequence)
            else [int(max_tokens)] * len(prompts)
        )
        if len(caps) != len(prompts):
            raise ValueError(f"{len(caps)} max_tokens for {len(prompts)} prompts")
        params = [
            SamplingParams(
                max_tokens=caps[i],
                temperature=temperature,
                top_p=top_p,
                top_k=-1,
                repetition_penalty=1.0,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                logprobs=0,  # logprob of the sampled token only
                seed=None if seeds is None else int(seeds[i]),
            )
            for i in range(len(prompts))
        ]
        outs = self.llm.generate(
            [TokensPrompt(prompt_token_ids=list(p)) for p in prompts],
            params,
            use_tqdm=False,
            lora_request=self.lora_request,
        )
        results: list[GenResult] = []
        for o in outs:
            c = o.outputs[0]
            lps: list[float] = []
            for tok, lp in zip(c.token_ids, c.logprobs or []):
                lps.append(float(lp[tok].logprob) if lp and tok in lp else 0.0)
            results.append(
                GenResult(
                    token_ids=list(c.token_ids),
                    logprobs=lps,
                    finish_reason=c.finish_reason or "",
                    text=c.text,
                )
            )
        ready_at = time.perf_counter()
        for result in results:
            result.ready_at = ready_at
        return results

    # ------------------------------------------------------------------

    def rollout_group(
        self,
        task: Task,
        *,
        schedule: str = "lockstep",
        **kwargs,
    ) -> tuple[list[Trajectory], BatchTimings]:
        """Run ``group_size`` rollouts on one task under the chosen schedule.

        ``schedule="lockstep"`` uses the synchronized schedule described in the module
        docstring and is the default, so every existing caller keeps the exact
        behaviour it had. ``schedule="async"`` gives each rollout its own turn
        loop and grades each trajectory at its own finish.

        The two paths sample identically. A rollout's generation request at
        turn ``t`` is ``(context + turn opener, request_seed(base_seed,
        group_id, rollout_id, t))`` under both, and both paths append the
        returned ids verbatim, so the recorded trajectories are
        token-identical for a given engine. What
        differs is when each request is issued, which fields of
        :class:`BatchTimings` are wall-clock, and when ``on_verdict`` fires.
        """
        impls = {
            "lockstep": self._rollout_group_lockstep,
            "async": self._rollout_group_async,
        }
        if schedule not in impls:
            raise ValueError(
                f"unknown schedule {schedule!r}; expected one of {sorted(impls)}"
            )
        return impls[schedule](task, **kwargs)

    # ------------------------------------------------------------------

    def _rollout_group_lockstep(
        self,
        task: Task,
        *,
        group_size: int,
        group_id: int = 0,
        max_steps: int = 300,
        max_tokens_per_step: int = 1024,
        max_generated_tokens: int = 16384,
        max_observation_tokens: int = 4096,
        max_prompt_tokens: int = 16384,
        temperature: float = 1.0,
        sandbox_prefix: str = "thundersync",
        base_seed: int = 0,
        tool_workers: int = 8,
        repo_context: bool = True,
        score: bool = True,
        verify_broken: bool = True,
        coupled_prefix_steps: int = 0,
        turn_open: str = "<|im_start|>assistant\n",
        context_kwargs: dict | None = None,
        on_prompt=None,
        on_block=None,
        on_step=None,
        on_trajectory_complete=None,
        on_verdict=None,
        sandbox_factory: Callable[[str, str], object] | None = None,
        grade_fn: Callable[[object, Task], object] | None = None,
    ) -> tuple[list[Trajectory], BatchTimings]:
        """Run ``group_size`` rollouts on one task, advancing in lockstep.

        Each trajectory ends at the first of: a submit, ``max_steps`` turns,
        a context that leaves no room for ``max_tokens_per_step`` more tokens
        within ``max_model_len`` (``context_limit``), or ``max_generated_tokens``
        policy-generated tokens (``generation_limit``). A turn's request caps
        its generation at ``min(max_tokens_per_step, max_generated_tokens -
        generated)``; earlier turns are never trimmed. A tool observation keeps
        its first and last ``max_observation_tokens // 2`` tokens
        (:func:`truncate_observation_tokens`). An initial task prompt (system
        and user turns) longer than ``max_prompt_tokens`` raises ``ValueError``
        before any generation. Turn ``t`` of rollout ``i`` samples with
        ``request_seed(base_seed, group_id, i, t)``; a coupled-prefix turn
        samples once, with the seed of ``COUPLED_ROLLOUT_SLOT``.

        ``turn_open`` is the assistant-turn opener; overriding it lets a caller
        force a chat template's non-thinking mode (``...<think>\\n\\n</think>\\n\\n``).
        ``on_prompt(prompt_token_ids)`` fires once, after the shared task prompt
        is tokenized and before any generation. ``on_block(events)`` exposes
        immutable action blocks after generation and observation blocks after
        tool execution, where each event is ``(rollout_id, CommittedBlock)``.
        ``on_step(step_idx, events)``
        fires after every lockstep round with ``events = [(rollout_id, Step)]``
        for each rollout that appended a step this round -- the hook a streaming
        trainer uses to forward turns at emission rather than at batch end.
        ``on_trajectory_complete(rollout_id, trajectory)`` fires after the
        final turn and before scoring. The callbacks run synchronously; their
        cost shows up in generation wall-clock.

        ``on_verdict(rollout_id, trajectory)`` fires once per rollout after
        scoring. On this path all verdicts land together at the group barrier
        and the callback is invoked in rollout_id order; the async path fires it
        at each trajectory's own finish. ``sandbox_factory(image, name)`` and
        ``grade_fn(sandbox, task)`` default to :class:`DockerSandbox` and
        :func:`run_tests`; sandboxes are created through :func:`new_sandbox`,
        which carries ``task.requires_network``.
        """
        _check_prompt_cap(max_prompt_tokens)
        timings = BatchTimings(schedule="lockstep")
        timings.verdict_at = [0.0] * group_size
        timings.trajectory_completed_at = [0.0] * group_size
        _new_box = sandbox_factory or DockerSandbox
        _score_one = grade_fn or run_tests
        boxes = [
            new_sandbox(_new_box, task, f"{sandbox_prefix}-{group_id}-{i}")
            for i in range(group_size)
        ]
        bug_bases: list[str | None] = [None] * group_size
        try:
            t_setup = time.perf_counter()
            for i, box in enumerate(boxes):
                box.start()
                if task.bug_patch:
                    # The image ships the clean repository. The dataset patch
                    # is sealed into history before policy commands can run.
                    box.apply_bug_patch(task.bug_patch)
                    box.assert_bug_not_revertible()
                bug_bases[i] = _capture_bug_base(
                    box, task, trusted_host_snapshot=True
                )
            timings.tool_s += time.perf_counter() - t_setup

            if verify_broken and task.bug_patch:
                # fail loudly if the task is not actually broken: otherwise every
                # rollout scores 1.0 for doing nothing and the solve rate is noise
                broken_reward, _reason = _reward_and_reason(
                    _score_one(boxes[0], task)
                )
                if broken_reward != 0.0:
                    raise RuntimeError(
                        f"{task.instance_id}: FAIL_TO_PASS passes before any edit; "
                        "the bug patch did not take effect"
                    )

            if repo_context:
                # gathered once: identical for every rollout, so it lands in the
                # shared prefix rather than being duplicated G times
                t0 = time.perf_counter()
                # context_kwargs reaches build_repo_context unchanged.
                tree, files = build_repo_context(
                    boxes[0], task.problem_statement, **(context_kwargs or {})
                )
                timings.tool_s += time.perf_counter() - t0
                user_text = CONTEXT_TEMPLATE.format(
                    problem_statement=task.problem_statement, tree=tree, files=files
                )
            else:
                user_text = TASK_TEMPLATE.format(problem_statement=task.problem_statement)

            header = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{user_text}<|im_end|>\n"
            )
            prompt_ids = self.encode(header)
            _check_prompt_length(task, prompt_ids, max_prompt_tokens)
            open_ids = self.encode(turn_open)
            if on_prompt is not None:
                on_prompt(prompt_ids)

            trajs = [
                Trajectory(
                    task_id=task.instance_id,
                    group_id=group_id,
                    rollout_id=i,
                    prompt_token_ids=list(prompt_ids),
                )
                for i in range(group_size)
            ]
            contexts = [list(prompt_ids) for _ in range(group_size)]
            live = list(range(group_size))
            generated = [0] * group_size
            last_content_ready_at: list[float | None] = [None] * group_size
            t_wall = time.perf_counter()

            for step_idx in range(max_steps):
                if not live:
                    break
                budget = self.max_model_len - max_tokens_per_step - len(open_ids)
                live = [i for i in live if len(contexts[i]) < budget]
                for i in set(range(group_size)) - set(live):
                    if not trajs[i].finish_reason:
                        trajs[i].finish_reason = "context_limit"
                live = [i for i in live if generated[i] < max_generated_tokens]
                for i in set(range(group_size)) - set(live):
                    if not trajs[i].finish_reason:
                        trajs[i].finish_reason = "generation_limit"
                if not live:
                    break
                caps = [
                    min(max_tokens_per_step, max_generated_tokens - generated[i])
                    for i in live
                ]

                # --- Coupled prefix: one physical model row for the whole group ---
                # During the coupled prefix every replica shares one sampled action.
                # Contexts are identical here, so a single generation is exactly what
                # all G replicas would have received had they been drawn together --
                # this is the maximal-coupling end of the partition, not an
                # approximation of it. Tool execution still runs per sandbox, which
                # is correct: the compression claim is about model work.
                coupled = step_idx < coupled_prefix_steps
                t0 = time.perf_counter()
                if coupled:
                    one = self.generate(
                        [contexts[live[0]] + open_ids],
                        max_tokens=caps[0],
                        temperature=temperature,
                        seeds=[
                            request_seed(
                                base_seed, group_id, COUPLED_ROLLOUT_SLOT, step_idx
                            )
                        ],
                    )
                    gens = [one[0] for _ in live]
                    timings.model_rows += 1
                else:
                    gens = self.generate(
                        [contexts[i] + open_ids for i in live],
                        max_tokens=caps[0] if len(set(caps)) == 1 else caps,
                        temperature=temperature,
                        seeds=[
                            request_seed(base_seed, group_id, i, step_idx)
                            for i in live
                        ],
                    )
                    timings.model_rows += len(live)
                prompts = [contexts[i] + open_ids for i in live]
                timings.logical_rows += len(live)
                gen_s = time.perf_counter() - t0
                timings.generation_s += gen_s
                timings.generation_batch_sizes.append(1 if coupled else len(live))
                timings.n_steps += 1

                action_ready_times = [
                    result.ready_at for result in gens
                ]
                if any(value is None for value in action_ready_times):
                    action_committed_at = time.perf_counter()
                else:
                    action_committed_at = max(
                        float(value) for value in action_ready_times
                    )
                actions = [parse_action(g.text) for g in gens]
                for i, g in zip(live, gens, strict=True):
                    generated[i] += len(g.token_ids)
                    last_content_ready_at[i] = action_committed_at

                if on_block is not None:
                    on_block(
                        [
                            (
                                i,
                                CommittedBlock(
                                    step_index=step_idx,
                                    kind="action",
                                    token_ids=tuple(open_ids + g.token_ids),
                                    scored=(False,) * len(open_ids)
                                    + (True,) * len(g.token_ids),
                                    committed_at=action_committed_at,
                                    terminal_reason_at_commit=(
                                        "submit"
                                        if action == "submit"
                                        else (
                                            "max_steps"
                                            if step_idx == max_steps - 1
                                            else None
                                        )
                                    ),
                                ),
                            )
                            for i, g, action in zip(live, gens, actions, strict=True)
                        ]
                    )

                # --- tool phase: the GPU is idle for all of this ---
                t1 = time.perf_counter()
                needs_tool = [
                    (k, i, a)
                    for k, (i, a) in enumerate(zip(live, actions, strict=True))
                    if a is not None and a != "submit"
                ]
                tool_positions = {k for k, _i, _action in needs_tool}
                observations: dict[int, str] = {}
                observation_tokens: dict[int, list[int]] = {}
                observation_committed_at: dict[int, float] = {}

                # An invalid action has a deterministic observation with no
                # environment dependency.  Tokenize it before waiting for
                # unrelated tool calls so its source-readiness timestamp stays
                # local to that rollout.
                for k, action in enumerate(actions):
                    if action is None:
                        fallback = PARSE_ERROR_OBSERVATION
                        obs_text = (
                            f"<|im_end|>\n<|im_start|>user\n{fallback}"
                            f"<|im_end|>\n"
                        )
                        observations[k] = fallback
                        observation_tokens[k] = self.encode(obs_text)
                        observation_committed_at[k] = time.perf_counter()
                if needs_tool:
                    with ThreadPoolExecutor(max_workers=tool_workers) as pool:
                        futs = {
                            pool.submit(command_output, boxes[i], a): k
                            for k, i, a in needs_tool
                        }
                        for future in as_completed(futs):
                            k = futs[future]
                            observation = truncate_observation_tokens(
                                future.result(),
                                self.encode,
                                self.decode,
                                max_observation_tokens,
                            )
                            observations[k] = observation
                            obs_text = (
                                f"<|im_end|>\n<|im_start|>user\n{observation}"
                                f"<|im_end|>\n"
                            )
                            observation_tokens[k] = self.encode(obs_text)
                            observation_committed_at[k] = time.perf_counter()
                tool_s = time.perf_counter() - t1
                timings.tool_s += tool_s
                timings.tool_calls += len(needs_tool)

                still_live: list[int] = []
                round_events: list[tuple[int, Step]] = []
                for k, i in enumerate(live):
                    g, action = gens[k], actions[k]
                    step = Step(
                        action=action or g.text[:200],
                        observation=observations.get(k, ""),
                        action_token_ids=open_ids + g.token_ids,
                        action_logprobs=[0.0] * len(open_ids) + g.logprobs,
                        generation_s=gen_s / max(len(live), 1),
                        tool_s=(
                            tool_s / max(len(needs_tool), 1)
                            if k in tool_positions
                            else 0.0
                        ),
                        prompt_len=len(prompts[k]),
                    )

                    if action == "submit":
                        trajs[i].steps.append(step)
                        trajs[i].finish_reason = "submit"
                        round_events.append((i, step))
                        continue

                    step.observation_token_ids = observation_tokens[k]
                    last_content_ready_at[i] = observation_committed_at[k]
                    if on_block is not None:
                        on_block(
                            [
                                (
                                    i,
                                    CommittedBlock(
                                        step_index=step_idx,
                                        kind="observation",
                                        token_ids=tuple(step.observation_token_ids),
                                        scored=(False,)
                                        * len(step.observation_token_ids),
                                        committed_at=observation_committed_at[k],
                                    ),
                                )
                            ]
                        )
                    trajs[i].steps.append(step)
                    contexts[i] = (
                        contexts[i] + step.action_token_ids + step.observation_token_ids
                    )
                    round_events.append((i, step))
                    still_live.append(i)
                live = still_live
                if on_step is not None and round_events:
                    on_step(step_idx, round_events)

            for t in trajs:
                if not t.finish_reason:
                    t.finish_reason = "max_steps"

            for i, trajectory in enumerate(trajs):
                trajectory.completed_at = (
                    last_content_ready_at[i]
                    if last_content_ready_at[i] is not None
                    else time.perf_counter()
                )
            timings.trajectory_completed_at = [
                float(trajectory.completed_at) for trajectory in trajs
            ]
            if on_trajectory_complete is not None:
                for i in range(group_size):
                    on_trajectory_complete(i, trajs[i])

            if score:
                t0 = time.perf_counter()

                def _score(i: int) -> tuple[float, str, str]:
                    # capture the source diff *before* grading, since grading
                    # restores test files and would erase the evidence
                    diff = _source_diff(boxes[i], bug_bases[i])
                    reward, reason = _reward_and_reason(
                        _score_one(boxes[i], task)
                    )
                    return reward, diff.strip(), reason

                with ThreadPoolExecutor(max_workers=tool_workers) as pool:
                    futs = {pool.submit(_score, i): i for i in range(group_size)}
                    for f, i in futs.items():
                        (
                            trajs[i].reward,
                            trajs[i].final_diff,
                            trajs[i].verdict_reason,
                        ) = f.result()
                timings.verify_s += time.perf_counter() - t0

                # a reward with no source change means the tests passed without
                # the model fixing anything: the reward channel is compromised
                free_riders = [
                    t.rollout_id for t in trajs if t.reward > 0 and not t.final_diff
                ]
                if free_riders:
                    raise RuntimeError(
                        f"{task.instance_id}: rollouts {free_riders} scored 1.0 with an "
                        "empty source diff -- reward obtainable without fixing anything"
                    )
                timings.graded_at = time.perf_counter()
                # every verdict on this path lands at the same barrier, so every
                # entry carries the same timestamp; the async path spreads them
                timings.verdict_at = [timings.graded_at] * group_size
                if on_verdict is not None:
                    for i in range(group_size):
                        on_verdict(i, trajs[i])
            timings.rollout_wall_s = time.perf_counter() - t_wall
        finally:
            for box in boxes:
                box.close()

        return trajs, timings

    # ------------------------------------------------------------------

    def _rollout_group_async(
        self,
        task: Task,
        *,
        group_size: int,
        group_id: int = 0,
        max_steps: int = 300,
        max_tokens_per_step: int = 1024,
        max_generated_tokens: int = 16384,
        max_observation_tokens: int = 4096,
        max_prompt_tokens: int = 16384,
        temperature: float = 1.0,
        sandbox_prefix: str = "thundersync",
        base_seed: int = 0,
        tool_workers: int = 8,
        repo_context: bool = True,
        score: bool = True,
        verify_broken: bool = True,
        coupled_prefix_steps: int = 0,
        turn_open: str = "<|im_start|>assistant\n",
        context_kwargs: dict | None = None,
        on_prompt=None,
        on_block=None,
        on_step=None,
        on_trajectory_complete=None,
        on_verdict=None,
        sandbox_factory: Callable[[str, str], object] | None = None,
        grade_fn: Callable[[object, Task], object] | None = None,
        broker: _GenerationBroker | None = None,
        callback_lock: threading.Lock | None = None,
        tool_semaphore: threading.Semaphore | None = None,
        verifier_semaphore: threading.Semaphore | None = None,
        gen_linger_s: float = 0.002,
    ) -> tuple[list[Trajectory], BatchTimings]:
        """Run ``group_size`` rollouts on one task, each advancing on its own.

        Turn limits, the generated-token cap, the initial prompt cap,
        observation truncation and request seeds are those of
        :meth:`_rollout_group_lockstep`.

        One worker thread per rollout owns that rollout's turn loop, so a rollout
        issues its next generation request as soon as its own sandbox returns.
        Generation is still batched, through :class:`_GenerationBroker`; pass an
        external ``broker`` (as :meth:`rollout_groups` does) to batch across
        groups as well.

        ``on_trajectory_complete`` runs after the final emitted turn and before
        scoring. Scoring runs inside the worker at that rollout's finish, so
        ``on_verdict(rollout_id, trajectory)`` fires while slower siblings are
        still taking turns, so a streaming trainer's per-trajectory backward can
        start at the trajectory's own verdict instead of the group's.

        Callback contract, held identical to lockstep so the two are comparable:
        ``on_prompt(prompt_token_ids)`` fires once before any generation,
        ``on_block([(rollout_id, CommittedBlock)])`` fires at action and
        observation readiness, and
        ``on_step(step_idx, [(rollout_id, Step)])`` fires with the same argument
        shape. The event list carries exactly one entry here, because a step
        completing is a per-rollout event once the round barrier is gone. All
        callbacks are serialized on one lock, so a trainer sees them one at a
        time and in each trajectory's own step order; what is not preserved is
        the interleaving BETWEEN trajectories.

        Reported timings differ in kind, not only in value. ``generation_s``,
        ``tool_s`` and ``verify_s`` become sums over rollouts of overlapping
        intervals; ``rollout_wall_s`` is the makespan and is the only field on
        this path that may be compared against a lockstep wall-clock.
        """
        if coupled_prefix_steps:
            # The coupled prefix hands one sampled action to every replica,
            # which requires the replicas to be at the same turn at the same
            # time. Honouring it here would reinstate the round barrier for the
            # coupled steps; approximating it would change what is sampled.
            raise NotImplementedError(
                "coupled_prefix_steps requires the group to advance together; "
                'use schedule="lockstep" for coupled prefixes'
            )

        _check_prompt_cap(max_prompt_tokens)
        timings = BatchTimings(schedule="async")
        timings.verdict_at = [0.0] * group_size
        timings.trajectory_completed_at = [0.0] * group_size
        _new_box = sandbox_factory or DockerSandbox
        _score_one = grade_fn or run_tests
        boxes = [
            new_sandbox(_new_box, task, f"{sandbox_prefix}-{group_id}-{i}")
            for i in range(group_size)
        ]
        bug_bases: list[str | None] = [None] * group_size
        try:
            # ---- group prologue. Shared by construction: the bug patch, the
            # brokenness check and the repo context all produce state every
            # rollout starts from, so they precede the fan-out on both paths.
            # Sandbox setup itself is per-rollout and independent, so it runs
            # concurrently here where lockstep runs it one container at a time.
            t_setup = time.perf_counter()

            def _setup(item) -> None:
                rollout_id, box = item
                def execute() -> None:
                    box.start()
                    if task.bug_patch:
                        box.apply_bug_patch(task.bug_patch)
                        box.assert_bug_not_revertible()
                    bug_bases[rollout_id] = _capture_bug_base(
                        box, task, trusted_host_snapshot=True
                    )

                _limited_call(tool_semaphore, execute)

            with ThreadPoolExecutor(max_workers=max(tool_workers, 1)) as pool:
                list(pool.map(_setup, enumerate(boxes)))
            timings.tool_s += time.perf_counter() - t_setup

            if verify_broken and task.bug_patch:
                broken_reward, _reason = _reward_and_reason(
                    _limited_call(verifier_semaphore, _score_one, boxes[0], task)
                )
                if broken_reward != 0.0:
                    raise RuntimeError(
                        f"{task.instance_id}: FAIL_TO_PASS passes before any edit; "
                        "the bug patch did not take effect"
                    )

            if repo_context:
                t0 = time.perf_counter()
                # context_kwargs reaches build_repo_context unchanged.
                tree, files = _limited_call(
                    tool_semaphore,
                    build_repo_context,
                    boxes[0],
                    task.problem_statement,
                    **(context_kwargs or {}),
                )
                timings.tool_s += time.perf_counter() - t0
                user_text = CONTEXT_TEMPLATE.format(
                    problem_statement=task.problem_statement, tree=tree, files=files
                )
            else:
                user_text = TASK_TEMPLATE.format(problem_statement=task.problem_statement)

            header = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{user_text}<|im_end|>\n"
            )
            prompt_ids = self.encode(header)
            _check_prompt_length(task, prompt_ids, max_prompt_tokens)
            open_ids = self.encode(turn_open)
            cb_lock = callback_lock or threading.Lock()
            if on_prompt is not None:
                with cb_lock:
                    on_prompt(prompt_ids)

            trajs = [
                Trajectory(
                    task_id=task.instance_id,
                    group_id=group_id,
                    rollout_id=i,
                    prompt_token_ids=list(prompt_ids),
                )
                for i in range(group_size)
            ]

            # Trainer callbacks are serialized: on_step/on_verdict reach into a
            # trainer that owns CUDA state. ``rollout_groups`` supplies one lock
            # for the complete generation batch; a direct ``rollout_group`` call
            # receives a group-local lock.
            tim_lock = threading.Lock()
            own_broker = broker is None
            brk = broker or _GenerationBroker(self, linger_s=gen_linger_s).start()

            def _emit(step_idx: int, rollout_id: int, step: Step) -> None:
                if on_step is None:
                    return
                with cb_lock:
                    on_step(step_idx, [(rollout_id, step)])

            def _emit_block(rollout_id: int, block: CommittedBlock) -> None:
                if on_block is None:
                    return
                with cb_lock:
                    on_block([(rollout_id, block)])

            def _work(i: int) -> None:
                traj = trajs[i]
                context = list(prompt_ids)
                generated = 0
                gen_total = tool_total = 0.0
                n_turns = n_tools = 0
                last_content_ready_at: float | None = None

                for step_idx in range(max_steps):
                    budget = self.max_model_len - max_tokens_per_step - len(open_ids)
                    if len(context) >= budget:
                        traj.finish_reason = "context_limit"
                        break
                    if generated >= max_generated_tokens:
                        traj.finish_reason = "generation_limit"
                        break

                    prompt = context + open_ids
                    t0 = time.perf_counter()
                    g = brk.submit(
                        prompt,
                        seed=request_seed(base_seed, group_id, i, step_idx),
                        max_tokens=min(
                            max_tokens_per_step, max_generated_tokens - generated
                        ),
                        temperature=temperature,
                    )
                    gen_s = time.perf_counter() - t0
                    if g.ready_at is None:
                        raise RuntimeError(
                            "generation broker omitted the source-readiness "
                            "timestamp"
                        )
                    action_committed_at = g.ready_at
                    last_content_ready_at = action_committed_at
                    gen_total += gen_s
                    generated += len(g.token_ids)
                    n_turns += 1

                    action = parse_action(g.text)
                    _emit_block(
                        i,
                        CommittedBlock(
                            step_index=step_idx,
                            kind="action",
                            token_ids=tuple(open_ids + g.token_ids),
                            scored=(False,) * len(open_ids)
                            + (True,) * len(g.token_ids),
                            committed_at=action_committed_at,
                            terminal_reason_at_commit=(
                                "submit"
                                if action == "submit"
                                else (
                                    "max_steps"
                                    if step_idx == max_steps - 1
                                    else None
                                )
                            ),
                        ),
                    )

                    # None means "no tool ran", which is distinct from a tool
                    # that ran and produced empty output; lockstep encodes the
                    # same distinction as key-present-in-`observations`.
                    observation: str | None = (
                        PARSE_ERROR_OBSERVATION if action is None else None
                    )
                    tool_s = 0.0
                    if action is not None and action != "submit":
                        t1 = time.perf_counter()
                        observation = truncate_observation_tokens(
                            _limited_call(
                                tool_semaphore, command_output, boxes[i], action
                            ),
                            self.encode,
                            self.decode,
                            max_observation_tokens,
                        )
                        tool_s = time.perf_counter() - t1
                        tool_total += tool_s
                        n_tools += 1

                    step = Step(
                        action=action or g.text[:200],
                        observation=observation or "",
                        action_token_ids=open_ids + g.token_ids,
                        action_logprobs=[0.0] * len(open_ids) + g.logprobs,
                        generation_s=gen_s,
                        tool_s=tool_s,
                        prompt_len=len(prompt),
                    )

                    if action == "submit":
                        traj.steps.append(step)
                        traj.finish_reason = "submit"
                        _emit(step_idx, i, step)
                        break

                    obs_body = (
                        observation if observation is not None else PARSE_ERROR_OBSERVATION
                    )
                    obs_text = (
                        f"<|im_end|>\n<|im_start|>user\n{obs_body}<|im_end|>\n"
                    )
                    step.observation_token_ids = self.encode(obs_text)
                    observation_committed_at = time.perf_counter()
                    last_content_ready_at = observation_committed_at
                    _emit_block(
                        i,
                        CommittedBlock(
                            step_index=step_idx,
                            kind="observation",
                            token_ids=tuple(step.observation_token_ids),
                            scored=(False,) * len(step.observation_token_ids),
                            committed_at=observation_committed_at,
                        ),
                    )
                    traj.steps.append(step)
                    context = context + step.action_token_ids + step.observation_token_ids
                    _emit(step_idx, i, step)

                if not traj.finish_reason:
                    traj.finish_reason = "max_steps"

                completed_at = (
                    last_content_ready_at
                    if last_content_ready_at is not None
                    else time.perf_counter()
                )
                traj.completed_at = completed_at
                with tim_lock:
                    timings.trajectory_completed_at[i] = completed_at
                if on_trajectory_complete is not None:
                    with cb_lock:
                        on_trajectory_complete(i, traj)

                with tim_lock:
                    timings.generation_s += gen_total
                    timings.tool_s += tool_total
                    timings.tool_calls += n_tools
                    timings.model_rows += n_turns
                    timings.logical_rows += n_turns
                    timings.n_steps = max(timings.n_steps, n_turns)

                if not score:
                    return

                # ---- this trajectory's verdict, at this trajectory's finish
                t0 = time.perf_counter()
                diff = _limited_call(
                    tool_semaphore, _source_diff, boxes[i], bug_bases[i]
                )
                reward, reason = _reward_and_reason(
                    _limited_call(verifier_semaphore, _score_one, boxes[i], task)
                )
                verify_s = time.perf_counter() - t0
                traj.reward = reward
                traj.final_diff = diff.strip()
                traj.verdict_reason = reason
                at = time.perf_counter()

                # checked here rather than at a group barrier so a compromised
                # reward can never reach the trainer through on_verdict
                if traj.reward > 0 and not traj.final_diff:
                    raise RuntimeError(
                        f"{task.instance_id}: rollouts [{i}] scored 1.0 with an "
                        "empty source diff -- reward obtainable without fixing anything"
                    )

                with tim_lock:
                    timings.verify_s += verify_s
                    timings.verdict_at[i] = at
                    timings.graded_at = max(timings.graded_at, at)

                if on_verdict is not None:
                    with cb_lock:
                        on_verdict(i, traj)

            t_wall = time.perf_counter()
            try:
                with ThreadPoolExecutor(max_workers=max(group_size, 1)) as pool:
                    futures = [pool.submit(_work, i) for i in range(group_size)]
                    errors = []
                    for f in futures:
                        try:
                            f.result()
                        except BaseException as exc:  # noqa: BLE001
                            errors.append(exc)
                if errors:
                    raise errors[0]
            finally:
                timings.rollout_wall_s = time.perf_counter() - t_wall
                if own_broker:
                    brk.close()
        finally:
            with ThreadPoolExecutor(max_workers=max(tool_workers, 1)) as pool:
                list(
                    pool.map(
                        lambda b: _limited_call(tool_semaphore, b.close), boxes
                    )
                )

        return trajs, timings

    # ------------------------------------------------------------------

    def rollout_groups(
        self,
        specs: Sequence[dict],
        *,
        schedule: str = "async",
        max_concurrent_groups: int | None = None,
        max_concurrent_tools: int | None = None,
        max_concurrent_verifiers: int | None = None,
        gen_linger_s: float = 0.002,
        **common,
    ) -> list[tuple[list[Trajectory], BatchTimings]]:
        """Run several groups at once. Results are returned in ``specs`` order.

        Each entry of ``specs`` is the per-group keyword dict for
        :meth:`rollout_group` -- at minimum ``task`` and ``group_id``, plus that
        group's own ``on_prompt`` / ``on_block`` / ``on_step`` /
        ``on_trajectory_complete`` /
        ``on_verdict`` closures, which
        the trainer binds per group. ``common`` supplies the keywords shared by
        every group.

        ``schedule="async"`` runs every group at once against ONE shared broker,
        so rollouts of different tasks coalesce into the same ``generate`` call.
        ``schedule="lockstep"`` runs the groups one after another with lockstep
        rounds, sequential groups, and grading at a group barrier. Keeping it
        sequential also
        keeps ``generate`` called from a single thread, which is what the vLLM
        offline engine supports; on the async path the broker's dispatcher is
        the only caller, so that holds there too.
        """
        if schedule not in ("async", "lockstep"):
            raise ValueError(f"unknown schedule {schedule!r}")
        if max_concurrent_tools is not None and max_concurrent_tools <= 0:
            raise ValueError("max_concurrent_tools must be positive")
        if max_concurrent_verifiers is not None and max_concurrent_verifiers <= 0:
            raise ValueError("max_concurrent_verifiers must be positive")

        n = len(specs)
        if not n:
            return []
        results: list[tuple[list[Trajectory], BatchTimings] | None] = [None] * n

        def _kw(idx: int) -> tuple[Task, dict]:
            kw = dict(common)
            kw.update(specs[idx])
            return kw.pop("task"), kw

        if schedule == "lockstep":
            for i in range(n):
                task, kw = _kw(i)
                results[i] = self.rollout_group(task, schedule="lockstep", **kw)
            return [r for r in results if r is not None]

        brk = _GenerationBroker(self, linger_s=gen_linger_s).start()
        shared_callback_lock = threading.Lock()
        shared_tool_semaphore = (
            threading.BoundedSemaphore(max_concurrent_tools)
            if max_concurrent_tools is not None
            else None
        )
        shared_verifier_semaphore = (
            threading.BoundedSemaphore(max_concurrent_verifiers)
            if max_concurrent_verifiers is not None
            else None
        )
        try:
            def _one(idx: int):
                task, kw = _kw(idx)
                kw["broker"] = brk
                kw["callback_lock"] = shared_callback_lock
                kw["tool_semaphore"] = shared_tool_semaphore
                kw["verifier_semaphore"] = shared_verifier_semaphore
                return self.rollout_group(task, schedule="async", **kw)

            workers = max_concurrent_groups or n
            with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
                futures = {pool.submit(_one, i): i for i in range(n)}
                errors = []
                for f, i in futures.items():
                    try:
                        results[i] = f.result()
                    except BaseException as exc:  # noqa: BLE001
                        errors.append(exc)
            if errors:
                raise errors[0]
        finally:
            brk.close()

        for result in results:
            if result is None:
                continue
            timings = result[1]
            timings.shared_generation_s = brk.generation_s
            timings.shared_generation_batch_sizes = list(brk.batch_sizes)
            timings.shared_generation_prompt_tokens = brk.prompt_tokens
            timings.shared_generation_completion_tokens = brk.completion_tokens

        return [r for r in results if r is not None]
