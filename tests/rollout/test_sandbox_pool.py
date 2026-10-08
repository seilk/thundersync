"""Sandbox pool gates: reuse only behind the trusted pristine check.

Failure modes each test guards:

* a reused container carrying tracked edits, untracked leftovers, or a
  previous use's cached policy patch into the next verification;
* a container whose in-container Git was mangled slipping back into the
  free list because the reset "succeeded";
* the trusted gate depending on the trusted store's INDEX, which
  policy-patch capture legitimately mutates during a trajectory;
* pooled grade_isolated paying (or double-paying) container setup, or
  closing a pool-owned container.

The fake sandbox executes the pool's reset command against a real Git
worktree on disk, so reset and gate behavior are exercised for real.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


from thundersync.rollout import engine as eng  # noqa: E402
from thundersync.rollout.engine import Task  # noqa: E402
from thundersync.rollout.sandbox_pool import (  # noqa: E402
    SandboxPool,
    SandboxPoolError,
    trusted_tree_pristine,
)


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "HOME": str(cwd),
        },
    )


def make_worktree(root: Path) -> Path:
    workdir = root / "testbed"
    workdir.mkdir(parents=True)
    (workdir / "solver.py").write_text("def solve():\n    return 41\n")
    (workdir / "test_solver.py").write_text(
        "from solver import solve\n\ndef test_solve():\n    assert solve() == 42\n"
    )
    _git("init", "-q", cwd=workdir)
    _git("add", "-A", cwd=workdir)
    _git("commit", "-qm", "sealed bug state", cwd=workdir)
    return workdir


class FakeSandbox:
    """Host-backed sandbox whose container IS its host worktree."""

    def __init__(self, workdir: Path, name: str, *, reset_behavior: str = "real"):
        self.host_workdir = str(workdir)
        self.name = name
        self.reset_behavior = reset_behavior
        self.started = False
        self.closed = False
        self.exec_commands: list[str] = []

    def start(self) -> None:
        self.started = True

    def apply_bug_patch(self, patch: str) -> None:
        raise AssertionError("tests use tasks without a bug patch")

    def assert_bug_not_revertible(self) -> None:
        raise AssertionError("tests use tasks without a bug patch")

    def exec(self, command: str) -> str:
        self.exec_commands.append(command)
        if self.reset_behavior == "raise":
            raise RuntimeError("container shell is gone")
        if self.reset_behavior == "noop":
            return "THUNDERSYNC_POOL_RESET_OK"
        result = subprocess.run(
            ["bash", "-c", command],
            cwd=self.host_workdir,
            capture_output=True,
            text=True,
        )
        return result.stdout

    def close(self) -> None:
        self.closed = True


def make_pool(root: Path, *, reset_behavior: str = "real", max_uses: int = 0):
    created: list[FakeSandbox] = []

    def factory(image: str, name: str) -> FakeSandbox:
        workdir = make_worktree(root / f"c{len(created)}")
        sandbox = FakeSandbox(workdir, name, reset_behavior=reset_behavior)
        created.append(sandbox)
        return sandbox

    pool = SandboxPool(factory, name_prefix="test-pool", max_uses=max_uses)
    return pool, created


TASK = Task(
    instance_id="demo__task.0001",
    problem_statement="make solve return 42",
    image_name="demo-image",
)


def test_acquire_creates_once_and_release_reuses(tmp_path: Path) -> None:
    pool, created = make_pool(tmp_path)
    first = pool.acquire(TASK)
    assert first.started
    pool.release(TASK, first)
    second = pool.acquire(TASK)
    assert second is first
    assert len(created) == 1
    report = pool.report()
    assert report["created"] == 1
    assert report["reused"] == 1
    assert report["discarded"] == 0


def test_release_restores_tracked_edits_and_removes_untracked(tmp_path: Path) -> None:
    pool, _created = make_pool(tmp_path)
    sandbox = pool.acquire(TASK)
    workdir = Path(sandbox.host_workdir)
    (workdir / "solver.py").write_text("def solve():\n    return 42\n")
    (workdir / "scratch.txt").write_text("leftover")
    pool.release(TASK, sandbox)
    assert pool.acquire(TASK) is sandbox
    assert "return 41" in (workdir / "solver.py").read_text()
    assert not (workdir / "scratch.txt").exists()


def test_mangled_container_shell_discards(tmp_path: Path) -> None:
    pool, created = make_pool(tmp_path, reset_behavior="raise")
    sandbox = pool.acquire(TASK)
    pool.release(TASK, sandbox)
    assert sandbox.closed
    replacement = pool.acquire(TASK)
    assert replacement is not sandbox
    assert len(created) == 2
    assert pool.report()["pristine_failures"] == 1


def test_noop_reset_with_drift_fails_the_trusted_gate(tmp_path: Path) -> None:
    pool, _created = make_pool(tmp_path, reset_behavior="noop")
    sandbox = pool.acquire(TASK)
    Path(sandbox.host_workdir, "solver.py").write_text("def solve():\n    return 0\n")
    pool.release(TASK, sandbox)
    assert sandbox.closed
    assert pool.report()["pristine_failures"] == 1


def test_gate_survives_policy_patch_index_mutation(tmp_path: Path) -> None:
    pool, _created = make_pool(tmp_path)
    sandbox = pool.acquire(TASK)
    workdir = Path(sandbox.host_workdir)
    (workdir / "solver.py").write_text("def solve():\n    return 42\n")
    # A trajectory captures its policy patch through the trusted store,
    # legitimately re-staging the index against the dirty worktree.
    patch = eng._capture_policy_patch(sandbox, max_bytes=1_000_000)
    assert patch
    pool.release(TASK, sandbox)
    # The gate compares against the immutable tree, not the mutated index:
    # after the reset the member must be reusable.
    assert pool.acquire(TASK) is sandbox


def test_release_clears_stale_policy_patch_state(tmp_path: Path) -> None:
    pool, _created = make_pool(tmp_path)
    sandbox = pool.acquire(TASK)
    sandbox._thundersync_policy_patch_cache = b"stale"
    sandbox._thundersync_policy_patch_rejection = "stale"
    sandbox._thundersync_policy_patch_capture_error = "stale"
    pool.release(TASK, sandbox)
    assert not hasattr(sandbox, "_thundersync_policy_patch_cache")
    assert not hasattr(sandbox, "_thundersync_policy_patch_rejection")
    assert not hasattr(sandbox, "_thundersync_policy_patch_capture_error")


def test_max_uses_recycles_the_member(tmp_path: Path) -> None:
    pool, created = make_pool(tmp_path, max_uses=2)
    first = pool.acquire(TASK)
    pool.release(TASK, first)
    second = pool.acquire(TASK)
    assert second is first
    pool.release(TASK, second)
    assert first.closed
    third = pool.acquire(TASK)
    assert third is not first
    assert len(created) == 2


def test_concurrent_acquires_get_distinct_members(tmp_path: Path) -> None:
    pool, created = make_pool(tmp_path)
    first = pool.acquire(TASK)
    second = pool.acquire(TASK)
    assert first is not second
    assert len(created) == 2
    pool.release(TASK, first)
    pool.release(TASK, second)
    assert pool.report()["idle"] == 2


def test_close_closes_idle_and_rejects_acquire(tmp_path: Path) -> None:
    pool, _created = make_pool(tmp_path)
    sandbox = pool.acquire(TASK)
    pool.release(TASK, sandbox)
    pool.close()
    assert sandbox.closed
    with pytest.raises(SandboxPoolError):
        pool.acquire(TASK)


def test_retire_closes_the_pool_and_removes_nothing(tmp_path: Path) -> None:
    """The shutdown form: the containers stay for the next launch's prune."""
    pool, _created = make_pool(tmp_path)
    sandbox = pool.acquire(TASK)
    pool.release(TASK, sandbox)
    assert pool.report()["idle"] == 1
    pool.retire()
    assert not sandbox.closed
    assert pool.report()["idle"] == 0
    with pytest.raises(SandboxPoolError):
        pool.acquire(TASK)


def test_gate_fails_closed_without_trusted_attributes(tmp_path: Path) -> None:
    workdir = make_worktree(tmp_path)
    sandbox = FakeSandbox(workdir, "bare")
    assert trusted_tree_pristine(sandbox) is False


def test_pooled_grade_isolated_skips_setup_and_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    class StubPool:
        def __init__(self, verifier: FakeSandbox) -> None:
            self.verifier = verifier

        def acquire(self, task: Task) -> FakeSandbox:
            calls.append("acquire")
            return self.verifier

        def release(self, task: Task, sandbox: FakeSandbox) -> None:
            calls.append("release")

    policy = FakeSandbox(make_worktree(tmp_path / "policy"), "policy")
    policy._thundersync_bug_base = "0" * 40
    policy._thundersync_policy_patch_cache = b""
    verifier = FakeSandbox(make_worktree(tmp_path / "verifier"), "verifier")
    monkeypatch.setattr(
        eng, "grade", lambda sandbox, task, **kwargs: eng.Verdict(1.0, reason="ok")
    )
    verdict = eng.grade_isolated(
        policy,
        TASK,
        sandbox_factory=lambda image, name: pytest.fail(
            "pooled grading must not create a fresh verifier"
        ),
        verifier_name="unused",
        verifier_pool=StubPool(verifier),
    )
    assert verdict.reward == 1.0
    assert calls == ["acquire", "release"]
    assert verifier.started is False
    assert verifier.closed is False


def make_capped_pool(root: Path, *, max_members: int):
    created: list[FakeSandbox] = []

    def factory(image: str, name: str) -> FakeSandbox:
        workdir = make_worktree(root / f"c{len(created)}")
        sandbox = FakeSandbox(workdir, name, reset_behavior="real")
        created.append(sandbox)
        return sandbox

    pool = SandboxPool(
        factory, name_prefix="capped-pool", max_members=max_members
    )
    return pool, created


def _task(number: int) -> Task:
    return Task(
        instance_id=f"demo__task.{number:04d}",
        problem_statement="make solve return 42",
        image_name="demo-image",
    )


def test_member_cap_evicts_an_idle_member_of_another_task(tmp_path: Path) -> None:
    # The pool footprint (gigabytes per container) must stay bounded however
    # many distinct tasks flow through; without a cap, task-keyed idle lists
    # grow with task spread until the store fills the disk.
    pool, created = make_capped_pool(tmp_path, max_members=2)
    first = pool.acquire(_task(1))
    second = pool.acquire(_task(2))
    pool.release(_task(1), first)
    pool.release(_task(2), second)
    third = pool.acquire(_task(3))
    assert third.started
    assert len(created) == 3
    report = pool.report()
    assert report["members"] == 2
    assert report["evicted"] == 1
    assert report["discarded"] == 1


def test_member_cap_blocks_until_a_busy_member_frees(tmp_path: Path) -> None:
    import threading as _threading

    pool, created = make_capped_pool(tmp_path, max_members=1)
    first = pool.acquire(_task(1))
    acquired: list = []

    def second_acquire() -> None:
        acquired.append(pool.acquire(_task(1)))

    waiter = _threading.Thread(target=second_acquire)
    waiter.start()
    waiter.join(timeout=0.5)
    assert waiter.is_alive(), "acquire must block at the cap"
    pool.release(_task(1), first)
    waiter.join(timeout=5)
    assert not waiter.is_alive()
    assert acquired and acquired[0] is first
    assert pool.report()["members"] == 1


def test_setup_failure_frees_the_reserved_member_slot(tmp_path: Path) -> None:
    created: list[FakeSandbox] = []

    def factory(image: str, name: str) -> FakeSandbox:
        workdir = make_worktree(tmp_path / f"c{len(created)}")
        sandbox = FakeSandbox(workdir, name, reset_behavior="real")
        created.append(sandbox)
        return sandbox

    def failing_setup(sandbox, task) -> None:
        raise RuntimeError("setup exploded")

    pool = SandboxPool(
        factory, name_prefix="capped-pool", max_members=1, setup=failing_setup
    )
    with pytest.raises(RuntimeError, match="setup exploded"):
        pool.acquire(_task(1))
    assert pool.report()["members"] == 0


def test_report_accounts_creation_phases_waits_and_discards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_start = FakeSandbox.start

    def start_with_timings(self: FakeSandbox) -> None:
        original_start(self)
        self.start_timings = {"copy_s": 1.5, "probe_s": 0.25}

    monkeypatch.setattr(FakeSandbox, "start", start_with_timings)
    pool, _created = make_pool(tmp_path)
    first = pool.acquire(TASK)
    second = pool.acquire(TASK)
    pool.release(TASK, first)
    pool.release(TASK, second)
    pool.close()
    report = pool.report()
    assert report["created"] == 2
    assert report["create_s"] >= report["create_max_s"] > 0
    assert report["discard_s"] >= 0
    assert report["wait_s"] == 0
    assert report["start_phase_s"] == {"copy_s": 3.0, "probe_s": 0.5}
