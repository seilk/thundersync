from __future__ import annotations

import inspect
import os
import stat
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest


from thundersync.rollout import udocker_sandbox  # noqa: E402
from thundersync.rollout.docker_sandbox import DockerSandbox, bounded_name  # noqa: E402
from thundersync.rollout.udocker_sandbox import (  # noqa: E402
    STAGING_DIRNAME,
    UDockerSandbox,
    remove_tree,
)


def test_exposes_the_docker_sandbox_interface(tmp_path):
    assert issubclass(UDockerSandbox, DockerSandbox)
    for name in (
        "start",
        "exec",
        "close",
        "apply_bug_patch",
        "assert_bug_not_revertible",
    ):
        assert callable(getattr(UDockerSandbox, name))
    assert inspect.signature(UDockerSandbox.exec) == inspect.signature(
        DockerSandbox.exec
    )
    assert UDockerSandbox.apply_bug_patch is DockerSandbox.apply_bug_patch
    assert (
        UDockerSandbox.assert_bug_not_revertible
        is DockerSandbox.assert_bug_not_revertible
    )


def test_seed_name_is_image_bound_and_sandbox_name_is_unique(tmp_path):
    left = UDockerSandbox("repo/image:tag", "same", udocker_dir=tmp_path)
    right = UDockerSandbox("repo/image:tag", "same", udocker_dir=tmp_path)
    other = UDockerSandbox("repo/other:tag", "same", udocker_dir=tmp_path)
    assert left.seed_name == right.seed_name
    assert left.seed_name != other.seed_name
    assert left.container_name != right.container_name


def test_bounded_name_contains_no_runtime_metacharacters():
    value = bounded_name("thundersync", "repo/image:tag with spaces", "identity")
    assert "/" not in value
    assert ":" not in value
    assert "." not in value
    assert " " not in value


def test_exec_uses_private_mounts_and_network_guard(monkeypatch, tmp_path):
    box = UDockerSandbox(
        "repo/image",
        "unit",
        udocker_dir=tmp_path / "store",
        scratch_dir=tmp_path / "scratch",
    )
    box._started = True
    calls = []

    def fake_run(args, *, timeout, network_guard=False, **_limits):
        calls.append((args, timeout, network_guard))
        return subprocess.CompletedProcess(args, 0, "ok", "")

    monkeypatch.setattr(box, "_run", fake_run)
    assert box.exec("echo ok") == "ok"
    args, _timeout, guarded = calls[0]
    assert guarded
    assert "--nosysdirs" in args
    assert f"--volume={box.host_workdir}:/testbed" in args
    assert f"--volume={box.host_tmpdir}:/tmp" in args
    assert f"--volume={box.host_runtime_dir / 'hosts'}:/etc/hosts" in args
    assert (
        f"--volume={box.host_runtime_dir / 'resolv.conf'}:/etc/resolv.conf" in args
    )
    for device in ("null", "zero", "random", "urandom"):
        assert f"--volume=/dev/{device}:/dev/{device}" in args
    assert args[-3] == "/bin/bash"
    assert any("HOME=/root" in arg for arg in args)
    shell_command = args[-1]
    assert "GOROOT=/usr/local/go" in shell_command
    assert "GOTELEMETRY=off" in shell_command


@pytest.mark.parametrize("requires_network", [False, True])
def test_network_profile_uses_private_name_service_and_filters_unless_required(
    monkeypatch, tmp_path, requires_network
):
    box = UDockerSandbox(
        "repo/image",
        "loopback",
        udocker_dir=tmp_path / "store",
        scratch_dir=tmp_path / "scratch",
        requires_network=requires_network,
    )
    box._started = True
    calls = []

    def fake_run(args, *, timeout, network_guard=False, **_limits):
        calls.append((args, timeout, network_guard))
        return subprocess.CompletedProcess(args, 0, "ok", "")

    monkeypatch.setattr(box, "_run", fake_run)
    assert box.exec("go version") == "ok"
    args, _timeout, guarded = calls[0]
    assert guarded is not requires_network
    assert f"--volume={box.host_runtime_dir / 'hosts'}:/etc/hosts" in args
    assert (
        f"--volume={box.host_runtime_dir / 'resolv.conf'}:/etc/resolv.conf" in args
    )


def test_missing_image_fails_closed(monkeypatch, tmp_path):
    box = UDockerSandbox(
        "missing/image",
        "unit",
        udocker_dir=tmp_path / "store",
        containers_dir=tmp_path / "containers",
    )

    monkeypatch.setattr(box, "_inspect_root", lambda _name: None)

    def fake_run(args, *, timeout, network_guard=False, **_limits):
        assert args[0] == "create"
        return subprocess.CompletedProcess(args, 1, "", "image missing")

    monkeypatch.setattr(box, "_run", fake_run)
    with pytest.raises(RuntimeError, match="unavailable"):
        box._ensure_seed()


def _seed_container(tmp_path, *, execmode=None):
    """A seed container directory laid out as uDocker 1.3.17 extracts one."""
    seed = tmp_path / "store" / "containers" / "0f0e0d0c-seed"
    (seed / "ROOT" / "testbed").mkdir(parents=True)
    (seed / "ROOT" / "testbed" / "module.py").write_text("x = 1\n")
    (seed / "ROOT" / "bin").mkdir()
    (seed / "ROOT" / "bin" / "sh").write_text("#!/bin/true\n")
    (seed / "container.json").write_text('{"architecture": "amd64"}')
    (seed / "imagerepo.name").write_text("repo/image:latest")
    (seed / "thundersync-image.txt").write_text("repo/image\n")
    if execmode is not None:
        (seed / "execmode").write_text(execmode)
    return seed


def _startable_box(tmp_path, monkeypatch, name="unit", *, calls=None, seed=None):
    """A sandbox whose start() runs the real copy, publish and retract.

    The uDocker CLI is faked the way 1.3.17 answers: ``setup`` succeeds and
    ``inspect -p NAME`` prints ``<containers>/NAME/ROOT`` only when that
    directory exists. The host worktree copy and the probes are faked.
    """
    seed = seed or _seed_container(tmp_path)
    box = UDockerSandbox(
        "repo/image",
        name,
        udocker_dir=tmp_path / "store",
        scratch_dir=tmp_path / "scratch",
    )
    calls = [] if calls is None else calls

    def fake_run(args, *, timeout, network_guard=False, **_limits):
        calls.append(args[0])
        if args[0] == "inspect":
            root = box.store_containers_dir / args[-1] / "ROOT"
            if root.is_dir():
                return subprocess.CompletedProcess(args, 0, f"{root}\n", "")
            return subprocess.CompletedProcess(args, 1, "", "Error: not found")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(box, "_run", fake_run)
    monkeypatch.setattr(box, "_ensure_seed", lambda: seed)
    monkeypatch.setattr(box, "_prepare_host_paths", lambda: None)
    monkeypatch.setattr(
        box, "exec", lambda _cmd: "THUNDERSYNC_WORKDIR_OK THUNDERSYNC_TOOLCHAIN=pytest"
    )
    return box


def _forbid_exclusive_store_lock(monkeypatch, box):
    @contextmanager
    def forbidden():
        raise AssertionError("sandbox provisioning took the exclusive store lock")
        yield  # pragma: no cover

    monkeypatch.setattr(box, "_store_lock", forbidden)


def _staged(tmp_path):
    staging = tmp_path / "store" / "containers" / STAGING_DIRNAME
    return sorted(path.name for path in staging.iterdir()) if staging.is_dir() else []


def test_start_publishes_a_complete_copy_under_the_sandbox_name(
    monkeypatch, tmp_path
):
    calls = []
    box = _startable_box(tmp_path, monkeypatch, calls=calls)
    _forbid_exclusive_store_lock(monkeypatch, box)
    box.start()
    published = box.store_containers_dir / box.container_name
    # A plain directory uDocker resolves by its own name: no alias symlink,
    # so no alias ever has to be removed while other runs scan them.
    assert published.is_dir() and not published.is_symlink()
    assert (published / "ROOT" / "testbed" / "module.py").read_text() == "x = 1\n"
    assert (published / "container.json").read_text() == '{"architecture": "amd64"}'
    assert box.host_root == published / "ROOT"
    assert "clone" not in calls and "rm" not in calls
    assert calls[:2] == ["setup", "inspect"]
    assert _staged(tmp_path) == []
    assert set(box.start_timings) == {
        "seed_s",
        "copy_s",
        "register_s",
        "prepare_s",
        "probe_s",
    }


def test_close_retracts_the_copy_without_udocker_rm_or_the_store_lock(
    monkeypatch, tmp_path
):
    calls = []
    box = _startable_box(tmp_path, monkeypatch, calls=calls)
    _forbid_exclusive_store_lock(monkeypatch, box)
    box.start()
    published = box.store_containers_dir / box.container_name
    # A policy can leave read-only directories and plant links to host paths.
    outside = tmp_path / "host-data"
    outside.mkdir()
    (outside / "keep.txt").write_text("host")
    frozen = published / "ROOT" / "go" / "pkg" / "mod"
    frozen.mkdir(parents=True)
    (frozen / "cached.go").write_text("package x\n")
    frozen.chmod(0o555)
    (published / "ROOT" / "go").chmod(0o555)
    (published / "ROOT" / "escape").symlink_to(outside)
    box.close()
    assert not os.path.lexists(published)
    assert _staged(tmp_path) == []
    assert (outside / "keep.txt").read_text() == "host"
    assert "rm" not in calls


def test_remove_tree_empties_read_only_directories_and_never_follows_links(
    tmp_path,
):
    tree = tmp_path / "tree"
    locked = tree / "a" / "b"
    locked.mkdir(parents=True)
    (locked / "file").write_text("x")
    locked.chmod(0o500)
    (tree / "a").chmod(0o500)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("x")
    outside.chmod(0o500)
    (tree / "link").symlink_to(outside, target_is_directory=True)
    try:
        assert remove_tree(tree)
        assert not tree.exists()
        assert (outside / "keep").read_text() == "x"
        assert stat.S_IMODE(outside.stat().st_mode) == 0o500
        assert remove_tree(tree), "an absent tree is already removed"
    finally:
        outside.chmod(0o700)


def _copy_command_failing(times, log):
    """A copy that leaves a partial tree and fails its first ``times`` runs."""

    def command(source, destination):
        script = (
            'n=$(cat "$3" 2>/dev/null || echo 0); n=$((n+1)); echo "$n" > "$3"; '
            f'if [ "$n" -le {times} ]; then touch "$2/partial"; exit 1; fi; '
            'cp -R "$1/." "$2"'
        )
        return ["sh", "-c", script, "sh", str(source), str(destination), str(log)]

    return command


def test_a_failed_copy_retries_then_fails_closed_without_publishing(
    monkeypatch, tmp_path
):
    box = _startable_box(tmp_path, monkeypatch)
    log = tmp_path / "copies"
    monkeypatch.setattr(
        udocker_sandbox, "copy_tree_command", _copy_command_failing(99, log)
    )
    monkeypatch.setattr(udocker_sandbox.time, "sleep", lambda _s: None)
    with pytest.raises(RuntimeError, match="copy failed"):
        box.start()
    assert log.read_text().strip() == "5"
    assert not os.path.lexists(box.store_containers_dir / box.container_name)
    assert _staged(tmp_path) == []


def test_a_transient_copy_failure_recovers_on_retry(monkeypatch, tmp_path):
    # A copy that fails under store pressure can succeed when reissued. One
    # flaky copy must not kill a run, and its partial tree must not survive
    # under any name.
    box = _startable_box(tmp_path, monkeypatch)
    log = tmp_path / "copies"
    monkeypatch.setattr(
        udocker_sandbox, "copy_tree_command", _copy_command_failing(1, log)
    )
    monkeypatch.setattr(udocker_sandbox.time, "sleep", lambda _s: None)
    box.start()
    published = box.store_containers_dir / box.container_name
    assert log.read_text().strip() == "2"
    assert not (published / "partial").exists()
    assert (published / "ROOT" / "testbed" / "module.py").is_file()
    assert _staged(tmp_path) == []


def test_setup_failure_retracts_the_copy_and_raises(monkeypatch, tmp_path):
    box = _startable_box(tmp_path, monkeypatch)
    succeed = box._run

    def failing_setup(args, *, timeout, network_guard=False):
        if args[0] == "setup":
            return subprocess.CompletedProcess(args, 1, "", "setup broke")
        return succeed(args, timeout=timeout, network_guard=network_guard)

    monkeypatch.setattr(box, "_run", failing_setup)
    with pytest.raises(RuntimeError, match="P1 setup failed"):
        box.start()
    assert not os.path.lexists(box.store_containers_dir / box.container_name)
    assert _staged(tmp_path) == []


def test_a_copy_udocker_does_not_resolve_fails_closed(monkeypatch, tmp_path):
    box = _startable_box(tmp_path, monkeypatch)
    elsewhere = tmp_path / "elsewhere" / "ROOT"
    elsewhere.mkdir(parents=True)
    monkeypatch.setattr(box, "_inspect_root", lambda _name: elsewhere)
    with pytest.raises(RuntimeError, match="does not resolve"):
        box.start()
    assert not os.path.lexists(box.store_containers_dir / box.container_name)


def test_a_seed_outside_the_proot_modes_is_refused(monkeypatch, tmp_path):
    for mode, refused in (("F3", True), ("P1", False), (None, False)):
        seed = _seed_container(tmp_path / (mode or "default"), execmode=mode)
        store = tmp_path / (mode or "default") / "store"
        box = UDockerSandbox("repo/image", "unit", udocker_dir=store)
        monkeypatch.setattr(box, "_inspect_root", lambda _name, s=seed: s / "ROOT")
        if refused:
            with pytest.raises(RuntimeError, match="execution mode"):
                box._ensure_seed()
        else:
            assert box._ensure_seed() == seed


def _timed_copy(log):
    """A copy that records its start and end and takes half a second."""

    clock = f"$('{sys.executable}' -c 'import time; print(time.time())')"

    def command(source, destination):
        script = (
            f'echo "start {clock}" >> "$3"; sleep 0.5; cp -R "$1/." "$2"; '
            f'echo "end {clock}" >> "$3"'
        )
        return ["sh", "-c", script, "sh", str(source), str(destination), str(log)]

    return command


def _copy_times(log):
    times = {"start": [], "end": []}
    for line in log.read_text().splitlines():
        kind, value = line.split()
        times[kind].append(float(value))
    return sorted(times["start"]), sorted(times["end"])


def test_copies_overlap_with_the_real_store_locks_live(monkeypatch, tmp_path):
    seed = _seed_container(tmp_path)
    log = tmp_path / "copies.log"
    monkeypatch.setattr(udocker_sandbox, "copy_tree_command", _timed_copy(log))
    boxes = [
        _startable_box(tmp_path, monkeypatch, f"unit-{index}", seed=seed)
        for index in range(2)
    ]
    threads = [threading.Thread(target=box.start) for box in boxes]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    for box in boxes:
        assert (box.store_containers_dir / box.container_name).is_dir()
    starts, ends = _copy_times(log)
    assert len(starts) == 2 and len(ends) == 2
    assert starts[1] < ends[0], "the second copy waited for the first"


def test_seed_creation_still_excludes_every_copy(monkeypatch, tmp_path):
    log = tmp_path / "copies.log"
    monkeypatch.setattr(udocker_sandbox, "copy_tree_command", _timed_copy(log))
    box = _startable_box(tmp_path, monkeypatch)
    worker = threading.Thread(target=box.start)
    with box._store_lock():
        worker.start()
        time.sleep(0.5)
        assert not log.exists(), "a copy ran beside a seed's creation"
    worker.join(timeout=30)
    assert (box.store_containers_dir / box.container_name).is_dir()


def _sandbox_running_script(tmp_path, body):
    script = tmp_path / "fake-udocker"
    script.write_text(f"#!/bin/sh\n{body}")
    script.chmod(0o755)
    return UDockerSandbox(
        "repo/image",
        "unit",
        udocker_dir=tmp_path / "store",
        scratch_dir=tmp_path / "scratch",
        bin_path=str(script),
    )


def _exited(pid, timeout=10.0):
    # A killed descendant that was reparented to a subreaper (a test runner
    # under a process-tree supervisor) stays a zombie until that reaper
    # collects it, and os.kill(pid, 0) still succeeds on a zombie. Read
    # the /proc state where it exists; a zombie has exited for this purpose.
    deadline = time.monotonic() + timeout
    stat = Path("/proc") / str(pid) / "stat"
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        try:
            state = stat.read_text().rsplit(")", 1)[1].split()[0]
        except (OSError, IndexError):
            state = ""
        if state in {"Z", "X"}:
            return True
        if time.monotonic() >= deadline:
            os.kill(pid, 9)
            return False
        time.sleep(0.05)


def test_a_backgrounded_descendant_does_not_outlive_its_exec(tmp_path):
    # A udocker/PRoot descendant of a rollout exec must not outlive it in the
    # worker's process group. The exec owns its subtree, so it cannot.
    box = _sandbox_running_script(
        tmp_path, 'sleep 300 >/dev/null 2>&1 &\necho "$!" > "$1"\nexit 0\n'
    )
    pid_path = tmp_path / "background.pid"
    result = box._run([str(pid_path)], timeout=30)
    assert result.returncode == 0
    assert _exited(int(pid_path.read_text())), "the exec left a live descendant"


def test_a_timed_out_exec_kills_its_whole_group(tmp_path):
    box = _sandbox_running_script(
        tmp_path, 'sleep 300 >/dev/null 2>&1 &\necho "$!" > "$1"\nsleep 300\n'
    )
    pid_path = tmp_path / "background.pid"
    with pytest.raises(subprocess.TimeoutExpired):
        box._run([str(pid_path)], timeout=2)
    assert _exited(int(pid_path.read_text())), "a timeout left a live descendant"


def test_child_environment_does_not_reinterpret_cli_path_as_helper_directory(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("UDOCKER_BIN", "/opt/udocker/bin/udocker")
    monkeypatch.setenv("NCCL_NET", "Socket")
    box = UDockerSandbox("repo/image", "unit", udocker_dir=tmp_path / "store")
    environment = box._environment()
    assert "UDOCKER_BIN" not in environment
    assert "NCCL_NET" not in environment
    assert environment["UDOCKER_DIR"] == str(tmp_path / "store")
