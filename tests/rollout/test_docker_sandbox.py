"""Docker sandbox backend, exercised against a stateful fake ``docker``.

Failure modes each test guards:

* a sandbox without a host-backed worktree (trusted capture, isolated
  grading and the pool's pristine gate all read it on the host);
* resource limits or network isolation silently dropped from ``docker run``;
* a missing task image pulled implicitly at sandbox start instead of being
  refused (images are verified up front by the launcher);
* a timed-out command left running inside the container;
* cleanup removing a container this code did not create, or leaving its own;
* concurrent sandboxes of one image racing the seed copy.
"""

from __future__ import annotations

import inspect
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


from thundersync.rollout.docker_sandbox import (  # noqa: E402
    DEFAULT_PIDS_LIMIT,
    RUN_CAPABILITIES,
    DockerBindMountError,
    DockerImageError,
    DockerSandbox,
    ensure_images,
    prune_owned_containers,
    retire_sandboxes,
)
from thundersync.rollout.engine import Task, _capture_bug_base, _capture_policy_patch  # noqa: E402
from thundersync.rollout.sandbox_pool import SandboxPool, trusted_tree_pristine  # noqa: E402
from thundersync.rollout.udocker_sandbox import UDockerSandbox  # noqa: E402
from tests.rollout import fake_docker  # noqa: E402

IMAGE = "repo/swe-image:tag"
BUG_PATCH = """\
diff --git a/solver.py b/solver.py
--- a/solver.py
+++ b/solver.py
@@ -1,2 +1,2 @@
 def solve():
-    return 42
+    return 41
"""
GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
}


def _image_tree(tmp_path: Path, image: str = IMAGE) -> Path:
    testbed = (
        tmp_path / "fake-docker-images" / fake_docker.image_directory(image) / "testbed"
    )
    testbed.mkdir(parents=True)
    (testbed / "solver.py").write_text("def solve():\n    return 42\n")
    env = {**GIT_ENV, "PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(tmp_path)}
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "clean"]):
        subprocess.run(["git", *args], cwd=testbed, check=True, capture_output=True, env=env)
    return testbed


@pytest.fixture
def docker_site(tmp_path, monkeypatch):
    state = fake_docker.install(tmp_path, monkeypatch, images=(IMAGE,))
    for key, value in GIT_ENV.items():
        monkeypatch.setenv(key, value)
    _image_tree(tmp_path)
    return state


def _box(tmp_path: Path, name: str = "unit", **kwargs) -> DockerSandbox:
    return DockerSandbox(
        IMAGE, name, scratch_dir=tmp_path / "scratch", verify_toolchain=False, **kwargs
    )


def test_udocker_shares_the_docker_sandbox_interface():
    assert issubclass(UDockerSandbox, DockerSandbox)
    assert inspect.signature(UDockerSandbox.exec) == inspect.signature(DockerSandbox.exec)
    assert DockerSandbox.backend == "docker" and UDockerSandbox.backend == "udocker"


def test_constructing_a_sandbox_touches_neither_docker_nor_the_disk(tmp_path, docker_site):
    box = _box(tmp_path)
    assert fake_docker.calls(docker_site) == []
    assert not (tmp_path / "scratch").exists()
    assert box.container_name != _box(tmp_path).container_name


def test_run_isolates_the_network_and_takes_its_limits_from_arguments(tmp_path, docker_site):
    with _box(tmp_path, memory="6g", cpus="3", pids_limit=512) as box:
        assert "ok" in box.exec("echo ok")
        run = next(call for call in fake_docker.calls(docker_site) if call[0] == "run")
    assert run[run.index("--network") + 1] == "none"
    assert run[run.index("--memory") + 1] == "6g"
    assert run[run.index("--cpus") + 1] == "3"
    assert run[run.index("--pids-limit") + 1] == "512"
    assert f"{box.host_workdir}:/testbed" in run
    assert f"{box.host_tmpdir}:/tmp" in run
    assert "thundersync.sandbox=1" in run
    assert "--init=false" in run


def test_default_limits(tmp_path, docker_site):
    with _box(tmp_path) as box:
        box.exec("true")
    run = next(call for call in fake_docker.calls(docker_site) if call[0] == "run")
    assert run[run.index("--memory") + 1] == "4g"
    assert run[run.index("--cpus") + 1] == "2"
    assert run[run.index("--pids-limit") + 1] == str(DEFAULT_PIDS_LIMIT) == "512"


def test_limits_can_be_left_to_the_daemon(tmp_path, docker_site):
    with _box(tmp_path, memory=None, cpus=None, pids_limit=None) as box:
        box.exec("true")
    run = next(call for call in fake_docker.calls(docker_site) if call[0] == "run")
    assert "--memory" not in run and "--cpus" not in run and "--pids-limit" not in run


def test_run_drops_every_capability_but_the_sandbox_set(tmp_path, docker_site):
    with _box(tmp_path) as box:
        box.exec("true")
    run = next(call for call in fake_docker.calls(docker_site) if call[0] == "run")
    assert run[run.index("--security-opt") + 1] == "no-new-privileges"
    assert "--cap-drop=ALL" in run
    added = [arg.removeprefix("--cap-add=") for arg in run if arg.startswith("--cap-add=")]
    assert added == list(RUN_CAPABILITIES)
    assert run.index("--cap-drop=ALL") < run.index(f"--cap-add={RUN_CAPABILITIES[0]}")
    assert run.index("--cap-drop=ALL") < run.index(IMAGE)


def test_worktree_is_host_backed_and_seeded_from_the_image(tmp_path, docker_site):
    with _box(tmp_path) as box:
        host = Path(box.host_workdir)
        assert (host / "solver.py").read_text().endswith("return 42\n")
        box.exec("echo changed > solver.py")
        assert (host / "solver.py").read_text() == "changed\n"
    assert not host.exists()


def test_bug_sealing_and_trusted_capture_run_on_the_docker_backend(tmp_path, docker_site):
    task = Task("demo__task.1", "fix it", IMAGE, bug_patch=BUG_PATCH)
    with _box(tmp_path) as box:
        box.apply_bug_patch(BUG_PATCH)
        box.assert_bug_not_revertible()
        base = _capture_bug_base(box, task, trusted_host_snapshot=True)
        assert base and len(base) == 40
        assert trusted_tree_pristine(box)
        box.exec("sed -i.bak 's/41/42/' solver.py && rm solver.py.bak")
        assert not trusted_tree_pristine(box)
        patch = _capture_policy_patch(box, max_bytes=1_000_000)
        assert b"+    return 42" in patch
        trusted_root = box._thundersync_trusted_root
    assert not Path(trusted_root).exists()


def test_the_pool_reuses_a_pristine_docker_sandbox_and_closes_it(tmp_path, docker_site):
    task = Task("demo__task.1", "fix it", IMAGE, bug_patch=BUG_PATCH)
    pool = SandboxPool(
        lambda image, name: DockerSandbox(
            image, name, scratch_dir=tmp_path / "scratch", verify_toolchain=False
        ),
        name_prefix="pool",
    )
    first = pool.acquire(task)
    first.exec("echo stray > untracked.txt")
    pool.release(task, first)
    assert pool.acquire(task) is first
    pool.release(task, first)
    pool.close()
    assert pool.report()["reused"] == 1
    assert list((docker_site / "containers").glob("*.json")) == []


def test_a_missing_image_is_refused_not_pulled_at_start(tmp_path, monkeypatch):
    state = fake_docker.install(tmp_path, monkeypatch, images=(), pullable=(IMAGE,))
    with pytest.raises(DockerImageError, match="absent"):
        _box(tmp_path).start()
    assert not any(call[0] in {"pull", "run"} for call in fake_docker.calls(state))


def test_ensure_images_pulls_only_when_allowed_and_refuses_the_rest(tmp_path, monkeypatch):
    fake_docker.install(tmp_path, monkeypatch, images=("present:1",), pullable=("pullable:1",))
    with pytest.raises(DockerImageError, match="pullable:1"):
        ensure_images(["present:1", "pullable:1"], pull=False)
    found = ensure_images(["present:1", "pullable:1"], pull=True)
    assert set(found) == {"present:1", "pullable:1"}
    assert all(value.startswith("sha256:") for value in found.values())
    with pytest.raises(DockerImageError, match="absent:1"):
        ensure_images(["absent:1"], pull=True)


def test_a_timeout_kills_the_command_inside_the_container(tmp_path, docker_site):
    with _box(tmp_path, timeout=1) as box:
        output = box.exec("sleep 20")
        assert output == "<command timed out after 1s>"
        killed = (docker_site / "killed.txt").read_text().split()
        assert killed == [box.container_name]


def test_exit_statuses_of_timeout_wrappers_read_as_timeouts(tmp_path, docker_site):
    with _box(tmp_path, timeout=30) as box:
        assert box.exec("echo partial; exit 124").endswith(
            "partial\n<command timed out after 30s>"
        )
        assert "timed out" not in box.exec("exit 3")


def _rootful(monkeypatch) -> None:
    """The fake daemon writes as this user; a rootful daemon writes as root."""
    monkeypatch.setattr(DockerSandbox, "_probe_owner", lambda self: 0)


def test_close_hands_files_back_and_removes_only_its_container(
    tmp_path, docker_site, monkeypatch
):
    _rootful(monkeypatch)
    (docker_site / "containers").mkdir(exist_ok=True)
    foreign = docker_site / "containers" / "someone-else.json"
    foreign.write_text('{"image": "x", "labels": [],"mounts": {}, "workdir": "/", "running": true}')
    box = _box(tmp_path)
    box.start()
    box.close()
    removals = [call for call in fake_docker.calls(docker_site) if call[:2] == ["rm", "-f"]]
    assert ["rm", "-f", box.container_name] in removals
    assert all("someone-else" not in call for call in removals)
    assert (docker_site / "chowned.txt").read_text().split() == [box.container_name]
    assert foreign.exists()


def test_a_failed_start_cleans_up_after_itself(tmp_path, docker_site, monkeypatch):
    from thundersync.rollout import docker_sandbox

    monkeypatch.setattr(docker_sandbox, "_TOOLCHAIN_PROBE", "echo no-toolchain")
    box = DockerSandbox(IMAGE, "unit", scratch_dir=tmp_path / "scratch")
    with pytest.raises(RuntimeError, match="no supported pytest or Go verifier"):
        box.start()
    assert not Path(box.host_workdir).exists()
    assert not (docker_site / "containers" / f"{box.container_name}.json").exists()


def test_concurrent_sandboxes_of_one_image_copy_the_seed_once(tmp_path, docker_site):
    def run(index: int) -> str:
        with _box(tmp_path, name=f"c{index}") as box:
            return box.exec("cat solver.py")

    with ThreadPoolExecutor(max_workers=6) as pool:
        outputs = list(pool.map(run, range(6)))
    assert all("return 42" in output for output in outputs)
    creates = [call for call in fake_docker.calls(docker_site) if call[0] == "create"]
    assert len(creates) == 1


def test_prune_removes_only_this_users_sandbox_containers(tmp_path, docker_site):
    box = _box(tmp_path)
    box.start()
    foreign = docker_site / "containers" / "someone-else.json"
    foreign.write_text('{"image": "x", "labels": ["thundersync.sandbox=1"], "mounts": {}, "workdir": "/", "running": true}')
    assert prune_owned_containers() == 1
    assert foreign.exists()
    assert not (docker_site / "containers" / f"{box.container_name}.json").exists()


def test_retiring_a_pool_removes_its_idle_members_in_one_call(
    tmp_path, docker_site, monkeypatch
):
    _rootful(monkeypatch)
    task = Task("demo__task.1", "fix it", IMAGE)
    pool = SandboxPool(
        lambda image, name: DockerSandbox(
            image, name, scratch_dir=tmp_path / "scratch", verify_toolchain=False
        ),
        name_prefix="pool",
    )
    first, second = pool.acquire(task), pool.acquire(task)
    pool.release(task, first)
    pool.release(task, second)
    retired = pool.retire()
    assert {box.container_name for box in retired} == {
        first.container_name,
        second.container_name,
    }
    assert retire_sandboxes(retired) == 2
    removals = [call for call in fake_docker.calls(docker_site) if call[:2] == ["rm", "-f"]]
    assert sorted(removals[-1][2:]) == sorted([first.container_name, second.container_name])
    reclaim = [call for call in fake_docker.calls(docker_site) if call[:2] == ["run", "--rm"]]
    assert len(reclaim) == 1 and f"{tmp_path / 'scratch'}:/thundersync-reclaim" in reclaim[0]
    assert "--cap-drop=ALL" in reclaim[0] and "--cap-add=CHOWN" in reclaim[0]
    assert not first.container_created and not second.container_created
    assert list((docker_site / "containers").glob("thundersync-pool-*.json")) == []
    assert not Path(first.host_workdir).exists() and not Path(second.host_workdir).exists()


def test_a_rootless_daemon_needs_no_ownership_hand_back(tmp_path, docker_site):
    box = _box(tmp_path)
    box.start()
    assert box.reclaims_owner is False
    box.close()
    assert not (docker_site / "chowned.txt").exists()
    assert not Path(box.host_workdir).exists()


@pytest.mark.parametrize(
    ("owner", "message"),
    [(None, "remote DOCKER_HOST"), (12345, "user-namespace remapping")],
)
def test_bind_mounts_that_the_host_cannot_own_refuse_the_start(
    tmp_path, docker_site, monkeypatch, owner, message
):
    monkeypatch.setattr(DockerSandbox, "_probe_owner", lambda self: owner)
    box = _box(tmp_path)
    with pytest.raises(DockerBindMountError, match=message):
        box.start()
    assert not (docker_site / "containers" / f"{box.container_name}.json").exists()


def test_seeds_default_to_the_shared_seed_directory(tmp_path, docker_site, monkeypatch):
    shared = tmp_path / "shared-seeds"
    monkeypatch.setenv("THUNDERSYNC_DOCKER_SEED_DIR", str(shared))
    for index in range(2):
        with DockerSandbox(
            IMAGE, f"s{index}", scratch_dir=tmp_path / f"run{index}", verify_toolchain=False
        ) as box:
            assert box.seed_dir == shared
    creates = [call for call in fake_docker.calls(docker_site) if call[0] == "create"]
    assert len(creates) == 1
    assert (shared / box.seed_name / "workdir" / "solver.py").is_file()


def test_a_seed_directory_outside_the_scratch_root_is_private(tmp_path, docker_site):
    seeds = tmp_path / "elsewhere" / "seeds"
    with _box(tmp_path, seed_dir=seeds) as box:
        box.exec("true")
    assert seeds.stat().st_mode & 0o777 == 0o700


def test_a_world_writable_seed_directory_is_refused(tmp_path, docker_site):
    seeds = tmp_path / "shared"
    seeds.mkdir()
    seeds.chmod(0o777)
    with pytest.raises(RuntimeError, match="writable by other users"):
        _box(tmp_path, seed_dir=seeds).start()
    assert not any(call[0] in {"create", "run"} for call in fake_docker.calls(docker_site))
