"""Docker backend for rollout and verifier sandboxes.

A site whose Docker daemon answers the current user runs its sandboxes here;
every other site runs them on udocker (``udocker_sandbox.py``). Both expose
one interface: ``start``/``exec``/``close``, the context manager, bug-patch
sealing, and a host-backed worktree (``host_workdir``) that the trusted
patch capture, isolated grading and the sandbox pool's pristine gate read
with the host's Git.

Each sandbox is a container of its own, named uniquely and labeled
``thundersync.sandbox=1`` plus the owning user id, so cleanup removes exactly the
containers this code created. The image's work directory is copied out once
per image into a host seed (serialized per image under a file lock) and
every sandbox receives a private copy of that seed, bind-mounted at the
work directory, with a private host directory at ``/tmp``.

Isolation: ``--network none`` leaves a loopback interface only, which also
serves task suites that need TCP loopback (``requires_network`` changes
nothing here). Commands run as the image's default user (root unless the
image sets ``USER``) with every capability dropped except ``RUN_CAPABILITIES``
(file ownership and permission overrides, set-uid/gid and kill: what Git, a
package install and the ownership hand-back need) and with
``no-new-privileges``. The container is given a system-wide
``safe.directory`` so its Git accepts the host-owned worktree. Memory, CPU
and PID limits are constructor arguments; the PID limit defaults to
``DEFAULT_PIDS_LIMIT``.

Ownership: right after ``docker run`` a probe file written inside the
container is read on the host. Owned by this user (rootless docker), the
host can remove everything the sandbox writes. Owned by root (a rootful
daemon), ``close`` hands the files back with an in-container ``chown``
before the host removes them. Any other owner (user-namespace remapping) or
no file at all (a daemon whose bind mounts land on another machine) refuses
the start: the host-side Git, patch capture and pristine gate could not work.

Environment: ``THUNDERSYNC_DOCKER_BIN`` (default ``docker``),
``THUNDERSYNC_SANDBOX_SCRATCH`` and ``THUNDERSYNC_DOCKER_SEED_DIR`` are read
when a sandbox is constructed or a module function is called.

Seeds: ``seed_dir`` (default ``THUNDERSYNC_DOCKER_SEED_DIR``, else ``seeds`` under
the scratch root) holds one host copy of each image's work directory, bound
to the image's content ID, so prewarm and every later run on the host reuse
it.

Timeouts: the host-side deadline is authoritative. On expiry the ``docker``
client's process group is killed and every process in the container except
its keepalive (PID 1) is killed with it; a sandbox runs one command at a
time, so nothing else is lost.

Output: every ``docker`` call is read through ``bounded_output.run_bounded``.
A command that writes more than ``output_limits.kill_bytes`` is killed the
same way a timed-out one is. ``exec_output`` takes a ``memory_guard`` that
fails that one command, never the caller, while the caller is over its
budget.
"""

from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from secrets import token_hex
from typing import Iterable

from thundersync.rollout.bounded_output import (
    BoundedCompletedProcess,
    CommandOutput,
    MemoryGuard,
    OutputLimits,
    WriteGuard,
    completed_process,
    run_bounded,
)
from thundersync.rollout.scratch_guard import ensure_private_dir

logger = logging.getLogger(__name__)

SANDBOX_LABEL = "thundersync.sandbox"
OWNER_LABEL = "thundersync.sandbox.owner"
DEFAULT_MEMORY = "4g"
DEFAULT_CPUS = "2"
DEFAULT_PIDS_LIMIT = 512
RUN_CAPABILITIES = ("CHOWN", "DAC_OVERRIDE", "FOWNER", "SETUID", "SETGID", "KILL")
"""The capabilities a sandbox container keeps; every other one is dropped."""
_SEED_MARKER = "thundersync-image.txt"
_OWNER_PROBE = ".thundersync-owner-probe"
_GO_ENVIRONMENT = (
    "if [ -x /usr/local/go/bin/go ]; then "
    "export PATH=/go/bin:/usr/local/go/bin:$PATH "
    "GOROOT=/usr/local/go GOPATH=/go GOTOOLCHAIN=local "
    "GOTELEMETRY=off; fi; "
)
_TOOLCHAIN_PROBE = (
    "if python -m pytest --version >/dev/null 2>&1; then "
    "echo THUNDERSYNC_TOOLCHAIN=pytest; "
    "elif command -v go >/dev/null 2>&1 && "
    "go version >/dev/null 2>&1; "
    "then echo THUNDERSYNC_TOOLCHAIN=go; fi"
)


class DockerImageError(RuntimeError):
    """A task image is absent from the daemon and was not (or could not be) pulled."""


class DockerBindMountError(RuntimeError):
    """The container's bind mounts are not this host's paths as this user."""


def docker_bin() -> str:
    """The docker CLI: ``THUNDERSYNC_DOCKER_BIN``, else ``docker``."""
    return os.environ.get("THUNDERSYNC_DOCKER_BIN", "docker")


def default_docker_scratch_dir() -> Path:
    """``THUNDERSYNC_SANDBOX_SCRATCH``, else a per-user directory under ``/tmp``."""
    return Path(
        os.environ.get(
            "THUNDERSYNC_SANDBOX_SCRATCH", f"/tmp/thundersync-docker-{os.getuid()}"
        )
    )


def _capability_arguments() -> list[str]:
    return ["--cap-drop=ALL", *(f"--cap-add={cap}" for cap in RUN_CAPABILITIES)]


def report_kill(name: str, command: str, output: CommandOutput) -> None:
    """Log a sandbox command that was killed, with its output sizes."""
    logger.warning(
        "sandbox %s command %s: stdout_bytes=%s stderr_bytes=%s command=%r",
        name,
        output.killed,
        output.stdout_bytes,
        output.stderr_bytes,
        command[:200],
    )


def bounded_name(prefix: str, value: str, identity: str) -> str:
    """A runtime-safe name: ``prefix``, a cleaned ``value`` and a digest of ``identity``."""
    clean = re.sub(r"[^A-Za-z0-9_-]", "-", value).strip("-") or "sandbox"
    digest = hashlib.sha256(identity.encode()).hexdigest()[:12]
    return f"{prefix}-{clean[:28]}-{digest}"


def copy_tree_command(source: Path, destination: Path) -> list[str]:
    """``cp`` argv copying the contents of ``source`` into ``destination``.

    GNU cp shares extents where the filesystem can; other platforms copy.
    """
    if sys.platform.startswith("linux"):
        return ["cp", "-a", "--reflink=auto", f"{source}/.", str(destination)]
    return ["cp", "-a", f"{source}/.", str(destination)]


def _run_docker(
    bin_path: str,
    args: list[str],
    *,
    timeout: float,
    limits: OutputLimits | None = None,
    keep_bytes: int | None = None,
    memory_guard: MemoryGuard | None = None,
    write_guard: WriteGuard | None = None,
) -> BoundedCompletedProcess:
    """One ``docker`` client call in a process group of its own.

    Raises ``TimeoutExpired`` at the deadline, as ``communicate`` does.
    """
    limits = limits or OutputLimits()
    run = run_bounded(
        [bin_path, *args],
        timeout=timeout,
        keep_bytes=limits.keep_for(keep_bytes),
        kill_bytes=limits.kill_bytes,
        memory_guard=memory_guard,
        write_guard=write_guard,
    )
    return completed_process(run, timeout=timeout)


def image_id(
    image: str, *, bin_path: str | None = None, timeout: float = 120
) -> str | None:
    """The daemon's content ID for ``image``, or ``None`` when it is absent."""
    bin_path = bin_path or docker_bin()
    result = _run_docker(
        bin_path, ["image", "inspect", "--format", "{{.Id}}", image], timeout=timeout
    )
    value = result.stdout.strip()
    return value.splitlines()[-1] if result.returncode == 0 and value else None


def ensure_images(
    images: Iterable[str],
    *,
    pull: bool,
    bin_path: str | None = None,
    pull_timeout: float = 3600,
) -> dict[str, str]:
    """Every image present in the daemon, pulled first when ``pull`` allows it.

    Returns image -> content ID. Refuses with the list of missing images.
    """
    bin_path = bin_path or docker_bin()
    found: dict[str, str] = {}
    missing: list[str] = []
    for image in images:
        identity = image_id(image, bin_path=bin_path)
        if identity is None and pull:
            pulled = _run_docker(bin_path, ["pull", image], timeout=pull_timeout)
            if pulled.returncode != 0:
                missing.append(f"{image} (pull failed: {(pulled.stderr or pulled.stdout)[-300:]})")
                continue
            identity = image_id(image, bin_path=bin_path)
        if identity is None:
            missing.append(image)
        else:
            found[image] = identity
    if missing:
        raise DockerImageError(
            "task images are absent from the docker daemon: " + "; ".join(missing)
        )
    return found


def owned_containers(
    *, bin_path: str | None = None, owner: int | None = None
) -> list[str]:
    """IDs of every sandbox container this user created, running or not."""
    bin_path = bin_path or docker_bin()
    owner = os.getuid() if owner is None else owner
    result = _run_docker(
        bin_path,
        [
            "ps",
            "-aq",
            "--filter",
            f"label={SANDBOX_LABEL}=1",
            "--filter",
            f"label={OWNER_LABEL}={owner}",
        ],
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"docker ps failed: {(result.stderr or result.stdout)[-300:]}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def prune_owned_containers(
    *, bin_path: str | None = None, owner: int | None = None
) -> int:
    """Remove this user's sandbox containers and nothing else."""
    bin_path = bin_path or docker_bin()
    ids = owned_containers(bin_path=bin_path, owner=owner)
    if ids:
        result = _run_docker(bin_path, ["rm", "-f", *ids], timeout=600)
        if result.returncode != 0:
            raise RuntimeError(
                f"docker rm failed: {(result.stderr or result.stdout)[-300:]}"
            )
    return len(ids)


def retire_sandboxes(
    sandboxes: Iterable["DockerSandbox"], *, bin_path: str | None = None
) -> int:
    """Remove retired sandboxes' containers in one call and reclaim their files.

    The shutdown form for a pool's idle members: one ``docker rm -f`` for
    all of them instead of one per member, then one short-lived container
    per scratch tree that hands root-written files back to this user, so
    the host can remove them. Returns the number of containers removed.
    """
    members = [box for box in sandboxes if box.container_created]
    if not members:
        return 0
    bin_path = bin_path or docker_bin()
    _run_docker(bin_path, ["rm", "-f", *(box.container_name for box in members)], timeout=600)
    for box in members:
        box.forget_container()
    scratch_images = {
        box.scratch_dir: box.image for box in members if box.reclaims_owner
    }
    for scratch, image in scratch_images.items():
        _run_docker(
            bin_path,
            [
                "run",
                "--rm",
                "--network",
                "none",
                "--security-opt",
                "no-new-privileges",
                *_capability_arguments(),
                "--label",
                f"{SANDBOX_LABEL}=1",
                "--label",
                f"{OWNER_LABEL}={os.getuid()}",
                "-v",
                f"{scratch}:/thundersync-reclaim",
                "--entrypoint",
                "/bin/sh",
                image,
                "-c",
                f"chown -R {os.getuid()}:{os.getgid()} /thundersync-reclaim",
            ],
            timeout=1800,
        )
    for box in members:
        box.remove_host_paths()
    return len(members)


class DockerSandbox:
    """A per-rollout container with a host-backed worktree.

    Each rollout needs its own container: two rollouts editing the same
    checkout would contaminate each other's observations and destroy the
    independence the group-relative advantage assumes.
    """

    backend = "docker"

    def __init__(
        self,
        image: str,
        name: str,
        workdir: str = "/testbed",
        timeout: int = 120,
        *,
        scratch_dir: Path | str | None = None,
        seed_dir: Path | str | None = None,
        memory: str | None = DEFAULT_MEMORY,
        cpus: str | None = DEFAULT_CPUS,
        pids_limit: int | None = DEFAULT_PIDS_LIMIT,
        verify_toolchain: bool = True,
        requires_network: bool = False,
        bin_path: str | None = None,
        output_limits: OutputLimits | None = None,
    ) -> None:
        # No docker call and no filesystem change here: subclasses and pools
        # construct sandboxes they may never start.
        self.image, self.name, self.workdir, self.timeout = image, name, workdir, timeout
        self._started = False
        self.output_limits = output_limits or OutputLimits()
        self.bin_path = bin_path or docker_bin()
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.verify_toolchain = verify_toolchain
        # Loopback is always present under ``--network none``; recorded for
        # interface parity with the udocker backend, where it matters.
        self.requires_network = requires_network
        self.scratch_dir = Path(scratch_dir or default_docker_scratch_dir())
        seed_default = os.environ.get("THUNDERSYNC_DOCKER_SEED_DIR")
        self.seed_dir = Path(
            seed_dir or seed_default or self.scratch_dir / "seeds"
        )
        unique = f"{name}\0{os.getpid()}\0{token_hex(8)}"
        self.container_name = bounded_name("thundersync", name, unique)
        self.seed_name = bounded_name("thundersync-seed", image, image)
        self.host_workdir = self.scratch_dir / f"work-{self.container_name}"
        self.host_tmpdir = self.scratch_dir / f"tmp-{self.container_name}"
        self._container_created = False
        # Whether files the container writes are root's and must be handed
        # back before the host can remove them; settled by the owner probe.
        self._reclaim_owner = True

    # ------------------------------------------------------------ state

    @property
    def container_created(self) -> bool:
        """Whether a container may exist under ``container_name``."""
        return self._container_created

    @property
    def reclaims_owner(self) -> bool:
        """Whether the container's writes are root's and need a hand-back."""
        return self._reclaim_owner

    def forget_container(self) -> None:
        """Record that the container was removed outside ``close``."""
        self._container_created = False
        self._started = False

    # ------------------------------------------------------------ helpers

    def _docker(
        self,
        args: list[str],
        *,
        timeout: float,
        keep_bytes: int | None = None,
        memory_guard: MemoryGuard | None = None,
        write_guard: WriteGuard | None = None,
    ) -> BoundedCompletedProcess:
        return _run_docker(
            self.bin_path,
            args,
            timeout=timeout,
            limits=self.output_limits,
            keep_bytes=keep_bytes,
            memory_guard=memory_guard,
            write_guard=write_guard,
        )

    @property
    def _seed_root(self) -> Path:
        return self.seed_dir / self.seed_name

    @contextmanager
    def _docker_seed_lock(self):
        # One lock per image seed. Each acquisition opens its own descriptor,
        # and flock excludes separate open descriptions, so this serializes
        # threads as well as processes without serializing other images.
        if self.seed_dir.is_relative_to(self.scratch_dir):
            ensure_private_dir(self.scratch_dir)
        ensure_private_dir(self.seed_dir)
        descriptor = os.open(
            self.seed_dir / f".{self.seed_name}.lock", os.O_CREAT | os.O_RDWR, 0o600
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _ensure_docker_seed(self) -> Path:
        """The image's work directory on the host, copied out once per image."""
        with self._docker_seed_lock():
            identity = image_id(self.image, bin_path=self.bin_path)
            if identity is None:
                raise DockerImageError(
                    f"docker image {self.image!r} is absent from the daemon; "
                    "pull it first (see ensure_images)"
                )
            root = self._seed_root
            marker = root / _SEED_MARKER
            expected = f"{self.image}\n{identity}\n"
            if marker.is_file() and marker.read_text() == expected:
                return root / "workdir"
            shutil.rmtree(root, ignore_errors=True)
            staging = self.seed_dir / f".{self.seed_name}.staging-{token_hex(4)}"
            shutil.rmtree(staging, ignore_errors=True)
            (staging / "workdir").mkdir(parents=True)
            copier = bounded_name("thundersync-seedcopy", self.image, f"{self.image}\0{token_hex(8)}")
            try:
                created = self._docker(
                    [
                        "create",
                        "--name",
                        copier,
                        "--label",
                        f"{SANDBOX_LABEL}=1",
                        "--label",
                        f"{OWNER_LABEL}={os.getuid()}",
                        "--network",
                        "none",
                        "--entrypoint",
                        "true",
                        self.image,
                    ],
                    timeout=600,
                )
                if created.returncode != 0:
                    raise RuntimeError(
                        f"docker create failed for {self.image}: "
                        f"{(created.stdout + created.stderr)[-600:]}"
                    )
                copied = self._docker(
                    ["cp", f"{copier}:{self.workdir}/.", str(staging / "workdir")],
                    timeout=3600,
                )
                if copied.returncode != 0:
                    raise RuntimeError(
                        f"workdir {self.workdir} could not be copied out of {self.image}: "
                        f"{(copied.stdout + copied.stderr)[-600:]}"
                    )
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise
            finally:
                self._docker(["rm", "-f", copier], timeout=600)
            (staging / _SEED_MARKER).write_text(expected)
            staging.rename(root)
            return root / "workdir"

    def _prepare_docker_host_paths(self, seed: Path) -> None:
        ensure_private_dir(self.scratch_dir)
        shutil.rmtree(self.host_workdir, ignore_errors=True)
        shutil.rmtree(self.host_tmpdir, ignore_errors=True)
        self.host_workdir.mkdir(parents=True)
        self.host_tmpdir.mkdir(parents=True)
        # The container's /tmp: world-writable with the sticky bit, as a
        # real /tmp is, so a non-root step inside the image can use it.
        os.chmod(self.host_tmpdir, 0o1777)
        copied = subprocess.run(
            copy_tree_command(seed, self.host_workdir),
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if copied.returncode != 0:
            raise RuntimeError(f"could not seed docker workdir: {copied.stderr[-400:]}")

    def _run_arguments(self) -> list[str]:
        args = [
            "run",
            "-d",
            "--name",
            self.container_name,
            "--label",
            f"{SANDBOX_LABEL}=1",
            "--label",
            f"{OWNER_LABEL}={os.getuid()}",
            "--network",
            "none",
            # PID 1 must be the keepalive, whatever the daemon's default
            # init: the timeout kill spares PID 1 and nothing else.
            "--init=false",
            # A setuid binary in the image cannot raise a policy command's
            # privileges past the container's starting set.
            "--security-opt",
            "no-new-privileges",
            *_capability_arguments(),
        ]
        if self.memory is not None:
            args += ["--memory", str(self.memory)]
        if self.cpus is not None:
            args += ["--cpus", str(self.cpus)]
        if self.pids_limit is not None:
            args += ["--pids-limit", str(self.pids_limit)]
        args += [
            "-v",
            f"{self.host_workdir}:{self.workdir}",
            "-v",
            f"{self.host_tmpdir}:/tmp",
            "-w",
            self.workdir,
            "-e",
            "HOME=/root",
            "--entrypoint",
            "sleep",
            self.image,
            "infinity",
        ]
        return args

    def _exec_arguments(self, command: str) -> list[str]:
        return [
            "exec",
            "-w",
            self.workdir,
            "-e",
            "HOME=/root",
            self.container_name,
            "/bin/bash",
            "-lc",
            f"export HOME=/root; {command}",
        ]

    def _probe_owner(self) -> int | None:
        """Owner uid, on the host, of a file the container just wrote to /tmp."""
        probe = self.host_tmpdir / _OWNER_PROBE
        try:
            return probe.stat().st_uid
        except FileNotFoundError:
            return None

    def _settle_ownership(self) -> None:
        written = self._docker(
            ["exec", self.container_name, "/bin/sh", "-c", f"touch /tmp/{_OWNER_PROBE}"],
            timeout=120,
        )
        owner = self._probe_owner() if written.returncode == 0 else None
        if owner is None:
            raise DockerBindMountError(
                "a file written in the container's /tmp did not appear at "
                f"{self.host_tmpdir}; the daemon's bind mounts are not this host's "
                "paths (a remote DOCKER_HOST is unsupported)"
            )
        if owner == os.getuid():
            self._reclaim_owner = False
        elif owner == 0:
            self._reclaim_owner = True
        else:
            raise DockerBindMountError(
                f"container root writes as uid {owner} on the host; user-namespace "
                "remapping is unsupported (the host must read and remove the worktree)"
            )
        (self.host_tmpdir / _OWNER_PROBE).unlink(missing_ok=True)

    def remove_host_paths(self) -> None:
        """Delete this sandbox's host worktree, ``/tmp`` and trusted snapshot."""
        shutil.rmtree(self.host_workdir, ignore_errors=True)
        shutil.rmtree(self.host_tmpdir, ignore_errors=True)
        trusted_root = getattr(self, "_thundersync_trusted_root", None)
        if trusted_root is not None:
            shutil.rmtree(trusted_root, ignore_errors=True)
            self._thundersync_trusted_root = None

    def _kill_in_container(self) -> None:
        # PID 1 is the keepalive; ``kill -1`` spares it and the sender.
        try:
            self._docker(
                ["exec", self.container_name, "/bin/bash", "-c", "kill -9 -1"],
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError):
            pass

    # --------------------------------------------------------- interface

    def start(self) -> None:
        try:
            seed = self._ensure_docker_seed()
            self._prepare_docker_host_paths(seed)
            self._container_created = True
            started = self._docker(self._run_arguments(), timeout=600)
            if started.returncode != 0:
                raise RuntimeError(
                    f"docker run failed for {self.image}: "
                    f"{(started.stdout + started.stderr)[-600:]}"
                )
            self._settle_ownership()
            self._started = True
            probe = self.exec(
                "git config --system --add safe.directory '*' >/dev/null 2>&1; "
                f"test -d {shlex.quote(self.workdir)} && echo THUNDERSYNC_WORKDIR_OK"
            )
            if "THUNDERSYNC_WORKDIR_OK" not in probe:
                raise RuntimeError(f"docker workdir probe failed: {probe[:300]}")
            if self.verify_toolchain:
                probe = self.exec(_TOOLCHAIN_PROBE)
                if "THUNDERSYNC_TOOLCHAIN=" not in probe:
                    raise RuntimeError(
                        f"no supported pytest or Go verifier in {self.image}: {probe[:300]}"
                    )
        except Exception:
            self.close()
            raise

    def apply_bug_patch(self, patch: str) -> None:
        """Introduce the instance's bug, and make it the repository's only state.

        Applying the patch as an uncommitted working-tree change is not enough:
        the image's git history still holds the clean code, so ``git checkout .``
        or ``git stash`` reverts the bug in one command and the tests pass
        without the model fixing anything.

        Amending the initial commit leaves a single commit containing the buggy
        code, so ``checkout``/``stash``/``reset --hard`` all become no-ops. This
        also matches SWE-bench proper, where the checkout contains the bug and
        the history does not contain the fix.

        Raises rather than warns: a silently unapplied patch leaves the repo
        working, so every rollout scores 1.0 and the solve rate measures nothing.

        The patch reaches the sandbox as a file written on the host into its
        ``/tmp`` mount under a fresh random name, so its size is not bounded
        by a command line and no file planted beforehand is applied.
        """
        if not patch.strip():
            raise ValueError("empty bug patch")
        if not self._started:
            self.start()
        patch_name = f"thundersync_bug_{token_hex(8)}.patch"
        host_patch = self.host_tmpdir / patch_name
        host_patch.write_bytes(patch.encode())
        try:
            out = self.exec(
                f"git apply --whitespace=nowarn /tmp/{patch_name} 2>&1; echo rc=$?"
            )
        finally:
            host_patch.unlink(missing_ok=True)
        if "rc=0" not in out:
            raise RuntimeError(f"bug patch did not apply in {self.name}: {out[:400]}")

        out = self.exec(
            "git -c user.email=setup@thundersync -c user.name=thundersync "
            "commit -a --amend --no-edit --allow-empty 2>&1; echo rc=$?"
        )
        if "rc=0" not in out:
            raise RuntimeError(f"could not seal bug into history in {self.name}: {out[:400]}")

        dirty = self.exec("git status --porcelain | head -3").strip()
        if dirty:
            raise RuntimeError(f"working tree still dirty after sealing: {dirty[:200]}")

    def assert_bug_not_revertible(self) -> None:
        """Check the obvious escape hatches are actually closed.

        Reject a sandbox whose history exposes the reference solution.
        """
        n_commits = self.exec("git rev-list --count HEAD").strip()
        if n_commits.isdigit() and int(n_commits) > 1:
            raise RuntimeError(
                f"{self.name}: history has {n_commits} commits; a reset would "
                "recover the un-bugged code"
            )

    def exec_output(
        self,
        command: str,
        *,
        keep_bytes: int | None = None,
        memory_guard: MemoryGuard | None = None,
        write_guard: WriteGuard | None = None,
    ) -> CommandOutput:
        """The command's bounded output; ``exec`` is its text.

        ``keep_bytes`` overrides how much of each stream is kept whole;
        ``memory_guard`` refuses or kills this command while the caller is
        over its budget; ``write_guard`` kills it once the sandbox's scratch
        has grown past its budget.
        """
        if not self._started:
            self.start()
        try:
            result = self._docker(
                self._exec_arguments(_GO_ENVIRONMENT + command),
                timeout=self.timeout,
                keep_bytes=keep_bytes,
                memory_guard=memory_guard,
                write_guard=write_guard,
            )
        except subprocess.TimeoutExpired as error:
            self._kill_in_container()
            return CommandOutput.timed_out_after(
                self.timeout, getattr(error, "bounded_run", None)
            )
        output = CommandOutput.from_process(result, timeout_s=self.timeout)
        if output.killed is not None:
            # Killing the docker client leaves the in-container command running.
            self._kill_in_container()
            report_kill(self.name, command, output)
        return output

    def exec(self, command: str) -> str:
        return self.exec_output(command).text

    def scratch_mounts(self) -> dict[Path, str]:
        """The host directories this sandbox writes, by their in-sandbox path."""
        return {self.host_workdir: self.workdir, self.host_tmpdir: "/tmp"}

    def stop_commands(self) -> None:
        """Kill every command still running in the sandbox."""
        if self._started:
            self._kill_in_container()

    def remove_in_sandbox(self, paths: Iterable[str]) -> None:
        """Remove in-sandbox files the host may not (a rootful daemon's)."""
        paths = list(paths)
        for start in range(0, len(paths), 64):
            quoted = " ".join(shlex.quote(path) for path in paths[start : start + 64])
            self.exec_output(f"rm -f -- {quoted}")

    def close(self) -> None:
        if self._container_created and self._reclaim_owner:
            try:
                # In-container writes land root-owned in the host scratch
                # tree; hand them back before the host removes it.
                self._docker(
                    [
                        "exec",
                        self.container_name,
                        "/bin/sh",
                        "-c",
                        f"chown -R {os.getuid()}:{os.getgid()} {shlex.quote(self.workdir)} /tmp",
                    ],
                    timeout=300,
                )
            except (OSError, subprocess.SubprocessError):
                pass
        if self._container_created:
            try:
                self._docker(["rm", "-f", self.container_name], timeout=600)
            except (OSError, subprocess.SubprocessError):
                pass
            self._container_created = False
        self.remove_host_paths()
        self._started = False

    def __enter__(self) -> "DockerSandbox":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
