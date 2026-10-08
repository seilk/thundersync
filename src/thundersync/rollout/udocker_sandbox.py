"""Unprivileged udocker backend for hosts without a usable Docker daemon.

The backend uses udocker's PRoot mode only as an execution mechanism.  Every
policy and verifier receives a distinct extracted container root, an external
host-backed worktree, and a private ``/tmp``.  Host ``/proc``, ``/sys``,
``/run``, and the `/dev` tree are not mounted; four standard character devices
are bound individually for Git and test-runner compatibility.  A seccomp
filter (``network_guard.py``) denies every socket family except AF_UNIX for
the complete command process tree.  A sandbox constructed with
``requires_network`` (a task whose integration tests need TCP loopback) is
the exception.  PRoot cannot create a network namespace, so such a sandbox
receives host networking with private localhost-only name-service files.

PRoot translates paths through ptrace; it is not a kernel namespace and not
a security boundary against hostile code. Unlike the Docker backend this one
sets no container memory, CPU or PID limit. Its isolation is the private
filesystem copy, the socket filter, and the host-side output and time limits.

Provisioning runs concurrently.  A uDocker store has no central index: a
container is a directory under ``containers/`` (plus optional alias symlinks),
and ``udocker clone`` is a whole-directory copy of the source container.  A
sandbox is therefore the image's seed directory copied into a private staging
directory and renamed to ``containers/<container_name>``, a name uDocker
resolves as a container id.  The rename is the only step that makes it
visible, so a partial copy never is; removal renames it out again before
deleting the tree.  Copies and removals hold the store lock shared, so they
run together; seed creation (``create`` and ``setup``) holds it exclusively.

Every ``udocker`` call is read through ``bounded_output.run_bounded``, with
the same output cap, and ``exec_output`` the same memory guard, as the
docker backend.

Environment: ``THUNDERSYNC_UDOCKER_BIN`` (default ``udocker``),
``UDOCKER_DIR`` (default ``~/.udocker``), ``THUNDERSYNC_SANDBOX_SCRATCH`` and
``PROOT_TMP_DIR`` are read when a sandbox is constructed.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import logging
import os
import shlex
import shutil
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from secrets import token_hex

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
from thundersync.rollout.docker_sandbox import (
    DockerSandbox,
    bounded_name,
    copy_tree_command,
    report_kill,
)


logger = logging.getLogger(__name__)


def udocker_bin() -> str:
    """The udocker CLI: ``THUNDERSYNC_UDOCKER_BIN``, else ``udocker``."""
    return os.environ.get("THUNDERSYNC_UDOCKER_BIN", "udocker")


def default_udocker_dir() -> Path:
    """``UDOCKER_DIR``, else ``~/.udocker``."""
    return Path(os.environ.get("UDOCKER_DIR", str(Path.home() / ".udocker")))


def default_scratch_dir() -> Path:
    """``THUNDERSYNC_SANDBOX_SCRATCH``, else a per-user directory under ``/tmp``."""
    return Path(
        os.environ.get(
            "THUNDERSYNC_SANDBOX_SCRATCH", f"/tmp/thundersync-udocker-{os.getuid()}"
        )
    )


def default_proot_tmp_dir() -> Path:
    """``PROOT_TMP_DIR``, else ``proot-tmp`` under the default scratch directory."""
    return Path(
        os.environ.get("PROOT_TMP_DIR", str(default_scratch_dir() / "proot-tmp"))
    )


_UDOCKER_STORE_THREAD_LOCK = threading.Lock()
# Copies are assembled, and removals emptied, in this directory inside the
# containers directory, so every publishing or retracting rename stays on one
# filesystem. A store prune deletes whatever a killed process left here.
STAGING_DIRNAME = ".thundersync-staging"
# Sandbox copies are directories named SANDBOX_PREFIX-...; seeds are
# SEED_PREFIX-... aliases. A store prune tells them apart by these.
SANDBOX_PREFIX = "thundersync"
SEED_PREFIX = "thundersync-seed"
# One copy plus four retries: a copy that fails while the store is under
# pressure is reissued after a backoff (2, 4, 8, 16 s) before failing closed.
_COPY_ATTEMPTS = 5


def _grant_owner_access(root: Path) -> None:
    """Give the owner rwx on every real directory below ``root``.

    ``lstat`` decides what a directory is, so a symlink is never chmodded or
    descended into.
    """
    pending = [str(root)]
    while pending:
        current = pending.pop()
        try:
            mode = os.lstat(current).st_mode
        except OSError:
            continue
        if not stat.S_ISDIR(mode):
            continue
        if mode & stat.S_IRWXU != stat.S_IRWXU:
            try:
                os.chmod(current, stat.S_IMODE(mode) | stat.S_IRWXU)
            except OSError:
                continue
        try:
            with os.scandir(current) as entries:
                pending.extend(
                    entry.path
                    for entry in entries
                    if entry.is_dir(follow_symlinks=False)
                )
        except OSError:
            continue


def remove_tree(path: Path | str) -> bool:
    """Delete a sandbox tree without following symlinks; True when it is gone.

    Commands in a sandbox can leave directories without owner write or search
    permission (read-only module caches, a policy's ``chmod``), which
    ``rmtree`` alone cannot empty; ``udocker rm -f`` chmods the tree first for
    the same reason. Only real directories are made owner-accessible and
    ``rmtree`` never follows a link, so a symlink planted in a policy-writable
    tree cannot redirect either step to a host path. A residue is logged: on a
    memory-backed store it is gigabytes.
    """
    path = Path(path)
    for attempt in range(2):
        try:
            shutil.rmtree(path)
            return True
        except OSError as error:
            if not os.path.lexists(path):
                return True
            if attempt == 0:
                _grant_owner_access(path)
                continue
            logger.warning("could not remove sandbox tree %s: %s", path, error)
    return False


def _same_directory(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except OSError:
        return False


class UDockerSandbox(DockerSandbox):
    """A fresh udocker PRoot tree exposing the ``DockerSandbox`` interface."""

    backend = "udocker"

    def __init__(
        self,
        image: str,
        name: str,
        workdir: str = "/testbed",
        timeout: int = 120,
        *,
        udocker_dir: Path | str | None = None,
        containers_dir: Path | str | None = None,
        scratch_dir: Path | str | None = None,
        proot_tmp_dir: Path | str | None = None,
        verify_toolchain: bool = True,
        isolate_network: bool = True,
        requires_network: bool = False,
        bin_path: str | None = None,
        output_limits: OutputLimits | None = None,
    ) -> None:
        bin_path = bin_path or udocker_bin()
        super().__init__(
            image,
            name,
            workdir=workdir,
            timeout=timeout,
            memory=None,
            cpus=None,
            pids_limit=None,
            bin_path=bin_path,
            output_limits=output_limits,
        )
        self.udocker_dir = Path(udocker_dir or default_udocker_dir())
        self.containers_dir = (
            Path(containers_dir) if containers_dir is not None else None
        )
        self.scratch_dir = Path(scratch_dir or default_scratch_dir())
        # uDocker/PRoot may materialize or clean helper files below
        # PROOT_TMP_DIR during startup.  A rollout worker starts several
        # sandboxes concurrently, so sharing one directory across those
        # starts can make one PRoot probe observe another start's transient
        # state (reported by uDocker as "proot executable not found").
        # Keep the CLI path as a run-level base, but give every sandbox a
        # private child after its unique container name is known.
        proot_tmp_base = Path(proot_tmp_dir or default_proot_tmp_dir())
        self.verify_toolchain = verify_toolchain
        self.isolate_network = isolate_network
        # True runs commands without the socket filter, on host networking.
        self.requires_network = requires_network
        unique = f"{name}\0{os.getpid()}\0{token_hex(8)}"
        self.container_name = bounded_name(SANDBOX_PREFIX, name, unique)
        self.proot_tmp_dir = proot_tmp_base / self.container_name
        self.seed_name = bounded_name(SEED_PREFIX, image, image)
        self.host_root: Path | None = None
        self.host_workdir = self.scratch_dir / f"work-{self.container_name}"
        self.host_tmpdir = self.scratch_dir / f"tmp-{self.container_name}"
        self.host_runtime_dir = self.scratch_dir / f"runtime-{self.container_name}"
        # Wall seconds of each start() phase; the sandbox pool aggregates
        # them into its report.
        self.start_timings: dict[str, float] = {}

    @property
    def store_containers_dir(self) -> Path:
        """The directory uDocker reads containers from (UDOCKER_CONTAINERS)."""
        return self.containers_dir or (self.udocker_dir / "containers")

    def _environment(self) -> dict[str, str]:
        environment = {
            # NCCL is a trainer-process transport setting.  Passing it into
            # the verifier changes uDocker/PRoot command setup on some
            # images (notably ``NCCL_NET=Socket``) and can make the sandbox
            # reject its own executable before the task starts.  Keep the
            # trainer's environment unchanged; verifier commands do not need
            # collective-transport variables.
            **{
                key: value
                for key, value in os.environ.items()
                if not key.startswith("NCCL_")
            },
            "UDOCKER_DIR": str(self.udocker_dir),
            "PROOT_TMP_DIR": str(self.proot_tmp_dir),
            "UDOCKER_LOGLEVEL": "2",
        }
        # uDocker interprets UDOCKER_BIN as its *helper directory* (the
        # directory containing proot/patchelf); an inherited value naming the
        # CLI executable makes PRoot discovery search a file path and report
        # the misleading "proot executable not found" error.  The CLI path is
        # already supplied directly via ``self.bin_path`` in ``_run``; keep
        # the variable out of the child environment so uDocker derives its
        # helper directory from UDOCKER_DIR/bin.
        environment.pop("UDOCKER_BIN", None)
        if self.containers_dir is not None:
            environment["UDOCKER_CONTAINERS"] = str(self.containers_dir)
        return environment

    @contextmanager
    def _seed_lock(self):
        lock_dir = self.store_containers_dir
        lock_dir.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            lock_dir / f".{self.seed_name}.lock", os.O_CREAT | os.O_RDWR, 0o600
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _store_lock_path(self) -> Path:
        lock_path = self.udocker_dir / ".thundersync-container-start.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        return lock_path

    @contextmanager
    def _store_lock(self):
        """Hold the shared store exclusively, across threads and processes.

        Only seed creation takes it: ``create`` extracts an image into a new
        container and ``setup`` configures it, and neither ran cleanly beside
        a clone. Sandbox copies and removals hold the same lock shared
        (``_shared_store_lock``), so any number of them run together and none
        beside a seed's creation.
        """
        with _UDOCKER_STORE_THREAD_LOCK:
            descriptor = os.open(
                self._store_lock_path(), os.O_CREAT | os.O_RDWR, 0o600
            )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    @contextmanager
    def _shared_store_lock(self):
        """The store lock in shared mode; see ``_store_lock``.

        ``flock`` binds to the open file description, so a descriptor opened
        here conflicts with an exclusive holder in this process as well as in
        another one, and shares with every other shared holder.
        """
        descriptor = os.open(self._store_lock_path(), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _run(
        self,
        args: list[str],
        *,
        timeout: int,
        network_guard: bool = False,
        keep_bytes: int | None = None,
        memory_guard: MemoryGuard | None = None,
        write_guard: WriteGuard | None = None,
    ) -> BoundedCompletedProcess:
        """One udocker call; raises ``TimeoutExpired`` at the deadline."""
        command = [self.bin_path, *args]
        if network_guard:
            command = [
                sys.executable,
                str(Path(__file__).with_name("network_guard.py")),
                *command,
            ]
        # Own the exec's subtree.  Without a session of its own the child's
        # udocker/PRoot descendants inherit the rollout worker's process
        # group; one that outlives the exec then outlives the worker.
        # ``network_guard`` execs itself in place, so the guarded command
        # leads the same group.
        run = run_bounded(
            command,
            timeout=timeout,
            keep_bytes=self.output_limits.keep_for(keep_bytes),
            kill_bytes=self.output_limits.kill_bytes,
            env=self._environment(),
            memory_guard=memory_guard,
            write_guard=write_guard,
        )
        return completed_process(run, timeout=timeout)

    def _inspect_root(self, name: str) -> Path | None:
        result = self._run(["inspect", "-p", name], timeout=120)
        if result.returncode != 0:
            return None
        value = result.stdout.strip().splitlines()
        if not value:
            return None
        root = Path(value[-1])
        return root if root.is_dir() else None

    def _ensure_seed(self) -> Path:
        """Return the image's seed container directory, creating it once."""
        with self._seed_lock():
            root = self._inspect_root(self.seed_name)
            if root is None:
                self._create_seed()
                root = self._inspect_root(self.seed_name)
                if root is None:
                    raise RuntimeError("udocker seed exists but has no root path")
            seed_dir = root.parent
            marker = seed_dir / "thundersync-image.txt"
            if not marker.is_file() or marker.read_text().strip() != self.image:
                raise RuntimeError(
                    f"udocker seed {self.seed_name!r} is not bound to {self.image!r}"
                )
            # A sandbox is a verbatim copy of the seed. That is what
            # ``udocker clone`` does for PRoot modes; for the Fakechroot
            # modes clone also re-patches absolute container paths, which a
            # copy does not. No execmode file means uDocker's default, which
            # the seed's own ``setup --execmode=P1`` found to be P1.
            execmode = seed_dir / "execmode"
            mode = execmode.read_text().strip() if execmode.is_file() else ""
            if mode and not mode.startswith("P"):
                raise RuntimeError(
                    f"udocker seed {self.seed_name!r} is in execution mode "
                    f"{mode!r}; sandbox copies are sound only for PRoot modes"
                )
            return seed_dir

    def _create_seed(self) -> None:
        """Extract and configure the image's seed container, once.

        ``create`` and ``setup`` rewrite the shared store, so seed creation
        holds the store lock; a seed's directory lacks ``container.json``
        until extraction finishes. The inspection is repeated under the lock:
        a transient inspect failure outside it must not recreate a seed that
        exists.
        """
        with self._store_lock():
            root = self._inspect_root(self.seed_name)
            if root is None:
                created = self._run(
                    ["create", f"--name={self.seed_name}", self.image], timeout=3600
                )
                if created.returncode != 0:
                    raise RuntimeError(
                        f"udocker image {self.image!r} is unavailable or could not "
                        f"be extracted: {(created.stdout + created.stderr)[-600:]}"
                    )
                configured = self._run(
                    ["setup", "--execmode=P1", self.seed_name], timeout=600
                )
                if configured.returncode != 0:
                    raise RuntimeError(
                        "udocker could not configure the P1 seed: "
                        f"{(configured.stdout + configured.stderr)[-600:]}"
                    )
                root = self._inspect_root(self.seed_name)
                if root is None:
                    raise RuntimeError("udocker seed exists but has no root path")
                (root.parent / "thundersync-image.txt").write_text(self.image + "\n")

    def _prepare_host_paths(self) -> None:
        if self.host_root is None:
            raise RuntimeError("udocker sandbox root is unresolved; start() registers it")
        source = self.host_root / self.workdir.lstrip("/")
        if not source.is_dir():
            raise RuntimeError(f"workdir {self.workdir} missing in {self.image}")
        ensure_private_dir(self.scratch_dir)
        ensure_private_dir(self.proot_tmp_dir)
        shutil.rmtree(self.host_workdir, ignore_errors=True)
        shutil.rmtree(self.host_tmpdir, ignore_errors=True)
        shutil.rmtree(self.host_runtime_dir, ignore_errors=True)
        self.host_workdir.mkdir(parents=True)
        self.host_tmpdir.mkdir(parents=True)
        self.host_runtime_dir.mkdir(parents=True)
        (self.host_runtime_dir / "hosts").write_text(
            "127.0.0.1 localhost\n::1 localhost\n"
        )
        (self.host_runtime_dir / "resolv.conf").write_text(
            "options attempts:0 timeout:1\n"
        )
        copied = subprocess.run(
            [
                "cp",
                "-a",
                "--reflink=auto",
                f"{source}/.",
                str(self.host_workdir),
            ],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if copied.returncode != 0:
            raise RuntimeError(
                f"could not seed udocker workdir: {copied.stderr[-400:]}"
            )

    def _publish_copy(self, seed_dir: Path) -> Path:
        """Copy the seed and rename the complete copy into the store.

        The copy is assembled under ``STAGING_DIRNAME`` and becomes a
        container only through the rename to ``containers/<container_name>``,
        so a failed or partial copy is never visible under the sandbox's name.
        No alias symlink is created: uDocker resolves the directory's own
        name, and every ``udocker run`` scans the aliases with an unguarded
        ``islink``/``readlink`` pair that an alias removed mid-scan would
        crash. A failed attempt removes its staging tree and is retried after
        a backoff; the last failure raises.
        """
        store = self.store_containers_dir
        published = store / self.container_name
        if os.path.lexists(published):
            raise RuntimeError(
                f"udocker sandbox {self.container_name!r} already exists"
            )
        detail = ""
        for attempt in range(_COPY_ATTEMPTS):
            if attempt:
                logger.warning(
                    "udocker copy of %s: attempt %d failed, retrying: %s",
                    self.container_name,
                    attempt,
                    detail[-300:],
                )
                time.sleep(2.0 * (2 ** (attempt - 1)))
            staging = (
                store / STAGING_DIRNAME / f"copy-{self.container_name}-{token_hex(4)}"
            )
            try:
                with self._shared_store_lock():
                    ensure_private_dir(staging.parent)
                    staging.mkdir()
                    copied = subprocess.run(
                        copy_tree_command(seed_dir, staging),
                        capture_output=True,
                        text=True,
                        timeout=3600,
                    )
                    if copied.returncode == 0:
                        os.rename(staging, published)
                        return published
                    detail = copied.stdout + copied.stderr
            except (OSError, subprocess.SubprocessError) as error:
                detail = str(error)
            remove_tree(staging)
        raise RuntimeError(
            f"udocker sandbox copy failed for {self.image}: {detail[-600:]}"
        )

    def _retract(self) -> None:
        """Rename the published copy out of the store, then delete its tree."""
        store = self.store_containers_dir
        published = store / self.container_name
        if published.is_symlink() or not published.is_dir():
            return
        trash = (
            store / STAGING_DIRNAME / f"trash-{self.container_name}-{token_hex(4)}"
        )
        try:
            with self._shared_store_lock():
                ensure_private_dir(trash.parent)
                os.rename(published, trash)
        except OSError:
            trash = published
        remove_tree(trash)

    def start(self) -> None:
        self.udocker_dir.mkdir(parents=True, exist_ok=True)
        timings: dict[str, float] = {}
        mark = time.perf_counter()

        def lap(phase: str) -> None:
            nonlocal mark
            now = time.perf_counter()
            timings[phase] = now - mark
            mark = now

        seed_dir = self._ensure_seed()
        lap("seed_s")
        try:
            published = self._publish_copy(seed_dir)
            lap("copy_s")
            # A no-op for a copy of a P1 seed; it converts any other PRoot
            # mode and fails closed on a copy uDocker cannot configure.
            configured = self._run(
                ["setup", "--execmode=P1", self.container_name], timeout=600
            )
            if configured.returncode != 0:
                raise RuntimeError(
                    f"udocker P1 setup failed: "
                    f"{(configured.stdout + configured.stderr)[-600:]}"
                )
            # uDocker must resolve the name to the published copy and parse
            # its metadata; anything else fails closed.
            self.host_root = self._inspect_root(self.container_name)
            if self.host_root is None or not _same_directory(
                self.host_root.parent, published
            ):
                raise RuntimeError(
                    f"udocker does not resolve {self.container_name!r} to its "
                    f"published copy (inspect gave {self.host_root})"
                )
            lap("register_s")
            self._prepare_host_paths()
            lap("prepare_s")
            self._started = True
            # The startup probes run unlocked, like every later exec: on
            # uDocker 1.3.17 a run writes only its own container's
            # ``.mountpoints`` and uniquely named temp files, and its helper
            # discovery only reads UDOCKER_DIR/bin. PROOT_TMP_DIR is private
            # per sandbox and UDOCKER_BIN is not forwarded (__init__,
            # _environment).
            probe = self.exec(f"test -d {shlex.quote(self.workdir)} && echo THUNDERSYNC_WORKDIR_OK")
            if "THUNDERSYNC_WORKDIR_OK" not in probe:
                raise RuntimeError(f"udocker workdir probe failed: {probe[:300]}")
            if self.verify_toolchain:
                probe = self.exec(
                    "if python -m pytest --version >/dev/null 2>&1; then "
                    "echo THUNDERSYNC_TOOLCHAIN=pytest; "
                    "elif command -v go >/dev/null 2>&1 && "
                    "go version >/dev/null 2>&1; "
                    "then echo THUNDERSYNC_TOOLCHAIN=go; fi"
                )
                if "THUNDERSYNC_TOOLCHAIN=" not in probe:
                    raise RuntimeError(
                        f"no supported pytest or Go verifier in "
                        f"{self.image}: {probe[:300]}"
                    )
            lap("probe_s")
        except Exception:
            self.close()
            raise
        self.start_timings = timings

    def _exec_command(self, command: str) -> list[str]:
        args = [
            "run",
            "--nobanner",
            "--nosysdirs",
            f"--workdir={self.workdir}",
            "--env=HOME=/root",
            f"--volume={self.host_workdir}:{self.workdir}",
            f"--volume={self.host_tmpdir}:/tmp",
            f"--volume={self.host_runtime_dir / 'hosts'}:/etc/hosts",
            f"--volume={self.host_runtime_dir / 'resolv.conf'}:/etc/resolv.conf",
            "--volume=/dev/null:/dev/null",
            "--volume=/dev/zero:/dev/zero",
            "--volume=/dev/random:/dev/random",
            "--volume=/dev/urandom:/dev/urandom",
            self.container_name,
        ]
        # The host-side ``run_bounded`` deadline in ``_run`` is the
        # authoritative deadline.  Do not put an image-provided ``timeout``
        # in front of the command: some SWE-Smith images contain a binary
        # that PRoot cannot execute (even when the path is present), which
        # otherwise rejects every verifier command before it starts.
        # Keep the command prefix to the shell itself.  Minimal task images
        # can also omit ``env``; exporting HOME in the shell preserves the
        # same behavior without adding another image binary dependency.
        # Use the absolute image path. PRoot's command lookup can
        # intermittently reject the bare ``bash`` name when many cloned
        # sandboxes start concurrently even though /bin/bash exists and the
        # same image passed prewarm.
        args.extend(["/bin/bash", "-lc", f"export HOME=/root; {command}"])
        return args

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
        environment = (
            "if [ -x /usr/local/go/bin/go ]; then "
            "export PATH=/go/bin:/usr/local/go/bin:$PATH "
            "GOROOT=/usr/local/go GOPATH=/go GOTOOLCHAIN=local "
            "GOTELEMETRY=off; fi; "
        )
        try:
            result = self._run(
                self._exec_command(environment + command),
                timeout=self.timeout + 45,
                network_guard=self.isolate_network and not self.requires_network,
                keep_bytes=keep_bytes,
                memory_guard=memory_guard,
                write_guard=write_guard,
            )
        except subprocess.TimeoutExpired as error:
            return CommandOutput.timed_out_after(
                self.timeout, getattr(error, "bounded_run", None)
            )
        output = CommandOutput.from_process(result, timeout_s=self.timeout)
        if output.killed is not None:
            report_kill(self.name, command, output)
        return output

    def stop_commands(self) -> None:
        """Nothing outlives an exec: its process group is killed when it ends."""

    def close(self) -> None:
        # Synchronous: a pool frees the member's slot only after close()
        # returns, which is what bounds the store's footprint. No `udocker rm`:
        # it rescans every alias with the same unguarded readlink as `run`.
        self._retract()
        shutil.rmtree(self.host_workdir, ignore_errors=True)
        shutil.rmtree(self.host_tmpdir, ignore_errors=True)
        shutil.rmtree(self.host_runtime_dir, ignore_errors=True)
        trusted_root = getattr(self, "_thundersync_trusted_root", None)
        if trusted_root is not None:
            shutil.rmtree(trusted_root, ignore_errors=True)
            self._thundersync_trusted_root = None
        self.host_root = None
        self._started = False

    def __enter__(self) -> "UDockerSandbox":
        self.start()
        return self
