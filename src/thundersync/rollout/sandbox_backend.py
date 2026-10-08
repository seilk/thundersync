"""Which container runtime a site's sandboxes run on.

The rule: Docker when the host's daemon answers this user, udocker
otherwise. ``auto`` applies the rule; an explicit ``docker`` or ``udocker``
request is honored only when that runtime is usable and is refused
otherwise, never swapped for the other one. The backend is a site property:
resolve it once and hand the resolved name to every process, so all of them
run the same backend.

``python -m thundersync.rollout.sandbox_backend --requested auto`` prints the
resolution record as JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

RECORD_SCHEMA = "thundersync-sandbox-backend-v1"
BACKENDS = ("docker", "udocker")
REQUESTS = ("auto", *BACKENDS)
DEFAULT_PROBE_TIMEOUT_S = 15.0


class SandboxBackendError(RuntimeError):
    """The requested sandbox backend cannot run on this host."""


@dataclass(frozen=True, slots=True)
class Probe:
    """One runtime check: whether it passed, the version, or why it failed."""

    usable: bool
    version: str | None
    detail: str

    def as_record(self) -> dict[str, Any]:
        return {"usable": self.usable, "version": self.version, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class ResolvedSandboxBackend:
    requested: str
    backend: str
    decided_by: str
    docker: Probe | None
    udocker: Probe | None

    @property
    def version(self) -> str | None:
        probe = self.docker if self.backend == "docker" else self.udocker
        return probe.version if probe is not None else None

    def as_record(self) -> dict[str, Any]:
        return {
            "schema": RECORD_SCHEMA,
            "requested": self.requested,
            "backend": self.backend,
            "decided_by": self.decided_by,
            "version": self.version,
            "docker": self.docker.as_record() if self.docker is not None else None,
            "udocker": self.udocker.as_record() if self.udocker is not None else None,
        }


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _resolve_executable(bin_path: str) -> str | None:
    found = shutil.which(bin_path)
    if found is not None:
        return found
    path = Path(bin_path)
    return str(path) if path.is_file() and os.access(path, os.X_OK) else None


def _call(
    runner: Runner, argv: list[str], timeout_s: float, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str] | str:
    try:
        if env is None:
            return runner(argv, capture_output=True, text=True, timeout=timeout_s)
        return runner(argv, capture_output=True, text=True, timeout=timeout_s, env=env)
    except subprocess.TimeoutExpired:
        return f"`{' '.join(argv)}` did not answer within {timeout_s:g} s"
    except OSError as error:
        return f"`{' '.join(argv)}` could not start: {error}"


def probe_docker(
    bin_path: str = "docker",
    *,
    timeout_s: float = DEFAULT_PROBE_TIMEOUT_S,
    runner: Runner = subprocess.run,
) -> Probe:
    """Does the Docker daemon answer this user?

    ``docker version`` with a server template fails when the CLI is present
    but the daemon is absent, stopped, or refuses this user's socket access,
    which is exactly the condition that must fall back to udocker.
    """

    executable = _resolve_executable(bin_path)
    if executable is None:
        return Probe(False, None, f"no {bin_path!r} executable on PATH")
    result = _call(
        runner, [executable, "version", "--format", "{{.Server.Version}}"], timeout_s
    )
    if isinstance(result, str):
        return Probe(False, None, result)
    version = result.stdout.strip()
    if result.returncode != 0 or not version:
        reason = (result.stderr or result.stdout).strip().splitlines()
        return Probe(
            False,
            None,
            f"docker daemon unusable (exit {result.returncode}): "
            f"{reason[-1][:300] if reason else 'no server version'}",
        )
    return Probe(True, version.splitlines()[-1], f"server {executable}")


def probe_udocker(
    bin_path: str,
    *,
    timeout_s: float = DEFAULT_PROBE_TIMEOUT_S,
    runner: Runner = subprocess.run,
) -> Probe:
    executable = _resolve_executable(bin_path)
    if executable is None:
        return Probe(False, None, f"no udocker executable at {bin_path!r}")
    # udocker initializes whatever UDOCKER_DIR names on first use; the probe
    # runs against a throwaway directory so it never leaves a skeleton where
    # the real store is to be staged.
    with tempfile.TemporaryDirectory(prefix="udocker-probe-") as scratch:
        env = {**os.environ, "UDOCKER_DIR": scratch}
        result = _call(runner, [executable, "version"], timeout_s, env=env)
    if isinstance(result, str):
        return Probe(False, None, result)
    if result.returncode != 0:
        reason = (result.stderr or result.stdout).strip().splitlines()
        return Probe(
            False,
            None,
            f"udocker unusable (exit {result.returncode}): "
            f"{reason[-1][:300] if reason else 'no output'}",
        )
    version = next(
        (
            line.split(":", 1)[1].strip()
            for line in result.stdout.splitlines()
            if line.lower().startswith("version")
        ),
        result.stdout.strip().splitlines()[0] if result.stdout.strip() else "unknown",
    )
    return Probe(True, version, f"executable {executable}")


def resolve_sandbox_backend(
    requested: str,
    *,
    docker_bin: str | None = None,
    udocker_bin: str | None = None,
    timeout_s: float = DEFAULT_PROBE_TIMEOUT_S,
    runner: Runner = subprocess.run,
) -> ResolvedSandboxBackend:
    """Apply the site rule; refuse rather than substitute.

    ``docker_bin`` and ``udocker_bin`` default to ``THUNDERSYNC_DOCKER_BIN``
    and ``THUNDERSYNC_UDOCKER_BIN``, else ``docker`` and ``udocker``.
    """

    if requested not in REQUESTS:
        raise SandboxBackendError(
            f"sandbox backend {requested!r} is not one of {', '.join(REQUESTS)}"
        )
    docker_bin = docker_bin or os.environ.get("THUNDERSYNC_DOCKER_BIN", "docker")
    udocker_bin = udocker_bin or os.environ.get("THUNDERSYNC_UDOCKER_BIN", "udocker")
    if requested == "docker":
        docker = probe_docker(docker_bin, timeout_s=timeout_s, runner=runner)
        if not docker.usable:
            raise SandboxBackendError(
                f"sandbox backend docker was requested and is not usable: {docker.detail}"
            )
        return ResolvedSandboxBackend(requested, "docker", "explicit_request", docker, None)
    if requested == "udocker":
        udocker = probe_udocker(udocker_bin, timeout_s=timeout_s, runner=runner)
        if not udocker.usable:
            raise SandboxBackendError(
                f"sandbox backend udocker was requested and is not usable: {udocker.detail}"
            )
        return ResolvedSandboxBackend(requested, "udocker", "explicit_request", None, udocker)
    docker = probe_docker(docker_bin, timeout_s=timeout_s, runner=runner)
    if docker.usable:
        return ResolvedSandboxBackend(
            requested, "docker", "auto_docker_daemon_usable", docker, None
        )
    udocker = probe_udocker(udocker_bin, timeout_s=timeout_s, runner=runner)
    if udocker.usable:
        return ResolvedSandboxBackend(
            requested, "udocker", "auto_docker_unusable_udocker_available", docker, udocker
        )
    raise SandboxBackendError(
        "no sandbox backend is usable: "
        f"docker: {docker.detail}; udocker: {udocker.detail}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--requested", choices=REQUESTS, required=True)
    parser.add_argument("--docker-bin", default=None)
    parser.add_argument("--udocker-bin", default=None)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_PROBE_TIMEOUT_S)
    args = parser.parse_args(argv)
    try:
        resolved = resolve_sandbox_backend(
            args.requested,
            docker_bin=args.docker_bin,
            udocker_bin=args.udocker_bin,
            timeout_s=args.timeout_s,
        )
    except SandboxBackendError as error:
        print(f"sandbox backend: {error}", file=sys.stderr)
        return 2
    print(json.dumps(resolved.as_record(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
