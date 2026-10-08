"""A stateful stand-in for the ``docker`` CLI, for hosts without a daemon.

``install(tmp_path, monkeypatch)`` puts an executable named ``docker`` first
on PATH. It records every argv as a JSON line and emulates what the sandbox
code calls: ``version``, ``image inspect``, ``pull``, ``create``, ``cp``,
``run -d``, ``exec``, ``rm -f`` and ``ps -aq --filter``. An ``exec`` runs
its command on the host in the bind-mounted directory that backs the
container path, with the ``/tmp`` mount substituted, so bug sealing,
trusted capture and the pool's pristine gate run end to end.

Commands that must never reach the host are intercepted: ``kill -9 -1``
and ``chown -R`` are recorded and answered, and the container Git's
``--system`` configuration becomes a no-op.

Environment: ``FAKE_DOCKER_STATE`` (state directory), ``FAKE_DOCKER_DAEMON``
(``down`` makes the daemon unreachable), ``FAKE_DOCKER_IMAGES`` (comma list
present in the daemon), ``FAKE_DOCKER_PULLABLE`` (comma list a pull adds),
``FAKE_DOCKER_IMAGE_ROOT`` (per-image filesystem trees for ``cp``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SERVER_VERSION = "27.0.1-fake"


def image_directory(image: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", image)


def install(tmp_path: Path, monkeypatch, *, images=(), pullable=(), daemon="up") -> Path:
    """Install the fake on PATH; returns the state directory."""
    bin_dir = tmp_path / "fake-docker-bin"
    bin_dir.mkdir(exist_ok=True)
    state = tmp_path / "fake-docker-state"
    state.mkdir(exist_ok=True)
    script = bin_dir / "docker"
    script.write_text(
        f"#!/bin/sh\nexec {sys.executable} {Path(__file__).resolve()} \"$@\"\n"
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_DOCKER_STATE", str(state))
    monkeypatch.setenv("FAKE_DOCKER_DAEMON", daemon)
    monkeypatch.setenv("FAKE_DOCKER_IMAGES", ",".join(images))
    monkeypatch.setenv("FAKE_DOCKER_PULLABLE", ",".join(pullable))
    monkeypatch.setenv("FAKE_DOCKER_IMAGE_ROOT", str(tmp_path / "fake-docker-images"))
    return state


def calls(state: Path) -> list[list[str]]:
    log = state / "calls.jsonl"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line]


# ------------------------------------------------------------- emulation


def _state() -> Path:
    return Path(os.environ["FAKE_DOCKER_STATE"])


def _images() -> set[str]:
    present = {item for item in os.environ.get("FAKE_DOCKER_IMAGES", "").split(",") if item}
    pulled = _state() / "pulled.txt"
    if pulled.exists():
        present |= set(pulled.read_text().split())
    return present


def _containers() -> Path:
    path = _state() / "containers"
    path.mkdir(exist_ok=True)
    return path


def _load(name: str) -> dict | None:
    path = _containers() / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _save(name: str, record: dict) -> None:
    (_containers() / f"{name}.json").write_text(json.dumps(record))


def _options(args: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    """Split ``--flag value`` pairs (and ``-d``) from positionals."""
    flags: dict[str, list[str]] = {}
    rest: list[str] = []
    index = 0
    valued = {
        "--name", "--label", "--network", "--memory", "--cpus", "--pids-limit", "--security-opt",
        "-v", "-w", "-e", "--entrypoint", "--format", "--filter",
    }
    while index < len(args):
        item = args[index]
        if item in valued:
            flags.setdefault(item, []).append(args[index + 1])
            index += 2
        elif item.startswith("-") and not rest:
            flags.setdefault(item, []).append("")
            index += 1
        else:
            rest.append(item)
            index += 1
    return flags, rest


def _create(args: list[str], *, running: bool) -> int:
    flags, rest = _options(args)
    image = rest[0]
    if image not in _images():
        print(f"Unable to find image '{image}' locally", file=sys.stderr)
        return 125
    if "--rm" in flags:
        # A one-shot container: nothing persists, and its command (a chown
        # of a host tree) is not run on the host.
        return 0
    name = flags["--name"][0]
    if _load(name) is not None:
        print(f"Conflict. The container name \"/{name}\" is already in use", file=sys.stderr)
        return 125
    mounts = {}
    for spec in flags.get("-v", []):
        host, container = spec.split(":", 1)
        mounts[container] = host
    _save(
        name,
        {
            "image": image,
            "labels": flags.get("--label", []),
            "mounts": mounts,
            "workdir": (flags.get("-w") or ["/"])[0],
            "running": running,
            "flags": flags,
        },
    )
    if running:
        print("0" * 64)
    return 0


def _cp(args: list[str]) -> int:
    source, destination = args
    name, path = source.split(":", 1)
    record = _load(name)
    if record is None:
        print(f"No such container: {name}", file=sys.stderr)
        return 1
    root = Path(os.environ["FAKE_DOCKER_IMAGE_ROOT"]) / image_directory(record["image"])
    tree = root / path.lstrip("/").removesuffix("/.")
    if not tree.is_dir():
        print(f"Could not find the file {path} in container {name}", file=sys.stderr)
        return 1
    shutil.copytree(tree, destination, symlinks=True, dirs_exist_ok=True)
    return 0


def _exec(args: list[str]) -> int:
    flags, rest = _options(args)
    name, argv = rest[0], rest[1:]
    record = _load(name)
    if record is None or not record["running"]:
        print(f"Error response from daemon: container {name} is not running", file=sys.stderr)
        return 1
    command = argv[-1]
    if "kill -9 -1" in command:
        (_state() / "killed.txt").open("a").write(f"{name}\n")
        return 0
    if command.startswith("chown -R"):
        (_state() / "chowned.txt").open("a").write(f"{name}\n")
        return 0
    command = command.replace("git config --system --add safe.directory '*'", "true")
    # The container's HOME is /root; on the host that is another user's
    # directory, and git warns on stderr when it cannot read it.
    home = _state() / "home"
    home.mkdir(exist_ok=True)
    command = command.replace("HOME=/root", f"HOME={home}")
    # One pass over every mount: a host path substituted for one mount must
    # not be rewritten again by another (host paths live under /tmp on Linux).
    mounts = record["mounts"]
    if mounts:
        alternatives = "|".join(
            re.escape(path) for path in sorted(mounts, key=len, reverse=True)
        )
        command = re.sub(
            rf"(?<![\w/.])({alternatives})(?=[/\s;'\"]|$)",
            lambda match: mounts[match.group(1)],
            command,
        )
    workdir = (flags.get("-w") or [record["workdir"]])[0]
    cwd = record["mounts"].get(workdir)
    if cwd is None:
        print(f"no mount backs {workdir}", file=sys.stderr)
        return 126
    completed = subprocess.run(["/bin/bash", "-c", command], cwd=cwd)
    return completed.returncode


def _rm(args: list[str]) -> int:
    for name in (item for item in args if not item.startswith("-")):
        path = _containers() / f"{name}.json"
        if path.exists():
            path.unlink()
    return 0


def _ps(args: list[str]) -> int:
    flags, _ = _options(args)
    wanted = [item.removeprefix("label=") for item in flags.get("--filter", [])]
    for path in sorted(_containers().glob("*.json")):
        record = json.loads(path.read_text())
        if all(label in record["labels"] for label in wanted):
            print(path.stem)
    return 0


def main(argv: list[str]) -> int:
    with (_state() / "calls.jsonl").open("a") as log:
        log.write(json.dumps(argv) + "\n")
    if os.environ.get("FAKE_DOCKER_DAEMON") == "down":
        if argv[:1] == ["version"]:
            print("Client:\n Version: fake")
        print(
            "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. "
            "Is the docker daemon running?",
            file=sys.stderr,
        )
        return 1
    command, args = argv[0], argv[1:]
    if command == "version":
        print(SERVER_VERSION)
        return 0
    if command == "image" and args[:1] == ["inspect"]:
        image = args[-1]
        if image not in _images():
            print(f"Error: No such image: {image}", file=sys.stderr)
            return 1
        print("sha256:" + hashlib.sha256(image.encode()).hexdigest())
        return 0
    if command == "pull":
        image = args[-1]
        pullable = os.environ.get("FAKE_DOCKER_PULLABLE", "").split(",")
        if image not in pullable:
            print(f"Error response from daemon: pull access denied for {image}", file=sys.stderr)
            return 1
        with (_state() / "pulled.txt").open("a") as handle:
            handle.write(image + "\n")
        return 0
    if command == "create":
        return _create(args, running=False)
    if command == "run":
        return _create(args, running=True)
    if command == "cp":
        return _cp(args)
    if command == "exec":
        return _exec(args)
    if command == "rm":
        return _rm(args)
    if command == "ps":
        return _ps(args)
    print(f"fake docker: unsupported command {argv}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
