from __future__ import annotations

import ctypes.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


needs_seccomp = pytest.mark.skipif(
    not sys.platform.startswith("linux") or ctypes.util.find_library("seccomp") is None,
    reason="requires Linux and libseccomp",
)


@needs_seccomp
def test_guard_denies_inet_and_preserves_unix_sockets():
    program = """
import socket

unix = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
unix.close()
print("AF_UNIX_OK")
try:
    socket.socket(socket.AF_INET, socket.SOCK_STREAM)
except PermissionError:
    print("AF_INET_BLOCKED")
else:
    raise SystemExit("AF_INET socket unexpectedly opened")
"""
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "thundersync.rollout.network_guard",
            sys.executable,
            "-c",
            program,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["AF_UNIX_OK", "AF_INET_BLOCKED"]
