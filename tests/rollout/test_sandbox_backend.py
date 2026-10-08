"""Sandbox backend resolution: docker when usable, udocker otherwise.

Failure modes each test guards:

* ``auto`` choosing udocker on a host whose daemon answers, or docker on a
  host whose CLI exists but whose daemon does not;
* an explicit request silently swapped for the other runtime;
* a refusal that does not say which checks failed;
* a record missing the facts a run records.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


from thundersync.rollout import sandbox_backend as sb  # noqa: E402
from tests.rollout import fake_docker  # noqa: E402


@pytest.fixture
def udocker_bin(tmp_path: Path) -> str:
    path = tmp_path / "udocker"
    path.write_text("#!/bin/sh\necho 'version: 1.3.17'\necho 'repository: example'\n")
    path.chmod(0o755)
    return str(path)


def test_auto_picks_docker_when_the_daemon_answers(tmp_path, monkeypatch, udocker_bin):
    fake_docker.install(tmp_path, monkeypatch)
    resolved = sb.resolve_sandbox_backend("auto", udocker_bin=udocker_bin)
    assert resolved.backend == "docker"
    assert resolved.decided_by == "auto_docker_daemon_usable"
    assert resolved.version == fake_docker.SERVER_VERSION
    assert resolved.udocker is None


def test_auto_falls_back_to_udocker_when_the_daemon_is_down(tmp_path, monkeypatch, udocker_bin):
    fake_docker.install(tmp_path, monkeypatch, daemon="down")
    resolved = sb.resolve_sandbox_backend("auto", udocker_bin=udocker_bin)
    assert resolved.backend == "udocker"
    assert resolved.decided_by == "auto_docker_unusable_udocker_available"
    assert resolved.version == "1.3.17"
    record = resolved.as_record()
    assert record["docker"]["usable"] is False
    assert "Cannot connect" in record["docker"]["detail"]


def test_auto_falls_back_when_no_docker_cli_exists(tmp_path, monkeypatch, udocker_bin):
    resolved = sb.resolve_sandbox_backend(
        "auto", docker_bin=str(tmp_path / "absent-docker"), udocker_bin=udocker_bin
    )
    assert resolved.backend == "udocker"


def test_auto_refuses_when_neither_runtime_is_usable(tmp_path, monkeypatch):
    fake_docker.install(tmp_path, monkeypatch, daemon="down")
    with pytest.raises(sb.SandboxBackendError) as caught:
        sb.resolve_sandbox_backend("auto", udocker_bin=str(tmp_path / "no-udocker"))
    message = str(caught.value)
    assert "docker:" in message and "udocker:" in message


def test_an_explicit_docker_request_is_refused_not_swapped(tmp_path, monkeypatch, udocker_bin):
    fake_docker.install(tmp_path, monkeypatch, daemon="down")
    with pytest.raises(sb.SandboxBackendError, match="docker was requested"):
        sb.resolve_sandbox_backend("docker", udocker_bin=udocker_bin)


def test_an_explicit_udocker_request_is_refused_not_swapped(tmp_path, monkeypatch):
    fake_docker.install(tmp_path, monkeypatch)
    with pytest.raises(sb.SandboxBackendError, match="udocker was requested"):
        sb.resolve_sandbox_backend("udocker", udocker_bin=str(tmp_path / "no-udocker"))


def test_an_explicit_request_that_is_usable_is_honored(tmp_path, monkeypatch, udocker_bin):
    fake_docker.install(tmp_path, monkeypatch)
    assert sb.resolve_sandbox_backend("udocker", udocker_bin=udocker_bin).backend == "udocker"
    resolved = sb.resolve_sandbox_backend("docker", udocker_bin=udocker_bin)
    assert (resolved.backend, resolved.decided_by) == ("docker", "explicit_request")


def test_an_unknown_request_is_refused():
    with pytest.raises(sb.SandboxBackendError, match="not one of"):
        sb.resolve_sandbox_backend("podman")


def test_a_hung_daemon_is_unusable_within_the_timeout(tmp_path, udocker_bin):
    hung = tmp_path / "docker"
    hung.write_text("#!/bin/sh\nsleep 30\n")
    hung.chmod(0o755)
    resolved = sb.resolve_sandbox_backend(
        "auto", docker_bin=str(hung), udocker_bin=udocker_bin, timeout_s=0.5
    )
    assert resolved.backend == "udocker"
    assert "did not answer" in resolved.docker.detail


def test_the_cli_prints_the_record(tmp_path, monkeypatch, capsys, udocker_bin):
    fake_docker.install(tmp_path, monkeypatch)
    assert sb.main(["--requested", "auto", "--udocker-bin", udocker_bin]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["schema"] == sb.RECORD_SCHEMA
    assert (record["requested"], record["backend"]) == ("auto", "docker")


def test_the_cli_refuses_with_a_nonzero_status(tmp_path, monkeypatch, capsys):
    fake_docker.install(tmp_path, monkeypatch, daemon="down")
    assert sb.main(["--requested", "docker"]) == 2
    assert "not usable" in capsys.readouterr().err


def test_the_udocker_probe_never_touches_the_configured_store(tmp_path, monkeypatch):
    """udocker creates its tree wherever UDOCKER_DIR points on first use; a
    probe run against the real store would leave a skeleton there."""
    store = tmp_path / "udocker-store-v1"
    seen = tmp_path / "seen"
    probe = tmp_path / "udocker"
    probe.write_text(
        "#!/bin/sh\nmkdir -p \"$UDOCKER_DIR/containers\"\necho \"$UDOCKER_DIR\" > " + str(seen)
        + "\necho 'version: 1.3.17'\n"
    )
    probe.chmod(0o755)
    monkeypatch.setenv("UDOCKER_DIR", str(store))
    result = sb.probe_udocker(str(probe))
    assert result.usable and result.version == "1.3.17"
    assert not store.exists()
    used = Path(seen.read_text().strip())
    assert used != store and not used.exists()
