import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]

torch = pytest.importorskip("torch")

from thundersync.accel import capability  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_probes():
    capability.reset_kernel_probes()
    yield
    capability.reset_kernel_probes()


@pytest.fixture
def fake_cuda(monkeypatch):
    """Let the trial machinery run without a device."""

    monkeypatch.setattr(capability.torch.cuda, "device", lambda _: contextlib.nullcontext())
    monkeypatch.setattr(capability.torch.cuda, "synchronize", lambda _=None: None)
    monkeypatch.setattr(capability, "device_identity", lambda _=None: {"name": "stub"})
    return SimpleNamespace(
        is_cuda=True, device=SimpleNamespace(index=0), dtype=torch.bfloat16
    )


def test_a_cpu_tensor_never_runs_an_optional_kernel() -> None:
    calls = []
    assert not capability.kernel_runs("any", torch.zeros(1), 64, lambda *a: calls.append(a))
    assert calls == []


def test_each_key_is_tried_once_and_cached(fake_cuda) -> None:
    calls = []

    def trial(device, dtype, head_dim):
        calls.append(head_dim)

    assert capability.kernel_runs("k", fake_cuda, 64, trial)
    assert capability.kernel_runs("k", fake_cuda, 64, trial)
    assert capability.kernel_runs("k", fake_cuda, 128, trial)
    assert calls == [64, 128]


def test_a_failing_trial_reports_false_and_records_why(fake_cuda) -> None:
    def trial(device, dtype, head_dim):
        raise RuntimeError("no kernel image is available for execution on the device")

    assert not capability.kernel_runs("k", fake_cuda, 256, trial)
    (record,) = capability.kernel_probe_records()
    assert record["kernel"] == "k" and record["runs"] is False
    assert record["shape_key"] == 256 and record["dtype"] == "torch.bfloat16"
    assert "no kernel image" in record["error"]
    assert record["device"] == {"name": "stub"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_the_device_identity_names_the_capability() -> None:
    identity = capability.device_identity(0)
    major, minor = torch.cuda.get_device_capability(0)
    assert identity["compute_capability"] == f"{major}.{minor}"
    assert identity["total_memory_mib"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_every_attention_path_is_probed_or_declined_on_this_device() -> None:
    from thundersync.engine import streaming

    query = torch.randn(1, 2, 8, 128, device="cuda", dtype=torch.bfloat16)
    streaming._fa4_bottom_right_is_available(query, 8)
    streaming._split_bottom_right_flash_is_available(query, 8)
    streaming._split_bottom_right_is_available(query, 8)
    streaming._ragged_attention_runs(query, 128)
    probed = {record["kernel"] for record in capability.kernel_probe_records()}
    # FA4 is probed only when its build imports; the torch paths always are.
    assert {"cudnn_split", "ragged_flash"} <= probed or not streaming._CUDNN_SDPA
    for record in capability.kernel_probe_records():
        assert record["device"]["compute_capability"]
