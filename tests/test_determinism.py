"""The trainer's determinism setting: what each value applies and what a rank records.

The parsing and the refusals need no torch. The settings themselves are
torch's process state, so those tests import torch when they run (skipped
where there is no real torch) and restore the state they found.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

from thundersync import determinism  # noqa: E402


def test_a_setting_that_is_not_a_bool_is_refused_before_torch_is_touched() -> None:
    for value in ("on", 1, None):
        with pytest.raises(TypeError, match="must be a bool"):
            determinism.apply_trainer_determinism(value)


def _torch_is_real() -> bool:
    """Another test may have installed a stub under the torch name that
    answers every attribute; the stub has no import spec."""
    try:
        spec = importlib.util.find_spec("torch")
    except ValueError:  # sys.modules["torch"].__spec__ is None: the stub
        return False
    return spec is not None and spec.origin is not None


@pytest.fixture
def torch_state():
    if not _torch_is_real():
        pytest.skip("torch's process state needs a real torch")
    import torch
    import torch.utils.deterministic

    saved = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.utils.deterministic.fill_uninitialized_memory,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    yield torch
    torch.use_deterministic_algorithms(saved[0], warn_only=saved[1])
    torch.utils.deterministic.fill_uninitialized_memory = saved[2]
    torch.backends.cudnn.deterministic = saved[3]
    torch.backends.cudnn.benchmark = saved[4]


def test_on_is_exactly_what_the_trainer_always_ran_under(torch_state) -> None:
    torch = torch_state
    torch.backends.cudnn.benchmark = True
    applied = determinism.apply_trainer_determinism(True)
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.utils.deterministic.fill_uninitialized_memory is False
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert applied == determinism.trainer_determinism() == {
        "deterministic_algorithms": True,
        "warn_only": False,
        "fill_uninitialized_memory": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }


def test_off_drops_the_deterministic_kernels_and_keeps_autotuning_off(torch_state) -> None:
    torch = torch_state
    determinism.apply_trainer_determinism(True)
    applied = determinism.apply_trainer_determinism(False)
    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is False
    assert applied == {
        "deterministic_algorithms": False,
        "warn_only": False,
        "fill_uninitialized_memory": False,
        "cudnn_deterministic": False,
        "cudnn_benchmark": False,
    }


class _ModelLoadReached(Exception):
    pass


@pytest.mark.parametrize("enabled", [True, False])
def test_load_trainer_applies_the_configured_determinism_before_the_model_loads(
    torch_state, monkeypatch, enabled
) -> None:
    from types import SimpleNamespace

    from thundersync.grpo import runtime
    from thundersync.grpo.config import GrpoTrainerConfig

    determinism.apply_trainer_determinism(not enabled)
    seen = []

    def load_policy_model(*_args, **_kwargs):
        seen.append(determinism.trainer_determinism()["deterministic_algorithms"])
        raise _ModelLoadReached

    monkeypatch.setattr(runtime, "load_policy_model", load_policy_model)
    config = GrpoTrainerConfig.from_namespace(
        SimpleNamespace(
            model_path="/m",
            learning_rate=1e-5,
            rebuild_block_tokens=1024,
            deterministic_algorithms=enabled,
        )
    )
    with pytest.raises(_ModelLoadReached):
        runtime.load_trainer(config, torch_state.device("cpu"))
    assert seen == [enabled]
