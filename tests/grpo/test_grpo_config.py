"""The GRPO row's typed trainer configuration (no torch).

``GrpoTrainerConfig.from_namespace`` is where a GRPO entry point's parsed
arguments meet the trainer builder; these pin that it fails there, by
name, and that its defaults are the ones the builder always carried.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]

from thundersync.grpo.config import GrpoTrainerConfig  # noqa: E402
from thundersync.config import ConfigError  # noqa: E402


TRAINER_REQUIRED = {
    "model_path": Path("/models/snapshot"),
    "learning_rate": 1e-5,
    "rebuild_block_tokens": 1024,
}


def _trainer_namespace(**overrides: Any) -> SimpleNamespace:
    fields = {**TRAINER_REQUIRED, **overrides}
    return SimpleNamespace(**fields)


def test_a_minimal_namespace_becomes_a_grpo_trainer_config() -> None:
    config = GrpoTrainerConfig.from_namespace(_trainer_namespace())
    assert config.policy.model_path == Path("/models/snapshot")
    assert config.policy.learning_rate == 1e-5
    assert config.streaming.rebuild_block_tokens == 1024


def test_the_advantage_epsilon_defaults_to_the_objective_default() -> None:
    from thundersync.grpo.config import DEFAULT_GRPO_EPSILON

    assert DEFAULT_GRPO_EPSILON == 1e-6
    assert GrpoTrainerConfig.from_namespace(_trainer_namespace()).grpo_epsilon == 1e-6
    assert (
        GrpoTrainerConfig.from_namespace(_trainer_namespace(grpo_epsilon=1e-4)).grpo_epsilon
        == 1e-4
    )
    # The objective and its data-parallel trainer default to the same value;
    # read from source so this test stays free of torch.
    for module in ("reward_linear.py", "data_parallel.py"):
        source = (ROOT / "src" / "thundersync" / "grpo" / module).read_text()
        assert "eps: float = DEFAULT_GRPO_EPSILON," in source


@pytest.mark.parametrize("name", sorted(TRAINER_REQUIRED))
def test_a_missing_trainer_field_is_refused_by_name(name: str) -> None:
    fields = dict(TRAINER_REQUIRED)
    del fields[name]
    with pytest.raises(ConfigError, match=name):
        GrpoTrainerConfig.from_namespace(SimpleNamespace(**fields))


def test_the_trainer_defaults_are_the_ones_the_builder_carried() -> None:
    """An entry point that defines none of the optional flags must build
    the trainer the probing builder built, value for value."""

    config = GrpoTrainerConfig.from_namespace(_trainer_namespace())
    assert config.policy.disable_fla_causal_conv is False
    assert config.policy.lora_rank == 0
    assert config.policy.lora_alpha is None
    assert config.policy.fused_lora is False
    assert config.policy.compile_pointwise_modules is False
    assert config.policy.optimizer_state_sharding is False
    assert config.policy.grad_clip == 1.0
    assert config.streaming.rebuild_plain_reserve_gib == 0.0
    assert config.streaming.gradient_accumulation_offload_device is None
    assert config.streaming.source_offload_device is None
    assert config.streaming.source_offload_mode == "blocking"
    assert config.streaming.source_offload_window_mib == 256
    assert config.streaming.source_offload_hbm_threshold == 0.65
    assert config.streaming.source_offload_pinned_windows == 2
    assert config.streaming.source_reduction_mode == "unrestricted_statistics"
    assert config.streaming.source_reduction_pack_size == 1
    assert config.streaming.source_reduction_worker_count == 2
    assert config.streaming.source_offload_worker_count == 2
    assert config.streaming.source_spill_directory is None
    assert config.streaming.statistic_device is None
    assert config.streaming.statistic_dtype == "fp32"
    assert config.policy.deterministic_algorithms is True
    assert config.streaming.direct_when_group_bound is False


def test_the_trainer_determinism_is_a_bool_the_entry_point_may_turn_off() -> None:
    """Absent, the trainer is deterministic, as it always was; the entry
    point's parsed bool passes through, and nothing else is coerced (the
    string "off" would read as true)."""

    off = GrpoTrainerConfig.from_namespace(_trainer_namespace(deterministic_algorithms=False))
    assert off.policy.deterministic_algorithms is False
    for value in ("off", 0, None):
        with pytest.raises(ConfigError, match="deterministic_algorithms"):
            GrpoTrainerConfig.from_namespace(_trainer_namespace(deterministic_algorithms=value))



def test_the_k0_direct_path_is_a_bool_the_entry_point_may_turn_on() -> None:
    on = GrpoTrainerConfig.from_namespace(_trainer_namespace(direct_when_group_bound=True))
    assert on.streaming.direct_when_group_bound is True
    for value in ("on", 1, None):
        with pytest.raises(ConfigError, match="direct_when_group_bound"):
            GrpoTrainerConfig.from_namespace(_trainer_namespace(direct_when_group_bound=value))


def test_the_pinned_arena_is_sized_from_the_residency_gates() -> None:
    """The slab count must hold every source a gated admission can leave
    unfolded, max(max_held_sources, prepare_pack_max); without a held
    gate only the host budget bounds it."""

    ungated = GrpoTrainerConfig.from_namespace(_trainer_namespace())
    assert ungated.streaming.source_pinned_host_fraction == 0.8
    assert ungated.streaming.source_pinned_min_slabs == 1
    assert ungated.streaming.source_pinned_max_slabs is None
    assert ungated.streaming.statistic_open_groups == 0
    gated = GrpoTrainerConfig.from_namespace(
        _trainer_namespace(
            max_held_sources=2,
            prepare_pack_max=3,
            max_concurrent_groups=2,
            pinned_host_fraction=0.5,
        )
    )
    assert gated.streaming.source_pinned_min_slabs == 3
    assert gated.streaming.source_pinned_max_slabs == 3
    assert gated.streaming.statistic_open_groups == 2
    assert gated.streaming.source_pinned_host_fraction == 0.5
    off = GrpoTrainerConfig.from_namespace(
        _trainer_namespace(pinned_host_fraction=0.0)
    )
    assert off.streaming.source_pinned_host_fraction == 0.0
    # one rank fed by two workers of one group each holds two groups open
    fed_by_two = GrpoTrainerConfig.from_namespace(
        _trainer_namespace(max_concurrent_groups=1, statistic_open_groups=2)
    )
    assert fed_by_two.streaming.statistic_open_groups == 2
    with pytest.raises(ConfigError, match="source_pinned_host_fraction"):
        GrpoTrainerConfig.from_namespace(
            _trainer_namespace(pinned_host_fraction=1.5)
        )


def test_the_none_device_spelling_becomes_an_absent_device() -> None:
    """Both device selections are argparse choices whose 'off' value is a
    string; the builder takes the absence.  The source offload target is
    deliberately not one of them -- the executor reads 'none' itself."""

    config = GrpoTrainerConfig.from_namespace(
        _trainer_namespace(
            gradient_accumulation_offload_device="none",
            statistic_device="none",
            source_offload_device="none",
        )
    )
    assert config.streaming.gradient_accumulation_offload_device is None
    assert config.streaming.statistic_device is None
    assert config.streaming.source_offload_device == "none"

    selected = GrpoTrainerConfig.from_namespace(
        _trainer_namespace(
            gradient_accumulation_offload_device="cpu",
            statistic_device="cuda",
        )
    )
    assert selected.streaming.gradient_accumulation_offload_device == "cpu"
    assert selected.streaming.statistic_device == "cuda"


def test_statistic_device_auto_places_device_resident_sources_only() -> None:
    """auto sizes device statistic homes for device-resident sources; any
    other offload mode has no device fold to size them for."""

    config = GrpoTrainerConfig.from_namespace(
        _trainer_namespace(
            statistic_device="auto",
            source_offload_mode="device_resident",
            device_statistic_reserve_gib=20.0,
        )
    )
    assert config.streaming.statistic_device == "auto"
    assert config.streaming.device_statistic_reserve_gib == 20.0
    assert (
        GrpoTrainerConfig.from_namespace(_trainer_namespace())
        .streaming.device_statistic_reserve_gib
        == 0.0
    )
    with pytest.raises(ConfigError, match="device_resident"):
        GrpoTrainerConfig.from_namespace(
            _trainer_namespace(
                statistic_device="auto",
                source_offload_mode="pinned_nonblocking",
            )
        )
    with pytest.raises(ConfigError, match="reserve"):
        GrpoTrainerConfig.from_namespace(
            _trainer_namespace(
                statistic_device="auto",
                source_offload_mode="device_resident",
                device_statistic_reserve_gib=-1.0,
            )
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"source_offload_mode": "streaming"},
        {"source_reduction_mode": "restricted"},
        {"statistic_dtype": "bf16"},
        {"learning_rate": 0.0},
        {"lora_rank": -1},
        {"rebuild_block_tokens": 0},
    ],
)
def test_an_invalid_trainer_value_is_refused(overrides: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        GrpoTrainerConfig.from_namespace(_trainer_namespace(**overrides))


def test_every_accepted_offload_and_statistic_enumeration_is_admitted() -> None:
    for mode in (
        "blocking",
        "pinned_nonblocking",
        "pinned_streaming",
        "device_resident",
    ):
        assert (
            GrpoTrainerConfig.from_namespace(
                _trainer_namespace(source_offload_mode=mode)
            ).streaming.source_offload_mode
            == mode
        )
    for dtype in ("fp32", "source"):
        assert (
            GrpoTrainerConfig.from_namespace(
                _trainer_namespace(statistic_dtype=dtype)
            ).streaming.statistic_dtype
            == dtype
        )


def test_a_grpo_trainer_config_cannot_be_mutated_after_it_is_built() -> None:
    config = GrpoTrainerConfig.from_namespace(_trainer_namespace())
    for instance, field in (
        (config, "grpo_epsilon"),
        (config.policy, "learning_rate"),
        (config.policy, "lora_rank"),
        (config.streaming, "source_offload_mode"),
    ):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(instance, field, 1)


def test_the_rebuild_checkpoint_storage_device_defaults_to_the_device() -> None:
    from types import SimpleNamespace

    base = dict(
        model_path="/m", learning_rate=1e-5, rebuild_block_tokens=2048
    )
    config = GrpoTrainerConfig.from_namespace(SimpleNamespace(**base))
    assert config.streaming.rebuild_checkpoint_storage_device is None
    config = GrpoTrainerConfig.from_namespace(
        SimpleNamespace(**base, rebuild_checkpoint_storage_device="none")
    )
    assert config.streaming.rebuild_checkpoint_storage_device is None
    config = GrpoTrainerConfig.from_namespace(
        SimpleNamespace(**base, rebuild_checkpoint_storage_device="cpu")
    )
    assert config.streaming.rebuild_checkpoint_storage_device == "cpu"


def test_the_retained_bytes_log_cadence_defaults_to_never_and_refuses_negatives() -> None:
    from types import SimpleNamespace

    import pytest

    base = dict(model_path="/m", learning_rate=1e-5, rebuild_block_tokens=2048)
    assert GrpoTrainerConfig.from_namespace(SimpleNamespace(**base)).streaming.retained_bytes_log_every == 0
    assert GrpoTrainerConfig.from_namespace(
        SimpleNamespace(**base, retained_bytes_log_every=8)
    ).streaming.retained_bytes_log_every == 8
    with pytest.raises(ConfigError):
        GrpoTrainerConfig.from_namespace(SimpleNamespace(**base, retained_bytes_log_every=-1))
