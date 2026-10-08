"""The streaming attention backend switch: THUNDERSYNC_FA4 = require | off | auto.

Runs under real torch (the streaming module imports it); the FA4 import is
faked in both directions so the node's PYTHONPATH does not decide the test.
"""

from __future__ import annotations

import importlib.metadata
import sys
import types

import pytest

pytest.importorskip("torch")
try:
    from thundersync.engine import streaming
except Exception as error:  # noqa: BLE001 - the local torch stub
    pytest.skip(f"streaming needs real torch: {error!r}", allow_module_level=True)


@pytest.fixture
def unresolved(monkeypatch):
    monkeypatch.setattr(streaming, "_FA4_RESOLVED", False)
    monkeypatch.setattr(streaming, "_FA4_FUNC", None)
    monkeypatch.setattr(streaming, "_FA4_BUILD", None)
    monkeypatch.setattr(streaming, "_FA4_SETTING", None)
    # a None entry makes the import raise ImportError whatever the path holds
    monkeypatch.setitem(sys.modules, "flash_attn", None)
    monkeypatch.setitem(sys.modules, "flash_attn.cute", None)
    monkeypatch.setitem(sys.modules, "flash_attn.cute.interface", None)
    yield
    monkeypatch.setattr(streaming, "_FA4_RESOLVED", False)


def test_off_skips_fa4_without_importing_it(unresolved, monkeypatch) -> None:
    monkeypatch.setenv(streaming.FA4_ENV, "off")
    assert streaming._fa4_flash_attn_func() is None
    assert streaming.fa4_build() is None
    assert streaming._FA4_RESOLVED is True


def test_require_without_a_build_raises_on_every_dispatch(unresolved, monkeypatch) -> None:
    monkeypatch.setenv(streaming.FA4_ENV, "require")
    for _ in range(2):
        with pytest.raises(RuntimeError, match="THUNDERSYNC_FA4=require"):
            streaming._fa4_flash_attn_func()
    assert streaming._FA4_RESOLVED is False
    assert streaming.fa4_build() is None


def test_auto_without_a_build_leaves_fa4_absent(unresolved, monkeypatch) -> None:
    monkeypatch.delenv(streaming.FA4_ENV, raising=False)
    assert streaming.fa4_setting() == "auto"
    assert streaming._fa4_flash_attn_func() is None
    assert streaming._FA4_RESOLVED is True


def test_an_unknown_setting_is_refused(unresolved, monkeypatch) -> None:
    monkeypatch.setenv(streaming.FA4_ENV, "maybe")
    with pytest.raises(ValueError, match="THUNDERSYNC_FA4"):
        streaming._fa4_flash_attn_func()


def test_a_present_build_is_resolved_and_named(unresolved, monkeypatch) -> None:
    package = types.ModuleType("flash_attn")
    package.__version__ = "9.9.9"
    interface = types.ModuleType("flash_attn.cute.interface")
    interface.flash_attn_func = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "flash_attn", package)
    monkeypatch.setitem(sys.modules, "flash_attn.cute", types.ModuleType("flash_attn.cute"))
    monkeypatch.setitem(sys.modules, "flash_attn.cute.interface", interface)
    monkeypatch.setenv(streaming.FA4_ENV, "require")
    assert streaming._fa4_flash_attn_func() is interface.flash_attn_func
    assert streaming.fa4_build() == "flash_attn 9.9.9"


def test_a_namespace_build_is_named_by_its_distribution(unresolved, monkeypatch) -> None:
    package = types.ModuleType("flash_attn")
    cute_package = types.ModuleType("flash_attn.cute")
    cute_package.__version__ = "0.0.0"
    interface = types.ModuleType("flash_attn.cute.interface")
    interface.flash_attn_func = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "flash_attn", package)
    monkeypatch.setitem(sys.modules, "flash_attn.cute", cute_package)
    monkeypatch.setitem(sys.modules, "flash_attn.cute.interface", interface)
    monkeypatch.setattr(
        importlib.metadata, "packages_distributions", lambda: {"flash_attn": ["flash_attn_4"]}
    )
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "4.0.0b27"
        if name == "flash_attn_4"
        else (_ for _ in ()).throw(importlib.metadata.PackageNotFoundError(name)),
    )
    monkeypatch.setenv(streaming.FA4_ENV, "require")
    assert streaming._fa4_flash_attn_func() is interface.flash_attn_func
    assert streaming.fa4_build() == "flash_attn_4 4.0.0b27"


def test_a_build_without_metadata_falls_back_to_cute_then_unknown(unresolved, monkeypatch) -> None:
    package = types.ModuleType("flash_attn")
    cute_package = types.ModuleType("flash_attn.cute")
    cute_package.__version__ = "1.2.3"
    interface = types.ModuleType("flash_attn.cute.interface")
    interface.flash_attn_func = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "flash_attn", package)
    monkeypatch.setitem(sys.modules, "flash_attn.cute", cute_package)
    monkeypatch.setitem(sys.modules, "flash_attn.cute.interface", interface)
    monkeypatch.setattr(importlib.metadata, "packages_distributions", lambda: {})
    monkeypatch.setenv(streaming.FA4_ENV, "require")
    assert streaming.fa4_build() == "flash_attn.cute 1.2.3"

    cute_package.__version__ = "0.0.0"
    monkeypatch.setattr(streaming, "_FA4_RESOLVED", False)
    monkeypatch.setattr(streaming, "_FA4_BUILD", None)
    assert streaming.fa4_build() == "flash_attn unknown"
