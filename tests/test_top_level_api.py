"""The top-level entry points: each resolves to its submodule's object, and
``import thundersync`` alone imports no torch."""

import importlib
import subprocess
import sys

import pytest

import thundersync


def test_import_thundersync_does_not_import_torch():
    code = "import sys, thundersync; assert 'torch' not in sys.modules, 'torch imported'"
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("name", sorted(thundersync._EXPORTS))
def test_each_top_level_name_is_its_submodule_object(name):
    pytest.importorskip("torch")
    module = importlib.import_module(thundersync._EXPORTS[name])
    assert getattr(thundersync, name) is getattr(module, name)
    assert name in thundersync.__all__
    assert name in dir(thundersync)


def test_an_unknown_name_raises_attribute_error():
    with pytest.raises(AttributeError, match="no_such_name"):
        thundersync.no_such_name  # noqa: B018


def test_all_lists_exactly_the_lazy_exports_and_the_version():
    assert sorted(thundersync.__all__) == sorted([*thundersync._EXPORTS, "__version__"])
