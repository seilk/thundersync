"""The sandbox scratch root is private to the user who runs the sandboxes."""

import os
import stat

import pytest

from thundersync.rollout.scratch_guard import ensure_private_dir


def test_a_new_scratch_root_is_created_owner_only(tmp_path) -> None:
    root = tmp_path / "parent" / "scratch"
    ensure_private_dir(root)
    assert stat.S_IMODE(os.stat(root).st_mode) == 0o700
    ensure_private_dir(root)


def test_a_symlinked_scratch_root_is_refused(tmp_path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(RuntimeError, match="not a plain directory"):
        ensure_private_dir(link)


def test_a_world_writable_scratch_root_is_refused(tmp_path) -> None:
    root = tmp_path / "scratch"
    root.mkdir()
    os.chmod(root, 0o777)
    with pytest.raises(RuntimeError, match="writable by other users"):
        ensure_private_dir(root)
