"""Retrieved-file ordering inside the repository context, and the block layout.

``build_repo_context(..., ordering=...)`` arranges the blocks inside
``<relevant_files>``; the file SET a task receives is fixed by retrieval and
is identical under every ordering, because that set is what the model can
read.

``CONTEXT_TEMPLATE`` lays its blocks out most-shared first: exact prefix
sharing stops at the first divergent token, so nothing task-specific may
precede the retrieved files.
"""

from __future__ import annotations

import inspect
import os
import shlex

import pytest

from thundersync.rollout.agent import (  # noqa: E402
    CONTEXT_TEMPLATE,
    ORDERINGS,
    RetrievedFile,
    _display_path,
    build_repo_context,
    order_retrieved_files,
    render_files_block,
    truncate_observation,
)

BODIES = {
    "pkg/core.py": "C" * 1000,
    "pkg/util.py": "U" * 500,
    "pkg/io.py": "I" * 100,
    "pkg/net.py": "N" * 50,
    "pkg/cli.py": "L" * 50,
}


def _task_files(paths):
    return [RetrievedFile(path=p, body=BODIES[p]) for p in paths]


# --------------------------------------------------------------------------
# a container stub: canned output for the commands retrieval issues
# --------------------------------------------------------------------------


class FakeSandbox:
    """Replays fixed command output so retrieval can run without Docker."""

    def __init__(self, tree: str, ranked: list[str], bodies: dict[str, str]):
        self.tree, self.ranked, self.bodies = tree, ranked, bodies
        self.commands: list[str] = []

    def exec(self, command: str) -> str:
        self.commands.append(command)
        if command.startswith("cat "):
            path = shlex.split(command[len("cat "):].split(" 2>/dev/null")[0])[0]
            return self.bodies.get(path, "")
        if command.startswith("grep -rEc"):
            return "\n".join(self.ranked) + "\n"
        if command.startswith("find ."):
            return self.tree
        raise AssertionError(f"unexpected command: {command}")


PROBLEM = (
    "TokenBucket.consume raises AttributeError when refill_rate is zero.\n"
    "Repro: TokenBucket(capacity=10, refill_rate=0).consume(1) -- consume "
    "then calls _refill and _refill touches refill_rate.\n"
)
RANKED = ["./pkg/io.py", "./pkg/util.py", "./pkg/core.py"]
"""Grep-count order for the stub task."""
TREE = "pkg/core.py\npkg/util.py\npkg/io.py\npkg/net.py\npkg/cli.py"


def _sandbox():
    return FakeSandbox(TREE, RANKED, {"./" + p: BODIES[p] for p in BODIES})


def _blocks(files_text: str) -> list[tuple[str, str]]:
    out = []
    for chunk in files_text.split("--- "):
        if not chunk.strip():
            continue
        header, _, body = chunk.partition(" ---\n")
        out.append((header, body.rstrip("\n")))
    return out


# --------------------------------------------------------------------------
# 1. the file set is what the model reads, and no ordering may change it
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ordering", ORDERINGS)
def test_rendered_blocks_are_a_permutation_under_every_ordering(ordering):
    files = _task_files(["pkg/io.py", "pkg/core.py", "pkg/util.py", "pkg/net.py"])
    reordered = order_retrieved_files(files, ordering=ordering)
    assert len(reordered) == len(files)
    assert sorted(render_files_block([f]) for f in reordered) == sorted(
        render_files_block([f]) for f in files
    )


def test_path_ordering_sorts_by_display_path():
    files = _task_files(["pkg/io.py", "pkg/core.py", "pkg/util.py"])
    assert [f.path for f in order_retrieved_files(files, ordering="path")] == [
        "pkg/core.py",
        "pkg/io.py",
        "pkg/util.py",
    ]


def test_unknown_ordering_is_refused():
    files = _task_files(["pkg/io.py"])
    with pytest.raises(ValueError, match="unknown ordering"):
        order_retrieved_files(files, ordering="nonesuch")
    with pytest.raises(ValueError, match="unknown ordering"):
        build_repo_context(_sandbox(), PROBLEM, extensions=["py"], ordering="nonesuch")


def test_grep_is_the_default_ordering():
    assert ORDERINGS == ("grep", "path")
    assert inspect.signature(order_retrieved_files).parameters["ordering"].default == "grep"
    assert inspect.signature(build_repo_context).parameters["ordering"].default == "grep"


# --------------------------------------------------------------------------
# 2. retrieval end to end
# --------------------------------------------------------------------------


def test_default_context_is_grep_order_with_display_paths():
    box = _sandbox()
    tree, files_text = build_repo_context(
        box, PROBLEM, extensions=["py"], max_files=6, max_file_chars=8000
    )
    assert tree == TREE
    expected = "\n\n".join(
        f"--- {path} ---\n{truncate_observation(BODIES[path], 8000)}"
        for path in ("pkg/io.py", "pkg/util.py", "pkg/core.py")
    )
    assert files_text == expected


def test_path_ordering_changes_order_not_content():
    default = build_repo_context(_sandbox(), PROBLEM, extensions=["py"])[1]
    ordered = build_repo_context(_sandbox(), PROBLEM, extensions=["py"], ordering="path")[1]
    assert set(_blocks(default)) == set(_blocks(ordered))
    assert [p for p, _ in _blocks(ordered)] == ["pkg/core.py", "pkg/io.py", "pkg/util.py"]


@pytest.mark.parametrize(
    ("raw", "shown"),
    [
        (".github/ci.yml", ".github/ci.yml"),
        (".env", ".env"),
        ("./a/b.py", "a/b.py"),
        ("././x", "x"),
    ],
)
def test_display_path_keeps_leading_dots_of_names(raw, shown):
    assert _display_path(raw) == shown


def test_display_path_strips_only_a_leading_dot_slash():
    bodies = {
        "./.github/ci.yml": "on: push",
        "./.env": "KEY=value",
        "./a/b.py": "x = 1",
    }
    box = FakeSandbox("", list(bodies), bodies)
    files_text = build_repo_context(box, PROBLEM, extensions=["py"])[1]
    assert [p for p, _ in _blocks(files_text)] == [".github/ci.yml", ".env", "a/b.py"]


def test_retrieval_issues_the_same_container_commands():
    """The stub asserts on anything unexpected; this pins the command count."""
    box = _sandbox()
    build_repo_context(box, PROBLEM, extensions=["py"])
    assert sum(c.startswith("find .") for c in box.commands) == 1
    assert sum(c.startswith("grep -rEc") for c in box.commands) == 1
    assert sum(c.startswith("cat ") for c in box.commands) == len(RANKED)


# --------------------------------------------------------------------------
# 3. the block layout
# --------------------------------------------------------------------------

REPO_INVARIANT_BLOCKS = ("{tree}",)
"""Identical for every task drawn from one repository."""

TASK_SPECIFIC_BLOCKS = ("{files}", "{problem_statement}")
"""Differ from task to task; ``{files}`` overlaps across tasks, the issue does
not overlap at all."""


def test_context_template_puts_every_repo_invariant_block_first():
    for block in REPO_INVARIANT_BLOCKS + TASK_SPECIFIC_BLOCKS:
        assert CONTEXT_TEMPLATE.count(block) == 1, block
    last_invariant = max(CONTEXT_TEMPLATE.index(b) for b in REPO_INVARIANT_BLOCKS)
    first_specific = min(CONTEXT_TEMPLATE.index(b) for b in TASK_SPECIFIC_BLOCKS)
    assert last_invariant < first_specific


def test_context_template_puts_retrieved_files_before_the_issue():
    """The issue text is shared by nothing, so any block after it is
    unreachable as a shared prefix."""
    assert CONTEXT_TEMPLATE.index("{files}") < CONTEXT_TEMPLATE.index("{problem_statement}")
    assert CONTEXT_TEMPLATE.index("<file_tree>") < CONTEXT_TEMPLATE.index("<relevant_files>")
    assert CONTEXT_TEMPLATE.index("<relevant_files>") < CONTEXT_TEMPLATE.index("<issue>")


def test_context_template_renders_two_tasks_sharing_a_prefix_past_the_tree():
    """Two tasks of one repo with the same first retrieved file share a
    prefix that runs past the tree and into that file's block."""
    shared_block = f"--- pkg/core.py ---\n{BODIES['pkg/core.py']}"
    a = CONTEXT_TEMPLATE.format(
        tree=TREE,
        files=shared_block + "\n\n--- pkg/util.py ---\n" + BODIES["pkg/util.py"],
        problem_statement="issue A",
    )
    b = CONTEXT_TEMPLATE.format(
        tree=TREE,
        files=shared_block + "\n\n--- pkg/io.py ---\n" + BODIES["pkg/io.py"],
        problem_statement="issue B",
    )
    common = os.path.commonprefix([a, b])
    assert len(common) > len(TREE) + len(shared_block)
    assert "issue A" not in common and "issue B" not in common
