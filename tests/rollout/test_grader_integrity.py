"""The grader must never score correct work as failure.

A false negative is worse than a false positive here. It silently changes ``k``,
which changes ``2*sqrt(k(G-k))``, which changes the advantage of *every* rollout
in the group -- including the ones that were graded correctly. One mislabelled
rollout corrupts the whole group's update and every aggregate built on it.

A grader that matches words in pytest's console output has a real false
negative: a passing test that emits a warning containing "ValueError" scores
0.0. These tests exercise that class of failure directly against live containers.

Marked ``docker`` because they run real SWE-Smith images.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


from thundersync.rollout.agent import Task  # noqa: E402
from thundersync.rollout.docker_sandbox import DockerSandbox  # noqa: E402
from thundersync.rollout.engine import grade, run_tests  # noqa: E402

IMAGE = "jyangballin/swesmith.x86_64.pytest-dev_1776_iniconfig.16793ead"


def docker_available() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=15).returncode == 0
    except Exception:
        return False


# These tests pull a SWE-smith image and run real containers, so they run
# only when asked for.
pytestmark = pytest.mark.skipif(
    os.environ.get("THUNDERSYNC_DOCKER_TESTS") != "1" or not docker_available(),
    reason="set THUNDERSYNC_DOCKER_TESTS=1 with a docker daemon",
)


def write_tests(box: DockerSandbox, body: str, path: str = "gradert/test_g.py") -> None:
    box.exec(f"mkdir -p /testbed/{Path(path).parent}")
    box.exec(f"cat > /testbed/{path} <<'THUNDERSYNC_EOF'\n{body}\nTHUNDERSYNC_EOF")


def synthetic_task(node_ids: list[str]) -> Task:
    return Task(
        instance_id="synthetic",
        problem_statement="",
        image_name=IMAGE,
        fail_to_pass=node_ids,
    )


# ----------------------------------------------------------------------
# false negatives: correct work must score 1.0
# ----------------------------------------------------------------------


def test_passing_test_emitting_error_warning_scores_one():
    """A passing test that warns about a 'ValueError' scores 1.0.

    A console-output grader scores it 0.0 because the word 'error' appears.
    """
    body = (
        "import warnings\n"
        "def test_ok():\n"
        "    warnings.warn('ValueError is deprecated')\n"
        "    assert True\n"
    )
    with DockerSandbox(IMAGE, "grader-warn") as box:
        write_tests(box, body)
        v = grade(box, synthetic_task(["gradert/test_g.py::test_ok"]), restore_tests=False)
        assert v.reward == 1.0, f"correct work scored {v.reward}: {v.reason} {v.failed}"


def test_passing_test_named_error_scores_one():
    body = "def test_error_handling():\n    assert True\n"
    with DockerSandbox(IMAGE, "grader-name") as box:
        write_tests(box, body)
        v = grade(box, synthetic_task(["gradert/test_g.py::test_error_handling"]), restore_tests=False)
        assert v.reward == 1.0, f"correct work scored {v.reward}: {v.reason}"


def test_passing_test_printing_failed_scores_one():
    """Console text the test itself prints must not be read as a verdict."""
    body = (
        "def test_prints():\n"
        "    print('1 failed, 3 errors')\n"
        "    assert True\n"
    )
    with DockerSandbox(IMAGE, "grader-print") as box:
        write_tests(box, body)
        v = grade(box, synthetic_task(["gradert/test_g.py::test_prints"]), restore_tests=False)
        assert v.reward == 1.0, f"correct work scored {v.reward}: {v.reason}"


def test_many_passing_tests_are_all_run_not_truncated():
    """More than one batch of tests must all be graded, not the first 40.

    Truncation is a false *positive* risk in the other direction: untested tests
    could be failing while the reward reads 1.0.
    """
    n = 95
    body = "".join(f"def test_n{i}():\n    assert True\n" for i in range(n))
    ids = [f"gradert/test_g.py::test_n{i}" for i in range(n)]
    with DockerSandbox(IMAGE, "grader-many") as box:
        write_tests(box, body)
        v = grade(box, synthetic_task(ids), restore_tests=False)
        assert v.reward == 1.0, f"{v.reason} failed={v.failed[:3]} missing={v.missing[:3]}"
        assert len(v.passed) == n, f"only graded {len(v.passed)} of {n}"


def test_slow_but_passing_test_scores_one():
    body = "import time\ndef test_slow():\n    time.sleep(3)\n    assert True\n"
    with DockerSandbox(IMAGE, "grader-slow") as box:
        write_tests(box, body)
        v = grade(box, synthetic_task(["gradert/test_g.py::test_slow"]), restore_tests=False)
        assert v.reward == 1.0, f"slow correct work scored {v.reward}: {v.reason}"


# ----------------------------------------------------------------------
# true negatives: broken work must score 0.0
# ----------------------------------------------------------------------


def test_failing_test_scores_zero():
    body = "def test_bad():\n    assert False\n"
    with DockerSandbox(IMAGE, "grader-fail") as box:
        write_tests(box, body)
        v = grade(box, synthetic_task(["gradert/test_g.py::test_bad"]), restore_tests=False)
        assert v.reward == 0.0 and v.failed


def test_partial_pass_scores_zero():
    """All FAIL_TO_PASS must pass; a majority is not a fix."""
    body = "def test_a():\n    assert True\ndef test_b():\n    assert False\n"
    ids = ["gradert/test_g.py::test_a", "gradert/test_g.py::test_b"]
    with DockerSandbox(IMAGE, "grader-partial") as box:
        write_tests(box, body)
        v = grade(box, synthetic_task(ids), restore_tests=False)
        assert v.reward == 0.0
        assert v.passed == ["gradert/test_g.py::test_a"]
        assert v.failed == ["gradert/test_g.py::test_b"]


def test_deleted_test_is_missing_not_passing():
    """Deleting the graded test must not read as success."""
    with DockerSandbox(IMAGE, "grader-deleted") as box:
        v = grade(box, synthetic_task(["gradert/nonexistent.py::test_gone"]), restore_tests=False)
        assert v.reward == 0.0
        assert not v.conclusive or v.missing, "a vanished test must never score 1.0"


def test_skipped_test_is_not_a_pass():
    """A skip is an absence of evidence, not evidence of a fix."""
    body = "import pytest\n@pytest.mark.skip(reason='x')\ndef test_skipped():\n    assert True\n"
    with DockerSandbox(IMAGE, "grader-skip") as box:
        write_tests(box, body)
        v = grade(box, synthetic_task(["gradert/test_g.py::test_skipped"]), restore_tests=False)
        assert v.reward == 0.0, "a skipped test was counted as passing"


def test_collection_error_is_not_a_pass():
    body = "import does_not_exist_anywhere\ndef test_x():\n    assert True\n"
    with DockerSandbox(IMAGE, "grader-collect") as box:
        write_tests(box, body)
        v = grade(box, synthetic_task(["gradert/test_g.py::test_x"]), restore_tests=False)
        assert v.reward == 0.0


# ----------------------------------------------------------------------
# the standard oracle: FAIL_TO_PASS *and* PASS_TO_PASS
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_task() -> Task:
    from datasets import load_dataset

    ds = load_dataset("SWE-bench/SWE-smith", split="train")
    r = next(x for x in ds if x["instance_id"].startswith("pytest-dev__iniconfig"))
    return Task(
        instance_id=r["instance_id"],
        problem_statement=r["problem_statement"],
        image_name=r["image_name"],
        fail_to_pass=r["FAIL_TO_PASS"],
        pass_to_pass=r["PASS_TO_PASS"],
        repo=r["repo"],
        bug_patch=r["patch"],
    )


def test_buggy_repo_scores_zero(real_task: Task):
    with DockerSandbox(real_task.image_name, "oracle-buggy") as box:
        box.apply_bug_patch(real_task.bug_patch)
        v = grade(box, real_task)
        assert v.reward == 0.0 and v.failed


def test_gold_fix_scores_one(real_task: Task):
    """Reverting the injected bug is the reference correct fix.

    If the oracle cannot recognise the gold patch, it cannot recognise a real
    one either, and every solve rate is a floor of unknown depth.
    """
    with DockerSandbox(real_task.image_name, "oracle-gold") as box:
        box.apply_bug_patch(real_task.bug_patch)
        assert grade(box, real_task).reward == 0.0
        # undo the bug: the gold fix
        import base64

        b64 = base64.b64encode(real_task.bug_patch.encode()).decode()
        box.exec(f"printf '%s' '{b64}' | base64 -d > /tmp/undo.patch")
        out = box.exec("git apply -R --whitespace=nowarn /tmp/undo.patch 2>&1; echo rc=$?")
        assert "rc=0" in out, f"could not revert the bug: {out[:200]}"

        v = grade(box, real_task)
        assert v.reward == 1.0, (
            f"gold fix scored {v.reward}: reason={v.reason!r} "
            f"failed={v.failed[:3]} regressed={v.regressed[:3]}"
        )
        assert v.n_pass_to_pass > 0, "PASS_TO_PASS was not actually checked"
        assert not v.regressed


def test_fix_that_breaks_other_tests_scores_zero(real_task: Task):
    """The false positive the old grader allowed.

    Repairing the target tests while breaking unrelated behaviour is not a fix.
    Checking FAIL_TO_PASS alone would score this 1.0.
    """
    import base64

    with DockerSandbox(real_task.image_name, "oracle-regress") as box:
        box.apply_bug_patch(real_task.bug_patch)
        b64 = base64.b64encode(real_task.bug_patch.encode()).decode()
        box.exec(f"printf '%s' '{b64}' | base64 -d > /tmp/undo.patch")
        box.exec("git apply -R --whitespace=nowarn /tmp/undo.patch")
        assert grade(box, real_task).reward == 1.0, "setup: gold fix must pass first"

        # now sabotage a source file the PASS_TO_PASS suite covers
        src = box.exec(
            "git ls-files 'src/*.py' '*/__init__.py' | grep -v test | head -1"
        ).strip()
        assert src, "no source file found to sabotage"
        box.exec(f"printf '\\nraise RuntimeError(\"sabotage\")\\n' >> {src!r}")

        v = grade(box, real_task)
        assert v.reward == 0.0, (
            "a fix that breaks unrelated behaviour was scored as success"
        )


def test_regressions_are_only_checked_when_they_can_change_the_answer(real_task: Task):
    """Cost discipline: don't run 655 tests for a rollout that already failed."""
    with DockerSandbox(real_task.image_name, "oracle-shortcircuit") as box:
        box.apply_bug_patch(real_task.bug_patch)
        v = grade(box, real_task)
        assert v.reward == 0.0
        assert v.n_pass_to_pass == 0, "PASS_TO_PASS ran despite FAIL_TO_PASS failing"


# ----------------------------------------------------------------------
# the verdict must explain itself
# ----------------------------------------------------------------------


def test_verdict_reports_which_tests_failed():
    """A bare float cannot be audited; the verdict names the tests."""
    body = "def test_a():\n    assert True\ndef test_b():\n    assert False\n"
    ids = ["gradert/test_g.py::test_a", "gradert/test_g.py::test_b"]
    with DockerSandbox(IMAGE, "grader-explain") as box:
        write_tests(box, body)
        v = grade(box, synthetic_task(ids), restore_tests=False)
        assert v.reason == "failing_tests"
        assert set(v.passed) | set(v.failed) == set(ids)


def test_run_tests_wrapper_matches_grade():
    body = "def test_ok():\n    assert True\n"
    with DockerSandbox(IMAGE, "grader-wrapper") as box:
        write_tests(box, body)
        task = synthetic_task(["gradert/test_g.py::test_ok"])
        assert run_tests(box, task, restore_tests=False) == grade(
            box, task, restore_tests=False
        ).reward
