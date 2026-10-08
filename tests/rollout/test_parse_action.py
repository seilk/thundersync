"""``parse_action``: the bash command a reply carries in its fenced block."""

from __future__ import annotations

from thundersync.rollout.agent import parse_action


def test_a_normal_block_yields_its_command():
    reply = "Let me look.\n```bash\nls -la src\n```\nThen I will edit."
    assert parse_action(reply) == "ls -la src"


def test_the_closing_fence_may_carry_trailing_blanks():
    assert parse_action("```bash \ngit status\n```  \n") == "git status"


def test_a_reply_without_a_block_has_no_action():
    assert parse_action("I think the bug is in parser.py.") is None
    assert parse_action("```python\nprint(1)\n```") is None


def test_a_block_cut_before_its_closing_fence_has_no_action():
    assert parse_action("```bash\necho unfinished\n``") is None


def test_backticks_inside_a_line_do_not_close_the_block():
    command = (
        "cat > NOTES.md <<'EOF'\n"
        "Run it with ```make test``` first.\n"
        "Inline: ``` is a fence marker.\n"
        "EOF"
    )
    assert parse_action(f"```bash\n{command}\n```\n") == command


def test_the_first_block_is_the_action():
    reply = "```bash\nsubmit\n```\n```bash\nrm -rf /\n```"
    assert parse_action(reply) == "submit"
