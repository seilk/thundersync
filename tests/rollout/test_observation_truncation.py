"""Token-based observation truncation, against fake tokenizers."""

from __future__ import annotations

import pytest

from thundersync.rollout.agent import truncate_observation_tokens
from thundersync.rollout.bounded_output import CapturedText, CommandOutput


class CharTokenizer:
    """One token per character; records the longest text it was asked to encode."""

    def __init__(self) -> None:
        self.longest = 0

    def encode(self, text: str) -> list[int]:
        self.longest = max(self.longest, len(text))
        return [ord(c) for c in text]

    def decode(self, ids) -> str:
        return "".join(chr(i) for i in ids)


def byte_encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def byte_decode(ids) -> str:
    return bytes(ids).decode("utf-8", "replace")


def truncate(output, tok=None, **kw):
    tok = tok or CharTokenizer()
    return truncate_observation_tokens(output, tok.encode, tok.decode, **kw)


def test_an_observation_within_the_limit_is_unchanged():
    text = "a" * 4096
    assert truncate(text) == text


def test_keeps_2048_tokens_at_each_end_and_counts_the_elided_tokens():
    text = "h" * 2048 + "m" * 1000 + "t" * 2048
    assert truncate(text) == (
        "h" * 2048 + "\n\n... <1000 tokens elided> ...\n\n" + "t" * 2048
    )


def test_a_mid_size_observation_counts_the_elided_tokens_exactly():
    text = "h" * 2048 + "m" * 50_000 + "t" * 2048
    assert truncate(text) == (
        "h" * 2048 + "\n\n... <50000 tokens elided> ...\n\n" + "t" * 2048
    )


def test_a_long_observation_is_tokenized_in_bounded_windows():
    tok = CharTokenizer()
    text = "h" * 2048 + "m" * 1_000_000 + "t" * 2048
    out = truncate(text, tok)
    assert out == "h" * 2048 + "\n\n... <1000000 chars elided> ...\n\n" + "t" * 2048
    assert tok.longest < 100_000


def test_an_incomplete_output_takes_each_end_from_what_was_kept():
    part = CapturedText(
        head="h" * 2048 + "x" * 952, tail="y" * 952 + "t" * 2048, chars=1_000_000
    )
    out = truncate(CommandOutput((part, CapturedText.of_text("<note>"))))
    assert out == (
        "h" * 2048
        + f"\n\n... <{1_000_006 - 4096} chars elided> ...\n\n"
        + "t" * 2042
        + "<note>"
    )


def test_a_cut_inside_a_multibyte_character_is_trimmed_to_a_clean_boundary():
    text = "a" + "é" * 3000 + "b"
    out = truncate_observation_tokens(text, byte_encode, byte_decode)
    assert "�" not in out
    head, tail = out.split("\n\n... <")[0], out.split("> ...\n\n")[1]
    assert text.startswith(head) and text.endswith(tail)
    assert len(head.encode()) == 2047 and len(tail.encode()) == 2047


def test_max_tokens_splits_evenly_between_head_and_tail():
    text = "".join(chr(0x4E00 + i) for i in range(100))
    out = truncate(text, max_tokens=10)
    assert out == text[:5] + "\n\n... <90 tokens elided> ...\n\n" + text[-5:]


def test_rejects_a_limit_below_two_tokens():
    with pytest.raises(ValueError, match="at least 2"):
        truncate("abc", max_tokens=1)
