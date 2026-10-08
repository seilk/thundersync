"""Minimal bash-only SWE agent with token-exact trajectory recording.

Modelled on mini-swe-agent: no structured tool API, the model emits a bash
command in a fenced block and gets stdout back.

The one non-standard requirement is that trajectories are recorded as an
append-only token sequence. The prefix tree needs to know exactly which physical
tokens two rollouts share, so the context is assembled from token ids directly
rather than by re-rendering a chat template each turn (which can retokenise
across a boundary and manufacture spurious divergence).
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

from thundersync.rollout.bounded_output import CommandOutput, command_output


class ExecSandbox(Protocol):
    """What repository retrieval needs from a sandbox: a shell command's text.

    Both sandbox backends satisfy it; ``command_output`` uses their bounded
    ``exec_output`` when present.
    """

    def exec(self, command: str) -> str: ...

SYSTEM_PROMPT = """You are a software engineer fixing a bug in a repository.

You interact with the repository by emitting exactly one bash command per reply,
inside a fenced block:

```bash
your command here
```

The command runs in the repository root and you get its combined output back.
Work incrementally: look at the relevant code, make a focused edit, and verify.

When the fix is complete, reply with exactly:

```bash
submit
```
"""

TASK_TEMPLATE = """Fix the following issue in the repository at /testbed.

<issue>
{problem_statement}
</issue>

Begin by inspecting the relevant files."""

# Block order is load-bearing: exact prefix sharing stops at the first divergent
# token, so the blocks are laid out most-shared first, in three tiers.
#
#   1. REPO-INVARIANT   <file_tree>       identical for every task of one repo
#   2. PARTIALLY SHARED <relevant_files>  retrieved per task, with overlap
#                                         across tasks from one repository
#   3. TASK-UNIQUE      <issue>           shared by nothing
#
# Every repo-invariant block therefore precedes every task-specific block, and
# among the task-specific blocks the retrieved files precede the issue.
CONTEXT_TEMPLATE = """The repository at /testbed has been indexed for you.

<file_tree>
{tree}
</file_tree>

<relevant_files>
{files}
</relevant_files>

Fix the following issue.

<issue>
{problem_statement}
</issue>

Verify the behaviour yourself before editing; the retrieval above is a starting
point, not ground truth."""

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "when", "then", "have",
    "does", "not", "but", "are", "was", "were", "which", "should", "would",
    "will", "can", "issue", "problem", "error", "bug", "code", "file", "line",
    "expected", "actual", "description", "example", "following", "here", "there",
}
_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b")


def extract_keywords(problem_statement: str, limit: int = 12) -> list[str]:
    """Identifiers from the issue text, most frequent first.

    A frequency heuristic: retrieval builds a large prompt shared across the
    group, as SWE agents commonly do; it does not aim at retrieval quality.
    """
    counts: dict[str, int] = {}
    for m in _IDENT_RE.finditer(problem_statement):
        w = m.group(0)
        if w.lower() in _STOPWORDS:
            continue
        counts[w] = counts.get(w, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], -len(kv[0])))
    return [w for w, _ in ranked[:limit]]


SOURCE_EXTENSIONS = (
    "py", "go", "js", "ts", "jsx", "tsx", "java", "rb", "rs", "c", "h",
    "cc", "cpp", "hpp", "cs", "php", "scala", "kt", "swift", "m", "sh",
)


def detect_source_extensions(sandbox: ExecSandbox, top: int = 3) -> list[str]:
    """Find which languages a repository is actually written in.

    Language is detected from the checkout rather than assumed so retrieval
    does not silently become empty for non-Python repositories.
    """
    out = sandbox.exec(
        "find . -type f -not -path './.git/*' -not -path './build/*' "
        rf"| grep -Eo '\.({'|'.join(SOURCE_EXTENSIONS)})$' "
        "| sort | uniq -c | sort -rn | head -" + str(top)
    )
    exts = []
    for line in out.splitlines():
        parts = line.split(".")
        if len(parts) >= 2 and parts[-1].strip():
            exts.append(parts[-1].strip())
    return exts or ["py"]


ORDERINGS = ("grep", "path")
"""How the retrieved file blocks are ordered inside <relevant_files>.

The file set is the same under both values; only block order changes.

  "grep"  grep-count order, as the retriever returned it (the default).
  "path"  sorted by display path.
"""


def _display_path(path: str) -> str:
    """Path as it appears in a file block header: without leading ``./``."""
    while path.startswith("./"):
        path = path[2:]
    return path


@dataclass(frozen=True)
class RetrievedFile:
    """One retrieved source file, already truncated to the caller's budget."""

    path: str
    """Display path, as rendered in the block header."""
    body: str


def order_retrieved_files(
    files: Sequence[RetrievedFile], *, ordering: str = "grep"
) -> list[RetrievedFile]:
    """Reorder one task's retrieved files; see :data:`ORDERINGS`.

    The set of files is preserved exactly.
    """
    if ordering == "grep":
        return list(files)
    if ordering == "path":
        return sorted(files, key=lambda f: f.path)
    raise ValueError(f"unknown ordering {ordering!r}; expected one of {ORDERINGS}")


def render_files_block(files: Sequence[RetrievedFile]) -> str:
    """The text that goes between the <relevant_files> tags."""
    body = "\n\n".join(f"--- {f.path} ---\n{f.body}" for f in files)
    return body or "(no files matched)"


def retrieve_repo_files(
    sandbox: ExecSandbox,
    problem_statement: str,
    *,
    max_tree_entries: int = 400,
    max_files: int = 6,
    max_file_chars: int = 8000,
    extensions: list[str] | None = None,
) -> tuple[str, list[RetrievedFile]]:
    """The file tree and the keyword-retrieved files, in grep-count order."""
    exts = extensions or detect_source_extensions(sandbox)
    include = " ".join(shlex.quote(f"--include=*.{e}") for e in exts)
    name_expr = " -o ".join(f"-name {shlex.quote(f'*.{e}')}" for e in exts)

    tree = sandbox.exec(
        f"find . \\( {name_expr} \\) -not -path './.git/*' -not -path './build/*' "
        f"| sed 's|^\\./||' | sort | head -{max_tree_entries}"
    ).strip()

    keywords = extract_keywords(problem_statement)
    files: list[RetrievedFile] = []
    if keywords:
        pattern = "|".join(re.escape(k) for k in keywords)
        ranked = sandbox.exec(
            f"grep -rEc {shlex.quote(pattern)} {include} . 2>/dev/null "
            "| grep -v ':0$' | sort -t: -k2 -rn | head -"
            f"{max_files} | cut -d: -f1"
        ).strip()
        for path in [p for p in ranked.splitlines() if p.strip()][:max_files]:
            body = command_output(sandbox, f"cat {shlex.quote(path)} 2>/dev/null")
            if body.text.strip():
                files.append(
                    RetrievedFile(
                        path=_display_path(path),
                        body=truncate_observation(body, max_file_chars),
                    )
                )

    return tree or "(empty)", files


def build_repo_context(
    sandbox: ExecSandbox,
    problem_statement: str,
    *,
    max_tree_entries: int = 400,
    max_files: int = 6,
    max_file_chars: int = 8000,
    extensions: list[str] | None = None,
    ordering: str = "grep",
) -> tuple[str, str]:
    """Gather a file tree and the most relevant sources from the container.

    Called once per group, not once per rollout: every rollout on a task
    starts from the same repository state, so the resulting text is identical
    across the group and lands in the shared prefix.

    ``ordering`` selects how the retrieved blocks are arranged inside
    <relevant_files>; see :data:`ORDERINGS`.
    """
    if ordering not in ORDERINGS:
        raise ValueError(f"unknown ordering {ordering!r}; expected one of {ORDERINGS}")
    tree, files = retrieve_repo_files(
        sandbox,
        problem_statement,
        max_tree_entries=max_tree_entries,
        max_files=max_files,
        max_file_chars=max_file_chars,
        extensions=extensions,
    )
    return tree, render_files_block(order_retrieved_files(files, ordering=ordering))


# The closing fence is a line holding only ``` (trailing blanks allowed), so a
# command whose body contains ``` inside a line, such as a heredoc that writes
# Markdown, is not cut short.
ACTION_RE = re.compile(r"```bash[ \t]*\n(.*?)^```[ \t]*$", re.DOTALL | re.MULTILINE)
TEST_FRAMEWORKS = ("pytest", "go")


@dataclass
class Task:
    instance_id: str
    problem_statement: str
    image_name: str
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    repo: str = ""
    bug_patch: str = ""
    """SWE-smith's ``patch`` field: the diff that *introduces* the bug.

    One image serves many instances of the same repository, so the image ships
    the clean checkout and the per-instance bug has to be applied at setup. This
    is the opposite of the SWE-bench convention, where ``patch`` is the gold fix
    and the checkout already contains the bug -- and getting it backwards is
    silent: FAIL_TO_PASS passes on the untouched repo, so every rollout scores
    1.0 without doing anything.
    """
    test_framework: str = "pytest"
    """Structured verifier for the test IDs: ``"pytest"`` (JUnit XML per
    selector batch) or ``"go"`` (one ``go test -json ./...`` run graded by
    test name)."""
    requires_network: bool = False
    """The task's tests need socket families beyond AF_UNIX (TCP loopback).
    Set on every sandbox the engine and the sandbox pool create for the
    task, before ``start`` (``engine.new_sandbox``)."""

    def __post_init__(self) -> None:
        if self.test_framework not in TEST_FRAMEWORKS:
            raise ValueError(
                f"unknown test_framework {self.test_framework!r}; "
                f"expected one of {TEST_FRAMEWORKS}"
            )


@dataclass
class Step:
    action: str
    observation: str
    action_token_ids: list[int] = field(default_factory=list)
    action_logprobs: list[float] = field(default_factory=list)
    observation_token_ids: list[int] = field(default_factory=list)
    generation_s: float = 0.0
    tool_s: float = 0.0
    prompt_len: int = 0
    """Context length at the moment this action was generated."""


@dataclass
class Trajectory:
    task_id: str
    group_id: int
    rollout_id: int
    prompt_token_ids: list[int] = field(default_factory=list)
    """The shared task prompt. Identical for every rollout in the group."""
    steps: list[Step] = field(default_factory=list)
    reward: float = 0.0
    finish_reason: str = ""
    completed_at: float | None = None
    """Monotonic source timestamp immediately before the completion callback."""
    final_diff: str = ""
    """Changes from the sealed buggy commit, captured before grading.

    The baseline is captured before policy actions, so this includes both
    committed and uncommitted work. A reward with an empty diff means the tests
    passed without a repository change.
    """
    verdict_reason: str = ""
    """Structured grader reason, retained so infrastructure failures are visible."""

    @property
    def token_ids(self) -> list[int]:
        """Full append-only trajectory: prompt, then action/observation turns."""
        out = list(self.prompt_token_ids)
        for s in self.steps:
            out.extend(s.action_token_ids)
            out.extend(s.observation_token_ids)
        return out

    @property
    def action_mask(self) -> list[bool]:
        """True exactly on tokens the policy generated -- the trainable ones."""
        out = [False] * len(self.prompt_token_ids)
        for s in self.steps:
            out.extend([True] * len(s.action_token_ids))
            out.extend([False] * len(s.observation_token_ids))
        return out

    @property
    def logprobs(self) -> list[float]:
        out = [0.0] * len(self.prompt_token_ids)
        for s in self.steps:
            out.extend(s.action_logprobs)
            out.extend([0.0] * len(s.observation_token_ids))
        return out

    @property
    def n_action_tokens(self) -> int:
        return sum(len(s.action_token_ids) for s in self.steps)

    @property
    def generation_s(self) -> float:
        return sum(s.generation_s for s in self.steps)

    @property
    def tool_s(self) -> float:
        return sum(s.tool_s for s in self.steps)


def truncate_observation(text: str | CommandOutput, max_chars: int = 4000) -> str:
    """Keep both ends: the head usually says what ran, the tail says what broke.

    A ``CommandOutput`` whose capture kept only the ends of a long stream
    renders exactly as its complete text would: the same ends, the same
    count of elided characters.
    """
    if isinstance(text, CommandOutput):
        if text.complete:
            return truncate_observation(text.text, max_chars)
        half = max_chars // 2
        if half < 1 or text.chars <= max_chars:
            raise ValueError(
                f"an incomplete output of {text.chars} chars cannot render at {max_chars}"
            )
        return (
            f"{text.head(half)}\n\n... <{text.chars - max_chars} chars elided> ...\n\n"
            f"{text.tail(half)}"
        )
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return f"{text[:half]}\n\n... <{len(text) - max_chars} chars elided> ...\n\n{text[-half:]}"


# Up to this many characters an observation is tokenized whole, so the
# elision marker counts tokens exactly; a longer one is tokenized in a
# window at each end.
_WHOLE_TOKENIZE_CHARS = 1 << 16
# Tokens tokenized past a window's needed count, so the counted tokens
# lie clear of the window's cut, where tokenization can differ from the
# whole text's.
_WINDOW_MARGIN_TOKENS = 64
_INITIAL_CHARS_PER_TOKEN = 4
_WINDOW_GROWTH = 4
# A byte-level token can end inside a multibyte character; at most three
# tokens are dropped to reach a clean character boundary.
_MAX_SEAM_TRIM = 3


def _window_tokens(
    text: str, count: int, encode: Callable[[str], list[int]], *, from_end: bool
) -> tuple[list[int], int]:
    """Tokens of a char window at one end of ``text``, and the window's width.

    The window grows until it holds ``count`` tokens plus a margin, or
    covers ``text``; a width of at least ``len(text)`` means all of it.
    """
    width = _INITIAL_CHARS_PER_TOKEN * (count + _WINDOW_MARGIN_TOKENS)
    while True:
        if width >= len(text):
            return encode(text), len(text)
        ids = encode(text[-width:] if from_end else text[:width])
        if len(ids) >= count + _WINDOW_MARGIN_TOKENS:
            return ids, width
        width *= _WINDOW_GROWTH


def _clean_end(
    ids: list[int], text: str, decode: Callable[[Sequence[int]], str], *, from_end: bool
) -> str:
    """``decode(ids)``, trimmed off a split character at its cut when possible."""
    matches = text.endswith if from_end else text.startswith
    for trim in range(_MAX_SEAM_TRIM + 1):
        kept = ids[trim:] if from_end else ids[: len(ids) - trim]
        piece = decode(kept)
        if matches(piece):
            return piece
    return decode(ids)


def truncate_observation_tokens(
    output: str | CommandOutput,
    encode: Callable[[str], list[int]],
    decode: Callable[[Sequence[int]], str],
    max_tokens: int = 4096,
) -> str:
    """Keep the first and last ``max_tokens // 2`` tokens of an observation.

    An observation of at most ``max_tokens`` tokens is returned unchanged.
    A text of at most ``_WHOLE_TOKENIZE_CHARS`` characters is tokenized
    whole and the marker between the ends counts the elided tokens. A
    longer one has each end tokenized in a char window that grows only
    until it holds its half, so the cost is bounded by the windows and not
    the output's length; the marker then counts the elided characters,
    unless the two windows together cover the text. An incomplete
    ``CommandOutput`` takes each end from what its capture kept and its
    marker counts characters.
    """
    if max_tokens < 2:
        raise ValueError(f"max_tokens must be at least 2, not {max_tokens}")
    head_count = max_tokens // 2
    tail_count = max_tokens - head_count
    if isinstance(output, CommandOutput) and not output.complete:
        head_text, tail_text = output.kept_head(), output.kept_tail()
        total_chars = output.chars
        head_ids, _ = _window_tokens(head_text, head_count, encode, from_end=False)
        tail_ids, _ = _window_tokens(tail_text, tail_count, encode, from_end=True)
    else:
        text = output.text if isinstance(output, CommandOutput) else output
        head_text = tail_text = text
        total_chars = len(text)
        if len(text) <= _WHOLE_TOKENIZE_CHARS:
            head_ids, head_width = encode(text), len(text)
        else:
            head_ids, head_width = _window_tokens(
                text, head_count, encode, from_end=False
            )
        whole: list[int] | None = head_ids if head_width >= len(text) else None
        if whole is None:
            tail_ids, tail_width = _window_tokens(
                text, tail_count, encode, from_end=True
            )
            if head_width + tail_width >= len(text):
                whole = encode(text)
        if whole is not None:
            if len(whole) <= max_tokens:
                return text
            head = _clean_end(whole[:head_count], text, decode, from_end=False)
            tail = _clean_end(whole[-tail_count:], text, decode, from_end=True)
            elided = len(whole) - head_count - tail_count
            return f"{head}\n\n... <{elided} tokens elided> ...\n\n{tail}"
    head = _clean_end(head_ids[:head_count], head_text, decode, from_end=False)
    tail = _clean_end(tail_ids[-tail_count:], tail_text, decode, from_end=True)
    elided = total_chars - len(head) - len(tail)
    return f"{head}\n\n... <{elided} chars elided> ...\n\n{tail}"


def parse_action(text: str) -> str | None:
    """The body of the first ```bash block, stripped, or None without one."""
    m = ACTION_RE.search(text)
    return m.group(1).strip() if m else None

