"""Bounded, streaming capture of a command's output.

A sandbox command's output is policy-controlled, and nothing bounds it:
``yes``, ``cat /dev/zero`` or a runaway loop writes as fast as a pipe
drains. ``Popen.communicate`` keeps every byte in the calling process
until the command exits or its deadline passes, and at pipe speed that is
gigabytes per second of the caller's memory.

``run_bounded`` drains both pipes as they fill and keeps, per stream:

* every byte while the stream holds at most ``keep_bytes``. Such a stream
  decodes exactly as ``Popen(text=True, encoding="utf-8",
  errors="replace")`` decodes it: one whole-buffer decode, then ``\\r\\n``
  and ``\\r`` to ``\\n``.
* beyond ``keep_bytes``, its first ``keep_bytes // 2`` and last
  ``keep_bytes - keep_bytes // 2`` bytes and the exact number of
  characters the whole stream decodes to; the bytes between are read and
  discarded.

A command whose two streams together write more than ``kill_bytes`` has
its process group killed; exactly ``kill_bytes`` bytes stand as its
output, so what is kept depends on the byte stream, not on when the kill
lands. A ``MemoryGuard`` reads the calling process's resident set while a
command runs and kills the command, never the caller, once the resident
set is over the guard's budget; a command is not started while it is. A
``write_guard`` (``thundersync.rollout.scratch_guard.ScratchGuard``) measures the
command's sandbox scratch at its own interval while the command runs and
kills the command once the scratch has grown past its budget.

``CommandOutput`` is what a sandbox's ``exec_output`` returns: the text's
parts (stdout, stderr, then any note such as a timeout), each with its
head, tail and exact character count. ``thundersync.rollout.agent.
truncate_observation``, the character rule, renders an incomplete output
exactly as it renders the complete text. The rollout engine renders tool
observations with ``thundersync.rollout.agent.truncate_observation_tokens``,
which keeps the same leading and trailing tokens of an incomplete output
as of its complete text whenever each kept end holds at least that many
tokens; the default ``keep_bytes`` keeps 8 MiB per end, far above the
2,048 tokens per end it retains. Its elision marker can count only what
the capture kept, so it states the elided characters for an incomplete
output.
"""

from __future__ import annotations

import codecs
import io
import os
import selectors
import signal
import subprocess
import time
from collections import deque
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

__all__ = [
    "BoundedCompletedProcess",
    "BoundedRun",
    "CapturedText",
    "CommandOutput",
    "DEFAULT_KEEP_BYTES",
    "DEFAULT_KILL_BYTES",
    "DEFAULT_RSS_BUDGET_FRACTION",
    "MemoryGuard",
    "OutputLimits",
    "StreamCapture",
    "WriteGuard",
    "command_output",
    "completed_process",
    "decode_like_popen",
    "default_rss_budget",
    "kill_process_group",
    "minimum_keep_bytes",
    "resident_set_bytes",
    "run_bounded",
    "visible_memory_bytes",
]

# Kept whole per stream: about seven times the longest tool output any
# recorded rollout produced (2.3 M characters).
DEFAULT_KEEP_BYTES = 16 << 20
# Read from a command, both streams together, before it is killed.
DEFAULT_KILL_BYTES = 256 << 20
# A rollout worker's resident-set budget as a fraction of the memory the
# host (or the worker's memory cgroup) makes visible; never below the floor.
DEFAULT_RSS_BUDGET_FRACTION = 0.125
MINIMUM_RSS_BUDGET_BYTES = 4 << 30

_READ_BYTES = 1 << 16
_GUARD_INTERVAL_S = 0.2
_GUARD_INTERVAL_BYTES = 8 << 20
# A UTF-8 sequence has at most three continuation bytes after its lead.
_MAX_CONTINUATION = 3
_UNLIMITED_CGROUP = 1 << 60

_Selector = getattr(selectors, "PollSelector", selectors.SelectSelector)


def kill_process_group(pid: int) -> None:
    """Kill every member of the process group led by ``pid``."""
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def decode_like_popen(data: bytes) -> str:
    """The text ``Popen(text=True, encoding="utf-8", errors="replace")`` reads."""
    return data.decode("utf-8", "replace").replace("\r\n", "\n").replace("\r", "\n")


def _text_decoder() -> io.IncrementalNewlineDecoder:
    return io.IncrementalNewlineDecoder(
        codecs.getincrementaldecoder("utf-8")("replace"), translate=True
    )


def _decoded_prefix(data: bytes) -> str:
    """A prefix of the stream's text: incomplete trailing input is held back."""
    return _text_decoder().decode(data, final=False)


def _decoded_suffix(data: bytes) -> str:
    """A suffix of the stream's text, from a cut at an arbitrary byte.

    Up to three leading continuation bytes may belong to a sequence that
    began before the cut. Past them the whole-stream decode restarts too:
    a byte that is not a continuation byte always starts a new sequence,
    and a fourth continuation byte cannot belong to a sequence that began
    before the cut. A ``\\n`` whose ``\\r`` fell before the cut is the
    same one character either way.
    """
    skip = 0
    while skip < min(_MAX_CONTINUATION, len(data)) and 0x80 <= data[skip] < 0xC0:
        skip += 1
    return _text_decoder().decode(data[skip:], final=True)


def minimum_keep_bytes(max_chars: int) -> int:
    """The least ``keep_bytes`` that renders any output exactly at ``max_chars``.

    A spilled stream keeps ``keep_bytes // 2`` bytes at each end. Its head
    may hold back three bytes of an incomplete sequence and its tail drop
    three continuation bytes, and a character takes at most four bytes, so
    each end then holds at least ``max_chars // 2`` characters, and a
    spilled stream is longer than ``max_chars``.
    """
    return 8 * (max_chars // 2) + 8


@dataclass(frozen=True)
class OutputLimits:
    """Per-command capture bounds; see the module docstring."""

    keep_bytes: int = DEFAULT_KEEP_BYTES
    kill_bytes: int = DEFAULT_KILL_BYTES

    def __post_init__(self) -> None:
        if self.keep_bytes < 2:
            raise ValueError(f"keep_bytes must be at least 2, not {self.keep_bytes}")
        if self.kill_bytes < self.keep_bytes:
            raise ValueError(
                f"kill_bytes ({self.kill_bytes}) is below keep_bytes ({self.keep_bytes})"
            )

    def keep_for(self, override: int | None) -> int:
        """``keep_bytes``, or a per-call override no larger than the kill cap."""
        return self.keep_bytes if override is None else min(override, self.kill_bytes)


class CapturedText:
    """One part of a command's text: whole, or its head, tail and length.

    A whole part keeps its raw bytes and decodes them on first use; a
    literal part (a note such as a timeout) is text from the start.
    """

    def __init__(
        self,
        *,
        raw: bytes | None = None,
        literal: str | None = None,
        head: str = "",
        tail: str = "",
        chars: int = 0,
        nbytes: int = 0,
    ) -> None:
        self.raw = raw
        self._literal = literal
        self._head = head
        self._tail = tail
        self._chars = chars
        self.nbytes = nbytes

    @classmethod
    def of_text(cls, text: str) -> "CapturedText":
        return cls(literal=text)

    @property
    def complete(self) -> bool:
        return self.raw is not None or self._literal is not None

    @cached_property
    def text(self) -> str:
        if self._literal is not None:
            return self._literal
        if self.raw is not None:
            return decode_like_popen(self.raw)
        raise ValueError("an incomplete part has no whole text")

    @property
    def chars(self) -> int:
        return len(self.text) if self.complete else self._chars

    def head(self, count: int) -> str:
        if self.complete:
            return self.text[:count]
        if count > len(self._head):
            raise ValueError(f"{count} head characters requested, {len(self._head)} kept")
        return self._head[:count]

    def tail(self, count: int) -> str:
        if count <= 0:
            return ""
        if self.complete:
            return self.text[-count:]
        if count > len(self._tail):
            raise ValueError(f"{count} tail characters requested, {len(self._tail)} kept")
        return self._tail[-count:]

    @property
    def kept_head(self) -> str:
        """The longest prefix of this part's text the capture kept."""
        return self.text if self.complete else self._head

    @property
    def kept_tail(self) -> str:
        """The longest suffix of this part's text the capture kept."""
        return self.text if self.complete else self._tail

    def rendered(self) -> str:
        """The whole text, or its kept ends around a count of the rest."""
        if self.complete:
            return self.text
        elided = self._chars - len(self._head) - len(self._tail)
        return f"{self._head}\n\n... <{elided} chars elided> ...\n\n{self._tail}"


class StreamCapture:
    """One pipe's bytes: all of them up to ``keep_bytes``, then head and tail."""

    def __init__(self, keep_bytes: int) -> None:
        self._keep = keep_bytes
        self._head_limit = keep_bytes // 2
        self._tail_limit = keep_bytes - keep_bytes // 2
        self._buffer = bytearray()
        self._head: bytes | None = None
        self._tail: deque[bytes] = deque()
        self._tail_bytes = 0
        self._decoder: io.IncrementalNewlineDecoder | None = None
        self._chars = 0
        self.nbytes = 0

    @property
    def spilled(self) -> bool:
        return self._head is not None

    def feed(self, data: bytes) -> None:
        if not data:
            return
        self.nbytes += len(data)
        if self._head is None:
            self._buffer += data
            if len(self._buffer) > self._keep:
                self._spill()
            return
        self._count(data)
        self._append_tail(data)

    def _spill(self) -> None:
        # In read-sized slices, so the spill holds no second copy of the
        # buffer and no text the size of it.
        with memoryview(self._buffer) as view:
            self._head = bytes(view[: self._head_limit])
            self._decoder = _text_decoder()
            for start in range(0, len(view), _READ_BYTES):
                self._count(view[start : start + _READ_BYTES])
            tail_start = max(self._head_limit, len(view) - self._tail_limit)
            self._append_tail(bytes(view[tail_start:]))
        self._buffer = bytearray()

    def _count(self, data: bytes | memoryview) -> None:
        if self._decoder is None:
            raise RuntimeError("stream capture counted characters before spilling")
        self._chars += len(self._decoder.decode(data))

    def _append_tail(self, data: bytes) -> None:
        self._tail.append(data)
        self._tail_bytes += len(data)
        while self._tail and self._tail_bytes - len(self._tail[0]) >= self._tail_limit:
            self._tail_bytes -= len(self._tail.popleft())

    def finish(self) -> CapturedText:
        if self._head is None:
            return CapturedText(raw=bytes(self._buffer), nbytes=self.nbytes)
        if self._decoder is None:
            raise RuntimeError("stream capture spilled without a decoder")
        self._chars += len(self._decoder.decode(b"", final=True))
        tail = b"".join(self._tail)[-self._tail_limit :]
        return CapturedText(
            head=_decoded_prefix(self._head),
            tail=_decoded_suffix(tail),
            chars=self._chars,
            nbytes=self.nbytes,
        )


def resident_set_bytes() -> int | None:
    """This process's resident set, or None where /proc does not report it."""
    try:
        fields = Path("/proc/self/statm").read_text().split()
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def _cgroup_limits() -> list[int]:
    limits: list[int] = []
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return limits
    root = Path("/sys/fs/cgroup")
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _hierarchy, controllers, path = parts
        if controllers == "":
            # cgroup v2: every ancestor's memory.max bounds this one.
            current = root / path.lstrip("/")
            while True:
                limits.extend(_read_limit(current / "memory.max"))
                if current == root:
                    break
                current = current.parent
        elif "memory" in controllers.split(","):
            limits.extend(
                _read_limit(root / "memory" / path.lstrip("/") / "memory.limit_in_bytes")
            )
    return limits


def _read_limit(path: Path) -> list[int]:
    try:
        value = path.read_text().strip()
    except OSError:
        return []
    if not value.isdigit():
        return []
    limit = int(value)
    return [limit] if 0 < limit < _UNLIMITED_CGROUP else []


def visible_memory_bytes() -> int | None:
    """Host memory, or the memory cgroup's limit when that is lower."""
    candidates = _cgroup_limits()
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                candidates.append(int(line.split()[1]) * 1024)
                break
    except (OSError, ValueError, IndexError):
        pass
    return min(candidates) if candidates else None


def default_rss_budget(
    fraction: float = DEFAULT_RSS_BUDGET_FRACTION, *, visible: int | None = None
) -> int | None:
    """``fraction`` of the visible memory, at least the floor; None disables."""
    if fraction <= 0:
        return None
    if fraction > 1:
        raise ValueError(f"an RSS budget fraction is at most 1, not {fraction}")
    visible = visible_memory_bytes() if visible is None else visible
    if visible is None:
        return None
    return min(visible, max(int(visible * fraction), MINIMUM_RSS_BUDGET_BYTES))


class WriteGuard(Protocol):
    """What ``run_bounded`` reads of a sandbox scratch budget."""

    budget_bytes: int

    @property
    def check_interval_s(self) -> float: ...

    def over_budget(self) -> int | None: ...


class MemoryGuard:
    """A soft resident-set budget for the process that runs commands."""

    def __init__(
        self,
        budget_bytes: int,
        *,
        rss: Callable[[], int | None] = resident_set_bytes,
    ) -> None:
        if budget_bytes <= 0:
            raise ValueError(f"an RSS budget is positive, not {budget_bytes}")
        self.budget_bytes = budget_bytes
        self._rss = rss

    def over_budget(self) -> int | None:
        """The resident set when it exceeds the budget, else None."""
        rss = self._rss()
        return rss if rss is not None and rss > self.budget_bytes else None


@dataclass
class BoundedRun:
    """What ``run_bounded`` observed of one command."""

    args: list[str]
    returncode: int | None
    stdout: CapturedText
    stderr: CapturedText
    timed_out: bool = False
    # "output_cap", "memory_budget", "write_budget", or None.
    killed: str | None = None
    # The memory guard refused the command before it started.
    refused: bool = False
    kill_bytes: int = DEFAULT_KILL_BYTES
    rss_bytes: int | None = None
    budget_bytes: int | None = None
    elapsed_s: float = 0.0
    # The write guard's scratch growth at the kill, and its budget.
    write_bytes: int | None = None
    write_budget_bytes: int | None = None


def run_bounded(
    command: Sequence[str],
    *,
    timeout: float,
    keep_bytes: int = DEFAULT_KEEP_BYTES,
    kill_bytes: int = DEFAULT_KILL_BYTES,
    env: Mapping[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    memory_guard: MemoryGuard | None = None,
    write_guard: WriteGuard | None = None,
) -> BoundedRun:
    """Run ``command`` in a session of its own with bounded output capture.

    Like ``communicate``, the run waits for both pipes to close and then
    for the command to exit, all within one deadline. The command's process
    group is killed when the deadline passes, when the command writes more
    than ``kill_bytes``, when ``memory_guard`` finds the caller over its
    budget, when ``write_guard`` finds its scratch grown past its budget,
    and after the command exits, which takes any descendant that outlived
    it. Both guards are read while the pipes are open and while the run
    waits for the exit. ``write_guard`` is not read before the start: the
    caller that owns the sandbox checks it then. Any other exception kills
    the group and propagates.
    """
    OutputLimits(keep_bytes, kill_bytes)
    started = time.monotonic()
    args = list(command)
    if memory_guard is not None:
        rss = memory_guard.over_budget()
        if rss is not None:
            empty = CapturedText(raw=b"")
            return BoundedRun(
                args, None, empty, empty, refused=True, killed="memory_budget",
                kill_bytes=kill_bytes, rss_bytes=rss,
                budget_bytes=memory_guard.budget_bytes,
            )
    deadline = started + timeout
    captures = {"stdout": StreamCapture(keep_bytes), "stderr": StreamCapture(keep_bytes)}
    timed_out = False
    killed: str | None = None
    rss: int | None = None
    written: int | None = None
    next_memory_check = started + _GUARD_INTERVAL_S
    next_write_check = (
        started + write_guard.check_interval_s if write_guard is not None else None
    )

    def guard_wait(remaining: float) -> float:
        wait = remaining
        if memory_guard is not None:
            wait = min(wait, _GUARD_INTERVAL_S)
        if next_write_check is not None:
            wait = min(wait, max(0.0, next_write_check - time.monotonic()))
        return wait

    def write_check() -> str | None:
        nonlocal next_write_check, written
        if write_guard is None or time.monotonic() < next_write_check:
            return None
        written = write_guard.over_budget()
        next_write_check = time.monotonic() + write_guard.check_interval_s
        return "write_budget" if written is not None else None

    with subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=None if env is None else dict(env),
        cwd=cwd,
        start_new_session=True,
    ) as process:
        try:
            streams = {process.stdout: captures["stdout"], process.stderr: captures["stderr"]}
            total = 0
            checked_at = 0
            with _Selector() as selector:
                for pipe in streams:
                    selector.register(pipe, selectors.EVENT_READ)
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        timed_out = True
                        break
                    for key, _events in selector.select(guard_wait(remaining)):
                        data = os.read(key.fd, _READ_BYTES)
                        if not data:
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                            continue
                        room = kill_bytes - total
                        if len(data) > room:
                            streams[key.fileobj].feed(data[:room])
                            total = kill_bytes
                            killed = "output_cap"
                            break
                        streams[key.fileobj].feed(data)
                        total += len(data)
                    if killed is not None:
                        break
                    if memory_guard is not None and (
                        time.monotonic() >= next_memory_check
                        or total - checked_at >= _GUARD_INTERVAL_BYTES
                    ):
                        next_memory_check = time.monotonic() + _GUARD_INTERVAL_S
                        checked_at = total
                        rss = memory_guard.over_budget()
                        if rss is not None:
                            killed = "memory_budget"
                            break
                    killed = write_check()
                    if killed is not None:
                        break
            # Both pipes closed; the command may still run (a descendant
            # that closed its streams), so the guards are still read.
            while not timed_out and killed is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    process.wait(timeout=guard_wait(remaining))
                    break
                except subprocess.TimeoutExpired:
                    pass
                if memory_guard is not None and time.monotonic() >= next_memory_check:
                    next_memory_check = time.monotonic() + _GUARD_INTERVAL_S
                    rss = memory_guard.over_budget()
                    if rss is not None:
                        killed = "memory_budget"
                        break
                killed = write_check()
            # Every way out kills the group: the deadline, a guard, or the
            # exit, which takes any descendant that outlived the command.
            kill_process_group(process.pid)
            process.wait()
        except BaseException:
            # The leader is unreaped until wait(), so the group is still ours.
            kill_process_group(process.pid)
            process.wait()
            raise
    return BoundedRun(
        args,
        process.returncode,
        captures["stdout"].finish(),
        captures["stderr"].finish(),
        timed_out=timed_out,
        killed=killed,
        kill_bytes=kill_bytes,
        rss_bytes=rss,
        budget_bytes=None if memory_guard is None else memory_guard.budget_bytes,
        elapsed_s=time.monotonic() - started,
        write_bytes=written,
        write_budget_bytes=None if write_guard is None else write_guard.budget_bytes,
    )


class BoundedCompletedProcess(subprocess.CompletedProcess):
    """A ``CompletedProcess`` whose text came from a ``BoundedRun``."""

    def __init__(self, run: BoundedRun) -> None:
        super().__init__(
            run.args, run.returncode, run.stdout.rendered(), run.stderr.rendered()
        )
        self.bounded = run


def completed_process(run: BoundedRun, *, timeout: float) -> BoundedCompletedProcess:
    """``run`` as ``communicate`` reports it: a timeout raises, the rest returns.

    The raised ``TimeoutExpired`` carries the run as ``bounded_run``.
    """
    if run.timed_out:
        error = subprocess.TimeoutExpired(run.args, timeout)
        error.bounded_run = run  # type: ignore[attr-defined]
        raise error
    return BoundedCompletedProcess(run)


def _note(text: str) -> CapturedText:
    return CapturedText.of_text(text)


@dataclass(frozen=True)
class CommandOutput:
    """A sandbox command's text in parts, with what bounded its capture."""

    parts: tuple[CapturedText, ...]
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    returncode: int | None = None
    timed_out: bool = False
    killed: str | None = None
    # Set when the sandbox's scratch went past its write budget: what was
    # measured and what was removed (``ScratchGuard.event``).
    scratch: Mapping[str, Any] | None = None

    @classmethod
    def of_text(cls, text: str) -> "CommandOutput":
        return cls((_note(text),))

    @classmethod
    def timed_out_after(cls, timeout_s: Any, run: BoundedRun | None = None) -> "CommandOutput":
        """A sandbox deadline: the partial output is dropped, as it always was."""
        return cls(
            (_note(f"<command timed out after {timeout_s}s>"),),
            stdout_bytes=0 if run is None else run.stdout.nbytes,
            stderr_bytes=0 if run is None else run.stderr.nbytes,
            timed_out=True,
        )

    @classmethod
    def from_process(
        cls, result: subprocess.CompletedProcess, *, timeout_s: Any
    ) -> "CommandOutput":
        """A sandbox exec's text: stdout, stderr, then any note.

        Exit 124 or 137 appends the timeout note as it always did. A kill
        for the output cap or the memory budget appends a note saying so.
        A ``CompletedProcess`` that did not come from ``run_bounded`` is
        whole text by construction.
        """
        run = getattr(result, "bounded", None)
        if run is None:
            parts: tuple[CapturedText, ...] = (
                _note(result.stdout or ""), _note(result.stderr or "")
            )
            stdout_bytes = len((result.stdout or "").encode())
            stderr_bytes = len((result.stderr or "").encode())
            killed = None
        else:
            parts = (run.stdout, run.stderr)
            stdout_bytes, stderr_bytes = run.stdout.nbytes, run.stderr.nbytes
            killed = run.killed
        if run is not None and run.refused:
            parts = (
                _note(
                    "<command not run: the rollout worker's resident memory "
                    f"({run.rss_bytes} bytes) is over its {run.budget_bytes} byte budget>"
                ),
            )
        elif killed == "output_cap":
            parts += (
                _note(
                    f"<command output exceeded {run.kill_bytes} bytes; the command was killed>"
                ),
            )
        elif killed == "memory_budget":
            parts += (
                _note(
                    "<command killed: the rollout worker's resident memory "
                    f"({run.rss_bytes} bytes) exceeded its {run.budget_bytes} byte budget>"
                ),
            )
        elif killed == "write_budget":
            parts += (
                _note(
                    "<command killed: its sandbox's files grew by "
                    f"{run.write_bytes} bytes, past the {run.write_budget_bytes} "
                    "byte write budget>"
                ),
            )
        elif result.returncode in {124, 137}:
            parts += (_note(f"<command timed out after {timeout_s}s>"),)
        return cls(
            parts,
            stdout_bytes=stdout_bytes,
            stderr_bytes=stderr_bytes,
            returncode=result.returncode,
            killed=killed,
        )

    def with_scratch(
        self, note: str, event: Mapping[str, Any], *, refused: bool = False
    ) -> "CommandOutput":
        """This output after a write-budget breach: ``note`` follows its text.

        A refused command has no text of its own, so the note is all of it.
        """
        parts = (_note(note),) if refused else self.parts + (_note(note),)
        return replace(self, parts=parts, killed="write_budget", scratch=dict(event))

    @property
    def complete(self) -> bool:
        return all(part.complete for part in self.parts)

    @property
    def chars(self) -> int:
        return sum(part.chars for part in self.parts)

    @property
    def text(self) -> str:
        """The whole text, or each incomplete part's kept ends around a count."""
        return "".join(part.rendered() for part in self.parts)

    def head(self, count: int) -> str:
        pieces: list[str] = []
        for part in self.parts:
            if count <= 0:
                break
            piece = part.head(min(count, part.chars))
            pieces.append(piece)
            count -= len(piece)
        return "".join(pieces)

    def tail(self, count: int) -> str:
        pieces: list[str] = []
        for part in reversed(self.parts):
            if count <= 0:
                break
            piece = part.tail(min(count, part.chars))
            pieces.append(piece)
            count -= len(piece)
        return "".join(reversed(pieces))

    def kept_head(self) -> str:
        """The longest prefix of the text the capture kept."""
        pieces: list[str] = []
        for part in self.parts:
            pieces.append(part.kept_head)
            if not part.complete:
                break
        return "".join(pieces)

    def kept_tail(self) -> str:
        """The longest suffix of the text the capture kept."""
        pieces: list[str] = []
        for part in reversed(self.parts):
            pieces.append(part.kept_tail)
            if not part.complete:
                break
        return "".join(reversed(pieces))

    def evidence(self) -> dict[str, Any]:
        """What bounded the capture, for a trajectory record."""
        return {
            "stdout_bytes": self.stdout_bytes,
            "stderr_bytes": self.stderr_bytes,
            "chars": self.chars,
            "complete": self.complete,
            "timed_out": self.timed_out,
            "killed": self.killed,
            **({} if self.scratch is None else {"scratch": dict(self.scratch)}),
        }


def command_output(sandbox: Any, command: str, **kwargs: Any) -> CommandOutput:
    """``sandbox.exec_output(command)``, or its ``exec`` text as a whole output.

    ``exec_output`` is the bounded form both sandbox backends provide; a
    sandbox with only ``exec`` returns whole text, which is what it is.
    """
    exec_output = getattr(sandbox, "exec_output", None)
    if exec_output is not None:
        return exec_output(command, **kwargs)
    return CommandOutput.of_text(sandbox.exec(command))
