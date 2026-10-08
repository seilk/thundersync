"""An opt-in torch.profiler window around selected close packs.

Diagnosis only. ``THUNDERSYNC_GRPO_PROFILE_DIR`` names a directory; while it is
unset, :meth:`ClosePackProfiler.from_env` returns None and a close pays one
``is None`` test. When set, the first ``THUNDERSYNC_GRPO_PROFILE_PACKS``
(default 1) close packs of step ``THUNDERSYNC_GRPO_PROFILE_STEP`` (default 3)
on rank ``THUNDERSYNC_GRPO_PROFILE_RANK`` (default 0) whose longest branch has
at least ``THUNDERSYNC_GRPO_PROFILE_MIN_TOKENS`` tokens run under ``torch.profiler`` (CPU and CUDA activities, no shapes;
``THUNDERSYNC_GRPO_PROFILE_STACK=1`` adds Python stacks). Each window writes
``rank<r>-step<s>-pack<k>.json.gz``, a Chrome trace, into the directory.
The window changes no value the close computes.

The token floor defaults to ``DEFAULT_PROFILE_MIN_TOKENS``, a starting point
that selects long branches; tune it to the branch lengths of the workload.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch

PROFILE_DIR_ENV = "THUNDERSYNC_GRPO_PROFILE_DIR"
DEFAULT_PROFILE_MIN_TOKENS = 14000


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


class ClosePackProfiler:
    """Selects close packs and runs one profiler window around each."""

    def __init__(
        self,
        directory: Path,
        *,
        step: int,
        rank: int,
        min_tokens: int,
        packs: int,
        with_stack: bool,
    ) -> None:
        self.directory = directory
        self.step = step
        self.rank = rank
        self.min_tokens = min_tokens
        self.packs = packs
        self.with_stack = with_stack
        self.taken = 0
        self._active: Any | None = None
        self._active_record: dict[str, Any] | None = None

    @classmethod
    def from_env(cls) -> ClosePackProfiler | None:
        directory = os.environ.get(PROFILE_DIR_ENV, "").strip()
        if not directory:
            return None
        return cls(
            Path(directory),
            step=_int_env("THUNDERSYNC_GRPO_PROFILE_STEP", 3),
            rank=_int_env("THUNDERSYNC_GRPO_PROFILE_RANK", 0),
            min_tokens=_int_env(
                "THUNDERSYNC_GRPO_PROFILE_MIN_TOKENS", DEFAULT_PROFILE_MIN_TOKENS
            ),
            packs=_int_env("THUNDERSYNC_GRPO_PROFILE_PACKS", 1),
            with_stack=os.environ.get("THUNDERSYNC_GRPO_PROFILE_STACK", "") == "1",
        )

    def selects(self, *, rank: int, step: int, branch_tokens: list[int]) -> bool:
        return (
            self._active is None
            and self.taken < self.packs
            and rank == self.rank
            and step == self.step
            and max(branch_tokens, default=0) >= self.min_tokens
        )

    def start(
        self, *, rank: int, step: int, trajectory_ids: list[int], branch_tokens: list[int]
    ) -> None:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
            torch.cuda.synchronize()
        profiler = torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            with_stack=self.with_stack,
        )
        profiler.__enter__()
        self._active = profiler
        self._active_record = {
            "rank": rank,
            "step": step,
            "pack": self.taken,
            "trajectory_ids": list(trajectory_ids),
            "branch_tokens": list(branch_tokens),
            "started_at": time.perf_counter(),
        }

    def stop(self) -> None:
        profiler, record = self._active, self._active_record
        if profiler is None or record is None:
            return
        self._active = None
        self._active_record = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        record["window_s"] = time.perf_counter() - record["started_at"]
        profiler.__exit__(None, None, None)
        self.directory.mkdir(parents=True, exist_ok=True)
        stem = f"rank{record['rank']}-step{record['step']}-pack{record['pack']}"
        profiler.export_chrome_trace(str(self.directory / f"{stem}.json.gz"))
        (self.directory / f"{stem}.meta.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )
        self.taken += 1
