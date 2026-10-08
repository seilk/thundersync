"""Work claims over a real process-group store (CPU, gloo, torchrun).

* race: four processes claim the same 500 items at once through the
  default store's compare_set; every item has exactly one winner;
* protocol: two ranks run the claiming admission over the default store for
  three steps, each ending in dist.barrier(): every trajectory runs once on
  one rank, the idle rank claims part of the busy rank's final work, the
  owner cedes exactly that, and no rank is left claiming at the barrier.
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest


def _real_torch() -> bool:
    try:
        spec = importlib.util.find_spec("torch")
    except ValueError:
        return False
    if spec is None:
        return False
    import torch

    return hasattr(torch, "__version__") and hasattr(torch.distributed, "TCPStore")


pytestmark = pytest.mark.skipif(not _real_torch(), reason="needs a real torch")

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "tests" / "_claims_worker.py"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run(mode: str, nproc: int, tmp_path: Path) -> list[dict]:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    command = [
        sys.executable, "-m", "torch.distributed.run", f"--nproc-per-node={nproc}",
        "--master-addr=127.0.0.1", f"--master-port={_free_port()}",
        str(WORKER), "--mode", mode, "--out", str(tmp_path),
    ]
    proc = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    return [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(nproc)]


def test_every_item_has_exactly_one_winner_across_processes(tmp_path):
    results = _run("race", 4, tmp_path)
    won = [item for result in results for item in result["won"]]
    assert sorted(won) == list(range(500))
    # Every rank contested every item. CAS guarantees one winner, not fair
    # scheduling: a fast rank is allowed to win every item.
    assert all(result["claim_attempts"] == 500 for result in results)


def test_two_ranks_claim_and_reach_the_barrier(tmp_path):
    busy, idle = _run("protocol", 2, tmp_path)
    for step_busy, step_idle in zip(busy["steps"], idle["steps"], strict=True):
        everything = step_busy["owned"] + step_idle["owned"]
        ran = (
            step_busy["ran_own"] + step_busy["ran_claimed"]
            + step_idle["ran_own"] + step_idle["ran_claimed"]
        )
        assert sorted(ran) == sorted(everything)
        assert step_idle["ran_claimed"] and step_idle["ran_claimed"] == step_busy["ceded"]
        assert not step_busy["ran_claimed"] and not step_idle["ceded"]
        assert step_busy["closed_groups"] == [0, 2] and step_idle["closed_groups"] == [1]
        assert step_busy["report"]["done"] and step_idle["report"]["done"]
        assert step_idle["report"]["stolen"] == len(step_idle["ran_claimed"])
