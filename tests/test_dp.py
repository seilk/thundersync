"""Data-parallel gates: DP must change NOTHING but the wall clock.

Each test shells out `torchrun --nproc-per-node=2 tests/_dp_worker.py` and
asserts on its exit code plus a JSON verdict rank 0 writes -- collectives
cannot run inside a single pytest process, and a subprocess keeps NCCL's
lifecycle (and any hang, via timeout) isolated from the suite.

The three gates, each with the failure it guards against:

* equality -- 2 ranks x 1 group each through DataParallelThunderSync.step() equals a
  single-process streamed run over BOTH groups with grads averaged over the
  2 groups (the mean-over-groups semantics dp.py documents). This is the gate
  that proves DP changes nothing: same update, to gradient tolerance.
* broadcast -- ranks constructed from different seeds are bitwise identical
  after init_from_env (and identical to rank 0's init specifically).
* guard -- an optimizer step with open groups raises on the offending rank
  BEFORE any collective (loud, no deadlock), and the trainer recovers.

Fewer than two visible GPUs skips the file.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "tests" / "_dp_worker.py"
NPROC = 2
TIMEOUT_S = 600


def _eligible_devices() -> list[str]:
    """Device ids torchrun may use: the visible devices."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        return [d.strip() for d in cvd.split(",") if d.strip()]
    # device_count() queries the driver without creating a context
    return [str(i) for i in range(torch.cuda.device_count())]


DEVICES = _eligible_devices()

pytestmark = pytest.mark.skipif(
    len(DEVICES) < NPROC,
    reason=f"needs {NPROC} GPUs, have {DEVICES}",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_worker(
    mode: str,
    tmp_path: Path,
    extra_env: dict[str, str] | None = None,
) -> dict:
    out = tmp_path / f"dp_{mode}.json"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(DEVICES[:NPROC])
    env.setdefault("NCCL_DEBUG", "WARN")
    if extra_env:
        env.update(extra_env)
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc-per-node={NPROC}",
        "--master-addr=127.0.0.1",
        f"--master-port={_free_port()}",
        str(WORKER),
        "--mode",
        mode,
        "--out",
        str(out),
    ]
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    assert proc.returncode == 0, (
        f"torchrun[{mode}] exited {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout[-4000:]}\n"
        f"--- stderr ---\n{proc.stderr[-4000:]}"
    )
    assert out.exists(), f"worker[{mode}] exited 0 but wrote no verdict"
    verdict = json.loads(out.read_text())
    assert verdict["ok"] is True and verdict["mode"] == mode
    return verdict["result"]


def test_dp_step_equals_single_process_run(tmp_path):
    """THE equality gate: DP over {rank r streams group r} == one process
    streaming both groups, grads averaged over groups, same SGD step."""
    result = _run_worker("equality", tmp_path)
    assert result["replicas_identical"], "ranks diverged after an averaged step"
    assert result["n_params_checked"] > 5
    assert result["n_groups_total"] == NPROC
    # the worker already assert_close'd at rtol 2e-3 / atol 1e-5; the recorded
    # worsts document how much margin the gate passed with
    assert result["max_abs_update_diff"] <= 1e-4
    assert result["stats"]["world_size"] == NPROC
    assert result["stats"]["n_local_groups"] == 1
    assert result["stats"]["grad_clip"] == 0.05
    assert result["stats"]["global_grad_norm"] >= 0.0


def test_dp_step_equals_single_process_run_device_resident(tmp_path):
    """The equality gate again with device-resident sources on both sides:
    2-rank DP folds on each rank's parameter device and must still equal the
    single-process device-resident run at the same tolerances."""
    result = _run_worker(
        "equality",
        tmp_path,
        extra_env={"THUNDERSYNC_DP_WORKER_SOURCE_OFFLOAD_MODE": "device_resident"},
    )
    assert result["replicas_identical"], "ranks diverged after an averaged step"
    assert result["n_params_checked"] > 5
    assert result["n_groups_total"] == NPROC
    assert result["max_abs_update_diff"] <= 1e-4
    assert result["stats"]["world_size"] == NPROC


def test_dp_unequal_group_counts_equal_single_process_run(tmp_path):
    result = _run_worker("unequal", tmp_path)
    assert result["replicas_identical"]
    assert result["n_groups_total"] == 3
    assert result["max_abs_update_diff"] <= 1e-4
    assert result["stats"]["group_counts_by_rank"] == [2, 1]
    assert result["stats"]["n_global_groups"] == 3


def test_init_from_env_broadcasts_rank0_state(tmp_path):
    result = _run_worker("broadcast", tmp_path)
    assert result["world"] == NPROC
    assert result["hash"]  # rank 0's init hash, matched by every rank


def test_step_with_open_groups_raises_per_rank(tmp_path):
    result = _run_worker("guard", tmp_path)
    assert result["raised"] and result["recovered"]


def test_two_sequential_batches_rearm_and_shutdown(tmp_path):
    result = _run_worker("fresh_batches", tmp_path)
    assert result["replicas_identical"]
    assert result["run_was_rearmed"]
    assert result["steps"] == 2
    assert result["local_groups_closed_after_each_step"] == [0, 0]
    assert result["shutdown"] is True
