"""torchrun worker for tests/grpo/test_work_claims_store.py -- NOT a pytest file.

Launched as ``torchrun --nproc-per-node=N tests/_claims_worker.py --mode M
--out DIR`` on CPU (gloo). Every rank writes ``DIR/rank-<r>.json``; any rank
raising makes torchrun exit nonzero.

Modes:
  race      every rank claims every one of 500 items through the default
            store's compare_set, all at once; each writes what it won.
  protocol  the claiming admission on two ranks over the default store:
            rank 0 has two groups of busy final work, rank 1 one small group;
            each runs its step to its end and enters dist.barrier(), three
            steps in a row; each writes what it ran, ceded and claimed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "grpo"))

from thundersync.scheduling.work_claims import StoreWorkClaims  # noqa: E402


def _claims(rank: int, world: int, namespace: str) -> StoreWorkClaims:
    return StoreWorkClaims(
        dist.distributed_c10d._get_default_store(),
        rank=rank,
        world_size=world,
        namespace=namespace,
    )


def mode_race(rank: int, world: int) -> dict:
    claims = _claims(rank, world, "race")
    claims.begin_step(1)
    dist.barrier()
    won = [item for item in range(500) if claims._try_claim(item)]
    dist.barrier()
    return {"won": won, "claim_attempts": claims.report()["claim_attempts"]}


def mode_protocol(rank: int, world: int) -> dict:
    from test_work_claims_admission import ClaimTrainer, Recorder, _claiming, _run_rank

    claims = _claims(rank, world, "protocol")
    steps = []
    for step in range(3):
        if rank == 0:
            trainer = ClaimTrainer(
                {
                    0: {step * 1000 + i: 100 * (i % 8 + 1) for i in range(8)},
                    2: {step * 1000 + 100 + i: 100 * (i % 8 + 1) for i in range(8)},
                },
                seconds_per_token=5e-5,
                cap=1000,
            )
            group_size = 8
        else:
            trainer = ClaimTrainer(
                {1: {step * 1000 + 500 + i: 50 for i in range(4)}},
                seconds_per_token=5e-5,
                cap=1000,
            )
            group_size = 4
        recorder = Recorder()
        _run_rank(_claiming(trainer, recorder, claims, group_size), trainer)
        # the driver's pre-update barrier: a rank still claiming would hang it
        dist.barrier()
        steps.append(
            {
                "owned": sorted(trainer.tokens),
                "ran_own": sorted(trainer.ran_own),
                "ran_claimed": sorted(trainer.ran_claimed),
                "ceded": sorted(trainer.ceded),
                "closed_groups": sorted(trainer.closed_groups),
                "report": trainer.claims_report,
            }
        )
    return {"steps": steps}


MODES = {"race": mode_race, "protocol": mode_protocol}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=sorted(MODES))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    result = MODES[args.mode](rank, world)
    (args.out / f"rank-{rank}.json").write_text(json.dumps(result))
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    raise SystemExit(main())
