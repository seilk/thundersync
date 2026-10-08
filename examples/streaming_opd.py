"""One OPD update from teacher-scored actions; no downloads required."""

import json
import sys
from pathlib import Path

import torch

# Run as `python examples/streaming_opd.py`: make the sibling helper importable
# even under `python -I`, which leaves the script's directory off sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _tiny_model import tiny_qwen3  # noqa: E402

from thundersync import OPDTurnStream, StreamingRun  # noqa: E402


def main():
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Eval mode, not train mode: turn_boundary_rebuild (below) replays turns
    # and requires model.eval() so the replay is deterministic.
    model = tiny_qwen3(device).eval()
    # The OPD objective is a divergence from the teacher, a loss to
    # decrease, so the optimizer minimizes (unlike the GRPO example).
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
    )
    optimizer.zero_grad()
    # boundary_cut: trajectories read a detached copy of the shared prompt's
    # state, so each backward stops at the prompt. stream_turns: every turn
    # is forwarded as it arrives, so its teacher-scored loss can backward
    # at once. turn_boundary_rebuild: each turn's outgoing state is cut too;
    # closing the trajectory rebuilds the turns in reverse and carries the
    # later turns' gradients back into the earlier ones.
    run = StreamingRun(model, boundary_cut=True, stream_turns=True, turn_boundary_rebuild=True)
    objective = OPDTurnStream(run, model)

    # Group 0 has prompt tokens [1, 2, 3]; trajectory 0 arrives in two turns.
    # Token ids are arbitrary ids below the tiny model's vocabulary size.
    objective.open_group(0, [1, 2, 3])
    # Replace these illustrative scores with the teacher's log-probabilities
    # for exactly the generated tokens marked True.
    objective.append_scored_turn(
        0, 0, [4, 5], [True, True],
        teacher_logprobs=torch.tensor([-1.5, -1.2], device=device),
    )
    if not any(p.grad is not None and bool(p.grad.abs().sum()) for p in model.parameters()):
        raise RuntimeError("the first scored turn produced no gradient")
    objective.append_scored_turn(
        0, 0, [6, 7, 8], [False, True, True],
        teacher_logprobs=torch.tensor([-1.4, -1.1], device=device),
    )
    objective.close_trajectory(0)
    objective.close_group(0)
    objective.assert_safe_to_step()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not (torch.isfinite(norm) and norm > 0):
        raise RuntimeError(f"expected a finite, nonzero gradient norm, got {float(norm)}")
    optimizer.step()
    print(json.dumps({"objective": "opd", "updates": 1, "gradient_norm": float(norm)}))


if __name__ == "__main__":
    main()
