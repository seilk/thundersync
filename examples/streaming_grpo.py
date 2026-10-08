"""One GRPO update from two arriving trajectories; no downloads required."""

import json
import sys
from pathlib import Path

import torch

# Run as `python examples/streaming_grpo.py`: make the sibling helper importable
# even under `python -I`, which leaves the script's directory off sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _tiny_model import tiny_qwen3  # noqa: E402

from thundersync import RewardLinearBackward, StreamingRun  # noqa: E402


def main():
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tiny_qwen3(device).train()
    # The GRPO objective is the advantage-weighted log-probability of the
    # sampled tokens, a quantity to increase, so the optimizer maximizes.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-6, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.0, maximize=True,
    )
    optimizer.zero_grad()
    # boundary_cut: trajectories read a detached copy of the shared prompt's
    # state, so each trajectory's backward stops at the prompt and the
    # prompt is backpropagated once, when the group closes. The
    # reward-linear backward requires it.
    run = StreamingRun(model, boundary_cut=True)
    objective = RewardLinearBackward(run, model, source_offload_device="cpu")

    # Group 0 has prompt tokens [1, 2, 3] and two trajectories, 0 and 1.
    # Token ids are arbitrary ids below the tiny model's vocabulary size.
    objective.open_group(0, [1, 2, 3], [0, 1])
    # Each turn's mask marks the tokens the policy generated (scored).
    run.append_turn(0, 0, [4, 5], [True, True])
    objective.close_trajectory(0, 0.0)
    run.append_turn(0, 1, [6, 7, 8], [True, True, True])
    objective.close_trajectory(1, 1.0)
    objective.close_group(0)
    objective.assert_safe_to_step()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not (torch.isfinite(norm) and norm > 0):
        raise RuntimeError(f"expected a finite, nonzero gradient norm, got {float(norm)}")
    optimizer.step()
    print(json.dumps({"objective": "grpo", "updates": 1, "gradient_norm": float(norm)}))


if __name__ == "__main__":
    main()
