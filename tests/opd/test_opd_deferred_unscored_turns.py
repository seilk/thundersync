"""An unscored turn is forwarded only when a later turn needs its state.

A turn with no scored token has no loss. Its forward yields the K/V and the
last hidden state that the trajectory's later turns attend to and predict
from, and nothing else reads them: its replay boundary can only receive an
adjoint from a descendant. So its forward waits for the next arrival of its
trajectory, which runs it first as the same single-turn call at the same
positions, ancestor state and parameters, and a turn that ends its
trajectory is never forwarded. The update is the one the arrival-time
forward produced.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]

from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from thundersync.engine import streaming  # noqa: E402
from thundersync.opd.objectives import OPDTurnStream  # noqa: E402
from thundersync.engine.streaming import (  # noqa: E402
    STREAM_ATTENTION_NAME,
    StreamingRun,
    register_stream_attention,
)

_FORWARD_PACKED = StreamingRun._forward_packed

# observation-only opener, action, observation-only middle, action, and
# the trailing observation every recorded trajectory ends with
TURNS = [
    ([3, 4], [False, False]),
    ([5, 6, 7], [False, True, True]),
    ([8, 9, 10, 11], [False] * 4),
    ([12, 13, 14], [False, True, True]),
    ([15, 16, 17], [False] * 3),
]


def _drive(monkeypatch, *, defer: bool, dtype):
    monkeypatch.setattr(streaming, "_DEFER_UNSCORED_TURNS", defer)
    register_stream_attention()
    torch.manual_seed(29)
    config = AutoConfig.for_model(
        "qwen3", hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        vocab_size=64, max_position_embeddings=128, tie_word_embeddings=False,
    )
    model = AutoModelForCausalLM.from_config(
        config, attn_implementation=STREAM_ATTENTION_NAME, dtype=dtype,
    ).eval()
    run = StreamingRun(model, boundary_cut=True, checkpoint=True, stream_turns=True,
                       turn_boundary_rebuild=True, rebuild_block_tokens=8)
    forwards = []

    def packed(self, segments, tokens, *args, **kwargs):
        forwards.append(list(tokens))
        return _FORWARD_PACKED(self, segments, tokens, *args, **kwargs)

    monkeypatch.setattr(StreamingRun, "_forward_packed", packed)
    objective = OPDTurnStream(run, model, normalization="token_mean")
    objective.open_group(0, [1, 2])
    losses = []
    for tokens, scored in TURNS:
        for trajectory in (0, 1):
            losses.append(objective.append_scored_turn(
                0, trajectory, [t + 20 * trajectory for t in tokens], scored,
                teacher_logprobs=torch.linspace(-1.5, -0.5, sum(scored), dtype=dtype),
            ))
    direct = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    arrivals = list(forwards)
    for trajectory in (0, 1):
        objective.close_trajectory(trajectory)
    objective.close_group(0)
    objective.assert_safe_to_step()
    final = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return losses, direct, final, arrivals, run.turn_path_counts, objective.loss_value


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_deferral_preserves_losses_and_gradients_exactly(monkeypatch, dtype):
    eager = _drive(monkeypatch, defer=False, dtype=dtype)
    deferred = _drive(monkeypatch, defer=True, dtype=dtype)
    assert deferred[0] == eager[0]
    for got, want in ((deferred[1], eager[1]), (deferred[2], eager[2])):
        assert got.keys() == want.keys()
        for name in want:
            torch.testing.assert_close(got[name], want[name], rtol=0, atol=0, msg=name)
    assert deferred[5] == eager[5]


def test_only_the_trailing_unscored_turn_is_left_unforwarded(monkeypatch):
    eager = _drive(monkeypatch, defer=False, dtype=torch.float64)
    deferred = _drive(monkeypatch, defer=True, dtype=torch.float64)
    prompt = [[1, 2]]
    # forwards before any replay, in arrival order: every turn as it lands
    assert eager[3] == prompt + [
        [t + 20 * trajectory for t in tokens]
        for tokens, _scored in TURNS for trajectory in (0, 1)
    ]
    # the opener and the middle observation run just before their successor;
    # the trailing observation never runs
    expected = prompt
    for index in (1, 3):
        for trajectory in (0, 1):
            for tokens, _scored in (TURNS[index - 1], TURNS[index]):
                expected.append([t + 20 * trajectory for t in tokens])
    assert deferred[3] == expected
    assert deferred[4]["unscored"] == {"deferred": 6, "dropped": 2}
    assert eager[4]["unscored"] == {"deferred": 0, "dropped": 0}


def test_a_closed_trajectory_holds_no_deferred_turn(monkeypatch):
    monkeypatch.setattr(streaming, "_DEFER_UNSCORED_TURNS", True)
    register_stream_attention()
    torch.manual_seed(3)
    config = AutoConfig.for_model(
        "qwen3", hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        vocab_size=64, max_position_embeddings=128, tie_word_embeddings=False,
    )
    model = AutoModelForCausalLM.from_config(
        config, attn_implementation=STREAM_ATTENTION_NAME, dtype=torch.float64,
    ).eval()
    run = StreamingRun(model, boundary_cut=True, checkpoint=True, stream_turns=True,
                       turn_boundary_rebuild=True, rebuild_block_tokens=8)
    objective = OPDTurnStream(run, model)
    objective.open_group(0, [1, 2])
    objective.append_scored_turn(0, 0, [3, 4], [True, True],
                                 teacher_logprobs=torch.full((2,), -1.0, dtype=torch.float64))
    objective.append_scored_turn(0, 0, [5, 6], [False, False],
                                 teacher_logprobs=torch.empty(0, dtype=torch.float64))
    state = run._trajs[0]
    assert state.deferred_turns == [([5, 6], [False, False])]
    assert state.turns[-1] == ([5, 6], [False, False])
    assert state.n_tokens == 4
    objective.close_trajectory(0)
    assert not state.deferred_turns
    with pytest.raises(RuntimeError, match="already closed"):
        objective.append_scored_turn(0, 0, [7], [True],
                                     teacher_logprobs=torch.full((1,), -1.0, dtype=torch.float64))
