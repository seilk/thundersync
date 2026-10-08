"""Turn cuts preserve descendant contributions without delaying direct losses."""

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from thundersync.opd.objectives import OPDTurnStream
from thundersync.engine.streaming import (
    STREAM_ATTENTION_NAME, StreamingRun, _TurnReplayBoundary, register_stream_attention,
)


def make_run(dtype=torch.float32, *, explicit=False, capture=False,
             reserve=None, normalization="sum"):
    register_stream_attention()
    torch.manual_seed(71)
    config = AutoConfig.for_model(
        "qwen3", hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        vocab_size=32, max_position_embeddings=128, tie_word_embeddings=False,
    )
    model = AutoModelForCausalLM.from_config(
        config, attn_implementation=STREAM_ATTENTION_NAME, dtype=dtype,
    ).eval()
    run = StreamingRun(
        model, boundary_cut=True, checkpoint=True, stream_turns=True,
        turn_boundary_rebuild=True, rebuild_block_tokens=8,
        rebuild_plain_reserve_bytes=reserve,
        parameter_adjoint_storage_device="cpu" if capture else None,
        explicit_adjoint_vjp=explicit,
    )
    objective = OPDTurnStream(run, model, normalization=normalization)
    objective.open_group(0, [1, 2, 3])
    return model, run, objective


def finish(model, run, objective):
    objective.close_trajectory(0)
    objective.close_group(0)
    objective.assert_safe_to_step()
    assert not run._trajs[0].turn_replay_boundaries
    assert run.retained_device_bytes()["turn_replay_grad"] == 0
    return {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}


def injected_descendant_gradient(coefficients):
    model, run, objective = make_run(torch.bfloat16)
    objective.append_scored_turn(0, 0, [4, 5], [False, False],
                                 teacher_logprobs=torch.empty(0))
    # The injected cotangents stand in for a descendant turn, whose arrival
    # forwards the deferred observation first.
    run.forward_deferred_turns(0)
    predictor = run._trajs[0].turn_replay_boundaries[-1].proxy[-1]
    for coefficient in coefficients:
        run.backward_with_boundary_adjoint_capture(
            predictor[0] * coefficient, source=("trajectory_turn", 0),
        )
    return finish(model, run, objective)


@pytest.mark.parametrize("plain", [False, True])
def test_bf16_turn_state_does_not_lose_a_small_descendant_contribution(monkeypatch, plain):
    monkeypatch.setattr(StreamingRun, "_plain_turn_fits", lambda self, needs, **_: plain)
    expected = injected_descendant_gradient([1.0])
    actual = injected_descendant_gradient([256.0, 1.0, -256.0])
    assert any(bool(value.abs().sum()) for value in expected.values())
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0, msg=name)


def streamed_gradients(explicit):
    model, run, objective = make_run(explicit=explicit, capture=True)
    for tokens, mask in [([4, 5], [True, True]), ([6, 7], [False, True]), ([8], [False])]:
        objective.append_scored_turn(0, 0, tokens, mask,
                                     teacher_logprobs=torch.full((sum(mask),), -1.5))
    # The direct objective has differentiated before trajectory closure.
    assert any(value is not None and bool(value.abs().sum()) for value in run._parameter_adjoints)
    return finish(model, run, objective)


def test_explicit_vjp_preserves_descendant_turn_state_gradients():
    expected = streamed_gradients(False)
    actual = streamed_gradients(True)
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=3e-5, atol=1e-6, msg=name)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_turn_ledger_owns_only_the_leaf_sized_gradient(dtype):
    proxy = torch.zeros(3, dtype=dtype, requires_grad=True)
    boundary = _TurnReplayBoundary(cid=1, parent_chain=[], tokens=[1], pos0=0, proxy=[proxy])
    backing = torch.ones(4096, dtype=dtype)
    proxy.backward(backing[:3])
    ledger = boundary.adjoints._adjoints[0]
    assert proxy.grad is None
    assert ledger.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    assert ledger.untyped_storage().nbytes() == ledger.numel() * ledger.element_size()
    boundary.adjoints.release()
    assert not boundary.adjoints._hook_handles
    assert not boundary.proxy


def test_unreplayed_turn_state_cannot_be_freed():
    model, run, objective = make_run()
    objective.append_scored_turn(0, 0, [4], [True], teacher_logprobs=torch.tensor([-1.5]))
    with pytest.raises(RuntimeError, match="finalize turn adjoints"):
        run.free_trajectory(0)
    with pytest.raises(RuntimeError, match="unreplayed turn adjoints"):
        run.free_group(0)
    finish(model, run, objective)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_aliased_learner_reference_still_rejects_nonfinite_scores(monkeypatch, value):
    _, run, objective = make_run()
    consume = run.consume_latest_turn_logprobs

    def nonfinite(trajectory_id):
        lp, _ = consume(trajectory_id)
        invalid = lp * value
        return invalid, invalid.detach()

    monkeypatch.setattr(run, "consume_latest_turn_logprobs", nonfinite)
    with pytest.raises(RuntimeError, match="learner logprobs must be finite"):
        objective.append_scored_turn(0, 0, [4], [True],
                                     teacher_logprobs=torch.tensor([-1.5]), return_loss=False)


@pytest.mark.parametrize("difference", [0.0, 0.25])
def test_nonaliased_learner_reference_keeps_its_value_check(monkeypatch, difference):
    model, run, objective = make_run()
    consume = run.consume_latest_turn_logprobs

    def copied(trajectory_id):
        lp, old = consume(trajectory_id)
        return lp, old.clone() + difference

    monkeypatch.setattr(run, "consume_latest_turn_logprobs", copied)
    if difference:
        with pytest.raises(RuntimeError, match="on-policy"):
            objective.append_scored_turn(0, 0, [4], [True], teacher_logprobs=torch.tensor([-1.5]))
    else:
        objective.append_scored_turn(0, 0, [4], [True], teacher_logprobs=torch.tensor([-1.5]))
        finish(model, run, objective)


def test_deferred_reporting_preserves_loss_and_gradients():
    def evaluate(return_loss):
        model, run, objective = make_run()
        returned = []
        for tokens, mask in [([4, 5], [True, True]), ([6, 7], [False, True])]:
            returned.append(objective.append_scored_turn(
                0, 0, tokens, mask, teacher_logprobs=torch.full((sum(mask),), -1.5),
                return_loss=return_loss,
            ))
        gradients = finish(model, run, objective)
        assert objective._loss_numerator_device is None
        return objective.loss_value, gradients, returned

    expected_loss, expected_gradients, immediate = evaluate(True)
    loss, gradients, deferred = evaluate(False)
    assert expected_loss == sum(immediate)
    assert loss == expected_loss
    assert deferred == [None, None]
    for name, expected in expected_gradients.items():
        torch.testing.assert_close(gradients[name], expected, rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_transient_replay_preserves_full_normalized_update_and_arrival_work(monkeypatch, dtype):
    results = []
    turns = [([4, 5], [True, True]), ([6, 7, 8], [False] * 3),
             ([9, 10, 11], [True] * 3), ([12], [False])]
    for plain in (False, True):
        phase = "prompt"
        calls, fit_inputs, arrival_inputs = [], [], []
        original_packed = StreamingRun._forward_packed

        def packed(self, *args, **kwargs):
            calls.append((phase, kwargs.get("plain", False)))
            return original_packed(self, *args, **kwargs)

        def fits(self, needs, *, replay=False, memory=None):
            assert len(needs) == 1
            if phase == "turn":
                # Arrival asks the same memory rule; held checkpointed here so
                # the two runs differ only in the replay's executor.
                assert not replay
                arrival_inputs.append(list(needs[0].tokens))
                return False
            assert phase == "replay" and replay
            fit_inputs.append(list(needs[0].tokens))
            return plain

        with monkeypatch.context() as scoped:
            scoped.setattr(StreamingRun, "_forward_packed", packed)
            # Only capacity is simulated; both executors run the real CPU model.
            scoped.setattr(StreamingRun, "_plain_turn_fits", fits)
            model, run, objective = make_run(dtype, reserve=24 << 30,
                                              normalization="token_mean")
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            phase = "turn"
            for tokens, mask in turns:
                for tid in (0, 1):
                    objective.append_scored_turn(
                        0, tid, [token + 8 * tid for token in tokens], mask,
                        teacher_logprobs=torch.full((sum(mask),), -1.5),
                        return_loss=False,
                    )
                    if sum(mask):
                        assert any(p.grad is not None and bool(p.grad.abs().sum())
                                   for p in model.parameters())
            assert not fit_inputs and not objective._closed
            # An unscored turn is forwarded when its successor arrives; the
            # trailing one never is.
            assert arrival_inputs == [
                [4, 5], [12, 13], [6, 7, 8], [9, 10, 11], [14, 15, 16], [17, 18, 19],
            ]
            direct = {name: p.grad.clone() for name, p in model.named_parameters()}
            boundaries = [boundary for tid in (0, 1)
                          for boundary in run._trajs[tid].turn_replay_boundaries]
            phase = "replay"
            for tid in (0, 1):
                objective.close_trajectory(tid)
            phase = "group"
            objective.close_group(0)
            objective.assert_safe_to_step()
            assert objective.normalization_divisor == 10
            assert fit_inputs == [[6, 7, 8], [4, 5], [14, 15, 16], [12, 13]]
            assert [flag for stage, flag in calls if stage == "replay"] == [plain] * 4
            assert [flag for stage, flag in calls if stage != "replay"] == [False] * 7
            assert run.turn_path_counts["unscored"] == {"deferred": 4, "dropped": 2}
            assert all(not b.proxy and not b.adjoints._adjoints for b in boundaries)
            assert not run.open_group_ids()
            assert all(not run._trajs[tid].turn_replay_boundaries for tid in (0, 1))
            retained = run.retained_device_bytes()
            for field in ("kv", "turn_replay_proxy", "turn_replay_grad", "boundary_real",
                          "boundary_proxy", "boundary_ledger", "trajectory_state"):
                assert retained[field] == 0, field
            gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
            assert any(not torch.equal(gradients[name] * 10, direct[name]) for name in direct)
            initial = {name: p.detach().clone() for name, p in model.named_parameters()}
            optimizer.step()
            parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
            assert any(not torch.equal(parameters[name], initial[name]) for name in initial)
            moments = {name: {key: value.clone() for key, value in optimizer.state[p].items()}
                       for name, p in model.named_parameters()}
            results.append((objective.loss_value, direct, gradients, parameters, moments))

    expected, actual = results
    assert actual[0] == expected[0]
    for name in expected[1]:
        torch.testing.assert_close(actual[1][name], expected[1][name], rtol=0, atol=0)
    rtol, atol = (3e-5, 1e-6) if dtype == torch.float32 else (1e-10, 1e-12)
    for left, right in zip(actual[2:4], expected[2:4]):
        assert left.keys() == right.keys()
        for name in right:
            torch.testing.assert_close(left[name], right[name], rtol=rtol, atol=atol, msg=name)
    for name in expected[4]:
        assert actual[4][name].keys() == expected[4][name].keys()
        for key in expected[4][name]:
            torch.testing.assert_close(actual[4][name][key], expected[4][name][key],
                                       rtol=rtol, atol=atol, msg=f"{name}.{key}")


@pytest.mark.parametrize("channel", range(5))
def test_transient_replay_preserves_each_layer_kv_and_last_hidden_adjoint(monkeypatch, channel):
    results = []
    for plain in (False, True):
        model, run, objective = make_run(torch.float64)
        monkeypatch.setattr(run, "_plain_turn_fits", lambda needs, **_: plain)
        # A scored action populates the native gradient inventory before an
        # observation-only turn receives one isolated descendant cotangent.
        for tokens, mask in (([4, 5], [True, True]), ([6, 7], [False, False])):
            objective.append_scored_turn(0, 0, tokens, mask,
                                         teacher_logprobs=torch.full((sum(mask),), -1.5))
        # The injected cotangent stands in for a descendant turn, whose
        # arrival forwards the deferred observation first.
        run.forward_deferred_turns(0)
        proxy = run._trajs[0].turn_replay_boundaries[-1].proxy[channel]
        cotangent = torch.linspace(0.2, 0.8, proxy.numel(), dtype=proxy.dtype).reshape_as(proxy)
        run.backward_with_boundary_adjoint_capture((proxy * cotangent).sum(),
                                                   source=("trajectory_turn", 0))
        results.append(finish(model, run, objective))
    expected, actual = results
    assert expected.keys() == actual.keys()
    assert any(bool(value.abs().sum()) for value in expected.values())
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-10, atol=1e-12, msg=name)


@pytest.mark.parametrize("reserve", [None, 24 << 30])
def test_transient_replay_keeps_default_and_cpu_memory_fallback(monkeypatch, reserve):
    model, run, objective = make_run(reserve=reserve)
    original_forward = run._forward
    replay_flags = []

    def forward(*args, **kwargs):
        replay_flags.append(kwargs.get("plain", False))
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_: pytest.fail("CPU queried CUDA memory"))
    for tokens in ([4, 5], [6, 7]):
        objective.append_scored_turn(0, 0, tokens, [True, True],
                                     teacher_logprobs=torch.full((2,), -1.5))
    monkeypatch.setattr(run, "_forward", forward)
    finish(model, run, objective)
    assert replay_flags == [False]


def test_transient_replay_failure_keeps_unconsumed_state_and_forbids_release(monkeypatch):
    _, run, objective = make_run()
    for tokens in ([4, 5], [6, 7]):
        objective.append_scored_turn(0, 0, tokens, [True, True],
                                     teacher_logprobs=torch.full((2,), -1.5))
    boundary = run._trajs[0].turn_replay_boundaries[0]
    monkeypatch.setattr(run, "_plain_turn_fits", lambda needs, **_: True)

    def fail(*args, **kwargs):
        assert kwargs["plain"] is True
        raise RuntimeError("plain replay failed")

    monkeypatch.setattr(run, "_forward", fail)
    with pytest.raises(RuntimeError, match="plain replay failed"):
        objective.close_trajectory(0)
    assert run._trajs[0].turn_replay_boundaries == [boundary]
    assert boundary.adjoints.has_buffered_adjoint() and not boundary.adjoints._consumed
    with pytest.raises(RuntimeError, match="finalize turn adjoints"):
        run.free_trajectory(0)
    with pytest.raises(RuntimeError, match="unreplayed turn adjoints"):
        run.free_group(0)
    with pytest.raises(RuntimeError, match="open groups"):
        objective.assert_safe_to_step()


def test_transient_replay_rejects_changed_parameters_before_capacity_decision(monkeypatch):
    model, run, objective = make_run()
    objective.append_scored_turn(0, 0, [4, 5], [True, True],
                                 teacher_logprobs=torch.full((2,), -1.5))
    monkeypatch.setattr(run, "_plain_turn_fits", lambda needs, **_: pytest.fail("replayed changed weights"))
    with torch.no_grad():
        next(model.parameters()).add_(1)
    with pytest.raises(RuntimeError, match="parameter was modified"):
        objective.close_trajectory(0)
