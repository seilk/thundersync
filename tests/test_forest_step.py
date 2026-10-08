"""End to end on a real transformer: same logprobs, same gradients, less work.

The whole design rests on one claim -- that sharing the forward, packing the
stream and gating the projection are *invisible* to the objective. So this runs
an actual `Qwen3ForCausalLM` three ways over the same trajectories:

    reference  one trajectory at a time, full [T, V] log-softmax
    baseline   [B, T_max] rectangle, projection over everything, mask after
    forest     one shared forward, projection only where the objective reads

and requires all three to agree in value *and* in parameter gradient. Values
alone are not enough: an earlier version of this project matched losses exactly
while producing wrong gradients, because a per-token weight had been folded into
a place that looked equivalent and was not.
"""

from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from thundersync.engine.segments import Segment, Trajectory, build_forest
from thundersync.engine.plan import compile_plan# noqa: E402
from tests.reference_logprobs import baseline_logprobs, reference_logprobs  # noqa: E402
from thundersync.engine.step import (  # noqa: E402
    ATTENTION_NAME,
    forest_logprobs,
    register_attention,
    split_by_trajectory,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
register_attention()


def build_models(vocab=512, seed=0):
    """Two copies of one model: one using forest attention, one using sdpa.

    Each gets its *own* config object. `from_config(cfg, attn_implementation=...)`
    writes the choice back into `cfg`, and the attention modules hold a reference
    to that same object -- so sharing one config between the two silently reverts
    the first model to the second's implementation, and the comparison then comes
    out equal for the wrong reason.
    """
    torch.manual_seed(seed)

    def cfg():
        return AutoConfig.for_model(
            "qwen3",
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            vocab_size=vocab,
            max_position_embeddings=8192,
            tie_word_embeddings=False,
        )

    g = AutoModelForCausalLM.from_config(
        cfg(), attn_implementation=ATTENTION_NAME, dtype=torch.float32
    ).to(DEV)
    r = AutoModelForCausalLM.from_config(
        cfg(), attn_implementation="sdpa", dtype=torch.float32
    ).to(DEV)
    r.load_state_dict(g.state_dict())
    assert g.model.layers[0].self_attn.config._attn_implementation == ATTENTION_NAME
    assert r.model.layers[0].self_attn.config._attn_implementation == "sdpa"
    return g.eval(), r.eval()


def make_trajectories(n=4, prompt_len=32, vocab=512, seed=1):
    g = torch.Generator().manual_seed(seed)
    r = lambda n_: torch.randint(0, vocab, (n_,), generator=g).tolist()  # noqa: E731
    prompt = r(prompt_len)
    out = []
    for i in range(n):
        segs = [Segment.prompt(prompt)]
        for k in range(1 + i % 3):
            segs.append(Segment.action(r(5 + k)))
            segs.append(Segment.observation(r(9 + i)))
        segs.append(Segment.action(r(4)))
        out.append(Trajectory(segs, group_id=0, traj_id=i))
    return out


def total_loss(per_traj: dict[int, torch.Tensor], seed=7):
    """A stand-in objective: arbitrary detached per-token coefficients.

    Deliberately not GRPO or PPO. The core must not care, and a test that used a
    real objective could hide a bug behind that objective's own structure.
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    loss = 0.0
    for tid in sorted(per_traj):
        lp = per_traj[tid]
        c = torch.randn(lp.shape, generator=g, device=DEV)
        loss = loss + (c * lp).sum()
    return loss


# ----------------------------------------------------------------------


def test_matches_reference_values():
    gm, rm = build_models()
    trajs = make_trajectories()
    plan = compile_plan(build_forest(trajs), device=DEV)

    got = split_by_trajectory(plan, forest_logprobs(gm, plan))
    want = reference_logprobs(rm, trajs)

    for tid in want:
        assert got[tid].shape == want[tid].shape, f"traj {tid} scored count differs"
        torch.testing.assert_close(got[tid], want[tid], rtol=1e-4, atol=1e-5)


def test_matches_the_standard_baseline():
    """Agreement with the padded, project-everything path we are racing."""
    gm, rm = build_models(seed=3)
    trajs = make_trajectories(seed=5)
    plan = compile_plan(build_forest(trajs), device=DEV)

    got = split_by_trajectory(plan, forest_logprobs(gm, plan))
    want = baseline_logprobs(rm, trajs)
    for tid in want:
        torch.testing.assert_close(got[tid], want[tid], rtol=1e-4, atol=1e-5)


def test_matches_reference_gradients():
    """The claim that values alone cannot establish."""
    gm, rm = build_models(seed=11)
    trajs = make_trajectories(seed=13)
    plan = compile_plan(build_forest(trajs), device=DEV)

    total_loss(split_by_trajectory(plan, forest_logprobs(gm, plan))).backward()
    total_loss(reference_logprobs(rm, trajs)).backward()

    gg = dict(gm.named_parameters())
    rg = dict(rm.named_parameters())
    checked = 0
    for name, p in gg.items():
        q = rg[name]
        if p.grad is None and q.grad is None:
            continue
        assert p.grad is not None and q.grad is not None, f"{name}: one side has no grad"
        torch.testing.assert_close(
            p.grad, q.grad, rtol=2e-3, atol=2e-5, msg=lambda m, n=name: f"{n}: {m}"
        )
        checked += 1
    assert checked > 5, f"only {checked} parameters carried gradient; test is vacuous"


def test_shared_prompt_gradient_accumulates_over_trajectories():
    """A shared node is used by every branch; its gradient must sum, not overwrite."""
    gm, rm = build_models(seed=17)
    trajs = make_trajectories(n=6, prompt_len=64, seed=19)
    plan = compile_plan(build_forest(trajs), device=DEV)

    # the prompt exists once physically but is on six trajectories' paths
    assert plan.forest.nodes[plan.forest.roots[0]].multiplicity == 6
    total_loss(split_by_trajectory(plan, forest_logprobs(gm, plan))).backward()
    total_loss(reference_logprobs(rm, trajs)).backward()

    a = gm.model.embed_tokens.weight.grad
    b = rm.model.embed_tokens.weight.grad
    torch.testing.assert_close(a, b, rtol=2e-3, atol=2e-5)


def test_projects_far_fewer_positions_than_the_baseline():
    gm, _ = build_models(seed=23)
    trajs = make_trajectories(n=8, prompt_len=512, seed=29)
    plan = compile_plan(build_forest(trajs), device=DEV)
    _, stats = forest_logprobs(gm, plan, return_stats=True)

    assert stats.token_compression > 2.0, str(stats)
    assert stats.projection_reduction > 10.0, str(stats)
    # every logical scored token is still accounted for
    assert stats.logical_scored == sum(t.n_scored for t in trajs)


@pytest.mark.parametrize("share", [True, False])
@pytest.mark.parametrize("gate", [True, False])
def test_every_share_and_gate_setting_is_the_same_computation(share, gate):
    """Sharing and gating change cost, never what is computed."""
    from thundersync.engine.segments import unshared

    gm, rm = build_models(seed=37)
    trajs = make_trajectories(n=5, seed=41)
    plan = compile_plan(
        build_forest(trajs if share else unshared(trajs)), device=DEV
    )

    got = split_by_trajectory(plan, forest_logprobs(gm, plan, gate=gate))
    want = reference_logprobs(rm, trajs)
    for tid in want:
        torch.testing.assert_close(got[tid], want[tid], rtol=1e-4, atol=1e-5)


def test_unshared_forest_really_shares_nothing():
    from thundersync.engine.segments import unshared

    trajs = make_trajectories(n=6, prompt_len=128, seed=43)
    shared = compile_plan(build_forest(trajs), device=DEV)
    flat = compile_plan(build_forest(unshared(trajs)), device=DEV)
    assert flat.n_physical == sum(t.n_tokens for t in trajs)
    assert shared.n_physical < flat.n_physical
    assert len(flat.forest.roots) == len(trajs)


def test_activation_checkpointing_recompute_still_sees_the_plan():
    """The recompute runs inside backward, possibly on an autograd engine thread.

    A `ContextVar` bound on the main thread would not be visible there, and the
    recomputed forward would attend with no plan -- either raising, or worse,
    quietly using a different attention than the forward did.
    """
    from thundersync.engine.step import forest_context

    gm, rm = build_models(seed=53)
    gm.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    gm.train()
    trajs = make_trajectories(n=4, seed=59)
    plan = compile_plan(build_forest(trajs), device=DEV)

    with forest_context(plan):
        got = split_by_trajectory(plan, forest_logprobs(gm, plan))
        total_loss(got).backward()
    total_loss(reference_logprobs(rm, trajs)).backward()

    a = dict(gm.named_parameters())
    b = dict(rm.named_parameters())
    checked = 0
    for name, p in a.items():
        if p.grad is None:
            continue
        torch.testing.assert_close(
            p.grad, b[name].grad, rtol=2e-3, atol=2e-5,
            msg=lambda m, n=name: f"{n} after recompute: {m}",
        )
        checked += 1
    assert checked > 5, "checkpointing test carried almost no gradient"


def test_hybrid_models_are_refused_not_silently_miscomputed():
    """Recurrent layers would run state from one branch into the next."""
    from thundersync.engine.step import assert_supported

    clean, _ = build_models(seed=47)
    assert_supported(clean)  # dense qwen3 is fine

    # A *separate* instance, mutated before its first check: `assert_supported`
    # memoises per model, so reusing `clean` would hit the cache and pass for the
    # wrong reason.
    hybrid, _ = build_models(seed=49)
    hybrid.config.layer_types = ["full_attention", "linear_attention"]
    with pytest.raises(NotImplementedError, match="linear_attention"):
        assert_supported(hybrid)


def test_supported_model_cache_does_not_keep_destroyed_models_alive():
    from thundersync.engine.step import assert_supported

    model = torch.nn.Linear(2, 2)
    model.config = SimpleNamespace(layer_types=None)
    model_ref = weakref.ref(model)
    assert_supported(model)
    del model
    gc.collect()

    assert model_ref() is None


def test_attention_outside_a_plan_context_is_an_error():
    """Silently attending across trajectories would be the worst failure mode."""
    gm, _ = build_models(seed=31)
    ids = torch.randint(0, 512, (1, 16), device=DEV)
    with pytest.raises(RuntimeError, match="forest_context"):
        gm(input_ids=ids)
