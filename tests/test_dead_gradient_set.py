"""Which positions are actually dead in attention backward, and which only look it.

Skipping `dQ` at every zero-coefficient position is unsafe: although `dQ_i`
flows only through `O_i`, the objective
never reads position `i`. The first half is true. The second does not follow:
an unscored position's hidden state is read as K/V by later *scored* positions,
so gradient reaches it by that route, and `dO_i` is non-zero at every layer
except the last. Applying that shortcut would sever the cross-chunk channel
while leaving the loss value apparently correct.

This locks in the real dead set:

    positions after the last scored one   dQ, dK and dV are zero in EVERY layer
    unscored positions before it          dQ is zero ONLY in the final layer
    position 0                            degenerate; it attends only to itself,
                                          so softmax is constant in Q_0

The exploitable saving is therefore the final layer's dQ plus the trailing
suffix, and neither grows with context length.
"""

from __future__ import annotations


import torch

from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

NL = 4
T = 24
SCORED = [10, 13, 15]
ZERO = 1e-20


def _model_and_grads():
    """Run one backward whose loss reads only SCORED, capturing q/k/v grads."""
    torch.manual_seed(0)
    cfg = AutoConfig.for_model(
        "qwen3",
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=NL,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=512,
        max_position_embeddings=8192,
        tie_word_embeddings=False,
    )
    model = AutoModelForCausalLM.from_config(
        cfg, attn_implementation="sdpa", dtype=torch.float32
    )

    captured: list[dict[str, torch.Tensor]] = [{} for _ in range(NL)]

    def hook(layer_idx: str, name: str):
        def fn(_mod, _inp, out):
            out.retain_grad()
            captured[layer_idx][name] = out

        return fn

    for li, layer in enumerate(model.model.layers):
        for name in ("q_proj", "k_proj", "v_proj"):
            getattr(layer.self_attn, name).register_forward_hook(hook(li, name))

    ids = torch.randint(0, 512, (1, T))
    logits = model(input_ids=ids).logits[0]
    # the objective reads logits at scored positions only -- what gated
    # projection (R2) enforces in the real trainer
    logits[SCORED].logsumexp(-1).sum().backward()
    return captured


def test_dq_is_live_at_unscored_positions_except_the_final_layer():
    """The section 6 premise, stated as the test that refutes it."""
    caps = _model_and_grads()
    last = max(SCORED)
    # position 0 attends only to itself, so its dQ is structurally zero for a
    # reason unrelated to the loss mask
    unscored_before = [i for i in range(1, last) if i not in SCORED]

    for li in range(NL - 1):
        n = caps[li]["q_proj"].grad[0, unscored_before].norm().item()
        assert n > ZERO, f"layer {li}: dQ at unscored positions must stay live, got {n}"

    n_last = caps[NL - 1]["q_proj"].grad[0, unscored_before].norm().item()
    assert n_last <= ZERO, f"final layer dQ must be dead, got {n_last}"


def test_suffix_after_last_scored_position_is_dead_in_every_layer():
    """The one region where skipping attention backward is exact."""
    caps = _model_and_grads()
    after = list(range(max(SCORED) + 1, T))
    assert after, "test needs trailing unscored positions"

    for li in range(NL):
        for name in ("q_proj", "k_proj", "v_proj"):
            n = caps[li][name].grad[0, after].norm().item()
            assert n <= ZERO, f"layer {li} {name}: suffix must be dead, got {n}"


def test_dk_dv_stay_live_at_unscored_positions_before_the_last_scored_one():
    """Why only dQ was ever a candidate: K/V are read by later scored positions."""
    caps = _model_and_grads()
    last = max(SCORED)
    unscored_before = [i for i in range(1, last) if i not in SCORED]

    for li in range(NL):
        for name in ("k_proj", "v_proj"):
            n = caps[li][name].grad[0, unscored_before].norm().item()
            assert n > ZERO, f"layer {li} {name} must stay live, got {n}"
