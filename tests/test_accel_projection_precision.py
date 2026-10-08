"""Projection and chunk reduction must not erase representable logit differences."""

import pytest
import torch

from thundersync.accel.fused_logprob import fused_logprob, fused_logprob_entropy


@pytest.mark.parametrize("with_entropy", [False, True])
@pytest.mark.parametrize("dtype,offset", [(torch.bfloat16, 256), (torch.float16, 2048)])
def test_projection_retains_the_fp32_dot_product_before_log_softmax(with_entropy, dtype, offset):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden = torch.tensor([[1, 1]], dtype=dtype, device=device)
    weight = torch.tensor([[offset, 1], [offset, 0]], dtype=dtype, device=device)
    target = torch.tensor([0], device=device)
    logits = torch.nn.functional.linear(hidden.float(), weight.float())
    expected = torch.log_softmax(logits, -1)
    if with_entropy:
        actual, entropy = fused_logprob_entropy(hidden, weight, target, chunk=1)
        expected_entropy = -(expected.exp() * expected).sum(-1)
        torch.testing.assert_close(entropy, expected_entropy, rtol=5e-4, atol=5e-4)
    else:
        actual = fused_logprob(hidden, weight, target, chunk=1)
    torch.testing.assert_close(actual, expected[:, 0], rtol=5e-4, atol=5e-4)


@pytest.mark.parametrize("with_entropy", [False, True])
@pytest.mark.parametrize("dtype,large", [(torch.bfloat16, 256), (torch.float16, 2048)])
def test_weight_gradient_does_not_lose_small_chunks(with_entropy, dtype, large):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden = torch.ones(3, 1, dtype=dtype, device=device)
    weight = torch.zeros(2, 1, dtype=dtype, device=device, requires_grad=True)
    target = torch.zeros(3, dtype=torch.long, device=device)
    coefficients = torch.tensor([2 * large, 2, -2 * large], dtype=torch.float32, device=device)
    if with_entropy:
        logprobs, _ = fused_logprob_entropy(hidden, weight, target, chunk=1)
    else:
        logprobs = fused_logprob(hidden, weight, target, chunk=1)
    (coefficients * logprobs).sum().backward()
    torch.testing.assert_close(weight.grad, torch.tensor([[1], [-1]], dtype=dtype, device=device),
                               rtol=0, atol=0)


@pytest.mark.parametrize("with_entropy", [False, True])
def test_logprob_subtraction_is_stable_under_a_common_logit_offset(with_entropy):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden = torch.ones(1, 1, device=device)
    weight = torch.tensor([[100.0], [88.0]], device=device)
    target = torch.tensor([0], device=device)
    expected = torch.log_softmax(hidden @ weight.T, -1)[:, 0]
    if with_entropy:
        actual, _ = fused_logprob_entropy(hidden, weight, target)
    else:
        actual = fused_logprob(hidden, weight, target)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("with_entropy", [False, True])
def test_autocast_does_not_downcast_fp32_head_values_or_gradients(with_entropy):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden = torch.tensor([[1.0, 1.0]], device=device, requires_grad=True)
    weight = torch.tensor([[1.0, 1.0], [1.0, 0.0]], device=device, requires_grad=True)
    target = torch.tensor([0], device=device)
    reference_h = hidden.detach().clone().requires_grad_(True)
    reference_w = weight.detach().clone().requires_grad_(True)
    reference = torch.log_softmax(reference_h @ reference_w.T, -1)
    expected = reference[:, 0]
    if with_entropy:
        expected = expected + 0.3 * -(reference.exp() * reference).sum(-1)
    expected.sum().backward()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        if with_entropy:
            lp, entropy = fused_logprob_entropy(hidden, weight, target)
            actual = lp + 0.3 * entropy
        else:
            actual = fused_logprob(hidden, weight, target)
        actual.sum().backward()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(hidden.grad, reference_h.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(weight.grad, reference_w.grad, rtol=1e-5, atol=1e-6)
