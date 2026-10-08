"""The FP32 boundary ledger holds the bits of widening each adjoint and adding.

`_ledger_accumulate` makes the first contribution the ledger with one widening
copy and adds later ones with a mixed-dtype ``add_``. These gates pin that
the ledger equals the reference that widens every contribution first and
then adds in FP32 (FP64 for FP64 adjoints), bit for bit, through the direct
helper, the post-accumulate hook of a boundary and its consume and drain.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _torch_is_real() -> bool:
    """A test elsewhere installs a stub under the torch name where torch is
    absent; the stub has no import spec."""
    try:
        spec = importlib.util.find_spec("torch")
    except ValueError:
        return False
    return spec is not None and spec.origin is not None


if not _torch_is_real():
    pytest.skip("needs a real torch", allow_module_level=True)
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

from thundersync.engine import streaming# noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
DTYPES = [torch.bfloat16, torch.float16, torch.float32, torch.float64]


def _contributions(dtype, device, count=5, shape=(3, 257)):
    generator = torch.Generator().manual_seed(7)
    values = []
    for index in range(count):
        # magnitudes spread over several binades, so FP32 rounding happens
        scale = 10.0 ** (index - 2)
        values.append(
            (torch.randn(shape, generator=generator, dtype=torch.float64) * scale)
            .to(dtype)
            .to(device)
        )
    return values


def _reference(values):
    wide = torch.float64 if values[0].dtype == torch.float64 else torch.float32
    total = values[0].to(wide).clone()
    for value in values[1:]:
        total.add_(value.to(wide))
    return total


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_the_ledger_is_bitwise_the_widened_sum(device, dtype):
    values = _contributions(dtype, device)
    ledgers = [None]
    for value in values:
        streaming._ledger_accumulate(ledgers, 0, value)
    expected = _reference(values)
    assert ledgers[0].dtype == expected.dtype
    assert torch.equal(ledgers[0], expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_a_first_contribution_in_the_ledger_dtype_is_copied(device, dtype):
    value = _contributions(dtype, device, count=1)[0]
    ledgers = [None]
    streaming._ledger_accumulate(ledgers, 0, value)
    assert ledgers[0].data_ptr() != value.data_ptr()
    value.zero_()
    assert ledgers[0].abs().sum() > 0


def _boundary(dtype, device, count=2):
    proxies = [
        torch.zeros(3, 257, dtype=dtype, device=device, requires_grad=True)
        for _ in range(count)
    ]
    reals = [proxy.detach().clone() for proxy in proxies]
    source = {"value": ("turn", 0)}
    boundary = streaming._Boundary(
        real=reals, proxy=proxies, capture_source=lambda: source["value"]
    )
    return boundary, proxies


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_the_hook_ledger_matches_the_widened_sum(device, dtype):
    boundary, proxies = _boundary(dtype, device)
    per_proxy = [_contributions(dtype, device, count=4) for _ in proxies]
    for step in range(4):
        loss = sum(
            (proxy * values[step]).sum() for proxy, values in zip(proxies, per_proxy)
        )
        loss.backward()
        assert all(proxy.grad is None for proxy in proxies)
    for index, values in enumerate(per_proxy):
        assert torch.equal(boundary._adjoints[index], _reference(values))
    assert boundary.adjoint_diagnostics()["contribution_count"] == 8


@pytest.mark.parametrize("device", DEVICES)
def test_consume_adds_a_native_gradient_exactly(device):
    dtype = torch.bfloat16
    boundary, proxies = _boundary(dtype, device)
    values = _contributions(dtype, device, count=3)
    for value in values[:2]:
        boundary.accumulate_explicit_adjoints([value, None], source=("turn", 0))
    # a native gradient left on the proxy is folded in at consumption
    proxies[0].grad = values[2].clone()
    outputs, gradients = boundary.consume_adjoint_pairs()
    assert len(outputs) == 1
    expected = _reference(values).to(dtype)
    assert torch.equal(gradients[0], expected)
