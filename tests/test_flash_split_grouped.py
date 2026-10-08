"""The flash split reads grouped K/V in place; values and gradients hold.

The expanded flash split repeats every key/value head to the query's heads
and copies each half before the kernel. The kernel maps query head h to
key/value head h // (Hq // Hkv) itself and takes the store's layout, so the
grouped path hands it views. Forward values are the same arithmetic on the
same numbers and match bit for bit. dQ is the same arithmetic too, but the
kernel accumulates it atomically, so two runs of either path already differ
in a few elements by a rounding step; dK/dV differ only in how the group's
partial sums are rounded. The path is taken only where its own trial passes.
"""

from __future__ import annotations


import importlib.util

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

from thundersync.accel import capability  # noqa: E402
from thundersync.engine import streaming# noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or streaming._FLASH_SDPA is None,
    reason="the flash split needs CUDA",
)

HEADS, KV_HEADS, HEAD_DIM = 32, 8, 128
# the student's heads over its key/value heads (groups of four) and the
# teacher's (groups of five)
GROUPINGS = [(32, 8), (40, 8)]


def _inputs(tail: int, chunks: list[int], seed: int, heads: int, kv_heads: int):
    """A query in the projection's layout and per-chunk K/V in the store's."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    query = torch.randn(1, tail, heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16,
                        generator=generator).transpose(1, 2)
    keys = [torch.randn(n, kv_heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16,
                        generator=generator) for n in chunks]
    values = [torch.randn(n, kv_heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16,
                          generator=generator) for n in chunks]
    return query, keys, values


def _run(monkeypatch, grouped: bool, tail: int, chunks: list[int], seed: int,
         heads: int = HEADS, kv_heads: int = KV_HEADS):
    monkeypatch.setattr(streaming, "_FLASH_SPLIT_GROUPED_KV", grouped)
    query, keys, values = _inputs(tail, chunks, seed, heads, kv_heads)
    taken = []
    original = streaming._SplitBottomRightFlashGrouped.apply
    # counts the measured calls only, not the grouped trial's own small ones
    monkeypatch.setattr(
        streaming._SplitBottomRightFlashGrouped, "apply",
        lambda *args: (taken.append(True) if args[0].shape[1] == heads else None)
        or original(*args),
    )
    scale = HEAD_DIM ** -0.5
    before = dict(streaming.attention_path_counts())
    with torch.no_grad():
        primal = streaming._causal_sdpa(query, keys, values, scale)
    leaves = [query.detach().requires_grad_()]
    leaves += [k.detach().requires_grad_() for k in keys]
    leaves += [v.detach().requires_grad_() for v in values]
    n = len(keys)
    out = streaming._causal_sdpa(leaves[0], leaves[1:1 + n], leaves[1 + n:], scale)
    cotangent = torch.randn(out.shape, device="cuda", dtype=out.dtype,
                            generator=torch.Generator(device="cuda").manual_seed(seed + 1))
    grads = torch.autograd.grad(out, leaves, cotangent)
    after = streaming.attention_path_counts()
    counted = {name: after.get(name, 0) - before.get(name, 0) for name in after}
    monkeypatch.setattr(streaming._SplitBottomRightFlashGrouped, "apply", original)
    counted["grouped_apply"] = len(taken)
    return primal, out.detach(), grads, counted


# (query rows, key/value chunk lengths): one turn over a prompt and earlier
# turns, a one-token turn, a prompt-only prefix, a single packed chunk
SHAPES = [
    (157, [492, 157, 157, 157]),
    (1, [492, 124, 1]),
    (42, [300, 42]),
    (64, [640]),
]


@pytest.mark.parametrize("heads,kv_heads", GROUPINGS)
@pytest.mark.parametrize("tail,chunks", SHAPES)
def test_grouped_split_matches_the_expanded_split(monkeypatch, tail, chunks, heads, kv_heads):
    shape = dict(heads=heads, kv_heads=kv_heads)
    expanded = _run(monkeypatch, False, tail, chunks, seed=len(chunks) + tail, **shape)
    grouped = _run(monkeypatch, True, tail, chunks, seed=len(chunks) + tail, **shape)
    for result in (expanded, grouped):
        assert result[3].get("flash_split_primal") == 1
        assert result[3].get("flash_split") == 1
    assert expanded[3]["grouped_apply"] == 0
    # one query row keeps the expanded split (see _grouped_flash_split)
    assert grouped[3]["grouped_apply"] == (1 if tail > 1 else 0)
    assert torch.equal(grouped[0], expanded[0])
    assert torch.equal(grouped[1], expanded[1])
    assert torch.equal(grouped[0], grouped[1])
    # dQ: atomic accumulation, so the expanded path against itself is the floor
    again = _run(monkeypatch, False, tail, chunks, seed=len(chunks) + tail, **shape)
    floor = float((again[2][0].float() - expanded[2][0].float()).norm())
    base = float(expanded[2][0].float().norm())
    difference = float((grouped[2][0].float() - expanded[2][0].float()).norm())
    assert difference <= max(4.0 * floor, 1e-4 * base), (difference, floor, base)
    for got, want in zip(grouped[2][1:], expanded[2][1:], strict=True):
        # dK/dV: the group's partial sums rounded once instead of per head
        torch.testing.assert_close(got, want)
        relative = float((got.float() - want.float()).norm() / want.float().norm())
        assert relative < 4e-3, relative


def test_the_grouped_trial_is_recorded_and_gates_the_path(monkeypatch):
    capability.reset_kernel_probes()
    _run(monkeypatch, True, 8, [40, 8], seed=3)
    records = {record["kernel"]: record for record in capability.kernel_probe_records()}
    assert records["flash_split_grouped"]["runs"] is True
    assert records["flash_split_grouped"]["shape_key"] == HEAD_DIM

    capability.reset_kernel_probes()

    def refuse(*_args):
        raise RuntimeError("grouped heads refused")

    monkeypatch.setattr(streaming, "_flash_split_grouped_trial", refuse)
    calls = []
    original = streaming._SplitBottomRightFlash.apply

    def expanded(*args):
        calls.append(args[1].shape[1])
        return original(*args)

    monkeypatch.setattr(streaming._SplitBottomRightFlash, "apply", expanded)
    primal, out, _grads, counted = _run(monkeypatch, True, 8, [40, 8], seed=3)
    records = {record["kernel"]: record for record in capability.kernel_probe_records()}
    assert records["flash_split_grouped"]["runs"] is False
    # the re-run flash_split trial calls the expanded split on its own
    # two-head input first; the measured call gets the expanded heads
    assert calls[-1] == HEADS
    assert counted.get("flash_split") == 1
    capability.reset_kernel_probes()
