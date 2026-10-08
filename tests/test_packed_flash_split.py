"""A plain pack's attention runs as one op; each segment computes the same.

Where every segment of a plain turn pack takes the grouped flash split,
`_PackedFlashSplit` runs the pack's attention: each segment's forward is its
single-turn op's (the same flash calls on the same numbers), the halves of
all segments merge at once, and one pair of variable-length flash backward
calls serves the whole pack from the saved output and log-sum-exp. Driven
through the pack's attention entry point against the per-segment ops: the
forward and dK/dV match bit for bit, the shared prompt's gradient sums the
segments' contributions in the order the per-segment ops delivered them,
and dQ, which the kernel accumulates atomically, stays within the floor two
per-segment runs already show. A pack with a segment the split does not
take keeps the per-segment ops.
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest


def _torch_is_real() -> bool:
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
    not torch.cuda.is_available()
    or streaming._FLASH_SDPA is None
    or streaming._FLASH_VARLEN_BACKWARD is None,
    reason="the flash split needs CUDA",
)

HEAD_DIM = 128
GROUPINGS = [(32, 8), (40, 8)]
# per segment: (own tokens, the turn chunks after the shared prompt)
PACKS = [
    [(157, [157, 157]), (157, [42, 157, 68]), (42, []), (157, [157])],
    [(64, [8000]), (512, [1000, 157]), (1024, [])],
    [(2, [5]), (3, [])],
]
PROMPT = 492


def _pack(pack, seed, heads, kv_heads):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def leaf(*shape):
        return torch.randn(
            *shape, device="cuda", dtype=torch.bfloat16, generator=generator
        ).requires_grad_()

    total = sum(length for length, _chunks in pack)
    # the projections' layout: [1, T, H, D], attention reads [1, H, T, D]
    query = leaf(1, total, heads, HEAD_DIM)
    key, value = leaf(1, total, kv_heads, HEAD_DIM), leaf(1, total, kv_heads, HEAD_DIM)
    prompt = (leaf(PROMPT, kv_heads, HEAD_DIM), leaf(PROMPT, kv_heads, HEAD_DIM))
    turns = [
        [(leaf(n, kv_heads, HEAD_DIM), leaf(n, kv_heads, HEAD_DIM)) for n in chunks]
        for _length, chunks in pack
    ]
    cotangent = torch.randn(
        1, total, heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    return query, key, value, prompt, turns, cotangent


def _attend(monkeypatch, packed, pack, seed, heads=32, kv_heads=8):
    monkeypatch.setattr(streaming, "_PACKED_FLASH_SPLIT", packed)
    query, key, value, prompt, turns, cotangent = _pack(pack, seed, heads, kv_heads)
    kv = streaming._StreamKV()
    kv.put(0, 0, *prompt)
    segments, cid = [], 1
    for (length, _chunks), chunks in zip(pack, turns, strict=True):
        chain = [0]
        for k, v in chunks:
            kv.put(0, cid, k, v)
            chain.append(cid)
            cid += 1
        chain.append(cid)
        segments.append((cid, chain, length))
        cid += 1
    run = SimpleNamespace(_active_segments=segments, _kv=kv, _active_pack_layouts={})
    calls = []
    original = streaming._PackedFlashSplit.apply
    # the measured call only, not the op's own kernel trial
    monkeypatch.setattr(
        streaming._PackedFlashSplit, "apply",
        lambda *args: (calls.append(True) if args[4].counted else None) or original(*args),
    )
    before = dict(streaming.attention_path_counts())
    out, _ = streaming._plain_packed_attention(
        run, 0, query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
        HEAD_DIM ** -0.5,
    )
    leaves = [query, key, value, *prompt] + [t for chunks in turns for kv_ in chunks for t in kv_]
    grads = torch.autograd.grad(out, leaves, cotangent)
    after = streaming.attention_path_counts()
    monkeypatch.setattr(streaming._PackedFlashSplit, "apply", original)
    counts = {name: after.get(name, 0) - before.get(name, 0) for name in after}
    return out.detach(), grads, {k: v for k, v in counts.items() if v}, calls


@pytest.mark.parametrize("heads,kv_heads", GROUPINGS)
@pytest.mark.parametrize("pack", PACKS)
def test_the_packed_op_is_each_segments_split(monkeypatch, pack, heads, kv_heads):
    seed = sum(length for length, _ in pack)
    shape = dict(heads=heads, kv_heads=kv_heads)
    per_segment = _attend(monkeypatch, False, pack, seed, **shape)
    again = _attend(monkeypatch, False, pack, seed, **shape)
    packed = _attend(monkeypatch, True, pack, seed, **shape)
    assert packed[3] == [True] and per_segment[3] == []
    assert packed[2] == per_segment[2] == {
        "flash_split": len(pack), "flash_split_primal": len(pack),
    }
    assert torch.equal(packed[0], per_segment[0])
    # the output is the projection's input layout: its reshape is a view
    assert packed[0].is_contiguous()
    floor = float((again[1][0].float() - per_segment[1][0].float()).norm())
    base = float(per_segment[1][0].float().norm())
    difference = float((packed[1][0].float() - per_segment[1][0].float()).norm())
    assert difference <= max(4.0 * floor, 1e-4 * base), (difference, floor, base)
    # own K/V, the shared prompt (one contribution per segment) and every turn
    for index, (got, want) in enumerate(zip(packed[1][1:], per_segment[1][1:], strict=True)):
        assert torch.equal(got, want), index


def test_a_segment_the_split_does_not_take_keeps_the_per_segment_ops(monkeypatch):
    # one query row: the single-turn op takes the expanded split
    pack = [(157, [157]), (1, [42])]
    packed = _attend(monkeypatch, True, pack, 11)
    per_segment = _attend(monkeypatch, False, pack, 11)
    assert packed[3] == []
    assert torch.equal(packed[0], per_segment[0])


def test_the_packed_trial_is_recorded(monkeypatch):
    capability.reset_kernel_probes()
    _attend(monkeypatch, True, [(8, [4]), (6, [])], 3, heads=4, kv_heads=2)
    records = {record["kernel"]: record for record in capability.kernel_probe_records()}
    assert records["flash_split_packed"]["runs"] is True
    assert records["flash_split_packed"]["shape_key"] == HEAD_DIM
    capability.reset_kernel_probes()

    def refuse(*_args):
        raise RuntimeError("varlen backward refused")

    monkeypatch.setattr(streaming, "_packed_flash_split_trial", refuse)
    refused = _attend(monkeypatch, True, [(8, [4]), (6, [])], 3, heads=4, kv_heads=2)
    assert refused[3] == []
    records = {record["kernel"]: record for record in capability.kernel_probe_records()}
    assert records["flash_split_packed"]["runs"] is False
    capability.reset_kernel_probes()
