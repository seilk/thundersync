"""Forest attention must equal running every trajectory on its own.

Two independent checks, because they fail differently:

* against `reference_attention` -- catches errors in the LSE merge, since the
  reference builds the same visibility mask and does a plain softmax;
* against a genuinely separate per-trajectory causal forward -- catches errors
  in the *plan*, which the reference would share and therefore never expose.

The second is the one that certifies the sharing itself.
"""

from __future__ import annotations

import math

import pytest
import torch

from thundersync.engine.segments import Segment, Trajectory, build_forest
from thundersync.engine.plan import compile_plan# noqa: E402
from thundersync.engine.attention import forest_attention, reference_attention  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="flash attention needs a GPU"
)

DEV = "cuda"
DT = torch.float16  # flash requires half precision


def make_group(n_traj, prompt_len, seed=0, groups=1, scale=1):
    """`scale` lengthens the private branches without lengthening the prompt."""
    g = torch.Generator().manual_seed(seed)
    trajs, tid = [], 0
    for gi in range(groups):
        prompt = torch.randint(0, 900, (prompt_len,), generator=g).tolist()
        for i in range(n_traj):
            segs = [Segment.prompt(prompt)]
            for k in range(1 + i % 3):
                a = (4 + k + i) * scale
                o = (7 + i) * scale
                segs.append(Segment.action(torch.randint(0, 900, (a,), generator=g).tolist()))
                segs.append(
                    Segment.observation(torch.randint(0, 900, (o,), generator=g).tolist())
                )
            trajs.append(Trajectory(segs, group_id=gi, traj_id=tid))
            tid += 1
    return trajs


def qkv(p, heads, kv_heads, dim, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    mk = lambda h: torch.randn(  # noqa: E731
        p, h, dim, generator=g, device=DEV, dtype=DT, requires_grad=True
    )
    return mk(heads), mk(kv_heads), mk(kv_heads)


@pytest.mark.parametrize("heads,kv_heads", [(4, 4), (8, 2)])
def test_matches_dense_reference(heads, kv_heads):
    plan = compile_plan(build_forest(make_group(5, 32)), device=DEV)
    q, k, v = qkv(plan.n_physical, heads, kv_heads, 32)
    got = forest_attention(q, k, v, plan)
    want = reference_attention(q, k, v, plan)
    torch.testing.assert_close(got.float(), want.float(), rtol=2e-3, atol=2e-3)


def test_matches_independent_per_trajectory_forward():
    """The claim that justifies sharing at all, checked without sharing."""
    trajs = make_group(6, 48, seed=3)
    plan = compile_plan(build_forest(trajs), device=DEV)
    heads, kv_heads, dim = 8, 2, 64
    q, k, v = qkv(plan.n_physical, heads, kv_heads, dim, seed=5)
    shared = forest_attention(q, k, v, plan)

    scale = 1.0 / math.sqrt(dim)
    for t in trajs:
        path = plan.physical_path(t.traj_id)
        idx = torch.tensor(path, device=DEV)
        # a plain causal attention over this trajectory alone
        qi, ki, vi = q[idx], k[idx], v[idx]
        if heads != kv_heads:
            ki = ki.repeat_interleave(heads // kv_heads, dim=1)
            vi = vi.repeat_interleave(heads // kv_heads, dim=1)
        n = len(path)
        causal = torch.tril(torch.ones(n, n, dtype=torch.bool, device=DEV))
        s = torch.einsum("qhd,khd->hqk", qi.float(), ki.float()) * scale
        s = s.masked_fill(~causal.unsqueeze(0), float("-inf"))
        want = torch.einsum("hqk,khd->qhd", s.softmax(-1), vi.float())
        torch.testing.assert_close(
            shared[idx].float(), want, rtol=2e-3, atol=2e-3,
            msg=f"trajectory {t.traj_id} differs from its independent forward",
        )


def test_gradients_match_the_reference():
    plan = compile_plan(build_forest(make_group(4, 24, seed=7)), device=DEV)
    q, k, v = qkv(plan.n_physical, 4, 4, 32, seed=9)
    w = torch.randn_like(q)

    (forest_attention(q, k, v, plan).float() * w.float()).sum().backward()
    ga = [x.grad.clone() for x in (q, k, v)]
    for x in (q, k, v):
        x.grad = None
    (reference_attention(q, k, v, plan).float() * w.float()).sum().backward()
    gb = [x.grad.clone() for x in (q, k, v)]

    for name, a, b in zip("qkv", ga, gb):
        torch.testing.assert_close(
            a.float(), b.float(), rtol=5e-2, atol=5e-3, msg=f"d{name} differs"
        )


def test_multiple_groups_stay_isolated():
    """Separate tasks share nothing; a leak across groups must not be possible."""
    plan = compile_plan(build_forest(make_group(3, 20, seed=11, groups=3)), device=DEV)
    q, k, v = qkv(plan.n_physical, 4, 4, 32, seed=13)
    got = forest_attention(q, k, v, plan)
    want = reference_attention(q, k, v, plan)
    torch.testing.assert_close(got.float(), want.float(), rtol=2e-3, atol=2e-3)


def test_no_quadratic_allocation():
    """The failure mode of the previous attempt: the mask itself was the cost.

    Needs P large enough that a `[P, P]` matrix dominates the linear tensors,
    otherwise the bound is vacuous -- so the branches are lengthened rather than
    the prompt, which is also the regime the claim is about.
    """
    plan = compile_plan(
        build_forest(make_group(8, 4096, seed=17, scale=256)), device=DEV
    )
    p = plan.n_physical
    assert p > 20_000, f"P={p} is too small for this bound to mean anything"
    q, k, v = qkv(p, 8, 2, 64, seed=19)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    forest_attention(q, k, v, plan).sum().backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    quadratic = p * p * 2  # a [P,P] fp16 score matrix, once
    assert peak < quadratic, (
        f"peak {peak/2**30:.2f} GiB is above a single [P,P] matrix "
        f"({quadratic/2**30:.2f} GiB) at P={p}; something is materialising the mask"
    )
