"""Correctness gates for shared-node consumer-adjoint reduction."""

from __future__ import annotations


import pytest

torch = pytest.importorskip("torch")
if getattr(torch, "__spec__", None) is None:
    pytest.skip("torch is a stand-in stub here", allow_module_level=True)


from thundersync.engine.boundary_adjoint import (  # noqa: E402
    reduce_boundary_adjoints,
    triton_boundary_adjoint_available,
)


@pytest.mark.parametrize("degree", [1, 2, 5, 16])
@pytest.mark.parametrize("shape", [(17,), (3, 11), (2, 3, 7)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_torch_reduction_varied_degrees_shapes_and_dtypes(degree, shape, dtype):
    generator = torch.Generator().manual_seed(1000 + degree)
    adjoints = [torch.randn(shape, generator=generator).to(dtype) for _ in range(degree)]
    existing = torch.randn(shape, generator=generator).to(dtype)

    got = reduce_boundary_adjoints(adjoints, existing=existing, backend="torch")
    expected = existing.clone()
    for adjoint in adjoints:
        expected.add_(adjoint)

    assert got is not None
    torch.testing.assert_close(got, expected, rtol=0.0, atol=0.0)
    assert got.data_ptr() != existing.data_ptr()


def test_zero_and_missing_adjoints_are_supported():
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    got = reduce_boundary_adjoints(
        [None, torch.zeros_like(base), base, None],
        backend="torch",
    )
    torch.testing.assert_close(got, base, rtol=0.0, atol=0.0)


def test_all_missing_adjoints_preserve_existing_parent_adjoint():
    existing = torch.randn(9)
    got = reduce_boundary_adjoints([None, None], existing=existing)
    assert got is not None
    torch.testing.assert_close(got, existing, rtol=0.0, atol=0.0)
    assert got.data_ptr() != existing.data_ptr()
    assert reduce_boundary_adjoints([None, None]) is None


def test_input_mismatch_is_rejected():
    with pytest.raises(ValueError, match="shape"):
        reduce_boundary_adjoints([torch.zeros(2), torch.zeros(3)])
    with pytest.raises(ValueError, match="dtype"):
        reduce_boundary_adjoints(
            [torch.zeros(2, dtype=torch.float32)],
            existing=torch.zeros(2, dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="backend"):
        reduce_boundary_adjoints([torch.zeros(2)], backend="cuda")


def test_unsupported_float64_triton_request_uses_reference():
    inputs = [torch.randn(13, dtype=torch.float64) for _ in range(3)]
    expected = reduce_boundary_adjoints(inputs, backend="torch")
    got = reduce_boundary_adjoints(inputs, backend="triton")
    torch.testing.assert_close(got, expected, rtol=0.0, atol=0.0)


def test_higher_order_semantics_use_differentiable_reference():
    a = torch.randn(7, requires_grad=True)
    b = torch.randn(7, requires_grad=True)
    existing = torch.randn(7, requires_grad=True)
    upstream = torch.linspace(-1.0, 1.0, 7)

    out = reduce_boundary_adjoints(
        [a, None, b], existing=existing, backend="triton"
    )
    assert out is not None
    (out * upstream).sum().backward()

    torch.testing.assert_close(a.grad, upstream, rtol=0.0, atol=0.0)
    torch.testing.assert_close(b.grad, upstream, rtol=0.0, atol=0.0)
    torch.testing.assert_close(existing.grad, upstream, rtol=0.0, atol=0.0)

    c = torch.randn(7, requires_grad=True)
    reduced = reduce_boundary_adjoints([c], backend="triton")
    assert reduced is not None
    reduced.sum().backward()
    torch.testing.assert_close(c.grad, torch.ones_like(c), rtol=0.0, atol=0.0)


@pytest.mark.skipif(
    not triton_boundary_adjoint_available(),
    reason="requires CUDA and Triton",
)
@pytest.mark.parametrize("degree", [1, 2, 5, 16])
@pytest.mark.parametrize("shape", [(257,), (7, 513), (2, 3, 65)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_triton_reduction_matches_reference(degree, shape, dtype):
    generator = torch.Generator(device="cuda").manual_seed(2000 + degree)
    adjoints = [
        torch.randn(shape, generator=generator, device="cuda", dtype=dtype)
        for _ in range(degree)
    ]
    # Exercise missing adjoints and accumulation into an existing parent.
    inputs: list[torch.Tensor | None] = [None, *adjoints, None]
    existing = torch.randn(
        shape, generator=generator, device="cuda", dtype=dtype
    )
    sequential = reduce_boundary_adjoints(
        inputs, existing=existing, backend="torch"
    )
    got = reduce_boundary_adjoints(
        inputs, existing=existing, backend="triton"
    )

    # The Triton program accumulates the input dtype's exactly represented
    # values in fp32 and casts once at the output. Sequential in-place PyTorch
    # accumulation casts after every addition for fp16/bf16, so it is an
    # unsuitable numerical oracle at higher node degrees. Construct the oracle
    # with the kernel's declared accumulation semantics and compare after the
    # one required output cast.
    oracle_fp32 = existing.float()
    for adjoint in adjoints:
        oracle_fp32 = oracle_fp32 + adjoint.float()
    oracle = oracle_fp32.to(dtype)

    if dtype is torch.float32:
        rtol = atol = 2e-5
    else:
        # Two output-dtype epsilons cover the final cast near zero and relative
        # quantization away from zero. Intermediate accumulation is fp32.
        rtol = atol = 2.0 * torch.finfo(dtype).eps
    torch.testing.assert_close(got, oracle, rtol=rtol, atol=atol)

    if dtype is torch.bfloat16 and degree == 16:
        # The 2026-08-12 reference-site audit showed a stable separation over all
        # three shapes: Triton relative L2 error 0.00164-0.00167 versus
        # 0.00510-0.00528 for sequential bf16 accumulation.
        scale = torch.linalg.vector_norm(oracle_fp32).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        triton_error = torch.linalg.vector_norm(got.float() - oracle_fp32) / scale
        sequential_error = (
            torch.linalg.vector_norm(sequential.float() - oracle_fp32) / scale
        )
        assert triton_error <= sequential_error
