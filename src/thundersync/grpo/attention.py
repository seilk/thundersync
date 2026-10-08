"""Model-owned cuDNN graphs for GRPO source attention.

The caller opts in after a device forward/backward trial. No process-wide
attention dispatch or kernel flags are changed; graph descriptors are cached
by geometry while CUDA tensors and workspaces belong to each invocation.
"""

from __future__ import annotations

import time

import torch

def graph_attention(scale, *, shared_handle=None, workspace_provider=None, query_chunk=0,
                          kv_bucket=0, planning_records=None, padded_query=0, padded_kv=0):
    """Native lower-right attention with explicit deterministic backward selection.

    Query chunks trim causally invisible keys; optional padding reuses plans
    while the actual lengths keep each chunk's causal diagonal unchanged.
    Workspaces are allocated per operation, never retained by the plan cache.
    """
    import cudnn

    if query_chunk:
        chunks = {}

        def chunked(q, k, v):
            prefix = k.shape[2] - q.shape[2]
            outputs = []
            for start in range(0, q.shape[2], query_chunk):
                end = min(start + query_chunk, q.shape[2])
                # Rows after this chunk are causally invisible. Trimming K/V
                # keeps the chunk's bottom-right diagonal at prefix + start.
                inputs = (q[:, :, start:end], k[:, :, :prefix + end], v[:, :, :prefix + end])
                query_capacity = min(query_chunk, 1 << (end - start - 1).bit_length())
                key_capacity = (max(query_capacity, ((inputs[1].shape[2] + kv_bucket - 1) // kv_bucket) * kv_bucket)
                                if kv_bucket else 0)
                signature = (key_capacity, query_capacity) if kv_bucket else tuple((tuple(x.shape), x.stride()) for x in inputs)
                if signature not in chunks:
                    chunks[signature] = graph_attention(
                        scale, shared_handle=shared_handle, workspace_provider=workspace_provider,
                        padded_query=query_capacity if kv_bucket else 0, padded_kv=key_capacity,
                        planning_records=planning_records)
                outputs.append(chunks[signature](*inputs))
            return torch.cat(outputs, dim=2) if len(outputs) > 1 else outputs[0]

        return chunked

    cache = None

    class Attention(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v):
            nonlocal cache
            ctx.original_lengths = q.shape[2], k.shape[2]
            lengths = []
            if padded_query:
                originals = q, k, v
                padded = [torch.zeros((*x.shape[:2], capacity, x.shape[3]), device=x.device, dtype=x.dtype)
                          for x, capacity in zip(originals, (padded_query, padded_kv, padded_kv), strict=True)]
                for destination, original in zip(padded, originals, strict=True):
                    destination[:, :, :original.shape[2]].copy_(original)
                q, k, v = padded
                lengths = [torch.full((1, 1, 1, 1), value, device=q.device, dtype=torch.int32)
                           for value in ctx.original_lengths]
            out = torch.empty_like(q)
            stats = torch.empty((*q.shape[:3], 1), device=q.device, dtype=torch.float32)
            if cache is None:
                planning_started = time.perf_counter()
                handle = cudnn.create_handle() if shared_handle is None else shared_handle
                graphs, bindings = [], []
                for backward in (False, True):
                    graph = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16,
                                          intermediate_data_type=cudnn.data_type.FLOAT,
                                          compute_data_type=cudnn.data_type.FLOAT,
                                          handle=handle)
                    samples = [q, k, v, out, torch.empty_like(out, memory_format=torch.contiguous_format), stats] if backward else [q, k, v]
                    inputs = [graph.tensor_like(value.detach()) for value in samples]
                    options = dict(attn_scale=scale, diagonal_alignment=cudnn.diagonal_alignment.BOTTOM_RIGHT,
                                   diagonal_band_right_bound=0)
                    if lengths:
                        length_inputs = [graph.tensor_like(value.detach()) for value in lengths]
                        options.update(use_padding_mask=True, seq_len_q=length_inputs[0], seq_len_kv=length_inputs[1])
                    else:
                        length_inputs = []
                    if backward:
                        outputs = graph.sdpa_backward(*inputs, **options,
                                                      use_deterministic_algorithm=torch.are_deterministic_algorithms_enabled())
                        destinations = [torch.empty_like(value) for value in (q, k, v)]
                    else:
                        outputs = graph.sdpa(*inputs, **options, is_inference=False)
                        destinations = [out, stats]
                    for tensor, destination in zip(outputs, destinations, strict=True):
                        tensor.set_output(True).set_dim(destination.shape).set_stride(destination.stride())
                        if destination.dtype == torch.float32:
                            tensor.set_data_type(cudnn.data_type.FLOAT)
                    graph.validate()
                    graph.build_operation_graph()
                    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
                    graph.check_support()
                    graph.build_plans()
                    graphs.append(graph)
                    bindings.append((inputs, outputs, length_inputs))
                cache = handle, graphs, bindings
                if planning_records is not None:
                    planning_records.append({"s": time.perf_counter() - planning_started,
                                             "q_shape": list(q.shape), "k_shape": list(k.shape),
                                             "q_stride": list(q.stride()), "k_stride": list(k.stride()),
                                             "workspace_bytes": [graph.get_workspace_size() for graph in graphs]})
            handle, graphs, bindings = cache
            required = graphs[0].get_workspace_size()
            workspace = (torch.empty(required, device=q.device, dtype=torch.uint8)
                         if workspace_provider is None else workspace_provider(required, q.device))
            cudnn.set_stream(handle, torch.cuda.current_stream(q.device).cuda_stream)
            inputs, outputs, length_inputs = bindings[0]
            graphs[0].execute(dict(zip([*inputs, *outputs, *length_inputs],
                                      [value.detach() for value in (q, k, v, out, stats, *lengths)], strict=True)),
                              workspace, handle=handle)
            ctx.save_for_backward(q, k, v, out, stats, *lengths)
            return out[:, :, :ctx.original_lengths[0]] if padded_query else out

        @staticmethod
        def backward(ctx, dout):
            q, k, v, out, stats, *lengths = ctx.saved_tensors
            if padded_query:
                padded_dout = torch.zeros_like(out, memory_format=torch.contiguous_format)
                padded_dout[:, :, :dout.shape[2]].copy_(dout)
                dout = padded_dout
            gradients = [torch.empty_like(value) for value in (q, k, v)]
            handle, graphs, bindings = cache
            required = graphs[1].get_workspace_size()
            workspace = (torch.empty(required, device=q.device, dtype=torch.uint8)
                         if workspace_provider is None else workspace_provider(required, q.device))
            cudnn.set_stream(handle, torch.cuda.current_stream(q.device).cuda_stream)
            inputs, outputs, length_inputs = bindings[1]
            graphs[1].execute(dict(zip([*inputs, *outputs, *length_inputs],
                                      [value.detach() for value in (q, k, v, out, dout.contiguous(), stats, *gradients, *lengths)],
                                      strict=True)),
                              workspace, handle=handle)
            if padded_query:
                return tuple(value[:, :, :length] for value, length in zip(
                    gradients, (ctx.original_lengths[0], ctx.original_lengths[1], ctx.original_lengths[1]), strict=True))
            return tuple(gradients)

    return Attention.apply


def source_graph_attention(*, query_chunk: int, kv_bucket: int):
    """One model's source backend, handle and descriptor cache."""
    import cudnn
    from thundersync.engine.streaming import count_attention_path, grouped_key_value

    handle = cudnn.create_handle()
    cache = {}
    planning_records = []

    def attention(query, keys, values, scale):
        if query.dtype != torch.bfloat16:
            raise RuntimeError("cuDNN graph source attention requires bfloat16 inputs")
        key, value = grouped_key_value(keys, values)
        geometry = (query.dtype, query.shape[1], key.shape[1], query.shape[-1]) if kv_bucket else tuple(
            (tensor.dtype, tuple(tensor.shape), tensor.stride()) for tensor in (query, key, value))
        signature = (query.device, scale, torch.are_deterministic_algorithms_enabled(), geometry)
        if signature not in cache:
            cache[signature] = graph_attention(scale, shared_handle=handle,
                                               query_chunk=query_chunk, kv_bucket=kv_bucket,
                                               planning_records=planning_records)
        count_attention_path("cudnn_graph_lower_right")
        return cache[signature](query, key, value)

    attention.planning_records = planning_records
    return attention
