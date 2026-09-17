"""Correctness of the Triton QSA decode kernels against their torch references.

Covers the paged indexer MQA logits kernel and the split-KV sparse-GQA decode
kernel, both of which run inside the decode CUDA graph.
"""

import math

import pytest
import torch

from sglang.srt.layers.attention.qsa.mqa import (
    torch_qsa_mqa_decode,
    triton_qsa_mqa_decode,
)
from sglang.srt.layers.attention.qsa.sparse_attn import (
    qwen_sparse_fa2_cu_seqlens_triton,
    sparse_gqa_packed_decode_triton,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=60, stage="stage-b", runner_config="1-gpu-large-amd")

PAGE_SIZE = 64
HEAD_DIM = 128
NUM_HEADS = 4
CACHE_PAGES = 512

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a GPU"
)


def _decode_inputs(batch, max_pages, context_lens, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        batch,
        NUM_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    k_cache = torch.randn(
        CACHE_PAGES,
        PAGE_SIZE,
        1,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    page_table = torch.randint(
        0,
        CACHE_PAGES,
        (batch, max_pages),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    lengths = torch.tensor(context_lens, dtype=torch.int32, device="cuda")
    return q, k_cache, page_table, lengths


def _assert_matches_reference(q, k_cache, page_table, lengths, max_model_len):
    expected = torch_qsa_mqa_decode(q, k_cache, page_table, lengths, max_model_len)
    actual = triton_qsa_mqa_decode(q, k_cache, page_table, lengths, max_model_len)
    finite = torch.isfinite(expected)
    assert torch.equal(torch.isfinite(actual), finite)
    # Both paths multiply bf16 inputs and accumulate in fp32; only the
    # summation order differs, so the tolerance is at fp32 rounding level.
    torch.testing.assert_close(actual[finite], expected[finite], rtol=1e-5, atol=1e-4)


@requires_gpu
@pytest.mark.parametrize("batch", [1, 2, 8, 32])
def test_triton_mqa_decode_matches_torch_on_ragged_contexts(batch):
    max_pages = 24
    lengths = [1 + (index * 137) % (max_pages * PAGE_SIZE) for index in range(batch)]
    q, k_cache, page_table, context_lens = _decode_inputs(batch, max_pages, lengths)
    _assert_matches_reference(
        q, k_cache, page_table, context_lens, max_pages * PAGE_SIZE
    )


@requires_gpu
def test_triton_mqa_decode_masks_page_table_padding():
    """A negative page id is padding, not a cache slot: the torch reference
    clamps it to page 0 and the context length masks the row away."""

    max_pages = 8
    q, k_cache, page_table, context_lens = _decode_inputs(
        2, max_pages, [PAGE_SIZE * 3, 100]
    )
    page_table[:, 3:] = -1
    _assert_matches_reference(
        q, k_cache, page_table, context_lens, max_pages * PAGE_SIZE
    )


@requires_gpu
@pytest.mark.parametrize("max_model_len", [PAGE_SIZE, 320, 8 * PAGE_SIZE + 96])
def test_triton_mqa_decode_honours_output_width(max_model_len):
    """Rows are -inf past the page table even when the context reaches further,
    and are truncated when the output is narrower than the table."""

    max_pages = 8
    q, k_cache, page_table, context_lens = _decode_inputs(
        3, max_pages, [1, 4 * PAGE_SIZE, 1 << 20]
    )
    _assert_matches_reference(q, k_cache, page_table, context_lens, max_model_len)


@requires_gpu
def test_triton_mqa_decode_replays_under_cuda_graph():
    """Decode runs inside a replayed graph, so the kernel must not allocate on
    the host, sync, or read shapes off device tensors."""

    max_pages = 16
    max_model_len = max_pages * PAGE_SIZE
    q, k_cache, page_table, context_lens = _decode_inputs(
        4, max_pages, [PAGE_SIZE, 300, 900, 1024]
    )

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        triton_qsa_mqa_decode(q, k_cache, page_table, context_lens, max_model_len)
    torch.cuda.current_stream().wait_stream(warmup)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = triton_qsa_mqa_decode(
            q, k_cache, page_table, context_lens, max_model_len
        )
    context_lens.copy_(
        torch.tensor([1, 1023, 64, 512], dtype=torch.int32, device="cuda")
    )
    graph.replay()
    torch.cuda.synchronize()

    expected = torch_qsa_mqa_decode(q, k_cache, page_table, context_lens, max_model_len)
    finite = torch.isfinite(expected)
    assert torch.equal(torch.isfinite(captured), finite)
    torch.testing.assert_close(captured[finite], expected[finite], rtol=1e-5, atol=1e-4)


@requires_gpu
def test_triton_mqa_decode_applies_score_scale():
    max_pages = 4
    q, k_cache, page_table, context_lens = _decode_inputs(
        2, max_pages, [PAGE_SIZE, 200]
    )
    scale = 3.5
    actual = triton_qsa_mqa_decode(
        q, k_cache, page_table, context_lens, max_pages * PAGE_SIZE, score_scale=scale
    )
    default = triton_qsa_mqa_decode(
        q, k_cache, page_table, context_lens, max_pages * PAGE_SIZE
    )
    finite = torch.isfinite(default)
    torch.testing.assert_close(
        actual[finite],
        default[finite] * (math.sqrt(HEAD_DIM) / scale),
        rtol=1e-5,
        atol=1e-4,
    )


NUM_Q_HEADS = 24
NUM_KV_HEADS = 2
ATTN_HEAD_DIM = 256
TOPK = 2051
ATTN_SCALE = ATTN_HEAD_DIM**-0.5


def _packed_decode_inputs(valid_counts, identity, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    batch = len(valid_counts)
    counts = torch.tensor(valid_counts, dtype=torch.int32, device="cuda")
    cu_q = torch.arange(batch + 1, dtype=torch.int32, device="cuda")
    cu_k = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
    cu_k[1:] = torch.cumsum(counts, 0)
    packed = max(int(cu_k[-1].item()), 1)
    q = torch.randn(
        batch,
        NUM_Q_HEADS,
        ATTN_HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    k = torch.randn(
        packed,
        NUM_KV_HEADS,
        ATTN_HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    v = torch.randn(
        packed,
        NUM_KV_HEADS,
        ATTN_HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    indices = None
    if not identity:
        indices = torch.arange(TOPK, dtype=torch.int32, device="cuda").expand(batch, -1)
        indices = indices.masked_fill(indices >= counts[:, None], -1).contiguous()
    return q, k, v, indices, cu_q, cu_k, counts


def _explicit_gqa(q, k, v, cu_k, counts):
    expected = torch.empty_like(q)
    group_size = NUM_Q_HEADS // NUM_KV_HEADS
    for row, length in enumerate(counts.tolist()):
        start, end = int(cu_k[row]), int(cu_k[row + 1])
        for head in range(NUM_Q_HEADS):
            kv_head = head // group_size
            scores = (
                q[row, head].float() @ k[start:end][:length, kv_head].float().T
            ) * ATTN_SCALE
            expected[row, head] = (
                scores.softmax(-1) @ v[start:end][:length, kv_head].float()
            ).to(q.dtype)
    return expected


@requires_gpu
@pytest.mark.parametrize(
    "valid_counts",
    [
        [2048],
        [2048, 1],
        [2048, 1, 17, 64, 65, 1000, TOPK, 3],
        [1 + (index * 97) % TOPK for index in range(32)],
    ],
)
def test_split_decode_matches_explicit_gqa(valid_counts):
    """The flash-decoding rescale must reproduce one-pass softmax; a wrong
    combine shows up as a per-row scale error, not as garbage."""

    q, k, v, indices, cu_q, cu_k, counts = _packed_decode_inputs(
        valid_counts, identity=True
    )
    actual = sparse_gqa_packed_decode_triton(
        q, k, v, indices, cu_q, cu_k, counts, ATTN_SCALE, identity_topk=TOPK
    )
    torch.testing.assert_close(
        actual, _explicit_gqa(q, k, v, cu_k, counts), rtol=2e-2, atol=2e-2
    )


@requires_gpu
def test_split_decode_identity_path_matches_materialized_indices():
    """The identity fast path replaces a [batch, topk] index tensor the host
    used to build every replay; both must select the same rows."""

    valid_counts = [2048, 1, 17, 64, 65, 1000, TOPK, 3]
    args = _packed_decode_inputs(valid_counts, identity=False)
    q, k, v, indices, cu_q, cu_k, counts = args
    materialized = sparse_gqa_packed_decode_triton(
        q, k, v, indices, cu_q, cu_k, counts, ATTN_SCALE
    )
    identity = sparse_gqa_packed_decode_triton(
        q, k, v, None, cu_q, cu_k, counts, ATTN_SCALE, identity_topk=TOPK
    )
    assert torch.equal(identity, materialized)


@requires_gpu
def test_split_decode_accepts_non_contiguous_query():
    """The backend hands the kernel a query slice without a contiguity copy."""

    q, k, v, _, cu_q, cu_k, counts = _packed_decode_inputs([2048, 300], identity=True)
    padded = torch.zeros(
        q.shape[0], NUM_Q_HEADS, 2 * ATTN_HEAD_DIM, dtype=q.dtype, device=q.device
    )
    padded[:, :, :ATTN_HEAD_DIM] = q
    strided = padded[:, :, :ATTN_HEAD_DIM]
    assert not strided.is_contiguous()
    actual = sparse_gqa_packed_decode_triton(
        strided, k, v, None, cu_q, cu_k, counts, ATTN_SCALE, identity_topk=TOPK
    )
    torch.testing.assert_close(
        actual, _explicit_gqa(q, k, v, cu_k, counts), rtol=2e-2, atol=2e-2
    )


@requires_gpu
def test_split_decode_replays_under_cuda_graph():
    q, k, v, _, cu_q, cu_k, counts = _packed_decode_inputs([2048, 900], identity=True)

    def call():
        return sparse_gqa_packed_decode_triton(
            q, k, v, None, cu_q, cu_k, counts, ATTN_SCALE, identity_topk=TOPK
        )

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        call()
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    counts.copy_(torch.tensor([64, 2048], dtype=torch.int32, device="cuda"))
    cu_k.copy_(torch.tensor([0, 64, 2112], dtype=torch.int32, device="cuda"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        captured, _explicit_gqa(q, k, v, cu_k, counts), rtol=2e-2, atol=2e-2
    )


@requires_gpu
@pytest.mark.parametrize("batch", [1, 4, 8, 9, 16, 64])
def test_fa2_cu_seqlens_counts_and_prefix(batch):
    """The fused single-program path is selected only below a batch threshold;
    both sides of that switch must produce identical counts and offsets."""

    generator = torch.Generator(device="cuda").manual_seed(3)
    seq_lens = torch.randint(
        1, 12000, (batch,), dtype=torch.int32, device="cuda", generator=generator
    )
    indices = torch.randint(
        -1, 12000, (batch, TOPK), dtype=torch.int32, device="cuda", generator=generator
    )
    counts = torch.empty(batch, dtype=torch.int32, device="cuda")
    cu_k = torch.empty(batch + 1, dtype=torch.int32, device="cuda")
    qwen_sparse_fa2_cu_seqlens_triton(seq_lens, indices, counts, cu_k, batch, TOPK)

    expected_counts = ((indices >= 0) & (indices < seq_lens[:, None])).sum(1)
    assert torch.equal(counts, expected_counts.to(torch.int32))
    assert int(cu_k[0]) == 0
    assert torch.equal(cu_k[1:], torch.cumsum(expected_counts, 0).to(torch.int32))
