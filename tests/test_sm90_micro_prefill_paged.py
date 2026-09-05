import pytest
import torch

from stream_attention.paged import PagedKVCache
from stream_attention.backends.sm90.micro_prefill_paged import validate_paged_micro_prefill
from stream_attention.backends.sm90.micro_prefill_paged_sources import CPP_SOURCE, paged_cuda_source
from benchmarks.profile_sm90_micro_prefill_paged import experiment_cases, reference, set_metadata


def buffers(layout="NHD"):
    q = torch.zeros(2, 3, 8, 64, dtype=torch.float16)
    shape = (4, 16, 2, 64) if layout == "NHD" else (4, 2, 16, 64)
    cache = PagedKVCache(
        torch.zeros(shape, dtype=q.dtype), torch.zeros(shape, dtype=q.dtype),
        torch.tensor([[3, -1], [-1, -1]], dtype=torch.int32),
        torch.tensor([15, 0], dtype=torch.int32), layout,
    )
    return q, cache, torch.tensor([3, 0], dtype=torch.int32)


@pytest.mark.parametrize("layout", ["HND", "NHD"])
def test_paged_contract(layout):
    q, cache, ql = buffers(layout)
    validate_paged_micro_prefill(q, cache, ql)
    with pytest.raises(ValueError, match="decode query"):
        cache.validate(q)
    with pytest.raises(ValueError, match="positive"):
        cache.validate(q[:, :1].contiguous())
    cache.sequence_lengths[1] = 1
    with pytest.raises(ValueError, match="physical pages"):
        validate_paged_micro_prefill(q, cache, ql)


@pytest.mark.parametrize("field,value,message", [
    ("query", -1, "query_lengths"), ("query", 4, "query_lengths"),
    ("kv", -1, "nonnegative"), ("kv", 33, "capacity"),
    ("page", 4, "physical pages"),
])
def test_invalid_metadata(field, value, message):
    q, cache, ql = buffers()
    target = ql if field == "query" else cache.sequence_lengths if field == "kv" else cache.page_table[0]
    target[0] = value
    with pytest.raises(ValueError, match=message):
        validate_paged_micro_prefill(q, cache, ql)


def test_logical_position_capacity_is_not_physical_pages():
    q, cache, ql = buffers()
    qp = torch.zeros(2, 3, dtype=torch.int64)
    kp = torch.zeros(2, 32, dtype=torch.int64)
    validate_paged_micro_prefill(q, cache, ql, causal=True, query_positions=qp, key_positions=kp)
    with pytest.raises(ValueError, match="key_positions"):
        validate_paged_micro_prefill(q, cache, ql, causal=True, query_positions=qp, key_positions=kp[:, :16])
    with pytest.raises(ValueError, match="positions require"):
        validate_paged_micro_prefill(q, cache, ql, query_positions=qp)


@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("causal", [False, True])
def test_native_source_reuses_mma_and_zero_fills(d, dtype, causal):
    source = paged_cuda_source(d, dtype, causal)
    assert "SM80_CP_ASYNC_CACHEGLOBAL_ZFILL" in source
    assert "if (valid) {\n      const int page = table[" in source
    assert "partial_kernel<16,true,kNHD>" in source
    assert "partial_kernel<kNHD><<<" in source
    assert "qi >= valid_queries || ki >= sequence_length" in source
    assert "query_begin >= valid_queries" in source
    assert "query_lengths[cache_group / kv_heads]" in source
    assert "streamattn_micro_load_page16<kNHD, true, false>" in source
    assert "torch::Tensor pt" in CPP_SOURCE
    assert "gather" not in source.lower()


def test_cases_cover_both_layouts_and_empty_cache_mutation():
    cases = experiment_cases("full")
    assert len(cases) == 144
    assert {c["layout"] for c in cases} == {"HND", "NHD"}
    assert {c["m"] for c in cases} >= {2, 3, 17, 32, 64}
    for layout in ("HND", "NHD"):
        q, cache, ql = buffers(layout)
        c = dict(batch=2, m=3, pages=2, mask="append")
        qp, kp = torch.empty(2, 3, dtype=torch.int64), torch.empty(2, 32, dtype=torch.int64)
        for mutate, empty in ((False, False), (True, False), (True, True)):
            set_metadata(c, q, cache, ql, qp, kp, mutate=mutate, empty=empty)
            validate_paged_micro_prefill(q, cache, ql)
            output, lse = reference(q, cache, ql, qp, kp, True)
            assert torch.isfinite(output).all()
            assert not torch.isnan(lse).any()
            if empty:
                assert (output == 0).all() and torch.isneginf(lse).all()


def test_b4_mutated_queries_cannot_exceed_capacity():
    q = torch.empty(4, 3, 16, 64, dtype=torch.float16)
    cache = PagedKVCache(
        torch.empty(23, 16, 4, 64, dtype=q.dtype), torch.empty(23, 16, 4, 64, dtype=q.dtype),
        torch.empty(4, 5, dtype=torch.int32), torch.empty(4, dtype=torch.int32),
    )
    ql = torch.empty(4, dtype=torch.int32)
    qp, kp = torch.empty(4, 3, dtype=torch.int64), torch.empty(4, 80, dtype=torch.int64)
    c = dict(batch=4, m=3, pages=5, mask="permuted")
    set_metadata(c, q, cache, ql, qp, kp, mutate=True)
    assert ql.tolist() == [3, 3, 2, 1]
    validate_paged_micro_prefill(q, cache, ql)


@pytest.mark.parametrize("layout", ["HND", "NHD"])
def test_h100_paged_graph_replay(tmp_path, layout):
    from types import SimpleNamespace
    from stream_attention.backends.sm90.transposed_gqa_exact import resolve_cutlass_root
    from benchmarks.profile_sm90_micro_prefill_paged import profile_case

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires an SM90 GPU")
    try:
        root = resolve_cutlass_root()
    except FileNotFoundError:
        pytest.skip("requires CUTLASS headers")
    case = next(c for c in experiment_cases("smoke") if c["layout"] == layout)
    args = SimpleNamespace(seed=9103, cutlass_root=root, build_dir=tmp_path)
    assert profile_case(case, args)["passed"]
