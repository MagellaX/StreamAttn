import inspect
import math
from pathlib import Path

import pytest
import torch

import stream_attention as stream_attn
from stream_attention.paged import (
    PAGED_EXACT_NATIVE_BACKEND,
    PAGED_EXACT_SM80_CP_ASYNC_BACKEND,
    PAGED_EXACT_SM80_GROUPED_BACKEND,
    PAGED_EXACT_SM100_GROUPED_BACKEND,
    PAGED_EXACT_SM100_TGV_BACKEND,
    PROMOTED_PAGED_EXACT_SM100_TGV_SPLITS,
    PAGED_EXACT_SM90_FRAGMENTED_BACKEND,
    PAGED_EXACT_SM90_FRAGMENTED_RAGGED_BACKEND,
    PAGED_EXACT_SM90_NHD_FRAGMENTED_BACKEND,
    PAGED_EXACT_SM90_NHD_FRAGMENTED_RAGGED_BACKEND,
    PROMOTED_PAGED_EXACT_SPLITS,
    PROMOTED_PAGED_EXACT_PAGE16_D128_G4_RAGGED_SPLITS,
    PROMOTED_PAGED_EXACT_PAGE16_D128_G4_SPLITS,
    PROMOTED_PAGED_EXACT_PAGE16_D128_G8_RAGGED_SPLITS,
    PROMOTED_PAGED_EXACT_PAGE16_D128_G8_SPLITS,
    PROMOTED_PAGED_EXACT_PAGE16_NHD_D128_G8_RAGGED_SPLITS,
    PROMOTED_PAGED_EXACT_PAGE16_NHD_D128_G8_SPLITS,
    PROMOTED_PAGED_EXACT_PAGE16_NHD_RAGGED_SHAPES,
    PROMOTED_PAGED_EXACT_PAGE16_NHD_SHAPES,
    PROMOTED_PAGED_EXACT_PAGE16_SPLITS,
    PROMOTED_PAGED_EXACT_PAGE16_RAGGED_SPLITS,
    PROMOTED_PAGED_EXACT_PAGE16_RAGGED_SHAPES,
    PROMOTED_PAGED_EXACT_PAGE16_SHAPES,
    PagedExactDecodePlan,
    PagedKVCache,
    choose_paged_exact_splits,
    paged_exact_reference,
)
from stream_attention.backends.sm90.transposed_gqa_exact_sources import (
    CPP_SOURCE,
    CUDA_SOURCE,
    cuda_source_for_head_dim,
)
from stream_attention.backends.sm80.paged_gqa_exact_sources import (
    CPP_SOURCE as SM80_CPP_SOURCE,
    CUDA_SOURCE as SM80_CUDA_SOURCE,
)


def _make_paged_inputs(*, layout: str = "NHD", device: str = "cpu"):
    torch.manual_seed(17)
    batch = 2
    query_heads = 4
    kv_heads = 2
    dim = 8
    page_size = 4
    num_pages = 5
    query = torch.randn(batch, 1, query_heads, dim, device=device, dtype=torch.float32)
    key_nhd = torch.randn(
        num_pages, page_size, kv_heads, dim, device=device, dtype=torch.float32
    )
    value_nhd = torch.randn_like(key_nhd)
    page_table = torch.tensor([[3, 0, -1], [1, 4, 2]], device=device, dtype=torch.int32)
    sequence_lengths = torch.tensor([7, 10], device=device, dtype=torch.int32)
    if layout == "NHD":
        key, value = key_nhd, value_nhd
    else:
        key = key_nhd.permute(0, 2, 1, 3).contiguous()
        value = value_nhd.permute(0, 2, 1, 3).contiguous()
    cache = PagedKVCache(
        key=key,
        value=value,
        page_table=page_table,
        sequence_lengths=sequence_lengths,
        layout=layout,
    )
    return query, cache


def _dense_expected(query: torch.Tensor, cache: PagedKVCache) -> torch.Tensor:
    output = torch.empty_like(query)
    group_size = query.shape[2] // cache.kv_heads
    scale = 1.0 / math.sqrt(float(query.shape[3]))
    for batch_idx in range(query.shape[0]):
        sequence_length = int(cache.sequence_lengths[batch_idx].item())
        pages = []
        active_pages = (sequence_length + cache.page_size - 1) // cache.page_size
        for logical_page in range(active_pages):
            physical_page = int(cache.page_table[batch_idx, logical_page].item())
            if cache.normalized_layout == "NHD":
                pages.append(cache.key[physical_page])
            else:
                pages.append(cache.key[physical_page].transpose(0, 1))
        keys = torch.cat(pages, dim=0)[:sequence_length]
        pages = []
        for logical_page in range(active_pages):
            physical_page = int(cache.page_table[batch_idx, logical_page].item())
            if cache.normalized_layout == "NHD":
                pages.append(cache.value[physical_page])
            else:
                pages.append(cache.value[physical_page].transpose(0, 1))
        values = torch.cat(pages, dim=0)[:sequence_length]
        for head_idx in range(query.shape[2]):
            kv_head = head_idx // group_size
            scores = (
                keys[:, kv_head].float() @ query[batch_idx, 0, head_idx].float()
            ) * scale
            output[batch_idx, 0, head_idx] = (
                scores.softmax(dim=0)[:, None] * values[:, kv_head].float()
            ).sum(dim=0)
    return output


@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_paged_reference_matches_dense_without_repacking_runtime(layout):
    query, cache = _make_paged_inputs(layout=layout)

    output = paged_exact_reference(query, cache)
    expected = _dense_expected(query, cache)

    torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)


def test_paged_plan_reuses_output_and_observes_page_mutation():
    query, cache = _make_paged_inputs()
    plan = PagedExactDecodePlan.build(query, cache)

    first = plan.run().clone()
    output_pointer = plan.output.data_ptr()
    cache.key[3].add_(0.5)
    second = plan.run()

    assert second.data_ptr() == output_pointer
    assert not torch.equal(first, second)
    assert plan.workspace_bytes == 0
    assert plan.backend == "torch_paged_exact_reference"


def test_public_decode_accepts_paged_cache_and_verified_auto_fails_closed_exact():
    query, cache = _make_paged_inputs()

    output, info = stream_attn.decode(query, cache, mode="verified_auto")
    expected = _dense_expected(query, cache)

    torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)
    assert info.backend_used == "torch_paged_exact_reference"
    assert info.plan_reason == "paged_kv_exact_only"
    assert info.seed_only_enabled is False
    assert info.stats["page_size"] == 4


def test_public_decode_rejects_seed_only_and_redundant_value_cache():
    query, cache = _make_paged_inputs()

    with pytest.raises(ValueError, match="does not yet support paged KV"):
        stream_attn.decode(query, cache, mode="seed_only_native")
    with pytest.raises(ValueError, match="value_cache must be omitted"):
        stream_attn.decode(query, cache, torch.empty_like(query), mode="exact_native")


def test_paged_metadata_validation_rejects_bad_active_page_and_length():
    query, cache = _make_paged_inputs()
    bad_table = cache.page_table.clone()
    bad_table[0, 1] = cache.num_pages
    bad_page_cache = PagedKVCache(
        cache.key,
        cache.value,
        bad_table,
        cache.sequence_lengths,
        cache.layout,
    )
    with pytest.raises(ValueError, match="active page_table"):
        bad_page_cache.validate(query)

    bad_lengths = cache.sequence_lengths.clone()
    bad_lengths[1] = cache.max_sequence_length + 1
    bad_length_cache = PagedKVCache(
        cache.key,
        cache.value,
        cache.page_table,
        bad_lengths,
        cache.layout,
    )
    with pytest.raises(ValueError, match="exceed page_table capacity"):
        bad_length_cache.validate(query)


def test_paged_split_rule_targets_producer_parallelism():
    assert (
        choose_paged_exact_splits(batch=1, query_heads=16, max_pages_per_request=2048)
        == 32
    )
    assert (
        choose_paged_exact_splits(batch=4, query_heads=16, max_pages_per_request=2048)
        == 8
    )
    assert (
        choose_paged_exact_splits(batch=8, query_heads=16, max_pages_per_request=2048)
        == 4
    )


def test_promoted_paged_sm90_cells_and_source_contract():
    assert PROMOTED_PAGED_EXACT_SPLITS[(1, 16384)] == 64
    assert PROMOTED_PAGED_EXACT_SPLITS[(4, 32768)] == 32
    assert PROMOTED_PAGED_EXACT_SPLITS[(8, 65536)] == 32
    assert len(PROMOTED_PAGED_EXACT_SPLITS) == 12
    assert PROMOTED_PAGED_EXACT_PAGE16_SPLITS[(1, 32768)] == 64
    assert PROMOTED_PAGED_EXACT_PAGE16_SPLITS[(4, 32768)] == 64
    assert PROMOTED_PAGED_EXACT_PAGE16_SPLITS[(8, 65536)] == 32
    assert len(PROMOTED_PAGED_EXACT_PAGE16_SPLITS) == 12
    assert PROMOTED_PAGED_EXACT_PAGE16_RAGGED_SPLITS == (
        PROMOTED_PAGED_EXACT_PAGE16_SPLITS
    )
    assert PROMOTED_PAGED_EXACT_PAGE16_D128_G8_SPLITS[(1, 65536)] == 128
    assert PROMOTED_PAGED_EXACT_PAGE16_D128_G8_SPLITS[(8, 65536)] == 16
    assert PROMOTED_PAGED_EXACT_PAGE16_D128_G8_RAGGED_SPLITS[(8, 65536)] == 24
    assert PROMOTED_PAGED_EXACT_PAGE16_D128_G4_SPLITS[(1, 65536)] == 32
    assert PROMOTED_PAGED_EXACT_PAGE16_D128_G4_SPLITS[(8, 65536)] == 8
    assert PROMOTED_PAGED_EXACT_PAGE16_D128_G4_RAGGED_SPLITS[(4, 32768)] == 12
    assert PROMOTED_PAGED_EXACT_PAGE16_D128_G4_RAGGED_SPLITS[(4, 65536)] == 16
    assert PROMOTED_PAGED_EXACT_PAGE16_D128_G4_RAGGED_SPLITS[(8, 32768)] == 12
    assert PROMOTED_PAGED_EXACT_PAGE16_D128_G4_RAGGED_SPLITS[(8, 65536)] == 16
    assert PROMOTED_PAGED_EXACT_PAGE16_SHAPES == {
        (16, 2, 8, 64): PROMOTED_PAGED_EXACT_PAGE16_SPLITS,
        (16, 2, 8, 128): PROMOTED_PAGED_EXACT_PAGE16_D128_G8_SPLITS,
        (32, 8, 4, 128): PROMOTED_PAGED_EXACT_PAGE16_D128_G4_SPLITS,
    }
    assert PROMOTED_PAGED_EXACT_PAGE16_RAGGED_SHAPES == {
        (16, 2, 8, 64): PROMOTED_PAGED_EXACT_PAGE16_RAGGED_SPLITS,
        (16, 2, 8, 128): PROMOTED_PAGED_EXACT_PAGE16_D128_G8_RAGGED_SPLITS,
        (32, 8, 4, 128): PROMOTED_PAGED_EXACT_PAGE16_D128_G4_RAGGED_SPLITS,
    }
    assert PROMOTED_PAGED_EXACT_PAGE16_NHD_D128_G8_SPLITS[(1, 32768)] == 128
    assert PROMOTED_PAGED_EXACT_PAGE16_NHD_D128_G8_SPLITS[(8, 65536)] == 32
    assert PROMOTED_PAGED_EXACT_PAGE16_NHD_D128_G8_RAGGED_SPLITS[(8, 16384)] == 16
    assert PROMOTED_PAGED_EXACT_PAGE16_NHD_SHAPES == {
        (16, 2, 8, 128): PROMOTED_PAGED_EXACT_PAGE16_NHD_D128_G8_SPLITS,
    }
    assert PROMOTED_PAGED_EXACT_PAGE16_NHD_RAGGED_SHAPES == {
        (16, 2, 8, 128): PROMOTED_PAGED_EXACT_PAGE16_NHD_D128_G8_RAGGED_SPLITS,
    }
    assert "paged_exact_decode_out" in CPP_SOURCE
    assert "paged_fragmented_exact_decode_out" in CPP_SOURCE
    assert "paged_fragmented_ragged_exact_decode_out" in CPP_SOURCE
    assert "paged_fragmented_nhd_exact_decode_out" in CPP_SOURCE
    assert "paged_fragmented_nhd_ragged_exact_decode_out" in CPP_SOURCE
    assert "streamattn_exact_tile_ptr<kPagedPageSize>" in CUDA_SOURCE
    assert "streamattn_transposed_wgmma_exact_partial_kernel<64>" in CUDA_SOURCE
    assert "streamattn_transposed_wgmma_exact_partial_kernel<16>" in CUDA_SOURCE
    assert "streamattn_transposed_wgmma_exact_partial_kernel<16, true>" in CUDA_SOURCE
    assert "using SmemLayoutPaged16" in CUDA_SOURCE
    assert "streamattn_copy_paged16_tile" in CUDA_SOURCE
    assert "make_stride(token_stride, _1{})" in CUDA_SOURCE
    assert "16, kVariableLength, true" in CUDA_SOURCE
    assert "static constexpr int kHeadDim = 128;" in cuda_source_for_head_dim(128)
    assert CUDA_SOURCE.count("q_group must have shape [B,Hkv,4|8,D]") >= 2
    assert PAGED_EXACT_SM90_FRAGMENTED_BACKEND.endswith("fragmented_exact")
    assert PAGED_EXACT_SM90_FRAGMENTED_RAGGED_BACKEND.endswith(
        "fragmented_ragged_exact"
    )
    assert PAGED_EXACT_SM90_NHD_FRAGMENTED_BACKEND.endswith("nhd_fragmented_exact")
    assert PAGED_EXACT_SM90_NHD_FRAGMENTED_RAGGED_BACKEND.endswith(
        "nhd_fragmented_ragged_exact"
    )
    assert PAGED_EXACT_SM80_GROUPED_BACKEND.endswith("sm80_grouped_exact")
    assert PAGED_EXACT_SM80_CP_ASYNC_BACKEND.endswith("sm80_cp_async_exact")
    assert PAGED_EXACT_SM100_GROUPED_BACKEND.endswith("sm100_grouped_exact")
    assert PAGED_EXACT_SM100_TGV_BACKEND.endswith("sm100_tgv_exact")
    assert PROMOTED_PAGED_EXACT_SM100_TGV_SPLITS == {
        (1, 32768): 16,
        (2, 32768): 16,
        (2, 65536): 16,
        (4, 32768): 8,
        (4, 65536): 8,
        (8, 32768): 4,
    }


def test_sm90_plan_reports_native_64_token_logical_tile():
    source = inspect.getsource(PagedExactDecodePlan.build)
    assert "tokens_per_tile=64" in source


def test_sm100_tgv_source_contract():
    from stream_attention.backends.sm100.paged_gqa_exact import _cutlass_candidates
    from stream_attention.backends.sm100.paged_gqa_exact_sources import (
        CPP_SOURCE as SM100_CPP_SOURCE,
        CUDA_SOURCE as SM100_CUDA_SOURCE,
    )

    assert "gqa_paged_separate_host" in SM100_CUDA_SOURCE
    assert "[B,2,8,128]" in SM100_CUDA_SOURCE
    assert "max_pages+64" in SM100_CUDA_SOURCE
    assert "num_splits == 16" in SM100_CUDA_SOURCE
    assert "paged_exact_decode_out" in SM100_CPP_SOURCE
    assert any(
        str(path).replace("\\", "/") == "/opt/cutlass"
        for path in _cutlass_candidates()
    )
    header = (
        Path(__file__).parents[1]
        / "stream_attention/backends/sm100/csrc/tgv_gqa_paged.cuh"
    ).read_text(encoding="utf-8")
    assert "gqa_paged_separate_host" in header
    assert "TypeQKV* device_ptr_K" in header
    assert "TypeQKV* device_ptr_V" in header
    assert "Tensor mK_nhd = make_tensor" in header
    assert "Tensor mV_nhd = make_tensor" in header
    assert "no K/V data is moved" in header


def test_sm80_cp_async_source_contract():
    assert "paged_exact_decode_out" in SM80_CPP_SOURCE
    assert "SM80_CP_ASYNC_CACHEGLOBAL" in SM80_CUDA_SOURCE
    assert "SM80_16x8x16_F32BF16BF16F32_TN" in SM80_CUDA_SOURCE
    assert "SM75_U32x4_LDSM_N" in SM80_CUDA_SOURCE
    assert "streamattn_group_max" in SM80_CUDA_SOURCE
    assert "streamattn_sm80_exact_merge_warp_kernel" in SM80_CUDA_SOURCE
    assert "num_splits <= 512" in SM80_CUDA_SOURCE


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA",
)
def test_cuda_paged_exact_matches_dense_and_uses_native_backend():
    from stream_attention.kernels.paged_exact_triton import TRITON_AVAILABLE

    if not TRITON_AVAILABLE:
        pytest.skip("requires Triton")
    query, cache = _make_paged_inputs(device="cuda")
    output, info = stream_attn.decode(query, cache, mode="exact_native")
    expected = _dense_expected(query, cache)

    torch.testing.assert_close(output, expected, atol=2e-4, rtol=2e-4)
    assert info.backend_used == PAGED_EXACT_NATIVE_BACKEND
    assert info.stats["workspace_bytes"] > 0
