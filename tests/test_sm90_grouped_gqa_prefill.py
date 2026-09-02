from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from stream_attention.backends.sm90.grouped_gqa_prefill import (
    GroupedRSPrefillPlan,
    GroupedWgmmaPrefillPlan,
    decode_grouped_prefill_resources,
    decode_grouped_rs_prefill_resources,
    supports_grouped_rs_prefill,
    supports_grouped_wgmma_prefill,
)


def test_grouped_prefill_source_uses_natural_m64n64_wgmma():
    source = (
        Path(__file__).parents[1]
        / "stream_attention/backends/sm90/transposed_gqa_exact_sources.py"
    ).read_text(encoding="utf-8")
    assert "SM90_64x64x16_F32BF16BF16_SS" in source
    assert "kPrefillConsumerGroups = 2" in source
    assert "__launch_bounds__(256, 1)" in source
    assert "streamattn_grouped_wgmma_prefill_kernel" in source
    assert "key_position > query_position" in source
    assert "GroupedPrefillSharedStorage" in source
    assert "int(query.shape[3]) != 128" in inspect.getsource(
        supports_grouped_wgmma_prefill
    )


def test_grouped_prefill_resource_decoder_names_all_fields():
    assert decode_grouped_prefill_resources(torch.arange(5)) == {
        "registers_per_thread": 0,
        "static_shared_bytes": 1,
        "dynamic_shared_bytes": 2,
        "blocks_per_sm": 3,
        "max_threads_per_block": 4,
    }
    with pytest.raises(ValueError, match="expected 5"):
        decode_grouped_prefill_resources(torch.arange(4))


def test_grouped_rs_prefill_source_keeps_probability_in_registers():
    source = (
        Path(__file__).parents[1]
        / "stream_attention/backends/sm90/transposed_gqa_exact_sources.py"
    ).read_text(encoding="utf-8")
    assert "streamattn_grouped_rs_prefill_kernel" in source
    assert "PrefillRSTiledMmaPV" in source
    assert "GMMA::rs_op_selector" in source
    assert "streamattn_convert_layout_acc_aregs<PrefillRSTiledMmaPV>" in source
    assert "streamattn_quad_sum" in source
    assert "grouped_rs_prefill_out" in source


def test_grouped_rs_resource_decoder_includes_local_memory():
    assert decode_grouped_rs_prefill_resources(torch.arange(6)) == {
        "registers_per_thread": 0,
        "static_shared_bytes": 1,
        "dynamic_shared_bytes": 2,
        "blocks_per_sm": 3,
        "max_threads_per_block": 4,
        "local_bytes_per_thread": 5,
    }
    with pytest.raises(ValueError, match="expected 6"):
        decode_grouped_rs_prefill_resources(torch.arange(5))


def test_grouped_prefill_rejects_cpu_tensors():
    q = torch.empty(1, 64, 16, 128, dtype=torch.bfloat16)
    k = torch.empty(1, 64, 2, 128, dtype=torch.bfloat16)
    assert not supports_grouped_wgmma_prefill(q, k, k)
    assert not supports_grouped_rs_prefill(q, k, k)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (9, 0),
    reason="SM90 CUDA device required",
)
@pytest.mark.parametrize(("group_size", "sequence_length"), ((4, 128), (8, 128)))
def test_grouped_wgmma_prefill_matches_flash_sdpa(
    group_size: int,
    sequence_length: int,
):
    torch.manual_seed(91 + group_size)
    q_heads = 16
    kv_heads = q_heads // group_size
    query = torch.randn(
        1,
        sequence_length,
        q_heads,
        128,
        device="cuda",
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        1,
        sequence_length,
        kv_heads,
        128,
        device="cuda",
        dtype=torch.bfloat16,
    )
    value = torch.randn_like(key)
    plan = GroupedWgmmaPrefillPlan.build(query, key, value)
    actual = plan.run()
    expected = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        is_causal=True,
        dropout_p=0.0,
        enable_gqa=True,
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected, rtol=4e-2, atol=4e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (9, 0),
    reason="SM90 CUDA device required",
)
@pytest.mark.parametrize("group_size", (4, 8))
def test_grouped_rs_prefill_matches_flash_sdpa(group_size: int):
    torch.manual_seed(191 + group_size)
    sequence_length = 128
    q_heads = 16
    kv_heads = q_heads // group_size
    query = torch.randn(
        1,
        sequence_length,
        q_heads,
        128,
        device="cuda",
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        1,
        sequence_length,
        kv_heads,
        128,
        device="cuda",
        dtype=torch.bfloat16,
    )
    value = torch.randn_like(key)
    plan = GroupedRSPrefillPlan.build(query, key, value)
    actual = plan.run()
    expected = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        is_causal=True,
        dropout_p=0.0,
        enable_gqa=True,
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected, rtol=4e-2, atol=4e-2)
