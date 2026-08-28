from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import stream_attention as stream_attn
from stream_attention.backends.sm100.gqa_prefill import (
    PROMOTED_SM100_GQA_PREFILL_CELLS,
    Sm100GqaPrefillPlan,
    TILE_VARIANTS,
    is_promoted_sm100_gqa_prefill,
    supports_sm100_gqa_prefill,
)


def test_sm100_gqa_prefill_source_keeps_row_specific_causal_mask():
    root = Path(__file__).resolve().parents[1]
    header = (root / "stream_attention/backends/sm100/csrc/tgv_gqa.cuh").read_text(
        encoding="utf-8"
    )
    binding = (
        root / "stream_attention/backends/sm100/gqa_prefill_sources.py"
    ).read_text(encoding="utf-8")

    assert "bool Causal = false" in header
    assert "q_position = work_tile_info.qL_idx * CTA_qL + q_local" in header
    assert "kv_seq_len > q_position" in header
    assert "causal_query_end = (qL_idx + 1) * CTA_qL" in header
    assert "workload_seq_len = cute::min(workload_seq_len, causal_query_end)" in header
    assert "TileQueryHeads, TileQueryLength" in binding
    assert "1, 1," in binding
    assert "true>(" in binding
    assert set(TILE_VARIANTS) == {"h8_q1", "h8_q2", "h8_q4"}
    assert PROMOTED_SM100_GQA_PREFILL_CELLS == frozenset(
        {(1, 64), (1, 128), (1, 256), (1, 384), (2, 64)}
    )


def test_sm100_gqa_prefill_rejects_cpu_tensors():
    q = torch.empty(1, 128, 16, 128, dtype=torch.bfloat16)
    k = torch.empty(1, 128, 2, 128, dtype=torch.bfloat16)
    assert not supports_sm100_gqa_prefill(q, k, k)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 0),
    reason="B200-class SM100a GPU required",
)
@pytest.mark.parametrize("tile", tuple(TILE_VARIANTS))
def test_sm100_gqa_prefill_matches_causal_sdpa(tile: str):
    torch.manual_seed(43)
    q = torch.randn(1, 128, 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 128, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    plan = Sm100GqaPrefillPlan.build(q, k, v, tile=tile)
    output = plan.run()
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        is_causal=True,
        dropout_p=0.0,
        enable_gqa=True,
    ).transpose(1, 2)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 0),
    reason="B200-class SM100a GPU required",
)
@pytest.mark.parametrize(
    ("batch", "seq_len"), sorted(PROMOTED_SM100_GQA_PREFILL_CELLS)
)
def test_promoted_sm100_gqa_prefill_cells_match_causal_sdpa(batch: int, seq_len: int):
    torch.manual_seed(47 + batch + seq_len)
    q = torch.randn(batch, seq_len, 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, seq_len, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    assert is_promoted_sm100_gqa_prefill(q, k, v)
    plan = Sm100GqaPrefillPlan.build(q, k, v, tile="h8_q2")
    output = plan.run()
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        is_causal=True,
        dropout_p=0.0,
        enable_gqa=True,
    ).transpose(1, 2)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 0),
    reason="B200-class SM100a GPU required",
)
def test_public_prefill_routes_promoted_sm100_cell():
    torch.manual_seed(53)
    q = torch.randn(1, 64, 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 64, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    with torch.no_grad():
        output, info = stream_attn.prefill(q, k, v, return_info=True)
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        is_causal=True,
        dropout_p=0.0,
        enable_gqa=True,
    ).transpose(1, 2)
    torch.testing.assert_close(output, expected, rtol=3e-2, atol=3e-2)
    assert info.backend_used == "sm100_tgv_gqa_causal_prefill"
    assert info.backend_plan.reason == "promoted_exact_b200_prefill_cell"
