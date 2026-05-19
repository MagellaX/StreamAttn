import pytest
import torch

from benchmarks.profile_gate0_candidate_filters import (
    _project_query,
    _projection_matrix,
    _projection_metadata,
)
from benchmarks.profile_gate1_inline_projection import _summarize_inline_stats
from benchmarks.profile_gate1_inline_projection import _summarize_inline_stats_per_head
from stream_attention.gate1 import dense_attention_forward
from stream_attention.kernels.gate1_inline_projection_fwd_triton import (
    TRITON_AVAILABLE,
    gate1_inline_projection_attention_triton_forward,
)


def test_inline_projection_stats_summary():
    raw = torch.zeros(1, 2, 8, dtype=torch.int32)
    raw[0, 0, 0] = 3
    raw[0, 0, 3] = 5
    raw[0, 0, 4] = 8
    raw[0, 0, 6] = 6
    raw[0, 1, 0] = 1
    raw[0, 1, 3] = 7
    raw[0, 1, 4] = 8
    raw[0, 1, 6] = 6

    stats = _summarize_inline_stats(raw)

    assert stats["projection_skipped_blocks"] == 4
    assert stats["pv_executed_blocks"] == 12
    assert stats["total_blocks"] == 16
    assert stats["middle_blocks"] == 12
    assert stats["projection_skip_fraction"] == pytest.approx(4 / 12)
    assert stats["pv_executed_fraction"] == pytest.approx(12 / 16)

    per_head = _summarize_inline_stats_per_head(raw)
    assert per_head["projection_skip_fraction_mean"] == pytest.approx(((3 / 6) + (1 / 6)) / 2)
    assert per_head["pv_executed_fraction_mean"] == pytest.approx(((5 / 8) + (7 / 8)) / 2)
    assert len(per_head["per_head"]) == 2


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
def test_inline_projection_matches_dense_when_skips_disabled():
    torch.manual_seed(0)
    q = torch.randn(1, 1, 2, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 128, 2, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 128, 2, 64, device="cuda", dtype=torch.float16)
    projection = _projection_matrix(
        "random",
        dim=64,
        rank=8,
        seed=0,
        device=torch.device("cuda"),
    )
    proj_min, proj_max = _projection_metadata(k, block_size=16, projection=projection, metadata_dtype=torch.float16)
    q_proj = _project_query(q, projection)

    actual, raw_stats = gate1_inline_projection_attention_triton_forward(
        q,
        k,
        v,
        q_proj,
        proj_min,
        proj_max,
        error_budget=0.0,
        filter_margin=-1.0e9,
        block_size=16,
        sink_blocks=1,
        recent_blocks=1,
        middle_seed_blocks=0,
        block_order="sink_recent_first",
        return_raw_stats=True,
    )
    expected = dense_attention_forward(q, k, v, causal=False)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    stats = _summarize_inline_stats(raw_stats)
    assert stats["projection_skipped_blocks"] == 0
    assert stats["gate1_post_qk_skipped_blocks"] == 0
