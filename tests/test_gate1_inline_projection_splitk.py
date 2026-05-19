import pytest
import torch

from benchmarks.profile_gate0_candidate_filters import (
    _project_query,
    _projection_matrix,
    _projection_metadata,
)
from stream_attention.gate1 import dense_attention_forward
from stream_attention.kernels.gate1_inline_projection_splitk_triton import (
    TRITON_AVAILABLE,
    _block_order_id,
    _seed_strategy_id,
    gate1_inline_projection_splitk_attention_triton_forward,
)


def test_splitk_block_order_ids():
    assert _block_order_id("sequential") == 0
    assert _block_order_id("recent_first") == 1
    assert _block_order_id("sink_recent_first") == 2
    with pytest.raises(ValueError, match="block_order"):
        _block_order_id("bad")


def test_splitk_seed_strategy_ids():
    assert _seed_strategy_id("separate") == 0
    assert _seed_strategy_id("recompute_seed") == 1
    with pytest.raises(ValueError, match="seed_strategy"):
        _seed_strategy_id("bad")


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
def test_splitk_inline_projection_matches_dense_when_skips_disabled():
    torch.manual_seed(2)
    q = torch.randn(1, 1, 2, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 256, 2, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 256, 2, 64, device="cuda", dtype=torch.float16)
    projection = _projection_matrix(
        "random",
        dim=64,
        rank=8,
        seed=2,
        device=torch.device("cuda"),
    )
    proj_min, proj_max = _projection_metadata(k, block_size=16, projection=projection, metadata_dtype=torch.float16)
    q_proj = _project_query(q, projection)

    actual, _ = gate1_inline_projection_splitk_attention_triton_forward(
        q,
        k,
        v,
        q_proj,
        proj_min,
        proj_max,
        num_chunks=4,
        error_budget=0.0,
        filter_margin=-1.0e9,
        block_size=16,
        sink_blocks=1,
        recent_blocks=1,
        middle_seed_blocks=2,
        block_order="recent_first",
    )
    expected = dense_attention_forward(q, k, v, causal=False)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
def test_splitk_recompute_seed_matches_dense_when_skips_disabled():
    torch.manual_seed(4)
    q = torch.randn(1, 1, 2, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 256, 2, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 256, 2, 64, device="cuda", dtype=torch.float16)
    projection = _projection_matrix(
        "random",
        dim=64,
        rank=8,
        seed=4,
        device=torch.device("cuda"),
    )
    proj_min, proj_max = _projection_metadata(k, block_size=16, projection=projection, metadata_dtype=torch.float16)
    q_proj = _project_query(q, projection)

    actual, _ = gate1_inline_projection_splitk_attention_triton_forward(
        q,
        k,
        v,
        q_proj,
        proj_min,
        proj_max,
        num_chunks=4,
        error_budget=0.0,
        filter_margin=-1.0e9,
        block_size=16,
        sink_blocks=1,
        recent_blocks=1,
        middle_seed_blocks=2,
        block_order="recent_first",
        seed_strategy="recompute_seed",
    )
    expected = dense_attention_forward(q, k, v, causal=False)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
def test_splitk_fused_qproj_matches_precomputed_when_skips_disabled():
    torch.manual_seed(3)
    q = torch.randn(1, 1, 2, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 256, 2, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 256, 2, 64, device="cuda", dtype=torch.float16)
    projection = _projection_matrix(
        "random",
        dim=64,
        rank=8,
        seed=3,
        device=torch.device("cuda"),
    )
    proj_min, proj_max = _projection_metadata(k, block_size=16, projection=projection, metadata_dtype=torch.float16)
    q_proj = _project_query(q, projection)

    precomputed, _ = gate1_inline_projection_splitk_attention_triton_forward(
        q,
        k,
        v,
        q_proj,
        proj_min,
        proj_max,
        num_chunks=4,
        error_budget=0.0,
        filter_margin=-1.0e9,
        block_size=16,
        sink_blocks=1,
        recent_blocks=1,
        middle_seed_blocks=2,
        block_order="recent_first",
    )
    fused, _ = gate1_inline_projection_splitk_attention_triton_forward(
        q,
        k,
        v,
        None,
        proj_min,
        proj_max,
        projection=projection,
        compute_qproj=True,
        num_chunks=4,
        error_budget=0.0,
        filter_margin=-1.0e9,
        block_size=16,
        sink_blocks=1,
        recent_blocks=1,
        middle_seed_blocks=2,
        block_order="recent_first",
    )

    torch.testing.assert_close(fused, precomputed, rtol=2e-2, atol=2e-2)
