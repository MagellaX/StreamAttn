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
    make_splitk_workspace,
)
from benchmarks.profile_gate1_inline_projection_splitk import _parse_head_indices, _per_head_error


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


def test_splitk_profile_parse_head_indices():
    assert _parse_head_indices("", heads=4) == []
    assert _parse_head_indices("3,1,3", heads=4) == [1, 3]
    assert _parse_head_indices("-1", heads=4) == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="outside"):
        _parse_head_indices("4", heads=4)


def test_splitk_profile_reports_per_head_error():
    expected = torch.zeros((1, 1, 3, 2), dtype=torch.float32)
    actual = expected.clone()
    actual[:, :, 1, 0] = 0.5
    actual[:, :, 2, :] = 0.25

    payload = _per_head_error(actual, expected)

    assert payload["worst_head"] == 1
    assert payload["per_head"][0]["max_abs_error"] == 0.0
    assert payload["per_head"][1]["max_abs_error"] == 0.5
    assert payload["per_head"][2]["mean_abs_error"] == 0.25


def test_make_splitk_workspace_shapes_cpu():
    q = torch.empty(1, 1, 3, 64)
    workspace = make_splitk_workspace(q, rank=8, num_chunks=4, seed_strategy="recompute_seed")

    assert workspace["output"].shape == q.shape
    assert workspace["chunk_max"].shape == (1, 3, 5)
    assert workspace["chunk_num"].shape == (1, 3, 5, 64)
    assert workspace["raw_stats"].shape == (1, 3, 4, 8)


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
def test_splitk_workspace_matches_allocating_path_when_skips_disabled():
    torch.manual_seed(6)
    q = torch.randn(1, 1, 2, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 256, 2, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 256, 2, 64, device="cuda", dtype=torch.float16)
    projection = _projection_matrix(
        "random",
        dim=64,
        rank=8,
        seed=6,
        device=torch.device("cuda"),
    )
    proj_min, proj_max = _projection_metadata(k, block_size=16, projection=projection, metadata_dtype=torch.float16)
    q_proj = _project_query(q, projection)
    workspace = make_splitk_workspace(q, rank=8, num_chunks=4, seed_strategy="recompute_seed")

    allocating, _ = gate1_inline_projection_splitk_attention_triton_forward(
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
    workspace_out, _ = gate1_inline_projection_splitk_attention_triton_forward(
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
        workspace=workspace,
    )

    torch.testing.assert_close(workspace_out, allocating, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
def test_splitk_recompute_seed_with_chunk_anchors_matches_dense_when_skips_disabled():
    torch.manual_seed(5)
    q = torch.randn(1, 1, 2, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 256, 2, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 256, 2, 64, device="cuda", dtype=torch.float16)
    projection = _projection_matrix(
        "random",
        dim=64,
        rank=8,
        seed=5,
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
        chunk_anchor_blocks=2,
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
