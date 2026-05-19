import math

import pytest
import torch

from benchmarks.profile_gate0_candidate_filters import (
    _project_query,
    _projection_matrix,
    _projection_metadata,
    _projection_scores,
    _unpack_bitmask,
)
from stream_attention.kernels.gate0_projection_bitmask_triton import (
    TRITON_AVAILABLE,
    gate0_projection_bitmask_triton,
    gate0_projection_bitmask_static_threshold_triton,
)


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
def test_gate0_projection_bitmask_triton_matches_score_threshold():
    torch.manual_seed(0)
    q = torch.randn(1, 1, 2, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 160, 2, 64, device="cuda", dtype=torch.float16)
    projection = _projection_matrix(
        "random",
        dim=64,
        rank=8,
        seed=0,
        device=torch.device("cuda"),
    )
    proj_min, proj_max = _projection_metadata(k, block_size=16, projection=projection)
    q_proj = _project_query(q, projection)

    selected_blocks = list(range(1, 9))
    scores = _projection_scores(
        q,
        projection=projection,
        proj_min=proj_min,
        proj_max=proj_max,
        selected_blocks=selected_blocks,
    )
    thresholds = torch.full_like(scores, 0.25)
    has_state = torch.ones_like(scores, dtype=torch.bool)
    block_lengths = torch.full((10,), 16, device="cuda", dtype=torch.long)
    block_log_lengths = block_lengths.float().log()
    expected = scores.isfinite() & (scores + math.log(16) <= thresholds)

    packed = gate0_projection_bitmask_triton(
        q_proj,
        proj_min,
        proj_max,
        thresholds,
        has_state,
        block_log_lengths,
        dim=64,
        filter_margin=0.0,
        scan_start=1,
        scan_end=9,
    )
    actual = _unpack_bitmask(packed, num_blocks=10)

    torch.testing.assert_close(actual, expected)

    reusable = torch.zeros_like(packed)
    reused = gate0_projection_bitmask_triton(
        q_proj,
        proj_min,
        proj_max,
        thresholds,
        has_state,
        block_log_lengths,
        dim=64,
        filter_margin=0.0,
        scan_start=1,
        scan_end=9,
        words_per_program=2,
        output=reusable,
        clear_output=False,
    )
    assert reused.data_ptr() == reusable.data_ptr()
    torch.testing.assert_close(_unpack_bitmask(reused, num_blocks=10), expected)


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
def test_gate0_projection_bitmask_multiword_and_static_threshold_match_scores():
    torch.manual_seed(1)
    q = torch.randn(1, 1, 2, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2048, 2, 64, device="cuda", dtype=torch.float16)
    projection = _projection_matrix(
        "random",
        dim=64,
        rank=8,
        seed=1,
        device=torch.device("cuda"),
    )
    proj_min, proj_max = _projection_metadata(k, block_size=16, projection=projection)
    q_proj = _project_query(q, projection)

    selected_blocks = list(range(33, 97))
    scores = _projection_scores(
        q,
        projection=projection,
        proj_min=proj_min,
        proj_max=proj_max,
        selected_blocks=selected_blocks,
    )
    thresholds = torch.full_like(scores, 0.25)
    has_state = torch.ones_like(scores, dtype=torch.bool)
    block_lengths = torch.full((128,), 16, device="cuda", dtype=torch.long)
    block_log_lengths = block_lengths.float().log()
    expected = scores.isfinite() & (scores + math.log(16) <= thresholds)

    packed = gate0_projection_bitmask_triton(
        q_proj,
        proj_min,
        proj_max,
        thresholds,
        has_state,
        block_log_lengths,
        dim=64,
        filter_margin=0.0,
        scan_start=33,
        scan_end=97,
        words_per_program=2,
    )
    torch.testing.assert_close(_unpack_bitmask(packed, num_blocks=128), expected)

    static_thresholds = torch.full((1, 2, 1), 0.25, device="cuda")
    static_packed = gate0_projection_bitmask_static_threshold_triton(
        q_proj,
        proj_min,
        proj_max,
        static_thresholds,
        block_log_lengths,
        dim=64,
        filter_margin=0.0,
        scan_start=33,
        scan_end=97,
        words_per_program=2,
    )
    torch.testing.assert_close(_unpack_bitmask(static_packed, num_blocks=128), expected)
