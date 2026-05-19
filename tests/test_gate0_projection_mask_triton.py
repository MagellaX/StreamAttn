import math

import pytest
import torch

from benchmarks.profile_gate0_candidate_filters import (
    _candidate_metrics,
    _candidate_metrics_from_prediction,
    _project_query,
    _projection_matrix,
    _projection_metadata,
    _projection_scores,
)
from stream_attention.kernels.gate0_projection_mask_triton import (
    TRITON_AVAILABLE,
    gate0_projection_mask_triton,
    gate0_projection_mask_static_threshold_triton,
)


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
def test_gate0_projection_mask_triton_matches_score_threshold():
    torch.manual_seed(0)
    q = torch.randn(1, 1, 2, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 128, 2, 64, device="cuda", dtype=torch.float16)
    projection = _projection_matrix(
        "random",
        dim=64,
        rank=8,
        seed=0,
        device=torch.device("cuda"),
    )
    proj_min, proj_max = _projection_metadata(k, block_size=16, projection=projection)
    q_proj = _project_query(q, projection)

    scores = _projection_scores(
        q,
        projection=projection,
        proj_min=proj_min,
        proj_max=proj_max,
        selected_blocks=list(range(2, 6)),
    )
    thresholds = torch.full_like(scores, 0.25)
    has_state = torch.ones_like(scores, dtype=torch.bool)
    block_lengths = torch.full((8,), 16, device="cuda", dtype=torch.long)
    block_log_lengths = block_lengths.float().log()
    expected = scores.isfinite() & (scores + math.log(16) <= thresholds)

    actual = gate0_projection_mask_triton(
        q_proj,
        proj_min,
        proj_max,
        thresholds,
        has_state,
        block_log_lengths,
        dim=64,
        filter_margin=0.0,
        scan_start=2,
        scan_end=6,
        blocks_per_program=16,
    ).bool()

    torch.testing.assert_close(actual, expected)

    reusable = torch.zeros_like(actual, dtype=torch.uint8)
    reused_actual = gate0_projection_mask_triton(
        q_proj,
        proj_min,
        proj_max,
        thresholds,
        has_state,
        block_log_lengths,
        dim=64,
        filter_margin=0.0,
        scan_start=2,
        scan_end=6,
        blocks_per_program=16,
        output=reusable,
        clear_output=False,
    )
    assert reused_actual.data_ptr() == reusable.data_ptr()
    torch.testing.assert_close(reused_actual.bool(), expected)

    actual_skip = torch.zeros_like(expected)
    score_metrics = _candidate_metrics(
        scores=scores,
        actual_skip=actual_skip,
        has_state=has_state,
        thresholds=thresholds,
        block_lengths=block_lengths,
        filter_margin=0.0,
        sink_blocks=1,
        recent_blocks=1,
    )
    mask_metrics = _candidate_metrics_from_prediction(
        predicted=actual,
        actual_skip=actual_skip,
        sink_blocks=1,
        recent_blocks=1,
    )
    assert mask_metrics["predicted_skip_count"] == score_metrics["predicted_skip_count"]


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
def test_gate0_projection_mask_static_threshold_triton_matches_score_threshold():
    torch.manual_seed(0)
    q = torch.randn(1, 1, 2, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 128, 2, 64, device="cuda", dtype=torch.float16)
    projection = _projection_matrix(
        "random",
        dim=64,
        rank=8,
        seed=0,
        device=torch.device("cuda"),
    )
    proj_min, proj_max = _projection_metadata(k, block_size=16, projection=projection)
    q_proj = _project_query(q, projection)

    scores = _projection_scores(
        q,
        projection=projection,
        proj_min=proj_min,
        proj_max=proj_max,
        selected_blocks=list(range(2, 6)),
    )
    static_thresholds = torch.full((1, 2, 1), 0.25, device="cuda")
    block_log_lengths = torch.full((8,), 16, device="cuda", dtype=torch.float32).log()
    expected = scores.isfinite() & (scores + math.log(16) <= static_thresholds[..., None])

    actual = gate0_projection_mask_static_threshold_triton(
        q_proj,
        proj_min,
        proj_max,
        static_thresholds,
        block_log_lengths,
        dim=64,
        filter_margin=0.0,
        scan_start=2,
        scan_end=6,
        blocks_per_program=16,
    ).bool()

    torch.testing.assert_close(actual, expected)
