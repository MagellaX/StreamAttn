import pytest
import torch

from benchmarks.profile_gate0_candidate_filters import (
    _project_query,
    _projection_matrix,
    _projection_metadata,
    _projection_scores,
)
from stream_attention.kernels.gate0_projection_scan_triton import (
    TRITON_AVAILABLE,
    gate0_projection_scan_triton,
)


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
def test_gate0_projection_scan_triton_matches_torch_reference():
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

    expected = _projection_scores(
        q,
        projection=projection,
        proj_min=proj_min,
        proj_max=proj_max,
        selected_blocks=list(range(2, 6)),
    )
    actual = gate0_projection_scan_triton(
        q_proj,
        proj_min,
        proj_max,
        dim=64,
        scan_start=2,
        scan_end=6,
        blocks_per_program=16,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    reusable = torch.full_like(expected, float("inf"))
    reused_actual = gate0_projection_scan_triton(
        q_proj,
        proj_min,
        proj_max,
        dim=64,
        scan_start=2,
        scan_end=6,
        blocks_per_program=16,
        output=reusable,
        clear_output=False,
    )
    assert reused_actual.data_ptr() == reusable.data_ptr()
    torch.testing.assert_close(reused_actual, expected, rtol=1e-3, atol=1e-3)
