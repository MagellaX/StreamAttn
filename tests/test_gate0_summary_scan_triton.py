import math

import pytest
import torch

from stream_attention.certified.summaries import build_block_summaries
from stream_attention.kernels.gate0_summary_scan_triton import (
    TRITON_AVAILABLE,
    gate0_summary_scan_triton,
)


def _torch_summary_scan(q, summaries, *, scale):
    q_bhsd = q.permute(0, 2, 1, 3).contiguous().float()
    dot_centroid = torch.einsum("bhsd,bhkd->bhsk", q_bhsd, summaries.centroid)
    q_norm = torch.linalg.vector_norm(q_bhsd, dim=-1)
    residual = (dot_centroid + q_norm[..., None] * summaries.radius[:, :, None, :]) * scale
    if summaries.outlier_keys is None or summaries.outlier_mask is None:
        return residual

    outlier_scores = torch.einsum("bhsd,bhkod->bhsko", q_bhsd, summaries.outlier_keys) * scale
    outlier_scores = outlier_scores.masked_fill(
        ~summaries.outlier_mask[:, :, None, :, :],
        -float("inf"),
    )
    return torch.maximum(residual, outlier_scores.amax(dim=-1))


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
def test_gate0_summary_scan_triton_matches_torch_outlier2():
    torch.manual_seed(0)
    q = torch.randn(1, 1, 3, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 128, 3, 64, device="cuda", dtype=torch.float16)
    summaries = build_block_summaries(k, block_size=16, num_outliers=2)
    scale = 1.0 / math.sqrt(q.shape[-1])

    expected = _torch_summary_scan(q, summaries, scale=scale)
    actual = gate0_summary_scan_triton(
        q,
        summaries.centroid,
        summaries.radius,
        outlier_keys=summaries.outlier_keys,
        outlier_mask=summaries.outlier_mask,
        scale=scale,
        blocks_per_program=16,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
