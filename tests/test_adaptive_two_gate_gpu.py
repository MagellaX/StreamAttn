"""GPU mechanism checks; CPU contributor CI skips these without CUDA/Triton."""

import math

import pytest
import torch

from benchmarks.profile_adaptive_two_gate import make_case, reference
from stream_attention.certified import build_block_summaries
from stream_attention.kernels.certified_fwd_triton import (
    TRITON_AVAILABLE,
    certified_attention_triton_forward,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and TRITON_AVAILABLE), reason="CUDA + Triton required"
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["cumulative", "peaked", "post_only", "mixed_rows"])
def test_two_gate_native_mechanisms(kind, dtype):
    q, k, v, options = make_case(kind, dtype)
    summaries = build_block_summaries(k, v, block_size=32)
    output = torch.empty_like(q)
    bound = torch.empty(q.shape[:-1], device=q.device)
    stats = torch.empty(q.shape[0], q.shape[2], math.ceil(q.shape[1] / 16), 6,
                        dtype=torch.int32, device=q.device)
    common = dict(options, block_size=32, tile_size_q=16, summaries=summaries,
                  out=output, error_bound_out=bound, raw_stats_out=stats, return_raw_stats=True)
    got, raw = certified_attention_triton_forward(q, k, v, **common)
    assert got.data_ptr() == output.data_ptr() and raw.data_ptr() == stats.data_ptr()
    ref = reference(q, k, v, False)
    allowance = 0.006 if dtype == torch.bfloat16 else 0.001
    error = torch.linalg.vector_norm(got.float() - ref, dim=-1)
    assert torch.all(error <= bound + allowance)
    assert bound.max().item() <= options["error_budget"] + 2e-6
    counts = raw.sum(dim=(0, 1, 2)).tolist()
    if kind == "peaked":
        assert counts[0] > 0 and counts[4] < counts[3] and counts[5] < counts[3]
    elif kind == "post_only":
        assert counts[0] == 0 and counts[1] > 0
        assert counts[4] == counts[3] and counts[5] < counts[3]
    elif kind == "mixed_rows":
        assert counts[0] == counts[1] == 0
        assert counts[4] == counts[3] and counts[5] == counts[3]
        assert bound.max().item() == 0
    else:
        assert 0 < counts[0] + counts[1] < 127 * q.shape[2]
    expected = got.clone()
    certified_attention_triton_forward(q, k, v, **common, materialize_skipped_work=True)
    torch.testing.assert_close(got, expected, atol=0, rtol=0)


def test_two_gate_native_rejects_incompatible_buffers():
    q, k, v, options = make_case("cumulative", torch.float16)
    summaries = build_block_summaries(k, v, block_size=32)
    with pytest.raises(ValueError, match="invalid output buffer"):
        certified_attention_triton_forward(q, k, v, summaries=summaries, block_size=32,
                                           out=torch.empty_like(q, dtype=torch.float32), **options)


def test_two_gate_native_rejects_missing_value_evidence():
    q, k, v, options = make_case("cumulative", torch.float16)
    summaries = build_block_summaries(k, block_size=32)
    with pytest.raises(ValueError, match="value-bound metadata is missing"):
        certified_attention_triton_forward(q, k, v, summaries=summaries, block_size=32, **options)
