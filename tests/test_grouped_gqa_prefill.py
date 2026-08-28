from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from stream_attention.kernels.grouped_gqa_prefill_triton import (
    TRITON_AVAILABLE,
    effective_kv_reuse,
    grouped_gqa_prefill,
)


def _reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    group_size = q.shape[2] // k.shape[2]
    kh = k.transpose(1, 2).repeat_interleave(group_size, dim=1)
    vh = v.transpose(1, 2).repeat_interleave(group_size, dim=1)
    return F.scaled_dot_product_attention(
        q.transpose(1, 2),
        kh,
        vh,
        is_causal=True,
        dropout_p=0.0,
    ).transpose(1, 2)


def _reference_lse(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    group_size = q.shape[2] // k.shape[2]
    qh = q.transpose(1, 2).float()
    kh = k.transpose(1, 2).repeat_interleave(group_size, dim=1).float()
    scores = torch.matmul(qh, kh.transpose(-1, -2)) / q.shape[-1] ** 0.5
    rows = torch.arange(q.shape[1], device=q.device)[:, None]
    columns = torch.arange(k.shape[1], device=q.device)[None, :]
    scores = scores.masked_fill(columns > rows, -torch.inf)
    return torch.logsumexp(scores, dim=-1)


def test_effective_kv_reuse_exposes_non_reusing_schedules():
    assert effective_kv_reuse(
        heads_per_program=2,
        tile_m=32,
        reference_tile_m=64,
    ) == 1.0
    assert effective_kv_reuse(
        heads_per_program=4,
        tile_m=32,
        reference_tile_m=64,
    ) == 2.0
    with pytest.raises(ValueError, match="positive"):
        effective_kv_reuse(
            heads_per_program=4,
            tile_m=0,
            reference_tile_m=64,
        )


def test_grouped_prefill_source_truncates_causal_stream():
    from pathlib import Path

    source = (
        Path(__file__).parents[1]
        / "stream_attention/kernels/grouped_gqa_prefill_triton.py"
    ).read_text(encoding="utf-8")
    assert "end_n = tl.minimum(N, (pid_m + 1) * TILE_M + q_start)" in source
    assert "for start_n in range(0, end_n, TILE_N)" in source


@pytest.mark.skipif(
    not torch.cuda.is_available() or not TRITON_AVAILABLE,
    reason="CUDA and Triton are required for grouped GQA prefill",
)
@pytest.mark.parametrize(("heads_per_program", "tile_m"), ((2, 32), (4, 16)))
@pytest.mark.parametrize("head_dim", (64, 128))
@pytest.mark.parametrize(("dtype", "atol"), ((torch.float32, 1e-2), (torch.bfloat16, 3e-2)))
def test_grouped_gqa_prefill_matches_exact_sdpa(
    heads_per_program: int,
    tile_m: int,
    head_dim: int,
    dtype: torch.dtype,
    atol: float,
):
    torch.manual_seed(31)
    q = torch.randn(1, 128, 8, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(1, 128, 2, head_dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)

    output, lse = grouped_gqa_prefill(
        q,
        k,
        v,
        heads_per_program=heads_per_program,
        tile_m=tile_m,
        return_lse=True,
    )
    expected = _reference(q, k, v)
    torch.testing.assert_close(output, expected, rtol=atol, atol=atol)
    torch.testing.assert_close(
        lse,
        _reference_lse(q, k),
        rtol=max(atol, 5e-2),
        atol=max(atol, 5e-2),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not TRITON_AVAILABLE,
    reason="CUDA and Triton are required for grouped GQA prefill",
)
def test_grouped_gqa_prefill_reuses_output_buffers():
    torch.manual_seed(37)
    q = torch.randn(1, 64, 8, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 64, 2, 64, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    output = torch.empty_like(q)
    lse = torch.empty((1, 8, 64), device=q.device, dtype=torch.float32)
    returned_output, returned_lse = grouped_gqa_prefill(
        q,
        k,
        v,
        heads_per_program=2,
        tile_m=32,
        return_lse=True,
        output=output,
        lse=lse,
    )
    assert returned_output.data_ptr() == output.data_ptr()
    assert returned_lse.data_ptr() == lse.data_ptr()
