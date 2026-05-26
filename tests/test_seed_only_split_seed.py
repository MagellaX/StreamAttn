import pytest
import torch

from stream_attention.kernels.gate0_seed_only_triton import (
    TRITON_AVAILABLE,
    gate0_seed_only_attention_triton_forward_out,
    gate0_seed_only_split_seed_triton_forward_out,
    make_gate0_seed_only_split_seed_workspace,
)


def test_split_seed_workspace_shapes():
    q = torch.randn(4, 1, 14, 64)

    workspace = make_gate0_seed_only_split_seed_workspace(q, seed_chunks=6)

    assert workspace["partial_m"].shape == (4, 14, 6)
    assert workspace["partial_l"].shape == (4, 14, 6)
    assert workspace["partial_num"].shape == (4, 14, 6, 64)
    assert workspace["partial_num"].dtype is torch.float32


def test_split_seed_workspace_rejects_bad_chunks():
    q = torch.randn(4, 1, 14, 64)

    with pytest.raises(ValueError, match="seed_chunks"):
        make_gate0_seed_only_split_seed_workspace(q, seed_chunks=0)


@pytest.mark.skipif(
    not (TRITON_AVAILABLE and torch.cuda.is_available()),
    reason="split-seed Triton correctness test requires CUDA and Triton",
)
def test_split_seed_matches_direct_seed_cuda():
    torch.manual_seed(0)
    device = torch.device("cuda")
    q = torch.randn(2, 1, 14, 64, device=device, dtype=torch.float16)
    k = torch.randn(2, 512, 2, 64, device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    direct = torch.empty_like(q)
    split = torch.empty_like(q)

    gate0_seed_only_attention_triton_forward_out(
        q,
        k,
        v,
        direct,
        block_size=32,
        sink_blocks=2,
        recent_blocks=2,
        middle_seed_blocks=4,
        block_order="recent_first",
    )
    gate0_seed_only_split_seed_triton_forward_out(
        q,
        k,
        v,
        split,
        seed_tile_tokens=64,
        block_size=32,
        sink_blocks=2,
        recent_blocks=2,
        middle_seed_blocks=4,
        block_order="recent_first",
    )

    torch.testing.assert_close(split, direct, rtol=2e-3, atol=2e-3)
