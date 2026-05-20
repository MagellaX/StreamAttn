import torch
import pytest

from stream_attention.gate0_fused_hybrid import (
    Gate0FusedHybridPolicy,
    build_gate0_projection_matrix,
    build_gate0_projection_metadata,
    stream_attn_gate0_fused_hybrid,
    summarize_gate0_fused_hybrid_raw_stats,
)
from stream_attention.gate1 import dense_attention_forward
from stream_attention.kernels.gate1_inline_projection_splitk_triton import TRITON_AVAILABLE


def _entry():
    return {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "layer_id": 8,
        "kv_len_bucket": 32768,
        "safety_budget": {"name": "moderate"},
        "runtime": {
            "head_modes": [1, 0, 1],
            "trusted_sparse_heads": [1],
            "exact_heads": [0, 2],
            "block_size": 4,
            "sink_blocks": 1,
            "recent_blocks": 1,
            "middle_seed_blocks": 1,
            "chunk_anchor_blocks": 0,
            "block_order": "recent_first",
            "num_chunks": 2,
            "seed_strategy": "recompute_seed",
            "filter_margin": 16.0,
            "error_budget": 0.01,
            "projection_kind": "random",
            "projection_dim": 2,
            "projection_seed": 3,
            "projection_metadata_dtype": "fp16",
            "splitk_workspace": "reuse",
        },
    }


def test_policy_loads_from_fused_hybrid_entry():
    policy = Gate0FusedHybridPolicy.from_entry(_entry())

    assert policy.head_modes == (1, 0, 1)
    assert policy.trusted_sparse_heads == (1,)
    assert policy.exact_heads == (0, 2)
    assert policy.block_size == 4
    assert policy.projection_dim == 2
    assert policy.safety_budget_name == "moderate"


def test_projection_matrix_is_deterministic():
    a = build_gate0_projection_matrix(kind="random", dim=8, rank=3, seed=5, device="cpu")
    b = build_gate0_projection_matrix(kind="random", dim=8, rank=3, seed=5, device="cpu")
    c = build_gate0_projection_matrix(kind="random", dim=8, rank=3, seed=6, device="cpu")

    torch.testing.assert_close(a, b)
    assert not torch.allclose(a, c)
    torch.testing.assert_close(a.norm(dim=-1), torch.ones(3), rtol=1e-6, atol=1e-6)


def test_projection_metadata_shapes_and_values():
    policy = Gate0FusedHybridPolicy.from_entry(_entry())
    k = torch.arange(1 * 8 * 3 * 4, dtype=torch.float32).reshape(1, 8, 3, 4)
    projection = torch.eye(2, 4)

    metadata = build_gate0_projection_metadata(k, policy, projection=projection)

    assert metadata.proj_min.shape == (1, 3, 2, 2)
    assert metadata.proj_max.shape == (1, 3, 2, 2)
    assert metadata.proj_min.dtype is torch.float16
    torch.testing.assert_close(metadata.proj_min[0, 0, 0].float(), torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(metadata.proj_max[0, 0, 0].float(), torch.tensor([36.0, 37.0]))


def test_cpu_dense_fallback_matches_dense_attention():
    policy = Gate0FusedHybridPolicy.from_entry(_entry())
    q = torch.randn(1, 1, 3, 4)
    k = torch.randn(1, 8, 3, 4)
    v = torch.randn(1, 8, 3, 4)

    actual, info = stream_attn_gate0_fused_hybrid(
        q,
        k,
        v,
        policy=policy,
        fallback="dense",
        return_info=True,
    )
    expected = dense_attention_forward(q, k, v, causal=False)

    torch.testing.assert_close(actual, expected)
    assert info.stats is None
    assert info.policy is policy


def test_raw_stats_summary_uses_splitk_counter_layout():
    raw = torch.zeros(1, 2, 3, 8, dtype=torch.int32)
    raw[..., 0] = 2
    raw[..., 1] = 3
    raw[..., 2] = 4
    raw[..., 3] = 5
    raw[..., 4] = 10
    raw[..., 5] = 1
    raw[..., 6] = 3
    raw[..., 7] = 2

    stats = summarize_gate0_fused_hybrid_raw_stats(raw)

    assert stats.projection_skipped_blocks == 12
    assert stats.projection_computed_blocks == 18
    assert stats.gate1_post_qk_skipped_blocks == 24
    assert stats.pv_executed_blocks == 30
    assert stats.middle_blocks == 60
    assert stats.projection_skip_fraction == 0.2
    assert stats.pv_executed_fraction == 0.5


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
def test_cuda_runtime_wrapper_dispatches_fused_kernel_with_telemetry():
    policy = Gate0FusedHybridPolicy.from_entry(_entry())
    q = torch.randn(1, 1, 3, 4, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 64, 3, 4, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 64, 3, 4, device="cuda", dtype=torch.float16)

    out, info = stream_attn_gate0_fused_hybrid(
        q,
        k,
        v,
        policy=policy,
        return_info=True,
    )

    assert out.shape == q.shape
    assert info.stats is not None
    assert info.per_head_stats is not None
    assert len(info.per_head_stats) == 3
