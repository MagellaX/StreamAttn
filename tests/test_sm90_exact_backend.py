import pytest
import torch

from stream_attention.backends.sm90.transposed_gqa_exact import (
    ExactDecodePlan,
    PROMOTED_EXACT_D128_G4_SPLITS,
    PROMOTED_EXACT_G4_SPLITS,
    PROMOTED_EXACT_SPLITS,
    _shape_reasons,
    choose_num_splits,
    resolve_cutlass_root,
    supports_transposed_gqa_exact,
)
from stream_attention.backends.sm90.transposed_gqa_exact_sources import (
    CPP_SOURCE,
    CUDA_SOURCE,
    cuda_source_for_head_dim,
)


def test_split_rule_preserves_256_producer_cta_target():
    assert choose_num_splits(batch=2, kv_heads=2, kv_len=32768) == 64
    assert choose_num_splits(batch=4, kv_heads=2, kv_len=32768) == 32
    assert choose_num_splits(batch=8, kv_heads=2, kv_len=32768) == 16


def test_promoted_shape_contract_requires_head_major_bf16_buffers():
    assert PROMOTED_EXACT_SPLITS == {
        (2, 16384): 64,
        (4, 16384): 64,
        (4, 32768): 64,
        (4, 65536): 64,
        (8, 16384): 32,
        (8, 32768): 32,
        (8, 65536): 32,
    }
    q = torch.empty(4, 1, 16, 64, dtype=torch.bfloat16, device="meta")
    k_head_major = torch.empty(
        4, 2, 32768, 64, dtype=torch.bfloat16, device="meta"
    )
    v_head_major = torch.empty_like(k_head_major)
    assert _shape_reasons(q, k_head_major, v_head_major, promoted_only=True) == []

    k_token_major = torch.empty(
        4, 32768, 2, 64, dtype=torch.bfloat16, device="meta"
    )
    reasons = _shape_reasons(q, k_token_major, k_token_major, promoted_only=True)
    assert "gqa" in reasons
    assert "unpromoted_shape" in reasons
    assert not supports_transposed_gqa_exact(q, k_head_major, v_head_major)

    q_b2 = torch.empty(2, 1, 16, 64, dtype=torch.bfloat16, device="meta")
    k_b2_16k = torch.empty(
        2, 2, 16384, 64, dtype=torch.bfloat16, device="meta"
    )
    assert _shape_reasons(
        q_b2, k_b2_16k, k_b2_16k, promoted_only=True
    ) == []
    k_b2_64k = torch.empty(
        2, 2, 65536, 64, dtype=torch.bfloat16, device="meta"
    )
    assert "unpromoted_shape" in _shape_reasons(
        q_b2, k_b2_64k, k_b2_64k, promoted_only=True
    )


def test_group_size_four_uses_a_discrete_promoted_region():
    assert PROMOTED_EXACT_G4_SPLITS == {
        (1, 16384): 64,
        (1, 32768): 128,
        (2, 16384): 64,
        (2, 32768): 64,
        (2, 65536): 64,
        (4, 16384): 32,
        (4, 32768): 32,
        (4, 65536): 32,
        (8, 16384): 16,
        (8, 32768): 16,
        (8, 65536): 16,
        (16, 16384): 8,
        (16, 32768): 8,
        (16, 65536): 8,
    }
    q = torch.empty(1, 1, 16, 64, dtype=torch.bfloat16, device="meta")
    k = torch.empty(1, 4, 65536, 64, dtype=torch.bfloat16, device="meta")
    v = torch.empty_like(k)

    assert _shape_reasons(q, k, v, promoted_only=False) == []
    promoted_reasons = _shape_reasons(q, k, v, promoted_only=True)
    assert "unpromoted_shape" in promoted_reasons

    q_green = torch.empty(4, 1, 16, 64, dtype=torch.bfloat16, device="meta")
    k_green = torch.empty(4, 4, 32768, 64, dtype=torch.bfloat16, device="meta")
    assert _shape_reasons(q_green, k_green, k_green, promoted_only=True) == []


def test_exact_source_pads_g4_inside_the_wgmma_producer():
    assert "head < active_heads" in CUDA_SOURCE
    assert "groups * active_heads" in CUDA_SOURCE
    assert "[B,Hkv,4|8,64]" in CUDA_SOURCE


def test_d128_source_and_shape_use_a_discrete_promoted_region():
    d128_source = cuda_source_for_head_dim(128)
    assert "static constexpr int kHeadDim = 128;" in d128_source
    assert "dim0 += 64" in d128_source
    assert "tSrKRead" in d128_source
    assert cuda_source_for_head_dim(64) is CUDA_SOURCE

    assert PROMOTED_EXACT_D128_G4_SPLITS == {
        (4, 32768): 8,
        (4, 65536): 8,
        (8, 16384): 4,
        (8, 65536): 4,
        (16, 32768): 2,
        (16, 65536): 2,
    }

    q = torch.empty(4, 1, 32, 128, dtype=torch.bfloat16, device="meta")
    k = torch.empty(4, 8, 32768, 128, dtype=torch.bfloat16, device="meta")
    assert _shape_reasons(q, k, k, promoted_only=False) == []
    assert _shape_reasons(q, k, k, promoted_only=True) == []

    q_red = torch.empty(4, 1, 32, 128, dtype=torch.bfloat16, device="meta")
    k_red = torch.empty(4, 8, 16384, 128, dtype=torch.bfloat16, device="meta")
    assert "unpromoted_shape" in _shape_reasons(
        q_red, k_red, k_red, promoted_only=True
    )

    q_g8 = torch.empty(4, 1, 32, 128, dtype=torch.bfloat16, device="meta")
    k_g8 = torch.empty(4, 4, 32768, 128, dtype=torch.bfloat16, device="meta")
    assert "unpromoted_shape" in _shape_reasons(
        q_g8, k_g8, k_g8, promoted_only=True
    )

    q_fragile = torch.empty(8, 1, 32, 128, dtype=torch.bfloat16, device="meta")
    k_fragile = torch.empty(8, 8, 32768, 128, dtype=torch.bfloat16, device="meta")
    assert "unpromoted_shape" in _shape_reasons(
        q_fragile, k_fragile, k_fragile, promoted_only=True
    )


def test_combined_exact_dispatch_is_bound_and_plan_keeps_two_call_control():
    assert 'm.def("exact_decode_out"' in CPP_SOURCE
    assert 'm.def("exact_merge_warp_out"' in CPP_SOURCE
    assert "streamattn_transposed_wgmma_exact_decode_out_cuda" in CUDA_SOURCE
    assert "streamattn_transposed_wgmma_exact_merge_warp_kernel" in CUDA_SOURCE

    calls: list[str] = []

    def partial(*_args) -> None:
        calls.append("partial")

    def merge(*_args) -> None:
        calls.append("merge")

    def combined(*_args) -> None:
        calls.append("combined")

    def warp_merge(*_args) -> None:
        calls.append("warp_merge")

    tensor = torch.empty(1)
    plan = ExactDecodePlan(
        query=tensor,
        key_cache=tensor,
        value_cache=tensor,
        output=tensor,
        query_group=tensor,
        output_group=tensor,
        partial_output=tensor,
        partial_lse=tensor,
        num_splits=1,
        extension=None,
        partial_launch=partial,
        merge_launch=merge,
        warp_merge_launch=warp_merge,
        combined_launch=combined,
    )

    assert plan.run_two_call() is tensor
    assert calls == ["partial", "merge"]
    calls.clear()
    assert plan.run_combined() is tensor
    assert calls == ["combined"]
    calls.clear()
    assert plan.run_warp_merge() is tensor
    assert calls == ["partial", "warp_merge"]
    calls.clear()
    assert plan.run() is tensor
    assert calls == ["partial", "warp_merge"]


def _has_h100_and_cutlass() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        return False
    try:
        resolve_cutlass_root()
    except FileNotFoundError:
        return False
    return True


@pytest.mark.skipif(
    not _has_h100_and_cutlass(),
    reason="requires H100 and CUTLASS/CUTE headers",
)
def test_promoted_exact_plan_reuses_workspace_and_tracks_query_mutation():
    device = torch.device("cuda")
    q = torch.randn(4, 1, 16, 64, device=device, dtype=torch.bfloat16)
    k = torch.randn(4, 2, 32768, 64, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    plan = ExactDecodePlan.build(q, k, v)

    out = plan.run()
    first = out.clone()
    output_ptr = out.data_ptr()
    torch.cuda.synchronize()

    q_group = q.view(4, 2, 8, 64).float()
    scores = torch.einsum("bhgd,bhnd->bhgn", q_group, k.float()) * 0.125
    expected = torch.einsum("bhgn,bhnd->bhgd", scores.softmax(dim=-1), v.float())
    torch.testing.assert_close(
        out.view(4, 2, 8, 64).float(), expected, atol=5e-3, rtol=5e-3
    )

    q.add_(0.25)
    second = plan.run()
    torch.cuda.synchronize()
    assert second.data_ptr() == output_ptr
    assert not torch.equal(first, second)
    assert plan.num_splits == 64
    assert plan.workspace_bytes > 0
