import pytest
import torch

from stream_attention.backends.sm90.transposed_gqa_exact import (
    ExactDecodePlan,
    _shape_reasons,
    choose_num_splits,
    resolve_cutlass_root,
    supports_transposed_gqa_exact,
)
from stream_attention.backends.sm90.transposed_gqa_exact_sources import (
    CPP_SOURCE,
    CUDA_SOURCE,
)


def test_split_rule_preserves_256_producer_cta_target():
    assert choose_num_splits(batch=2, kv_heads=2, kv_len=32768) == 64
    assert choose_num_splits(batch=4, kv_heads=2, kv_len=32768) == 32
    assert choose_num_splits(batch=8, kv_heads=2, kv_len=32768) == 16


def test_promoted_shape_contract_requires_head_major_bf16_buffers():
    q = torch.empty(4, 1, 16, 64, dtype=torch.bfloat16)
    k_head_major = torch.empty(4, 2, 32768, 64, dtype=torch.bfloat16)
    v_head_major = torch.empty_like(k_head_major)
    assert _shape_reasons(q, k_head_major, v_head_major, promoted_only=True) == []

    k_token_major = torch.empty(4, 32768, 2, 64, dtype=torch.bfloat16)
    reasons = _shape_reasons(q, k_token_major, k_token_major, promoted_only=True)
    assert "gqa" in reasons
    assert "unpromoted_shape" in reasons
    assert not supports_transposed_gqa_exact(q, k_head_major, v_head_major)


def test_combined_exact_dispatch_is_bound_and_plan_keeps_two_call_control():
    assert 'm.def("exact_decode_out"' in CPP_SOURCE
    assert "streamattn_transposed_wgmma_exact_decode_out_cuda" in CUDA_SOURCE

    calls: list[str] = []

    def partial(*_args) -> None:
        calls.append("partial")

    def merge(*_args) -> None:
        calls.append("merge")

    def combined(*_args) -> None:
        calls.append("combined")

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
        combined_launch=combined,
    )

    assert plan.run_two_call() is tensor
    assert calls == ["partial", "merge"]
    calls.clear()
    assert plan.run_combined() is tensor
    assert calls == ["combined"]


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
    assert plan.num_splits == 32
    assert plan.workspace_bytes > 0
