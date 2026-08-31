import torch

from stream_attention.backends.sm80.tk_grouped_exact import (
    ExactDecodePlan,
    PROMOTED_EXACT_G4_SPLITS,
    PROMOTED_EXACT_G8_SPLITS,
    _shape_reasons,
    supports_grouped_exact,
)
from stream_attention.backends.sm80.tk_grouped_exact_sources import (
    CPP_SOURCE,
    CUDA_SOURCE,
)


def test_promoted_sm80_phase_table_matches_a100_gate():
    assert PROMOTED_EXACT_G8_SPLITS == {
        (1, 32768): 128,
        (2, 32768): 128,
        (4, 16384): 128,
        (4, 32768): 128,
        (4, 65536): 128,
        (8, 32768): 64,
    }
    assert PROMOTED_EXACT_G4_SPLITS == {(4, 32768): 64}


def test_sm80_shape_contract_is_head_major_bf16_d64():
    q = torch.empty(4, 1, 16, 64, dtype=torch.bfloat16, device="meta")
    k_g8 = torch.empty(4, 2, 32768, 64, dtype=torch.bfloat16, device="meta")
    assert _shape_reasons(q, k_g8, k_g8, promoted_only=True) == []

    k_g4 = torch.empty(4, 4, 32768, 64, dtype=torch.bfloat16, device="meta")
    assert _shape_reasons(q, k_g4, k_g4, promoted_only=True) == []

    k_token_major = torch.empty(
        4, 32768, 2, 64, dtype=torch.bfloat16, device="meta"
    )
    assert "unpromoted_shape" in _shape_reasons(
        q, k_token_major, k_token_major, promoted_only=True
    )

    q_d128 = torch.empty(4, 1, 16, 128, dtype=torch.bfloat16, device="meta")
    k_d128 = torch.empty(4, 2, 32768, 128, dtype=torch.bfloat16, device="meta")
    assert "head_dim" in _shape_reasons(
        q_d128, k_d128, k_d128, promoted_only=True
    )
    assert not supports_grouped_exact(q, k_g8, k_g8)


def test_sm80_source_contains_fused_boundary_and_allocation_free_entrypoint():
    assert 'm.def("exact_decode_chunk_merged_staged_grouped_direct_out"' in CPP_SOURCE
    assert "streamattn_tc_load_grouped_query" in CUDA_SOURCE
    assert "streamattn_tk_tc_exact_warp_merge_direct_kernel" in CUDA_SOURCE
    assert "warp::load_async" in CUDA_SOURCE
    assert "producer_warps = 4" in CUDA_SOURCE


def test_sm80_plan_reuses_bound_workspace_and_output():
    calls = []

    def launch(*args):
        calls.append(args)

    tensor = torch.empty(1)
    plan = ExactDecodePlan(
        query=tensor,
        key_cache=tensor,
        value_cache=tensor,
        output=tensor,
        query_flat=tensor,
        output_flat=tensor,
        partial_output=tensor,
        partial_lse=tensor,
        num_chunks=64,
        extension=None,
        launch=launch,
    )

    assert plan.run() is tensor
    assert len(calls) == 1
    assert calls[0][-1] == 64
    assert plan.workspace_bytes == 2 * tensor.element_size()
