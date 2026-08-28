from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from stream_attention.exact_compiler import (
    AccumulatorSpace,
    AlgebraOrientation,
    BackwardRole,
    CandidateResourceEstimate,
    GuaranteeClass,
    LoadEngine,
    MmaEngine,
    Ownership,
    ScheduleCandidate,
    SchedulerKind,
    UniversalExactManifest,
    default_architecture_resource_models,
    load_universal_exact_manifest,
    matching_kernel_families,
    registered_exact_kernel_families,
)
from stream_attention.planning import (
    ATTENTION_GUARANTEE_SCHEDULE_EXACT,
    AttentionProblem,
    AttentionTilePlan,
)


def _sm100_candidate() -> ScheduleCandidate:
    return ScheduleCandidate(
        family_id="sm100_tgv_gqa_causal_prefill",
        architecture="sm100",
        phase_family="prefill",
        guarantee=GuaranteeClass.FULL_CONTEXT_EXACT,
        algebra_orientation=AlgebraOrientation.QK,
        ownership=Ownership.KV_GROUP,
        q_heads_per_cta=8,
        q_positions_per_cta=2,
        kv_tokens_per_tile=128,
        head_dim_stage=128,
        split_q=1,
        split_kv=1,
        producer_ctas=1,
        load_engine=LoadEngine.TMA,
        mma_engine=MmaEngine.TCGEN05,
        accumulator_space=AccumulatorSpace.TMEM,
        num_load_warps=4,
        num_consumer_warpgroups=1,
        pipeline_stages=3,
        scheduler=SchedulerKind.STATIC,
        cluster_shape=(1, 1, 1),
        softmax_variant="online_exp2",
        merge_variant="none",
        epilogue_variant="direct_bshd",
        backward_role=BackwardRole.FORWARD,
    )


def test_universal_exact_manifest_freezes_all_surfaces_architectures_and_phases():
    manifest = load_universal_exact_manifest()

    assert manifest.summary() == {
        "manifest_id": "universal_exact_v1_20260828",
        "schema_version": 1,
        "cell_count": 30,
        "surfaces": {"boundary": 10, "feature": 8, "real": 12},
        "architectures": {"sm100": 12, "sm80": 7, "sm90": 11},
        "phases": {"decode": 7, "prefill": 17, "train": 6},
        "dtypes": {"bfloat16": 23, "float16": 7},
        "head_dims": {"128": 25, "256": 2, "64": 3},
    }
    assert all(cell.guarantee is GuaranteeClass.FULL_CONTEXT_EXACT for cell in manifest.cells)
    assert all(cell.baseline_candidates for cell in manifest.cells)


@pytest.mark.parametrize("scheme", ["trace", "stratified", "boundary"])
def test_manifest_weight_schemes_normalize_without_hiding_zero_weight_cells(scheme: str):
    manifest = load_universal_exact_manifest()
    weights = manifest.normalized_weights(scheme)

    assert set(weights) == {cell.cell_id for cell in manifest.cells}
    assert sum(weights.values()) == pytest.approx(1.0)


def test_manifest_cells_lower_into_existing_attention_problem_contract():
    manifest = load_universal_exact_manifest()

    for cell in manifest.cells:
        problem = cell.to_attention_problem(device="cuda")
        assert isinstance(problem, AttentionProblem)
        assert problem.guarantee == "exact"
        assert problem.group_size == cell.group_size
        assert problem.kv_lengths == cell.kv_lengths


def test_manifest_rejects_duplicate_ids():
    manifest = load_universal_exact_manifest()

    with pytest.raises(ValueError, match="cell IDs must be unique"):
        UniversalExactManifest(
            schema_version=manifest.schema_version,
            manifest_id=manifest.manifest_id,
            cells=manifest.cells + (manifest.cells[0],),
            acceptance=manifest.acceptance,
        )


def test_schedule_exact_is_distinct_from_full_context_and_verified_guarantees():
    q = torch.randn(1, 1, 4, 32)
    k = torch.randn(1, 64, 2, 32)
    problem = AttentionProblem.from_contiguous(
        q,
        k,
        k,
        guarantee=ATTENTION_GUARANTEE_SCHEDULE_EXACT,
    )
    plan = AttentionTilePlan.selected(
        problem,
        logical_tile_size=32,
        tile_ids_per_row=((0,),),
        policy_id="explicit-schedule",
        reason="schedule_contract_test",
    )

    assert plan.problem.guarantee == "schedule_exact"
    assert plan.tile_coverage == 0.5


def test_candidate_key_is_stable_and_instruction_shaping():
    candidate = _sm100_candidate()

    assert candidate.kernel_key == _sm100_candidate().kernel_key
    assert candidate.kernel_key.startswith("sm100:sm100_tgv_gqa_causal_prefill:")
    assert "batch_size" not in candidate.as_dict()
    assert "query_len" not in candidate.as_dict()
    assert "kv_len" not in candidate.as_dict()
    assert replace(candidate, pipeline_stages=2).kernel_key != candidate.kernel_key


def test_sm100_resource_model_accepts_legal_tmem_and_rejects_structural_overflow():
    model = default_architecture_resource_models()["sm100"]
    candidate = _sm100_candidate()
    legal = model.check(
        candidate,
        CandidateResourceEstimate(
            threads_per_cta=256,
            dynamic_shared_memory_bytes=180_000,
            registers_per_thread=128,
            tmem_columns=64,
        ),
    )
    illegal = model.check(
        candidate,
        CandidateResourceEstimate(
            threads_per_cta=256,
            dynamic_shared_memory_bytes=180_000,
            registers_per_thread=128,
            tmem_columns=160,
        ),
    )

    assert legal.legal is True
    assert legal.estimated_active_ctas_per_sm == 1
    assert illegal.legal is False
    assert "tmem_columns" in illegal.reasons


def test_spills_are_a_hard_legality_failure_before_benchmarking():
    model = default_architecture_resource_models()["sm90"]
    legality = model.check(
        replace(
            _sm100_candidate(),
            architecture="sm90",
            load_engine=LoadEngine.TMA,
            mma_engine=MmaEngine.WGMMA,
            accumulator_space=AccumulatorSpace.REGISTERS,
        ),
        CandidateResourceEstimate(
            threads_per_cta=256,
            dynamic_shared_memory_bytes=100_000,
            registers_per_thread=128,
            spill_bytes=16,
        ),
    )

    assert legality.legal is False
    assert "register_spill" in legality.reasons


def test_existing_native_assets_are_registered_as_families_not_copied():
    manifest = load_universal_exact_manifest()
    families = registered_exact_kernel_families()
    by_id = {family.family_id: family for family in families}

    assert by_id["sm90_transposed_gqa_exact_decode"].maturity == "promoted"
    assert by_id["sm100_tgv_gqa_causal_prefill"].maturity == "calibration"
    assert by_id["explicit_external_exact_fallback"].native is False
    missing_native = [
        cell.cell_id
        for cell in manifest.cells
        if not matching_kernel_families(cell, families, native_only=True)
    ]
    assert missing_native == [
        "sm80_train_dropout_mha_d128",
        "sm90_train_deterministic_g4_d128",
    ]
    assert matching_kernel_families(
        next(cell for cell in manifest.cells if cell.cell_id == missing_native[0]),
        families,
    )[-1].family_id == "explicit_external_exact_fallback"


def test_promoted_b200_prefill_and_h100_decode_match_specific_native_families():
    manifest = load_universal_exact_manifest()
    families = registered_exact_kernel_families()
    cells = {cell.cell_id: cell for cell in manifest.cells}

    b200 = {
        family.family_id
        for family in matching_kernel_families(
            cells["sm100_prefill_b1_256_g8_d128"], families, native_only=True
        )
    }
    h100 = {
        family.family_id
        for family in matching_kernel_families(
            cells["sm90_decode_b1_32k_g8_d128_hnd"], families, native_only=True
        )
    }

    assert "sm100_tgv_gqa_causal_prefill" in b200
    assert "sm90_transposed_gqa_exact_decode" in h100
