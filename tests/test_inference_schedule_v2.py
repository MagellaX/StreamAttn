from __future__ import annotations

import pytest

from stream_attention.exact_compiler import GuaranteeClass
from stream_attention.inference_schedule import (
    BarrierTopology,
    ConsumerOverlap,
    HierarchicalExecutionPlan,
    MacroExecutionPlan,
    MacroPlanKind,
    OperandSource,
    OutputMode,
    PhysicalScheduleV2,
    ProducerTopology,
    TaskGranularity,
    WorkCohort,
)
from stream_attention.inference_workload import AttentionBatchV2


def _workload() -> AttentionBatchV2:
    requests = [
        {"request_id": "r0", "phase": "decode", "query_len": 1, "kv_len": 32768},
        {"request_id": "r1", "phase": "verify", "query_len": 4, "kv_len": 16384},
        {
            "request_id": "r2",
            "phase": "micro_prefill",
            "query_len": 32,
            "kv_len": 4096,
        },
    ]
    return AttentionBatchV2.from_dict(
        {
            "batch_id": "mixed-plan",
            "architecture": "sm90",
            "phase": "mixed",
            "requests": requests,
            "attention_kind": "gqa",
            "q_heads": 16,
            "kv_heads": 2,
            "d_qk": 128,
            "d_v": 128,
            "q_dtype": "bf16",
            "kv_dtype": "bf16",
            "output_dtype": "bf16",
            "cache_kind": "contiguous",
            "cache_layout": "bshd",
            "mask_kind": "noncausal",
            "execution_mode": "cuda_graph",
            "fixed_workspace_bytes": 1 << 20,
            "maximum_captured_batch": 8,
            "objective": "latency",
        }
    )


def _macro(workload: AttentionBatchV2) -> MacroExecutionPlan:
    return MacroExecutionPlan(
        workload_sha256=workload.fingerprint,
        plan_kind=MacroPlanKind.QUERY_LENGTH_COHORT,
        cohorts=(
            WorkCohort("decode", (0,), "decode", 1, 1),
            WorkCohort("verify", (1,), "verify", 2, 8),
            WorkCohort("micro", (2,), "micro_prefill", 9, 64),
        ),
        task_granularity=TaskGranularity.Q_TILE,
        launch_count=3,
        persistent_queue=False,
        split_kv=True,
        graph_safe=True,
        fixed_workspace_bytes=1 << 20,
        planner_version="universal_inference_v2",
    )


def _physical(cohort_id: str) -> PhysicalScheduleV2:
    return PhysicalScheduleV2(
        family_id=f"sm90_{cohort_id}",
        architecture="sm90",
        cohort_id=cohort_id,
        guarantee=GuaranteeClass.FULL_CONTEXT_EXACT,
        qk_operand_source=OperandSource.SS,
        pv_operand_source=OperandSource.RS,
        producer_topology=ProducerTopology.PRODUCER_WARPGROUP,
        barrier_topology=BarrierTopology.STAGE_LOCAL_MBARRIER,
        consumer_overlap=ConsumerOverlap.INTER_WARPGROUP_PING_PONG,
        task_granularity=TaskGranularity.Q_TILE,
        output_mode=OutputMode.DIRECT,
        q_heads_per_cta=8,
        q_positions_per_cta=16,
        kv_tokens_per_tile=128,
        split_q=1,
        split_kv=4,
        pipeline_stages=3,
        producer_warps=4,
        consumer_warpgroups=1,
        cluster_shape=(1, 1, 1),
        accumulator_space="register",
        load_engine="tma",
        mma_engine="wgmma",
        softmax_variant="online_fp32",
        merge_variant="exact_state",
        epilogue_variant="direct",
    )


def test_hierarchical_plan_partitions_batch_and_has_stable_keys():
    workload = _workload()
    macro = _macro(workload)
    plan = HierarchicalExecutionPlan(
        macro=macro,
        physical_schedules=tuple(_physical(name) for name in ("decode", "verify", "micro")),
    )
    plan.validate_for(workload)
    assert macro.plan_key == _macro(workload).plan_key
    assert _physical("verify").kernel_key == _physical("verify").kernel_key


def test_macro_plan_rejects_missing_requests():
    workload = _workload()
    macro = MacroExecutionPlan(
        workload_sha256=workload.fingerprint,
        plan_kind=MacroPlanKind.SPLIT_PHASE,
        cohorts=(WorkCohort("bad", (0,), "decode", 1, 1),),
        task_granularity=TaskGranularity.REQUEST,
        launch_count=1,
        persistent_queue=False,
        split_kv=False,
        graph_safe=True,
        fixed_workspace_bytes=0,
        planner_version="test",
    )
    with pytest.raises(ValueError, match="partition"):
        macro.validate_for(workload)


def test_v2_pipeline_fields_change_kernel_identity_without_touching_v1():
    base = _physical("verify")
    changed = PhysicalScheduleV2(
        **{
            **base.__dict__,
            "barrier_topology": BarrierTopology.NAMED,
        }
    )
    assert changed.kernel_key != base.kernel_key
