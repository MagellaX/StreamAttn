import torch
import pytest
from pathlib import Path

from stream_attention.decode import (
    DecodeCostEntry,
    DecodeCostKey,
    DecodeCostModel,
    Gate0SeedOnlyBatchedPolicy,
    STREAMATTN_EXACT_NATIVE_BACKEND,
    StreamAttnDecodePolicy,
    StreamAttnDecodeWorkspace,
    StreamAttnDecodeWrapper,
    StreamAttnSeedOnlyDecodeService,
    decode_cost_model_from_profile_rows,
    find_packaged_gate0_seed_only_batched_policies,
    list_packaged_gate0_seed_only_batched_policies,
    load_packaged_gate0_seed_only_batched_policy,
    packaged_gate0_seed_only_batched_policy_registry,
    stream_attn_decode_plan,
    stream_attn_decode_run,
    stream_attn_exact_native_decode,
    stream_attn_seed_only_decode,
)
from stream_attention.gate0_fused_hybrid import Gate0FusedHybridPolicy
from stream_attention.gate1 import dense_attention_forward


def _tensors(kv_len: int = 4096):
    q = torch.randn(1, 1, 2, 8)
    k = torch.randn(1, kv_len, 2, 8)
    v = torch.randn(1, kv_len, 2, 8)
    return q, k, v


def _key(q, k) -> DecodeCostKey:
    return DecodeCostKey.from_tensors(
        q,
        k,
        block_size=128,
        tile_size_q=16,
        num_warps=4,
        num_stages=3,
    )


def test_decode_cost_entry_predicts_with_gate1_dense_equiv():
    entry = DecodeCostEntry.from_measurement(
        dense_ms=1.0,
        qk_scan_ms=0.2,
        gate1_dense_equiv_ms=1.8,
        gate1_mass_ms=0.42,
        active_fraction=0.125,
        gate1_value_bound_ms=0.47,
    )

    assert entry.gate1_pv_ms == 1.6
    assert abs(entry.predicate_overhead_ms - 0.02) < 1.0e-6
    assert abs(entry.predict_mass_ms(0.125) - 0.42) < 1.0e-6
    assert abs(entry.predict_value_bound_ms(0.125) - 0.47) < 1.0e-6


def test_decode_cost_model_json_roundtrip(tmp_path):
    q, k, _ = _tensors()
    model = DecodeCostModel()
    model.update(
        _key(q, k),
        DecodeCostEntry.from_measurement(
            dense_ms=1.0,
            qk_scan_ms=0.2,
            gate1_dense_equiv_ms=1.8,
            gate1_mass_ms=0.42,
            active_fraction=0.125,
        ),
    )

    path = tmp_path / "decode_cost.json"
    model.to_json(path)
    loaded = DecodeCostModel.from_json(path)

    assert loaded.lookup(_key(q, k)).predict_mass_ms(0.125) == model.lookup(
        _key(q, k)
    ).predict_mass_ms(0.125)


def test_decode_plan_chooses_mass_when_calibrated_profitable():
    q, k, _ = _tensors()
    model = DecodeCostModel()
    model.update(
        _key(q, k),
        DecodeCostEntry.from_measurement(
            dense_ms=1.0,
            qk_scan_ms=0.2,
            gate1_dense_equiv_ms=1.8,
            gate1_mass_ms=0.32,
            active_fraction=0.0625,
        ),
    )

    plan = stream_attn_decode_plan(
        q,
        k,
        decode_cost_model=model,
        policy=StreamAttnDecodePolicy(allow_value_bound=False),
        active_fraction_hint=0.0625,
        block_size=128,
        tile_size_q=16,
    )

    assert plan.backend == "gate1_mass"
    assert plan.skip_predicate == "mass"
    assert plan.predicted_active_fraction == 0.0625


def test_decode_plan_falls_back_dense_when_cost_says_dense():
    q, k, _ = _tensors()
    model = DecodeCostModel()
    model.update(
        _key(q, k),
        DecodeCostEntry.from_measurement(
            dense_ms=1.0,
            qk_scan_ms=0.2,
            gate1_dense_equiv_ms=1.8,
            gate1_mass_ms=1.1,
            active_fraction=0.25,
        ),
    )

    plan = stream_attn_decode_plan(
        q,
        k,
        decode_cost_model=model,
        active_fraction_hint=0.25,
        block_size=128,
        tile_size_q=16,
    )

    assert plan.backend == "dense"
    assert plan.reason == "dense_fallback"


def test_decode_run_dense_plan_matches_dense_shape():
    q, k, v = _tensors(kv_len=64)
    plan = stream_attn_decode_plan(q, k, active_fraction_hint=1.0)

    out = stream_attn_decode_run(q, k, v, plan=plan)

    assert out.shape == q.shape


def test_decode_cost_model_from_profile_rows_groups_active_curve():
    row_base = {
        "device": "NVIDIA H100 80GB HBM3",
        "shape": {
            "query_len": 1,
            "kv_len": 8192,
            "heads": 16,
            "kv_heads": 16,
            "dim": 128,
            "dtype": "fp16",
            "attention_type": "mha",
        },
        "block_size": 128,
        "tile_size_q": 16,
        "num_warps": 4,
        "num_stages": 3,
        "causal_mode": "none",
        "dense_decode_ms": 0.14,
        "gate1_qk_scan_ms": 0.06,
        "gate1_dense_equiv_ms": 0.34,
        "metadata_update_wall_ms": 0.02,
        "metadata_full_build_ms": 0.04,
    }
    rows = [
        {
            **row_base,
            "active_pv_fraction_mass": 0.0625,
            "gate1_mass_ms": 0.09,
            "gate1_value_bound_ms": 0.10,
        },
        {
            **row_base,
            "active_pv_fraction_mass": 0.25,
            "gate1_mass_ms": 0.16,
            "gate1_value_bound_ms": 0.18,
        },
    ]

    model = decode_cost_model_from_profile_rows(rows)
    key = DecodeCostKey(
        device_class="sm90_h100",
        dtype="fp16",
        query_len=1,
        kv_bucket=8192,
        heads=16,
        kv_heads=16,
        dim=128,
        attention_type="mha",
        block_size=128,
        tile_size_q=16,
        num_warps=4,
        num_stages=3,
        causal=False,
    )

    entry = model.lookup(key)
    assert entry is not None
    assert entry.sample_count == 2
    assert entry.metadata_update_ms == 0.02


def test_decode_workspace_allocation_validates_dimensions_and_device():
    workspace = StreamAttnDecodeWorkspace.allocate(
        device="cpu",
        max_batch=1,
        max_query_len=2,
        max_kv_len=64,
        max_heads=2,
        head_dim=8,
        block_size=16,
        dtype=torch.float32,
    )
    q = torch.randn(1, 2, 2, 8)
    k = torch.randn(1, 64, 2, 8)
    v = torch.randn(1, 64, 2, 8)

    workspace.validate(q, k, v)

    assert workspace.raw_stats.shape == (1, 2, 1, 6)
    assert workspace.output.shape == (1, 2, 2, 8)
    assert workspace.metadata_update_scratch.shape == (1, 2, 4)
    with pytest.raises(ValueError):
        workspace.validate(batch=2)


def test_decode_wrapper_falls_back_dense_without_cost_entry():
    q, k, v = _tensors(kv_len=64)
    workspace = StreamAttnDecodeWorkspace.allocate(
        device="cpu",
        max_batch=1,
        max_query_len=1,
        max_kv_len=64,
        max_heads=2,
        head_dim=8,
        block_size=16,
        dtype=q.dtype,
    )
    wrapper = StreamAttnDecodeWrapper(workspace)
    wrapper.plan(
        query_shape=q.shape,
        kv_shape=k.shape,
        block_size=16,
        tile_size_q=1,
    )

    out = wrapper.run(q, k, v, active_fraction_hint=1.0)

    assert out.shape == q.shape
    assert wrapper.last_plan.backend == "dense"
    assert wrapper.last_plan.reason == "missing_decode_cost"


def test_decode_wrapper_plan_step_chooses_mass_when_calibrated():
    q, k, _ = _tensors()
    model = DecodeCostModel()
    model.update(
        _key(q, k),
        DecodeCostEntry.from_measurement(
            dense_ms=1.0,
            qk_scan_ms=0.2,
            gate1_dense_equiv_ms=1.8,
            gate1_mass_ms=0.32,
            active_fraction=0.0625,
        ),
    )
    workspace = StreamAttnDecodeWorkspace.allocate(
        device="cpu",
        max_batch=1,
        max_query_len=1,
        max_kv_len=4096,
        max_heads=2,
        head_dim=8,
        block_size=128,
        dtype=q.dtype,
    )
    wrapper = StreamAttnDecodeWrapper(
        workspace,
        policy=StreamAttnDecodePolicy(allow_value_bound=False),
        decode_cost_model=model,
    )
    wrapper.plan(
        query_shape=q.shape,
        kv_shape=k.shape,
        block_size=128,
        tile_size_q=16,
    )

    plan = wrapper.plan_step(q, k, active_fraction_hint=0.0625)

    assert plan.backend == "gate1_mass"
    assert plan.reason == "calibrated_mass_profitable"


def test_decode_wrapper_reuses_last_active_fraction_for_step_plan():
    q, k, _ = _tensors()
    model = DecodeCostModel()
    model.update(
        _key(q, k),
        DecodeCostEntry.from_measurement(
            dense_ms=1.0,
            qk_scan_ms=0.2,
            gate1_dense_equiv_ms=1.8,
            gate1_mass_ms=0.32,
            active_fraction=0.0625,
        ),
    )
    workspace = StreamAttnDecodeWorkspace.allocate(
        device="cpu",
        max_batch=1,
        max_query_len=1,
        max_kv_len=4096,
        max_heads=2,
        head_dim=8,
        block_size=128,
        dtype=q.dtype,
    )
    wrapper = StreamAttnDecodeWrapper(
        workspace,
        policy=StreamAttnDecodePolicy(allow_value_bound=False),
        decode_cost_model=model,
    )
    wrapper.plan(
        query_shape=q.shape,
        kv_shape=k.shape,
        block_size=128,
        tile_size_q=16,
    )
    wrapper.observe_active_fraction(0.0625)

    plan = wrapper.plan_step(q, k)

    assert plan.backend == "gate1_mass"
    assert plan.predicted_active_fraction == 0.0625


def _gate0_policy(q, k, *, speedup: float = 1.4, error: float = 0.005):
    dense_ms = 1.0
    return Gate0FusedHybridPolicy(
        head_modes=tuple(0 if head == 1 else 1 for head in range(q.shape[2])),
        trusted_sparse_heads=(1,),
        exact_heads=tuple(head for head in range(q.shape[2]) if head != 1),
        block_size=16,
        sink_blocks=1,
        recent_blocks=1,
        middle_seed_blocks=1,
        num_chunks=2,
        seed_strategy="recompute_seed",
        filter_margin=32.0,
        error_budget=0.01,
        projection_dim=2,
        projection_metadata_dtype="fp16",
        expected_dense_ms=dense_ms,
        expected_fused_hybrid_ms=dense_ms / speedup,
        expected_speedup_vs_dense=speedup,
        expected_max_abs_error=error,
        expected_mean_abs_error=error / 10,
        kv_len_bucket=k.shape[1],
    )


def _seed_only_policy(
    q,
    k,
    *,
    speedup: float = 1.2,
    min_batch: int = 4,
):
    dense_ms = 1.0
    return Gate0SeedOnlyBatchedPolicy(
        model_id="test-model",
        layer_id=8,
        policy_id="test-seed-only-batched",
        dtype="fp32",
        kv_len_bucket=k.shape[1],
        min_batch=min_batch,
        heads=q.shape[2],
        kv_heads=k.shape[2],
        dim=q.shape[3],
        block_size=16,
        sink_blocks=1,
        recent_blocks=1,
        middle_seed_blocks=1,
        block_order="recent_first",
        expected_dense_ms=dense_ms,
        expected_seed_only_ms=dense_ms / speedup,
        expected_speedup_vs_dense=speedup,
        max_kl=1.0e-5,
        max_logit_delta=0.2,
    )


def test_decode_plan_chooses_gate0_fused_hybrid_from_policy_without_cost_model():
    q, k, _ = _tensors(kv_len=32768)
    gate0_policy = _gate0_policy(q, k, speedup=1.45, error=0.005)

    plan = stream_attn_decode_plan(
        q,
        k,
        gate0_fused_hybrid_policy=gate0_policy,
        active_fraction_hint=1.0,
        block_size=16,
        tile_size_q=16,
        error_budget=0.01,
    )

    assert plan.backend == "gate0_fused_hybrid"
    assert plan.reason == "calibrated_gate0_fused_hybrid_policy"
    assert plan.gate0_fused_hybrid_policy is gate0_policy
    assert plan.projection_metadata_required is True
    assert plan.metadata_update_required is True


def test_decode_plan_rejects_gate0_when_policy_error_exceeds_budget():
    q, k, _ = _tensors(kv_len=32768)
    gate0_policy = _gate0_policy(q, k, speedup=1.45, error=0.02)

    plan = stream_attn_decode_plan(
        q,
        k,
        gate0_fused_hybrid_policy=gate0_policy,
        active_fraction_hint=1.0,
        block_size=16,
        tile_size_q=16,
        error_budget=0.01,
    )

    assert plan.backend == "dense"
    assert plan.reason == "missing_decode_cost"


def test_decode_run_gate0_plan_uses_dense_fallback_on_cpu():
    q, k, v = _tensors(kv_len=32768)
    gate0_policy = _gate0_policy(q, k, speedup=1.45, error=0.005)
    plan = stream_attn_decode_plan(
        q,
        k,
        gate0_fused_hybrid_policy=gate0_policy,
        active_fraction_hint=1.0,
        block_size=16,
        tile_size_q=16,
        error_budget=0.01,
    )

    out, info = stream_attn_decode_run(q, k, v, plan=plan, return_info=True)
    expected = dense_attention_forward(q, k, v, causal=False)

    torch.testing.assert_close(out, expected)
    assert info.stats is None


def test_decode_wrapper_can_plan_gate0_fused_hybrid_backend():
    q, k, v = _tensors(kv_len=32768)
    gate0_policy = _gate0_policy(q, k, speedup=1.45, error=0.005)
    workspace = StreamAttnDecodeWorkspace.allocate(
        device="cpu",
        max_batch=1,
        max_query_len=1,
        max_kv_len=32768,
        max_heads=2,
        head_dim=8,
        block_size=16,
        dtype=q.dtype,
    )
    wrapper = StreamAttnDecodeWrapper(
        workspace,
        gate0_fused_hybrid_policy=gate0_policy,
    )
    wrapper.plan(
        query_shape=q.shape,
        kv_shape=k.shape,
        block_size=16,
        tile_size_q=16,
        error_budget=0.01,
    )

    out = wrapper.run(q, k, v, active_fraction_hint=1.0)

    assert out.shape == q.shape
    assert wrapper.last_plan.backend == "gate0_fused_hybrid"


def test_decode_plan_chooses_gate0_seed_only_batched_from_policy_without_cost_model():
    q, k, _ = _tensors(kv_len=128)
    q = q.repeat(4, 1, 1, 1)
    k = k.repeat(4, 1, 1, 1)
    seed_policy = _seed_only_policy(q, k, speedup=1.2)

    plan = stream_attn_decode_plan(
        q,
        k,
        gate0_seed_only_batched_policy=seed_policy,
        policy=StreamAttnDecodePolicy(min_kv_len_for_gate0_seed_only=1),
        active_fraction_hint=1.0,
        block_size=16,
        tile_size_q=16,
    )

    assert plan.backend == "gate0_seed_only_batched"
    assert plan.reason == "calibrated_gate0_seed_only_batched_policy"
    assert plan.gate0_seed_only_batched_policy is seed_policy
    assert plan.projection_metadata_required is False
    assert plan.metadata_update_required is False


def test_decode_plan_rejects_gate0_seed_only_batched_below_min_batch():
    q, k, _ = _tensors(kv_len=128)
    seed_policy = _seed_only_policy(q, k, speedup=1.2, min_batch=4)

    plan = stream_attn_decode_plan(
        q,
        k,
        gate0_seed_only_batched_policy=seed_policy,
        policy=StreamAttnDecodePolicy(min_kv_len_for_gate0_seed_only=1),
        active_fraction_hint=1.0,
        block_size=16,
        tile_size_q=16,
    )

    assert plan.backend == "dense"
    assert plan.reason == "missing_decode_cost"
    assert plan.fallback_reason == "batch_below_min"


def test_decode_run_gate0_seed_only_batched_uses_dense_fallback_on_cpu():
    q, k, v = _tensors(kv_len=128)
    q = q.repeat(4, 1, 1, 1)
    k = k.repeat(4, 1, 1, 1)
    v = v.repeat(4, 1, 1, 1)
    seed_policy = _seed_only_policy(q, k, speedup=1.2)
    plan = stream_attn_decode_plan(
        q,
        k,
        gate0_seed_only_batched_policy=seed_policy,
        policy=StreamAttnDecodePolicy(min_kv_len_for_gate0_seed_only=1),
        active_fraction_hint=1.0,
        block_size=16,
        tile_size_q=16,
    )

    out, info = stream_attn_decode_run(q, k, v, plan=plan, return_info=True)
    expected = dense_attention_forward(q, k, v, causal=False)

    torch.testing.assert_close(out, expected)
    assert info.backend_used == "dense"
    assert info.fallback_reason == "seed_only_batched_requires_cuda"
    assert info.stats is None


def test_decode_run_dense_plan_uses_injected_fallback():
    q, k, v = _tensors(kv_len=128)
    plan = stream_attn_decode_plan(
        q,
        k,
        policy=StreamAttnDecodePolicy(min_kv_len_for_gate1=4096),
        active_fraction_hint=1.0,
        block_size=16,
        tile_size_q=16,
    )

    out = stream_attn_decode_run(
        q,
        k,
        v,
        plan=plan,
        dense_fallback=lambda query, key, value: torch.zeros_like(query),
    )

    assert plan.backend == "dense"
    torch.testing.assert_close(out, torch.zeros_like(q))


def test_decode_wrapper_can_plan_gate0_seed_only_batched_backend():
    q, k, v = _tensors(kv_len=128)
    q = q.repeat(4, 1, 1, 1)
    k = k.repeat(4, 1, 1, 1)
    v = v.repeat(4, 1, 1, 1)
    seed_policy = _seed_only_policy(q, k, speedup=1.2)
    workspace = StreamAttnDecodeWorkspace.allocate(
        device="cpu",
        max_batch=4,
        max_query_len=1,
        max_kv_len=128,
        max_heads=2,
        head_dim=8,
        block_size=16,
        dtype=q.dtype,
    )
    wrapper = StreamAttnDecodeWrapper(
        workspace,
        policy=StreamAttnDecodePolicy(min_kv_len_for_gate0_seed_only=1),
        gate0_seed_only_batched_policy=seed_policy,
    )
    wrapper.plan(
        query_shape=q.shape,
        kv_shape=k.shape,
        block_size=16,
        tile_size_q=16,
    )

    out = wrapper.run(q, k, v, active_fraction_hint=1.0)

    assert out.shape == q.shape
    assert wrapper.last_plan.backend == "gate0_seed_only_batched"
    assert wrapper.last_info.backend_used == "dense"
    assert wrapper.runtime_counters()["backend_counts"] == {"dense": 1}
    assert wrapper.runtime_counters()["fallback_reasons"] == {
        "seed_only_batched_requires_cuda": 1
    }


def test_decode_wrapper_counts_injected_dense_fallback_backend():
    q, k, v = _tensors(kv_len=128)
    seed_policy = _seed_only_policy(q, k, speedup=1.2, min_batch=4)
    workspace = StreamAttnDecodeWorkspace.allocate(
        device="cpu",
        max_batch=1,
        max_query_len=1,
        max_kv_len=128,
        max_heads=2,
        head_dim=8,
        block_size=16,
        dtype=q.dtype,
    )
    wrapper = StreamAttnDecodeWrapper(
        workspace,
        policy=StreamAttnDecodePolicy(min_kv_len_for_gate0_seed_only=1),
        gate0_seed_only_batched_policy=seed_policy,
        dense_fallback=lambda query, key, value: torch.zeros_like(query),
        dense_fallback_backend="flashinfer_dense",
    )
    wrapper.plan(
        query_shape=q.shape,
        kv_shape=k.shape,
        block_size=16,
        tile_size_q=16,
    )

    out = wrapper.run(q, k, v, active_fraction_hint=1.0)

    torch.testing.assert_close(out, torch.zeros_like(q))
    assert wrapper.last_plan.backend == "dense"
    assert wrapper.runtime_counters()["backend_counts"] == {"flashinfer_dense": 1}
    assert wrapper.runtime_counters()["fallback_reasons"] == {"batch_below_min": 1}


def test_gate0_seed_only_batched_policy_loads_captured_batch_artifact_entry():
    entry = {
        "model_id": "Qwen2.5-0.5B",
        "layer_id": 8,
        "mode": "all_seed_only",
        "shape": {
            "batch": 4,
            "q_heads": 14,
            "true_kv_heads": 2,
            "dim": 64,
            "dtype": "fp16",
            "kv_len": 32768,
        },
        "seed_config": {
            "block_size": 32,
            "sink_blocks": 2,
            "recent_blocks": 2,
            "middle_seed_blocks": 8,
            "block_order": "recent_first",
            "num_warps": 4,
            "num_stages": 2,
        },
        "timing": {
            "seed_direct_full_prealloc_ms": 0.0552,
            "flashinfer_batch_tc_exact_ms": 0.0651,
            "speedup_vs_flashinfer_batch": 1.18,
        },
        "quality": {
            "seed_vs_flashinfer_exact": {
                "max_abs_error": 0.2429,
                "mean_abs_error": 0.00958,
            }
        },
    }

    policy = Gate0SeedOnlyBatchedPolicy.from_entry(entry)

    assert policy.model_id == "Qwen2.5-0.5B"
    assert policy.layer_id == 8
    assert policy.min_batch == 4
    assert policy.heads == 14
    assert policy.kv_heads == 2
    assert policy.dim == 64
    assert policy.dtype == "fp16"
    assert policy.kv_len_bucket == 32768
    assert policy.expected_seed_only_ms == 0.0552
    assert policy.expected_dense_ms == 0.0651
    assert policy.expected_speedup_vs_dense == 1.18
    assert policy.expected_max_abs_error == 0.2429


def test_gate0_seed_only_batched_policy_loads_packaged_l8_artifact():
    path = (
        Path(__file__).resolve().parents[1]
        / "stream_attention"
        / "policies"
        / "qwen25_05b_l8_32k_seed_only_batched.json"
    )

    policy = Gate0SeedOnlyBatchedPolicy.from_json(path)

    assert policy.policy_id == "qwen25_05b_l8_32k_fp16_b4_seed_only_v2"
    assert policy.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert policy.layer_id == 8
    assert policy.min_batch == 4
    assert policy.heads == 14
    assert policy.kv_heads == 2
    assert policy.dim == 64
    assert policy.dtype == "fp16"
    assert policy.kv_len_bucket == 32768
    assert policy.expected_seed_only_ms == 0.0254828811
    assert policy.expected_dense_ms == 0.0343097591
    assert policy.expected_speedup_vs_dense == 1.34638462
    assert policy.max_kl == 0.0001
    assert policy.max_logprob_delta == 0.001


def test_gate0_seed_only_batched_policy_loads_packaged_default():
    policy = load_packaged_gate0_seed_only_batched_policy()
    via_class = Gate0SeedOnlyBatchedPolicy.from_packaged(
        "qwen25_05b_l8_32k_fp16_b4_seed_only_v2"
    )
    via_compat_alias = Gate0SeedOnlyBatchedPolicy.from_packaged(
        "qwen25_05b_l8_32k_fp16_b8_seed_only_v1"
    )

    assert policy.policy_id == "qwen25_05b_l8_32k_fp16_b4_seed_only_v2"
    assert via_class.policy_id == policy.policy_id
    assert via_compat_alias.policy_id == policy.policy_id
    assert policy.min_batch == 4


def test_gate0_seed_only_batched_policy_loads_packaged_l6_artifact():
    policy = load_packaged_gate0_seed_only_batched_policy(
        "qwen25_05b_l6_32k_seed_only_batched"
    )
    via_id = Gate0SeedOnlyBatchedPolicy.from_packaged(
        "qwen25_05b_l6_32k_fp16_b4_seed_only_v1"
    )

    assert policy.policy_id == "qwen25_05b_l6_32k_fp16_b4_seed_only_v1"
    assert via_id.policy_id == policy.policy_id
    assert policy.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert policy.layer_id == 6
    assert policy.min_batch == 4
    assert policy.min_topk_overlap == 4
    assert policy.max_kl == 0.0001
    assert policy.max_logprob_delta == 0.001


def test_gate0_seed_only_batched_policy_loads_packaged_expansion_cells():
    expected = {
        "qwen25_05b_l1_32k_seed_only_batched": (
            "qwen25_05b_l1_32k_fp16_b4_seed_only_v1",
            1,
            0.002,
        ),
        "qwen25_05b_l2_32k_seed_only_batched": (
            "qwen25_05b_l2_32k_fp16_b4_seed_only_v1",
            2,
            0.001,
        ),
        "qwen25_05b_l5_32k_seed_only_batched": (
            "qwen25_05b_l5_32k_fp16_b4_seed_only_v1",
            5,
            0.001,
        ),
        "qwen25_05b_l18_32k_seed_only_batched": (
            "qwen25_05b_l18_32k_fp16_b4_seed_only_v1",
            18,
            0.001,
        ),
        "qwen25_15b_l0_32k_seed_only_batched": (
            "qwen25_15b_l0_32k_fp16_b4_seed_only_v1",
            0,
            0.005,
        ),
        "qwen25_15b_l3_32k_seed_only_batched": (
            "qwen25_15b_l3_32k_fp16_b4_seed_only_v1",
            3,
            0.001,
        ),
        "qwen25_3b_l0_32k_seed_only_batched": (
            "qwen25_3b_l0_32k_fp16_b4_seed_only_v1",
            0,
            0.001,
        ),
        "qwen25_3b_l14_32k_seed_only_batched": (
            "qwen25_3b_l14_32k_fp16_b4_seed_only_v1",
            14,
            0.001,
        ),
        "qwen25_3b_l16_32k_seed_only_batched": (
            "qwen25_3b_l16_32k_fp16_b4_seed_only_v1",
            16,
            0.001,
        ),
        "qwen25_3b_l24_32k_seed_only_batched": (
            "qwen25_3b_l24_32k_fp16_b4_seed_only_v1",
            24,
            0.001,
        ),
        "qwen25_3b_l26_32k_seed_only_batched": (
            "qwen25_3b_l26_32k_fp16_b4_seed_only_v1",
            26,
            0.001,
        ),
        "qwen25_3b_l27_32k_seed_only_batched": (
            "qwen25_3b_l27_32k_fp16_b4_seed_only_v1",
            27,
            0.001,
        ),
        "qwen25_3b_l29_32k_seed_only_batched": (
            "qwen25_3b_l29_32k_fp16_b4_seed_only_v1",
            29,
            0.002,
        ),
        "qwen25_3b_l35_32k_seed_only_batched": (
            "qwen25_3b_l35_32k_fp16_b4_seed_only_v1",
            35,
            0.001,
        ),
    }

    for name, (policy_id, layer_id, max_logprob_delta) in expected.items():
        policy = load_packaged_gate0_seed_only_batched_policy(name)
        via_id = Gate0SeedOnlyBatchedPolicy.from_packaged(policy_id)

        assert policy.policy_id == policy_id
        assert via_id.policy_id == policy.policy_id
        assert policy.layer_id == layer_id
        assert policy.min_batch == 4
        assert policy.min_topk_overlap == 4
        assert policy.max_kl == 0.0001
        assert policy.max_logprob_delta == max_logprob_delta
        if name.startswith("qwen25_15b"):
            assert policy.model_id == "Qwen/Qwen2.5-1.5B-Instruct"
            assert policy.heads == 12
            assert policy.kv_heads == 2
            assert policy.dim == 128
        if name.startswith("qwen25_3b"):
            assert policy.model_id == "Qwen/Qwen2.5-3B-Instruct"
            assert policy.heads == 16
            assert policy.kv_heads == 2
            assert policy.dim == 128


def test_gate0_seed_only_batched_policy_registry_lists_green_cells():
    registry = packaged_gate0_seed_only_batched_policy_registry()
    names = list_packaged_gate0_seed_only_batched_policies()
    names_with_aliases = list_packaged_gate0_seed_only_batched_policies(include_aliases=True)

    assert registry["schema"] == "streamattn.policy_registry.v1"
    assert registry["default"] == "qwen25_05b_l8_32k_seed_only_batched"
    expected_names = [
        "qwen25_05b_l1_32k_seed_only_batched",
        "qwen25_05b_l2_32k_seed_only_batched",
        "qwen25_05b_l5_32k_seed_only_batched",
        "qwen25_05b_l6_32k_seed_only_batched",
        "qwen25_05b_l8_32k_seed_only_batched",
        "qwen25_05b_l18_32k_seed_only_batched",
        "qwen25_15b_l0_32k_seed_only_batched",
        "qwen25_15b_l3_32k_seed_only_batched",
        "qwen25_3b_l0_32k_seed_only_batched",
        "qwen25_3b_l14_32k_seed_only_batched",
        "qwen25_3b_l16_32k_seed_only_batched",
        "qwen25_3b_l24_32k_seed_only_batched",
        "qwen25_3b_l26_32k_seed_only_batched",
        "qwen25_3b_l27_32k_seed_only_batched",
        "qwen25_3b_l29_32k_seed_only_batched",
        "qwen25_3b_l35_32k_seed_only_batched",
    ]

    assert names == expected_names
    assert "qwen25_05b_l1_32k_fp16_b4_seed_only_v1" in names_with_aliases
    assert "qwen25_05b_l2_32k_fp16_b4_seed_only_v1" in names_with_aliases
    assert "qwen25_05b_l5_32k_fp16_b4_seed_only_v1" in names_with_aliases
    assert "qwen25_05b_l6_32k_fp16_b4_seed_only_v1" in names_with_aliases
    assert "qwen25_05b_l8_32k_fp16_b4_seed_only_v2" in names_with_aliases
    assert "qwen25_05b_l8_32k_fp16_b8_seed_only_v1" in names_with_aliases
    assert "qwen25_05b_l18_32k_fp16_b4_seed_only_v1" in names_with_aliases
    assert "qwen25_15b_l0_32k_fp16_b4_seed_only_v1" in names_with_aliases
    assert "qwen25_15b_l3_32k_fp16_b4_seed_only_v1" in names_with_aliases
    assert "qwen25_3b_l0_32k_fp16_b4_seed_only_v1" in names_with_aliases
    assert "qwen25_3b_l14_32k_fp16_b4_seed_only_v1" in names_with_aliases
    assert "qwen25_3b_l16_32k_fp16_b4_seed_only_v1" in names_with_aliases
    assert "qwen25_3b_l24_32k_fp16_b4_seed_only_v1" in names_with_aliases
    assert "qwen25_3b_l26_32k_fp16_b4_seed_only_v1" in names_with_aliases
    assert "qwen25_3b_l27_32k_fp16_b4_seed_only_v1" in names_with_aliases
    assert "qwen25_3b_l29_32k_fp16_b4_seed_only_v1" in names_with_aliases
    assert "qwen25_3b_l35_32k_fp16_b4_seed_only_v1" in names_with_aliases


def test_gate0_seed_only_batched_policy_registry_finds_matching_cells():
    l8_matches = find_packaged_gate0_seed_only_batched_policies(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        layer_id=8,
        dtype="float16",
        kv_len_bucket=32768,
        min_batch=4,
    )
    all_matches = find_packaged_gate0_seed_only_batched_policies(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        dtype="fp16",
        kv_len_bucket=32768,
        min_batch=4,
    )
    too_small_batch = find_packaged_gate0_seed_only_batched_policies(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        layer_id=8,
        dtype="fp16",
        kv_len_bucket=32768,
        min_batch=2,
    )
    wrong_layer = find_packaged_gate0_seed_only_batched_policies(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        layer_id=9,
        dtype="fp16",
        kv_len_bucket=32768,
        min_batch=4,
    )
    qwen15_matches = find_packaged_gate0_seed_only_batched_policies(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        dtype="fp16",
        kv_len_bucket=32768,
        min_batch=4,
    )
    qwen3_matches = find_packaged_gate0_seed_only_batched_policies(
        model_id="Qwen/Qwen2.5-3B-Instruct",
        dtype="fp16",
        kv_len_bucket=32768,
        min_batch=4,
    )

    assert l8_matches == ["qwen25_05b_l8_32k_seed_only_batched"]
    assert all_matches == [
        "qwen25_05b_l1_32k_seed_only_batched",
        "qwen25_05b_l2_32k_seed_only_batched",
        "qwen25_05b_l5_32k_seed_only_batched",
        "qwen25_05b_l6_32k_seed_only_batched",
        "qwen25_05b_l8_32k_seed_only_batched",
        "qwen25_05b_l18_32k_seed_only_batched",
    ]
    assert qwen15_matches == [
        "qwen25_15b_l0_32k_seed_only_batched",
        "qwen25_15b_l3_32k_seed_only_batched",
    ]
    assert qwen3_matches == [
        "qwen25_3b_l0_32k_seed_only_batched",
        "qwen25_3b_l14_32k_seed_only_batched",
        "qwen25_3b_l16_32k_seed_only_batched",
        "qwen25_3b_l24_32k_seed_only_batched",
        "qwen25_3b_l26_32k_seed_only_batched",
        "qwen25_3b_l27_32k_seed_only_batched",
        "qwen25_3b_l29_32k_seed_only_batched",
        "qwen25_3b_l35_32k_seed_only_batched",
    ]
    assert too_small_batch == []
    assert wrong_layer == []


def test_gate0_seed_only_batched_policy_reports_fail_closed_mismatches():
    q, k, _ = _tensors(kv_len=128)
    q = q.repeat(4, 1, 1, 1)
    k = k.repeat(4, 1, 1, 1)
    policy = _seed_only_policy(q, k, min_batch=8)

    reasons = policy.mismatch_reasons(q, k, min_kv_len=1, layer_id=9)

    assert "batch_below_min" in reasons
    assert "layer_mismatch" in reasons
    assert not policy.matches_tensors(q, k, min_kv_len=1, layer_id=9)


def test_gate0_seed_only_batched_policy_accepts_matching_tensors():
    q, k, _ = _tensors(kv_len=128)
    q = q.repeat(4, 1, 1, 1)
    k = k.repeat(4, 1, 1, 1)
    policy = _seed_only_policy(q, k, min_batch=4)

    assert policy.mismatch_reasons(q, k, min_kv_len=1, layer_id=8) == []
    assert policy.matches_tensors(q, k, min_kv_len=1, layer_id=8)


def test_gate0_seed_only_batched_policy_reports_dtype_and_bucket_mismatch():
    q, k, _ = _tensors(kv_len=128)
    q = q.repeat(4, 1, 1, 1)
    k = k.repeat(4, 1, 1, 1)
    policy = Gate0SeedOnlyBatchedPolicy(
        model_id="test-model",
        layer_id=8,
        dtype="fp16",
        kv_len_bucket=32768,
        min_batch=4,
        heads=q.shape[2],
        kv_heads=k.shape[2],
        dim=q.shape[3],
    )

    reasons = policy.mismatch_reasons(q, k, min_kv_len=1)

    assert "dtype_mismatch" in reasons
    assert "kv_len_bucket_mismatch" in reasons


def test_seed_only_decode_service_fails_closed_when_backend_unavailable():
    q, k, v = _tensors(kv_len=128)
    q = q.repeat(4, 1, 1, 1)
    k = k.repeat(4, 1, 1, 1)
    v = v.repeat(4, 1, 1, 1)
    policy = _seed_only_policy(q, k, speedup=1.2, min_batch=4)
    service = StreamAttnSeedOnlyDecodeService(
        policy=policy,
        decode_policy=StreamAttnDecodePolicy(min_kv_len_for_gate0_seed_only=1),
        dense_fallback=lambda query, key, value: torch.zeros_like(query),
        dense_fallback_backend="flashinfer_dense",
    )

    out, info = service.run(q, k, v)

    torch.testing.assert_close(out, torch.zeros_like(q))
    assert info.backend_used == "flashinfer_dense"
    assert info.fallback_reason == "backend_unavailable"
    assert info.plan_backend == "dense"
    assert info.seed_only_enabled is False
    assert info.safety_policy_matched is True
    assert info.runtime_counters["fallback_reasons"] == {"backend_unavailable": 1}
    assert info.to_dict()["policy_id"] == "test-seed-only-batched"


def test_seed_only_decode_service_direct_plan_rejects_unavailable_backend():
    q, k, v = _tensors(kv_len=128)
    q = q.repeat(4, 1, 1, 1)
    k = k.repeat(4, 1, 1, 1)
    v = v.repeat(4, 1, 1, 1)
    policy = _seed_only_policy(q, k, speedup=1.2, min_batch=4)
    service = StreamAttnSeedOnlyDecodeService(
        policy=policy,
        decode_policy=StreamAttnDecodePolicy(min_kv_len_for_gate0_seed_only=1),
    )

    with pytest.raises(ValueError, match="backend_unavailable"):
        service.plan_direct_seed_only(q, k, v)


def test_seed_only_decode_service_defaults_to_exact_native():
    q, k, v = _tensors(kv_len=128)
    q = q.repeat(4, 1, 1, 1)
    k = k.repeat(4, 1, 1, 1)
    v = v.repeat(4, 1, 1, 1)
    policy = _seed_only_policy(q, k, speedup=1.2, min_batch=4)
    service = StreamAttnSeedOnlyDecodeService(
        policy=policy,
        decode_policy=StreamAttnDecodePolicy(min_kv_len_for_gate0_seed_only=1),
    )

    out, info = service.run(q, k, v)
    expected = dense_attention_forward(q, k, v, causal=False)

    torch.testing.assert_close(out, expected)
    assert info.backend_used == STREAMATTN_EXACT_NATIVE_BACKEND
    assert info.fallback_reason == "backend_unavailable"
    assert info.plan_backend == "dense"
    assert info.runtime_counters["backend_counts"] == {
        STREAMATTN_EXACT_NATIVE_BACKEND: 1
    }


def test_seed_only_decode_service_reports_policy_mismatch():
    q, k, v = _tensors(kv_len=128)
    q = q.repeat(4, 1, 1, 1)
    k = k.repeat(4, 1, 1, 1)
    v = v.repeat(4, 1, 1, 1)
    policy = _seed_only_policy(q, k, speedup=1.2, min_batch=4)
    service = StreamAttnSeedOnlyDecodeService(
        policy=policy,
        decode_policy=StreamAttnDecodePolicy(min_kv_len_for_gate0_seed_only=1),
        dense_fallback=lambda query, key, value: torch.ones_like(query),
        dense_fallback_backend="flashinfer_dense",
    )

    out, info = service.run(q, k, v, layer_id=9)

    torch.testing.assert_close(out, torch.ones_like(q))
    assert info.backend_used == "flashinfer_dense"
    assert info.fallback_reason == "layer_mismatch"
    assert info.safety_policy_matched is False


def test_seed_only_decode_service_manual_dense_mode():
    q, k, v = _tensors(kv_len=128)
    q = q.repeat(4, 1, 1, 1)
    k = k.repeat(4, 1, 1, 1)
    v = v.repeat(4, 1, 1, 1)
    policy = _seed_only_policy(q, k, speedup=1.2, min_batch=4)
    service = StreamAttnSeedOnlyDecodeService(
        policy=policy,
        decode_policy=StreamAttnDecodePolicy(min_kv_len_for_gate0_seed_only=1),
        dense_fallback=lambda query, key, value: torch.full_like(query, 2.0),
        dense_fallback_backend="flashinfer_dense",
    )

    out, info = service.run(q, k, v, mode="dense")

    torch.testing.assert_close(out, torch.full_like(q, 2.0))
    assert info.backend_used == "flashinfer_dense"
    assert info.fallback_reason == "manual_dense"
    assert info.safety_policy_matched is True


def test_seed_only_decode_service_manual_exact_native_mode():
    q, k, v = _tensors(kv_len=128)
    q = q.repeat(4, 1, 1, 1)
    k = k.repeat(4, 1, 1, 1)
    v = v.repeat(4, 1, 1, 1)
    policy = _seed_only_policy(q, k, speedup=1.2, min_batch=4)
    service = StreamAttnSeedOnlyDecodeService(
        policy=policy,
        decode_policy=StreamAttnDecodePolicy(min_kv_len_for_gate0_seed_only=1),
    )

    out, info = service.run(q, k, v, mode="dense")
    expected = stream_attn_exact_native_decode(q, k, v)

    torch.testing.assert_close(out, expected)
    assert info.backend_used == STREAMATTN_EXACT_NATIVE_BACKEND
    assert info.fallback_reason == "manual_dense"
    assert info.safety_policy_matched is True


def test_stream_attn_seed_only_decode_one_shot_helper():
    q, k, v = _tensors(kv_len=128)
    q = q.repeat(4, 1, 1, 1)
    k = k.repeat(4, 1, 1, 1)
    v = v.repeat(4, 1, 1, 1)
    policy = _seed_only_policy(q, k, speedup=1.2, min_batch=4)

    out, info = stream_attn_seed_only_decode(
        q,
        k,
        v,
        policy=policy,
        decode_policy=StreamAttnDecodePolicy(min_kv_len_for_gate0_seed_only=1),
        dense_fallback=lambda query, key, value: torch.zeros_like(query),
        dense_fallback_backend="flashinfer_dense",
    )

    torch.testing.assert_close(out, torch.zeros_like(q))
    assert info.policy_id == "test-seed-only-batched"
    assert info.backend_used == "flashinfer_dense"
    assert info.fallback_reason == "backend_unavailable"


def test_stream_attn_seed_only_decode_one_shot_defaults_to_exact_native():
    q, k, v = _tensors(kv_len=128)
    q = q.repeat(4, 1, 1, 1)
    k = k.repeat(4, 1, 1, 1)
    v = v.repeat(4, 1, 1, 1)
    policy = _seed_only_policy(q, k, speedup=1.2, min_batch=4)

    out, info = stream_attn_seed_only_decode(
        q,
        k,
        v,
        policy=policy,
        decode_policy=StreamAttnDecodePolicy(min_kv_len_for_gate0_seed_only=1),
    )
    expected = stream_attn_exact_native_decode(q, k, v)

    torch.testing.assert_close(out, expected)
    assert info.policy_id == "test-seed-only-batched"
    assert info.backend_used == STREAMATTN_EXACT_NATIVE_BACKEND
    assert info.fallback_reason == "backend_unavailable"
