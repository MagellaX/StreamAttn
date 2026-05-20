import torch
import pytest

from stream_attention.decode import (
    DecodeCostEntry,
    DecodeCostKey,
    DecodeCostModel,
    StreamAttnDecodePolicy,
    StreamAttnDecodeWorkspace,
    StreamAttnDecodeWrapper,
    decode_cost_model_from_profile_rows,
    stream_attn_decode_plan,
    stream_attn_decode_run,
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
