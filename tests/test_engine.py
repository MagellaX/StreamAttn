import pytest
import torch
from types import SimpleNamespace

import stream_attention as stream_attn
from stream_attention.decode import (
    STREAMATTN_EXACT_NATIVE_BACKEND,
    STREAMATTN_MODE_EXACT_NATIVE,
    STREAMATTN_MODE_SEED_ONLY_NATIVE,
    STREAMATTN_MODE_VERIFIED_AUTO,
    Gate0SeedOnlyBatchedPolicy,
    StreamAttnDecodePolicy,
)
from stream_attention.engine import StreamAttnEngine
from stream_attention.gate1 import dense_attention_forward
from stream_attention.kernels.gate0_exact_refresh_triton import TRITON_AVAILABLE
from stream_attention.planning import (
    ATTENTION_GUARANTEE_DISTRIBUTION_VERIFIED,
    ATTENTION_GUARANTEE_EXACT,
    ATTENTION_SCHEDULE_ALL,
    ATTENTION_SCHEDULE_SELECTED,
    ATTENTION_ROUTE_GRANULARITY_Q_HEAD,
    AttentionProblem,
    AttentionTilePlan,
)
from stream_attention.paged import (
    PagedDynamicSelectedDecodePlan,
    PagedKVCache,
    PagedQuerySelectedDecodePlan,
    PagedSelectedDecodePlan,
)


def _tensors():
    q = torch.randn(2, 1, 4, 8)
    k = torch.randn(2, 32, 2, 8)
    v = torch.randn_like(k)
    return q, k, v


def _policy(q, k):
    return Gate0SeedOnlyBatchedPolicy(
        policy_id="engine-test",
        model_id="test-model",
        layer_id=3,
        dtype="fp32",
        kv_len_bucket=32,
        min_batch=2,
        heads=q.shape[2],
        kv_heads=k.shape[2],
        dim=q.shape[3],
        block_size=8,
        sink_blocks=1,
        recent_blocks=1,
        middle_seed_blocks=1,
        expected_dense_ms=1.0,
        expected_seed_only_ms=0.5,
        expected_speedup_vs_dense=2.0,
    )


def test_engine_exact_native_plan_and_public_decode_alias():
    q, k, v = _tensors()
    engine = StreamAttnEngine(
        policy=_policy(q, k),
        decode_policy=StreamAttnDecodePolicy(min_kv_len_for_gate0_seed_only=1),
    )

    plan = engine.plan(q, k, v, mode=STREAMATTN_MODE_EXACT_NATIVE)
    out, info = plan.run()
    alias_out, alias_info = stream_attn.decode(
        q,
        k,
        v,
        mode=STREAMATTN_MODE_EXACT_NATIVE,
        engine=engine,
    )

    torch.testing.assert_close(out, alias_out)
    assert plan.backend == STREAMATTN_EXACT_NATIVE_BACKEND
    assert plan.uses_fixed_buffers is False
    assert plan.attention_problem.guarantee == ATTENTION_GUARANTEE_EXACT
    assert plan.tile_plan.schedule.kind == ATTENTION_SCHEDULE_ALL
    assert plan.backend_plan.backend == STREAMATTN_EXACT_NATIVE_BACKEND
    assert plan.summary()["tile_plan"]["schedule"]["kind"] == ATTENTION_SCHEDULE_ALL
    assert info.backend_used == STREAMATTN_EXACT_NATIVE_BACKEND
    assert alias_info.backend_used == STREAMATTN_EXACT_NATIVE_BACKEND


def test_verified_auto_fails_closed_when_native_backend_is_unavailable():
    q, k, v = _tensors()
    engine = StreamAttnEngine(
        policy=_policy(q, k),
        decode_policy=StreamAttnDecodePolicy(min_kv_len_for_gate0_seed_only=1),
    )

    plan = engine.plan(
        q,
        k,
        v,
        mode=STREAMATTN_MODE_VERIFIED_AUTO,
        model_id="test-model",
        layer_id=3,
    )
    _, info = plan.run()

    assert plan.backend == STREAMATTN_EXACT_NATIVE_BACKEND
    assert "backend_unavailable" in plan.reason
    assert plan.tile_plan.schedule.kind == ATTENTION_SCHEDULE_ALL
    assert info.backend_used == STREAMATTN_EXACT_NATIVE_BACKEND


def test_explicit_seed_only_native_rejects_unavailable_backend():
    q, k, v = _tensors()
    engine = StreamAttnEngine(
        policy=_policy(q, k),
        decode_policy=StreamAttnDecodePolicy(min_kv_len_for_gate0_seed_only=1),
    )

    with pytest.raises(ValueError, match="backend_unavailable"):
        engine.plan(
            q,
            k,
            v,
            mode=STREAMATTN_MODE_SEED_ONLY_NATIVE,
            model_id="test-model",
            layer_id=3,
        )


def test_engine_lowers_verified_policy_to_selected_logical_tiles(monkeypatch):
    q, k, v = _tensors()
    policy = _policy(q, k)
    engine = StreamAttnEngine(
        policy=policy,
        decode_policy=StreamAttnDecodePolicy(min_kv_len_for_gate0_seed_only=1),
    )

    class _Runner:
        def __init__(self):
            self.policy = policy

        def run(self):  # pragma: no cover - planning test only
            raise AssertionError("planning test must not execute the backend")

    monkeypatch.setattr(engine.service, "plan_direct_seed_only", lambda *args, **kwargs: _Runner())
    plan = engine.plan(
        q,
        k,
        v,
        mode=STREAMATTN_MODE_VERIFIED_AUTO,
        model_id="test-model",
        layer_id=3,
    )

    assert plan.attention_problem.guarantee == ATTENTION_GUARANTEE_DISTRIBUTION_VERIFIED
    assert plan.tile_plan.schedule.kind == ATTENTION_SCHEDULE_SELECTED
    assert plan.tile_plan.schedule.selected_tile_ids == ((0, 2, 3), (0, 2, 3))
    assert plan.tile_plan.tile_coverage == 0.75
    assert plan.backend_plan.backend == "gate0_seed_only_triton"


def test_engine_binds_selected_paged_tile_plan_to_native_runner(monkeypatch):
    query = torch.randn(1, 1, 8, 8)
    cache = PagedKVCache(
        key=torch.randn(4, 16, 1, 8),
        value=torch.randn(4, 16, 1, 8),
        page_table=torch.arange(4, dtype=torch.int32).view(1, 4),
        sequence_lengths=torch.tensor([64], dtype=torch.int32),
        layout="NHD",
    )
    problem = AttentionProblem.from_paged(
        query,
        cache,
        guarantee=ATTENTION_GUARANTEE_DISTRIBUTION_VERIFIED,
    )
    tile_plan = AttentionTilePlan.selected(
        problem,
        logical_tile_size=16,
        tile_ids_per_row=((0, 1, 2, 3),),
        policy_id="engine-selected-paged-test",
        reason="verified_test_schedule",
        schedule_epoch=2,
    )

    def fake_build(_query, _cache, routes, **_kwargs):
        return SimpleNamespace(
            backend="fake_sm90_selected",
            routes=routes,
            max_routes_per_row=1,
            workspace_bytes=4096,
            run=lambda: query,
        )

    monkeypatch.setattr(PagedSelectedDecodePlan, "build", staticmethod(fake_build))
    engine = StreamAttnEngine(policy=_policy(*_tensors()[:2]))
    plan = engine.plan_selected_paged(query, cache, tile_plan)
    output, info = plan.run()

    assert output is query
    assert plan.tile_plan is tile_plan
    assert plan.backend == "fake_sm90_selected"
    assert plan.uses_fixed_buffers is True
    assert plan.backend_plan.workspace_bytes == 4096
    assert info.policy_id == "engine-selected-paged-test"
    assert info.stats["route_count"] == 1
    assert info.stats["max_routes_per_row"] == 1


def test_engine_binds_mutable_qhead_routes_to_dynamic_native_runner(monkeypatch):
    query = torch.randn(1, 1, 8, 8)
    cache = PagedKVCache(
        key=torch.randn(8, 16, 1, 8),
        value=torch.randn(8, 16, 1, 8),
        page_table=torch.arange(8, dtype=torch.int32).view(1, 8),
        sequence_lengths=torch.tensor([128], dtype=torch.int32),
        layout="NHD",
    )
    problem = AttentionProblem.from_paged(
        query,
        cache,
        guarantee=ATTENTION_GUARANTEE_DISTRIBUTION_VERIFIED,
    )
    tile_plan = AttentionTilePlan.selected(
        problem,
        logical_tile_size=64,
        tile_ids_per_row=tuple((head & 1,) for head in range(8)),
        policy_id="engine-dynamic-selected-test",
        reason="gpu_query_selector",
        route_granularity=ATTENTION_ROUTE_GRANULARITY_Q_HEAD,
        schedule_epoch=3,
    ).with_device_routes(device="cpu")

    def fake_build(_query, _cache, routes, **_kwargs):
        return SimpleNamespace(
            backend="fake_sm90_dynamic_selected",
            routes=routes,
            max_routes_per_group=8,
            producer_ctas=8,
            metadata_bytes=1024,
            workspace_bytes=4096,
            run=lambda: query,
        )

    monkeypatch.setattr(
        PagedDynamicSelectedDecodePlan,
        "build",
        staticmethod(fake_build),
    )
    engine = StreamAttnEngine(policy=_policy(*_tensors()[:2]))
    plan = engine.plan_dynamic_selected_paged(query, cache, tile_plan)
    output, info = plan.run()

    assert output is query
    assert plan.tile_plan is tile_plan
    assert plan.backend == "fake_sm90_dynamic_selected"
    assert info.policy_id == "engine-dynamic-selected-test"
    assert info.stats["route_preparation"] == (
        "gpu_bounded_membership_warp_compaction_no_host_readback"
    )
    assert info.stats["max_routes_per_group"] == 8


def test_engine_query_selected_plan_exposes_research_safety_scope(monkeypatch):
    query = torch.randn(1, 1, 8, 8)
    cache = PagedKVCache(
        key=torch.randn(8, 16, 1, 8),
        value=torch.randn(8, 16, 1, 8),
        page_table=torch.arange(8, dtype=torch.int32).view(1, 8),
        sequence_lengths=torch.tensor([128], dtype=torch.int32),
        layout="NHD",
    )

    def fake_build(_query, _cache, **_kwargs):
        return SimpleNamespace(
            backend="fake_sm90_query_selected",
            support_metadata_bytes=2048,
            selector_workspace_bytes=512,
            selected_plan=SimpleNamespace(
                workspace_bytes=4096,
                max_routes_per_group=48,
            ),
            run=lambda: query,
        )

    monkeypatch.setattr(
        PagedQuerySelectedDecodePlan,
        "build",
        staticmethod(fake_build),
    )
    engine = StreamAttnEngine(policy=_policy(*_tensors()[:2]))
    plan = engine.plan_query_selected_paged(
        query,
        cache,
        selected_atoms=2,
        support_width=4,
    )
    output, info = plan.run()

    assert output is query
    assert plan.backend == "fake_sm90_query_selected"
    assert plan.backend_plan.workspace_bytes == 4608
    assert info.stats["support_width"] == 4
    assert info.stats["safety_scope"] == "caller_distribution_verified"


@pytest.mark.skipif(
    not torch.cuda.is_available() or not TRITON_AVAILABLE,
    reason="requires CUDA and Triton",
)
def test_exact_native_cuda_plan_reuses_buffers_and_tracks_mutation():
    q = torch.randn(2, 1, 4, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 64, 2, 64, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    engine = StreamAttnEngine(policy=_policy(q, k))

    plan = engine.plan(q, k, v, mode=STREAMATTN_MODE_EXACT_NATIVE)
    out, info = plan.run()
    expected = dense_attention_forward(q, k, v, causal=False)
    first = out.clone()
    output_ptr = out.data_ptr()

    torch.testing.assert_close(out, expected, atol=5e-3, rtol=5e-3)
    assert plan.uses_fixed_buffers is True
    assert info.backend_used == STREAMATTN_EXACT_NATIVE_BACKEND

    q.add_(0.5)
    second = plan.run(return_info=False)
    assert second.data_ptr() == output_ptr
    assert not torch.equal(first, second)
