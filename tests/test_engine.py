import pytest
import torch

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
