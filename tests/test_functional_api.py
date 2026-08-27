import pytest
import torch
import torch.nn.functional as F

import stream_attention as stream_attn
from stream_attention.core.fused_online_attention import TRITON_AVAILABLE
from stream_attention.planning import (
    ATTENTION_GUARANTEE_EXACT,
    ATTENTION_PHASE_PREFILL,
    ATTENTION_PHASE_TRAIN,
    ATTENTION_SCHEDULE_ALL,
    AttentionProblem,
)


def _sdpa_reference(q, k, v, *, causal):
    qh = q.transpose(1, 2)
    kh = k.transpose(1, 2)
    vh = v.transpose(1, 2)
    return F.scaled_dot_product_attention(
        qh,
        kh,
        vh,
        is_causal=causal,
        dropout_p=0.0,
    ).transpose(1, 2).contiguous()


def test_attention_problem_supports_prefill_and_train_phases():
    q = torch.randn(2, 7, 4, 16)
    k = torch.randn(2, 11, 2, 16)
    v = torch.randn_like(k)

    prefill = AttentionProblem.from_qkv(
        q,
        k,
        v,
        phase=ATTENTION_PHASE_PREFILL,
    )
    training = AttentionProblem.from_qkv(
        q,
        k,
        v,
        phase=ATTENTION_PHASE_TRAIN,
        mask="custom",
    )

    assert prefill.phase == ATTENTION_PHASE_PREFILL
    assert prefill.guarantee == ATTENTION_GUARANTEE_EXACT
    assert prefill.query_len == 7
    assert prefill.group_size == 2
    assert prefill.kv_lengths == (11, 11)
    assert training.phase == ATTENTION_PHASE_TRAIN
    assert training.mask == "custom"


def test_public_prefill_matches_sdpa_and_reports_shared_plan():
    torch.manual_seed(7)
    q = torch.randn(2, 9, 4, 16)
    k = torch.randn(2, 9, 4, 16)
    v = torch.randn(2, 9, 4, 16)

    output, info = stream_attn.prefill(q, k, v, return_info=True)
    expected = _sdpa_reference(q, k, v, causal=True)

    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    assert info.phase == ATTENTION_PHASE_PREFILL
    assert info.backend_used == "torch_sdpa"
    assert info.attention_problem.guarantee == ATTENTION_GUARANTEE_EXACT
    assert info.tile_plan.schedule.kind == ATTENTION_SCHEDULE_ALL
    assert info.tile_plan.tile_coverage == 1.0


def test_public_train_matches_sdpa_forward_and_gradients():
    torch.manual_seed(11)
    q = torch.randn(1, 8, 2, 8, requires_grad=True)
    k = torch.randn(1, 8, 2, 8, requires_grad=True)
    v = torch.randn(1, 8, 2, 8, requires_grad=True)

    output, info = stream_attn.train(q, k, v, return_info=True)
    grad = torch.randn_like(output)
    output.backward(grad)
    actual_grads = (q.grad.clone(), k.grad.clone(), v.grad.clone())

    qr = q.detach().clone().requires_grad_(True)
    kr = k.detach().clone().requires_grad_(True)
    vr = v.detach().clone().requires_grad_(True)
    expected = _sdpa_reference(qr, kr, vr, causal=True)
    expected.backward(grad)

    torch.testing.assert_close(output.detach(), expected.detach(), rtol=1e-5, atol=1e-5)
    for actual, reference in zip(actual_grads, (qr.grad, kr.grad, vr.grad), strict=True):
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)
    assert info.phase == ATTENTION_PHASE_TRAIN
    assert info.backend_used == "torch_sdpa"


def test_engine_plans_prefill_and_train_through_same_contract():
    engine = stream_attn.StreamAttnEngine()
    q = torch.randn(1, 5, 2, 8)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    prefill = engine.plan_prefill(q, k, v)
    training = engine.plan_train(q, k, v)

    assert prefill.attention_problem.phase == ATTENTION_PHASE_PREFILL
    assert training.attention_problem.phase == ATTENTION_PHASE_TRAIN
    assert prefill.tile_plan.schedule.kind == ATTENTION_SCHEDULE_ALL
    assert training.tile_plan.schedule.kind == ATTENTION_SCHEDULE_ALL
    assert prefill.backend_plan.backend == "torch_sdpa"
    assert training.backend_plan.backend == "torch_sdpa"
    assert engine._service is None


def test_functional_api_rejects_unlowered_gqa_and_inference_dropout():
    q = torch.randn(1, 5, 4, 8)
    k = torch.randn(1, 5, 2, 8)
    v = torch.randn_like(k)

    with pytest.raises(ValueError, match="GQA prefill lowering is not implemented"):
        stream_attn.prefill(q, k, v)

    k_mha = torch.randn_like(q)
    v_mha = torch.randn_like(q)
    with pytest.raises(ValueError, match="inference-only"):
        stream_attn.prefill(q, k_mha, v_mha, dropout_p=0.1)

    q_grad = q.detach().clone().requires_grad_(True)
    with pytest.raises(ValueError, match="use torch.no_grad"):
        stream_attn.prefill(q_grad, k_mha, v_mha)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not TRITON_AVAILABLE,
    reason="CUDA and Triton are required for the native functional API gate",
)
@pytest.mark.parametrize(
    ("dtype", "atol"),
    ((torch.float32, 1e-2), (torch.bfloat16, 2e-2)),
)
def test_native_prefill_and_train_match_sdpa_on_cuda(dtype, atol):
    torch.manual_seed(17)
    shape = (1, 128, 4, 64)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    with torch.no_grad():
        output, prefill_info = stream_attn.prefill(q, k, v, return_info=True)
        expected = _sdpa_reference(q, k, v, causal=True)
    torch.testing.assert_close(output, expected, rtol=atol, atol=atol)
    assert prefill_info.backend_used == "triton_online_softmax"

    qt = q.detach().clone().requires_grad_(True)
    kt = k.detach().clone().requires_grad_(True)
    vt = v.detach().clone().requires_grad_(True)
    train_output, train_info = stream_attn.train(qt, kt, vt, return_info=True)
    grad = torch.randn_like(train_output)
    train_output.backward(grad)

    qr = q.detach().clone().requires_grad_(True)
    kr = k.detach().clone().requires_grad_(True)
    vr = v.detach().clone().requires_grad_(True)
    reference = _sdpa_reference(qr, kr, vr, causal=True)
    reference.backward(grad)

    torch.testing.assert_close(train_output, reference, rtol=atol, atol=atol)
    torch.testing.assert_close(qt.grad, qr.grad, rtol=atol, atol=atol)
    torch.testing.assert_close(kt.grad, kr.grad, rtol=atol, atol=atol)
    torch.testing.assert_close(vt.grad, vr.grad, rtol=atol, atol=atol)
    assert train_info.backend_used == "triton_online_softmax_autograd"
