import pytest
import torch

from stream_attention.gate1 import (
    dense_attention_forward,
    make_route_request,
    stream_attn_gate1,
    summarize_gate1_raw_stats,
)
from stream_attention.router import CostEntry, CostKey, Gate1CostModel, StreamAttnPolicy, StreamAttnRouter
from stream_attention.telemetry import Prediction


def test_summarize_gate1_raw_stats():
    raw = torch.zeros(1, 2, 3, 6, dtype=torch.int32)
    raw[..., 0] = 1
    raw[..., 1] = 2
    raw[..., 2] = 3
    raw[..., 3] = 4
    raw[..., 4] = 5
    raw[..., 5] = 6

    stats = summarize_gate1_raw_stats(raw)

    assert stats.row_skips == 6
    assert stats.row_computes == 12
    assert stats.cta_tiles_total == 18
    assert stats.cta_pv_skipped == 24
    assert stats.cta_pv_executed == 30
    assert stats.force_mode_sum == 36
    assert stats.active_pv_fraction == pytest.approx(30 / 18)


def test_dense_attention_forward_matches_sdpa_layout():
    torch.manual_seed(0)
    q = torch.randn(1, 5, 2, 4)
    k = torch.randn(1, 5, 2, 4)
    v = torch.randn(1, 5, 2, 4)

    out = dense_attention_forward(q, k, v, causal=False)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3),
        k.permute(0, 2, 1, 3),
        v.permute(0, 2, 1, 3),
        is_causal=False,
    ).permute(0, 2, 1, 3)

    torch.testing.assert_close(out, ref)


def test_auto_runtime_falls_back_without_router():
    q = torch.randn(1, 4, 1, 4)
    k = torch.randn(1, 4, 1, 4)
    v = torch.randn(1, 4, 1, 4)

    out, info = stream_attn_gate1(
        q,
        k,
        v,
        causal=False,
        mode="auto",
        return_info=True,
    )

    assert info.decision.backend == "dense"
    assert info.decision.reason == "missing_router"
    assert info.stats is None
    torch.testing.assert_close(out, dense_attention_forward(q, k, v, causal=False))


def test_auto_runtime_falls_back_without_value_metadata():
    q = torch.randn(1, 4, 1, 4)
    k = torch.randn(1, 4, 1, 4)
    v = torch.randn(1, 4, 1, 4)
    request = make_route_request(q, k, causal=False, block_size=4, tile_size_q=4)
    cost_model = Gate1CostModel()
    cost_model.update(
        CostKey.from_request(request),
        CostEntry(dense_ms=0.10, qk_only_ms=0.05),
    )
    router = StreamAttnRouter(
        policy=StreamAttnPolicy(min_confidence=0.0, history_min_observations=1),
        cost_model=cost_model,
    )
    router.observe(request, cta_pv_executed=1, cta_tiles_total=10)

    out, info = stream_attn_gate1(
        q,
        k,
        v,
        causal=False,
        mode="auto",
        router=router,
        request=request,
        skip_predicate="value_bound",
        return_info=True,
    )

    assert info.decision.backend == "dense"
    assert info.decision.reason == "missing_value_norm_bounds"
    torch.testing.assert_close(out, dense_attention_forward(q, k, v, causal=False))


def test_auto_runtime_uses_injected_prediction_for_router():
    q = torch.randn(1, 4, 1, 4)
    k = torch.randn(1, 4, 1, 4)
    request = make_route_request(q, k, causal=False, block_size=4, tile_size_q=4)
    cost_model = Gate1CostModel()
    cost_model.update(
        CostKey.from_request(request),
        CostEntry(dense_ms=0.10, qk_only_ms=0.05),
    )
    router = StreamAttnRouter(
        policy=StreamAttnPolicy(min_confidence=0.0),
        cost_model=cost_model,
    )

    decision = router.choose(
        request,
        prediction=Prediction(active_frac_hat=0.10, confidence=1.0, source="test"),
    )

    assert decision.backend == "gate1"
