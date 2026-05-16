import pytest

from stream_attention.router import (
    AttentionRouteRequest,
    CostEntry,
    CostKey,
    Gate1CostModel,
    StreamAttnPolicy,
    StreamAttnRouter,
    router_regret,
)
from stream_attention.telemetry import ActiveFractionTelemetry, Prediction


def _request():
    return AttentionRouteRequest(
        batch=1,
        seq_q=1024,
        seq_k=1024,
        heads=4,
        dim=64,
        dtype="fp16",
        device="A10G",
        tile_size_q=64,
        block_size=64,
        causal=False,
        model_id="test",
        layer_id=3,
        head_id=2,
    )


def _router_with_cost():
    req = _request()
    cost_model = Gate1CostModel()
    cost_model.update(
        CostKey.from_request(req),
        CostEntry(dense_ms=0.10, qk_only_ms=0.05),
    )
    router = StreamAttnRouter(
        policy=StreamAttnPolicy(min_confidence=0.7, safety_margin=1.05),
        telemetry=ActiveFractionTelemetry(min_observations=4),
        cost_model=cost_model,
    )
    return router, req


def test_router_falls_back_to_dense_when_history_is_not_confident():
    router, req = _router_with_cost()

    decision = router.choose(req)

    assert decision.backend == "dense"
    assert decision.reason == "low_confidence"


def test_router_uses_confident_sparse_history_when_profitable():
    router, req = _router_with_cost()
    for _ in range(4):
        router.observe(req, cta_pv_executed=8, cta_tiles_total=100)

    decision = router.choose(req)

    assert decision.backend == "gate1"
    assert decision.reason == "profitable_history"
    assert decision.prediction.active_frac_hat == 0.08
    assert decision.predicted_gate1_ms < decision.dense_ms


def test_router_rejects_gate1_when_cost_model_margin_fails():
    req = _request()
    cost_model = Gate1CostModel()
    cost_model.update(
        CostKey.from_request(req),
        CostEntry(dense_ms=0.10, qk_only_ms=0.09),
    )
    router = StreamAttnRouter(
        policy=StreamAttnPolicy(min_confidence=0.7, safety_margin=1.10),
        telemetry=ActiveFractionTelemetry(min_observations=4),
        cost_model=cost_model,
    )
    for _ in range(4):
        router.observe(req, cta_pv_executed=10, cta_tiles_total=100)

    decision = router.choose(req)

    assert decision.backend == "dense"
    assert decision.reason == "not_profitable_with_margin"


def test_router_hysteresis_keeps_gate1_until_disable_threshold():
    router, req = _router_with_cost()

    first = router.choose(
        req,
        prediction=Prediction(
            active_frac_hat=0.25,
            confidence=1.0,
            source="test",
        ),
    )
    second = router.choose(
        req,
        prediction=Prediction(
            active_frac_hat=0.40,
            confidence=1.0,
            source="test",
        ),
    )
    third = router.choose(
        req,
        prediction=Prediction(
            active_frac_hat=0.46,
            confidence=1.0,
            source="test",
        ),
    )

    assert first.backend == "gate1"
    assert second.backend == "gate1"
    assert third.backend == "dense"


def test_router_regret_reports_oracle_gap():
    regret, relative = router_regret(
        dense_ms=0.10,
        gate1_ms=0.06,
        chosen_backend="dense",
    )

    assert regret == pytest.approx(0.04)
    assert relative == pytest.approx(0.04 / 0.06)
