"""Hardware-aware router for dense versus Gate-1 attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .telemetry import ActiveFractionKey, ActiveFractionTelemetry, Prediction, seq_bucket


@dataclass(frozen=True)
class StreamAttnPolicy:
    """Conservative defaults for ``mode="auto"`` routing."""

    gate1_active_threshold: float = 0.30
    gate1_disable_threshold: float = 0.45
    min_confidence: float = 0.70
    probe_min_seq: int = 4096
    safety_margin: float = 1.10
    history_min_observations: int = 8

    def __post_init__(self) -> None:
        if not 0.0 <= self.gate1_active_threshold <= 1.0:
            raise ValueError("gate1_active_threshold must be in [0, 1]")
        if not 0.0 <= self.gate1_disable_threshold <= 1.0:
            raise ValueError("gate1_disable_threshold must be in [0, 1]")
        if self.gate1_disable_threshold < self.gate1_active_threshold:
            raise ValueError("disable threshold must be >= active threshold")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if self.safety_margin < 1.0:
            raise ValueError("safety_margin must be >= 1")


@dataclass(frozen=True)
class AttentionRouteRequest:
    """Shape and identity fields needed for a routing decision."""

    batch: int
    seq_q: int
    seq_k: int
    heads: int
    dim: int
    dtype: str
    device: str
    tile_size_q: int
    block_size: int
    causal: bool
    model_id: str = "default"
    layer_id: int = 0
    head_id: int = 0
    phase: str = "prefill"

    @property
    def seq_q_bucket(self) -> int:
        return seq_bucket(self.seq_q)

    @property
    def seq_k_bucket(self) -> int:
        return seq_bucket(self.seq_k)

    def active_key(self) -> ActiveFractionKey:
        return ActiveFractionKey(
            model_id=self.model_id,
            layer_id=self.layer_id,
            head_id=self.head_id,
            phase=self.phase,
            seq_bucket=self.seq_k_bucket,
            dtype=self.dtype,
            device=self.device,
            tile_size_q=self.tile_size_q,
            block_size=self.block_size,
            causal=self.causal,
        )


@dataclass(frozen=True)
class CostKey:
    device: str
    dtype: str
    seq_q_bucket: int
    seq_k_bucket: int
    heads: int
    dim: int
    tile_size_q: int
    block_size: int
    causal: bool

    @classmethod
    def from_request(cls, request: AttentionRouteRequest) -> "CostKey":
        return cls(
            device=request.device,
            dtype=request.dtype,
            seq_q_bucket=request.seq_q_bucket,
            seq_k_bucket=request.seq_k_bucket,
            heads=request.heads,
            dim=request.dim,
            tile_size_q=request.tile_size_q,
            block_size=request.block_size,
            causal=request.causal,
        )


@dataclass(frozen=True)
class CostEntry:
    """Measured cost model entry for one shape/device/tile config."""

    dense_ms: float
    qk_only_ms: float
    predicate_overhead_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.dense_ms <= 0.0:
            raise ValueError("dense_ms must be positive")
        if self.qk_only_ms <= 0.0:
            raise ValueError("qk_only_ms must be positive")
        if self.qk_only_ms > self.dense_ms:
            raise ValueError("qk_only_ms cannot exceed dense_ms")
        if self.predicate_overhead_ms < 0.0:
            raise ValueError("predicate_overhead_ms must be non-negative")

    @property
    def pv_ms(self) -> float:
        return max(0.0, self.dense_ms - self.qk_only_ms)

    def predict_gate1_ms(self, active_fraction: float) -> float:
        active_fraction = min(1.0, max(0.0, float(active_fraction)))
        return self.qk_only_ms + active_fraction * self.pv_ms + self.predicate_overhead_ms

    def profitable(self, active_fraction: float, *, safety_margin: float) -> bool:
        return self.predict_gate1_ms(active_fraction) * safety_margin < self.dense_ms


class Gate1CostModel:
    """Lookup table populated from profiler runs."""

    def __init__(self) -> None:
        self._entries: Dict[CostKey, CostEntry] = {}

    def update(self, key: CostKey, entry: CostEntry) -> None:
        self._entries[key] = entry

    def lookup(self, request: AttentionRouteRequest) -> Optional[CostEntry]:
        return self._entries.get(CostKey.from_request(request))


@dataclass(frozen=True)
class BackendDecision:
    backend: str
    reason: str
    prediction: Prediction
    dense_ms: Optional[float] = None
    predicted_gate1_ms: Optional[float] = None
    active_threshold: Optional[float] = None


class StreamAttnRouter:
    """Choose dense or Gate-1 using active-fraction telemetry and costs."""

    def __init__(
        self,
        *,
        policy: Optional[StreamAttnPolicy] = None,
        telemetry: Optional[ActiveFractionTelemetry] = None,
        cost_model: Optional[Gate1CostModel] = None,
    ) -> None:
        self.policy = policy or StreamAttnPolicy()
        self.telemetry = telemetry or ActiveFractionTelemetry(
            min_observations=self.policy.history_min_observations
        )
        self.cost_model = cost_model or Gate1CostModel()
        self._last_backend: Dict[ActiveFractionKey, str] = {}

    def observe(
        self,
        request: AttentionRouteRequest,
        *,
        cta_pv_executed: int,
        cta_tiles_total: int,
    ) -> float:
        active = self.telemetry.active_fraction_from_counts(
            cta_pv_executed=cta_pv_executed,
            cta_tiles_total=cta_tiles_total,
        )
        self.telemetry.update(request.active_key(), active)
        return active

    def choose(
        self,
        request: AttentionRouteRequest,
        *,
        prediction: Optional[Prediction] = None,
    ) -> BackendDecision:
        key = request.active_key()
        pred = prediction or self.telemetry.predict(key, use_p90=True)
        cost = self.cost_model.lookup(request)

        if pred.confidence < self.policy.min_confidence:
            self._last_backend[key] = "dense"
            return BackendDecision(
                backend="dense",
                reason="low_confidence",
                prediction=pred,
                dense_ms=cost.dense_ms if cost else None,
                predicted_gate1_ms=(
                    cost.predict_gate1_ms(pred.active_frac_hat) if cost else None
                ),
            )

        last_backend = self._last_backend.get(key, "dense")
        active_threshold = (
            self.policy.gate1_disable_threshold
            if last_backend == "gate1"
            else self.policy.gate1_active_threshold
        )
        if pred.active_frac_hat > active_threshold:
            self._last_backend[key] = "dense"
            return BackendDecision(
                backend="dense",
                reason="active_fraction_above_threshold",
                prediction=pred,
                dense_ms=cost.dense_ms if cost else None,
                predicted_gate1_ms=(
                    cost.predict_gate1_ms(pred.active_frac_hat) if cost else None
                ),
                active_threshold=active_threshold,
            )

        if cost is not None:
            predicted_gate1 = cost.predict_gate1_ms(pred.active_frac_hat)
            if not cost.profitable(
                pred.active_frac_hat,
                safety_margin=self.policy.safety_margin,
            ):
                self._last_backend[key] = "dense"
                return BackendDecision(
                    backend="dense",
                    reason="not_profitable_with_margin",
                    prediction=pred,
                    dense_ms=cost.dense_ms,
                    predicted_gate1_ms=predicted_gate1,
                    active_threshold=active_threshold,
                )
            self._last_backend[key] = "gate1"
            return BackendDecision(
                backend="gate1",
                reason="profitable_history",
                prediction=pred,
                dense_ms=cost.dense_ms,
                predicted_gate1_ms=predicted_gate1,
                active_threshold=active_threshold,
            )

        self._last_backend[key] = "gate1"
        return BackendDecision(
            backend="gate1",
            reason="below_threshold_no_cost_model",
            prediction=pred,
            active_threshold=active_threshold,
        )


def router_regret(
    *,
    dense_ms: float,
    gate1_ms: float,
    chosen_backend: str,
) -> Tuple[float, float]:
    """Return absolute and relative regret versus the oracle backend."""

    oracle = min(dense_ms, gate1_ms)
    chosen = gate1_ms if chosen_backend == "gate1" else dense_ms
    regret = chosen - oracle
    return regret, regret / oracle if oracle > 0.0 else 0.0
