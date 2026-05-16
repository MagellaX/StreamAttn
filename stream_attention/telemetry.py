"""Telemetry primitives for routing sparse attention backends."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional


def _clamp_fraction(value: float) -> float:
    if math.isnan(value):
        raise ValueError("active fraction cannot be NaN")
    return min(1.0, max(0.0, float(value)))


def seq_bucket(seq_len: int) -> int:
    """Return a power-of-two sequence bucket."""

    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    return 1 << (seq_len - 1).bit_length()


@dataclass(frozen=True)
class ActiveFractionKey:
    """Key for per-model/layer/head active-PV telemetry."""

    model_id: str
    layer_id: int
    head_id: int
    kv_head_id: int
    q_group_id: int
    phase: str
    seq_bucket: int
    dtype: str
    device: str
    device_class: str
    tile_size_q: int
    block_size: int
    causal: bool


@dataclass(frozen=True)
class Prediction:
    """Predicted active PV fraction for a router decision."""

    active_frac_hat: float
    confidence: float
    source: str
    upper_bound: Optional[float] = None

    @property
    def confident(self) -> bool:
        return self.confidence > 0.0


@dataclass
class RunningActiveFraction:
    """EWMA plus recent-window p90 for active PV fraction."""

    alpha: float = 0.1
    window_size: int = 64
    count: int = 0
    ewma: float = 1.0
    ewma_square: float = 1.0
    recent: Deque[float] = field(default_factory=deque)

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")

    def update(self, active_fraction: float) -> None:
        value = _clamp_fraction(active_fraction)
        if self.count == 0:
            self.ewma = value
            self.ewma_square = value * value
        else:
            self.ewma = (1.0 - self.alpha) * self.ewma + self.alpha * value
            self.ewma_square = (
                (1.0 - self.alpha) * self.ewma_square + self.alpha * value * value
            )
        self.count += 1
        self.recent.append(value)
        while len(self.recent) > self.window_size:
            self.recent.popleft()

    @property
    def variance(self) -> float:
        return max(0.0, self.ewma_square - self.ewma * self.ewma)

    @property
    def p90(self) -> float:
        if not self.recent:
            return 1.0
        values = sorted(self.recent)
        idx = max(0, math.ceil(0.9 * len(values)) - 1)
        return values[idx]

    def confidence(self, *, min_observations: int = 8) -> float:
        if self.count == 0:
            return 0.0
        count_conf = min(1.0, self.count / max(1, min_observations))
        stability = max(0.0, 1.0 - 2.0 * math.sqrt(self.variance))
        return count_conf * stability


class ActiveFractionTelemetry:
    """Store and predict active-PV fractions from previous Gate-1 runs."""

    def __init__(
        self,
        *,
        alpha: float = 0.1,
        window_size: int = 64,
        min_observations: int = 8,
    ) -> None:
        self.alpha = alpha
        self.window_size = window_size
        self.min_observations = min_observations
        self._profiles: Dict[ActiveFractionKey, RunningActiveFraction] = {}

    def update(self, key: ActiveFractionKey, active_fraction: float) -> RunningActiveFraction:
        profile = self._profiles.get(key)
        if profile is None:
            profile = RunningActiveFraction(
                alpha=self.alpha,
                window_size=self.window_size,
            )
            self._profiles[key] = profile
        profile.update(active_fraction)
        return profile

    def get(self, key: ActiveFractionKey) -> Optional[RunningActiveFraction]:
        return self._profiles.get(key)

    def predict(self, key: ActiveFractionKey, *, use_p90: bool = True) -> Prediction:
        profile = self._profiles.get(key)
        if profile is None:
            return Prediction(
                active_frac_hat=1.0,
                confidence=0.0,
                source="history",
                upper_bound=None,
            )
        return Prediction(
            active_frac_hat=profile.p90 if use_p90 else profile.ewma,
            confidence=profile.confidence(min_observations=self.min_observations),
            source="history",
            upper_bound=profile.p90,
        )

    def active_fraction_from_counts(self, *, cta_pv_executed: int, cta_tiles_total: int) -> float:
        if cta_tiles_total <= 0:
            raise ValueError("cta_tiles_total must be positive")
        return _clamp_fraction(cta_pv_executed / cta_tiles_total)
