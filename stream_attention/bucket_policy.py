"""Bucket-conditioned route policy helpers.

The Qwen3B stress runs showed that the 7-layer seed-only bundle is not globally
safe.  This module keeps that status explicit: product-strict routing falls back
to exact-native on adversarial stress buckets, while research mode can still
return the reduced route used for follow-up repair screens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


QWEN25_3B_FULL_SEED_LAYERS: Tuple[int, ...] = (0, 2, 14, 16, 24, 26, 27, 35)
QWEN25_3B_REDUCED_STRESS_SEED_LAYERS: Tuple[int, ...] = (0, 14, 16, 24, 35)

QWEN25_3B_POLICY_BY_LAYER: Dict[int, str] = {
    0: "qwen25_3b_l0_32k_seed_only_batched",
    2: "qwen25_3b_l2_s416_32k_seed_only_batched",
    14: "qwen25_3b_l14_32k_seed_only_batched",
    16: "qwen25_3b_l16_32k_seed_only_batched",
    24: "qwen25_3b_l24_32k_seed_only_batched",
    26: "qwen25_3b_l26_32k_seed_only_batched",
    27: "qwen25_3b_l27_32k_seed_only_batched",
    35: "qwen25_3b_l35_32k_seed_only_batched",
}

QWEN25_3B_STRESS_RISK_BUCKETS = frozenset(
    {
        "chat_instruction",
        "json_tool",
        "needle_rag",
        "noisy_neartie",
    }
)
QWEN25_3B_MARGIN_EXACT_BUCKETS = frozenset({"noisy_neartie"})
QWEN25_3B_VALIDATED_DEFAULT_BUCKETS = frozenset(
    {
        "needle",
        "code",
        "long_doc",
        "chat_doc",
        "default",
        "mixed_validated",
    }
)
QWEN25_3B_RISK_TIERS = frozenset({"bucket_only", "validated", "stress", "unknown"})


@dataclass(frozen=True)
class BucketRouteDecision:
    model_id: str
    prompt_bucket: str
    mode: str
    seed_only_layers: Tuple[int, ...]
    exact_layers: str
    policy_names: Tuple[str, ...]
    strict_gate_passed: bool
    status: str
    reason: str
    evidence: str
    risk_tier: str = "bucket_only"


def qwen25_3b_bucket_route_decision(
    prompt_bucket: Optional[str],
    *,
    product_strict: bool = True,
    risk_tier: Optional[str] = None,
) -> BucketRouteDecision:
    """Return the Qwen2.5-3B bucket-conditioned route decision.

    `product_strict=True` is the serving-safe selector: stress-risk buckets use
    exact-native because the stress gate failed.  `product_strict=False` returns
    the reduced stress route for research screens, with L26/L27 exact.
    """

    bucket = str(prompt_bucket or "default")
    tier = str(risk_tier or "bucket_only")
    if tier not in QWEN25_3B_RISK_TIERS:
        raise ValueError(f"unsupported Qwen2.5-3B route risk tier {tier!r}")
    if tier in {"stress", "unknown"}:
        return BucketRouteDecision(
            model_id="Qwen/Qwen2.5-3B-Instruct",
            prompt_bucket=bucket,
            mode="exact_native",
            seed_only_layers=(),
            exact_layers="all",
            policy_names=(),
            strict_gate_passed=True,
            status="product_exact_fallback",
            reason="risk_tier_requires_exact",
            evidence=(
                "32-step hybrid stress matrix: no tested L27/L26/L24 hybrid route "
                "passed the strict gate; coarse bucket names are insufficient"
            ),
            risk_tier=tier,
        )
    if tier == "validated" and bucket not in QWEN25_3B_VALIDATED_DEFAULT_BUCKETS:
        return BucketRouteDecision(
            model_id="Qwen/Qwen2.5-3B-Instruct",
            prompt_bucket=bucket,
            mode="exact_native",
            seed_only_layers=(),
            exact_layers="all",
            policy_names=(),
            strict_gate_passed=True,
            status="product_exact_fallback",
            reason="bucket_not_validated_for_validated_tier",
            evidence="validated risk tier can only use explicitly validated buckets",
            risk_tier=tier,
        )
    if bucket in QWEN25_3B_MARGIN_EXACT_BUCKETS:
        return BucketRouteDecision(
            model_id="Qwen/Qwen2.5-3B-Instruct",
            prompt_bucket=bucket,
            mode="exact_native",
            seed_only_layers=(),
            exact_layers="all",
            policy_names=(),
            strict_gate_passed=True,
            status="product_exact_fallback",
            reason="margin_sensitive_stress_bucket",
            evidence="minimal repair sweep: noisy_neartie still changed top1 under minus_l26_l27",
            risk_tier=tier,
        )
    if bucket in QWEN25_3B_STRESS_RISK_BUCKETS:
        if product_strict:
            return BucketRouteDecision(
                model_id="Qwen/Qwen2.5-3B-Instruct",
                prompt_bucket=bucket,
                mode="exact_native",
                seed_only_layers=(),
                exact_layers="all",
                policy_names=(),
                strict_gate_passed=True,
                status="product_exact_fallback",
                reason="stress_bucket_failed_strict_gate",
                evidence="32-step hybrid stress matrix: no tested hybrid route passed strict gate",
                risk_tier=tier,
            )
        layers = QWEN25_3B_REDUCED_STRESS_SEED_LAYERS
        return BucketRouteDecision(
            model_id="Qwen/Qwen2.5-3B-Instruct",
            prompt_bucket=bucket,
            mode="reduced_seed_only_bundle",
            seed_only_layers=layers,
            exact_layers="all non-listed layers plus L26/L27",
            policy_names=tuple(QWEN25_3B_POLICY_BY_LAYER[layer] for layer in layers),
            strict_gate_passed=False,
            status="research_only",
            reason="late_layers_exact_for_repair_screen",
            evidence=(
                "research-only reduced route; 32-step hybrid stress matrix confirmed "
                "that even L24/L26/L27 exact did not pass"
            ),
            risk_tier=tier,
        )

    if product_strict and bucket not in QWEN25_3B_VALIDATED_DEFAULT_BUCKETS:
        return BucketRouteDecision(
            model_id="Qwen/Qwen2.5-3B-Instruct",
            prompt_bucket=bucket,
            mode="exact_native",
            seed_only_layers=(),
            exact_layers="all",
            policy_names=(),
            strict_gate_passed=True,
            status="product_exact_fallback",
            reason="bucket_not_validated_by_stress_or_default_gate",
            evidence="unknown buckets fail closed until separately validated",
            risk_tier=tier,
        )

    layers = QWEN25_3B_FULL_SEED_LAYERS
    return BucketRouteDecision(
        model_id="Qwen/Qwen2.5-3B-Instruct",
        prompt_bucket=bucket,
        mode="seed_only_bundle",
        seed_only_layers=layers,
        exact_layers="all non-listed layers",
        policy_names=tuple(QWEN25_3B_POLICY_BY_LAYER[layer] for layer in layers),
        strict_gate_passed=bucket in QWEN25_3B_VALIDATED_DEFAULT_BUCKETS,
        status="product_candidate" if bucket in QWEN25_3B_VALIDATED_DEFAULT_BUCKETS else "unknown_bucket_exact_recommended",
        reason=(
            "validated_default_bucket"
            if bucket in QWEN25_3B_VALIDATED_DEFAULT_BUCKETS
            else "bucket_not_validated_by_stress_or_default_gate"
        ),
        evidence=(
            "Qwen3B L2-S416 8-layer route passed the validated-bucket 32-step gate; "
            "128-step long-horizon screen stayed top1/sample stable but exceeded max-KL/top-k strictness"
        ),
        risk_tier=tier,
    )


def qwen25_3b_policy_names_for_bucket(
    prompt_bucket: Optional[str],
    *,
    product_strict: bool = True,
    risk_tier: Optional[str] = None,
) -> Tuple[str, ...]:
    """Return packaged policy names for the selected bucket route."""

    return qwen25_3b_bucket_route_decision(
        prompt_bucket,
        product_strict=product_strict,
        risk_tier=risk_tier,
    ).policy_names
