"""Profile router decisions against dense, Gate-1, and oracle choices.

This is a lightweight offline benchmark for the routing rule. It uses measured
or supplied ``dense_ms`` and ``qk_only_ms`` values plus a list of active PV
fractions, then reports router regret relative to the oracle backend.
"""

import argparse
import json

from stream_attention.router import (
    AttentionRouteRequest,
    CostEntry,
    CostKey,
    Gate1CostModel,
    StreamAttnPolicy,
    StreamAttnRouter,
    router_regret,
)
from stream_attention.telemetry import Prediction


def _parse_floats(raw: str):
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-fracs", default="0.0625,0.125,0.25,0.5,0.75,1.0")
    parser.add_argument("--dense-ms", type=float, default=0.095)
    parser.add_argument("--qk-only-ms", type=float, default=0.050)
    parser.add_argument("--predicate-overhead-ms", type=float, default=0.0)
    parser.add_argument("--safety-margin", type=float, default=1.10)
    parser.add_argument("--gate1-active-threshold", type=float, default=0.30)
    parser.add_argument("--gate1-disable-threshold", type=float, default=0.45)
    parser.add_argument("--confidence", type=float, default=1.0)
    parser.add_argument("--seq-q", type=int, default=1024)
    parser.add_argument("--seq-k", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--device", default="A10G")
    parser.add_argument("--tile-size-q", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--causal", action="store_true")
    args = parser.parse_args()

    request = AttentionRouteRequest(
        batch=1,
        seq_q=args.seq_q,
        seq_k=args.seq_k,
        heads=args.heads,
        dim=args.dim,
        dtype=args.dtype,
        device=args.device,
        tile_size_q=args.tile_size_q,
        block_size=args.block_size,
        causal=args.causal,
    )
    cost_entry = CostEntry(
        dense_ms=args.dense_ms,
        qk_only_ms=args.qk_only_ms,
        predicate_overhead_ms=args.predicate_overhead_ms,
    )
    cost_model = Gate1CostModel()
    cost_model.update(CostKey.from_request(request), cost_entry)
    router = StreamAttnRouter(
        policy=StreamAttnPolicy(
            gate1_active_threshold=args.gate1_active_threshold,
            gate1_disable_threshold=args.gate1_disable_threshold,
            safety_margin=args.safety_margin,
            min_confidence=0.0,
        ),
        cost_model=cost_model,
    )

    rows = []
    totals = {
        "dense_always_ms": 0.0,
        "gate1_always_ms": 0.0,
        "router_ms": 0.0,
        "oracle_ms": 0.0,
        "router_regret_ms": 0.0,
    }

    for active_fraction in _parse_floats(args.active_fracs):
        gate1_ms = cost_entry.predict_gate1_ms(active_fraction)
        decision = router.choose(
            request,
            prediction=Prediction(
                active_frac_hat=active_fraction,
                confidence=args.confidence,
                source="profile",
            ),
        )
        chosen_ms = gate1_ms if decision.backend == "gate1" else args.dense_ms
        oracle_ms = min(args.dense_ms, gate1_ms)
        regret_ms, regret_relative = router_regret(
            dense_ms=args.dense_ms,
            gate1_ms=gate1_ms,
            chosen_backend=decision.backend,
        )
        row = {
            "active_pv_fraction": active_fraction,
            "dense_ms": args.dense_ms,
            "gate1_ms": gate1_ms,
            "oracle_backend": "gate1" if gate1_ms < args.dense_ms else "dense",
            "router_backend": decision.backend,
            "router_reason": decision.reason,
            "router_ms": chosen_ms,
            "oracle_ms": oracle_ms,
            "regret_ms": regret_ms,
            "regret_relative": regret_relative,
        }
        rows.append(row)
        totals["dense_always_ms"] += args.dense_ms
        totals["gate1_always_ms"] += gate1_ms
        totals["router_ms"] += chosen_ms
        totals["oracle_ms"] += oracle_ms
        totals["router_regret_ms"] += regret_ms

    totals["router_regret_relative"] = (
        totals["router_regret_ms"] / totals["oracle_ms"]
        if totals["oracle_ms"] > 0.0
        else 0.0
    )

    print(
        json.dumps(
            {
                "cost_entry": {
                    "dense_ms": cost_entry.dense_ms,
                    "qk_only_ms": cost_entry.qk_only_ms,
                    "pv_ms": cost_entry.pv_ms,
                    "predicate_overhead_ms": cost_entry.predicate_overhead_ms,
                },
                "policy": {
                    "gate1_active_threshold": args.gate1_active_threshold,
                    "gate1_disable_threshold": args.gate1_disable_threshold,
                    "safety_margin": args.safety_margin,
                },
                "rows": rows,
                "totals": totals,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
