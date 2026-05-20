"""Plan higher-risk Gate-0 split-K frontier experiments from policy evidence."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Sequence, Tuple


SCHEMA = "streamattn.gate0.splitk_frontier_plan.v1"


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    return int(value)


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _entry_heads(entry: Dict[str, Any]) -> List[int]:
    heads = ((entry.get("runtime") or {}).get("head_indices") or [])
    if heads:
        return [int(item) for item in heads]
    raw = ((entry.get("runtime") or {}).get("head_group") or "")
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def _head_group(heads: Iterable[int]) -> str:
    return ",".join(str(item) for item in sorted(set(int(head) for head in heads)))


def _context_key(entry: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        entry.get("model_id"),
        entry.get("layer_id"),
        entry.get("kv_len_bucket"),
        (entry.get("safety_budget") or {}).get("name"),
    )


def _experiment(
    *,
    name: str,
    entry: Dict[str, Any],
    head_group: str,
    risk: str,
    rationale: str,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    runtime = dict(entry.get("runtime") or {})
    overrides = overrides or {}
    runtime.update(overrides)
    runtime["head_group"] = head_group
    runtime["head_indices"] = [int(item) for item in head_group.split(",") if item]
    runtime["selected_head_count"] = len(runtime["head_indices"])
    return {
        "name": name,
        "risk": risk,
        "rationale": rationale,
        "model_id": entry.get("model_id"),
        "prompt_type": entry.get("prompt_type"),
        "layer_id": entry.get("layer_id"),
        "kv_len": entry.get("kv_len_bucket"),
        "budget": (entry.get("safety_budget") or {}).get("name"),
        "runtime": runtime,
    }


def _median_config(entries: Sequence[Dict[str, Any]], key: str) -> Any:
    values = [((entry.get("runtime") or {}).get(key)) for entry in entries]
    values = [value for value in values if value is not None]
    if not values:
        return None
    if all(isinstance(value, (int, float)) for value in values):
        return median(values)
    return values[0]


def _pathway(
    *,
    name: str,
    status: str,
    rationale: str,
    experiments: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "rationale": rationale,
        "experiments": list(experiments),
    }


def _group_by_context(entries: Sequence[Dict[str, Any]]) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[_context_key(entry)].append(entry)
    return grouped


def _baseline_path(entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    experiments = [
        _experiment(
            name="reproduce_policy_entry",
            entry=entry,
            head_group=_head_group(_entry_heads(entry)),
            risk="low",
            rationale="re-run known passing calibrated group as the control row",
        )
        for entry in entries
    ]
    return _pathway(
        name="selective_splitk_policy_baseline",
        status="go" if experiments else "blocked",
        rationale="Use only policy-passing prompt/layer/KV/head groups; dense fallback handles everything else.",
        experiments=experiments,
    )


def _intersection_union_paths(entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    intersection_experiments: List[Dict[str, Any]] = []
    union_experiments: List[Dict[str, Any]] = []
    for _, group in _group_by_context(entries).items():
        prompt_types = {entry.get("prompt_type") for entry in group}
        if len(prompt_types) < 2:
            continue
        head_sets = [set(_entry_heads(entry)) for entry in group]
        if not head_sets:
            continue
        intersection = set.intersection(*head_sets)
        union = set.union(*head_sets)
        anchor = max(
            group,
            key=lambda entry: _as_float((entry.get("quality") or {}).get("speedup_vs_dense"), 0.0),
        )
        common_overrides = {
            "num_chunks": int(_median_config(group, "num_chunks") or (anchor.get("runtime") or {}).get("num_chunks")),
            "filter_margin": float(
                min(
                    _as_float((entry.get("runtime") or {}).get("filter_margin"), 1.0e9)
                    for entry in group
                )
            ),
        }
        if len(intersection) >= 2:
            intersection_experiments.append(
                _experiment(
                    name="prompt_agnostic_intersection",
                    entry=anchor,
                    head_group=_head_group(intersection),
                    risk="medium",
                    rationale="try heads that passed across multiple prompt regimes to reduce prompt brittleness",
                    overrides=common_overrides,
                )
            )
        if len(union) > len(intersection):
            union_experiments.append(
                _experiment(
                    name="aggressive_union_with_verification",
                    entry=anchor,
                    head_group=_head_group(union),
                    risk="high",
                    rationale="test whether sampled verification can make the larger union group usable",
                    overrides={**common_overrides, "verification": "sample_skipped_blocks"},
                )
            )
    return [
        _pathway(
            name="prompt_agnostic_head_intersection",
            status="probe" if intersection_experiments else "insufficient_evidence",
            rationale="If prompt-specific calibration is too brittle, run only the head intersection that survives multiple prompts.",
            experiments=intersection_experiments,
        ),
        _pathway(
            name="aggressive_union_with_online_verification",
            status="high_risk_probe" if union_experiments else "insufficient_evidence",
            rationale="If intersection is too slow, try the prompt-union group but require verification/telemetry fallback.",
            experiments=union_experiments,
        ),
    ]


def _strict_recovery_path(entries: Sequence[Dict[str, Any]], budgets: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    strict_has_entries = any((entry.get("safety_budget") or {}).get("name") == "strict" for entry in entries)
    moderate_entries = [entry for entry in entries if (entry.get("safety_budget") or {}).get("name") == "moderate"]
    experiments: List[Dict[str, Any]] = []
    if not strict_has_entries:
        for entry in moderate_entries:
            runtime = entry.get("runtime") or {}
            margin = _as_float(runtime.get("filter_margin"), 0.0)
            for next_margin in sorted({max(0.0, margin - 8.0), max(0.0, margin - 16.0)}):
                if next_margin == margin:
                    continue
                experiments.append(
                    _experiment(
                        name="strict_recovery_margin_down",
                        entry=entry,
                        head_group=_head_group(_entry_heads(entry)),
                        risk="medium",
                        rationale="tighten margin to reduce max/mean error and see if strict budget becomes reachable",
                        overrides={"filter_margin": next_margin},
                    )
                )
    return _pathway(
        name="strict_mode_recovery",
        status="probe" if experiments else "already_has_strict" if strict_has_entries else "blocked",
        rationale="Strict has no current policy entries; attempt smaller margins before changing metadata.",
        experiments=experiments[:16],
    )


def _cross_layer_path(entries: Sequence[Dict[str, Any]], target_layers: Sequence[int]) -> Dict[str, Any]:
    experiments: List[Dict[str, Any]] = []
    for entry in entries:
        current_layer = _as_int(entry.get("layer_id"), -1)
        for layer in target_layers:
            if layer == current_layer:
                continue
            cloned = dict(entry)
            cloned["layer_id"] = layer
            experiments.append(
                _experiment(
                    name="cross_layer_transfer",
                    entry=cloned,
                    head_group=_head_group(_entry_heads(entry)),
                    risk="high",
                    rationale="test whether calibrated sparse head sets transfer beyond the discovered layer",
                )
            )
    return _pathway(
        name="cross_layer_transfer_probe",
        status="high_risk_probe" if experiments else "blocked",
        rationale="If only L8 works, the path is narrow; test L4/L12 with the same groups before overfitting.",
        experiments=experiments[:24],
    )


def _long_kv_gate_path(entries: Sequence[Dict[str, Any]], *, min_kv_for_enable: int) -> Dict[str, Any]:
    experiments: List[Dict[str, Any]] = []
    for entry in entries:
        kv_len = _as_int(entry.get("kv_len_bucket"), 0)
        if kv_len >= min_kv_for_enable:
            continue
        cloned = dict(entry)
        cloned["kv_len_bucket"] = min_kv_for_enable
        experiments.append(
            _experiment(
                name="long_kv_only_retest",
                entry=cloned,
                head_group=_head_group(_entry_heads(entry)),
                risk="medium",
                rationale="enable sparse path only where seed/split overhead should amortize",
            )
        )
    return _pathway(
        name="long_kv_only_gate",
        status="go",
        rationale=f"Treat split-K projection as long-context only unless rows pass below {min_kv_for_enable}.",
        experiments=experiments,
    )


def plan_frontiers(
    policy: Dict[str, Any],
    *,
    target_layers: Sequence[int] = (4, 12),
    min_kv_for_enable: int = 32768,
) -> Dict[str, Any]:
    entries = list(policy.get("entries") or [])
    budgets = list(policy.get("budgets") or [])
    pathways: List[Dict[str, Any]] = [_baseline_path(entries)]
    pathways.extend(_intersection_union_paths(entries))
    pathways.append(_strict_recovery_path(entries, budgets))
    pathways.append(_cross_layer_path(entries, target_layers))
    pathways.append(_long_kv_gate_path(entries, min_kv_for_enable=min_kv_for_enable))
    return {
        "schema": SCHEMA,
        "summary": {
            "policy_entries": len(entries),
            "pathways": len(pathways),
            "experiments": sum(len(pathway.get("experiments") or []) for pathway in pathways),
            "target_layers": list(target_layers),
            "min_kv_for_enable": min_kv_for_enable,
        },
        "pathways": pathways,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _print_pathways(pathways: Sequence[Dict[str, Any]], *, limit: int) -> None:
    headers = ["pathway", "status", "experiments", "first"]
    print(" ".join(f"{header:>22}" for header in headers))
    for pathway in pathways[:limit]:
        experiments = pathway.get("experiments") or []
        first = experiments[0] if experiments else {}
        runtime = first.get("runtime") or {}
        first_text = f"L{first.get('layer_id')} kv={first.get('kv_len')} heads={runtime.get('head_group')}"
        print(
            " ".join(
                [
                    f"{_fmt(pathway.get('name'))[:22]:>22}",
                    f"{_fmt(pathway.get('status'))[:22]:>22}",
                    f"{len(experiments):>22}",
                    f"{first_text[:22]:>22}",
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("policy_json")
    parser.add_argument("--target-layers", default="4,12")
    parser.add_argument("--min-kv-for-enable", type=int, default=32768)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()

    policy = json.loads(Path(args.policy_json).read_text(encoding="utf-8"))
    target_layers = [int(item.strip()) for item in args.target_layers.split(",") if item.strip()]
    payload = plan_frontiers(
        policy,
        target_layers=target_layers,
        min_kv_for_enable=args.min_kv_for_enable,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    _print_pathways(payload["pathways"], limit=args.limit)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
