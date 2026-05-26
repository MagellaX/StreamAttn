"""Compile StreamAttn seed-only policy artifacts from validation evidence.

This is the artifact-driven bridge between research sweeps and serving policy
registration.  It consumes a layer sweep plus per-layer closed-loop rollout
artifacts, then emits green policy cells only for layers that pass both gates.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_MODEL_SLUG = "qwen25_05b"
DEFAULT_POLICY_DEFAULT = "qwen25_05b_l8_32k_seed_only_batched"
DEFAULT_TIMING = {
    "streamattn_seed_only_ms": 0.0254828811,
    "seed_direct_full_prealloc_ms": 0.0264505601,
    "flashinfer_batch_tc_exact_ms": 0.0343097591,
    "speedup_vs_flashinfer_batch": 1.34638462,
    "h100_service_b4_ms": 0.0254828811,
    "h100_flashinfer_b4_ms": 0.0343097591,
    "h100_service_b4_speedup": 1.34638462,
    "h100_planned_direct_b4_ms": 0.0242528009,
    "h100_planned_direct_b4_speedup": 1.41467203,
    "first_service_win_batch": 4,
    "first_planned_direct_win_batch": 4,
    "first_product_route_seed_only_batch": 4,
}


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    return int(value)


def _kv_bucket_label(kv_len: int) -> str:
    if kv_len % 1024 == 0:
        return f"{kv_len // 1024}k"
    return str(kv_len)


def _portable_path(path: Path) -> str:
    return path.as_posix()


def _budget_tier(observed: float, tiers: Sequence[float]) -> float:
    for tier in tiers:
        if observed <= tier:
            return float(tier)
    return float(tiers[-1])


def _summary_passes(
    summary: Dict[str, Any],
    *,
    max_kl: float,
    min_top5_overlap: int,
    require_top1_match: bool,
) -> bool:
    if require_top1_match and _as_int(summary.get("top1_changed_count")) != 0:
        return False
    if _as_int(summary.get("topk_overlap_min"), default=999) < min_top5_overlap:
        return False
    if _as_float(summary.get("kl_max"), default=math.inf) > max_kl:
        return False
    return True


def _closed_loop_passes(
    rollout: Dict[str, Any],
    *,
    max_kl: float,
    min_top5_overlap: int,
    require_top1_match: bool,
) -> bool:
    for key in ("teacher_forced", "greedy", "sampling"):
        summary = rollout.get(key, {}).get("summary")
        if not isinstance(summary, dict):
            return False
        if not _summary_passes(
            summary,
            max_kl=max_kl,
            min_top5_overlap=min_top5_overlap,
            require_top1_match=require_top1_match,
        ):
            return False

    greedy = rollout["greedy"]["summary"]
    sampling = rollout["sampling"]["summary"]
    if _as_int(greedy.get("diverged_row_count")) != 0:
        return False
    if _as_float(greedy.get("sequence_exact_match_rate")) < 1.0:
        return False
    if _as_int(sampling.get("sample_token_changed_count")) != 0:
        return False
    if _as_float(sampling.get("sample_sequence_exact_match_rate")) < 1.0:
        return False
    return True


def _discover_rollout_path(
    *,
    layer_id: int,
    min_batch: int,
    closed_loop_dir: Path,
) -> Optional[Path]:
    candidates = [
        closed_loop_dir / f"seed_only_l{layer_id}_b{min_batch}_closed_loop_h100.json",
        closed_loop_dir / f"seed_only_l{layer_id}_closed_loop_h100.json",
    ]
    if layer_id == 8:
        candidates.append(closed_loop_dir / f"seed_only_b{min_batch}_closed_loop_h100.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(
        closed_loop_dir.glob(f"seed_only_l{layer_id}_b{min_batch}_closed_loop*.json")
    )
    if not matches:
        matches = sorted(
            closed_loop_dir.glob(f"seed_only_*_l{layer_id}_b{min_batch}_closed_loop*.json")
        )
    return matches[0] if matches else None


@dataclass(frozen=True)
class CompiledSeedCell:
    layer_id: int
    name: str
    path: str
    policy_id: str
    aliases: List[str]
    policy: Dict[str, Any]
    sweep_summary: Dict[str, Any]
    rollout_path: str
    rollout: Dict[str, Any]
    status: str


def _policy_name(model_slug: str, layer_id: int, kv_len: int) -> str:
    return f"{model_slug}_l{layer_id}_{_kv_bucket_label(kv_len)}_seed_only_batched"


def _policy_id(model_slug: str, layer_id: int, kv_len: int, dtype: str, min_batch: int) -> str:
    return (
        f"{model_slug}_l{layer_id}_{_kv_bucket_label(kv_len)}_"
        f"{dtype}_b{min_batch}_seed_only_v1"
    )


def _existing_registry_by_name(registry_path: Path) -> Dict[str, Dict[str, Any]]:
    if not registry_path.exists():
        return {}
    registry = _read_json(registry_path)
    return {
        str(entry.get("name")): dict(entry)
        for entry in registry.get("policies") or []
        if entry.get("name")
    }


def _existing_registry_payload(registry_path: Path) -> Dict[str, Any]:
    if not registry_path.exists():
        return {
            "schema": "streamattn.policy_registry.v1",
            "default": DEFAULT_POLICY_DEFAULT,
            "policies": [],
        }
    return _read_json(registry_path)


def _seed_config_from_sweep(sweep: Dict[str, Any]) -> Dict[str, Any]:
    seed = dict(sweep.get("seed_config") or {})
    return {
        "block_size": _as_int(seed.get("block_size"), 32),
        "sink_blocks": _as_int(seed.get("sink_blocks"), 2),
        "recent_blocks": _as_int(seed.get("recent_blocks"), 2),
        "middle_seed_blocks": _as_int(seed.get("middle_seed_blocks"), 8),
        "block_order": str(seed.get("block_order") or "recent_first"),
        "num_warps": _as_int(seed.get("num_warps"), 4),
        "num_stages": _as_int(seed.get("num_stages"), 2),
    }


def _prompt_kinds(sweep: Dict[str, Any]) -> List[str]:
    prompt_kinds = sweep.get("sweep", {}).get("prompt_kinds")
    if isinstance(prompt_kinds, str):
        return [kind.strip() for kind in prompt_kinds.split(",") if kind.strip()]
    if isinstance(prompt_kinds, list):
        return [str(kind) for kind in prompt_kinds]
    return []


def _closed_loop_case_count(rollout: Dict[str, Any]) -> int:
    return _as_int(rollout.get("greedy", {}).get("summary", {}).get("case_count"))


def _build_policy_payload(
    *,
    sweep: Dict[str, Any],
    layer_result: Dict[str, Any],
    rollout: Dict[str, Any],
    rollout_artifact: str,
    policy_id: str,
    model_id: str,
    layer_id: int,
    dtype: str,
    kv_len: int,
    min_batch: int,
    min_top5_overlap: int,
    max_kl: float,
) -> Dict[str, Any]:
    shape = dict(layer_result.get("shape") or {})
    seed_config = _seed_config_from_sweep(sweep)
    prompt_kinds = _prompt_kinds(sweep)
    teacher = rollout["teacher_forced"]["summary"]
    greedy = rollout["greedy"]["summary"]
    sampling = rollout["sampling"]["summary"]
    sweep_summary = layer_result["policy"]["summary_vs_model_baseline"]
    max_observed_logit_delta = max(
        _as_float(sweep_summary.get("max_logit_delta")),
        _as_float(teacher.get("max_logit_delta")),
        _as_float(greedy.get("max_logit_delta")),
        _as_float(sampling.get("max_logit_delta")),
    )
    max_observed_logprob_delta = max(
        abs(_as_float(sweep_summary.get("target_next_token_logprob_delta_max_abs"))),
        abs(_as_float(teacher.get("reference_top1_logprob_delta_max_abs"))),
        abs(_as_float(greedy.get("reference_top1_logprob_delta_max_abs"))),
        abs(_as_float(sampling.get("reference_top1_logprob_delta_max_abs"))),
    )
    max_logit_delta = _budget_tier(max_observed_logit_delta, [0.2, 1.0, 2.0, 4.0])
    max_logprob_delta = _budget_tier(
        max_observed_logprob_delta,
        [0.001, 0.002, 0.005, 0.01],
    )
    case_count = _closed_loop_case_count(rollout)
    return {
        "schema": "streamattn.gate0.seed_only_batched_policy.v1",
        "policy_id": policy_id,
        "mode": "all_seed_only",
        "model_id": model_id,
        "layer_id": layer_id,
        "tensor_space": "post_rope",
        "dtype": dtype,
        "kv_len_bucket": kv_len,
        "min_batch": min_batch,
        "shape": {
            "batch": min_batch,
            "q_heads": _as_int(shape.get("q_heads"), 14),
            "true_kv_heads": _as_int(shape.get("true_kv_heads"), 2),
            "dim": _as_int(shape.get("dim"), 64),
            "kv_len": kv_len,
            "dtype": dtype,
        },
        "seed_config": seed_config,
        "safety": {
            "gate": "distribution_aware",
            "top1_must_match": True,
            "min_top5_overlap": min_top5_overlap,
            "max_kl": max_kl,
            "max_logit_delta": max_logit_delta,
            "max_logprob_delta": max_logprob_delta,
            "validated_rollout_steps": _as_int(greedy.get("step_count"), 32),
            "teacher_forced_last32": {
                "artifact": str(Path(sweep.get("_artifact_path", ""))),
                "prompt_kinds": prompt_kinds,
                "cases": _as_int(sweep_summary.get("case_count")),
                "top1_changes": _as_int(sweep_summary.get("top1_changed_count")),
                "top5_overlap_min": _as_int(sweep_summary.get("topk_overlap_min")),
                "kl_max": _as_float(sweep_summary.get("kl_max")),
                "max_logit_delta": _as_float(sweep_summary.get("max_logit_delta")),
                "target_logprob_delta_max": _as_float(
                    sweep_summary.get("target_next_token_logprob_delta_max_abs")
                ),
            },
            "greedy_closed_loop_32": {
                "artifact": rollout_artifact,
                "prompt_kinds": prompt_kinds,
                "cases": case_count,
                "diverged_rows": _as_int(greedy.get("diverged_row_count")),
                "sequence_exact_match_rate": _as_float(greedy.get("sequence_exact_match_rate")),
                "top1_changes": _as_int(greedy.get("top1_changed_count")),
                "top5_overlap_min": _as_int(greedy.get("topk_overlap_min")),
                "kl_max": _as_float(greedy.get("kl_max")),
                "max_logit_delta": _as_float(greedy.get("max_logit_delta")),
                "reference_top1_margin_min": _as_float(greedy.get("reference_top1_margin_min")),
                "reference_top1_logprob_delta_max": _as_float(
                    greedy.get("reference_top1_logprob_delta_max_abs")
                ),
            },
            "coupled_top_p_sampling_32": {
                "artifact": rollout_artifact,
                "prompt_kinds": prompt_kinds,
                "cases": case_count,
                "top_p": _as_float(
                    sampling.get("sampling", {}).get("top_p"),
                    0.95,
                ),
                "sample_token_changed_count": _as_int(sampling.get("sample_token_changed_count")),
                "sample_sequence_exact_match_rate": _as_float(
                    sampling.get("sample_sequence_exact_match_rate")
                ),
                "top1_changes": _as_int(sampling.get("top1_changed_count")),
                "top5_overlap_min": _as_int(sampling.get("topk_overlap_min")),
                "kl_max": _as_float(sampling.get("kl_max")),
                "reference_top1_logprob_delta_max": _as_float(
                    sampling.get("reference_top1_logprob_delta_max_abs")
                ),
            },
            "batch4_closed_loop_32": {
                "artifact": rollout_artifact,
                "prompt_kinds": prompt_kinds,
                "cases": case_count,
                "top1_changes": _as_int(greedy.get("top1_changed_count")),
                "top5_overlap_min": _as_int(greedy.get("topk_overlap_min")),
                "kl_max": _as_float(greedy.get("kl_max")),
                "max_logit_delta": _as_float(greedy.get("max_logit_delta")),
                "teacher_forced_top1_changes": _as_int(teacher.get("top1_changed_count")),
                "greedy_diverged_rows": _as_int(greedy.get("diverged_row_count")),
                "greedy_sequence_exact_match_rate": _as_float(
                    greedy.get("sequence_exact_match_rate")
                ),
                "sample_token_changed_count": _as_int(sampling.get("sample_token_changed_count")),
                "sample_sequence_exact_match_rate": _as_float(
                    sampling.get("sample_sequence_exact_match_rate")
                ),
                "reference_top1_margin_min": _as_float(greedy.get("reference_top1_margin_min")),
                "reference_top1_logprob_delta_max": _as_float(
                    greedy.get("reference_top1_logprob_delta_max_abs")
                ),
            },
        },
        "timing": dict(DEFAULT_TIMING),
        "kernel_modes": {
            "batch_ge_4": "head_private_direct_seed",
            "batch_lt_4": "exact_native",
        },
        "fallback": "dense",
    }


def _registry_entry(
    *,
    name: str,
    path: str,
    policy: Dict[str, Any],
    aliases: Sequence[str],
) -> Dict[str, Any]:
    shape = policy["shape"]
    return {
        "name": name,
        "path": path,
        "policy_id": policy["policy_id"],
        "aliases": list(aliases),
        "status": "green",
        "model_id": policy["model_id"],
        "layer_id": policy["layer_id"],
        "mode": policy["mode"],
        "tensor_space": policy["tensor_space"],
        "dtype": policy["dtype"],
        "kv_len_bucket": policy["kv_len_bucket"],
        "min_batch": policy["min_batch"],
        "q_heads": shape["q_heads"],
        "kv_heads": shape["true_kv_heads"],
        "head_dim": shape["dim"],
        "attention_type": "true_gqa",
        "kernel_modes": dict(policy["kernel_modes"]),
    }


def compile_seed_policy_cells(
    *,
    sweep_json: Path,
    closed_loop_dir: Path,
    policy_dir: Path,
    registry_json: Path,
    model_slug: str = DEFAULT_MODEL_SLUG,
    model_id: Optional[str] = None,
    default_policy_name: Optional[str] = None,
    min_batch: int = 4,
    dtype: Optional[str] = None,
    max_kl: Optional[float] = None,
    min_top5_overlap: Optional[int] = None,
    require_top1_match: Optional[bool] = None,
) -> Dict[str, Any]:
    sweep = _read_json(sweep_json)
    sweep["_artifact_path"] = _portable_path(sweep_json)
    sweep_cfg = sweep.get("sweep") or {}
    gate = sweep.get("safety_gate") or {}
    compiled_model_id = model_id or str(sweep_cfg.get("model") or DEFAULT_MODEL_ID)
    compiled_dtype = dtype or str(sweep_cfg.get("dtype") or "fp16")
    kv_len = _as_int(sweep_cfg.get("kv_len"), 32768)
    compiled_max_kl = float(max_kl if max_kl is not None else gate.get("max_kl", 0.0001))
    compiled_min_top5 = int(
        min_top5_overlap
        if min_top5_overlap is not None
        else gate.get("min_topk_overlap", 4)
    )
    compiled_require_top1 = bool(
        require_top1_match
        if require_top1_match is not None
        else gate.get("require_top1_match", True)
    )
    existing = _existing_registry_by_name(registry_json)
    green_cells: List[CompiledSeedCell] = []
    rejected: List[Dict[str, Any]] = []

    for layer_key, layer_result in sorted(
        (sweep.get("results_by_layer") or {}).items(),
        key=lambda item: int(item[0]),
    ):
        layer_id = int(layer_key)
        policy = dict(layer_result.get("policy") or {})
        sweep_summary = dict(policy.get("summary_vs_model_baseline") or {})
        reasons: List[str] = []
        if not policy.get("passes_distribution_gate", False):
            reasons.append("sweep_distribution_gate_failed")
        if not _summary_passes(
            sweep_summary,
            max_kl=compiled_max_kl,
            min_top5_overlap=compiled_min_top5,
            require_top1_match=compiled_require_top1,
        ):
            reasons.append("sweep_summary_failed")

        rollout_path = _discover_rollout_path(
            layer_id=layer_id,
            min_batch=min_batch,
            closed_loop_dir=closed_loop_dir,
        )
        rollout: Optional[Dict[str, Any]] = None
        if rollout_path is None:
            reasons.append("missing_closed_loop_artifact")
        else:
            rollout = _read_json(rollout_path)
            if not _closed_loop_passes(
                rollout,
                max_kl=compiled_max_kl,
                min_top5_overlap=compiled_min_top5,
                require_top1_match=compiled_require_top1,
            ):
                reasons.append("closed_loop_gate_failed")

        if reasons:
            rejected.append({"layer_id": layer_id, "reasons": reasons})
            continue

        name = _policy_name(model_slug, layer_id, kv_len)
        existing_entry = existing.get(name, {})
        relative_path = str(
            existing_entry.get("path")
            or f"policies/{name}.json"
        )
        compiled_policy_id = str(
            existing_entry.get("policy_id")
            or _policy_id(model_slug, layer_id, kv_len, compiled_dtype, min_batch)
        )
        aliases = [str(alias) for alias in existing_entry.get("aliases") or []]
        policy_payload = _build_policy_payload(
            sweep=sweep,
            layer_result=layer_result,
            rollout=rollout or {},
            rollout_artifact=_portable_path(rollout_path),
            policy_id=compiled_policy_id,
            model_id=compiled_model_id,
            layer_id=layer_id,
            dtype=compiled_dtype,
            kv_len=kv_len,
            min_batch=min_batch,
            min_top5_overlap=compiled_min_top5,
            max_kl=compiled_max_kl,
        )
        green_cells.append(
            CompiledSeedCell(
                layer_id=layer_id,
                name=name,
                path=relative_path,
                policy_id=compiled_policy_id,
                aliases=aliases,
                policy=policy_payload,
                sweep_summary=sweep_summary,
                rollout_path=_portable_path(rollout_path),
                rollout=rollout or {},
                status="green",
            )
        )

    green_names = [cell.name for cell in green_cells]
    resolved_default_policy = default_policy_name
    if resolved_default_policy is None and DEFAULT_POLICY_DEFAULT in green_names:
        resolved_default_policy = DEFAULT_POLICY_DEFAULT
    if resolved_default_policy is None and green_cells:
        resolved_default_policy = green_cells[0].name
    registry = {
        "schema": "streamattn.policy_registry.v1",
        "default": resolved_default_policy or "",
        "policies": [
            _registry_entry(
                name=cell.name,
                path=cell.path,
                policy=cell.policy,
                aliases=cell.aliases,
            )
            for cell in green_cells
        ],
    }
    return {
        "schema": "streamattn.seed_policy_compiler.v1",
        "sweep_json": _portable_path(sweep_json),
        "closed_loop_dir": _portable_path(closed_loop_dir),
        "model_id": compiled_model_id,
        "model_slug": model_slug,
        "dtype": compiled_dtype,
        "kv_len_bucket": kv_len,
        "min_batch": min_batch,
        "gates": {
            "max_kl": compiled_max_kl,
            "min_top5_overlap": compiled_min_top5,
            "require_top1_match": compiled_require_top1,
        },
        "green_layers": [cell.layer_id for cell in green_cells],
        "rejected_layers": rejected,
        "registry": registry,
        "policies": [
            {
                "layer_id": cell.layer_id,
                "name": cell.name,
                "path": cell.path,
                "policy_id": cell.policy_id,
                "rollout_artifact": cell.rollout_path,
                "sweep_kl_max": _as_float(cell.sweep_summary.get("kl_max")),
                "sweep_top5_overlap_min": _as_int(cell.sweep_summary.get("topk_overlap_min")),
                "closed_loop_kl_max": _as_float(
                    cell.rollout["greedy"]["summary"].get("kl_max")
                ),
                "closed_loop_top5_overlap_min": _as_int(
                    cell.rollout["greedy"]["summary"].get("topk_overlap_min")
                ),
                "max_logprob_delta_budget": cell.policy["safety"]["max_logprob_delta"],
                "max_logit_delta_budget": cell.policy["safety"]["max_logit_delta"],
            }
            for cell in green_cells
        ],
        "_compiled_cells": green_cells,
    }


def _write_compiled_outputs(
    compiled: Dict[str, Any],
    *,
    policy_dir: Path,
    registry_json: Path,
    write_existing: bool,
) -> Dict[str, Any]:
    existing_payload = _existing_registry_payload(registry_json)
    existing = _existing_registry_by_name(registry_json)
    writes: List[Dict[str, Any]] = []
    for cell in compiled["_compiled_cells"]:
        destination = policy_dir.parent / cell.path
        exists = destination.exists()
        if exists and not write_existing:
            action = "preserved_existing"
        else:
            _write_json(destination, cell.policy)
            action = "wrote"
        writes.append({"path": _portable_path(destination), "action": action})
        existing[cell.name] = _registry_entry(
            name=cell.name,
            path=cell.path,
            policy=cell.policy,
            aliases=cell.aliases,
        )
    merged_registry = {
        "schema": "streamattn.policy_registry.v1",
        "default": (
            existing_payload.get("default")
            if existing_payload.get("default") in existing
            else compiled["registry"].get("default")
        ),
        "policies": sorted(
            existing.values(),
            key=lambda entry: (
                str(entry.get("model_id", "")),
                int(entry.get("kv_len_bucket", 0)),
                int(entry.get("min_batch", 0)),
                int(entry.get("layer_id", 0)),
                str(entry.get("name", "")),
            ),
        ),
    }
    _write_json(registry_json, merged_registry)
    writes.append({"path": _portable_path(registry_json), "action": "wrote"})
    return {"writes": writes}


def _public_summary(compiled: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in compiled.items() if key != "_compiled_cells"}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-json",
        type=Path,
        default=Path("artifacts/gate0/seed_only_layer_sweep_l0_l23_b4_h100.json"),
    )
    parser.add_argument("--closed-loop-dir", type=Path, default=Path("artifacts/gate0"))
    parser.add_argument(
        "--policy-dir",
        type=Path,
        default=Path("stream_attention/policies"),
    )
    parser.add_argument(
        "--registry-json",
        type=Path,
        default=Path("stream_attention/policies/registry.json"),
    )
    parser.add_argument("--model-slug", default=DEFAULT_MODEL_SLUG)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--default-policy-name", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--min-batch", type=int, default=4)
    parser.add_argument("--max-kl", type=float, default=None)
    parser.add_argument("--min-top5-overlap", type=int, default=None)
    parser.add_argument("--allow-top1-changes", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--write-existing", action="store_true")
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    compiled = compile_seed_policy_cells(
        sweep_json=args.sweep_json,
        closed_loop_dir=args.closed_loop_dir,
        policy_dir=args.policy_dir,
        registry_json=args.registry_json,
        model_slug=args.model_slug,
        model_id=args.model_id,
        default_policy_name=args.default_policy_name,
        min_batch=args.min_batch,
        dtype=args.dtype,
        max_kl=args.max_kl,
        min_top5_overlap=args.min_top5_overlap,
        require_top1_match=not args.allow_top1_changes,
    )
    write_summary: Dict[str, Any] = {}
    if args.write:
        write_summary = _write_compiled_outputs(
            compiled,
            policy_dir=args.policy_dir,
            registry_json=args.registry_json,
            write_existing=args.write_existing,
        )
    summary = _public_summary(compiled)
    if write_summary:
        summary["write_summary"] = write_summary
    if args.summary_json:
        _write_json(args.summary_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
