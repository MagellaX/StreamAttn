"""Targeted repair sweep for Qwen3B stress-policy failures.

This runner is intentionally narrow.  It tests layer-specific seed widening and
late-layer exact gating only on buckets that prompt-aware coverage identified as
unsafe for L24/L26/L27.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from itertools import cycle, islice
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_seed_only_stress_attribution import (  # noqa: E402
    QWEN3B_POLICY_BY_LAYER,
    failure_score,
)
from benchmarks.summarize_seed_policy_stress_replay import summarize_payload  # noqa: E402


STRICT_LAYERS = (0, 14, 16, 24, 26, 27, 35)
SEED_CONFIGS = {
    "s512": "sink=2,recent=4,middle=10,block=32",
    "s640": "sink=2,recent=4,middle=14,block=32",
    "s768": "sink=2,recent=6,middle=16,block=32",
}


@dataclass(frozen=True)
class RepairVariant:
    name: str
    layers: tuple[int, ...]
    overrides: str = ""

    @property
    def policy_names(self) -> str:
        missing = [layer for layer in self.layers if layer not in QWEN3B_POLICY_BY_LAYER]
        if missing:
            raise KeyError(f"no packaged Qwen3B policy mapping for layers {missing}")
        return ",".join(QWEN3B_POLICY_BY_LAYER[layer] for layer in self.layers)


def _parse_csv(text: str) -> List[str]:
    return [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]


def _override(*layers: int, seed: str) -> str:
    config = SEED_CONFIGS[seed]
    return ";".join(f"{int(layer)}:{config}" for layer in layers)


def build_repair_variants(variant_set: str) -> List[RepairVariant]:
    variants = [
        RepairVariant("strict_base", STRICT_LAYERS),
        RepairVariant("minus_l27", tuple(layer for layer in STRICT_LAYERS if layer != 27)),
        RepairVariant("minus_l26_l27", tuple(layer for layer in STRICT_LAYERS if layer not in {26, 27})),
        RepairVariant("l26_s640", STRICT_LAYERS, _override(26, seed="s640")),
        RepairVariant("l27_s640", STRICT_LAYERS, _override(27, seed="s640")),
        RepairVariant("l26_l27_s640", STRICT_LAYERS, _override(26, 27, seed="s640")),
        RepairVariant("l24_l26_l27_s640", STRICT_LAYERS, _override(24, 26, 27, seed="s640")),
    ]
    if variant_set == "minimal":
        keep = {"strict_base", "minus_l26_l27", "l26_l27_s640"}
        return [variant for variant in variants if variant.name in keep]
    if variant_set == "wide":
        variants.extend(
            [
                RepairVariant("l26_l27_s512", STRICT_LAYERS, _override(26, 27, seed="s512")),
                RepairVariant("l26_l27_s768", STRICT_LAYERS, _override(26, 27, seed="s768")),
                RepairVariant("l24_json_s640", STRICT_LAYERS, _override(24, seed="s640")),
                RepairVariant(
                    "minus_l27_l26_s640",
                    tuple(layer for layer in STRICT_LAYERS if layer != 27),
                    _override(26, seed="s640"),
                ),
            ]
        )
    return variants


def build_bucket_prompt_pack(
    *,
    source_prompt_file: Path,
    target_buckets: Sequence[str],
    batch_size: int,
    output_path: Path,
) -> List[Dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in source_prompt_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_bucket: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        bucket = str(row.get("bucket") or row.get("kind") or "")
        if bucket in target_buckets and bucket not in by_bucket:
            by_bucket[bucket] = row
    missing = [bucket for bucket in target_buckets if bucket not in by_bucket]
    if missing:
        raise ValueError(f"prompt file is missing target buckets: {missing}")
    selected = []
    for idx, row in enumerate(islice(cycle([by_bucket[bucket] for bucket in target_buckets]), batch_size)):
        clone = dict(row)
        clone["id"] = f"{row.get('id', row.get('bucket', 'row'))}_repairrep{idx:02d}"
        clone["repair_source_id"] = row.get("id", "")
        selected.append(clone)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    return selected


def _bucket_summary(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    buckets = payload.get("safety", {}).get("by_prompt_bucket", {}) or {}
    result = {}
    for bucket, row in buckets.items():
        result[bucket] = {
            "case_count": int(row.get("case_count", 0) or 0),
            "kl_max": float(row.get("kl_max", 0.0) or 0.0),
            "kl_p99": float(row.get("kl_p99", 0.0) or 0.0),
            "top1_changes": int(row.get("top1_changed_count", 0) or 0),
            "sample_changes": int(row.get("sample_token_changed_count", 0) or 0),
            "topk_overlap_min": int(row.get("topk_overlap_min", 0) or 0),
            "target_logprob_delta_max_abs": float(
                row.get("reference_top1_logprob_delta_max_abs", 0.0) or 0.0
            ),
        }
    return result


def _run_cmd(cmd: List[str], *, env: Dict[str, str]) -> None:
    print(f"[repair-sweep] running {' '.join(cmd[:6])} ...", flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


def _route_cmd(args: argparse.Namespace, variant: RepairVariant, prompt_path: Path, output_path: Path) -> List[str]:
    cmd = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "benchmarks" / "profile_seed_only_route_bundle_decode.py"),
        "--model",
        args.model,
        "--policy-names",
        variant.policy_names,
        "--prompt-file",
        str(prompt_path),
        "--prompt-truncation-side",
        args.prompt_truncation_side,
        "--max-prompts",
        str(args.batch_size),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--max-seq",
        str(args.max_seq),
        "--steps",
        str(args.steps),
        "--warmup-steps",
        str(args.warmup_steps),
        "--top-k",
        str(args.top_k),
        "--max-kl",
        str(args.max_kl),
        "--min-topk-overlap",
        str(args.min_topk_overlap),
        "--max-logprob-delta",
        str(args.max_logprob_delta),
        "--sample-temperature",
        str(args.sample_temperature),
        "--sample-top-p",
        str(args.sample_top_p),
        "--sample-top-k",
        str(args.sample_top_k),
        "--sample-seed",
        str(args.sample_seed),
        "--q-heads",
        str(args.q_heads),
        "--kv-heads",
        str(args.kv_heads),
        "--head-dim",
        str(args.head_dim),
        "--allow-mixed-seed-configs",
        "--output-json",
        str(output_path),
    ]
    if variant.overrides:
        cmd.extend(["--layer-seed-overrides", variant.overrides])
    if args.native_routed_cache:
        cmd.append("--native-routed-cache")
    if args.fused_rope_append_seed:
        cmd.append("--fused-rope-append-seed")
    if args.packed_qkv_projection:
        cmd.append("--packed-qkv-projection")
    return cmd


def run_sweep(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_buckets = _parse_csv(args.target_buckets)
    prompt_pack_path = output_dir / "repair_prompt_pack.jsonl"
    prompt_rows = build_bucket_prompt_pack(
        source_prompt_file=Path(args.prompt_file),
        target_buckets=target_buckets,
        batch_size=args.batch_size,
        output_path=prompt_pack_path,
    )
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    route_summaries = []
    for variant in build_repair_variants(args.variant_set):
        output_path = output_dir / f"{variant.name}.json"
        if args.reuse_existing and output_path.exists():
            print(f"[repair-sweep] reusing {output_path}", flush=True)
        else:
            _run_cmd(_route_cmd(args, variant, prompt_pack_path, output_path), env=env)
        summary = summarize_payload(output_path)
        summary["route_name"] = variant.name
        summary["layers"] = list(variant.layers)
        summary["overrides"] = variant.overrides
        summary["score"] = failure_score(
            summary,
            max_kl=args.max_kl,
            max_logprob_delta=args.max_logprob_delta,
        )
        summary["by_bucket"] = _bucket_summary(output_path)
        route_summaries.append(summary)

    route_summaries.sort(key=lambda row: (float(row.get("score", 0.0)), -float(row.get("speedup_vs_dense_decode", 0.0))))
    result = {
        "schema": "streamattn.seed_policy_repair_sweep.v1",
        "prompt_file": args.prompt_file,
        "repair_prompt_pack": str(prompt_pack_path),
        "target_buckets": target_buckets,
        "batch_size": int(args.batch_size),
        "steps": int(args.steps),
        "variant_set": args.variant_set,
        "prompts": [
            {
                key: row.get(key, "")
                for key in ("id", "repair_source_id", "bucket", "kind", "risk")
                if row.get(key, "") != ""
            }
            for row in prompt_rows
        ],
        "routes": route_summaries,
        "best_route": route_summaries[0] if route_summaries else {},
    }
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--prompt-file", default="benchmarks/prompts/qwen3b_32k_stress_pack_v1_b8.jsonl")
    parser.add_argument("--target-buckets", default="chat_instruction,json_tool,needle_rag,noisy_neartie")
    parser.add_argument("--variant-set", choices=["minimal", "focused", "wide"], default="focused")
    parser.add_argument("--prompt-truncation-side", choices=["left", "right"], default="left")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--max-seq", type=int, default=32768)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-kl", type=float, default=1.0e-4)
    parser.add_argument("--min-topk-overlap", type=int, default=4)
    parser.add_argument("--max-logprob-delta", type=float, default=2.0e-3)
    parser.add_argument("--sample-temperature", type=float, default=0.8)
    parser.add_argument("--sample-top-p", type=float, default=0.95)
    parser.add_argument("--sample-top-k", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=1234)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--native-routed-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fused-rope-append-seed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--packed-qkv-projection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--output-dir", default="artifacts/gate0/qwen25_3b_32k_b8_repair_sweep")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    result = run_sweep(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
