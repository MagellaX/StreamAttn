"""Route-level stress attribution for seed-only policy bundles.

This runner executes the actual model-decode route benchmark over a small set of
route variants, then scores each route by safety damage.  It is deliberately
oriented around attribution, not final promotion: use shorter step counts for
screening, then rerun promising repaired routes with the full gate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.summarize_seed_policy_stress_replay import summarize_payload  # noqa: E402


QWEN3B_POLICY_BY_LAYER = {
    0: "qwen25_3b_l0_32k_seed_only_batched",
    2: "qwen25_3b_l2_s640_32k_seed_only_batched",
    14: "qwen25_3b_l14_32k_seed_only_batched",
    16: "qwen25_3b_l16_32k_seed_only_batched",
    24: "qwen25_3b_l24_32k_seed_only_batched",
    26: "qwen25_3b_l26_32k_seed_only_batched",
    27: "qwen25_3b_l27_32k_seed_only_batched",
    29: "qwen25_3b_l29_32k_seed_only_batched",
    35: "qwen25_3b_l35_32k_seed_only_batched",
}


@dataclass(frozen=True)
class RouteSpec:
    name: str
    layers: tuple[int, ...]

    @property
    def policy_names(self) -> str:
        missing = [layer for layer in self.layers if layer not in QWEN3B_POLICY_BY_LAYER]
        if missing:
            raise KeyError(f"no packaged policy mapping for layers {missing}")
        return ",".join(QWEN3B_POLICY_BY_LAYER[layer] for layer in self.layers)

    @property
    def allow_mixed_seed_configs(self) -> bool:
        seed_tokens = {640 if layer == 2 else 384 for layer in self.layers}
        return len(seed_tokens) > 1


def _parse_layers(text: str) -> tuple[int, ...]:
    layers = tuple(sorted({int(part.strip()) for part in text.split(",") if part.strip()}))
    if not layers:
        raise ValueError("layer list must not be empty")
    return layers


def build_route_specs(
    *,
    base_layers: Sequence[int],
    route_set: str,
    add_layers: Sequence[int],
) -> List[RouteSpec]:
    base = tuple(sorted(int(layer) for layer in base_layers))
    specs: List[RouteSpec] = [RouteSpec("strict_base", base)]
    if route_set in {"leaveout", "full"}:
        for layer in base:
            specs.append(
                RouteSpec(
                    f"minus_l{layer}",
                    tuple(layer_id for layer_id in base if layer_id != layer),
                )
            )
    if route_set in {"single", "full"}:
        for layer in base:
            specs.append(RouteSpec(f"single_l{layer}", (layer,)))
    if route_set in {"candidate", "full"}:
        for layer in add_layers:
            if layer in base:
                continue
            specs.append(RouteSpec(f"plus_l{layer}", tuple(sorted((*base, int(layer))))))
            specs.append(RouteSpec(f"single_l{layer}", (int(layer),)))
    return specs


def failure_score(
    summary: Dict[str, Any],
    *,
    max_kl: float = 1.0e-4,
    max_logprob_delta: float = 2.0e-3,
) -> float:
    kl = float(summary.get("kl_max", 0.0) or 0.0)
    logprob = float(summary.get("target_logprob_delta_max_abs", 0.0) or 0.0)
    top1 = int(summary.get("top1_changes", 0) or 0)
    sample = int(summary.get("sample_changes", 0) or 0)
    topk = int(summary.get("topk_overlap_min", 5) or 5)
    return (
        10.0 * top1
        + 10.0 * sample
        + 1000.0 * max(0.0, kl - max_kl)
        + 100.0 * max(0.0, logprob - max_logprob_delta)
        + 5.0 * max(0, 4 - topk)
    )


def _run_cmd(cmd: List[str], *, env: Dict[str, str]) -> None:
    print(f"[stress-attribution] running {' '.join(cmd[:6])} ...", flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


def _route_cmd(args: argparse.Namespace, spec: RouteSpec, output_path: Path) -> List[str]:
    cmd = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "benchmarks" / "profile_seed_only_route_bundle_decode.py"),
        "--model",
        args.model,
        "--policy-names",
        spec.policy_names,
        "--prompt-file",
        args.prompt_file,
        "--prompt-truncation-side",
        args.prompt_truncation_side,
        "--max-prompts",
        str(args.max_prompts),
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
        "--output-json",
        str(output_path),
    ]
    if args.native_routed_cache:
        cmd.append("--native-routed-cache")
    if args.fused_rope_append_seed:
        cmd.append("--fused-rope-append-seed")
    if args.packed_qkv_projection:
        cmd.append("--packed-qkv-projection")
    if spec.allow_mixed_seed_configs:
        cmd.append("--allow-mixed-seed-configs")
    return cmd


def run_attribution(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = build_route_specs(
        base_layers=_parse_layers(args.base_layers),
        route_set=args.route_set,
        add_layers=_parse_layers(args.add_layers) if args.add_layers.strip() else (),
    )
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    route_summaries: List[Dict[str, Any]] = []
    for spec in specs:
        output_path = output_dir / f"{spec.name}.json"
        if args.reuse_existing and output_path.exists():
            print(f"[stress-attribution] reusing {output_path}", flush=True)
        else:
            _run_cmd(_route_cmd(args, spec, output_path), env=env)
        summary = summarize_payload(output_path)
        summary["route_name"] = spec.name
        summary["score"] = failure_score(
            summary,
            max_kl=args.max_kl,
            max_logprob_delta=args.max_logprob_delta,
        )
        route_summaries.append(summary)

    by_name = {row["route_name"]: row for row in route_summaries}
    base = by_name.get("strict_base", {})
    base_score = float(base.get("score", 0.0) or 0.0)
    attribution = []
    for row in route_summaries:
        name = str(row["route_name"])
        if not name.startswith("minus_l"):
            continue
        layer = int(name.removeprefix("minus_l"))
        attribution.append(
            {
                "layer": layer,
                "leaveout_route": name,
                "leaveout_score": float(row.get("score", 0.0) or 0.0),
                "leaveout_gain": base_score - float(row.get("score", 0.0) or 0.0),
                "leaveout_top1_changes": int(row.get("top1_changes", 0) or 0),
                "leaveout_sample_changes": int(row.get("sample_changes", 0) or 0),
                "leaveout_kl_max": float(row.get("kl_max", 0.0) or 0.0),
                "worst_bucket": row.get("worst_bucket", {}),
            }
        )
    attribution.sort(key=lambda row: float(row["leaveout_gain"]), reverse=True)

    result = {
        "schema": "streamattn.seed_only_stress_attribution.v1",
        "base_layers": list(_parse_layers(args.base_layers)),
        "route_set": args.route_set,
        "prompt_file": args.prompt_file,
        "prompt_truncation_side": args.prompt_truncation_side,
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "failure_score": {
            "formula": (
                "10*top1_changes + 10*sample_changes + "
                "1000*max(0,KLmax-gate) + 100*max(0,logprob-gate) + "
                "5*max(0,4-top5_overlap_min)"
            ),
            "base_score": base_score,
        },
        "routes": route_summaries,
        "leaveout_attribution": attribution,
    }
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--base-layers", default="0,14,16,24,26,27,35")
    parser.add_argument("--add-layers", default="2,29")
    parser.add_argument("--route-set", choices=["leaveout", "single", "candidate", "full"], default="leaveout")
    parser.add_argument("--prompt-file", default="benchmarks/prompts/qwen3b_32k_stress_pack_v1_b8.jsonl")
    parser.add_argument("--prompt-truncation-side", choices=["left", "right"], default="left")
    parser.add_argument("--max-prompts", type=int, default=8)
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    result = run_attribution(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
