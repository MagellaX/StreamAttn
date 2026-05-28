"""Evaluate cheap dynamic seed selector proxies on stress coverage rows.

This is a thin, purpose-specific wrapper around the prompt-aware attention
coverage profiler.  It defaults to selector profiles that can plausibly become
runtime selectors, plus qk/exact/value oracles as ceilings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_seed_policy_attention_coverage import profile  # noqa: E402


DEFAULT_SELECTOR_PROFILES = ",".join(
    [
        "fixed_policy",
        "block_mean_proxy",
        "block_l2_bound_proxy",
        "support_top2_norm",
        "support_top4_norm",
        "support_top2_norm_refine16",
        "support_top2_norm_refine32",
        "support_top4_norm_refine16",
        "support_top4_norm_refine32",
        "qk_block_max",
        "exact_mass_oracle",
        "value_residual_oracle",
    ]
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--prompt-file", default="benchmarks/prompts/qwen3b_32k_stress_pack_v1_b8.jsonl")
    parser.add_argument("--prompt-truncation-side", choices=["left", "right"], default="left")
    parser.add_argument("--max-prompts", type=int, default=8)
    parser.add_argument("--prompt-kinds", default="needle,code,long_doc,chat_doc")
    parser.add_argument("--prompt-repeat", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq", type=int, default=32768)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--target-layers", default="26,27")
    parser.add_argument("--routed-layers", default="0,14,16,24,26,27,35")
    parser.add_argument("--target-buckets", default="chat_instruction,noisy_neartie,json_tool,needle_rag")
    parser.add_argument("--selector-profiles", default=DEFAULT_SELECTOR_PROFILES)
    parser.add_argument(
        "--failure-artifact",
        default="artifacts/gate0/qwen25_3b_32k_b8_model_decode/strict7_stress_pack_left_b8_h100.json",
    )
    parser.add_argument("--include-step0", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--use-safetensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    result = profile(args)
    result["schema"] = "streamattn.seed_selector_proxy.v1"
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
