"""Actual-model marginal optimizer for seed-only decode routes.

This benchmark ranks route layers by full-model decode utility, not by isolated
attention-call speed.  It loads the model once, builds one dense decode
reference, then evaluates route cases such as:

* current base bundle
* base bundle plus each candidate layer
* base bundle with each layer removed
* optional single-layer routes

Each candidate still runs the actual `use_cache=True` model decode loop with
patched attention modules, so timings include model-runner overhead, cache
updates, MLPs, norms, and non-routed layers.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_gate0_seed_only_closed_loop_rollout import _prompts_from_args  # noqa: E402
from benchmarks.profile_seed_only_route_bundle_decode import (  # noqa: E402
    _batch_tokens,
    _compare_decode_logits,
    _decode_loop,
    _native_cache_from_hf_cache,
    _native_cache_mask_bookkeeping,
    _native_cache_max_len,
    _patched_seed_only_decode_modules,
    _prefill,
    _safety_decision,
    _warmup_decode,
    parse_layer_id_set,
)
from benchmarks.profile_real_llm_gate1_heads import _import_transformers  # noqa: E402
from benchmarks.profile_stream_attn_gate0_wrapper import _dtype  # noqa: E402
from stream_attention.decode import Gate0SeedOnlyBatchedPolicy  # noqa: E402


def _cuda_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


@dataclass(frozen=True)
class MarginalCase:
    name: str
    kind: str
    layers: List[int]
    candidate_layer: Optional[int] = None


@dataclass(frozen=True)
class SimpleRouteBundle:
    policy_names: List[str]
    policies: List[Gate0SeedOnlyBatchedPolicy]
    artifacts: List[Dict[str, Any]]
    layer_ids: List[int]


def _parse_layer_ids(text: str) -> List[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    return sorted(set(values))


def _parse_case_modes(text: str) -> List[str]:
    allowed = {"base", "add", "leave_one_out", "single"}
    modes = [part.strip() for part in text.split(",") if part.strip()]
    unknown = sorted(set(modes) - allowed)
    if unknown:
        raise ValueError(f"unknown case modes: {unknown}")
    return modes


def build_marginal_cases(
    *,
    base_layers: Sequence[int],
    candidate_layers: Sequence[int],
    modes: Sequence[str],
) -> List[MarginalCase]:
    base = sorted(set(int(layer) for layer in base_layers))
    candidates = sorted(set(int(layer) for layer in candidate_layers))
    cases: List[MarginalCase] = []
    seen: set[tuple[int, ...]] = set()

    def add_case(case: MarginalCase) -> None:
        key = tuple(case.layers)
        if not key or key in seen:
            return
        seen.add(key)
        cases.append(case)

    if "base" in modes:
        add_case(MarginalCase(name="base", kind="base", layers=base))
    if "add" in modes:
        for layer in candidates:
            if layer in base:
                continue
            layers = sorted(set([*base, layer]))
            add_case(MarginalCase(name=f"base_plus_l{layer}", kind="add", layers=layers, candidate_layer=layer))
    if "leave_one_out" in modes:
        for layer in base:
            layers = [value for value in base if value != layer]
            add_case(
                MarginalCase(
                    name=f"base_minus_l{layer}",
                    kind="leave_one_out",
                    layers=layers,
                    candidate_layer=layer,
                )
            )
    if "single" in modes:
        for layer in candidates:
            add_case(MarginalCase(name=f"single_l{layer}", kind="single", layers=[layer], candidate_layer=layer))
    return cases


def _make_policy(args: argparse.Namespace, layer_id: int) -> Gate0SeedOnlyBatchedPolicy:
    return Gate0SeedOnlyBatchedPolicy(
        policy_id=f"ad_hoc_model_decode_l{layer_id}_s{args.block_size}_{args.sink_blocks}_{args.recent_blocks}_{args.middle_seed_blocks}",
        model_id=args.model,
        layer_id=layer_id,
        dtype=args.dtype,
        kv_len_bucket=args.max_seq,
        min_batch=args.batch_size,
        heads=args.q_heads,
        kv_heads=args.kv_heads,
        dim=args.head_dim,
        block_size=args.block_size,
        sink_blocks=args.sink_blocks,
        recent_blocks=args.recent_blocks,
        middle_seed_blocks=args.middle_seed_blocks,
        block_order=args.block_order,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
        max_kl=args.max_kl,
        min_topk_overlap=args.min_topk_overlap,
        max_logprob_delta=args.max_logprob_delta,
    )


def _bundle_for_layers(args: argparse.Namespace, layers: Sequence[int]) -> SimpleRouteBundle:
    policies = [_make_policy(args, layer_id) for layer_id in sorted(set(layers))]
    return SimpleRouteBundle(
        policy_names=[policy.policy_id for policy in policies],
        policies=policies,
        artifacts=[],
        layer_ids=[int(policy.layer_id) for policy in policies],
    )


def _patch_counts(patches) -> Dict[str, Dict[str, Any]]:
    return {
        layer_id: {
            "forward_calls": patch.call_count,
            "seed_only_decode_calls": patch.seed_call_count,
            "native_cache_update_calls": patch.native_cache_update_count,
            "hf_sync_update_calls": patch.hf_sync_update_count,
            "fused_rope_append_seed": bool(patch.fused_rope_append_seed),
            "packed_qkv_projection": bool(patch.packed_qkv_projection),
            "packed_qkv_fused_input": bool(patch.packed_qkv_fused_input),
            "direct_o_proj": bool(patch.direct_o_proj),
            "triton_o_proj": bool(patch.triton_o_proj),
            "fallback_reasons": dict(patch.fallback_reasons),
            "fallback_samples": patch.fallback_samples,
        }
        for layer_id, patch in patches.items()
    }


def _case_metric(
    *,
    case: MarginalCase,
    dense_total_ms: float,
    dense_logits: Sequence[torch.Tensor],
    seed_result: Dict[str, Any],
    patch_counts: Dict[str, Any],
    prompt_rows: Sequence[Dict[str, str]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    comparison = _compare_decode_logits(
        dense_logits,
        seed_result["logits_by_step"],
        prompt_rows=prompt_rows,
        args=args,
    )
    safety = comparison["summary"]
    decision = _safety_decision(safety, args=args)
    total_ms = float(seed_result["total_ms"])
    saved_ms = dense_total_ms - total_ms
    return {
        "name": case.name,
        "kind": case.kind,
        "candidate_layer": case.candidate_layer,
        "layers": case.layers,
        "layer_count": len(case.layers),
        "total_ms": total_ms,
        "ms_per_token": float(seed_result["ms_per_token"]),
        "speedup_vs_dense": dense_total_ms / max(total_ms, 1.0e-12),
        "saved_ms_total": saved_ms,
        "saved_ms_per_token": saved_ms / max(1, int(args.steps)),
        "safety": safety,
        "decision": decision,
        "patch_counts": patch_counts,
    }


def score_cases(cases: Sequence[Dict[str, Any]], *, base_name: str = "base") -> List[Dict[str, Any]]:
    by_name = {case["name"]: case for case in cases}
    base = by_name.get(base_name)
    scored: List[Dict[str, Any]] = []
    for case in cases:
        row = dict(case)
        row["marginal_vs_base_ms_total"] = None
        row["marginal_vs_base_kl"] = None
        row["recommendation"] = "observe"
        if base is not None and case["name"] != base_name:
            row["marginal_vs_base_ms_total"] = float(base["total_ms"]) - float(case["total_ms"])
            row["marginal_vs_base_kl"] = float(case["safety"]["kl_max"]) - float(base["safety"]["kl_max"])
            if case.get("kind") == "leave_one_out":
                if not case["decision"]["passed"]:
                    row["recommendation"] = "keep_layer_safety"
                elif row["marginal_vs_base_ms_total"] > 0:
                    row["recommendation"] = "candidate_remove"
                else:
                    row["recommendation"] = "keep_layer_runtime"
            elif not case["decision"]["passed"]:
                row["recommendation"] = "reject_safety"
            elif row["marginal_vs_base_ms_total"] <= 0:
                row["recommendation"] = "reject_runtime"
            else:
                row["recommendation"] = "candidate_add"
        elif case["decision"]["passed"] and float(case["saved_ms_total"]) > 0:
            row["recommendation"] = "keep"
        elif not case["decision"]["passed"]:
            row["recommendation"] = "reject_safety"
        else:
            row["recommendation"] = "reject_runtime"
        scored.append(row)
    return scored


def _coverage_targets(*, speedup: float, region_speedup: float, targets: Iterable[float]) -> Dict[str, Any]:
    def eff(total_speedup: float) -> float:
        return (1.0 - (1.0 / total_speedup)) / (1.0 - (1.0 / region_speedup))

    return {
        "assumed_region_speedup": region_speedup,
        "effective_fraction": eff(speedup),
        "target_required_fractions": [
            {
                "target_speedup": target,
                "required_routed_fraction": eff(float(target)),
            }
            for target in targets
        ],
    }


def _run_case(
    *,
    model,
    tokens: Dict[str, torch.Tensor],
    first_token: torch.Tensor,
    dense_input_tokens: Sequence[torch.Tensor],
    dense_logits: Sequence[torch.Tensor],
    prompt_rows: Sequence[Dict[str, str]],
    dense_total_ms: float,
    case: MarginalCase,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    bundle = _bundle_for_layers(args, case.layers)
    _cuda_cleanup()
    prefill = _prefill(model, tokens)
    native_cache = None
    if args.native_routed_cache:
        native_cache = _native_cache_from_hf_cache(
            prefill.past_key_values,
            bundle,
            max_len=_native_cache_max_len(tokens, args),
            attach_hf_views=args.native_cache_attach_hf_views,
        )
    with _native_cache_mask_bookkeeping(prefill.past_key_values, enabled=args.native_routed_cache):
        with _patched_seed_only_decode_modules(
            model,
            bundle,
            native_cache=native_cache,
            native_cache_hf_sync_layers=parse_layer_id_set(args.native_cache_hf_sync_layers),
            native_attention_module=args.native_attention_module,
            fused_rope_append_seed=args.fused_rope_append_seed,
            packed_qkv_projection=args.packed_qkv_projection,
            packed_qkv_fused_input=args.packed_qkv_fused_input,
            direct_o_proj=args.direct_o_proj,
            triton_o_proj=args.triton_o_proj,
        ) as patches:
            seed = _decode_loop(
                model=model,
                past_key_values=prefill.past_key_values,
                attention_mask=tokens["attention_mask"],
                first_token=first_token,
                fixed_input_tokens=dense_input_tokens,
                prompt_rows=prompt_rows,
                args=args,
            )
    result = _case_metric(
        case=case,
        dense_total_ms=dense_total_ms,
        dense_logits=dense_logits,
        seed_result=seed,
        patch_counts=_patch_counts(patches),
        prompt_rows=prompt_rows,
        args=args,
    )
    del prefill
    del seed
    del native_cache
    del bundle
    _cuda_cleanup()
    return result


def profile(args: argparse.Namespace) -> Dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    prompt_rows = _prompts_from_args(args)
    base_layers = _parse_layer_ids(args.base_layers)
    candidate_layers = _parse_layer_ids(args.candidate_layers)
    case_modes = _parse_case_modes(args.case_modes)
    cases = build_marginal_cases(base_layers=base_layers, candidate_layers=candidate_layers, modes=case_modes)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    AutoModelForCausalLM, AutoTokenizer = _import_transformers()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {
        "torch_dtype": dtype,
        "use_safetensors": args.use_safetensors,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).to(device)
    model.eval()
    tokens = _batch_tokens(tokenizer, prompt_rows, max_seq=args.max_seq, device=device)

    warmup: Dict[str, Any] = {"steps": int(args.warmup_steps)}
    if args.warmup_steps > 0:
        print("[marginals] warmup dense decode", flush=True)
        dense_warmup = _warmup_decode(
            model=model,
            tokens=tokens,
            prompt_rows=prompt_rows,
            args=args,
            bundle=None,
        )
        warmup["dense_ms_per_token"] = dense_warmup["decode"]["ms_per_token"]
        del dense_warmup
        _cuda_cleanup()

        union_layers = sorted(set(layer for case in cases for layer in case.layers))
        print(f"[marginals] warmup seed route for union layers {union_layers}", flush=True)
        seed_warmup = _warmup_decode(
            model=model,
            tokens=tokens,
            prompt_rows=prompt_rows,
            args=args,
            bundle=_bundle_for_layers(args, union_layers),
        )
        warmup["seed_union_ms_per_token"] = seed_warmup["decode"]["ms_per_token"]
        warmup["seed_union_patch_counts"] = seed_warmup["patch_counts"]
        del seed_warmup
        _cuda_cleanup()

    print("[marginals] dense reference prefill/decode", flush=True)
    dense_prefill = _prefill(model, tokens)
    first_token = torch.argmax(dense_prefill.logits[:, -1, :], dim=-1, keepdim=True)
    dense = _decode_loop(
        model=model,
        past_key_values=dense_prefill.past_key_values,
        attention_mask=tokens["attention_mask"],
        first_token=first_token,
        fixed_input_tokens=None,
        prompt_rows=prompt_rows,
        args=args,
    )
    del dense_prefill
    _cuda_cleanup()

    case_results: List[Dict[str, Any]] = []
    for case in cases:
        print(f"[marginals] case {case.name}: layers={case.layers}", flush=True)
        case_results.append(
            _run_case(
                model=model,
                tokens=tokens,
                first_token=first_token,
                dense_input_tokens=dense["input_tokens"],
                dense_logits=dense["logits_by_step"],
                prompt_rows=prompt_rows,
                dense_total_ms=float(dense["total_ms"]),
                case=case,
                args=args,
            )
        )

    scored = score_cases(case_results)
    base_case = next((case for case in scored if case["name"] == "base"), None)
    coverage = None
    if base_case is not None:
        coverage = _coverage_targets(
            speedup=float(base_case["speedup_vs_dense"]),
            region_speedup=float(args.assumed_region_speedup),
            targets=[float(row) for row in args.target_speedups.split(",") if row.strip()],
        )

    return {
        "schema": "streamattn.seed_only_model_decode_marginals.v1",
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "model": {
            "model_id": args.model,
            "attn_implementation": args.attn_implementation or "default",
        },
        "shape": {
            "batch": len(prompt_rows),
            "prompt_seq_len": int(tokens["input_ids"].shape[1]),
            "steps": int(args.steps),
            "dtype": args.dtype,
        },
        "seed_config": {
            "block_size": args.block_size,
            "sink_blocks": args.sink_blocks,
            "recent_blocks": args.recent_blocks,
            "middle_seed_blocks": args.middle_seed_blocks,
            "block_order": args.block_order,
        },
        "runtime_config": {
            "native_routed_cache": bool(args.native_routed_cache),
            "native_cache_hf_sync_layers": sorted(parse_layer_id_set(args.native_cache_hf_sync_layers)),
            "native_cache_attach_hf_views": bool(args.native_cache_attach_hf_views),
            "native_attention_module": bool(args.native_attention_module),
            "fused_rope_append_seed": bool(args.fused_rope_append_seed),
            "packed_qkv_projection": bool(args.packed_qkv_projection or args.native_attention_module),
            "packed_qkv_fused_input": bool(args.packed_qkv_fused_input),
            "direct_o_proj": bool(args.direct_o_proj),
            "triton_o_proj": bool(args.triton_o_proj),
        },
        "base_layers": base_layers,
        "candidate_layers": candidate_layers,
        "case_modes": case_modes,
        "dense": {
            "total_ms": dense["total_ms"],
            "ms_per_token": dense["ms_per_token"],
        },
        "warmup": warmup,
        "cases": scored,
        "coverage": coverage,
        "prompts": [{"row": idx, "kind": row["kind"]} for idx, row in enumerate(prompt_rows)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-layers", default="0,14,16,24,26,27,35")
    parser.add_argument("--candidate-layers", default="0,2,14,16,18,24,26,27,29,35")
    parser.add_argument("--case-modes", default="base,add,leave_one_out")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--prompt-kinds", default="needle,code,long_doc,chat_doc")
    parser.add_argument("--prompt-repeat", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--max-seq", type=int, default=32768)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument("--sample-temperature", type=float, default=0.8)
    parser.add_argument("--sample-top-p", type=float, default=0.95)
    parser.add_argument("--sample-top-k", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=1234)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--recent-blocks", type=int, default=2)
    parser.add_argument("--middle-seed-blocks", type=int, default=8)
    parser.add_argument("--block-order", choices=["sequential", "recent_first", "sink_recent_first"], default="recent_first")
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-kl", type=float, default=1.0e-4)
    parser.add_argument("--min-topk-overlap", type=int, default=4)
    parser.add_argument("--max-logprob-delta", type=float, default=2.0e-3)
    parser.add_argument("--assumed-region-speedup", type=float, default=3.0)
    parser.add_argument("--target-speedups", default="1.05,1.10,1.20")
    parser.add_argument("--use-safetensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--explicit-cache-position", action="store_true")
    parser.add_argument("--native-routed-cache", action="store_true")
    parser.add_argument("--native-cache-hf-sync-layers", default="")
    parser.add_argument("--native-cache-attach-hf-views", action="store_true")
    parser.add_argument("--native-attention-module", action="store_true")
    parser.add_argument("--fused-rope-append-seed", action="store_true")
    parser.add_argument("--packed-qkv-projection", action="store_true")
    parser.add_argument("--packed-qkv-fused-input", action="store_true")
    parser.add_argument("--direct-o-proj", action="store_true")
    parser.add_argument("--triton-o-proj", action="store_true")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    result = profile(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
