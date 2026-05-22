"""Profile downstream error for Gate-0 seed-only policies.

The earlier repair studies used per-head attention-output max error as the
safety target. This profiler asks a model-aware question: how much of that
head-local error survives the attention output projection ``o_proj``?

It loads captured post-RoPE Q/K/V tensors, computes dense and seed-only decode
outputs for the target layer, applies the model's attention ``o_proj``, and
compares:

* raw per-head attention-output error;
* per-head projected impact through the corresponding ``W_o`` slice;
* full post-``o_proj`` output error for fixed and repaired policies;
* repair rankings by raw error versus output-projection impact.

This is not a runtime benchmark. It is a safety-target / policy-economics
profiler.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_gate0_kv_group_repair_real import (  # noqa: E402
    parse_fixed_repair_policy,
    q_heads_for_kv_group,
)
from benchmarks.profile_gate0_true_gqa import _dense_true_gqa, _true_gqa_kv  # noqa: E402
from benchmarks.profile_real_llm_gate1_heads import (  # noqa: E402
    _attention_modules,
    _import_transformers,
)
from benchmarks.profile_stream_attn_gate0_wrapper import (  # noqa: E402
    _dtype,
    _load_tensor,
)
from stream_attention.kernels.gate0_seed_only_triton import (  # noqa: E402
    gate0_seed_only_attention_triton_forward,
)


def _parse_ints(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _error_summary(actual: torch.Tensor, expected: torch.Tensor) -> Dict[str, float]:
    diff = (actual - expected).detach().float()
    abs_diff = diff.abs()
    actual_l2 = torch.linalg.vector_norm(diff)
    expected_l2 = torch.linalg.vector_norm(expected.detach().float())
    return {
        "max_abs_error": float(abs_diff.max().item()),
        "mean_abs_error": float(abs_diff.mean().item()),
        "l2_error": float(actual_l2.item()),
        "relative_l2_error": float(actual_l2.item() / max(expected_l2.item(), 1.0e-6)),
    }


def _per_head_raw_error(candidate: torch.Tensor, dense: torch.Tensor) -> List[Dict[str, Any]]:
    rows = []
    for head in range(dense.shape[2]):
        err = _error_summary(candidate[:, :, head, :], dense[:, :, head, :])
        err["head"] = head
        rows.append(err)
    return rows


def _flatten_heads(heads: torch.Tensor) -> torch.Tensor:
    return heads.contiguous().view(heads.shape[0], heads.shape[1], heads.shape[2] * heads.shape[3])


def _project_attention_output(module: torch.nn.Module, heads: torch.Tensor) -> torch.Tensor:
    # Captured/seed tensors are often inference tensors. Clone to a normal
    # tensor so Linear does not try to save inference tensors for autograd.
    with torch.no_grad():
        flat = _flatten_heads(heads).detach().clone()
        return module.o_proj(flat)


def _wo_projected_per_head(
    *,
    module: torch.nn.Module,
    candidate: torch.Tensor,
    dense: torch.Tensor,
) -> List[Dict[str, Any]]:
    weight = module.o_proj.weight.detach().float()
    _batch, _query, heads, dim = dense.shape
    rows = []
    for head in range(heads):
        delta = (candidate[:, :, head, :] - dense[:, :, head, :]).detach().float().reshape(-1, dim)
        start = head * dim
        end = start + dim
        projected = delta @ weight[:, start:end].T
        projected_l2 = torch.linalg.vector_norm(projected)
        delta_l2 = torch.linalg.vector_norm(delta)
        rows.append(
            {
                "head": head,
                "raw_max_abs_error": float(delta.abs().max().item()),
                "raw_l2_error": float(delta_l2.item()),
                "wo_projected_max_abs_error": float(projected.abs().max().item()),
                "wo_projected_l2_error": float(projected_l2.item()),
            }
        )
    return rows


def _rank_heads(rows: Sequence[Dict[str, Any]], metric: str, candidates: Iterable[int]) -> List[int]:
    allowed = set(int(head) for head in candidates)
    return [
        int(row["head"])
        for row in sorted(rows, key=lambda item: float(item.get(metric) or 0.0), reverse=True)
        if int(row["head"]) in allowed
    ]


def _replace_heads(base: torch.Tensor, source: torch.Tensor, heads: Sequence[int]) -> torch.Tensor:
    if not heads:
        return base.clone()
    out = base.clone()
    index = torch.tensor(list(heads), device=base.device, dtype=torch.long)
    out.index_copy_(2, index, source.index_select(2, index))
    return out


def _policy_metrics(
    *,
    name: str,
    candidate: torch.Tensor,
    dense_heads: torch.Tensor,
    dense_post_wo: torch.Tensor,
    module: torch.nn.Module,
    seed_heads: Sequence[int],
    repair_heads: Sequence[int],
) -> Dict[str, Any]:
    post_wo = _project_attention_output(module, candidate)
    raw = _error_summary(candidate, dense_heads)
    post = _error_summary(post_wo, dense_post_wo)
    per_head = _per_head_raw_error(candidate, dense_heads)
    projected = _wo_projected_per_head(module=module, candidate=candidate, dense=dense_heads)
    return {
        "name": name,
        "seed_heads": list(seed_heads),
        "repair_heads": list(repair_heads),
        "raw_attention_error": raw,
        "post_o_proj_error": post,
        "per_head_raw_error": per_head,
        "per_head_wo_projected_error": projected,
        "top_heads_by_raw_l2": _rank_heads(projected, "raw_l2_error", range(dense_heads.shape[2])),
        "top_heads_by_wo_l2": _rank_heads(projected, "wo_projected_l2_error", range(dense_heads.shape[2])),
    }


def _repair_sweep(
    *,
    base_name: str,
    base_seed_heads: Sequence[int],
    dense_heads: torch.Tensor,
    seed_heads_all: torch.Tensor,
    dense_post_wo: torch.Tensor,
    module: torch.nn.Module,
    repair_counts: Sequence[int],
) -> List[Dict[str, Any]]:
    base = _replace_heads(dense_heads, seed_heads_all, base_seed_heads)
    projected = _wo_projected_per_head(module=module, candidate=base, dense=dense_heads)
    rankings = {
        "raw_l2": _rank_heads(projected, "raw_l2_error", base_seed_heads),
        "raw_max": _rank_heads(projected, "raw_max_abs_error", base_seed_heads),
        "wo_l2": _rank_heads(projected, "wo_projected_l2_error", base_seed_heads),
        "wo_max": _rank_heads(projected, "wo_projected_max_abs_error", base_seed_heads),
    }
    rows: List[Dict[str, Any]] = []
    for rank_name, order in rankings.items():
        for count in repair_counts:
            repair = order[: min(int(count), len(order))]
            candidate = _replace_heads(base, dense_heads, repair)
            rows.append(
                _policy_metrics(
                    name=f"{base_name}_repair_{rank_name}_top{len(repair)}",
                    candidate=candidate,
                    dense_heads=dense_heads,
                    dense_post_wo=dense_post_wo,
                    module=module,
                    seed_heads=[head for head in base_seed_heads if head not in set(repair)],
                    repair_heads=repair,
                )
            )
    return rows


def _load_attention_module(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    AutoModelForCausalLM, _AutoTokenizer = _import_transformers()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        use_safetensors=args.use_safetensors,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()
    modules = _attention_modules(model)
    if args.layer_id < 0 or args.layer_id >= len(modules):
        raise ValueError(f"layer_id={args.layer_id} outside available attention modules")
    _layer_id, module_name, module = modules[args.layer_id]
    return model, module_name, module


def profile(args: argparse.Namespace) -> Dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    dtype = _dtype(args.dtype)

    q = _load_tensor(args.q_path, key="q", device=device, dtype=dtype)
    k_expanded = _load_tensor(args.k_path, key="k", device=device, dtype=dtype)
    v_expanded = _load_tensor(args.v_path, key="v", device=device, dtype=dtype)
    k_true = _true_gqa_kv(k_expanded, true_kv_heads=args.true_kv_heads)
    v_true = _true_gqa_kv(v_expanded, true_kv_heads=args.true_kv_heads)
    if q.shape[0] != 1 or q.shape[1] != 1:
        raise ValueError("downstream profiler currently supports B=1, M=1")
    q_heads = int(q.shape[2])
    kv_heads = int(k_true.shape[2])
    if q_heads % kv_heads != 0:
        raise ValueError("q_heads must be divisible by true_kv_heads")

    print("[downstream-error] loading attention output projection", flush=True)
    model, module_name, module = _load_attention_module(args, device, dtype)
    del model
    if not hasattr(module, "o_proj"):
        raise ValueError(f"attention module {module_name} has no o_proj")

    print("[downstream-error] computing dense and seed-only attention outputs", flush=True)
    with torch.inference_mode():
        dense_heads = _dense_true_gqa(q, k_true, v_true)
        seed_heads_all, _ = gate0_seed_only_attention_triton_forward(
            q,
            k_true,
            v_true,
            block_size=args.block_size,
            sink_blocks=args.sink_blocks,
            recent_blocks=args.recent_blocks,
            middle_seed_blocks=args.middle_seed_blocks,
            block_order=args.block_order,
            return_raw_stats=False,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        )
        dense_post_wo = _project_attention_output(module, dense_heads)

    group_heads = {
        kv_head: q_heads_for_kv_group(kv_head, q_heads=q_heads, kv_heads=kv_heads)
        for kv_head in range(kv_heads)
    }
    all_heads = list(range(q_heads))
    seed_kv_groups = _parse_ints(args.seed_kv_groups)
    seed_heads_from_groups = sorted(
        {head for kv_head in seed_kv_groups for head in group_heads[kv_head]}
    )
    trusted_seed_heads = _parse_ints(args.trusted_seed_heads)
    repair_policy = parse_fixed_repair_policy(args.repair_policy)
    repair_heads = sorted({head for heads in repair_policy.values() for head in heads})
    trusted_policy_heads = (
        trusted_seed_heads
        if trusted_seed_heads
        else [head for head in seed_heads_from_groups if head not in set(repair_heads)]
    )

    fixed_policies: List[Dict[str, Any]] = []
    candidates = [
        ("dense_reference", dense_heads, [], all_heads),
        (
            f"seed_kv_groups_{'_'.join(str(x) for x in seed_kv_groups) or 'none'}",
            _replace_heads(dense_heads, seed_heads_all, seed_heads_from_groups),
            seed_heads_from_groups,
            [head for head in all_heads if head not in set(seed_heads_from_groups)],
        ),
        ("all_seed_only", seed_heads_all, all_heads, []),
        (
            "trusted_cross_prompt_policy",
            _replace_heads(dense_heads, seed_heads_all, trusted_policy_heads),
            trusted_policy_heads,
            [head for head in all_heads if head not in set(trusted_policy_heads)],
        ),
    ]
    for name, candidate, seed_heads, exact_heads in candidates:
        fixed_policies.append(
            _policy_metrics(
                name=name,
                candidate=candidate,
                dense_heads=dense_heads,
                dense_post_wo=dense_post_wo,
                module=module,
                seed_heads=seed_heads,
                repair_heads=exact_heads,
            )
        )

    repair_counts = _parse_ints(args.repair_counts)
    sweeps: Dict[str, List[Dict[str, Any]]] = {}
    for kv_head, heads in group_heads.items():
        sweeps[f"kv_group_{kv_head}_seed_repair"] = _repair_sweep(
            base_name=f"kv_group_{kv_head}_seed",
            base_seed_heads=heads,
            dense_heads=dense_heads,
            seed_heads_all=seed_heads_all,
            dense_post_wo=dense_post_wo,
            module=module,
            repair_counts=repair_counts,
        )
    sweeps["all_seed_repair"] = _repair_sweep(
        base_name="all_seed",
        base_seed_heads=all_heads,
        dense_heads=dense_heads,
        seed_heads_all=seed_heads_all,
        dense_post_wo=dense_post_wo,
        module=module,
        repair_counts=repair_counts,
    )

    base_seed_projected = _wo_projected_per_head(
        module=module,
        candidate=seed_heads_all,
        dense=dense_heads,
    )
    output_weight = module.o_proj.weight.detach()
    return {
        "schema": "streamattn.gate0.downstream_error.v1",
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "shape": {
            "batch": int(q.shape[0]),
            "query_len": int(q.shape[1]),
            "kv_len": int(k_true.shape[1]),
            "q_heads": q_heads,
            "true_kv_heads": kv_heads,
            "group_size": q_heads // kv_heads,
            "dim": int(q.shape[3]),
            "hidden_size": int(output_weight.shape[0]),
            "dtype": args.dtype,
        },
        "model": {
            "model_id": args.model,
            "layer_id": args.layer_id,
            "attention_module": module_name,
            "o_proj_weight_shape": list(output_weight.shape),
        },
        "seed_config": {
            "block_size": args.block_size,
            "sink_blocks": args.sink_blocks,
            "recent_blocks": args.recent_blocks,
            "middle_seed_blocks": args.middle_seed_blocks,
            "block_order": args.block_order,
        },
        "policy_inputs": {
            "seed_kv_groups": seed_kv_groups,
            "trusted_seed_heads": trusted_policy_heads,
            "repair_policy": repair_policy,
            "repair_counts": repair_counts,
        },
        "fixed_policies": fixed_policies,
        "repair_sweeps": sweeps,
        "all_seed_per_head_projected_impact": base_seed_projected,
        "logit_metrics": {
            "available": False,
            "reason": "This profiler currently stops at attention o_proj; residual/logit replay needs a separate model-forward hook.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-path", required=True)
    parser.add_argument("--k-path", required=True)
    parser.add_argument("--v-path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer-id", type=int, required=True)
    parser.add_argument("--true-kv-heads", type=int, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--seed-kv-groups", default="0")
    parser.add_argument("--trusted-seed-heads", default="2,3,4")
    parser.add_argument("--repair-policy", default="0:0,1,5,6;1:7,8,9,10,11,12,13")
    parser.add_argument("--repair-counts", default="0,1,2,3,4,7,11,14")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--recent-blocks", type=int, default=2)
    parser.add_argument("--middle-seed-blocks", type=int, default=8)
    parser.add_argument("--block-order", choices=["sequential", "recent_first", "sink_recent_first"], default="recent_first")
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--use-safetensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--summary-json-out", default="")
    args = parser.parse_args()

    result = profile(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.summary_json_out:
        path = Path(args.summary_json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
