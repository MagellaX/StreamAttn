"""Profile Qwen routed-layer projection floors.

This benchmark isolates the next model-level StreamAttn question:

    after native cache ownership, are routed layers dominated by real GEMM
    work or by avoidable Q/K/V module dispatch and projection layout overhead?

It compares the existing Hugging Face-style separate q_proj/k_proj/v_proj
path against direct ``F.linear`` calls and a single packed QKV projection.
The inputs are decode-shaped random tensors; the measured quantity is the
projection floor for routed attention modules, not model quality.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_real_llm_gate1_heads import _attention_modules, _import_transformers  # noqa: E402
from benchmarks.profile_stream_attn_gate0_wrapper import _dtype, _error, _sync, _time_cuda  # noqa: E402


def parse_layers(raw: str) -> List[int]:
    layers: list[int] = []
    seen: set[int] = set()
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        layer_id = int(part)
        if layer_id < 0:
            raise ValueError("layer ids must be non-negative")
        if layer_id not in seen:
            layers.append(layer_id)
            seen.add(layer_id)
    if not layers:
        raise ValueError("at least one layer id is required")
    return layers


def parse_batch_sizes(raw: str) -> List[int]:
    batches: list[int] = []
    seen: set[int] = set()
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        batch = int(part)
        if batch <= 0:
            raise ValueError("batch sizes must be positive")
        if batch not in seen:
            batches.append(batch)
            seen.add(batch)
    if not batches:
        raise ValueError("at least one batch size is required")
    return batches


def _linear_bias(linear: torch.nn.Module) -> Optional[torch.Tensor]:
    bias = getattr(linear, "bias", None)
    return bias if isinstance(bias, torch.Tensor) else None


def packed_qkv_params(module: torch.nn.Module) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[int, int, int]]:
    """Return packed QKV weight/bias and split sizes for a Qwen attention module."""

    weights = [module.q_proj.weight, module.k_proj.weight, module.v_proj.weight]
    weight = torch.cat(weights, dim=0).contiguous()
    biases = [_linear_bias(module.q_proj), _linear_bias(module.k_proj), _linear_bias(module.v_proj)]
    if any(bias is not None for bias in biases):
        packed_biases = [
            bias
            if bias is not None
            else torch.zeros(
                linear.out_features,
                device=weight.device,
                dtype=weight.dtype,
            )
            for bias, linear in zip(biases, (module.q_proj, module.k_proj, module.v_proj))
        ]
        bias = torch.cat(packed_biases, dim=0).contiguous()
    else:
        bias = None
    sizes = (
        int(module.q_proj.out_features),
        int(module.k_proj.out_features),
        int(module.v_proj.out_features),
    )
    return weight, bias, sizes


def _first_attr(module: torch.nn.Module, names: Iterable[str], default: Any) -> Any:
    for name in names:
        if hasattr(module, name):
            value = getattr(module, name)
            if value is not None:
                return value
    return default


def _infer_shape(module: torch.nn.Module) -> Dict[str, int]:
    hidden_size = int(module.q_proj.in_features)
    q_out = int(module.q_proj.out_features)
    k_out = int(module.k_proj.out_features)
    v_out = int(module.v_proj.out_features)
    head_dim = int(_first_attr(module, ("head_dim",), 0))
    if head_dim <= 0:
        num_heads = int(_first_attr(module, ("num_heads",), 0))
        head_dim = q_out // num_heads if num_heads > 0 else 128
    q_heads = int(_first_attr(module, ("num_heads", "num_attention_heads"), q_out // head_dim))
    kv_heads = int(_first_attr(module, ("num_key_value_heads", "num_kv_heads"), k_out // head_dim))
    return {
        "hidden_size": hidden_size,
        "q_out": q_out,
        "k_out": k_out,
        "v_out": v_out,
        "qkv_out": q_out + k_out + v_out,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "group_size": q_heads // max(1, kv_heads),
    }


def _tensor_errors(
    actual: Sequence[torch.Tensor],
    expected: Sequence[torch.Tensor],
) -> Dict[str, Dict[str, float]]:
    names = ("q", "k", "v")
    return {name: _error(a, e) for name, a, e in zip(names, actual, expected)}


def _time_layer_batch(
    *,
    module: torch.nn.Module,
    layer_id: int,
    module_name: str,
    batch: int,
    dtype_name: str,
    device: torch.device,
    warmup: int,
    iters: int,
    seed: int,
) -> Dict[str, Any]:
    shape = _infer_shape(module)
    torch.manual_seed(seed + layer_id * 1009 + batch)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + layer_id * 1009 + batch)

    hidden = torch.randn(batch, 1, shape["hidden_size"], device=device, dtype=_dtype(dtype_name))
    attn_flat = torch.randn(batch, 1, shape["q_out"], device=device, dtype=_dtype(dtype_name))
    qkv_weight, qkv_bias, qkv_sizes = packed_qkv_params(module)

    module.eval()
    for linear in (module.q_proj, module.k_proj, module.v_proj, module.o_proj):
        linear.eval()

    def qkv_module() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return module.q_proj(hidden), module.k_proj(hidden), module.v_proj(hidden)

    def qkv_module_views() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = qkv_module()
        q = q.view(batch, 1, shape["q_heads"], shape["head_dim"]).transpose(1, 2)
        k = k.view(batch, 1, shape["kv_heads"], shape["head_dim"]).transpose(1, 2)
        v = v.view(batch, 1, shape["kv_heads"], shape["head_dim"]).transpose(1, 2)
        return q, k, v

    def qkv_flinear() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            F.linear(hidden, module.q_proj.weight, _linear_bias(module.q_proj)),
            F.linear(hidden, module.k_proj.weight, _linear_bias(module.k_proj)),
            F.linear(hidden, module.v_proj.weight, _linear_bias(module.v_proj)),
        )

    def qkv_packed() -> torch.Tensor:
        return F.linear(hidden, qkv_weight, qkv_bias)

    def qkv_packed_split() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.split(qkv_packed(), qkv_sizes, dim=-1)

    def qkv_packed_views() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = qkv_packed_split()
        q = q.view(batch, 1, shape["q_heads"], shape["head_dim"]).transpose(1, 2)
        k = k.view(batch, 1, shape["kv_heads"], shape["head_dim"]).transpose(1, 2)
        v = v.view(batch, 1, shape["kv_heads"], shape["head_dim"]).transpose(1, 2)
        return q, k, v

    def o_module() -> torch.Tensor:
        return module.o_proj(attn_flat)

    def o_flinear() -> torch.Tensor:
        return F.linear(attn_flat, module.o_proj.weight, _linear_bias(module.o_proj))

    with torch.inference_mode():
        qkv_ref = tuple(t.detach().clone() for t in qkv_module())
        qkv_flinear_ref = tuple(t.detach().clone() for t in qkv_flinear())
        qkv_packed_ref = tuple(t.detach().clone() for t in qkv_packed_split())
        o_ref = o_module().detach().clone()
        o_flinear_ref = o_flinear().detach().clone()
        _sync(device)

        qkv_module_ms = _time_cuda(qkv_module, device=device, warmup=warmup, iters=iters)
        qkv_module_views_ms = _time_cuda(qkv_module_views, device=device, warmup=warmup, iters=iters)
        qkv_flinear_ms = _time_cuda(qkv_flinear, device=device, warmup=warmup, iters=iters)
        qkv_packed_ms = _time_cuda(qkv_packed, device=device, warmup=warmup, iters=iters)
        qkv_packed_views_ms = _time_cuda(qkv_packed_views, device=device, warmup=warmup, iters=iters)
        o_module_ms = _time_cuda(o_module, device=device, warmup=warmup, iters=iters)
        o_flinear_ms = _time_cuda(o_flinear, device=device, warmup=warmup, iters=iters)

    qkv_packed_total_ms = qkv_packed_views_ms
    module_projection_floor_ms = qkv_module_views_ms + o_module_ms
    packed_projection_floor_ms = qkv_packed_total_ms + o_flinear_ms
    return {
        "layer_id": int(layer_id),
        "module_name": module_name,
        "batch": int(batch),
        "shape": shape,
        "timing": {
            "qkv_module_ms": qkv_module_ms,
            "qkv_module_with_decode_views_ms": qkv_module_views_ms,
            "qkv_flinear_separate_ms": qkv_flinear_ms,
            "qkv_packed_flinear_ms": qkv_packed_ms,
            "qkv_packed_with_decode_views_ms": qkv_packed_views_ms,
            "o_module_ms": o_module_ms,
            "o_flinear_ms": o_flinear_ms,
            "module_qkv_o_floor_ms": module_projection_floor_ms,
            "packed_qkv_o_floor_ms": packed_projection_floor_ms,
            "qkv_packed_speedup_vs_module": qkv_module_views_ms / qkv_packed_views_ms,
            "o_flinear_speedup_vs_module": o_module_ms / o_flinear_ms,
            "packed_floor_speedup_vs_module_floor": module_projection_floor_ms
            / packed_projection_floor_ms,
            "packed_floor_savings_ms": module_projection_floor_ms - packed_projection_floor_ms,
        },
        "errors": {
            "qkv_flinear_vs_module": _tensor_errors(qkv_flinear_ref, qkv_ref),
            "qkv_packed_vs_module": _tensor_errors(qkv_packed_ref, qkv_ref),
            "o_flinear_vs_module": _error(o_flinear_ref, o_ref),
        },
    }


def _summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_batch: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_batch.setdefault(int(row["batch"]), []).append(row)
    summary_by_batch = {}
    for batch, batch_rows in sorted(by_batch.items()):
        timings = [row["timing"] for row in batch_rows]
        module_sum = sum(item["module_qkv_o_floor_ms"] for item in timings)
        packed_sum = sum(item["packed_qkv_o_floor_ms"] for item in timings)
        summary_by_batch[str(batch)] = {
            "layers": [int(row["layer_id"]) for row in batch_rows],
            "layer_count": len(batch_rows),
            "route_module_qkv_o_floor_ms": module_sum,
            "route_packed_qkv_o_floor_ms": packed_sum,
            "route_packed_projection_savings_ms": module_sum - packed_sum,
            "route_packed_projection_speedup": module_sum / packed_sum,
            "mean_layer_qkv_module_views_ms": mean(
                item["qkv_module_with_decode_views_ms"] for item in timings
            ),
            "mean_layer_qkv_packed_views_ms": mean(
                item["qkv_packed_with_decode_views_ms"] for item in timings
            ),
            "mean_layer_o_module_ms": mean(item["o_module_ms"] for item in timings),
            "mean_layer_o_flinear_ms": mean(item["o_flinear_ms"] for item in timings),
            "mean_layer_packed_floor_savings_ms": mean(
                item["packed_floor_savings_ms"] for item in timings
            ),
        }
    best = max(
        rows,
        key=lambda row: row["timing"]["packed_floor_speedup_vs_module_floor"],
    )
    return {
        "by_batch": summary_by_batch,
        "best_layer_batch": {
            "layer_id": int(best["layer_id"]),
            "batch": int(best["batch"]),
            "timing": best["timing"],
        },
    }


def profile(args: argparse.Namespace) -> Dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    dtype = _dtype(args.dtype)
    layers = parse_layers(args.layers)
    batches = parse_batch_sizes(args.batch_sizes)

    AutoModelForCausalLM, _ = _import_transformers()
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "use_safetensors": args.use_safetensors,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).to(device)
    model.eval()

    modules_by_layer = {layer_id: (name, module) for layer_id, name, module in _attention_modules(model)}
    missing = sorted(set(layers) - set(modules_by_layer))
    if missing:
        raise ValueError(f"model has no attention modules for layers: {missing}")

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in batches:
            for layer_id in layers:
                name, module = modules_by_layer[layer_id]
                rows.append(
                    _time_layer_batch(
                        module=module,
                        layer_id=layer_id,
                        module_name=name,
                        batch=batch,
                        dtype_name=args.dtype,
                        device=device,
                        warmup=args.warmup,
                        iters=args.iters,
                        seed=args.seed,
                    )
                )

    return {
        "schema": "streamattn.qwen_routed_projection_floor.v1",
        "model": args.model,
        "device": str(device),
        "dtype": args.dtype,
        "layers": layers,
        "batch_sizes": batches,
        "summary": _summarize_rows(rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--layers", default="0,14,16,24,26,27,35")
    parser.add_argument("--batch-sizes", default="4,8,16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument("--no-use-safetensors", dest="use_safetensors", action="store_false")
    parser.set_defaults(use_safetensors=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234)
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
