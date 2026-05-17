"""Profile Gate-1 active fractions on real Hugging Face LLM activations.

This benchmark answers the BLASST-vs-certification and head-heterogeneity
questions:

* which layers/heads have low active PV fractions under mass mode;
* where value-bound mode is stricter than mass mode;
* whether per-head grouped routing has enough oracle upside;
* how much output error mass/value-bound introduce versus dense SDPA.

It is intentionally optional-dependency friendly. Normal repo tests do not
require ``transformers``; install it only on machines used for this benchmark.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch

from stream_attention import StreamAttnMetadataCache, dense_attention_forward, stream_attn_gate1


@dataclass
class CapturedAttentionInput:
    layer_id: int
    module_name: str
    module: torch.nn.Module
    hidden_states: torch.Tensor
    kwargs: dict


def _import_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "profile_real_llm_gate1_heads.py requires transformers. "
            "Install transformers on the profiling host."
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer


def _parse_int_list(raw: Optional[str]) -> Optional[set[int]]:
    if raw is None or raw.strip() == "":
        return None
    values = set()
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.add(int(item))
    return values


def _load_prompts(args) -> list[str]:
    prompts: list[str] = []
    if args.prompt:
        prompts.extend(args.prompt)
    if args.prompt_file:
        path = Path(args.prompt_file)
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            if path.suffix.lower() == ".jsonl":
                payload = json.loads(line)
                prompts.append(str(payload.get("text") or payload.get("prompt") or ""))
            else:
                prompts.append(line)
    if not prompts:
        prompts = [
            "In a long technical report about attention kernels, explain why "
            "hardware-aligned sparse routing matters for long-context inference."
        ]
    return prompts[: args.max_prompts]


def _attention_modules(model: torch.nn.Module) -> list[tuple[int, str, torch.nn.Module]]:
    modules = []
    for name, module in model.named_modules():
        has_qkv = all(hasattr(module, attr) for attr in ("q_proj", "k_proj", "v_proj"))
        if not has_qkv:
            continue
        layer_id = len(modules)
        modules.append((layer_id, name, module))
    return modules


def _capture_attention_inputs(model: torch.nn.Module, selected_layers: Optional[set[int]]):
    captured: list[CapturedAttentionInput] = []
    handles = []
    for layer_id, name, module in _attention_modules(model):
        if selected_layers is not None and layer_id not in selected_layers:
            continue

        def hook(mod, args, kwargs, *, _layer_id=layer_id, _name=name):
            if args:
                hidden_states = args[0]
            else:
                hidden_states = kwargs.get("hidden_states")
            if hidden_states is None:
                return
            captured.append(
                CapturedAttentionInput(
                    layer_id=_layer_id,
                    module_name=_name,
                    module=mod,
                    hidden_states=hidden_states.detach(),
                    kwargs=dict(kwargs),
                )
            )

        handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
    return captured, handles


def _first_attr(module: torch.nn.Module, names: Iterable[str], default=None):
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    config = getattr(module, "config", None)
    if config is not None:
        for name in names:
            if hasattr(config, name):
                return getattr(config, name)
    return default


def _shape_qkv(
    capture: CapturedAttentionInput,
    *,
    apply_rope: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    module = capture.module
    hidden = capture.hidden_states
    batch, seq_len, _ = hidden.shape
    num_heads = int(_first_attr(module, ["num_heads", "num_attention_heads"]))
    num_kv_heads = int(
        _first_attr(
            module,
            ["num_key_value_heads", "num_kv_heads"],
            num_heads,
        )
    )
    head_dim = int(_first_attr(module, ["head_dim"], module.q_proj.out_features // num_heads))

    q = module.q_proj(hidden).view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    k = module.k_proj(hidden).view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v = module.v_proj(hidden).view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    rope_applied = False
    rope_error = None
    if apply_rope:
        try:
            q, k = _apply_rope_if_available(capture, q, k)
            rope_applied = True
        except Exception as exc:  # pragma: no cover - adapter/version dependent
            rope_error = str(exc)

    if num_kv_heads != num_heads:
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads={num_heads} is not divisible by num_kv_heads={num_kv_heads}"
            )
        repeat = num_heads // num_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    else:
        repeat = 1

    # Gate-1 uses [batch, seq, heads, dim].
    q_bshd = q.transpose(1, 2).contiguous()
    k_bshd = k.transpose(1, 2).contiguous()
    v_bshd = v.transpose(1, 2).contiguous()
    meta = {
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "q_per_kv": repeat,
        "head_dim": head_dim,
        "rope_applied": rope_applied,
        "rope_error": rope_error,
    }
    return q_bshd, k_bshd, v_bshd, meta


def _apply_rope_if_available(
    capture: CapturedAttentionInput,
    q: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    kwargs = capture.kwargs
    position_embeddings = kwargs.get("position_embeddings")
    if position_embeddings is not None:
        cos, sin = position_embeddings
    else:
        position_ids = kwargs.get("position_ids")
        if position_ids is None:
            raise RuntimeError("missing position_ids/position_embeddings")
        rotary = getattr(capture.module, "rotary_emb", None)
        if rotary is None:
            raise RuntimeError("attention module has no rotary_emb")
        cos, sin = rotary(k, position_ids)

    # Try common HF helper locations. They share the same signature for the
    # Llama/Qwen2/Mistral family.
    helpers = [
        "transformers.models.llama.modeling_llama",
        "transformers.models.qwen2.modeling_qwen2",
        "transformers.models.mistral.modeling_mistral",
    ]
    last_error = None
    for module_name in helpers:
        try:
            mod = __import__(module_name, fromlist=["apply_rotary_pos_emb"])
            return mod.apply_rotary_pos_emb(q, k, cos, sin)
        except Exception as exc:  # pragma: no cover - version dependent
            last_error = exc
    raise RuntimeError(f"could not apply RoPE with known HF helpers: {last_error}")


def _error_summary(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    diff = (candidate - reference).detach().float()
    abs_diff = diff.abs()
    token_l2 = torch.linalg.vector_norm(diff, dim=-1)
    ref_l2 = torch.linalg.vector_norm(reference.detach().float(), dim=-1)
    rel = token_l2 / torch.clamp(ref_l2, min=1.0e-6)
    return {
        "max_abs_error": float(abs_diff.max().item()),
        "mean_abs_error": float(abs_diff.mean().item()),
        "p99_token_l2_error": float(torch.quantile(token_l2.flatten(), 0.99).item()),
        "relative_l2_error": float(
            torch.linalg.vector_norm(diff).item()
            / max(torch.linalg.vector_norm(reference.detach().float()).item(), 1.0e-6)
        ),
        "p99_relative_token_error": float(torch.quantile(rel.flatten(), 0.99).item()),
    }


def _per_head_error(reference: torch.Tensor, candidate: torch.Tensor) -> list[dict]:
    rows = []
    for head_idx in range(reference.shape[2]):
        err = _error_summary(reference[:, :, head_idx, :], candidate[:, :, head_idx, :])
        err["head_id"] = head_idx
        rows.append(err)
    return rows


def _profile_capture(capture: CapturedAttentionInput, args, prompt_id: int) -> dict:
    q, k, v, meta = _shape_qkv(capture, apply_rope=not args.no_rope)
    if q.shape[3] < 16:
        return {
            "skipped": True,
            "reason": "head_dim_too_small_for_triton_dot",
            "layer_id": capture.layer_id,
            "module_name": capture.module_name,
            "seq": q.shape[1],
            "head_dim": q.shape[3],
            "meta": meta,
        }
    if q.shape[1] < args.min_seq:
        return {
            "skipped": True,
            "reason": "seq_too_short",
            "layer_id": capture.layer_id,
            "module_name": capture.module_name,
            "seq": q.shape[1],
        }
    if q.shape[1] > args.max_seq:
        q = q[:, -args.max_seq :, :, :].contiguous()
        k = k[:, -args.max_seq :, :, :].contiguous()
        v = v[:, -args.max_seq :, :, :].contiguous()

    metadata = StreamAttnMetadataCache.from_value(v, block_size=args.block_size)
    dense = dense_attention_forward(q, k, v, causal=args.causal)
    mass, mass_info = stream_attn_gate1(
        q,
        k,
        v,
        causal=args.causal,
        mode="gate1",
        skip_predicate="mass",
        error_budget=args.error_budget,
        block_size=args.block_size,
        tile_size_q=args.tile_size_q,
        telemetry=True,
        return_info=True,
    )
    value_bound, value_info = stream_attn_gate1(
        q,
        k,
        v,
        causal=args.causal,
        mode="gate1",
        metadata=metadata,
        skip_predicate="value_bound",
        error_budget=args.error_budget,
        block_size=args.block_size,
        tile_size_q=args.tile_size_q,
        telemetry=True,
        return_info=True,
    )

    mass_heads = mass_info.per_head_stats or ()
    value_heads = value_info.per_head_stats or ()
    per_head_rows = []
    mass_head_errors = _per_head_error(dense, mass)
    value_head_errors = _per_head_error(dense, value_bound)
    for head_idx in range(q.shape[2]):
        mass_active = (
            mass_heads[head_idx].active_pv_fraction if head_idx < len(mass_heads) else None
        )
        value_active = (
            value_heads[head_idx].active_pv_fraction if head_idx < len(value_heads) else None
        )
        kv_head = head_idx // max(1, meta["q_per_kv"])
        per_head_rows.append(
            {
                "prompt_id": prompt_id,
                "layer_id": capture.layer_id,
                "module_name": capture.module_name,
                "head_id": head_idx,
                "kv_head_id": kv_head,
                "q_group_id": head_idx % max(1, meta["q_per_kv"]),
                "active_mass": mass_active,
                "active_value_bound": value_active,
                "mass_minus_value_bound_active": (
                    None
                    if mass_active is None or value_active is None
                    else mass_active - value_active
                ),
                **{f"mass_{key}": value for key, value in mass_head_errors[head_idx].items() if key != "head_id"},
                **{
                    f"value_bound_{key}": value
                    for key, value in value_head_errors[head_idx].items()
                    if key != "head_id"
                },
            }
        )

    return {
        "skipped": False,
        "prompt_id": prompt_id,
        "layer_id": capture.layer_id,
        "module_name": capture.module_name,
        "shape": {
            "batch": q.shape[0],
            "seq": q.shape[1],
            "heads": q.shape[2],
            "dim": q.shape[3],
            "dtype": str(q.dtype).replace("torch.", ""),
        },
        "meta": meta,
        "mass": {
            "active_pv_fraction": mass_info.active_pv_fraction,
            **_error_summary(dense, mass),
        },
        "value_bound": {
            "active_pv_fraction": value_info.active_pv_fraction,
            **_error_summary(dense, value_bound),
        },
        "per_head": per_head_rows,
    }


def _write_csv(path: str, rows: list[dict]) -> None:
    import csv

    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--max-prompts", type=int, default=1)
    parser.add_argument("--layers", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--max-seq", type=int, default=4096)
    parser.add_argument("--min-seq", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--tile-size-q", type=int, default=64)
    parser.add_argument("--error-budget", type=float, default=1e-3)
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-rope", action="store_true")
    parser.add_argument("--use-safetensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--per-head-csv-out", default=None)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Gate-1 real-LLM profiling")

    AutoModelForCausalLM, AutoTokenizer = _import_transformers()
    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        use_safetensors=args.use_safetensors,
        trust_remote_code=args.trust_remote_code,
    ).to(args.device)
    model.eval()

    selected_layers = _parse_int_list(args.layers)
    prompts = _load_prompts(args)
    results = []
    per_head_rows = []
    for prompt_id, prompt in enumerate(prompts):
        captured, handles = _capture_attention_inputs(model, selected_layers)
        try:
            tokens = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_seq,
            ).to(args.device)
            with torch.inference_mode():
                model(**tokens, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

        for capture in captured:
            with torch.inference_mode():
                row = _profile_capture(capture, args, prompt_id)
            results.append(row)
            per_head_rows.extend(row.get("per_head", []))

    payload = {
        "model": args.model,
        "prompts": len(prompts),
        "block_size": args.block_size,
        "tile_size_q": args.tile_size_q,
        "error_budget": args.error_budget,
        "results": results,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n", encoding="utf-8")
    if args.per_head_csv_out:
        _write_csv(args.per_head_csv_out, per_head_rows)
    print(text)


if __name__ == "__main__":
    main()
