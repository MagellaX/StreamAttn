"""Capture real post-RoPE Q/K tensors for Gate-0 summary-bound profiling.

This script is intentionally capture-only. It does not run StreamAttn kernels.
It saves tensors that can be fed into ``profile_gate0_summary_bounds.py`` to
answer whether real model K blocks have tight enough summaries for pre-QK skip.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_real_llm_gate1_heads import (
    CapturedAttentionInput,
    _capture_attention_inputs,
    _import_transformers,
    _load_prompts,
    _parse_int_list,
    _shape_qkv,
)


def _dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def _safe_name(value: str) -> str:
    value = value.strip().replace("\\", "/")
    value = value.split("/")[-1] if "/" in value else value
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "model"


def _crop_decode_tensors(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    query_len: int,
    kv_len: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if query_len <= 0:
        raise ValueError("query_len must be positive")
    if query_len > q.shape[1]:
        raise ValueError(f"query_len={query_len} exceeds captured sequence length {q.shape[1]}")
    q = q[:, -query_len:, :, :].contiguous()
    if kv_len is not None and kv_len > 0:
        if kv_len > k.shape[1]:
            raise ValueError(f"kv_len={kv_len} exceeds captured sequence length {k.shape[1]}")
        k = k[:, -kv_len:, :, :].contiguous()
        v = v[:, -kv_len:, :, :].contiguous()
    return q, k, v


def _save_tensor(path: Path, key: str, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: tensor.detach().cpu().contiguous()}, path)


def _gate0_command(row: dict) -> str:
    pieces = [
        "python benchmarks/profile_gate0_summary_bounds.py",
        f"--q-path {row['q_path']}",
        f"--k-path {row['k_path']}",
        f"--tensor-space {row['tensor_space']}",
        f"--model-id {row['model_id']}",
        f"--layer-id {row['layer_id']}",
        "--per-head",
        "--block-size 64 128",
        "--summary-outliers 0 1 2 4",
        "--block-order sequential recent_first sink_recent_first summary_desc",
    ]
    if row.get("v_path"):
        pieces.insert(3, f"--v-path {row['v_path']}")
    return " ".join(pieces)


def _capture_one(
    capture: CapturedAttentionInput,
    args,
    *,
    prompt_id: int,
    prompt_token_count: int,
    output_dir: Path,
) -> dict:
    apply_rope = args.tensor_space == "post_rope"
    q, k, v, meta = _shape_qkv(capture, apply_rope=apply_rope)
    if apply_rope and not meta.get("rope_applied"):
        if not args.allow_rope_fallback:
            return {
                "skipped": True,
                "reason": "rope_unavailable",
                "rope_error": meta.get("rope_error"),
                "prompt_id": prompt_id,
                "layer_id": capture.layer_id,
                "module_name": capture.module_name,
            }
        tensor_space = "pre_rope"
    else:
        tensor_space = args.tensor_space

    q, k, v = _crop_decode_tensors(q, k, v, query_len=args.query_len, kv_len=args.kv_len)
    model_name = _safe_name(args.model)
    prefix = f"{args.prefix}_" if args.prefix else ""
    stem = f"{prefix}prompt{prompt_id}_layer{capture.layer_id}_{tensor_space}"
    layer_dir = output_dir / model_name / f"prompt_{prompt_id}" / f"layer_{capture.layer_id}"

    q_key = "post_rope_q" if tensor_space == "post_rope" else "pre_rope_q"
    k_key = "post_rope_k" if tensor_space == "post_rope" else "pre_rope_k"
    q_path = layer_dir / f"{stem}_q.pt"
    k_path = layer_dir / f"{stem}_k.pt"
    v_path = layer_dir / f"{stem}_v.pt" if args.save_v else None

    _save_tensor(q_path, q_key, q)
    _save_tensor(k_path, k_key, k)
    if v_path is not None:
        _save_tensor(v_path, "v", v)

    row = {
        "skipped": False,
        "model_id": args.model,
        "prompt_id": prompt_id,
        "prompt_token_count": prompt_token_count,
        "layer_id": capture.layer_id,
        "module_name": capture.module_name,
        "tensor_space": tensor_space,
        "q_path": str(q_path),
        "k_path": str(k_path),
        "v_path": None if v_path is None else str(v_path),
        "shape": {
            "batch": int(q.shape[0]),
            "query_len": int(q.shape[1]),
            "kv_len": int(k.shape[1]),
            "heads": int(q.shape[2]),
            "dim": int(q.shape[3]),
            "dtype": args.dtype,
        },
        "meta": {
            **meta,
            "gqa_expanded_for_gate0_profile": bool(meta.get("q_per_kv", 1) != 1),
            "saved_heads": int(q.shape[2]),
            "logical_num_kv_heads": int(meta.get("num_kv_heads", q.shape[2])),
            "logical_num_q_heads": int(meta.get("num_heads", q.shape[2])),
        },
        "capture_policy": {
            "max_seq": args.max_seq,
            "kv_len": args.kv_len,
            "query_len": args.query_len,
            "save_v": args.save_v,
        },
    }
    row["gate0_profile_command"] = _gate0_command(row)
    return row


def _write_metadata(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--max-prompts", type=int, default=1)
    parser.add_argument("--layers", default="0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--max-seq", type=int, default=4096)
    parser.add_argument("--kv-len", type=int, default=0)
    parser.add_argument("--query-len", type=int, default=1)
    parser.add_argument("--tensor-space", choices=["pre_rope", "post_rope"], default="post_rope")
    parser.add_argument("--allow-rope-fallback", action="store_true")
    parser.add_argument("--save-v", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="artifacts/gate0/real_qk")
    parser.add_argument("--metadata-json-out", default="")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--use-safetensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if args.query_len <= 0:
        raise ValueError("--query-len must be positive")

    AutoModelForCausalLM, AutoTokenizer = _import_transformers()
    dtype = _dtype(args.dtype)
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
    output_dir = Path(args.output_dir)
    rows = []
    errors = []

    for prompt_id, prompt in enumerate(prompts):
        captured, handles = _capture_attention_inputs(model, selected_layers)
        try:
            tokens = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_seq,
            ).to(args.device)
            prompt_token_count = int(tokens["input_ids"].shape[1])
            with torch.inference_mode():
                model(**tokens, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

        for capture in captured:
            try:
                with torch.inference_mode():
                    rows.append(
                        _capture_one(
                            capture,
                            args,
                            prompt_id=prompt_id,
                            prompt_token_count=prompt_token_count,
                            output_dir=output_dir,
                        )
                    )
            except Exception as exc:
                errors.append(
                    {
                        "error": f"{type(exc).__name__}: {exc}",
                        "prompt_id": prompt_id,
                        "layer_id": capture.layer_id,
                        "module_name": capture.module_name,
                    }
                )

    payload = {
        "model_id": args.model,
        "tensor_space": args.tensor_space,
        "rows": rows,
        "errors": errors,
    }
    metadata_path = (
        Path(args.metadata_json_out)
        if args.metadata_json_out
        else output_dir / _safe_name(args.model) / "metadata.json"
    )
    _write_metadata(metadata_path, payload)
    print(json.dumps({**payload, "metadata_json": str(metadata_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
