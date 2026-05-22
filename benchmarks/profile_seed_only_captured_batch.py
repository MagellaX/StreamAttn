"""Profile batched seed-only on captured real Qwen Q/K/V rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_gate0_true_gqa import _true_gqa_kv  # noqa: E402
from benchmarks.profile_seed_only_batch_scaling import (  # noqa: E402
    _make_flashinfer_wrapper,
    _make_paged_kv_cache,
)
from benchmarks.profile_stream_attn_gate0_wrapper import _dtype, _error, _load_tensor, _sync, _time_cuda  # noqa: E402
from stream_attention.kernels.gate0_seed_only_triton import gate0_seed_only_attention_triton_forward_out  # noqa: E402


def _rows_for_layer(metadata: dict[str, Any], layer_id: int) -> list[dict[str, Any]]:
    rows = [
        row
        for row in metadata.get("rows", [])
        if not row.get("skipped") and int(row.get("layer_id")) == int(layer_id)
    ]
    if not rows:
        raise ValueError(f"metadata has no usable rows for layer {layer_id}")
    return rows


def _stack_captures(rows: list[dict[str, Any]], *, device: torch.device, dtype: torch.dtype):
    q_parts = []
    k_parts = []
    v_parts = []
    true_kv_heads = None
    prompt_ids = []
    for row in rows:
        row_true_kv_heads = int((row.get("meta") or {}).get("logical_num_kv_heads") or row["shape"]["heads"])
        true_kv_heads = row_true_kv_heads if true_kv_heads is None else true_kv_heads
        if row_true_kv_heads != true_kv_heads:
            raise ValueError("all rows must have the same true KV head count")
        q_parts.append(_load_tensor(row["q_path"], key="q", device=device, dtype=dtype))
        k_expanded = _load_tensor(row["k_path"], key="k", device=device, dtype=dtype)
        v_expanded = _load_tensor(row["v_path"], key="v", device=device, dtype=dtype)
        k_parts.append(_true_gqa_kv(k_expanded, true_kv_heads=true_kv_heads).contiguous())
        v_parts.append(_true_gqa_kv(v_expanded, true_kv_heads=true_kv_heads).contiguous())
        prompt_ids.append(row.get("prompt_id"))
    return torch.cat(q_parts, dim=0), torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0), true_kv_heads, prompt_ids


def profile(args: argparse.Namespace) -> Dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    dtype = _dtype(args.dtype)
    metadata = json.loads(Path(args.metadata_json).read_text(encoding="utf-8"))
    rows = _rows_for_layer(metadata, args.layer_id)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    q, k, v, true_kv_heads, prompt_ids = _stack_captures(rows, device=device, dtype=dtype)
    out = torch.empty_like(q)
    kv_cache, indptr, indices, last_page_len = _make_paged_kv_cache(k, v, page_size=args.page_size)
    workspace = torch.empty(args.workspace_mb * 1024 * 1024, device=device, dtype=torch.uint8)
    wrapper = _make_flashinfer_wrapper(
        workspace=workspace,
        indptr=indptr,
        indices=indices,
        last_page_len=last_page_len,
        q_heads=q.shape[2],
        kv_heads=k.shape[2],
        dim=q.shape[3],
        page_size=args.page_size,
        dtype=dtype,
        backend=args.flashinfer_backend,
        use_tensor_cores=args.flashinfer_tensor_cores,
        disable_split_kv=args.disable_split_kv,
    )

    def seed_run() -> torch.Tensor:
        return gate0_seed_only_attention_triton_forward_out(
            q,
            k,
            v,
            out,
            block_size=args.block_size,
            sink_blocks=args.sink_blocks,
            recent_blocks=args.recent_blocks,
            middle_seed_blocks=args.middle_seed_blocks,
            block_order=args.block_order,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        )

    def flashinfer_run() -> torch.Tensor:
        return wrapper.run(q[:, 0].contiguous(), kv_cache).view_as(q)

    seed_ref = seed_run().clone()
    flash_ref = flashinfer_run().clone()
    _sync(device)
    seed_ms = _time_cuda(seed_run, device=device, warmup=args.warmup, iters=args.iters)
    flash_ms = _time_cuda(flashinfer_run, device=device, warmup=args.warmup, iters=args.iters)
    batch = int(q.shape[0])
    return {
        "schema": "streamattn.gate0.seed_only_captured_batch.v1",
        "capture": {
            "metadata_json": args.metadata_json,
            "layer_id": args.layer_id,
            "prompt_ids": prompt_ids,
            "row_count": batch,
        },
        "shape": {
            "batch": batch,
            "query_len": int(q.shape[1]),
            "q_heads": int(q.shape[2]),
            "true_kv_heads": int(true_kv_heads),
            "group_size": int(q.shape[2] // true_kv_heads),
            "kv_len": int(k.shape[1]),
            "dim": int(q.shape[3]),
            "dtype": args.dtype,
        },
        "seed_config": {
            "block_size": args.block_size,
            "sink_blocks": args.sink_blocks,
            "recent_blocks": args.recent_blocks,
            "middle_seed_blocks": args.middle_seed_blocks,
            "seed_blocks": args.sink_blocks + args.recent_blocks + args.middle_seed_blocks,
            "seed_tokens": (args.sink_blocks + args.recent_blocks + args.middle_seed_blocks)
            * args.block_size,
            "block_order": args.block_order,
            "num_warps": args.num_warps,
            "num_stages": args.num_stages,
        },
        "timing": {
            "seed_direct_full_prealloc_ms": seed_ms,
            "flashinfer_batch_tc_exact_ms": flash_ms,
            "speedup_vs_flashinfer_batch": flash_ms / seed_ms,
            "seed_per_row_us": seed_ms * 1000.0 / batch,
            "flashinfer_per_row_us": flash_ms * 1000.0 / batch,
        },
        "quality": {
            "seed_vs_flashinfer_exact": _error(seed_ref, flash_ref),
        },
        "decision": "captured_seed_batch_beats_flashinfer" if seed_ms < flash_ms else "captured_seed_batch_loses",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-json", required=True)
    parser.add_argument("--layer-id", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--recent-blocks", type=int, default=2)
    parser.add_argument("--middle-seed-blocks", type=int, default=8)
    parser.add_argument("--block-order", default="recent_first", choices=["sequential", "recent_first", "sink_recent_first"])
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--flashinfer-backend", default="auto")
    parser.add_argument("--flashinfer-tensor-cores", action="store_true")
    parser.add_argument("--disable-split-kv", action="store_true")
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--workspace-mb", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
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
