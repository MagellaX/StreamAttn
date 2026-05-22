"""Batch-size scaling for seed-only direct output vs FlashInfer batch decode.

The single-row seed-only path is stuck near FlashInfer's launch floor.  This
profiler tests the serving shape where that floor can be amortized: multiple
independent decode rows in one launch.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_stream_attn_gate0_wrapper import _error, _sync, _time_cuda  # noqa: E402
from stream_attention.kernels.gate0_seed_only_triton import (  # noqa: E402
    gate0_seed_only_attention_triton_forward_out,
)

try:
    import flashinfer

    HAS_FLASHINFER = True
except Exception:  # pragma: no cover - optional dependency
    flashinfer = None
    HAS_FLASHINFER = False


def _parse_ints(raw: str) -> List[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"empty integer list: {raw!r}")
    return values


def _dtype(raw: str) -> torch.dtype:
    if raw == "fp16":
        return torch.float16
    if raw == "bf16":
        return torch.bfloat16
    if raw == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {raw}")


def _make_paged_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    page_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if k.shape != v.shape or k.ndim != 4:
        raise ValueError(f"expected matching [B,N,H,D] K/V, got {k.shape=} {v.shape=}")
    batch, kv_len, kv_heads, dim = k.shape
    pages_per_req = math.ceil(kv_len / page_size)
    total_pages = batch * pages_per_req
    cache = torch.zeros(total_pages, 2, page_size, kv_heads, dim, device=k.device, dtype=k.dtype)
    for batch_idx in range(batch):
        for local_page in range(pages_per_req):
            start = local_page * page_size
            end = min(kv_len, start + page_size)
            global_page = batch_idx * pages_per_req + local_page
            cache[global_page, 0, : end - start] = k[batch_idx, start:end]
            cache[global_page, 1, : end - start] = v[batch_idx, start:end]
    indptr = torch.arange(0, total_pages + 1, pages_per_req, device=k.device, dtype=torch.int32)
    indices = torch.arange(total_pages, device=k.device, dtype=torch.int32)
    last_page_len = torch.full(
        (batch,),
        kv_len - (pages_per_req - 1) * page_size,
        device=k.device,
        dtype=torch.int32,
    )
    return cache, indptr, indices, last_page_len


def _make_flashinfer_wrapper(
    *,
    workspace: torch.Tensor,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    last_page_len: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    dim: int,
    page_size: int,
    dtype: torch.dtype,
    backend: str,
    use_tensor_cores: bool,
    disable_split_kv: bool,
):
    if not HAS_FLASHINFER:
        raise RuntimeError("FlashInfer is not available")
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        use_tensor_cores=use_tensor_cores,
        backend=backend,
    )
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        q_heads,
        kv_heads,
        dim,
        page_size,
        pos_encoding_mode="NONE",
        q_data_type=dtype,
        kv_data_type=dtype,
        o_data_type=dtype,
        sm_scale=1.0 / math.sqrt(float(dim)),
        disable_split_kv=disable_split_kv,
    )
    return wrapper


def profile(args: argparse.Namespace) -> Dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not HAS_FLASHINFER:
        raise RuntimeError("FlashInfer is required")
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    dtype = _dtype(args.dtype)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    rows: list[dict[str, Any]] = []
    for batch in _parse_ints(args.batch_sizes):
        q = torch.randn(batch, 1, args.q_heads, args.dim, device=device, dtype=dtype)
        k = torch.randn(batch, args.kv_len, args.kv_heads, args.dim, device=device, dtype=dtype)
        v = torch.randn_like(k)
        seed_out = torch.empty_like(q)
        kv_cache, indptr, indices, last_page_len = _make_paged_kv_cache(k, v, page_size=args.page_size)
        workspace = torch.empty(args.workspace_mb * 1024 * 1024, device=device, dtype=torch.uint8)
        wrapper = _make_flashinfer_wrapper(
            workspace=workspace,
            indptr=indptr,
            indices=indices,
            last_page_len=last_page_len,
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            dim=args.dim,
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
                seed_out,
                block_size=args.block_size,
                sink_blocks=args.sink_blocks,
                recent_blocks=args.recent_blocks,
                middle_seed_blocks=args.middle_seed_blocks,
                block_order=args.block_order,
                num_warps=args.num_warps,
                num_stages=args.num_stages,
            )

        def flashinfer_batch_run() -> torch.Tensor:
            return wrapper.run(q[:, 0].contiguous(), kv_cache).view(batch, 1, args.q_heads, args.dim)

        seed_ref = seed_run().clone()
        flash_ref = flashinfer_batch_run().clone()
        _sync(device)
        seed_ms = _time_cuda(seed_run, device=device, warmup=args.warmup, iters=args.iters)
        flash_ms = _time_cuda(flashinfer_batch_run, device=device, warmup=args.warmup, iters=args.iters)
        rows.append(
            {
                "batch": batch,
                "seed_direct_full_prealloc_ms": seed_ms,
                "flashinfer_batch_tc_exact_ms": flash_ms,
                "speedup_vs_flashinfer_batch": flash_ms / seed_ms,
                "seed_per_row_us": seed_ms * 1000.0 / batch,
                "flashinfer_per_row_us": flash_ms * 1000.0 / batch,
                "seed_vs_flashinfer_exact": _error(seed_ref, flash_ref),
            }
        )

    winning = [row for row in rows if row["seed_direct_full_prealloc_ms"] < row["flashinfer_batch_tc_exact_ms"]]
    return {
        "schema": "streamattn.gate0.seed_only_batch_scaling.v1",
        "shape": {
            "query_len": 1,
            "q_heads": args.q_heads,
            "true_kv_heads": args.kv_heads,
            "group_size": args.q_heads // args.kv_heads,
            "kv_len": args.kv_len,
            "dim": args.dim,
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
        "flashinfer": {
            "backend": args.flashinfer_backend,
            "use_tensor_cores": args.flashinfer_tensor_cores,
            "disable_split_kv": args.disable_split_kv,
            "page_size": args.page_size,
        },
        "search": {
            "batch_sizes": _parse_ints(args.batch_sizes),
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed,
        },
        "rows": rows,
        "break_even_batch": min((row["batch"] for row in winning), default=None),
        "decision": "seed_only_wins_at_some_batch" if winning else "seed_only_never_wins_in_batch_sweep",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32")
    parser.add_argument("--q-heads", type=int, default=14)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--kv-len", type=int, default=32768)
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
