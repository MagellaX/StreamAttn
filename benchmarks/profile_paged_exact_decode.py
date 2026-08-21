"""Benchmark StreamAttn native paged exact decode against FlashInfer."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stream_attention import PagedExactDecodePlan, PagedKVCache
from stream_attention.paged import PAGED_EXACT_SM90_BACKEND


def _dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError("dtype must be fp16 or bf16")


def _time_cuda(
    function: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeats: int,
) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def _flashinfer_runner(
    query: torch.Tensor,
    cache: PagedKVCache,
    *,
    workspace_mb: int,
):
    try:
        import flashinfer
    except Exception as exc:
        raise RuntimeError("FlashInfer is required for the paired benchmark") from exc
    batch = int(query.shape[0])
    pages_per_request = cache.max_pages_per_request
    indptr = torch.arange(
        0,
        (batch + 1) * pages_per_request,
        pages_per_request,
        device=query.device,
        dtype=torch.int32,
    )
    indices = cache.page_table.reshape(-1).to(dtype=torch.int32)
    last_page_len = (
        (cache.sequence_lengths - 1) % cache.page_size + 1
    ).to(dtype=torch.int32)
    combined_cache = torch.stack((cache.key, cache.value), dim=1).contiguous()
    workspace = torch.empty(
        workspace_mb * 1024 * 1024,
        device=query.device,
        dtype=torch.uint8,
    )
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        cache.normalized_layout,
        use_tensor_cores=True,
        backend="auto",
    )
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        int(query.shape[2]),
        cache.kv_heads,
        int(query.shape[3]),
        cache.page_size,
        pos_encoding_mode="NONE",
        q_data_type=query.dtype,
        kv_data_type=cache.key.dtype,
        o_data_type=query.dtype,
        sm_scale=1.0 / math.sqrt(float(query.shape[3])),
        disable_split_kv=False,
    )
    output = torch.empty(
        query.shape[0],
        query.shape[2],
        query.shape[3],
        device=query.device,
        dtype=query.dtype,
    )

    def run() -> torch.Tensor:
        return wrapper.run(query[:, 0], combined_cache, out=output)

    return run


def profile(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.kv_len <= 0 or args.kv_len % args.page_size:
        raise ValueError("kv_len must be positive and divisible by page_size")
    if args.q_heads % args.kv_heads:
        raise ValueError("q_heads must be divisible by kv_heads")

    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    pages_per_request = args.kv_len // args.page_size
    total_pages = args.batch * pages_per_request
    query = torch.randn(
        args.batch,
        1,
        args.q_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    if args.layout == "NHD":
        cache_shape = (
            total_pages,
            args.page_size,
            args.kv_heads,
            args.head_dim,
        )
    else:
        cache_shape = (
            total_pages,
            args.kv_heads,
            args.page_size,
            args.head_dim,
        )
    key = torch.randn(*cache_shape, device=device, dtype=dtype)
    value = torch.randn_like(key)
    page_table = torch.randperm(
        total_pages, device=device, dtype=torch.int64
    ).to(torch.int32).view(args.batch, pages_per_request)
    sequence_lengths = torch.full(
        (args.batch,),
        args.kv_len,
        device=device,
        dtype=torch.int32,
    )
    cache = PagedKVCache(
        key=key,
        value=value,
        page_table=page_table,
        sequence_lengths=sequence_lengths,
        layout=args.layout,
    )
    plan = PagedExactDecodePlan.build(
        query,
        cache,
        splits=args.splits,
        tokens_per_tile=args.tokens_per_tile,
        partial_num_warps=args.partial_num_warps,
    )
    flashinfer_run = _flashinfer_runner(
        query,
        cache,
        workspace_mb=args.workspace_mb,
    )

    stream_output = plan.run().clone()
    flashinfer_output = flashinfer_run().view_as(query).clone()
    torch.cuda.synchronize()
    max_abs_error = float(
        (stream_output.float() - flashinfer_output.float()).abs().max().item()
    )
    mean_abs_error = float(
        (stream_output.float() - flashinfer_output.float()).abs().mean().item()
    )
    if max_abs_error > args.atol:
        raise RuntimeError(
            f"paged exact correctness failed: {max_abs_error} > {args.atol}"
        )

    stream_samples = _time_cuda(
        plan.run,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    flashinfer_samples = _time_cuda(
        flashinfer_run,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    stream_ms = float(statistics.median(stream_samples))
    flashinfer_ms = float(statistics.median(flashinfer_samples))
    paired_trials: list[dict[str, float | int | str]] = []
    for trial in range(args.paired_trials):
        if trial % 2 == 0:
            stream_trial = _time_cuda(
                plan.run, warmup=0, repeats=args.paired_repeats
            )
            flashinfer_trial = _time_cuda(
                flashinfer_run, warmup=0, repeats=args.paired_repeats
            )
            order = "streamattn_first"
        else:
            flashinfer_trial = _time_cuda(
                flashinfer_run, warmup=0, repeats=args.paired_repeats
            )
            stream_trial = _time_cuda(
                plan.run, warmup=0, repeats=args.paired_repeats
            )
            order = "flashinfer_first"
        stream_trial_ms = float(statistics.median(stream_trial))
        flashinfer_trial_ms = float(statistics.median(flashinfer_trial))
        paired_trials.append(
            {
                "trial": trial,
                "order": order,
                "streamattn_ms": stream_trial_ms,
                "flashinfer_ms": flashinfer_trial_ms,
                "speedup_vs_flashinfer": flashinfer_trial_ms / stream_trial_ms,
            }
        )
    paired_speedups = [
        float(trial["speedup_vs_flashinfer"]) for trial in paired_trials
    ]
    producer_groups = (
        args.batch * args.kv_heads
        if plan.backend == PAGED_EXACT_SM90_BACKEND
        else args.batch * args.q_heads
    )
    return {
        "schema": "streamattn.paged_exact_decode_profile.v1",
        "timestamp_unix": time.time(),
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch_version": torch.__version__,
        "batch": args.batch,
        "kv_len": args.kv_len,
        "q_heads": args.q_heads,
        "kv_heads": args.kv_heads,
        "group_size": args.q_heads // args.kv_heads,
        "head_dim": args.head_dim,
        "dtype": args.dtype,
        "page_size": args.page_size,
        "layout": args.layout,
        "pages_per_request": pages_per_request,
        "physical_page_order": "randomized",
        "splits": plan.splits,
        "tokens_per_tile": plan.tokens_per_tile,
        "partial_num_warps": plan.partial_num_warps,
        "backend_variant": plan.backend,
        "producer_ctas": producer_groups * plan.splits,
        "workspace_bytes": plan.workspace_bytes,
        "streamattn_ms": stream_ms,
        "flashinfer_ms": flashinfer_ms,
        "speedup_vs_flashinfer": flashinfer_ms / stream_ms,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "streamattn_samples_ms": stream_samples,
        "flashinfer_samples_ms": flashinfer_samples,
        "paired": {
            "trials": paired_trials,
            "speedup_median": float(statistics.median(paired_speedups)),
            "speedup_min": float(min(paired_speedups)),
            "wins": sum(speedup > 1.0 for speedup in paired_speedups),
            "trial_count": len(paired_speedups),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--kv-len", type=int, default=32768)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--layout", choices=("NHD", "HND"), default="NHD")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--splits", type=int)
    parser.add_argument("--tokens-per-tile", type=int, default=512)
    parser.add_argument("--partial-num-warps", type=int, default=4)
    parser.add_argument("--workspace-mb", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--paired-trials", type=int, default=9)
    parser.add_argument("--paired-repeats", type=int, default=10)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    result = profile(args)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
