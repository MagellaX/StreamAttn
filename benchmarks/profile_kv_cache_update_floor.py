"""Profile KV-cache update floors for StreamAttn routed decode layers.

The actual-model route timing showed that `past_key_value.update` dominates
the routed seed-only patch. This benchmark isolates that cost from attention:

* HF DynamicCache append path;
* HF StaticLayer index_copy path;
* direct preallocated tensor writes;
* optional Triton append writes.

The target shape is Qwen-style decode with BHND cache tensors:

    cache: [layers, batch, kv_heads, max_len, head_dim]
    new:   [layers, batch, kv_heads, 1,       head_dim]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_stream_attn_gate0_wrapper import _dtype  # noqa: E402

try:  # pragma: no cover - optional CUDA/Triton dependency
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - optional CUDA/Triton dependency
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:  # pragma: no cover - exercised on GPU only

    @triton.jit
    def _kv_append_bhnd_kernel(
        k_new,
        v_new,
        k_cache,
        v_cache,
        cache_pos: tl.constexpr,
        total: tl.constexpr,
        batch: tl.constexpr,
        kv_heads: tl.constexpr,
        max_len: tl.constexpr,
        head_dim: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < total
        d = offsets % head_dim
        tmp = offsets // head_dim
        h = tmp % kv_heads
        tmp = tmp // kv_heads
        b = tmp % batch
        layer = tmp // batch

        src = (((layer * batch + b) * kv_heads + h) * head_dim) + d
        dst = ((((layer * batch + b) * kv_heads + h) * max_len + cache_pos) * head_dim) + d
        tl.store(k_cache + dst, tl.load(k_new + src, mask=mask), mask=mask)
        tl.store(v_cache + dst, tl.load(v_new + src, mask=mask), mask=mask)


def cache_bytes(
    *,
    layer_count: int,
    batch_size: int,
    kv_heads: int,
    max_len: int,
    head_dim: int,
    dtype_bytes: int,
) -> int:
    return 2 * layer_count * batch_size * kv_heads * max_len * head_dim * dtype_bytes


def _stats(samples: Sequence[float]) -> Dict[str, float]:
    values = sorted(float(value) for value in samples)
    if not values:
        return {"min_ms": 0.0, "median_ms": 0.0, "p90_ms": 0.0, "mean_ms": 0.0}
    p90_idx = int(round((len(values) - 1) * 0.90))
    return {
        "min_ms": values[0],
        "median_ms": values[len(values) // 2],
        "p90_ms": values[max(0, min(p90_idx, len(values) - 1))],
        "mean_ms": sum(values) / len(values),
    }


def _cuda_time_ms(fn: Callable[[], None]) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end))
    start_time = time.perf_counter()
    fn()
    return (time.perf_counter() - start_time) * 1000.0


def _summarize_method(
    name: str,
    samples: Sequence[float],
    *,
    update_steps: int,
    layer_count: int,
) -> Dict[str, Any]:
    stats = _stats(samples)
    median = stats["median_ms"]
    return {
        "method": name,
        "samples_ms": [float(value) for value in samples],
        **stats,
        "median_ms_per_decode_step": median / max(1, update_steps),
        "median_us_per_layer_step": 1000.0 * median / max(1, update_steps * layer_count),
    }


def _allocate_inputs(args: argparse.Namespace, *, device: torch.device, dtype: torch.dtype):
    max_len = args.max_seq + args.update_steps + 1
    shape = (args.layer_count, args.batch_size, args.kv_heads, args.max_seq, args.head_dim)
    update_shape = (args.layer_count, args.batch_size, args.kv_heads, 1, args.head_dim)
    base_k = torch.empty(shape, device=device, dtype=dtype)
    base_v = torch.empty(shape, device=device, dtype=dtype)
    k_new = torch.empty(update_shape, device=device, dtype=dtype)
    v_new = torch.empty(update_shape, device=device, dtype=dtype)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return base_k, base_v, k_new, v_new, max_len


def _bench_hf_dynamic_cache(args, base_k, base_v, k_new, v_new, device) -> List[float]:
    from transformers.cache_utils import DynamicCache

    samples = []
    for sample_idx in range(args.samples + args.warmup_samples):
        cache = DynamicCache()
        for layer in range(args.layer_count):
            cache.update(base_k[layer], base_v[layer], layer)
        if device.type == "cuda":
            torch.cuda.synchronize()

        def run():
            for step in range(args.update_steps):
                cache_position = torch.tensor([args.max_seq + step], device=device, dtype=torch.long)
                kwargs = {"cache_position": cache_position}
                for layer in range(args.layer_count):
                    cache.update(k_new[layer], v_new[layer], layer, kwargs)

        elapsed = _cuda_time_ms(run)
        if sample_idx >= args.warmup_samples:
            samples.append(elapsed)
        del cache
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return samples


def _bench_static_layer_index_copy(args, k_new, v_new, device, dtype) -> List[float]:
    max_len = args.max_seq + args.update_steps + 1
    k_cache = torch.empty(
        (args.layer_count, args.batch_size, args.kv_heads, max_len, args.head_dim),
        device=device,
        dtype=dtype,
    )
    v_cache = torch.empty_like(k_cache)
    samples = []
    for sample_idx in range(args.samples + args.warmup_samples):
        if device.type == "cuda":
            torch.cuda.synchronize()

        def run():
            for step in range(args.update_steps):
                cache_position = torch.tensor([args.max_seq + step], device=device, dtype=torch.long)
                for layer in range(args.layer_count):
                    k_cache[layer].index_copy_(2, cache_position, k_new[layer])
                    v_cache[layer].index_copy_(2, cache_position, v_new[layer])

        elapsed = _cuda_time_ms(run)
        if sample_idx >= args.warmup_samples:
            samples.append(elapsed)
    return samples


def _bench_direct_slice_layer_loop(args, k_new, v_new, device, dtype, max_len: int) -> List[float]:
    k_cache = torch.empty(
        (args.layer_count, args.batch_size, args.kv_heads, max_len, args.head_dim),
        device=device,
        dtype=dtype,
    )
    v_cache = torch.empty_like(k_cache)
    samples = []
    for sample_idx in range(args.samples + args.warmup_samples):
        if device.type == "cuda":
            torch.cuda.synchronize()

        def run():
            for step in range(args.update_steps):
                pos = args.max_seq + step
                for layer in range(args.layer_count):
                    k_cache[layer, :, :, pos : pos + 1, :].copy_(k_new[layer])
                    v_cache[layer, :, :, pos : pos + 1, :].copy_(v_new[layer])

        elapsed = _cuda_time_ms(run)
        if sample_idx >= args.warmup_samples:
            samples.append(elapsed)
    return samples


def _bench_direct_slice_batched_layers(args, k_new, v_new, device, dtype, max_len: int) -> List[float]:
    k_cache = torch.empty(
        (args.layer_count, args.batch_size, args.kv_heads, max_len, args.head_dim),
        device=device,
        dtype=dtype,
    )
    v_cache = torch.empty_like(k_cache)
    samples = []
    for sample_idx in range(args.samples + args.warmup_samples):
        if device.type == "cuda":
            torch.cuda.synchronize()

        def run():
            for step in range(args.update_steps):
                pos = args.max_seq + step
                k_cache[:, :, :, pos : pos + 1, :].copy_(k_new)
                v_cache[:, :, :, pos : pos + 1, :].copy_(v_new)

        elapsed = _cuda_time_ms(run)
        if sample_idx >= args.warmup_samples:
            samples.append(elapsed)
    return samples


def _bench_triton_append_batched_layers(args, k_new, v_new, device, dtype, max_len: int) -> List[float]:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if device.type != "cuda":
        raise RuntimeError("Triton append requires CUDA")
    k_cache = torch.empty(
        (args.layer_count, args.batch_size, args.kv_heads, max_len, args.head_dim),
        device=device,
        dtype=dtype,
    )
    v_cache = torch.empty_like(k_cache)
    total = args.layer_count * args.batch_size * args.kv_heads * args.head_dim
    grid = (triton.cdiv(total, args.triton_block),)

    def launch(pos: int):
        _kv_append_bhnd_kernel[grid](
            k_new,
            v_new,
            k_cache,
            v_cache,
            pos,
            total,
            args.batch_size,
            args.kv_heads,
            max_len,
            args.head_dim,
            block=args.triton_block,
        )

    launch(args.max_seq)
    torch.cuda.synchronize()
    samples = []
    for sample_idx in range(args.samples + args.warmup_samples):
        def run():
            for step in range(args.update_steps):
                launch(args.max_seq + step)

        elapsed = _cuda_time_ms(run)
        if sample_idx >= args.warmup_samples:
            samples.append(elapsed)
    return samples


def profile(args: argparse.Namespace) -> Dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    dtype_bytes = torch.empty((), dtype=dtype).element_size()
    base_k, base_v, k_new, v_new, max_len = _allocate_inputs(args, device=device, dtype=dtype)

    methods = [part.strip() for part in args.methods.split(",") if part.strip()]
    runners = {
        "hf_dynamic_cache": lambda: _bench_hf_dynamic_cache(args, base_k, base_v, k_new, v_new, device),
        "static_layer_index_copy": lambda: _bench_static_layer_index_copy(args, k_new, v_new, device, dtype),
        "direct_slice_layer_loop": lambda: _bench_direct_slice_layer_loop(
            args, k_new, v_new, device, dtype, max_len
        ),
        "direct_slice_batched_layers": lambda: _bench_direct_slice_batched_layers(
            args, k_new, v_new, device, dtype, max_len
        ),
        "triton_append_batched_layers": lambda: _bench_triton_append_batched_layers(
            args, k_new, v_new, device, dtype, max_len
        ),
    }
    unknown = sorted(set(methods) - set(runners))
    if unknown:
        raise ValueError(f"unknown cache update methods: {unknown}")

    results = []
    for method in methods:
        if method == "triton_append_batched_layers" and not TRITON_AVAILABLE:
            results.append({"method": method, "skipped": True, "reason": "triton_unavailable"})
            continue
        print(f"[kv-cache-update-floor] profiling {method}", flush=True)
        samples = runners[method]()
        results.append(
            _summarize_method(
                method,
                samples,
                update_steps=args.update_steps,
                layer_count=args.layer_count,
            )
        )

    by_method = {row["method"]: row for row in results if not row.get("skipped")}
    hf = by_method.get("hf_dynamic_cache")
    for row in results:
        if row.get("skipped") or hf is None or row["method"] == "hf_dynamic_cache":
            continue
        row["speedup_vs_hf_dynamic_cache"] = hf["median_ms"] / max(row["median_ms"], 1.0e-12)

    return {
        "schema": "streamattn.kv_cache_update_floor.v1",
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "shape": {
            "layer_count": args.layer_count,
            "batch_size": args.batch_size,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "max_seq": args.max_seq,
            "max_len_allocated": max_len,
            "update_steps": args.update_steps,
            "warmup_samples": args.warmup_samples,
            "timed_samples": args.samples,
            "dtype": args.dtype,
        },
        "memory": {
            "cache_bytes_per_kv_pair": cache_bytes(
                layer_count=args.layer_count,
                batch_size=args.batch_size,
                kv_heads=args.kv_heads,
                max_len=max_len,
                head_dim=args.head_dim,
                dtype_bytes=dtype_bytes,
            ),
            "dtype_bytes": dtype_bytes,
        },
        "triton_available": TRITON_AVAILABLE,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--layer-count", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--max-seq", type=int, default=32768)
    parser.add_argument("--update-steps", type=int, default=8)
    parser.add_argument("--warmup-samples", type=int, default=1)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--triton-block", type=int, default=256)
    parser.add_argument(
        "--methods",
        default=(
            "hf_dynamic_cache,static_layer_index_copy,direct_slice_layer_loop,"
            "direct_slice_batched_layers,triton_append_batched_layers"
        ),
    )
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
