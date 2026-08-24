"""Benchmark StreamAttn native paged exact decode against FlashInfer."""

from __future__ import annotations

import argparse
import importlib.metadata
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

from stream_attention import PagedExactDecodePlan, PagedKVCache  # noqa: E402
from stream_attention.paged import (  # noqa: E402
    PAGED_EXACT_SM90_BACKEND,
    PAGED_EXACT_SM90_FRAGMENTED_BACKEND,
    PAGED_EXACT_SM90_FRAGMENTED_RAGGED_BACKEND,
)


_SM90_PAGED_BACKENDS = {
    PAGED_EXACT_SM90_BACKEND,
    PAGED_EXACT_SM90_FRAGMENTED_BACKEND,
    PAGED_EXACT_SM90_FRAGMENTED_RAGGED_BACKEND,
}


def _dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError("dtype must be fp16 or bf16")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _profile_sequence_lengths(
    profile: str,
    *,
    batch: int,
    kv_len: int,
    page_size: int,
    device: torch.device,
) -> torch.Tensor:
    fractions = {
        "full": (1.0,),
        "tail": (1.0,),
        "mild": (1.0, 0.875, 0.75, 0.625),
        "severe": (1.0, 0.75, 0.5, 0.25),
        "short": (0.5, 0.375, 0.25, 0.125),
        "floor": (0.125,),
        "tiny": (0.015625,),
        "minimum": (0.0,),
    }
    if profile not in fractions:
        raise ValueError(f"unknown length profile: {profile}")
    if profile == "full":
        values = [kv_len] * batch
    elif profile == "minimum":
        values = [1] * batch
    else:
        tail_offsets = (1, 7, 13, 3)
        profile_fractions = fractions[profile]
        values = []
        for row in range(batch):
            fraction = profile_fractions[row % len(profile_fractions)]
            length = int(kv_len * fraction) - tail_offsets[row % 4]
            values.append(max(page_size, length))
    return torch.tensor(values, device=device, dtype=torch.int32)


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
    backend: str,
):
    try:
        import flashinfer
    except Exception as exc:
        raise RuntimeError("FlashInfer is required for the paired benchmark") from exc
    lengths_cpu = cache.sequence_lengths.detach().to("cpu", torch.int64).tolist()
    active_pages = [
        (int(length) + cache.page_size - 1) // cache.page_size for length in lengths_cpu
    ]
    indptr_values = [0]
    for count in active_pages:
        indptr_values.append(indptr_values[-1] + count)
    indptr = torch.tensor(indptr_values, device=query.device, dtype=torch.int32)
    indices = torch.cat(
        [cache.page_table[row, :count] for row, count in enumerate(active_pages)]
    ).to(dtype=torch.int32)
    last_page_len = ((cache.sequence_lengths - 1) % cache.page_size + 1).to(
        dtype=torch.int32
    )
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
        backend=backend,
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

    resolved_backend = getattr(wrapper, "_backend", None)
    return run, {
        "version": importlib.metadata.version("flashinfer-python"),
        "cubin_version": _package_version("flashinfer-cubin"),
        "jit_cache_version": _package_version("flashinfer-jit-cache"),
        "requested_backend": backend,
        "resolved_backend": (
            None if resolved_backend is None else str(resolved_backend)
        ),
    }


def _dense_paged_reference(
    query: torch.Tensor,
    cache: PagedKVCache,
) -> torch.Tensor:
    """Materialize short rows and evaluate dense attention in FP32."""

    batch, _query_length, query_heads, head_dim = map(int, query.shape)
    group_size = query_heads // cache.kv_heads
    kv_head_indices = torch.arange(query_heads, device=query.device) // group_size
    output = torch.empty_like(query, dtype=torch.float32)
    scale = 1.0 / math.sqrt(float(head_dim))
    for row in range(batch):
        length = int(cache.sequence_lengths[row].item())
        page_count = (length + cache.page_size - 1) // cache.page_size
        physical_pages = cache.page_table[row, :page_count].to(torch.int64)
        key_pages = cache.key.index_select(0, physical_pages)
        value_pages = cache.value.index_select(0, physical_pages)
        if cache.normalized_layout == "HND":
            key_pages = key_pages.permute(0, 2, 1, 3)
            value_pages = value_pages.permute(0, 2, 1, 3)
        keys = key_pages.reshape(-1, cache.kv_heads, head_dim)[:length].float()
        values = value_pages.reshape(-1, cache.kv_heads, head_dim)[:length].float()
        keys = keys.index_select(1, kv_head_indices)
        values = values.index_select(1, kv_head_indices)
        q = query[row, 0].float()
        scores = torch.einsum("nhd,hd->hn", keys, q) * scale
        probabilities = torch.softmax(scores, dim=-1)
        output[row, 0] = torch.einsum("hn,nhd->hd", probabilities, values)
    return output


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
    page_table = (
        torch.randperm(total_pages, device=device, dtype=torch.int64)
        .to(torch.int32)
        .view(args.batch, pages_per_request)
        .clone()
    )
    sequence_lengths = _profile_sequence_lengths(
        args.length_profile,
        batch=args.batch,
        kv_len=args.kv_len,
        page_size=args.page_size,
        device=device,
    )
    lengths_cpu = sequence_lengths.detach().to("cpu", torch.int64).tolist()
    for row, length in enumerate(lengths_cpu):
        active_pages = (int(length) + args.page_size - 1) // args.page_size
        page_table[row, active_pages:] = -1
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
        sm90_fragmented_experimental=args.sm90_fragmented_experimental,
        sm90_fragmented_ragged_experimental=(args.sm90_fragmented_ragged_experimental),
    )
    stream_output = plan.run().clone()
    torch.cuda.synchronize()
    reference_output = None
    stream_reference_max_error = None
    stream_reference_mean_error = None
    if max(lengths_cpu) <= 2048:
        reference_output = _dense_paged_reference(query, cache)
        stream_reference_error = (stream_output.float() - reference_output).abs()
        stream_reference_max_error = float(stream_reference_error.max().item())
        stream_reference_mean_error = float(stream_reference_error.mean().item())
    stream_samples = _time_cuda(
        plan.run,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    stream_ms = float(statistics.median(stream_samples))

    requested_flashinfer_backends = [
        backend.strip()
        for backend in getattr(args, "flashinfer_backends", "auto").split(",")
        if backend.strip()
    ]
    if not requested_flashinfer_backends:
        raise ValueError("flashinfer_backends must contain at least one backend")
    flashinfer_candidates: list[dict[str, Any]] = []
    valid_flashinfer_candidates: list[dict[str, Any]] = []
    for requested_backend in requested_flashinfer_backends:
        try:
            candidate_run, candidate_info = _flashinfer_runner(
                query,
                cache,
                workspace_mb=args.workspace_mb,
                backend=requested_backend,
            )
            candidate_output = candidate_run().view_as(query).clone()
            torch.cuda.synchronize()
            candidate_max_error = float(
                (stream_output.float() - candidate_output.float()).abs().max().item()
            )
            candidate_mean_error = float(
                (stream_output.float() - candidate_output.float()).abs().mean().item()
            )
            candidate_reference_max_error = None
            candidate_reference_mean_error = None
            correct = candidate_max_error <= args.atol
            if reference_output is not None:
                candidate_reference_error = (
                    candidate_output.float() - reference_output
                ).abs()
                candidate_reference_max_error = float(
                    candidate_reference_error.max().item()
                )
                candidate_reference_mean_error = float(
                    candidate_reference_error.mean().item()
                )
                correct = bool(
                    stream_reference_max_error <= args.atol
                    and candidate_reference_max_error <= args.atol
                )
            candidate_samples = _time_cuda(
                candidate_run,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            candidate_ms = float(statistics.median(candidate_samples))
            candidate = {
                **candidate_info,
                "status": "correct" if correct else "mismatch",
                "max_abs_error": candidate_max_error,
                "mean_abs_error": candidate_mean_error,
                "reference_max_abs_error": candidate_reference_max_error,
                "reference_mean_abs_error": candidate_reference_mean_error,
                "median_ms": candidate_ms,
                "samples_ms": candidate_samples,
                "run": candidate_run,
            }
            flashinfer_candidates.append(candidate)
            if correct:
                valid_flashinfer_candidates.append(candidate)
        except Exception as exc:
            torch.cuda.synchronize()
            flashinfer_candidates.append(
                {
                    "requested_backend": requested_backend,
                    "status": "unsupported",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    if not valid_flashinfer_candidates:
        raise RuntimeError(
            "no correct FlashInfer backend was available: "
            + "; ".join(
                f"{candidate['requested_backend']}={candidate['status']}"
                for candidate in flashinfer_candidates
            )
        )
    selected_flashinfer = min(
        valid_flashinfer_candidates,
        key=lambda candidate: float(candidate["median_ms"]),
    )
    flashinfer_run = selected_flashinfer.pop("run")
    for candidate in flashinfer_candidates:
        candidate.pop("run", None)
    flashinfer_info = {
        key: value
        for key, value in selected_flashinfer.items()
        if key not in {"samples_ms", "median_ms", "max_abs_error", "mean_abs_error"}
    }
    flashinfer_info["selection"] = "fastest_correct_initial_median"
    max_abs_error = float(selected_flashinfer["max_abs_error"])
    mean_abs_error = float(selected_flashinfer["mean_abs_error"])
    flashinfer_samples = list(selected_flashinfer["samples_ms"])
    flashinfer_ms = float(statistics.median(flashinfer_samples))
    paired_trials: list[dict[str, float | int | str]] = []
    for trial in range(args.paired_trials):
        if trial % 2 == 0:
            stream_trial = _time_cuda(plan.run, warmup=0, repeats=args.paired_repeats)
            flashinfer_trial = _time_cuda(
                flashinfer_run, warmup=0, repeats=args.paired_repeats
            )
            order = "streamattn_first"
        else:
            flashinfer_trial = _time_cuda(
                flashinfer_run, warmup=0, repeats=args.paired_repeats
            )
            stream_trial = _time_cuda(plan.run, warmup=0, repeats=args.paired_repeats)
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
    paired_speedups = [float(trial["speedup_vs_flashinfer"]) for trial in paired_trials]
    producer_groups = (
        args.batch * args.kv_heads
        if plan.backend in _SM90_PAGED_BACKENDS
        else args.batch * args.q_heads
    )
    active_producer_ctas = producer_groups * plan.splits
    if plan.backend == PAGED_EXACT_SM90_FRAGMENTED_RAGGED_BACKEND:
        active_producer_ctas = args.kv_heads * sum(
            math.ceil(
                math.ceil(int(length) / 64)
                / math.ceil(math.ceil(int(length) / 64) / plan.splits)
            )
            for length in lengths_cpu
        )
    return {
        "schema": "streamattn.paged_exact_decode_profile.v2",
        "timestamp_unix": time.time(),
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch_version": torch.__version__,
        "flashinfer": flashinfer_info,
        "flashinfer_candidates": flashinfer_candidates,
        "batch": args.batch,
        "kv_len": args.kv_len,
        "length_profile": args.length_profile,
        "sequence_lengths": [int(length) for length in lengths_cpu],
        "total_sequence_tokens": int(sum(lengths_cpu)),
        "batch_capacity_utilization": float(
            sum(lengths_cpu) / (args.batch * args.kv_len)
        ),
        "q_heads": args.q_heads,
        "kv_heads": args.kv_heads,
        "group_size": args.q_heads // args.kv_heads,
        "head_dim": args.head_dim,
        "dtype": args.dtype,
        "page_size": args.page_size,
        "layout": args.layout,
        "pages_per_request": pages_per_request,
        "physical_page_order": "randomized",
        "physical_pages_per_compute_tile": (
            64 // args.page_size if plan.backend in _SM90_PAGED_BACKENDS else None
        ),
        "splits": plan.splits,
        "tokens_per_tile": plan.tokens_per_tile,
        "partial_num_warps": plan.partial_num_warps,
        "backend_variant": plan.backend,
        "producer_ctas": producer_groups * plan.splits,
        "active_producer_ctas": active_producer_ctas,
        "workspace_bytes": plan.workspace_bytes,
        "streamattn_ms": stream_ms,
        "flashinfer_ms": flashinfer_ms,
        "speedup_vs_flashinfer": flashinfer_ms / stream_ms,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "reference": {
            "available": reference_output is not None,
            "max_sequence_length": int(max(lengths_cpu)),
            "streamattn_max_abs_error": stream_reference_max_error,
            "streamattn_mean_abs_error": stream_reference_mean_error,
            "flashinfer_max_abs_error": selected_flashinfer["reference_max_abs_error"],
            "flashinfer_mean_abs_error": selected_flashinfer[
                "reference_mean_abs_error"
            ],
        },
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
    parser.add_argument("--sm90-fragmented-experimental", action="store_true")
    parser.add_argument("--sm90-fragmented-ragged-experimental", action="store_true")
    parser.add_argument(
        "--length-profile",
        choices=(
            "full",
            "tail",
            "mild",
            "severe",
            "short",
            "floor",
            "tiny",
            "minimum",
        ),
        default="full",
    )
    parser.add_argument("--workspace-mb", type=int, default=128)
    parser.add_argument("--flashinfer-backends", default="auto")
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
