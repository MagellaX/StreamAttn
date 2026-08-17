"""Benchmark StreamAttn's transposed Hopper WGMMA exact-decode path.

This is the first implementation gate for the exact-native transposed dataflow:

    K_tile [64, 64] @ Q_group.T [64, 8] -> scores [64, 8]

The context axis maps to WGMMA M=64 and a true-GQA group maps to N=8. Unlike
the earlier ThunderKittens spike, this uses the native m64n8k16 atom and does
not pad the query group to 16 rows. The benchmark retains isolated QK and QK+PV
floors, then measures the complete online-softmax partial and exact LSE merge.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_stream_attn_gate0_wrapper import _error, _time_cuda  # noqa: E402

try:
    import flashinfer

    FLASHINFER_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on benchmark environment
    flashinfer = None
    FLASHINFER_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


from stream_attention.backends.sm90.transposed_gqa_exact import (  # noqa: E402
    ExactDecodePlan,
    compile_transposed_gqa_exact_extension,
)



def _parse_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one split count is required")
    return values


def _compile_extension(*, cutlass_root: Path, build_dir: Path | None, verbose: bool):
    resolved_build_dir = (
        Path(tempfile.mkdtemp(prefix="streamattn_transposed_wgmma_qk_"))
        if build_dir is None
        else build_dir.expanduser().resolve()
    )
    started = time.perf_counter()
    extension = compile_transposed_gqa_exact_extension(
        cutlass_root=cutlass_root,
        build_dir=resolved_build_dir,
        verbose=verbose,
    )
    return extension, time.perf_counter() - started


def _time_repeated(fn, *, warmup: int, iters: int, repeats: int) -> tuple[float, list[float]]:
    samples = [
        _time_cuda(fn, device=torch.device("cuda"), warmup=warmup, iters=iters)
        for _ in range(repeats)
    ]
    return float(statistics.median(samples)), samples


def _paired_cuda_ratio(
    candidate,
    reference,
    *,
    device: torch.device,
    warmup: int,
    iters: int,
    repeats: int,
) -> dict[str, Any]:
    """Time candidate/reference in alternating order and report reference/candidate."""

    candidate_ms: list[float] = []
    reference_ms: list[float] = []
    ratios: list[float] = []
    for pair_idx in range(repeats):
        if pair_idx % 2 == 0:
            candidate_value = _time_cuda(
                candidate, device=device, warmup=warmup, iters=iters
            )
            reference_value = _time_cuda(
                reference, device=device, warmup=warmup, iters=iters
            )
        else:
            reference_value = _time_cuda(
                reference, device=device, warmup=warmup, iters=iters
            )
            candidate_value = _time_cuda(
                candidate, device=device, warmup=warmup, iters=iters
            )
        candidate_ms.append(float(candidate_value))
        reference_ms.append(float(reference_value))
        ratios.append(float(reference_value / candidate_value))
    return {
        "candidate_ms": candidate_ms,
        "reference_ms": reference_ms,
        "ratios": ratios,
        "ratio_median": float(statistics.median(ratios)),
        "ratio_min": float(min(ratios)),
        "wins": int(sum(ratio > 1.0 for ratio in ratios)),
        "trials": len(ratios),
    }


def _flashinfer_batched_runner(
    q: torch.Tensor,
    k_nhd: torch.Tensor,
    v_nhd: torch.Tensor,
    *,
    page_size: int,
):
    if flashinfer is None:
        raise RuntimeError(f"FlashInfer import failed: {FLASHINFER_IMPORT_ERROR}")
    batch, kv_len, kv_heads, dim = k_nhd.shape
    if kv_len % page_size:
        raise ValueError("kv_len must be divisible by page_size")
    pages_per_request = kv_len // page_size
    key_pages = k_nhd.view(batch * pages_per_request, page_size, kv_heads, dim)
    value_pages = v_nhd.view(batch * pages_per_request, page_size, kv_heads, dim)
    cache = torch.stack((key_pages, value_pages), dim=1).contiguous()
    total_pages = batch * pages_per_request
    indptr = torch.arange(
        0,
        total_pages + 1,
        pages_per_request,
        device=q.device,
        dtype=torch.int32,
    )
    indices = torch.arange(total_pages, device=q.device, dtype=torch.int32)
    last_page_len = torch.full((batch,), page_size, device=q.device, dtype=torch.int32)
    workspace = torch.empty(128 * 1024 * 1024, device=q.device, dtype=torch.uint8)
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        use_tensor_cores=True,
        backend="auto",
    )
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        q.shape[1],
        kv_heads,
        dim,
        page_size,
        pos_encoding_mode="NONE",
        q_data_type=q.dtype,
        kv_data_type=q.dtype,
        o_data_type=q.dtype,
        sm_scale=1.0 / math.sqrt(float(dim)),
        disable_split_kv=False,
    )
    out = torch.empty_like(q)

    def run() -> torch.Tensor:
        return wrapper.run(q, cache, out=out)

    return run


def profile(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.head_dim != 64 or args.q_heads // args.kv_heads != 8:
        raise ValueError("this milestone requires D64 and true-GQA group size 8")
    if args.kv_len % 64:
        raise ValueError("kv_len must be divisible by 64")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    q = torch.randn(args.batch, args.q_heads, args.head_dim, device=device, dtype=torch.bfloat16)
    k_nhd = torch.randn(
        args.batch,
        args.kv_len,
        args.kv_heads,
        args.head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    v_nhd = torch.randn_like(k_nhd)
    q_group = q.view(args.batch, args.kv_heads, 8, args.head_dim).contiguous()
    k_group = k_nhd.permute(0, 2, 1, 3).contiguous()
    scores = torch.empty(
        args.batch,
        args.kv_heads,
        args.kv_len,
        8,
        device=device,
        dtype=torch.float32,
    )

    cutlass_root = Path(args.cutlass_root).expanduser().resolve()
    if not (cutlass_root / "include" / "cute" / "tensor.hpp").is_file():
        raise FileNotFoundError(f"invalid CUTLASS root: {cutlass_root}")
    print(f"[transposed-qk] compiling from {cutlass_root}", flush=True)
    os.environ["STREAMATTN_CUTLASS_ROOT"] = str(cutlass_root)
    if args.build_dir:
        os.environ["STREAMATTN_EXACT_BUILD_DIR"] = str(
            Path(args.build_dir).expanduser().resolve()
        )
    ext, compile_s = _compile_extension(
        cutlass_root=cutlass_root,
        build_dir=Path(args.build_dir) if args.build_dir else None,
        verbose=args.compile_verbose,
    )
    print(f"[transposed-qk] compile_s={compile_s:.3f}", flush=True)

    split_counts = _parse_ints(args.num_splits_list)
    timings: dict[str, float] = {}
    samples: dict[str, list[float]] = {}
    checksum_timings: dict[str, float] = {}
    checksum_samples: dict[str, list[float]] = {}
    async_checksum_timings: dict[str, float] = {}
    async_checksum_samples: dict[str, list[float]] = {}
    qkpv_timings: dict[str, float] = {}
    qkpv_samples: dict[str, list[float]] = {}
    exact_partial_timings: dict[str, float] = {}
    exact_partial_samples: dict[str, list[float]] = {}
    exact_merge_timings: dict[str, float] = {}
    exact_merge_samples: dict[str, list[float]] = {}
    exact_end_to_end_timings: dict[str, float] = {}
    exact_end_to_end_samples: dict[str, list[float]] = {}
    quality: dict[str, dict[str, Any]] = {}
    checksum_quality: dict[str, dict[str, float]] = {}
    async_checksum_quality: dict[str, dict[str, float]] = {}
    qkpv_quality: dict[str, dict[str, float]] = {}
    exact_partial_quality: dict[str, dict[str, float]] = {}
    exact_merged_quality: dict[str, dict[str, float]] = {}
    reference = torch.einsum("bhgd,bhnd->bhng", q_group.float(), k_group.float())
    reference_flat = reference.view(args.batch * args.kv_heads, args.kv_len, 8)
    v_group = v_nhd.permute(0, 2, 1, 3).contiguous()
    v_flat = v_group.view(
        args.batch * args.kv_heads, args.kv_len, args.head_dim
    )
    # The floor kernel intentionally stages QK through a BF16 P tile before
    # WGMMA PV.  Validate against those staged semantics rather than an FP32
    # QK tensor, otherwise the checksum aggregates harmless BF16 rounding over
    # N * Hq * D products and reports a misleadingly large absolute error.
    qkpv_reference_flat = reference_flat.to(torch.bfloat16).float()
    qkpv_token_contribution = (
        qkpv_reference_flat.sum(dim=2) * v_flat.float().sum(dim=2)
    )
    exact_probabilities = torch.softmax(reference_flat * 0.125, dim=1)
    exact_reference_out = torch.einsum(
        "gnh,gnd->ghd", exact_probabilities, v_flat.float()
    )
    for splits in split_counts:
        def run(splits: int = splits) -> torch.Tensor:
            ext.qk_out(q_group, k_group, scores, splits)
            return scores

        scores.fill_(float("nan"))
        run()
        torch.cuda.synchronize()
        first_output = scores.clone()
        run()
        torch.cuda.synchronize()
        repeat_diff = (scores - first_output).abs()
        finite = torch.isfinite(scores)
        finite_count = int(finite.sum().item())
        total_count = scores.numel()
        if finite_count == total_count:
            diff = (scores - reference).abs()
            flat_max = int(diff.reshape(-1).argmax().item())
            quality[str(splits)] = {
                **_error(scores, reference),
                "nonfinite_count": 0,
                "large_error_count_gt_1e-4": int((diff > 1.0e-4).sum().item()),
                "max_error_flat_index": flat_max,
                "repeat_max_abs_diff": float(repeat_diff.max().item()),
                "repeat_changed_count_gt_1e-6": int((repeat_diff > 1.0e-6).sum().item()),
            }
        else:
            quality[str(splits)] = {
                "max_abs_error": None,
                "mean_abs_error": None,
                "nonfinite_count": total_count - finite_count,
                "large_error_count_gt_1e-4": None,
                "max_error_flat_index": None,
                "repeat_max_abs_diff": None,
                "repeat_changed_count_gt_1e-6": None,
            }
        median_ms, raw_samples = _time_repeated(
            run,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        timings[str(splits)] = median_ms
        samples[str(splits)] = raw_samples

        checksums = torch.empty(
            args.batch * args.kv_heads,
            splits,
            device=device,
            dtype=torch.float32,
        )
        num_tiles = args.kv_len // 64
        tiles_per_split = (num_tiles + splits - 1) // splits
        reference_checksums = torch.zeros_like(checksums)
        qkpv_reference_checksums = torch.zeros_like(checksums)
        for split in range(splits):
            token_begin = split * tiles_per_split * 64
            token_end = min(args.kv_len, token_begin + tiles_per_split * 64)
            if token_begin < token_end:
                reference_checksums[:, split] = reference_flat[:, token_begin:token_end].sum(dim=(1, 2))
                qkpv_reference_checksums[:, split] = qkpv_token_contribution[:, token_begin:token_end].sum(dim=1)

        def run_checksum(splits: int = splits) -> torch.Tensor:
            ext.qk_checksum_out(q_group, k_group, checksums, splits)
            return checksums

        run_checksum()
        torch.cuda.synchronize()
        checksum_quality[str(splits)] = _error(checksums, reference_checksums)
        checksum_ms, checksum_raw_samples = _time_repeated(
            run_checksum,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        checksum_timings[str(splits)] = checksum_ms
        checksum_samples[str(splits)] = checksum_raw_samples

        def run_async_checksum(splits: int = splits) -> torch.Tensor:
            ext.qk_async_checksum_out(q_group, k_group, checksums, splits)
            return checksums

        run_async_checksum()
        torch.cuda.synchronize()
        async_checksum_quality[str(splits)] = _error(checksums, reference_checksums)
        async_checksum_ms, async_checksum_raw_samples = _time_repeated(
            run_async_checksum,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        async_checksum_timings[str(splits)] = async_checksum_ms
        async_checksum_samples[str(splits)] = async_checksum_raw_samples

        def run_qkpv(splits: int = splits) -> torch.Tensor:
            ext.qkpv_async_checksum_out(q_group, k_group, v_group, checksums, splits)
            return checksums

        run_qkpv()
        torch.cuda.synchronize()
        first_qkpv = checksums.clone()
        run_qkpv()
        torch.cuda.synchronize()
        qkpv_error = _error(checksums, qkpv_reference_checksums)
        reference_scale = float(qkpv_reference_checksums.abs().max().item())
        qkpv_quality[str(splits)] = {
            **qkpv_error,
            "max_abs_reference": reference_scale,
            "normalized_max_abs_error": (
                qkpv_error["max_abs_error"] / max(reference_scale, 1.0e-12)
            ),
            "repeat_max_abs_diff": float((checksums - first_qkpv).abs().max().item()),
        }
        qkpv_ms, qkpv_raw_samples = _time_repeated(
            run_qkpv,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        qkpv_timings[str(splits)] = qkpv_ms
        qkpv_samples[str(splits)] = qkpv_raw_samples

        partial_o = torch.empty(
            args.batch * args.kv_heads,
            splits,
            8,
            args.head_dim,
            device=device,
            dtype=torch.float32,
        )
        partial_lse = torch.empty(
            args.batch * args.kv_heads,
            splits,
            8,
            device=device,
            dtype=torch.float32,
        )

        def run_exact_partial(splits: int = splits) -> torch.Tensor:
            ext.exact_partial_out(
                q_group, k_group, v_group, partial_o, partial_lse, splits
            )
            return partial_o

        run_exact_partial()
        torch.cuda.synchronize()
        first_partial_o = partial_o.clone()
        first_partial_lse = partial_lse.clone()
        run_exact_partial()
        torch.cuda.synchronize()
        merge_max = partial_lse.max(dim=1, keepdim=True).values
        merge_weights = torch.exp2(partial_lse - merge_max)
        merged_out = (
            (partial_o * merge_weights.unsqueeze(-1)).sum(dim=1)
            / merge_weights.sum(dim=1).unsqueeze(-1)
        )
        exact_error = _error(merged_out, exact_reference_out)
        exact_partial_quality[str(splits)] = {
            **exact_error,
            "partial_o_repeat_max_abs_diff": float(
                (partial_o - first_partial_o).abs().max().item()
            ),
            "partial_lse_repeat_max_abs_diff": float(
                (partial_lse - first_partial_lse).abs().max().item()
            ),
            "partial_o_nonfinite_count": int(
                (~torch.isfinite(partial_o)).sum().item()
            ),
            "partial_lse_nonfinite_count": int(
                (~torch.isfinite(partial_lse)).sum().item()
            ),
        }
        exact_partial_ms, exact_partial_raw_samples = _time_repeated(
            run_exact_partial,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        exact_partial_timings[str(splits)] = exact_partial_ms
        exact_partial_samples[str(splits)] = exact_partial_raw_samples

        exact_output = torch.empty(
            args.batch * args.kv_heads,
            8,
            args.head_dim,
            device=device,
            dtype=torch.bfloat16,
        )

        def run_exact_merge() -> torch.Tensor:
            ext.exact_merge_out(partial_o, partial_lse, exact_output)
            return exact_output

        run_exact_merge()
        torch.cuda.synchronize()
        first_exact_output = exact_output.clone()
        run_exact_merge()
        torch.cuda.synchronize()
        merged_error = _error(exact_output.float(), exact_reference_out)
        exact_merged_quality[str(splits)] = {
            **merged_error,
            "repeat_max_abs_diff": float(
                (exact_output - first_exact_output).abs().max().item()
            ),
            "nonfinite_count": int((~torch.isfinite(exact_output)).sum().item()),
        }
        exact_merge_ms, exact_merge_raw_samples = _time_repeated(
            run_exact_merge,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        exact_merge_timings[str(splits)] = exact_merge_ms
        exact_merge_samples[str(splits)] = exact_merge_raw_samples

        def run_exact_end_to_end() -> torch.Tensor:
            run_exact_partial()
            run_exact_merge()
            return exact_output

        exact_end_to_end_ms, exact_end_to_end_raw_samples = _time_repeated(
            run_exact_end_to_end,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        exact_end_to_end_timings[str(splits)] = exact_end_to_end_ms
        exact_end_to_end_samples[str(splits)] = exact_end_to_end_raw_samples
        print(
            f"[transposed-qk] splits={splits:>3} ctas={args.batch * args.kv_heads * splits:>4} "
            f"scores_ms={median_ms:.6f} checksum_ms={checksum_ms:.6f} "
            f"async_checksum_ms={async_checksum_ms:.6f} "
            f"qkpv_ms={qkpv_ms:.6f} "
            f"exact_partial_ms={exact_partial_ms:.6f} "
            f"merge_ms={exact_merge_ms:.6f} "
            f"exact_e2e_ms={exact_end_to_end_ms:.6f} "
            f"exact_out_err={exact_error['max_abs_error']:.6g} "
            f"max_err={quality[str(splits)]['max_abs_error']} "
            f"nonfinite={quality[str(splits)]['nonfinite_count']}",
            flush=True,
        )

    flashinfer_error = None
    flashinfer_ms = None
    try:
        flashinfer_run = _flashinfer_batched_runner(q, k_nhd, v_nhd, page_size=args.page_size)
        flashinfer_ms, flashinfer_samples = _time_repeated(
            flashinfer_run,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        flashinfer_output = flashinfer_run().clone()
        torch.cuda.synchronize()
        flashinfer_quality = _error(
            exact_output.view(args.batch, args.q_heads, args.head_dim).float(),
            flashinfer_output.float(),
        )
    except Exception as exc:  # pragma: no cover - depends on installed FlashInfer
        flashinfer_error = f"{type(exc).__name__}: {exc}"
        flashinfer_samples = []
        flashinfer_quality = None

    best_splits = min(timings, key=timings.get)
    best_ms = timings[best_splits]
    best_checksum_splits = min(checksum_timings, key=checksum_timings.get)
    best_checksum_ms = checksum_timings[best_checksum_splits]
    best_async_checksum_splits = min(async_checksum_timings, key=async_checksum_timings.get)
    best_async_checksum_ms = async_checksum_timings[best_async_checksum_splits]
    best_qkpv_splits = min(qkpv_timings, key=qkpv_timings.get)
    best_qkpv_ms = qkpv_timings[best_qkpv_splits]
    best_exact_partial_splits = min(exact_partial_timings, key=exact_partial_timings.get)
    best_exact_partial_ms = exact_partial_timings[best_exact_partial_splits]
    best_exact_merge_splits = min(exact_merge_timings, key=exact_merge_timings.get)
    best_exact_merge_ms = exact_merge_timings[best_exact_merge_splits]
    best_exact_end_to_end_splits = min(
        exact_end_to_end_timings, key=exact_end_to_end_timings.get
    )
    best_exact_end_to_end_ms = exact_end_to_end_timings[
        best_exact_end_to_end_splits
    ]
    backend_plan = ExactDecodePlan.build(
        q.unsqueeze(1),
        k_group,
        v_group,
        num_splits=int(best_exact_end_to_end_splits),
        cutlass_root=cutlass_root,
        build_dir=Path(args.build_dir) if args.build_dir else None,
        promoted_only=False,
    )
    backend_plan.run()
    torch.cuda.synchronize()
    first_backend_output = backend_plan.output.clone()
    backend_plan.run()
    torch.cuda.synchronize()
    backend_plan_ms, backend_plan_samples = _time_repeated(
        backend_plan.run,
        warmup=args.warmup,
        iters=args.iters,
        repeats=args.repeats,
    )
    backend_plan_quality = {
        **_error(
            backend_plan.output.view(
                args.batch * args.kv_heads, 8, args.head_dim
            ).float(),
            exact_reference_out,
        ),
        "repeat_max_abs_diff": float(
            (backend_plan.output - first_backend_output).abs().max().item()
        ),
        "nonfinite_count": int(
            (~torch.isfinite(backend_plan.output)).sum().item()
        ),
        "workspace_bytes": backend_plan.workspace_bytes,
        "backend": backend_plan.backend,
    }
    backend_plan.run_combined()
    torch.cuda.synchronize()
    first_combined_output = backend_plan.output.clone()
    backend_plan.run_combined()
    torch.cuda.synchronize()
    backend_plan_combined_ms, backend_plan_combined_samples = _time_repeated(
        backend_plan.run_combined,
        warmup=args.warmup,
        iters=args.iters,
        repeats=args.repeats,
    )
    backend_plan_combined_quality = {
        **_error(
            backend_plan.output.view(
                args.batch * args.kv_heads, 8, args.head_dim
            ).float(),
            exact_reference_out,
        ),
        "repeat_max_abs_diff": float(
            (backend_plan.output - first_combined_output).abs().max().item()
        ),
        "nonfinite_count": int((~torch.isfinite(backend_plan.output)).sum().item()),
        "workspace_bytes": backend_plan.workspace_bytes,
        "backend": backend_plan.backend,
    }
    combined_vs_two_call = _paired_cuda_ratio(
        backend_plan.run_combined,
        backend_plan.run_two_call,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
        repeats=max(9, args.repeats),
    )
    serving_dispatch_quality = None
    serving_dispatch_ms = None
    serving_dispatch_samples: list[float] = []
    if (
        args.batch == 4
        and args.q_heads == 16
        and args.kv_heads == 2
        and args.kv_len == 32768
        and args.head_dim == 64
    ):
        from stream_attention.decode import StreamAttnExactNativeDirectRunner

        serving_output = torch.empty_like(q.unsqueeze(1))
        serving_runner = StreamAttnExactNativeDirectRunner(
            query=q.unsqueeze(1),
            key_cache=k_group,
            value_cache=v_group,
            output=serving_output,
            info=None,
        )
        serving_runner.run()
        torch.cuda.synchronize()
        serving_dispatch_quality = {
            **_error(
                serving_output.view(
                    args.batch * args.kv_heads, 8, args.head_dim
                ).float(),
                exact_reference_out,
            ),
            "backend_variant": serving_runner.backend_variant,
            "nonfinite_count": int((~torch.isfinite(serving_output)).sum().item()),
        }
        serving_dispatch_ms, serving_dispatch_samples = _time_repeated(
            serving_runner.run,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
    paired_exact_ms: list[float] = []
    paired_flashinfer_ms: list[float] = []
    paired_speedups: list[float] = []
    paired_combined_vs_flashinfer = None
    if flashinfer_ms is not None:
        paired_two_call_vs_flashinfer = _paired_cuda_ratio(
            backend_plan.run_two_call,
            flashinfer_run,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
            repeats=max(9, args.repeats),
        )
        paired_exact_ms = paired_two_call_vs_flashinfer["candidate_ms"]
        paired_flashinfer_ms = paired_two_call_vs_flashinfer["reference_ms"]
        paired_speedups = paired_two_call_vs_flashinfer["ratios"]
        paired_speedup_median = paired_two_call_vs_flashinfer["ratio_median"]
        paired_speedup_min = paired_two_call_vs_flashinfer["ratio_min"]
        paired_combined_vs_flashinfer = _paired_cuda_ratio(
            backend_plan.run_combined,
            flashinfer_run,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
            repeats=max(9, args.repeats),
        )
    else:
        paired_speedup_median = None
        paired_speedup_min = None
    result: dict[str, Any] = {
        "schema": "streamattn.transposed_wgmma_exact_decode.v1",
        "device": torch.cuda.get_device_name(device),
        "shape": {
            "batch": args.batch,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "group_size": 8,
            "kv_len": args.kv_len,
            "head_dim": args.head_dim,
            "dtype": "bf16",
            "wgmma_atom": "m64n8k16.f32.bf16.bf16",
            "qk_orientation": "K[64,64] @ Q.T[64,8]",
            "split_state_lse": "log2",
        },
        "compile_s": compile_s,
        "timing": {
            "qk_ms_by_splits": timings,
            "qk_samples_ms_by_splits": samples,
            "qk_checksum_ms_by_splits": checksum_timings,
            "qk_checksum_samples_ms_by_splits": checksum_samples,
            "qk_async_checksum_ms_by_splits": async_checksum_timings,
            "qk_async_checksum_samples_ms_by_splits": async_checksum_samples,
            "qkpv_async_checksum_ms_by_splits": qkpv_timings,
            "qkpv_async_checksum_samples_ms_by_splits": qkpv_samples,
            "exact_partial_ms_by_splits": exact_partial_timings,
            "exact_partial_samples_ms_by_splits": exact_partial_samples,
            "exact_merge_ms_by_splits": exact_merge_timings,
            "exact_merge_samples_ms_by_splits": exact_merge_samples,
            "exact_end_to_end_ms_by_splits": exact_end_to_end_timings,
            "exact_end_to_end_samples_ms_by_splits": exact_end_to_end_samples,
            "best_splits": int(best_splits),
            "best_qk_ms": best_ms,
            "best_cta_count": args.batch * args.kv_heads * int(best_splits),
            "best_checksum_splits": int(best_checksum_splits),
            "best_qk_checksum_ms": best_checksum_ms,
            "best_checksum_cta_count": args.batch * args.kv_heads * int(best_checksum_splits),
            "best_async_checksum_splits": int(best_async_checksum_splits),
            "best_qk_async_checksum_ms": best_async_checksum_ms,
            "best_async_checksum_cta_count": (
                args.batch * args.kv_heads * int(best_async_checksum_splits)
            ),
            "best_qkpv_splits": int(best_qkpv_splits),
            "best_qkpv_ms": best_qkpv_ms,
            "best_qkpv_cta_count": args.batch * args.kv_heads * int(best_qkpv_splits),
            "best_exact_partial_splits": int(best_exact_partial_splits),
            "best_exact_partial_ms": best_exact_partial_ms,
            "best_exact_partial_cta_count": (
                args.batch * args.kv_heads * int(best_exact_partial_splits)
            ),
            "best_exact_merge_splits": int(best_exact_merge_splits),
            "best_exact_merge_ms": best_exact_merge_ms,
            "best_exact_end_to_end_splits": int(best_exact_end_to_end_splits),
            "best_exact_end_to_end_ms": best_exact_end_to_end_ms,
            "backend_plan_ms": backend_plan_ms,
            "backend_plan_samples_ms": backend_plan_samples,
            "backend_plan_two_call_ms": backend_plan_ms,
            "backend_plan_two_call_samples_ms": backend_plan_samples,
            "backend_plan_combined_ms": backend_plan_combined_ms,
            "backend_plan_combined_samples_ms": backend_plan_combined_samples,
            "serving_dispatch_ms": serving_dispatch_ms,
            "serving_dispatch_samples_ms": serving_dispatch_samples,
            "paired_combined_vs_two_call": combined_vs_two_call,
            "paired_combined_vs_flashinfer": paired_combined_vs_flashinfer,
            "flashinfer_batched_exact_ms": flashinfer_ms,
            "flashinfer_samples_ms": flashinfer_samples,
            "qk_budget_fraction_of_flashinfer": (
                best_ms / flashinfer_ms if flashinfer_ms is not None else None
            ),
            "qk_checksum_budget_fraction_of_flashinfer": (
                best_checksum_ms / flashinfer_ms if flashinfer_ms is not None else None
            ),
            "qk_async_checksum_budget_fraction_of_flashinfer": (
                best_async_checksum_ms / flashinfer_ms if flashinfer_ms is not None else None
            ),
            "qkpv_budget_fraction_of_flashinfer": (
                best_qkpv_ms / flashinfer_ms if flashinfer_ms is not None else None
            ),
            "exact_partial_budget_fraction_of_flashinfer": (
                best_exact_partial_ms / flashinfer_ms
                if flashinfer_ms is not None
                else None
            ),
            "exact_end_to_end_speedup_vs_flashinfer": (
                flashinfer_ms / best_exact_end_to_end_ms
                if flashinfer_ms is not None
                else None
            ),
            "paired_exact_ms": paired_exact_ms,
            "paired_flashinfer_ms": paired_flashinfer_ms,
            "paired_speedups": paired_speedups,
            "paired_speedup_median": paired_speedup_median,
            "paired_speedup_min": paired_speedup_min,
        },
        "quality": quality,
        "checksum_quality": checksum_quality,
        "async_checksum_quality": async_checksum_quality,
        "qkpv_quality": qkpv_quality,
        "exact_partial_quality": exact_partial_quality,
        "exact_merged_quality": exact_merged_quality,
        "backend_plan_quality": backend_plan_quality,
        "backend_plan_combined_quality": backend_plan_combined_quality,
        "serving_dispatch_quality": serving_dispatch_quality,
        "exact_vs_flashinfer_quality": flashinfer_quality,
        "flashinfer_error": flashinfer_error,
        "flashinfer_import_error": FLASHINFER_IMPORT_ERROR,
        "decision": {
            "qk_gate": (
                "pass"
                if flashinfer_ms is not None and best_async_checksum_ms <= 0.5 * flashinfer_ms
                else "fail"
            ),
            "criterion": "cp.async storeless QK milestone <= 50% of matching batched FlashInfer exact",
            "exact_native_gate": (
                "pass"
                if paired_speedup_median is not None
                and paired_speedup_median > 1.0
                and paired_speedup_min > 0.98
                and flashinfer_quality is not None
                and flashinfer_quality["max_abs_error"] <= 5.0e-4
                and exact_merged_quality[best_exact_end_to_end_splits]["nonfinite_count"] == 0
                and exact_merged_quality[best_exact_end_to_end_splits]["repeat_max_abs_diff"] == 0.0
                and backend_plan_quality["nonfinite_count"] == 0
                and backend_plan_quality["repeat_max_abs_diff"] == 0.0
                and backend_plan_quality["max_abs_error"] <= 5.0e-4
                else "fail"
            ),
            "combined_dispatch_gate": (
                "pass"
                if combined_vs_two_call["ratio_median"] > 1.0
                and combined_vs_two_call["ratio_min"] > 0.995
                and backend_plan_combined_quality["nonfinite_count"] == 0
                and backend_plan_combined_quality["repeat_max_abs_diff"] == 0.0
                and backend_plan_combined_quality["max_abs_error"] <= 5.0e-4
                else "fail"
            ),
            "combined_dispatch_criterion": (
                "paired median faster than two-call plan, paired min >=0.995x, "
                "max output delta <=5e-4, finite and deterministic"
            ),
            "exact_native_criterion": (
                "paired median faster than matching FlashInfer, paired min >=0.98x, "
                "max output delta <=5e-4, finite and deterministic"
            ),
        },
    }
    print(json.dumps(result, indent=2), flush=True)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--kv-len", type=int, default=32768)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--num-splits-list", default="8,16,17,32,33,64,128,256,512")
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cutlass-root", required=True)
    parser.add_argument("--build-dir", default="")
    parser.add_argument("--compile-verbose", action="store_true")
    parser.add_argument("--output-json", default="")
    profile(parser.parse_args())


if __name__ == "__main__":
    main()
