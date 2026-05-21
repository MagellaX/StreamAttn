"""Measure KV-group seed-only repair economics in the TK scheduler.

The compact TK head-mode kernel already supports the row modes needed for
repair:

* mode 0: exact row, active on every KV chunk;
* mode 1: trusted seed-only row, active only on seed chunks;
* mode 2: padding/inactive row.

This benchmark isolates one true-GQA KV group, sweeps the number of exact
repair rows inside that group, and reports the break-even point where the
group no longer beats exact decode.  It is a scheduler economics probe; it
uses synthetic BF16 tensors by default so the timing question is not blocked
on model capture.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_gate0_true_gqa import _dense_true_gqa  # noqa: E402
from benchmarks.profile_head_mode_decode_cuda import _flashinfer_exact, _torch_head_mode_reference  # noqa: E402
from benchmarks.profile_head_mode_decode_cuda import _compile_extension as _compile_scalar_repair_extension  # noqa: E402
from benchmarks.profile_stream_attn_gate0_wrapper import _dtype, _error, _time_cuda  # noqa: E402
from benchmarks.profile_tk_tensor_core_exact_decode import (  # noqa: E402
    _compile_extension,
    _find_or_clone_tk,
    _pack_active_chunks_by_kv_group,
    _pack_kv_head_major,
    _pack_q_by_kv_group,
    _pack_row_modes_by_kv_group,
    _unpack_q_by_kv_group,
)


def _parse_ints(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _seed_tiles(
    *,
    kv_len: int,
    block_size: int,
    sink_blocks: int,
    recent_blocks: int,
    middle_seed_blocks: int,
    block_order: str,
) -> set[int]:
    from benchmarks.profile_tk_tensor_core_exact_decode import _tile_is_seed

    return {
        tile
        for tile in range(kv_len // 16)
        if _tile_is_seed(
            tile=tile,
            kv_len=kv_len,
            block_size=block_size,
            sink_blocks=sink_blocks,
            recent_blocks=recent_blocks,
            middle_seed_blocks=middle_seed_blocks,
            block_order=block_order,
        )
    }


def repair_work_model(
    *,
    q_heads: int,
    kv_len: int,
    num_chunks: int,
    seed_heads: Sequence[int],
    block_size: int,
    sink_blocks: int,
    recent_blocks: int,
    middle_seed_blocks: int,
    block_order: str,
) -> Dict[str, Any]:
    """Return scheduler-visible work for a single KV group."""
    tiles = kv_len // 16
    if tiles % num_chunks != 0:
        raise ValueError("num_chunks must divide kv_len/16")
    tiles_per_chunk = tiles // num_chunks
    seed_set = set(seed_heads)
    exact_rows = [head for head in range(q_heads) if head not in seed_set]
    trusted_seed_rows = [head for head in range(q_heads) if head in seed_set]
    seed_tiles = _seed_tiles(
        kv_len=kv_len,
        block_size=block_size,
        sink_blocks=sink_blocks,
        recent_blocks=recent_blocks,
        middle_seed_blocks=middle_seed_blocks,
        block_order=block_order,
    )
    active_chunks: list[int] = []
    active_tile_count = 0
    active_row_tile_work = 0
    for chunk in range(num_chunks):
        chunk_has_work = False
        for tile in range(chunk * tiles_per_chunk, (chunk + 1) * tiles_per_chunk):
            seed_tile = tile in seed_tiles
            active_rows = len(exact_rows) + (len(trusted_seed_rows) if seed_tile else 0)
            if active_rows:
                chunk_has_work = True
                active_tile_count += 1
                active_row_tile_work += active_rows
        if chunk_has_work:
            active_chunks.append(chunk)

    dense_row_tile_work = q_heads * tiles
    return {
        "repair_rows": exact_rows,
        "trusted_seed_rows": trusted_seed_rows,
        "seed_tile_count": len(seed_tiles),
        "active_chunk_count": len(active_chunks),
        "active_chunks_preview": _preview_ints(active_chunks),
        "kv_tiles_loaded": active_tile_count,
        "q_row_tiles_computed": active_row_tile_work,
        "dense_kv_tiles": tiles,
        "dense_q_row_tiles": dense_row_tile_work,
        "kv_tile_load_reduction": 1.0 - (active_tile_count / tiles if tiles else 0.0),
        "row_work_reduction": 1.0 - (active_row_tile_work / dense_row_tile_work if dense_row_tile_work else 0.0),
    }


def _per_head_errors(actual: torch.Tensor, expected: torch.Tensor) -> List[Dict[str, float]]:
    diff = (actual - expected).detach().abs().float()
    rows = []
    for head in range(diff.shape[1]):
        head_diff = diff[:, head, :]
        rows.append(
            {
                "head": head,
                "max_abs_error": float(head_diff.max().item()),
                "mean_abs_error": float(head_diff.mean().item()),
            }
        )
    return rows


def _preview_ints(values: Sequence[int], *, limit: int = 8) -> Dict[str, Any]:
    if len(values) <= limit * 2:
        return {"count": len(values), "values": list(values)}
    return {"count": len(values), "first": list(values[:limit]), "last": list(values[-limit:])}


def profile(args: argparse.Namespace) -> Dict[str, Any]:
    if args.dtype != "bf16":
        raise ValueError("this benchmark currently supports --dtype bf16 only")
    if args.kv_heads != 1:
        raise ValueError("repair economics benchmark isolates one KV group; use --kv-heads 1")
    if args.q_heads > 16:
        raise ValueError("q_heads must fit in the 16-row TK tile")
    if args.head_dim not in (64, 128):
        raise ValueError("head_dim must be 64 or 128")
    if args.kv_len % 16 != 0:
        raise ValueError("kv_len must be divisible by 16")
    if (args.kv_len // 16) % args.num_chunks != 0:
        raise ValueError("num_chunks must divide kv_len/16")

    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    torch.manual_seed(args.seed)

    q = torch.randn((1, args.q_heads, args.head_dim), device=device, dtype=dtype)
    k = torch.randn((1, args.kv_len, args.kv_heads, args.head_dim), device=device, dtype=dtype)
    v = torch.randn_like(k)
    q_group = _pack_q_by_kv_group(q, args.kv_heads, padded_rows=16)
    k_group = _pack_kv_head_major(k)
    v_group = _pack_kv_head_major(v)

    repair_counts = _parse_ints(args.repair_counts)
    repair_order = _parse_ints(args.repair_order) if args.repair_order else list(range(args.q_heads))
    if sorted(repair_order) != list(range(args.q_heads)):
        raise ValueError("repair_order must contain each local row exactly once")

    tk_root = _find_or_clone_tk(args)
    print(
        "[tk-repair] compiling extension "
        f"D={args.head_dim} q_heads={args.q_heads} kv_len={args.kv_len}",
        flush=True,
    )
    compile_start = time.perf_counter()
    ext = _compile_extension(
        tk_root=tk_root,
        cuda_arch=args.cuda_arch,
        torch_cuda_arch_list=args.torch_cuda_arch_list,
        verbose=args.compile_verbose,
    )
    compile_s = time.perf_counter() - compile_start
    print(f"[tk-repair] compile finished in {compile_s:.2f}s", flush=True)
    print("[tk-repair] compiling compact scalar repair extension", flush=True)
    scalar_compile_start = time.perf_counter()
    scalar_repair_ext = _compile_scalar_repair_extension(verbose=args.compile_verbose)
    scalar_compile_s = time.perf_counter() - scalar_compile_start
    print(f"[tk-repair] scalar repair compile finished in {scalar_compile_s:.2f}s", flush=True)

    def dense_true() -> torch.Tensor:
        return _dense_true_gqa(q[:, None, :, :], k, v)[:, 0]

    def exact_group_raw() -> torch.Tensor:
        return ext.exact_decode_chunk_merged(q_group, k_group, v_group, args.num_chunks)

    print("[tk-repair] building references", flush=True)
    dense_ref = dense_true()
    exact_out = _unpack_q_by_kv_group(exact_group_raw(), args.q_heads)
    exact_quality = _error(exact_out[:, None, :, :], dense_ref[:, None, :, :])

    def _time_flashinfer_exact(q_in: torch.Tensor, *, use_tensor_cores: bool) -> tuple[float | None, str | None]:
        try:
            return (
                _time_cuda(
                    lambda: _flashinfer_exact(q_in, k, v, use_tensor_cores=use_tensor_cores),
                    device=device,
                    warmup=args.warmup,
                    iters=args.iters,
                ),
                None,
            )
        except Exception as exc:  # pragma: no cover - depends on optional backend
            return None, repr(exc)

    flashinfer_tc_ms, flashinfer_tc_error = _time_flashinfer_exact(q, use_tensor_cores=True)
    flashinfer_no_tc_ms, flashinfer_no_tc_error = _time_flashinfer_exact(q, use_tensor_cores=False)
    flashinfer_candidates = [value for value in (flashinfer_tc_ms, flashinfer_no_tc_ms) if value is not None]
    flashinfer_ms = min(flashinfer_candidates) if flashinfer_candidates else None
    flashinfer_error = flashinfer_tc_error or flashinfer_no_tc_error

    dense_ms = _time_cuda(dense_true, device=device, warmup=args.warmup, iters=args.iters)
    exact_ms = _time_cuda(exact_group_raw, device=device, warmup=args.warmup, iters=args.iters)
    block_order_id = 0 if args.block_order == "sequential" else 1

    rows: list[dict[str, Any]] = []
    print("[tk-repair] sweeping repair counts", flush=True)
    for repair_count in repair_counts:
        repair_heads = repair_order[:repair_count]
        trusted_seed_heads = [head for head in range(args.q_heads) if head not in set(repair_heads)]
        row_modes = _pack_row_modes_by_kv_group(
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            seed_heads=trusted_seed_heads,
            padded_rows=16,
            device=device,
        )
        active_chunks, active_counts, flat_active_chunks, active_offsets, active_by_kv = _pack_active_chunks_by_kv_group(
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            seed_heads=trusted_seed_heads,
            kv_len=args.kv_len,
            num_chunks=args.num_chunks,
            block_size=args.block_size,
            sink_blocks=args.sink_blocks,
            recent_blocks=args.recent_blocks,
            middle_seed_blocks=args.middle_seed_blocks,
            block_order=args.block_order,
            device=device,
        )

        def tk_repair_group() -> torch.Tensor:
            return ext.head_mode_compact_chunk_merged(
                q_group,
                k_group,
                v_group,
                row_modes,
                active_chunks,
                active_counts,
                flat_active_chunks,
                active_offsets,
                args.num_chunks,
                args.block_size,
                args.sink_blocks,
                args.recent_blocks,
                args.middle_seed_blocks,
                block_order_id,
            )

        head_modes = torch.zeros(args.q_heads, device=device, dtype=torch.int32)
        if trusted_seed_heads:
            head_modes[torch.tensor(trusted_seed_heads, device=device, dtype=torch.long)] = 1

        def torch_repair_reference() -> torch.Tensor:
            return _torch_head_mode_reference(
                q,
                k,
                v,
                head_modes,
                block_size=args.block_size,
                sink_blocks=args.sink_blocks,
                recent_blocks=args.recent_blocks,
                middle_seed_blocks=args.middle_seed_blocks,
                block_order=args.block_order,
            )

        tk_out = _unpack_q_by_kv_group(tk_repair_group(), args.q_heads)
        ref_out = torch_repair_reference()
        timing_ms = _time_cuda(tk_repair_group, device=device, warmup=args.warmup, iters=args.iters)
        if repair_heads and flashinfer_ms is not None:
            repair_index = torch.tensor(repair_heads, device=device, dtype=torch.long)
            q_repair = q.index_select(1, repair_index).contiguous()
            repair_modes = torch.zeros(len(repair_heads), device=device, dtype=torch.int32)

            exact_repair_tc_ms, exact_repair_tc_error = _time_flashinfer_exact(q_repair, use_tensor_cores=True)
            exact_repair_no_tc_ms, exact_repair_no_tc_error = _time_flashinfer_exact(q_repair, use_tensor_cores=False)
            exact_repair_candidates = [
                value for value in (exact_repair_tc_ms, exact_repair_no_tc_ms) if value is not None
            ]
            exact_repair_ms = min(exact_repair_candidates) if exact_repair_candidates else None
            exact_repair_error = exact_repair_tc_error or exact_repair_no_tc_error

            def compact_scalar_repair() -> torch.Tensor:
                return scalar_repair_ext.head_mode_decode(
                    q_repair,
                    k,
                    v,
                    repair_modes,
                    args.block_size,
                    args.sink_blocks,
                    args.recent_blocks,
                    args.middle_seed_blocks,
                    block_order_id,
                    args.repair_threads,
                )

            try:
                compact_scalar_repair_ms = _time_cuda(
                    compact_scalar_repair,
                    device=device,
                    warmup=args.warmup,
                    iters=args.iters,
                )
                compact_scalar_repair_error = None
            except Exception as exc:  # pragma: no cover - optional dtype/backend constraints
                compact_scalar_repair_ms = None
                compact_scalar_repair_error = repr(exc)

            def compact_triton_repair() -> torch.Tensor:
                from stream_attention.kernels.gate0_compact_repair_triton import compact_repair_splitk_triton_forward

                return compact_repair_splitk_triton_forward(
                    q_repair,
                    k,
                    v,
                    num_chunks=args.repair_num_chunks,
                    block_d=args.repair_block_d,
                )

            try:
                compact_triton_repair_ms = _time_cuda(
                    compact_triton_repair,
                    device=device,
                    warmup=args.warmup,
                    iters=args.iters,
                )
                compact_triton_repair_error = None
            except Exception as exc:  # pragma: no cover - optional backend constraints
                compact_triton_repair_ms = None
                compact_triton_repair_error = repr(exc)
        else:
            exact_repair_ms = 0.0 if not repair_heads else None
            exact_repair_error = None if not repair_heads else flashinfer_error
            exact_repair_tc_ms = exact_repair_ms
            exact_repair_no_tc_ms = exact_repair_ms
            compact_scalar_repair_ms = 0.0 if not repair_heads else None
            compact_scalar_repair_error = None if not repair_heads else flashinfer_error
            compact_triton_repair_ms = 0.0 if not repair_heads else None
            compact_triton_repair_error = None if not repair_heads else flashinfer_error
        repair_candidates = {
            "flashinfer_best": exact_repair_ms,
            "scalar_cuda": compact_scalar_repair_ms,
            "triton_splitk": compact_triton_repair_ms,
        }
        available_repair_candidates = {
            name: value for name, value in repair_candidates.items() if value is not None
        }
        compact_repair_best_name = (
            min(available_repair_candidates, key=lambda name: float(available_repair_candidates[name]))
            if available_repair_candidates
            else None
        )
        compact_repair_best_ms = (
            float(available_repair_candidates[compact_repair_best_name]) if compact_repair_best_name else None
        )
        per_head = _per_head_errors(tk_out, dense_ref)
        trusted_errors = [row["max_abs_error"] for row in per_head if row["head"] in set(trusted_seed_heads)]
        repair_errors = [row["max_abs_error"] for row in per_head if row["head"] in set(repair_heads)]
        model = repair_work_model(
            q_heads=args.q_heads,
            kv_len=args.kv_len,
            num_chunks=args.num_chunks,
            seed_heads=trusted_seed_heads,
            block_size=args.block_size,
            sink_blocks=args.sink_blocks,
            recent_blocks=args.recent_blocks,
            middle_seed_blocks=args.middle_seed_blocks,
            block_order=args.block_order,
        )
        rows.append(
            {
                "repair_count": repair_count,
                "repair_heads": repair_heads,
                "trusted_seed_heads": trusted_seed_heads,
                "timing_ms": timing_ms,
                "exact_repair_flashinfer_ms": exact_repair_ms,
                "exact_repair_flashinfer_tc_ms": exact_repair_tc_ms,
                "exact_repair_flashinfer_no_tc_ms": exact_repair_no_tc_ms,
                "exact_repair_flashinfer_error": exact_repair_error,
                "compact_scalar_repair_ms": compact_scalar_repair_ms,
                "compact_scalar_repair_error": compact_scalar_repair_error,
                "compact_triton_repair_ms": compact_triton_repair_ms,
                "compact_triton_repair_error": compact_triton_repair_error,
                "compact_repair_best_ms": compact_repair_best_ms,
                "compact_repair_best_backend": compact_repair_best_name,
                "speedup_vs_dense_torch": dense_ms / timing_ms if timing_ms else None,
                "speedup_vs_tk_exact": exact_ms / timing_ms if timing_ms else None,
                "speedup_vs_flashinfer": flashinfer_ms / timing_ms if flashinfer_ms and timing_ms else None,
                "beats_flashinfer": bool(flashinfer_ms is not None and timing_ms < flashinfer_ms),
                "beats_tk_exact": bool(timing_ms < exact_ms),
                "active_chunks_by_kv_group": [_preview_ints(chunks) for chunks in active_by_kv],
                "work_model": model,
                "quality_vs_torch_head_mode_reference": _error(tk_out[:, None, :, :], ref_out[:, None, :, :]),
                "quality_vs_dense_true_gqa": _error(tk_out[:, None, :, :], dense_ref[:, None, :, :]),
                "max_trusted_seed_error_after_repair": max(trusted_errors) if trusted_errors else 0.0,
                "max_repair_row_error": max(repair_errors) if repair_errors else 0.0,
                "per_head_error_vs_dense": per_head,
            }
        )

    seed_only_whole_group_ms = next((row["timing_ms"] for row in rows if row["repair_count"] == 0), None)
    for row in rows:
        exact_repair_ms = row.get("compact_repair_best_ms")
        if seed_only_whole_group_ms is not None and exact_repair_ms is not None:
            serial_ms = seed_only_whole_group_ms + float(exact_repair_ms)
            parallel_oracle_ms = max(seed_only_whole_group_ms, float(exact_repair_ms))
            row["external_repair_serial_ms"] = serial_ms
            row["external_repair_parallel_oracle_ms"] = parallel_oracle_ms
            row["external_repair_serial_speedup_vs_flashinfer"] = (
                flashinfer_ms / serial_ms if flashinfer_ms and serial_ms else None
            )
            row["external_repair_parallel_oracle_speedup_vs_flashinfer"] = (
                flashinfer_ms / parallel_oracle_ms if flashinfer_ms and parallel_oracle_ms else None
            )
            row["external_repair_parallel_oracle_beats_flashinfer"] = bool(
                flashinfer_ms is not None and parallel_oracle_ms < flashinfer_ms
            )
        else:
            row["external_repair_serial_ms"] = None
            row["external_repair_parallel_oracle_ms"] = None
            row["external_repair_serial_speedup_vs_flashinfer"] = None
            row["external_repair_parallel_oracle_speedup_vs_flashinfer"] = None
            row["external_repair_parallel_oracle_beats_flashinfer"] = False

    break_even = [
        row["repair_count"]
        for row in rows
        if row.get("beats_flashinfer")
    ]
    external_break_even = [
        row["repair_count"]
        for row in rows
        if row.get("external_repair_parallel_oracle_beats_flashinfer")
    ]
    payload = {
        "schema": "streamattn.tk_kv_group_repair.v1",
        "shape": {
            "batch": 1,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "group_size": args.q_heads // args.kv_heads,
            "head_dim": args.head_dim,
            "kv_len": args.kv_len,
            "dtype": args.dtype,
            "num_chunks": args.num_chunks,
            "block_size": args.block_size,
            "sink_blocks": args.sink_blocks,
            "recent_blocks": args.recent_blocks,
            "middle_seed_blocks": args.middle_seed_blocks,
            "block_order": args.block_order,
            "repair_order": repair_order,
            "repair_num_chunks": args.repair_num_chunks,
            "repair_block_d": args.repair_block_d,
        },
        "compile": {
            "compile_s": compile_s,
            "scalar_repair_compile_s": scalar_compile_s,
            "tk_root": str(tk_root),
            "cuda_arch": args.cuda_arch,
        },
        "baselines": {
            "torch_dense_true_gqa_ms": dense_ms,
            "tk_exact_chunk_merged_ms": exact_ms,
            "flashinfer_exact_ms": flashinfer_ms,
            "flashinfer_exact_tc_ms": flashinfer_tc_ms,
            "flashinfer_exact_no_tc_ms": flashinfer_no_tc_ms,
            "flashinfer_error": flashinfer_error,
            "tk_exact_quality_vs_dense": exact_quality,
        },
        "summary": {
            "repair_counts_tested": repair_counts,
            "flashinfer_break_even_repair_count": max(break_even) if break_even else None,
            "external_repair_parallel_oracle_break_even_repair_count": max(external_break_even)
            if external_break_even
            else None,
            "rows_beating_flashinfer": len(break_even),
            "external_repair_parallel_oracle_rows_beating_flashinfer": len(external_break_even),
        },
        "rows": rows,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-len", type=int, default=32768)
    parser.add_argument("--q-heads", type=int, default=7)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--num-chunks", type=int, default=256)
    parser.add_argument("--repair-threads", type=int, default=128)
    parser.add_argument("--repair-num-chunks", type=int, default=256)
    parser.add_argument("--repair-block-d", type=int, default=32)
    parser.add_argument("--repair-counts", default="0,1,2,3,4,7")
    parser.add_argument("--repair-order", default="")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--recent-blocks", type=int, default=2)
    parser.add_argument("--middle-seed-blocks", type=int, default=2)
    parser.add_argument("--block-order", default="recent_first", choices=["sequential", "recent_first"])
    parser.add_argument("--tk-root", default="")
    parser.add_argument("--checkout-dir", default="")
    parser.add_argument("--cuda-arch", default="sm_90a")
    parser.add_argument("--torch-cuda-arch-list", default="9.0a")
    parser.add_argument("--compile-verbose", action="store_true")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    payload = profile(args)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
