"""Audit component floors for the KV-group hybrid Gate-0 backend.

This benchmark is intentionally narrower than the hybrid backend realism runner.
It answers whether the underlying pieces are fast enough before any fused
runtime work:

* FlashInfer all-head exact floor;
* TK whole-KV-group seed-only floor;
* TK seed empty/no-active-chunk floor;
* exact repair rows for the seeded KV group;
* exact full rows for fallback KV groups;
* component oracle and serial lower bounds.

Every component is timed as a repeated distribution so timing outliers and
backend path changes are visible instead of hidden by one mean.
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

from benchmarks.profile_gate0_kv_group_repair_real import (  # noqa: E402
    parse_fixed_repair_policy,
    q_heads_for_kv_group,
)
from benchmarks.profile_gate0_seed_only_true_gqa import (  # noqa: E402
    HAS_FLASHINFER,
    _dense_selected_heads,
    _flashinfer_single_decode,
    _per_head_error,
    _select_kv_head,
    _select_q_heads,
)
from benchmarks.profile_gate0_true_gqa import _dense_true_gqa, _true_gqa_kv  # noqa: E402
from benchmarks.profile_stream_attn_gate0_wrapper import (  # noqa: E402
    _dtype,
    _error,
    _load_tensor,
    _sync,
    _time_cuda,
)


DEFAULT_REPAIR_POLICY = "0:0,1,5,6;1:7,8,9,10,11,12,13"


def _parse_ints(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def _distribution(samples: Sequence[float]) -> Dict[str, Any]:
    ordered = sorted(float(value) for value in samples)
    return {
        "samples_ms": list(samples),
        "min_ms": ordered[0] if ordered else None,
        "p10_ms": _quantile(ordered, 0.10) if ordered else None,
        "median_ms": _quantile(ordered, 0.50) if ordered else None,
        "p90_ms": _quantile(ordered, 0.90) if ordered else None,
        "max_ms": ordered[-1] if ordered else None,
        "mean_ms": sum(ordered) / len(ordered) if ordered else None,
        "repeat_count": len(ordered),
    }


def _time_distribution(
    fn: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    iters: int,
    repeats: int,
) -> Dict[str, Any]:
    samples = [
        _time_cuda(fn, device=device, warmup=warmup, iters=iters)
        for _ in range(max(1, repeats))
    ]
    return _distribution(samples)


def _tensor_info(tensor: torch.Tensor) -> Dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        "is_contiguous": bool(tensor.is_contiguous()),
        "storage_offset": int(tensor.storage_offset()),
    }


def _best_timing(rows: Dict[str, Dict[str, Any]]) -> tuple[str | None, float | None]:
    available = {
        name: row.get("median_ms")
        for name, row in rows.items()
        if row.get("median_ms") is not None
    }
    if not available:
        return None, None
    key = min(available, key=lambda name: float(available[name]))
    return key, float(available[key])


def _time_flashinfer_variants(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    device: torch.device,
    warmup: int,
    iters: int,
    repeats: int,
) -> Dict[str, Any]:
    rows: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, str] = {}
    if not HAS_FLASHINFER:
        return {"rows": rows, "errors": {"flashinfer": "FlashInfer is not available"}, "best_backend": None, "best_median_ms": None}
    for backend, use_tc in (("flashinfer_tc", True), ("flashinfer_no_tc", False)):
        try:
            rows[backend] = _time_distribution(
                lambda use_tc=use_tc: _flashinfer_single_decode(q, k, v, use_tensor_cores=use_tc),
                device=device,
                warmup=warmup,
                iters=iters,
                repeats=repeats,
            )
        except Exception as exc:  # pragma: no cover - backend/shape dependent
            errors[backend] = f"{type(exc).__name__}: {exc}"
    best_backend, best_median = _best_timing(rows)
    return {"rows": rows, "errors": errors, "best_backend": best_backend, "best_median_ms": best_median}


def _compile_tk(args: argparse.Namespace) -> tuple[Any, Dict[str, Any]]:
    from benchmarks.profile_tk_tensor_core_exact_decode import (  # noqa: WPS433
        _compile_extension,
        _find_or_clone_tk,
    )

    tk_root = _find_or_clone_tk(args)
    start = time.perf_counter()
    ext = _compile_extension(
        tk_root=tk_root,
        cuda_arch=args.cuda_arch,
        torch_cuda_arch_list=args.torch_cuda_arch_list,
        verbose=args.compile_verbose,
    )
    return ext, {
        "compile_s": time.perf_counter() - start,
        "tk_root": str(tk_root),
        "cuda_arch": args.cuda_arch,
    }


def _make_tk_seed_floor_fns(
    *,
    ext: Any,
    q_group: torch.Tensor,
    k_group: torch.Tensor,
    v_group: torch.Tensor,
    block_size: int,
    sink_blocks: int,
    recent_blocks: int,
    middle_seed_blocks: int,
    block_order: str,
    num_chunks: int,
    device: torch.device,
) -> tuple[Dict[str, Callable[[], torch.Tensor]], Dict[str, Any]]:
    from benchmarks.profile_tk_tensor_core_exact_decode import (  # noqa: WPS433
        _pack_active_chunks_by_kv_group,
        _pack_kv_head_major,
        _pack_q_by_kv_group,
        _pack_row_modes_by_kv_group,
        _unpack_q_by_kv_group,
    )

    local_q_heads = int(q_group.shape[2])
    q_bf16 = q_group[:, 0].contiguous().to(torch.bfloat16)
    k_bf16 = k_group.contiguous().to(torch.bfloat16)
    v_bf16 = v_group.contiguous().to(torch.bfloat16)
    q_packed = _pack_q_by_kv_group(q_bf16, 1, padded_rows=16)
    k_packed = _pack_kv_head_major(k_bf16)
    v_packed = _pack_kv_head_major(v_bf16)
    row_modes = _pack_row_modes_by_kv_group(
        q_heads=local_q_heads,
        kv_heads=1,
        seed_heads=list(range(local_q_heads)),
        padded_rows=16,
        device=device,
    )
    active_chunks, active_counts, flat_active_chunks, active_offsets, active_by_kv = (
        _pack_active_chunks_by_kv_group(
            q_heads=local_q_heads,
            kv_heads=1,
            seed_heads=list(range(local_q_heads)),
            kv_len=int(k_group.shape[1]),
            num_chunks=num_chunks,
            block_size=block_size,
            sink_blocks=sink_blocks,
            recent_blocks=recent_blocks,
            middle_seed_blocks=middle_seed_blocks,
            block_order=block_order,
            device=device,
        )
    )
    empty_chunks = torch.zeros((1, 1), device=device, dtype=torch.int32)
    empty_counts = torch.zeros((1,), device=device, dtype=torch.int32)
    empty_flat = torch.zeros((1,), device=device, dtype=torch.int32)
    empty_offsets = torch.zeros((2,), device=device, dtype=torch.int32)
    block_order_id = 0 if block_order == "sequential" else 1

    def raw_seed() -> torch.Tensor:
        return ext.head_mode_compact_chunk_merged(
            q_packed,
            k_packed,
            v_packed,
            row_modes,
            active_chunks,
            active_counts,
            flat_active_chunks,
            active_offsets,
            num_chunks,
            block_size,
            sink_blocks,
            recent_blocks,
            middle_seed_blocks,
            block_order_id,
        )

    def unpacked_seed() -> torch.Tensor:
        return _unpack_q_by_kv_group(raw_seed(), local_q_heads)[:, None]

    def raw_empty() -> torch.Tensor:
        return ext.head_mode_compact_chunk_merged(
            q_packed,
            k_packed,
            v_packed,
            row_modes,
            empty_chunks,
            empty_counts,
            empty_flat,
            empty_offsets,
            num_chunks,
            block_size,
            sink_blocks,
            recent_blocks,
            middle_seed_blocks,
            block_order_id,
        )

    return {
        "tk_seed_raw": raw_seed,
        "tk_seed_unpacked": unpacked_seed,
        "tk_seed_empty_no_active_chunks_raw": raw_empty,
    }, {
        "active_chunks": list(active_by_kv[0]),
        "active_chunk_count": len(active_by_kv[0]),
        "num_chunks": num_chunks,
        "packed_tensors": {
            "q_packed": _tensor_info(q_packed),
            "k_packed": _tensor_info(k_packed),
            "v_packed": _tensor_info(v_packed),
        },
    }


def profile(args: argparse.Namespace) -> Dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    dtype = _dtype(args.dtype)
    q = _load_tensor(args.q_path, key="q", device=device, dtype=dtype)
    k_expanded = _load_tensor(args.k_path, key="k", device=device, dtype=dtype)
    v_expanded = _load_tensor(args.v_path, key="v", device=device, dtype=dtype)
    k_true = _true_gqa_kv(k_expanded, true_kv_heads=args.true_kv_heads)
    v_true = _true_gqa_kv(v_expanded, true_kv_heads=args.true_kv_heads)
    if q.shape[0] != 1 or q.shape[1] != 1:
        raise ValueError("component floor profiler currently supports B=1, M=1")
    q_heads = int(q.shape[2])
    kv_heads = int(k_true.shape[2])
    if q_heads % kv_heads != 0:
        raise ValueError("q_heads must be divisible by true_kv_heads")
    seed_kv_groups = _parse_ints(args.seed_kv_groups)
    if len(seed_kv_groups) != 1:
        raise ValueError("component floor profiler currently expects one seeded KV group")
    seed_kv_head = seed_kv_groups[0]
    repair_policy = parse_fixed_repair_policy(args.repair_policy)

    print("[component-floors] materializing dense reference", flush=True)
    dense_ref = _dense_true_gqa(q, k_true, v_true)

    print("[component-floors] compiling TK", flush=True)
    tk_ext, tk_compile = _compile_tk(args)

    group_heads = q_heads_for_kv_group(seed_kv_head, q_heads=q_heads, kv_heads=kv_heads)
    q_group0 = _select_q_heads(q, group_heads)
    k_group0 = _select_kv_head(k_true, seed_kv_head)
    v_group0 = _select_kv_head(v_true, seed_kv_head)
    seed_fns, seed_info = _make_tk_seed_floor_fns(
        ext=tk_ext,
        q_group=q_group0,
        k_group=k_group0,
        v_group=v_group0,
        block_size=args.block_size,
        sink_blocks=args.sink_blocks,
        recent_blocks=args.recent_blocks,
        middle_seed_blocks=args.middle_seed_blocks,
        block_order=args.block_order,
        num_chunks=args.tk_num_chunks,
        device=device,
    )

    component_rows: Dict[str, Dict[str, Any]] = {}
    quality_rows: Dict[str, Any] = {}

    print("[component-floors] timing FlashInfer all-head exact", flush=True)
    flash_all = _time_flashinfer_variants(
        q,
        k_true,
        v_true,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
        repeats=args.repeats,
    )
    component_rows["flashinfer_all_exact"] = flash_all
    if flash_all["best_backend"]:
        out = _flashinfer_single_decode(
            q,
            k_true,
            v_true,
            use_tensor_cores=flash_all["best_backend"] == "flashinfer_tc",
        )
        quality_rows["flashinfer_all_vs_dense"] = _error(out, dense_ref)

    print("[component-floors] timing TK seed floors", flush=True)
    seed_component_rows: Dict[str, Any] = {}
    for name, fn in seed_fns.items():
        try:
            seed_component_rows[name] = _time_distribution(
                fn,
                device=device,
                warmup=args.warmup,
                iters=args.iters,
                repeats=args.repeats,
            )
        except Exception as exc:  # pragma: no cover - optional spike path
            seed_component_rows[name] = {"error": f"{type(exc).__name__}: {exc}"}
    component_rows["tk_seed_group0"] = {"rows": seed_component_rows, "info": seed_info}
    seed_out = seed_fns["tk_seed_unpacked"]().to(dtype)
    dense_group0 = dense_ref.index_select(
        2,
        torch.tensor(group_heads, device=device, dtype=torch.long),
    )
    quality_rows["tk_seed_group0_vs_dense"] = _error(seed_out, dense_group0)
    quality_rows["tk_seed_group0_vs_dense_per_head"] = _per_head_error(seed_out, dense_group0)["per_head"]

    exact_component_medians: List[float] = []
    exact_rows = []
    repair_heads = sorted(set(repair_policy.get(seed_kv_head, [])))
    if repair_heads:
        print(f"[component-floors] timing repair heads {repair_heads}", flush=True)
        q_repair = _select_q_heads(q, repair_heads)
        repair = _time_flashinfer_variants(
            q_repair,
            k_group0,
            v_group0,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        exact_rows.append({"scope": "repair_seed_kv_group", "kv_head": seed_kv_head, "q_heads": repair_heads, **repair})
        if repair["best_median_ms"] is not None:
            exact_component_medians.append(float(repair["best_median_ms"]))
        if repair["best_backend"]:
            repair_out = _flashinfer_single_decode(
                q_repair,
                k_group0,
                v_group0,
                use_tensor_cores=repair["best_backend"] == "flashinfer_tc",
            )
            dense_repair = dense_ref.index_select(
                2,
                torch.tensor(repair_heads, device=device, dtype=torch.long),
            )
            quality_rows["repair_seed_kv_group_vs_dense"] = _error(repair_out, dense_repair)

    for kv_head in range(kv_heads):
        if kv_head == seed_kv_head:
            continue
        exact_heads = sorted(set(repair_policy.get(kv_head, [])))
        if not exact_heads:
            exact_heads = q_heads_for_kv_group(kv_head, q_heads=q_heads, kv_heads=kv_heads)
        print(f"[component-floors] timing exact kv_group={kv_head} heads={exact_heads}", flush=True)
        q_exact = _select_q_heads(q, exact_heads)
        k_exact = _select_kv_head(k_true, kv_head)
        v_exact = _select_kv_head(v_true, kv_head)
        exact = _time_flashinfer_variants(
            q_exact,
            k_exact,
            v_exact,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        exact_rows.append({"scope": "exact_fallback_kv_group", "kv_head": kv_head, "q_heads": exact_heads, **exact})
        if exact["best_median_ms"] is not None:
            exact_component_medians.append(float(exact["best_median_ms"]))
        if exact["best_backend"]:
            exact_out = _flashinfer_single_decode(
                q_exact,
                k_exact,
                v_exact,
                use_tensor_cores=exact["best_backend"] == "flashinfer_tc",
            )
            dense_exact = dense_ref.index_select(
                2,
                torch.tensor(exact_heads, device=device, dtype=torch.long),
            )
            quality_rows[f"exact_kv_group_{kv_head}_vs_dense"] = _error(exact_out, dense_exact)
    component_rows["exact_selected_components"] = exact_rows

    exact_heads_all = sorted(set(repair_heads + [head for row in exact_rows for head in row.get("q_heads", [])]))
    exact_selected_one_component = None
    try:
        print("[component-floors] timing torch exact selected as one component", flush=True)
        exact_selected_one_component = _time_distribution(
            lambda: _dense_selected_heads(q, k_true, v_true, exact_heads_all),
            device=device,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
    except Exception as exc:  # pragma: no cover - diagnostic only
        exact_selected_one_component = {"error": f"{type(exc).__name__}: {exc}"}
    component_rows["torch_exact_selected_one_component"] = exact_selected_one_component

    seed_median = seed_component_rows.get("tk_seed_raw", {}).get("median_ms")
    exact_oracle = max(exact_component_medians) if exact_component_medians else None
    component_oracle = (
        max(float(seed_median), float(exact_oracle))
        if seed_median is not None and exact_oracle is not None
        else None
    )
    component_serial = (
        float(seed_median) + sum(exact_component_medians)
        if seed_median is not None and exact_component_medians
        else None
    )
    flash_all_best = flash_all.get("best_median_ms")
    gate = {
        "component_oracle_ms": component_oracle,
        "component_serial_sum_ms": component_serial,
        "flashinfer_all_best_median_ms": flash_all_best,
        "component_oracle_speedup_vs_flashinfer_all": (
            flash_all_best / component_oracle
            if flash_all_best is not None and component_oracle
            else None
        ),
        "component_oracle_margin_vs_flashinfer_all_ms": (
            flash_all_best - component_oracle
            if flash_all_best is not None and component_oracle is not None
            else None
        ),
        "component_oracle_positive": bool(
            flash_all_best is not None
            and component_oracle is not None
            and component_oracle < flash_all_best
        ),
        "component_oracle_strong_positive": bool(
            flash_all_best is not None
            and component_oracle is not None
            and component_oracle <= flash_all_best - args.strong_margin_ms
        ),
        "strong_margin_ms": args.strong_margin_ms,
    }

    return {
        "schema": "streamattn.gate0.hybrid_component_floors.v1",
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "flashinfer_available": HAS_FLASHINFER,
        "shape": {
            "batch": int(q.shape[0]),
            "query_len": int(q.shape[1]),
            "kv_len": int(k_true.shape[1]),
            "q_heads": q_heads,
            "true_kv_heads": kv_heads,
            "group_size": q_heads // kv_heads,
            "dim": int(q.shape[3]),
            "dtype": args.dtype,
        },
        "tensor_layout": {
            "q": _tensor_info(q),
            "k_expanded": _tensor_info(k_expanded),
            "v_expanded": _tensor_info(v_expanded),
            "k_true": _tensor_info(k_true),
            "v_true": _tensor_info(v_true),
        },
        "policy": {
            "seed_kv_groups": seed_kv_groups,
            "repair_policy": repair_policy,
            "trusted_seed_heads": [head for head in group_heads if head not in set(repair_heads)],
            "repair_heads": repair_heads,
            "block_size": args.block_size,
            "sink_blocks": args.sink_blocks,
            "recent_blocks": args.recent_blocks,
            "middle_seed_blocks": args.middle_seed_blocks,
            "block_order": args.block_order,
        },
        "benchmark": {
            "warmup": args.warmup,
            "iters": args.iters,
            "repeats": args.repeats,
        },
        "components": component_rows,
        "quality": quality_rows,
        "gate": gate,
        "tk_compile": tk_compile,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-path", required=True)
    parser.add_argument("--k-path", required=True)
    parser.add_argument("--v-path", required=True)
    parser.add_argument("--true-kv-heads", type=int, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--seed-kv-groups", default="0")
    parser.add_argument("--repair-policy", default=DEFAULT_REPAIR_POLICY)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--recent-blocks", type=int, default=2)
    parser.add_argument("--middle-seed-blocks", type=int, default=8)
    parser.add_argument("--block-order", choices=["sequential", "recent_first"], default="recent_first")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--tk-num-chunks", type=int, default=256)
    parser.add_argument("--strong-margin-ms", type=float, default=0.005)
    parser.add_argument("--tk-root", default="")
    parser.add_argument("--checkout-dir", default="")
    parser.add_argument("--cuda-arch", default="sm_90a")
    parser.add_argument("--torch-cuda-arch-list", default="9.0a")
    parser.add_argument("--compile-verbose", action="store_true")
    parser.add_argument("--summary-json-out", default="")
    args = parser.parse_args()

    result = profile(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.summary_json_out:
        path = Path(args.summary_json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
