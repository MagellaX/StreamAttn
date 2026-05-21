"""Profile real-Qwen KV-group seed-only speculation plus exact repair.

The synthetic TK repair probes showed that whole-KV-group seed-only scheduling
is fast, but same-tile row repair is the wrong shape.  This profiler moves the
question back onto captured model tensors:

* speculate an entire true-GQA KV group as seed-only;
* identify Q rows that need exact repair for each error budget;
* time compact selected-row exact repair lower bounds on the real tensors.

It is an evidence tool.  It does not compose a production runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_gate0_seed_only_true_gqa import (  # noqa: E402
    HAS_FLASHINFER,
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
from stream_attention.kernels.gate0_seed_only_triton import (  # noqa: E402
    gate0_seed_only_attention_triton_forward,
)


DEFAULT_BUDGETS = "strict:1e-2,moderate:1.5e-2,research:5e-2"


def parse_budgets(raw: str) -> List[Dict[str, Any]]:
    budgets: List[Dict[str, Any]] = []
    for item in raw.split(","):
        if not item.strip():
            continue
        name, value = item.split(":", 1)
        budgets.append({"name": name.strip(), "max_abs_error": float(value)})
    return budgets


def parse_ints(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_fixed_repair_policy(raw: str) -> Dict[int, List[int]]:
    """Parse ``kv_head:head,head;kv_head:head`` repair policy syntax."""
    policy: Dict[int, List[int]] = {}
    if not raw.strip():
        return policy
    for item in raw.split(";"):
        if not item.strip():
            continue
        kv_raw, heads_raw = item.split(":", 1)
        policy[int(kv_raw.strip())] = parse_ints(heads_raw)
    return policy


def q_heads_for_kv_group(kv_head: int, *, q_heads: int, kv_heads: int) -> List[int]:
    if q_heads % kv_heads != 0:
        raise ValueError("q_heads must be divisible by kv_heads")
    group_size = q_heads // kv_heads
    start = kv_head * group_size
    return list(range(start, start + group_size))


def select_repair_heads(
    per_head_rows: Sequence[Dict[str, Any]],
    *,
    budget: float,
) -> List[int]:
    return [
        int(row["head"])
        for row in per_head_rows
        if float(row.get("max_abs_error") or 0.0) > budget
    ]


def corrected_max_error(
    per_head_rows: Sequence[Dict[str, Any]],
    *,
    repair_heads: Sequence[int],
) -> float:
    repair_set = set(int(head) for head in repair_heads)
    return max(
        (
            float(row.get("max_abs_error") or 0.0)
            for row in per_head_rows
            if int(row["head"]) not in repair_set
        ),
        default=0.0,
    )


def repair_work_summary(
    *,
    group_size: int,
    repair_count: int,
    kv_len: int,
    block_size: int,
    sink_blocks: int,
    recent_blocks: int,
    middle_seed_blocks: int,
) -> Dict[str, Any]:
    num_blocks = (kv_len + block_size - 1) // block_size
    seed_blocks = min(num_blocks, sink_blocks + recent_blocks + middle_seed_blocks)
    seed_fraction = seed_blocks / num_blocks if num_blocks else 0.0
    dense_row_blocks = group_size * num_blocks
    external_row_blocks = group_size * seed_blocks + repair_count * num_blocks
    same_kernel_row_blocks = repair_count * num_blocks + (group_size - repair_count) * seed_blocks
    return {
        "num_blocks": num_blocks,
        "seed_blocks": seed_blocks,
        "seed_fraction": seed_fraction,
        "repair_count": repair_count,
        "external_seed_plus_repair_row_block_fraction": (
            external_row_blocks / dense_row_blocks if dense_row_blocks else 0.0
        ),
        "same_kernel_row_mode_row_block_fraction": (
            same_kernel_row_blocks / dense_row_blocks if dense_row_blocks else 0.0
        ),
        "whole_group_seed_only_can_skip_nonseed_kv": True,
        "same_kernel_repair_forces_nonseed_kv_when_repair_count_positive": repair_count > 0,
    }


def _time_flashinfer(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    use_tensor_cores: bool,
    device: torch.device,
    warmup: int,
    iters: int,
) -> tuple[float | None, str | None]:
    if not HAS_FLASHINFER:
        return None, "FlashInfer is not available"
    try:
        return (
            _time_cuda(
                lambda: _flashinfer_single_decode(
                    q,
                    k,
                    v,
                    use_tensor_cores=use_tensor_cores,
                ),
                device=device,
                warmup=warmup,
                iters=iters,
            ),
            None,
        )
    except Exception as exc:  # pragma: no cover - backend/shape dependent
        return None, f"{type(exc).__name__}: {exc}"


def _best_timing(candidates: Dict[str, float | None]) -> tuple[str | None, float | None]:
    available = {name: value for name, value in candidates.items() if value is not None}
    if not available:
        return None, None
    name = min(available, key=lambda key: float(available[key]))
    return name, float(available[name])


def _measure_repair_set(
    *,
    q: torch.Tensor,
    k_group: torch.Tensor,
    v_group: torch.Tensor,
    dense_group_ref: torch.Tensor,
    global_heads: Sequence[int],
    group_heads: Sequence[int],
    device: torch.device,
    warmup: int,
    iters: int,
    measure_triton_repair: bool,
    repair_num_chunks: int,
    repair_block_d: int,
) -> Dict[str, Any]:
    repair_heads = list(global_heads)
    if not repair_heads:
        return {
            "repair_heads": [],
            "repair_count": 0,
            "flashinfer_tc_ms": 0.0,
            "flashinfer_no_tc_ms": 0.0,
            "flashinfer_tc_error": None,
            "flashinfer_no_tc_error": None,
            "triton_splitk_ms": 0.0,
            "triton_splitk_error": None,
            "best_backend": "none",
            "best_ms": 0.0,
            "quality": {"flashinfer_best_vs_dense": None, "triton_splitk_vs_dense": None},
        }

    q_repair = _select_q_heads(q, repair_heads)
    dense_local_positions = [group_heads.index(head) for head in repair_heads]
    dense_repair = dense_group_ref.index_select(
        2,
        torch.tensor(dense_local_positions, device=device, dtype=torch.long),
    ).contiguous()

    tc_ms, tc_error = _time_flashinfer(
        q_repair,
        k_group,
        v_group,
        use_tensor_cores=True,
        device=device,
        warmup=warmup,
        iters=iters,
    )
    no_tc_ms, no_tc_error = _time_flashinfer(
        q_repair,
        k_group,
        v_group,
        use_tensor_cores=False,
        device=device,
        warmup=warmup,
        iters=iters,
    )
    flashinfer_best_backend, flashinfer_best_ms = _best_timing(
        {"flashinfer_tc": tc_ms, "flashinfer_no_tc": no_tc_ms}
    )
    flashinfer_quality = None
    if flashinfer_best_backend is not None:
        out = _flashinfer_single_decode(
            q_repair,
            k_group,
            v_group,
            use_tensor_cores=flashinfer_best_backend == "flashinfer_tc",
        )
        flashinfer_quality = _error(out, dense_repair)

    triton_ms = None
    triton_error = None
    triton_quality = None
    if measure_triton_repair:
        try:
            from stream_attention.kernels.gate0_compact_repair_triton import (
                compact_repair_splitk_triton_forward,
            )

            def triton_repair() -> torch.Tensor:
                return compact_repair_splitk_triton_forward(
                    q_repair[:, 0].contiguous(),
                    k_group,
                    v_group,
                    num_chunks=repair_num_chunks,
                    block_d=repair_block_d,
                )

            triton_ms = _time_cuda(
                triton_repair,
                device=device,
                warmup=warmup,
                iters=iters,
            )
            triton_quality = _error(triton_repair()[:, None], dense_repair)
        except Exception as exc:  # pragma: no cover - optional Triton path
            triton_error = f"{type(exc).__name__}: {exc}"

    best_backend, best_ms = _best_timing(
        {
            "flashinfer_tc": tc_ms,
            "flashinfer_no_tc": no_tc_ms,
            "triton_splitk": triton_ms,
        }
    )
    return {
        "repair_heads": repair_heads,
        "repair_count": len(repair_heads),
        "flashinfer_tc_ms": tc_ms,
        "flashinfer_no_tc_ms": no_tc_ms,
        "flashinfer_tc_error": tc_error,
        "flashinfer_no_tc_error": no_tc_error,
        "triton_splitk_ms": triton_ms,
        "triton_splitk_error": triton_error,
        "best_backend": best_backend,
        "best_ms": best_ms,
        "quality": {
            "flashinfer_best_vs_dense": flashinfer_quality,
            "triton_splitk_vs_dense": triton_quality,
        },
    }


def _repair_order_from_errors(per_head_rows: Sequence[Dict[str, Any]]) -> List[int]:
    return [
        int(row["head"])
        for row in sorted(
            per_head_rows,
            key=lambda item: float(item.get("max_abs_error") or 0.0),
            reverse=True,
        )
    ]


def _measure_tk_bf16_group(
    *,
    ext: Any,
    q_group: torch.Tensor,
    k_group: torch.Tensor,
    v_group: torch.Tensor,
    dense_group_ref: torch.Tensor,
    block_size: int,
    sink_blocks: int,
    recent_blocks: int,
    middle_seed_blocks: int,
    block_order: str,
    num_chunks: int,
    device: torch.device,
    warmup: int,
    iters: int,
) -> Dict[str, Any]:
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
    seed_heads = list(range(local_q_heads))
    row_modes = _pack_row_modes_by_kv_group(
        q_heads=local_q_heads,
        kv_heads=1,
        seed_heads=seed_heads,
        padded_rows=16,
        device=device,
    )
    active_chunks, active_counts, flat_active_chunks, active_offsets, active_by_kv = _pack_active_chunks_by_kv_group(
        q_heads=local_q_heads,
        kv_heads=1,
        seed_heads=seed_heads,
        kv_len=int(k_group.shape[1]),
        num_chunks=num_chunks,
        block_size=block_size,
        sink_blocks=sink_blocks,
        recent_blocks=recent_blocks,
        middle_seed_blocks=middle_seed_blocks,
        block_order=block_order,
        device=device,
    )
    block_order_id = 0 if block_order == "sequential" else 1

    def tk_seed() -> torch.Tensor:
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

    def tk_exact() -> torch.Tensor:
        return ext.exact_decode_chunk_merged(q_packed, k_packed, v_packed, num_chunks)

    seed_ms = _time_cuda(tk_seed, device=device, warmup=warmup, iters=iters)
    exact_ms = _time_cuda(tk_exact, device=device, warmup=warmup, iters=iters)
    seed_out = _unpack_q_by_kv_group(tk_seed(), local_q_heads)[:, None]
    exact_out = _unpack_q_by_kv_group(tk_exact(), local_q_heads)[:, None]
    return {
        "seed_only_whole_group_ms": seed_ms,
        "exact_group_ms": exact_ms,
        "seed_only_vs_dense_group": _error(seed_out, dense_group_ref),
        "seed_only_vs_dense_group_per_head": _per_head_error(seed_out, dense_group_ref)["per_head"],
        "exact_vs_dense_group": _error(exact_out, dense_group_ref),
        "active_chunks": list(active_by_kv[0]),
        "active_chunk_count": len(active_by_kv[0]),
    }


def profile(args: argparse.Namespace) -> Dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    dtype = _dtype(args.dtype)
    budgets = parse_budgets(args.budgets)

    print("[kv-repair-real] loading captured tensors", flush=True)
    q = _load_tensor(args.q_path, key="q", device=device, dtype=dtype)
    k_expanded = _load_tensor(args.k_path, key="k", device=device, dtype=dtype)
    v_expanded = _load_tensor(args.v_path, key="v", device=device, dtype=dtype)
    k_true = _true_gqa_kv(k_expanded, true_kv_heads=args.true_kv_heads)
    v_true = _true_gqa_kv(v_expanded, true_kv_heads=args.true_kv_heads)
    if q.shape[0] != 1 or q.shape[1] != 1:
        raise ValueError("real KV-group repair profiler currently supports B=1, M=1")
    q_heads = int(q.shape[2])
    kv_heads = int(k_true.shape[2])
    if q_heads % kv_heads != 0:
        raise ValueError("q heads must be divisible by true KV heads")

    if args.kv_groups:
        kv_groups = parse_ints(args.kv_groups)
    else:
        kv_groups = list(range(kv_heads))
    for kv_head in kv_groups:
        if kv_head < 0 or kv_head >= kv_heads:
            raise ValueError(f"kv_group {kv_head} outside [0, {kv_heads})")

    repair_counts = parse_ints(args.repair_counts)
    fixed_repair_policy = parse_fixed_repair_policy(args.fixed_repair_policy)
    tk_ext = None
    tk_compile: Dict[str, Any] | None = None
    if args.measure_tk_bf16:
        from benchmarks.profile_tk_tensor_core_exact_decode import (  # noqa: WPS433
            _compile_extension,
            _find_or_clone_tk,
        )

        print("[kv-repair-real] compiling TK BF16 extension", flush=True)
        import time

        tk_root = _find_or_clone_tk(args)
        compile_start = time.perf_counter()
        tk_ext = _compile_extension(
            tk_root=tk_root,
            cuda_arch=args.cuda_arch,
            torch_cuda_arch_list=args.torch_cuda_arch_list,
            verbose=args.compile_verbose,
        )
        tk_compile = {
            "compile_s": time.perf_counter() - compile_start,
            "tk_root": str(tk_root),
            "cuda_arch": args.cuda_arch,
        }
        print(f"[kv-repair-real] TK compile finished in {tk_compile['compile_s']:.2f}s", flush=True)
    print(
        "[kv-repair-real] profiling "
        f"q_heads={q_heads} kv_heads={kv_heads} D={q.shape[3]} kv_len={k_true.shape[1]}",
        flush=True,
    )

    def dense_all() -> torch.Tensor:
        return _dense_true_gqa(q, k_true, v_true)

    dense_all_ms = _time_cuda(dense_all, device=device, warmup=args.warmup, iters=args.iters)
    flashinfer_all_tc_ms, flashinfer_all_tc_error = _time_flashinfer(
        q,
        k_true,
        v_true,
        use_tensor_cores=True,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    flashinfer_all_no_tc_ms, flashinfer_all_no_tc_error = _time_flashinfer(
        q,
        k_true,
        v_true,
        use_tensor_cores=False,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    _, flashinfer_all_best_ms = _best_timing(
        {"flashinfer_tc": flashinfer_all_tc_ms, "flashinfer_no_tc": flashinfer_all_no_tc_ms}
    )
    reference_all_ms = flashinfer_all_best_ms if flashinfer_all_best_ms is not None else dense_all_ms

    group_rows: List[Dict[str, Any]] = []
    for kv_head in kv_groups:
        group_heads = q_heads_for_kv_group(kv_head, q_heads=q_heads, kv_heads=kv_heads)
        q_group = _select_q_heads(q, group_heads)
        k_group = _select_kv_head(k_true, kv_head)
        v_group = _select_kv_head(v_true, kv_head)
        print(f"[kv-repair-real] kv_group={kv_head} heads={group_heads}", flush=True)

        def dense_group() -> torch.Tensor:
            return _dense_true_gqa(q_group, k_group, v_group)

        def seed_only_group() -> torch.Tensor:
            return gate0_seed_only_attention_triton_forward(
                q_group,
                k_group,
                v_group,
                block_size=args.block_size,
                sink_blocks=args.sink_blocks,
                recent_blocks=args.recent_blocks,
                middle_seed_blocks=args.middle_seed_blocks,
                block_order=args.block_order,
                num_warps=args.num_warps,
                num_stages=args.num_stages,
            )[0]

        dense_group_ref = dense_group()
        seed_out = seed_only_group()
        _sync(device)
        dense_group_ms = _time_cuda(dense_group, device=device, warmup=args.group_warmup, iters=args.group_iters)
        seed_group_ms = _time_cuda(
            seed_only_group,
            device=device,
            warmup=args.group_warmup,
            iters=args.group_iters,
        )
        group_tc_ms, group_tc_error = _time_flashinfer(
            q_group,
            k_group,
            v_group,
            use_tensor_cores=True,
            device=device,
            warmup=args.group_warmup,
            iters=args.group_iters,
        )
        group_no_tc_ms, group_no_tc_error = _time_flashinfer(
            q_group,
            k_group,
            v_group,
            use_tensor_cores=False,
            device=device,
            warmup=args.group_warmup,
            iters=args.group_iters,
        )
        group_reference_backend, group_reference_ms = _best_timing(
            {
                "flashinfer_tc": group_tc_ms,
                "flashinfer_no_tc": group_no_tc_ms,
                "torch_sdpa": dense_group_ms,
            }
        )

        local_errors = _per_head_error(seed_out, dense_group_ref)["per_head"]
        per_head_rows = []
        for local_row in local_errors:
            local_head = int(local_row["head"])
            copied = dict(local_row)
            copied["local_head"] = local_head
            copied["head"] = int(group_heads[local_head])
            copied["kv_head"] = kv_head
            per_head_rows.append(copied)
        tk_bf16_payload = (
            _measure_tk_bf16_group(
                ext=tk_ext,
                q_group=q_group,
                k_group=k_group,
                v_group=v_group,
                dense_group_ref=dense_group_ref,
                block_size=args.block_size,
                sink_blocks=args.sink_blocks,
                recent_blocks=args.recent_blocks,
                middle_seed_blocks=args.middle_seed_blocks,
                block_order=args.block_order,
                num_chunks=args.tk_num_chunks,
                device=device,
                warmup=args.group_warmup,
                iters=args.group_iters,
            )
            if tk_ext is not None
            else None
        )
        tk_seed_ms = (
            float(tk_bf16_payload["seed_only_whole_group_ms"]) if tk_bf16_payload is not None else None
        )
        tk_per_head_rows = None
        if tk_bf16_payload is not None:
            tk_per_head_rows = []
            for local_row in tk_bf16_payload["seed_only_vs_dense_group_per_head"]:
                local_head = int(local_row["head"])
                copied = dict(local_row)
                copied["local_head"] = local_head
                copied["head"] = int(group_heads[local_head])
                copied["kv_head"] = kv_head
                tk_per_head_rows.append(copied)
            tk_bf16_payload["seed_only_vs_dense_group_per_head"] = tk_per_head_rows

        repair_cache: Dict[tuple[int, ...], Dict[str, Any]] = {}

        def repair_measure(heads: Sequence[int]) -> Dict[str, Any]:
            key = tuple(sorted(int(head) for head in heads))
            if key not in repair_cache:
                repair_cache[key] = _measure_repair_set(
                    q=q,
                    k_group=k_group,
                    v_group=v_group,
                    dense_group_ref=dense_group_ref,
                    global_heads=key,
                    group_heads=group_heads,
                    device=device,
                    warmup=args.group_warmup,
                    iters=args.group_iters,
                    measure_triton_repair=args.measure_triton_repair,
                    repair_num_chunks=args.repair_num_chunks,
                    repair_block_d=args.repair_block_d,
                )
            return repair_cache[key]

        budget_rows = []
        for budget in budgets:
            repair_heads = select_repair_heads(per_head_rows, budget=float(budget["max_abs_error"]))
            repair = repair_measure(repair_heads)
            repair_ms = repair.get("best_ms")
            parallel_oracle_ms = max(seed_group_ms, float(repair_ms)) if repair_ms is not None else None
            serial_ms = seed_group_ms + float(repair_ms) if repair_ms is not None else None
            tk_repair_heads = (
                select_repair_heads(tk_per_head_rows, budget=float(budget["max_abs_error"]))
                if tk_per_head_rows is not None
                else repair_heads
            )
            tk_repair = repair_measure(tk_repair_heads)
            tk_repair_ms = tk_repair.get("best_ms")
            tk_parallel_oracle_ms = (
                max(tk_seed_ms, float(tk_repair_ms))
                if tk_seed_ms is not None and tk_repair_ms is not None
                else None
            )
            budget_rows.append(
                {
                    "budget": budget["name"],
                    "max_abs_error_budget": float(budget["max_abs_error"]),
                    "repair_heads": repair_heads,
                    "trusted_seed_heads": [head for head in group_heads if head not in set(repair_heads)],
                    "corrected_max_abs_error": corrected_max_error(per_head_rows, repair_heads=repair_heads),
                    "repair": repair,
                    "external_parallel_oracle_ms": parallel_oracle_ms,
                    "external_serial_ms": serial_ms,
                    "tk_bf16_repair_heads": tk_repair_heads,
                    "tk_bf16_trusted_seed_heads": [
                        head for head in group_heads if head not in set(tk_repair_heads)
                    ],
                    "tk_bf16_corrected_max_abs_error": (
                        corrected_max_error(tk_per_head_rows, repair_heads=tk_repair_heads)
                        if tk_per_head_rows is not None
                        else None
                    ),
                    "tk_bf16_repair": tk_repair,
                    "tk_bf16_external_parallel_oracle_ms": tk_parallel_oracle_ms,
                    "tk_bf16_external_parallel_oracle_speedup_vs_group_reference": (
                        group_reference_ms / tk_parallel_oracle_ms
                        if group_reference_ms and tk_parallel_oracle_ms
                        else None
                    ),
                    "tk_bf16_external_parallel_oracle_beats_group_reference": bool(
                        group_reference_ms is not None
                        and tk_parallel_oracle_ms is not None
                        and tk_parallel_oracle_ms < group_reference_ms
                    ),
                    "external_parallel_oracle_speedup_vs_group_reference": (
                        group_reference_ms / parallel_oracle_ms
                        if group_reference_ms and parallel_oracle_ms
                        else None
                    ),
                    "external_parallel_oracle_beats_group_reference": bool(
                        group_reference_ms is not None
                        and parallel_oracle_ms is not None
                        and parallel_oracle_ms < group_reference_ms
                    ),
                    "external_serial_speedup_vs_group_reference": (
                        group_reference_ms / serial_ms if group_reference_ms and serial_ms else None
                    ),
                    "work": repair_work_summary(
                        group_size=len(group_heads),
                        repair_count=len(repair_heads),
                        kv_len=int(k_true.shape[1]),
                        block_size=args.block_size,
                        sink_blocks=args.sink_blocks,
                        recent_blocks=args.recent_blocks,
                        middle_seed_blocks=args.middle_seed_blocks,
                    ),
                }
            )

        fixed_policy_row = None
        if kv_head in fixed_repair_policy:
            fixed_repair_heads = sorted(set(fixed_repair_policy[kv_head]))
            invalid_heads = [head for head in fixed_repair_heads if head not in set(group_heads)]
            if invalid_heads:
                raise ValueError(
                    f"fixed repair policy for kv_group {kv_head} contains heads outside group: {invalid_heads}"
                )
            fixed_repair = repair_measure(fixed_repair_heads)
            fixed_repair_ms = fixed_repair.get("best_ms")
            fixed_triton_parallel_ms = (
                max(seed_group_ms, float(fixed_repair_ms)) if fixed_repair_ms is not None else None
            )
            fixed_tk_parallel_ms = (
                max(tk_seed_ms, float(fixed_repair_ms))
                if tk_seed_ms is not None and fixed_repair_ms is not None
                else None
            )
            fixed_policy_row = {
                "repair_heads": fixed_repair_heads,
                "trusted_seed_heads": [head for head in group_heads if head not in set(fixed_repair_heads)],
                "corrected_max_abs_error": corrected_max_error(per_head_rows, repair_heads=fixed_repair_heads),
                "tk_bf16_corrected_max_abs_error": (
                    corrected_max_error(tk_per_head_rows, repair_heads=fixed_repair_heads)
                    if tk_per_head_rows is not None
                    else None
                ),
                "repair": fixed_repair,
                "triton_external_parallel_oracle_ms": fixed_triton_parallel_ms,
                "triton_external_parallel_oracle_speedup_vs_group_reference": (
                    group_reference_ms / fixed_triton_parallel_ms
                    if group_reference_ms and fixed_triton_parallel_ms
                    else None
                ),
                "tk_bf16_external_parallel_oracle_ms": fixed_tk_parallel_ms,
                "tk_bf16_external_parallel_oracle_speedup_vs_group_reference": (
                    group_reference_ms / fixed_tk_parallel_ms
                    if group_reference_ms and fixed_tk_parallel_ms
                    else None
                ),
                "work": repair_work_summary(
                    group_size=len(group_heads),
                    repair_count=len(fixed_repair_heads),
                    kv_len=int(k_true.shape[1]),
                    block_size=args.block_size,
                    sink_blocks=args.sink_blocks,
                    recent_blocks=args.recent_blocks,
                    middle_seed_blocks=args.middle_seed_blocks,
                ),
            }

        repair_order = _repair_order_from_errors(per_head_rows)
        sweep_rows = []
        for count in repair_counts:
            if count < 0 or count > len(group_heads):
                continue
            repair_heads = repair_order[:count]
            repair = repair_measure(repair_heads)
            repair_ms = repair.get("best_ms")
            parallel_oracle_ms = max(seed_group_ms, float(repair_ms)) if repair_ms is not None else None
            serial_ms = seed_group_ms + float(repair_ms) if repair_ms is not None else None
            tk_parallel_oracle_ms = (
                max(tk_seed_ms, float(repair_ms))
                if tk_seed_ms is not None and repair_ms is not None
                else None
            )
            sweep_rows.append(
                {
                    "repair_count": count,
                    "repair_heads": repair_heads,
                    "trusted_seed_heads": [head for head in group_heads if head not in set(repair_heads)],
                    "corrected_max_abs_error": corrected_max_error(per_head_rows, repair_heads=repair_heads),
                    "repair": repair,
                    "external_parallel_oracle_ms": parallel_oracle_ms,
                    "external_serial_ms": serial_ms,
                    "tk_bf16_external_parallel_oracle_ms": tk_parallel_oracle_ms,
                    "tk_bf16_external_parallel_oracle_speedup_vs_group_reference": (
                        group_reference_ms / tk_parallel_oracle_ms
                        if group_reference_ms and tk_parallel_oracle_ms
                        else None
                    ),
                    "tk_bf16_external_parallel_oracle_beats_group_reference": bool(
                        group_reference_ms is not None
                        and tk_parallel_oracle_ms is not None
                        and tk_parallel_oracle_ms < group_reference_ms
                    ),
                    "external_parallel_oracle_speedup_vs_group_reference": (
                        group_reference_ms / parallel_oracle_ms
                        if group_reference_ms and parallel_oracle_ms
                        else None
                    ),
                    "external_parallel_oracle_beats_group_reference": bool(
                        group_reference_ms is not None
                        and parallel_oracle_ms is not None
                        and parallel_oracle_ms < group_reference_ms
                    ),
                    "work": repair_work_summary(
                        group_size=len(group_heads),
                        repair_count=count,
                        kv_len=int(k_true.shape[1]),
                        block_size=args.block_size,
                        sink_blocks=args.sink_blocks,
                        recent_blocks=args.recent_blocks,
                        middle_seed_blocks=args.middle_seed_blocks,
                    ),
                }
            )

        group_rows.append(
            {
                "kv_head": kv_head,
                "q_heads": group_heads,
                "timing_ms": {
                    "seed_only_whole_group_ms": seed_group_ms,
                    "torch_dense_group_ms": dense_group_ms,
                    "flashinfer_group_tc_ms": group_tc_ms,
                    "flashinfer_group_no_tc_ms": group_no_tc_ms,
                    "flashinfer_group_tc_error": group_tc_error,
                    "flashinfer_group_no_tc_error": group_no_tc_error,
                    "reference_group_backend": group_reference_backend,
                    "reference_group_ms": group_reference_ms,
                    "seed_only_speedup_vs_group_reference": (
                        group_reference_ms / seed_group_ms if group_reference_ms and seed_group_ms else None
                    ),
                    "seed_only_speedup_vs_dense_all_reference": (
                        reference_all_ms / seed_group_ms if reference_all_ms and seed_group_ms else None
                    ),
                },
                "quality": {
                    "seed_only_vs_dense_group": _error(seed_out, dense_group_ref),
                    "seed_only_vs_dense_group_per_head": per_head_rows,
                },
                "tk_bf16": tk_bf16_payload,
                "repair_order_by_seed_error": repair_order,
                "budget_rows": budget_rows,
                "fixed_policy": fixed_policy_row,
                "repair_sweep": sweep_rows,
            }
        )

    budget_summary = []
    for budget in budgets:
        name = str(budget["name"])
        group_budget_rows = []
        for group in group_rows:
            match = next((row for row in group["budget_rows"] if row["budget"] == name), None)
            if match is not None:
                group_budget_rows.append(match)
        tk_values = [
            row.get("tk_bf16_external_parallel_oracle_ms")
            for row in group_budget_rows
            if row.get("tk_bf16_external_parallel_oracle_ms") is not None
        ]
        triton_values = [
            row.get("external_parallel_oracle_ms")
            for row in group_budget_rows
            if row.get("external_parallel_oracle_ms") is not None
        ]
        tk_layer_ms = max(float(value) for value in tk_values) if tk_values else None
        triton_layer_ms = max(float(value) for value in triton_values) if triton_values else None
        budget_summary.append(
            {
                "budget": name,
                "max_abs_error_budget": float(budget["max_abs_error"]),
                "tk_bf16_layer_parallel_oracle_ms": tk_layer_ms,
                "tk_bf16_layer_parallel_oracle_speedup_vs_reference_all": (
                    reference_all_ms / tk_layer_ms if reference_all_ms and tk_layer_ms else None
                ),
                "triton_layer_parallel_oracle_ms": triton_layer_ms,
                "triton_layer_parallel_oracle_speedup_vs_reference_all": (
                    reference_all_ms / triton_layer_ms if reference_all_ms and triton_layer_ms else None
                ),
                "repair_heads_by_group": [
                    {
                        "kv_head": group["kv_head"],
                        "repair_heads": row["repair_heads"],
                        "corrected_max_abs_error": row["corrected_max_abs_error"],
                        "tk_bf16_repair_heads": row.get("tk_bf16_repair_heads"),
                        "tk_bf16_corrected_max_abs_error": row.get("tk_bf16_corrected_max_abs_error"),
                    }
                    for group, row in zip(group_rows, group_budget_rows)
                ],
            }
        )

    fixed_group_rows = [
        {"kv_head": group["kv_head"], **group["fixed_policy"]}
        for group in group_rows
        if group.get("fixed_policy") is not None
    ]
    fixed_tk_values = [
        row.get("tk_bf16_external_parallel_oracle_ms")
        for row in fixed_group_rows
        if row.get("tk_bf16_external_parallel_oracle_ms") is not None
    ]
    fixed_triton_values = [
        row.get("triton_external_parallel_oracle_ms")
        for row in fixed_group_rows
        if row.get("triton_external_parallel_oracle_ms") is not None
    ]
    fixed_tk_layer_ms = max(float(value) for value in fixed_tk_values) if fixed_tk_values else None
    fixed_triton_layer_ms = (
        max(float(value) for value in fixed_triton_values) if fixed_triton_values else None
    )
    fixed_policy_summary = {
        "repair_policy": fixed_repair_policy,
        "group_rows": fixed_group_rows,
        "tk_bf16_layer_parallel_oracle_ms": fixed_tk_layer_ms,
        "tk_bf16_layer_parallel_oracle_speedup_vs_reference_all": (
            reference_all_ms / fixed_tk_layer_ms if reference_all_ms and fixed_tk_layer_ms else None
        ),
        "tk_bf16_layer_corrected_max_abs_error": max(
            (
                float(row.get("tk_bf16_corrected_max_abs_error") or 0.0)
                for row in fixed_group_rows
            ),
            default=None,
        )
        if fixed_group_rows
        else None,
        "triton_layer_parallel_oracle_ms": fixed_triton_layer_ms,
        "triton_layer_parallel_oracle_speedup_vs_reference_all": (
            reference_all_ms / fixed_triton_layer_ms
            if reference_all_ms and fixed_triton_layer_ms
            else None
        ),
        "triton_layer_corrected_max_abs_error": max(
            (float(row.get("corrected_max_abs_error") or 0.0) for row in fixed_group_rows),
            default=None,
        )
        if fixed_group_rows
        else None,
    }

    payload = {
        "schema": "streamattn.gate0.kv_group_repair_real.v1",
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
        "policy": {
            "kv_groups": kv_groups,
            "budgets": budgets,
            "block_size": args.block_size,
            "sink_blocks": args.sink_blocks,
            "recent_blocks": args.recent_blocks,
            "middle_seed_blocks": args.middle_seed_blocks,
            "block_order": args.block_order,
            "repair_counts": repair_counts,
        },
        "baselines": {
            "torch_dense_all_ms": dense_all_ms,
            "flashinfer_all_tc_ms": flashinfer_all_tc_ms,
            "flashinfer_all_no_tc_ms": flashinfer_all_no_tc_ms,
            "flashinfer_all_tc_error": flashinfer_all_tc_error,
            "flashinfer_all_no_tc_error": flashinfer_all_no_tc_error,
            "reference_all_ms": reference_all_ms,
        },
        "summary": {"budget_rows": budget_summary, "fixed_policy": fixed_policy_summary},
        "tk_compile": tk_compile,
        "kv_group_rows": group_rows,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-path", required=True)
    parser.add_argument("--k-path", required=True)
    parser.add_argument("--v-path", required=True)
    parser.add_argument("--true-kv-heads", type=int, required=True)
    parser.add_argument("--kv-groups", default="")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--budgets", default=DEFAULT_BUDGETS)
    parser.add_argument("--repair-counts", default="0,1,2,3,4,7")
    parser.add_argument("--fixed-repair-policy", default="")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--recent-blocks", type=int, default=2)
    parser.add_argument("--middle-seed-blocks", type=int, default=8)
    parser.add_argument(
        "--block-order",
        choices=["sequential", "recent_first", "sink_recent_first"],
        default="recent_first",
    )
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--group-warmup", type=int, default=3)
    parser.add_argument("--group-iters", type=int, default=10)
    parser.add_argument("--measure-triton-repair", action="store_true")
    parser.add_argument("--repair-num-chunks", type=int, default=256)
    parser.add_argument("--repair-block-d", type=int, default=32)
    parser.add_argument("--measure-tk-bf16", action="store_true")
    parser.add_argument("--tk-num-chunks", type=int, default=256)
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
