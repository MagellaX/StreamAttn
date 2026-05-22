"""Profile a fixed KV-group seed+repair hybrid backend shape.

This is a backend realism benchmark, not the final runtime.  It evaluates the
current product-shaped StreamAttn policy:

* one true-GQA KV group runs whole-group seed-only;
* selected rows in that group are exactly repaired;
* the remaining KV group runs exact decode.

The benchmark keeps the policy KV-group structured and reports the gap between
the component oracle and real executable compositions.  That gap is the backend
overhead budget a one-call C++/CUDA/TK implementation must close.
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
    _flashinfer_single_decode,
    _per_head_error,
    _select_kv_head,
    _select_q_heads,
    _time_cuda_graph_replay,
    _time_parallel_groups,
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


def _best_timing(candidates: Dict[str, float | None]) -> tuple[str | None, float | None]:
    available = {name: value for name, value in candidates.items() if value is not None}
    if not available:
        return None, None
    key = min(available, key=lambda name: float(available[name]))
    return key, float(available[key])


def _time_flashinfer(
    fn: Callable[[bool], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    iters: int,
) -> Dict[str, Any]:
    timings: Dict[str, float | None] = {"flashinfer_tc": None, "flashinfer_no_tc": None}
    errors: Dict[str, str] = {}
    if not HAS_FLASHINFER:
        errors["flashinfer"] = "FlashInfer is not available"
        return {"timings_ms": timings, "errors": errors, "best_backend": None, "best_ms": None}
    for backend, use_tc in (("flashinfer_tc", True), ("flashinfer_no_tc", False)):
        try:
            timings[backend] = _time_cuda(
                lambda use_tc=use_tc: fn(use_tc),
                device=device,
                warmup=warmup,
                iters=iters,
            )
        except Exception as exc:  # pragma: no cover - backend/shape dependent
            errors[backend] = f"{type(exc).__name__}: {exc}"
    best_backend, best_ms = _best_timing(timings)
    return {
        "timings_ms": timings,
        "errors": errors,
        "best_backend": best_backend,
        "best_ms": best_ms,
    }


def _time_cuda_graph_replay_wall(
    fn: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    iters: int,
) -> tuple[float | None, str | None, torch.Tensor | None]:
    """CUDA graph replay timed with host wall clock and explicit sync.

    CUDA event timing can undercount if a captured library uses a stream in a
    way the event pair does not observe.  This intentionally measures the
    replay from the host side as a sanity check for sub-50us claims.
    """

    if device.type != "cuda":
        return None, "CUDA graph replay requires CUDA", None
    _sync(device)
    warmup_stream = torch.cuda.Stream(device=device)
    warmup_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(warmup_stream):
        for _ in range(max(1, warmup)):
            fn()
    torch.cuda.current_stream(device).wait_stream(warmup_stream)
    _sync(device)

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            output = fn()
    except Exception as exc:  # pragma: no cover - depends on CUDA graph support
        _sync(device)
        return None, repr(exc), None
    _sync(device)

    start = time.perf_counter()
    for _ in range(max(1, iters)):
        graph.replay()
    _sync(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(1, iters)
    return elapsed_ms, None, output


def _cuda_graph_mutation_check(
    fn: Callable[[], torch.Tensor],
    q: torch.Tensor,
    *,
    device: torch.device,
    warmup: int,
    delta: float,
) -> Dict[str, Any]:
    """Check whether graph replay reuses stale capture outputs.

    Some library calls can allocate/return tensors during capture in ways that
    make the replayed graph mostly copy capture-time outputs.  Mutating Q after
    capture should change the replay output if the graph really re-executes the
    attention kernels against the current input pointers.
    """

    if device.type != "cuda":
        return {"checked": False, "error": "CUDA graph replay requires CUDA"}
    _sync(device)
    for _ in range(max(1, warmup)):
        fn()
    _sync(device)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            output = fn()
    except Exception as exc:  # pragma: no cover - depends on CUDA graph support
        _sync(device)
        return {"checked": False, "error": repr(exc)}
    _sync(device)
    graph.replay()
    _sync(device)
    baseline = output.detach().clone()
    with torch.no_grad():
        q.add_(delta)
    graph.replay()
    _sync(device)
    mutated = output.detach().clone()
    with torch.no_grad():
        q.sub_(delta)
    graph.replay()
    _sync(device)
    restored = output.detach().clone()
    return {
        "checked": True,
        "delta": delta,
        "max_change_after_q_mutation": float((mutated - baseline).detach().abs().float().max().item()),
        "max_change_after_restore": float((restored - baseline).detach().abs().float().max().item()),
    }


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


def _make_tk_seed_group_fn(
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
) -> tuple[Callable[[], torch.Tensor], Dict[str, Any]]:
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
    active_chunks, active_counts, flat_active_chunks, active_offsets, active_by_kv = (
        _pack_active_chunks_by_kv_group(
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
    )
    block_order_id = 0 if block_order == "sequential" else 1

    def tk_seed_group() -> torch.Tensor:
        packed = ext.head_mode_compact_chunk_merged(
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
        return _unpack_q_by_kv_group(packed, local_q_heads)[:, None]

    return tk_seed_group, {
        "backend": "tk_bf16_head_mode_compact",
        "active_chunks": list(active_by_kv[0]),
        "active_chunk_count": len(active_by_kv[0]),
        "num_chunks": num_chunks,
    }


def _policy_rows(
    *,
    repair_policy: Dict[int, List[int]],
    q_heads: int,
    kv_heads: int,
    seed_kv_groups: Sequence[int],
) -> Dict[str, Any]:
    groups = []
    exact_heads: List[int] = []
    trusted_seed_heads: List[int] = []
    repair_heads: List[int] = []
    seed_kv_set = set(seed_kv_groups)
    for kv_head in range(kv_heads):
        group_heads = q_heads_for_kv_group(kv_head, q_heads=q_heads, kv_heads=kv_heads)
        group_repair = sorted(set(repair_policy.get(kv_head, [])))
        invalid = [head for head in group_repair if head not in set(group_heads)]
        if invalid:
            raise ValueError(f"repair policy for kv_group={kv_head} contains invalid heads {invalid}")
        if kv_head in seed_kv_set:
            trusted = [head for head in group_heads if head not in set(group_repair)]
            trusted_seed_heads.extend(trusted)
            repair_heads.extend(group_repair)
            groups.append(
                {
                    "kv_head": kv_head,
                    "mode": "seed_only_with_exact_repair",
                    "trusted_seed_q_heads": trusted,
                    "repair_q_heads": group_repair,
                }
            )
        else:
            exact = group_repair if group_repair else group_heads
            exact_heads.extend(exact)
            groups.append({"kv_head": kv_head, "mode": "exact_full", "exact_q_heads": exact})
    return {
        "groups": groups,
        "trusted_seed_heads": trusted_seed_heads,
        "repair_heads": repair_heads,
        "exact_heads": exact_heads,
    }


def profile(args: argparse.Namespace) -> Dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.seed_backend != "tk_bf16":
        raise ValueError("this backend spike currently supports --seed-backend tk_bf16 only")
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
        raise ValueError("hybrid backend spike currently supports B=1, M=1")
    q_heads = int(q.shape[2])
    kv_heads = int(k_true.shape[2])
    if q_heads % kv_heads != 0:
        raise ValueError("q_heads must be divisible by true_kv_heads")
    seed_kv_groups = _parse_ints(args.seed_kv_groups)
    repair_policy = parse_fixed_repair_policy(args.repair_policy)
    policy = _policy_rows(
        repair_policy=repair_policy,
        q_heads=q_heads,
        kv_heads=kv_heads,
        seed_kv_groups=seed_kv_groups,
    )

    print("[kv-hybrid-backend] loading baselines", flush=True)
    dense_ref = _dense_true_gqa(q, k_true, v_true)

    def dense_all() -> torch.Tensor:
        return _dense_true_gqa(q, k_true, v_true)

    def flashinfer_all(use_tc: bool) -> torch.Tensor:
        return _flashinfer_single_decode(q, k_true, v_true, use_tensor_cores=use_tc)

    dense_all_ms = _time_cuda(dense_all, device=device, warmup=args.warmup, iters=args.iters)
    flashinfer_all = _time_flashinfer(
        flashinfer_all,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    reference_all_ms = (
        flashinfer_all["best_ms"] if flashinfer_all["best_ms"] is not None else dense_all_ms
    )
    reference_all_backend = flashinfer_all["best_backend"] or "torch_sdpa"

    print("[kv-hybrid-backend] compiling TK seed component", flush=True)
    tk_ext, tk_compile = _compile_tk(args)

    seed_components: List[Dict[str, Any]] = []
    exact_components: List[Dict[str, Any]] = []
    for kv_head in range(kv_heads):
        group_heads = q_heads_for_kv_group(kv_head, q_heads=q_heads, kv_heads=kv_heads)
        k_group = _select_kv_head(k_true, kv_head)
        v_group = _select_kv_head(v_true, kv_head)
        if kv_head in seed_kv_groups:
            q_group = _select_q_heads(q, group_heads)
            seed_fn, seed_info = _make_tk_seed_group_fn(
                ext=tk_ext,
                q_group=q_group,
                k_group=k_group,
                v_group=v_group,
                block_size=args.block_size,
                sink_blocks=args.sink_blocks,
                recent_blocks=args.recent_blocks,
                middle_seed_blocks=args.middle_seed_blocks,
                block_order=args.block_order,
                num_chunks=args.tk_num_chunks,
                device=device,
            )
            seed_idx = torch.tensor(group_heads, device=device, dtype=torch.long)
            seed_components.append(
                {
                    "kv_head": kv_head,
                    "q_heads": group_heads,
                    "index": seed_idx,
                    "fn": seed_fn,
                    "info": seed_info,
                }
            )
        repair_heads = sorted(set(repair_policy.get(kv_head, [])))
        if kv_head not in seed_kv_groups and not repair_heads:
            repair_heads = group_heads
        if repair_heads:
            q_exact = _select_q_heads(q, repair_heads)
            index = torch.tensor(repair_heads, device=device, dtype=torch.long)

            def make_exact(qe: torch.Tensor, kg: torch.Tensor, vg: torch.Tensor) -> Callable[[bool], torch.Tensor]:
                return lambda use_tc: _flashinfer_single_decode(
                    qe,
                    kg,
                    vg,
                    use_tensor_cores=use_tc,
                )

            exact_components.append(
                {
                    "kv_head": kv_head,
                    "q_heads": repair_heads,
                    "index": index,
                    "fn_by_backend": make_exact(q_exact, k_group, v_group),
                }
            )

    print("[kv-hybrid-backend] timing components", flush=True)
    seed_rows = []
    for component in seed_components:
        ms = _time_cuda(
            component["fn"],
            device=device,
            warmup=args.group_warmup,
            iters=args.group_iters,
        )
        seed_out = component["fn"]().to(dtype)
        dense_group_ref = dense_ref.index_select(2, component["index"])
        seed_rows.append(
            {
                "kv_head": component["kv_head"],
                "q_heads": component["q_heads"],
                "seed_only_ms": ms,
                "quality_vs_dense_group": _error(seed_out, dense_group_ref),
                "quality_vs_dense_group_per_head": _per_head_error(seed_out, dense_group_ref)["per_head"],
                **component["info"],
            }
        )
        component["best_ms"] = ms

    exact_rows = []
    for component in exact_components:
        timing = _time_flashinfer(
            component["fn_by_backend"],
            device=device,
            warmup=args.group_warmup,
            iters=args.group_iters,
        )
        best_backend = timing["best_backend"]
        if best_backend is None:
            raise RuntimeError(f"no exact backend available for component {component['q_heads']}: {timing}")
        use_tc = best_backend == "flashinfer_tc"
        exact_out = component["fn_by_backend"](use_tc)
        dense_group_ref = dense_ref.index_select(2, component["index"])
        exact_rows.append(
            {
                "kv_head": component["kv_head"],
                "q_heads": component["q_heads"],
                "timings_ms": timing["timings_ms"],
                "errors": timing["errors"],
                "best_backend": best_backend,
                "best_ms": timing["best_ms"],
                "quality_vs_dense": _error(exact_out, dense_group_ref),
            }
        )
        component["best_backend"] = best_backend
        component["best_ms"] = timing["best_ms"]

    out_static = torch.empty_like(q)

    def hybrid_static() -> torch.Tensor:
        for component in seed_components:
            seed_out = component["fn"]().to(dtype)
            out_static.index_copy_(2, component["index"], seed_out)
        for component in exact_components:
            use_tc = component["best_backend"] == "flashinfer_tc"
            exact_out = component["fn_by_backend"](use_tc)
            out_static.index_copy_(2, component["index"], exact_out)
        return out_static

    def hybrid_allocating() -> torch.Tensor:
        out = torch.empty_like(q)
        for component in seed_components:
            seed_out = component["fn"]().to(dtype)
            out.index_copy_(2, component["index"], seed_out)
        for component in exact_components:
            use_tc = component["best_backend"] == "flashinfer_tc"
            exact_out = component["fn_by_backend"](use_tc)
            out.index_copy_(2, component["index"], exact_out)
        return out

    hybrid_out = hybrid_static()
    _sync(device)
    hybrid_quality = _error(hybrid_out, dense_ref)
    hybrid_per_head = _per_head_error(hybrid_out, dense_ref)["per_head"]

    print("[kv-hybrid-backend] timing executable compositions", flush=True)
    serial_ms = _time_cuda(hybrid_allocating, device=device, warmup=args.warmup, iters=args.iters)
    static_serial_ms = _time_cuda(hybrid_static, device=device, warmup=args.warmup, iters=args.iters)
    graph_ms, graph_error = (
        _time_cuda_graph_replay(
            hybrid_static,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
        if args.measure_cuda_graph
        else (None, None)
    )
    graph_wall_ms, graph_wall_error, graph_wall_output = (
        _time_cuda_graph_replay_wall(
            hybrid_static,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
        if args.measure_cuda_graph_wall
        else (None, None, None)
    )
    graph_wall_quality = _error(graph_wall_output, dense_ref) if graph_wall_output is not None else None
    graph_mutation = (
        _cuda_graph_mutation_check(
            hybrid_static,
            q,
            device=device,
            warmup=args.warmup,
            delta=args.graph_mutation_delta,
        )
        if args.measure_graph_mutation_check
        else {"checked": False}
    )
    parallel_stream_ms = (
        _time_parallel_groups(
            [component["fn"] for component in seed_components]
            + [
                (
                    lambda component=component: component["fn_by_backend"](
                        component["best_backend"] == "flashinfer_tc"
                    )
                )
                for component in exact_components
            ],
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
        if args.measure_parallel_streams
        else None
    )

    component_ms = [float(component["best_ms"]) for component in seed_components + exact_components]
    oracle_ms = max(component_ms) if component_ms else 0.0
    serial_component_sum_ms = sum(component_ms)
    graph_dynamic_inputs = (
        not graph_mutation.get("checked")
        or float(graph_mutation.get("max_change_after_q_mutation") or 0.0) > args.graph_dynamic_epsilon
    )
    usable_graph_ms = graph_ms if graph_dynamic_inputs else None
    usable_graph_wall_ms = graph_wall_ms if graph_dynamic_inputs else None
    real_candidates = {
        "python_allocating_serial": serial_ms,
        "python_static_serial": static_serial_ms,
        "cuda_graph_replay": usable_graph_ms,
        "cuda_graph_replay_wall": usable_graph_wall_ms,
        "parallel_stream_components_only": parallel_stream_ms,
    }
    best_real_backend, best_real_ms = _best_timing(real_candidates)

    return {
        "schema": "streamattn.gate0.kv_group_hybrid_backend.v1",
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
            "seed_backend": args.seed_backend,
        },
        "policy": {
            "mode": "kv_group_seed_with_exact_repair",
            "seed_kv_groups": seed_kv_groups,
            "repair_policy": repair_policy,
            **policy,
            "block_size": args.block_size,
            "sink_blocks": args.sink_blocks,
            "recent_blocks": args.recent_blocks,
            "middle_seed_blocks": args.middle_seed_blocks,
            "block_order": args.block_order,
        },
        "baselines": {
            "torch_dense_all_ms": dense_all_ms,
            "flashinfer_all": flashinfer_all,
            "reference_all_backend": reference_all_backend,
            "reference_all_ms": reference_all_ms,
        },
        "components": {
            "seed": seed_rows,
            "exact": exact_rows,
            "component_parallel_oracle_ms": oracle_ms,
            "component_serial_sum_ms": serial_component_sum_ms,
        },
        "backend_realism": {
            "python_allocating_serial_ms": serial_ms,
            "python_static_serial_ms": static_serial_ms,
            "cuda_graph_replay_ms": graph_ms,
            "cuda_graph_error": graph_error,
            "cuda_graph_replay_wall_ms": graph_wall_ms,
            "cuda_graph_wall_error": graph_wall_error,
            "cuda_graph_wall_quality_vs_dense": graph_wall_quality,
            "cuda_graph_mutation_check": graph_mutation,
            "cuda_graph_replay_uses_dynamic_inputs": graph_dynamic_inputs,
            "cuda_graph_replay_excluded_reason": (
                None if graph_dynamic_inputs else "graph replay output did not change after q mutation"
            ),
            "parallel_stream_components_only_ms": parallel_stream_ms,
            "best_real_backend": best_real_backend,
            "best_real_ms": best_real_ms,
            "best_real_overhead_vs_oracle_ms": (
                best_real_ms - oracle_ms if best_real_ms is not None else None
            ),
            "best_real_speedup_vs_reference_all": (
                reference_all_ms / best_real_ms if reference_all_ms and best_real_ms else None
            ),
            "oracle_speedup_vs_reference_all": (
                reference_all_ms / oracle_ms if reference_all_ms and oracle_ms else None
            ),
            "oracle_margin_vs_reference_all_ms": (
                reference_all_ms - oracle_ms if reference_all_ms is not None else None
            ),
            "best_real_beats_reference_all": bool(
                reference_all_ms is not None and best_real_ms is not None and best_real_ms < reference_all_ms
            ),
        },
        "quality": {
            "hybrid_vs_dense": hybrid_quality,
            "hybrid_vs_dense_per_head": hybrid_per_head,
        },
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
    parser.add_argument("--seed-backend", choices=["tk_bf16"], default="tk_bf16")
    parser.add_argument("--seed-kv-groups", default="0")
    parser.add_argument("--repair-policy", default=DEFAULT_REPAIR_POLICY)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--recent-blocks", type=int, default=2)
    parser.add_argument("--middle-seed-blocks", type=int, default=8)
    parser.add_argument("--block-order", choices=["sequential", "recent_first"], default="recent_first")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--group-warmup", type=int, default=3)
    parser.add_argument("--group-iters", type=int, default=10)
    parser.add_argument("--tk-num-chunks", type=int, default=256)
    parser.add_argument("--measure-cuda-graph", action="store_true")
    parser.add_argument("--measure-cuda-graph-wall", action="store_true")
    parser.add_argument("--measure-graph-mutation-check", action="store_true")
    parser.add_argument("--graph-mutation-delta", type=float, default=1.0)
    parser.add_argument("--graph-dynamic-epsilon", type=float, default=1e-6)
    parser.add_argument("--measure-parallel-streams", action="store_true")
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
