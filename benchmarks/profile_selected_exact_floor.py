"""Audit selected exact decode floors for true-GQA StreamAttn policies.

The hybrid Gate-0 policy only wins if exact repair / fallback rows can be
computed substantially cheaper than FlashInfer all-head exact decode. This
benchmark isolates why selected exact can fail to scale:

* FlashInfer all-head exact vs KV-group exact vs repair-row exact;
* pre-compacted tensors vs view inputs that trigger hidden ``contiguous`` work;
* explicit Q/K/V selection and copy costs;
* per-backend timing distributions and tensor layout diagnostics.

It intentionally does not run seed-only kernels. The only question here is
whether the exact side has a profitable floor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_gate0_hybrid_component_floors import (  # noqa: E402
    _distribution,
    _tensor_info,
    _time_distribution,
)
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
    _time_cuda,
)


def _parse_heads(raw: str) -> List[int]:
    return sorted(set(int(item.strip()) for item in raw.split(",") if item.strip()))


def _best_timing(rows: Dict[str, Dict[str, Any]]) -> tuple[str | None, float | None]:
    candidates = {
        name: row.get("median_ms")
        for name, row in rows.items()
        if row.get("median_ms") is not None
    }
    if not candidates:
        return None, None
    best = min(candidates, key=lambda name: float(candidates[name]))
    return best, float(candidates[best])


def _flashinfer_variants(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dense_ref: torch.Tensor,
    device: torch.device,
    warmup: int,
    iters: int,
    repeats: int,
) -> Dict[str, Any]:
    rows: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, str] = {}
    quality: Dict[str, Any] = {}
    if not HAS_FLASHINFER:
        return {
            "rows": rows,
            "errors": {"flashinfer": "FlashInfer is not available"},
            "quality": quality,
            "best_backend": None,
            "best_median_ms": None,
        }

    for backend, use_tc in (("flashinfer_tc", True), ("flashinfer_no_tc", False)):
        try:
            rows[backend] = _time_distribution(
                lambda use_tc=use_tc: _flashinfer_single_decode(
                    q,
                    k,
                    v,
                    use_tensor_cores=use_tc,
                ),
                device=device,
                warmup=warmup,
                iters=iters,
                repeats=repeats,
            )
            out = _flashinfer_single_decode(q, k, v, use_tensor_cores=use_tc)
            quality[backend] = _error(out, dense_ref)
            quality[f"{backend}_per_head"] = _per_head_error(out, dense_ref)
        except Exception as exc:  # pragma: no cover - backend/shape dependent
            errors[backend] = f"{type(exc).__name__}: {exc}"

    best_backend, best_median = _best_timing(rows)
    return {
        "rows": rows,
        "errors": errors,
        "quality": quality,
        "best_backend": best_backend,
        "best_median_ms": best_median,
    }


def _slice_q_group(q: torch.Tensor, kv_head: int, group_size: int) -> torch.Tensor:
    start = kv_head * group_size
    return q[:, :, start : start + group_size, :]


def _slice_kv_group(tensor: torch.Tensor, kv_head: int) -> torch.Tensor:
    return tensor[:, :, kv_head : kv_head + 1, :]


def _group_heads(kv_head: int, group_size: int) -> List[int]:
    start = kv_head * group_size
    return list(range(start, start + group_size))


def _scope_dense(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return _dense_true_gqa(q, k, v)


def _time_copy(
    name: str,
    fn: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    iters: int,
    repeats: int,
) -> Dict[str, Any]:
    try:
        sample = fn()
        return {
            "name": name,
            "timing": _time_distribution(
                fn,
                device=device,
                warmup=warmup,
                iters=iters,
                repeats=repeats,
            ),
            "result": _tensor_info(sample),
        }
    except Exception as exc:  # pragma: no cover - diagnostic only
        return {"name": name, "error": f"{type(exc).__name__}: {exc}"}


def _profile_flash_scope(
    *,
    scope: str,
    input_mode: str,
    q_heads: Sequence[int],
    kv_head: int | None,
    q_scope: torch.Tensor,
    k_scope: torch.Tensor,
    v_scope: torch.Tensor,
    device: torch.device,
    warmup: int,
    iters: int,
    repeats: int,
) -> Dict[str, Any]:
    dense_ref = _scope_dense(q_scope, k_scope, v_scope)
    timing = _flashinfer_variants(
        q_scope,
        k_scope,
        v_scope,
        dense_ref=dense_ref,
        device=device,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
    )
    return {
        "scope": scope,
        "input_mode": input_mode,
        "q_heads": list(q_heads),
        "kv_head": kv_head,
        "q": _tensor_info(q_scope),
        "k": _tensor_info(k_scope),
        "v": _tensor_info(v_scope),
        **timing,
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
        raise ValueError("selected exact floor profiler currently supports B=1, M=1")
    q_heads = int(q.shape[2])
    kv_heads = int(k_true.shape[2])
    if q_heads % kv_heads != 0:
        raise ValueError("q_heads must be divisible by true_kv_heads")
    group_size = q_heads // kv_heads

    repair_heads = _parse_heads(args.repair_heads)
    if not repair_heads:
        raise ValueError("--repair-heads must contain at least one head")
    repair_kv_heads = sorted({head // group_size for head in repair_heads})
    if len(repair_kv_heads) != 1:
        raise ValueError(f"repair heads must belong to one KV group; got {repair_kv_heads}")
    repair_kv_head = repair_kv_heads[0]

    print("[selected-exact-floor] materializing reference outputs", flush=True)
    dense_all = _dense_true_gqa(q, k_true, v_true)

    flash_scopes: List[Dict[str, Any]] = []
    print("[selected-exact-floor] timing FlashInfer all-head exact", flush=True)
    flash_scopes.append(
        _profile_flash_scope(
            scope="all_heads",
            input_mode="original_true_gqa_contiguous",
            q_heads=list(range(q_heads)),
            kv_head=None,
            q_scope=q,
            k_scope=k_true,
            v_scope=v_true,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
    )

    for kv_head in range(kv_heads):
        heads = _group_heads(kv_head, group_size)
        q_view = _slice_q_group(q, kv_head, group_size)
        k_view = _slice_kv_group(k_true, kv_head)
        v_view = _slice_kv_group(v_true, kv_head)
        print(f"[selected-exact-floor] timing KV group {kv_head} view inputs", flush=True)
        flash_scopes.append(
            _profile_flash_scope(
                scope=f"kv_group_{kv_head}",
                input_mode="view_internal_contiguous",
                q_heads=heads,
                kv_head=kv_head,
                q_scope=q_view,
                k_scope=k_view,
                v_scope=v_view,
                device=device,
                warmup=args.warmup,
                iters=args.iters,
                repeats=args.repeats,
            )
        )

        q_compact = _select_q_heads(q, heads)
        k_compact = _select_kv_head(k_true, kv_head)
        v_compact = _select_kv_head(v_true, kv_head)
        print(f"[selected-exact-floor] timing KV group {kv_head} precompact inputs", flush=True)
        flash_scopes.append(
            _profile_flash_scope(
                scope=f"kv_group_{kv_head}",
                input_mode="precompact_contiguous",
                q_heads=heads,
                kv_head=kv_head,
                q_scope=q_compact,
                k_scope=k_compact,
                v_scope=v_compact,
                device=device,
                warmup=args.warmup,
                iters=args.iters,
                repeats=args.repeats,
            )
        )

    q_repair = _select_q_heads(q, repair_heads)
    k_repair = _select_kv_head(k_true, repair_kv_head)
    v_repair = _select_kv_head(v_true, repair_kv_head)
    print(f"[selected-exact-floor] timing repair heads {repair_heads}", flush=True)
    flash_scopes.append(
        _profile_flash_scope(
            scope="repair_rows",
            input_mode="precompact_contiguous",
            q_heads=repair_heads,
            kv_head=repair_kv_head,
            q_scope=q_repair,
            k_scope=k_repair,
            v_scope=v_repair,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
    )

    repair_prefix_scopes: List[Dict[str, Any]] = []
    for count in range(1, min(args.repair_prefix_count, len(repair_heads)) + 1):
        heads = repair_heads[:count]
        q_prefix = _select_q_heads(q, heads)
        print(f"[selected-exact-floor] timing repair prefix count={count} heads={heads}", flush=True)
        repair_prefix_scopes.append(
            _profile_flash_scope(
                scope=f"repair_prefix_{count}",
                input_mode="precompact_contiguous",
                q_heads=heads,
                kv_head=repair_kv_head,
                q_scope=q_prefix,
                k_scope=k_repair,
                v_scope=v_repair,
                device=device,
                warmup=args.warmup,
                iters=args.iters,
                repeats=args.repeats,
            )
        )

    copy_costs = [
        _time_copy(
            "select_repair_q_index_select_contiguous",
            lambda: _select_q_heads(q, repair_heads),
            device=device,
            warmup=args.copy_warmup,
            iters=args.copy_iters,
            repeats=args.repeats,
        ),
        _time_copy(
            "select_repair_kv_k_contiguous",
            lambda: _select_kv_head(k_true, repair_kv_head),
            device=device,
            warmup=args.copy_warmup,
            iters=args.copy_iters,
            repeats=args.repeats,
        ),
        _time_copy(
            "select_repair_kv_v_contiguous",
            lambda: _select_kv_head(v_true, repair_kv_head),
            device=device,
            warmup=args.copy_warmup,
            iters=args.copy_iters,
            repeats=args.repeats,
        ),
    ]
    for kv_head in range(kv_heads):
        copy_costs.extend(
            [
                _time_copy(
                    f"kv_group_{kv_head}_q_view_contiguous",
                    lambda kv_head=kv_head: _slice_q_group(q, kv_head, group_size).contiguous(),
                    device=device,
                    warmup=args.copy_warmup,
                    iters=args.copy_iters,
                    repeats=args.repeats,
                ),
                _time_copy(
                    f"kv_group_{kv_head}_k_view_contiguous",
                    lambda kv_head=kv_head: _slice_kv_group(k_true, kv_head).contiguous(),
                    device=device,
                    warmup=args.copy_warmup,
                    iters=args.copy_iters,
                    repeats=args.repeats,
                ),
                _time_copy(
                    f"kv_group_{kv_head}_v_view_contiguous",
                    lambda kv_head=kv_head: _slice_kv_group(v_true, kv_head).contiguous(),
                    device=device,
                    warmup=args.copy_warmup,
                    iters=args.copy_iters,
                    repeats=args.repeats,
                ),
            ]
        )

    dense_checks = {
        "dense_all_vs_self": _error(dense_all, dense_all),
        "dense_all_per_head": _per_head_error(dense_all, dense_all),
    }

    all_best = next(row for row in flash_scopes if row["scope"] == "all_heads")
    all_best_ms = all_best.get("best_median_ms")
    group_best = [
        row
        for row in flash_scopes
        if row["scope"].startswith("kv_group_") and row["input_mode"] == "precompact_contiguous"
    ]
    group_medians = [
        float(row["best_median_ms"])
        for row in group_best
        if row.get("best_median_ms") is not None
    ]
    repair_row = next(row for row in flash_scopes if row["scope"] == "repair_rows")
    repair_best_ms = repair_row.get("best_median_ms")
    group_parallel_oracle = max(group_medians) if group_medians else None
    group_serial_sum = sum(group_medians) if group_medians else None

    ratios: Dict[str, Any] = {
        "all_flashinfer_best_ms": all_best_ms,
        "two_kv_group_parallel_oracle_ms": group_parallel_oracle,
        "two_kv_group_serial_sum_ms": group_serial_sum,
        "two_kv_group_oracle_speedup_vs_all": (
            all_best_ms / group_parallel_oracle
            if all_best_ms is not None and group_parallel_oracle
            else None
        ),
        "repair_best_ms": repair_best_ms,
        "repair_speedup_vs_all": (
            all_best_ms / repair_best_ms
            if all_best_ms is not None and repair_best_ms
            else None
        ),
    }
    for row in flash_scopes:
        if row.get("best_median_ms") is not None and all_best_ms is not None:
            ratios[f"{row['scope']}:{row['input_mode']}:fraction_of_all"] = (
                float(row["best_median_ms"]) / float(all_best_ms)
            )

    return {
        "schema": "streamattn.gate0.selected_exact_floor.v1",
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "flashinfer_available": HAS_FLASHINFER,
        "shape": {
            "batch": int(q.shape[0]),
            "query_len": int(q.shape[1]),
            "kv_len": int(k_true.shape[1]),
            "q_heads": q_heads,
            "true_kv_heads": kv_heads,
            "group_size": group_size,
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
        "policy_probe": {
            "repair_heads": repair_heads,
            "repair_kv_head": repair_kv_head,
        },
        "benchmark": {
            "warmup": args.warmup,
            "iters": args.iters,
            "copy_warmup": args.copy_warmup,
            "copy_iters": args.copy_iters,
            "repeats": args.repeats,
        },
        "flashinfer_scopes": flash_scopes,
        "repair_prefix_scopes": repair_prefix_scopes,
        "copy_costs": copy_costs,
        "dense_checks": dense_checks,
        "ratios": ratios,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-path", required=True)
    parser.add_argument("--k-path", required=True)
    parser.add_argument("--v-path", required=True)
    parser.add_argument("--true-kv-heads", type=int, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--repair-heads", default="0,1,5,6")
    parser.add_argument("--repair-prefix-count", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--copy-warmup", type=int, default=5)
    parser.add_argument("--copy-iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=7)
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
