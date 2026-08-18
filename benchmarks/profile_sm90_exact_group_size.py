"""Gate experimental SM90 exact decode group sizes against FlashInfer.

The native producer always uses WGMMA m64n8k16. For G4, four columns are real
query heads and four are zero-filled inside the kernel. This benchmark measures
whether the additional KV-group CTAs compensate for that 50% column utilization.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_stream_attn_gate0_wrapper import _error, _time_cuda  # noqa: E402
from benchmarks.profile_transposed_wgmma_exact_qk import (  # noqa: E402
    FLASHINFER_IMPORT_ERROR,
    _flashinfer_batched_runner,
    _paired_cuda_ratio,
)
from stream_attention.backends.sm90.transposed_gqa_exact import (  # noqa: E402
    ExactDecodePlan,
    supports_transposed_gqa_exact,
)
from stream_attention.decode import StreamAttnExactNativeDirectRunner  # noqa: E402


def _parse_ints(raw: str) -> list[int]:
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not values:
        raise ValueError("at least one split count is required")
    return values


def _time_repeated(
    fn, *, warmup: int, iters: int, repeats: int
) -> tuple[float, list[float]]:
    samples = [
        float(_time_cuda(fn, device=torch.device("cuda"), warmup=warmup, iters=iters))
        for _ in range(repeats)
    ]
    return float(statistics.median(samples)), samples


def _reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    batch, q_heads, dim = q.shape
    kv_heads = int(k.shape[1])
    group_size = q_heads // kv_heads
    q_group = q.view(batch, kv_heads, group_size, dim).float()
    scores = torch.einsum("bhgd,bhnd->bhgn", q_group, k.float())
    scores.mul_(1.0 / math.sqrt(float(dim)))
    probs = scores.softmax(dim=-1)
    return torch.einsum("bhgn,bhnd->bhgd", probs, v.float())


def profile(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    if capability != (9, 0):
        raise RuntimeError(f"SM90 is required, got capability={capability}")
    if args.q_heads % args.kv_heads:
        raise ValueError("q_heads must be divisible by kv_heads")
    group_size = args.q_heads // args.kv_heads
    if group_size not in (4, 8):
        raise ValueError("this experiment supports group_size 4 or 8")
    if args.head_dim not in (64, 128):
        raise ValueError("this experiment supports D64 or D128")
    if args.kv_len % 64:
        raise ValueError("kv_len must be divisible by 64")

    torch.manual_seed(args.seed)
    q = torch.randn(
        args.batch,
        args.q_heads,
        args.head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        args.batch,
        args.kv_heads,
        args.kv_len,
        args.head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)
    k_nhd = k.permute(0, 2, 1, 3).contiguous()
    v_nhd = v.permute(0, 2, 1, 3).contiguous()

    reference = _reference(q, k, v)
    plans: dict[int, ExactDecodePlan] = {}
    sweep: dict[str, dict[str, Any]] = {}
    for splits in _parse_ints(args.num_splits_list):
        if splits > args.kv_len // 64:
            continue
        plan = ExactDecodePlan.build(
            q.unsqueeze(1),
            k,
            v,
            num_splits=splits,
            cutlass_root=Path(args.cutlass_root),
            build_dir=Path(args.build_dir) if args.build_dir else None,
            promoted_only=False,
        )
        plan.run()
        torch.cuda.synchronize()
        first = plan.output.clone()
        plan.run()
        torch.cuda.synchronize()
        quality = _error(plan.output.view_as(reference).float(), reference)
        quality.update(
            {
                "repeat_max_abs_diff": float(
                    (plan.output - first).abs().max().item()
                ),
                "nonfinite_count": int((~torch.isfinite(plan.output)).sum().item()),
            }
        )
        total_ms, total_samples = _time_repeated(
            plan.run,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        partial_call = lambda p=plan: p.partial_launch(  # noqa: E731
            p.query_group,
            p.key_cache,
            p.value_cache,
            p.partial_output,
            p.partial_lse,
            p.num_splits,
        )
        merge_call = lambda p=plan: p.warp_merge_launch(  # noqa: E731
            p.partial_output, p.partial_lse, p.output_group
        )
        partial_ms, partial_samples = _time_repeated(
            partial_call,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        merge_ms, merge_samples = _time_repeated(
            merge_call,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        plans[splits] = plan
        sweep[str(splits)] = {
            "producer_ctas": args.batch * args.kv_heads * splits,
            "tiles_per_split": math.ceil((args.kv_len // 64) / splits),
            "workspace_bytes": plan.workspace_bytes,
            "total_ms": total_ms,
            "total_samples_ms": total_samples,
            "partial_ms": partial_ms,
            "partial_samples_ms": partial_samples,
            "merge_ms": merge_ms,
            "merge_samples_ms": merge_samples,
            "quality": quality,
        }
        print(
            f"[gqa-exact] C={splits} CTAs={args.batch * args.kv_heads * splits} "
            f"partial={partial_ms:.6f} merge={merge_ms:.6f} total={total_ms:.6f}",
            flush=True,
        )

    if not sweep:
        raise ValueError("no valid split counts")
    best_splits = min(sweep, key=lambda key: float(sweep[key]["total_ms"]))
    best_plan = plans[int(best_splits)]

    q_before = q.clone()
    best_plan.run()
    torch.cuda.synchronize()
    output_before = best_plan.output.clone()
    q.add_(0.125)
    best_plan.run()
    torch.cuda.synchronize()
    mutation_max_abs_diff = float(
        (best_plan.output - output_before).abs().max().item()
    )
    q.copy_(q_before)
    best_plan.run()
    torch.cuda.synchronize()

    flashinfer_error = None
    flashinfer_run = None
    flashinfer_quality = None
    try:
        flashinfer_run = _flashinfer_batched_runner(
            q, k_nhd, v_nhd, page_size=args.page_size
        )
        flashinfer_output = flashinfer_run().clone()
        torch.cuda.synchronize()
        flashinfer_quality = _error(flashinfer_output.float(), reference.view_as(q))
    except Exception as exc:  # pragma: no cover - benchmark environment dependent
        flashinfer_error = f"{type(exc).__name__}: {exc}"

    paired = None
    if flashinfer_run is not None:
        paired = _paired_cuda_ratio(
            best_plan.run,
            flashinfer_run,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
            repeats=max(9, args.paired_repeats),
        )

    serving = None
    serving_paired = None
    if supports_transposed_gqa_exact(
        q.unsqueeze(1), k, v, require_cutlass=False
    ):
        serving_output = torch.empty_like(q.unsqueeze(1))
        serving_runner = StreamAttnExactNativeDirectRunner(
            query=q.unsqueeze(1),
            key_cache=k,
            value_cache=v,
            output=serving_output,
            info=None,
        )
        serving_runner.run()
        torch.cuda.synchronize()
        serving = {
            "backend_variant": serving_runner.backend_variant,
            "num_splits": serving_runner._sm90_plan.num_splits,
            "quality": _error(serving_output.view_as(reference).float(), reference),
        }
        if flashinfer_run is not None:
            serving_paired = _paired_cuda_ratio(
                serving_runner.run,
                flashinfer_run,
                device=device,
                warmup=args.warmup,
                iters=args.iters,
                repeats=max(9, args.paired_repeats),
            )

    quality = sweep[best_splits]["quality"]
    correctness_pass = (
        quality["max_abs_error"] <= args.max_abs_error
        and quality["nonfinite_count"] == 0
        and quality["repeat_max_abs_diff"] == 0.0
        and mutation_max_abs_diff > 0.0
    )
    paired_pass = bool(
        paired is not None
        and paired["wins"] == paired["trials"]
        and paired["ratio_min"] > 1.0
    )
    serving_pass = bool(
        serving_paired is not None
        and serving_paired["wins"] == serving_paired["trials"]
        and serving_paired["ratio_min"] > 1.0
    )
    return {
        "schema": "streamattn.sm90_exact_group_size.v1",
        "device": torch.cuda.get_device_name(device),
        "shape": {
            "batch": args.batch,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "group_size": group_size,
            "kv_len": args.kv_len,
            "head_dim": args.head_dim,
            "dtype": "bf16",
        },
        "kernel": {
            "wgmma_atom": "m64n8k16.f32.bf16.bf16",
            "qk_k_steps": args.head_dim // 16,
            "pv_m_tiles": args.head_dim // 64,
            "active_wgmma_columns": group_size,
            "physical_wgmma_columns": 8,
            "column_utilization": group_size / 8.0,
            "inactive_columns_zero_filled_in_kernel": group_size < 8,
            "estimated_static_shared_bytes": (
                2 * (2 if args.head_dim == 64 else 1) * 64 * args.head_dim * 2
                + 8 * args.head_dim * 2
                + 64 * 8 * 2
                + (4 * 8 + 8 + 8) * 4
            ),
        },
        "benchmark": {
            "warmup": args.warmup,
            "iters": args.iters,
            "repeats": args.repeats,
            "paired_repeats": max(9, args.paired_repeats),
        },
        "sweep": sweep,
        "best": {
            "num_splits": int(best_splits),
            **sweep[best_splits],
            "mutation_max_abs_diff": mutation_max_abs_diff,
        },
        "flashinfer": {
            "import_error": FLASHINFER_IMPORT_ERROR,
            "run_error": flashinfer_error,
            "quality": flashinfer_quality,
        },
        "paired_vs_flashinfer": paired,
        "serving": serving,
        "serving_paired_vs_flashinfer": serving_paired,
        "decision": {
            "correctness": "pass" if correctness_pass else "fail",
            "paired_gate": "pass" if paired_pass else "fail",
            "experimental_promotion_candidate": correctness_pass and paired_pass,
            "serving_gate": (
                "pass" if serving_pass else "fail" if serving is not None else "not_promoted"
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--kv-len", type=int, default=32768)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--num-splits-list", default="8,16,32,64")
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--paired-repeats", type=int, default=9)
    parser.add_argument("--max-abs-error", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cutlass-root", required=True)
    parser.add_argument("--build-dir", default="")
    parser.add_argument("--output-json", default="")
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = profile(args)
    payload = json.dumps(result, indent=2)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
