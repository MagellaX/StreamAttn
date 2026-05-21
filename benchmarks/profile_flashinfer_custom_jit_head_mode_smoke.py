"""Smoke-test FlashInfer custom-JIT as a StreamAttn head-mode attach point.

The StreamAttn runtime path cannot survive as a separate sparse Triton launch
beside FlashInfer.  This benchmark checks the next, narrower question:

Can FlashInfer's batch-decode custom JIT path carry an extra per-Q-head
``head_modes`` tensor while preserving exact default attention behavior and
near-default latency?  In the riskier mode, can a custom logits-mask variant
turn that tensor into a seed-only sparse policy inside the FlashInfer launch?

The exact-equivalent mode is deliberately boring: the custom variant still uses
FlashInfer's default attention math; ``head_modes`` is passed through the JIT
signature but not consumed yet.  The seed-only mode is an evidence probe.  It
uses a custom mask to make selected heads attend only to sink/recent/seed
blocks.  If that does not reduce runtime, FlashInfer's variant hook is too late
in the schedule and StreamAttn needs a scheduler-level backend change.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_stream_attn_gate0_wrapper import _error, _time_cuda  # noqa: E402

try:
    import flashinfer

    HAS_FLASHINFER = True
except Exception:  # pragma: no cover - optional dependency
    flashinfer = None
    HAS_FLASHINFER = False


DEFAULT_PREFILL_TENSOR_NAMES = [
    "maybe_custom_mask",
    "maybe_mask_indptr",
    "maybe_alibi_slopes",
    "maybe_prefix_len_ptr",
    "maybe_token_pos_in_items_ptr",
    "maybe_max_item_len_ptr",
    "maybe_k_cache_sf",
    "maybe_v_cache_sf",
]

DEFAULT_PREFILL_TENSOR_DTYPES = [
    "uint8_t",
    "int32_t",
    "float",
    "uint32_t",
    "uint16_t",
    "uint16_t",
    "uint8_t",
    "uint8_t",
]

DEFAULT_PREFILL_SCALAR_NAMES = [
    "logits_soft_cap",
    "sm_scale",
    "rope_rcp_scale",
    "rope_rcp_theta",
    "token_pos_in_items_len",
]

DEFAULT_PREFILL_SCALAR_DTYPES = ["double", "double", "double", "double", "int64_t"]

SEED_ONLY_SCALAR_NAMES = [
    "block_size",
    "sink_blocks",
    "recent_blocks",
    "middle_seed_blocks",
    "block_order",
]

SEED_ONLY_SCALAR_DTYPES = ["int64_t", "int64_t", "int64_t", "int64_t", "int64_t"]

HEAD_MODE_SEED_ONLY_VARIANT_DECL = r"""
#include<flashinfer/attention/variants.cuh>

struct StreamAttnHeadModeSeedOnly : AttentionVariantBase {
  static constexpr bool use_softmax = true;

  int32_t* head_modes;
  uint32_t qo_len, kv_len;
  uint32_t window_left;
  uint32_t block_size, sink_blocks, recent_blocks, middle_seed_blocks, block_order;
  float sm_scale_log2;

  template <typename Params>
  __device__ __host__ StreamAttnHeadModeSeedOnly(
      const Params& params, uint32_t batch_idx, uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    head_modes = params.head_modes;
    block_size = static_cast<uint32_t>(params.block_size);
    sink_blocks = static_cast<uint32_t>(params.sink_blocks);
    recent_blocks = static_cast<uint32_t>(params.recent_blocks);
    middle_seed_blocks = static_cast<uint32_t>(params.middle_seed_blocks);
    block_order = static_cast<uint32_t>(params.block_order);
    window_left = (params.window_left >= 0) ? params.window_left : kv_len;
    sm_scale_log2 = params.sm_scale * math::log2e;
  }

  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    bool seed_keep = true;
    const bool seed_only = head_modes[qo_head_idx] != 0;
    if (seed_only) {
      const uint32_t bs = block_size == 0 ? 1 : block_size;
      const uint32_t num_blocks = (kv_len + bs - 1) / bs;
      const uint32_t sink_end = min(sink_blocks * bs, kv_len);
      const uint32_t recent_start = recent_blocks >= num_blocks
          ? 0
          : (num_blocks - recent_blocks) * bs;
      const uint32_t middle_seed_tokens = middle_seed_blocks * bs;

      seed_keep = (kv_idx < sink_end) || (kv_idx >= recent_start);
      if (middle_seed_blocks > 0) {
        if (block_order == 0) {
          const uint32_t middle_start = sink_end;
          const uint32_t middle_end = min(middle_start + middle_seed_tokens, recent_start);
          seed_keep = seed_keep || ((kv_idx >= middle_start) && (kv_idx < middle_end));
        } else {
          uint32_t middle_end = recent_start;
          uint32_t middle_start = middle_end > middle_seed_tokens
              ? middle_end - middle_seed_tokens
              : sink_end;
          middle_start = max(middle_start, sink_end);
          seed_keep = seed_keep || ((kv_idx >= middle_start) && (kv_idx < middle_end));
        }
      }
    }
    if ((kv_idx >= kv_len) || (seed_only && !seed_keep)) {
      logits = -50000.0f;
    }
    return logits;
  })

  REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    bool mask = (qo_idx < qo_len) && (kv_idx < kv_len);
    const bool seed_only = head_modes[qo_head_idx] != 0;
    if (seed_only) {
      const uint32_t bs = block_size == 0 ? 1 : block_size;
      const uint32_t num_blocks = (kv_len + bs - 1) / bs;
      const uint32_t sink_end = min(sink_blocks * bs, kv_len);
      const uint32_t recent_start = recent_blocks >= num_blocks
          ? 0
          : (num_blocks - recent_blocks) * bs;
      const uint32_t middle_seed_tokens = middle_seed_blocks * bs;

      bool keep = (kv_idx < sink_end) || (kv_idx >= recent_start);
      if (middle_seed_blocks > 0) {
        if (block_order == 0) {
          const uint32_t middle_start = sink_end;
          const uint32_t middle_end = min(middle_start + middle_seed_tokens, recent_start);
          keep = keep || ((kv_idx >= middle_start) && (kv_idx < middle_end));
        } else {
          uint32_t middle_end = recent_start;
          uint32_t middle_start = middle_end > middle_seed_tokens
              ? middle_end - middle_seed_tokens
              : sink_end;
          middle_start = max(middle_start, sink_end);
          keep = keep || ((kv_idx >= middle_start) && (kv_idx < middle_end));
        }
      }
      mask = mask && keep;
    }
    return mask;
  })
};
"""


def _dtype(raw: str) -> torch.dtype:
    if raw == "fp16":
        return torch.float16
    if raw == "bf16":
        return torch.bfloat16
    if raw == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {raw}")


def _make_paged_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    page_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack dense [B, N, Hkv, D] K/V into FlashInfer NHD paged layout."""

    if k.shape != v.shape or k.ndim != 4:
        raise ValueError(f"expected matching [B,N,H,D] K/V, got {k.shape=} {v.shape=}")
    batch, kv_len, kv_heads, head_dim = k.shape
    num_pages_per_req = math.ceil(kv_len / page_size)
    num_pages_total = batch * num_pages_per_req
    cache = torch.zeros(
        num_pages_total,
        2,
        page_size,
        kv_heads,
        head_dim,
        device=k.device,
        dtype=k.dtype,
    )
    for batch_idx in range(batch):
        for page_idx in range(num_pages_per_req):
            start = page_idx * page_size
            end = min(kv_len, start + page_size)
            width = end - start
            global_page = batch_idx * num_pages_per_req + page_idx
            cache[global_page, 0, :width] = k[batch_idx, start:end]
            cache[global_page, 1, :width] = v[batch_idx, start:end]
    indptr = torch.arange(
        0,
        num_pages_total + 1,
        num_pages_per_req,
        device=k.device,
        dtype=torch.int32,
    )
    indices = torch.arange(num_pages_total, device=k.device, dtype=torch.int32)
    final_page_len = kv_len - (num_pages_per_req - 1) * page_size
    last_page_len = torch.full(
        (batch,),
        final_page_len,
        device=k.device,
        dtype=torch.int32,
    )
    return cache, indptr, indices, last_page_len


def _make_jit_args(
    *,
    backend: str,
    dtype: torch.dtype,
    head_dim: int,
    mode: str,
    uri_suffix: str,
) -> List[Any]:
    if backend not in {"fa2", "fa3"}:
        raise ValueError("custom JIT smoke requires backend fa2 or fa3")
    uri = (
        f"streamattn_head_mode_{mode}_{backend}_"
        f"{str(dtype).replace('torch.', '')}_d{head_dim}_{uri_suffix}"
    )
    tensor_names = ["head_modes", *DEFAULT_PREFILL_TENSOR_NAMES]
    tensor_dtypes = ["int32_t", *DEFAULT_PREFILL_TENSOR_DTYPES]
    scalar_names = list(DEFAULT_PREFILL_SCALAR_NAMES)
    scalar_dtypes = list(DEFAULT_PREFILL_SCALAR_DTYPES)
    if mode == "exact_equiv":
        variant_name = "DefaultAttention<use_custom_mask, false, false, false>"
        variant_decl = "#include<flashinfer/attention/variants.cuh>"
    elif mode == "seed_only":
        scalar_names.extend(SEED_ONLY_SCALAR_NAMES)
        scalar_dtypes.extend(SEED_ONLY_SCALAR_DTYPES)
        variant_name = "StreamAttnHeadModeSeedOnly"
        variant_decl = HEAD_MODE_SEED_ONLY_VARIANT_DECL
    else:
        raise ValueError(f"unsupported mode: {mode}")
    return [
        uri,
        dtype,
        dtype,
        dtype,
        torch.int32,
        head_dim,
        head_dim,
        tensor_names,
        tensor_dtypes,
        scalar_names,
        scalar_dtypes,
        variant_name,
        variant_decl,
        0,  # pos_encoding_mode: NONE
        False,  # use_sliding_window
        False,  # use_logits_soft_cap
        False,  # use_fp16_qk_reduction
        False,  # fp8_enabled
    ]


def _block_order_id(block_order: str) -> int:
    if block_order == "sequential":
        return 0
    if block_order == "recent_first":
        return 1
    raise ValueError("--block-order must be sequential or recent_first")


def _custom_run_args(
    head_modes: torch.Tensor,
    *,
    head_dim: int,
    mode: str,
    block_size: int,
    sink_blocks: int,
    recent_blocks: int,
    middle_seed_blocks: int,
    block_order: str,
) -> List[Any]:
    sm_scale = 1.0 / math.sqrt(float(head_dim))
    run_args = [
        head_modes,
        None,  # maybe_prefix_len_ptr
        None,  # maybe_token_pos_in_items_ptr
        None,  # maybe_max_item_len_ptr
        None,  # maybe_k_cache_sf
        None,  # maybe_v_cache_sf
        0.0,  # logits_soft_cap, unused when use_logits_soft_cap=false
        sm_scale,
        1.0,  # rope_rcp_scale, unused with pos_encoding_mode=NONE
        1.0 / 10000.0,  # rope_rcp_theta, unused with pos_encoding_mode=NONE
        0,  # token_pos_in_items_len
    ]
    if mode == "seed_only":
        run_args.extend(
            [
                int(block_size),
                int(sink_blocks),
                int(recent_blocks),
                int(middle_seed_blocks),
                _block_order_id(block_order),
            ]
        )
    return run_args


def _seed_keep_mask(
    *,
    kv_len: int,
    block_size: int,
    sink_blocks: int,
    recent_blocks: int,
    middle_seed_blocks: int,
    block_order: str,
    device: torch.device,
) -> torch.Tensor:
    positions = torch.arange(kv_len, device=device)
    num_blocks = math.ceil(kv_len / block_size)
    sink_end = min(sink_blocks * block_size, kv_len)
    recent_start = 0 if recent_blocks >= num_blocks else (num_blocks - recent_blocks) * block_size
    keep = (positions < sink_end) | (positions >= recent_start)
    if middle_seed_blocks > 0:
        middle_seed_tokens = middle_seed_blocks * block_size
        if block_order == "sequential":
            middle_start = sink_end
            middle_end = min(middle_start + middle_seed_tokens, recent_start)
        else:
            middle_end = recent_start
            middle_start = max(sink_end, middle_end - middle_seed_tokens)
        keep = keep | ((positions >= middle_start) & (positions < middle_end))
    return keep


def _torch_head_mode_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    head_modes: torch.Tensor,
    *,
    block_size: int,
    sink_blocks: int,
    recent_blocks: int,
    middle_seed_blocks: int,
    block_order: str,
) -> torch.Tensor:
    batch, q_heads, dim = q.shape
    kv_len = k.shape[1]
    kv_heads = k.shape[2]
    group_size = q_heads // kv_heads
    scale = 1.0 / math.sqrt(float(dim))
    seed_mask = _seed_keep_mask(
        kv_len=kv_len,
        block_size=block_size,
        sink_blocks=sink_blocks,
        recent_blocks=recent_blocks,
        middle_seed_blocks=middle_seed_blocks,
        block_order=block_order,
        device=q.device,
    )
    out = torch.empty_like(q)
    for batch_idx in range(batch):
        for q_head in range(q_heads):
            kv_head = q_head // group_size
            logits = torch.matmul(
                k[batch_idx, :, kv_head, :].float(),
                q[batch_idx, q_head, :].float(),
            ) * scale
            if int(head_modes[q_head].item()) != 0:
                logits = logits.masked_fill(~seed_mask, -float("inf"))
            probs = torch.softmax(logits, dim=0)
            out[batch_idx, q_head, :] = torch.matmul(
                probs.to(v.dtype),
                v[batch_idx, :, kv_head, :],
            )
    return out


def _make_wrapper(
    *,
    workspace: torch.Tensor,
    backend: str,
    use_tensor_cores: bool,
    jit_args: List[Any] | None,
):
    if not HAS_FLASHINFER:
        raise RuntimeError("FlashInfer is not available")
    return flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        use_tensor_cores=use_tensor_cores,
        backend=backend,
        jit_args=jit_args,
    )


def _plan_wrapper(
    wrapper,
    *,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    last_page_len: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    page_size: int,
    dtype: torch.dtype,
    disable_split_kv: bool,
) -> None:
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        q_heads,
        kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        q_data_type=dtype,
        kv_data_type=dtype,
        o_data_type=dtype,
        sm_scale=1.0 / math.sqrt(float(head_dim)),
        disable_split_kv=disable_split_kv,
    )


def profile(args: argparse.Namespace) -> Dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not HAS_FLASHINFER:
        raise RuntimeError("FlashInfer is required for this benchmark")

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    dtype = _dtype(args.dtype)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    q = torch.randn(args.batch_size, args.q_heads, args.head_dim, device=device, dtype=dtype)
    k = torch.randn(args.batch_size, args.kv_len, args.kv_heads, args.head_dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    kv_cache, indptr, indices, last_page_len = _make_paged_kv_cache(k, v, page_size=args.page_size)
    head_modes = torch.zeros(args.q_heads, device=device, dtype=torch.int32)
    if args.seed_heads:
        for item in args.seed_heads.split(","):
            item = item.strip()
            if item:
                head_modes[int(item)] = 1

    default_workspace = torch.empty(args.workspace_mb * 1024 * 1024, dtype=torch.uint8, device=device)
    custom_workspace = torch.empty_like(default_workspace)
    jit_args = _make_jit_args(
        backend=args.backend,
        dtype=dtype,
        head_dim=args.head_dim,
        mode=args.mode,
        uri_suffix=args.uri_suffix,
    )

    default_build_start = time.perf_counter()
    default_wrapper = _make_wrapper(
        workspace=default_workspace,
        backend=args.backend,
        use_tensor_cores=args.use_tensor_cores,
        jit_args=None,
    )
    _plan_wrapper(
        default_wrapper,
        indptr=indptr,
        indices=indices,
        last_page_len=last_page_len,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        page_size=args.page_size,
        dtype=dtype,
        disable_split_kv=args.disable_split_kv,
    )
    default_build_plan_ms = (time.perf_counter() - default_build_start) * 1000.0

    custom_build_start = time.perf_counter()
    custom_wrapper = _make_wrapper(
        workspace=custom_workspace,
        backend=args.backend,
        use_tensor_cores=args.use_tensor_cores,
        jit_args=jit_args,
    )
    _plan_wrapper(
        custom_wrapper,
        indptr=indptr,
        indices=indices,
        last_page_len=last_page_len,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        page_size=args.page_size,
        dtype=dtype,
        disable_split_kv=args.disable_split_kv,
    )
    custom_build_plan_ms = (time.perf_counter() - custom_build_start) * 1000.0

    run_args = _custom_run_args(
        head_modes,
        head_dim=args.head_dim,
        mode=args.mode,
        block_size=args.block_size,
        sink_blocks=args.sink_blocks,
        recent_blocks=args.recent_blocks,
        middle_seed_blocks=args.middle_seed_blocks,
        block_order=args.block_order,
    )

    def default_run() -> torch.Tensor:
        return default_wrapper.run(q, kv_cache)

    def custom_run() -> torch.Tensor:
        return custom_wrapper.run(q, kv_cache, *run_args)

    default_out = default_run()
    custom_out = custom_run()
    reference_out = _torch_head_mode_reference(
        q,
        k,
        v,
        head_modes if args.mode == "seed_only" else torch.zeros_like(head_modes),
        block_size=args.block_size,
        sink_blocks=args.sink_blocks,
        recent_blocks=args.recent_blocks,
        middle_seed_blocks=args.middle_seed_blocks,
        block_order=args.block_order,
    )
    torch.cuda.synchronize(device) if device.type == "cuda" else None

    default_ms = _time_cuda(default_run, device=device, warmup=args.warmup, iters=args.iters)
    custom_ms = _time_cuda(custom_run, device=device, warmup=args.warmup, iters=args.iters)
    reference_error = _error(
        custom_out.view(args.batch_size, 1, args.q_heads, args.head_dim),
        reference_out.view(args.batch_size, 1, args.q_heads, args.head_dim),
    )
    approximation_error = _error(
        custom_out.view(args.batch_size, 1, args.q_heads, args.head_dim),
        default_out.view(args.batch_size, 1, args.q_heads, args.head_dim),
    )

    reference_ok = reference_error["max_abs_error"] <= args.reference_error_budget
    overhead_ok = custom_ms <= default_ms * args.max_overhead_ratio
    scheduler_skip_observed = args.mode == "seed_only" and custom_ms < default_ms * args.scheduler_speedup_gate
    if args.mode == "seed_only" and reference_ok and not scheduler_skip_observed:
        recommended_next_path = "scheduler_level_head_mode_backend_required"
    elif reference_ok and overhead_ok:
        recommended_next_path = "custom_jit_attach_point_viable"
    else:
        recommended_next_path = "custom_jit_attach_point_failed"

    return {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "flashinfer": getattr(flashinfer, "__version__", None),
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "cuda_capability": torch.cuda.get_device_capability(device) if device.type == "cuda" else None,
        },
        "shape": {
            "batch_size": args.batch_size,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "group_size": args.q_heads // args.kv_heads,
            "head_dim": args.head_dim,
            "kv_len": args.kv_len,
            "page_size": args.page_size,
            "dtype": args.dtype,
            "head_modes_nonzero": int((head_modes != 0).sum().item()),
            "block_size": args.block_size,
            "sink_blocks": args.sink_blocks,
            "recent_blocks": args.recent_blocks,
            "middle_seed_blocks": args.middle_seed_blocks,
            "block_order": args.block_order,
        },
        "backend": {
            "backend": args.backend,
            "mode": args.mode,
            "use_tensor_cores": args.use_tensor_cores,
            "disable_split_kv": args.disable_split_kv,
            "jit_uri": jit_args[0],
            "additional_tensor_names": jit_args[7],
            "additional_scalar_names": jit_args[9],
        },
        "timing": {
            "default_build_plan_ms": default_build_plan_ms,
            "custom_build_plan_ms": custom_build_plan_ms,
            "default_ms": default_ms,
            "custom_exact_equiv_ms": custom_ms,
            "custom_over_default": custom_ms / default_ms if default_ms else None,
            "custom_minus_default_ms": custom_ms - default_ms,
            "custom_speedup_vs_default": default_ms / custom_ms if custom_ms else None,
        },
        "quality": {
            "custom_vs_reference": reference_error,
            "custom_vs_default": approximation_error,
            "max_abs_custom_default": float((custom_out - default_out).abs().float().max().item()),
        },
        "decision": {
            "custom_jit_passed": bool(torch.isfinite(custom_out.float()).all().item()),
            "math_attach_point_viable": bool(reference_ok and overhead_ok),
            "scheduler_skip_observed": bool(scheduler_skip_observed),
            "recommended_next_path": recommended_next_path,
            "max_overhead_ratio": args.max_overhead_ratio,
            "scheduler_speedup_gate": args.scheduler_speedup_gate,
            "reference_error_budget": args.reference_error_budget,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=14)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--kv-len", type=int, default=32768)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--workspace-mb", type=int, default=128)
    parser.add_argument("--backend", choices=["fa2", "fa3"], default="fa2")
    parser.add_argument("--mode", choices=["exact_equiv", "seed_only"], default="exact_equiv")
    parser.add_argument("--use-tensor-cores", action="store_true")
    parser.add_argument("--disable-split-kv", action="store_true")
    parser.add_argument("--seed-heads", default="2,3,4,6,7")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--recent-blocks", type=int, default=2)
    parser.add_argument("--middle-seed-blocks", type=int, default=2)
    parser.add_argument("--block-order", choices=["sequential", "recent_first"], default="recent_first")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--max-overhead-ratio", type=float, default=1.15)
    parser.add_argument("--scheduler-speedup-gate", type=float, default=0.95)
    parser.add_argument("--reference-error-budget", type=float, default=1e-2)
    parser.add_argument("--uri-suffix", default="v1")
    parser.add_argument("--summary-json-out", default="")
    args = parser.parse_args()
    if args.q_heads % args.kv_heads != 0:
        raise ValueError("--q-heads must be divisible by --kv-heads for true GQA")

    result = profile(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.summary_json_out:
        path = Path(args.summary_json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
