"""ThunderKittens scheduler-level head-mode decode spike.

This is the first attention-shaped TK prototype after the FlashInfer custom JIT
and scalar CUDA probes.  It is intentionally a scheduler/correctness spike, not
a performance claim:

* one CUDA launch processes true-GQA Q heads;
* ``EXACT_FULL`` heads load/score every KV token;
* ``SEED_ONLY`` heads branch before K/V loads and only load/score seed ranges;
* per-head counters prove how many KV tokens were scheduled.

The exact branch is still scalar.  If this passes, the next work is replacing
that scalar exact path with a TK tensor-core/WGMMA path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_gate0_seed_only_true_gqa import _parse_heads  # noqa: E402
from benchmarks.profile_gate0_true_gqa import _dense_true_gqa  # noqa: E402
from benchmarks.profile_head_mode_decode_cuda import (  # noqa: E402
    _block_order_id,
    _flashinfer_exact,
    _make_inputs,
    _seed_keep_mask,
    _torch_head_mode_reference,
)
from benchmarks.profile_stream_attn_gate0_wrapper import _dtype, _error, _time_cuda  # noqa: E402
from benchmarks.profile_thunderkittens_extension_smoke import (  # noqa: E402
    _clone_tk,
    _find_tk_root,
    _tk_arch_define,
)


CPP_SOURCE = r"""
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> streamattn_tk_head_mode_decode_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor head_modes,
    int64_t block_size,
    int64_t sink_blocks,
    int64_t recent_blocks,
    int64_t middle_seed_blocks,
    int64_t block_order,
    int64_t threads);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("head_mode_decode", &streamattn_tk_head_mode_decode_cuda,
        "StreamAttn ThunderKittens scheduler-level true-GQA head-mode decode");
}
"""


CUDA_SOURCE = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>
#include "kittens.cuh"

constexpr int MAX_D = 128;

template <typename scalar_t>
__device__ __forceinline__ float streamattn_load_as_float(const scalar_t* ptr, int64_t offset) {
  return static_cast<float>(ptr[offset]);
}

__device__ __forceinline__ bool streamattn_tk_keep_token(
    int n,
    int N,
    int block_size,
    int sink_blocks,
    int recent_blocks,
    int middle_seed_blocks,
    int block_order) {
  const int bs = block_size <= 0 ? 1 : block_size;
  const int num_blocks = (N + bs - 1) / bs;
  const int sink_end = min(sink_blocks * bs, N);
  const int recent_start = recent_blocks >= num_blocks ? 0 : (num_blocks - recent_blocks) * bs;
  bool keep = (n < sink_end) || (n >= recent_start);
  if (middle_seed_blocks > 0) {
    const int middle_seed_tokens = middle_seed_blocks * bs;
    if (block_order == 0) {
      const int middle_start = sink_end;
      const int middle_end = min(middle_start + middle_seed_tokens, recent_start);
      keep = keep || (n >= middle_start && n < middle_end);
    } else {
      const int middle_end = recent_start;
      const int middle_start = max(sink_end, middle_end - middle_seed_tokens);
      keep = keep || (n >= middle_start && n < middle_end);
    }
  }
  return keep;
}

template <typename scalar_t>
__global__ void streamattn_tk_head_mode_decode_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int32_t* __restrict__ head_modes,
    scalar_t* __restrict__ out,
    int64_t* __restrict__ stats,
    int B,
    int Hq,
    int Hkv,
    int N,
    int D,
    int block_size,
    int sink_blocks,
    int recent_blocks,
    int middle_seed_blocks,
    int block_order,
    float scale) {
  const int bh = blockIdx.x;
  const int b = bh / Hq;
  const int h = bh - b * Hq;
  const int tid = threadIdx.x;
  const int group_size = Hq / Hkv;
  const int kvh = h / group_size;
  const bool seed_only = head_modes[h] != 0;

  extern __shared__ float smem[];
  float* s_reduce = smem;
  float* s_acc = smem + blockDim.x;

  float q_vec[MAX_D];
  #pragma unroll
  for (int d = 0; d < MAX_D; ++d) {
    q_vec[d] = 0.0f;
  }
  for (int d = 0; d < D; ++d) {
    q_vec[d] = streamattn_load_as_float(q, (b * Hq + h) * D + d);
  }

  float local_max = -INFINITY;
  for (int n = tid; n < N; n += blockDim.x) {
    if (seed_only && !streamattn_tk_keep_token(
          n, N, block_size, sink_blocks, recent_blocks, middle_seed_blocks, block_order)) {
      continue;
    }
    float dot = 0.0f;
    const int64_t base = ((int64_t)b * N * Hkv + (int64_t)n * Hkv + kvh) * D;
    for (int d = 0; d < D; ++d) {
      dot += q_vec[d] * streamattn_load_as_float(k, base + d);
    }
    local_max = fmaxf(local_max, dot * scale);
  }
  s_reduce[tid] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_reduce[tid] = fmaxf(s_reduce[tid], s_reduce[tid + stride]);
    }
    __syncthreads();
  }
  const float max_score = s_reduce[0];

  float acc[MAX_D];
  #pragma unroll
  for (int d = 0; d < MAX_D; ++d) {
    acc[d] = 0.0f;
  }
  float local_den = 0.0f;
  if (isfinite(max_score)) {
    for (int n = tid; n < N; n += blockDim.x) {
      if (seed_only && !streamattn_tk_keep_token(
            n, N, block_size, sink_blocks, recent_blocks, middle_seed_blocks, block_order)) {
        continue;
      }
      float dot = 0.0f;
      const int64_t base = ((int64_t)b * N * Hkv + (int64_t)n * Hkv + kvh) * D;
      for (int d = 0; d < D; ++d) {
        dot += q_vec[d] * streamattn_load_as_float(k, base + d);
      }
      const float p = expf(dot * scale - max_score);
      local_den += p;
      for (int d = 0; d < D; ++d) {
        acc[d] += p * streamattn_load_as_float(v, base + d);
      }
    }
  }

  s_reduce[tid] = local_den;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_reduce[tid] += s_reduce[tid + stride];
    }
    __syncthreads();
  }
  const float den = s_reduce[0];

  for (int d = 0; d < D; ++d) {
    s_acc[d * blockDim.x + tid] = acc[d];
  }
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      for (int d = 0; d < D; ++d) {
        s_acc[d * blockDim.x + tid] += s_acc[d * blockDim.x + tid + stride];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    int scheduled = 0;
    if (seed_only) {
      for (int n = 0; n < N; ++n) {
        scheduled += streamattn_tk_keep_token(
          n, N, block_size, sink_blocks, recent_blocks, middle_seed_blocks, block_order) ? 1 : 0;
      }
    } else {
      scheduled = N;
    }

    for (int d = 0; d < D; ++d) {
      const float value = den > 0.0f ? s_acc[d * blockDim.x] / den : 0.0f;
      out[(b * Hq + h) * D + d] = static_cast<scalar_t>(value);
    }

    // [mode, kv_head, scheduled_tokens, skipped_tokens, qk_token_passes, tk_reached]
    const int64_t stats_base = ((int64_t)b * Hq + h) * 6;
    stats[stats_base + 0] = seed_only ? 1 : 0;
    stats[stats_base + 1] = kvh;
    stats[stats_base + 2] = scheduled;
    stats[stats_base + 3] = N - scheduled;
    stats[stats_base + 4] = 2 * scheduled;  // two-pass scalar online-softmax reference path.
    stats[stats_base + 5] = (::kittens::laneid() == 0 && ::kittens::warpid() == 0) ? 1 : 0;
  }
}

std::vector<torch::Tensor> streamattn_tk_head_mode_decode_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor head_modes,
    int64_t block_size,
    int64_t sink_blocks,
    int64_t recent_blocks,
    int64_t middle_seed_blocks,
    int64_t block_order,
    int64_t threads) {
  TORCH_CHECK(q.is_cuda(), "q must be CUDA");
  TORCH_CHECK(k.is_cuda(), "k must be CUDA");
  TORCH_CHECK(v.is_cuda(), "v must be CUDA");
  TORCH_CHECK(head_modes.is_cuda(), "head_modes must be CUDA");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous [B,Hq,D]");
  TORCH_CHECK(k.is_contiguous(), "k must be contiguous [B,N,Hkv,D]");
  TORCH_CHECK(v.is_contiguous(), "v must be contiguous [B,N,Hkv,D]");
  TORCH_CHECK(head_modes.is_contiguous(), "head_modes must be contiguous [Hq]");
  TORCH_CHECK(q.dim() == 3, "q must have shape [B,Hq,D]");
  TORCH_CHECK(k.dim() == 4, "k must have shape [B,N,Hkv,D]");
  TORCH_CHECK(v.sizes() == k.sizes(), "v must match k shape");
  TORCH_CHECK(head_modes.scalar_type() == at::ScalarType::Int, "head_modes must be int32");
  const int B = q.size(0);
  const int Hq = q.size(1);
  const int D = q.size(2);
  const int N = k.size(1);
  const int Hkv = k.size(2);
  TORCH_CHECK(D <= MAX_D, "head_dim exceeds MAX_D=128");
  TORCH_CHECK(k.size(0) == B && k.size(3) == D, "K shape incompatible with Q");
  TORCH_CHECK(Hq % Hkv == 0, "Hq must be divisible by Hkv");
  TORCH_CHECK(head_modes.size(0) == Hq, "head_modes length must equal Hq");
  TORCH_CHECK(threads > 0 && (threads & (threads - 1)) == 0, "threads must be power of two");

  auto out = torch::empty_like(q);
  auto stats = torch::empty({B, Hq, 6}, q.options().dtype(torch::kInt64));
  const dim3 grid(B * Hq);
  const dim3 block(threads);
  const size_t shared_bytes = (threads + D * threads) * sizeof(float);
  const float scale = 1.0f / sqrtf(static_cast<float>(D));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      q.scalar_type(),
      "streamattn_tk_head_mode_decode_cuda",
      [&] {
        auto kernel = streamattn_tk_head_mode_decode_kernel<scalar_t>;
        cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes));
        kernel<<<grid, block, shared_bytes>>>(
            q.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            head_modes.data_ptr<int32_t>(),
            out.data_ptr<scalar_t>(),
            stats.data_ptr<int64_t>(),
            B, Hq, Hkv, N, D,
            static_cast<int>(block_size),
            static_cast<int>(sink_blocks),
            static_cast<int>(recent_blocks),
            static_cast<int>(middle_seed_blocks),
            static_cast<int>(block_order),
            scale);
      });
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
  return {out, stats};
}
"""


def _compile_extension(
    *,
    tk_root: Path,
    cuda_arch: str,
    torch_cuda_arch_list: str,
    verbose: bool = False,
):
    from torch.utils.cpp_extension import load_inline

    previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = torch_cuda_arch_list
    try:
        build_dir = tempfile.mkdtemp(prefix="streamattn_tk_head_mode_decode_")
        return load_inline(
            name="streamattn_tk_head_mode_decode",
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            build_directory=build_dir,
            verbose=verbose,
            with_cuda=True,
            extra_include_paths=[str(tk_root / "include")],
            extra_cflags=["-std=c++20"],
            extra_cuda_cflags=[
                "-std=c++20",
                "-O3",
                "--use_fast_math",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                f"-D{_tk_arch_define(cuda_arch)}",
            ],
        )
    finally:
        if previous_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch


def _seed_counts(
    kv_len: int,
    *,
    block_size: int,
    sink_blocks: int,
    recent_blocks: int,
    middle_seed_blocks: int,
    block_order: str,
    device: torch.device,
) -> dict[str, int]:
    mask = _seed_keep_mask(
        kv_len,
        block_size=block_size,
        sink_blocks=sink_blocks,
        recent_blocks=recent_blocks,
        middle_seed_blocks=middle_seed_blocks,
        block_order=block_order,
        device=device,
    )
    blocks = mask.view(-1, block_size) if kv_len % block_size == 0 else None
    if blocks is None:
        block_count = 0
        for start in range(0, kv_len, block_size):
            if bool(mask[start : start + block_size].any().item()):
                block_count += 1
    else:
        block_count = int(blocks.any(dim=1).sum().item())
    token_count = int(mask.sum().item())
    return {
        "seed_tokens": token_count,
        "seed_blocks": block_count,
        "skipped_tokens": kv_len - token_count,
        "total_blocks": (kv_len + block_size - 1) // block_size,
    }


def _stats_summary(stats: torch.Tensor, *, seed_heads: list[int], kv_len: int) -> dict[str, Any]:
    cpu = stats.detach().cpu()
    rows = []
    for batch_idx in range(cpu.shape[0]):
        for head in range(cpu.shape[1]):
            rows.append(
                {
                    "batch": batch_idx,
                    "head": head,
                    "mode": int(cpu[batch_idx, head, 0].item()),
                    "kv_head": int(cpu[batch_idx, head, 1].item()),
                    "scheduled_tokens": int(cpu[batch_idx, head, 2].item()),
                    "skipped_tokens": int(cpu[batch_idx, head, 3].item()),
                    "qk_token_passes": int(cpu[batch_idx, head, 4].item()),
                    "tk_device_helper_reached": bool(cpu[batch_idx, head, 5].item()),
                }
            )
    seed_set = set(seed_heads)
    seed_rows = [row for row in rows if row["head"] in seed_set]
    exact_rows = [row for row in rows if row["head"] not in seed_set]
    return {
        "per_head": rows,
        "seed_scheduled_tokens_max": max((row["scheduled_tokens"] for row in seed_rows), default=0),
        "seed_skipped_tokens_min": min((row["skipped_tokens"] for row in seed_rows), default=0),
        "exact_scheduled_tokens_min": min((row["scheduled_tokens"] for row in exact_rows), default=0),
        "exact_skipped_tokens_max": max((row["skipped_tokens"] for row in exact_rows), default=0),
        "all_tk_helpers_reached": all(row["tk_device_helper_reached"] for row in rows),
        "all_exact_heads_full_range": all(row["scheduled_tokens"] == kv_len for row in exact_rows),
    }


def profile(args: argparse.Namespace) -> Dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    dtype = _dtype(args.dtype)

    tk_root = _find_tk_root(args.thunderkittens_root)
    clone_result = {"requested": False}
    if tk_root is None and args.clone_thunderkittens:
        clone_result = _clone_tk(Path(args.checkout_dir).expanduser() / "ThunderKittens")
        tk_root = _find_tk_root(str(Path(args.checkout_dir).expanduser() / "ThunderKittens"))
    if tk_root is None:
        raise RuntimeError("ThunderKittens root not found")

    q, k, v = _make_inputs(args, device, dtype)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    seed_heads = _parse_heads(args.seed_heads)
    head_modes = torch.zeros(q.shape[1], device=device, dtype=torch.int32)
    if seed_heads:
        head_modes[torch.tensor(seed_heads, device=device, dtype=torch.long)] = 1

    compile_start = time.perf_counter()
    module = _compile_extension(
        tk_root=tk_root,
        cuda_arch=args.cuda_arch,
        torch_cuda_arch_list=args.torch_cuda_arch_list,
        verbose=args.verbose_compile,
    )
    compile_ms = (time.perf_counter() - compile_start) * 1000.0

    def tk_mixed():
        return module.head_mode_decode(
            q,
            k,
            v,
            head_modes,
            args.block_size,
            args.sink_blocks,
            args.recent_blocks,
            args.middle_seed_blocks,
            _block_order_id(args.block_order),
            args.threads,
        )

    def torch_dense() -> torch.Tensor:
        return _dense_true_gqa(q[:, None, :, :], k, v)[:, 0]

    def flashinfer_exact() -> torch.Tensor:
        return _flashinfer_exact(q, k, v, use_tensor_cores=args.flashinfer_tensor_cores)

    ref = _torch_head_mode_reference(
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
    dense_ref = torch_dense()
    tk_out, tk_stats = tk_mixed()

    tk_ms = _time_cuda(lambda: tk_mixed()[0], device=device, warmup=args.warmup, iters=args.iters)
    torch_dense_ms = _time_cuda(torch_dense, device=device, warmup=args.warmup, iters=args.iters)
    flashinfer_ms = None
    flashinfer_error = None
    flashinfer_failure = None
    try:
        flashinfer_ms = _time_cuda(
            flashinfer_exact,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
        flashinfer_error = _error(flashinfer_exact()[:, None, :, :], dense_ref[:, None, :, :])
    except Exception as exc:
        flashinfer_failure = f"{type(exc).__name__}: {exc}"

    seed_counts = _seed_counts(
        int(k.shape[1]),
        block_size=args.block_size,
        sink_blocks=args.sink_blocks,
        recent_blocks=args.recent_blocks,
        middle_seed_blocks=args.middle_seed_blocks,
        block_order=args.block_order,
        device=device,
    )
    stats_summary = _stats_summary(tk_stats, seed_heads=seed_heads, kv_len=int(k.shape[1]))
    scheduler_counters_correct = (
        stats_summary["all_exact_heads_full_range"]
        and stats_summary["exact_skipped_tokens_max"] == 0
        and stats_summary["seed_scheduled_tokens_max"] == seed_counts["seed_tokens"]
        and stats_summary["seed_skipped_tokens_min"] == seed_counts["skipped_tokens"]
        and stats_summary["all_tk_helpers_reached"]
    )
    error_vs_ref = _error(tk_out[:, None, :, :], ref[:, None, :, :])
    error_vs_dense = _error(tk_out[:, None, :, :], dense_ref[:, None, :, :])
    scheduler_correct = error_vs_ref["max_abs_error"] <= args.error_budget
    if scheduler_correct and scheduler_counters_correct:
        next_path = "replace_scalar_exact_with_thunderkittens_tensor_core_path"
    elif scheduler_correct:
        next_path = "fix_scheduler_counters_before_tensor_core_work"
    else:
        next_path = "fix_tk_head_mode_correctness"

    return {
        "schema": "streamattn.tk_head_mode_decode_spike.v1",
        "tk_root": str(tk_root),
        "clone_result": clone_result,
        "compile_ms": compile_ms,
        "shape": {
            "batch": int(q.shape[0]),
            "q_heads": int(q.shape[1]),
            "kv_heads": int(k.shape[2]),
            "kv_len": int(k.shape[1]),
            "head_dim": int(q.shape[2]),
            "dtype": str(dtype),
        },
        "policy": {
            "seed_heads": seed_heads,
            "exact_heads": [head for head in range(q.shape[1]) if head not in set(seed_heads)],
            "block_size": args.block_size,
            "sink_blocks": args.sink_blocks,
            "recent_blocks": args.recent_blocks,
            "middle_seed_blocks": args.middle_seed_blocks,
            "block_order": args.block_order,
        },
        "timings_ms": {
            "tk_head_mode_ms": tk_ms,
            "torch_dense_ms": torch_dense_ms,
            "flashinfer_exact_ms": flashinfer_ms,
        },
        "errors": {
            "tk_vs_head_mode_reference": error_vs_ref,
            "tk_vs_dense": error_vs_dense,
            "flashinfer_vs_dense": flashinfer_error,
            "flashinfer_failure": flashinfer_failure,
        },
        "scheduler": {
            "seed_counts": seed_counts,
            "stats_summary": stats_summary,
            "scheduler_correct": scheduler_correct,
            "scheduler_counters_correct": scheduler_counters_correct,
            "seed_only_heads_skip_before_qk": scheduler_counters_correct,
        },
        "recommended_next_path": next_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--q-path", default="")
    parser.add_argument("--k-path", default="")
    parser.add_argument("--v-path", default="")
    parser.add_argument("--true-kv-heads", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--kv-len", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=14)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--seed-heads", default="2,3,4,6,7")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--recent-blocks", type=int, default=2)
    parser.add_argument("--middle-seed-blocks", type=int, default=2)
    parser.add_argument("--block-order", default="recent_first", choices=["sequential", "recent_first"])
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--error-budget", type=float, default=2e-3)
    parser.add_argument("--flashinfer-tensor-cores", action="store_true")
    parser.add_argument("--thunderkittens-root", default="")
    parser.add_argument("--checkout-dir", default="artifacts/backend_sources")
    parser.add_argument("--clone-thunderkittens", action="store_true")
    parser.add_argument("--cuda-arch", default="sm_90a")
    parser.add_argument("--torch-cuda-arch-list", default="9.0a")
    parser.add_argument("--verbose-compile", action="store_true")
    parser.add_argument("--summary-json-out", default="")
    args = parser.parse_args()
    result = profile(args)
    text = json.dumps(result, indent=2, sort_keys=True, default=str)
    if args.summary_json_out:
        path = Path(args.summary_json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
