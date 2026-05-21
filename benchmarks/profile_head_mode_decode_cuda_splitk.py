"""Split-K CUDA scheduler-level head-mode decode prototype.

``profile_head_mode_decode_cuda.py`` proved the branch point is correct but its
one-block-per-head exact path is far too serial.  This benchmark moves the
same scheduler-level idea to split-K:

* one chunk kernel over ``batch x q_head x chunk``;
* seed-only heads skip non-seed tokens before K/V work inside each chunk;
* one merge kernel combines online-softmax partial states.

This is still scalar CUDA, not a dense-quality tensor-core backend.  The goal is
to measure how much of the 85x scalar gap is scheduling parallelism versus the
need for a CuTe/ThunderKittens-style exact path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_head_mode_decode_cuda import (  # noqa: E402
    _block_order_id,
    _flashinfer_exact,
    _make_inputs,
    _torch_head_mode_reference,
)
from benchmarks.profile_gate0_true_gqa import _dense_true_gqa  # noqa: E402
from benchmarks.profile_stream_attn_gate0_wrapper import _dtype, _error, _time_cuda  # noqa: E402
from benchmarks.profile_gate0_seed_only_true_gqa import _parse_heads  # noqa: E402


CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor streamattn_head_mode_splitk_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor head_modes,
    int64_t num_chunks,
    int64_t block_size,
    int64_t sink_blocks,
    int64_t recent_blocks,
    int64_t middle_seed_blocks,
    int64_t block_order,
    int64_t threads);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("head_mode_splitk", &streamattn_head_mode_splitk_cuda,
        "StreamAttn split-K scheduler-level true-GQA head-mode decode");
}
"""


CUDA_SOURCE = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cmath>

constexpr int MAX_D = 128;

template <typename scalar_t>
__device__ __forceinline__ float load_as_float(const scalar_t* ptr, int64_t offset) {
  return static_cast<float>(ptr[offset]);
}

__device__ __forceinline__ bool streamattn_keep_token_splitk(
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
__global__ void streamattn_splitk_chunk_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int32_t* __restrict__ head_modes,
    float* __restrict__ chunk_m,
    float* __restrict__ chunk_l,
    float* __restrict__ chunk_num,
    int B,
    int Hq,
    int Hkv,
    int N,
    int D,
    int C,
    int block_size,
    int sink_blocks,
    int recent_blocks,
    int middle_seed_blocks,
    int block_order,
    float scale) {
  const int pid = blockIdx.x;
  const int c = pid % C;
  const int h = (pid / C) % Hq;
  const int b = pid / (C * Hq);
  const int tid = threadIdx.x;
  const int group_size = Hq / Hkv;
  const int kvh = h / group_size;
  const bool seed_only = head_modes[h] != 0;
  const int chunk_start = (int)(((int64_t)N * c) / C);
  const int chunk_end = (int)(((int64_t)N * (c + 1)) / C);

  extern __shared__ float smem[];
  float* s_reduce = smem;
  float* s_acc = smem + blockDim.x;

  float q_vec[MAX_D];
  #pragma unroll
  for (int d = 0; d < MAX_D; ++d) {
    q_vec[d] = 0.0f;
  }
  for (int d = 0; d < D; ++d) {
    q_vec[d] = load_as_float(q, (b * Hq + h) * D + d);
  }

  float local_max = -INFINITY;
  for (int n = chunk_start + tid; n < chunk_end; n += blockDim.x) {
    if (seed_only && !streamattn_keep_token_splitk(
          n, N, block_size, sink_blocks, recent_blocks, middle_seed_blocks, block_order)) {
      continue;
    }
    float dot = 0.0f;
    const int64_t base = ((int64_t)b * N * Hkv + (int64_t)n * Hkv + kvh) * D;
    for (int d = 0; d < D; ++d) {
      dot += q_vec[d] * load_as_float(k, base + d);
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
    for (int n = chunk_start + tid; n < chunk_end; n += blockDim.x) {
      if (seed_only && !streamattn_keep_token_splitk(
            n, N, block_size, sink_blocks, recent_blocks, middle_seed_blocks, block_order)) {
        continue;
      }
      float dot = 0.0f;
      const int64_t base = ((int64_t)b * N * Hkv + (int64_t)n * Hkv + kvh) * D;
      for (int d = 0; d < D; ++d) {
        dot += q_vec[d] * load_as_float(k, base + d);
      }
      const float p = expf(dot * scale - max_score);
      local_den += p;
      for (int d = 0; d < D; ++d) {
        acc[d] += p * load_as_float(v, base + d);
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
    const int64_t state = ((int64_t)b * Hq + h) * C + c;
    chunk_m[state] = den > 0.0f ? max_score : -INFINITY;
    chunk_l[state] = den;
    for (int d = 0; d < D; ++d) {
      chunk_num[state * D + d] = den > 0.0f ? s_acc[d * blockDim.x] : 0.0f;
    }
  }
}

template <typename scalar_t>
__global__ void streamattn_splitk_merge_kernel(
    const float* __restrict__ chunk_m,
    const float* __restrict__ chunk_l,
    const float* __restrict__ chunk_num,
    scalar_t* __restrict__ out,
    int B,
    int Hq,
    int D,
    int C) {
  const int bh = blockIdx.x;
  const int b = bh / Hq;
  const int h = bh - b * Hq;
  const int tid = threadIdx.x;

  extern __shared__ float smem[];
  float* s_reduce = smem;
  float* s_acc = smem + blockDim.x;

  float local_max = -INFINITY;
  for (int c = tid; c < C; c += blockDim.x) {
    const int64_t state = ((int64_t)b * Hq + h) * C + c;
    const float den = chunk_l[state];
    if (den > 0.0f) {
      local_max = fmaxf(local_max, chunk_m[state]);
    }
  }
  s_reduce[tid] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_reduce[tid] = fmaxf(s_reduce[tid], s_reduce[tid + stride]);
    }
    __syncthreads();
  }
  const float global_max = s_reduce[0];

  float acc[MAX_D];
  #pragma unroll
  for (int d = 0; d < MAX_D; ++d) {
    acc[d] = 0.0f;
  }
  float local_den = 0.0f;
  if (isfinite(global_max)) {
    for (int c = tid; c < C; c += blockDim.x) {
      const int64_t state = ((int64_t)b * Hq + h) * C + c;
      const float den = chunk_l[state];
      if (den > 0.0f) {
        const float w = expf(chunk_m[state] - global_max);
        local_den += den * w;
        for (int d = 0; d < D; ++d) {
          acc[d] += chunk_num[state * D + d] * w;
        }
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
  const float total_den = s_reduce[0];

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
    for (int d = 0; d < D; ++d) {
      const float value = total_den > 0.0f ? s_acc[d * blockDim.x] / total_den : 0.0f;
      out[(b * Hq + h) * D + d] = static_cast<scalar_t>(value);
    }
  }
}

torch::Tensor streamattn_head_mode_splitk_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor head_modes,
    int64_t num_chunks,
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
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(q.dim() == 3 && k.dim() == 4 && v.dim() == 4, "expected q [B,Hq,D], k/v [B,N,Hkv,D]");
  TORCH_CHECK(v.sizes() == k.sizes(), "v must match k");
  const int B = q.size(0);
  const int Hq = q.size(1);
  const int D = q.size(2);
  const int N = k.size(1);
  const int Hkv = k.size(2);
  const int C = static_cast<int>(num_chunks);
  TORCH_CHECK(D <= MAX_D, "head_dim exceeds MAX_D=128");
  TORCH_CHECK(C > 0, "num_chunks must be positive");
  TORCH_CHECK(k.size(0) == B && k.size(3) == D, "K shape incompatible with Q");
  TORCH_CHECK(Hq % Hkv == 0, "Hq must be divisible by Hkv");
  TORCH_CHECK(head_modes.is_contiguous() && head_modes.size(0) == Hq, "head_modes must be contiguous [Hq]");
  TORCH_CHECK(threads > 0 && (threads & (threads - 1)) == 0, "threads must be power of two");

  auto opts_f = q.options().dtype(torch::kFloat32);
  auto chunk_m = torch::empty({B, Hq, C}, opts_f);
  auto chunk_l = torch::empty({B, Hq, C}, opts_f);
  auto chunk_num = torch::empty({B, Hq, C, D}, opts_f);
  auto out = torch::empty_like(q);
  const size_t shared_bytes = (threads + D * threads) * sizeof(float);
  const float scale = 1.0f / sqrtf(static_cast<float>(D));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "streamattn_head_mode_splitk_cuda", [&] {
    auto chunk_kernel = streamattn_splitk_chunk_kernel<scalar_t>;
    auto merge_kernel = streamattn_splitk_merge_kernel<scalar_t>;
    cudaFuncSetAttribute(
        chunk_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes));
    cudaFuncSetAttribute(
        merge_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes));
    chunk_kernel<<<B * Hq * C, threads, shared_bytes>>>(
        q.data_ptr<scalar_t>(),
        k.data_ptr<scalar_t>(),
        v.data_ptr<scalar_t>(),
        head_modes.data_ptr<int32_t>(),
        chunk_m.data_ptr<float>(),
        chunk_l.data_ptr<float>(),
        chunk_num.data_ptr<float>(),
        B, Hq, Hkv, N, D, C,
        static_cast<int>(block_size),
        static_cast<int>(sink_blocks),
        static_cast<int>(recent_blocks),
        static_cast<int>(middle_seed_blocks),
        static_cast<int>(block_order),
        scale);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
    merge_kernel<<<B * Hq, threads, shared_bytes>>>(
        chunk_m.data_ptr<float>(),
        chunk_l.data_ptr<float>(),
        chunk_num.data_ptr<float>(),
        out.data_ptr<scalar_t>(),
        B, Hq, D, C);
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
  });
  return out;
}
"""


def _compile_extension(*, verbose: bool = False):
    from torch.utils.cpp_extension import load_inline

    previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    try:
        build_dir = tempfile.mkdtemp(prefix="streamattn_head_mode_splitk_cuda_")
        return load_inline(
            name="streamattn_head_mode_splitk_cuda",
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            build_directory=build_dir,
            verbose=verbose,
            with_cuda=True,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
    finally:
        if previous_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch


def profile(args: argparse.Namespace) -> Dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    dtype = _dtype(args.dtype)
    q, k, v = _make_inputs(args, device, dtype)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    seed_heads = _parse_heads(args.seed_heads)
    head_modes = torch.zeros(q.shape[1], device=device, dtype=torch.int32)
    if seed_heads:
        head_modes[torch.tensor(seed_heads, device=device, dtype=torch.long)] = 1
    all_exact_modes = torch.zeros_like(head_modes)
    all_seed_modes = torch.ones_like(head_modes)

    compile_start = time.perf_counter()
    module = _compile_extension(verbose=args.verbose_compile)
    compile_ms = (time.perf_counter() - compile_start) * 1000.0
    order_id = _block_order_id(args.block_order)

    def splitk_head_mode() -> torch.Tensor:
        return module.head_mode_splitk(
            q,
            k,
            v,
            head_modes,
            args.num_chunks,
            args.block_size,
            args.sink_blocks,
            args.recent_blocks,
            args.middle_seed_blocks,
            order_id,
            args.threads,
        )

    def splitk_all_exact() -> torch.Tensor:
        return module.head_mode_splitk(
            q,
            k,
            v,
            all_exact_modes,
            args.num_chunks,
            args.block_size,
            args.sink_blocks,
            args.recent_blocks,
            args.middle_seed_blocks,
            order_id,
            args.threads,
        )

    def splitk_all_seed() -> torch.Tensor:
        return module.head_mode_splitk(
            q,
            k,
            v,
            all_seed_modes,
            args.num_chunks,
            args.block_size,
            args.sink_blocks,
            args.recent_blocks,
            args.middle_seed_blocks,
            order_id,
            args.threads,
        )

    def torch_dense() -> torch.Tensor:
        return _dense_true_gqa(q[:, None, :, :], k, v)[:, 0]

    def flashinfer_exact() -> torch.Tensor:
        return _flashinfer_exact(q, k, v, use_tensor_cores=args.flashinfer_tensor_cores)

    dense_ref = torch_dense()
    head_mode_ref = _torch_head_mode_reference(
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
    splitk_head_out = splitk_head_mode()
    splitk_exact_out = splitk_all_exact()

    flashinfer_ms = None
    flashinfer_error = None
    flashinfer_available = True
    flashinfer_failure = None
    try:
        flashinfer_ms = _time_cuda(flashinfer_exact, device=device, warmup=args.warmup, iters=args.iters)
        flashinfer_error = _error(flashinfer_exact()[:, None, :, :], dense_ref[:, None, :, :])
    except Exception as exc:
        flashinfer_available = False
        flashinfer_failure = f"{type(exc).__name__}: {exc}"

    splitk_head_ms = _time_cuda(splitk_head_mode, device=device, warmup=args.warmup, iters=args.iters)
    splitk_exact_ms = _time_cuda(splitk_all_exact, device=device, warmup=args.warmup, iters=args.iters)
    splitk_seed_ms = _time_cuda(splitk_all_seed, device=device, warmup=args.warmup, iters=args.iters)
    torch_dense_ms = _time_cuda(torch_dense, device=device, warmup=args.warmup, iters=args.iters)

    scheduler_branch_error = _error(splitk_head_out[:, None, :, :], head_mode_ref[:, None, :, :])
    exact_error = _error(splitk_exact_out[:, None, :, :], dense_ref[:, None, :, :])
    reference_ms = flashinfer_ms if flashinfer_ms is not None else torch_dense_ms
    scheduler_branch_correct = scheduler_branch_error["max_abs_error"] <= args.error_budget
    exact_equiv_correct = exact_error["max_abs_error"] <= args.error_budget
    if scheduler_branch_correct and splitk_head_ms < reference_ms:
        next_path = "optimize_splitk_scheduler_backend"
    elif scheduler_branch_correct and splitk_head_ms < reference_ms * 3.0:
        next_path = "splitk_parallelism_helped_but_tensor_core_exact_path_needed"
    elif scheduler_branch_correct:
        next_path = "scalar_splitk_still_too_slow_jump_to_cute_or_thunderkittens"
    else:
        next_path = "fix_splitk_scheduler_correctness"

    return {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "cuda_capability": torch.cuda.get_device_capability(device) if device.type == "cuda" else None,
        },
        "shape": {
            "batch": int(q.shape[0]),
            "q_heads": int(q.shape[1]),
            "kv_heads": int(k.shape[2]),
            "group_size": int(q.shape[1] // k.shape[2]),
            "kv_len": int(k.shape[1]),
            "head_dim": int(q.shape[2]),
            "dtype": args.dtype,
        },
        "policy": {
            "seed_heads": seed_heads,
            "exact_heads": [head for head in range(int(q.shape[1])) if head not in set(seed_heads)],
            "num_chunks": args.num_chunks,
            "block_size": args.block_size,
            "sink_blocks": args.sink_blocks,
            "recent_blocks": args.recent_blocks,
            "middle_seed_blocks": args.middle_seed_blocks,
            "block_order": args.block_order,
        },
        "timing": {
            "compile_ms": compile_ms,
            "splitk_head_mode_ms": splitk_head_ms,
            "splitk_all_exact_ms": splitk_exact_ms,
            "splitk_all_seed_ms": splitk_seed_ms,
            "torch_dense_true_gqa_ms": torch_dense_ms,
            "flashinfer_exact_ms": flashinfer_ms,
            "splitk_head_mode_speedup_vs_flashinfer": (
                flashinfer_ms / splitk_head_ms if flashinfer_ms and splitk_head_ms else None
            ),
            "splitk_head_mode_speedup_vs_torch_dense": torch_dense_ms / splitk_head_ms if splitk_head_ms else None,
        },
        "quality": {
            "splitk_head_mode_vs_head_mode_reference": scheduler_branch_error,
            "splitk_all_exact_vs_dense": exact_error,
            "splitk_head_mode_vs_dense": _error(splitk_head_out[:, None, :, :], dense_ref[:, None, :, :]),
            "flashinfer_vs_dense": flashinfer_error,
        },
        "backend": {
            "threads": args.threads,
            "flashinfer_available": flashinfer_available,
            "flashinfer_failure": flashinfer_failure,
            "flashinfer_tensor_cores": args.flashinfer_tensor_cores,
        },
        "decision": {
            "scheduler_branch_correct": bool(scheduler_branch_correct),
            "exact_equiv_correct": bool(exact_equiv_correct),
            "single_launch_backend_viable": bool(scheduler_branch_correct and splitk_head_ms < reference_ms),
            "next_path": next_path,
            "error_budget": args.error_budget,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-path", default="")
    parser.add_argument("--k-path", default="")
    parser.add_argument("--v-path", default="")
    parser.add_argument("--true-kv-heads", type=int, default=2)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=14)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--kv-len", type=int, default=32768)
    parser.add_argument("--seed-heads", default="2,3,4,6,7")
    parser.add_argument("--num-chunks", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--recent-blocks", type=int, default=2)
    parser.add_argument("--middle-seed-blocks", type=int, default=2)
    parser.add_argument("--block-order", choices=["sequential", "recent_first"], default="recent_first")
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--error-budget", type=float, default=2e-3)
    parser.add_argument("--flashinfer-tensor-cores", action="store_true")
    parser.add_argument("--verbose-compile", action="store_true")
    parser.add_argument("--summary-json-out", default="")
    args = parser.parse_args()
    if bool(args.q_path) != bool(args.k_path) or bool(args.q_path) != bool(args.v_path):
        raise ValueError("--q-path/--k-path/--v-path must be provided together")
    if args.q_heads % args.kv_heads != 0:
        raise ValueError("--q-heads must be divisible by --kv-heads")

    result = profile(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.summary_json_out:
        path = Path(args.summary_json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
