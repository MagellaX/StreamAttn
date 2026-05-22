"""Profile a specialized direct-output seed-only CUDA decode kernel.

This is the lower-overhead counterpart to the Triton seed-only kernel:

* one CTA per Q head, preserving enough parallelism for Hq=14;
* no selected-head indirection;
* no per-call output allocation;
* no head-mode branch;
* direct final [B, 1, Hq, D] output write.

The target question is whether a specialized CUDA kernel can beat the current
Triton preallocated seed path and FlashInfer TC exact on logit-safe layers.
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

from benchmarks.profile_gate0_seed_only_true_gqa import _flashinfer_single_decode  # noqa: E402
from benchmarks.profile_gate0_true_gqa import _true_gqa_kv  # noqa: E402
from benchmarks.profile_stream_attn_gate0_wrapper import (  # noqa: E402
    _dtype,
    _error,
    _load_tensor,
    _sync,
    _time_cuda,
)
from stream_attention.kernels.gate0_launch_floor_triton import (  # noqa: E402
    gate0_launch_floor_empty_triton_forward,
)
from stream_attention.kernels.gate0_seed_only_triton import (  # noqa: E402
    gate0_seed_only_attention_triton_forward_out,
)


CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor streamattn_seed_only_direct_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor out,
    int64_t block_size,
    int64_t sink_blocks,
    int64_t recent_blocks,
    int64_t middle_seed_blocks,
    int64_t block_order,
    int64_t threads);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("seed_only_direct", &streamattn_seed_only_direct_cuda,
        "StreamAttn direct-output seed-only true-GQA decode");
}
"""


CUDA_SOURCE = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

constexpr int MAX_D = 128;

template <typename scalar_t>
__device__ __forceinline__ float load_as_float(const scalar_t* ptr, int64_t offset) {
  return static_cast<float>(ptr[offset]);
}

__device__ __forceinline__ int seed_slot_to_token(
    int slot,
    int N,
    int block_size,
    int sink_blocks,
    int recent_blocks,
    int middle_seed_blocks,
    int block_order) {
  const int bs = block_size <= 0 ? 1 : block_size;
  const int num_blocks = (N + bs - 1) / bs;
  const int block_pos = slot / bs;
  const int offset = slot - block_pos * bs;
  const int seed_blocks = sink_blocks + recent_blocks + middle_seed_blocks;
  if (block_pos < 0 || block_pos >= seed_blocks) {
    return -1;
  }

  int block_idx = -1;
  if (block_pos < sink_blocks) {
    block_idx = block_pos;
  } else if (block_pos < sink_blocks + recent_blocks) {
    const int recent_pos = block_pos - sink_blocks;
    block_idx = num_blocks - recent_blocks + recent_pos;
  } else {
    const int middle_pos = block_pos - sink_blocks - recent_blocks;
    if (block_order == 0) {
      block_idx = sink_blocks + middle_pos;
    } else {
      block_idx = num_blocks - recent_blocks - 1 - middle_pos;
    }
  }
  const int token = block_idx * bs + offset;
  return (block_idx >= 0 && block_idx < num_blocks && token >= 0 && token < N) ? token : -1;
}

template <typename scalar_t>
__global__ void seed_only_direct_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ out,
    int B,
    int Hq,
    int Hkv,
    int N,
    int D,
    int group_size,
    int seed_slots,
    int block_size,
    int sink_blocks,
    int recent_blocks,
    int middle_seed_blocks,
    int block_order,
    float scale) {
  const int bh = blockIdx.x;
  const int b = bh / Hq;
  const int qh = bh - b * Hq;
  const int kvh = qh / group_size;
  const int tid = threadIdx.x;

  extern __shared__ float smem[];
  float* q_sh = smem;
  float* reduce = q_sh + D;
  float* acc_sh = reduce + blockDim.x;

  for (int d = tid; d < D; d += blockDim.x) {
    q_sh[d] = load_as_float(q, ((int64_t)b * Hq + qh) * D + d);
  }
  __syncthreads();

  float local_max = -INFINITY;
  for (int slot = tid; slot < seed_slots; slot += blockDim.x) {
    const int n = seed_slot_to_token(
        slot, N, block_size, sink_blocks, recent_blocks, middle_seed_blocks, block_order);
    if (n < 0) {
      continue;
    }
    const int64_t base = ((int64_t)b * N * Hkv + (int64_t)n * Hkv + kvh) * D;
    float dot = 0.0f;
    for (int d = 0; d < D; ++d) {
      dot += q_sh[d] * load_as_float(k, base + d);
    }
    local_max = fmaxf(local_max, dot * scale);
  }
  reduce[tid] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
    }
    __syncthreads();
  }
  const float max_score = reduce[0];

  float acc[MAX_D];
  #pragma unroll
  for (int d = 0; d < MAX_D; ++d) {
    acc[d] = 0.0f;
  }
  float local_den = 0.0f;
  for (int slot = tid; slot < seed_slots; slot += blockDim.x) {
    const int n = seed_slot_to_token(
        slot, N, block_size, sink_blocks, recent_blocks, middle_seed_blocks, block_order);
    if (n < 0) {
      continue;
    }
    const int64_t base = ((int64_t)b * N * Hkv + (int64_t)n * Hkv + kvh) * D;
    float dot = 0.0f;
    for (int d = 0; d < D; ++d) {
      dot += q_sh[d] * load_as_float(k, base + d);
    }
    const float p = expf(dot * scale - max_score);
    local_den += p;
    for (int d = 0; d < D; ++d) {
      acc[d] += p * load_as_float(v, base + d);
    }
  }

  reduce[tid] = local_den;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }
  const float den = reduce[0];

  for (int d = 0; d < D; ++d) {
    acc_sh[d * blockDim.x + tid] = acc[d];
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      for (int d = 0; d < D; ++d) {
        acc_sh[d * blockDim.x + tid] += acc_sh[d * blockDim.x + tid + stride];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    for (int d = 0; d < D; ++d) {
      const float value = den > 0.0f ? acc_sh[d * blockDim.x] / den : 0.0f;
      out[((int64_t)b * Hq + qh) * D + d] = static_cast<scalar_t>(value);
    }
  }
}

torch::Tensor streamattn_seed_only_direct_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor out,
    int64_t block_size,
    int64_t sink_blocks,
    int64_t recent_blocks,
    int64_t middle_seed_blocks,
    int64_t block_order,
    int64_t threads) {
  TORCH_CHECK(q.is_cuda(), "q must be CUDA");
  TORCH_CHECK(k.is_cuda(), "k must be CUDA");
  TORCH_CHECK(v.is_cuda(), "v must be CUDA");
  TORCH_CHECK(out.is_cuda(), "out must be CUDA");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous [B,1,Hq,D]");
  TORCH_CHECK(k.is_contiguous(), "k must be contiguous [B,N,Hkv,D]");
  TORCH_CHECK(v.is_contiguous(), "v must be contiguous [B,N,Hkv,D]");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous [B,1,Hq,D]");
  TORCH_CHECK(q.dim() == 4 && q.size(1) == 1, "q must have shape [B,1,Hq,D]");
  TORCH_CHECK(k.dim() == 4, "k must have shape [B,N,Hkv,D]");
  TORCH_CHECK(v.sizes() == k.sizes(), "v must match k shape");
  TORCH_CHECK(out.sizes() == q.sizes(), "out must match q shape");
  const int B = q.size(0);
  const int Hq = q.size(2);
  const int D = q.size(3);
  const int N = k.size(1);
  const int Hkv = k.size(2);
  TORCH_CHECK(k.size(0) == B && k.size(3) == D, "K shape incompatible with Q");
  TORCH_CHECK(Hq % Hkv == 0, "Hq must be divisible by Hkv");
  TORCH_CHECK(D > 0 && D <= MAX_D, "head_dim must be in (0, 128]");
  const int group_size = Hq / Hkv;
  const int seed_slots = static_cast<int>((sink_blocks + recent_blocks + middle_seed_blocks) * block_size);
  TORCH_CHECK(seed_slots > 0, "seed slot count must be positive");
  TORCH_CHECK(threads > 0 && (threads & (threads - 1)) == 0, "threads must be a power of two");

  const dim3 grid(B * Hq);
  const dim3 block(static_cast<int>(threads));
  const size_t shared_bytes = (D + static_cast<int>(threads) + D * static_cast<int>(threads)) * sizeof(float);
  const float scale = 1.0f / sqrtf(static_cast<float>(D));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      q.scalar_type(),
      "streamattn_seed_only_direct_cuda",
      [&] {
    auto kernel = seed_only_direct_kernel<scalar_t>;
    cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes));
    kernel<<<grid, block, shared_bytes>>>(
        q.data_ptr<scalar_t>(),
        k.data_ptr<scalar_t>(),
        v.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        B,
        Hq,
        Hkv,
        N,
        D,
        group_size,
        seed_slots,
        static_cast<int>(block_size),
        static_cast<int>(sink_blocks),
        static_cast<int>(recent_blocks),
        static_cast<int>(middle_seed_blocks),
        static_cast<int>(block_order),
        scale);
  });
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
  return out;
}
"""


def _compile_extension(*, verbose: bool = False):
    from torch.utils.cpp_extension import load_inline

    previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    try:
        build_dir = tempfile.mkdtemp(prefix="streamattn_seed_only_direct_cuda_")
        return load_inline(
            name="streamattn_seed_only_direct_cuda",
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


def _block_order_id(block_order: str) -> int:
    if block_order == "sequential":
        return 0
    if block_order in {"recent_first", "sink_recent_first"}:
        return 1
    raise ValueError("--block-order must be sequential, recent_first, or sink_recent_first")


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
    k_true = _true_gqa_kv(k_expanded, true_kv_heads=args.true_kv_heads).contiguous()
    v_true = _true_gqa_kv(v_expanded, true_kv_heads=args.true_kv_heads).contiguous()
    q = q.contiguous()

    compile_start = time.perf_counter()
    ext = _compile_extension(verbose=args.verbose_compile)
    compile_ms = (time.perf_counter() - compile_start) * 1000.0

    cuda_out = torch.empty_like(q)
    triton_out = torch.empty_like(q)
    selected_heads = torch.arange(q.shape[2], device=device, dtype=torch.int32)

    def cuda_seed() -> torch.Tensor:
        return ext.seed_only_direct(
            q,
            k_true,
            v_true,
            cuda_out,
            args.block_size,
            args.sink_blocks,
            args.recent_blocks,
            args.middle_seed_blocks,
            _block_order_id(args.block_order),
            args.threads,
        )

    def triton_seed() -> torch.Tensor:
        return gate0_seed_only_attention_triton_forward_out(
            q,
            k_true,
            v_true,
            triton_out,
            block_size=args.block_size,
            sink_blocks=args.sink_blocks,
            recent_blocks=args.recent_blocks,
            middle_seed_blocks=args.middle_seed_blocks,
            block_order=args.block_order,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        )

    def empty_launch() -> torch.Tensor:
        return gate0_launch_floor_empty_triton_forward(q, selected_heads)

    def flashinfer_exact() -> torch.Tensor:
        return _flashinfer_single_decode(
            q,
            k_true,
            v_true,
            use_tensor_cores=args.flashinfer_tensor_cores,
        )

    cuda_ref = cuda_seed().clone()
    triton_ref = triton_seed().clone()
    flash_ref = flashinfer_exact().clone()
    _sync(device)

    timing = {
        "cuda_seed_direct_ms": _time_cuda(cuda_seed, device=device, warmup=args.warmup, iters=args.iters),
        "triton_seed_prealloc_ms": _time_cuda(triton_seed, device=device, warmup=args.warmup, iters=args.iters),
        "flashinfer_tc_exact_ms": _time_cuda(
            flashinfer_exact,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        ),
        "empty_launch_ms": _time_cuda(empty_launch, device=device, warmup=args.warmup, iters=args.iters),
    }
    timing["cuda_seed_speedup_vs_flashinfer"] = timing["flashinfer_tc_exact_ms"] / timing["cuda_seed_direct_ms"]
    timing["cuda_seed_speedup_vs_triton_seed"] = timing["triton_seed_prealloc_ms"] / timing["cuda_seed_direct_ms"]

    seed_blocks = args.sink_blocks + args.recent_blocks + args.middle_seed_blocks
    return {
        "schema": "streamattn.gate0.seed_only_direct_cuda.v1",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
        "shape": {
            "batch": int(q.shape[0]),
            "query_len": int(q.shape[1]),
            "q_heads": int(q.shape[2]),
            "true_kv_heads": int(k_true.shape[2]),
            "group_size": int(q.shape[2] // k_true.shape[2]),
            "kv_len": int(k_true.shape[1]),
            "dim": int(q.shape[3]),
            "dtype": args.dtype,
        },
        "seed_config": {
            "block_size": args.block_size,
            "sink_blocks": args.sink_blocks,
            "recent_blocks": args.recent_blocks,
            "middle_seed_blocks": args.middle_seed_blocks,
            "seed_blocks": seed_blocks,
            "seed_slots": seed_blocks * args.block_size,
            "block_order": args.block_order,
        },
        "backend": {
            "threads": args.threads,
            "num_warps": args.num_warps,
            "num_stages": args.num_stages,
            "compile_ms": compile_ms,
        },
        "timing": timing,
        "quality": {
            "cuda_seed_vs_triton_seed": _error(cuda_ref, triton_ref),
            "cuda_seed_vs_flashinfer_exact": _error(cuda_ref, flash_ref),
            "triton_seed_vs_flashinfer_exact": _error(triton_ref, flash_ref),
        },
        "decision": (
            "cuda_direct_seed_beats_flashinfer"
            if timing["cuda_seed_direct_ms"] < timing["flashinfer_tc_exact_ms"]
            else "cuda_direct_seed_still_slower_than_flashinfer"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-path", required=True)
    parser.add_argument("--k-path", required=True)
    parser.add_argument("--v-path", required=True)
    parser.add_argument("--true-kv-heads", type=int, required=True)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--recent-blocks", type=int, default=2)
    parser.add_argument("--middle-seed-blocks", type=int, default=8)
    parser.add_argument("--block-order", default="recent_first", choices=["sequential", "recent_first", "sink_recent_first"])
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--flashinfer-tensor-cores", action="store_true")
    parser.add_argument("--verbose-compile", action="store_true")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    result = profile(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
