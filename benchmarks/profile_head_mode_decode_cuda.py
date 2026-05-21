"""Prototype a scheduler-level StreamAttn head-mode decode backend in CUDA.

The FlashInfer custom-JIT smoke proved that logits/math hooks are too late for
StreamAttn: they can change the result, but they do not prevent dense KV/QK
scheduling.  This benchmark compiles a minimal CUDA extension where the
``head_modes`` branch happens before the KV loop:

* ``EXACT_FULL`` heads visit every KV token;
* ``SEED_ONLY`` heads visit only sink/recent/middle-seed token ranges.

This is not intended to beat FlashInfer yet.  It is a scheduler-level prototype
that answers whether a one-launch true-GQA head-mode backend is correct and
what part remains slow once the branch is in the right place.
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
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_gate0_seed_only_true_gqa import _flashinfer_single_decode, _parse_heads  # noqa: E402
from benchmarks.profile_gate0_true_gqa import _dense_true_gqa, _true_gqa_kv  # noqa: E402
from benchmarks.profile_stream_attn_gate0_wrapper import (  # noqa: E402
    _dtype,
    _error,
    _load_tensor,
    _time_cuda,
)


CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor streamattn_head_mode_decode_cuda(
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
  m.def("head_mode_decode", &streamattn_head_mode_decode_cuda,
        "StreamAttn scheduler-level true-GQA head-mode decode");
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

__device__ __forceinline__ bool streamattn_keep_token(
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
__global__ void streamattn_head_mode_decode_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int32_t* __restrict__ head_modes,
    scalar_t* __restrict__ out,
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
    q_vec[d] = load_as_float(q, (b * Hq + h) * D + d);
  }

  float local_max = -INFINITY;
  for (int n = tid; n < N; n += blockDim.x) {
    if (seed_only && !streamattn_keep_token(
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
  for (int n = tid; n < N; n += blockDim.x) {
    if (seed_only && !streamattn_keep_token(
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
    for (int d = 0; d < D; ++d) {
      const float value = den > 0.0f ? s_acc[d * blockDim.x] / den : 0.0f;
      out[(b * Hq + h) * D + d] = static_cast<scalar_t>(value);
    }
  }
}

torch::Tensor streamattn_head_mode_decode_cuda(
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
  const dim3 grid(B * Hq);
  const dim3 block(threads);
  const size_t shared_bytes = (threads + D * threads) * sizeof(float);
  const float scale = 1.0f / sqrtf(static_cast<float>(D));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      q.scalar_type(),
      "streamattn_head_mode_decode_cuda",
      [&] {
    auto kernel = streamattn_head_mode_decode_kernel<scalar_t>;
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
  return out;
}
"""


def _compile_extension(*, verbose: bool = False):
    from torch.utils.cpp_extension import load_inline

    previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    try:
        build_dir = tempfile.mkdtemp(prefix="streamattn_head_mode_decode_cuda_")
        return load_inline(
            name="streamattn_head_mode_decode_cuda",
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
    if block_order == "recent_first":
        return 1
    raise ValueError("--block-order must be sequential or recent_first")


def _seed_keep_mask(
    kv_len: int,
    *,
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
        kv_len,
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
            logits = torch.matmul(k[batch_idx, :, kv_head, :].float(), q[batch_idx, q_head, :].float()) * scale
            if int(head_modes[q_head].item()) != 0:
                logits = logits.masked_fill(~seed_mask, -float("inf"))
            probs = torch.softmax(logits, dim=0)
            out[batch_idx, q_head, :] = torch.matmul(probs.to(v.dtype), v[batch_idx, :, kv_head, :])
    return out


def _make_inputs(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    if args.q_path:
        q_loaded = _load_tensor(args.q_path, key="q", device=device, dtype=dtype)
        k_loaded = _load_tensor(args.k_path, key="k", device=device, dtype=dtype)
        v_loaded = _load_tensor(args.v_path, key="v", device=device, dtype=dtype)
        q = q_loaded[:, -1].contiguous()
        k = _true_gqa_kv(k_loaded, true_kv_heads=args.true_kv_heads).contiguous()
        v = _true_gqa_kv(v_loaded, true_kv_heads=args.true_kv_heads).contiguous()
        return q, k, v

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    q = torch.randn(args.batch_size, args.q_heads, args.head_dim, device=device, dtype=dtype)
    k = torch.randn(args.batch_size, args.kv_len, args.kv_heads, args.head_dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    return q, k, v


def _flashinfer_exact(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, use_tensor_cores: bool) -> torch.Tensor:
    return _flashinfer_single_decode(
        q[:, None, :, :].contiguous(),
        k,
        v,
        use_tensor_cores=use_tensor_cores,
    )[:, 0]


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

    def cuda_head_mode() -> torch.Tensor:
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

    def cuda_all_exact() -> torch.Tensor:
        return module.head_mode_decode(
            q,
            k,
            v,
            all_exact_modes,
            args.block_size,
            args.sink_blocks,
            args.recent_blocks,
            args.middle_seed_blocks,
            _block_order_id(args.block_order),
            args.threads,
        )

    def cuda_all_seed() -> torch.Tensor:
        return module.head_mode_decode(
            q,
            k,
            v,
            all_seed_modes,
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
    dense_ref = torch_dense()
    cuda_head_out = cuda_head_mode()
    cuda_exact_out = cuda_all_exact()

    flashinfer_ms = None
    flashinfer_error = None
    flashinfer_available = True
    flashinfer_failure = None
    try:
        flashinfer_ms = _time_cuda(flashinfer_exact, device=device, warmup=args.warmup, iters=args.iters)
        flashinfer_error = _error(
            flashinfer_exact()[:, None, :, :],
            dense_ref[:, None, :, :],
        )
    except Exception as exc:
        flashinfer_available = False
        flashinfer_failure = f"{type(exc).__name__}: {exc}"

    cuda_head_ms = _time_cuda(cuda_head_mode, device=device, warmup=args.warmup, iters=args.iters)
    cuda_exact_ms = _time_cuda(cuda_all_exact, device=device, warmup=args.warmup, iters=args.iters)
    cuda_seed_ms = _time_cuda(cuda_all_seed, device=device, warmup=args.warmup, iters=args.iters)
    torch_dense_ms = _time_cuda(torch_dense, device=device, warmup=args.warmup, iters=args.iters)

    expected_seed_tokens = (
        args.sink_blocks + args.recent_blocks + args.middle_seed_blocks
    ) * args.block_size
    expected_seed_tokens = min(expected_seed_tokens, int(k.shape[1]))

    scheduler_branch_correct = _error(
        cuda_head_out[:, None, :, :],
        head_mode_ref[:, None, :, :],
    )["max_abs_error"] <= args.error_budget
    exact_equiv_correct = _error(
        cuda_exact_out[:, None, :, :],
        dense_ref[:, None, :, :],
    )["max_abs_error"] <= args.error_budget
    reference_ms = flashinfer_ms if flashinfer_ms is not None else torch_dense_ms
    if scheduler_branch_correct and cuda_head_ms < reference_ms:
        next_path = "optimize_and_harden_scheduler_level_backend"
    elif scheduler_branch_correct:
        next_path = "scheduler_branch_works_but_exact_branch_needs_dense_quality_rewrite"
    else:
        next_path = "fix_scheduler_correctness_before_backend_work"

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
            "block_size": args.block_size,
            "sink_blocks": args.sink_blocks,
            "recent_blocks": args.recent_blocks,
            "middle_seed_blocks": args.middle_seed_blocks,
            "block_order": args.block_order,
            "expected_seed_tokens_per_seed_head": expected_seed_tokens,
        },
        "timing": {
            "compile_ms": compile_ms,
            "cuda_head_mode_ms": cuda_head_ms,
            "cuda_all_exact_ms": cuda_exact_ms,
            "cuda_all_seed_ms": cuda_seed_ms,
            "torch_dense_true_gqa_ms": torch_dense_ms,
            "flashinfer_exact_ms": flashinfer_ms,
            "cuda_head_mode_speedup_vs_flashinfer": (
                flashinfer_ms / cuda_head_ms if flashinfer_ms and cuda_head_ms else None
            ),
            "cuda_head_mode_speedup_vs_torch_dense": torch_dense_ms / cuda_head_ms if cuda_head_ms else None,
        },
        "quality": {
            "cuda_head_mode_vs_head_mode_reference": _error(
                cuda_head_out[:, None, :, :],
                head_mode_ref[:, None, :, :],
            ),
            "cuda_all_exact_vs_dense": _error(
                cuda_exact_out[:, None, :, :],
                dense_ref[:, None, :, :],
            ),
            "cuda_head_mode_vs_dense": _error(
                cuda_head_out[:, None, :, :],
                dense_ref[:, None, :, :],
            ),
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
            "single_launch_backend_viable": bool(scheduler_branch_correct and cuda_head_ms < reference_ms),
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
