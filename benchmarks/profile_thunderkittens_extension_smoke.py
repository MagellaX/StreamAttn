"""Compile and run a minimal ThunderKittens-backed PyTorch extension.

``probe_tensor_core_backend_paths.py`` checks that ThunderKittens headers can be
compiled by ``nvcc``.  This smoke goes one step closer to the StreamAttn backend:
it builds a PyTorch CUDA extension, includes ``kittens.cuh``, uses TK device
helpers, accepts a per-Q-head mode table, and validates true-GQA head mapping.

This is still not an attention benchmark.  Its purpose is to prove the build/run
plumbing for the next tensor-core head-mode decode spike.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
THUNDERKITTENS_REPO = "https://github.com/HazyResearch/ThunderKittens.git"


CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor streamattn_tk_head_mode_smoke_cuda(
    torch::Tensor q,
    torch::Tensor head_modes,
    int64_t kv_heads);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("head_mode_smoke", &streamattn_tk_head_mode_smoke_cuda,
        "StreamAttn ThunderKittens head-mode smoke");
}
"""


CUDA_SOURCE = r"""
#include <torch/extension.h>
#include "kittens.cuh"

__global__ void streamattn_tk_head_mode_smoke_kernel(
    const float* __restrict__ q,
    const int32_t* __restrict__ head_modes,
    float* __restrict__ out,
    int B,
    int Hq,
    int Hkv,
    int D) {
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int group_size = Hq / Hkv;
  const int kv_head = h / group_size;

  float q_sum = 0.0f;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    q_sum += q[(b * Hq + h) * D + d];
  }

  extern __shared__ float smem[];
  smem[threadIdx.x] = q_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (::kittens::laneid() == 0 && ::kittens::warpid() == 0) {
    const int64_t base = ((int64_t)b * Hq + h) * 4;
    out[base + 0] = float(head_modes[h]);
    out[base + 1] = float(kv_head);
    out[base + 2] = smem[0];
    out[base + 3] = 1.0f;  // TK device helper path reached.
  }
}

torch::Tensor streamattn_tk_head_mode_smoke_cuda(
    torch::Tensor q,
    torch::Tensor head_modes,
    int64_t kv_heads) {
  TORCH_CHECK(q.is_cuda(), "q must be CUDA");
  TORCH_CHECK(head_modes.is_cuda(), "head_modes must be CUDA");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(head_modes.is_contiguous(), "head_modes must be contiguous");
  TORCH_CHECK(q.scalar_type() == at::ScalarType::Float, "q must be float32 for smoke");
  TORCH_CHECK(head_modes.scalar_type() == at::ScalarType::Int, "head_modes must be int32");
  TORCH_CHECK(q.dim() == 3, "q must be [B, Hq, D]");
  TORCH_CHECK(head_modes.dim() == 1, "head_modes must be [Hq]");
  const int B = (int)q.size(0);
  const int Hq = (int)q.size(1);
  const int D = (int)q.size(2);
  const int Hkv = (int)kv_heads;
  TORCH_CHECK(Hkv > 0 && Hq % Hkv == 0, "Hq must be divisible by Hkv");
  TORCH_CHECK(head_modes.size(0) == Hq, "head_modes length must match Hq");

  auto out = torch::empty({B, Hq, 4}, q.options().dtype(torch::kFloat32));
  const int threads = 128;
  const dim3 grid(Hq, B);
  const size_t smem = threads * sizeof(float);
  streamattn_tk_head_mode_smoke_kernel<<<grid, threads, smem>>>(
      q.data_ptr<float>(),
      head_modes.data_ptr<int32_t>(),
      out.data_ptr<float>(),
      B,
      Hq,
      Hkv,
      D);
  return out;
}
"""


def _run(cmd: list[str], *, timeout: int = 900) -> dict[str, Any]:
    result = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=timeout,
    )
    return {
        "cmd": cmd,
        "returncode": result.returncode,
        "output_tail": result.stdout[-12000:],
    }


def _clone_tk(root: Path) -> dict[str, Any]:
    if root.exists():
        return {"requested": True, "available": True, "path": str(root), "reason": "already_exists"}
    root.parent.mkdir(parents=True, exist_ok=True)
    result = _run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            THUNDERKITTENS_REPO,
            str(root),
        ],
        timeout=1800,
    )
    return {
        "requested": True,
        "available": root.exists() and result["returncode"] == 0,
        "path": str(root),
        "returncode": result["returncode"],
        "output_tail": result["output_tail"],
    }


def _find_tk_root(path_text: str) -> Optional[Path]:
    candidates = []
    if path_text:
        candidates.append(Path(path_text).expanduser())
    candidates.extend(
        [
            REPO_ROOT / "third_party" / "ThunderKittens",
            REPO_ROOT / "artifacts" / "backend_sources" / "ThunderKittens",
            Path("/tmp/streamattn_backend_sources/ThunderKittens"),
        ]
    )
    for root in candidates:
        if (root / "include" / "kittens.cuh").exists():
            return root.resolve()
    return None


def _tk_arch_define(cuda_arch: str) -> str:
    arch = cuda_arch or os.environ.get("STREAMATTN_CUDA_ARCH") or "sm_90a"
    if arch in {"sm_90", "sm_90a", "compute_90", "compute_90a"}:
        return "KITTENS_SM90"
    if arch in {"sm_100", "sm_100a", "compute_100", "compute_100a"}:
        return "KITTENS_SM100"
    if arch in {"sm_103", "sm_103a", "compute_103", "compute_103a"}:
        return "KITTENS_SM103"
    if arch in {"sm_120", "sm_120a", "compute_120", "compute_120a"}:
        return "KITTENS_SM120"
    return "KITTENS_SM80"


def _time_cuda(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / max(1, iters)


def profile(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "available": False,
            "reason": "CUDA is not available",
            "recommended_next_path": "run_on_h100_modal",
        }

    tk_root = _find_tk_root(args.thunderkittens_root)
    clone_result = {"requested": False}
    if tk_root is None and args.clone_thunderkittens:
        target = Path(args.checkout_dir).expanduser() / "ThunderKittens"
        clone_result = _clone_tk(target)
        tk_root = _find_tk_root(str(target))
    if tk_root is None:
        return {
            "available": False,
            "reason": "ThunderKittens root not found",
            "clone_result": clone_result,
            "recommended_next_path": "clone_thunderkittens_and_retry",
        }

    from torch.utils.cpp_extension import load_inline

    previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = args.torch_cuda_arch_list
    build_dir_obj = tempfile.TemporaryDirectory(prefix="streamattn_tk_extension_")
    build_dir = Path(build_dir_obj.name)
    compile_start = time.perf_counter()
    try:
        module = load_inline(
            name="streamattn_tk_head_mode_smoke",
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            build_directory=str(build_dir),
            extra_include_paths=[str(tk_root / "include")],
            extra_cflags=["-std=c++20"],
            extra_cuda_cflags=[
                "-std=c++20",
                "-O3",
                "--use_fast_math",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                f"-D{_tk_arch_define(args.cuda_arch)}",
            ],
            with_cuda=True,
            verbose=args.verbose_compile,
        )
        compile_ms = (time.perf_counter() - compile_start) * 1000.0

        q = torch.arange(
            args.batch * args.q_heads * args.head_dim,
            device="cuda",
            dtype=torch.float32,
        ).reshape(args.batch, args.q_heads, args.head_dim)
        modes = torch.ones(args.q_heads, device="cuda", dtype=torch.int32)
        for head in [int(item) for item in args.seed_heads.split(",") if item.strip()]:
            modes[head] = 0

        def run():
            return module.head_mode_smoke(q, modes, args.kv_heads)

        ms = _time_cuda(run, warmup=args.warmup, iters=args.iters)
        out = run()
        torch.cuda.synchronize()

        expected_kv = torch.arange(args.q_heads, device="cuda", dtype=torch.float32) // (
            args.q_heads // args.kv_heads
        )
        mode_error = float((out[0, :, 0] - modes.float()).abs().max().item())
        kv_error = float((out[0, :, 1] - expected_kv).abs().max().item())
        reached = bool((out[:, :, 3] == 1.0).all().item())
        return {
            "available": True,
            "schema": "streamattn.thunderkittens_extension_smoke.v1",
            "tk_root": str(tk_root),
            "clone_result": clone_result,
            "compile_ms": compile_ms,
            "run_ms": ms,
            "shape": {
                "batch": args.batch,
                "q_heads": args.q_heads,
                "kv_heads": args.kv_heads,
                "head_dim": args.head_dim,
            },
            "mode_error": mode_error,
            "kv_head_mapping_error": kv_error,
            "tk_device_helper_reached": reached,
            "recommended_next_path": (
                "implement_thunderkittens_head_mode_attention_spike"
                if mode_error == 0.0 and kv_error == 0.0 and reached
                else "fix_thunderkittens_extension_plumbing"
            ),
        }
    except Exception as exc:
        return {
            "available": False,
            "schema": "streamattn.thunderkittens_extension_smoke.v1",
            "tk_root": str(tk_root),
            "clone_result": clone_result,
            "compile_elapsed_ms": (time.perf_counter() - compile_start) * 1000.0,
            "reason": repr(exc),
            "recommended_next_path": "fix_thunderkittens_extension_compile",
        }
    finally:
        build_dir_obj.cleanup()
        if previous_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thunderkittens-root", default="")
    parser.add_argument("--checkout-dir", default="artifacts/backend_sources")
    parser.add_argument("--clone-thunderkittens", action="store_true")
    parser.add_argument("--cuda-arch", default="sm_90a")
    parser.add_argument("--torch-cuda-arch-list", default="9.0a")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=14)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--seed-heads", default="2,3,4,6,7")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
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
