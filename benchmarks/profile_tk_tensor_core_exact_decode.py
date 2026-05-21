"""ThunderKittens tensor-core exact true-GQA decode baseline.

This spike answers one narrow backend question after the scalar TK head-mode
prototype: can a small true-GQA decode group use TK MMA instead of a scalar
per-head loop?

It intentionally starts with a compact, testable shape:

* B=1/M=1 decode;
* Q heads are packed by KV group into a 16-row tensor-core tile;
* K/V are packed as [B, Hkv, N, D] for the spike;
* one warp processes one KV group exactly.

This is not the final scheduler.  It is the exact-branch floor we need before
adding row masks and seed-only tile gating.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_gate0_true_gqa import _dense_true_gqa  # noqa: E402
from benchmarks.profile_head_mode_decode_cuda import _flashinfer_exact  # noqa: E402
from benchmarks.profile_stream_attn_gate0_wrapper import _dtype, _error, _time_cuda  # noqa: E402
from benchmarks.profile_thunderkittens_extension_smoke import (  # noqa: E402
    _clone_tk,
    _find_tk_root,
    _tk_arch_define,
)


CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor streamattn_tk_tc_exact_decode_cuda(
    torch::Tensor q_group,
    torch::Tensor k_group,
    torch::Tensor v_group);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("exact_decode", &streamattn_tk_tc_exact_decode_cuda,
        "StreamAttn TK tensor-core exact true-GQA decode baseline");
}
"""


CUDA_SOURCE = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "kittens.cuh"

using namespace kittens;

template <int D>
struct streamattn_tc_exact_tiles {};

template <>
struct streamattn_tc_exact_tiles<128> {
  using q_tile = rt_bf<16, 128>;
  using k_tile = rt_bf<16, 128>;
  using v_tile = rt_bf<16, 128, ducks::rt_layout::col>;
  using scores_fl = rt_fl<16, 16>;
  using scores_bf = rt_bf<16, 16>;
  using out_tile = rt_fl<16, 128>;
  using score_col = col_vec<scores_fl>;
};

struct streamattn_tc_exact_globals {
  using q_gl = gl<bf16, -1, -1, -1, -1>;
  using kv_gl = gl<bf16, -1, -1, -1, -1>;
  q_gl q;
  kv_gl k;
  kv_gl v;
  q_gl out;
  int N;
  int Hkv;
};

template <int D>
__global__ void streamattn_tk_tc_exact_decode_kernel(
    const __grid_constant__ streamattn_tc_exact_globals g) {
  using T = streamattn_tc_exact_tiles<D>;
  const int bh = blockIdx.x;
  const int b = bh / g.Hkv;
  const int kvh = bh - b * g.Hkv;
  if (threadIdx.x >= 32) return;

  typename T::q_tile q_reg;
  typename T::k_tile k_reg;
  typename T::v_tile v_reg;
  typename T::scores_fl scores;
  typename T::scores_bf scores_mma;
  typename T::out_tile acc;
  typename T::score_col max_vec;
  typename T::score_col norm_vec;
  typename T::score_col max_vec_last_scaled;
  typename T::score_col max_vec_scaled;

  warp::load(q_reg, g.q, {b, kvh, 0, 0});
  warp::zero(acc);
  warp::zero(norm_vec);
  warp::neg_infty(max_vec);

  constexpr float scale_log2 = 0.08838834764f * 1.44269504089f;
  const int tiles = g.N / 16;
  for (int tile = 0; tile < tiles; ++tile) {
    warp::load(k_reg, g.k, {b, kvh, tile, 0});
    warp::zero(scores);
    warp::mma_ABt(scores, q_reg, k_reg, scores);

    warp::copy(max_vec_last_scaled, max_vec);
    warp::mul(max_vec_last_scaled, max_vec_last_scaled, scale_log2);
    warp::row_max(max_vec, scores, max_vec);
    warp::mul(scores, scores, scale_log2);
    warp::mul(max_vec_scaled, max_vec, scale_log2);
    warp::sub_row(scores, scores, max_vec_scaled);
    warp::exp2(scores, scores);
    warp::sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
    warp::exp2(max_vec_last_scaled, max_vec_last_scaled);
    warp::mul(norm_vec, norm_vec, max_vec_last_scaled);
    warp::row_sum(norm_vec, scores, norm_vec);
    warp::copy(scores_mma, scores);
    warp::mul_row(acc, acc, max_vec_last_scaled);

    warp::load(v_reg, g.v, {b, kvh, tile, 0});
    warp::mma_AB(acc, scores_mma, v_reg, acc);
  }

  warp::div_row(acc, acc, norm_vec);
  warp::store(g.out, acc, {b, kvh, 0, 0});
}

torch::Tensor streamattn_tk_tc_exact_decode_cuda(
    torch::Tensor q_group,
    torch::Tensor k_group,
    torch::Tensor v_group) {
  TORCH_CHECK(q_group.is_cuda(), "q_group must be CUDA");
  TORCH_CHECK(k_group.is_cuda(), "k_group must be CUDA");
  TORCH_CHECK(v_group.is_cuda(), "v_group must be CUDA");
  TORCH_CHECK(q_group.is_contiguous(), "q_group must be contiguous [B,Hkv,16,D]");
  TORCH_CHECK(k_group.is_contiguous(), "k_group must be contiguous [B,Hkv,N,D]");
  TORCH_CHECK(v_group.is_contiguous(), "v_group must be contiguous [B,Hkv,N,D]");
  TORCH_CHECK(q_group.scalar_type() == at::ScalarType::BFloat16, "q_group must be bf16 for this spike");
  TORCH_CHECK(k_group.scalar_type() == at::ScalarType::BFloat16, "k_group must be bf16 for this spike");
  TORCH_CHECK(v_group.scalar_type() == at::ScalarType::BFloat16, "v_group must be bf16 for this spike");
  TORCH_CHECK(q_group.dim() == 4, "q_group must have shape [B,Hkv,16,D]");
  TORCH_CHECK(k_group.dim() == 4, "k_group must have shape [B,Hkv,N,D]");
  TORCH_CHECK(v_group.sizes() == k_group.sizes(), "v_group must match k_group shape");
  const int B = q_group.size(0);
  const int Hkv = q_group.size(1);
  const int padded_rows = q_group.size(2);
  const int D = q_group.size(3);
  const int N = k_group.size(2);
  TORCH_CHECK(k_group.size(0) == B && k_group.size(1) == Hkv && k_group.size(3) == D,
              "K shape incompatible with Q");
  TORCH_CHECK(D == 128, "only D=128 is implemented");
  TORCH_CHECK(padded_rows == 16, "only 16 padded Q rows are implemented");
  TORCH_CHECK(N % 16 == 0, "N must be divisible by 16");

  auto out = torch::empty_like(q_group);
  const dim3 grid(B * Hkv);
  const dim3 block(32);
  using q_gl = streamattn_tc_exact_globals::q_gl;
  using kv_gl = streamattn_tc_exact_globals::kv_gl;
  streamattn_tc_exact_globals g{
      q_gl{reinterpret_cast<bf16*>(q_group.data_ptr<at::BFloat16>()),
           static_cast<unsigned long>(B),
           static_cast<unsigned long>(Hkv),
           static_cast<unsigned long>(padded_rows),
           static_cast<unsigned long>(D)},
      kv_gl{reinterpret_cast<bf16*>(k_group.data_ptr<at::BFloat16>()),
            static_cast<unsigned long>(B),
            static_cast<unsigned long>(Hkv),
            static_cast<unsigned long>(N),
            static_cast<unsigned long>(D)},
      kv_gl{reinterpret_cast<bf16*>(v_group.data_ptr<at::BFloat16>()),
            static_cast<unsigned long>(B),
            static_cast<unsigned long>(Hkv),
            static_cast<unsigned long>(N),
            static_cast<unsigned long>(D)},
      q_gl{reinterpret_cast<bf16*>(out.data_ptr<at::BFloat16>()),
           static_cast<unsigned long>(B),
           static_cast<unsigned long>(Hkv),
           static_cast<unsigned long>(padded_rows),
           static_cast<unsigned long>(D)},
      N,
      Hkv};
  streamattn_tk_tc_exact_decode_kernel<128><<<grid, block>>>(g);
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
  return out;
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
        build_dir = tempfile.mkdtemp(prefix="streamattn_tk_tc_exact_decode_")
        return load_inline(
            name="streamattn_tk_tc_exact_decode",
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


def _pack_q_by_kv_group(q: torch.Tensor, kv_heads: int, padded_rows: int = 16) -> torch.Tensor:
    if q.dim() != 3:
        raise ValueError("q must have shape [B,Hq,D]")
    batch, q_heads, dim = q.shape
    if q_heads % kv_heads != 0:
        raise ValueError("q_heads must be divisible by kv_heads")
    group_size = q_heads // kv_heads
    if group_size > padded_rows:
        raise ValueError("group_size exceeds padded_rows")
    packed = torch.zeros((batch, kv_heads, padded_rows, dim), device=q.device, dtype=q.dtype)
    for kv_head in range(kv_heads):
        start = kv_head * group_size
        end = start + group_size
        packed[:, kv_head, :group_size, :] = q[:, start:end, :]
    return packed.contiguous()


def _unpack_q_by_kv_group(packed: torch.Tensor, q_heads: int) -> torch.Tensor:
    batch, kv_heads, _, dim = packed.shape
    if q_heads % kv_heads != 0:
        raise ValueError("q_heads must be divisible by kv_heads")
    group_size = q_heads // kv_heads
    out = torch.empty((batch, q_heads, dim), device=packed.device, dtype=packed.dtype)
    for kv_head in range(kv_heads):
        start = kv_head * group_size
        end = start + group_size
        out[:, start:end, :] = packed[:, kv_head, :group_size, :]
    return out


def _pack_kv_head_major(kv: torch.Tensor) -> torch.Tensor:
    if kv.dim() != 4:
        raise ValueError("kv must have shape [B,N,Hkv,D]")
    return kv.permute(0, 2, 1, 3).contiguous()


def _reference_from_packed(q_group: torch.Tensor, k_group: torch.Tensor, v_group: torch.Tensor) -> torch.Tensor:
    # Reference over the padded rows; caller can unpack actual Q heads.
    batch, kv_heads, padded_rows, dim = q_group.shape
    outputs = []
    scale = dim**-0.5
    for kv_head in range(kv_heads):
        qh = q_group[:, kv_head, :, :].float()
        kh = k_group[:, kv_head, :, :].float()
        vh = v_group[:, kv_head, :, :].float()
        scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.matmul(probs, vh).to(q_group.dtype))
    return torch.stack(outputs, dim=1).contiguous()


def _find_or_clone_tk(args: argparse.Namespace) -> Path:
    tk_root = _find_tk_root(args.tk_root)
    if tk_root is not None:
        return tk_root
    if not args.checkout_dir:
        raise RuntimeError("ThunderKittens root not found; pass --tk-root or --checkout-dir")
    clone = _clone_tk(Path(args.checkout_dir).expanduser() / "ThunderKittens")
    tk_root = _find_tk_root(str(Path(args.checkout_dir).expanduser() / "ThunderKittens"))
    if tk_root is None:
        raise RuntimeError(f"ThunderKittens clone failed: {clone}")
    return tk_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-len", type=int, default=32768)
    parser.add_argument("--q-heads", type=int, default=14)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--tk-root", default="")
    parser.add_argument("--checkout-dir", default="")
    parser.add_argument("--cuda-arch", default="sm_90a")
    parser.add_argument("--torch-cuda-arch-list", default="9.0a")
    parser.add_argument("--compile-verbose", action="store_true")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    if args.dtype != "bf16":
        raise ValueError("this spike currently supports --dtype bf16 only")
    if args.head_dim != 128:
        raise ValueError("this spike currently supports --head-dim 128 only")

    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    torch.manual_seed(args.seed)
    q = torch.randn((1, args.q_heads, args.head_dim), device=device, dtype=dtype)
    k = torch.randn((1, args.kv_len, args.kv_heads, args.head_dim), device=device, dtype=dtype)
    v = torch.randn_like(k)
    q_group = _pack_q_by_kv_group(q, args.kv_heads, padded_rows=16)
    k_group = _pack_kv_head_major(k)
    v_group = _pack_kv_head_major(v)

    tk_root = _find_or_clone_tk(args)
    compile_start = time.perf_counter()
    ext = _compile_extension(
        tk_root=tk_root,
        cuda_arch=args.cuda_arch,
        torch_cuda_arch_list=args.torch_cuda_arch_list,
        verbose=args.compile_verbose,
    )
    compile_s = time.perf_counter() - compile_start

    def tk_exact() -> torch.Tensor:
        return ext.exact_decode(q_group, k_group, v_group)

    tk_out_group = tk_exact()
    torch_ref_group = _reference_from_packed(q_group, k_group, v_group)
    tk_out = _unpack_q_by_kv_group(tk_out_group, args.q_heads)

    def dense_true() -> torch.Tensor:
        return _dense_true_gqa(q[:, None, :, :], k, v)[:, 0]

    dense_ref = dense_true()
    try:
        flashinfer_ms = _time_cuda(
            lambda: _flashinfer_exact(q, k, v, use_tensor_cores=True),
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
    except Exception as exc:  # pragma: no cover - depends on optional backend
        flashinfer_ms = None
        flashinfer_error = str(exc)
    else:
        flashinfer_error = None

    tk_ms = _time_cuda(tk_exact, device=device, warmup=args.warmup, iters=args.iters)
    dense_ms = _time_cuda(dense_true, device=device, warmup=args.warmup, iters=args.iters)
    output = {
        "schema": "streamattn.tk_tensor_core_exact_decode.v1",
        "shape": {
            "batch": 1,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "group_size": args.q_heads // args.kv_heads,
            "padded_group_rows": 16,
            "kv_len": args.kv_len,
            "head_dim": args.head_dim,
            "dtype": args.dtype,
            "kv_layout_runtime": "B,Hkv,N,D",
            "note": "spike packs NHD KV to head-major layout before the TK kernel",
        },
        "compile": {
            "tk_root": str(tk_root),
            "compile_s": compile_s,
            "cuda_arch": args.cuda_arch,
        },
        "timing": {
            "tk_tensor_core_exact_ms": tk_ms,
            "torch_dense_true_gqa_ms": dense_ms,
            "flashinfer_exact_ms": flashinfer_ms,
            "tk_speedup_vs_torch_dense": dense_ms / tk_ms if tk_ms else None,
            "tk_speedup_vs_flashinfer": flashinfer_ms / tk_ms if flashinfer_ms and tk_ms else None,
        },
        "quality": {
            "tk_vs_packed_torch_ref": _error(tk_out_group, torch_ref_group),
            "tk_vs_dense_true_gqa": _error(tk_out[:, None, :, :], dense_ref[:, None, :, :]),
        },
        "flashinfer_error": flashinfer_error,
        "next_path": "add_row_mask_exact_if_tensor_core_baseline_is_close"
        if (flashinfer_ms is not None and tk_ms <= flashinfer_ms * 2.0)
        else "inspect_tk_exact_scheduling_or_flashinfer_scheduler_integration",
    }
    text = json.dumps(output, indent=2, sort_keys=True)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
