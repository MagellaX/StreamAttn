"""Benchmark StreamAttn's transposed Hopper WGMMA exact-decode path.

This is the first implementation gate for the exact-native transposed dataflow:

    K_tile [64, 64] @ Q_group.T [64, 8] -> scores [64, 8]

The context axis maps to WGMMA M=64 and a true-GQA group maps to N=8. Unlike
the earlier ThunderKittens spike, this uses the native m64n8k16 atom and does
not pad the query group to 16 rows. The benchmark retains isolated QK and QK+PV
floors, then measures the complete online-softmax partial and exact LSE merge.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_stream_attn_gate0_wrapper import _error, _time_cuda  # noqa: E402

try:
    import flashinfer

    FLASHINFER_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on benchmark environment
    flashinfer = None
    FLASHINFER_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


CPP_SOURCE = r"""
#include <torch/extension.h>

void streamattn_transposed_wgmma_qk_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_cache,
    torch::Tensor scores,
    int64_t num_splits);

void streamattn_transposed_wgmma_qk_checksum_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_cache,
    torch::Tensor checksums,
    int64_t num_splits);

void streamattn_transposed_wgmma_qk_async_checksum_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_cache,
    torch::Tensor checksums,
    int64_t num_splits);

void streamattn_transposed_wgmma_qkpv_async_checksum_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor checksums,
    int64_t num_splits);

void streamattn_transposed_wgmma_exact_partial_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor partial_o,
    torch::Tensor partial_lse,
    int64_t num_splits);

void streamattn_transposed_wgmma_exact_merge_out_cuda(
    torch::Tensor partial_o,
    torch::Tensor partial_lse,
    torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qk_out", &streamattn_transposed_wgmma_qk_out_cuda,
        "StreamAttn transposed m64n8k16 exact QK (out variant)");
  m.def("qk_checksum_out", &streamattn_transposed_wgmma_qk_checksum_out_cuda,
        "StreamAttn transposed m64n8k16 exact QK (storeless checksum variant)");
  m.def("qk_async_checksum_out", &streamattn_transposed_wgmma_qk_async_checksum_out_cuda,
        "StreamAttn transposed m64n8k16 exact QK (cp.async double-buffer checksum variant)");
  m.def("qkpv_async_checksum_out", &streamattn_transposed_wgmma_qkpv_async_checksum_out_cuda,
        "StreamAttn transposed m64n8k16 QK+PV floor (cp.async checksum variant)");
  m.def("exact_partial_out", &streamattn_transposed_wgmma_exact_partial_out_cuda,
        "StreamAttn transposed m64n8k16 exact attention partial states");
  m.def("exact_merge_out", &streamattn_transposed_wgmma_exact_merge_out_cuda,
        "StreamAttn exact split-state merge");
}
"""


CUDA_SOURCE = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

using namespace cute;

using Element = cutlass::bfloat16_t;
using Accum = float;

static constexpr int kBlockM = 64;
static constexpr int kBlockN = 8;
static constexpr int kHeadDim = 64;

using SmemLayoutK = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<Element>{},
    Shape<Int<kBlockM>, Int<kHeadDim>>{}));
using SmemLayoutQ = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<Element>{},
    Shape<Int<kBlockN>, Int<kHeadDim>>{}));
using SmemLayoutK2 = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<Element>{},
    Shape<Int<kBlockM>, Int<kHeadDim>, _2>{}));
using TiledMma = decltype(make_tiled_mma(
    SM90_64x8x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{}));
using TiledMmaO = decltype(make_tiled_mma(
    SM90_64x8x16_F32BF16BF16_SS<GMMA::Major::MN, GMMA::Major::MN>{}));
using SmemLayoutV = SmemLayoutK;
using SmemLayoutVt = decltype(composition(
    SmemLayoutV{},
    make_layout(Shape<Int<kHeadDim>, Int<kBlockM>>{}, GenRowMajor{})));
// PV consumes P as an [N=8,K=64] Major-MN operand.  The wider swizzled
// atoms require N >= 16/32/64, while the interleaved atom is canonical for
// the native m64n8 WGMMA.  Expose a transposed [M=64,N=8] alias so the QK
// accumulator can write the exact same storage without materializing a
// second score tile.
using SmemLayoutP = decltype(tile_to_shape(
    GMMA::Layout_MN_INTER_Atom<Element>{},
    Shape<Int<kBlockN>, Int<kBlockM>>{}));
using SmemLayoutPOrigin = decltype(composition(
    SmemLayoutP{},
    make_layout(Shape<Int<kBlockM>, Int<kBlockN>>{}, GenRowMajor{})));

struct alignas(128) SharedStorage {
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> k;
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> q;
};

struct alignas(128) AsyncSharedStorage {
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutK2>> k;
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> q;
};

struct alignas(128) AsyncQKPVSharedStorage {
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> k0;
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> k1;
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> v0;
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> v1;
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> q;
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutPOrigin>> p;
  Accum row_reduce[4][kBlockN];
  Accum row_max[kBlockN];
  Accum row_sum[kBlockN];
};

using GmemCopyK = decltype(make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>{},
    Layout<Shape<_16, _8>, Stride<_8, _1>>{},
    Layout<Shape<_1, _8>>{}));

template <typename To, typename Engine, typename Layout>
__forceinline__ __device__ auto streamattn_convert_type(Tensor<Engine, Layout> const& tensor) {
  using From = typename Engine::value_type;
  constexpr int numel = decltype(size(tensor))::value;
  cutlass::NumericArrayConverter<To, From, numel> convert_op;
  auto fragment = convert_op(
      *reinterpret_cast<const cutlass::Array<From, numel>*>(tensor.data()));
  return make_tensor(make_rmem_ptr<To>(&fragment), tensor.layout());
}

template <bool Transposed = false, typename Layout0>
__forceinline__ __device__ auto streamattn_acc_rowcol(Layout0 layout) {
  static_assert(decltype(rank<0>(layout))::value == 3);
  static_assert(decltype(size<0, 0>(layout))::value == 2);
  static_assert(decltype(size<0, 1>(layout))::value == 2);
  if constexpr (!Transposed) {
    return make_layout(
        make_layout(get<0, 1>(layout), get<1>(layout)),
        make_layout(get<0, 0>(layout), get<0, 2>(layout), get<2>(layout)));
  } else {
    return make_layout(
        make_layout(get<0, 0>(layout), get<0, 2>(layout), get<2>(layout)),
        make_layout(get<0, 1>(layout), get<1>(layout)));
  }
}

__forceinline__ __device__ Accum streamattn_group_max(Accum value) {
  const int group = (threadIdx.x & 31) & 3;
  const unsigned mask = 0x11111111u << group;
  value = fmaxf(value, __shfl_xor_sync(mask, value, 16));
  value = fmaxf(value, __shfl_xor_sync(mask, value, 8));
  value = fmaxf(value, __shfl_xor_sync(mask, value, 4));
  return value;
}

__forceinline__ __device__ Accum streamattn_group_sum(Accum value) {
  const int group = (threadIdx.x & 31) & 3;
  const unsigned mask = 0x11111111u << group;
  value += __shfl_xor_sync(mask, value, 16);
  value += __shfl_xor_sync(mask, value, 8);
  value += __shfl_xor_sync(mask, value, 4);
  return value;
}

__global__ __launch_bounds__(128)
void streamattn_transposed_wgmma_qk_kernel(
    const Element* __restrict__ q_group,
    const Element* __restrict__ k_cache,
    Accum* __restrict__ scores,
    int groups,
    int kv_len,
    int num_splits) {
  const int work = blockIdx.x;
  const int group = work / num_splits;
  const int split = work - group * num_splits;
  if (group >= groups) {
    return;
  }

  __shared__ SharedStorage storage;
  Tensor sK = make_tensor(make_smem_ptr(storage.k.data()), SmemLayoutK{});
  Tensor sQ = make_tensor(make_smem_ptr(storage.q.data()), SmemLayoutQ{});

  const Element* q_ptr = q_group + static_cast<int64_t>(group) * kBlockN * kHeadDim;
  for (int idx = threadIdx.x; idx < kBlockN * kHeadDim; idx += blockDim.x) {
    const int row = idx / kHeadDim;
    const int col = idx - row * kHeadDim;
    sQ(row, col) = q_ptr[idx];
  }
  cutlass::arch::fence_view_async_shared();
  __syncthreads();

  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
  Tensor tSrK = thr_mma.partition_fragment_A(sK);
  Tensor tSrQ = thr_mma.partition_fragment_B(sQ);

  const int num_tiles = kv_len / kBlockM;
  const int tiles_per_split = (num_tiles + num_splits - 1) / num_splits;
  const int tile_begin = split * tiles_per_split;
  const int tile_end = min(num_tiles, tile_begin + tiles_per_split);

  for (int tile = tile_begin; tile < tile_end; ++tile) {
    const int token_begin = tile * kBlockM;
    const Element* k_ptr = k_cache
        + static_cast<int64_t>(group) * kv_len * kHeadDim
        + static_cast<int64_t>(token_begin) * kHeadDim;

    for (int idx = threadIdx.x; idx < kBlockM * kHeadDim; idx += blockDim.x) {
      const int row = idx / kHeadDim;
      const int col = idx - row * kHeadDim;
      sK(row, col) = k_ptr[idx];
    }
    cutlass::arch::fence_view_async_shared();
    __syncthreads();

    Tensor tCrS = partition_fragment_C(
        tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    clear(tCrS);
    warpgroup_fence_operand(tCrS);
    warpgroup_arrive();
    cute::gemm(tiled_mma, tSrK, tSrQ, tCrS);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrS);

    Accum* score_ptr = scores
        + static_cast<int64_t>(group) * kv_len * kBlockN
        + static_cast<int64_t>(token_begin) * kBlockN;
    Tensor gS = make_tensor(
        make_gmem_ptr(score_ptr),
        Shape<Int<kBlockM>, Int<kBlockN>>{},
        make_stride(Int<kBlockN>{}, _1{}));
    Tensor tSgS = thr_mma.partition_C(gS);
    cute::copy(tCrS, tSgS);
    __syncthreads();
  }
}

__global__ __launch_bounds__(128)
void streamattn_transposed_wgmma_qk_checksum_kernel(
    const Element* __restrict__ q_group,
    const Element* __restrict__ k_cache,
    Accum* __restrict__ checksums,
    int groups,
    int kv_len,
    int num_splits) {
  const int work = blockIdx.x;
  const int group = work / num_splits;
  const int split = work - group * num_splits;
  if (group >= groups) {
    return;
  }

  __shared__ SharedStorage storage;
  __shared__ Accum reduction[128];
  Tensor sK = make_tensor(make_smem_ptr(storage.k.data()), SmemLayoutK{});
  Tensor sQ = make_tensor(make_smem_ptr(storage.q.data()), SmemLayoutQ{});

  const Element* q_ptr = q_group + static_cast<int64_t>(group) * kBlockN * kHeadDim;
  for (int idx = threadIdx.x; idx < kBlockN * kHeadDim; idx += blockDim.x) {
    const int row = idx / kHeadDim;
    const int col = idx - row * kHeadDim;
    sQ(row, col) = q_ptr[idx];
  }
  cutlass::arch::fence_view_async_shared();
  __syncthreads();

  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
  Tensor tSrK = thr_mma.partition_fragment_A(sK);
  Tensor tSrQ = thr_mma.partition_fragment_B(sQ);

  const int num_tiles = kv_len / kBlockM;
  const int tiles_per_split = (num_tiles + num_splits - 1) / num_splits;
  const int tile_begin = split * tiles_per_split;
  const int tile_end = min(num_tiles, tile_begin + tiles_per_split);
  Accum local_sum = 0.0f;

  for (int tile = tile_begin; tile < tile_end; ++tile) {
    const int token_begin = tile * kBlockM;
    const Element* k_ptr = k_cache
        + static_cast<int64_t>(group) * kv_len * kHeadDim
        + static_cast<int64_t>(token_begin) * kHeadDim;
    for (int idx = threadIdx.x; idx < kBlockM * kHeadDim; idx += blockDim.x) {
      const int row = idx / kHeadDim;
      const int col = idx - row * kHeadDim;
      sK(row, col) = k_ptr[idx];
    }
    cutlass::arch::fence_view_async_shared();
    __syncthreads();

    Tensor tCrS = partition_fragment_C(
        tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    clear(tCrS);
    warpgroup_fence_operand(tCrS);
    warpgroup_arrive();
    cute::gemm(tiled_mma, tSrK, tSrQ, tCrS);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrS);
    CUTE_UNROLL
    for (int idx = 0; idx < size(tCrS); ++idx) {
      local_sum += tCrS(idx);
    }
  }

  reduction[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = 64; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    checksums[static_cast<int64_t>(group) * num_splits + split] = reduction[0];
  }
}

__global__ __launch_bounds__(128)
void streamattn_transposed_wgmma_qk_async_checksum_kernel(
    const Element* __restrict__ q_group,
    const Element* __restrict__ k_cache,
    Accum* __restrict__ checksums,
    int groups,
    int kv_len,
    int num_splits) {
  const int work = blockIdx.x;
  const int group = work / num_splits;
  const int split = work - group * num_splits;
  if (group >= groups) {
    return;
  }

  __shared__ AsyncSharedStorage storage;
  __shared__ Accum reduction[128];
  Tensor sK = make_tensor(make_smem_ptr(storage.k.data()), SmemLayoutK2{});
  Tensor sQ = make_tensor(make_smem_ptr(storage.q.data()), SmemLayoutQ{});

  const Element* q_ptr = q_group + static_cast<int64_t>(group) * kBlockN * kHeadDim;
  for (int idx = threadIdx.x; idx < kBlockN * kHeadDim; idx += blockDim.x) {
    const int row = idx / kHeadDim;
    const int col = idx - row * kHeadDim;
    sQ(row, col) = q_ptr[idx];
  }
  cutlass::arch::fence_view_async_shared();
  __syncthreads();

  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
  Tensor tSrK = thr_mma.partition_fragment_A(sK);
  Tensor tSrQ = thr_mma.partition_fragment_B(sQ);
  GmemCopyK copy_k;
  auto thr_copy_k = copy_k.get_thread_slice(threadIdx.x);
  Tensor tKsK = thr_copy_k.partition_D(sK);

  const int num_tiles = kv_len / kBlockM;
  const int tiles_per_split = (num_tiles + num_splits - 1) / num_splits;
  const int tile_begin = split * tiles_per_split;
  const int tile_end = min(num_tiles, tile_begin + tiles_per_split);
  const Element* group_k = k_cache + static_cast<int64_t>(group) * kv_len * kHeadDim;
  Accum local_sum = 0.0f;

  if (tile_begin < tile_end) {
    const Element* first_k = group_k + static_cast<int64_t>(tile_begin) * kBlockM * kHeadDim;
    Tensor gK = make_tensor(
        make_gmem_ptr(first_k),
        Shape<Int<kBlockM>, Int<kHeadDim>>{},
        make_stride(Int<kHeadDim>{}, _1{}));
    Tensor tKgK = thr_copy_k.partition_S(gK);
    cute::copy(copy_k, tKgK, tKsK(_, _, _, 0));
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();
  }

  int read_pipe = 0;
  for (int tile = tile_begin; tile < tile_end; ++tile) {
    const int next_tile = tile + 1;
    const int write_pipe = read_pipe ^ 1;
    if (next_tile < tile_end) {
      const Element* next_k = group_k + static_cast<int64_t>(next_tile) * kBlockM * kHeadDim;
      Tensor gKNext = make_tensor(
          make_gmem_ptr(next_k),
          Shape<Int<kBlockM>, Int<kHeadDim>>{},
          make_stride(Int<kHeadDim>{}, _1{}));
      Tensor tKgKNext = thr_copy_k.partition_S(gKNext);
      cute::copy(copy_k, tKgKNext, tKsK(_, _, _, write_pipe));
      cute::cp_async_fence();
    }

    Tensor tCrS = partition_fragment_C(
        tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    clear(tCrS);
    warpgroup_fence_operand(tCrS);
    warpgroup_arrive();
    cute::gemm(tiled_mma, tSrK(_, _, _, read_pipe), tSrQ, tCrS);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrS);
    CUTE_UNROLL
    for (int idx = 0; idx < size(tCrS); ++idx) {
      local_sum += tCrS(idx);
    }

    if (next_tile < tile_end) {
      cute::cp_async_wait<0>();
      __syncthreads();
      read_pipe = write_pipe;
    }
  }

  reduction[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = 64; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    checksums[static_cast<int64_t>(group) * num_splits + split] = reduction[0];
  }
}

__global__ __launch_bounds__(128)
void streamattn_transposed_wgmma_qkpv_async_checksum_kernel(
    const Element* __restrict__ q_group,
    const Element* __restrict__ k_cache,
    const Element* __restrict__ v_cache,
    Accum* __restrict__ checksums,
    int groups,
    int kv_len,
    int num_splits) {
  const int work = blockIdx.x;
  const int group = work / num_splits;
  const int split = work - group * num_splits;
  if (group >= groups) {
    return;
  }

  const int num_tiles = kv_len / kBlockM;
  const int tiles_per_split = (num_tiles + num_splits - 1) / num_splits;
  const int tile_begin = split * tiles_per_split;
  const int tile_end = min(num_tiles, tile_begin + tiles_per_split);
  if (tile_begin >= tile_end) {
    if (threadIdx.x == 0) {
      checksums[static_cast<int64_t>(group) * num_splits + split] = 0.0f;
    }
    return;
  }

  __shared__ AsyncQKPVSharedStorage storage;
  __shared__ Accum reduction[128];
  Tensor sK0 = make_tensor(make_smem_ptr(storage.k0.data()), SmemLayoutK{});
  Tensor sK1 = make_tensor(make_smem_ptr(storage.k1.data()), SmemLayoutK{});
  Tensor sV0 = make_tensor(make_smem_ptr(storage.v0.data()), SmemLayoutV{});
  Tensor sV1 = make_tensor(make_smem_ptr(storage.v1.data()), SmemLayoutV{});
  Tensor sVt0 = make_tensor(make_smem_ptr(storage.v0.data()), SmemLayoutVt{});
  Tensor sVt1 = make_tensor(make_smem_ptr(storage.v1.data()), SmemLayoutVt{});
  Tensor sQ = make_tensor(make_smem_ptr(storage.q.data()), SmemLayoutQ{});
  Tensor sPOrigin = make_tensor(make_smem_ptr(storage.p.data()), SmemLayoutPOrigin{});
  Tensor sP = make_tensor(make_smem_ptr(storage.p.data()), SmemLayoutP{});

  const Element* q_ptr = q_group + static_cast<int64_t>(group) * kBlockN * kHeadDim;
  for (int idx = threadIdx.x; idx < kBlockN * kHeadDim; idx += blockDim.x) {
    const int row = idx / kHeadDim;
    const int col = idx - row * kHeadDim;
    sQ(row, col) = q_ptr[idx];
  }
  cutlass::arch::fence_view_async_shared();
  __syncthreads();

  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
  Tensor tSrK0 = thr_mma.partition_fragment_A(sK0);
  Tensor tSrK1 = thr_mma.partition_fragment_A(sK1);
  Tensor tSrQ = thr_mma.partition_fragment_B(sQ);
  Tensor tPsP = thr_mma.partition_C(sPOrigin);

  TiledMmaO tiled_mma_o;
  auto thr_mma_o = tiled_mma_o.get_thread_slice(threadIdx.x);
  Tensor tOrV0 = thr_mma_o.partition_fragment_A(sVt0);
  Tensor tOrV1 = thr_mma_o.partition_fragment_A(sVt1);
  Tensor tOrP = thr_mma_o.partition_fragment_B(sP);
  Tensor tOrO = partition_fragment_C(
      tiled_mma_o, Shape<Int<kHeadDim>, Int<kBlockN>>{});
  clear(tOrO);

  GmemCopyK copy_kv;
  auto thr_copy_kv = copy_kv.get_thread_slice(threadIdx.x);
  Tensor tK0sK0 = thr_copy_kv.partition_D(sK0);
  Tensor tK1sK1 = thr_copy_kv.partition_D(sK1);
  Tensor tV0sV0 = thr_copy_kv.partition_D(sV0);
  Tensor tV1sV1 = thr_copy_kv.partition_D(sV1);

  const Element* group_k = k_cache + static_cast<int64_t>(group) * kv_len * kHeadDim;
  const Element* group_v = v_cache + static_cast<int64_t>(group) * kv_len * kHeadDim;

  if (tile_begin < tile_end) {
    const Element* first_k = group_k + static_cast<int64_t>(tile_begin) * kBlockM * kHeadDim;
    const Element* first_v = group_v + static_cast<int64_t>(tile_begin) * kBlockM * kHeadDim;
    Tensor gK = make_tensor(make_gmem_ptr(first_k), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(Int<kHeadDim>{}, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(first_v), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(Int<kHeadDim>{}, _1{}));
    Tensor tKgK = thr_copy_kv.partition_S(gK);
    Tensor tVgV = thr_copy_kv.partition_S(gV);
    cute::copy(copy_kv, tKgK, tK0sK0);
    cute::copy(copy_kv, tVgV, tV0sV0);
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();
  }

  int read_pipe = 0;
  for (int tile = tile_begin; tile < tile_end; ++tile) {
    const int next_tile = tile + 1;
    const int write_pipe = read_pipe ^ 1;
    if (next_tile < tile_end) {
      const Element* next_k = group_k + static_cast<int64_t>(next_tile) * kBlockM * kHeadDim;
      const Element* next_v = group_v + static_cast<int64_t>(next_tile) * kBlockM * kHeadDim;
      Tensor gKNext = make_tensor(make_gmem_ptr(next_k), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                  make_stride(Int<kHeadDim>{}, _1{}));
      Tensor gVNext = make_tensor(make_gmem_ptr(next_v), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                  make_stride(Int<kHeadDim>{}, _1{}));
      Tensor tKgKNext = thr_copy_kv.partition_S(gKNext);
      Tensor tVgVNext = thr_copy_kv.partition_S(gVNext);
      if (write_pipe == 0) {
        cute::copy(copy_kv, tKgKNext, tK0sK0);
        cute::copy(copy_kv, tVgVNext, tV0sV0);
      } else {
        cute::copy(copy_kv, tKgKNext, tK1sK1);
        cute::copy(copy_kv, tVgVNext, tV1sV1);
      }
      cute::cp_async_fence();
    }

    Tensor tCrS = partition_fragment_C(
        tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    clear(tCrS);
    warpgroup_fence_operand(tCrS);
    warpgroup_arrive();
    if (read_pipe == 0) {
      cute::gemm(tiled_mma, tSrK0, tSrQ, tCrS);
    } else {
      cute::gemm(tiled_mma, tSrK1, tSrQ, tCrS);
    }
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrS);

    Tensor rP = streamattn_convert_type<Element>(tCrS);
    cute::copy(rP, tPsP);
    cutlass::arch::fence_view_async_shared();
    __syncthreads();

    warpgroup_fence_operand(tOrO);
    warpgroup_arrive();
    if (read_pipe == 0) {
      cute::gemm(tiled_mma_o, tOrV0, tOrP, tOrO);
    } else {
      cute::gemm(tiled_mma_o, tOrV1, tOrP, tOrO);
    }
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tOrO);

    if (next_tile < tile_end) {
      cute::cp_async_wait<0>();
      __syncthreads();
      read_pipe = write_pipe;
    }
  }

  Accum local_sum = 0.0f;
  CUTE_UNROLL
  for (int idx = 0; idx < size(tOrO); ++idx) {
    local_sum += tOrO(idx);
  }
  reduction[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = 64; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    checksums[static_cast<int64_t>(group) * num_splits + split] = reduction[0];
  }
}

__global__ __launch_bounds__(128)
void streamattn_transposed_wgmma_exact_partial_kernel(
    const Element* __restrict__ q_group,
    const Element* __restrict__ k_cache,
    const Element* __restrict__ v_cache,
    Accum* __restrict__ partial_o,
    Accum* __restrict__ partial_lse,
    int groups,
    int kv_len,
    int num_splits) {
  const int work = blockIdx.x;
  const int group = work / num_splits;
  const int split = work - group * num_splits;
  if (group >= groups) {
    return;
  }

  const int num_tiles = kv_len / kBlockM;
  const int tiles_per_split = (num_tiles + num_splits - 1) / num_splits;
  const int tile_begin = split * tiles_per_split;
  const int tile_end = min(num_tiles, tile_begin + tiles_per_split);
  if (tile_begin >= tile_end) {
    for (int idx = threadIdx.x; idx < kBlockN * kHeadDim; idx += blockDim.x) {
      partial_o[(static_cast<int64_t>(work) * kBlockN * kHeadDim) + idx] = 0.0f;
    }
    if (threadIdx.x < kBlockN) {
      partial_lse[static_cast<int64_t>(work) * kBlockN + threadIdx.x] = -INFINITY;
    }
    return;
  }

  __shared__ AsyncQKPVSharedStorage storage;
  Tensor sK0 = make_tensor(make_smem_ptr(storage.k0.data()), SmemLayoutK{});
  Tensor sK1 = make_tensor(make_smem_ptr(storage.k1.data()), SmemLayoutK{});
  Tensor sV0 = make_tensor(make_smem_ptr(storage.v0.data()), SmemLayoutV{});
  Tensor sV1 = make_tensor(make_smem_ptr(storage.v1.data()), SmemLayoutV{});
  Tensor sVt0 = make_tensor(make_smem_ptr(storage.v0.data()), SmemLayoutVt{});
  Tensor sVt1 = make_tensor(make_smem_ptr(storage.v1.data()), SmemLayoutVt{});
  Tensor sQ = make_tensor(make_smem_ptr(storage.q.data()), SmemLayoutQ{});
  Tensor sPOrigin = make_tensor(make_smem_ptr(storage.p.data()), SmemLayoutPOrigin{});
  Tensor sP = make_tensor(make_smem_ptr(storage.p.data()), SmemLayoutP{});

  const Element* q_ptr = q_group + static_cast<int64_t>(group) * kBlockN * kHeadDim;
  for (int idx = threadIdx.x; idx < kBlockN * kHeadDim; idx += blockDim.x) {
    sQ(idx / kHeadDim, idx % kHeadDim) = q_ptr[idx];
  }
  cutlass::arch::fence_view_async_shared();
  __syncthreads();

  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
  Tensor tSrK0 = thr_mma.partition_fragment_A(sK0);
  Tensor tSrK1 = thr_mma.partition_fragment_A(sK1);
  Tensor tSrQ = thr_mma.partition_fragment_B(sQ);
  Tensor tPsP = thr_mma.partition_C(sPOrigin);
  Tensor cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
  Tensor tScS = thr_mma.partition_C(cS);
  Tensor tScSRowCol = make_tensor(
      tScS.data(), streamattn_acc_rowcol<true>(tScS.layout()));

  TiledMmaO tiled_mma_o;
  auto thr_mma_o = tiled_mma_o.get_thread_slice(threadIdx.x);
  Tensor tOrV0 = thr_mma_o.partition_fragment_A(sVt0);
  Tensor tOrV1 = thr_mma_o.partition_fragment_A(sVt1);
  Tensor tOrP = thr_mma_o.partition_fragment_B(sP);
  Tensor tOrO = partition_fragment_C(
      tiled_mma_o, Shape<Int<kHeadDim>, Int<kBlockN>>{});
  Tensor tOrORowCol = make_tensor(
      tOrO.data(), streamattn_acc_rowcol<true>(tOrO.layout()));
  Tensor cO = make_identity_tensor(Shape<Int<kHeadDim>, Int<kBlockN>>{});
  Tensor tOcO = thr_mma_o.partition_C(cO);
  clear(tOrO);

  constexpr int kRowsPerThread = decltype(size<0>(tOrORowCol))::value;
  static_assert(kRowsPerThread == 2, "m64n8 output must expose two query rows per thread");
  Accum row_max[kRowsPerThread];
  Accum row_sum[kRowsPerThread];
  CUTE_UNROLL
  for (int row = 0; row < kRowsPerThread; ++row) {
    row_max[row] = -INFINITY;
    row_sum[row] = 0.0f;
  }

  GmemCopyK copy_kv;
  auto thr_copy_kv = copy_kv.get_thread_slice(threadIdx.x);
  Tensor tK0sK0 = thr_copy_kv.partition_D(sK0);
  Tensor tK1sK1 = thr_copy_kv.partition_D(sK1);
  Tensor tV0sV0 = thr_copy_kv.partition_D(sV0);
  Tensor tV1sV1 = thr_copy_kv.partition_D(sV1);

  const Element* group_k = k_cache + static_cast<int64_t>(group) * kv_len * kHeadDim;
  const Element* group_v = v_cache + static_cast<int64_t>(group) * kv_len * kHeadDim;

  if (tile_begin < tile_end) {
    const Element* first_k = group_k + static_cast<int64_t>(tile_begin) * kBlockM * kHeadDim;
    const Element* first_v = group_v + static_cast<int64_t>(tile_begin) * kBlockM * kHeadDim;
    Tensor gK = make_tensor(make_gmem_ptr(first_k), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(Int<kHeadDim>{}, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(first_v), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(Int<kHeadDim>{}, _1{}));
    cute::copy(copy_kv, thr_copy_kv.partition_S(gK), tK0sK0);
    cute::copy(copy_kv, thr_copy_kv.partition_S(gV), tV0sV0);
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();
  }

  constexpr Accum kSoftmaxScaleLog2 = 0.18033688011112042f;
  int read_pipe = 0;
  for (int tile = tile_begin; tile < tile_end; ++tile) {
    const int next_tile = tile + 1;
    const int write_pipe = read_pipe ^ 1;
    if (next_tile < tile_end) {
      const Element* next_k = group_k + static_cast<int64_t>(next_tile) * kBlockM * kHeadDim;
      const Element* next_v = group_v + static_cast<int64_t>(next_tile) * kBlockM * kHeadDim;
      Tensor gKNext = make_tensor(make_gmem_ptr(next_k), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                  make_stride(Int<kHeadDim>{}, _1{}));
      Tensor gVNext = make_tensor(make_gmem_ptr(next_v), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                  make_stride(Int<kHeadDim>{}, _1{}));
      if (write_pipe == 0) {
        cute::copy(copy_kv, thr_copy_kv.partition_S(gKNext), tK0sK0);
        cute::copy(copy_kv, thr_copy_kv.partition_S(gVNext), tV0sV0);
      } else {
        cute::copy(copy_kv, thr_copy_kv.partition_S(gKNext), tK1sK1);
        cute::copy(copy_kv, thr_copy_kv.partition_S(gVNext), tV1sV1);
      }
      cute::cp_async_fence();
    }

    Tensor tCrS = partition_fragment_C(
        tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    clear(tCrS);
    warpgroup_fence_operand(tCrS);
    warpgroup_arrive();
    if (read_pipe == 0) {
      cute::gemm(tiled_mma, tSrK0, tSrQ, tCrS);
    } else {
      cute::gemm(tiled_mma, tSrK1, tSrQ, tCrS);
    }
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrS);

    Tensor scores = make_tensor(
        tCrS.data(), streamattn_acc_rowcol<true>(tCrS.layout()));
    Accum scale_o[kRowsPerThread];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int group_row = lane >> 2;
    CUTE_UNROLL
    for (int row = 0; row < kRowsPerThread; ++row) {
      Accum tile_max = -INFINITY;
      CUTE_UNROLL
      for (int col = 0; col < size<1>(scores); ++col) {
        tile_max = fmaxf(tile_max, scores(row, col));
      }
      tile_max = streamattn_group_max(tile_max);
      const int head = int(get<1>(tScSRowCol(row, 0)));
      if (group_row == 0) {
        storage.row_reduce[warp][head] = tile_max;
      }
    }
    __syncthreads();

    CUTE_UNROLL
    for (int row = 0; row < kRowsPerThread; ++row) {
      const int head = int(get<1>(tScSRowCol(row, 0)));
      Accum tile_max = storage.row_reduce[0][head];
      CUTE_UNROLL
      for (int warp_idx = 1; warp_idx < 4; ++warp_idx) {
        tile_max = fmaxf(tile_max, storage.row_reduce[warp_idx][head]);
      }
      const Accum next_max = fmaxf(row_max[row], tile_max);
      const Accum alpha = row_max[row] == -INFINITY
          ? 0.0f
          : exp2f((row_max[row] - next_max) * kSoftmaxScaleLog2);
      row_max[row] = next_max;
      row_sum[row] *= alpha;
      scale_o[row] = alpha;

      Accum local_sum = 0.0f;
      const Accum max_scaled = next_max * kSoftmaxScaleLog2;
      CUTE_UNROLL
      for (int col = 0; col < size<1>(scores); ++col) {
        const Accum probability = exp2f(
            scores(row, col) * kSoftmaxScaleLog2 - max_scaled);
        scores(row, col) = probability;
        local_sum += probability;
      }
      row_sum[row] += local_sum;
      CUTE_UNROLL
      for (int col = 0; col < size<1>(tOrORowCol); ++col) {
        tOrORowCol(row, col) *= alpha;
      }
    }

    Tensor rP = streamattn_convert_type<Element>(tCrS);
    cute::copy(rP, tPsP);
    cutlass::arch::fence_view_async_shared();
    __syncthreads();

    warpgroup_fence_operand(tOrO);
    warpgroup_arrive();
    if (read_pipe == 0) {
      cute::gemm(tiled_mma_o, tOrV0, tOrP, tOrO);
    } else {
      cute::gemm(tiled_mma_o, tOrV1, tOrP, tOrO);
    }
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tOrO);

    if (next_tile < tile_end) {
      cute::cp_async_wait<0>();
      __syncthreads();
      read_pipe = write_pipe;
    }
  }

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int group_row = lane >> 2;
  CUTE_UNROLL
  for (int row = 0; row < kRowsPerThread; ++row) {
    const int head = int(get<1>(tScSRowCol(row, 0)));
    Accum total = streamattn_group_sum(row_sum[row]);
    if (group_row == 0) {
      storage.row_reduce[warp][head] = total;
    }
  }
  __syncthreads();
  if (warp == 0 && group_row == 0) {
    CUTE_UNROLL
    for (int row = 0; row < kRowsPerThread; ++row) {
      const int head = int(get<1>(tScSRowCol(row, 0)));
      Accum total = 0.0f;
      CUTE_UNROLL
      for (int warp_idx = 0; warp_idx < 4; ++warp_idx) {
        total += storage.row_reduce[warp_idx][head];
      }
      storage.row_max[head] = row_max[row];
      storage.row_sum[head] = total;
      partial_lse[(static_cast<int64_t>(work) * kBlockN) + head] =
          row_max[row] * 0.125f + logf(total);
    }
  }
  __syncthreads();

  Tensor tOcORowCol = make_tensor(
      tOcO.data(), streamattn_acc_rowcol<true>(tOcO.layout()));
  CUTE_UNROLL
  for (int row = 0; row < size<0>(tOrORowCol); ++row) {
    CUTE_UNROLL
    for (int col = 0; col < size<1>(tOrORowCol); ++col) {
      const auto coord = tOcORowCol(row, col);
      const int dim = int(get<0>(coord));
      const int head = int(get<1>(coord));
      partial_o[((static_cast<int64_t>(work) * kBlockN + head) * kHeadDim) + dim] =
          tOrORowCol(row, col) / storage.row_sum[head];
    }
  }
}

__global__ __launch_bounds__(64)
void streamattn_transposed_wgmma_exact_merge_kernel(
    const Accum* __restrict__ partial_o,
    const Accum* __restrict__ partial_lse,
    Element* __restrict__ output,
    int groups,
    int num_splits) {
  const int row = blockIdx.x;
  if (row >= groups * kBlockN) {
    return;
  }
  const int group = row / kBlockN;
  const int head = row - group * kBlockN;
  const int lane = threadIdx.x & 31;
  __shared__ Accum weights[512];
  __shared__ Accum normalizer;
  __shared__ Accum row_max;

  Accum local_max = -INFINITY;
  if (threadIdx.x < 32) {
    for (int split = lane; split < num_splits; split += 32) {
      local_max = fmaxf(
          local_max,
          partial_lse[(static_cast<int64_t>(group) * num_splits + split) * kBlockN + head]);
    }
    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, 16));
    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, 8));
    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, 4));
    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, 2));
    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, 1));
    if (lane == 0) {
      row_max = local_max;
    }
  }
  __syncthreads();

  Accum local_sum = 0.0f;
  if (threadIdx.x < 32) {
    for (int split = lane; split < num_splits; split += 32) {
      const Accum weight = __expf(
          partial_lse[(static_cast<int64_t>(group) * num_splits + split) * kBlockN + head]
          - row_max);
      weights[split] = weight;
      local_sum += weight;
    }
    local_sum += __shfl_xor_sync(0xffffffffu, local_sum, 16);
    local_sum += __shfl_xor_sync(0xffffffffu, local_sum, 8);
    local_sum += __shfl_xor_sync(0xffffffffu, local_sum, 4);
    local_sum += __shfl_xor_sync(0xffffffffu, local_sum, 2);
    local_sum += __shfl_xor_sync(0xffffffffu, local_sum, 1);
    if (lane == 0) {
      normalizer = local_sum;
    }
  }
  __syncthreads();

  const int dim = threadIdx.x;
  Accum value = 0.0f;
  for (int split = 0; split < num_splits; ++split) {
    value += weights[split] * partial_o[
        ((static_cast<int64_t>(group) * num_splits + split) * kBlockN + head)
        * kHeadDim + dim];
  }
  output[static_cast<int64_t>(row) * kHeadDim + dim] = Element(value / normalizer);
}

void streamattn_transposed_wgmma_qk_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_cache,
    torch::Tensor scores,
    int64_t num_splits) {
  TORCH_CHECK(q_group.is_cuda() && k_cache.is_cuda() && scores.is_cuda(),
              "q_group, k_cache, and scores must be CUDA tensors");
  TORCH_CHECK(q_group.is_contiguous() && k_cache.is_contiguous() && scores.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(q_group.scalar_type() == at::ScalarType::BFloat16,
              "q_group must be bf16");
  TORCH_CHECK(k_cache.scalar_type() == at::ScalarType::BFloat16,
              "k_cache must be bf16");
  TORCH_CHECK(scores.scalar_type() == at::ScalarType::Float,
              "scores must be fp32");
  TORCH_CHECK(q_group.dim() == 4 && q_group.size(2) == kBlockN && q_group.size(3) == kHeadDim,
              "q_group must have shape [B,Hkv,8,64]");
  TORCH_CHECK(k_cache.dim() == 4 && k_cache.size(3) == kHeadDim,
              "k_cache must have shape [B,Hkv,N,64]");
  TORCH_CHECK(k_cache.size(0) == q_group.size(0) && k_cache.size(1) == q_group.size(1),
              "q_group and k_cache batch/KV-head dimensions must match");
  TORCH_CHECK(k_cache.size(2) % kBlockM == 0,
              "kv_len must be divisible by 64");
  TORCH_CHECK(scores.sizes() == torch::IntArrayRef(
                  {q_group.size(0), q_group.size(1), k_cache.size(2), kBlockN}),
              "scores must have shape [B,Hkv,N,8]");

  const int groups = static_cast<int>(q_group.size(0) * q_group.size(1));
  const int kv_len = static_cast<int>(k_cache.size(2));
  const int num_tiles = kv_len / kBlockM;
  TORCH_CHECK(num_splits > 0 && num_splits <= num_tiles,
              "num_splits must be in [1, kv_len/64]");

  const dim3 grid(groups * static_cast<int>(num_splits));
  const dim3 block(128);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  streamattn_transposed_wgmma_qk_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const Element*>(q_group.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(k_cache.data_ptr<at::BFloat16>()),
      scores.data_ptr<float>(),
      groups,
      kv_len,
      static_cast<int>(num_splits));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void streamattn_transposed_wgmma_qk_checksum_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_cache,
    torch::Tensor checksums,
    int64_t num_splits) {
  TORCH_CHECK(q_group.is_cuda() && k_cache.is_cuda() && checksums.is_cuda(),
              "q_group, k_cache, and checksums must be CUDA tensors");
  TORCH_CHECK(q_group.is_contiguous() && k_cache.is_contiguous() && checksums.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(q_group.scalar_type() == at::ScalarType::BFloat16 &&
              k_cache.scalar_type() == at::ScalarType::BFloat16,
              "q_group and k_cache must be bf16");
  TORCH_CHECK(checksums.scalar_type() == at::ScalarType::Float,
              "checksums must be fp32");
  TORCH_CHECK(q_group.dim() == 4 && q_group.size(2) == kBlockN && q_group.size(3) == kHeadDim,
              "q_group must have shape [B,Hkv,8,64]");
  TORCH_CHECK(k_cache.dim() == 4 && k_cache.size(3) == kHeadDim,
              "k_cache must have shape [B,Hkv,N,64]");
  TORCH_CHECK(k_cache.size(0) == q_group.size(0) && k_cache.size(1) == q_group.size(1),
              "q_group and k_cache batch/KV-head dimensions must match");
  TORCH_CHECK(k_cache.size(2) % kBlockM == 0,
              "kv_len must be divisible by 64");

  const int groups = static_cast<int>(q_group.size(0) * q_group.size(1));
  const int kv_len = static_cast<int>(k_cache.size(2));
  const int num_tiles = kv_len / kBlockM;
  TORCH_CHECK(num_splits > 0 && num_splits <= num_tiles,
              "num_splits must be in [1, kv_len/64]");
  TORCH_CHECK(checksums.numel() == static_cast<int64_t>(groups) * num_splits,
              "checksums must have B*Hkv*num_splits elements");

  const dim3 grid(groups * static_cast<int>(num_splits));
  const dim3 block(128);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  streamattn_transposed_wgmma_qk_checksum_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const Element*>(q_group.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(k_cache.data_ptr<at::BFloat16>()),
      checksums.data_ptr<float>(),
      groups,
      kv_len,
      static_cast<int>(num_splits));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void streamattn_transposed_wgmma_qk_async_checksum_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_cache,
    torch::Tensor checksums,
    int64_t num_splits) {
  TORCH_CHECK(q_group.is_cuda() && k_cache.is_cuda() && checksums.is_cuda(),
              "q_group, k_cache, and checksums must be CUDA tensors");
  TORCH_CHECK(q_group.is_contiguous() && k_cache.is_contiguous() && checksums.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(q_group.scalar_type() == at::ScalarType::BFloat16 &&
              k_cache.scalar_type() == at::ScalarType::BFloat16,
              "q_group and k_cache must be bf16");
  TORCH_CHECK(checksums.scalar_type() == at::ScalarType::Float,
              "checksums must be fp32");
  TORCH_CHECK(q_group.dim() == 4 && q_group.size(2) == kBlockN && q_group.size(3) == kHeadDim,
              "q_group must have shape [B,Hkv,8,64]");
  TORCH_CHECK(k_cache.dim() == 4 && k_cache.size(3) == kHeadDim,
              "k_cache must have shape [B,Hkv,N,64]");
  TORCH_CHECK(k_cache.size(0) == q_group.size(0) && k_cache.size(1) == q_group.size(1),
              "q_group and k_cache batch/KV-head dimensions must match");
  TORCH_CHECK(k_cache.size(2) % kBlockM == 0,
              "kv_len must be divisible by 64");

  const int groups = static_cast<int>(q_group.size(0) * q_group.size(1));
  const int kv_len = static_cast<int>(k_cache.size(2));
  const int num_tiles = kv_len / kBlockM;
  TORCH_CHECK(num_splits > 0 && num_splits <= num_tiles,
              "num_splits must be in [1, kv_len/64]");
  TORCH_CHECK(checksums.numel() == static_cast<int64_t>(groups) * num_splits,
              "checksums must have B*Hkv*num_splits elements");

  const dim3 grid(groups * static_cast<int>(num_splits));
  const dim3 block(128);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  streamattn_transposed_wgmma_qk_async_checksum_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const Element*>(q_group.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(k_cache.data_ptr<at::BFloat16>()),
      checksums.data_ptr<float>(),
      groups,
      kv_len,
      static_cast<int>(num_splits));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void streamattn_transposed_wgmma_qkpv_async_checksum_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor checksums,
    int64_t num_splits) {
  TORCH_CHECK(q_group.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda() && checksums.is_cuda(),
              "all tensors must be CUDA tensors");
  TORCH_CHECK(q_group.is_contiguous() && k_cache.is_contiguous() &&
              v_cache.is_contiguous() && checksums.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(q_group.scalar_type() == at::ScalarType::BFloat16 &&
              k_cache.scalar_type() == at::ScalarType::BFloat16 &&
              v_cache.scalar_type() == at::ScalarType::BFloat16,
              "q_group, k_cache, and v_cache must be bf16");
  TORCH_CHECK(checksums.scalar_type() == at::ScalarType::Float,
              "checksums must be fp32");
  TORCH_CHECK(q_group.dim() == 4 && q_group.size(2) == kBlockN && q_group.size(3) == kHeadDim,
              "q_group must have shape [B,Hkv,8,64]");
  TORCH_CHECK(k_cache.sizes() == v_cache.sizes(), "k_cache and v_cache must match");
  TORCH_CHECK(k_cache.dim() == 4 && k_cache.size(3) == kHeadDim,
              "K/V must have shape [B,Hkv,N,64]");
  TORCH_CHECK(k_cache.size(0) == q_group.size(0) && k_cache.size(1) == q_group.size(1),
              "q_group and K/V batch/KV-head dimensions must match");
  TORCH_CHECK(k_cache.size(2) % kBlockM == 0, "kv_len must be divisible by 64");

  const int groups = static_cast<int>(q_group.size(0) * q_group.size(1));
  const int kv_len = static_cast<int>(k_cache.size(2));
  const int num_tiles = kv_len / kBlockM;
  TORCH_CHECK(num_splits > 0 && num_splits <= num_tiles,
              "num_splits must be in [1, kv_len/64]");
  TORCH_CHECK(checksums.numel() == static_cast<int64_t>(groups) * num_splits,
              "checksums must have B*Hkv*num_splits elements");

  const dim3 grid(groups * static_cast<int>(num_splits));
  const dim3 block(128);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  streamattn_transposed_wgmma_qkpv_async_checksum_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const Element*>(q_group.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(k_cache.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(v_cache.data_ptr<at::BFloat16>()),
      checksums.data_ptr<float>(),
      groups,
      kv_len,
      static_cast<int>(num_splits));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void streamattn_transposed_wgmma_exact_partial_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor partial_o,
    torch::Tensor partial_lse,
    int64_t num_splits) {
  TORCH_CHECK(q_group.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda() &&
              partial_o.is_cuda() && partial_lse.is_cuda(),
              "all tensors must be CUDA tensors");
  TORCH_CHECK(q_group.is_contiguous() && k_cache.is_contiguous() &&
              v_cache.is_contiguous() && partial_o.is_contiguous() &&
              partial_lse.is_contiguous(), "all tensors must be contiguous");
  TORCH_CHECK(q_group.scalar_type() == at::ScalarType::BFloat16 &&
              k_cache.scalar_type() == at::ScalarType::BFloat16 &&
              v_cache.scalar_type() == at::ScalarType::BFloat16,
              "q_group, k_cache, and v_cache must be bf16");
  TORCH_CHECK(partial_o.scalar_type() == at::ScalarType::Float &&
              partial_lse.scalar_type() == at::ScalarType::Float,
              "partial outputs must be fp32");
  TORCH_CHECK(q_group.dim() == 4 && q_group.size(2) == kBlockN &&
              q_group.size(3) == kHeadDim,
              "q_group must have shape [B,Hkv,8,64]");
  TORCH_CHECK(k_cache.sizes() == v_cache.sizes(), "k_cache and v_cache must match");
  TORCH_CHECK(k_cache.dim() == 4 && k_cache.size(3) == kHeadDim,
              "K/V must have shape [B,Hkv,N,64]");
  TORCH_CHECK(k_cache.size(0) == q_group.size(0) &&
              k_cache.size(1) == q_group.size(1),
              "q_group and K/V batch/KV-head dimensions must match");
  TORCH_CHECK(k_cache.size(2) % kBlockM == 0, "kv_len must be divisible by 64");

  const int groups = static_cast<int>(q_group.size(0) * q_group.size(1));
  const int kv_len = static_cast<int>(k_cache.size(2));
  const int num_tiles = kv_len / kBlockM;
  TORCH_CHECK(num_splits > 0 && num_splits <= num_tiles,
              "num_splits must be in [1, kv_len/64]");
  TORCH_CHECK(partial_o.sizes() == torch::IntArrayRef(
                  {groups, num_splits, kBlockN, kHeadDim}),
              "partial_o must have shape [B*Hkv,num_splits,8,64]");
  TORCH_CHECK(partial_lse.sizes() == torch::IntArrayRef(
                  {groups, num_splits, kBlockN}),
              "partial_lse must have shape [B*Hkv,num_splits,8]");

  const dim3 grid(groups * static_cast<int>(num_splits));
  const dim3 block(128);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  streamattn_transposed_wgmma_exact_partial_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const Element*>(q_group.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(k_cache.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(v_cache.data_ptr<at::BFloat16>()),
      partial_o.data_ptr<float>(),
      partial_lse.data_ptr<float>(),
      groups,
      kv_len,
      static_cast<int>(num_splits));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void streamattn_transposed_wgmma_exact_merge_out_cuda(
    torch::Tensor partial_o,
    torch::Tensor partial_lse,
    torch::Tensor output) {
  TORCH_CHECK(partial_o.is_cuda() && partial_lse.is_cuda() && output.is_cuda(),
              "all tensors must be CUDA tensors");
  TORCH_CHECK(partial_o.is_contiguous() && partial_lse.is_contiguous() &&
              output.is_contiguous(), "all tensors must be contiguous");
  TORCH_CHECK(partial_o.scalar_type() == at::ScalarType::Float &&
              partial_lse.scalar_type() == at::ScalarType::Float,
              "partial inputs must be fp32");
  TORCH_CHECK(output.scalar_type() == at::ScalarType::BFloat16,
              "output must be bf16");
  TORCH_CHECK(partial_o.dim() == 4 && partial_o.size(2) == kBlockN &&
              partial_o.size(3) == kHeadDim,
              "partial_o must have shape [groups,num_splits,8,64]");
  TORCH_CHECK(partial_lse.sizes() == torch::IntArrayRef(
                  {partial_o.size(0), partial_o.size(1), kBlockN}),
              "partial_lse must have shape [groups,num_splits,8]");
  TORCH_CHECK(output.sizes() == torch::IntArrayRef(
                  {partial_o.size(0), kBlockN, kHeadDim}),
              "output must have shape [groups,8,64]");
  TORCH_CHECK(partial_o.size(1) <= 512, "num_splits must be <= 512");

  const int groups = static_cast<int>(partial_o.size(0));
  const int num_splits = static_cast<int>(partial_o.size(1));
  const dim3 grid(groups * kBlockN);
  const dim3 block(kHeadDim);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  streamattn_transposed_wgmma_exact_merge_kernel<<<grid, block, 0, stream>>>(
      partial_o.data_ptr<float>(),
      partial_lse.data_ptr<float>(),
      reinterpret_cast<Element*>(output.data_ptr<at::BFloat16>()),
      groups,
      num_splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


def _parse_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one split count is required")
    return values


def _compile_extension(*, cutlass_root: Path, build_dir: Path | None, verbose: bool):
    from torch.utils.cpp_extension import load_inline

    previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    try:
        if build_dir is None:
            resolved_build_dir = Path(tempfile.mkdtemp(prefix="streamattn_transposed_wgmma_qk_"))
        else:
            resolved_build_dir = build_dir.expanduser().resolve()
            resolved_build_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        extension = load_inline(
            name="streamattn_transposed_wgmma_qk",
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            build_directory=str(resolved_build_dir),
            extra_include_paths=[str(cutlass_root / "include")],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++17",
                "--use_fast_math",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "-gencode=arch=compute_90a,code=sm_90a",
            ],
            with_cuda=True,
            verbose=verbose,
        )
        return extension, time.perf_counter() - started
    finally:
        if previous_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch


def _time_repeated(fn, *, warmup: int, iters: int, repeats: int) -> tuple[float, list[float]]:
    samples = [
        _time_cuda(fn, device=torch.device("cuda"), warmup=warmup, iters=iters)
        for _ in range(repeats)
    ]
    return float(statistics.median(samples)), samples


def _flashinfer_batched_runner(
    q: torch.Tensor,
    k_nhd: torch.Tensor,
    v_nhd: torch.Tensor,
    *,
    page_size: int,
):
    if flashinfer is None:
        raise RuntimeError(f"FlashInfer import failed: {FLASHINFER_IMPORT_ERROR}")
    batch, kv_len, kv_heads, dim = k_nhd.shape
    if kv_len % page_size:
        raise ValueError("kv_len must be divisible by page_size")
    pages_per_request = kv_len // page_size
    key_pages = k_nhd.view(batch * pages_per_request, page_size, kv_heads, dim)
    value_pages = v_nhd.view(batch * pages_per_request, page_size, kv_heads, dim)
    cache = torch.stack((key_pages, value_pages), dim=1).contiguous()
    total_pages = batch * pages_per_request
    indptr = torch.arange(
        0,
        total_pages + 1,
        pages_per_request,
        device=q.device,
        dtype=torch.int32,
    )
    indices = torch.arange(total_pages, device=q.device, dtype=torch.int32)
    last_page_len = torch.full((batch,), page_size, device=q.device, dtype=torch.int32)
    workspace = torch.empty(128 * 1024 * 1024, device=q.device, dtype=torch.uint8)
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        use_tensor_cores=True,
        backend="auto",
    )
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        q.shape[1],
        kv_heads,
        dim,
        page_size,
        pos_encoding_mode="NONE",
        q_data_type=q.dtype,
        kv_data_type=q.dtype,
        o_data_type=q.dtype,
        sm_scale=1.0 / math.sqrt(float(dim)),
        disable_split_kv=False,
    )
    out = torch.empty_like(q)

    def run() -> torch.Tensor:
        return wrapper.run(q, cache, out=out)

    return run


def profile(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.head_dim != 64 or args.q_heads // args.kv_heads != 8:
        raise ValueError("this milestone requires D64 and true-GQA group size 8")
    if args.kv_len % 64:
        raise ValueError("kv_len must be divisible by 64")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    q = torch.randn(args.batch, args.q_heads, args.head_dim, device=device, dtype=torch.bfloat16)
    k_nhd = torch.randn(
        args.batch,
        args.kv_len,
        args.kv_heads,
        args.head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    v_nhd = torch.randn_like(k_nhd)
    q_group = q.view(args.batch, args.kv_heads, 8, args.head_dim).contiguous()
    k_group = k_nhd.permute(0, 2, 1, 3).contiguous()
    scores = torch.empty(
        args.batch,
        args.kv_heads,
        args.kv_len,
        8,
        device=device,
        dtype=torch.float32,
    )

    cutlass_root = Path(args.cutlass_root).expanduser().resolve()
    if not (cutlass_root / "include" / "cute" / "tensor.hpp").is_file():
        raise FileNotFoundError(f"invalid CUTLASS root: {cutlass_root}")
    print(f"[transposed-qk] compiling from {cutlass_root}", flush=True)
    ext, compile_s = _compile_extension(
        cutlass_root=cutlass_root,
        build_dir=Path(args.build_dir) if args.build_dir else None,
        verbose=args.compile_verbose,
    )
    print(f"[transposed-qk] compile_s={compile_s:.3f}", flush=True)

    split_counts = _parse_ints(args.num_splits_list)
    timings: dict[str, float] = {}
    samples: dict[str, list[float]] = {}
    checksum_timings: dict[str, float] = {}
    checksum_samples: dict[str, list[float]] = {}
    async_checksum_timings: dict[str, float] = {}
    async_checksum_samples: dict[str, list[float]] = {}
    qkpv_timings: dict[str, float] = {}
    qkpv_samples: dict[str, list[float]] = {}
    exact_partial_timings: dict[str, float] = {}
    exact_partial_samples: dict[str, list[float]] = {}
    exact_merge_timings: dict[str, float] = {}
    exact_merge_samples: dict[str, list[float]] = {}
    exact_end_to_end_timings: dict[str, float] = {}
    exact_end_to_end_samples: dict[str, list[float]] = {}
    quality: dict[str, dict[str, Any]] = {}
    checksum_quality: dict[str, dict[str, float]] = {}
    async_checksum_quality: dict[str, dict[str, float]] = {}
    qkpv_quality: dict[str, dict[str, float]] = {}
    exact_partial_quality: dict[str, dict[str, float]] = {}
    exact_merged_quality: dict[str, dict[str, float]] = {}
    reference = torch.einsum("bhgd,bhnd->bhng", q_group.float(), k_group.float())
    reference_flat = reference.view(args.batch * args.kv_heads, args.kv_len, 8)
    v_group = v_nhd.permute(0, 2, 1, 3).contiguous()
    v_flat = v_group.view(
        args.batch * args.kv_heads, args.kv_len, args.head_dim
    )
    # The floor kernel intentionally stages QK through a BF16 P tile before
    # WGMMA PV.  Validate against those staged semantics rather than an FP32
    # QK tensor, otherwise the checksum aggregates harmless BF16 rounding over
    # N * Hq * D products and reports a misleadingly large absolute error.
    qkpv_reference_flat = reference_flat.to(torch.bfloat16).float()
    qkpv_token_contribution = (
        qkpv_reference_flat.sum(dim=2) * v_flat.float().sum(dim=2)
    )
    exact_probabilities = torch.softmax(reference_flat * 0.125, dim=1)
    exact_reference_out = torch.einsum(
        "gnh,gnd->ghd", exact_probabilities, v_flat.float()
    )
    for splits in split_counts:
        def run(splits: int = splits) -> torch.Tensor:
            ext.qk_out(q_group, k_group, scores, splits)
            return scores

        scores.fill_(float("nan"))
        run()
        torch.cuda.synchronize()
        first_output = scores.clone()
        run()
        torch.cuda.synchronize()
        repeat_diff = (scores - first_output).abs()
        finite = torch.isfinite(scores)
        finite_count = int(finite.sum().item())
        total_count = scores.numel()
        if finite_count == total_count:
            diff = (scores - reference).abs()
            flat_max = int(diff.reshape(-1).argmax().item())
            quality[str(splits)] = {
                **_error(scores, reference),
                "nonfinite_count": 0,
                "large_error_count_gt_1e-4": int((diff > 1.0e-4).sum().item()),
                "max_error_flat_index": flat_max,
                "repeat_max_abs_diff": float(repeat_diff.max().item()),
                "repeat_changed_count_gt_1e-6": int((repeat_diff > 1.0e-6).sum().item()),
            }
        else:
            quality[str(splits)] = {
                "max_abs_error": None,
                "mean_abs_error": None,
                "nonfinite_count": total_count - finite_count,
                "large_error_count_gt_1e-4": None,
                "max_error_flat_index": None,
                "repeat_max_abs_diff": None,
                "repeat_changed_count_gt_1e-6": None,
            }
        median_ms, raw_samples = _time_repeated(
            run,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        timings[str(splits)] = median_ms
        samples[str(splits)] = raw_samples

        checksums = torch.empty(
            args.batch * args.kv_heads,
            splits,
            device=device,
            dtype=torch.float32,
        )
        num_tiles = args.kv_len // 64
        tiles_per_split = (num_tiles + splits - 1) // splits
        reference_checksums = torch.zeros_like(checksums)
        qkpv_reference_checksums = torch.zeros_like(checksums)
        for split in range(splits):
            token_begin = split * tiles_per_split * 64
            token_end = min(args.kv_len, token_begin + tiles_per_split * 64)
            if token_begin < token_end:
                reference_checksums[:, split] = reference_flat[:, token_begin:token_end].sum(dim=(1, 2))
                qkpv_reference_checksums[:, split] = qkpv_token_contribution[:, token_begin:token_end].sum(dim=1)

        def run_checksum(splits: int = splits) -> torch.Tensor:
            ext.qk_checksum_out(q_group, k_group, checksums, splits)
            return checksums

        run_checksum()
        torch.cuda.synchronize()
        checksum_quality[str(splits)] = _error(checksums, reference_checksums)
        checksum_ms, checksum_raw_samples = _time_repeated(
            run_checksum,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        checksum_timings[str(splits)] = checksum_ms
        checksum_samples[str(splits)] = checksum_raw_samples

        def run_async_checksum(splits: int = splits) -> torch.Tensor:
            ext.qk_async_checksum_out(q_group, k_group, checksums, splits)
            return checksums

        run_async_checksum()
        torch.cuda.synchronize()
        async_checksum_quality[str(splits)] = _error(checksums, reference_checksums)
        async_checksum_ms, async_checksum_raw_samples = _time_repeated(
            run_async_checksum,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        async_checksum_timings[str(splits)] = async_checksum_ms
        async_checksum_samples[str(splits)] = async_checksum_raw_samples

        def run_qkpv(splits: int = splits) -> torch.Tensor:
            ext.qkpv_async_checksum_out(q_group, k_group, v_group, checksums, splits)
            return checksums

        run_qkpv()
        torch.cuda.synchronize()
        first_qkpv = checksums.clone()
        run_qkpv()
        torch.cuda.synchronize()
        qkpv_error = _error(checksums, qkpv_reference_checksums)
        reference_scale = float(qkpv_reference_checksums.abs().max().item())
        qkpv_quality[str(splits)] = {
            **qkpv_error,
            "max_abs_reference": reference_scale,
            "normalized_max_abs_error": (
                qkpv_error["max_abs_error"] / max(reference_scale, 1.0e-12)
            ),
            "repeat_max_abs_diff": float((checksums - first_qkpv).abs().max().item()),
        }
        qkpv_ms, qkpv_raw_samples = _time_repeated(
            run_qkpv,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        qkpv_timings[str(splits)] = qkpv_ms
        qkpv_samples[str(splits)] = qkpv_raw_samples

        partial_o = torch.empty(
            args.batch * args.kv_heads,
            splits,
            8,
            args.head_dim,
            device=device,
            dtype=torch.float32,
        )
        partial_lse = torch.empty(
            args.batch * args.kv_heads,
            splits,
            8,
            device=device,
            dtype=torch.float32,
        )

        def run_exact_partial(splits: int = splits) -> torch.Tensor:
            ext.exact_partial_out(
                q_group, k_group, v_group, partial_o, partial_lse, splits
            )
            return partial_o

        run_exact_partial()
        torch.cuda.synchronize()
        first_partial_o = partial_o.clone()
        first_partial_lse = partial_lse.clone()
        run_exact_partial()
        torch.cuda.synchronize()
        merge_max = partial_lse.max(dim=1, keepdim=True).values
        merge_weights = torch.exp(partial_lse - merge_max)
        merged_out = (
            (partial_o * merge_weights.unsqueeze(-1)).sum(dim=1)
            / merge_weights.sum(dim=1).unsqueeze(-1)
        )
        exact_error = _error(merged_out, exact_reference_out)
        exact_partial_quality[str(splits)] = {
            **exact_error,
            "partial_o_repeat_max_abs_diff": float(
                (partial_o - first_partial_o).abs().max().item()
            ),
            "partial_lse_repeat_max_abs_diff": float(
                (partial_lse - first_partial_lse).abs().max().item()
            ),
            "partial_o_nonfinite_count": int(
                (~torch.isfinite(partial_o)).sum().item()
            ),
            "partial_lse_nonfinite_count": int(
                (~torch.isfinite(partial_lse)).sum().item()
            ),
        }
        exact_partial_ms, exact_partial_raw_samples = _time_repeated(
            run_exact_partial,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        exact_partial_timings[str(splits)] = exact_partial_ms
        exact_partial_samples[str(splits)] = exact_partial_raw_samples

        exact_output = torch.empty(
            args.batch * args.kv_heads,
            8,
            args.head_dim,
            device=device,
            dtype=torch.bfloat16,
        )

        def run_exact_merge() -> torch.Tensor:
            ext.exact_merge_out(partial_o, partial_lse, exact_output)
            return exact_output

        run_exact_merge()
        torch.cuda.synchronize()
        first_exact_output = exact_output.clone()
        run_exact_merge()
        torch.cuda.synchronize()
        merged_error = _error(exact_output.float(), exact_reference_out)
        exact_merged_quality[str(splits)] = {
            **merged_error,
            "repeat_max_abs_diff": float(
                (exact_output - first_exact_output).abs().max().item()
            ),
            "nonfinite_count": int((~torch.isfinite(exact_output)).sum().item()),
        }
        exact_merge_ms, exact_merge_raw_samples = _time_repeated(
            run_exact_merge,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        exact_merge_timings[str(splits)] = exact_merge_ms
        exact_merge_samples[str(splits)] = exact_merge_raw_samples

        def run_exact_end_to_end() -> torch.Tensor:
            run_exact_partial()
            run_exact_merge()
            return exact_output

        exact_end_to_end_ms, exact_end_to_end_raw_samples = _time_repeated(
            run_exact_end_to_end,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        exact_end_to_end_timings[str(splits)] = exact_end_to_end_ms
        exact_end_to_end_samples[str(splits)] = exact_end_to_end_raw_samples
        print(
            f"[transposed-qk] splits={splits:>3} ctas={args.batch * args.kv_heads * splits:>4} "
            f"scores_ms={median_ms:.6f} checksum_ms={checksum_ms:.6f} "
            f"async_checksum_ms={async_checksum_ms:.6f} "
            f"qkpv_ms={qkpv_ms:.6f} "
            f"exact_partial_ms={exact_partial_ms:.6f} "
            f"merge_ms={exact_merge_ms:.6f} "
            f"exact_e2e_ms={exact_end_to_end_ms:.6f} "
            f"exact_out_err={exact_error['max_abs_error']:.6g} "
            f"max_err={quality[str(splits)]['max_abs_error']} "
            f"nonfinite={quality[str(splits)]['nonfinite_count']}",
            flush=True,
        )

    flashinfer_error = None
    flashinfer_ms = None
    try:
        flashinfer_run = _flashinfer_batched_runner(q, k_nhd, v_nhd, page_size=args.page_size)
        flashinfer_ms, flashinfer_samples = _time_repeated(
            flashinfer_run,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        flashinfer_output = flashinfer_run().clone()
        torch.cuda.synchronize()
        flashinfer_quality = _error(
            exact_output.view(args.batch, args.q_heads, args.head_dim).float(),
            flashinfer_output.float(),
        )
    except Exception as exc:  # pragma: no cover - depends on installed FlashInfer
        flashinfer_error = f"{type(exc).__name__}: {exc}"
        flashinfer_samples = []
        flashinfer_quality = None

    best_splits = min(timings, key=timings.get)
    best_ms = timings[best_splits]
    best_checksum_splits = min(checksum_timings, key=checksum_timings.get)
    best_checksum_ms = checksum_timings[best_checksum_splits]
    best_async_checksum_splits = min(async_checksum_timings, key=async_checksum_timings.get)
    best_async_checksum_ms = async_checksum_timings[best_async_checksum_splits]
    best_qkpv_splits = min(qkpv_timings, key=qkpv_timings.get)
    best_qkpv_ms = qkpv_timings[best_qkpv_splits]
    best_exact_partial_splits = min(exact_partial_timings, key=exact_partial_timings.get)
    best_exact_partial_ms = exact_partial_timings[best_exact_partial_splits]
    best_exact_merge_splits = min(exact_merge_timings, key=exact_merge_timings.get)
    best_exact_merge_ms = exact_merge_timings[best_exact_merge_splits]
    best_exact_end_to_end_splits = min(
        exact_end_to_end_timings, key=exact_end_to_end_timings.get
    )
    best_exact_end_to_end_ms = exact_end_to_end_timings[
        best_exact_end_to_end_splits
    ]
    paired_exact_ms: list[float] = []
    paired_flashinfer_ms: list[float] = []
    paired_speedups: list[float] = []
    if flashinfer_ms is not None:
        paired_splits = int(best_exact_end_to_end_splits)
        paired_partial_o = torch.empty(
            args.batch * args.kv_heads,
            paired_splits,
            8,
            args.head_dim,
            device=device,
            dtype=torch.float32,
        )
        paired_partial_lse = torch.empty(
            args.batch * args.kv_heads,
            paired_splits,
            8,
            device=device,
            dtype=torch.float32,
        )
        paired_output = torch.empty(
            args.batch * args.kv_heads,
            8,
            args.head_dim,
            device=device,
            dtype=torch.bfloat16,
        )

        def run_paired_exact() -> torch.Tensor:
            ext.exact_partial_out(
                q_group,
                k_group,
                v_group,
                paired_partial_o,
                paired_partial_lse,
                paired_splits,
            )
            ext.exact_merge_out(paired_partial_o, paired_partial_lse, paired_output)
            return paired_output

        for pair_idx in range(max(5, args.repeats)):
            if pair_idx % 2 == 0:
                exact_ms = _time_cuda(
                    run_paired_exact,
                    device=device,
                    warmup=args.warmup,
                    iters=args.iters,
                )
                fi_ms = _time_cuda(
                    flashinfer_run,
                    device=device,
                    warmup=args.warmup,
                    iters=args.iters,
                )
            else:
                fi_ms = _time_cuda(
                    flashinfer_run,
                    device=device,
                    warmup=args.warmup,
                    iters=args.iters,
                )
                exact_ms = _time_cuda(
                    run_paired_exact,
                    device=device,
                    warmup=args.warmup,
                    iters=args.iters,
                )
            paired_exact_ms.append(float(exact_ms))
            paired_flashinfer_ms.append(float(fi_ms))
            paired_speedups.append(float(fi_ms / exact_ms))
        paired_speedup_median = float(statistics.median(paired_speedups))
        paired_speedup_min = float(min(paired_speedups))
    else:
        paired_speedup_median = None
        paired_speedup_min = None
    result: dict[str, Any] = {
        "schema": "streamattn.transposed_wgmma_exact_decode.v1",
        "device": torch.cuda.get_device_name(device),
        "shape": {
            "batch": args.batch,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "group_size": 8,
            "kv_len": args.kv_len,
            "head_dim": args.head_dim,
            "dtype": "bf16",
            "wgmma_atom": "m64n8k16.f32.bf16.bf16",
            "qk_orientation": "K[64,64] @ Q.T[64,8]",
        },
        "compile_s": compile_s,
        "timing": {
            "qk_ms_by_splits": timings,
            "qk_samples_ms_by_splits": samples,
            "qk_checksum_ms_by_splits": checksum_timings,
            "qk_checksum_samples_ms_by_splits": checksum_samples,
            "qk_async_checksum_ms_by_splits": async_checksum_timings,
            "qk_async_checksum_samples_ms_by_splits": async_checksum_samples,
            "qkpv_async_checksum_ms_by_splits": qkpv_timings,
            "qkpv_async_checksum_samples_ms_by_splits": qkpv_samples,
            "exact_partial_ms_by_splits": exact_partial_timings,
            "exact_partial_samples_ms_by_splits": exact_partial_samples,
            "exact_merge_ms_by_splits": exact_merge_timings,
            "exact_merge_samples_ms_by_splits": exact_merge_samples,
            "exact_end_to_end_ms_by_splits": exact_end_to_end_timings,
            "exact_end_to_end_samples_ms_by_splits": exact_end_to_end_samples,
            "best_splits": int(best_splits),
            "best_qk_ms": best_ms,
            "best_cta_count": args.batch * args.kv_heads * int(best_splits),
            "best_checksum_splits": int(best_checksum_splits),
            "best_qk_checksum_ms": best_checksum_ms,
            "best_checksum_cta_count": args.batch * args.kv_heads * int(best_checksum_splits),
            "best_async_checksum_splits": int(best_async_checksum_splits),
            "best_qk_async_checksum_ms": best_async_checksum_ms,
            "best_async_checksum_cta_count": (
                args.batch * args.kv_heads * int(best_async_checksum_splits)
            ),
            "best_qkpv_splits": int(best_qkpv_splits),
            "best_qkpv_ms": best_qkpv_ms,
            "best_qkpv_cta_count": args.batch * args.kv_heads * int(best_qkpv_splits),
            "best_exact_partial_splits": int(best_exact_partial_splits),
            "best_exact_partial_ms": best_exact_partial_ms,
            "best_exact_partial_cta_count": (
                args.batch * args.kv_heads * int(best_exact_partial_splits)
            ),
            "best_exact_merge_splits": int(best_exact_merge_splits),
            "best_exact_merge_ms": best_exact_merge_ms,
            "best_exact_end_to_end_splits": int(best_exact_end_to_end_splits),
            "best_exact_end_to_end_ms": best_exact_end_to_end_ms,
            "flashinfer_batched_exact_ms": flashinfer_ms,
            "flashinfer_samples_ms": flashinfer_samples,
            "qk_budget_fraction_of_flashinfer": (
                best_ms / flashinfer_ms if flashinfer_ms is not None else None
            ),
            "qk_checksum_budget_fraction_of_flashinfer": (
                best_checksum_ms / flashinfer_ms if flashinfer_ms is not None else None
            ),
            "qk_async_checksum_budget_fraction_of_flashinfer": (
                best_async_checksum_ms / flashinfer_ms if flashinfer_ms is not None else None
            ),
            "qkpv_budget_fraction_of_flashinfer": (
                best_qkpv_ms / flashinfer_ms if flashinfer_ms is not None else None
            ),
            "exact_partial_budget_fraction_of_flashinfer": (
                best_exact_partial_ms / flashinfer_ms
                if flashinfer_ms is not None
                else None
            ),
            "exact_end_to_end_speedup_vs_flashinfer": (
                flashinfer_ms / best_exact_end_to_end_ms
                if flashinfer_ms is not None
                else None
            ),
            "paired_exact_ms": paired_exact_ms,
            "paired_flashinfer_ms": paired_flashinfer_ms,
            "paired_speedups": paired_speedups,
            "paired_speedup_median": paired_speedup_median,
            "paired_speedup_min": paired_speedup_min,
        },
        "quality": quality,
        "checksum_quality": checksum_quality,
        "async_checksum_quality": async_checksum_quality,
        "qkpv_quality": qkpv_quality,
        "exact_partial_quality": exact_partial_quality,
        "exact_merged_quality": exact_merged_quality,
        "exact_vs_flashinfer_quality": flashinfer_quality,
        "flashinfer_error": flashinfer_error,
        "flashinfer_import_error": FLASHINFER_IMPORT_ERROR,
        "decision": {
            "qk_gate": (
                "pass"
                if flashinfer_ms is not None and best_async_checksum_ms <= 0.5 * flashinfer_ms
                else "fail"
            ),
            "criterion": "cp.async storeless QK milestone <= 50% of matching batched FlashInfer exact",
            "exact_native_gate": (
                "pass"
                if paired_speedup_median is not None
                and paired_speedup_median > 1.0
                and paired_speedup_min > 0.98
                and flashinfer_quality is not None
                and flashinfer_quality["max_abs_error"] <= 5.0e-4
                and exact_merged_quality[best_exact_end_to_end_splits]["nonfinite_count"] == 0
                and exact_merged_quality[best_exact_end_to_end_splits]["repeat_max_abs_diff"] == 0.0
                else "fail"
            ),
            "exact_native_criterion": (
                "paired median faster than matching FlashInfer, paired min >=0.98x, "
                "max output delta <=5e-4, finite and deterministic"
            ),
        },
    }
    print(json.dumps(result, indent=2), flush=True)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--kv-len", type=int, default=32768)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--num-splits-list", default="8,16,17,32,33,64,128,256,512")
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cutlass-root", required=True)
    parser.add_argument("--build-dir", default="")
    parser.add_argument("--compile-verbose", action="store_true")
    parser.add_argument("--output-json", default="")
    profile(parser.parse_args())


if __name__ == "__main__":
    main()
