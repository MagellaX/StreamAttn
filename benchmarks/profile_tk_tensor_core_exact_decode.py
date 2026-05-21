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
from benchmarks.profile_head_mode_decode_cuda import _flashinfer_exact, _torch_head_mode_reference  # noqa: E402
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

torch::Tensor streamattn_tk_tc_exact_decode_chunks_cuda(
    torch::Tensor q_group,
    torch::Tensor k_group,
    torch::Tensor v_group,
    int64_t num_chunks);

std::vector<torch::Tensor> streamattn_tk_tc_exact_decode_chunk_states_cuda(
    torch::Tensor q_group,
    torch::Tensor k_group,
    torch::Tensor v_group,
    int64_t num_chunks);

torch::Tensor streamattn_tk_tc_exact_decode_chunk_merged_cuda(
    torch::Tensor q_group,
    torch::Tensor k_group,
    torch::Tensor v_group,
    int64_t num_chunks);

torch::Tensor streamattn_tk_tc_head_mode_chunk_merged_cuda(
    torch::Tensor q_group,
    torch::Tensor k_group,
    torch::Tensor v_group,
    torch::Tensor row_modes,
    int64_t num_chunks,
    int64_t block_size,
    int64_t sink_blocks,
    int64_t recent_blocks,
    int64_t middle_seed_blocks,
    int64_t block_order);

torch::Tensor streamattn_tk_tc_head_mode_compact_chunk_merged_cuda(
    torch::Tensor q_group,
    torch::Tensor k_group,
    torch::Tensor v_group,
    torch::Tensor row_modes,
    torch::Tensor active_chunks,
    torch::Tensor active_counts,
    torch::Tensor flat_active_chunks,
    torch::Tensor active_offsets,
    int64_t logical_num_chunks,
    int64_t block_size,
    int64_t sink_blocks,
    int64_t recent_blocks,
    int64_t middle_seed_blocks,
    int64_t block_order);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("exact_decode", &streamattn_tk_tc_exact_decode_cuda,
        "StreamAttn TK tensor-core exact true-GQA decode baseline");
  m.def("exact_decode_chunks", &streamattn_tk_tc_exact_decode_chunks_cuda,
        "StreamAttn TK tensor-core exact true-GQA chunk-only decode baseline");
  m.def("exact_decode_chunk_states", &streamattn_tk_tc_exact_decode_chunk_states_cuda,
        "StreamAttn TK tensor-core exact true-GQA chunk states baseline");
  m.def("exact_decode_chunk_merged", &streamattn_tk_tc_exact_decode_chunk_merged_cuda,
        "StreamAttn TK tensor-core exact true-GQA chunk+merge baseline");
  m.def("head_mode_chunk_merged", &streamattn_tk_tc_head_mode_chunk_merged_cuda,
        "StreamAttn TK tensor-core true-GQA head-mode chunk+merge baseline");
  m.def("head_mode_compact_chunk_merged", &streamattn_tk_tc_head_mode_compact_chunk_merged_cuda,
        "StreamAttn TK tensor-core true-GQA compact head-mode chunk+merge baseline");
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
struct streamattn_tc_exact_tiles<64> {
  using q_tile = rt_bf<16, 64>;
  using k_tile = rt_bf<16, 64>;
  using v_tile = rt_bf<16, 64, ducks::rt_layout::col>;
  using scores_fl = rt_fl<16, 16>;
  using scores_bf = rt_bf<16, 16>;
  using out_tile = rt_fl<16, 64>;
  using score_col = col_vec<scores_fl>;
};

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

struct streamattn_tc_chunk_globals {
  using q_gl = gl<bf16, -1, -1, -1, -1>;
  using kv_gl = gl<bf16, -1, -1, -1, -1>;
  using lse_gl = gl<float, -1, -1, -1, -1>;
  q_gl q;
  kv_gl k;
  kv_gl v;
  q_gl partial_out;
  lse_gl partial_lse;
  const int32_t* row_modes;
  const int32_t* active_chunks;
  const int32_t* active_counts;
  const int32_t* flat_active_chunks;
  const int32_t* active_offsets;
  int N;
  int Hkv;
  int num_chunks;
  int tiles_per_chunk;
  int total_active_entries;
  int block_size;
  int sink_blocks;
  int recent_blocks;
  int middle_seed_blocks;
  int block_order;
  int use_head_modes;
  int compact_chunks;
};

__device__ __forceinline__ bool streamattn_tc_tile_is_seed(
    int tile,
    int N,
    int block_size,
    int sink_blocks,
    int recent_blocks,
    int middle_seed_blocks,
    int block_order) {
  const int tile_tokens = 16;
  const int token_start = tile * tile_tokens;
  const int bs = block_size <= 0 ? tile_tokens : block_size;
  const int num_blocks = (N + bs - 1) / bs;
  const int sink_end = min(sink_blocks * bs, N);
  const int recent_start = recent_blocks >= num_blocks ? 0 : (num_blocks - recent_blocks) * bs;
  bool keep = token_start < sink_end || token_start >= recent_start;
  if (middle_seed_blocks > 0) {
    const int middle_seed_tokens = middle_seed_blocks * bs;
    if (block_order == 0) {
      const int middle_start = sink_end;
      const int middle_end = min(middle_start + middle_seed_tokens, recent_start);
      keep = keep || (token_start >= middle_start && token_start < middle_end);
    } else {
      const int middle_end = recent_start;
      const int middle_start = max(sink_end, middle_end - middle_seed_tokens);
      keep = keep || (token_start >= middle_start && token_start < middle_end);
    }
  }
  return keep;
}

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

  const float scale = rsqrtf(static_cast<float>(D));
  const float scale_log2 = scale * 1.44269504089f;
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

template <int D>
__global__ void streamattn_tk_tc_exact_decode_chunk_kernel(
    const __grid_constant__ streamattn_tc_chunk_globals g) {
  using T = streamattn_tc_exact_tiles<D>;
  const int pid = blockIdx.x;
  int chunk_slot = pid % g.num_chunks;
  int bh = pid / g.num_chunks;
  int b = bh / g.Hkv;
  int kvh = bh - b * g.Hkv;
  int chunk = chunk_slot;
  if (g.compact_chunks) {
    const int entry = pid % g.total_active_entries;
    b = pid / g.total_active_entries;
    kvh = 0;
    #pragma unroll
    for (int candidate = 0; candidate < 16; ++candidate) {
      if (candidate < g.Hkv &&
          entry >= g.active_offsets[candidate] &&
          entry < g.active_offsets[candidate + 1]) {
        kvh = candidate;
      }
    }
    chunk_slot = entry - g.active_offsets[kvh];
    chunk = g.flat_active_chunks[entry];
  }
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

  const float scale = rsqrtf(static_cast<float>(D));
  const float scale_log2 = scale * 1.44269504089f;
  const int tile_begin = chunk * g.tiles_per_chunk;
  const int tile_end = min(tile_begin + g.tiles_per_chunk, g.N / 16);
  for (int tile = tile_begin; tile < tile_end; ++tile) {
    bool tile_seed = true;
    bool has_active_rows = true;
    if (g.use_head_modes) {
      tile_seed = streamattn_tc_tile_is_seed(
          tile,
          g.N,
          g.block_size,
          g.sink_blocks,
          g.recent_blocks,
          g.middle_seed_blocks,
          g.block_order);
      has_active_rows = false;
      #pragma unroll
      for (int row = 0; row < 16; ++row) {
        const int mode = g.row_modes[kvh * 16 + row];
        has_active_rows = has_active_rows || (mode == 0) || (mode == 1 && tile_seed);
      }
    }
    if (!has_active_rows) {
      continue;
    }
    warp::load(k_reg, g.k, {b, kvh, tile, 0});
    warp::zero(scores);
    warp::mma_ABt(scores, q_reg, k_reg, scores);
    if (g.use_head_modes) {
      scores = warp::apply(scores, [row_modes = g.row_modes, kvh, tile_seed] __device__ (int row, int col, float val) {
        const int mode = row_modes[kvh * 16 + row];
        const bool active = (mode == 0) || (mode == 1 && tile_seed);
        return active ? val : -1.0e20f;
      });
    }

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
  warp::store(g.partial_out, acc, {b, kvh, chunk_slot, 0});
  warp::mul(max_vec_scaled, max_vec, scale);
  warp::log(norm_vec, norm_vec);
  warp::add(norm_vec, norm_vec, max_vec_scaled);
  warp::store(g.partial_lse, norm_vec, {b, kvh, chunk_slot, 0});
}

#define STREAMATTN_TK_TC_DISPATCH_D(D_VALUE, KERNEL_NAME, GRID, BLOCK, GLOBALS) \
  do { \
    if ((D_VALUE) == 64) { \
      KERNEL_NAME<64><<<(GRID), (BLOCK)>>>(GLOBALS); \
    } else if ((D_VALUE) == 128) { \
      KERNEL_NAME<128><<<(GRID), (BLOCK)>>>(GLOBALS); \
    } else { \
      TORCH_CHECK(false, "only D=64 or D=128 is implemented"); \
    } \
  } while (0)

__global__ void streamattn_tk_tc_exact_merge_kernel(
    const bf16* __restrict__ partial_out,
    const float* __restrict__ partial_lse,
    bf16* __restrict__ out,
    const int32_t* __restrict__ active_counts,
    int B,
    int Hkv,
    int num_chunks,
    int D) {
  const int row_pid = blockIdx.x;
  const int row = row_pid % 16;
  const int kvh = (row_pid / 16) % Hkv;
  const int b = row_pid / (16 * Hkv);
  const int tid = threadIdx.x;
  const int chunks_to_merge = active_counts == nullptr ? num_chunks : active_counts[kvh];

  float max_lse = -INFINITY;
  for (int chunk = 0; chunk < chunks_to_merge; ++chunk) {
    const int64_t lse_idx = (((int64_t)b * Hkv + kvh) * num_chunks + chunk) * 16 + row;
    const float lse = partial_lse[lse_idx];
    if (isfinite(lse)) {
      max_lse = fmaxf(max_lse, lse);
    }
  }
  float den = 0.0f;
  if (isfinite(max_lse)) {
    for (int chunk = 0; chunk < chunks_to_merge; ++chunk) {
      const int64_t lse_idx = (((int64_t)b * Hkv + kvh) * num_chunks + chunk) * 16 + row;
      const float lse = partial_lse[lse_idx];
      if (isfinite(lse)) {
        den += expf(lse - max_lse);
      }
    }
  }
  for (int d = tid; d < D; d += blockDim.x) {
    float num = 0.0f;
    if (den > 0.0f && isfinite(max_lse)) {
      for (int chunk = 0; chunk < chunks_to_merge; ++chunk) {
        const int64_t lse_idx = (((int64_t)b * Hkv + kvh) * num_chunks + chunk) * 16 + row;
        const float lse = partial_lse[lse_idx];
        if (!isfinite(lse)) {
          continue;
        }
        const float w = expf(lse - max_lse);
        const int64_t out_idx = ((((int64_t)b * Hkv + kvh) * (num_chunks * 16) + chunk * 16 + row) * D) + d;
        const float value = __bfloat162float(partial_out[out_idx]);
        num += isfinite(value) ? (w * value) : 0.0f;
      }
    }
    const int64_t dst_idx = ((((int64_t)b * Hkv + kvh) * 16 + row) * D) + d;
    out[dst_idx] = __float2bfloat16(den > 0.0f ? (num / den) : 0.0f);
  }
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
  TORCH_CHECK(D == 64 || D == 128, "only D=64 or D=128 is implemented");
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
  STREAMATTN_TK_TC_DISPATCH_D(D, streamattn_tk_tc_exact_decode_kernel, grid, block, g);
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
  return out;
}

torch::Tensor streamattn_tk_tc_exact_decode_chunks_cuda(
    torch::Tensor q_group,
    torch::Tensor k_group,
    torch::Tensor v_group,
    int64_t num_chunks) {
  TORCH_CHECK(q_group.is_cuda(), "q_group must be CUDA");
  TORCH_CHECK(k_group.is_cuda(), "k_group must be CUDA");
  TORCH_CHECK(v_group.is_cuda(), "v_group must be CUDA");
  TORCH_CHECK(q_group.is_contiguous(), "q_group must be contiguous [B,Hkv,16,D]");
  TORCH_CHECK(k_group.is_contiguous(), "k_group must be contiguous [B,Hkv,N,D]");
  TORCH_CHECK(v_group.is_contiguous(), "v_group must be contiguous [B,Hkv,N,D]");
  TORCH_CHECK(q_group.scalar_type() == at::ScalarType::BFloat16, "q_group must be bf16 for this spike");
  TORCH_CHECK(k_group.scalar_type() == at::ScalarType::BFloat16, "k_group must be bf16 for this spike");
  TORCH_CHECK(v_group.scalar_type() == at::ScalarType::BFloat16, "v_group must be bf16 for this spike");
  const int B = q_group.size(0);
  const int Hkv = q_group.size(1);
  const int padded_rows = q_group.size(2);
  const int D = q_group.size(3);
  const int N = k_group.size(2);
  TORCH_CHECK(D == 64 || D == 128, "only D=64 or D=128 is implemented");
  TORCH_CHECK(padded_rows == 16, "only 16 padded Q rows are implemented");
  TORCH_CHECK(N % 16 == 0, "N must be divisible by 16");
  TORCH_CHECK(num_chunks > 0, "num_chunks must be positive");
  TORCH_CHECK((N / 16) % num_chunks == 0, "num_chunks must divide N/16 for this spike");
  const int chunks = static_cast<int>(num_chunks);
  const int tiles_per_chunk = (N / 16) / chunks;

  auto partial = torch::empty({B, Hkv, chunks * padded_rows, D}, q_group.options());
  auto partial_lse = torch::empty({B, Hkv, chunks, padded_rows}, q_group.options().dtype(torch::kFloat32));
  const dim3 grid(B * Hkv * chunks);
  const dim3 block(32);
  using q_gl = streamattn_tc_chunk_globals::q_gl;
  using kv_gl = streamattn_tc_chunk_globals::kv_gl;
  streamattn_tc_chunk_globals g{
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
      q_gl{reinterpret_cast<bf16*>(partial.data_ptr<at::BFloat16>()),
           static_cast<unsigned long>(B),
           static_cast<unsigned long>(Hkv),
           static_cast<unsigned long>(chunks * padded_rows),
           static_cast<unsigned long>(D)},
      streamattn_tc_chunk_globals::lse_gl{partial_lse.data_ptr<float>(),
           static_cast<unsigned long>(B),
           static_cast<unsigned long>(Hkv),
           static_cast<unsigned long>(chunks),
           static_cast<unsigned long>(padded_rows)},
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      N,
      Hkv,
      chunks,
      tiles_per_chunk,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0};
  STREAMATTN_TK_TC_DISPATCH_D(D, streamattn_tk_tc_exact_decode_chunk_kernel, grid, block, g);
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
  return partial;
}

std::vector<torch::Tensor> streamattn_tk_tc_exact_decode_chunk_states_cuda(
    torch::Tensor q_group,
    torch::Tensor k_group,
    torch::Tensor v_group,
    int64_t num_chunks) {
  TORCH_CHECK(q_group.is_cuda(), "q_group must be CUDA");
  TORCH_CHECK(k_group.is_cuda(), "k_group must be CUDA");
  TORCH_CHECK(v_group.is_cuda(), "v_group must be CUDA");
  TORCH_CHECK(q_group.is_contiguous(), "q_group must be contiguous [B,Hkv,16,D]");
  TORCH_CHECK(k_group.is_contiguous(), "k_group must be contiguous [B,Hkv,N,D]");
  TORCH_CHECK(v_group.is_contiguous(), "v_group must be contiguous [B,Hkv,N,D]");
  TORCH_CHECK(q_group.scalar_type() == at::ScalarType::BFloat16, "q_group must be bf16 for this spike");
  const int B = q_group.size(0);
  const int Hkv = q_group.size(1);
  const int padded_rows = q_group.size(2);
  const int D = q_group.size(3);
  const int N = k_group.size(2);
  TORCH_CHECK(D == 64 || D == 128, "only D=64 or D=128 is implemented");
  TORCH_CHECK(padded_rows == 16, "only 16 padded Q rows are implemented");
  TORCH_CHECK(N % 16 == 0, "N must be divisible by 16");
  TORCH_CHECK(num_chunks > 0, "num_chunks must be positive");
  TORCH_CHECK((N / 16) % num_chunks == 0, "num_chunks must divide N/16 for this spike");
  const int chunks = static_cast<int>(num_chunks);
  const int tiles_per_chunk = (N / 16) / chunks;

  auto partial = torch::empty({B, Hkv, chunks * padded_rows, D}, q_group.options());
  auto partial_lse = torch::empty({B, Hkv, chunks, padded_rows}, q_group.options().dtype(torch::kFloat32));
  const dim3 grid(B * Hkv * chunks);
  const dim3 block(32);
  using q_gl = streamattn_tc_chunk_globals::q_gl;
  using kv_gl = streamattn_tc_chunk_globals::kv_gl;
  streamattn_tc_chunk_globals g{
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
      q_gl{reinterpret_cast<bf16*>(partial.data_ptr<at::BFloat16>()),
           static_cast<unsigned long>(B),
           static_cast<unsigned long>(Hkv),
           static_cast<unsigned long>(chunks * padded_rows),
           static_cast<unsigned long>(D)},
      streamattn_tc_chunk_globals::lse_gl{partial_lse.data_ptr<float>(),
           static_cast<unsigned long>(B),
           static_cast<unsigned long>(Hkv),
           static_cast<unsigned long>(chunks),
           static_cast<unsigned long>(padded_rows)},
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      N,
      Hkv,
      chunks,
      tiles_per_chunk,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0};
  STREAMATTN_TK_TC_DISPATCH_D(D, streamattn_tk_tc_exact_decode_chunk_kernel, grid, block, g);
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
  return {partial, partial_lse};
}

torch::Tensor streamattn_tk_tc_exact_decode_chunk_merged_cuda(
    torch::Tensor q_group,
    torch::Tensor k_group,
    torch::Tensor v_group,
    int64_t num_chunks) {
  auto states = streamattn_tk_tc_exact_decode_chunk_states_cuda(q_group, k_group, v_group, num_chunks);
  auto partial = states[0];
  auto partial_lse = states[1];
  const int B = q_group.size(0);
  const int Hkv = q_group.size(1);
  const int padded_rows = q_group.size(2);
  const int D = q_group.size(3);
  const int chunks = static_cast<int>(num_chunks);
  auto out = torch::empty_like(q_group);
  const dim3 grid(B * Hkv * padded_rows);
  const dim3 block(128);
  streamattn_tk_tc_exact_merge_kernel<<<grid, block>>>(
      reinterpret_cast<const bf16*>(partial.data_ptr<at::BFloat16>()),
      partial_lse.data_ptr<float>(),
      reinterpret_cast<bf16*>(out.data_ptr<at::BFloat16>()),
      nullptr,
      B,
      Hkv,
      chunks,
      D);
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
  return out;
}

torch::Tensor streamattn_tk_tc_head_mode_chunk_merged_cuda(
    torch::Tensor q_group,
    torch::Tensor k_group,
    torch::Tensor v_group,
    torch::Tensor row_modes,
    int64_t num_chunks,
    int64_t block_size,
    int64_t sink_blocks,
    int64_t recent_blocks,
    int64_t middle_seed_blocks,
    int64_t block_order) {
  TORCH_CHECK(q_group.is_cuda(), "q_group must be CUDA");
  TORCH_CHECK(k_group.is_cuda(), "k_group must be CUDA");
  TORCH_CHECK(v_group.is_cuda(), "v_group must be CUDA");
  TORCH_CHECK(row_modes.is_cuda(), "row_modes must be CUDA");
  TORCH_CHECK(q_group.is_contiguous(), "q_group must be contiguous [B,Hkv,16,D]");
  TORCH_CHECK(k_group.is_contiguous(), "k_group must be contiguous [B,Hkv,N,D]");
  TORCH_CHECK(v_group.is_contiguous(), "v_group must be contiguous [B,Hkv,N,D]");
  TORCH_CHECK(row_modes.is_contiguous(), "row_modes must be contiguous [Hkv,16]");
  TORCH_CHECK(row_modes.scalar_type() == at::ScalarType::Int, "row_modes must be int32");
  TORCH_CHECK(q_group.scalar_type() == at::ScalarType::BFloat16, "q_group must be bf16 for this spike");
  const int B = q_group.size(0);
  const int Hkv = q_group.size(1);
  const int padded_rows = q_group.size(2);
  const int D = q_group.size(3);
  const int N = k_group.size(2);
  TORCH_CHECK(D == 64 || D == 128, "only D=64 or D=128 is implemented");
  TORCH_CHECK(padded_rows == 16, "only 16 padded Q rows are implemented");
  TORCH_CHECK(row_modes.size(0) == Hkv && row_modes.size(1) == padded_rows, "row_modes shape mismatch");
  TORCH_CHECK(N % 16 == 0, "N must be divisible by 16");
  TORCH_CHECK(num_chunks > 0, "num_chunks must be positive");
  TORCH_CHECK((N / 16) % num_chunks == 0, "num_chunks must divide N/16 for this spike");
  const int chunks = static_cast<int>(num_chunks);
  const int tiles_per_chunk = (N / 16) / chunks;

  auto partial = torch::empty({B, Hkv, chunks * padded_rows, D}, q_group.options());
  auto partial_lse = torch::empty({B, Hkv, chunks, padded_rows}, q_group.options().dtype(torch::kFloat32));
  const dim3 chunk_grid(B * Hkv * chunks);
  const dim3 chunk_block(32);
  using q_gl = streamattn_tc_chunk_globals::q_gl;
  using kv_gl = streamattn_tc_chunk_globals::kv_gl;
  streamattn_tc_chunk_globals g{
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
      q_gl{reinterpret_cast<bf16*>(partial.data_ptr<at::BFloat16>()),
           static_cast<unsigned long>(B),
           static_cast<unsigned long>(Hkv),
           static_cast<unsigned long>(chunks * padded_rows),
           static_cast<unsigned long>(D)},
      streamattn_tc_chunk_globals::lse_gl{partial_lse.data_ptr<float>(),
           static_cast<unsigned long>(B),
           static_cast<unsigned long>(Hkv),
           static_cast<unsigned long>(chunks),
           static_cast<unsigned long>(padded_rows)},
      row_modes.data_ptr<int32_t>(),
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      N,
      Hkv,
      chunks,
      tiles_per_chunk,
      0,
      static_cast<int>(block_size),
      static_cast<int>(sink_blocks),
      static_cast<int>(recent_blocks),
      static_cast<int>(middle_seed_blocks),
      static_cast<int>(block_order),
      1,
      0};
  STREAMATTN_TK_TC_DISPATCH_D(D, streamattn_tk_tc_exact_decode_chunk_kernel, chunk_grid, chunk_block, g);
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));

  auto out = torch::empty_like(q_group);
  const dim3 merge_grid(B * Hkv * padded_rows);
  const dim3 merge_block(128);
  streamattn_tk_tc_exact_merge_kernel<<<merge_grid, merge_block>>>(
      reinterpret_cast<const bf16*>(partial.data_ptr<at::BFloat16>()),
      partial_lse.data_ptr<float>(),
      reinterpret_cast<bf16*>(out.data_ptr<at::BFloat16>()),
      nullptr,
      B,
      Hkv,
      chunks,
      D);
  err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
  return out;
}

torch::Tensor streamattn_tk_tc_head_mode_compact_chunk_merged_cuda(
    torch::Tensor q_group,
    torch::Tensor k_group,
    torch::Tensor v_group,
    torch::Tensor row_modes,
    torch::Tensor active_chunks,
    torch::Tensor active_counts,
    torch::Tensor flat_active_chunks,
    torch::Tensor active_offsets,
    int64_t logical_num_chunks,
    int64_t block_size,
    int64_t sink_blocks,
    int64_t recent_blocks,
    int64_t middle_seed_blocks,
    int64_t block_order) {
  TORCH_CHECK(q_group.is_cuda(), "q_group must be CUDA");
  TORCH_CHECK(k_group.is_cuda(), "k_group must be CUDA");
  TORCH_CHECK(v_group.is_cuda(), "v_group must be CUDA");
  TORCH_CHECK(row_modes.is_cuda(), "row_modes must be CUDA");
  TORCH_CHECK(active_chunks.is_cuda(), "active_chunks must be CUDA");
  TORCH_CHECK(active_counts.is_cuda(), "active_counts must be CUDA");
  TORCH_CHECK(flat_active_chunks.is_cuda(), "flat_active_chunks must be CUDA");
  TORCH_CHECK(active_offsets.is_cuda(), "active_offsets must be CUDA");
  TORCH_CHECK(q_group.is_contiguous(), "q_group must be contiguous [B,Hkv,16,D]");
  TORCH_CHECK(k_group.is_contiguous(), "k_group must be contiguous [B,Hkv,N,D]");
  TORCH_CHECK(v_group.is_contiguous(), "v_group must be contiguous [B,Hkv,N,D]");
  TORCH_CHECK(row_modes.is_contiguous(), "row_modes must be contiguous [Hkv,16]");
  TORCH_CHECK(active_chunks.is_contiguous(), "active_chunks must be contiguous [Hkv,max_active_chunks]");
  TORCH_CHECK(active_counts.is_contiguous(), "active_counts must be contiguous [Hkv]");
  TORCH_CHECK(flat_active_chunks.is_contiguous(), "flat_active_chunks must be contiguous [total_active_entries]");
  TORCH_CHECK(active_offsets.is_contiguous(), "active_offsets must be contiguous [Hkv+1]");
  TORCH_CHECK(row_modes.scalar_type() == at::ScalarType::Int, "row_modes must be int32");
  TORCH_CHECK(active_chunks.scalar_type() == at::ScalarType::Int, "active_chunks must be int32");
  TORCH_CHECK(active_counts.scalar_type() == at::ScalarType::Int, "active_counts must be int32");
  TORCH_CHECK(flat_active_chunks.scalar_type() == at::ScalarType::Int, "flat_active_chunks must be int32");
  TORCH_CHECK(active_offsets.scalar_type() == at::ScalarType::Int, "active_offsets must be int32");
  TORCH_CHECK(q_group.scalar_type() == at::ScalarType::BFloat16, "q_group must be bf16 for this spike");
  const int B = q_group.size(0);
  const int Hkv = q_group.size(1);
  const int padded_rows = q_group.size(2);
  const int D = q_group.size(3);
  const int N = k_group.size(2);
  const int max_active_chunks = active_chunks.size(1);
  const int total_active_entries = flat_active_chunks.size(0);
  TORCH_CHECK(D == 64 || D == 128, "only D=64 or D=128 is implemented");
  TORCH_CHECK(padded_rows == 16, "only 16 padded Q rows are implemented");
  TORCH_CHECK(row_modes.size(0) == Hkv && row_modes.size(1) == padded_rows, "row_modes shape mismatch");
  TORCH_CHECK(active_chunks.size(0) == Hkv, "active_chunks Hkv mismatch");
  TORCH_CHECK(active_counts.size(0) == Hkv, "active_counts Hkv mismatch");
  TORCH_CHECK(active_offsets.size(0) == Hkv + 1, "active_offsets shape mismatch");
  TORCH_CHECK(N % 16 == 0, "N must be divisible by 16");
  TORCH_CHECK(logical_num_chunks > 0, "logical_num_chunks must be positive");
  TORCH_CHECK(max_active_chunks > 0, "active_chunks must contain at least one slot");
  TORCH_CHECK(total_active_entries > 0, "flat_active_chunks must contain at least one entry");
  TORCH_CHECK((N / 16) % logical_num_chunks == 0, "logical_num_chunks must divide N/16 for this spike");
  const int logical_chunks = static_cast<int>(logical_num_chunks);
  const int compact_chunks = static_cast<int>(max_active_chunks);
  const int tiles_per_chunk = (N / 16) / logical_chunks;

  auto partial = torch::empty({B, Hkv, compact_chunks * padded_rows, D}, q_group.options());
  auto partial_lse = torch::empty({B, Hkv, compact_chunks, padded_rows}, q_group.options().dtype(torch::kFloat32));
  const dim3 chunk_grid(B * total_active_entries);
  const dim3 chunk_block(32);
  using q_gl = streamattn_tc_chunk_globals::q_gl;
  using kv_gl = streamattn_tc_chunk_globals::kv_gl;
  streamattn_tc_chunk_globals g{
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
      q_gl{reinterpret_cast<bf16*>(partial.data_ptr<at::BFloat16>()),
           static_cast<unsigned long>(B),
           static_cast<unsigned long>(Hkv),
           static_cast<unsigned long>(compact_chunks * padded_rows),
           static_cast<unsigned long>(D)},
      streamattn_tc_chunk_globals::lse_gl{partial_lse.data_ptr<float>(),
           static_cast<unsigned long>(B),
           static_cast<unsigned long>(Hkv),
           static_cast<unsigned long>(compact_chunks),
           static_cast<unsigned long>(padded_rows)},
      row_modes.data_ptr<int32_t>(),
      active_chunks.data_ptr<int32_t>(),
      active_counts.data_ptr<int32_t>(),
      flat_active_chunks.data_ptr<int32_t>(),
      active_offsets.data_ptr<int32_t>(),
      N,
      Hkv,
      compact_chunks,
      tiles_per_chunk,
      total_active_entries,
      static_cast<int>(block_size),
      static_cast<int>(sink_blocks),
      static_cast<int>(recent_blocks),
      static_cast<int>(middle_seed_blocks),
      static_cast<int>(block_order),
      1,
      1};
  STREAMATTN_TK_TC_DISPATCH_D(D, streamattn_tk_tc_exact_decode_chunk_kernel, chunk_grid, chunk_block, g);
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));

  auto out = torch::empty_like(q_group);
  const dim3 merge_grid(B * Hkv * padded_rows);
  const dim3 merge_block(128);
  streamattn_tk_tc_exact_merge_kernel<<<merge_grid, merge_block>>>(
      reinterpret_cast<const bf16*>(partial.data_ptr<at::BFloat16>()),
      partial_lse.data_ptr<float>(),
      reinterpret_cast<bf16*>(out.data_ptr<at::BFloat16>()),
      active_counts.data_ptr<int32_t>(),
      B,
      Hkv,
      compact_chunks,
      D);
  err = cudaGetLastError();
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


def _parse_heads(raw: str) -> list[int]:
    return sorted(set(int(item.strip()) for item in raw.split(",") if item.strip()))


def _pack_row_modes_by_kv_group(
    *,
    q_heads: int,
    kv_heads: int,
    seed_heads: list[int],
    padded_rows: int = 16,
    device: torch.device,
) -> torch.Tensor:
    if q_heads % kv_heads != 0:
        raise ValueError("q_heads must be divisible by kv_heads")
    group_size = q_heads // kv_heads
    if group_size > padded_rows:
        raise ValueError("group_size exceeds padded_rows")
    seed_set = set(seed_heads)
    modes = torch.full((kv_heads, padded_rows), 2, device=device, dtype=torch.int32)
    for kv_head in range(kv_heads):
        for row in range(group_size):
            q_head = kv_head * group_size + row
            modes[kv_head, row] = 1 if q_head in seed_set else 0
    return modes.contiguous()


def _tile_is_seed(
    *,
    tile: int,
    kv_len: int,
    block_size: int,
    sink_blocks: int,
    recent_blocks: int,
    middle_seed_blocks: int,
    block_order: str,
) -> bool:
    token_start = tile * 16
    block_size = block_size or 16
    num_blocks = (kv_len + block_size - 1) // block_size
    sink_end = min(sink_blocks * block_size, kv_len)
    recent_start = 0 if recent_blocks >= num_blocks else (num_blocks - recent_blocks) * block_size
    keep = token_start < sink_end or token_start >= recent_start
    if middle_seed_blocks > 0:
        middle_seed_tokens = middle_seed_blocks * block_size
        if block_order == "sequential":
            middle_start = sink_end
            middle_end = min(middle_start + middle_seed_tokens, recent_start)
        else:
            middle_end = recent_start
            middle_start = max(sink_end, middle_end - middle_seed_tokens)
        keep = keep or (middle_start <= token_start < middle_end)
    return keep


def _pack_active_chunks_by_kv_group(
    *,
    q_heads: int,
    kv_heads: int,
    seed_heads: list[int],
    kv_len: int,
    num_chunks: int,
    block_size: int,
    sink_blocks: int,
    recent_blocks: int,
    middle_seed_blocks: int,
    block_order: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]]]:
    if q_heads % kv_heads != 0:
        raise ValueError("q_heads must be divisible by kv_heads")
    tiles = kv_len // 16
    if tiles % num_chunks != 0:
        raise ValueError("num_chunks must divide kv_len/16")
    group_size = q_heads // kv_heads
    seed_set = set(seed_heads)
    tiles_per_chunk = tiles // num_chunks
    active_by_kv: list[list[int]] = []
    for kv_head in range(kv_heads):
        q_start = kv_head * group_size
        q_group = range(q_start, q_start + group_size)
        has_exact = any(q_head not in seed_set for q_head in q_group)
        has_seed = any(q_head in seed_set for q_head in q_group)
        if has_exact:
            active = list(range(num_chunks))
        elif has_seed:
            active = [
                chunk
                for chunk in range(num_chunks)
                if any(
                    _tile_is_seed(
                        tile=tile,
                        kv_len=kv_len,
                        block_size=block_size,
                        sink_blocks=sink_blocks,
                        recent_blocks=recent_blocks,
                        middle_seed_blocks=middle_seed_blocks,
                        block_order=block_order,
                    )
                    for tile in range(chunk * tiles_per_chunk, (chunk + 1) * tiles_per_chunk)
                )
            ]
        else:
            active = []
        active_by_kv.append(active)

    max_active = max((len(chunks) for chunks in active_by_kv), default=0)
    if max_active == 0:
        max_active = 1
    active_chunks = torch.zeros((kv_heads, max_active), device=device, dtype=torch.int32)
    active_counts = torch.empty((kv_heads,), device=device, dtype=torch.int32)
    flat_chunks: list[int] = []
    offsets = [0]
    for kv_head, chunks in enumerate(active_by_kv):
        active_counts[kv_head] = len(chunks)
        if chunks:
            active_chunks[kv_head, : len(chunks)] = torch.tensor(chunks, device=device, dtype=torch.int32)
            flat_chunks.extend(chunks)
        offsets.append(len(flat_chunks))
    if not flat_chunks:
        flat_chunks = [0]
    flat_active_chunks = torch.tensor(flat_chunks, device=device, dtype=torch.int32)
    active_offsets = torch.tensor(offsets, device=device, dtype=torch.int32)
    return (
        active_chunks.contiguous(),
        active_counts.contiguous(),
        flat_active_chunks.contiguous(),
        active_offsets.contiguous(),
        active_by_kv,
    )


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
    parser.add_argument("--num-chunks", type=int, default=64)
    parser.add_argument("--num-chunks-list", default="")
    parser.add_argument("--seed-heads", default="2,3,4,6,7")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--recent-blocks", type=int, default=2)
    parser.add_argument("--middle-seed-blocks", type=int, default=2)
    parser.add_argument("--block-order", default="recent_first", choices=["sequential", "recent_first"])
    parser.add_argument("--tk-root", default="")
    parser.add_argument("--checkout-dir", default="")
    parser.add_argument("--cuda-arch", default="sm_90a")
    parser.add_argument("--torch-cuda-arch-list", default="9.0a")
    parser.add_argument("--compile-verbose", action="store_true")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    if args.dtype != "bf16":
        raise ValueError("this spike currently supports --dtype bf16 only")
    if args.head_dim not in (64, 128):
        raise ValueError("this spike currently supports --head-dim 64 or 128 only")

    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    torch.manual_seed(args.seed)
    q = torch.randn((1, args.q_heads, args.head_dim), device=device, dtype=dtype)
    k = torch.randn((1, args.kv_len, args.kv_heads, args.head_dim), device=device, dtype=dtype)
    v = torch.randn_like(k)
    q_group = _pack_q_by_kv_group(q, args.kv_heads, padded_rows=16)
    k_group = _pack_kv_head_major(k)
    v_group = _pack_kv_head_major(v)
    seed_heads = _parse_heads(args.seed_heads)
    row_modes = _pack_row_modes_by_kv_group(
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        seed_heads=seed_heads,
        padded_rows=16,
        device=device,
    )
    active_chunks, active_counts, flat_active_chunks, active_offsets, active_by_kv = _pack_active_chunks_by_kv_group(
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        seed_heads=seed_heads,
        kv_len=args.kv_len,
        num_chunks=args.num_chunks,
        block_size=args.block_size,
        sink_blocks=args.sink_blocks,
        recent_blocks=args.recent_blocks,
        middle_seed_blocks=args.middle_seed_blocks,
        block_order=args.block_order,
        device=device,
    )
    block_order_id = 0 if args.block_order == "sequential" else 1

    tk_root = _find_or_clone_tk(args)
    print(
        "[tk-tc] compiling extension "
        f"head_dim={args.head_dim} dtype={args.dtype} q_heads={args.q_heads} "
        f"kv_heads={args.kv_heads} kv_len={args.kv_len}",
        flush=True,
    )
    compile_start = time.perf_counter()
    ext = _compile_extension(
        tk_root=tk_root,
        cuda_arch=args.cuda_arch,
        torch_cuda_arch_list=args.torch_cuda_arch_list,
        verbose=args.compile_verbose,
    )
    compile_s = time.perf_counter() - compile_start
    print(f"[tk-tc] compile finished in {compile_s:.2f}s", flush=True)

    def tk_exact() -> torch.Tensor:
        return ext.exact_decode(q_group, k_group, v_group)

    def _chunk_counts() -> list[int]:
        counts = [args.num_chunks]
        if args.num_chunks_list:
            counts.extend(int(item.strip()) for item in args.num_chunks_list.split(",") if item.strip())
        return sorted(set(counts))

    chunk_counts = _chunk_counts()
    compact_inputs_by_chunk: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]]]] = {}
    for count in chunk_counts:
        compact_inputs_by_chunk[count] = _pack_active_chunks_by_kv_group(
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            seed_heads=seed_heads,
            kv_len=args.kv_len,
            num_chunks=count,
            block_size=args.block_size,
            sink_blocks=args.sink_blocks,
            recent_blocks=args.recent_blocks,
            middle_seed_blocks=args.middle_seed_blocks,
            block_order=args.block_order,
            device=device,
        )

    def tk_chunk_only(num_chunks: int) -> torch.Tensor:
        return ext.exact_decode_chunks(q_group, k_group, v_group, num_chunks)

    def tk_chunk_merged(num_chunks: int) -> torch.Tensor:
        return ext.exact_decode_chunk_merged(q_group, k_group, v_group, num_chunks)

    def tk_head_mode_merged(num_chunks: int) -> torch.Tensor:
        return ext.head_mode_chunk_merged(
            q_group,
            k_group,
            v_group,
            row_modes,
            num_chunks,
            args.block_size,
            args.sink_blocks,
            args.recent_blocks,
            args.middle_seed_blocks,
            block_order_id,
        )

    def tk_head_mode_compact(num_chunks: int) -> torch.Tensor:
        compact_chunks, compact_counts, compact_flat_chunks, compact_offsets, _ = compact_inputs_by_chunk[num_chunks]
        return ext.head_mode_compact_chunk_merged(
            q_group,
            k_group,
            v_group,
            row_modes,
            compact_chunks,
            compact_counts,
            compact_flat_chunks,
            compact_offsets,
            num_chunks,
            args.block_size,
            args.sink_blocks,
            args.recent_blocks,
            args.middle_seed_blocks,
            block_order_id,
        )

    print("[tk-tc] running correctness references", flush=True)
    tk_out_group = tk_exact()
    partial_group = tk_chunk_only(args.num_chunks)
    merged_group = tk_chunk_merged(args.num_chunks)
    head_mode_group = tk_head_mode_merged(args.num_chunks)
    compact_head_mode_group = ext.head_mode_compact_chunk_merged(
        q_group,
        k_group,
        v_group,
        row_modes,
        active_chunks,
        active_counts,
        flat_active_chunks,
        active_offsets,
        args.num_chunks,
        args.block_size,
        args.sink_blocks,
        args.recent_blocks,
        args.middle_seed_blocks,
        block_order_id,
    )
    torch_ref_group = _reference_from_packed(q_group, k_group, v_group)
    tk_out = _unpack_q_by_kv_group(tk_out_group, args.q_heads)
    merged_out = _unpack_q_by_kv_group(merged_group, args.q_heads)
    head_mode_out = _unpack_q_by_kv_group(head_mode_group, args.q_heads)
    compact_head_mode_out = _unpack_q_by_kv_group(compact_head_mode_group, args.q_heads)

    def dense_true() -> torch.Tensor:
        return _dense_true_gqa(q[:, None, :, :], k, v)[:, 0]

    head_modes = torch.zeros(args.q_heads, device=device, dtype=torch.int32)
    if seed_heads:
        head_modes[torch.tensor(seed_heads, device=device, dtype=torch.long)] = 1

    def head_mode_ref() -> torch.Tensor:
        return _torch_head_mode_reference(
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

    dense_ref = dense_true()
    head_ref = head_mode_ref()
    print("[tk-tc] timing kernels", flush=True)
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
    chunk_sweep: Dict[str, float] = {}
    merged_sweep: Dict[str, float] = {}
    head_mode_sweep: Dict[str, float] = {}
    compact_head_mode_sweep: Dict[str, float] = {}
    for num_chunks in chunk_counts:
        chunk_sweep[str(num_chunks)] = _time_cuda(
            lambda c=num_chunks: tk_chunk_only(c),
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
        merged_sweep[str(num_chunks)] = _time_cuda(
            lambda c=num_chunks: tk_chunk_merged(c),
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
        head_mode_sweep[str(num_chunks)] = _time_cuda(
            lambda c=num_chunks: tk_head_mode_merged(c),
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
        compact_head_mode_sweep[str(num_chunks)] = _time_cuda(
            lambda c=num_chunks: tk_head_mode_compact(c),
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
    chunk_ms = chunk_sweep[str(args.num_chunks)]
    merged_ms = merged_sweep[str(args.num_chunks)]
    dense_ms = _time_cuda(dense_true, device=device, warmup=args.warmup, iters=args.iters)
    output = {
        "schema": "streamattn.tk_tensor_core_exact_decode.v1",
        "shape": {
            "batch": 1,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "group_size": args.q_heads // args.kv_heads,
            "padded_group_rows": 16,
            "num_chunks": args.num_chunks,
            "seed_heads": seed_heads,
            "block_size": args.block_size,
            "seed_tile_blocks": {
                "sink_blocks": args.sink_blocks,
                "recent_blocks": args.recent_blocks,
                "middle_seed_blocks": args.middle_seed_blocks,
                "block_order": args.block_order,
            },
            "active_chunks_by_kv_group": active_by_kv,
            "active_chunk_counts_by_kv_group": [len(chunks) for chunks in active_by_kv],
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
            "tk_tensor_core_chunk_only_ms": chunk_ms,
            "tk_tensor_core_chunk_merged_ms": merged_ms,
            "tk_tensor_core_chunk_only_sweep_ms": chunk_sweep,
            "tk_tensor_core_chunk_merged_sweep_ms": merged_sweep,
            "tk_tensor_core_head_mode_merged_sweep_ms": head_mode_sweep,
            "tk_tensor_core_head_mode_compact_sweep_ms": compact_head_mode_sweep,
            "tk_tensor_core_best_chunk_only_ms": min(chunk_sweep.values()) if chunk_sweep else None,
            "tk_tensor_core_best_chunk_count": int(min(chunk_sweep, key=chunk_sweep.get)) if chunk_sweep else None,
            "tk_tensor_core_best_merged_ms": min(merged_sweep.values()) if merged_sweep else None,
            "tk_tensor_core_best_merged_chunk_count": int(min(merged_sweep, key=merged_sweep.get))
            if merged_sweep
            else None,
            "tk_tensor_core_best_head_mode_ms": min(head_mode_sweep.values()) if head_mode_sweep else None,
            "tk_tensor_core_best_head_mode_chunk_count": int(min(head_mode_sweep, key=head_mode_sweep.get))
            if head_mode_sweep
            else None,
            "tk_tensor_core_best_head_mode_compact_ms": min(compact_head_mode_sweep.values())
            if compact_head_mode_sweep
            else None,
            "tk_tensor_core_best_head_mode_compact_chunk_count": int(
                min(compact_head_mode_sweep, key=compact_head_mode_sweep.get)
            )
            if compact_head_mode_sweep
            else None,
            "torch_dense_true_gqa_ms": dense_ms,
            "flashinfer_exact_ms": flashinfer_ms,
            "tk_speedup_vs_torch_dense": dense_ms / tk_ms if tk_ms else None,
            "tk_speedup_vs_flashinfer": flashinfer_ms / tk_ms if flashinfer_ms and tk_ms else None,
            "chunk_only_speedup_vs_tk_serial": tk_ms / chunk_ms if chunk_ms else None,
            "chunk_only_speedup_vs_flashinfer": flashinfer_ms / chunk_ms if flashinfer_ms and chunk_ms else None,
        },
        "quality": {
            "tk_vs_packed_torch_ref": _error(tk_out_group, torch_ref_group),
            "tk_vs_dense_true_gqa": _error(tk_out[:, None, :, :], dense_ref[:, None, :, :]),
            "merged_vs_packed_torch_ref": _error(merged_group, torch_ref_group),
            "merged_vs_dense_true_gqa": _error(merged_out[:, None, :, :], dense_ref[:, None, :, :]),
            "head_mode_vs_reference": _error(head_mode_out, head_ref),
            "head_mode_vs_dense_true_gqa": _error(head_mode_out[:, None, :, :], dense_ref[:, None, :, :]),
            "compact_head_mode_vs_reference": _error(compact_head_mode_out, head_ref),
            "compact_head_mode_vs_dense_true_gqa": _error(
                compact_head_mode_out[:, None, :, :],
                dense_ref[:, None, :, :],
            ),
            "partial_group_shape": list(partial_group.shape),
        },
        "flashinfer_error": flashinfer_error,
        "next_path": "calibrate_kv_group_coherent_seed_policies_and_reduce_merge_overhead"
        if (flashinfer_ms is not None and min(merged_sweep.values()) <= flashinfer_ms * 1.5)
        else "optimize_partial_state_merge_or_embed_head_modes_in_flashinfer_scheduler",
    }
    text = json.dumps(output, indent=2, sort_keys=True)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
