"""CUDA sources for the promoted SM80 grouped exact decode backend."""

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

std::vector<torch::Tensor> streamattn_tk_tc_exact_decode_chunk_states_warpgroup_cuda(
    torch::Tensor q_group,
    torch::Tensor k_group,
    torch::Tensor v_group,
    int64_t num_chunks,
    int64_t producer_warps);

std::vector<torch::Tensor> streamattn_tk_tc_exact_decode_chunk_states_staged_cuda(
    torch::Tensor q_group,
    torch::Tensor k_group,
    torch::Tensor v_group,
    int64_t num_chunks);

std::vector<torch::Tensor> streamattn_tk_tc_exact_decode_chunk_states_staged_grouped_cuda(
    torch::Tensor q_group,
    torch::Tensor k_group,
    torch::Tensor v_group,
    int64_t num_chunks);

torch::Tensor streamattn_tk_tc_exact_decode_chunk_merged_cuda(
    torch::Tensor q_group,
    torch::Tensor k_group,
    torch::Tensor v_group,
    int64_t num_chunks);

torch::Tensor streamattn_tk_tc_exact_decode_chunk_merged_warpgroup_cuda(
    torch::Tensor q_group,
    torch::Tensor k_group,
    torch::Tensor v_group,
    int64_t num_chunks,
    int64_t producer_warps);

torch::Tensor streamattn_tk_tc_exact_decode_chunk_merged_staged_cuda(
    torch::Tensor q_group,
    torch::Tensor k_group,
    torch::Tensor v_group,
    int64_t num_chunks);

torch::Tensor streamattn_tk_tc_exact_decode_chunk_merged_staged_grouped_cuda(
    torch::Tensor q_group,
    torch::Tensor k_group,
    torch::Tensor v_group,
    int64_t num_chunks);

torch::Tensor streamattn_tk_tc_exact_decode_chunk_merged_staged_grouped_direct_cuda(
    torch::Tensor q,
    torch::Tensor k_group,
    torch::Tensor v_group,
    int64_t num_chunks);

torch::Tensor streamattn_tk_tc_exact_decode_chunk_merged_staged_grouped_direct_out_cuda(
    torch::Tensor q,
    torch::Tensor k_group,
    torch::Tensor v_group,
    torch::Tensor partial,
    torch::Tensor partial_lse,
    torch::Tensor out,
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
  m.def("exact_decode_chunk_states_warpgroup",
        &streamattn_tk_tc_exact_decode_chunk_states_warpgroup_cuda,
        "StreamAttn TK exact chunk states with multiple independent producer warps per CTA");
  m.def("exact_decode_chunk_states_staged",
        &streamattn_tk_tc_exact_decode_chunk_states_staged_cuda,
        "StreamAttn TK exact chunk states with double-buffered cp.async K/V staging");
  m.def("exact_decode_chunk_states_staged_grouped",
        &streamattn_tk_tc_exact_decode_chunk_states_staged_grouped_cuda,
        "StreamAttn TK exact staged chunk states with four-way intra-CTA reduction");
  m.def("exact_decode_chunk_merged", &streamattn_tk_tc_exact_decode_chunk_merged_cuda,
        "StreamAttn TK tensor-core exact true-GQA chunk+merge baseline");
  m.def("exact_decode_chunk_merged_warpgroup",
        &streamattn_tk_tc_exact_decode_chunk_merged_warpgroup_cuda,
        "StreamAttn TK exact chunk+merge with multiple independent producer warps per CTA");
  m.def("exact_decode_chunk_merged_staged",
        &streamattn_tk_tc_exact_decode_chunk_merged_staged_cuda,
        "StreamAttn TK exact chunk+merge with double-buffered cp.async K/V staging");
  m.def("exact_decode_chunk_merged_staged_grouped",
        &streamattn_tk_tc_exact_decode_chunk_merged_staged_grouped_cuda,
        "StreamAttn TK exact staged chunk+merge with four-way intra-CTA reduction");
  m.def("exact_decode_chunk_merged_staged_grouped_direct",
        &streamattn_tk_tc_exact_decode_chunk_merged_staged_grouped_direct_cuda,
        "StreamAttn TK exact grouped decode with fused standard-Q input and output");
  m.def("exact_decode_chunk_merged_staged_grouped_direct_out",
        &streamattn_tk_tc_exact_decode_chunk_merged_staged_grouped_direct_out_cuda,
        "Allocation-free StreamAttn TK exact grouped decode");
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
  using kv_smem = st_bf<16, 64>;
  using lse_smem = col_vec<st_fl<16, 64>>;
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
  using kv_smem = st_bf<16, 128>;
  using lse_smem = col_vec<st_fl<16, 128>>;
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
  int producer_warps;
  int total_tasks;
};

struct streamattn_tc_grouped_direct_globals {
  using kv_gl = gl<bf16, -1, -1, -1, -1>;
  using partial_gl = gl<bf16, -1, -1, -1, -1>;
  using lse_gl = gl<float, -1, -1, -1, -1>;
  const bf16* q;
  kv_gl k;
  kv_gl v;
  partial_gl partial_out;
  lse_gl partial_lse;
  int Hq;
  int Hkv;
  int group_size;
  int N;
  int num_chunks;
  int tiles_per_chunk;
};

template <int D>
__device__ __forceinline__ void streamattn_tc_load_grouped_query(
    typename streamattn_tc_exact_tiles<D>::q_tile& dst,
    const bf16* __restrict__ q,
    int b,
    int Hq,
    int kvh,
    int group_size) {
  const int lane = threadIdx.x & 31;
  const int row_lo = lane >> 2;
  const int row_hi = row_lo + 8;
  const int pair_col = 2 * (lane & 3);
  const bf16* base = q + ((static_cast<int64_t>(b) * Hq + kvh * group_size) * D);
  const bf16_2 zero = __float22bfloat162_rn(make_float2(0.0f, 0.0f));

  #pragma unroll
  for (int tile_col = 0; tile_col < D / 16; ++tile_col) {
    const int col = tile_col * 16 + pair_col;
    dst.tiles[0][tile_col].data[0] = row_lo < group_size
        ? *reinterpret_cast<const bf16_2*>(base + row_lo * D + col)
        : zero;
    dst.tiles[0][tile_col].data[2] = row_lo < group_size
        ? *reinterpret_cast<const bf16_2*>(base + row_lo * D + col + 8)
        : zero;
    dst.tiles[0][tile_col].data[1] = row_hi < group_size
        ? *reinterpret_cast<const bf16_2*>(base + row_hi * D + col)
        : zero;
    dst.tiles[0][tile_col].data[3] = row_hi < group_size
        ? *reinterpret_cast<const bf16_2*>(base + row_hi * D + col + 8)
        : zero;
  }
}

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
  const int producer_warp = threadIdx.x >> 5;
  if (producer_warp >= g.producer_warps) return;
  const int pid = blockIdx.x * g.producer_warps + producer_warp;
  if (pid >= g.total_tasks) return;
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

template <int D>
__global__ void streamattn_tk_tc_exact_decode_chunk_staged_kernel(
    const __grid_constant__ streamattn_tc_chunk_globals g) {
  using T = streamattn_tc_exact_tiles<D>;
  const int pid = blockIdx.x;
  if (threadIdx.x >= 32 || pid >= g.total_tasks) return;
  const int chunk = pid % g.num_chunks;
  const int bh = pid / g.num_chunks;
  const int b = bh / g.Hkv;
  const int kvh = bh - b * g.Hkv;

  __shared__ typename T::kv_smem k_smem[2];
  __shared__ typename T::kv_smem v_smem[2];
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
  int stage = 0;

  if (tile_begin < tile_end) {
    warp::load_async(k_smem[stage], g.k, {b, kvh, tile_begin, 0});
    warp::load_async(v_smem[stage], g.v, {b, kvh, tile_begin, 0});
    warp::load_async_wait();
  }

  for (int tile = tile_begin; tile < tile_end; ++tile) {
    warp::load(k_reg, k_smem[stage]);
    warp::load(v_reg, v_smem[stage]);

    const int next_tile = tile + 1;
    const int next_stage = stage ^ 1;
    if (next_tile < tile_end) {
      warp::load_async(k_smem[next_stage], g.k, {b, kvh, next_tile, 0});
      warp::load_async(v_smem[next_stage], g.v, {b, kvh, next_tile, 0});
    }

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
    warp::mma_AB(acc, scores_mma, v_reg, acc);

    if (next_tile < tile_end) {
      warp::load_async_wait();
      stage = next_stage;
    }
  }

  warp::div_row(acc, acc, norm_vec);
  warp::store(g.partial_out, acc, {b, kvh, chunk, 0});
  warp::mul(max_vec_scaled, max_vec, scale);
  warp::log(norm_vec, norm_vec);
  warp::add(norm_vec, norm_vec, max_vec_scaled);
  warp::store(g.partial_lse, norm_vec, {b, kvh, chunk, 0});
}

template <int D>
__global__ void streamattn_tk_tc_exact_decode_chunk_staged_grouped_kernel(
    const __grid_constant__ streamattn_tc_chunk_globals g) {
  using T = streamattn_tc_exact_tiles<D>;
  constexpr int producer_warps = 4;
  const int producer_warp = threadIdx.x >> 5;
  const int grouped_pid = blockIdx.x;
  const int grouped_chunk = grouped_pid % g.num_chunks;
  const int bh = grouped_pid / g.num_chunks;
  const int b = bh / g.Hkv;
  const int kvh = bh - b * g.Hkv;
  const int logical_chunk = grouped_chunk * producer_warps + producer_warp;

  __shared__ typename T::kv_smem k_smem[producer_warps][2];
  __shared__ typename T::kv_smem v_smem[producer_warps][2];
  __shared__ typename T::kv_smem out_smem[producer_warps];
  __shared__ typename T::lse_smem lse_smem[producer_warps];
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
  const int tile_begin = logical_chunk * g.tiles_per_chunk;
  const int tile_end = min(tile_begin + g.tiles_per_chunk, g.N / 16);
  int stage = 0;

  warp::load_async(k_smem[producer_warp][stage], g.k, {b, kvh, tile_begin, 0});
  warp::load_async(v_smem[producer_warp][stage], g.v, {b, kvh, tile_begin, 0});
  warp::load_async_wait();

  for (int tile = tile_begin; tile < tile_end; ++tile) {
    warp::load(k_reg, k_smem[producer_warp][stage]);
    warp::load(v_reg, v_smem[producer_warp][stage]);

    const int next_tile = tile + 1;
    const int next_stage = stage ^ 1;
    if (next_tile < tile_end) {
      warp::load_async(k_smem[producer_warp][next_stage], g.k, {b, kvh, next_tile, 0});
      warp::load_async(v_smem[producer_warp][next_stage], g.v, {b, kvh, next_tile, 0});
    }

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
    warp::mma_AB(acc, scores_mma, v_reg, acc);

    if (next_tile < tile_end) {
      warp::load_async_wait();
      stage = next_stage;
    }
  }

  warp::div_row(acc, acc, norm_vec);
  warp::mul(max_vec_scaled, max_vec, scale);
  warp::log(norm_vec, norm_vec);
  warp::add(norm_vec, norm_vec, max_vec_scaled);
  warp::store(out_smem[producer_warp], acc);
  warp::store(lse_smem[producer_warp], norm_vec);
  __syncthreads();

  if (producer_warp == 0) {
    typename T::out_tile merged_out;
    typename T::out_tile partial_out;
    typename T::score_col merged_lse;
    typename T::score_col partial_lse;
    typename T::score_col new_lse;
    typename T::score_col merged_weight;
    typename T::score_col partial_weight;
    typename T::score_col weight_sum;
    warp::load(merged_out, out_smem[0]);
    warp::load(merged_lse, lse_smem[0]);

    #pragma unroll
    for (int producer = 1; producer < producer_warps; ++producer) {
      warp::load(partial_out, out_smem[producer]);
      warp::load(partial_lse, lse_smem[producer]);
      warp::max(new_lse, merged_lse, partial_lse);
      warp::sub(merged_weight, merged_lse, new_lse);
      warp::exp(merged_weight, merged_weight);
      warp::sub(partial_weight, partial_lse, new_lse);
      warp::exp(partial_weight, partial_weight);
      warp::add(weight_sum, merged_weight, partial_weight);
      warp::div(merged_weight, merged_weight, weight_sum);
      warp::div(partial_weight, partial_weight, weight_sum);
      warp::log(weight_sum, weight_sum);
      warp::add(new_lse, new_lse, weight_sum);
      warp::mul_row(merged_out, merged_out, merged_weight);
      warp::mul_row(partial_out, partial_out, partial_weight);
      warp::add(merged_out, merged_out, partial_out);
      warp::copy(merged_lse, new_lse);
    }

    warp::store(g.partial_out, merged_out, {b, kvh, grouped_chunk, 0});
    warp::store(g.partial_lse, merged_lse, {b, kvh, grouped_chunk, 0});
  }
}

template <int D, int producer_warps>
__global__ void streamattn_tk_tc_exact_decode_chunk_staged_grouped_direct_kernel(
    const __grid_constant__ streamattn_tc_grouped_direct_globals g) {
  using T = streamattn_tc_exact_tiles<D>;
  const int producer_warp = threadIdx.x >> 5;
  const int grouped_pid = blockIdx.x;
  const int grouped_chunk = grouped_pid % g.num_chunks;
  const int bh = grouped_pid / g.num_chunks;
  const int b = bh / g.Hkv;
  const int kvh = bh - b * g.Hkv;
  const int logical_chunk = grouped_chunk * producer_warps + producer_warp;

  __shared__ typename T::kv_smem k_smem[producer_warps][2];
  __shared__ typename T::kv_smem v_smem[producer_warps][2];
  __shared__ typename T::kv_smem out_smem[producer_warps];
  __shared__ typename T::lse_smem lse_smem[producer_warps];
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

  streamattn_tc_load_grouped_query<D>(
      q_reg, g.q, b, g.Hq, kvh, g.group_size);
  warp::zero(acc);
  warp::zero(norm_vec);
  warp::neg_infty(max_vec);

  const float scale = rsqrtf(static_cast<float>(D));
  const float scale_log2 = scale * 1.44269504089f;
  const int tile_begin = logical_chunk * g.tiles_per_chunk;
  const int tile_end = min(tile_begin + g.tiles_per_chunk, g.N / 16);
  int stage = 0;

  warp::load_async(k_smem[producer_warp][stage], g.k, {b, kvh, tile_begin, 0});
  warp::load_async(v_smem[producer_warp][stage], g.v, {b, kvh, tile_begin, 0});
  warp::load_async_wait();

  for (int tile = tile_begin; tile < tile_end; ++tile) {
    warp::load(k_reg, k_smem[producer_warp][stage]);
    warp::load(v_reg, v_smem[producer_warp][stage]);

    const int next_tile = tile + 1;
    const int next_stage = stage ^ 1;
    if (next_tile < tile_end) {
      warp::load_async(k_smem[producer_warp][next_stage], g.k, {b, kvh, next_tile, 0});
      warp::load_async(v_smem[producer_warp][next_stage], g.v, {b, kvh, next_tile, 0});
    }

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
    warp::mma_AB(acc, scores_mma, v_reg, acc);

    if (next_tile < tile_end) {
      warp::load_async_wait();
      stage = next_stage;
    }
  }

  warp::div_row(acc, acc, norm_vec);
  warp::mul(max_vec_scaled, max_vec, scale);
  warp::log(norm_vec, norm_vec);
  warp::add(norm_vec, norm_vec, max_vec_scaled);
  warp::store(out_smem[producer_warp], acc);
  warp::store(lse_smem[producer_warp], norm_vec);
  __syncthreads();

  if (producer_warp == 0) {
    typename T::out_tile merged_out;
    typename T::out_tile partial_out;
    typename T::score_col merged_lse;
    typename T::score_col partial_lse;
    typename T::score_col new_lse;
    typename T::score_col merged_weight;
    typename T::score_col partial_weight;
    typename T::score_col weight_sum;
    warp::load(merged_out, out_smem[0]);
    warp::load(merged_lse, lse_smem[0]);

    #pragma unroll
    for (int producer = 1; producer < producer_warps; ++producer) {
      warp::load(partial_out, out_smem[producer]);
      warp::load(partial_lse, lse_smem[producer]);
      warp::max(new_lse, merged_lse, partial_lse);
      warp::sub(merged_weight, merged_lse, new_lse);
      warp::exp(merged_weight, merged_weight);
      warp::sub(partial_weight, partial_lse, new_lse);
      warp::exp(partial_weight, partial_weight);
      warp::add(weight_sum, merged_weight, partial_weight);
      warp::div(merged_weight, merged_weight, weight_sum);
      warp::div(partial_weight, partial_weight, weight_sum);
      warp::log(weight_sum, weight_sum);
      warp::add(new_lse, new_lse, weight_sum);
      warp::mul_row(merged_out, merged_out, merged_weight);
      warp::mul_row(partial_out, partial_out, partial_weight);
      warp::add(merged_out, merged_out, partial_out);
      warp::copy(merged_lse, new_lse);
    }

    warp::store(g.partial_out, merged_out, {b, kvh, grouped_chunk, 0});
    warp::store(g.partial_lse, merged_lse, {b, kvh, grouped_chunk, 0});
  }
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

__global__ void streamattn_tk_tc_exact_warp_merge_kernel(
    const bf16* __restrict__ partial_out,
    const float* __restrict__ partial_lse,
    bf16* __restrict__ out,
    const int32_t* __restrict__ active_counts,
    int B,
    int Hkv,
    int num_chunks,
    int D) {
  constexpr int warps_per_block = 4;
  constexpr int max_chunks = 128;
  __shared__ float weights[warps_per_block][max_chunks];

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row_pid = blockIdx.x * warps_per_block + warp;
  const int total_rows = B * Hkv * 16;
  if (row_pid >= total_rows) return;

  const int row = row_pid % 16;
  const int kvh = (row_pid / 16) % Hkv;
  const int b = row_pid / (16 * Hkv);
  const int chunks_to_merge = active_counts == nullptr ? num_chunks : active_counts[kvh];

  float lane_max = -INFINITY;
  for (int chunk = lane; chunk < chunks_to_merge; chunk += 32) {
    const int64_t lse_idx = (((int64_t)b * Hkv + kvh) * num_chunks + chunk) * 16 + row;
    lane_max = fmaxf(lane_max, partial_lse[lse_idx]);
  }
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    lane_max = fmaxf(lane_max, __shfl_down_sync(0xffffffff, lane_max, offset));
  }
  const float max_lse = __shfl_sync(0xffffffff, lane_max, 0);

  float lane_den = 0.0f;
  for (int chunk = lane; chunk < chunks_to_merge; chunk += 32) {
    const int64_t lse_idx = (((int64_t)b * Hkv + kvh) * num_chunks + chunk) * 16 + row;
    const float lse = partial_lse[lse_idx];
    const float weight = isfinite(lse) && isfinite(max_lse) ? expf(lse - max_lse) : 0.0f;
    weights[warp][chunk] = weight;
    lane_den += weight;
  }
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    lane_den += __shfl_down_sync(0xffffffff, lane_den, offset);
  }
  const float den = __shfl_sync(0xffffffff, lane_den, 0);
  __syncwarp();

  for (int d = lane; d < D; d += 32) {
    float num = 0.0f;
    for (int chunk = 0; chunk < chunks_to_merge; ++chunk) {
      const int64_t out_idx = ((((int64_t)b * Hkv + kvh) * (num_chunks * 16) +
                                chunk * 16 + row) * D) + d;
      const float value = __bfloat162float(partial_out[out_idx]);
      num += weights[warp][chunk] * (isfinite(value) ? value : 0.0f);
    }
    const int64_t dst_idx = ((((int64_t)b * Hkv + kvh) * 16 + row) * D) + d;
    out[dst_idx] = __float2bfloat16(den > 0.0f ? (num / den) : 0.0f);
  }
}

__global__ void streamattn_tk_tc_exact_warp_merge_direct_kernel(
    const bf16* __restrict__ partial_out,
    const float* __restrict__ partial_lse,
    bf16* __restrict__ out,
    int B,
    int Hq,
    int Hkv,
    int group_size,
    int num_chunks,
    int D) {
  constexpr int warps_per_block = 4;
  constexpr int max_chunks = 128;
  __shared__ float weights[warps_per_block][max_chunks];

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row_pid = blockIdx.x * warps_per_block + warp;
  const int total_rows = B * Hq;
  if (row_pid >= total_rows) return;

  const int qh = row_pid % Hq;
  const int b = row_pid / Hq;
  const int kvh = qh / group_size;
  const int row = qh - kvh * group_size;

  float lane_max = -INFINITY;
  for (int chunk = lane; chunk < num_chunks; chunk += 32) {
    const int64_t lse_idx = (((int64_t)b * Hkv + kvh) * num_chunks + chunk) * 16 + row;
    lane_max = fmaxf(lane_max, partial_lse[lse_idx]);
  }
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    lane_max = fmaxf(lane_max, __shfl_down_sync(0xffffffff, lane_max, offset));
  }
  const float max_lse = __shfl_sync(0xffffffff, lane_max, 0);

  float lane_den = 0.0f;
  for (int chunk = lane; chunk < num_chunks; chunk += 32) {
    const int64_t lse_idx = (((int64_t)b * Hkv + kvh) * num_chunks + chunk) * 16 + row;
    const float lse = partial_lse[lse_idx];
    const float weight = isfinite(lse) && isfinite(max_lse) ? expf(lse - max_lse) : 0.0f;
    weights[warp][chunk] = weight;
    lane_den += weight;
  }
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    lane_den += __shfl_down_sync(0xffffffff, lane_den, offset);
  }
  const float den = __shfl_sync(0xffffffff, lane_den, 0);
  __syncwarp();

  for (int d = lane; d < D; d += 32) {
    float num = 0.0f;
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
      const int64_t partial_idx = ((((int64_t)b * Hkv + kvh) * (num_chunks * 16) +
                                    chunk * 16 + row) * D) + d;
      const float value = __bfloat162float(partial_out[partial_idx]);
      num += weights[warp][chunk] * (isfinite(value) ? value : 0.0f);
    }
    const int64_t dst_idx = ((static_cast<int64_t>(b) * Hq + qh) * D) + d;
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
      0,
      1,
      B * Hkv * chunks};
  STREAMATTN_TK_TC_DISPATCH_D(D, streamattn_tk_tc_exact_decode_chunk_kernel, grid, block, g);
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
  return partial;
}

std::vector<torch::Tensor> streamattn_tk_tc_exact_decode_chunk_states_warpgroup_cuda(
    torch::Tensor q_group,
    torch::Tensor k_group,
    torch::Tensor v_group,
    int64_t num_chunks,
    int64_t producer_warps) {
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
  TORCH_CHECK(producer_warps == 1 || producer_warps == 2 ||
              producer_warps == 4 || producer_warps == 8,
              "producer_warps must be one of 1, 2, 4, or 8");
  TORCH_CHECK((N / 16) % num_chunks == 0, "num_chunks must divide N/16 for this spike");
  const int chunks = static_cast<int>(num_chunks);
  const int tiles_per_chunk = (N / 16) / chunks;
  const int warps = static_cast<int>(producer_warps);
  const int total_tasks = B * Hkv * chunks;

  auto partial = torch::empty({B, Hkv, chunks * padded_rows, D}, q_group.options());
  auto partial_lse = torch::empty({B, Hkv, chunks, padded_rows}, q_group.options().dtype(torch::kFloat32));
  const dim3 grid((total_tasks + warps - 1) / warps);
  const dim3 block(32 * warps);
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
      0,
      warps,
      total_tasks};
  STREAMATTN_TK_TC_DISPATCH_D(D, streamattn_tk_tc_exact_decode_chunk_kernel, grid, block, g);
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
  return {partial, partial_lse};
}

std::vector<torch::Tensor> streamattn_tk_tc_exact_decode_chunk_states_cuda(
    torch::Tensor q_group,
    torch::Tensor k_group,
    torch::Tensor v_group,
    int64_t num_chunks) {
  return streamattn_tk_tc_exact_decode_chunk_states_warpgroup_cuda(
      q_group, k_group, v_group, num_chunks, 1);
}

std::vector<torch::Tensor> streamattn_tk_tc_exact_decode_chunk_states_staged_cuda(
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
  const int total_tasks = B * Hkv * chunks;

  auto partial = torch::empty({B, Hkv, chunks * padded_rows, D}, q_group.options());
  auto partial_lse = torch::empty({B, Hkv, chunks, padded_rows}, q_group.options().dtype(torch::kFloat32));
  const dim3 grid(total_tasks);
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
      0,
      1,
      total_tasks};
  STREAMATTN_TK_TC_DISPATCH_D(
      D, streamattn_tk_tc_exact_decode_chunk_staged_kernel, grid, block, g);
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
  return {partial, partial_lse};
}

std::vector<torch::Tensor> streamattn_tk_tc_exact_decode_chunk_states_staged_grouped_cuda(
    torch::Tensor q_group,
    torch::Tensor k_group,
    torch::Tensor v_group,
    int64_t num_chunks) {
  constexpr int producer_warps = 4;
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
  TORCH_CHECK(D == 64, "grouped staged spike currently supports D=64 only");
  TORCH_CHECK(padded_rows == 16, "only 16 padded Q rows are implemented");
  TORCH_CHECK(N % 16 == 0, "N must be divisible by 16");
  TORCH_CHECK(num_chunks > 0 && num_chunks % producer_warps == 0,
              "num_chunks must be positive and divisible by producer warps");
  TORCH_CHECK((N / 16) % num_chunks == 0, "num_chunks must divide N/16 for this spike");
  const int logical_chunks = static_cast<int>(num_chunks);
  const int grouped_chunks = logical_chunks / producer_warps;
  const int tiles_per_chunk = (N / 16) / logical_chunks;
  const int total_tasks = B * Hkv * grouped_chunks;

  auto partial = torch::empty({B, Hkv, grouped_chunks * padded_rows, D}, q_group.options());
  auto partial_lse = torch::empty(
      {B, Hkv, grouped_chunks, padded_rows}, q_group.options().dtype(torch::kFloat32));
  const dim3 grid(total_tasks);
  const dim3 block(32 * producer_warps);
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
           static_cast<unsigned long>(grouped_chunks * padded_rows),
           static_cast<unsigned long>(D)},
      streamattn_tc_chunk_globals::lse_gl{partial_lse.data_ptr<float>(),
           static_cast<unsigned long>(B),
           static_cast<unsigned long>(Hkv),
           static_cast<unsigned long>(grouped_chunks),
           static_cast<unsigned long>(padded_rows)},
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      N,
      Hkv,
      grouped_chunks,
      tiles_per_chunk,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      producer_warps,
      total_tasks};
  streamattn_tk_tc_exact_decode_chunk_staged_grouped_kernel<64><<<grid, block>>>(g);
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
  return {partial, partial_lse};
}

torch::Tensor streamattn_tk_tc_exact_decode_chunk_merged_staged_grouped_direct_out_cuda(
    torch::Tensor q,
    torch::Tensor k_group,
    torch::Tensor v_group,
    torch::Tensor partial,
    torch::Tensor partial_lse,
    torch::Tensor out,
    int64_t num_chunks) {
  TORCH_CHECK(q.is_cuda() && k_group.is_cuda() && v_group.is_cuda(),
              "Q/K/V must be CUDA");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous [B,Hq,D]");
  TORCH_CHECK(k_group.is_contiguous(), "k_group must be contiguous [B,Hkv,N,D]");
  TORCH_CHECK(v_group.is_contiguous(), "v_group must be contiguous [B,Hkv,N,D]");
  TORCH_CHECK(q.scalar_type() == at::ScalarType::BFloat16,
              "q must be bf16");
  TORCH_CHECK(k_group.scalar_type() == at::ScalarType::BFloat16 &&
              v_group.scalar_type() == at::ScalarType::BFloat16,
              "K/V must be bf16");
  TORCH_CHECK(q.dim() == 3 && k_group.dim() == 4 && v_group.sizes() == k_group.sizes(),
              "expected q [B,Hq,D] and matching K/V [B,Hkv,N,D]");
  const int B = q.size(0);
  const int Hq = q.size(1);
  const int D = q.size(2);
  const int Hkv = k_group.size(1);
  const int N = k_group.size(2);
  TORCH_CHECK(k_group.size(0) == B && k_group.size(3) == D,
              "K shape incompatible with Q");
  TORCH_CHECK(Hkv > 0 && Hq % Hkv == 0, "Hq must be divisible by Hkv");
  const int group_size = Hq / Hkv;
  TORCH_CHECK(group_size == 4 || group_size == 8,
              "direct grouped path currently supports G4 or G8");
  TORCH_CHECK(D == 64 || D == 128,
              "direct grouped path currently supports D64 or D128");
  const int producer_warps = D == 128 ? 2 : 4;
  TORCH_CHECK(N % 16 == 0, "N must be divisible by 16");
  TORCH_CHECK(num_chunks > 0 && num_chunks % producer_warps == 0,
              "num_chunks must be positive and divisible by producer warps");
  TORCH_CHECK((N / 16) % num_chunks == 0,
              "num_chunks must divide N/16");
  const int logical_chunks = static_cast<int>(num_chunks);
  const int grouped_chunks = logical_chunks / producer_warps;
  TORCH_CHECK(grouped_chunks <= 128, "grouped merge supports at most 128 chunks");
  const int tiles_per_chunk = (N / 16) / logical_chunks;
  TORCH_CHECK(partial.is_cuda() && partial_lse.is_cuda() && out.is_cuda(),
              "partial/LSE/output must be CUDA");
  TORCH_CHECK(partial.is_contiguous() && partial_lse.is_contiguous() && out.is_contiguous(),
              "partial/LSE/output must be contiguous");
  TORCH_CHECK(partial.scalar_type() == at::ScalarType::BFloat16,
              "partial output must be bf16");
  TORCH_CHECK(partial_lse.scalar_type() == at::ScalarType::Float,
              "partial LSE must be fp32");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::BFloat16,
              "output must be bf16");
  TORCH_CHECK(partial.sizes() == torch::IntArrayRef({B, Hkv, grouped_chunks * 16, D}),
              "partial output has the wrong shape");
  TORCH_CHECK(partial_lse.sizes() == torch::IntArrayRef({B, Hkv, grouped_chunks, 16}),
              "partial LSE has the wrong shape");
  TORCH_CHECK(out.sizes() == q.sizes(), "output must match Q shape");
  using kv_gl = streamattn_tc_grouped_direct_globals::kv_gl;
  using partial_gl = streamattn_tc_grouped_direct_globals::partial_gl;
  streamattn_tc_grouped_direct_globals g{
      reinterpret_cast<const bf16*>(q.data_ptr<at::BFloat16>()),
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
      partial_gl{reinterpret_cast<bf16*>(partial.data_ptr<at::BFloat16>()),
            static_cast<unsigned long>(B),
            static_cast<unsigned long>(Hkv),
            static_cast<unsigned long>(grouped_chunks * 16),
            static_cast<unsigned long>(D)},
      streamattn_tc_grouped_direct_globals::lse_gl{partial_lse.data_ptr<float>(),
            static_cast<unsigned long>(B),
            static_cast<unsigned long>(Hkv),
            static_cast<unsigned long>(grouped_chunks),
            16ul},
      Hq,
      Hkv,
      group_size,
      N,
      grouped_chunks,
      tiles_per_chunk};
  const dim3 producer_grid(B * Hkv * grouped_chunks);
  const dim3 producer_block(32 * producer_warps);
  if (D == 64) {
    streamattn_tk_tc_exact_decode_chunk_staged_grouped_direct_kernel<64, 4>
        <<<producer_grid, producer_block>>>(g);
  } else {
    streamattn_tk_tc_exact_decode_chunk_staged_grouped_direct_kernel<128, 2>
        <<<producer_grid, producer_block>>>(g);
  }
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));

  const dim3 merge_grid((B * Hq + 3) / 4);
  const dim3 merge_block(128);
  streamattn_tk_tc_exact_warp_merge_direct_kernel<<<merge_grid, merge_block>>>(
      reinterpret_cast<const bf16*>(partial.data_ptr<at::BFloat16>()),
      partial_lse.data_ptr<float>(),
      reinterpret_cast<bf16*>(out.data_ptr<at::BFloat16>()),
      B,
      Hq,
      Hkv,
      group_size,
      grouped_chunks,
      D);
  err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
  return out;
}

torch::Tensor streamattn_tk_tc_exact_decode_chunk_merged_staged_grouped_direct_cuda(
    torch::Tensor q,
    torch::Tensor k_group,
    torch::Tensor v_group,
    int64_t num_chunks) {
  TORCH_CHECK(q.dim() == 3 && k_group.dim() == 4,
              "expected q [B,Hq,D] and K [B,Hkv,N,D]");
  const int B = q.size(0);
  const int Hkv = k_group.size(1);
  const int D = q.size(2);
  TORCH_CHECK(D == 64 || D == 128,
              "direct grouped path currently supports D64 or D128");
  const int producer_warps = D == 128 ? 2 : 4;
  const int grouped_chunks = static_cast<int>(num_chunks) / producer_warps;
  auto partial = torch::empty({B, Hkv, grouped_chunks * 16, D}, q.options());
  auto partial_lse = torch::empty(
      {B, Hkv, grouped_chunks, 16}, q.options().dtype(torch::kFloat32));
  auto out = torch::empty_like(q);
  return streamattn_tk_tc_exact_decode_chunk_merged_staged_grouped_direct_out_cuda(
      q, k_group, v_group, partial, partial_lse, out, num_chunks);
}

torch::Tensor streamattn_tk_tc_exact_decode_chunk_merged_warpgroup_cuda(
    torch::Tensor q_group,
    torch::Tensor k_group,
    torch::Tensor v_group,
    int64_t num_chunks,
    int64_t producer_warps);

torch::Tensor streamattn_tk_tc_exact_decode_chunk_merged_cuda(
    torch::Tensor q_group,
    torch::Tensor k_group,
    torch::Tensor v_group,
    int64_t num_chunks) {
  return streamattn_tk_tc_exact_decode_chunk_merged_warpgroup_cuda(
      q_group, k_group, v_group, num_chunks, 1);
}

torch::Tensor streamattn_tk_tc_exact_decode_chunk_merged_warpgroup_cuda(
    torch::Tensor q_group,
    torch::Tensor k_group,
    torch::Tensor v_group,
    int64_t num_chunks,
    int64_t producer_warps) {
  auto states = streamattn_tk_tc_exact_decode_chunk_states_warpgroup_cuda(
      q_group, k_group, v_group, num_chunks, producer_warps);
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

torch::Tensor streamattn_tk_tc_exact_decode_chunk_merged_staged_cuda(
    torch::Tensor q_group,
    torch::Tensor k_group,
    torch::Tensor v_group,
    int64_t num_chunks) {
  auto states = streamattn_tk_tc_exact_decode_chunk_states_staged_cuda(
      q_group, k_group, v_group, num_chunks);
  auto partial = states[0];
  auto partial_lse = states[1];
  const int B = q_group.size(0);
  const int Hkv = q_group.size(1);
  const int padded_rows = q_group.size(2);
  const int D = q_group.size(3);
  const int chunks = static_cast<int>(num_chunks);
  TORCH_CHECK(chunks <= 128, "staged warp merge supports at most 128 chunks");
  auto out = torch::empty_like(q_group);
  const dim3 grid((B * Hkv * padded_rows + 3) / 4);
  const dim3 block(128);
  streamattn_tk_tc_exact_warp_merge_kernel<<<grid, block>>>(
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

torch::Tensor streamattn_tk_tc_exact_decode_chunk_merged_staged_grouped_cuda(
    torch::Tensor q_group,
    torch::Tensor k_group,
    torch::Tensor v_group,
    int64_t num_chunks) {
  auto states = streamattn_tk_tc_exact_decode_chunk_states_staged_grouped_cuda(
      q_group, k_group, v_group, num_chunks);
  auto partial = states[0];
  auto partial_lse = states[1];
  const int B = q_group.size(0);
  const int Hkv = q_group.size(1);
  const int padded_rows = q_group.size(2);
  const int D = q_group.size(3);
  const int grouped_chunks = static_cast<int>(num_chunks) / 4;
  auto out = torch::empty_like(q_group);
  const dim3 grid((B * Hkv * padded_rows + 3) / 4);
  const dim3 block(128);
  streamattn_tk_tc_exact_warp_merge_kernel<<<grid, block>>>(
      reinterpret_cast<const bf16*>(partial.data_ptr<at::BFloat16>()),
      partial_lse.data_ptr<float>(),
      reinterpret_cast<bf16*>(out.data_ptr<at::BFloat16>()),
      nullptr,
      B,
      Hkv,
      grouped_chunks,
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
      0,
      1,
      B * Hkv * chunks};
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
      1,
      1,
      B * total_active_entries};
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
