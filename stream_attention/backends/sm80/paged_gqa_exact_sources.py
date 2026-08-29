"""Embedded C++/CUDA source for Ampere page-16 NHD D128/G8 decode."""

CPP_SOURCE = r"""
#include <torch/extension.h>

void streamattn_sm80_paged_gqa_exact_decode_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor page_table,
    torch::Tensor sequence_lengths,
    torch::Tensor partial_o,
    torch::Tensor partial_lse,
    torch::Tensor output,
    int64_t num_splits,
    bool hnd_layout,
    int64_t merge_segments,
    int64_t producer_tile);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("paged_exact_decode_out",
        &streamattn_sm80_paged_gqa_exact_decode_out_cuda,
        "StreamAttn SM80 direct-NHD paged GQA exact decode");
}
"""


CUDA_SOURCE = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cute/tensor.hpp>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

using namespace cute;

using Element = cutlass::bfloat16_t;
using Accum = float;

static constexpr int kBlockM = 64;
static constexpr int kBlockN = 8;
static constexpr int kHeadDim = 128;
static constexpr int kThreads = 128;
static constexpr int kStages = 2;
static constexpr int kBlockM128 = 128;
static constexpr int kThreads128 = 256;

using SmemAtom = decltype(composition(
    Swizzle<3, 3, 3>{},
    Layout<Shape<_8, Shape<_8, _8>>, Stride<_8, Stride<_1, _64>>>{}));
using SmemLayoutK = decltype(tile_to_shape(
    SmemAtom{}, Shape<Int<kBlockM>, Int<kHeadDim>, Int<kStages>>{}));
using SmemLayoutV = decltype(tile_to_shape(
    SmemAtom{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
using SmemLayoutQ = decltype(tile_to_shape(
    SmemAtom{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));
using SmemLayoutP = decltype(tile_to_shape(
    SmemAtom{}, Shape<Int<kBlockN>, Int<kBlockM>>{}));
using SmemLayoutVt = decltype(composition(
    SmemLayoutV{},
    make_layout(Shape<Int<kHeadDim>, Int<kBlockM>>{}, GenRowMajor{})));
using SmemLayoutPOrigin = decltype(composition(
    SmemLayoutP{},
    make_layout(Shape<Int<kBlockM>, Int<kBlockN>>{}, GenRowMajor{})));

using TiledMmaQK = decltype(make_tiled_mma(
    SM80_16x8x16_F32BF16BF16F32_TN{},
    Layout<Shape<_4, _1>>{},
    Tile<Int<kBlockM>, Int<kBlockN>, _16>{}));
using TiledMmaPV = decltype(make_tiled_mma(
    SM80_16x8x16_F32BF16BF16F32_TN{},
    Layout<Shape<_4, _1>>{},
    Tile<_64, Int<kBlockN>, _16>{}));
using S2RAtomA = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
using S2RAtomB = Copy_Atom<SM75_U32x2_LDSM_N, Element>;
using S2RAtomPVA = Copy_Atom<UniversalCopy<Element>, Element>;

struct alignas(128) SharedStorage {
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> k;
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> v;
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> q;
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutPOrigin>> p;
  Accum warp_reduce[4][kBlockN];
  Accum row_sum[kBlockN];
};

using SmemLayoutK128 = decltype(tile_to_shape(
    SmemAtom{}, Shape<Int<kBlockM128>, Int<kHeadDim>, Int<kStages>>{}));
using SmemLayoutV128 = decltype(tile_to_shape(
    SmemAtom{}, Shape<Int<kBlockM128>, Int<kHeadDim>>{}));
using SmemLayoutP128 = decltype(tile_to_shape(
    SmemAtom{}, Shape<Int<kBlockN>, Int<kBlockM128>>{}));
using SmemLayoutVt128 = decltype(composition(
    SmemLayoutV128{},
    make_layout(Shape<Int<kHeadDim>, Int<kBlockM128>>{}, GenRowMajor{})));
using SmemLayoutPOrigin128 = decltype(composition(
    SmemLayoutP128{},
    make_layout(Shape<Int<kBlockM128>, Int<kBlockN>>{}, GenRowMajor{})));
using TiledMmaQK128 = decltype(make_tiled_mma(
    SM80_16x8x16_F32BF16BF16F32_TN{},
    Layout<Shape<_8, _1>>{},
    Tile<Int<kBlockM128>, Int<kBlockN>, _16>{}));
using TiledMmaPV128 = decltype(make_tiled_mma(
    SM80_16x8x16_F32BF16BF16F32_TN{},
    Layout<Shape<_8, _1>>{},
    Tile<Int<kHeadDim>, Int<kBlockN>, _16>{}));

struct alignas(128) SharedStorage128 {
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutK128>> k;
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutV128>> v;
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> q;
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutPOrigin128>> p;
  Accum warp_reduce[8][kBlockN];
  Accum row_sum[kBlockN];
};

__forceinline__ __device__ unsigned streamattn_head_pair_mask() {
  return 0x11111111u << ((threadIdx.x & 31) & 3);
}

__forceinline__ __device__ Accum streamattn_group_max(Accum value) {
  const unsigned mask = streamattn_head_pair_mask();
  value = fmaxf(value, __shfl_xor_sync(mask, value, 16));
  value = fmaxf(value, __shfl_xor_sync(mask, value, 8));
  value = fmaxf(value, __shfl_xor_sync(mask, value, 4));
  return value;
}

__forceinline__ __device__ Accum streamattn_group_sum(Accum value) {
  const unsigned mask = streamattn_head_pair_mask();
  value += __shfl_xor_sync(mask, value, 16);
  value += __shfl_xor_sync(mask, value, 8);
  value += __shfl_xor_sync(mask, value, 4);
  return value;
}

template <int kTileM, bool kHnd, class SmemTensor>
__forceinline__ __device__ void streamattn_cp_async_tile(
    const Element* base,
    const int* page_table,
    int batch,
    int kv_head,
    int kv_heads,
    int max_pages,
    int tile,
    SmemTensor const& destination) {
  constexpr int kVectorsPerRow = kHeadDim / 8;
  constexpr int kVectorsPerTile = kTileM * kVectorsPerRow;
  CUTE_UNROLL
  for (int vector = threadIdx.x; vector < kVectorsPerTile;
       vector += blockDim.x) {
    const int row = vector / kVectorsPerRow;
    const int dim = (vector - row * kVectorsPerRow) * 8;
    const int logical_token = tile * kTileM + row;
    const int logical_page = logical_token / 16;
    const int token_in_page = logical_token - logical_page * 16;
    // A warp covers two rows per vector iteration, and both rows are always in
    // the same 16-token page. Broadcast one metadata load instead of issuing
    // 32 identical page-table loads.
    int physical_page = 0;
    if ((threadIdx.x & 31) == 0) {
      physical_page = page_table[batch * max_pages + logical_page];
    }
    physical_page = __shfl_sync(0xffffffffu, physical_page, 0);
    const int64_t token_offset = kHnd
        ? ((static_cast<int64_t>(physical_page) * kv_heads + kv_head) * 16 +
           token_in_page)
        : ((static_cast<int64_t>(physical_page) * 16 + token_in_page) *
               kv_heads +
           kv_head);
    const Element* source = base + token_offset * kHeadDim + dim;
    auto const& source_vector =
        *reinterpret_cast<cute::uint128_t const*>(source);
    auto& destination_vector = *reinterpret_cast<cute::uint128_t*>(
        &destination(row, dim));
    SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>::copy(
        source_vector, destination_vector);
  }
}

template <class SmemTensor>
__forceinline__ __device__ void streamattn_copy_q(
    const Element* source, SmemTensor const& destination) {
  constexpr int kVectors = kBlockN * kHeadDim / 8;
  if (threadIdx.x < kVectors) {
    const int head = threadIdx.x / (kHeadDim / 8);
    const int dim = (threadIdx.x - head * (kHeadDim / 8)) * 8;
    *reinterpret_cast<cute::uint128_t*>(&destination(head, dim)) =
        *reinterpret_cast<cute::uint128_t const*>(
            source + static_cast<int64_t>(head) * kHeadDim + dim);
  }
}

template <bool kHnd>
__global__ __launch_bounds__(kThreads)
void streamattn_sm80_paged_gqa_partial_kernel(
    const Element* __restrict__ q_group,
    const Element* __restrict__ k_pages,
    const Element* __restrict__ v_pages,
    const int* __restrict__ page_table,
    const int* __restrict__ sequence_lengths,
    Accum* __restrict__ partial_o,
    Accum* __restrict__ partial_lse,
    int groups,
    int kv_heads,
    int max_pages,
    int num_splits) {
  const int work = blockIdx.x;
  const int group = work / num_splits;
  const int split = work - group * num_splits;
  if (group >= groups) {
    return;
  }
  const int batch = group / kv_heads;
  const int kv_head = group - batch * kv_heads;
  const int sequence_length = sequence_lengths[batch];
  const int num_tiles = (sequence_length + kBlockM - 1) / kBlockM;
  const int tiles_per_split = (num_tiles + num_splits - 1) / num_splits;
  const int tile_begin = split * tiles_per_split;
  const int tile_end = min(num_tiles, tile_begin + tiles_per_split);
  if (tile_begin >= tile_end) {
    for (int idx = threadIdx.x; idx < kBlockN * kHeadDim; idx += blockDim.x) {
      partial_o[static_cast<int64_t>(work) * kBlockN * kHeadDim + idx] = 0.0f;
    }
    if (threadIdx.x < kBlockN) {
      partial_lse[static_cast<int64_t>(work) * kBlockN + threadIdx.x] =
          -INFINITY;
    }
    return;
  }

  extern __shared__ char shared_bytes[];
  SharedStorage& storage = *reinterpret_cast<SharedStorage*>(shared_bytes);
  Tensor sK = make_tensor(make_smem_ptr(storage.k.data()), SmemLayoutK{});
  Tensor sV = make_tensor(make_smem_ptr(storage.v.data()), SmemLayoutV{});
  Tensor sVt = make_tensor(make_smem_ptr(storage.v.data()), SmemLayoutVt{});
  Tensor sVt0 = local_tile(
      sVt, Shape<_64, Int<kBlockM>>{}, make_coord(Int<0>{}, Int<0>{}));
  Tensor sVt1 = local_tile(
      sVt, Shape<_64, Int<kBlockM>>{}, make_coord(Int<1>{}, Int<0>{}));
  Tensor sQ = make_tensor(make_smem_ptr(storage.q.data()), SmemLayoutQ{});
  Tensor sPOrigin =
      make_tensor(make_smem_ptr(storage.p.data()), SmemLayoutPOrigin{});
  Tensor sP = make_tensor(make_smem_ptr(storage.p.data()), SmemLayoutP{});

  const Element* q_ptr = q_group +
      static_cast<int64_t>(group) * kBlockN * kHeadDim;
  streamattn_copy_q(q_ptr, sQ);
  streamattn_cp_async_tile<kBlockM, kHnd>(
      k_pages, page_table, batch, kv_head, kv_heads, max_pages, tile_begin,
      sK(_, _, 0));
  cute::cp_async_fence();
  cute::cp_async_wait<0>();
  __syncthreads();

  TiledMmaQK tiled_mma_qk;
  auto thr_mma_qk = tiled_mma_qk.get_thread_slice(threadIdx.x);
  S2RAtomA s2r_atom_a;
  S2RAtomB s2r_atom_b;
  auto s2r_qk_a = make_tiled_copy_A(s2r_atom_a, tiled_mma_qk);
  auto s2r_qk_b = make_tiled_copy_B(s2r_atom_b, tiled_mma_qk);
  auto thr_qk_a = s2r_qk_a.get_thread_slice(threadIdx.x);
  auto thr_qk_b = s2r_qk_b.get_thread_slice(threadIdx.x);
  Tensor tXrK = thr_qk_a.retile_D(
      thr_mma_qk.partition_fragment_A(sK(_, _, 0)));
  Tensor tXrQ = thr_qk_b.retile_D(thr_mma_qk.partition_fragment_B(sQ));
  Tensor tXsQ = thr_qk_b.partition_S(sQ);

  TiledMmaPV tiled_mma_pv;
  auto thr_mma_pv = tiled_mma_pv.get_thread_slice(threadIdx.x);
  Tensor tOrO0 = partition_fragment_C(
      tiled_mma_pv, Shape<_64, Int<kBlockN>>{});
  Tensor tOrO1 = partition_fragment_C(
      tiled_mma_pv, Shape<_64, Int<kBlockN>>{});
  clear(tOrO0);
  clear(tOrO1);

  constexpr int kRowsPerThread = 2;
  Accum row_max[kRowsPerThread];
  Accum row_sum[kRowsPerThread];
  CUTE_UNROLL
  for (int row = 0; row < kRowsPerThread; ++row) {
    row_max[row] = -INFINITY;
    row_sum[row] = 0.0f;
  }

  constexpr Accum kSoftmaxScaleLog2 = 0.12751743082459868f;
  int read_pipe = 0;
  for (int tile = tile_begin; tile < tile_end; ++tile) {
    const int next_tile = tile + 1;
    const int write_pipe = read_pipe ^ 1;
    if (next_tile < tile_end) {
      streamattn_cp_async_tile<kBlockM, kHnd>(
          k_pages, page_table, batch, kv_head, kv_heads, max_pages, next_tile,
          sK(_, _, write_pipe));
      cute::cp_async_fence();
    }

    Tensor tCrS = partition_fragment_C(
        tiled_mma_qk, Shape<Int<kBlockM>, Int<kBlockN>>{});
    clear(tCrS);
    auto tXsK = thr_qk_a.partition_S(sK(_, _, read_pipe));
    constexpr int kKBlocks = kHeadDim / 16;
    CUTE_UNROLL
    for (int k_block = 0; k_block < kKBlocks; ++k_block) {
      copy(s2r_atom_a, tXsK(_, _, k_block), tXrK(_, _, k_block));
      copy(s2r_atom_b, tXsQ(_, _, k_block), tXrQ(_, _, k_block));
      gemm(tiled_mma_qk, tXrK(_, _, k_block), tXrQ(_, _, k_block), tCrS);
    }

    streamattn_cp_async_tile<kBlockM, kHnd>(
        v_pages, page_table, batch, kv_head, kv_heads, max_pages, tile, sV);
    cute::cp_async_fence();

    Accum* scores = tCrS.data();
    Accum alpha[kRowsPerThread];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int lane_in_group = lane >> 2;
    const int head_pair = lane & 3;
    const int head0 = head_pair * 2;
    const int head1 = head0 + 1;
    const int token0 = warp * 16 + lane_in_group;
    const int token1 = token0 + 8;
    if (tile * kBlockM + token0 >= sequence_length) {
      scores[0] = -INFINITY;
      scores[1] = -INFINITY;
    }
    if (tile * kBlockM + token1 >= sequence_length) {
      scores[2] = -INFINITY;
      scores[3] = -INFINITY;
    }

    CUTE_UNROLL
    for (int row = 0; row < kRowsPerThread; ++row) {
      Accum tile_max = fmaxf(scores[row], scores[row + 2]);
      tile_max = streamattn_group_max(tile_max);
      const int head = head0 + row;
      if (lane_in_group == 0) {
        storage.warp_reduce[warp][head] = tile_max;
      }
    }
    __syncthreads();

    CUTE_UNROLL
    for (int row = 0; row < kRowsPerThread; ++row) {
      const int head = head0 + row;
      Accum tile_max = storage.warp_reduce[0][head];
      CUTE_UNROLL
      for (int warp_idx = 1; warp_idx < 4; ++warp_idx) {
        tile_max = fmaxf(tile_max, storage.warp_reduce[warp_idx][head]);
      }
      const Accum next_max = fmaxf(row_max[row], tile_max);
      alpha[row] = row_max[row] == -INFINITY
          ? 0.0f
          : exp2f((row_max[row] - next_max) * kSoftmaxScaleLog2);
      row_max[row] = next_max;
      row_sum[row] *= alpha[row];
      const Accum max_scaled = next_max * kSoftmaxScaleLog2;
      scores[row] = exp2f(scores[row] * kSoftmaxScaleLog2 - max_scaled);
      scores[row + 2] =
          exp2f(scores[row + 2] * kSoftmaxScaleLog2 - max_scaled);
      row_sum[row] += scores[row] + scores[row + 2];
    }
    Accum* output_values0 = tOrO0.data();
    Accum* output_values1 = tOrO1.data();
    CUTE_UNROLL
    for (int value = 0; value < 4; ++value) {
      const Accum scale = alpha[value & 1];
      output_values0[value] *= scale;
      output_values1[value] *= scale;
    }

    sPOrigin(token0, head0) = Element(scores[0]);
    sPOrigin(token0, head1) = Element(scores[1]);
    sPOrigin(token1, head0) = Element(scores[2]);
    sPOrigin(token1, head1) = Element(scores[3]);
    cute::cp_async_wait<0>();
    __syncthreads();

    S2RAtomPVA s2r_atom_pv_a;
    auto s2r_pv_a = make_tiled_copy_A(s2r_atom_pv_a, tiled_mma_pv);
    auto s2r_pv_b = make_tiled_copy_B(s2r_atom_b, tiled_mma_pv);
    auto thr_pv_a = s2r_pv_a.get_thread_slice(threadIdx.x);
    auto thr_pv_b = s2r_pv_b.get_thread_slice(threadIdx.x);
    Tensor tXrV0 =
        thr_pv_a.retile_D(thr_mma_pv.partition_fragment_A(sVt0));
    Tensor tXrV1 =
        thr_pv_a.retile_D(thr_mma_pv.partition_fragment_A(sVt1));
    Tensor tXrP = thr_pv_b.retile_D(thr_mma_pv.partition_fragment_B(sP));
    Tensor tXsV0 = thr_pv_a.partition_S(sVt0);
    Tensor tXsV1 = thr_pv_a.partition_S(sVt1);
    Tensor tXsP = thr_pv_b.partition_S(sP);
    constexpr int kPBlocks = kBlockM / 16;
    CUTE_UNROLL
    for (int k_block = 0; k_block < kPBlocks; ++k_block) {
      copy(s2r_atom_pv_a, tXsV0(_, _, k_block), tXrV0(_, _, k_block));
      copy(s2r_atom_pv_a, tXsV1(_, _, k_block), tXrV1(_, _, k_block));
      copy(s2r_atom_b, tXsP(_, _, k_block), tXrP(_, _, k_block));
      gemm(
          tiled_mma_pv, tXrV0(_, _, k_block), tXrP(_, _, k_block), tOrO0);
      gemm(
          tiled_mma_pv, tXrV1(_, _, k_block), tXrP(_, _, k_block), tOrO1);
    }

    if (next_tile < tile_end) {
      read_pipe = write_pipe;
    }
  }

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int lane_in_group = lane >> 2;
  const int head_pair = lane & 3;
  const int head0 = head_pair * 2;
  const int head1 = head0 + 1;
  CUTE_UNROLL
  for (int row = 0; row < kRowsPerThread; ++row) {
    const int head = head0 + row;
    const Accum total = streamattn_group_sum(row_sum[row]);
    if (lane_in_group == 0) {
      storage.warp_reduce[warp][head] = total;
    }
  }
  __syncthreads();
  if (warp == 0 && lane_in_group == 0) {
    CUTE_UNROLL
    for (int row = 0; row < kRowsPerThread; ++row) {
      const int head = head0 + row;
      Accum total = 0.0f;
      CUTE_UNROLL
      for (int warp_idx = 0; warp_idx < 4; ++warp_idx) {
        total += storage.warp_reduce[warp_idx][head];
      }
      storage.row_sum[head] = total;
      partial_lse[static_cast<int64_t>(work) * kBlockN + head] =
          row_max[row] * kSoftmaxScaleLog2 + log2f(total);
    }
  }
  __syncthreads();

  Accum* output_values0 = tOrO0.data();
  Accum* output_values1 = tOrO1.data();
  const int base_dim = warp * 16 + lane_in_group;
  CUTE_UNROLL
  for (int half = 0; half < 2; ++half) {
    Accum* output_values = half == 0 ? output_values0 : output_values1;
    const int dim = base_dim + half * 64;
    partial_o[((static_cast<int64_t>(work) * kBlockN + head0) * kHeadDim) +
              dim] = output_values[0] / storage.row_sum[head0];
    partial_o[((static_cast<int64_t>(work) * kBlockN + head1) * kHeadDim) +
              dim] = output_values[1] / storage.row_sum[head1];
    partial_o[((static_cast<int64_t>(work) * kBlockN + head0) * kHeadDim) +
              dim + 8] = output_values[2] / storage.row_sum[head0];
    partial_o[((static_cast<int64_t>(work) * kBlockN + head1) * kHeadDim) +
              dim + 8] = output_values[3] / storage.row_sum[head1];
  }
}

template <bool kHnd>
__global__ __launch_bounds__(kThreads128)
void streamattn_sm80_paged_gqa_partial_128_kernel(
    const Element* __restrict__ q_group,
    const Element* __restrict__ k_pages,
    const Element* __restrict__ v_pages,
    const int* __restrict__ page_table,
    const int* __restrict__ sequence_lengths,
    Accum* __restrict__ partial_o,
    Accum* __restrict__ partial_lse,
    int groups,
    int kv_heads,
    int max_pages,
    int num_splits) {
  const int work = blockIdx.x;
  const int group = work / num_splits;
  const int split = work - group * num_splits;
  if (group >= groups) {
    return;
  }
  const int batch = group / kv_heads;
  const int kv_head = group - batch * kv_heads;
  const int sequence_length = sequence_lengths[batch];
  const int num_tiles =
      (sequence_length + kBlockM128 - 1) / kBlockM128;
  const int tiles_per_split = (num_tiles + num_splits - 1) / num_splits;
  const int tile_begin = split * tiles_per_split;
  const int tile_end = min(num_tiles, tile_begin + tiles_per_split);
  if (tile_begin >= tile_end) {
    for (int idx = threadIdx.x; idx < kBlockN * kHeadDim; idx += blockDim.x) {
      partial_o[static_cast<int64_t>(work) * kBlockN * kHeadDim + idx] = 0.0f;
    }
    if (threadIdx.x < kBlockN) {
      partial_lse[static_cast<int64_t>(work) * kBlockN + threadIdx.x] =
          -INFINITY;
    }
    return;
  }

  extern __shared__ char shared_bytes[];
  SharedStorage128& storage =
      *reinterpret_cast<SharedStorage128*>(shared_bytes);
  Tensor sK = make_tensor(make_smem_ptr(storage.k.data()), SmemLayoutK128{});
  Tensor sV = make_tensor(make_smem_ptr(storage.v.data()), SmemLayoutV128{});
  Tensor sVt = make_tensor(make_smem_ptr(storage.v.data()), SmemLayoutVt128{});
  Tensor sQ = make_tensor(make_smem_ptr(storage.q.data()), SmemLayoutQ{});
  Tensor sPOrigin = make_tensor(
      make_smem_ptr(storage.p.data()), SmemLayoutPOrigin128{});
  Tensor sP = make_tensor(make_smem_ptr(storage.p.data()), SmemLayoutP128{});

  const Element* q_ptr = q_group +
      static_cast<int64_t>(group) * kBlockN * kHeadDim;
  streamattn_copy_q(q_ptr, sQ);
  streamattn_cp_async_tile<kBlockM128, kHnd>(
      k_pages, page_table, batch, kv_head, kv_heads, max_pages, tile_begin,
      sK(_, _, 0));
  cute::cp_async_fence();
  cute::cp_async_wait<0>();
  __syncthreads();

  TiledMmaQK128 tiled_mma_qk;
  auto thr_mma_qk = tiled_mma_qk.get_thread_slice(threadIdx.x);
  S2RAtomA s2r_atom_a;
  S2RAtomB s2r_atom_b;
  auto s2r_qk_a = make_tiled_copy_A(s2r_atom_a, tiled_mma_qk);
  auto s2r_qk_b = make_tiled_copy_B(s2r_atom_b, tiled_mma_qk);
  auto thr_qk_a = s2r_qk_a.get_thread_slice(threadIdx.x);
  auto thr_qk_b = s2r_qk_b.get_thread_slice(threadIdx.x);
  Tensor tXrK = thr_qk_a.retile_D(
      thr_mma_qk.partition_fragment_A(sK(_, _, 0)));
  Tensor tXrQ = thr_qk_b.retile_D(thr_mma_qk.partition_fragment_B(sQ));
  Tensor tXsQ = thr_qk_b.partition_S(sQ);

  TiledMmaPV128 tiled_mma_pv;
  auto thr_mma_pv = tiled_mma_pv.get_thread_slice(threadIdx.x);
  Tensor tOrO = partition_fragment_C(
      tiled_mma_pv, Shape<Int<kHeadDim>, Int<kBlockN>>{});
  clear(tOrO);

  constexpr int kRowsPerThread = 2;
  Accum row_max[kRowsPerThread];
  Accum row_sum[kRowsPerThread];
  CUTE_UNROLL
  for (int row = 0; row < kRowsPerThread; ++row) {
    row_max[row] = -INFINITY;
    row_sum[row] = 0.0f;
  }

  constexpr Accum kSoftmaxScaleLog2 = 0.12751743082459868f;
  int read_pipe = 0;
  for (int tile = tile_begin; tile < tile_end; ++tile) {
    const int next_tile = tile + 1;
    const int write_pipe = read_pipe ^ 1;
    if (next_tile < tile_end) {
      streamattn_cp_async_tile<kBlockM128, kHnd>(
          k_pages, page_table, batch, kv_head, kv_heads, max_pages, next_tile,
          sK(_, _, write_pipe));
      cute::cp_async_fence();
    }

    Tensor tCrS = partition_fragment_C(
        tiled_mma_qk, Shape<Int<kBlockM128>, Int<kBlockN>>{});
    clear(tCrS);
    auto tXsK = thr_qk_a.partition_S(sK(_, _, read_pipe));
    constexpr int kKBlocks = kHeadDim / 16;
    CUTE_UNROLL
    for (int k_block = 0; k_block < kKBlocks; ++k_block) {
      copy(s2r_atom_a, tXsK(_, _, k_block), tXrK(_, _, k_block));
      copy(s2r_atom_b, tXsQ(_, _, k_block), tXrQ(_, _, k_block));
      gemm(tiled_mma_qk, tXrK(_, _, k_block), tXrQ(_, _, k_block), tCrS);
    }

    streamattn_cp_async_tile<kBlockM128, kHnd>(
        v_pages, page_table, batch, kv_head, kv_heads, max_pages, tile, sV);
    cute::cp_async_fence();

    Accum* scores = tCrS.data();
    Accum alpha[kRowsPerThread];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int lane_in_group = lane >> 2;
    const int head_pair = lane & 3;
    const int head0 = head_pair * 2;
    const int head1 = head0 + 1;
    const int token0 = warp * 16 + lane_in_group;
    const int token1 = token0 + 8;
    if (tile * kBlockM128 + token0 >= sequence_length) {
      scores[0] = -INFINITY;
      scores[1] = -INFINITY;
    }
    if (tile * kBlockM128 + token1 >= sequence_length) {
      scores[2] = -INFINITY;
      scores[3] = -INFINITY;
    }

    CUTE_UNROLL
    for (int row = 0; row < kRowsPerThread; ++row) {
      Accum tile_max = fmaxf(scores[row], scores[row + 2]);
      tile_max = streamattn_group_max(tile_max);
      const int head = head0 + row;
      if (lane_in_group == 0) {
        storage.warp_reduce[warp][head] = tile_max;
      }
    }
    __syncthreads();

    CUTE_UNROLL
    for (int row = 0; row < kRowsPerThread; ++row) {
      const int head = head0 + row;
      Accum tile_max = storage.warp_reduce[0][head];
      CUTE_UNROLL
      for (int warp_idx = 1; warp_idx < 8; ++warp_idx) {
        tile_max = fmaxf(tile_max, storage.warp_reduce[warp_idx][head]);
      }
      const Accum next_max = fmaxf(row_max[row], tile_max);
      alpha[row] = row_max[row] == -INFINITY
          ? 0.0f
          : exp2f((row_max[row] - next_max) * kSoftmaxScaleLog2);
      row_max[row] = next_max;
      row_sum[row] *= alpha[row];
      const Accum max_scaled = next_max * kSoftmaxScaleLog2;
      scores[row] = exp2f(scores[row] * kSoftmaxScaleLog2 - max_scaled);
      scores[row + 2] =
          exp2f(scores[row + 2] * kSoftmaxScaleLog2 - max_scaled);
      row_sum[row] += scores[row] + scores[row + 2];
    }
    Accum* output_values = tOrO.data();
    CUTE_UNROLL
    for (int value = 0; value < 4; ++value) {
      output_values[value] *= alpha[value & 1];
    }

    sPOrigin(token0, head0) = Element(scores[0]);
    sPOrigin(token0, head1) = Element(scores[1]);
    sPOrigin(token1, head0) = Element(scores[2]);
    sPOrigin(token1, head1) = Element(scores[3]);
    cute::cp_async_wait<0>();
    __syncthreads();

    S2RAtomPVA s2r_atom_pv_a;
    auto s2r_pv_a = make_tiled_copy_A(s2r_atom_pv_a, tiled_mma_pv);
    auto s2r_pv_b = make_tiled_copy_B(s2r_atom_b, tiled_mma_pv);
    auto thr_pv_a = s2r_pv_a.get_thread_slice(threadIdx.x);
    auto thr_pv_b = s2r_pv_b.get_thread_slice(threadIdx.x);
    Tensor tXrV = thr_pv_a.retile_D(
        thr_mma_pv.partition_fragment_A(sVt));
    Tensor tXrP = thr_pv_b.retile_D(
        thr_mma_pv.partition_fragment_B(sP));
    Tensor tXsV = thr_pv_a.partition_S(sVt);
    Tensor tXsP = thr_pv_b.partition_S(sP);
    constexpr int kPBlocks = kBlockM128 / 16;
    CUTE_UNROLL
    for (int k_block = 0; k_block < kPBlocks; ++k_block) {
      copy(s2r_atom_pv_a, tXsV(_, _, k_block), tXrV(_, _, k_block));
      copy(s2r_atom_b, tXsP(_, _, k_block), tXrP(_, _, k_block));
      gemm(tiled_mma_pv, tXrV(_, _, k_block), tXrP(_, _, k_block), tOrO);
    }

    if (next_tile < tile_end) {
      read_pipe = write_pipe;
    }
  }

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int lane_in_group = lane >> 2;
  const int head_pair = lane & 3;
  const int head0 = head_pair * 2;
  const int head1 = head0 + 1;
  CUTE_UNROLL
  for (int row = 0; row < kRowsPerThread; ++row) {
    const int head = head0 + row;
    const Accum total = streamattn_group_sum(row_sum[row]);
    if (lane_in_group == 0) {
      storage.warp_reduce[warp][head] = total;
    }
  }
  __syncthreads();
  if (warp == 0 && lane_in_group == 0) {
    CUTE_UNROLL
    for (int row = 0; row < kRowsPerThread; ++row) {
      const int head = head0 + row;
      Accum total = 0.0f;
      CUTE_UNROLL
      for (int warp_idx = 0; warp_idx < 8; ++warp_idx) {
        total += storage.warp_reduce[warp_idx][head];
      }
      storage.row_sum[head] = total;
      partial_lse[static_cast<int64_t>(work) * kBlockN + head] =
          row_max[row] * kSoftmaxScaleLog2 + log2f(total);
    }
  }
  __syncthreads();

  Accum* output_values = tOrO.data();
  const int base_dim = warp * 16 + lane_in_group;
  partial_o[((static_cast<int64_t>(work) * kBlockN + head0) * kHeadDim) +
            base_dim] = output_values[0] / storage.row_sum[head0];
  partial_o[((static_cast<int64_t>(work) * kBlockN + head1) * kHeadDim) +
            base_dim] = output_values[1] / storage.row_sum[head1];
  partial_o[((static_cast<int64_t>(work) * kBlockN + head0) * kHeadDim) +
            base_dim + 8] = output_values[2] / storage.row_sum[head0];
  partial_o[((static_cast<int64_t>(work) * kBlockN + head1) * kHeadDim) +
            base_dim + 8] = output_values[3] / storage.row_sum[head1];
}

__global__ __launch_bounds__(32)
void streamattn_sm80_exact_merge_warp_kernel(
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
  const int lane = threadIdx.x;
  __shared__ Accum weights[512];

  Accum row_max = -INFINITY;
  for (int split = lane; split < num_splits; split += 32) {
    row_max = fmaxf(
        row_max,
        partial_lse[(static_cast<int64_t>(group) * num_splits + split) *
                        kBlockN +
                    head]);
  }
  row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffffu, row_max, 16));
  row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffffu, row_max, 8));
  row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffffu, row_max, 4));
  row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffffu, row_max, 2));
  row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffffu, row_max, 1));

  Accum normalizer = 0.0f;
  for (int split = lane; split < num_splits; split += 32) {
    const Accum weight = exp2f(
        partial_lse[(static_cast<int64_t>(group) * num_splits + split) *
                        kBlockN +
                    head] -
        row_max);
    weights[split] = weight;
    normalizer += weight;
  }
  normalizer += __shfl_xor_sync(0xffffffffu, normalizer, 16);
  normalizer += __shfl_xor_sync(0xffffffffu, normalizer, 8);
  normalizer += __shfl_xor_sync(0xffffffffu, normalizer, 4);
  normalizer += __shfl_xor_sync(0xffffffffu, normalizer, 2);
  normalizer += __shfl_xor_sync(0xffffffffu, normalizer, 1);
  __syncwarp();

  const Accum inverse_normalizer = 1.0f / normalizer;
  for (int dim0 = lane * 2; dim0 < kHeadDim; dim0 += 64) {
    Accum value0 = 0.0f;
    Accum value1 = 0.0f;
    for (int split = 0; split < num_splits; ++split) {
      const int64_t offset =
          ((static_cast<int64_t>(group) * num_splits + split) * kBlockN +
           head) *
              kHeadDim +
          dim0;
      value0 += weights[split] * partial_o[offset];
      value1 += weights[split] * partial_o[offset + 1];
    }
    output[static_cast<int64_t>(row) * kHeadDim + dim0] =
        Element(value0 * inverse_normalizer);
    output[static_cast<int64_t>(row) * kHeadDim + dim0 + 1] =
        Element(value1 * inverse_normalizer);
  }
}

template <int kMergeSegments>
__global__ __launch_bounds__(32)
void streamattn_sm80_exact_merge_segmented_kernel(
    const Accum* __restrict__ partial_o,
    const Accum* __restrict__ partial_lse,
    Element* __restrict__ output,
    int groups,
    int num_splits) {
  static_assert(kHeadDim % kMergeSegments == 0);
  const int row_segment = blockIdx.x;
  const int row = row_segment / kMergeSegments;
  const int segment = row_segment - row * kMergeSegments;
  if (row >= groups * kBlockN) {
    return;
  }
  const int group = row / kBlockN;
  const int head = row - group * kBlockN;
  const int lane = threadIdx.x;
  __shared__ Accum weights[512];

  Accum row_max = -INFINITY;
  for (int split = lane; split < num_splits; split += 32) {
    row_max = fmaxf(
        row_max,
        partial_lse[(static_cast<int64_t>(group) * num_splits + split) *
                        kBlockN +
                    head]);
  }
  row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffffu, row_max, 16));
  row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffffu, row_max, 8));
  row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffffu, row_max, 4));
  row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffffu, row_max, 2));
  row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffffu, row_max, 1));

  Accum normalizer = 0.0f;
  for (int split = lane; split < num_splits; split += 32) {
    const Accum weight = exp2f(
        partial_lse[(static_cast<int64_t>(group) * num_splits + split) *
                        kBlockN +
                    head] -
        row_max);
    weights[split] = weight;
    normalizer += weight;
  }
  normalizer += __shfl_xor_sync(0xffffffffu, normalizer, 16);
  normalizer += __shfl_xor_sync(0xffffffffu, normalizer, 8);
  normalizer += __shfl_xor_sync(0xffffffffu, normalizer, 4);
  normalizer += __shfl_xor_sync(0xffffffffu, normalizer, 2);
  normalizer += __shfl_xor_sync(0xffffffffu, normalizer, 1);
  __syncwarp();

  constexpr int kDimsPerSegment = kHeadDim / kMergeSegments;
  const int dim_begin = segment * kDimsPerSegment;
  const int dim_end = dim_begin + kDimsPerSegment;
  const Accum inverse_normalizer = 1.0f / normalizer;
  for (int dim = dim_begin + lane; dim < dim_end; dim += 32) {
    Accum value = 0.0f;
    for (int split = 0; split < num_splits; ++split) {
      const int64_t offset =
          ((static_cast<int64_t>(group) * num_splits + split) * kBlockN +
           head) *
              kHeadDim +
          dim;
      value += weights[split] * partial_o[offset];
    }
    output[static_cast<int64_t>(row) * kHeadDim + dim] =
        Element(value * inverse_normalizer);
  }
}

void streamattn_sm80_paged_gqa_exact_decode_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor page_table,
    torch::Tensor sequence_lengths,
    torch::Tensor partial_o,
    torch::Tensor partial_lse,
    torch::Tensor output,
    int64_t num_splits,
    bool hnd_layout,
    int64_t merge_segments,
    int64_t producer_tile) {
  TORCH_CHECK(q_group.is_cuda() && k_pages.is_cuda() && v_pages.is_cuda() &&
                  page_table.is_cuda() && sequence_lengths.is_cuda() &&
                  partial_o.is_cuda() && partial_lse.is_cuda() &&
                  output.is_cuda(),
              "all tensors must be CUDA tensors");
  TORCH_CHECK(q_group.is_contiguous() && k_pages.is_contiguous() &&
                  v_pages.is_contiguous() && page_table.is_contiguous() &&
                  sequence_lengths.is_contiguous() && partial_o.is_contiguous() &&
                  partial_lse.is_contiguous() && output.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(q_group.scalar_type() == at::ScalarType::BFloat16 &&
                  k_pages.scalar_type() == at::ScalarType::BFloat16 &&
                  v_pages.scalar_type() == at::ScalarType::BFloat16 &&
                  output.scalar_type() == at::ScalarType::BFloat16,
              "Q/K/V/output must be bf16");
  TORCH_CHECK(page_table.scalar_type() == at::ScalarType::Int &&
                  sequence_lengths.scalar_type() == at::ScalarType::Int,
              "page metadata must be int32");
  TORCH_CHECK(partial_o.scalar_type() == at::ScalarType::Float &&
                  partial_lse.scalar_type() == at::ScalarType::Float,
              "partial outputs must be fp32");
  TORCH_CHECK(q_group.dim() == 4 && q_group.size(2) == kBlockN &&
                  q_group.size(3) == kHeadDim,
              "q_group must have shape [B,Hkv,8,128]");
  TORCH_CHECK(k_pages.sizes() == v_pages.sizes() && k_pages.dim() == 4 &&
                  k_pages.size(3) == kHeadDim,
              "K/V pages must be rank-4 D128 tensors with matching shapes");
  if (hnd_layout) {
    TORCH_CHECK(k_pages.size(1) == q_group.size(1) &&
                    k_pages.size(2) == 16,
                "HND pages must have shape [pages,Hkv,16,128]");
  } else {
    TORCH_CHECK(k_pages.size(1) == 16 &&
                    k_pages.size(2) == q_group.size(1),
                "NHD pages must have shape [pages,16,Hkv,128]");
  }
  TORCH_CHECK(page_table.dim() == 2 &&
                  page_table.size(0) == q_group.size(0),
              "page_table must have shape [B,max_pages]");
  TORCH_CHECK(sequence_lengths.dim() == 1 &&
                  sequence_lengths.size(0) == q_group.size(0),
              "sequence_lengths must have shape [B]");

  const int batch = static_cast<int>(q_group.size(0));
  const int kv_heads = static_cast<int>(q_group.size(1));
  const int groups = batch * kv_heads;
  const int max_pages = static_cast<int>(page_table.size(1));
  TORCH_CHECK(producer_tile == 64 || producer_tile == 128,
              "producer_tile must be 64 or 128");
  const int pages_per_tile = static_cast<int>(producer_tile) / 16;
  const int logical_tiles =
      (max_pages + pages_per_tile - 1) / pages_per_tile;
  TORCH_CHECK(num_splits > 0 && num_splits <= logical_tiles &&
                  num_splits <= 512,
              "num_splits must be in "
              "[1,min(ceil(max_pages/pages_per_tile),512)]");
  TORCH_CHECK(merge_segments == 1 || merge_segments == 2 ||
                  merge_segments == 4 || merge_segments == 8,
              "merge_segments must be 1, 2, 4, or 8");
  TORCH_CHECK(partial_o.sizes() == torch::IntArrayRef(
                  {groups, num_splits, kBlockN, kHeadDim}),
              "partial_o shape mismatch");
  TORCH_CHECK(partial_lse.sizes() ==
                  torch::IntArrayRef({groups, num_splits, kBlockN}),
              "partial_lse shape mismatch");
  TORCH_CHECK(output.sizes() ==
                  torch::IntArrayRef({groups, kBlockN, kHeadDim}),
              "output shape mismatch");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (producer_tile == 64) {
    auto partial_kernel = hnd_layout
        ? streamattn_sm80_paged_gqa_partial_kernel<true>
        : streamattn_sm80_paged_gqa_partial_kernel<false>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        partial_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        sizeof(SharedStorage)));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        partial_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    partial_kernel<<<groups * static_cast<int>(num_splits), kThreads,
                     sizeof(SharedStorage), stream>>>(
        reinterpret_cast<const Element*>(q_group.data_ptr<at::BFloat16>()),
        reinterpret_cast<const Element*>(k_pages.data_ptr<at::BFloat16>()),
        reinterpret_cast<const Element*>(v_pages.data_ptr<at::BFloat16>()),
        page_table.data_ptr<int>(), sequence_lengths.data_ptr<int>(),
        partial_o.data_ptr<float>(), partial_lse.data_ptr<float>(), groups,
        kv_heads, max_pages, static_cast<int>(num_splits));
  } else {
    auto partial_kernel = hnd_layout
        ? streamattn_sm80_paged_gqa_partial_128_kernel<true>
        : streamattn_sm80_paged_gqa_partial_128_kernel<false>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        partial_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        sizeof(SharedStorage128)));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        partial_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    partial_kernel<<<groups * static_cast<int>(num_splits), kThreads128,
                     sizeof(SharedStorage128), stream>>>(
        reinterpret_cast<const Element*>(q_group.data_ptr<at::BFloat16>()),
        reinterpret_cast<const Element*>(k_pages.data_ptr<at::BFloat16>()),
        reinterpret_cast<const Element*>(v_pages.data_ptr<at::BFloat16>()),
        page_table.data_ptr<int>(), sequence_lengths.data_ptr<int>(),
        partial_o.data_ptr<float>(), partial_lse.data_ptr<float>(), groups,
        kv_heads, max_pages, static_cast<int>(num_splits));
  }
  if (merge_segments == 1) {
    streamattn_sm80_exact_merge_warp_kernel<<<groups * kBlockN, 32, 0, stream>>>(
        partial_o.data_ptr<float>(), partial_lse.data_ptr<float>(),
        reinterpret_cast<Element*>(output.data_ptr<at::BFloat16>()), groups,
        static_cast<int>(num_splits));
  } else if (merge_segments == 2) {
    streamattn_sm80_exact_merge_segmented_kernel<2>
        <<<groups * kBlockN * 2, 32, 0, stream>>>(
            partial_o.data_ptr<float>(), partial_lse.data_ptr<float>(),
            reinterpret_cast<Element*>(output.data_ptr<at::BFloat16>()), groups,
            static_cast<int>(num_splits));
  } else if (merge_segments == 4) {
    streamattn_sm80_exact_merge_segmented_kernel<4>
        <<<groups * kBlockN * 4, 32, 0, stream>>>(
            partial_o.data_ptr<float>(), partial_lse.data_ptr<float>(),
            reinterpret_cast<Element*>(output.data_ptr<at::BFloat16>()), groups,
            static_cast<int>(num_splits));
  } else {
    streamattn_sm80_exact_merge_segmented_kernel<8>
        <<<groups * kBlockN * 8, 32, 0, stream>>>(
            partial_o.data_ptr<float>(), partial_lse.data_ptr<float>(),
            reinterpret_cast<Element*>(output.data_ptr<at::BFloat16>()), groups,
            static_cast<int>(num_splits));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""
