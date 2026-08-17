"""CUDA extension sources for the Hopper transposed-GQA exact decode backend."""

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

void streamattn_transposed_wgmma_exact_merge_warp_out_cuda(
    torch::Tensor partial_o,
    torch::Tensor partial_lse,
    torch::Tensor output);

void streamattn_transposed_wgmma_exact_decode_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor partial_o,
    torch::Tensor partial_lse,
    torch::Tensor output,
    int64_t num_splits);

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
  m.def("exact_merge_warp_out", &streamattn_transposed_wgmma_exact_merge_warp_out_cuda,
        "StreamAttn one-warp exact split-state merge");
  m.def("exact_decode_out", &streamattn_transposed_wgmma_exact_decode_out_cuda,
        "StreamAttn exact producer and merge (single host dispatch)");
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
          row_max[row] * kSoftmaxScaleLog2 + log2f(total);
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
      const Accum weight = exp2f(
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

__global__ __launch_bounds__(32)
void streamattn_transposed_wgmma_exact_merge_warp_kernel(
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
                    kBlockN + head]);
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
                    kBlockN + head] - row_max);
    weights[split] = weight;
    normalizer += weight;
  }
  normalizer += __shfl_xor_sync(0xffffffffu, normalizer, 16);
  normalizer += __shfl_xor_sync(0xffffffffu, normalizer, 8);
  normalizer += __shfl_xor_sync(0xffffffffu, normalizer, 4);
  normalizer += __shfl_xor_sync(0xffffffffu, normalizer, 2);
  normalizer += __shfl_xor_sync(0xffffffffu, normalizer, 1);
  __syncwarp();

  Accum value0 = 0.0f;
  Accum value1 = 0.0f;
  const int dim0 = lane * 2;
  for (int split = 0; split < num_splits; ++split) {
    const int64_t base =
        ((static_cast<int64_t>(group) * num_splits + split) * kBlockN +
         head) * kHeadDim + dim0;
    const float2 pair = *reinterpret_cast<const float2*>(partial_o + base);
    const Accum weight = weights[split];
    value0 += weight * pair.x;
    value1 += weight * pair.y;
  }
  const Accum inverse_normalizer = 1.0f / normalizer;
  const int64_t output_base = static_cast<int64_t>(row) * kHeadDim + dim0;
  output[output_base] = Element(value0 * inverse_normalizer);
  output[output_base + 1] = Element(value1 * inverse_normalizer);
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

void streamattn_transposed_wgmma_exact_merge_warp_out_cuda(
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
  const dim3 block(32);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  streamattn_transposed_wgmma_exact_merge_warp_kernel<<<grid, block, 0, stream>>>(
      partial_o.data_ptr<float>(),
      partial_lse.data_ptr<float>(),
      reinterpret_cast<Element*>(output.data_ptr<at::BFloat16>()),
      groups,
      num_splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void streamattn_transposed_wgmma_exact_decode_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor partial_o,
    torch::Tensor partial_lse,
    torch::Tensor output,
    int64_t num_splits) {
  TORCH_CHECK(q_group.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda() &&
              partial_o.is_cuda() && partial_lse.is_cuda() && output.is_cuda(),
              "all tensors must be CUDA tensors");
  TORCH_CHECK(q_group.is_contiguous() && k_cache.is_contiguous() &&
              v_cache.is_contiguous() && partial_o.is_contiguous() &&
              partial_lse.is_contiguous() && output.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(q_group.scalar_type() == at::ScalarType::BFloat16 &&
              k_cache.scalar_type() == at::ScalarType::BFloat16 &&
              v_cache.scalar_type() == at::ScalarType::BFloat16,
              "q_group, k_cache, and v_cache must be bf16");
  TORCH_CHECK(partial_o.scalar_type() == at::ScalarType::Float &&
              partial_lse.scalar_type() == at::ScalarType::Float,
              "partial outputs must be fp32");
  TORCH_CHECK(output.scalar_type() == at::ScalarType::BFloat16,
              "output must be bf16");
  TORCH_CHECK(q_group.dim() == 4 && q_group.size(2) == kBlockN &&
              q_group.size(3) == kHeadDim,
              "q_group must have shape [B,Hkv,8,64]");
  TORCH_CHECK(k_cache.sizes() == v_cache.sizes(),
              "k_cache and v_cache must match");
  TORCH_CHECK(k_cache.dim() == 4 && k_cache.size(3) == kHeadDim,
              "K/V must have shape [B,Hkv,N,64]");
  TORCH_CHECK(k_cache.size(0) == q_group.size(0) &&
              k_cache.size(1) == q_group.size(1),
              "q_group and K/V batch/KV-head dimensions must match");
  TORCH_CHECK(k_cache.size(2) % kBlockM == 0,
              "kv_len must be divisible by 64");

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
  TORCH_CHECK(output.sizes() == torch::IntArrayRef(
                  {groups, kBlockN, kHeadDim}),
              "output must have shape [B*Hkv,8,64]");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 partial_grid(groups * static_cast<int>(num_splits));
  const dim3 partial_block(128);
  streamattn_transposed_wgmma_exact_partial_kernel<<<
      partial_grid, partial_block, 0, stream>>>(
      reinterpret_cast<const Element*>(q_group.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(k_cache.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(v_cache.data_ptr<at::BFloat16>()),
      partial_o.data_ptr<float>(),
      partial_lse.data_ptr<float>(),
      groups,
      kv_len,
      static_cast<int>(num_splits));

  const dim3 merge_grid(groups * kBlockN);
  const dim3 merge_block(kHeadDim);
  streamattn_transposed_wgmma_exact_merge_kernel<<<
      merge_grid, merge_block, 0, stream>>>(
      partial_o.data_ptr<float>(),
      partial_lse.data_ptr<float>(),
      reinterpret_cast<Element*>(output.data_ptr<at::BFloat16>()),
      groups,
      static_cast<int>(num_splits));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""
