"""CUDA sources for the SM90 grouped-prefill attention-epoch experiment."""

CPP_SOURCE = r"""
#include <torch/extension.h>

void streamattn_prefill_qk_floor_out_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor output, int64_t tiles_per_cta);
void streamattn_prefill_qk_softmax_floor_out_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor output, int64_t tiles_per_cta);
void streamattn_prefill_pv_ss_floor_out_cuda(
    torch::Tensor p, torch::Tensor v, torch::Tensor output, int64_t tiles_per_cta);
void streamattn_prefill_pv_rs_floor_out_cuda(
    torch::Tensor p, torch::Tensor v, torch::Tensor output, int64_t tiles_per_cta);
void streamattn_prefill_epoch_ss_floor_out_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor output, int64_t tiles_per_cta);
void streamattn_prefill_epoch_rs_floor_out_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor output, int64_t tiles_per_cta);
void streamattn_prefill_epoch_rs_reuse_q_floor_out_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor output, int64_t tiles_per_cta);
void streamattn_prefill_epoch_rs_tma_floor_out_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor output, int64_t tiles_per_cta);
void streamattn_prefill_epoch_rs_grouped2_serial_floor_out_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor output, int64_t tiles_per_cta);
void streamattn_prefill_epoch_rs_grouped2_tma_floor_out_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor output, int64_t tiles_per_cta);
void streamattn_prefill_epoch_rs_cluster2_independent_floor_out_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor output, int64_t tiles_per_group);
void streamattn_prefill_epoch_rs_cluster2_multicast_floor_out_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor output, int64_t tiles_per_group);
torch::Tensor streamattn_prefill_epoch_floor_resource_info_cuda();
torch::Tensor streamattn_prefill_epoch_rs_tma_resource_info_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v);
torch::Tensor streamattn_prefill_epoch_rs_grouped2_tma_resource_info_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v);
torch::Tensor streamattn_prefill_epoch_rs_cluster2_resource_info_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qk_out", &streamattn_prefill_qk_floor_out_cuda);
  m.def("qk_softmax_out", &streamattn_prefill_qk_softmax_floor_out_cuda);
  m.def("pv_ss_out", &streamattn_prefill_pv_ss_floor_out_cuda);
  m.def("pv_rs_out", &streamattn_prefill_pv_rs_floor_out_cuda);
  m.def("epoch_ss_out", &streamattn_prefill_epoch_ss_floor_out_cuda);
  m.def("epoch_rs_out", &streamattn_prefill_epoch_rs_floor_out_cuda);
  m.def("epoch_rs_reuse_q_out", &streamattn_prefill_epoch_rs_reuse_q_floor_out_cuda);
  m.def("epoch_rs_tma_out", &streamattn_prefill_epoch_rs_tma_floor_out_cuda);
  m.def("epoch_rs_grouped2_serial_out", &streamattn_prefill_epoch_rs_grouped2_serial_floor_out_cuda);
  m.def("epoch_rs_grouped2_tma_out", &streamattn_prefill_epoch_rs_grouped2_tma_floor_out_cuda);
  m.def("epoch_rs_cluster2_independent_out", &streamattn_prefill_epoch_rs_cluster2_independent_floor_out_cuda);
  m.def("epoch_rs_cluster2_multicast_out", &streamattn_prefill_epoch_rs_cluster2_multicast_floor_out_cuda);
  m.def("resource_info", &streamattn_prefill_epoch_floor_resource_info_cuda);
  m.def("epoch_rs_tma_resource_info", &streamattn_prefill_epoch_rs_tma_resource_info_cuda);
  m.def("epoch_rs_grouped2_tma_resource_info", &streamattn_prefill_epoch_rs_grouped2_tma_resource_info_cuda);
  m.def("epoch_rs_cluster2_resource_info", &streamattn_prefill_epoch_rs_cluster2_resource_info_cuda);
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
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cutlass/pipeline/pipeline.hpp>

using namespace cute;

using Element = cutlass::bfloat16_t;
using Accum = float;

static constexpr int kThreads = 128;
static constexpr int kTmaThreads = 256;
static constexpr int kProducerThreads = 128;
static constexpr int kConsumerThreads = 128;
static constexpr int kGroupedTmaThreads = 384;
static constexpr int kGroupedConsumerThreads = 256;
static constexpr int kConsumerGroups = 2;
static constexpr int kKStages = 2;
static constexpr int kVStages = 2;
static constexpr int kM = 64;
static constexpr int kN = 64;
static constexpr int kD = 128;
static constexpr int kPVN = 128;
static constexpr Accum kSoftmaxScaleLog2 = 0.12751743082459868f;
using ClusterShape2 = Shape<_2, _1, _1>;

using TileShapeQK = Shape<Int<kM>, Int<kN>, Int<kD>>;
using TileShapePV = Shape<Int<kM>, Int<kPVN>, Int<kN>>;
using TiledMmaQK = decltype(make_tiled_mma(
    GMMA::ss_op_selector<Element, Element, Accum, TileShapeQK>()));
using TiledMmaPVSS = decltype(make_tiled_mma(
    GMMA::ss_op_selector<
        Element, Element, Accum, TileShapePV,
        GMMA::Major::K, GMMA::Major::MN>()));
using TiledMmaPVRS = decltype(make_tiled_mma(
    GMMA::rs_op_selector<
        Element, Element, Accum, TileShapePV,
        GMMA::Major::K, GMMA::Major::MN>()));

using SmemLayoutQ = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<Element>{}, Shape<Int<kM>, Int<kD>>{}));
using SmemLayoutK = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<Element>{}, Shape<Int<kN>, Int<kD>>{}));
using SmemLayoutV = decltype(tile_to_shape(
    GMMA::Layout_MN_SW128_Atom<Element>{},
    Shape<Int<kPVN>, Int<kN>>{}, Step<_2, _1>{}));
using SmemLayoutP = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<Element>{}, Shape<Int<kM>, Int<kN>>{}));
using SmemLayoutKStages = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<Element>{},
    Shape<Int<kN>, Int<kD>, Int<kKStages>>{}));
using SmemLayoutVStages = decltype(tile_to_shape(
    GMMA::Layout_MN_SW128_Atom<Element>{},
    Shape<Int<kPVN>, Int<kN>, Int<kVStages>>{}, Step<_2, _1, _3>{}));
using SmemCopyAtomP = Copy_Atom<SM90_U32x4_STSM_N, Element>;
using GmemCopy = decltype(make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>{},
    Layout<Shape<_16, _8>, Stride<_8, _1>>{},
    Layout<Shape<_1, _8>>{}));
using GmemCopyV = decltype(make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>{},
    Layout<Shape<_8, _16>, Stride<_1, _8>>{},
    Layout<Shape<_8, _1>>{}));

struct alignas(128) QKSharedStorage {
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> q;
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> k;
  Accum reduction[kThreads];
};

template <bool RegisterPV>
struct alignas(128) PVSharedStorage {
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> v;
  cute::array_aligned<
      Element, RegisterPV ? 1 : cute::cosize_v<SmemLayoutP>> p;
  Accum reduction[kThreads];
};

template <bool RegisterPV>
struct alignas(128) EpochSharedStorage {
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> q;
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> k;
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> v;
  cute::array_aligned<
      Element, RegisterPV ? 1 : cute::cosize_v<SmemLayoutP>> p;
  Accum reduction[kThreads];
};

struct alignas(128) TmaEpochSharedStorage {
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> q;
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutKStages>> k;
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutVStages>> v;
  typename cutlass::PipelineTmaAsync<kKStages>::SharedStorage pipeline_k;
  typename cutlass::PipelineTmaAsync<kVStages>::SharedStorage pipeline_v;
  Accum reduction[kConsumerThreads];
};

struct alignas(128) GroupedTmaEpochSharedStorage {
  cute::array_aligned<
      Element, kConsumerGroups * cute::cosize_v<SmemLayoutQ>> q;
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutKStages>> k;
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutVStages>> v;
  typename cutlass::PipelineTmaAsync<kKStages>::SharedStorage pipeline_k;
  typename cutlass::PipelineTmaAsync<kVStages>::SharedStorage pipeline_v;
  Accum reduction[kGroupedConsumerThreads];
};

template <typename To, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(
    Tensor<Engine, Layout> const& tensor) {
  using From = typename Engine::value_type;
  constexpr int kElements = decltype(size(tensor))::value;
  cutlass::NumericArrayConverter<To, From, kElements> converter;
  auto fragment = converter(
      *reinterpret_cast<const cutlass::Array<From, kElements>*>(tensor.data()));
  return make_tensor(make_rmem_ptr<To>(&fragment), tensor.layout());
}

template <typename MmaTraits, typename Layout0>
__forceinline__ __device__ auto convert_layout_acc_aregs(Layout0 layout) {
  static_assert(decltype(rank<0>(layout))::value == 3);
  static_assert(decltype(size<0, 0>(layout))::value == 2);
  static_assert(decltype(size<0, 1>(layout))::value == 2);
  static_assert(sizeof(typename MmaTraits::ValTypeA) == 2);
  auto divided = logical_divide(get<0, 2>(layout), Tile<_2>{});
  return make_layout(
      make_layout(
          get<0, 0>(layout), get<0, 1>(layout), get<0, 0>(divided)),
      get<1>(layout),
      coalesce(make_layout(get<0, 1>(divided), get<2>(layout))));
}

template <typename Layout0>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout0 layout) {
  static_assert(decltype(rank<0>(layout))::value == 3);
  static_assert(decltype(size<0, 0>(layout))::value == 2);
  static_assert(decltype(size<0, 1>(layout))::value == 2);
  return make_layout(
      make_layout(get<0, 1>(layout), get<1>(layout)),
      make_layout(get<0, 0>(layout), get<0, 2>(layout), get<2>(layout)));
}

__forceinline__ __device__ Accum quad_max(Accum value) {
  value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 2));
  value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 1));
  return value;
}

__forceinline__ __device__ Accum quad_sum(Accum value) {
  value += __shfl_xor_sync(0xffffffffu, value, 2);
  value += __shfl_xor_sync(0xffffffffu, value, 1);
  return value;
}

template <typename Scores>
__forceinline__ __device__ auto softmax_in_place(Scores& scores_fragment) {
  Tensor scores = make_tensor(
      scores_fragment.data(), convert_layout_acc_rowcol(scores_fragment.layout()));
  constexpr int kRowsPerThread = decltype(size<0>(scores))::value;
  Tensor inverse_sum = make_tensor<Accum>(Shape<Int<kRowsPerThread>>{});
  CUTE_UNROLL
  for (int row = 0; row < kRowsPerThread; ++row) {
    Accum row_max = -INFINITY;
    CUTE_UNROLL
    for (int col = 0; col < size<1>(scores); ++col) {
      row_max = fmaxf(row_max, scores(row, col));
    }
    row_max = quad_max(row_max);
    Accum row_sum = 0.0f;
    CUTE_UNROLL
    for (int col = 0; col < size<1>(scores); ++col) {
      const Accum probability =
          exp2f(scores(row, col) * kSoftmaxScaleLog2
                - row_max * kSoftmaxScaleLog2);
      scores(row, col) = probability;
      row_sum += probability;
    }
    row_sum = quad_sum(row_sum);
    inverse_sum(row) = 1.0f / row_sum;
  }
  return inverse_sum;
}

template <typename Output, typename Scales>
__forceinline__ __device__ void scale_output(Output& output, Scales const& scales) {
  Tensor rows = make_tensor(
      output.data(), convert_layout_acc_rowcol(output.layout()));
  static_assert(decltype(size<0>(rows))::value == decltype(size(scales))::value);
  CUTE_UNROLL
  for (int row = 0; row < size<0>(rows); ++row) {
    CUTE_UNROLL
    for (int col = 0; col < size<1>(rows); ++col) {
      rows(row, col) *= scales(row);
    }
  }
}

template <typename Fragment>
__forceinline__ __device__ Accum fragment_sum(Fragment const& fragment) {
  Accum value = 0.0f;
  CUTE_UNROLL
  for (int idx = 0; idx < size(fragment); ++idx) {
    value += static_cast<Accum>(fragment(idx));
  }
  return value;
}

__forceinline__ __device__ void reduce_output(
    Accum local, Accum* reduction, Accum* output) {
  reduction[threadIdx.x] = local;
  __syncthreads();
  if (threadIdx.x == 0) {
    Accum total = 0.0f;
    CUTE_UNROLL
    for (int idx = 0; idx < kThreads; ++idx) {
      total += reduction[idx];
    }
    output[blockIdx.x] = total;
  }
}

template <bool ApplySoftmax>
__global__ __launch_bounds__(kThreads)
void qk_floor_kernel(
    const Element* __restrict__ q,
    const Element* __restrict__ k,
    Accum* __restrict__ output,
    int total_tiles,
    int tiles_per_cta) {
  __shared__ QKSharedStorage storage;
  Tensor sQ = make_tensor(make_smem_ptr(storage.q.data()), SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(storage.k.data()), SmemLayoutK{});
  GmemCopy copy;
  auto thread_copy = copy.get_thread_slice(threadIdx.x);
  TiledMmaQK tiled_mma;
  auto thread_mma = tiled_mma.get_thread_slice(threadIdx.x);
  Tensor rQ = thread_mma.partition_fragment_A(sQ);
  Tensor rK = thread_mma.partition_fragment_B(sK);
  const int begin = blockIdx.x * tiles_per_cta;
  const int end = min(total_tiles, begin + tiles_per_cta);
  Accum local = 0.0f;
  for (int tile = begin; tile < end; ++tile) {
    Tensor gQ = make_tensor(
        make_gmem_ptr(q + static_cast<int64_t>(tile) * kM * kD),
        Shape<Int<kM>, Int<kD>>{}, make_stride(Int<kD>{}, _1{}));
    Tensor gK = make_tensor(
        make_gmem_ptr(k + static_cast<int64_t>(tile) * kN * kD),
        Shape<Int<kN>, Int<kD>>{}, make_stride(Int<kD>{}, _1{}));
    cute::copy(copy, thread_copy.partition_S(gQ), thread_copy.partition_D(sQ));
    cute::copy(copy, thread_copy.partition_S(gK), thread_copy.partition_D(sK));
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();
    Tensor scores = partition_fragment_C(tiled_mma, Shape<Int<kM>, Int<kN>>{});
    clear(scores);
    warpgroup_fence_operand(scores);
    warpgroup_arrive();
    cute::gemm(tiled_mma, rQ, rK, scores);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(scores);
    if constexpr (ApplySoftmax) {
      auto inverse_sum = softmax_in_place(scores);
      CUTE_UNROLL
      for (int idx = 0; idx < size(inverse_sum); ++idx) {
        local += inverse_sum(idx);
      }
    }
    local += fragment_sum(scores);
    __syncthreads();
  }
  reduce_output(local, storage.reduction, output);
}

template <bool RegisterPV>
__global__ __launch_bounds__(kThreads)
void pv_floor_kernel(
    const Accum* __restrict__ p,
    const Element* __restrict__ v,
    Accum* __restrict__ output,
    int total_tiles,
    int tiles_per_cta) {
  __shared__ PVSharedStorage<RegisterPV> storage;
  Tensor sV = make_tensor(make_smem_ptr(storage.v.data()), SmemLayoutV{});
  Tensor sP = make_tensor(make_smem_ptr(storage.p.data()), SmemLayoutP{});
  GmemCopy copy;
  auto thread_copy = copy.get_thread_slice(threadIdx.x);
  TiledMmaQK tiled_qk;
  auto thread_qk = tiled_qk.get_thread_slice(threadIdx.x);
  auto copy_p = make_tiled_copy_C(SmemCopyAtomP{}, tiled_qk);
  auto thread_copy_p = copy_p.get_thread_slice(threadIdx.x);
  Tensor copy_destination_p = thread_copy_p.partition_D(sP);
  const int begin = blockIdx.x * tiles_per_cta;
  const int end = min(total_tiles, begin + tiles_per_cta);
  Accum local = 0.0f;
  for (int tile = begin; tile < end; ++tile) {
    const Element* tile_v = v + static_cast<int64_t>(tile) * kPVN * kN;
    for (int idx = threadIdx.x; idx < kPVN * kN; idx += kThreads) {
      const int row = idx / kN;
      const int col = idx - row * kN;
      sV(row, col) = tile_v[idx];
    }
    __syncthreads();

    Tensor gP = make_tensor(
        make_gmem_ptr(p + static_cast<int64_t>(tile) * kM * kN),
        Shape<Int<kM>, Int<kN>>{}, make_stride(Int<kN>{}, _1{}));
    Tensor scores = partition_fragment_C(tiled_qk, Shape<Int<kM>, Int<kN>>{});
    Tensor thread_gP = thread_qk.partition_C(gP);
    cute::copy(thread_gP, scores);
    Tensor p_acc = make_tensor(
        scores.data(), convert_layout_acc_aregs<TiledMmaPVRS>(scores.layout()));
    Tensor p_regs = convert_type<Element>(p_acc);

    if constexpr (RegisterPV) {
      TiledMmaPVRS tiled_pv;
      auto thread_pv = tiled_pv.get_thread_slice(threadIdx.x);
      Tensor rV = thread_pv.partition_fragment_B(sV);
      Tensor acc = partition_fragment_C(
          tiled_pv, Shape<Int<kM>, Int<kPVN>>{});
      clear(acc);
      warpgroup_fence_operand(p_regs);
      warpgroup_fence_operand(acc);
      warpgroup_arrive();
      cute::gemm(tiled_pv, p_regs, rV, acc);
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(p_regs);
      warpgroup_fence_operand(acc);
      local += fragment_sum(acc);
    } else {
      Tensor source_p = thread_copy_p.retile_S(p_regs);
      cute::copy(copy_p, source_p, copy_destination_p);
      cutlass::arch::fence_view_async_shared();
      cutlass::arch::NamedBarrier::sync(kThreads, 0);
      TiledMmaPVSS tiled_pv;
      auto thread_pv = tiled_pv.get_thread_slice(threadIdx.x);
      Tensor rP = thread_pv.partition_fragment_A(sP);
      Tensor rV = thread_pv.partition_fragment_B(sV);
      Tensor acc = partition_fragment_C(
          tiled_pv, Shape<Int<kM>, Int<kPVN>>{});
      clear(acc);
      warpgroup_fence_operand(acc);
      warpgroup_arrive();
      cute::gemm(tiled_pv, rP, rV, acc);
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(acc);
      local += fragment_sum(acc);
    }
    __syncthreads();
  }
  reduce_output(local, storage.reduction, output);
}

template <bool RegisterPV>
__global__ __launch_bounds__(kThreads)
void epoch_floor_kernel(
    const Element* __restrict__ q,
    const Element* __restrict__ k,
    const Element* __restrict__ v,
    Accum* __restrict__ output,
    int total_tiles,
    int tiles_per_cta) {
  extern __shared__ __align__(128) unsigned char shared_bytes[];
  auto& storage =
      *reinterpret_cast<EpochSharedStorage<RegisterPV>*>(shared_bytes);
  Tensor sQ = make_tensor(make_smem_ptr(storage.q.data()), SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(storage.k.data()), SmemLayoutK{});
  Tensor sV = make_tensor(make_smem_ptr(storage.v.data()), SmemLayoutV{});
  Tensor sP = make_tensor(make_smem_ptr(storage.p.data()), SmemLayoutP{});
  GmemCopy copy;
  auto thread_copy = copy.get_thread_slice(threadIdx.x);
  TiledMmaQK tiled_qk;
  auto thread_qk = tiled_qk.get_thread_slice(threadIdx.x);
  Tensor rQ = thread_qk.partition_fragment_A(sQ);
  Tensor rK = thread_qk.partition_fragment_B(sK);
  auto copy_p = make_tiled_copy_C(SmemCopyAtomP{}, tiled_qk);
  auto thread_copy_p = copy_p.get_thread_slice(threadIdx.x);
  Tensor copy_destination_p = thread_copy_p.partition_D(sP);
  const int begin = blockIdx.x * tiles_per_cta;
  const int end = min(total_tiles, begin + tiles_per_cta);
  Accum local = 0.0f;
  for (int tile = begin; tile < end; ++tile) {
    Tensor gQ = make_tensor(
        make_gmem_ptr(q + static_cast<int64_t>(tile) * kM * kD),
        Shape<Int<kM>, Int<kD>>{}, make_stride(Int<kD>{}, _1{}));
    Tensor gK = make_tensor(
        make_gmem_ptr(k + static_cast<int64_t>(tile) * kN * kD),
        Shape<Int<kN>, Int<kD>>{}, make_stride(Int<kD>{}, _1{}));
    cute::copy(copy, thread_copy.partition_S(gQ), thread_copy.partition_D(sQ));
    cute::copy(copy, thread_copy.partition_S(gK), thread_copy.partition_D(sK));
    const Element* tile_v = v + static_cast<int64_t>(tile) * kPVN * kN;
    for (int idx = threadIdx.x; idx < kPVN * kN; idx += kThreads) {
      const int row = idx / kN;
      const int col = idx - row * kN;
      sV(row, col) = tile_v[idx];
    }
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();

    Tensor scores = partition_fragment_C(tiled_qk, Shape<Int<kM>, Int<kN>>{});
    clear(scores);
    warpgroup_fence_operand(scores);
    warpgroup_arrive();
    cute::gemm(tiled_qk, rQ, rK, scores);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(scores);
    auto inverse_sum = softmax_in_place(scores);
    Tensor p_acc = make_tensor(
        scores.data(), convert_layout_acc_aregs<TiledMmaPVRS>(scores.layout()));
    Tensor p_regs = convert_type<Element>(p_acc);

    if constexpr (RegisterPV) {
      TiledMmaPVRS tiled_pv;
      auto thread_pv = tiled_pv.get_thread_slice(threadIdx.x);
      Tensor rV = thread_pv.partition_fragment_B(sV);
      Tensor acc = partition_fragment_C(
          tiled_pv, Shape<Int<kM>, Int<kPVN>>{});
      clear(acc);
      warpgroup_fence_operand(p_regs);
      warpgroup_fence_operand(acc);
      warpgroup_arrive();
      cute::gemm(tiled_pv, p_regs, rV, acc);
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(p_regs);
      warpgroup_fence_operand(acc);
      scale_output(acc, inverse_sum);
      local += fragment_sum(acc);
    } else {
      Tensor source_p = thread_copy_p.retile_S(p_regs);
      cute::copy(copy_p, source_p, copy_destination_p);
      cutlass::arch::fence_view_async_shared();
      cutlass::arch::NamedBarrier::sync(kThreads, 0);
      TiledMmaPVSS tiled_pv;
      auto thread_pv = tiled_pv.get_thread_slice(threadIdx.x);
      Tensor rP = thread_pv.partition_fragment_A(sP);
      Tensor rV = thread_pv.partition_fragment_B(sV);
      Tensor acc = partition_fragment_C(
          tiled_pv, Shape<Int<kM>, Int<kPVN>>{});
      clear(acc);
      warpgroup_fence_operand(acc);
      warpgroup_arrive();
      cute::gemm(tiled_pv, rP, rV, acc);
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(acc);
      scale_output(acc, inverse_sum);
      local += fragment_sum(acc);
    }
    __syncthreads();
  }
  reduce_output(local, storage.reduction, output);
}

__global__ __launch_bounds__(kThreads)
void epoch_rs_reuse_q_floor_kernel(
    const Element* __restrict__ q,
    const Element* __restrict__ k,
    const Element* __restrict__ v,
    Accum* __restrict__ output,
    int total_tiles,
    int tiles_per_cta) {
  extern __shared__ __align__(128) unsigned char shared_bytes[];
  auto& storage = *reinterpret_cast<EpochSharedStorage<true>*>(shared_bytes);
  Tensor sQ = make_tensor(make_smem_ptr(storage.q.data()), SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(storage.k.data()), SmemLayoutK{});
  Tensor sV = make_tensor(make_smem_ptr(storage.v.data()), SmemLayoutV{});
  GmemCopy copy;
  auto thread_copy = copy.get_thread_slice(threadIdx.x);
  GmemCopyV copy_v;
  auto thread_copy_v = copy_v.get_thread_slice(threadIdx.x);
  TiledMmaQK tiled_qk;
  auto thread_qk = tiled_qk.get_thread_slice(threadIdx.x);
  Tensor rQ = thread_qk.partition_fragment_A(sQ);
  Tensor rK = thread_qk.partition_fragment_B(sK);
  Tensor gQ = make_tensor(
      make_gmem_ptr(q + static_cast<int64_t>(blockIdx.x) * kM * kD),
      Shape<Int<kM>, Int<kD>>{}, make_stride(Int<kD>{}, _1{}));
  cute::copy(copy, thread_copy.partition_S(gQ), thread_copy.partition_D(sQ));
  cute::cp_async_fence();
  cute::cp_async_wait<0>();
  __syncthreads();

  const int begin = blockIdx.x * tiles_per_cta;
  const int end = min(total_tiles, begin + tiles_per_cta);
  Accum local = 0.0f;
  for (int tile = begin; tile < end; ++tile) {
    Tensor gK = make_tensor(
        make_gmem_ptr(k + static_cast<int64_t>(tile) * kN * kD),
        Shape<Int<kN>, Int<kD>>{}, make_stride(Int<kD>{}, _1{}));
    cute::copy(copy, thread_copy.partition_S(gK), thread_copy.partition_D(sK));
    Tensor gV = make_tensor(
        make_gmem_ptr(v + static_cast<int64_t>(tile) * kN * kPVN),
        Shape<Int<kPVN>, Int<kN>>{}, make_stride(_1{}, Int<kPVN>{}));
    cute::copy(
        copy_v, thread_copy_v.partition_S(gV), thread_copy_v.partition_D(sV));
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();

    Tensor scores = partition_fragment_C(tiled_qk, Shape<Int<kM>, Int<kN>>{});
    clear(scores);
    warpgroup_fence_operand(scores);
    warpgroup_arrive();
    cute::gemm(tiled_qk, rQ, rK, scores);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(scores);
    auto inverse_sum = softmax_in_place(scores);
    Tensor p_acc = make_tensor(
        scores.data(), convert_layout_acc_aregs<TiledMmaPVRS>(scores.layout()));
    Tensor p_regs = convert_type<Element>(p_acc);
    TiledMmaPVRS tiled_pv;
    auto thread_pv = tiled_pv.get_thread_slice(threadIdx.x);
    Tensor rV = thread_pv.partition_fragment_B(sV);
    Tensor acc = partition_fragment_C(tiled_pv, Shape<Int<kM>, Int<kPVN>>{});
    clear(acc);
    warpgroup_fence_operand(p_regs);
    warpgroup_fence_operand(acc);
    warpgroup_arrive();
    cute::gemm(tiled_pv, p_regs, rV, acc);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(p_regs);
    warpgroup_fence_operand(acc);
    scale_output(acc, inverse_sum);
    local += fragment_sum(acc);
    __syncthreads();
  }
  reduce_output(local, storage.reduction, output);
}

__global__ __launch_bounds__(kThreads)
void epoch_rs_grouped2_serial_floor_kernel(
    const Element* __restrict__ q,
    const Element* __restrict__ k,
    const Element* __restrict__ v,
    Accum* __restrict__ output,
    int total_tiles,
    int tiles_per_cta) {
  extern __shared__ __align__(128) unsigned char shared_bytes[];
  auto& storage = *reinterpret_cast<EpochSharedStorage<true>*>(shared_bytes);
  Tensor sQ = make_tensor(make_smem_ptr(storage.q.data()), SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(storage.k.data()), SmemLayoutK{});
  Tensor sV = make_tensor(make_smem_ptr(storage.v.data()), SmemLayoutV{});
  GmemCopy copy;
  auto thread_copy = copy.get_thread_slice(threadIdx.x);
  GmemCopyV copy_v;
  auto thread_copy_v = copy_v.get_thread_slice(threadIdx.x);
  TiledMmaQK tiled_qk;
  auto thread_qk = tiled_qk.get_thread_slice(threadIdx.x);
  Tensor rQ = thread_qk.partition_fragment_A(sQ);
  Tensor rK = thread_qk.partition_fragment_B(sK);
  Tensor gQ = make_tensor(
      make_gmem_ptr(q + static_cast<int64_t>(blockIdx.x) * kM * kD),
      Shape<Int<kM>, Int<kD>>{}, make_stride(Int<kD>{}, _1{}));
  cute::copy(copy, thread_copy.partition_S(gQ), thread_copy.partition_D(sQ));
  cute::cp_async_fence();
  cute::cp_async_wait<0>();
  __syncthreads();

  const int group = blockIdx.x / kConsumerGroups;
  const int begin = group * tiles_per_cta;
  const int end = min(total_tiles, begin + tiles_per_cta);
  Accum local = 0.0f;
  for (int tile = begin; tile < end; ++tile) {
    Tensor gK = make_tensor(
        make_gmem_ptr(k + static_cast<int64_t>(tile) * kN * kD),
        Shape<Int<kN>, Int<kD>>{}, make_stride(Int<kD>{}, _1{}));
    Tensor gV = make_tensor(
        make_gmem_ptr(v + static_cast<int64_t>(tile) * kN * kPVN),
        Shape<Int<kPVN>, Int<kN>>{}, make_stride(_1{}, Int<kPVN>{}));
    cute::copy(copy, thread_copy.partition_S(gK), thread_copy.partition_D(sK));
    cute::copy(
        copy_v, thread_copy_v.partition_S(gV), thread_copy_v.partition_D(sV));
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();

    Tensor scores = partition_fragment_C(tiled_qk, Shape<Int<kM>, Int<kN>>{});
    clear(scores);
    warpgroup_fence_operand(scores);
    warpgroup_arrive();
    cute::gemm(tiled_qk, rQ, rK, scores);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(scores);
    auto inverse_sum = softmax_in_place(scores);
    Tensor p_acc = make_tensor(
        scores.data(), convert_layout_acc_aregs<TiledMmaPVRS>(scores.layout()));
    Tensor p_regs = convert_type<Element>(p_acc);
    TiledMmaPVRS tiled_pv;
    auto thread_pv = tiled_pv.get_thread_slice(threadIdx.x);
    Tensor rV = thread_pv.partition_fragment_B(sV);
    Tensor acc = partition_fragment_C(tiled_pv, Shape<Int<kM>, Int<kPVN>>{});
    clear(acc);
    warpgroup_fence_operand(p_regs);
    warpgroup_fence_operand(acc);
    warpgroup_arrive();
    cute::gemm(tiled_pv, p_regs, rV, acc);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(p_regs);
    warpgroup_fence_operand(acc);
    scale_output(acc, inverse_sum);
    local += fragment_sum(acc);
    __syncthreads();
  }
  reduce_output(local, storage.reduction, output);
}

template <class TmaK, class TmaV>
__global__ __launch_bounds__(kTmaThreads, 1)
void epoch_rs_tma_floor_kernel(
    CUTLASS_GRID_CONSTANT TmaK const tma_k,
    CUTLASS_GRID_CONSTANT TmaV const tma_v,
    const Element* __restrict__ q,
    Accum* __restrict__ output,
    int k_rows,
    int v_cols,
    int total_tiles,
    int tiles_per_cta) {
  extern __shared__ __align__(128) unsigned char shared_bytes[];
  auto& storage = *reinterpret_cast<TmaEpochSharedStorage*>(shared_bytes);
  using PipelineK = cutlass::PipelineTmaAsync<kKStages>;
  using PipelineV = cutlass::PipelineTmaAsync<kVStages>;
  using StateK = typename PipelineK::PipelineState;

  const bool producer = threadIdx.x < kProducerThreads;
  typename PipelineK::Params params_k;
  params_k.transaction_bytes = kN * kD * sizeof(Element);
  params_k.role = producer ? PipelineK::ThreadCategory::Producer
                           : PipelineK::ThreadCategory::Consumer;
  params_k.is_leader = threadIdx.x == 0;
  params_k.num_consumers = kConsumerThreads;
  PipelineK pipeline_k(storage.pipeline_k, params_k, Shape<_1, _1, _1>{});
  typename PipelineV::Params params_v;
  params_v.transaction_bytes = kPVN * kN * sizeof(Element);
  params_v.role = producer ? PipelineV::ThreadCategory::Producer
                           : PipelineV::ThreadCategory::Consumer;
  params_v.is_leader = threadIdx.x == 0;
  params_v.num_consumers = kConsumerThreads;
  PipelineV pipeline_v(storage.pipeline_v, params_v, Shape<_1, _1, _1>{});
  __syncthreads();

  Tensor sQ = make_tensor(make_smem_ptr(storage.q.data()), SmemLayoutQ{});
  Tensor sKStages = make_tensor(
      make_smem_ptr(storage.k.data()), SmemLayoutKStages{});
  Tensor sVStages = make_tensor(
      make_smem_ptr(storage.v.data()), SmemLayoutVStages{});
  Tensor mK = tma_k.get_tma_tensor(make_shape(k_rows, Int<kD>{}));
  Tensor gK = local_tile(
      mK, make_shape(Int<kN>{}, Int<kD>{}), make_coord(_, 0));
  auto [tKgK, tKsK] = tma_partition(
      tma_k, _0{}, Layout<_1>{}, group_modes<0, 2>(sKStages),
      group_modes<0, 2>(gK));
  Tensor mV = tma_v.get_tma_tensor(make_shape(Int<kPVN>{}, v_cols));
  Tensor gV = local_tile(
      mV, make_shape(Int<kPVN>{}, Int<kN>{}), make_coord(0, _));
  auto [tVgV, tVsV] = tma_partition(
      tma_v, _0{}, Layout<_1>{}, group_modes<0, 2>(sVStages),
      group_modes<0, 2>(gV));

  const int begin = blockIdx.x * tiles_per_cta;
  const int end = min(total_tiles, begin + tiles_per_cta);
  if (producer) {
    cutlass::arch::warpgroup_reg_dealloc<56>();
    if (threadIdx.x == 0) {
      StateK write_k = cutlass::make_producer_start_state<PipelineK>();
      for (int tile = begin; tile < end; ++tile) {
        pipeline_k.producer_acquire(write_k);
        copy(
            tma_k.with(*pipeline_k.producer_get_barrier(write_k)),
            tKgK(_, tile), tKsK(_, write_k.index()));
        pipeline_v.producer_acquire(write_k);
        copy(
            tma_v.with(*pipeline_v.producer_get_barrier(write_k)),
            tVgV(_, tile), tVsV(_, write_k.index()));
        ++write_k;
      }
      pipeline_k.producer_tail(write_k);
      pipeline_v.producer_tail(write_k);
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<256>();
    const int consumer_idx = threadIdx.x - kProducerThreads;
    StateK read_k;
    GmemCopy q_copy;
    auto thread_q_copy = q_copy.get_thread_slice(consumer_idx);
    Tensor gQ = make_tensor(
        make_gmem_ptr(q + static_cast<int64_t>(blockIdx.x) * kM * kD),
        Shape<Int<kM>, Int<kD>>{}, make_stride(Int<kD>{}, _1{}));
    cute::copy(
        q_copy, thread_q_copy.partition_S(gQ), thread_q_copy.partition_D(sQ));
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    cutlass::arch::NamedBarrier::sync(kConsumerThreads, 2);
    TiledMmaQK tiled_qk;
    auto thread_qk = tiled_qk.get_thread_slice(consumer_idx);
    Tensor rQ = thread_qk.partition_fragment_A(sQ);
    Accum local = 0.0f;
    for (int tile = begin; tile < end; ++tile) {
      auto token_k = pipeline_k.consumer_try_wait(read_k);
      pipeline_k.consumer_wait(read_k, token_k);
      Tensor sK = make_tensor(
          make_smem_ptr(
              storage.k.data() + read_k.index() * cute::cosize_v<SmemLayoutK>),
          SmemLayoutK{});
      Tensor rK = thread_qk.partition_fragment_B(sK);
      Tensor scores = partition_fragment_C(tiled_qk, Shape<Int<kM>, Int<kN>>{});
      clear(scores);
      warpgroup_fence_operand(scores);
      warpgroup_arrive();
      cute::gemm(tiled_qk, rQ, rK, scores);
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(scores);
      pipeline_k.consumer_release(read_k);

      auto inverse_sum = softmax_in_place(scores);
      Tensor p_acc = make_tensor(
          scores.data(), convert_layout_acc_aregs<TiledMmaPVRS>(scores.layout()));
      Tensor p_regs = convert_type<Element>(p_acc);
      auto token_v = pipeline_v.consumer_try_wait(read_k);
      pipeline_v.consumer_wait(read_k, token_v);
      Tensor sV = make_tensor(
          make_smem_ptr(
              storage.v.data() + read_k.index() * cute::cosize_v<SmemLayoutV>),
          SmemLayoutV{});
      TiledMmaPVRS tiled_pv;
      auto thread_pv = tiled_pv.get_thread_slice(consumer_idx);
      Tensor rV = thread_pv.partition_fragment_B(sV);
      Tensor acc = partition_fragment_C(tiled_pv, Shape<Int<kM>, Int<kPVN>>{});
      clear(acc);
      warpgroup_fence_operand(p_regs);
      warpgroup_fence_operand(acc);
      warpgroup_arrive();
      cute::gemm(tiled_pv, p_regs, rV, acc);
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(p_regs);
      warpgroup_fence_operand(acc);
      pipeline_v.consumer_release(read_k);
      ++read_k;
      scale_output(acc, inverse_sum);
      local += fragment_sum(acc);
    }
    storage.reduction[consumer_idx] = local;
    cutlass::arch::NamedBarrier::sync(kConsumerThreads, 1);
    if (consumer_idx == 0) {
      Accum total = 0.0f;
      CUTE_UNROLL
      for (int idx = 0; idx < kConsumerThreads; ++idx) {
        total += storage.reduction[idx];
      }
      output[blockIdx.x] = total;
    }
  }
}

template <class TmaK, class TmaV>
__global__ __launch_bounds__(kGroupedTmaThreads, 1)
void epoch_rs_grouped2_tma_floor_kernel(
    CUTLASS_GRID_CONSTANT TmaK const tma_k,
    CUTLASS_GRID_CONSTANT TmaV const tma_v,
    const Element* __restrict__ q,
    Accum* __restrict__ output,
    int k_rows,
    int v_cols,
    int total_tiles,
    int tiles_per_cta) {
  extern __shared__ __align__(128) unsigned char shared_bytes[];
  auto& storage =
      *reinterpret_cast<GroupedTmaEpochSharedStorage*>(shared_bytes);
  using PipelineK = cutlass::PipelineTmaAsync<kKStages>;
  using PipelineV = cutlass::PipelineTmaAsync<kVStages>;
  using StateK = typename PipelineK::PipelineState;

  const bool producer = threadIdx.x < kProducerThreads;
  typename PipelineK::Params params_k;
  params_k.transaction_bytes = kN * kD * sizeof(Element);
  params_k.role = producer ? PipelineK::ThreadCategory::Producer
                           : PipelineK::ThreadCategory::Consumer;
  params_k.is_leader = threadIdx.x == 0;
  params_k.num_consumers = kGroupedConsumerThreads;
  PipelineK pipeline_k(storage.pipeline_k, params_k, Shape<_1, _1, _1>{});
  typename PipelineV::Params params_v;
  params_v.transaction_bytes = kPVN * kN * sizeof(Element);
  params_v.role = producer ? PipelineV::ThreadCategory::Producer
                           : PipelineV::ThreadCategory::Consumer;
  params_v.is_leader = threadIdx.x == 0;
  params_v.num_consumers = kGroupedConsumerThreads;
  PipelineV pipeline_v(storage.pipeline_v, params_v, Shape<_1, _1, _1>{});
  __syncthreads();

  Tensor sKStages = make_tensor(
      make_smem_ptr(storage.k.data()), SmemLayoutKStages{});
  Tensor sVStages = make_tensor(
      make_smem_ptr(storage.v.data()), SmemLayoutVStages{});
  Tensor mK = tma_k.get_tma_tensor(make_shape(k_rows, Int<kD>{}));
  Tensor gK = local_tile(
      mK, make_shape(Int<kN>{}, Int<kD>{}), make_coord(_, 0));
  auto [tKgK, tKsK] = tma_partition(
      tma_k, _0{}, Layout<_1>{}, group_modes<0, 2>(sKStages),
      group_modes<0, 2>(gK));
  Tensor mV = tma_v.get_tma_tensor(make_shape(Int<kPVN>{}, v_cols));
  Tensor gV = local_tile(
      mV, make_shape(Int<kPVN>{}, Int<kN>{}), make_coord(0, _));
  auto [tVgV, tVsV] = tma_partition(
      tma_v, _0{}, Layout<_1>{}, group_modes<0, 2>(sVStages),
      group_modes<0, 2>(gV));

  const int begin = blockIdx.x * tiles_per_cta;
  const int end = min(total_tiles, begin + tiles_per_cta);
  if (producer) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    if (threadIdx.x == 0) {
      StateK write_k = cutlass::make_producer_start_state<PipelineK>();
      for (int tile = begin; tile < end; ++tile) {
        pipeline_k.producer_acquire(write_k);
        copy(
            tma_k.with(*pipeline_k.producer_get_barrier(write_k)),
            tKgK(_, tile), tKsK(_, write_k.index()));
        pipeline_v.producer_acquire(write_k);
        copy(
            tma_v.with(*pipeline_v.producer_get_barrier(write_k)),
            tVgV(_, tile), tVsV(_, write_k.index()));
        ++write_k;
      }
      pipeline_k.producer_tail(write_k);
      pipeline_v.producer_tail(write_k);
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<240>();
    const int consumer_linear = threadIdx.x - kProducerThreads;
    const int consumer_group = consumer_linear / kConsumerThreads;
    const int consumer_idx = consumer_linear % kConsumerThreads;
    Tensor sQ = make_tensor(
        make_smem_ptr(
            storage.q.data()
            + consumer_group * cute::cosize_v<SmemLayoutQ>),
        SmemLayoutQ{});
    GmemCopy q_copy;
    auto thread_q_copy = q_copy.get_thread_slice(consumer_idx);
    Tensor gQ = make_tensor(
        make_gmem_ptr(
            q + static_cast<int64_t>(blockIdx.x * kConsumerGroups
                                      + consumer_group) * kM * kD),
        Shape<Int<kM>, Int<kD>>{}, make_stride(Int<kD>{}, _1{}));
    cute::copy(
        q_copy, thread_q_copy.partition_S(gQ), thread_q_copy.partition_D(sQ));
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    cutlass::arch::NamedBarrier::sync(
        kConsumerThreads, 2 + consumer_group);

    StateK read_k;
    TiledMmaQK tiled_qk;
    auto thread_qk = tiled_qk.get_thread_slice(consumer_idx);
    Tensor rQ = thread_qk.partition_fragment_A(sQ);
    Accum local = 0.0f;
    for (int tile = begin; tile < end; ++tile) {
      auto token_k = pipeline_k.consumer_try_wait(read_k);
      pipeline_k.consumer_wait(read_k, token_k);
      Tensor sK = make_tensor(
          make_smem_ptr(
              storage.k.data() + read_k.index() * cute::cosize_v<SmemLayoutK>),
          SmemLayoutK{});
      Tensor rK = thread_qk.partition_fragment_B(sK);
      Tensor scores = partition_fragment_C(tiled_qk, Shape<Int<kM>, Int<kN>>{});
      clear(scores);
      warpgroup_fence_operand(scores);
      warpgroup_arrive();
      cute::gemm(tiled_qk, rQ, rK, scores);
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(scores);
      pipeline_k.consumer_release(read_k);

      auto inverse_sum = softmax_in_place(scores);
      Tensor p_acc = make_tensor(
          scores.data(), convert_layout_acc_aregs<TiledMmaPVRS>(scores.layout()));
      Tensor p_regs = convert_type<Element>(p_acc);
      auto token_v = pipeline_v.consumer_try_wait(read_k);
      pipeline_v.consumer_wait(read_k, token_v);
      Tensor sV = make_tensor(
          make_smem_ptr(
              storage.v.data() + read_k.index() * cute::cosize_v<SmemLayoutV>),
          SmemLayoutV{});
      TiledMmaPVRS tiled_pv;
      auto thread_pv = tiled_pv.get_thread_slice(consumer_idx);
      Tensor rV = thread_pv.partition_fragment_B(sV);
      Tensor acc = partition_fragment_C(tiled_pv, Shape<Int<kM>, Int<kPVN>>{});
      clear(acc);
      warpgroup_fence_operand(p_regs);
      warpgroup_fence_operand(acc);
      warpgroup_arrive();
      cute::gemm(tiled_pv, p_regs, rV, acc);
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(p_regs);
      warpgroup_fence_operand(acc);
      pipeline_v.consumer_release(read_k);
      ++read_k;
      scale_output(acc, inverse_sum);
      local += fragment_sum(acc);
    }
    const int reduction_offset = consumer_group * kConsumerThreads;
    storage.reduction[reduction_offset + consumer_idx] = local;
    cutlass::arch::NamedBarrier::sync(
        kConsumerThreads, 4 + consumer_group);
    if (consumer_idx == 0) {
      Accum total = 0.0f;
      CUTE_UNROLL
      for (int idx = 0; idx < kConsumerThreads; ++idx) {
        total += storage.reduction[reduction_offset + idx];
      }
      output[blockIdx.x * kConsumerGroups + consumer_group] = total;
    }
  }
}

template <bool Multicast, class TmaK, class TmaV>
__global__ __launch_bounds__(kTmaThreads, 1)
void epoch_rs_cluster2_floor_kernel(
    CUTLASS_GRID_CONSTANT TmaK const tma_k,
    CUTLASS_GRID_CONSTANT TmaV const tma_v,
    const Element* __restrict__ q,
    Accum* __restrict__ output,
    int k_rows,
    int v_cols,
    int total_tiles,
    int tiles_per_group) {
  extern __shared__ __align__(128) unsigned char shared_bytes[];
  auto& storage = *reinterpret_cast<TmaEpochSharedStorage*>(shared_bytes);
  using PipelineK = cutlass::PipelineTmaAsync<kKStages>;
  using PipelineV = cutlass::PipelineTmaAsync<kVStages>;
  using StateK = typename PipelineK::PipelineState;
  using ActiveClusterShape = std::conditional_t<
      Multicast, ClusterShape2, Shape<_1, _1, _1>>;

  const bool producer = threadIdx.x < kProducerThreads;
  typename PipelineK::Params params_k;
  params_k.transaction_bytes = kN * kD * sizeof(Element);
  params_k.role = producer ? PipelineK::ThreadCategory::Producer
                           : PipelineK::ThreadCategory::Consumer;
  params_k.is_leader = threadIdx.x == 0;
  params_k.num_consumers = kConsumerThreads;
  PipelineK pipeline_k(
      storage.pipeline_k, params_k, ActiveClusterShape{});
  typename PipelineV::Params params_v;
  params_v.transaction_bytes = kPVN * kN * sizeof(Element);
  params_v.role = producer ? PipelineV::ThreadCategory::Producer
                           : PipelineV::ThreadCategory::Consumer;
  params_v.is_leader = threadIdx.x == 0;
  params_v.num_consumers = kConsumerThreads;
  PipelineV pipeline_v(
      storage.pipeline_v, params_v, ActiveClusterShape{});
  __syncthreads();
  if constexpr (Multicast) {
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
  }

  Tensor sQ = make_tensor(make_smem_ptr(storage.q.data()), SmemLayoutQ{});
  Tensor sKStages = make_tensor(
      make_smem_ptr(storage.k.data()), SmemLayoutKStages{});
  Tensor sVStages = make_tensor(
      make_smem_ptr(storage.v.data()), SmemLayoutVStages{});
  Tensor mK = tma_k.get_tma_tensor(make_shape(k_rows, Int<kD>{}));
  Tensor gK = local_tile(
      mK, make_shape(Int<kN>{}, Int<kD>{}), make_coord(_, 0));
  Tensor mV = tma_v.get_tma_tensor(make_shape(Int<kPVN>{}, v_cols));
  Tensor gV = local_tile(
      mV, make_shape(Int<kPVN>{}, Int<kN>{}), make_coord(0, _));
  const uint32_t block_rank = Multicast ? cute::block_rank_in_cluster() : 0;
  auto [tKgK, tKsK] = tma_partition(
      tma_k, block_rank, Layout<ActiveClusterShape>{},
      group_modes<0, 2>(sKStages), group_modes<0, 2>(gK));
  auto [tVgV, tVsV] = tma_partition(
      tma_v, block_rank, Layout<ActiveClusterShape>{},
      group_modes<0, 2>(sVStages), group_modes<0, 2>(gV));

  const int group = blockIdx.x / 2;
  const int begin = group * tiles_per_group;
  const int end = min(total_tiles, begin + tiles_per_group);
  uint16_t multicast_mask = 0;
  if constexpr (Multicast) {
    auto block_layout = Layout<ClusterShape2>{};
    CUTE_UNROLL
    for (int block = 0; block < size<0>(block_layout); ++block) {
      multicast_mask |= uint16_t(1) << block_layout(block, _0{}, _0{});
    }
  }

  if (producer) {
    cutlass::arch::warpgroup_reg_dealloc<56>();
    if (threadIdx.x == 0) {
      StateK write_k = cutlass::make_producer_start_state<PipelineK>();
      for (int tile = begin; tile < end; ++tile) {
        pipeline_k.producer_acquire(write_k);
        if constexpr (Multicast) {
          copy(
              tma_k.with(
                  *pipeline_k.producer_get_barrier(write_k), multicast_mask),
              tKgK(_, tile), tKsK(_, write_k.index()));
        } else {
          copy(
              tma_k.with(*pipeline_k.producer_get_barrier(write_k)),
              tKgK(_, tile), tKsK(_, write_k.index()));
        }
        pipeline_v.producer_acquire(write_k);
        if constexpr (Multicast) {
          copy(
              tma_v.with(
                  *pipeline_v.producer_get_barrier(write_k), multicast_mask),
              tVgV(_, tile), tVsV(_, write_k.index()));
        } else {
          copy(
              tma_v.with(*pipeline_v.producer_get_barrier(write_k)),
              tVgV(_, tile), tVsV(_, write_k.index()));
        }
        ++write_k;
      }
      pipeline_k.producer_tail(write_k);
      pipeline_v.producer_tail(write_k);
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<256>();
    const int consumer_idx = threadIdx.x - kProducerThreads;
    StateK read_k;
    GmemCopy q_copy;
    auto thread_q_copy = q_copy.get_thread_slice(consumer_idx);
    Tensor gQ = make_tensor(
        make_gmem_ptr(q + static_cast<int64_t>(blockIdx.x) * kM * kD),
        Shape<Int<kM>, Int<kD>>{}, make_stride(Int<kD>{}, _1{}));
    cute::copy(
        q_copy, thread_q_copy.partition_S(gQ), thread_q_copy.partition_D(sQ));
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    cutlass::arch::NamedBarrier::sync(kConsumerThreads, 2);
    TiledMmaQK tiled_qk;
    auto thread_qk = tiled_qk.get_thread_slice(consumer_idx);
    Tensor rQ = thread_qk.partition_fragment_A(sQ);
    Accum local = 0.0f;
    for (int tile = begin; tile < end; ++tile) {
      auto token_k = pipeline_k.consumer_try_wait(read_k);
      pipeline_k.consumer_wait(read_k, token_k);
      Tensor sK = make_tensor(
          make_smem_ptr(
              storage.k.data() + read_k.index() * cute::cosize_v<SmemLayoutK>),
          SmemLayoutK{});
      Tensor rK = thread_qk.partition_fragment_B(sK);
      Tensor scores = partition_fragment_C(tiled_qk, Shape<Int<kM>, Int<kN>>{});
      clear(scores);
      warpgroup_fence_operand(scores);
      warpgroup_arrive();
      cute::gemm(tiled_qk, rQ, rK, scores);
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(scores);
      pipeline_k.consumer_release(read_k);

      auto inverse_sum = softmax_in_place(scores);
      Tensor p_acc = make_tensor(
          scores.data(), convert_layout_acc_aregs<TiledMmaPVRS>(scores.layout()));
      Tensor p_regs = convert_type<Element>(p_acc);
      auto token_v = pipeline_v.consumer_try_wait(read_k);
      pipeline_v.consumer_wait(read_k, token_v);
      Tensor sV = make_tensor(
          make_smem_ptr(
              storage.v.data() + read_k.index() * cute::cosize_v<SmemLayoutV>),
          SmemLayoutV{});
      TiledMmaPVRS tiled_pv;
      auto thread_pv = tiled_pv.get_thread_slice(consumer_idx);
      Tensor rV = thread_pv.partition_fragment_B(sV);
      Tensor acc = partition_fragment_C(tiled_pv, Shape<Int<kM>, Int<kPVN>>{});
      clear(acc);
      warpgroup_fence_operand(p_regs);
      warpgroup_fence_operand(acc);
      warpgroup_arrive();
      cute::gemm(tiled_pv, p_regs, rV, acc);
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(p_regs);
      warpgroup_fence_operand(acc);
      pipeline_v.consumer_release(read_k);
      ++read_k;
      scale_output(acc, inverse_sum);
      local += fragment_sum(acc);
    }
    storage.reduction[consumer_idx] = local;
    cutlass::arch::NamedBarrier::sync(kConsumerThreads, 1);
    if (consumer_idx == 0) {
      Accum total = 0.0f;
      CUTE_UNROLL
      for (int idx = 0; idx < kConsumerThreads; ++idx) {
        total += storage.reduction[idx];
      }
      output[blockIdx.x] = total;
    }
  }
}

static void check_common(torch::Tensor output, int64_t tiles, int64_t tiles_per_cta) {
  TORCH_CHECK(output.is_cuda() && output.is_contiguous(), "output must be contiguous CUDA");
  TORCH_CHECK(output.scalar_type() == at::ScalarType::Float, "output must be fp32");
  TORCH_CHECK(tiles_per_cta > 0 && tiles % tiles_per_cta == 0,
              "tiles_per_cta must divide total tiles");
  TORCH_CHECK(output.numel() == tiles / tiles_per_cta,
              "output must have one checksum per CTA");
}

static void check_bf16_tile(torch::Tensor value, int64_t rows, int64_t cols, const char* name) {
  TORCH_CHECK(value.is_cuda() && value.is_contiguous(), name, " must be contiguous CUDA");
  TORCH_CHECK(value.scalar_type() == at::ScalarType::BFloat16, name, " must be bf16");
  TORCH_CHECK(value.dim() == 3 && value.size(1) == rows && value.size(2) == cols,
              name, " has an invalid shape");
}

void streamattn_prefill_qk_floor_out_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor output, int64_t tiles_per_cta) {
  check_bf16_tile(q, kM, kD, "q");
  check_bf16_tile(k, kN, kD, "k");
  TORCH_CHECK(q.size(0) == k.size(0), "q and k tile counts must match");
  check_common(output, q.size(0), tiles_per_cta);
  dim3 grid(output.numel());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  qk_floor_kernel<false><<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const Element*>(q.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(k.data_ptr<at::BFloat16>()),
      output.data_ptr<Accum>(), q.size(0), tiles_per_cta);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void streamattn_prefill_qk_softmax_floor_out_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor output, int64_t tiles_per_cta) {
  check_bf16_tile(q, kM, kD, "q");
  check_bf16_tile(k, kN, kD, "k");
  TORCH_CHECK(q.size(0) == k.size(0), "q and k tile counts must match");
  check_common(output, q.size(0), tiles_per_cta);
  dim3 grid(output.numel());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  qk_floor_kernel<true><<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const Element*>(q.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(k.data_ptr<at::BFloat16>()),
      output.data_ptr<Accum>(), q.size(0), tiles_per_cta);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool RegisterPV>
static void launch_pv(torch::Tensor p, torch::Tensor v, torch::Tensor output, int64_t tiles_per_cta) {
  TORCH_CHECK(p.is_cuda() && p.is_contiguous() && p.scalar_type() == at::ScalarType::Float,
              "p must be contiguous CUDA fp32");
  TORCH_CHECK(p.dim() == 3 && p.size(1) == kM && p.size(2) == kN,
              "p must have shape [tiles,64,64]");
  check_bf16_tile(v, kPVN, kN, "v");
  TORCH_CHECK(p.size(0) == v.size(0), "p and v tile counts must match");
  check_common(output, p.size(0), tiles_per_cta);
  dim3 grid(output.numel());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  pv_floor_kernel<RegisterPV><<<grid, kThreads, 0, stream>>>(
      p.data_ptr<Accum>(),
      reinterpret_cast<const Element*>(v.data_ptr<at::BFloat16>()),
      output.data_ptr<Accum>(), p.size(0), tiles_per_cta);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void streamattn_prefill_pv_ss_floor_out_cuda(
    torch::Tensor p, torch::Tensor v, torch::Tensor output, int64_t tiles_per_cta) {
  launch_pv<false>(p, v, output, tiles_per_cta);
}

void streamattn_prefill_pv_rs_floor_out_cuda(
    torch::Tensor p, torch::Tensor v, torch::Tensor output, int64_t tiles_per_cta) {
  launch_pv<true>(p, v, output, tiles_per_cta);
}

template <bool RegisterPV>
static void launch_epoch(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor output, int64_t tiles_per_cta) {
  check_bf16_tile(q, kM, kD, "q");
  check_bf16_tile(k, kN, kD, "k");
  check_bf16_tile(v, kPVN, kN, "v");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0),
              "q, k, and v tile counts must match");
  check_common(output, q.size(0), tiles_per_cta);
  constexpr int shared_bytes = sizeof(EpochSharedStorage<RegisterPV>);
  dim3 grid(output.numel());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto kernel = epoch_floor_kernel<RegisterPV>;
  if (shared_bytes > 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
  }
  kernel<<<grid, kThreads, shared_bytes, stream>>>(
      reinterpret_cast<const Element*>(q.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(k.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(v.data_ptr<at::BFloat16>()),
      output.data_ptr<Accum>(), q.size(0), tiles_per_cta);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void streamattn_prefill_epoch_ss_floor_out_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor output, int64_t tiles_per_cta) {
  launch_epoch<false>(q, k, v, output, tiles_per_cta);
}

void streamattn_prefill_epoch_rs_floor_out_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor output, int64_t tiles_per_cta) {
  launch_epoch<true>(q, k, v, output, tiles_per_cta);
}

void streamattn_prefill_epoch_rs_reuse_q_floor_out_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor output, int64_t tiles_per_cta) {
  check_bf16_tile(q, kM, kD, "q");
  check_bf16_tile(k, kN, kD, "k");
  check_bf16_tile(v, kN, kPVN, "v");
  TORCH_CHECK(k.size(0) == v.size(0), "k and v tile counts must match");
  check_common(output, k.size(0), tiles_per_cta);
  TORCH_CHECK(q.size(0) == output.numel(),
              "q must contain one tile per CTA");
  constexpr int shared_bytes = sizeof(EpochSharedStorage<true>);
  auto kernel = epoch_rs_reuse_q_floor_kernel;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
  kernel<<<output.numel(), kThreads, shared_bytes,
           at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const Element*>(q.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(k.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(v.data_ptr<at::BFloat16>()),
      output.data_ptr<Accum>(), k.size(0), tiles_per_cta);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static auto make_tma_k(torch::Tensor tensor) {
  Tensor global = make_tensor(
      reinterpret_cast<Element*>(tensor.data_ptr()),
      make_shape(int(tensor.size(0) * kN), Int<kD>{}),
      make_stride(Int<kD>{}, _1{}));
  return make_tma_atom(
      SM90_TMA_LOAD{}, global, SmemLayoutK{},
      make_shape(Int<kN>{}, Int<kD>{}));
}

static auto make_tma_v(torch::Tensor tensor) {
  Tensor global = make_tensor(
      reinterpret_cast<Element*>(tensor.data_ptr()),
      make_shape(Int<kPVN>{}, int(tensor.size(0) * kN)),
      make_stride(_1{}, Int<kPVN>{}));
  return make_tma_atom(
      SM90_TMA_LOAD{}, global, SmemLayoutV{},
      make_shape(Int<kPVN>{}, Int<kN>{}));
}

static auto make_tma_k_multicast(torch::Tensor tensor) {
  Tensor global = make_tensor(
      reinterpret_cast<Element*>(tensor.data_ptr()),
      make_shape(int(tensor.size(0) * kN), Int<kD>{}),
      make_stride(Int<kD>{}, _1{}));
  return make_tma_atom(
      SM90_TMA_LOAD_MULTICAST{}, global, SmemLayoutK{},
      make_shape(Int<kN>{}, Int<kD>{}), Int<2>{});
}

static auto make_tma_v_multicast(torch::Tensor tensor) {
  Tensor global = make_tensor(
      reinterpret_cast<Element*>(tensor.data_ptr()),
      make_shape(Int<kPVN>{}, int(tensor.size(0) * kN)),
      make_stride(_1{}, Int<kPVN>{}));
  return make_tma_atom(
      SM90_TMA_LOAD_MULTICAST{}, global, SmemLayoutV{},
      make_shape(Int<kPVN>{}, Int<kN>{}), Int<2>{});
}

void streamattn_prefill_epoch_rs_tma_floor_out_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor output, int64_t tiles_per_cta) {
  check_bf16_tile(q, kM, kD, "q");
  check_bf16_tile(k, kN, kD, "k");
  check_bf16_tile(v, kN, kPVN, "v");
  TORCH_CHECK(k.size(0) == v.size(0), "k and v tile counts must match");
  check_common(output, k.size(0), tiles_per_cta);
  TORCH_CHECK(q.size(0) == output.numel(),
              "q must contain one tile per CTA");
  auto tma_k = make_tma_k(k);
  auto tma_v = make_tma_v(v);
  auto kernel = epoch_rs_tma_floor_kernel<decltype(tma_k), decltype(tma_v)>;
  constexpr int shared_bytes = sizeof(TmaEpochSharedStorage);
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
  kernel<<<output.numel(), kTmaThreads, shared_bytes,
           at::cuda::getCurrentCUDAStream()>>>(
      tma_k, tma_v,
      reinterpret_cast<const Element*>(q.data_ptr<at::BFloat16>()),
      output.data_ptr<Accum>(), k.size(0) * kN, v.size(0) * kN,
      k.size(0), tiles_per_cta);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void streamattn_prefill_epoch_rs_grouped2_serial_floor_out_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor output, int64_t tiles_per_cta) {
  check_bf16_tile(q, kM, kD, "q");
  check_bf16_tile(k, kN, kD, "k");
  check_bf16_tile(v, kN, kPVN, "v");
  TORCH_CHECK(k.size(0) == v.size(0), "k and v tile counts must match");
  TORCH_CHECK(tiles_per_cta > 0 && k.size(0) % tiles_per_cta == 0,
              "tiles_per_cta must divide total tiles");
  const int groups = k.size(0) / tiles_per_cta;
  TORCH_CHECK(q.size(0) == groups * kConsumerGroups,
              "q must contain two tiles per K/V group");
  TORCH_CHECK(output.is_cuda() && output.is_contiguous()
              && output.scalar_type() == at::ScalarType::Float,
              "output must be contiguous CUDA fp32");
  TORCH_CHECK(output.numel() == q.size(0),
              "output must contain two checksums per K/V group");
  constexpr int shared_bytes = sizeof(EpochSharedStorage<true>);
  auto kernel = epoch_rs_grouped2_serial_floor_kernel;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
  kernel<<<output.numel(), kThreads, shared_bytes,
           at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const Element*>(q.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(k.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(v.data_ptr<at::BFloat16>()),
      output.data_ptr<Accum>(), k.size(0), tiles_per_cta);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void streamattn_prefill_epoch_rs_grouped2_tma_floor_out_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor output, int64_t tiles_per_cta) {
  check_bf16_tile(q, kM, kD, "q");
  check_bf16_tile(k, kN, kD, "k");
  check_bf16_tile(v, kN, kPVN, "v");
  TORCH_CHECK(k.size(0) == v.size(0), "k and v tile counts must match");
  TORCH_CHECK(tiles_per_cta > 0 && k.size(0) % tiles_per_cta == 0,
              "tiles_per_cta must divide total tiles");
  const int groups = k.size(0) / tiles_per_cta;
  TORCH_CHECK(q.size(0) == groups * kConsumerGroups,
              "q must contain two tiles per K/V group");
  TORCH_CHECK(output.is_cuda() && output.is_contiguous()
              && output.scalar_type() == at::ScalarType::Float,
              "output must be contiguous CUDA fp32");
  TORCH_CHECK(output.numel() == q.size(0),
              "output must contain two checksums per K/V group");
  auto tma_k = make_tma_k(k);
  auto tma_v = make_tma_v(v);
  auto kernel = epoch_rs_grouped2_tma_floor_kernel<
      decltype(tma_k), decltype(tma_v)>;
  constexpr int shared_bytes = sizeof(GroupedTmaEpochSharedStorage);
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
  kernel<<<groups, kGroupedTmaThreads, shared_bytes,
           at::cuda::getCurrentCUDAStream()>>>(
      tma_k, tma_v,
      reinterpret_cast<const Element*>(q.data_ptr<at::BFloat16>()),
      output.data_ptr<Accum>(), k.size(0) * kN, v.size(0) * kN,
      k.size(0), tiles_per_cta);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void check_cluster2_epoch(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor output, int64_t tiles_per_group) {
  check_bf16_tile(q, kM, kD, "q");
  check_bf16_tile(k, kN, kD, "k");
  check_bf16_tile(v, kN, kPVN, "v");
  TORCH_CHECK(k.size(0) == v.size(0), "k and v tile counts must match");
  TORCH_CHECK(tiles_per_group > 0 && k.size(0) % tiles_per_group == 0,
              "tiles_per_group must divide total tiles");
  const int groups = k.size(0) / tiles_per_group;
  TORCH_CHECK(q.size(0) == groups * 2,
              "q must contain two tiles per K/V group");
  TORCH_CHECK(output.is_cuda() && output.is_contiguous()
              && output.scalar_type() == at::ScalarType::Float,
              "output must be contiguous CUDA fp32");
  TORCH_CHECK(output.numel() == q.size(0),
              "output must contain two checksums per K/V group");
}

void streamattn_prefill_epoch_rs_cluster2_independent_floor_out_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor output, int64_t tiles_per_group) {
  check_cluster2_epoch(q, k, v, output, tiles_per_group);
  auto tma_k = make_tma_k(k);
  auto tma_v = make_tma_v(v);
  auto kernel = epoch_rs_cluster2_floor_kernel<
      false, decltype(tma_k), decltype(tma_v)>;
  constexpr int shared_bytes = sizeof(TmaEpochSharedStorage);
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
  kernel<<<output.numel(), kTmaThreads, shared_bytes,
           at::cuda::getCurrentCUDAStream()>>>(
      tma_k, tma_v,
      reinterpret_cast<const Element*>(q.data_ptr<at::BFloat16>()),
      output.data_ptr<Accum>(), k.size(0) * kN, v.size(0) * kN,
      k.size(0), tiles_per_group);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void streamattn_prefill_epoch_rs_cluster2_multicast_floor_out_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor output, int64_t tiles_per_group) {
  check_cluster2_epoch(q, k, v, output, tiles_per_group);
  auto tma_k = make_tma_k_multicast(k);
  auto tma_v = make_tma_v_multicast(v);
  auto kernel = epoch_rs_cluster2_floor_kernel<
      true, decltype(tma_k), decltype(tma_v)>;
  constexpr int shared_bytes = sizeof(TmaEpochSharedStorage);
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
  cutlass::ClusterLaunchParams params{
      dim3(output.numel()), dim3(kTmaThreads), dim3(2, 1, 1),
      shared_bytes, at::cuda::getCurrentCUDAStream()};
  void const* kernel_ptr = reinterpret_cast<void const*>(kernel);
  auto status = cutlass::launch_kernel_on_cluster(
      params, kernel_ptr, tma_k, tma_v,
      reinterpret_cast<const Element*>(q.data_ptr<at::BFloat16>()),
      output.data_ptr<Accum>(), int(k.size(0) * kN),
      int(v.size(0) * kN), int(k.size(0)), int(tiles_per_group));
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "cluster attention epoch launch failed");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename Kernel>
static void append_kernel_info(
    std::vector<int64_t>& values, Kernel kernel, int dynamic_shared_bytes,
    int threads = kThreads) {
  if (dynamic_shared_bytes > 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        dynamic_shared_bytes));
  }
  cudaFuncAttributes attributes{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
  int blocks = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, kernel, threads, dynamic_shared_bytes));
  values.push_back(attributes.numRegs);
  values.push_back(attributes.sharedSizeBytes);
  values.push_back(dynamic_shared_bytes);
  values.push_back(attributes.localSizeBytes);
  values.push_back(blocks);
  values.push_back(attributes.maxThreadsPerBlock);
}

torch::Tensor streamattn_prefill_epoch_floor_resource_info_cuda() {
  std::vector<int64_t> values = {
      static_cast<int64_t>(sizeof(QKSharedStorage)),
      static_cast<int64_t>(sizeof(PVSharedStorage<false>)),
      static_cast<int64_t>(sizeof(PVSharedStorage<true>)),
      static_cast<int64_t>(sizeof(EpochSharedStorage<false>)),
      static_cast<int64_t>(sizeof(EpochSharedStorage<true>)),
      static_cast<int64_t>(sizeof(TmaEpochSharedStorage)),
  };
  append_kernel_info(values, qk_floor_kernel<false>, 0);
  append_kernel_info(values, qk_floor_kernel<true>, 0);
  append_kernel_info(values, pv_floor_kernel<false>, 0);
  append_kernel_info(values, pv_floor_kernel<true>, 0);
  append_kernel_info(
      values, epoch_floor_kernel<false>, sizeof(EpochSharedStorage<false>));
  append_kernel_info(
      values, epoch_floor_kernel<true>, sizeof(EpochSharedStorage<true>));
  append_kernel_info(
      values, epoch_rs_reuse_q_floor_kernel, sizeof(EpochSharedStorage<true>));
  return torch::tensor(values, torch::TensorOptions().dtype(torch::kInt64));
}

torch::Tensor streamattn_prefill_epoch_rs_tma_resource_info_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  check_bf16_tile(q, kM, kD, "q");
  check_bf16_tile(k, kN, kD, "k");
  check_bf16_tile(v, kN, kPVN, "v");
  auto tma_k = make_tma_k(k);
  auto tma_v = make_tma_v(v);
  auto kernel = epoch_rs_tma_floor_kernel<decltype(tma_k), decltype(tma_v)>;
  std::vector<int64_t> values;
  append_kernel_info(
      values, kernel, sizeof(TmaEpochSharedStorage), kTmaThreads);
  return torch::tensor(values, torch::TensorOptions().dtype(torch::kInt64));
}

torch::Tensor streamattn_prefill_epoch_rs_grouped2_tma_resource_info_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  check_bf16_tile(q, kM, kD, "q");
  check_bf16_tile(k, kN, kD, "k");
  check_bf16_tile(v, kN, kPVN, "v");
  auto tma_k = make_tma_k(k);
  auto tma_v = make_tma_v(v);
  auto tma_kernel = epoch_rs_grouped2_tma_floor_kernel<
      decltype(tma_k), decltype(tma_v)>;
  std::vector<int64_t> values;
  append_kernel_info(
      values, epoch_rs_grouped2_serial_floor_kernel,
      sizeof(EpochSharedStorage<true>), kThreads);
  append_kernel_info(
      values, tma_kernel, sizeof(GroupedTmaEpochSharedStorage),
      kGroupedTmaThreads);
  return torch::tensor(values, torch::TensorOptions().dtype(torch::kInt64));
}

torch::Tensor streamattn_prefill_epoch_rs_cluster2_resource_info_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  check_bf16_tile(q, kM, kD, "q");
  check_bf16_tile(k, kN, kD, "k");
  check_bf16_tile(v, kN, kPVN, "v");
  auto tma_k = make_tma_k(k);
  auto tma_v = make_tma_v(v);
  auto tma_k_multicast = make_tma_k_multicast(k);
  auto tma_v_multicast = make_tma_v_multicast(v);
  auto independent = epoch_rs_cluster2_floor_kernel<
      false, decltype(tma_k), decltype(tma_v)>;
  auto multicast = epoch_rs_cluster2_floor_kernel<
      true, decltype(tma_k_multicast), decltype(tma_v_multicast)>;
  std::vector<int64_t> values;
  append_kernel_info(
      values, independent, sizeof(TmaEpochSharedStorage), kTmaThreads);
  append_kernel_info(
      values, multicast, sizeof(TmaEpochSharedStorage), kTmaThreads);
  return torch::tensor(values, torch::TensorOptions().dtype(torch::kInt64));
}
"""
