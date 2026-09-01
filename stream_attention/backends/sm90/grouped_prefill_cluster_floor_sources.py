"""CUDA sources for the SM90 two-CTA TMA multicast transport floor."""

CPP_SOURCE = r"""
#include <torch/extension.h>

void streamattn_prefill_cluster_independent_out_cuda(
    torch::Tensor k, torch::Tensor v, torch::Tensor output,
    int64_t tiles_per_group);
void streamattn_prefill_cluster_multicast_out_cuda(
    torch::Tensor k, torch::Tensor v, torch::Tensor output,
    int64_t tiles_per_group);
torch::Tensor streamattn_prefill_cluster_resource_info_cuda(
    torch::Tensor k, torch::Tensor v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("independent_out", &streamattn_prefill_cluster_independent_out_cuda);
  m.def("multicast_out", &streamattn_prefill_cluster_multicast_out_cuda);
  m.def("resource_info", &streamattn_prefill_cluster_resource_info_cuda);
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
#include <cutlass/numeric_types.h>
#include <cutlass/pipeline/pipeline.hpp>

using namespace cute;

using Element = cutlass::bfloat16_t;
using Accum = float;

static constexpr int kThreads = 256;
static constexpr int kProducerThreads = 128;
static constexpr int kConsumerThreads = 128;
static constexpr int kStages = 2;
static constexpr int kN = 64;
static constexpr int kD = 128;

using SingleShape = Shape<_1, _1, _1>;
using ClusterShape = Shape<_2, _1, _1>;
using SmemLayoutK = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<Element>{}, Shape<Int<kN>, Int<kD>>{}));
using SmemLayoutV = decltype(tile_to_shape(
    GMMA::Layout_MN_SW128_Atom<Element>{},
    Shape<Int<kD>, Int<kN>>{}, Step<_2, _1>{}));
using SmemLayoutKStages = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<Element>{},
    Shape<Int<kN>, Int<kD>, Int<kStages>>{}));
using SmemLayoutVStages = decltype(tile_to_shape(
    GMMA::Layout_MN_SW128_Atom<Element>{},
    Shape<Int<kD>, Int<kN>, Int<kStages>>{}, Step<_2, _1, _3>{}));

struct alignas(128) SharedStorage {
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutKStages>> k;
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutVStages>> v;
  typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
  typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
  Accum reduction[kConsumerThreads];
};

template <bool Multicast, class TmaK, class TmaV>
__global__ __launch_bounds__(kThreads, 1)
void cluster_transport_floor_kernel(
    CUTLASS_GRID_CONSTANT TmaK const tma_k,
    CUTLASS_GRID_CONSTANT TmaV const tma_v,
    Accum* __restrict__ output,
    int k_rows,
    int v_cols,
    int total_tiles,
    int tiles_per_group) {
  extern __shared__ __align__(128) unsigned char shared_bytes[];
  auto& storage = *reinterpret_cast<SharedStorage*>(shared_bytes);
  using Pipeline = cutlass::PipelineTmaAsync<kStages>;
  using State = typename Pipeline::PipelineState;
  using ActiveClusterShape = std::conditional_t<
      Multicast, ClusterShape, SingleShape>;

  const bool producer = threadIdx.x < kProducerThreads;
  typename Pipeline::Params params_k;
  params_k.transaction_bytes = kN * kD * sizeof(Element);
  params_k.role = producer ? Pipeline::ThreadCategory::Producer
                           : Pipeline::ThreadCategory::Consumer;
  params_k.is_leader = threadIdx.x == 0;
  params_k.num_consumers = kConsumerThreads;
  Pipeline pipeline_k(storage.pipeline_k, params_k, ActiveClusterShape{});
  typename Pipeline::Params params_v = params_k;
  Pipeline pipeline_v(storage.pipeline_v, params_v, ActiveClusterShape{});
  __syncthreads();
  if constexpr (Multicast) {
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
  }

  Tensor sKStages = make_tensor(
      make_smem_ptr(storage.k.data()), SmemLayoutKStages{});
  Tensor sVStages = make_tensor(
      make_smem_ptr(storage.v.data()), SmemLayoutVStages{});
  Tensor mK = tma_k.get_tma_tensor(make_shape(k_rows, Int<kD>{}));
  Tensor gK = local_tile(
      mK, make_shape(Int<kN>{}, Int<kD>{}), make_coord(_, 0));
  Tensor mV = tma_v.get_tma_tensor(make_shape(Int<kD>{}, v_cols));
  Tensor gV = local_tile(
      mV, make_shape(Int<kD>{}, Int<kN>{}), make_coord(0, _));
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
    auto block_layout = Layout<ClusterShape>{};
    CUTE_UNROLL
    for (int block = 0; block < size<0>(block_layout); ++block) {
      multicast_mask |= uint16_t(1) << block_layout(block, _0{}, _0{});
    }
  }

  if (producer) {
    cutlass::arch::warpgroup_reg_dealloc<56>();
    if (threadIdx.x == 0) {
      State write = cutlass::make_producer_start_state<Pipeline>();
      for (int tile = begin; tile < end; ++tile) {
        pipeline_k.producer_acquire(write);
        if constexpr (Multicast) {
          copy(
              tma_k.with(
                  *pipeline_k.producer_get_barrier(write), multicast_mask),
              tKgK(_, tile), tKsK(_, write.index()));
        } else {
          copy(
              tma_k.with(*pipeline_k.producer_get_barrier(write)),
              tKgK(_, tile), tKsK(_, write.index()));
        }
        pipeline_v.producer_acquire(write);
        if constexpr (Multicast) {
          copy(
              tma_v.with(
                  *pipeline_v.producer_get_barrier(write), multicast_mask),
              tVgV(_, tile), tVsV(_, write.index()));
        } else {
          copy(
              tma_v.with(*pipeline_v.producer_get_barrier(write)),
              tVgV(_, tile), tVsV(_, write.index()));
        }
        ++write;
      }
      pipeline_k.producer_tail(write);
      pipeline_v.producer_tail(write);
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<256>();
    const int consumer_idx = threadIdx.x - kProducerThreads;
    State read;
    Accum local = 0.0f;
    for (int tile = begin; tile < end; ++tile) {
      auto token_k = pipeline_k.consumer_try_wait(read);
      pipeline_k.consumer_wait(read, token_k);
      auto token_v = pipeline_v.consumer_try_wait(read);
      pipeline_v.consumer_wait(read, token_v);
      const int k_offset = read.index() * cute::cosize_v<SmemLayoutK>;
      const int v_offset = read.index() * cute::cosize_v<SmemLayoutV>;
      CUTE_UNROLL
      for (int idx = consumer_idx; idx < kN * kD; idx += kConsumerThreads) {
        local += static_cast<Accum>(storage.k[k_offset + idx]);
        local += static_cast<Accum>(storage.v[v_offset + idx]);
      }
      pipeline_k.consumer_release(read);
      pipeline_v.consumer_release(read);
      ++read;
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

static void check_tile(
    torch::Tensor value, int64_t rows, int64_t cols, const char* name) {
  TORCH_CHECK(
      value.is_cuda() && value.is_contiguous(), name,
      " must be contiguous CUDA");
  TORCH_CHECK(
      value.scalar_type() == at::ScalarType::BFloat16, name,
      " must be bf16");
  TORCH_CHECK(
      value.dim() == 3 && value.size(1) == rows && value.size(2) == cols,
      name, " has an invalid shape");
}

static void check_common(
    torch::Tensor k, torch::Tensor v, torch::Tensor output,
    int64_t tiles_per_group) {
  check_tile(k, kN, kD, "k");
  check_tile(v, kN, kD, "v");
  TORCH_CHECK(k.size(0) == v.size(0), "k and v tile counts must match");
  TORCH_CHECK(
      tiles_per_group > 0 && k.size(0) % tiles_per_group == 0,
      "tiles_per_group must divide total tiles");
  TORCH_CHECK(
      output.is_cuda() && output.is_contiguous()
          && output.scalar_type() == at::ScalarType::Float,
      "output must be contiguous CUDA fp32");
  TORCH_CHECK(
      output.numel() == 2 * k.size(0) / tiles_per_group,
      "output must contain two checksums per K/V group");
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
      make_shape(Int<kD>{}, int(tensor.size(0) * kN)),
      make_stride(_1{}, Int<kD>{}));
  return make_tma_atom(
      SM90_TMA_LOAD{}, global, SmemLayoutV{},
      make_shape(Int<kD>{}, Int<kN>{}));
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
      make_shape(Int<kD>{}, int(tensor.size(0) * kN)),
      make_stride(_1{}, Int<kD>{}));
  return make_tma_atom(
      SM90_TMA_LOAD_MULTICAST{}, global, SmemLayoutV{},
      make_shape(Int<kD>{}, Int<kN>{}), Int<2>{});
}

void streamattn_prefill_cluster_independent_out_cuda(
    torch::Tensor k, torch::Tensor v, torch::Tensor output,
    int64_t tiles_per_group) {
  check_common(k, v, output, tiles_per_group);
  auto tma_k = make_tma_k(k);
  auto tma_v = make_tma_v(v);
  auto kernel = cluster_transport_floor_kernel<
      false, decltype(tma_k), decltype(tma_v)>;
  constexpr int shared_bytes = sizeof(SharedStorage);
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
  kernel<<<output.numel(), kThreads, shared_bytes,
           at::cuda::getCurrentCUDAStream()>>>(
      tma_k, tma_v, output.data_ptr<Accum>(), k.size(0) * kN,
      v.size(0) * kN, k.size(0), tiles_per_group);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void streamattn_prefill_cluster_multicast_out_cuda(
    torch::Tensor k, torch::Tensor v, torch::Tensor output,
    int64_t tiles_per_group) {
  check_common(k, v, output, tiles_per_group);
  auto tma_k = make_tma_k_multicast(k);
  auto tma_v = make_tma_v_multicast(v);
  auto kernel = cluster_transport_floor_kernel<
      true, decltype(tma_k), decltype(tma_v)>;
  constexpr int shared_bytes = sizeof(SharedStorage);
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
  cutlass::ClusterLaunchParams params{
      dim3(output.numel()), dim3(kThreads), dim3(2, 1, 1), shared_bytes,
      at::cuda::getCurrentCUDAStream()};
  void const* kernel_ptr = reinterpret_cast<void const*>(kernel);
  auto status = cutlass::launch_kernel_on_cluster(
      params, kernel_ptr, tma_k, tma_v, output.data_ptr<Accum>(),
      int(k.size(0) * kN), int(v.size(0) * kN), int(k.size(0)),
      int(tiles_per_group));
  TORCH_CHECK(status == cutlass::Status::kSuccess, "cluster launch failed");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename Kernel>
static void append_kernel_info(
    std::vector<int64_t>& values, Kernel kernel, int dynamic_shared_bytes) {
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      dynamic_shared_bytes));
  cudaFuncAttributes attributes{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
  int blocks = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, kernel, kThreads, dynamic_shared_bytes));
  values.push_back(attributes.numRegs);
  values.push_back(attributes.sharedSizeBytes);
  values.push_back(dynamic_shared_bytes);
  values.push_back(attributes.localSizeBytes);
  values.push_back(blocks);
  values.push_back(attributes.maxThreadsPerBlock);
}

torch::Tensor streamattn_prefill_cluster_resource_info_cuda(
    torch::Tensor k, torch::Tensor v) {
  check_tile(k, kN, kD, "k");
  check_tile(v, kN, kD, "v");
  auto tma_k = make_tma_k(k);
  auto tma_v = make_tma_v(v);
  auto tma_k_multicast = make_tma_k_multicast(k);
  auto tma_v_multicast = make_tma_v_multicast(v);
  auto independent = cluster_transport_floor_kernel<
      false, decltype(tma_k), decltype(tma_v)>;
  auto multicast = cluster_transport_floor_kernel<
      true, decltype(tma_k_multicast), decltype(tma_v_multicast)>;
  std::vector<int64_t> values;
  append_kernel_info(values, independent, sizeof(SharedStorage));
  append_kernel_info(values, multicast, sizeof(SharedStorage));
  return torch::tensor(values, torch::TensorOptions().dtype(torch::kInt64));
}
"""
