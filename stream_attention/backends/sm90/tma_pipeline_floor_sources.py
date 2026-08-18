"""CUDA sources for the SM90 D128 TMA pipeline floor experiment.

This extension is intentionally separate from the promoted exact backend.  It
measures data movement and producer/consumer synchronization without changing
the attention math or public routing surface.
"""

CPP_SOURCE = r"""
#include <torch/extension.h>

void streamattn_cp_async_k_out_cuda(
    torch::Tensor k, torch::Tensor output, int64_t tiles_per_cta);
void streamattn_cp_async_kv_out_cuda(
    torch::Tensor k, torch::Tensor v, torch::Tensor output,
    int64_t tiles_per_cta);
void streamattn_tma_k_out_cuda(
    torch::Tensor k, torch::Tensor output, int64_t tiles_per_cta);
void streamattn_tma_kv_out_cuda(
    torch::Tensor k, torch::Tensor v, torch::Tensor output,
    int64_t tiles_per_cta);
torch::Tensor streamattn_tma_floor_resource_info_cuda(
    torch::Tensor k, torch::Tensor v, int64_t tiles_per_cta);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cp_async_k_out", &streamattn_cp_async_k_out_cuda);
  m.def("cp_async_kv_out", &streamattn_cp_async_kv_out_cuda);
  m.def("tma_k_out", &streamattn_tma_k_out_cuda);
  m.def("tma_kv_out", &streamattn_tma_kv_out_cuda);
  m.def("resource_info", &streamattn_tma_floor_resource_info_cuda);
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
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>
#include <cutlass/numeric_types.h>
#include <cutlass/pipeline/pipeline.hpp>

using namespace cute;

using Element = cutlass::bfloat16_t;

static constexpr int kTileRows = 64;
static constexpr int kHeadDim = 128;
static constexpr int kTileElements = kTileRows * kHeadDim;
static constexpr int kTileBytes = kTileElements * sizeof(Element);
static constexpr int kKStages = 2;
static constexpr int kVStages = 1;
static constexpr int kCpThreads = 128;
// WGMMA consumer warpgroups must begin on a 128-thread boundary.  Keep a full
// producer warpgroup so the consumer is the aligned thread range [128, 256).
// Only the first producer warp performs useful work; register deallocation
// makes the other producer warps inexpensive.
static constexpr int kTmaThreads = 256;
static constexpr int kConsumers = 128;

using SmemLayoutTile = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<Element>{},
    Shape<Int<kTileRows>, Int<kHeadDim>>{}));
using SmemLayoutK2 = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<Element>{},
    Shape<Int<kTileRows>, Int<kHeadDim>, Int<kKStages>>{}));
using SmemLayoutV1 = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<Element>{},
    Shape<Int<kTileRows>, Int<kHeadDim>, Int<kVStages>>{}));

using GmemCopy = decltype(make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>{},
    Layout<Shape<_16, _8>, Stride<_8, _1>>{},
    Layout<Shape<_1, _8>>{}));

template <int VStages>
struct alignas(128) TmaSharedStorage {
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutK2>> k;
  cute::array_aligned<
      Element,
      VStages == 1 ? cute::cosize_v<SmemLayoutV1> : 1> v;
  typename cutlass::PipelineTmaAsync<kKStages>::SharedStorage pipeline_k;
  typename cutlass::PipelineTmaAsync<kVStages>::SharedStorage pipeline_v;
  float reduction[kConsumers];
};

struct alignas(128) CpSharedStorage {
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutK2>> k;
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutV1>> v;
  float reduction[kCpThreads];
};

__device__ __forceinline__ float element_to_float(Element value) {
  return static_cast<float>(value);
}

template <typename TensorT>
__device__ __forceinline__ float consume_tile(TensorT tile, int consumer_idx) {
  float value = 0.0f;
  for (int idx = consumer_idx; idx < kTileElements; idx += kConsumers) {
    const int row = idx / kHeadDim;
    const int dim = idx - row * kHeadDim;
    value += element_to_float(tile(row, dim));
  }
  return value;
}

__device__ __forceinline__ void reduce_and_store(
    float value, float* reduction, float* output, int consumer_idx) {
  reduction[consumer_idx] = value;
  cutlass::arch::NamedBarrier::sync(kConsumers, 0);
  if (consumer_idx == 0) {
    float total = 0.0f;
    for (int idx = 0; idx < kConsumers; ++idx) {
      total += reduction[idx];
    }
    output[blockIdx.x] = total;
  }
}

template <bool LoadV>
__global__ __launch_bounds__(kCpThreads)
void cp_async_floor_kernel(
    const Element* __restrict__ k,
    const Element* __restrict__ v,
    float* __restrict__ output,
    int total_tiles,
    int tiles_per_cta) {
  extern __shared__ char shared_memory[];
  auto& storage = *reinterpret_cast<CpSharedStorage*>(shared_memory);
  Tensor sK0 = make_tensor(make_smem_ptr(storage.k.data()), SmemLayoutTile{});
  Tensor sK1 = make_tensor(
      make_smem_ptr(storage.k.data() + cute::cosize_v<SmemLayoutTile>),
      SmemLayoutTile{});
  Tensor sV0 = make_tensor(make_smem_ptr(storage.v.data()), SmemLayoutTile{});

  GmemCopy copy;
  auto thr_copy = copy.get_thread_slice(threadIdx.x);
  const int tile_begin = blockIdx.x * tiles_per_cta;
  const int tile_end = min(total_tiles, tile_begin + tiles_per_cta);
  float local = 0.0f;

  for (int tile = tile_begin; tile < tile_end; ++tile) {
    const int pipe = (tile - tile_begin) & 1;
    auto sKStage = pipe == 0 ? sK0 : sK1;
    Tensor gK = make_tensor(
        make_gmem_ptr(k + static_cast<int64_t>(tile) * kTileElements),
        Shape<Int<kTileRows>, Int<kHeadDim>>{},
        make_stride(Int<kHeadDim>{}, _1{}));
    cute::copy(copy, thr_copy.partition_S(gK), thr_copy.partition_D(sKStage));
    if constexpr (LoadV) {
      Tensor gV = make_tensor(
          make_gmem_ptr(v + static_cast<int64_t>(tile) * kTileElements),
          Shape<Int<kTileRows>, Int<kHeadDim>>{},
          make_stride(Int<kHeadDim>{}, _1{}));
      cute::copy(copy, thr_copy.partition_S(gV), thr_copy.partition_D(sV0));
    }
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();
    local += consume_tile(sKStage, threadIdx.x);
    if constexpr (LoadV) {
      local += consume_tile(sV0, threadIdx.x);
    }
    __syncthreads();
  }

  storage.reduction[threadIdx.x] = local;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
    for (int idx = 0; idx < kCpThreads; ++idx) {
      total += storage.reduction[idx];
    }
    output[blockIdx.x] = total;
  }
}

template <bool LoadV, class TmaK, class TmaV>
__global__ __launch_bounds__(kTmaThreads)
void tma_floor_kernel(
    CUTLASS_GRID_CONSTANT TmaK const tma_k,
    CUTLASS_GRID_CONSTANT TmaV const tma_v,
    float* __restrict__ output,
    int total_rows,
    int total_tiles,
    int tiles_per_cta) {
  extern __shared__ char shared_memory[];
  auto& storage = *reinterpret_cast<TmaSharedStorage<LoadV ? 1 : 0>*>(
      shared_memory);

  using PipelineK = cutlass::PipelineTmaAsync<kKStages>;
  using PipelineV = cutlass::PipelineTmaAsync<kVStages>;
  using StateK = typename PipelineK::PipelineState;
  using StateV = typename PipelineV::PipelineState;

  typename PipelineK::Params params_k;
  params_k.transaction_bytes = kTileBytes;
  params_k.role = threadIdx.x < 128
      ? PipelineK::ThreadCategory::Producer
      : PipelineK::ThreadCategory::Consumer;
  params_k.is_leader = threadIdx.x == 0;
  params_k.num_consumers = kConsumers;
  PipelineK pipeline_k(
      storage.pipeline_k, params_k, Shape<_1, _1, _1>{});

  typename PipelineV::Params params_v;
  params_v.transaction_bytes = kTileBytes;
  params_v.role = threadIdx.x < 128
      ? PipelineV::ThreadCategory::Producer
      : PipelineV::ThreadCategory::Consumer;
  params_v.is_leader = threadIdx.x == 0;
  params_v.num_consumers = kConsumers;
  PipelineV pipeline_v(
      storage.pipeline_v, params_v, Shape<_1, _1, _1>{});
  __syncthreads();

  Tensor sK = make_tensor(make_smem_ptr(storage.k.data()), SmemLayoutK2{});
  Tensor sK0 = make_tensor(make_smem_ptr(storage.k.data()), SmemLayoutTile{});
  Tensor sK1 = make_tensor(
      make_smem_ptr(storage.k.data() + cute::cosize_v<SmemLayoutTile>),
      SmemLayoutTile{});
  Tensor mK = tma_k.get_tma_tensor(
      make_shape(total_rows, Int<kHeadDim>{}));
  Tensor gK = local_tile(
      mK,
      make_shape(Int<kTileRows>{}, Int<kHeadDim>{}),
      make_coord(_, 0));
  auto [tKgK, tKsK] = tma_partition(
      tma_k,
      _0{},
      Layout<_1>{},
      group_modes<0, 2>(sK),
      group_modes<0, 2>(gK));

  Tensor sV = make_tensor(make_smem_ptr(storage.v.data()), SmemLayoutV1{});
  Tensor sV0 = make_tensor(make_smem_ptr(storage.v.data()), SmemLayoutTile{});
  Tensor mV = tma_v.get_tma_tensor(
      make_shape(total_rows, Int<kHeadDim>{}));
  Tensor gV = local_tile(
      mV,
      make_shape(Int<kTileRows>{}, Int<kHeadDim>{}),
      make_coord(_, 0));
  auto [tVgV, tVsV] = tma_partition(
      tma_v,
      _0{},
      Layout<_1>{},
      group_modes<0, 2>(sV),
      group_modes<0, 2>(gV));

  const int tile_begin = blockIdx.x * tiles_per_cta;
  const int tile_end = min(total_tiles, tile_begin + tiles_per_cta);

  if (threadIdx.x < 128) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    if (threadIdx.x == 0) {
      StateK write_k = cutlass::make_producer_start_state<PipelineK>();
      StateV write_v = cutlass::make_producer_start_state<PipelineV>();
      for (int tile = tile_begin; tile < tile_end; ++tile) {
        pipeline_k.producer_acquire(write_k);
        copy(
            tma_k.with(*pipeline_k.producer_get_barrier(write_k)),
            tKgK(_, tile),
            tKsK(_, write_k.index()));
        ++write_k;
        if constexpr (LoadV) {
          pipeline_v.producer_acquire(write_v);
          copy(
              tma_v.with(*pipeline_v.producer_get_barrier(write_v)),
              tVgV(_, tile),
              tVsV(_, write_v.index()));
          ++write_v;
        }
      }
      pipeline_k.producer_tail(write_k);
      if constexpr (LoadV) {
        pipeline_v.producer_tail(write_v);
      }
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<160>();
    const int consumer_idx = threadIdx.x - 128;
    StateK read_k;
    StateV read_v;
    float local = 0.0f;
    for (int tile = tile_begin; tile < tile_end; ++tile) {
      auto token_k = pipeline_k.consumer_try_wait(read_k);
      pipeline_k.consumer_wait(read_k, token_k);
      auto sKRead = read_k.index() == 0 ? sK0 : sK1;
      local += consume_tile(sKRead, consumer_idx);
      if constexpr (LoadV) {
        auto token_v = pipeline_v.consumer_try_wait(read_v);
        pipeline_v.consumer_wait(read_v, token_v);
        local += consume_tile(sV0, consumer_idx);
      }
      pipeline_k.consumer_release(read_k);
      ++read_k;
      if constexpr (LoadV) {
        pipeline_v.consumer_release(read_v);
        ++read_v;
      }
    }
    reduce_and_store(local, storage.reduction, output, consumer_idx);
  }
}

static void check_input(torch::Tensor tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16, name, " must be bf16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 2 && tensor.size(1) == kHeadDim,
              name, " must have shape [rows, 128]");
  TORCH_CHECK(tensor.size(0) % kTileRows == 0,
              name, " rows must be divisible by 64");
}

template <bool LoadV>
void launch_cp(
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor output,
    int64_t tiles_per_cta) {
  check_input(k, "k");
  if constexpr (LoadV) {
    check_input(v, "v");
    TORCH_CHECK(k.sizes() == v.sizes(), "k/v shapes must match");
  }
  TORCH_CHECK(tiles_per_cta > 0, "tiles_per_cta must be positive");
  const int total_tiles = int(k.size(0) / kTileRows);
  const int blocks = (total_tiles + int(tiles_per_cta) - 1) / int(tiles_per_cta);
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == torch::kFloat32,
              "output must be CUDA fp32");
  TORCH_CHECK(output.is_contiguous() && output.numel() == blocks,
              "output must be contiguous with one value per CTA");
  const int smem = int(sizeof(CpSharedStorage));
  auto kernel = cp_async_floor_kernel<LoadV>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  kernel<<<blocks, kCpThreads, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const Element*>(k.data_ptr()),
      LoadV ? reinterpret_cast<const Element*>(v.data_ptr()) : nullptr,
      output.data_ptr<float>(),
      total_tiles,
      int(tiles_per_cta));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool LoadV>
auto make_tma(torch::Tensor tensor) {
  const int rows = int(tensor.size(0));
  Tensor global = make_tensor(
      reinterpret_cast<Element*>(tensor.data_ptr()),
      make_shape(rows, Int<kHeadDim>{}),
      make_stride(Int<kHeadDim>{}, _1{}));
  return make_tma_atom(
      SM90_TMA_LOAD{},
      global,
      SmemLayoutTile{},
      make_shape(Int<kTileRows>{}, Int<kHeadDim>{}));
}

template <bool LoadV>
void launch_tma(
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor output,
    int64_t tiles_per_cta) {
  check_input(k, "k");
  if constexpr (LoadV) {
    check_input(v, "v");
    TORCH_CHECK(k.sizes() == v.sizes(), "k/v shapes must match");
  }
  TORCH_CHECK(tiles_per_cta > 0, "tiles_per_cta must be positive");
  const int total_rows = int(k.size(0));
  const int total_tiles = total_rows / kTileRows;
  const int blocks = (total_tiles + int(tiles_per_cta) - 1) / int(tiles_per_cta);
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == torch::kFloat32,
              "output must be CUDA fp32");
  TORCH_CHECK(output.is_contiguous() && output.numel() == blocks,
              "output must be contiguous with one value per CTA");
  auto tma_k = make_tma<LoadV>(k);
  auto tma_v = make_tma<LoadV>(LoadV ? v : k);
  auto kernel = tma_floor_kernel<LoadV, decltype(tma_k), decltype(tma_v)>;
  const int smem = int(sizeof(TmaSharedStorage<LoadV ? 1 : 0>));
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  kernel<<<blocks, kTmaThreads, smem, at::cuda::getCurrentCUDAStream()>>>(
      tma_k,
      tma_v,
      output.data_ptr<float>(),
      total_rows,
      total_tiles,
      int(tiles_per_cta));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void streamattn_cp_async_k_out_cuda(
    torch::Tensor k, torch::Tensor output, int64_t tiles_per_cta) {
  launch_cp<false>(k, torch::Tensor(), output, tiles_per_cta);
}

void streamattn_cp_async_kv_out_cuda(
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor output,
    int64_t tiles_per_cta) {
  launch_cp<true>(k, v, output, tiles_per_cta);
}

void streamattn_tma_k_out_cuda(
    torch::Tensor k, torch::Tensor output, int64_t tiles_per_cta) {
  launch_tma<false>(k, torch::Tensor(), output, tiles_per_cta);
}

void streamattn_tma_kv_out_cuda(
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor output,
    int64_t tiles_per_cta) {
  launch_tma<true>(k, v, output, tiles_per_cta);
}

template <typename Kernel>
void append_kernel_info(
    std::vector<int64_t>& values,
    Kernel kernel,
    int threads,
    int dynamic_smem) {
  cudaFuncAttributes attrs{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attrs, kernel));
  int blocks_per_sm = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, kernel, threads, dynamic_smem));
  values.push_back(attrs.numRegs);
  values.push_back(attrs.sharedSizeBytes);
  values.push_back(dynamic_smem);
  values.push_back(blocks_per_sm);
  values.push_back(attrs.maxThreadsPerBlock);
}

torch::Tensor streamattn_tma_floor_resource_info_cuda(
    torch::Tensor k, torch::Tensor v, int64_t tiles_per_cta) {
  check_input(k, "k");
  check_input(v, "v");
  auto tma_k = make_tma<true>(k);
  auto tma_v = make_tma<true>(v);
  auto tma_k_kernel = tma_floor_kernel<false, decltype(tma_k), decltype(tma_v)>;
  auto tma_kv_kernel = tma_floor_kernel<true, decltype(tma_k), decltype(tma_v)>;
  std::vector<int64_t> values;
  values.reserve(27);
  values.push_back(kTileBytes);
  values.push_back(sizeof(CpSharedStorage));
  values.push_back(sizeof(TmaSharedStorage<0>));
  values.push_back(sizeof(TmaSharedStorage<1>));
  values.push_back(2 * kTileBytes);
  values.push_back(3 * kTileBytes);
  values.push_back(4 * kTileBytes);
  append_kernel_info(
      values, cp_async_floor_kernel<false>, kCpThreads, sizeof(CpSharedStorage));
  append_kernel_info(
      values, cp_async_floor_kernel<true>, kCpThreads, sizeof(CpSharedStorage));
  append_kernel_info(
      values, tma_k_kernel, kTmaThreads, sizeof(TmaSharedStorage<0>));
  append_kernel_info(
      values, tma_kv_kernel, kTmaThreads, sizeof(TmaSharedStorage<1>));
  return torch::tensor(values, torch::TensorOptions().dtype(torch::kInt64));
}
"""
