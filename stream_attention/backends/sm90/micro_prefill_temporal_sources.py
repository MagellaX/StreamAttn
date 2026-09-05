"""Isolated R64 temporal producer, shared definitions and verbatim R64 merge."""

from .transposed_gqa_exact_sources import cuda_source_for_head_dim as _base_source


CPP_SOURCE = r"""
#include <torch/extension.h>
void streamattn_temporal_out_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor partial_o, torch::Tensor partial_lse, torch::Tensor output,
    int64_t splits, int64_t component, int64_t protocol);
torch::Tensor streamattn_temporal_resource_info_cuda(torch::Tensor q, int64_t protocol);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("out", &streamattn_temporal_out_cuda);
  m.def("resource_info", &streamattn_temporal_resource_info_cuda);
}
"""

_PREFIX_END = (
    "\n__global__ __launch_bounds__(256, 1)\n"
    "void streamattn_grouped_wgmma_prefill_kernel("
)
_MERGE_START = (
    "\n__global__ __launch_bounds__(128)\n"
    "void streamattn_natural_wgmma_micro_prefill_merge_kernel("
)
_MERGE_END = (
    "\n__global__ __launch_bounds__(128)\nvoid streamattn_transposed_wgmma_qk_kernel("
)


def original_merge_source(head_dim: int) -> str:
    """Fail closed if the original merge's extraction boundary changes."""
    base = _base_source(head_dim)
    if base.count(_MERGE_START) != 1 or base.count(_MERGE_END) != 1:
        raise ValueError(
            "R64 merge anchors changed; review temporal source composition"
        )
    begin, end = base.index(_MERGE_START), base.index(_MERGE_END)
    merge = base[begin:end]
    if (
        end <= begin
        or merge.count("__global__") != 1
        or not merge.rstrip().endswith("}")
    ):
        raise ValueError("R64 merge extraction contains unexpected source")
    return merge


_PRODUCER_SOURCE = r"""
#include <c10/cuda/CUDAGuard.h>
#include <ATen/MemoryOverlap.h>
#include <vector>

static_assert(kPrefillRowsPerWarpGroup == 64 && kBlockM == 64,
              "temporal canary keeps the original R64/K64 geometry");
static_assert(sizeof(GroupedRSPrefillSharedStorage) == 512 * kHeadDim,
              "temporal storage must remain Q64 + K64x2 + V64, all BF16");
static constexpr Accum kTemporalScaleLog2 = kHeadDim == 64
    ? 0.18033688011112042f : 0.12751743082459868f;

// No output or current P operand is reachable here while PV is outstanding.
template <class Scores, class State>
__device__ __forceinline__ void temporal_softmax_next(
    Scores& scores, State& row_max, State& row_sum, State& row_alpha) {
  Tensor score_rows = make_tensor(
      scores.data(), streamattn_acc_rowcol<false>(scores.layout()));
  static_assert(decltype(size<0>(score_rows))::value == decltype(size(row_max))::value);
  CUTE_UNROLL
  for (int row = 0; row < size<0>(score_rows); ++row) {
    Accum tile_max = -INFINITY;
    CUTE_UNROLL
    for (int col = 0; col < size<1>(score_rows); ++col) {
      tile_max = fmaxf(tile_max, score_rows(row, col));
    }
    tile_max = streamattn_quad_max(tile_max);
    const Accum next_max = fmaxf(row_max(row), tile_max);
    const Accum alpha = row_max(row) == -INFINITY ? 0.0f
        : exp2f((row_max(row) - next_max) * kTemporalScaleLog2);
    row_alpha(row) = alpha;
    row_max(row) = next_max;
    row_sum(row) *= alpha;
    Accum local_sum = 0.0f;
    const Accum max_scaled = next_max * kTemporalScaleLog2;
    CUTE_UNROLL
    for (int col = 0; col < size<1>(score_rows); ++col) {
      const Accum probability =
          exp2f(score_rows(row, col) * kTemporalScaleLog2 - max_scaled);
      score_rows(row, col) = probability;
      local_sum += probability;
    }
    row_sum(row) += streamattn_quad_sum(local_sum);
  }
}

template <class Scores, class Probability>
__device__ __forceinline__ void temporal_pack_p(
    Scores const& scores, Probability& probability) {
  Tensor source = make_tensor(scores.data(),
      streamattn_convert_layout_acc_aregs<PrefillRSTiledMmaPV>(scores.layout()));
  constexpr int count = decltype(size(source))::value;
  static_assert(count == decltype(size(probability))::value);
  // Same NumericArrayConverter as the original producer, into owned registers.
  cutlass::NumericArrayConverter<Element, Accum, count> convert;
  *reinterpret_cast<cutlass::Array<Element, count>*>(probability.data()) =
      convert(*reinterpret_cast<const cutlass::Array<Accum, count>*>(source.data()));
}

template <bool FullDrain>
__global__ __launch_bounds__(128)
void streamattn_temporal_micro_prefill_partial_kernel(
    const Element* __restrict__ query, const Element* __restrict__ key,
    const Element* __restrict__ value, Accum* __restrict__ partial_o,
    Accum* __restrict__ partial_lse, int query_length, int kv_length,
    int q_heads, int kv_heads, int group_size, int num_splits) {
  constexpr int kQueryRows = kPrefillRowsPerWarpGroup;
  const int positions = kQueryRows / group_size;
  const int query_tiles = (query_length + positions - 1) / positions;
  const int work = blockIdx.x;
  const int work_group = work / num_splits;
  const int split = work % num_splits;
  const int batch_kv_group = work_group / query_tiles;
  const int batch = batch_kv_group / kv_heads;
  const int kv_head = batch_kv_group % kv_heads;
  const int query_begin = (work_group % query_tiles) * positions;
  const int num_tiles = kv_length / kBlockM;
  // Same balanced intervals; widen the multiply before dividing.
  const int tile_begin = static_cast<int>(static_cast<int64_t>(split) * num_tiles / num_splits);
  const int tile_end = static_cast<int>(static_cast<int64_t>(split + 1) * num_tiles / num_splits);

  extern __shared__ __align__(128) unsigned char shared_bytes[];
  auto& storage = *reinterpret_cast<GroupedRSPrefillSharedStorage*>(shared_bytes);
  Tensor sQ = make_tensor(make_smem_ptr(storage.q.data()), PrefillSmemLayoutQ{});
  Tensor sK0 = make_tensor(make_smem_ptr(storage.k.data()), SmemLayoutK{});
  Tensor sK1 = make_tensor(
      make_smem_ptr(storage.k.data() + cute::cosize_v<SmemLayoutK>), SmemLayoutK{});
  Tensor sV = make_tensor(make_smem_ptr(storage.v.data()), PrefillRSSmemLayoutV{});
  for (int idx = threadIdx.x; idx < kQueryRows * kHeadDim; idx += 128) {
    const int row = idx / kHeadDim;
    const int dim = idx % kHeadDim;
    const int position = query_begin + row / group_size;
    Element item = Element(0.0f);
    if (position < query_length) {
      const int head = kv_head * group_size + row % group_size;
      item = query[((static_cast<int64_t>(batch) * query_length + position)
          * q_heads + head) * kHeadDim + dim];
    }
    sQ(row, dim) = item;
  }
  cutlass::arch::fence_view_async_shared();
  __syncthreads();

  PrefillTiledMma tiled_qk;
  auto thread_qk = tiled_qk.get_thread_slice(threadIdx.x);
  Tensor rQ = thread_qk.partition_fragment_A(sQ);
  Tensor rK0 = thread_qk.partition_fragment_B(sK0);
  Tensor rK1 = thread_qk.partition_fragment_B(sK1);
  Tensor scores = partition_fragment_C(tiled_qk, Shape<_64, _64>{});
  Tensor p_regs = make_tensor<Element>(
      streamattn_convert_layout_acc_aregs<PrefillRSTiledMmaPV>(scores.layout()));
  PrefillRSTiledMmaPV tiled_pv;
  auto thread_pv = tiled_pv.get_thread_slice(threadIdx.x);
  Tensor rV = thread_pv.partition_fragment_B(sV);
  Tensor output_acc = partition_fragment_C(tiled_pv, Shape<_64, Int<kHeadDim>>{});
  clear(output_acc);
  Tensor output_rows = make_tensor(
      output_acc.data(), streamattn_acc_rowcol<false>(output_acc.layout()));
  constexpr int kRowsPerThread = decltype(size<0>(output_rows))::value;
  Tensor row_max = make_tensor<Accum>(Shape<Int<kRowsPerThread>>{});
  Tensor row_sum = make_tensor<Accum>(Shape<Int<kRowsPerThread>>{});
  Tensor row_alpha = make_tensor<Accum>(Shape<Int<kRowsPerThread>>{});
  fill(row_max, -INFINITY);
  clear(row_sum);

  GmemCopyK copy_k;
  auto thread_copy_k = copy_k.get_thread_slice(threadIdx.x);
  Tensor dst_k0 = thread_copy_k.partition_D(sK0);
  Tensor dst_k1 = thread_copy_k.partition_D(sK1);
  GmemCopyPrefillRSV copy_v;
  auto thread_copy_v = copy_v.get_thread_slice(threadIdx.x);
  Tensor dst_v = thread_copy_v.partition_D(sV);
  auto copy_k_tile = [&](int tile, auto destination) {
    const Element* source = key
        + (static_cast<int64_t>(batch_kv_group) * kv_length + tile * kBlockM) * kHeadDim;
    Tensor global = make_tensor(make_gmem_ptr(source), Shape<_64, Int<kHeadDim>>{},
                                make_stride(Int<kHeadDim>{}, _1{}));
    cute::copy(copy_k, thread_copy_k.partition_S(global), destination);
  };
  auto copy_v_tile = [&](int tile) {
    const Element* source = value
        + (static_cast<int64_t>(batch_kv_group) * kv_length + tile * kBlockM) * kHeadDim;
    Tensor global = make_tensor(make_gmem_ptr(source), Shape<Int<kHeadDim>, _64>{},
                                make_stride(_1{}, Int<kHeadDim>{}));
    cute::copy(copy_v, thread_copy_v.partition_S(global), dst_v);
  };

  // Prologue: P0 and statistics, with K1/V0 staged exactly once.
  copy_k_tile(tile_begin, dst_k0);
  cute::cp_async_fence();
  cute::cp_async_wait<0>();
  __syncthreads();
  if (tile_begin + 1 < tile_end) {
    copy_k_tile(tile_begin + 1, dst_k1);
    cute::cp_async_fence();
  }
  clear(scores);
  warpgroup_fence_operand(scores);
  warpgroup_arrive();
  cute::gemm(tiled_qk, rQ, rK0, scores);
  warpgroup_commit_batch();
  warpgroup_wait<0>();
  warpgroup_fence_operand(scores);
  copy_v_tile(tile_begin);
  cute::cp_async_fence();
  temporal_softmax_next(scores, row_max, row_sum, row_alpha);
  temporal_pack_p(scores, p_regs);
  cute::cp_async_wait<0>();
  __syncthreads();

  int read_pipe = 0;
  for (int tile = tile_begin; tile + 1 < tile_end; ++tile) {
    // K[t] was retired before this iteration; prefetch K[t+2] into that slot.
    if (tile + 2 < tile_end) {
      if (read_pipe == 0) { copy_k_tile(tile + 2, dst_k0); }
      else { copy_k_tile(tile + 2, dst_k1); }
      cute::cp_async_fence();
    }
    clear(scores);
    warpgroup_fence_operand(scores);
    warpgroup_fence_operand(p_regs);
    warpgroup_fence_operand(output_acc);
    warpgroup_arrive();
    if (read_pipe == 0) { cute::gemm(tiled_qk, rQ, rK1, scores); }
    else { cute::gemm(tiled_qk, rQ, rK0, scores); }
    warpgroup_commit_batch();
    cute::gemm(tiled_pv, p_regs, rV, output_acc);
    warpgroup_commit_batch();
    if constexpr (FullDrain) { warpgroup_wait<0>(); }
    else { warpgroup_wait<1>(); }
    warpgroup_fence_operand(scores);
    temporal_softmax_next(scores, row_max, row_sum, row_alpha);

    // PV[t] must retire before rescaling O or replacing its register P operand.
    warpgroup_wait<0>();
    warpgroup_fence_operand(output_acc);
    warpgroup_fence_operand(p_regs);
    __syncthreads();
    copy_v_tile(tile + 1);
    cute::cp_async_fence();
    CUTE_UNROLL
    for (int row = 0; row < kRowsPerThread; ++row) {
      CUTE_UNROLL
      for (int col = 0; col < size<1>(output_rows); ++col) {
        output_rows(row, col) *= row_alpha(row);
      }
    }
    temporal_pack_p(scores, p_regs);
    cute::cp_async_wait<0>();
    __syncthreads();
    read_pipe ^= 1;
  }

  // Tail also handles a one-tile split; no lookahead K/V is read past tile_end.
  warpgroup_fence_operand(p_regs);
  warpgroup_fence_operand(output_acc);
  warpgroup_arrive();
  cute::gemm(tiled_pv, p_regs, rV, output_acc);
  warpgroup_commit_batch();
  warpgroup_wait<0>();
  warpgroup_fence_operand(p_regs);
  warpgroup_fence_operand(output_acc);
  Tensor identity = make_identity_tensor(Shape<_64, Int<kHeadDim>>{});
  Tensor coordinates = thread_pv.partition_C(identity);
  Tensor coords = make_tensor(
      coordinates.data(), streamattn_acc_rowcol<false>(coordinates.layout()));
  CUTE_UNROLL
  for (int row = 0; row < size<0>(output_rows); ++row) {
    CUTE_UNROLL
    for (int col = 0; col < size<1>(output_rows); ++col) {
      const auto coordinate = coords(row, col);
      const int local_query_row = int(get<0>(coordinate));
      const int dim = int(get<1>(coordinate));
      const int64_t state_row = static_cast<int64_t>(work) * kQueryRows + local_query_row;
      partial_o[state_row * kHeadDim + dim] = row_sum(row) > 0.0f
          ? output_rows(row, col) / row_sum(row) : 0.0f;
      if (dim == 0) {
        partial_lse[state_row] = row_sum(row) > 0.0f
            ? row_max(row) * kTemporalScaleLog2 + log2f(row_sum(row)) : -INFINITY;
      }
    }
  }
}
"""

_HOST_SOURCE = r"""
template <bool FullDrain>
static void temporal_launch(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor po, torch::Tensor pl, torch::Tensor o,
    int splits, int component, cudaStream_t stream) {
  const int group = q.size(2) / k.size(1);
  const int positions = kPrefillRowsPerWarpGroup / group;
  const int query_tiles = (q.size(1) + positions - 1) / positions;
  if (component != 2) {
    streamattn_temporal_micro_prefill_partial_kernel<FullDrain><<<
        q.size(0) * k.size(1) * query_tiles * splits, 128,
        sizeof(GroupedRSPrefillSharedStorage), stream>>>(
        reinterpret_cast<const Element*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const Element*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const Element*>(v.data_ptr<at::BFloat16>()),
        po.data_ptr<float>(), pl.data_ptr<float>(),
        q.size(1), k.size(2), q.size(2), k.size(1), group, splits);
  }
  if (component != 1) {
    streamattn_natural_wgmma_micro_prefill_merge_kernel<<<
        q.size(0) * q.size(1) * q.size(2), 128, 0, stream>>>(
        po.data_ptr<float>(), pl.data_ptr<float>(),
        reinterpret_cast<Element*>(o.data_ptr<at::BFloat16>()),
        q.size(0), q.size(1), q.size(2), k.size(1), group, splits);
  }
}

void streamattn_temporal_out_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor po, torch::Tensor pl, torch::Tensor o,
    int64_t splits, int64_t component, int64_t protocol) {
  TORCH_CHECK(q.is_cuda(), "query must be CUDA");
  const c10::cuda::CUDAGuard guard(q.device());
  for (const auto& t : {q, k, v, po, pl, o}) {
    TORCH_CHECK(t.device() == q.device() && t.is_contiguous(),
                "buffers must be contiguous on the query device");
  }
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && k.sizes() == v.sizes(), "invalid Q/K/V shape");
  TORCH_CHECK(q.size(0) > 0 && q.size(0) == k.size(0) && q.size(1) >= 1 && q.size(1) <= 64,
              "require B>0 and M in [1,64]");
  TORCH_CHECK(q.size(3) == kHeadDim && k.size(3) == kHeadDim, "head dimension mismatch");
  TORCH_CHECK(k.size(1) > 0 && q.size(2) % k.size(1) == 0, "invalid GQA");
  const int64_t group = q.size(2) / k.size(1);
  TORCH_CHECK(group == 4 || group == 8, "G must be 4 or 8");
  TORCH_CHECK(k.size(2) > 0 && k.size(2) % 64 == 0 && k.size(2) <= INT_MAX,
              "N must be a positive multiple of 64 within int32");
  TORCH_CHECK(splits > 0 && splits <= k.size(2) / 64 && splits <= 512, "invalid splits");
  TORCH_CHECK(protocol == 0 || protocol == 1, "invalid protocol");
  TORCH_CHECK(component >= 0 && component <= 2, "invalid component");
  for (const auto& t : {q, k, v, o}) {
    TORCH_CHECK(t.scalar_type() == at::ScalarType::BFloat16, "Q/K/V/output must be BF16");
  }
  TORCH_CHECK(po.scalar_type() == at::ScalarType::Float && pl.scalar_type() == at::ScalarType::Float,
              "partial state must be FP32");
  TORCH_CHECK(o.sizes() == q.sizes(), "output shape mismatch");
  const int64_t query_tiles = (q.size(1) + 64 / group - 1) / (64 / group);
  const int64_t groups = q.size(0) * k.size(1) * query_tiles;
  TORCH_CHECK(groups * splits <= INT_MAX && q.size(0) * q.size(1) * q.size(2) <= INT_MAX,
              "grid extent exceeds int32");
  TORCH_CHECK(po.sizes() == torch::IntArrayRef({groups, splits, int64_t(64), int64_t(kHeadDim)}),
              "partial output must have shape [B*Hkv*Qtiles,S,64,D]");
  TORCH_CHECK(pl.sizes() == torch::IntArrayRef({groups, splits, int64_t(64)}),
              "partial LSE must have shape [B*Hkv*Qtiles,S,64]");
  for (const auto& destination : {po, pl, o}) {
    for (const auto& source : {q, k, v}) { at::assert_no_overlap(destination, source); }
  }
  at::assert_no_overlap(po, pl); at::assert_no_overlap(po, o); at::assert_no_overlap(pl, o);
  const auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
  if (protocol == 0) { temporal_launch<false>(q,k,v,po,pl,o,splits,component,stream); }
  else { temporal_launch<true>(q,k,v,po,pl,o,splits,component,stream); }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <class Kernel>
static void temporal_append_resources(std::vector<int64_t>& values, Kernel kernel, int shared_bytes) {
  if (shared_bytes > 0) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
  }
  cudaFuncAttributes attributes{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
  int blocks = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, 128, shared_bytes));
  values.insert(values.end(), {attributes.numRegs, static_cast<int64_t>(attributes.sharedSizeBytes),
      shared_bytes, static_cast<int64_t>(attributes.localSizeBytes), blocks, attributes.maxThreadsPerBlock});
}

torch::Tensor streamattn_temporal_resource_info_cuda(torch::Tensor q, int64_t protocol) {
  TORCH_CHECK(q.is_cuda(), "query must be CUDA");
  TORCH_CHECK(q.dim() == 4 && q.size(3) == kHeadDim, "head dimension mismatch");
  TORCH_CHECK(protocol == 0 || protocol == 1, "invalid protocol");
  const c10::cuda::CUDAGuard guard(q.device());
  std::vector<int64_t> values;
  if (protocol == 0) {
    temporal_append_resources(values, streamattn_temporal_micro_prefill_partial_kernel<false>,
                              sizeof(GroupedRSPrefillSharedStorage));
  } else {
    temporal_append_resources(values, streamattn_temporal_micro_prefill_partial_kernel<true>,
                              sizeof(GroupedRSPrefillSharedStorage));
  }
  temporal_append_resources(values, streamattn_natural_wgmma_micro_prefill_merge_kernel, 0);
  return torch::tensor(values, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
}
"""


def cuda_source_for_head_dim(head_dim: int) -> str:
    base = _base_source(head_dim)
    if base.count(_PREFIX_END) != 1:
        raise ValueError("SM90 definition anchor changed; review temporal composition")
    prefix = base.split(_PREFIX_END, 1)[0]
    if "__global__" in prefix:
        raise ValueError("SM90 definitions unexpectedly contain a kernel")
    return prefix + _PRODUCER_SOURCE + original_merge_source(head_dim) + _HOST_SOURCE


CUDA_SOURCE = cuda_source_for_head_dim(64)
