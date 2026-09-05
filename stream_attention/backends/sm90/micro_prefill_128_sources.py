"""Isolated M128 SM90 canary; reuse definitions, never existing kernel bodies."""

from .transposed_gqa_exact_sources import cuda_source_for_head_dim as _base_source


CPP_SOURCE = r"""
#include <torch/extension.h>
void streamattn_micro128_out_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor partial_o, torch::Tensor partial_lse,
    torch::Tensor output, torch::Tensor lse,
    int64_t splits, int64_t component, int64_t protocol, bool direct);
torch::Tensor streamattn_micro128_resource_info_cuda(
    torch::Tensor q, int64_t group_size, int64_t protocol, bool direct);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("out", &streamattn_micro128_out_cuda);
  m.def("resource_info", &streamattn_micro128_resource_info_cuda);
}
"""

_PREFIX_END = (
    "\n__global__ __launch_bounds__(256, 1)\n"
    "void streamattn_grouped_wgmma_prefill_kernel("
)

_CANARY_SOURCE = r"""
#include <c10/cuda/CUDAGuard.h>
#include <ATen/MemoryOverlap.h>
#include <vector>

static constexpr int kMicro128Rows = 128;
static constexpr Accum kMicro128ScaleLog2 = kHeadDim == 64
    ? 0.18033688011112042f : 0.12751743082459868f;
static constexpr Accum kMicro128Ln2 = 0.6931471805599453f;

struct alignas(128) Micro128SharedStorage {
  cute::array_aligned<Element, 2 * cute::cosize_v<PrefillSmemLayoutQ>> q;
  cute::array_aligned<Element, cute::cosize_v<PrefillRSSmemLayoutKStages>> k;
  cute::array_aligned<Element, cute::cosize_v<PrefillRSSmemLayoutV>> v;
};
static_assert(sizeof(Micro128SharedStorage) == 640 * kHeadDim,
              "M128 storage must be Q128 + K64x2 + V64, all BF16");

template <class Scores, class Output, class State>
__device__ __forceinline__ void micro128_softmax(
    Scores& scores, Output& output, State& row_max, State& row_sum) {
  Tensor s = make_tensor(
      scores.data(), streamattn_acc_rowcol<false>(scores.layout()));
  Tensor o = make_tensor(
      output.data(), streamattn_acc_rowcol<false>(output.layout()));
  static_assert(decltype(size<0>(s))::value == decltype(size<0>(o))::value);
  CUTE_UNROLL
  for (int row = 0; row < size<0>(s); ++row) {
    Accum tile_max = -INFINITY;
    CUTE_UNROLL
    for (int col = 0; col < size<1>(s); ++col) {
      tile_max = fmaxf(tile_max, s(row, col));
    }
    const Accum next_max = fmaxf(row_max(row), streamattn_quad_max(tile_max));
    const Accum alpha = row_max(row) == -INFINITY ? 0.0f
        : exp2f((row_max(row) - next_max) * kMicro128ScaleLog2);
    row_max(row) = next_max;
    row_sum(row) *= alpha;
    CUTE_UNROLL
    for (int col = 0; col < size<1>(o); ++col) {
      o(row, col) *= alpha;
    }
    Accum local_sum = 0.0f;
    CUTE_UNROLL
    for (int col = 0; col < size<1>(s); ++col) {
      const Accum p = exp2f((s(row, col) - next_max) * kMicro128ScaleLog2);
      s(row, col) = p;
      local_sum += p;
    }
    row_sum(row) += streamattn_quad_sum(local_sum);
  }
}

template <class Scores, class Probability>
__device__ __forceinline__ void micro128_pack_p(
    Scores const& scores, Probability& probability) {
  Tensor source = make_tensor(
      scores.data(),
      streamattn_convert_layout_acc_aregs<PrefillRSTiledMmaPV>(scores.layout()));
  static_assert(decltype(size(source))::value == decltype(size(probability))::value);
  // Convert into one persistent register fragment, without a second FP32 array.
  CUTE_UNROLL
  for (int i = 0; i < size(probability); ++i) {
    probability(i) = Element(source(i));
  }
}

template <int Half, int GroupSize, bool Direct, class Output, class State, class Coords>
__device__ __forceinline__ void micro128_store(
    Output const& acc, State const& row_max, State const& row_sum,
    Coords const& coords, Accum* partial_o, Accum* partial_lse,
    Element* output, Accum* lse, int work, int batch,
    int query_begin, int query_length, int q_heads, int kv_head) {
  Tensor rows = make_tensor(
      acc.data(), streamattn_acc_rowcol<false>(acc.layout()));
  CUTE_UNROLL
  for (int row = 0; row < size<0>(rows); ++row) {
    CUTE_UNROLL
    for (int col = 0; col < size<1>(rows); ++col) {
      const auto coordinate = coords(row, col);
      const int packed_row = Half * 64 + int(get<0>(coordinate));
      const int dim = int(get<1>(coordinate));
      const int position = query_begin + packed_row / GroupSize;
      const bool valid = position < query_length && row_sum(row) > 0.0f;
      const Accum result = valid ? rows(row, col) / row_sum(row) : 0.0f;
      if constexpr (Direct) {
        if (position < query_length) {
          const int head = kv_head * GroupSize + packed_row % GroupSize;
          const int64_t destination =
              (static_cast<int64_t>(batch) * query_length + position) * q_heads + head;
          output[destination * kHeadDim + dim] = Element(result);
          if (dim == 0) {
            lse[destination] = valid
                ? (row_max(row) * kMicro128ScaleLog2 + log2f(row_sum(row))) * kMicro128Ln2
                : -INFINITY;
          }
        }
      } else {
        const int64_t state_row = static_cast<int64_t>(work) * kMicro128Rows + packed_row;
        partial_o[state_row * kHeadDim + dim] = result;
        if (dim == 0) {
          partial_lse[state_row] = valid
              ? row_max(row) * kMicro128ScaleLog2 + log2f(row_sum(row)) : -INFINITY;
        }
      }
    }
  }
}

template <int GroupSize, bool Overlap, bool Direct, bool DrainLoop = false>
__global__ __launch_bounds__(128)
void streamattn_micro128_kernel(
    const Element* __restrict__ query, const Element* __restrict__ key,
    const Element* __restrict__ value, Accum* __restrict__ partial_o,
    Accum* __restrict__ partial_lse, Element* __restrict__ output,
    Accum* __restrict__ lse, int query_length, int kv_length,
    int q_heads, int kv_heads, int num_splits) {
  constexpr int kPositions = kMicro128Rows / GroupSize;
  const int query_tiles = (query_length + kPositions - 1) / kPositions;
  const int work = blockIdx.x;
  const int work_group = work / num_splits;
  const int split = work % num_splits;
  const int batch_kv = work_group / query_tiles;
  const int batch = batch_kv / kv_heads;
  const int kv_head = batch_kv % kv_heads;
  const int query_begin = (work_group % query_tiles) * kPositions;
  const int num_tiles = kv_length / 64;
  const int tile_begin = static_cast<int>(static_cast<int64_t>(split) * num_tiles / num_splits);
  const int tile_end = static_cast<int>(static_cast<int64_t>(split + 1) * num_tiles / num_splits);

  extern __shared__ __align__(128) unsigned char shared_bytes[];
  auto& storage = *reinterpret_cast<Micro128SharedStorage*>(shared_bytes);
  Tensor sQA = make_tensor(make_smem_ptr(storage.q.data()), PrefillSmemLayoutQ{});
  Tensor sQB = make_tensor(
      make_smem_ptr(storage.q.data() + cute::cosize_v<PrefillSmemLayoutQ>),
      PrefillSmemLayoutQ{});
  Tensor sK0 = make_tensor(make_smem_ptr(storage.k.data()), SmemLayoutK{});
  Tensor sK1 = make_tensor(
      make_smem_ptr(storage.k.data() + cute::cosize_v<SmemLayoutK>), SmemLayoutK{});
  Tensor sV = make_tensor(make_smem_ptr(storage.v.data()), PrefillRSSmemLayoutV{});
  CUTE_UNROLL
  for (int half = 0; half < 2; ++half) {
    for (int idx = threadIdx.x; idx < 64 * kHeadDim; idx += 128) {
      const int row = idx / kHeadDim;
      const int dim = idx % kHeadDim;
      const int packed_row = half * 64 + row;
      const int position = query_begin + packed_row / GroupSize;
      Element item = Element(0.0f);
      if (position < query_length) {
        const int head = kv_head * GroupSize + packed_row % GroupSize;
        const int64_t source =
            ((static_cast<int64_t>(batch) * query_length + position) * q_heads + head)
            * kHeadDim + dim;
        item = query[source];
      }
      if (half == 0) { sQA(row, dim) = item; }
      else { sQB(row, dim) = item; }
    }
  }
  cutlass::arch::fence_view_async_shared();
  __syncthreads();

  PrefillTiledMma tiled_qk;
  auto thread_qk = tiled_qk.get_thread_slice(threadIdx.x);
  Tensor rQA = thread_qk.partition_fragment_A(sQA);
  Tensor rQB = thread_qk.partition_fragment_A(sQB);
  Tensor rK0 = thread_qk.partition_fragment_B(sK0);
  Tensor rK1 = thread_qk.partition_fragment_B(sK1);
  Tensor scores_a = partition_fragment_C(tiled_qk, Shape<_64, _64>{});
  Tensor scores_b = partition_fragment_C(tiled_qk, Shape<_64, _64>{});
  Tensor p_regs = make_tensor<Element>(
      streamattn_convert_layout_acc_aregs<PrefillRSTiledMmaPV>(scores_a.layout()));
  clear(p_regs);

  PrefillRSTiledMmaPV tiled_pv;
  auto thread_pv = tiled_pv.get_thread_slice(threadIdx.x);
  Tensor rV = thread_pv.partition_fragment_B(sV);
  Tensor output_a = partition_fragment_C(tiled_pv, Shape<_64, Int<kHeadDim>>{});
  Tensor output_b = partition_fragment_C(tiled_pv, Shape<_64, Int<kHeadDim>>{});
  clear(output_a);
  clear(output_b);
  Tensor output_rows = make_tensor(
      output_a.data(), streamattn_acc_rowcol<false>(output_a.layout()));
  constexpr int kRowsPerThread = decltype(size<0>(output_rows))::value;
  Tensor max_a = make_tensor<Accum>(Shape<Int<kRowsPerThread>>{});
  Tensor max_b = make_tensor<Accum>(Shape<Int<kRowsPerThread>>{});
  Tensor sum_a = make_tensor<Accum>(Shape<Int<kRowsPerThread>>{});
  Tensor sum_b = make_tensor<Accum>(Shape<Int<kRowsPerThread>>{});
  fill(max_a, -INFINITY);
  fill(max_b, -INFINITY);
  clear(sum_a);
  clear(sum_b);

  GmemCopyK copy_k;
  auto thread_copy_k = copy_k.get_thread_slice(threadIdx.x);
  Tensor dst_k0 = thread_copy_k.partition_D(sK0);
  Tensor dst_k1 = thread_copy_k.partition_D(sK1);
  GmemCopyPrefillRSV copy_v;
  auto thread_copy_v = copy_v.get_thread_slice(threadIdx.x);
  Tensor dst_v = thread_copy_v.partition_D(sV);
  auto load_k = [&](int tile, auto destination) {
    const Element* source = key
        + (static_cast<int64_t>(batch_kv) * kv_length + tile * 64) * kHeadDim;
    Tensor global = make_tensor(make_gmem_ptr(source), Shape<_64, Int<kHeadDim>>{},
                                make_stride(Int<kHeadDim>{}, _1{}));
    cute::copy(copy_k, thread_copy_k.partition_S(global), destination);
  };
  auto load_v = [&](int tile) {
    const Element* source = value
        + (static_cast<int64_t>(batch_kv) * kv_length + tile * 64) * kHeadDim;
    Tensor global = make_tensor(make_gmem_ptr(source), Shape<Int<kHeadDim>, _64>{},
                                make_stride(_1{}, Int<kHeadDim>{}));
    cute::copy(copy_v, thread_copy_v.partition_S(global), dst_v);
  };
  load_k(tile_begin, dst_k0);
  cute::cp_async_fence();
  cute::cp_async_wait<0>();
  __syncthreads();

  int read_pipe = 0;
  for (int tile = tile_begin; tile < tile_end; ++tile) {
    if (tile + 1 < tile_end) {
      if (read_pipe == 0) { load_k(tile + 1, dst_k1); }
      else { load_k(tile + 1, dst_k0); }
      cute::cp_async_fence();
    }
    clear(scores_a);
    clear(scores_b);
    warpgroup_fence_operand(scores_a);
    warpgroup_fence_operand(scores_b);
    warpgroup_arrive();
    if (read_pipe == 0) { cute::gemm(tiled_qk, rQA, rK0, scores_a); }
    else { cute::gemm(tiled_qk, rQA, rK1, scores_a); }
    warpgroup_commit_batch();
    if constexpr (!Overlap) { warpgroup_wait<0>(); }
    if (read_pipe == 0) { cute::gemm(tiled_qk, rQB, rK0, scores_b); }
    else { cute::gemm(tiled_qk, rQB, rK1, scores_b); }
    warpgroup_commit_batch();
    if constexpr (Overlap) { warpgroup_wait<1>(); }
    else { warpgroup_wait<0>(); }
    // QK(A) and previous PV(B) are retired; QK(B) may remain outstanding.
    warpgroup_fence_operand(scores_a);
    warpgroup_fence_operand(output_b);
    warpgroup_fence_operand(p_regs);
    load_v(tile);
    cute::cp_async_fence();
    micro128_softmax(scores_a, output_a, max_a, sum_a);
    micro128_pack_p(scores_a, p_regs);
    cute::cp_async_wait<0>();
    __syncthreads();
    warpgroup_fence_operand(p_regs);
    warpgroup_fence_operand(output_a);
    warpgroup_arrive();
    cute::gemm(tiled_pv, p_regs, rV, output_a);
    warpgroup_commit_batch();
    if constexpr (Overlap) { warpgroup_wait<1>(); }
    else { warpgroup_wait<0>(); }
    // QK(B) is retired; PV(A) can overlap B's independent softmax/rescale.
    warpgroup_fence_operand(scores_b);
    micro128_softmax(scores_b, output_b, max_b, sum_b);
    warpgroup_wait<0>();
    warpgroup_fence_operand(output_a);
    warpgroup_fence_operand(p_regs);
    micro128_pack_p(scores_b, p_regs);
    warpgroup_fence_operand(p_regs);
    warpgroup_fence_operand(output_b);
    warpgroup_arrive();
    cute::gemm(tiled_pv, p_regs, rV, output_b);
    warpgroup_commit_batch();
    if constexpr (!Overlap) { warpgroup_wait<0>(); }
    if constexpr (DrainLoop) {
      // Retire the last PV(B) at the footer, preserving intra-iteration overlap.
      warpgroup_wait<0>();
      warpgroup_fence_operand(p_regs);
      warpgroup_fence_operand(output_a);
      warpgroup_fence_operand(output_b);
    }
    read_pipe ^= 1;
  }
  warpgroup_wait<0>();
  warpgroup_fence_operand(p_regs);
  warpgroup_fence_operand(output_a);
  warpgroup_fence_operand(output_b);

  Tensor identity = make_identity_tensor(Shape<_64, Int<kHeadDim>>{});
  Tensor coordinates = thread_pv.partition_C(identity);
  Tensor coords = make_tensor(
      coordinates.data(), streamattn_acc_rowcol<false>(coordinates.layout()));
  micro128_store<0, GroupSize, Direct>(
      output_a, max_a, sum_a, coords, partial_o, partial_lse, output, lse,
      work, batch, query_begin, query_length, q_heads, kv_head);
  micro128_store<1, GroupSize, Direct>(
      output_b, max_b, sum_b, coords, partial_o, partial_lse, output, lse,
      work, batch, query_begin, query_length, q_heads, kv_head);
}

__global__ __launch_bounds__(128)
void streamattn_micro128_merge_kernel(
    const Accum* __restrict__ partial_o, const Accum* __restrict__ partial_lse,
    Element* __restrict__ output, Accum* __restrict__ lse,
    int query_length, int q_heads, int kv_heads, int group_size, int num_splits) {
  const int output_row = blockIdx.x;
  const int head = output_row % q_heads;
  const int batch_query = output_row / q_heads;
  const int position = batch_query % query_length;
  const int batch = batch_query / query_length;
  const int positions_per_tile = kMicro128Rows / group_size;
  const int query_tiles = (query_length + positions_per_tile - 1) / positions_per_tile;
  const int work_group = (batch * kv_heads + head / group_size) * query_tiles
      + position / positions_per_tile;
  const int row = (position % positions_per_tile) * group_size + head % group_size;
  const int lane = threadIdx.x & 31;
  __shared__ Accum weights[512];
  __shared__ Accum max_lse;
  __shared__ Accum normalizer;
  if (threadIdx.x < 32) {
    Accum m = -INFINITY;
    for (int split = lane; split < num_splits; split += 32) {
      const int64_t index =
          (static_cast<int64_t>(work_group) * num_splits + split) * kMicro128Rows + row;
      m = fmaxf(m, partial_lse[index]);
    }
    CUTE_UNROLL
    for (int offset = 16; offset > 0; offset >>= 1) {
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, offset));
    }
    if (lane == 0) { max_lse = m; }
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    Accum sum = 0.0f;
    for (int split = lane; split < num_splits; split += 32) {
      const int64_t index =
          (static_cast<int64_t>(work_group) * num_splits + split) * kMicro128Rows + row;
      const Accum weight = max_lse == -INFINITY ? 0.0f
          : exp2f(partial_lse[index] - max_lse);
      weights[split] = weight;
      sum += weight;
    }
    CUTE_UNROLL
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) {
      normalizer = sum;
      lse[output_row] = sum > 0.0f ? (max_lse + log2f(sum)) * kMicro128Ln2 : -INFINITY;
    }
  }
  __syncthreads();
  const int dim = threadIdx.x;
  if (dim < kHeadDim) {
    Accum result = 0.0f;
    for (int split = 0; split < num_splits; ++split) {
      const int64_t index =
          (static_cast<int64_t>(work_group) * num_splits + split) * kMicro128Rows + row;
      result += weights[split] * partial_o[index * kHeadDim + dim];
    }
    output[static_cast<int64_t>(output_row) * kHeadDim + dim] =
        Element(normalizer > 0.0f ? result / normalizer : 0.0f);
  }
}

template <int GroupSize, bool Overlap, bool Direct, bool DrainLoop = false>
static void micro128_launch(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor po, torch::Tensor pl, torch::Tensor o, torch::Tensor lse,
    int splits, int component, cudaStream_t stream) {
  const int query_tiles = (q.size(1) + kMicro128Rows / GroupSize - 1)
      / (kMicro128Rows / GroupSize);
  if (component != 2) {
    streamattn_micro128_kernel<GroupSize, Overlap, Direct, DrainLoop><<<
        q.size(0) * k.size(1) * query_tiles * splits, 128,
        sizeof(Micro128SharedStorage), stream>>>(
        reinterpret_cast<const Element*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const Element*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const Element*>(v.data_ptr<at::BFloat16>()),
        po.data_ptr<float>(), pl.data_ptr<float>(),
        reinterpret_cast<Element*>(o.data_ptr<at::BFloat16>()), lse.data_ptr<float>(),
        q.size(1), k.size(2), q.size(2), k.size(1), splits);
  }
  if constexpr (!Direct) {
    if (component != 1) {
      streamattn_micro128_merge_kernel<<<q.size(0) * q.size(1) * q.size(2), 128, 0, stream>>>(
          po.data_ptr<float>(), pl.data_ptr<float>(),
          reinterpret_cast<Element*>(o.data_ptr<at::BFloat16>()), lse.data_ptr<float>(),
          q.size(1), q.size(2), k.size(1), GroupSize, splits);
    }
  }
}

template <int GroupSize>
static void micro128_dispatch(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor po, torch::Tensor pl, torch::Tensor o, torch::Tensor lse,
    int splits, int component, int protocol, bool direct, cudaStream_t stream) {
  if (protocol == 0) {
    if (direct) { micro128_launch<GroupSize, true, true>(q,k,v,po,pl,o,lse,splits,component,stream); }
    else { micro128_launch<GroupSize, true, false>(q,k,v,po,pl,o,lse,splits,component,stream); }
  } else if (protocol == 2) {
    if (direct) { micro128_launch<GroupSize, true, true, true>(q,k,v,po,pl,o,lse,splits,component,stream); }
    else { micro128_launch<GroupSize, true, false, true>(q,k,v,po,pl,o,lse,splits,component,stream); }
  } else {
    if (direct) { micro128_launch<GroupSize, false, true>(q,k,v,po,pl,o,lse,splits,component,stream); }
    else { micro128_launch<GroupSize, false, false>(q,k,v,po,pl,o,lse,splits,component,stream); }
  }
}

void streamattn_micro128_out_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor po, torch::Tensor pl, torch::Tensor o, torch::Tensor lse,
    int64_t splits, int64_t component, int64_t protocol, bool direct) {
  TORCH_CHECK(q.is_cuda(), "query must be CUDA");
  const c10::cuda::CUDAGuard guard(q.device());
  for (const auto& t : {q, k, v, po, pl, o, lse}) {
    TORCH_CHECK(t.device() == q.device() && t.is_contiguous(),
                "buffers must be contiguous on the query device");
  }
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && k.sizes() == v.sizes(), "invalid Q/K/V ranks or shape");
  TORCH_CHECK(q.size(0) > 0 && q.size(0) == k.size(0) && q.size(1) >= 2 && q.size(1) <= 64,
              "require B>0 and M in [2,64]");
  TORCH_CHECK(q.size(3) == kHeadDim && k.size(3) == kHeadDim, "head dimension specialization mismatch");
  TORCH_CHECK(k.size(1) > 0 && q.size(2) % k.size(1) == 0, "invalid GQA");
  const int64_t group = q.size(2) / k.size(1);
  TORCH_CHECK(group == 4 || group == 8, "G must be 4 or 8");
  TORCH_CHECK(k.size(2) > 0 && k.size(2) % 64 == 0 && k.size(2) <= INT_MAX,
              "KV length must be a positive multiple of 64 within int32");
  TORCH_CHECK(splits > 0 && splits <= k.size(2) / 64 && splits <= 512, "invalid split count");
  TORCH_CHECK(protocol >= 0 && protocol <= 2, "invalid protocol");
  TORCH_CHECK(component >= 0 && component <= 2, "invalid component");
  TORCH_CHECK(!direct || (splits == 1 && component != 2), "direct mode requires S1 and has no merge");
  for (const auto& t : {q, k, v, o}) {
    TORCH_CHECK(t.scalar_type() == at::ScalarType::BFloat16, "Q/K/V/output must be BF16");
  }
  for (const auto& t : {po, pl, lse}) {
    TORCH_CHECK(t.scalar_type() == at::ScalarType::Float, "state/LSE must be FP32");
  }
  TORCH_CHECK(o.sizes() == q.sizes(), "output shape mismatch");
  TORCH_CHECK(lse.sizes() == torch::IntArrayRef({q.size(0), q.size(1), q.size(2)}), "LSE shape mismatch");
  const int64_t query_tiles = (q.size(1) + kMicro128Rows / group - 1) / (kMicro128Rows / group);
  const int64_t groups = q.size(0) * k.size(1) * query_tiles;
  TORCH_CHECK(groups * splits <= INT_MAX && q.size(0) * q.size(1) * q.size(2) <= INT_MAX,
              "grid extent exceeds int32");
  if (direct) {
    TORCH_CHECK(po.numel() == 0 && pl.numel() == 0, "direct mode requires empty partial buffers");
  } else {
    TORCH_CHECK(po.sizes() == torch::IntArrayRef({groups, splits, int64_t(kMicro128Rows), int64_t(kHeadDim)}),
                "partial output shape mismatch");
    TORCH_CHECK(pl.sizes() == torch::IntArrayRef({groups, splits, int64_t(kMicro128Rows)}),
                "partial LSE shape mismatch");
  }
  for (const auto& destination : {po, pl, o, lse}) {
    for (const auto& source : {q, k, v}) { at::assert_no_overlap(destination, source); }
  }
  at::assert_no_overlap(po, pl); at::assert_no_overlap(po, o); at::assert_no_overlap(po, lse);
  at::assert_no_overlap(pl, o); at::assert_no_overlap(pl, lse); at::assert_no_overlap(o, lse);
  const auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
  if (group == 4) { micro128_dispatch<4>(q,k,v,po,pl,o,lse,splits,component,protocol,direct,stream); }
  else { micro128_dispatch<8>(q,k,v,po,pl,o,lse,splits,component,protocol,direct,stream); }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <class Kernel>
static void micro128_append_resources(std::vector<int64_t>& values, Kernel kernel, int shared_bytes) {
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

template <int GroupSize>
static void micro128_resources(std::vector<int64_t>& values, int protocol, bool direct) {
  if (protocol == 0) {
    if (direct) { micro128_append_resources(values, streamattn_micro128_kernel<GroupSize, true, true>, sizeof(Micro128SharedStorage)); }
    else { micro128_append_resources(values, streamattn_micro128_kernel<GroupSize, true, false>, sizeof(Micro128SharedStorage)); }
  } else if (protocol == 2) {
    if (direct) { micro128_append_resources(values, streamattn_micro128_kernel<GroupSize, true, true, true>, sizeof(Micro128SharedStorage)); }
    else { micro128_append_resources(values, streamattn_micro128_kernel<GroupSize, true, false, true>, sizeof(Micro128SharedStorage)); }
  } else {
    if (direct) { micro128_append_resources(values, streamattn_micro128_kernel<GroupSize, false, true>, sizeof(Micro128SharedStorage)); }
    else { micro128_append_resources(values, streamattn_micro128_kernel<GroupSize, false, false>, sizeof(Micro128SharedStorage)); }
  }
}

torch::Tensor streamattn_micro128_resource_info_cuda(
    torch::Tensor q, int64_t group_size, int64_t protocol, bool direct) {
  TORCH_CHECK(q.is_cuda(), "query must be CUDA");
  TORCH_CHECK(group_size == 4 || group_size == 8, "G must be 4 or 8");
  TORCH_CHECK(protocol >= 0 && protocol <= 2, "invalid protocol");
  const c10::cuda::CUDAGuard guard(q.device());
  std::vector<int64_t> values;
  if (group_size == 4) { micro128_resources<4>(values, protocol, direct); }
  else { micro128_resources<8>(values, protocol, direct); }
  micro128_append_resources(values, streamattn_micro128_merge_kernel, 0);
  return torch::tensor(values, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
}
"""


def cuda_source_for_head_dim(head_dim: int) -> str:
    """Compose the shared CuTe definitions with only this candidate's kernels."""
    base = _base_source(head_dim)
    if base.count(_PREFIX_END) != 1:
        raise ValueError(
            "SM90 definitions boundary changed; review M128 source composition"
        )
    prefix = base.split(_PREFIX_END, 1)[0]
    if "__global__" in prefix:
        raise ValueError("definitions prefix unexpectedly contains a CUDA kernel")
    return prefix + _CANARY_SOURCE


CUDA_SOURCE = cuda_source_for_head_dim(64)
