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

void streamattn_transposed_wgmma_qkpv_ws_cp_async_checksum_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor checksums,
    int64_t num_splits,
    int64_t consumer_registers);

torch::Tensor streamattn_transposed_wgmma_qkpv_floor_resource_info_cuda(
    int64_t consumer_registers);

void streamattn_grouped_wgmma_prefill_out_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    torch::Tensor lse);

torch::Tensor streamattn_grouped_wgmma_prefill_resource_info_cuda();

void streamattn_grouped_rs_prefill_out_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    torch::Tensor lse);

torch::Tensor streamattn_grouped_rs_prefill_resource_info_cuda();

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

void streamattn_transposed_wgmma_paged_exact_decode_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor page_table,
    torch::Tensor partial_o,
    torch::Tensor partial_lse,
    torch::Tensor output,
    int64_t num_splits);

void streamattn_transposed_wgmma_paged_fragmented_exact_decode_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor page_table,
    torch::Tensor partial_o,
    torch::Tensor partial_lse,
    torch::Tensor output,
    int64_t num_splits);

void streamattn_transposed_wgmma_paged_fragmented_ragged_exact_decode_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor page_table,
    torch::Tensor sequence_lengths,
    torch::Tensor partial_o,
    torch::Tensor partial_lse,
    torch::Tensor output,
    int64_t num_splits);

void streamattn_transposed_wgmma_paged_fragmented_nhd_exact_decode_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor page_table,
    torch::Tensor partial_o,
    torch::Tensor partial_lse,
    torch::Tensor output,
    int64_t num_splits);

void streamattn_transposed_wgmma_paged_fragmented_nhd_ragged_exact_decode_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor page_table,
    torch::Tensor sequence_lengths,
    torch::Tensor partial_o,
    torch::Tensor partial_lse,
    torch::Tensor output,
    int64_t num_splits);

void streamattn_transposed_wgmma_paged_selected_fragmented_exact_decode_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor route_row_ptr,
    torch::Tensor physical_page_ids,
    torch::Tensor active_head_masks,
    torch::Tensor token_valid_masks,
    torch::Tensor partial_o,
    torch::Tensor partial_lse,
    torch::Tensor output,
    int64_t max_routes_per_row);

void streamattn_transposed_wgmma_paged_selected_fragmented_nhd_exact_decode_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor route_row_ptr,
    torch::Tensor physical_page_ids,
    torch::Tensor active_head_masks,
    torch::Tensor token_valid_masks,
    torch::Tensor partial_o,
    torch::Tensor partial_lse,
    torch::Tensor output,
    int64_t max_routes_per_row);

void streamattn_prepare_qhead_paged_routes64_out_cuda(
    torch::Tensor source_row_ptr,
    torch::Tensor source_atom_ids,
    torch::Tensor page_table,
    torch::Tensor sequence_lengths,
    torch::Tensor route_counts,
    torch::Tensor logical_atom_origins,
    torch::Tensor physical_page_ids,
    torch::Tensor atom_valid_masks,
    torch::Tensor active_head_masks,
    torch::Tensor token_valid_masks,
    torch::Tensor route_flags,
    torch::Tensor route_errors,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t num_pages,
    int64_t max_routes_per_group);

void streamattn_transposed_wgmma_paged_dynamic_qhead_fragmented_exact_decode_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor page_table,
    torch::Tensor sequence_lengths,
    torch::Tensor source_row_ptr,
    torch::Tensor source_atom_ids,
    torch::Tensor route_counts,
    torch::Tensor logical_atom_origins,
    torch::Tensor physical_page_ids,
    torch::Tensor atom_valid_masks,
    torch::Tensor active_head_masks,
    torch::Tensor token_valid_masks,
    torch::Tensor route_flags,
    torch::Tensor route_errors,
    torch::Tensor partial_o,
    torch::Tensor partial_lse,
    torch::Tensor output,
    int64_t max_routes_per_group);

void streamattn_transposed_wgmma_paged_dynamic_qhead_fragmented_nhd_exact_decode_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor page_table,
    torch::Tensor sequence_lengths,
    torch::Tensor source_row_ptr,
    torch::Tensor source_atom_ids,
    torch::Tensor route_counts,
    torch::Tensor logical_atom_origins,
    torch::Tensor physical_page_ids,
    torch::Tensor atom_valid_masks,
    torch::Tensor active_head_masks,
    torch::Tensor token_valid_masks,
    torch::Tensor route_flags,
    torch::Tensor route_errors,
    torch::Tensor partial_o,
    torch::Tensor partial_lse,
    torch::Tensor output,
    int64_t max_routes_per_group);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qk_out", &streamattn_transposed_wgmma_qk_out_cuda,
        "StreamAttn transposed m64n8k16 exact QK (out variant)");
  m.def("qk_checksum_out", &streamattn_transposed_wgmma_qk_checksum_out_cuda,
        "StreamAttn transposed m64n8k16 exact QK (storeless checksum variant)");
  m.def("qk_async_checksum_out", &streamattn_transposed_wgmma_qk_async_checksum_out_cuda,
        "StreamAttn transposed m64n8k16 exact QK (cp.async double-buffer checksum variant)");
  m.def("qkpv_async_checksum_out", &streamattn_transposed_wgmma_qkpv_async_checksum_out_cuda,
        "StreamAttn transposed m64n8k16 QK+PV floor (cp.async checksum variant)");
  m.def("qkpv_ws_cp_async_checksum_out",
        &streamattn_transposed_wgmma_qkpv_ws_cp_async_checksum_out_cuda,
        "StreamAttn transposed QK+PV floor (warp-specialized cp.async)");
  m.def("qkpv_floor_resource_info",
        &streamattn_transposed_wgmma_qkpv_floor_resource_info_cuda,
        "Compiled resources for cooperative and warp-specialized QK+PV floors");
  m.def("grouped_wgmma_prefill_out",
        &streamattn_grouped_wgmma_prefill_out_cuda,
        "StreamAttn natural m64n64 grouped exact causal prefill");
  m.def("grouped_wgmma_prefill_resource_info",
        &streamattn_grouped_wgmma_prefill_resource_info_cuda,
        "Compiled resources for natural m64n64 grouped exact causal prefill");
  m.def("grouped_rs_prefill_out",
        &streamattn_grouped_rs_prefill_out_cuda,
        "StreamAttn consumer-owned cp.async SS-QK/RS-PV exact causal prefill");
  m.def("grouped_rs_prefill_resource_info",
        &streamattn_grouped_rs_prefill_resource_info_cuda,
        "Compiled resources for consumer-owned grouped RS-PV prefill");
  m.def("exact_partial_out", &streamattn_transposed_wgmma_exact_partial_out_cuda,
        "StreamAttn transposed m64n8k16 exact attention partial states");
  m.def("exact_merge_out", &streamattn_transposed_wgmma_exact_merge_out_cuda,
        "StreamAttn exact split-state merge");
  m.def("exact_merge_warp_out", &streamattn_transposed_wgmma_exact_merge_warp_out_cuda,
        "StreamAttn one-warp exact split-state merge");
  m.def("exact_decode_out", &streamattn_transposed_wgmma_exact_decode_out_cuda,
        "StreamAttn exact producer and merge (single host dispatch)");
  m.def("paged_exact_decode_out",
        &streamattn_transposed_wgmma_paged_exact_decode_out_cuda,
        "StreamAttn HND-paged exact producer and merge (single host dispatch)");
  m.def("paged_fragmented_exact_decode_out",
        &streamattn_transposed_wgmma_paged_fragmented_exact_decode_out_cuda,
        "StreamAttn HND page-16 fragmented exact producer and merge");
  m.def("paged_fragmented_ragged_exact_decode_out",
        &streamattn_transposed_wgmma_paged_fragmented_ragged_exact_decode_out_cuda,
        "StreamAttn HND page-16 ragged fragmented exact producer and merge");
  m.def("paged_fragmented_nhd_exact_decode_out",
        &streamattn_transposed_wgmma_paged_fragmented_nhd_exact_decode_out_cuda,
        "StreamAttn direct NHD page-16 fragmented exact producer and merge");
  m.def("paged_fragmented_nhd_ragged_exact_decode_out",
        &streamattn_transposed_wgmma_paged_fragmented_nhd_ragged_exact_decode_out_cuda,
        "StreamAttn direct NHD page-16 ragged fragmented exact producer and merge");
  m.def("paged_selected_fragmented_exact_decode_out",
        &streamattn_transposed_wgmma_paged_selected_fragmented_exact_decode_out_cuda,
        "StreamAttn HND page-16 selected-route producer and static merge");
  m.def("paged_selected_fragmented_nhd_exact_decode_out",
        &streamattn_transposed_wgmma_paged_selected_fragmented_nhd_exact_decode_out_cuda,
        "StreamAttn direct NHD page-16 selected-route producer and static merge");
  m.def("prepare_qhead_paged_routes64_out",
        &streamattn_prepare_qhead_paged_routes64_out_cuda,
        "StreamAttn device-side Q-head CSR to row-local PackedRoute64 lowering");
  m.def("paged_dynamic_qhead_fragmented_exact_decode_out",
        &streamattn_transposed_wgmma_paged_dynamic_qhead_fragmented_exact_decode_out_cuda,
        "StreamAttn HND dynamic Q-head route preparation and selected decode");
  m.def("paged_dynamic_qhead_fragmented_nhd_exact_decode_out",
        &streamattn_transposed_wgmma_paged_dynamic_qhead_fragmented_nhd_exact_decode_out_cuda,
        "StreamAttn direct NHD dynamic Q-head route preparation and selected decode");
}
"""

CUDA_SOURCE = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <climits>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/pipeline/pipeline.hpp>

using namespace cute;

using Element = cutlass::bfloat16_t;
using Accum = float;

static constexpr int kBlockM = 64;
static constexpr int kBlockN = 8;
static constexpr int kHeadDim = 64;
static constexpr int kPipelineStages = 2;
static constexpr int kSeparateVStages = kHeadDim == 64 ? 2 : 0;

using SmemLayoutK = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<Element>{},
    Shape<Int<kBlockM>, Int<kHeadDim>>{}));
// Four physical page-16 fragments alias one logical 64-token WGMMA tile.
// Step<_1,_3,_2> preserves the exact swizzled address mapping used by
// SmemLayoutK while exposing a page-index mode to the copy producer.
using SmemLayoutPaged16 = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<Element>{},
    Shape<Int<16>, Int<kHeadDim>, Int<4>>{},
    Step<_1, _3, _2>{}));
static_assert(cute::cosize_v<SmemLayoutPaged16> == cute::cosize_v<SmemLayoutK>,
              "page-16 and WGMMA K layouts must alias the same shared tile");
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
static constexpr int kPrefillRowsPerWarpGroup = 64;
static constexpr int kPrefillConsumerGroups = 2;
static constexpr int kPrefillRows =
    kPrefillRowsPerWarpGroup * kPrefillConsumerGroups;
using PrefillTiledMma = decltype(make_tiled_mma(
    SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{}));
using PrefillTiledMmaO = decltype(make_tiled_mma(
    SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::MN, GMMA::Major::MN>{}));
using PrefillRSTileShapePV =
    Shape<Int<kPrefillRowsPerWarpGroup>, Int<kHeadDim>, Int<kBlockM>>;
using PrefillRSTiledMmaPV = decltype(make_tiled_mma(
    GMMA::rs_op_selector<
        Element,
        Element,
        Accum,
        PrefillRSTileShapePV,
        GMMA::Major::K,
        GMMA::Major::MN>()));
using SmemLayoutV = SmemLayoutK;
using SmemLayoutVt = decltype(composition(
    SmemLayoutV{},
    make_layout(Shape<Int<kHeadDim>, Int<kBlockM>>{}, GenRowMajor{})));
using PrefillRSSmemLayoutKStages = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<Element>{},
    Shape<Int<kBlockM>, Int<kHeadDim>, _2>{}));
using PrefillRSSmemLayoutV = decltype(tile_to_shape(
    GMMA::Layout_MN_SW128_Atom<Element>{},
    Shape<Int<kHeadDim>, Int<kBlockM>>{}, Step<_2, _1>{}));
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

using PrefillSmemLayoutQ = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<Element>{},
    Shape<Int<kPrefillRowsPerWarpGroup>, Int<kHeadDim>>{}));
using PrefillSmemLayoutP = decltype(tile_to_shape(
    GMMA::Layout_MN_SW128_Atom<Element>{},
    Shape<Int<kPrefillRowsPerWarpGroup>, Int<kBlockM>>{}));
using PrefillSmemLayoutPOrigin = decltype(composition(
    PrefillSmemLayoutP{},
    make_layout(
        Shape<Int<kBlockM>, Int<kPrefillRowsPerWarpGroup>>{}, GenRowMajor{})));

struct alignas(128) SharedStorage {
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> k;
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> q;
};

struct alignas(128) AsyncSharedStorage {
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutK2>> k;
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> q;
};

struct alignas(128) AsyncQKPVSharedStorage {
  cute::array_aligned<
      Element, cute::cosize_v<SmemLayoutK> * kPipelineStages> k;
  cute::array_aligned<
      Element, kSeparateVStages == 2
          ? cute::cosize_v<SmemLayoutV> * kSeparateVStages
          : 1> v;
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> q;
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutPOrigin>> p;
  Accum row_reduce[4][kBlockN];
  Accum row_max[kBlockN];
  Accum row_sum[kBlockN];
};

struct alignas(128) GroupedPrefillSharedStorage {
  cute::array_aligned<
      Element, cute::cosize_v<SmemLayoutK> * kPipelineStages> k;
  cute::array_aligned<
      Element, kSeparateVStages == 2
          ? cute::cosize_v<SmemLayoutV> * kSeparateVStages
          : 1> v;
  cute::array_aligned<
      Element,
      cute::cosize_v<PrefillSmemLayoutQ> * kPrefillConsumerGroups> q;
  cute::array_aligned<
      Element,
      cute::cosize_v<PrefillSmemLayoutPOrigin> * kPrefillConsumerGroups> p;
  Accum row_reduce[kPrefillConsumerGroups][4][kPrefillRowsPerWarpGroup];
  Accum row_max[kPrefillConsumerGroups][kPrefillRowsPerWarpGroup];
  Accum row_sum[kPrefillConsumerGroups][kPrefillRowsPerWarpGroup];
};

struct alignas(128) GroupedRSPrefillSharedStorage {
  cute::array_aligned<Element, cute::cosize_v<PrefillSmemLayoutQ>> q;
  cute::array_aligned<Element, cute::cosize_v<PrefillRSSmemLayoutKStages>> k;
  cute::array_aligned<Element, cute::cosize_v<PrefillRSSmemLayoutV>> v;
};

using WsPipelineK = cutlass::PipelineAsync<kPipelineStages>;
using WsPipelineV = cutlass::PipelineAsync<1>;

struct alignas(128) WsQKPVSharedStorage {
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutK> * 2> k;
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> v;
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> q;
  cute::array_aligned<Element, cute::cosize_v<SmemLayoutPOrigin>> p;
  typename WsPipelineK::SharedStorage pipeline_k;
  typename WsPipelineV::SharedStorage pipeline_v;
  Accum reduction[128];
};

using GmemCopyK = decltype(make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>{},
    Layout<Shape<_16, _8>, Stride<_8, _1>>{},
    Layout<Shape<_1, _8>>{}));
using GmemCopyPrefillRSV = decltype(make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>{},
    Layout<Shape<_8, _16>, Stride<_1, _8>>{},
    Layout<Shape<_8, _1>>{}));

template <typename To, typename Engine, typename Layout>
__forceinline__ __device__ auto streamattn_convert_type(Tensor<Engine, Layout> const& tensor) {
  using From = typename Engine::value_type;
  constexpr int numel = decltype(size(tensor))::value;
  cutlass::NumericArrayConverter<To, From, numel> convert_op;
  auto fragment = convert_op(
      *reinterpret_cast<const cutlass::Array<From, numel>*>(tensor.data()));
  return make_tensor(make_rmem_ptr<To>(&fragment), tensor.layout());
}

template <typename MmaTraits, typename Layout0>
__forceinline__ __device__ auto streamattn_convert_layout_acc_aregs(
    Layout0 layout) {
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

__forceinline__ __device__ Accum streamattn_quad_max(Accum value) {
  value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 2));
  value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 1));
  return value;
}

__forceinline__ __device__ Accum streamattn_quad_sum(Accum value) {
  value += __shfl_xor_sync(0xffffffffu, value, 2);
  value += __shfl_xor_sync(0xffffffffu, value, 1);
  return value;
}

__global__ __launch_bounds__(256, 1)
void streamattn_grouped_wgmma_prefill_kernel(
    const Element* __restrict__ query,
    const Element* __restrict__ key,
    const Element* __restrict__ value,
    Element* __restrict__ output,
    Accum* __restrict__ lse,
    int batch_size,
    int sequence_length,
    int q_heads,
    int kv_heads,
    int group_size) {
  const int query_positions = kPrefillRows / group_size;
  const int query_tiles =
      (sequence_length + query_positions - 1) / query_positions;
  const int work = blockIdx.x;
  const int group = work / query_tiles;
  const int query_tile = work - group * query_tiles;
  if (group >= batch_size * kv_heads) {
    return;
  }
  const int batch = group / kv_heads;
  const int kv_head = group - batch * kv_heads;
  const int query_begin = query_tile * query_positions;
  const int query_end = min(sequence_length, query_begin + query_positions);
  const int key_tiles = (query_end + kBlockM - 1) / kBlockM;
  const int consumer_group = threadIdx.x / 128;
  const int local_thread = threadIdx.x % 128;

  extern __shared__ __align__(128) unsigned char shared_bytes[];
  auto& storage =
      *reinterpret_cast<GroupedPrefillSharedStorage*>(shared_bytes);
  Element* k0_ptr = storage.k.data();
  Element* k1_ptr =
      storage.k.data() + cute::cosize_v<SmemLayoutK>;
  Element* v0_ptr =
      kSeparateVStages == 2 ? storage.v.data() : k0_ptr;
  Element* v1_ptr = kSeparateVStages == 2
      ? storage.v.data() + cute::cosize_v<SmemLayoutV>
      : k1_ptr;
  Tensor sK0 = make_tensor(make_smem_ptr(k0_ptr), SmemLayoutK{});
  Tensor sK1 = make_tensor(make_smem_ptr(k1_ptr), SmemLayoutK{});
  Tensor sV0 = make_tensor(make_smem_ptr(v0_ptr), SmemLayoutV{});
  Tensor sV1 = make_tensor(make_smem_ptr(v1_ptr), SmemLayoutV{});
  Tensor sVt0 = make_tensor(make_smem_ptr(v0_ptr), SmemLayoutVt{});
  Tensor sVt1 = make_tensor(make_smem_ptr(v1_ptr), SmemLayoutVt{});
  Element* q_ptr = storage.q.data()
      + consumer_group * cute::cosize_v<PrefillSmemLayoutQ>;
  Element* p_ptr = storage.p.data()
      + consumer_group * cute::cosize_v<PrefillSmemLayoutPOrigin>;
  Tensor sQ = make_tensor(make_smem_ptr(q_ptr), PrefillSmemLayoutQ{});
  Tensor sPOrigin = make_tensor(
      make_smem_ptr(p_ptr), PrefillSmemLayoutPOrigin{});
  Tensor sP = make_tensor(make_smem_ptr(p_ptr), PrefillSmemLayoutP{});

  for (int idx = local_thread;
       idx < kPrefillRowsPerWarpGroup * kHeadDim;
       idx += 128) {
    const int local_query_row = idx / kHeadDim;
    const int dim = idx - local_query_row * kHeadDim;
    const int global_query_row =
        consumer_group * kPrefillRowsPerWarpGroup + local_query_row;
    const int query_offset = global_query_row / group_size;
    const int head_offset = global_query_row - query_offset * group_size;
    const int query_position = query_begin + query_offset;
    const int q_head = kv_head * group_size + head_offset;
    Element item = Element(0.0f);
    if (query_position < sequence_length) {
      const int64_t source =
          ((static_cast<int64_t>(batch) * sequence_length + query_position)
               * q_heads
           + q_head)
              * kHeadDim
          + dim;
      item = query[source];
    }
    sQ(local_query_row, dim) = item;
  }
  cutlass::arch::fence_view_async_shared();
  __syncthreads();

  PrefillTiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(local_thread);
  Tensor tSrK0 = thr_mma.partition_fragment_A(sK0);
  Tensor tSrK1 = thr_mma.partition_fragment_A(sK1);
  Tensor tSrQ = thr_mma.partition_fragment_B(sQ);
  Tensor tPsP = thr_mma.partition_C(sPOrigin);
  Tensor cS = make_identity_tensor(
      Shape<Int<kBlockM>, Int<kPrefillRowsPerWarpGroup>>{});
  Tensor tScS = thr_mma.partition_C(cS);
  Tensor tScSRowCol = make_tensor(
      tScS.data(), streamattn_acc_rowcol<true>(tScS.layout()));

  PrefillTiledMmaO tiled_mma_o;
  auto thr_mma_o = tiled_mma_o.get_thread_slice(local_thread);
  Tensor tOrV0 = thr_mma_o.partition_fragment_A(sVt0);
  Tensor tOrV1 = thr_mma_o.partition_fragment_A(sVt1);
  Tensor tOrP = thr_mma_o.partition_fragment_B(sP);
  Tensor tOrO = partition_fragment_C(
      tiled_mma_o,
      Shape<Int<kHeadDim>, Int<kPrefillRowsPerWarpGroup>>{});
  Tensor tOrORowCol = make_tensor(
      tOrO.data(), streamattn_acc_rowcol<true>(tOrO.layout()));
  Tensor cO = make_identity_tensor(
      Shape<Int<kHeadDim>, Int<kPrefillRowsPerWarpGroup>>{});
  Tensor tOcO = thr_mma_o.partition_C(cO);
  clear(tOrO);

  constexpr int kRowsPerThread = decltype(size<0>(tOrORowCol))::value;
  static_assert(
      kRowsPerThread == decltype(size<0>(tScSRowCol))::value,
      "QK and PV fragments must expose the same query rows");
  Accum row_max[kRowsPerThread];
  Accum row_sum[kRowsPerThread];
  CUTE_UNROLL
  for (int row = 0; row < kRowsPerThread; ++row) {
    row_max[row] = -INFINITY;
    row_sum[row] = 0.0f;
  }

  GmemCopyK copy_kv;
  auto thr_copy_kv = copy_kv.get_thread_slice(local_thread);
  Tensor tK0sK0 = thr_copy_kv.partition_D(sK0);
  Tensor tK1sK1 = thr_copy_kv.partition_D(sK1);
  Tensor tV0sV0 = thr_copy_kv.partition_D(sV0);
  Tensor tV1sV1 = thr_copy_kv.partition_D(sV1);

  auto copy_tile = [&](const Element* source, int tile, auto destination) {
    Tensor global = make_tensor(
        make_gmem_ptr(source
            + ((static_cast<int64_t>(batch) * sequence_length
                + tile * kBlockM)
                   * kv_heads
               + kv_head)
                * kHeadDim),
        Shape<Int<kBlockM>, Int<kHeadDim>>{},
        make_stride(kv_heads * kHeadDim, _1{}));
    cute::copy(copy_kv, thr_copy_kv.partition_S(global), destination);
  };

  if (key_tiles > 0) {
    if (consumer_group == 0) {
      copy_tile(key, 0, tK0sK0);
      if constexpr (kSeparateVStages == 2) {
        copy_tile(value, 0, tV0sV0);
      }
      cute::cp_async_fence();
      cute::cp_async_wait<0>();
    }
    __syncthreads();
  }

  constexpr Accum kSoftmaxScaleLog2 = kHeadDim == 64
      ? 0.18033688011112042f
      : 0.12751743082459868f;
  int read_pipe = 0;
  for (int tile = 0; tile < key_tiles; ++tile) {
    const int next_tile = tile + 1;
    const int write_pipe = read_pipe ^ 1;
    if (consumer_group == 0 && next_tile < key_tiles) {
      if (write_pipe == 0) {
        copy_tile(key, next_tile, tK0sK0);
        if constexpr (kSeparateVStages == 2) {
          copy_tile(value, next_tile, tV0sV0);
        }
      } else {
        copy_tile(key, next_tile, tK1sK1);
        if constexpr (kSeparateVStages == 2) {
          copy_tile(value, next_tile, tV1sV1);
        }
      }
      cute::cp_async_fence();
    }

    Tensor tCrS = partition_fragment_C(
        tiled_mma,
        Shape<Int<kBlockM>, Int<kPrefillRowsPerWarpGroup>>{});
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

    if constexpr (kSeparateVStages == 0) {
      if (consumer_group == 0) {
        if (read_pipe == 0) {
          copy_tile(value, tile, tV0sV0);
        } else {
          copy_tile(value, tile, tV1sV1);
        }
        cute::cp_async_fence();
      }
    }

    Tensor scores = make_tensor(
        tCrS.data(), streamattn_acc_rowcol<true>(tCrS.layout()));
    CUTE_UNROLL
    for (int row = 0; row < size<0>(scores); ++row) {
      CUTE_UNROLL
      for (int col = 0; col < size<1>(scores); ++col) {
        const int token = int(get<0>(tScSRowCol(row, col)));
        const int local_query_row = int(get<1>(tScSRowCol(row, col)));
        const int global_query_row =
            consumer_group * kPrefillRowsPerWarpGroup + local_query_row;
        const int query_position = query_begin + global_query_row / group_size;
        const int key_position = tile * kBlockM + token;
        if (query_position >= sequence_length ||
            key_position > query_position ||
            key_position >= sequence_length) {
          scores(row, col) = -INFINITY;
        }
      }
    }

    const int lane = local_thread & 31;
    const int warp = local_thread >> 5;
    const int group_row = lane >> 2;
    CUTE_UNROLL
    for (int row = 0; row < kRowsPerThread; ++row) {
      Accum tile_max = -INFINITY;
      CUTE_UNROLL
      for (int col = 0; col < size<1>(scores); ++col) {
        tile_max = fmaxf(tile_max, scores(row, col));
      }
      tile_max = streamattn_group_max(tile_max);
      const int query_row = int(get<1>(tScSRowCol(row, 0)));
      if (group_row == 0) {
        storage.row_reduce[consumer_group][warp][query_row] = tile_max;
      }
    }
    __syncthreads();

    CUTE_UNROLL
    for (int row = 0; row < kRowsPerThread; ++row) {
      const int query_row = int(get<1>(tScSRowCol(row, 0)));
      Accum tile_max = storage.row_reduce[consumer_group][0][query_row];
      CUTE_UNROLL
      for (int warp_index = 1; warp_index < 4; ++warp_index) {
        tile_max = fmaxf(
            tile_max,
            storage.row_reduce[consumer_group][warp_index][query_row]);
      }
      const Accum next_max = fmaxf(row_max[row], tile_max);
      const Accum alpha = row_max[row] == -INFINITY
          ? 0.0f
          : exp2f((row_max[row] - next_max) * kSoftmaxScaleLog2);
      row_max[row] = next_max;
      row_sum[row] *= alpha;

      Accum local_sum = 0.0f;
      const Accum max_scaled = next_max * kSoftmaxScaleLog2;
      CUTE_UNROLL
      for (int col = 0; col < size<1>(scores); ++col) {
        const Accum probability = next_max == -INFINITY
            ? 0.0f
            : exp2f(scores(row, col) * kSoftmaxScaleLog2 - max_scaled);
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

    if constexpr (kSeparateVStages == 0) {
      if (consumer_group == 0) {
        cute::cp_async_wait<0>();
      }
      __syncthreads();
    }

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

    if (next_tile < key_tiles) {
      if (consumer_group == 0) {
        cute::cp_async_wait<0>();
      }
      __syncthreads();
      read_pipe = write_pipe;
    }
  }

  const int lane = local_thread & 31;
  const int warp = local_thread >> 5;
  const int group_row = lane >> 2;
  CUTE_UNROLL
  for (int row = 0; row < kRowsPerThread; ++row) {
    const int query_row = int(get<1>(tScSRowCol(row, 0)));
    const Accum total = streamattn_group_sum(row_sum[row]);
    if (group_row == 0) {
      storage.row_reduce[consumer_group][warp][query_row] = total;
    }
  }
  __syncthreads();
  if (warp == 0 && group_row == 0) {
    CUTE_UNROLL
    for (int row = 0; row < kRowsPerThread; ++row) {
      const int query_row = int(get<1>(tScSRowCol(row, 0)));
      Accum total = 0.0f;
      CUTE_UNROLL
      for (int warp_index = 0; warp_index < 4; ++warp_index) {
        total += storage.row_reduce[consumer_group][warp_index][query_row];
      }
      storage.row_max[consumer_group][query_row] = row_max[row];
      storage.row_sum[consumer_group][query_row] = total;
      const int global_query_row =
          consumer_group * kPrefillRowsPerWarpGroup + query_row;
      const int query_offset = global_query_row / group_size;
      const int head_offset = global_query_row - query_offset * group_size;
      const int query_position = query_begin + query_offset;
      if (query_position < sequence_length) {
        const int q_head = kv_head * group_size + head_offset;
        constexpr Accum kLn2 = 0.6931471805599453f;
        lse[(static_cast<int64_t>(batch) * sequence_length + query_position)
                * q_heads
            + q_head] = total > 0.0f
            ? (row_max[row] * kSoftmaxScaleLog2 + log2f(total)) * kLn2
            : -INFINITY;
      }
    }
  }
  __syncthreads();

  Tensor tOcORowCol = make_tensor(
      tOcO.data(), streamattn_acc_rowcol<true>(tOcO.layout()));
  CUTE_UNROLL
  for (int row = 0; row < size<0>(tOrORowCol); ++row) {
    CUTE_UNROLL
    for (int col = 0; col < size<1>(tOrORowCol); ++col) {
      const auto coordinate = tOcORowCol(row, col);
      const int dim = int(get<0>(coordinate));
      const int query_row = int(get<1>(coordinate));
      const int global_query_row =
          consumer_group * kPrefillRowsPerWarpGroup + query_row;
      const int query_offset = global_query_row / group_size;
      const int head_offset = global_query_row - query_offset * group_size;
      const int query_position = query_begin + query_offset;
      if (query_position < sequence_length) {
        const int q_head = kv_head * group_size + head_offset;
        const int64_t destination =
            ((static_cast<int64_t>(batch) * sequence_length + query_position)
                 * q_heads
             + q_head)
                * kHeadDim
            + dim;
        output[destination] = storage.row_sum[consumer_group][query_row] > 0.0f
            ? Element(
                tOrORowCol(row, col)
                / storage.row_sum[consumer_group][query_row])
            : Element(0.0f);
      }
    }
  }
}

__global__ __launch_bounds__(128)
void streamattn_grouped_rs_prefill_kernel(
    const Element* __restrict__ query,
    const Element* __restrict__ key,
    const Element* __restrict__ value,
    Element* __restrict__ output,
    Accum* __restrict__ lse,
    int batch_size,
    int sequence_length,
    int q_heads,
    int kv_heads,
    int group_size) {
  const int query_positions = kPrefillRowsPerWarpGroup / group_size;
  const int query_tiles =
      (sequence_length + query_positions - 1) / query_positions;
  const int work = blockIdx.x;
  const int work_group = work / query_tiles;
  const int query_tile = work - work_group * query_tiles;
  if (work_group >= batch_size * kv_heads) {
    return;
  }
  const int batch = work_group / kv_heads;
  const int kv_head = work_group - batch * kv_heads;
  const int query_begin = query_tile * query_positions;
  const int query_end = min(sequence_length, query_begin + query_positions);
  const int key_tiles = (query_end + kBlockM - 1) / kBlockM;

  extern __shared__ __align__(128) unsigned char shared_bytes[];
  auto& storage =
      *reinterpret_cast<GroupedRSPrefillSharedStorage*>(shared_bytes);
  Element* k0_ptr = storage.k.data();
  Element* k1_ptr = storage.k.data() + cute::cosize_v<SmemLayoutK>;
  Tensor sQ = make_tensor(
      make_smem_ptr(storage.q.data()), PrefillSmemLayoutQ{});
  Tensor sK0 = make_tensor(make_smem_ptr(k0_ptr), SmemLayoutK{});
  Tensor sK1 = make_tensor(make_smem_ptr(k1_ptr), SmemLayoutK{});
  Tensor sV = make_tensor(
      make_smem_ptr(storage.v.data()), PrefillRSSmemLayoutV{});

  for (int idx = threadIdx.x;
       idx < kPrefillRowsPerWarpGroup * kHeadDim;
       idx += 128) {
    const int local_query_row = idx / kHeadDim;
    const int dim = idx - local_query_row * kHeadDim;
    const int query_offset = local_query_row / group_size;
    const int head_offset = local_query_row - query_offset * group_size;
    const int query_position = query_begin + query_offset;
    const int q_head = kv_head * group_size + head_offset;
    Element item = Element(0.0f);
    if (query_position < sequence_length) {
      const int64_t source =
          ((static_cast<int64_t>(batch) * sequence_length + query_position)
               * q_heads
           + q_head)
              * kHeadDim
          + dim;
      item = query[source];
    }
    sQ(local_query_row, dim) = item;
  }
  cutlass::arch::fence_view_async_shared();
  __syncthreads();

  PrefillTiledMma tiled_qk;
  auto thread_qk = tiled_qk.get_thread_slice(threadIdx.x);
  Tensor rQ = thread_qk.partition_fragment_A(sQ);
  Tensor rK0 = thread_qk.partition_fragment_B(sK0);
  Tensor rK1 = thread_qk.partition_fragment_B(sK1);
  Tensor cScores = make_identity_tensor(
      Shape<Int<kPrefillRowsPerWarpGroup>, Int<kBlockM>>{});
  Tensor tScScores = thread_qk.partition_C(cScores);
  Tensor tScScoresRowCol = make_tensor(
      tScScores.data(), streamattn_acc_rowcol<false>(tScScores.layout()));

  PrefillRSTiledMmaPV tiled_pv;
  auto thread_pv = tiled_pv.get_thread_slice(threadIdx.x);
  Tensor rV = thread_pv.partition_fragment_B(sV);
  Tensor output_acc = partition_fragment_C(
      tiled_pv,
      Shape<Int<kPrefillRowsPerWarpGroup>, Int<kHeadDim>>{});
  clear(output_acc);
  Tensor output_rows = make_tensor(
      output_acc.data(), streamattn_acc_rowcol<false>(output_acc.layout()));
  Tensor cOutput = make_identity_tensor(
      Shape<Int<kPrefillRowsPerWarpGroup>, Int<kHeadDim>>{});
  Tensor tOcOutput = thread_pv.partition_C(cOutput);
  Tensor tOcOutputRowCol = make_tensor(
      tOcOutput.data(), streamattn_acc_rowcol<false>(tOcOutput.layout()));

  constexpr int kRowsPerThread = decltype(size<0>(output_rows))::value;
  Accum row_max[kRowsPerThread];
  Accum row_sum[kRowsPerThread];
  CUTE_UNROLL
  for (int row = 0; row < kRowsPerThread; ++row) {
    row_max[row] = -INFINITY;
    row_sum[row] = 0.0f;
  }

  GmemCopyK copy_k;
  auto thread_copy_k = copy_k.get_thread_slice(threadIdx.x);
  Tensor tK0sK0 = thread_copy_k.partition_D(sK0);
  Tensor tK1sK1 = thread_copy_k.partition_D(sK1);
  GmemCopyPrefillRSV copy_v;
  auto thread_copy_v = copy_v.get_thread_slice(threadIdx.x);
  Tensor tVsV = thread_copy_v.partition_D(sV);

  auto copy_k_tile = [&](int tile, auto destination) {
    Tensor global = make_tensor(
        make_gmem_ptr(
            key
            + ((static_cast<int64_t>(batch) * sequence_length
                + tile * kBlockM)
                   * kv_heads
               + kv_head)
                * kHeadDim),
        Shape<Int<kBlockM>, Int<kHeadDim>>{},
        make_stride(kv_heads * kHeadDim, _1{}));
    cute::copy(
        copy_k, thread_copy_k.partition_S(global), destination);
  };
  auto copy_v_tile = [&](int tile) {
    Tensor global = make_tensor(
        make_gmem_ptr(
            value
            + ((static_cast<int64_t>(batch) * sequence_length
                + tile * kBlockM)
                   * kv_heads
               + kv_head)
                * kHeadDim),
        Shape<Int<kHeadDim>, Int<kBlockM>>{},
        make_stride(_1{}, kv_heads * kHeadDim));
    cute::copy(copy_v, thread_copy_v.partition_S(global), tVsV);
  };

  if (key_tiles > 0) {
    copy_k_tile(0, tK0sK0);
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();
  }

  constexpr Accum kScaleLog2 = 0.12751743082459868f;
  int read_pipe = 0;
  for (int tile = 0; tile < key_tiles; ++tile) {
    const int next_tile = tile + 1;
    const int write_pipe = read_pipe ^ 1;
    if (next_tile < key_tiles) {
      if (write_pipe == 0) {
        copy_k_tile(next_tile, tK0sK0);
      } else {
        copy_k_tile(next_tile, tK1sK1);
      }
      cute::cp_async_fence();
    }

    Tensor scores = partition_fragment_C(
        tiled_qk,
        Shape<Int<kPrefillRowsPerWarpGroup>, Int<kBlockM>>{});
    clear(scores);
    warpgroup_fence_operand(scores);
    warpgroup_arrive();
    if (read_pipe == 0) {
      cute::gemm(tiled_qk, rQ, rK0, scores);
    } else {
      cute::gemm(tiled_qk, rQ, rK1, scores);
    }
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(scores);

    copy_v_tile(tile);
    cute::cp_async_fence();

    Tensor score_rows = make_tensor(
        scores.data(), streamattn_acc_rowcol<false>(scores.layout()));
    static_assert(
        kRowsPerThread == decltype(size<0>(score_rows))::value,
        "QK and PV fragments must expose the same query rows");
    CUTE_UNROLL
    for (int row = 0; row < size<0>(score_rows); ++row) {
      CUTE_UNROLL
      for (int col = 0; col < size<1>(score_rows); ++col) {
        const auto coordinate = tScScoresRowCol(row, col);
        const int local_query_row = int(get<0>(coordinate));
        const int token = int(get<1>(coordinate));
        const int query_position =
            query_begin + local_query_row / group_size;
        const int key_position = tile * kBlockM + token;
        if (query_position >= sequence_length ||
            key_position > query_position ||
            key_position >= sequence_length) {
          score_rows(row, col) = -INFINITY;
        }
      }
    }

    CUTE_UNROLL
    for (int row = 0; row < kRowsPerThread; ++row) {
      Accum tile_max = -INFINITY;
      CUTE_UNROLL
      for (int col = 0; col < size<1>(score_rows); ++col) {
        tile_max = fmaxf(tile_max, score_rows(row, col));
      }
      tile_max = streamattn_quad_max(tile_max);
      const Accum next_max = fmaxf(row_max[row], tile_max);
      const Accum alpha = row_max[row] == -INFINITY
          ? 0.0f
          : exp2f((row_max[row] - next_max) * kScaleLog2);
      row_max[row] = next_max;
      row_sum[row] *= alpha;
      CUTE_UNROLL
      for (int col = 0; col < size<1>(output_rows); ++col) {
        output_rows(row, col) *= alpha;
      }

      Accum local_sum = 0.0f;
      const Accum max_scaled = next_max * kScaleLog2;
      CUTE_UNROLL
      for (int col = 0; col < size<1>(score_rows); ++col) {
        const Accum probability = next_max == -INFINITY
            ? 0.0f
            : exp2f(score_rows(row, col) * kScaleLog2 - max_scaled);
        score_rows(row, col) = probability;
        local_sum += probability;
      }
      row_sum[row] += streamattn_quad_sum(local_sum);
    }

    Tensor p_acc = make_tensor(
        scores.data(),
        streamattn_convert_layout_acc_aregs<PrefillRSTiledMmaPV>(
            scores.layout()));
    Tensor p_regs = streamattn_convert_type<Element>(p_acc);
    cute::cp_async_wait<0>();
    __syncthreads();
    warpgroup_fence_operand(p_regs);
    warpgroup_fence_operand(output_acc);
    warpgroup_arrive();
    cute::gemm(tiled_pv, p_regs, rV, output_acc);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(p_regs);
    warpgroup_fence_operand(output_acc);
    __syncthreads();
    read_pipe = write_pipe;
  }

  constexpr Accum kLn2 = 0.6931471805599453f;
  CUTE_UNROLL
  for (int row = 0; row < size<0>(output_rows); ++row) {
    CUTE_UNROLL
    for (int col = 0; col < size<1>(output_rows); ++col) {
      const auto coordinate = tOcOutputRowCol(row, col);
      const int local_query_row = int(get<0>(coordinate));
      const int dim = int(get<1>(coordinate));
      const int query_offset = local_query_row / group_size;
      const int head_offset = local_query_row - query_offset * group_size;
      const int query_position = query_begin + query_offset;
      if (query_position < sequence_length) {
        const int q_head = kv_head * group_size + head_offset;
        const int64_t destination =
            ((static_cast<int64_t>(batch) * sequence_length + query_position)
                 * q_heads
             + q_head)
                * kHeadDim
            + dim;
        output[destination] = row_sum[row] > 0.0f
            ? Element(output_rows(row, col) / row_sum[row])
            : Element(0.0f);
        if (dim == 0) {
          lse[(static_cast<int64_t>(batch) * sequence_length + query_position)
                  * q_heads
              + q_head] = row_sum[row] > 0.0f
              ? (row_max[row] * kScaleLog2 + log2f(row_sum[row])) * kLn2
              : -INFINITY;
        }
      }
    }
  }
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
  Element* k0_ptr = storage.k.data();
  Element* k1_ptr = storage.k.data() +
      (kPipelineStages == 2 ? cute::cosize_v<SmemLayoutK> : 0);
  Element* v0_ptr = kSeparateVStages == 2 ? storage.v.data() : k0_ptr;
  Element* v1_ptr = kSeparateVStages == 2
      ? storage.v.data() + cute::cosize_v<SmemLayoutV>
      : k1_ptr;
  Tensor sK0 = make_tensor(make_smem_ptr(k0_ptr), SmemLayoutK{});
  Tensor sK1 = make_tensor(make_smem_ptr(k1_ptr), SmemLayoutK{});
  Tensor sV0 = make_tensor(make_smem_ptr(v0_ptr), SmemLayoutV{});
  Tensor sV1 = make_tensor(make_smem_ptr(v1_ptr), SmemLayoutV{});
  Tensor sVt0 = make_tensor(make_smem_ptr(v0_ptr), SmemLayoutVt{});
  Tensor sVt1 = make_tensor(make_smem_ptr(v1_ptr), SmemLayoutVt{});
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
    if constexpr (kSeparateVStages == 2) {
      cute::copy(copy_kv, tVgV, tV0sV0);
    }
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
        if constexpr (kSeparateVStages == 2) {
          cute::copy(copy_kv, tVgVNext, tV0sV0);
        }
      } else {
        cute::copy(copy_kv, tKgKNext, tK1sK1);
        if constexpr (kSeparateVStages == 2) {
          cute::copy(copy_kv, tVgVNext, tV1sV1);
        }
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

    if constexpr (kSeparateVStages == 0) {
      const Element* current_v =
          group_v + static_cast<int64_t>(tile) * kBlockM * kHeadDim;
      Tensor gVCurrent = make_tensor(
          make_gmem_ptr(current_v), Shape<Int<kBlockM>, Int<kHeadDim>>{},
          make_stride(Int<kHeadDim>{}, _1{}));
      if (read_pipe == 0) {
        cute::copy(copy_kv, thr_copy_kv.partition_S(gVCurrent), tV0sV0);
      } else {
        cute::copy(copy_kv, thr_copy_kv.partition_S(gVCurrent), tV1sV1);
      }
      cute::cp_async_fence();
    }

    Tensor rP = streamattn_convert_type<Element>(tCrS);
    cute::copy(rP, tPsP);
    cutlass::arch::fence_view_async_shared();
    __syncthreads();

    if constexpr (kSeparateVStages == 0) {
      cute::cp_async_wait<0>();
      __syncthreads();
    }

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
      if constexpr (kSeparateVStages == 2) {
        cute::cp_async_wait<0>();
        __syncthreads();
        read_pipe = write_pipe;
      } else {
        read_pipe = write_pipe;
      }
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

template <int ConsumerRegisters>
__global__ __launch_bounds__(256, 1)
void streamattn_transposed_wgmma_qkpv_ws_cp_async_checksum_kernel(
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

  extern __shared__ char shared_memory[];
  auto& storage = *reinterpret_cast<WsQKPVSharedStorage*>(shared_memory);
  Element* k0_ptr = storage.k.data();
  Element* k1_ptr = storage.k.data() + cute::cosize_v<SmemLayoutK>;
  Tensor sK0 = make_tensor(make_smem_ptr(k0_ptr), SmemLayoutK{});
  Tensor sK1 = make_tensor(make_smem_ptr(k1_ptr), SmemLayoutK{});
  Tensor sV = make_tensor(make_smem_ptr(storage.v.data()), SmemLayoutV{});
  Tensor sVt = make_tensor(make_smem_ptr(storage.v.data()), SmemLayoutVt{});
  Tensor sQ = make_tensor(make_smem_ptr(storage.q.data()), SmemLayoutQ{});
  Tensor sPOrigin = make_tensor(
      make_smem_ptr(storage.p.data()), SmemLayoutPOrigin{});
  Tensor sP = make_tensor(make_smem_ptr(storage.p.data()), SmemLayoutP{});

  typename WsPipelineK::Params params_k;
  params_k.role = threadIdx.x < 128
      ? WsPipelineK::ThreadCategory::Producer
      : WsPipelineK::ThreadCategory::Consumer;
  params_k.producer_arv_count = 128;
  params_k.consumer_arv_count = 128;
  WsPipelineK pipeline_k(storage.pipeline_k, params_k);

  typename WsPipelineV::Params params_v;
  params_v.role = threadIdx.x < 128
      ? WsPipelineV::ThreadCategory::Producer
      : WsPipelineV::ThreadCategory::Consumer;
  params_v.producer_arv_count = 128;
  params_v.consumer_arv_count = 128;
  WsPipelineV pipeline_v(storage.pipeline_v, params_v);

  if (threadIdx.x < 128) {
    const Element* q_ptr =
        q_group + static_cast<int64_t>(group) * kBlockN * kHeadDim;
    for (int idx = threadIdx.x; idx < kBlockN * kHeadDim; idx += 128) {
      const int row = idx / kHeadDim;
      const int col = idx - row * kHeadDim;
      sQ(row, col) = q_ptr[idx];
    }
  }
  cutlass::arch::fence_view_async_shared();
  __syncthreads();

  const Element* group_k =
      k_cache + static_cast<int64_t>(group) * kv_len * kHeadDim;
  const Element* group_v =
      v_cache + static_cast<int64_t>(group) * kv_len * kHeadDim;

  if (threadIdx.x < 128) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    GmemCopyK copy_kv;
    auto thr_copy = copy_kv.get_thread_slice(threadIdx.x);
    Tensor tK0sK0 = thr_copy.partition_D(sK0);
    Tensor tK1sK1 = thr_copy.partition_D(sK1);
    Tensor tVsV = thr_copy.partition_D(sV);
    typename WsPipelineK::PipelineState write_k =
        cutlass::make_producer_start_state<WsPipelineK>();
    typename WsPipelineV::PipelineState write_v =
        cutlass::make_producer_start_state<WsPipelineV>();

    for (int tile = tile_begin; tile < tile_end; ++tile) {
      const Element* k_ptr =
          group_k + static_cast<int64_t>(tile) * kBlockM * kHeadDim;
      Tensor gK = make_tensor(
          make_gmem_ptr(k_ptr),
          Shape<Int<kBlockM>, Int<kHeadDim>>{},
          make_stride(Int<kHeadDim>{}, _1{}));
      pipeline_k.producer_acquire(write_k);
      if (write_k.index() == 0) {
        cute::copy(copy_kv, thr_copy.partition_S(gK), tK0sK0);
      } else {
        cute::copy(copy_kv, thr_copy.partition_S(gK), tK1sK1);
      }
      cute::cp_async_fence();
      cute::cp_async_wait<0>();
      pipeline_k.producer_commit(write_k);
      ++write_k;

      const Element* v_ptr =
          group_v + static_cast<int64_t>(tile) * kBlockM * kHeadDim;
      Tensor gV = make_tensor(
          make_gmem_ptr(v_ptr),
          Shape<Int<kBlockM>, Int<kHeadDim>>{},
          make_stride(Int<kHeadDim>{}, _1{}));
      pipeline_v.producer_acquire(write_v);
      cute::copy(copy_kv, thr_copy.partition_S(gV), tVsV);
      cute::cp_async_fence();
      cute::cp_async_wait<0>();
      pipeline_v.producer_commit(write_v);
      ++write_v;
    }
    pipeline_k.producer_tail(write_k);
    pipeline_v.producer_tail(write_v);
  } else {
    cutlass::arch::warpgroup_reg_alloc<ConsumerRegisters>();
    const int consumer_idx = threadIdx.x - 128;
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(consumer_idx);
    Tensor tSrK0 = thr_mma.partition_fragment_A(sK0);
    Tensor tSrK1 = thr_mma.partition_fragment_A(sK1);
    Tensor tSrQ = thr_mma.partition_fragment_B(sQ);
    Tensor tPsP = thr_mma.partition_C(sPOrigin);

    TiledMmaO tiled_mma_o;
    auto thr_mma_o = tiled_mma_o.get_thread_slice(consumer_idx);
    Tensor tOrV = thr_mma_o.partition_fragment_A(sVt);
    Tensor tOrP = thr_mma_o.partition_fragment_B(sP);
    Tensor tOrO = partition_fragment_C(
        tiled_mma_o, Shape<Int<kHeadDim>, Int<kBlockN>>{});
    clear(tOrO);

    typename WsPipelineK::PipelineState read_k;
    typename WsPipelineV::PipelineState read_v;
    for (int tile = tile_begin; tile < tile_end; ++tile) {
      auto token_k = pipeline_k.consumer_try_wait(read_k);
      pipeline_k.consumer_wait(read_k, token_k);
      Tensor tCrS = partition_fragment_C(
          tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
      clear(tCrS);
      warpgroup_fence_operand(tCrS);
      warpgroup_arrive();
      if (read_k.index() == 0) {
        cute::gemm(tiled_mma, tSrK0, tSrQ, tCrS);
      } else {
        cute::gemm(tiled_mma, tSrK1, tSrQ, tCrS);
      }
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(tCrS);
      pipeline_k.consumer_release(read_k);
      ++read_k;

      Tensor rP = streamattn_convert_type<Element>(tCrS);
      cute::copy(rP, tPsP);
      cutlass::arch::fence_view_async_shared();
      cutlass::arch::NamedBarrier::sync(128, 0);

      auto token_v = pipeline_v.consumer_try_wait(read_v);
      pipeline_v.consumer_wait(read_v, token_v);
      warpgroup_fence_operand(tOrO);
      warpgroup_arrive();
      cute::gemm(tiled_mma_o, tOrV, tOrP, tOrO);
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(tOrO);
      pipeline_v.consumer_release(read_v);
      ++read_v;
    }

    Accum local_sum = 0.0f;
    CUTE_UNROLL
    for (int idx = 0; idx < size(tOrO); ++idx) {
      local_sum += tOrO(idx);
    }
    storage.reduction[consumer_idx] = local_sum;
    cutlass::arch::NamedBarrier::sync(128, 1);
    if (consumer_idx == 0) {
      Accum total = 0.0f;
      CUTE_UNROLL
      for (int idx = 0; idx < 128; ++idx) {
        total += storage.reduction[idx];
      }
      checksums[static_cast<int64_t>(group) * num_splits + split] = total;
    }
  }
}

template <int kPagedPageSize>
__forceinline__ __device__ const Element* streamattn_exact_tile_ptr(
    const Element* base,
    const int* page_table,
    int group,
    int tile,
    int kv_len,
    int max_pages,
    int kv_heads) {
  if constexpr (kPagedPageSize == kBlockM) {
    const int batch = group / kv_heads;
    const int kv_head = group - batch * kv_heads;
    const int physical_page = page_table[batch * max_pages + tile];
    return base +
        (static_cast<int64_t>(physical_page) * kv_heads + kv_head) *
            kBlockM * kHeadDim;
  } else {
    return base +
        (static_cast<int64_t>(group) * kv_len + tile * kBlockM) * kHeadDim;
  }
}

template <bool kNHD, bool kVariableLength,
          class SmemTensor, class TiledCopy, class ThrCopy>
__forceinline__ __device__ void streamattn_copy_paged16_tile(
    const Element* base,
    const int* page_table,
    int group,
    int tile,
    int max_pages,
    int kv_heads,
    int sequence_length,
    SmemTensor const& destination,
    TiledCopy const& tiled_copy,
    ThrCopy const& thread_copy) {
  const int batch = group / kv_heads;
  const int kv_head = group - batch * kv_heads;
  const int logical_page_base = tile * 4;
  const int active_pages = kVariableLength
      ? (sequence_length + 15) / 16
      : max_pages;
  CUTE_UNROLL
  for (int fragment = 0; fragment < 4; ++fragment) {
    const int logical_page = min(
        logical_page_base + fragment, active_pages - 1);
    const int physical_page = page_table[batch * max_pages + logical_page];
    const Element* page;
    int token_stride;
    if constexpr (kNHD) {
      page = base +
          (static_cast<int64_t>(physical_page) * 16 * kv_heads + kv_head) *
              kHeadDim;
      token_stride = kv_heads * kHeadDim;
    } else {
      page = base +
          (static_cast<int64_t>(physical_page) * kv_heads + kv_head) *
              16 * kHeadDim;
      token_stride = kHeadDim;
    }
    Tensor source = make_tensor(
        make_gmem_ptr(page), Shape<Int<16>, Int<kHeadDim>>{},
        make_stride(token_stride, _1{}));
    Tensor destination_fragment = destination(_, _, fragment);
    cute::copy(
        tiled_copy,
        thread_copy.partition_S(source),
        thread_copy.partition_D(destination_fragment));
  }
}

template <bool kNHD, class SmemTensor, class TiledCopy, class ThrCopy>
__forceinline__ __device__ void streamattn_copy_selected_paged16_route(
    const Element* base,
    const int* physical_page_ids,
    int route,
    int group,
    int kv_heads,
    SmemTensor const& destination,
    TiledCopy const& tiled_copy,
    ThrCopy const& thread_copy) {
  const int kv_head = group % kv_heads;
  CUTE_UNROLL
  for (int fragment = 0; fragment < 4; ++fragment) {
    // Invalid atoms carry zero head/token masks. Reading page zero keeps the
    // asynchronous copy address valid; score masking removes every value from
    // the online-softmax state before it can affect the result.
    const int encoded_page = physical_page_ids[route * 4 + fragment];
    const int physical_page = max(encoded_page, 0);
    const Element* page;
    int token_stride;
    if constexpr (kNHD) {
      page = base +
          (static_cast<int64_t>(physical_page) * 16 * kv_heads + kv_head) *
              kHeadDim;
      token_stride = kv_heads * kHeadDim;
    } else {
      page = base +
          (static_cast<int64_t>(physical_page) * kv_heads + kv_head) *
              16 * kHeadDim;
      token_stride = kHeadDim;
    }
    Tensor source = make_tensor(
        make_gmem_ptr(page), Shape<Int<16>, Int<kHeadDim>>{},
        make_stride(token_stride, _1{}));
    Tensor destination_fragment = destination(_, _, fragment);
    cute::copy(
        tiled_copy,
        thread_copy.partition_S(source),
        thread_copy.partition_D(destination_fragment));
  }
}

template <int kPagedPageSize = 0, bool kVariableLength = false,
          bool kNHD = false, bool kSelectedPaged = false,
          bool kSelectedRowLocal = false>
__global__ __launch_bounds__(128)
void streamattn_transposed_wgmma_exact_partial_kernel(
    const Element* __restrict__ q_group,
    const Element* __restrict__ k_cache,
    const Element* __restrict__ v_cache,
    Accum* __restrict__ partial_o,
    Accum* __restrict__ partial_lse,
    int groups,
    int kv_len,
    int num_splits,
    int active_heads,
    const int* __restrict__ page_table = nullptr,
    int max_pages = 0,
    int kv_heads = 0,
    const int* __restrict__ sequence_lengths = nullptr,
    const int* __restrict__ route_row_ptr = nullptr,
    const int* __restrict__ route_physical_page_ids = nullptr,
    const int* __restrict__ route_active_head_masks = nullptr,
    const int* __restrict__ route_token_valid_masks = nullptr,
    const int* __restrict__ route_counts = nullptr) {
  const int work = blockIdx.x;
  const int group = work / num_splits;
  const int split = work - group * num_splits;
  if (group >= groups) {
    return;
  }

  int sequence_length = kv_len;
  int selected_route = -1;
  if constexpr (kVariableLength) {
    static_assert(kPagedPageSize == 16,
                  "ragged exact specialization requires page-16 storage");
    sequence_length = sequence_lengths[group / kv_heads];
  }
  int tile_begin;
  int tile_end;
  if constexpr (kSelectedPaged) {
    static_assert(kPagedPageSize == 16,
                  "selected paged specialization requires page-16 storage");
    const int route_begin = kSelectedRowLocal
        ? group * num_splits
        : route_row_ptr[group];
    const int route_end = kSelectedRowLocal
        ? route_begin + route_counts[group]
        : route_row_ptr[group + 1];
    selected_route = route_begin + split;
    tile_begin = 0;
    tile_end = selected_route < route_end ? 1 : 0;
  } else {
    const int num_tiles = (sequence_length + kBlockM - 1) / kBlockM;
    const int tiles_per_split = (num_tiles + num_splits - 1) / num_splits;
    tile_begin = split * tiles_per_split;
    tile_end = min(num_tiles, tile_begin + tiles_per_split);
  }
  if (tile_begin >= tile_end) {
    if constexpr (kSelectedRowLocal) {
      return;
    }
    for (int idx = threadIdx.x; idx < kBlockN * kHeadDim; idx += blockDim.x) {
      partial_o[(static_cast<int64_t>(work) * kBlockN * kHeadDim) + idx] = 0.0f;
    }
    if (threadIdx.x < kBlockN) {
      partial_lse[static_cast<int64_t>(work) * kBlockN + threadIdx.x] = -INFINITY;
    }
    return;
  }

  __shared__ AsyncQKPVSharedStorage storage;
  Element* k0_ptr = storage.k.data();
  Element* k1_ptr = storage.k.data() +
      (kPipelineStages == 2 ? cute::cosize_v<SmemLayoutK> : 0);
  Element* v0_ptr = kSeparateVStages == 2 ? storage.v.data() : k0_ptr;
  Element* v1_ptr = kSeparateVStages == 2
      ? storage.v.data() + cute::cosize_v<SmemLayoutV>
      : k1_ptr;
  Tensor sK0 = make_tensor(make_smem_ptr(k0_ptr), SmemLayoutK{});
  Tensor sK1 = make_tensor(make_smem_ptr(k1_ptr), SmemLayoutK{});
  Tensor sK0Paged16 = make_tensor(make_smem_ptr(k0_ptr), SmemLayoutPaged16{});
  Tensor sK1Paged16 = make_tensor(make_smem_ptr(k1_ptr), SmemLayoutPaged16{});
  Tensor sV0 = make_tensor(make_smem_ptr(v0_ptr), SmemLayoutV{});
  Tensor sV1 = make_tensor(make_smem_ptr(v1_ptr), SmemLayoutV{});
  Tensor sV0Paged16 = make_tensor(make_smem_ptr(v0_ptr), SmemLayoutPaged16{});
  Tensor sV1Paged16 = make_tensor(make_smem_ptr(v1_ptr), SmemLayoutPaged16{});
  Tensor sVt0 = make_tensor(make_smem_ptr(v0_ptr), SmemLayoutVt{});
  Tensor sVt1 = make_tensor(make_smem_ptr(v1_ptr), SmemLayoutVt{});
  Tensor sQ = make_tensor(make_smem_ptr(storage.q.data()), SmemLayoutQ{});
  Tensor sPOrigin = make_tensor(make_smem_ptr(storage.p.data()), SmemLayoutPOrigin{});
  Tensor sP = make_tensor(make_smem_ptr(storage.p.data()), SmemLayoutP{});

  const Element* q_ptr =
      q_group + static_cast<int64_t>(group) * active_heads * kHeadDim;
  for (int idx = threadIdx.x; idx < kBlockN * kHeadDim; idx += blockDim.x) {
    const int head = idx / kHeadDim;
    const int dim = idx - head * kHeadDim;
    sQ(head, dim) = head < active_heads
        ? q_ptr[static_cast<int64_t>(head) * kHeadDim + dim]
        : Element(0.0f);
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

  if (tile_begin < tile_end) {
    if constexpr (kSelectedPaged) {
      streamattn_copy_selected_paged16_route<kNHD>(
          k_cache, route_physical_page_ids, selected_route, group, kv_heads,
          sK0Paged16, copy_kv, thr_copy_kv);
      if constexpr (kSeparateVStages == 2) {
        streamattn_copy_selected_paged16_route<kNHD>(
            v_cache, route_physical_page_ids, selected_route, group, kv_heads,
            sV0Paged16, copy_kv, thr_copy_kv);
      }
    } else if constexpr (kPagedPageSize == 16) {
      streamattn_copy_paged16_tile<kNHD, kVariableLength>(
          k_cache, page_table, group, tile_begin, max_pages, kv_heads,
          sequence_length,
          sK0Paged16, copy_kv, thr_copy_kv);
      if constexpr (kSeparateVStages == 2) {
        streamattn_copy_paged16_tile<kNHD, kVariableLength>(
            v_cache, page_table, group, tile_begin, max_pages, kv_heads,
            sequence_length,
            sV0Paged16, copy_kv, thr_copy_kv);
      }
    } else {
      const Element* first_k = streamattn_exact_tile_ptr<kPagedPageSize>(
          k_cache, page_table, group, tile_begin, kv_len, max_pages, kv_heads);
      const Element* first_v = streamattn_exact_tile_ptr<kPagedPageSize>(
          v_cache, page_table, group, tile_begin, kv_len, max_pages, kv_heads);
      Tensor gK = make_tensor(make_gmem_ptr(first_k), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                              make_stride(Int<kHeadDim>{}, _1{}));
      Tensor gV = make_tensor(make_gmem_ptr(first_v), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                              make_stride(Int<kHeadDim>{}, _1{}));
      cute::copy(copy_kv, thr_copy_kv.partition_S(gK), tK0sK0);
      if constexpr (kSeparateVStages == 2) {
        cute::copy(copy_kv, thr_copy_kv.partition_S(gV), tV0sV0);
      }
    }
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();
  }

  constexpr Accum kSoftmaxScaleLog2 = kHeadDim == 64
      ? 0.18033688011112042f
      : 0.12751743082459868f;
  int read_pipe = 0;
  for (int tile = tile_begin; tile < tile_end; ++tile) {
    const int next_tile = tile + 1;
    const int write_pipe = read_pipe ^ 1;
    if (next_tile < tile_end) {
      if constexpr (kSelectedPaged) {
        if (write_pipe == 0) {
          streamattn_copy_selected_paged16_route<kNHD>(
              k_cache, route_physical_page_ids, selected_route, group, kv_heads,
              sK0Paged16, copy_kv, thr_copy_kv);
          if constexpr (kSeparateVStages == 2) {
            streamattn_copy_selected_paged16_route<kNHD>(
                v_cache, route_physical_page_ids, selected_route, group, kv_heads,
                sV0Paged16, copy_kv, thr_copy_kv);
          }
        } else {
          streamattn_copy_selected_paged16_route<kNHD>(
              k_cache, route_physical_page_ids, selected_route, group, kv_heads,
              sK1Paged16, copy_kv, thr_copy_kv);
          if constexpr (kSeparateVStages == 2) {
            streamattn_copy_selected_paged16_route<kNHD>(
                v_cache, route_physical_page_ids, selected_route, group, kv_heads,
                sV1Paged16, copy_kv, thr_copy_kv);
          }
        }
      } else if constexpr (kPagedPageSize == 16) {
        if (write_pipe == 0) {
          streamattn_copy_paged16_tile<kNHD, kVariableLength>(
              k_cache, page_table, group, next_tile, max_pages, kv_heads,
              sequence_length,
              sK0Paged16, copy_kv, thr_copy_kv);
          if constexpr (kSeparateVStages == 2) {
            streamattn_copy_paged16_tile<kNHD, kVariableLength>(
                v_cache, page_table, group, next_tile, max_pages, kv_heads,
                sequence_length,
                sV0Paged16, copy_kv, thr_copy_kv);
          }
        } else {
          streamattn_copy_paged16_tile<kNHD, kVariableLength>(
              k_cache, page_table, group, next_tile, max_pages, kv_heads,
              sequence_length,
              sK1Paged16, copy_kv, thr_copy_kv);
          if constexpr (kSeparateVStages == 2) {
            streamattn_copy_paged16_tile<kNHD, kVariableLength>(
                v_cache, page_table, group, next_tile, max_pages, kv_heads,
                sequence_length,
                sV1Paged16, copy_kv, thr_copy_kv);
          }
        }
      } else {
        const Element* next_k = streamattn_exact_tile_ptr<kPagedPageSize>(
            k_cache, page_table, group, next_tile, kv_len, max_pages, kv_heads);
        const Element* next_v = streamattn_exact_tile_ptr<kPagedPageSize>(
            v_cache, page_table, group, next_tile, kv_len, max_pages, kv_heads);
        Tensor gKNext = make_tensor(make_gmem_ptr(next_k), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                    make_stride(Int<kHeadDim>{}, _1{}));
        Tensor gVNext = make_tensor(make_gmem_ptr(next_v), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                    make_stride(Int<kHeadDim>{}, _1{}));
        if constexpr (kSeparateVStages == 0) {
          auto tKsKWrite = write_pipe == 0 ? tK0sK0 : tK1sK1;
          cute::copy(copy_kv, thr_copy_kv.partition_S(gKNext), tKsKWrite);
        } else if (write_pipe == 0) {
          cute::copy(copy_kv, thr_copy_kv.partition_S(gKNext), tK0sK0);
          cute::copy(copy_kv, thr_copy_kv.partition_S(gVNext), tV0sV0);
        } else {
          cute::copy(copy_kv, thr_copy_kv.partition_S(gKNext), tK1sK1);
          cute::copy(copy_kv, thr_copy_kv.partition_S(gVNext), tV1sV1);
        }
      }
      cute::cp_async_fence();
    }

    Tensor tCrS = partition_fragment_C(
        tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    clear(tCrS);
    warpgroup_fence_operand(tCrS);
    warpgroup_arrive();
    if constexpr (kSeparateVStages == 0) {
      auto tSrKRead = read_pipe == 0 ? tSrK0 : tSrK1;
      cute::gemm(tiled_mma, tSrKRead, tSrQ, tCrS);
    } else if (read_pipe == 0) {
      cute::gemm(tiled_mma, tSrK0, tSrQ, tCrS);
    } else {
      cute::gemm(tiled_mma, tSrK1, tSrQ, tCrS);
    }
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrS);

    if constexpr (kSeparateVStages == 0) {
      if constexpr (kSelectedPaged) {
        if (read_pipe == 0) {
          streamattn_copy_selected_paged16_route<kNHD>(
              v_cache, route_physical_page_ids, selected_route, group, kv_heads,
              sV0Paged16, copy_kv, thr_copy_kv);
        } else {
          streamattn_copy_selected_paged16_route<kNHD>(
              v_cache, route_physical_page_ids, selected_route, group, kv_heads,
              sV1Paged16, copy_kv, thr_copy_kv);
        }
      } else if constexpr (kPagedPageSize == 16) {
        if (read_pipe == 0) {
          streamattn_copy_paged16_tile<kNHD, kVariableLength>(
              v_cache, page_table, group, tile, max_pages, kv_heads,
              sequence_length,
              sV0Paged16, copy_kv, thr_copy_kv);
        } else {
          streamattn_copy_paged16_tile<kNHD, kVariableLength>(
              v_cache, page_table, group, tile, max_pages, kv_heads,
              sequence_length,
              sV1Paged16, copy_kv, thr_copy_kv);
        }
      } else {
        const Element* current_v = streamattn_exact_tile_ptr<kPagedPageSize>(
            v_cache, page_table, group, tile, kv_len, max_pages, kv_heads);
        Tensor gVCurrent = make_tensor(
            make_gmem_ptr(current_v), Shape<Int<kBlockM>, Int<kHeadDim>>{},
            make_stride(Int<kHeadDim>{}, _1{}));
        auto tVsVRead = read_pipe == 0 ? tV0sV0 : tV1sV1;
        cute::copy(copy_kv, thr_copy_kv.partition_S(gVCurrent), tVsVRead);
      }
      cute::cp_async_fence();
    }

    Tensor scores = make_tensor(
        tCrS.data(), streamattn_acc_rowcol<true>(tCrS.layout()));
    if constexpr (kSelectedPaged) {
      CUTE_UNROLL
      for (int row = 0; row < size<0>(scores); ++row) {
        CUTE_UNROLL
        for (int col = 0; col < size<1>(scores); ++col) {
          const int token = int(get<0>(tScSRowCol(row, col)));
          const int head = int(get<1>(tScSRowCol(row, col)));
          const int atom = token >> 4;
          const int token_in_atom = token & 15;
          const unsigned int head_mask = static_cast<unsigned int>(
              route_active_head_masks[selected_route * 4 + atom]);
          const unsigned int token_mask = static_cast<unsigned int>(
              route_token_valid_masks[selected_route * 4 + atom]);
          const bool selected = ((head_mask >> head) & 1u) != 0u &&
              ((token_mask >> token_in_atom) & 1u) != 0u;
          if (!selected) {
            scores(row, col) = -INFINITY;
          }
        }
      }
    }
    if constexpr (kVariableLength) {
      CUTE_UNROLL
      for (int row = 0; row < size<0>(scores); ++row) {
        CUTE_UNROLL
        for (int col = 0; col < size<1>(scores); ++col) {
          const int token = int(get<0>(tScSRowCol(row, col)));
          if (tile * kBlockM + token >= sequence_length) {
            scores(row, col) = -INFINITY;
          }
        }
      }
    }
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
        const Accum probability = next_max == -INFINITY
            ? 0.0f
            : exp2f(scores(row, col) * kSoftmaxScaleLog2 - max_scaled);
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

    if constexpr (kSeparateVStages == 0) {
      cute::cp_async_wait<0>();
      __syncthreads();
    }

    warpgroup_fence_operand(tOrO);
    warpgroup_arrive();
    if constexpr (kSeparateVStages == 0) {
      auto tOrVRead = read_pipe == 0 ? tOrV0 : tOrV1;
      cute::gemm(tiled_mma_o, tOrVRead, tOrP, tOrO);
    } else if (read_pipe == 0) {
      cute::gemm(tiled_mma_o, tOrV0, tOrP, tOrO);
    } else {
      cute::gemm(tiled_mma_o, tOrV1, tOrP, tOrO);
    }
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tOrO);

    if (next_tile < tile_end) {
      if constexpr (kSeparateVStages == 2) {
        cute::cp_async_wait<0>();
        __syncthreads();
        read_pipe = write_pipe;
      } else {
        read_pipe = write_pipe;
      }
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
      partial_lse[(static_cast<int64_t>(work) * kBlockN) + head] = total > 0.0f
          ? row_max[row] * kSoftmaxScaleLog2 + log2f(total)
          : -INFINITY;
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
          storage.row_sum[head] > 0.0f
              ? tOrORowCol(row, col) / storage.row_sum[head]
              : 0.0f;
    }
  }
}

__global__ __launch_bounds__(128)
void streamattn_transposed_wgmma_exact_merge_kernel(
    const Accum* __restrict__ partial_o,
    const Accum* __restrict__ partial_lse,
    Element* __restrict__ output,
    int groups,
    int num_splits,
    int active_heads) {
  const int row = blockIdx.x;
  if (row >= groups * active_heads) {
    return;
  }
  const int group = row / active_heads;
  const int head = row - group * active_heads;
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
    int num_splits,
    int active_heads) {
  const int row = blockIdx.x;
  if (row >= groups * active_heads) {
    return;
  }
  const int group = row / active_heads;
  const int head = row - group * active_heads;
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

  const Accum inverse_normalizer = 1.0f / normalizer;
  for (int dim0 = lane * 2; dim0 < kHeadDim; dim0 += 64) {
    Accum value0 = 0.0f;
    Accum value1 = 0.0f;
    for (int split = 0; split < num_splits; ++split) {
      const int64_t base =
          ((static_cast<int64_t>(group) * num_splits + split) * kBlockN +
           head) * kHeadDim + dim0;
      const float2 pair = *reinterpret_cast<const float2*>(partial_o + base);
      const Accum weight = weights[split];
      value0 += weight * pair.x;
      value1 += weight * pair.y;
    }
    const int64_t output_base = static_cast<int64_t>(row) * kHeadDim + dim0;
    output[output_base] = Element(value0 * inverse_normalizer);
    output[output_base + 1] = Element(value1 * inverse_normalizer);
  }
}

__global__ __launch_bounds__(32)
void streamattn_transposed_wgmma_selected_row_local_merge_warp_kernel(
    const Accum* __restrict__ partial_o,
    const Accum* __restrict__ partial_lse,
    const int* __restrict__ route_counts,
    Element* __restrict__ output,
    int groups,
    int route_stride,
    int active_heads) {
  const int row = blockIdx.x;
  if (row >= groups * active_heads) {
    return;
  }
  const int group = row / active_heads;
  const int head = row - group * active_heads;
  const int lane = threadIdx.x;
  const int route_count = route_counts[group];
  __shared__ Accum weights[512];

  Accum row_max = -INFINITY;
  for (int route = lane; route < route_count; route += 32) {
    row_max = fmaxf(
        row_max,
        partial_lse[(static_cast<int64_t>(group) * route_stride + route) *
                    kBlockN + head]);
  }
  row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffffu, row_max, 16));
  row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffffu, row_max, 8));
  row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffffu, row_max, 4));
  row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffffu, row_max, 2));
  row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffffu, row_max, 1));

  Accum normalizer = 0.0f;
  for (int route = lane; route < route_count; route += 32) {
    const Accum weight = exp2f(
        partial_lse[(static_cast<int64_t>(group) * route_stride + route) *
                    kBlockN + head] - row_max);
    weights[route] = weight;
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
    for (int route = 0; route < route_count; ++route) {
      const int64_t base =
          ((static_cast<int64_t>(group) * route_stride + route) * kBlockN +
           head) * kHeadDim + dim0;
      const float2 pair = *reinterpret_cast<const float2*>(partial_o + base);
      const Accum weight = weights[route];
      value0 += weight * pair.x;
      value1 += weight * pair.y;
    }
    const int64_t output_base = static_cast<int64_t>(row) * kHeadDim + dim0;
    output[output_base] = Element(value0 * inverse_normalizer);
    output[output_base + 1] = Element(value1 * inverse_normalizer);
  }
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

template <int ConsumerRegisters>
void launch_streamattn_qkpv_ws_cp_async(
    torch::Tensor q_group,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor checksums,
    int groups,
    int kv_len,
    int num_splits) {
  const int smem = static_cast<int>(sizeof(WsQKPVSharedStorage));
  auto kernel =
      streamattn_transposed_wgmma_qkpv_ws_cp_async_checksum_kernel<
          ConsumerRegisters>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  const dim3 grid(groups * num_splits);
  const dim3 block(256);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  kernel<<<grid, block, smem, stream>>>(
      reinterpret_cast<const Element*>(q_group.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(k_cache.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(v_cache.data_ptr<at::BFloat16>()),
      checksums.data_ptr<float>(),
      groups,
      kv_len,
      num_splits);
}

void streamattn_transposed_wgmma_qkpv_ws_cp_async_checksum_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor checksums,
    int64_t num_splits,
    int64_t consumer_registers) {
  TORCH_CHECK(q_group.is_cuda() && k_cache.is_cuda() &&
              v_cache.is_cuda() && checksums.is_cuda(),
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
  TORCH_CHECK(q_group.dim() == 4 && q_group.size(2) == kBlockN &&
              q_group.size(3) == kHeadDim,
              "q_group must have shape [B,Hkv,8,D]");
  TORCH_CHECK(k_cache.sizes() == v_cache.sizes(),
              "k_cache and v_cache must match");
  TORCH_CHECK(k_cache.dim() == 4 && k_cache.size(3) == kHeadDim,
              "K/V must have shape [B,Hkv,N,D]");
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
  TORCH_CHECK(checksums.numel() == static_cast<int64_t>(groups) * num_splits,
              "checksums must have B*Hkv*num_splits elements");

  switch (consumer_registers) {
    case 96:
      launch_streamattn_qkpv_ws_cp_async<96>(
          q_group, k_cache, v_cache, checksums, groups, kv_len, int(num_splits));
      break;
    case 112:
      launch_streamattn_qkpv_ws_cp_async<112>(
          q_group, k_cache, v_cache, checksums, groups, kv_len, int(num_splits));
      break;
    case 128:
      launch_streamattn_qkpv_ws_cp_async<128>(
          q_group, k_cache, v_cache, checksums, groups, kv_len, int(num_splits));
      break;
    case 160:
      launch_streamattn_qkpv_ws_cp_async<160>(
          q_group, k_cache, v_cache, checksums, groups, kv_len, int(num_splits));
      break;
    default:
      TORCH_CHECK(false, "consumer_registers must be 96, 112, 128, or 160");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int ConsumerRegisters>
void append_ws_resource_info(std::vector<int64_t>& values) {
  auto kernel =
      streamattn_transposed_wgmma_qkpv_ws_cp_async_checksum_kernel<
          ConsumerRegisters>;
  const int smem = static_cast<int>(sizeof(WsQKPVSharedStorage));
  cudaFuncAttributes attrs{};
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attrs, kernel));
  int blocks = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, kernel, 256, smem));
  values.push_back(attrs.numRegs);
  values.push_back(attrs.sharedSizeBytes);
  values.push_back(smem);
  values.push_back(blocks);
  values.push_back(attrs.maxThreadsPerBlock);
}

torch::Tensor streamattn_transposed_wgmma_qkpv_floor_resource_info_cuda(
    int64_t consumer_registers) {
  cudaFuncAttributes cooperative{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(
      &cooperative, streamattn_transposed_wgmma_qkpv_async_checksum_kernel));
  const int specialized_smem = static_cast<int>(sizeof(WsQKPVSharedStorage));
  int cooperative_blocks = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &cooperative_blocks,
      streamattn_transposed_wgmma_qkpv_async_checksum_kernel,
      128,
      0));
  std::vector<int64_t> values = {
          cooperative.numRegs,
          static_cast<int64_t>(cooperative.sharedSizeBytes),
          int64_t(0),
          cooperative_blocks,
          cooperative.maxThreadsPerBlock,
      };
  switch (consumer_registers) {
    case 96:
      append_ws_resource_info<96>(values);
      break;
    case 112:
      append_ws_resource_info<112>(values);
      break;
    case 128:
      append_ws_resource_info<128>(values);
      break;
    case 160:
      append_ws_resource_info<160>(values);
      break;
    default:
      TORCH_CHECK(false, "consumer_registers must be 96, 112, 128, or 160");
  }
  values.push_back(static_cast<int64_t>(sizeof(AsyncQKPVSharedStorage)));
  values.push_back(specialized_smem);
  return torch::tensor(values, torch::TensorOptions().dtype(torch::kInt64));
}

torch::Tensor streamattn_grouped_wgmma_prefill_resource_info_cuda() {
  auto kernel = streamattn_grouped_wgmma_prefill_kernel;
  const int shared_bytes =
      static_cast<int>(sizeof(GroupedPrefillSharedStorage));
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
  cudaFuncAttributes attributes{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
  int blocks = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, kernel, 256, shared_bytes));
  return torch::tensor(
      {
          static_cast<int64_t>(attributes.numRegs),
          static_cast<int64_t>(attributes.sharedSizeBytes),
          static_cast<int64_t>(shared_bytes),
          static_cast<int64_t>(blocks),
          static_cast<int64_t>(attributes.maxThreadsPerBlock),
      },
      torch::TensorOptions().dtype(torch::kInt64));
}

torch::Tensor streamattn_grouped_rs_prefill_resource_info_cuda() {
  auto kernel = streamattn_grouped_rs_prefill_kernel;
  const int shared_bytes =
      static_cast<int>(sizeof(GroupedRSPrefillSharedStorage));
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
  cudaFuncAttributes attributes{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
  int blocks = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, kernel, 128, shared_bytes));
  return torch::tensor(
      {
          static_cast<int64_t>(attributes.numRegs),
          static_cast<int64_t>(attributes.sharedSizeBytes),
          static_cast<int64_t>(shared_bytes),
          static_cast<int64_t>(blocks),
          static_cast<int64_t>(attributes.maxThreadsPerBlock),
          static_cast<int64_t>(attributes.localSizeBytes),
      },
      torch::TensorOptions().dtype(torch::kInt64));
}

void streamattn_grouped_wgmma_prefill_out_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    torch::Tensor lse) {
  TORCH_CHECK(
      query.is_cuda() && key.is_cuda() && value.is_cuda() &&
          output.is_cuda() && lse.is_cuda(),
      "grouped prefill tensors must be CUDA tensors");
  TORCH_CHECK(
      query.is_contiguous() && key.is_contiguous() && value.is_contiguous() &&
          output.is_contiguous() && lse.is_contiguous(),
      "grouped prefill tensors must be contiguous");
  TORCH_CHECK(
      query.scalar_type() == at::ScalarType::BFloat16 &&
          key.scalar_type() == at::ScalarType::BFloat16 &&
          value.scalar_type() == at::ScalarType::BFloat16 &&
          output.scalar_type() == at::ScalarType::BFloat16,
      "grouped prefill Q/K/V/output must use bf16");
  TORCH_CHECK(
      lse.scalar_type() == at::ScalarType::Float,
      "grouped prefill LSE must use fp32");
  TORCH_CHECK(
      query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
      "grouped prefill Q/K/V must have shape [B,S,H,D]");
  TORCH_CHECK(key.sizes() == value.sizes(), "K/V shapes must match");
  TORCH_CHECK(
      query.size(0) == key.size(0) && query.size(1) == key.size(1) &&
          query.size(3) == key.size(3),
      "Q/K/V batch, sequence, and head dimensions must match");
  TORCH_CHECK(
      query.size(3) == kHeadDim,
      "Q/K/V head dimension does not match the compiled specialization");
  const int batch_size = static_cast<int>(query.size(0));
  const int sequence_length = static_cast<int>(query.size(1));
  const int q_heads = static_cast<int>(query.size(2));
  const int kv_heads = static_cast<int>(key.size(2));
  TORCH_CHECK(
      batch_size > 0 && sequence_length > 0 && kv_heads > 0 &&
          q_heads % kv_heads == 0,
      "grouped prefill requires positive dimensions and integral GQA groups");
  const int group_size = q_heads / kv_heads;
  TORCH_CHECK(
      group_size == 4 || group_size == 8,
      "grouped prefill currently supports G4 or G8");
  TORCH_CHECK(
      output.sizes() == query.sizes(),
      "grouped prefill output must match query shape");
  TORCH_CHECK(
      lse.sizes() == torch::IntArrayRef({batch_size, sequence_length, q_heads}),
      "grouped prefill LSE must have shape [B,S,Hq]");

  const int query_positions = kPrefillRows / group_size;
  const int query_tiles =
      (sequence_length + query_positions - 1) / query_positions;
  const int shared_bytes =
      static_cast<int>(sizeof(GroupedPrefillSharedStorage));
  auto kernel = streamattn_grouped_wgmma_prefill_kernel;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  kernel<<<batch_size * kv_heads * query_tiles, 256, shared_bytes, stream>>>(
      reinterpret_cast<const Element*>(query.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(key.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(value.data_ptr<at::BFloat16>()),
      reinterpret_cast<Element*>(output.data_ptr<at::BFloat16>()),
      lse.data_ptr<float>(),
      batch_size,
      sequence_length,
      q_heads,
      kv_heads,
      group_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void streamattn_grouped_rs_prefill_out_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    torch::Tensor lse) {
  TORCH_CHECK(
      query.is_cuda() && key.is_cuda() && value.is_cuda() &&
          output.is_cuda() && lse.is_cuda(),
      "grouped RS prefill tensors must be CUDA tensors");
  TORCH_CHECK(
      query.is_contiguous() && key.is_contiguous() && value.is_contiguous() &&
          output.is_contiguous() && lse.is_contiguous(),
      "grouped RS prefill tensors must be contiguous");
  TORCH_CHECK(
      query.scalar_type() == at::ScalarType::BFloat16 &&
          key.scalar_type() == at::ScalarType::BFloat16 &&
          value.scalar_type() == at::ScalarType::BFloat16 &&
          output.scalar_type() == at::ScalarType::BFloat16,
      "grouped RS prefill Q/K/V/output must use bf16");
  TORCH_CHECK(
      lse.scalar_type() == at::ScalarType::Float,
      "grouped RS prefill LSE must use fp32");
  TORCH_CHECK(
      query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
      "grouped RS prefill Q/K/V must have shape [B,S,H,D]");
  TORCH_CHECK(key.sizes() == value.sizes(), "grouped RS K/V shapes must match");
  TORCH_CHECK(
      query.size(0) == key.size(0) && query.size(1) == key.size(1) &&
          query.size(3) == key.size(3),
      "grouped RS Q/K/V batch, sequence, and head dimensions must match");
  TORCH_CHECK(
      kHeadDim == 128 && query.size(3) == kHeadDim,
      "grouped RS prefill is specialized for D128");
  const int batch_size = static_cast<int>(query.size(0));
  const int sequence_length = static_cast<int>(query.size(1));
  const int q_heads = static_cast<int>(query.size(2));
  const int kv_heads = static_cast<int>(key.size(2));
  TORCH_CHECK(
      batch_size > 0 && sequence_length > 0 && sequence_length % kBlockM == 0 &&
          kv_heads > 0 && q_heads % kv_heads == 0,
      "grouped RS prefill requires positive dimensions, S divisible by 64, "
      "and integral GQA groups");
  const int group_size = q_heads / kv_heads;
  TORCH_CHECK(
      group_size == 4 || group_size == 8,
      "grouped RS prefill currently supports G4 or G8");
  TORCH_CHECK(
      output.sizes() == query.sizes(),
      "grouped RS prefill output must match query shape");
  TORCH_CHECK(
      lse.sizes() == torch::IntArrayRef({batch_size, sequence_length, q_heads}),
      "grouped RS prefill LSE must have shape [B,S,Hq]");

  const int query_positions = kPrefillRowsPerWarpGroup / group_size;
  const int query_tiles =
      (sequence_length + query_positions - 1) / query_positions;
  const int shared_bytes =
      static_cast<int>(sizeof(GroupedRSPrefillSharedStorage));
  auto kernel = streamattn_grouped_rs_prefill_kernel;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  kernel<<<batch_size * kv_heads * query_tiles, 128, shared_bytes, stream>>>(
      reinterpret_cast<const Element*>(query.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(key.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(value.data_ptr<at::BFloat16>()),
      reinterpret_cast<Element*>(output.data_ptr<at::BFloat16>()),
      lse.data_ptr<float>(),
      batch_size,
      sequence_length,
      q_heads,
      kv_heads,
      group_size);
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
  TORCH_CHECK(q_group.dim() == 4 &&
              (q_group.size(2) == 4 || q_group.size(2) == kBlockN) &&
              q_group.size(3) == kHeadDim,
              "q_group must have shape [B,Hkv,4|8,64]");
  TORCH_CHECK(k_cache.sizes() == v_cache.sizes(), "k_cache and v_cache must match");
  TORCH_CHECK(k_cache.dim() == 4 && k_cache.size(3) == kHeadDim,
              "K/V must have shape [B,Hkv,N,64]");
  TORCH_CHECK(k_cache.size(0) == q_group.size(0) &&
              k_cache.size(1) == q_group.size(1),
              "q_group and K/V batch/KV-head dimensions must match");
  TORCH_CHECK(k_cache.size(2) % kBlockM == 0, "kv_len must be divisible by 64");

  const int groups = static_cast<int>(q_group.size(0) * q_group.size(1));
  const int active_heads = static_cast<int>(q_group.size(2));
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
  streamattn_transposed_wgmma_exact_partial_kernel<0><<<grid, block, 0, stream>>>(
      reinterpret_cast<const Element*>(q_group.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(k_cache.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(v_cache.data_ptr<at::BFloat16>()),
      partial_o.data_ptr<float>(),
      partial_lse.data_ptr<float>(),
      groups,
      kv_len,
      static_cast<int>(num_splits),
      active_heads);
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
  TORCH_CHECK(output.dim() == 3 && output.size(0) == partial_o.size(0) &&
              (output.size(1) == 4 || output.size(1) == kBlockN) &&
              output.size(2) == kHeadDim,
              "output must have shape [groups,4|8,64]");
  TORCH_CHECK(partial_o.size(1) <= 512, "num_splits must be <= 512");

  const int groups = static_cast<int>(partial_o.size(0));
  const int num_splits = static_cast<int>(partial_o.size(1));
  const int active_heads = static_cast<int>(output.size(1));
  const dim3 grid(groups * active_heads);
  const dim3 block(kHeadDim);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  streamattn_transposed_wgmma_exact_merge_kernel<<<grid, block, 0, stream>>>(
      partial_o.data_ptr<float>(),
      partial_lse.data_ptr<float>(),
      reinterpret_cast<Element*>(output.data_ptr<at::BFloat16>()),
      groups,
      num_splits,
      active_heads);
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
  TORCH_CHECK(output.dim() == 3 && output.size(0) == partial_o.size(0) &&
              (output.size(1) == 4 || output.size(1) == kBlockN) &&
              output.size(2) == kHeadDim,
              "output must have shape [groups,4|8,64]");
  TORCH_CHECK(partial_o.size(1) <= 512, "num_splits must be <= 512");

  const int groups = static_cast<int>(partial_o.size(0));
  const int num_splits = static_cast<int>(partial_o.size(1));
  const int active_heads = static_cast<int>(output.size(1));
  const dim3 grid(groups * active_heads);
  const dim3 block(32);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  streamattn_transposed_wgmma_exact_merge_warp_kernel<<<grid, block, 0, stream>>>(
      partial_o.data_ptr<float>(),
      partial_lse.data_ptr<float>(),
      reinterpret_cast<Element*>(output.data_ptr<at::BFloat16>()),
      groups,
      num_splits,
      active_heads);
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
  TORCH_CHECK(q_group.dim() == 4 &&
              (q_group.size(2) == 4 || q_group.size(2) == kBlockN) &&
              q_group.size(3) == kHeadDim,
              "q_group must have shape [B,Hkv,4|8,64]");
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
  const int active_heads = static_cast<int>(q_group.size(2));
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
                  {groups, active_heads, kHeadDim}),
              "output must have shape [B*Hkv,4|8,64]");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 partial_grid(groups * static_cast<int>(num_splits));
  const dim3 partial_block(128);
  streamattn_transposed_wgmma_exact_partial_kernel<0><<<
      partial_grid, partial_block, 0, stream>>>(
      reinterpret_cast<const Element*>(q_group.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(k_cache.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(v_cache.data_ptr<at::BFloat16>()),
      partial_o.data_ptr<float>(),
      partial_lse.data_ptr<float>(),
      groups,
      kv_len,
      static_cast<int>(num_splits),
      active_heads);

  const dim3 merge_grid(groups * active_heads);
  const dim3 merge_block(kHeadDim);
  streamattn_transposed_wgmma_exact_merge_kernel<<<
      merge_grid, merge_block, 0, stream>>>(
      partial_o.data_ptr<float>(),
      partial_lse.data_ptr<float>(),
      reinterpret_cast<Element*>(output.data_ptr<at::BFloat16>()),
      groups,
      static_cast<int>(num_splits),
      active_heads);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void streamattn_transposed_wgmma_paged_exact_decode_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor page_table,
    torch::Tensor partial_o,
    torch::Tensor partial_lse,
    torch::Tensor output,
    int64_t num_splits) {
  TORCH_CHECK(q_group.is_cuda() && k_pages.is_cuda() && v_pages.is_cuda() &&
              page_table.is_cuda() && partial_o.is_cuda() &&
              partial_lse.is_cuda() && output.is_cuda(),
              "all tensors must be CUDA tensors");
  TORCH_CHECK(q_group.is_contiguous() && k_pages.is_contiguous() &&
              v_pages.is_contiguous() && page_table.is_contiguous() &&
              partial_o.is_contiguous() && partial_lse.is_contiguous() &&
              output.is_contiguous(), "all tensors must be contiguous");
  TORCH_CHECK(q_group.scalar_type() == at::ScalarType::BFloat16 &&
              k_pages.scalar_type() == at::ScalarType::BFloat16 &&
              v_pages.scalar_type() == at::ScalarType::BFloat16,
              "q_group and K/V pages must be bf16");
  TORCH_CHECK(page_table.scalar_type() == at::ScalarType::Int,
              "page_table must be int32");
  TORCH_CHECK(partial_o.scalar_type() == at::ScalarType::Float &&
              partial_lse.scalar_type() == at::ScalarType::Float,
              "partial outputs must be fp32");
  TORCH_CHECK(output.scalar_type() == at::ScalarType::BFloat16,
              "output must be bf16");
  TORCH_CHECK(q_group.dim() == 4 && q_group.size(2) == kBlockN &&
              q_group.size(3) == kHeadDim,
              "q_group must have shape [B,Hkv,8,64]");
  TORCH_CHECK(k_pages.sizes() == v_pages.sizes(),
              "K/V page tensors must match");
  TORCH_CHECK(k_pages.dim() == 4 && k_pages.size(1) == q_group.size(1) &&
              k_pages.size(2) == kBlockM && k_pages.size(3) == kHeadDim,
              "HND K/V pages must have shape [pages,Hkv,64,64]");
  TORCH_CHECK(page_table.dim() == 2 &&
              page_table.size(0) == q_group.size(0),
              "page_table must have shape [B,max_pages]");

  const int batch = static_cast<int>(q_group.size(0));
  const int kv_heads = static_cast<int>(q_group.size(1));
  const int groups = batch * kv_heads;
  const int active_heads = kBlockN;
  const int max_pages = static_cast<int>(page_table.size(1));
  const int kv_len = max_pages * kBlockM;
  TORCH_CHECK(num_splits > 0 && num_splits <= max_pages,
              "num_splits must be in [1,max_pages]");
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
  streamattn_transposed_wgmma_exact_partial_kernel<64><<<
      partial_grid, partial_block, 0, stream>>>(
      reinterpret_cast<const Element*>(q_group.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(k_pages.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(v_pages.data_ptr<at::BFloat16>()),
      partial_o.data_ptr<float>(),
      partial_lse.data_ptr<float>(),
      groups,
      kv_len,
      static_cast<int>(num_splits),
      active_heads,
      page_table.data_ptr<int>(),
      max_pages,
      kv_heads);

  const dim3 merge_grid(groups * active_heads);
  const dim3 merge_block(32);
  streamattn_transposed_wgmma_exact_merge_warp_kernel<<<
      merge_grid, merge_block, 0, stream>>>(
      partial_o.data_ptr<float>(),
      partial_lse.data_ptr<float>(),
      reinterpret_cast<Element*>(output.data_ptr<at::BFloat16>()),
      groups,
      static_cast<int>(num_splits),
      active_heads);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void streamattn_transposed_wgmma_paged_fragmented_exact_decode_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor page_table,
    torch::Tensor partial_o,
    torch::Tensor partial_lse,
    torch::Tensor output,
    int64_t num_splits) {
  TORCH_CHECK(q_group.is_cuda() && k_pages.is_cuda() && v_pages.is_cuda() &&
              page_table.is_cuda() && partial_o.is_cuda() &&
              partial_lse.is_cuda() && output.is_cuda(),
              "all tensors must be CUDA tensors");
  TORCH_CHECK(q_group.is_contiguous() && k_pages.is_contiguous() &&
              v_pages.is_contiguous() && page_table.is_contiguous() &&
              partial_o.is_contiguous() && partial_lse.is_contiguous() &&
              output.is_contiguous(), "all tensors must be contiguous");
  TORCH_CHECK(q_group.scalar_type() == at::ScalarType::BFloat16 &&
              k_pages.scalar_type() == at::ScalarType::BFloat16 &&
              v_pages.scalar_type() == at::ScalarType::BFloat16,
              "q_group and K/V pages must be bf16");
  TORCH_CHECK(page_table.scalar_type() == at::ScalarType::Int,
              "page_table must be int32");
  TORCH_CHECK(partial_o.scalar_type() == at::ScalarType::Float &&
              partial_lse.scalar_type() == at::ScalarType::Float,
              "partial outputs must be fp32");
  TORCH_CHECK(output.scalar_type() == at::ScalarType::BFloat16,
              "output must be bf16");
  TORCH_CHECK(q_group.dim() == 4 &&
              (q_group.size(2) == 4 || q_group.size(2) == kBlockN) &&
              q_group.size(3) == kHeadDim,
              "q_group must have shape [B,Hkv,4|8,D]");
  TORCH_CHECK(k_pages.sizes() == v_pages.sizes(),
              "K/V page tensors must match");
  TORCH_CHECK(k_pages.dim() == 4 && k_pages.size(1) == q_group.size(1) &&
              k_pages.size(2) == 16 && k_pages.size(3) == kHeadDim,
              "HND K/V pages must have shape [pages,Hkv,16,D]");
  TORCH_CHECK(page_table.dim() == 2 &&
              page_table.size(0) == q_group.size(0),
              "page_table must have shape [B,max_pages]");

  const int batch = static_cast<int>(q_group.size(0));
  const int kv_heads = static_cast<int>(q_group.size(1));
  const int groups = batch * kv_heads;
  const int active_heads = static_cast<int>(q_group.size(2));
  const int max_pages = static_cast<int>(page_table.size(1));
  TORCH_CHECK(max_pages % 4 == 0,
              "page-16 tables must contain a multiple of four pages");
  const int kv_len = max_pages * 16;
  const int logical_tiles = max_pages / 4;
  TORCH_CHECK(num_splits > 0 && num_splits <= logical_tiles,
              "num_splits must be in [1,max_pages/4]");
  TORCH_CHECK(partial_o.sizes() == torch::IntArrayRef(
                  {groups, num_splits, kBlockN, kHeadDim}),
              "partial_o must have shape [B*Hkv,num_splits,8,D]");
  TORCH_CHECK(partial_lse.sizes() == torch::IntArrayRef(
                  {groups, num_splits, kBlockN}),
              "partial_lse must have shape [B*Hkv,num_splits,8]");
  TORCH_CHECK(output.sizes() == torch::IntArrayRef(
                  {groups, active_heads, kHeadDim}),
              "output must have shape [B*Hkv,4|8,D]");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 partial_grid(groups * static_cast<int>(num_splits));
  const dim3 partial_block(128);
  streamattn_transposed_wgmma_exact_partial_kernel<16><<<
      partial_grid, partial_block, 0, stream>>>(
      reinterpret_cast<const Element*>(q_group.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(k_pages.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(v_pages.data_ptr<at::BFloat16>()),
      partial_o.data_ptr<float>(),
      partial_lse.data_ptr<float>(),
      groups,
      kv_len,
      static_cast<int>(num_splits),
      active_heads,
      page_table.data_ptr<int>(),
      max_pages,
      kv_heads);

  const dim3 merge_grid(groups * active_heads);
  const dim3 merge_block(32);
  streamattn_transposed_wgmma_exact_merge_warp_kernel<<<
      merge_grid, merge_block, 0, stream>>>(
      partial_o.data_ptr<float>(),
      partial_lse.data_ptr<float>(),
      reinterpret_cast<Element*>(output.data_ptr<at::BFloat16>()),
      groups,
      static_cast<int>(num_splits),
      active_heads);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void streamattn_transposed_wgmma_paged_fragmented_ragged_exact_decode_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor page_table,
    torch::Tensor sequence_lengths,
    torch::Tensor partial_o,
    torch::Tensor partial_lse,
    torch::Tensor output,
    int64_t num_splits) {
  TORCH_CHECK(q_group.is_cuda() && k_pages.is_cuda() && v_pages.is_cuda() &&
              page_table.is_cuda() && sequence_lengths.is_cuda() &&
              partial_o.is_cuda() && partial_lse.is_cuda() && output.is_cuda(),
              "all tensors must be CUDA tensors");
  TORCH_CHECK(q_group.is_contiguous() && k_pages.is_contiguous() &&
              v_pages.is_contiguous() && page_table.is_contiguous() &&
              sequence_lengths.is_contiguous() && partial_o.is_contiguous() &&
              partial_lse.is_contiguous() && output.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(q_group.scalar_type() == at::ScalarType::BFloat16 &&
              k_pages.scalar_type() == at::ScalarType::BFloat16 &&
              v_pages.scalar_type() == at::ScalarType::BFloat16,
              "q_group and K/V pages must be bf16");
  TORCH_CHECK(page_table.scalar_type() == at::ScalarType::Int &&
              sequence_lengths.scalar_type() == at::ScalarType::Int,
              "page_table and sequence_lengths must be int32");
  TORCH_CHECK(partial_o.scalar_type() == at::ScalarType::Float &&
              partial_lse.scalar_type() == at::ScalarType::Float,
              "partial outputs must be fp32");
  TORCH_CHECK(output.scalar_type() == at::ScalarType::BFloat16,
              "output must be bf16");
  TORCH_CHECK(q_group.dim() == 4 &&
              (q_group.size(2) == 4 || q_group.size(2) == kBlockN) &&
              q_group.size(3) == kHeadDim,
              "q_group must have shape [B,Hkv,4|8,D]");
  TORCH_CHECK(k_pages.sizes() == v_pages.sizes(),
              "K/V page tensors must match");
  TORCH_CHECK(k_pages.dim() == 4 && k_pages.size(1) == q_group.size(1) &&
              k_pages.size(2) == 16 && k_pages.size(3) == kHeadDim,
              "HND K/V pages must have shape [pages,Hkv,16,D]");
  TORCH_CHECK(page_table.dim() == 2 &&
              page_table.size(0) == q_group.size(0),
              "page_table must have shape [B,max_pages]");
  TORCH_CHECK(sequence_lengths.dim() == 1 &&
              sequence_lengths.size(0) == q_group.size(0),
              "sequence_lengths must have shape [B]");

  const int batch = static_cast<int>(q_group.size(0));
  const int kv_heads = static_cast<int>(q_group.size(1));
  const int groups = batch * kv_heads;
  const int active_heads = static_cast<int>(q_group.size(2));
  const int max_pages = static_cast<int>(page_table.size(1));
  TORCH_CHECK(max_pages % 4 == 0,
              "page-16 tables must contain a multiple of four pages");
  const int kv_len = max_pages * 16;
  const int logical_tiles = max_pages / 4;
  TORCH_CHECK(num_splits > 0 && num_splits <= logical_tiles,
              "num_splits must be in [1,max_pages/4]");
  TORCH_CHECK(partial_o.sizes() == torch::IntArrayRef(
                  {groups, num_splits, kBlockN, kHeadDim}),
              "partial_o must have shape [B*Hkv,num_splits,8,D]");
  TORCH_CHECK(partial_lse.sizes() == torch::IntArrayRef(
                  {groups, num_splits, kBlockN}),
              "partial_lse must have shape [B*Hkv,num_splits,8]");
  TORCH_CHECK(output.sizes() == torch::IntArrayRef(
                  {groups, active_heads, kHeadDim}),
              "output must have shape [B*Hkv,4|8,D]");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 partial_grid(groups * static_cast<int>(num_splits));
  const dim3 partial_block(128);
  streamattn_transposed_wgmma_exact_partial_kernel<16, true><<<
      partial_grid, partial_block, 0, stream>>>(
      reinterpret_cast<const Element*>(q_group.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(k_pages.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(v_pages.data_ptr<at::BFloat16>()),
      partial_o.data_ptr<float>(),
      partial_lse.data_ptr<float>(),
      groups,
      kv_len,
      static_cast<int>(num_splits),
      active_heads,
      page_table.data_ptr<int>(),
      max_pages,
      kv_heads,
      sequence_lengths.data_ptr<int>());

  const dim3 merge_grid(groups * active_heads);
  const dim3 merge_block(32);
  streamattn_transposed_wgmma_exact_merge_warp_kernel<<<
      merge_grid, merge_block, 0, stream>>>(
      partial_o.data_ptr<float>(),
      partial_lse.data_ptr<float>(),
      reinterpret_cast<Element*>(output.data_ptr<at::BFloat16>()),
      groups,
      static_cast<int>(num_splits),
      active_heads);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool kVariableLength>
void streamattn_transposed_wgmma_paged_fragmented_nhd_impl(
    torch::Tensor q_group,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor page_table,
    const torch::Tensor* sequence_lengths,
    torch::Tensor partial_o,
    torch::Tensor partial_lse,
    torch::Tensor output,
    int64_t num_splits) {
  TORCH_CHECK(q_group.is_cuda() && k_pages.is_cuda() && v_pages.is_cuda() &&
              page_table.is_cuda() && partial_o.is_cuda() &&
              partial_lse.is_cuda() && output.is_cuda(),
              "all tensors must be CUDA tensors");
  TORCH_CHECK(q_group.is_contiguous() && k_pages.is_contiguous() &&
              v_pages.is_contiguous() && page_table.is_contiguous() &&
              partial_o.is_contiguous() && partial_lse.is_contiguous() &&
              output.is_contiguous(), "all tensors must be contiguous");
  TORCH_CHECK(q_group.scalar_type() == at::ScalarType::BFloat16 &&
              k_pages.scalar_type() == at::ScalarType::BFloat16 &&
              v_pages.scalar_type() == at::ScalarType::BFloat16,
              "q_group and K/V pages must be bf16");
  TORCH_CHECK(page_table.scalar_type() == at::ScalarType::Int,
              "page_table must be int32");
  TORCH_CHECK(partial_o.scalar_type() == at::ScalarType::Float &&
              partial_lse.scalar_type() == at::ScalarType::Float,
              "partial outputs must be fp32");
  TORCH_CHECK(output.scalar_type() == at::ScalarType::BFloat16,
              "output must be bf16");
  TORCH_CHECK(q_group.dim() == 4 &&
              (q_group.size(2) == 4 || q_group.size(2) == kBlockN) &&
              q_group.size(3) == kHeadDim,
              "q_group must have shape [B,Hkv,4|8,D]");
  TORCH_CHECK(k_pages.sizes() == v_pages.sizes(),
              "K/V page tensors must match");
  TORCH_CHECK(k_pages.dim() == 4 && k_pages.size(1) == 16 &&
              k_pages.size(2) == q_group.size(1) &&
              k_pages.size(3) == kHeadDim,
              "NHD K/V pages must have shape [pages,16,Hkv,D]");
  TORCH_CHECK(page_table.dim() == 2 &&
              page_table.size(0) == q_group.size(0),
              "page_table must have shape [B,max_pages]");
  if constexpr (kVariableLength) {
    TORCH_CHECK(sequence_lengths != nullptr && sequence_lengths->is_cuda() &&
                sequence_lengths->is_contiguous() &&
                sequence_lengths->scalar_type() == at::ScalarType::Int &&
                sequence_lengths->dim() == 1 &&
                sequence_lengths->size(0) == q_group.size(0),
                "sequence_lengths must be contiguous CUDA int32 [B]");
  }

  const int batch = static_cast<int>(q_group.size(0));
  const int kv_heads = static_cast<int>(q_group.size(1));
  const int groups = batch * kv_heads;
  const int active_heads = static_cast<int>(q_group.size(2));
  const int max_pages = static_cast<int>(page_table.size(1));
  TORCH_CHECK(max_pages % 4 == 0,
              "page-16 tables must contain a multiple of four pages");
  const int kv_len = max_pages * 16;
  const int logical_tiles = max_pages / 4;
  TORCH_CHECK(num_splits > 0 && num_splits <= logical_tiles,
              "num_splits must be in [1,max_pages/4]");
  TORCH_CHECK(partial_o.sizes() == torch::IntArrayRef(
                  {groups, num_splits, kBlockN, kHeadDim}),
              "partial_o must have shape [B*Hkv,num_splits,8,D]");
  TORCH_CHECK(partial_lse.sizes() == torch::IntArrayRef(
                  {groups, num_splits, kBlockN}),
              "partial_lse must have shape [B*Hkv,num_splits,8]");
  TORCH_CHECK(output.sizes() == torch::IntArrayRef(
                  {groups, active_heads, kHeadDim}),
              "output must have shape [B*Hkv,4|8,D]");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 partial_grid(groups * static_cast<int>(num_splits));
  const dim3 partial_block(128);
  streamattn_transposed_wgmma_exact_partial_kernel<
      16, kVariableLength, true><<<partial_grid, partial_block, 0, stream>>>(
      reinterpret_cast<const Element*>(q_group.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(k_pages.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(v_pages.data_ptr<at::BFloat16>()),
      partial_o.data_ptr<float>(),
      partial_lse.data_ptr<float>(),
      groups,
      kv_len,
      static_cast<int>(num_splits),
      active_heads,
      page_table.data_ptr<int>(),
      max_pages,
      kv_heads,
      kVariableLength ? sequence_lengths->data_ptr<int>() : nullptr);

  const dim3 merge_grid(groups * active_heads);
  const dim3 merge_block(32);
  streamattn_transposed_wgmma_exact_merge_warp_kernel<<<
      merge_grid, merge_block, 0, stream>>>(
      partial_o.data_ptr<float>(),
      partial_lse.data_ptr<float>(),
      reinterpret_cast<Element*>(output.data_ptr<at::BFloat16>()),
      groups,
      static_cast<int>(num_splits),
      active_heads);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void streamattn_transposed_wgmma_paged_fragmented_nhd_exact_decode_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor page_table,
    torch::Tensor partial_o,
    torch::Tensor partial_lse,
    torch::Tensor output,
    int64_t num_splits) {
  streamattn_transposed_wgmma_paged_fragmented_nhd_impl<false>(
      q_group, k_pages, v_pages, page_table, nullptr, partial_o, partial_lse,
      output, num_splits);
}

void streamattn_transposed_wgmma_paged_fragmented_nhd_ragged_exact_decode_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor page_table,
    torch::Tensor sequence_lengths,
    torch::Tensor partial_o,
    torch::Tensor partial_lse,
    torch::Tensor output,
    int64_t num_splits) {
  streamattn_transposed_wgmma_paged_fragmented_nhd_impl<true>(
      q_group, k_pages, v_pages, page_table, &sequence_lengths, partial_o,
      partial_lse, output, num_splits);
}

enum StreamAttnRoutePrepareError : int {
  kRoutePrepareInvalidAtom = 1 << 0,
  kRoutePrepareDuplicateAtom = 1 << 1,
  kRoutePrepareInvalidPage = 1 << 2,
  kRoutePrepareOverflow = 1 << 3,
  kRoutePrepareEmptyHead = 1 << 4,
};

__global__ __launch_bounds__(128)
void streamattn_prepare_qhead_paged_routes64_kernel(
    const int* __restrict__ source_row_ptr,
    const int* __restrict__ source_atom_ids,
    const int* __restrict__ page_table,
    const int* __restrict__ sequence_lengths,
    int* __restrict__ route_counts,
    int* __restrict__ logical_atom_origins,
    int* __restrict__ physical_page_ids,
    int* __restrict__ atom_valid_masks,
    int* __restrict__ active_head_masks,
    int* __restrict__ token_valid_masks,
    int* __restrict__ route_flags,
    int* __restrict__ route_errors,
    int groups,
    int q_heads,
    int kv_heads,
    int max_pages,
    int num_pages,
    int max_routes_per_group) {
  const int group = blockIdx.x;
  if (group >= groups) {
    return;
  }
  extern __shared__ unsigned int atom_head_masks[];
  __shared__ int warp_counts[4];
  __shared__ int warp_offsets[4];
  __shared__ int route_base;
  __shared__ int shared_error;

  const int thread = threadIdx.x;
  const int lane = thread & 31;
  const int warp = thread >> 5;
  const int group_size = q_heads / kv_heads;
  const int batch = group / kv_heads;
  const int kv_head = group - batch * kv_heads;
  const int first_q_row = batch * q_heads + kv_head * group_size;
  const int sequence_length = sequence_lengths[batch];
  const int logical_atoms = max_pages >> 2;
  const int valid_atoms = min(
      logical_atoms, (max(sequence_length, 0) + kBlockM - 1) / kBlockM);

  for (int atom = thread; atom < logical_atoms; atom += blockDim.x) {
    atom_head_masks[atom] = 0;
  }
  if (thread == 0) {
    route_base = 0;
    shared_error = 0;
  }
  __syncthreads();

  // Build a bounded membership map. The source rows are contiguous in CSR, so
  // every thread can own source entries independently while recovering the
  // corresponding local Q head from at most eight row boundaries.
  const int group_begin = source_row_ptr[first_q_row];
  const int group_end = source_row_ptr[first_q_row + group_size];
  for (int head = thread; head < group_size; head += blockDim.x) {
    if (source_row_ptr[first_q_row + head] >=
        source_row_ptr[first_q_row + head + 1]) {
      atomicOr(&shared_error, kRoutePrepareEmptyHead);
    }
  }
  for (int source = group_begin + thread; source < group_end;
       source += blockDim.x) {
    int head = 0;
    while (head + 1 < group_size &&
           source >= source_row_ptr[first_q_row + head + 1]) {
      ++head;
    }
    const int atom = source_atom_ids[source];
    if (atom < 0 || atom >= valid_atoms) {
      atomicOr(&shared_error, kRoutePrepareInvalidAtom);
    } else {
      const unsigned int head_bit = 1u << head;
      const unsigned int prior = atomicOr(&atom_head_masks[atom], head_bit);
      if ((prior & head_bit) != 0) {
        atomicOr(&shared_error, kRoutePrepareDuplicateAtom);
      }
    }
  }
  __syncthreads();

  // Compact active atoms in increasing logical order. Four warps process 128
  // atoms per round; warp ballots and a tiny block prefix preserve deterministic
  // CSR order without a global sort or host-visible route count.
  for (int atom_base = 0; atom_base < logical_atoms;
       atom_base += blockDim.x) {
    const int atom = atom_base + thread;
    const unsigned int head_mask =
        atom < logical_atoms ? atom_head_masks[atom] : 0;
    const unsigned int ballot = __ballot_sync(0xffffffffu, head_mask != 0);
    if (lane == 0) {
      warp_counts[warp] = __popc(ballot);
    }
    __syncthreads();
    if (warp == 0 && lane < 4) {
      int prefix = 0;
      for (int index = 0; index < lane; ++index) {
        prefix += warp_counts[index];
      }
      warp_offsets[lane] = prefix;
    }
    __syncthreads();

    if (head_mask != 0) {
      const unsigned int lower_lanes = lane == 0 ? 0 : ((1u << lane) - 1u);
      const int local_rank = warp_offsets[warp] + __popc(ballot & lower_lanes);
      const int route_index = route_base + local_rank;
      if (route_index >= max_routes_per_group) {
        atomicOr(&shared_error, kRoutePrepareOverflow);
      } else {
        const int route = group * max_routes_per_group + route_index;
        const int64_t logical_start = static_cast<int64_t>(atom) * kBlockM;
        int valid_mask = 0;
        bool all_heads = true;
        bool token_full = true;
        const unsigned int full_head_mask = (1u << group_size) - 1u;
        for (int fragment = 0; fragment < 4; ++fragment) {
          const int64_t logical_origin = logical_start + fragment * 16;
          const int metadata_index = route * 4 + fragment;
          int valid_tokens = sequence_length - static_cast<int>(logical_origin);
          valid_tokens = max(0, min(16, valid_tokens));
          int physical_page = -1;
          int token_mask = 0;
          unsigned int fragment_head_mask = 0;
          if (valid_tokens > 0) {
            const int logical_page = static_cast<int>(logical_origin >> 4);
            if (logical_page < 0 || logical_page >= max_pages) {
              atomicOr(&shared_error, kRoutePrepareInvalidPage);
            } else {
              physical_page = page_table[batch * max_pages + logical_page];
              if (physical_page < 0 || physical_page >= num_pages) {
                atomicOr(&shared_error, kRoutePrepareInvalidPage);
                physical_page = -1;
              } else {
                token_mask = valid_tokens == 16
                    ? 0xffff
                    : (1 << valid_tokens) - 1;
                fragment_head_mask = head_mask;
                valid_mask |= 1 << fragment;
              }
            }
          }
          logical_atom_origins[metadata_index] =
              physical_page >= 0 ? static_cast<int>(logical_origin) : -1;
          physical_page_ids[metadata_index] = physical_page;
          active_head_masks[metadata_index] =
              static_cast<int>(fragment_head_mask);
          token_valid_masks[metadata_index] = token_mask;
          all_heads = all_heads &&
              (physical_page < 0 || fragment_head_mask == full_head_mask);
          token_full = token_full &&
              (physical_page < 0 || token_mask == 0xffff);
        }
        atom_valid_masks[route] = valid_mask;
        route_flags[route] =
            (valid_mask == 0xf ? 1 : 0) + (all_heads ? 2 : 0) +
            (token_full ? 4 : 0);
      }
    }
    __syncthreads();
    if (thread == 0) {
      route_base += warp_counts[0] + warp_counts[1] +
                    warp_counts[2] + warp_counts[3];
    }
    __syncthreads();
  }

  if (thread == 0) {
    route_counts[group] = min(route_base, max_routes_per_group);
    route_errors[group] = shared_error;
  }
}

void streamattn_prepare_qhead_paged_routes64_out_cuda(
    torch::Tensor source_row_ptr,
    torch::Tensor source_atom_ids,
    torch::Tensor page_table,
    torch::Tensor sequence_lengths,
    torch::Tensor route_counts,
    torch::Tensor logical_atom_origins,
    torch::Tensor physical_page_ids,
    torch::Tensor atom_valid_masks,
    torch::Tensor active_head_masks,
    torch::Tensor token_valid_masks,
    torch::Tensor route_flags,
    torch::Tensor route_errors,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t num_pages,
    int64_t max_routes_per_group) {
  TORCH_CHECK(source_row_ptr.is_cuda() && source_atom_ids.is_cuda() &&
              page_table.is_cuda() && sequence_lengths.is_cuda() &&
              route_counts.is_cuda() && logical_atom_origins.is_cuda() &&
              physical_page_ids.is_cuda() && atom_valid_masks.is_cuda() &&
              active_head_masks.is_cuda() && token_valid_masks.is_cuda() &&
              route_flags.is_cuda() && route_errors.is_cuda(),
              "all route preparation tensors must be CUDA tensors");
  TORCH_CHECK(source_row_ptr.scalar_type() == at::ScalarType::Int &&
              source_atom_ids.scalar_type() == at::ScalarType::Int &&
              page_table.scalar_type() == at::ScalarType::Int &&
              sequence_lengths.scalar_type() == at::ScalarType::Int &&
              route_counts.scalar_type() == at::ScalarType::Int &&
              logical_atom_origins.scalar_type() == at::ScalarType::Int &&
              physical_page_ids.scalar_type() == at::ScalarType::Int &&
              atom_valid_masks.scalar_type() == at::ScalarType::Int &&
              active_head_masks.scalar_type() == at::ScalarType::Int &&
              token_valid_masks.scalar_type() == at::ScalarType::Int &&
              route_flags.scalar_type() == at::ScalarType::Int &&
              route_errors.scalar_type() == at::ScalarType::Int,
              "route preparation tensors must use int32");
  TORCH_CHECK(source_row_ptr.is_contiguous() && source_atom_ids.is_contiguous() &&
              page_table.is_contiguous() && sequence_lengths.is_contiguous() &&
              route_counts.is_contiguous() && logical_atom_origins.is_contiguous() &&
              physical_page_ids.is_contiguous() && atom_valid_masks.is_contiguous() &&
              active_head_masks.is_contiguous() && token_valid_masks.is_contiguous() &&
              route_flags.is_contiguous() && route_errors.is_contiguous(),
              "route preparation tensors must be contiguous");
  TORCH_CHECK(q_heads > 0 && kv_heads > 0 && q_heads % kv_heads == 0 &&
              (q_heads / kv_heads == 4 || q_heads / kv_heads == 8),
              "dynamic Q-head lowering requires G4 or G8");
  TORCH_CHECK(num_pages > 0, "num_pages must be positive");
  const int batch = static_cast<int>(sequence_lengths.numel());
  const int groups = batch * static_cast<int>(kv_heads);
  TORCH_CHECK(source_row_ptr.dim() == 1 &&
              source_row_ptr.numel() == batch * q_heads + 1,
              "source_row_ptr must have shape [B*Hq+1]");
  TORCH_CHECK(source_atom_ids.dim() == 1,
              "source_atom_ids must be one-dimensional");
  TORCH_CHECK(page_table.dim() == 2 && page_table.size(0) == batch,
              "page_table must have shape [B,max_pages]");
  TORCH_CHECK(page_table.size(1) % 4 == 0,
              "page-16 route preparation requires max_pages divisible by four");
  TORCH_CHECK(max_routes_per_group > 0 && max_routes_per_group <= 512,
              "max_routes_per_group must be in [1,512]");
  TORCH_CHECK(route_counts.sizes() == torch::IntArrayRef({groups}) &&
              route_errors.sizes() == torch::IntArrayRef({groups}),
              "route counts/errors must have shape [B*Hkv]");
  TORCH_CHECK(logical_atom_origins.sizes() == torch::IntArrayRef(
                  {groups, max_routes_per_group, 4}) &&
              physical_page_ids.sizes() == logical_atom_origins.sizes() &&
              active_head_masks.sizes() == logical_atom_origins.sizes() &&
              token_valid_masks.sizes() == logical_atom_origins.sizes(),
              "route atom metadata must have shape [B*Hkv,max_routes,4]");
  TORCH_CHECK(atom_valid_masks.sizes() == torch::IntArrayRef(
                  {groups, max_routes_per_group}) &&
              route_flags.sizes() == atom_valid_masks.sizes(),
              "route masks/flags must have shape [B*Hkv,max_routes]");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int logical_atoms = static_cast<int>(page_table.size(1) / 4);
  const size_t shared_bytes =
      static_cast<size_t>(logical_atoms) * sizeof(unsigned int);
  TORCH_CHECK(shared_bytes <= 48 * 1024,
              "dynamic route membership map exceeds 48 KiB shared memory");
  streamattn_prepare_qhead_paged_routes64_kernel<<<
      groups, 128, shared_bytes, stream>>>(
      source_row_ptr.data_ptr<int>(),
      source_atom_ids.data_ptr<int>(),
      page_table.data_ptr<int>(),
      sequence_lengths.data_ptr<int>(),
      route_counts.data_ptr<int>(),
      logical_atom_origins.data_ptr<int>(),
      physical_page_ids.data_ptr<int>(),
      atom_valid_masks.data_ptr<int>(),
      active_head_masks.data_ptr<int>(),
      token_valid_masks.data_ptr<int>(),
      route_flags.data_ptr<int>(),
      route_errors.data_ptr<int>(),
      groups,
      static_cast<int>(q_heads),
      static_cast<int>(kv_heads),
      static_cast<int>(page_table.size(1)),
      static_cast<int>(num_pages),
      static_cast<int>(max_routes_per_group));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool kNHD>
void streamattn_transposed_wgmma_paged_selected_fragmented_impl(
    torch::Tensor q_group,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor route_row_ptr,
    torch::Tensor physical_page_ids,
    torch::Tensor active_head_masks,
    torch::Tensor token_valid_masks,
    torch::Tensor partial_o,
    torch::Tensor partial_lse,
    torch::Tensor output,
    int64_t max_routes_per_row) {
  TORCH_CHECK(q_group.is_cuda() && k_pages.is_cuda() && v_pages.is_cuda() &&
              route_row_ptr.is_cuda() && physical_page_ids.is_cuda() &&
              active_head_masks.is_cuda() && token_valid_masks.is_cuda() &&
              partial_o.is_cuda() && partial_lse.is_cuda() && output.is_cuda(),
              "all tensors must be CUDA tensors");
  TORCH_CHECK(q_group.is_contiguous() && k_pages.is_contiguous() &&
              v_pages.is_contiguous() && route_row_ptr.is_contiguous() &&
              physical_page_ids.is_contiguous() &&
              active_head_masks.is_contiguous() &&
              token_valid_masks.is_contiguous() && partial_o.is_contiguous() &&
              partial_lse.is_contiguous() && output.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(q_group.scalar_type() == at::ScalarType::BFloat16 &&
              k_pages.scalar_type() == at::ScalarType::BFloat16 &&
              v_pages.scalar_type() == at::ScalarType::BFloat16,
              "q_group and K/V pages must be bf16");
  TORCH_CHECK(route_row_ptr.scalar_type() == at::ScalarType::Int &&
              physical_page_ids.scalar_type() == at::ScalarType::Int &&
              active_head_masks.scalar_type() == at::ScalarType::Int &&
              token_valid_masks.scalar_type() == at::ScalarType::Int,
              "selected route metadata must be int32");
  TORCH_CHECK(partial_o.scalar_type() == at::ScalarType::Float &&
              partial_lse.scalar_type() == at::ScalarType::Float,
              "partial outputs must be fp32");
  TORCH_CHECK(output.scalar_type() == at::ScalarType::BFloat16,
              "output must be bf16");
  TORCH_CHECK(q_group.dim() == 4 &&
              (q_group.size(2) == 4 || q_group.size(2) == kBlockN) &&
              q_group.size(3) == kHeadDim,
              "q_group must have shape [B,Hkv,4|8,D]");
  TORCH_CHECK(k_pages.sizes() == v_pages.sizes(),
              "K/V page tensors must match");
  if constexpr (kNHD) {
    TORCH_CHECK(k_pages.dim() == 4 && k_pages.size(1) == 16 &&
                k_pages.size(2) == q_group.size(1) &&
                k_pages.size(3) == kHeadDim,
                "NHD K/V pages must have shape [pages,16,Hkv,D]");
  } else {
    TORCH_CHECK(k_pages.dim() == 4 && k_pages.size(1) == q_group.size(1) &&
                k_pages.size(2) == 16 && k_pages.size(3) == kHeadDim,
                "HND K/V pages must have shape [pages,Hkv,16,D]");
  }

  const int batch = static_cast<int>(q_group.size(0));
  const int kv_heads = static_cast<int>(q_group.size(1));
  const int groups = batch * kv_heads;
  const int active_heads = static_cast<int>(q_group.size(2));
  const int routes = static_cast<int>(physical_page_ids.size(0));
  TORCH_CHECK(max_routes_per_row > 0,
              "max_routes_per_row must be positive");
  TORCH_CHECK(route_row_ptr.dim() == 1 && route_row_ptr.size(0) == groups + 1,
              "route_row_ptr must have shape [B*Hkv+1]");
  TORCH_CHECK(physical_page_ids.dim() == 2 &&
              physical_page_ids.size(1) == 4,
              "physical_page_ids must have shape [routes,4]");
  TORCH_CHECK(active_head_masks.sizes() == physical_page_ids.sizes() &&
              token_valid_masks.sizes() == physical_page_ids.sizes(),
              "selected masks must have shape [routes,4]");
  TORCH_CHECK(partial_o.sizes() == torch::IntArrayRef(
                  {groups, max_routes_per_row, kBlockN, kHeadDim}),
              "partial_o must have shape [B*Hkv,max_routes,8,D]");
  TORCH_CHECK(partial_lse.sizes() == torch::IntArrayRef(
                  {groups, max_routes_per_row, kBlockN}),
              "partial_lse must have shape [B*Hkv,max_routes,8]");
  TORCH_CHECK(output.sizes() == torch::IntArrayRef(
                  {groups, active_heads, kHeadDim}),
              "output must have shape [B*Hkv,4|8,D]");
  TORCH_CHECK(routes > 0, "selected route set must be non-empty");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 partial_grid(groups * static_cast<int>(max_routes_per_row));
  const dim3 partial_block(128);
  streamattn_transposed_wgmma_exact_partial_kernel<
      16, false, kNHD, true><<<partial_grid, partial_block, 0, stream>>>(
      reinterpret_cast<const Element*>(q_group.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(k_pages.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(v_pages.data_ptr<at::BFloat16>()),
      partial_o.data_ptr<float>(),
      partial_lse.data_ptr<float>(),
      groups,
      64,
      static_cast<int>(max_routes_per_row),
      active_heads,
      nullptr,
      0,
      kv_heads,
      nullptr,
      route_row_ptr.data_ptr<int>(),
      physical_page_ids.data_ptr<int>(),
      active_head_masks.data_ptr<int>(),
      token_valid_masks.data_ptr<int>());

  const dim3 merge_grid(groups * active_heads);
  const dim3 merge_block(32);
  streamattn_transposed_wgmma_exact_merge_warp_kernel<<<
      merge_grid, merge_block, 0, stream>>>(
      partial_o.data_ptr<float>(),
      partial_lse.data_ptr<float>(),
      reinterpret_cast<Element*>(output.data_ptr<at::BFloat16>()),
      groups,
      static_cast<int>(max_routes_per_row),
      active_heads);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void streamattn_transposed_wgmma_paged_selected_fragmented_exact_decode_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor route_row_ptr,
    torch::Tensor physical_page_ids,
    torch::Tensor active_head_masks,
    torch::Tensor token_valid_masks,
    torch::Tensor partial_o,
    torch::Tensor partial_lse,
    torch::Tensor output,
    int64_t max_routes_per_row) {
  streamattn_transposed_wgmma_paged_selected_fragmented_impl<false>(
      q_group, k_pages, v_pages, route_row_ptr, physical_page_ids,
      active_head_masks, token_valid_masks, partial_o, partial_lse, output,
      max_routes_per_row);
}

void streamattn_transposed_wgmma_paged_selected_fragmented_nhd_exact_decode_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor route_row_ptr,
    torch::Tensor physical_page_ids,
    torch::Tensor active_head_masks,
    torch::Tensor token_valid_masks,
    torch::Tensor partial_o,
    torch::Tensor partial_lse,
    torch::Tensor output,
    int64_t max_routes_per_row) {
  streamattn_transposed_wgmma_paged_selected_fragmented_impl<true>(
      q_group, k_pages, v_pages, route_row_ptr, physical_page_ids,
      active_head_masks, token_valid_masks, partial_o, partial_lse, output,
      max_routes_per_row);
}

template <bool kNHD>
void streamattn_transposed_wgmma_paged_dynamic_qhead_fragmented_impl(
    torch::Tensor q_group,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor page_table,
    torch::Tensor sequence_lengths,
    torch::Tensor source_row_ptr,
    torch::Tensor source_atom_ids,
    torch::Tensor route_counts,
    torch::Tensor logical_atom_origins,
    torch::Tensor physical_page_ids,
    torch::Tensor atom_valid_masks,
    torch::Tensor active_head_masks,
    torch::Tensor token_valid_masks,
    torch::Tensor route_flags,
    torch::Tensor route_errors,
    torch::Tensor partial_o,
    torch::Tensor partial_lse,
    torch::Tensor output,
    int64_t max_routes_per_group) {
  TORCH_CHECK(q_group.is_cuda() && k_pages.is_cuda() && v_pages.is_cuda() &&
              partial_o.is_cuda() && partial_lse.is_cuda() && output.is_cuda(),
              "dynamic selected decode tensors must be CUDA tensors");
  TORCH_CHECK(q_group.is_contiguous() && k_pages.is_contiguous() &&
              v_pages.is_contiguous() && partial_o.is_contiguous() &&
              partial_lse.is_contiguous() && output.is_contiguous(),
              "dynamic selected decode tensors must be contiguous");
  TORCH_CHECK(q_group.scalar_type() == at::ScalarType::BFloat16 &&
              k_pages.scalar_type() == at::ScalarType::BFloat16 &&
              v_pages.scalar_type() == at::ScalarType::BFloat16 &&
              output.scalar_type() == at::ScalarType::BFloat16,
              "dynamic selected Q/K/V/output must be bf16");
  TORCH_CHECK(partial_o.scalar_type() == at::ScalarType::Float &&
              partial_lse.scalar_type() == at::ScalarType::Float,
              "dynamic selected partials must be fp32");
  TORCH_CHECK(q_group.dim() == 4 &&
              (q_group.size(2) == 4 || q_group.size(2) == kBlockN) &&
              q_group.size(3) == kHeadDim,
              "q_group must have shape [B,Hkv,4|8,D]");
  TORCH_CHECK(k_pages.sizes() == v_pages.sizes(), "K/V pages must match");
  if constexpr (kNHD) {
    TORCH_CHECK(k_pages.dim() == 4 && k_pages.size(1) == 16 &&
                k_pages.size(2) == q_group.size(1) &&
                k_pages.size(3) == kHeadDim,
                "NHD K/V pages must have shape [pages,16,Hkv,D]");
  } else {
    TORCH_CHECK(k_pages.dim() == 4 && k_pages.size(1) == q_group.size(1) &&
                k_pages.size(2) == 16 && k_pages.size(3) == kHeadDim,
                "HND K/V pages must have shape [pages,Hkv,16,D]");
  }

  const int batch = static_cast<int>(q_group.size(0));
  const int kv_heads = static_cast<int>(q_group.size(1));
  const int active_heads = static_cast<int>(q_group.size(2));
  const int q_heads = kv_heads * active_heads;
  const int groups = batch * kv_heads;
  TORCH_CHECK(partial_o.sizes() == torch::IntArrayRef(
                  {groups, max_routes_per_group, kBlockN, kHeadDim}) &&
              partial_lse.sizes() == torch::IntArrayRef(
                  {groups, max_routes_per_group, kBlockN}),
              "dynamic partial workspace shape mismatch");
  TORCH_CHECK(output.sizes() == torch::IntArrayRef(
                  {groups, active_heads, kHeadDim}),
              "dynamic output must have shape [B*Hkv,4|8,D]");

  streamattn_prepare_qhead_paged_routes64_out_cuda(
      source_row_ptr,
      source_atom_ids,
      page_table,
      sequence_lengths,
      route_counts,
      logical_atom_origins,
      physical_page_ids,
      atom_valid_masks,
      active_head_masks,
      token_valid_masks,
      route_flags,
      route_errors,
      q_heads,
      kv_heads,
      k_pages.size(0),
      max_routes_per_group);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 partial_grid(groups * static_cast<int>(max_routes_per_group));
  streamattn_transposed_wgmma_exact_partial_kernel<
      16, false, kNHD, true, true><<<partial_grid, 128, 0, stream>>>(
      reinterpret_cast<const Element*>(q_group.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(k_pages.data_ptr<at::BFloat16>()),
      reinterpret_cast<const Element*>(v_pages.data_ptr<at::BFloat16>()),
      partial_o.data_ptr<float>(),
      partial_lse.data_ptr<float>(),
      groups,
      64,
      static_cast<int>(max_routes_per_group),
      active_heads,
      nullptr,
      0,
      kv_heads,
      nullptr,
      nullptr,
      physical_page_ids.data_ptr<int>(),
      active_head_masks.data_ptr<int>(),
      token_valid_masks.data_ptr<int>(),
      route_counts.data_ptr<int>());

  streamattn_transposed_wgmma_selected_row_local_merge_warp_kernel<<<
      groups * active_heads, 32, 0, stream>>>(
      partial_o.data_ptr<float>(),
      partial_lse.data_ptr<float>(),
      route_counts.data_ptr<int>(),
      reinterpret_cast<Element*>(output.data_ptr<at::BFloat16>()),
      groups,
      static_cast<int>(max_routes_per_group),
      active_heads);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void streamattn_transposed_wgmma_paged_dynamic_qhead_fragmented_exact_decode_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor page_table,
    torch::Tensor sequence_lengths,
    torch::Tensor source_row_ptr,
    torch::Tensor source_atom_ids,
    torch::Tensor route_counts,
    torch::Tensor logical_atom_origins,
    torch::Tensor physical_page_ids,
    torch::Tensor atom_valid_masks,
    torch::Tensor active_head_masks,
    torch::Tensor token_valid_masks,
    torch::Tensor route_flags,
    torch::Tensor route_errors,
    torch::Tensor partial_o,
    torch::Tensor partial_lse,
    torch::Tensor output,
    int64_t max_routes_per_group) {
  streamattn_transposed_wgmma_paged_dynamic_qhead_fragmented_impl<false>(
      q_group, k_pages, v_pages, page_table, sequence_lengths,
      source_row_ptr, source_atom_ids, route_counts, logical_atom_origins,
      physical_page_ids, atom_valid_masks, active_head_masks,
      token_valid_masks, route_flags, route_errors, partial_o, partial_lse,
      output, max_routes_per_group);
}

void streamattn_transposed_wgmma_paged_dynamic_qhead_fragmented_nhd_exact_decode_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor page_table,
    torch::Tensor sequence_lengths,
    torch::Tensor source_row_ptr,
    torch::Tensor source_atom_ids,
    torch::Tensor route_counts,
    torch::Tensor logical_atom_origins,
    torch::Tensor physical_page_ids,
    torch::Tensor atom_valid_masks,
    torch::Tensor active_head_masks,
    torch::Tensor token_valid_masks,
    torch::Tensor route_flags,
    torch::Tensor route_errors,
    torch::Tensor partial_o,
    torch::Tensor partial_lse,
    torch::Tensor output,
    int64_t max_routes_per_group) {
  streamattn_transposed_wgmma_paged_dynamic_qhead_fragmented_impl<true>(
      q_group, k_pages, v_pages, page_table, sequence_lengths,
      source_row_ptr, source_atom_ids, route_counts, logical_atom_origins,
      physical_page_ids, atom_valid_masks, active_head_masks,
      token_valid_masks, route_flags, route_errors, partial_o, partial_lse,
      output, max_routes_per_group);
}
"""


def cuda_source_for_head_dim(head_dim: int) -> str:
    """Return a separately compiled source specialization for D64 or D128."""

    if head_dim == 64:
        return CUDA_SOURCE
    if head_dim == 128:
        return CUDA_SOURCE.replace(
            "static constexpr int kHeadDim = 64;",
            "static constexpr int kHeadDim = 128;",
            1,
        )
    raise ValueError("SM90 exact source supports head_dim 64 or 128")
