"""Compact planned task launch for the retained exact natural producer."""

from .micro_prefill_paged_sources import CPP_SOURCE as PAGED_CPP, paged_cuda_source
from .micro_prefill_semantics_sources import _once, _between

_OLD = "int64_t splits, bool natural, bool nhd"
_NEW = "int64_t splits, bool natural, bool nhd, torch::Tensor tasks, torch::Tensor counts"
CPP_SOURCE = _once(PAGED_CPP, _OLD, _NEW)
_DECL = CPP_SOURCE.split("void micro_semantics_out(")[1].split(");", 1)[0]
CPP_SOURCE = _once(CPP_SOURCE, "PYBIND11_MODULE",
    "void micro_components_out(" + _DECL + ", int64_t component);\n"
    "std::vector<int64_t> micro_attributes(torch::Tensor q, bool nhd);\nPYBIND11_MODULE")
CPP_SOURCE = _once(CPP_SOURCE, '  m.def("out", &micro_semantics_out);',
    '  m.def("out", &micro_semantics_out);\n'
    '  m.def("components", &micro_components_out);\n'
    '  m.def("attributes", &micro_attributes);')


_PAGE_PAIR_LOADER = r"""
template <bool kNHD, bool kTranspose, class SmemTensor>
__forceinline__ __device__ void streamattn_micro_load_page16_pair(
    const Element* base, const int* table, int cache_group, int tile,
    int max_pages, int kv_heads, int length, SmemTensor destination) {
  static_assert(kHeadDim == 128, "page-pair reuse requires D128");
  const int batch = cache_group / kv_heads, head = cache_group % kv_heads;
  const int row_in_half = threadIdx.x / 16, dim = (threadIdx.x % 16) * 8;
  const int64_t half_page_stride = static_cast<int64_t>(8) * kHeadDim * (kNHD ? kv_heads : 1);
  CUTE_UNROLL
  for (int fragment = 0; fragment < 4; ++fragment) {
    const int token0 = fragment * 16 + row_in_half, token1 = token0 + 8;
    const int logical0 = tile * 64 + token0;
    const bool valid0 = logical0 < length, valid1 = logical0 + 8 < length;
    const Element* source0 = base;
    const Element* source1 = base;
    // A valid second copy implies a valid first copy on this same live page.
    if (valid0) {
      const int page = table[static_cast<int64_t>(batch) * max_pages + tile * 4 + fragment];
      const int64_t row = kNHD
          ? (static_cast<int64_t>(page) * 16 + row_in_half) * kv_heads + head
          : (static_cast<int64_t>(page) * kv_heads + head) * 16 + row_in_half;
      source0 = base + row * kHeadDim + dim;
      if (valid1) source1 = source0 + half_page_stride;
    }
    Element* target0;
    Element* target1;
    if constexpr (kTranspose) {
      target0 = &destination(dim, token0);
      target1 = &destination(dim, token1);
    } else {
      target0 = &destination(token0, dim);
      target1 = &destination(token1, dim);
    }
    cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>::copy(
        *reinterpret_cast<const cute::uint128_t*>(source0),
        *reinterpret_cast<cute::uint128_t*>(target0), valid0);
    cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>::copy(
        *reinterpret_cast<const cute::uint128_t*>(source1),
        *reinterpret_cast<cute::uint128_t*>(target1), valid1);
  }
}
"""


def ragged_cuda_source(head_dim, dtype, causal, affine_mode="none", q_vector_copy=False,
                       page_pair_reuse=False, unsigned_page_address=False):
    if affine_mode not in ("none", "index", "interior") or (affine_mode != "none" and not causal):
        raise ValueError("affine modes require causal attention")
    source = paged_cuda_source(head_dim, dtype, causal)
    if not isinstance(page_pair_reuse, bool) or (page_pair_reuse and not q_vector_copy):
        raise ValueError("page-pair experiment requires vector Q control")
    if not isinstance(unsigned_page_address, bool) or (unsigned_page_address and not page_pair_reuse):
        raise ValueError("unsigned page-address experiment requires page-pair control")
    begin = "\ntemplate <bool kNHD>\n__global__ __launch_bounds__(128)\nvoid streamattn_natural_wgmma_micro_prefill_partial_kernel("
    end = "\n__global__ __launch_bounds__(128)\nvoid streamattn_natural_wgmma_micro_prefill_merge_kernel("
    producer = _between(source, begin, end)
    old = producer
    producer = _once(producer, "const int* query_lengths, int max_pages) {",
                     "const int* query_lengths, int max_pages, const int* tasks) {")
    producer = _once(producer,
        "  const int work = blockIdx.x;\n  const int work_group = work / num_splits;\n"
        "  const int split = work - work_group * num_splits;",
        "  const int* task = tasks + static_cast<int64_t>(blockIdx.x) * 4;\n"
        "  const int work_group = task[0], split = task[1];\n"
        "  const int work = work_group * num_splits + split;")
    producer = _once(producer,
        "const int tile_begin = static_cast<int64_t>(split) * num_kv_tiles / num_splits;",
        "const int tile_begin = task[2];")
    producer = _once(producer,
        "const int tile_end = static_cast<int64_t>(split + 1) * num_kv_tiles / num_splits;",
        "const int tile_end = task[3];")
    if affine_mode != "none":
        producer = _once(producer,
            "(kPositionCausal &&\n"
            "               key_positions[static_cast<int64_t>(batch) * kv_length + ki] >\n"
            "               query_positions[static_cast<int64_t>(batch) * query_length + qi])",
            "ki > sequence_length - valid_queries + qi")
    if affine_mode == "interior":
        producer = _once(producer,
            "      {\n        CUTE_UNROLL\n        for (int col = 0; col < size<1>(score_rows); ++col)",
            "      if (!(query_begin + query_positions_per_tile <= valid_queries &&\n"
            "            tile * kBlockM + kBlockM - 1 <= sequence_length - valid_queries + query_begin)) {\n"
            "        CUTE_UNROLL\n        for (int col = 0; col < size<1>(score_rows); ++col)")
    if q_vector_copy:
        first = "  for (int idx = threadIdx.x; idx < kQueryRows * kHeadDim; idx += 128) {"
        last = "  cutlass::arch::fence_view_async_shared();"
        scalar = _between(producer, first, last)
        vector = r"""  // SW128 repeats after eight rows; tiling adds aligned atom offsets.
  static_assert([]() constexpr {
    constexpr auto layout = PrefillSmemLayoutQ{};
    for (int row = 0; row < 8; ++row) {
      for (int dim = 0; dim < kHeadDim; dim += 8) {
        const int base = layout(make_coord(row, dim));
        if (base % 8 != 0) return false;
        for (int i = 1; i < 8; ++i)
          if (layout(make_coord(row, dim + i)) != base + i) return false;
      }
    }
    return true;
  }(), "Q shared layout must preserve aligned 16-byte vectors");
  for (int idx = threadIdx.x * 8; idx < kQueryRows * kHeadDim; idx += 128 * 8) {
    const int local_query_row = idx / kHeadDim;
    const int dim = idx - local_query_row * kHeadDim;
    const int query_offset = local_query_row / group_size;
    const int head_offset = local_query_row - query_offset * group_size;
    const int query_position = query_begin + query_offset;
    const int q_head = kv_head * group_size + head_offset;
    uint4 item = make_uint4(0, 0, 0, 0);
    if (query_position < valid_queries) {
      const int64_t source =
          ((static_cast<int64_t>(batch) * query_length + query_position) * q_heads + q_head)
              * kHeadDim + dim;
      item = *reinterpret_cast<const uint4*>(query + source);
    }
    *reinterpret_cast<uint4*>(&sQ(local_query_row, dim)) = item;
  }
"""
        producer = _once(producer, scalar, vector)
    if page_pair_reuse and head_dim == 128:
        for transpose in ("false", "true"):
            producer = _once(producer,
                f"streamattn_micro_load_page16<kNHD, {transpose}, false>",
                f"streamattn_micro_load_page16_pair<kNHD, {transpose}>")
        loader = _PAGE_PAIR_LOADER
        if unsigned_page_address:
            loader = _once(loader, """      const int64_t row = kNHD
          ? (static_cast<int64_t>(page) * 16 + row_in_half) * kv_heads + head
          : (static_cast<int64_t>(page) * kv_heads + head) * 16 + row_in_half;""", """      // Active page IDs and head counts are nonnegative; widen BEFORE multiplication.
      const uint64_t page_heads = static_cast<uint64_t>(static_cast<uint32_t>(page))
          * static_cast<uint32_t>(kv_heads);
      const uint64_t row = kNHD
          ? page_heads * 16 + static_cast<uint64_t>(row_in_half)
              * static_cast<uint32_t>(kv_heads) + static_cast<uint32_t>(head)
          : (page_heads + static_cast<uint32_t>(head)) * 16 + row_in_half;""")
        producer = loader + producer
    source = _once(source, old, producer)
    merge = _between(source, end, "\ntemplate <int kPagedPageSize>\n")
    old = merge
    merge = _once(merge, "    int num_splits) {",
                  "    int num_splits, const int* counts, const int* query_lengths) {")
    merge = _once(merge, "  const int batch = batch_query / query_length;",
        "  const int batch = batch_query / query_length;\n"
        "  const int active_splits = counts[batch];\n"
        "  if (query_position >= query_lengths[batch] || active_splits == 0) {\n"
        "    if (threadIdx.x < kHeadDim)\n"
        "      output[static_cast<int64_t>(output_row) * kHeadDim + threadIdx.x] = Element(0.0f);\n"
        "    return;\n  }")
    merge = merge.replace("split < num_splits", "split < active_splits")
    source = _once(source, old, merge)
    source = _once(source, _OLD, _NEW)
    source = _once(source, "  auto stream=at::cuda::getCurrentCUDAStream();", r'''
  TORCH_CHECK(natural, "compact tasks require the natural family");
  TORCH_CHECK(tasks.device()==q.device() && counts.device()==q.device() &&
              tasks.is_contiguous() && counts.is_contiguous() &&
              tasks.scalar_type()==at::kInt && counts.scalar_type()==at::kInt &&
              tasks.dim()==2 && tasks.size(1)==4 && tasks.size(0)<=INT_MAX &&
              counts.sizes()==torch::IntArrayRef({B}), "invalid compact task metadata");
  for (auto out : {po,pl,o}) {
    at::assert_no_overlap(out,tasks); at::assert_no_overlap(out,counts);
  }
  auto stream=at::cuda::getCurrentCUDAStream();''')
    source = _once(source,
        "    streamattn_natural_wgmma_micro_prefill_partial_kernel<kNHD><<<groups*splits,128,shared,stream>>>(",
        "    if (tasks.size(0)>0) streamattn_natural_wgmma_micro_prefill_partial_kernel<kNHD><<<tasks.size(0),128,shared,stream>>>(")
    source = _once(source, "ql.data_ptr<int>(),pt.size(1));",
                  "ql.data_ptr<int>(),pt.size(1),tasks.data_ptr<int>());")
    source = _once(source, "optr,B,M,H,HK,G,splits);",
                  "optr,B,M,H,HK,G,splits,counts.data_ptr<int>(),ql.data_ptr<int>());")
    # Separate host launches without changing either device kernel or its math.
    source = _once(source, "void micro_semantics_out(", "void micro_components_out(")
    source = _once(source, _NEW + ") {", _NEW + ", int64_t component) {\n"
                   '  TORCH_CHECK(component >= 0 && component <= 2, "invalid component");')
    source = _once(source, "if (tasks.size(0)>0)", "if (component != 2 && tasks.size(0)>0)")
    source = _once(source,
        "    streamattn_natural_wgmma_micro_prefill_merge_kernel<<<",
        "    if (component != 1) streamattn_natural_wgmma_micro_prefill_merge_kernel<<<")
    source += "\nvoid micro_semantics_out(" + _DECL + r''') {
  micro_components_out(q,k,v,po,pl,o,qp,kp,pt,sl,ql,splits,natural,nhd,tasks,counts,0);
}

std::vector<int64_t> micro_attributes(torch::Tensor q, bool nhd) {
  TORCH_CHECK(q.is_cuda(), "query must be CUDA");
  c10::cuda::CUDAGuard guard(q.device());
  std::vector<int64_t> result;
  auto collect = [&](auto kernel, int shared) {
    if (shared) C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
    cudaFuncAttributes attr;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
    int resident = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, 128, shared));
    result.insert(result.end(), {attr.numRegs, static_cast<int64_t>(attr.localSizeBytes),
        static_cast<int64_t>(attr.sharedSizeBytes), shared, resident});
  };
  if (nhd) collect(streamattn_natural_wgmma_micro_prefill_partial_kernel<true>, sizeof(GroupedRSPrefillSharedStorage));
  else collect(streamattn_natural_wgmma_micro_prefill_partial_kernel<false>, sizeof(GroupedRSPrefillSharedStorage));
  collect(streamattn_natural_wgmma_micro_prefill_merge_kernel, 0);
  return result;
}
'''
    return source
