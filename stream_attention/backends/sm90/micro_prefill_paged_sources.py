"""Direct page-16 specializations of the retained micro-prefill producers."""

from .micro_prefill_semantics_sources import CPP_SOURCE as SEMANTIC_CPP
from .micro_prefill_semantics_sources import _between, _once, semantic_cuda_source


_ARGS = "torch::Tensor qp, torch::Tensor kp, int64_t splits, bool natural"
_PAGED_ARGS = (
    "torch::Tensor qp, torch::Tensor kp, torch::Tensor pt, "
    "torch::Tensor sl, torch::Tensor ql, int64_t splits, bool natural, bool nhd"
)
CPP_SOURCE = _once(SEMANTIC_CPP, _ARGS, _PAGED_ARGS)


_LOADER = r"""
// One 16-byte transaction owns eight contiguous feature values. Invalid
// tokens never dereference the page table or K/V, including poisoned tails.
template <bool kNHD, bool kTranspose, bool kPageAlias, class SmemTensor>
__forceinline__ __device__ void streamattn_micro_load_page16(
    const Element* base, const int* table, int cache_group, int tile,
    int max_pages, int kv_heads, int length, SmemTensor destination) {
  const int batch = cache_group / kv_heads, head = cache_group % kv_heads;
  CUTE_UNROLL
  for (int vector = threadIdx.x; vector < 64 * kHeadDim / 8; vector += 128) {
    const int token = vector / (kHeadDim / 8);
    const int dim = (vector % (kHeadDim / 8)) * 8;
    const int logical = tile * 64 + token;
    const bool valid = logical < length;
    const Element* source = base;
    if (valid) {
      const int page = table[static_cast<int64_t>(batch) * max_pages + logical / 16];
      const int offset = logical % 16;
      const int64_t row = kNHD
          ? (static_cast<int64_t>(page) * 16 + offset) * kv_heads + head
          : (static_cast<int64_t>(page) * kv_heads + head) * 16 + offset;
      source = base + row * kHeadDim + dim;
    }
    Element* target;
    if constexpr (kPageAlias) {
      target = &destination(token % 16, dim, token / 16);
    } else if constexpr (kTranspose) {
      target = &destination(dim, token);
    } else {
      target = &destination(token, dim);
    }
    cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>::copy(
        *reinterpret_cast<const cute::uint128_t*>(source),
        *reinterpret_cast<cute::uint128_t*>(target), valid);
  }
}
"""


def paged_cuda_source(head_dim: int, dtype: str, causal: bool) -> str:
    """Preserve MMA/softmax/merge; specialize only addressing and visibility."""
    source = semantic_cuda_source(head_dim, dtype, causal)
    natural_start = (
        "\n__global__ __launch_bounds__(128)\n"
        "void streamattn_natural_wgmma_micro_prefill_partial_kernel("
    )
    natural_end = (
        "\n__global__ __launch_bounds__(128)\n"
        "void streamattn_natural_wgmma_micro_prefill_merge_kernel("
    )
    natural = _between(source, natural_start, natural_end)
    original = natural
    natural = _once(natural, natural_start,
        "\ntemplate <bool kNHD>" + natural_start)
    natural = _once(natural,
        "const int64_t* __restrict__ key_positions) {",
        "const int64_t* __restrict__ key_positions,\n"
        "    const int* page_table, const int* sequence_lengths,\n"
        "    const int* query_lengths, int max_pages) {")
    natural = _once(natural, "const int num_kv_tiles = kv_length / kBlockM;",
        "const int sequence_length = sequence_lengths[batch];\n"
        "  const int valid_queries = query_lengths[batch];\n"
        "  const int num_kv_tiles = (sequence_length + kBlockM - 1) / kBlockM;")
    natural = _once(natural, "split * num_kv_tiles / num_splits;",
        "static_cast<int64_t>(split) * num_kv_tiles / num_splits;")
    natural = _once(natural, "(split + 1) * num_kv_tiles / num_splits;",
        "static_cast<int64_t>(split + 1) * num_kv_tiles / num_splits;")
    natural = _once(natural,
        "  extern __shared__ __align__(128) unsigned char shared_bytes[];", r"""
  // Ragged rows may have fewer tiles than the plan's split capacity.
  if (tile_begin >= tile_end || query_begin >= valid_queries) {
    const int64_t offset = static_cast<int64_t>(work) * kQueryRows;
    for (int i = threadIdx.x; i < kQueryRows * kHeadDim; i += 128)
      partial_o[offset * kHeadDim + i] = 0.0f;
    for (int i = threadIdx.x; i < kQueryRows; i += 128)
      partial_lse[offset + i] = -INFINITY;
    return;
  }
  extern __shared__ __align__(128) unsigned char shared_bytes[];""")
    natural = _once(natural, "if (query_position < query_length) {",
        "if (query_position < valid_queries) {")
    copy_begin = natural.index("  GmemCopyK copy_k;")
    copy_end = natural.index("  copy_k_tile(tile_begin, tK0sK0);")
    natural = natural[:copy_begin] + r"""
  auto copy_k_tile = [&](int tile, auto destination) {
    streamattn_micro_load_page16<kNHD, false, false>(
        key, page_table, batch_kv_group, tile, max_pages, kv_heads,
        sequence_length, destination);
  };
  auto copy_v_tile = [&](int tile) {
    streamattn_micro_load_page16<kNHD, true, false>(
        value, page_table, batch_kv_group, tile, max_pages, kv_heads,
        sequence_length, sV);
  };

""" + natural[copy_end:]
    natural = natural.replace("tK0sK0", "sK0").replace("tK1sK1", "sK1")
    natural = _once(natural, "if constexpr (kPositionCausal) {", "{")
    natural = _once(natural,
        "if (qi >= query_length ||\n"
        "              key_positions[static_cast<int64_t>(batch) * kv_length + ki] >\n"
        "              query_positions[static_cast<int64_t>(batch) * query_length + qi]) {",
        "if (qi >= valid_queries || ki >= sequence_length ||\n"
        "              (kPositionCausal &&\n"
        "               key_positions[static_cast<int64_t>(batch) * kv_length + ki] >\n"
        "               query_positions[static_cast<int64_t>(batch) * query_length + qi])) {")
    source = _once(source, original, _LOADER + natural)

    # Reuse all existing page-16 call sites, replacing the copy implementation
    # only in this extension. Promoted decode's loader is unchanged.
    loader_begin = source.index("__forceinline__ __device__ void streamattn_copy_paged16_tile(")
    body_begin = source.index(") {", loader_begin) + 3
    body_end = source.index("\n}\n", body_begin)
    source = source[:body_begin] + r"""
  streamattn_micro_load_page16<kNHD, false, true>(
      base, page_table, group, tile, max_pages, kv_heads,
      sequence_length, destination);
""" + source[body_end:]
    source = _once(source,
        "const int64_t* __restrict__ key_positions = nullptr) {",
        "const int64_t* __restrict__ key_positions = nullptr,\n"
        "    const int* query_lengths = nullptr) {")
    source = _once(source,
        "if (tile_begin >= tile_end) {",
        "if (tile_begin >= tile_end ||\n"
        "      (group / kv_heads) % query_positions_per_batch >=\n"
        "          query_lengths[cache_group / kv_heads]) {")
    source = _once(source,
        "if (key_positions[static_cast<int64_t>(batch) * kv_len + ki] > qpos) {",
        "if (ki >= sequence_length ||\n"
        "              key_positions[static_cast<int64_t>(batch) * kv_len + ki] > qpos) {")

    source = _once(source, _ARGS, _PAGED_ARGS)
    source = _once(source,
        'TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.sizes() == k.sizes(),',
        'TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.sizes() == k.sizes() &&\n'
        '              pt.dim() == 2,')
    source = _once(source, '"expected Q [B,M,Hq,D] and K/V [B,Hkv,N,D]"',
        '"expected Q [B,M,Hq,D], rank-4 pages and rank-2 page table"')
    source = _once(source, "const int64_t HK=k.size(1), N=k.size(2);",
        "const int64_t HK=k.size(nhd ? 2 : 1), N=pt.size(1)*16;\n"
        "  TORCH_CHECK(k.size(nhd ? 1 : 2)==16 && k.size(0)>0 && pt.size(0)==B,\n"
        '              "page-16 geometry mismatch");')
    source = _once(source, "D==kHeadDim && k.size(0)==B &&", "D==kHeadDim &&")
    source = _once(source, "N>0 && N%64==0 && splits>0 && splits<=N/64",
        "N>0 && splits>0 && splits<=(N+63)/64")
    source = source.replace("require N divisible by 64 and splits in [1,min(N/64,512)]",
        "require splits in [1,min(ceil(capacity/64),512)]")
    source = _once(source, "B*M*H<=INT_MAX && N<=INT_MAX && HK<=INT_MAX &&",
        "B*M*H<=INT_MAX && N<=INT_MAX-63 && k.size(0)<=INT_MAX && HK<=INT_MAX &&")
    source = _once(source, "for (auto t : {q,k,v,po,pl,o,qp,kp}) {",
        "for (auto t : {q,k,v,po,pl,o,qp,kp,pt,sl,ql}) {")
    source = _once(source, '  TORCH_CHECK(qp.scalar_type()==at::kLong', r"""
  TORCH_CHECK(pt.scalar_type()==at::kInt && sl.scalar_type()==at::kInt &&
              ql.scalar_type()==at::kInt && sl.sizes()==torch::IntArrayRef({B}) &&
              ql.sizes()==torch::IntArrayRef({B}), "page metadata must be int32");
  TORCH_CHECK(qp.scalar_type()==at::kLong""")
    source = _once(source, "for (auto in : {q,k,v,qp,kp})",
        "for (auto in : {q,k,v,qp,kp,pt,sl,ql})")
    source = _once(source, "  if (natural) {", r"""
  auto launch = [&](auto layout_tag) {
  constexpr bool kNHD = decltype(layout_tag)::value;
  if (natural) {""")
    source = _once(source,
        "        streamattn_natural_wgmma_micro_prefill_partial_kernel,",
        "        streamattn_natural_wgmma_micro_prefill_partial_kernel<kNHD>,")
    source = _once(source,
        "streamattn_natural_wgmma_micro_prefill_partial_kernel<<<",
        "streamattn_natural_wgmma_micro_prefill_partial_kernel<kNHD><<<")
    source = _once(source,
        "B,M,N,H,HK,G,splits,qp.data_ptr<int64_t>(),kp.data_ptr<int64_t>());",
        "B,M,N,H,HK,G,splits,qp.data_ptr<int64_t>(),kp.data_ptr<int64_t>(),\n"
        "        pt.data_ptr<int>(),sl.data_ptr<int>(),ql.data_ptr<int>(),pt.size(1));")
    source = _once(source, "streamattn_transposed_wgmma_exact_partial_kernel<0><<<",
        "streamattn_transposed_wgmma_exact_partial_kernel<16,true,kNHD><<<")
    source = _once(source,
        "nullptr,0,HK,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,M,\n"
        "        qp.data_ptr<int64_t>(),kp.data_ptr<int64_t>());",
        "pt.data_ptr<int>(),pt.size(1),HK,sl.data_ptr<int>(),\n"
        "        nullptr,nullptr,nullptr,nullptr,nullptr,M,\n"
        "        qp.data_ptr<int64_t>(),kp.data_ptr<int64_t>(),ql.data_ptr<int>());")
    source = _once(source, "  C10_CUDA_KERNEL_LAUNCH_CHECK();", r"""
  };
  if (nhd) launch(std::true_type{}); else launch(std::false_type{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();""")
    return source
