"""Compact planned task launch for the retained exact natural producer."""

from .micro_prefill_paged_sources import CPP_SOURCE as PAGED_CPP, paged_cuda_source
from .micro_prefill_semantics_sources import _once, _between

_OLD = "int64_t splits, bool natural, bool nhd"
_NEW = "int64_t splits, bool natural, bool nhd, torch::Tensor tasks, torch::Tensor counts"
CPP_SOURCE = _once(PAGED_CPP, _OLD, _NEW)


def ragged_cuda_source(head_dim, dtype, causal):
    source = paged_cuda_source(head_dim, dtype, causal)
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
    return source
