"""Dtype/position specializations of the retained two-family state machines.

Only the micro-prefill producers and merges are compiled here. Checked source
composition keeps the historical decode extension and its promoted ABI intact.
"""

from .transposed_gqa_exact_sources import cuda_source_for_head_dim


CPP_SOURCE = r"""
#include <torch/extension.h>
void micro_semantics_out(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor po, torch::Tensor pl, torch::Tensor o,
    torch::Tensor qp, torch::Tensor kp, int64_t splits, bool natural);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("out", &micro_semantics_out);
}
"""


def _between(source: str, begin: str, end: str) -> str:
    if source.count(begin) != 1 or source.count(end) != 1:
        raise ValueError("micro-prefill source anchors changed; review composition")
    start, stop = source.index(begin), source.index(end)
    if stop <= start:
        raise ValueError("micro-prefill source anchors are out of order")
    return source[start:stop]


def _once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"micro-prefill specialization anchor changed: {old!r}")
    return source.replace(old, new, 1)


def semantic_cuda_source(head_dim: int, dtype: str, causal: bool) -> str:
    if dtype not in ("bf16", "fp16") or not isinstance(causal, bool):
        raise ValueError("expected bf16/fp16 and a boolean causal specialization")
    base = cuda_source_for_head_dim(head_dim)
    prefix = base[:base.index(
        "\n__global__ __launch_bounds__(256, 1)\n"
        "void streamattn_grouped_wgmma_prefill_kernel("
    )]
    natural = _between(
        base,
        "\n__global__ __launch_bounds__(128)\n"
        "void streamattn_natural_wgmma_micro_prefill_partial_kernel(",
        "\n__global__ __launch_bounds__(128)\n"
        "void streamattn_transposed_wgmma_qk_kernel(",
    )
    transposed = _between(
        base, "\ntemplate <int kPagedPageSize>\n",
        "\n__global__ __launch_bounds__(128)\n"
        "void streamattn_transposed_wgmma_exact_merge_kernel(",
    )
    merge = _between(
        base,
        "\n__global__ __launch_bounds__(32)\n"
        "void streamattn_transposed_wgmma_exact_merge_warp_kernel(",
        "\n__global__ __launch_bounds__(32)\n"
        "void streamattn_transposed_wgmma_selected_row_local_merge_warp_kernel(",
    )
    # The natural producer and merge both take num_splits; isolate the producer.
    signature = "    int group_size,\n    int num_splits) {"
    producer, natural_merge = natural.split(
        "\n__global__ __launch_bounds__(128)\n"
        "void streamattn_natural_wgmma_micro_prefill_merge_kernel(", 1
    )
    producer = _once(producer, signature,
        "    int group_size, int num_splits,\n"
        "    const int64_t* __restrict__ query_positions,\n"
        "    const int64_t* __restrict__ key_positions) {")
    producer = _once(producer, "      Accum tile_max = -INFINITY;", r"""
      if constexpr (kPositionCausal) {
        CUTE_UNROLL
        for (int col = 0; col < size<1>(score_rows); ++col) {
          const auto coord = tScScoresRowCol(row, col);
          const int qi = query_begin + int(get<0>(coord)) / group_size;
          const int ki = tile * kBlockM + int(get<1>(coord));
          if (qi >= query_length ||
              key_positions[static_cast<int64_t>(batch) * kv_length + ki] >
              query_positions[static_cast<int64_t>(batch) * query_length + qi]) {
            score_rows(row, col) = -INFINITY;
          }
        }
      }
      Accum tile_max = -INFINITY;""")
    producer = _once(producer,
        "const Accum probability =\n            exp2f(",
        "const Accum probability = next_max == -INFINITY ? 0.0f\n            : exp2f(")
    natural_merge = _once(natural_merge,
        "const Accum weight = exp2f(lse - row_max);",
        "const Accum weight = row_max == -INFINITY ? 0.0f : exp2f(lse - row_max);")
    natural_merge = _once(natural_merge, "Element(result / normalizer);",
        "Element(normalizer > 0.0f ? result / normalizer : 0.0f);")
    natural = producer + (
        "\n__global__ __launch_bounds__(128)\n"
        "void streamattn_natural_wgmma_micro_prefill_merge_kernel("
    ) + natural_merge
    transposed = _once(transposed, "int query_positions_per_batch = 1) {",
        "int query_positions_per_batch = 1,\n"
        "    const int64_t* __restrict__ query_positions = nullptr,\n"
        "    const int64_t* __restrict__ key_positions = nullptr) {")
    transposed = _once(transposed, "    Accum scale_o[kRowsPerThread];", r"""
    if constexpr (kPositionCausal) {
      const int batch = group / (query_positions_per_batch * kv_heads);
      const int qi = (group / kv_heads) % query_positions_per_batch;
      const int64_t qpos = query_positions[
          static_cast<int64_t>(batch) * query_positions_per_batch + qi];
      CUTE_UNROLL
      for (int row = 0; row < size<0>(scores); ++row) {
        CUTE_UNROLL
        for (int col = 0; col < size<1>(scores); ++col) {
          const int ki = tile * kBlockM + int(get<0>(tScSRowCol(row, col)));
          if (key_positions[static_cast<int64_t>(batch) * kv_len + ki] > qpos) {
            scores(row, col) = -INFINITY;
          }
        }
      }
    }
    Accum scale_o[kRowsPerThread];""")
    merge = _once(merge, "const Accum weight = exp2f(",
        "const Accum weight = row_max == -INFINITY ? 0.0f : exp2f(")
    merge = _once(merge, "const Accum inverse_normalizer = 1.0f / normalizer;",
        "const Accum inverse_normalizer = normalizer > 0.0f ? 1.0f / normalizer : 0.0f;")
    if dtype == "fp16":
        prefix = _once(prefix, "using Element = cutlass::bfloat16_t;",
            "using Element = cutlass::half_t;")
        prefix = prefix.replace("F32BF16BF16", "F32F16F16")
    host = _HOST.replace("@SCALAR@", "Half" if dtype == "fp16" else "BFloat16")
    return (prefix + f"\nconstexpr bool kPositionCausal = {str(causal).lower()};\n"
            + natural + transposed + merge + host)


_HOST = r"""
#include <c10/cuda/CUDAGuard.h>
#include <ATen/MemoryOverlap.h>

void micro_semantics_out(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor po, torch::Tensor pl, torch::Tensor o,
    torch::Tensor qp, torch::Tensor kp, int64_t splits, bool natural) {
  TORCH_CHECK(q.is_cuda(), "query must be CUDA");
  c10::cuda::CUDAGuard guard(q.device());
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.sizes() == k.sizes(),
              "expected Q [B,M,Hq,D] and K/V [B,Hkv,N,D]");
  const int64_t B=q.size(0), M=q.size(1), H=q.size(2), D=q.size(3);
  const int64_t HK=k.size(1), N=k.size(2);
  TORCH_CHECK(B>0 && M>=2 && M<=64 && D==kHeadDim && k.size(0)==B &&
              k.size(3)==D && HK>0 && H%HK==0 && (H/HK==4 || H/HK==8),
              "unsupported micro-prefill geometry");
  TORCH_CHECK(N>0 && N%64==0 && splits>0 && splits<=N/64 && splits<=512,
              "require N divisible by 64 and splits in [1,min(N/64,512)]");
  TORCH_CHECK(B*M*H<=INT_MAX && N<=INT_MAX && HK<=INT_MAX &&
              B*M*HK*splits<=INT_MAX, "launch geometry exceeds int32");
  const int G=H/HK, qt=(M+64/G-1)/(64/G);
  const int groups=natural ? B*HK*qt : B*M*HK, rows=natural ? 64 : 8;
  for (auto t : {q,k,v,po,pl,o,qp,kp}) {
    TORCH_CHECK(t.device()==q.device() && t.is_contiguous(),
                "all tensors must be contiguous on the query device");
  }
  for (auto t : {q,k,v,o}) {
    TORCH_CHECK(t.scalar_type()==at::ScalarType::@SCALAR@,
                "Q/K/V/output dtype does not match compiled specialization");
  }
  TORCH_CHECK(po.scalar_type()==at::kFloat && pl.scalar_type()==at::kFloat &&
              po.sizes()==torch::IntArrayRef({groups,splits,rows,D}) &&
              pl.sizes()==torch::IntArrayRef({groups,splits,rows}) &&
              o.sizes()==q.sizes(), "output/workspace contract mismatch");
  TORCH_CHECK(qp.scalar_type()==at::kLong && kp.scalar_type()==at::kLong,
              "positions must be int64");
  if constexpr (kPositionCausal) {
    TORCH_CHECK(qp.sizes()==torch::IntArrayRef({B,M}) &&
                kp.sizes()==torch::IntArrayRef({B,N}), "position shape mismatch");
  } else {
    TORCH_CHECK(qp.numel()==0 && kp.numel()==0, "noncausal positions must be empty");
  }
  for (auto out : {po,pl,o}) {
    for (auto in : {q,k,v,qp,kp}) { at::assert_no_overlap(out,in); }
  }
  at::assert_no_overlap(po,pl);
  at::assert_no_overlap(po,o);
  at::assert_no_overlap(pl,o);
  auto stream=at::cuda::getCurrentCUDAStream();
  const auto qptr=reinterpret_cast<const Element*>(q.data_ptr<at::@SCALAR@>());
  const auto kptr=reinterpret_cast<const Element*>(k.data_ptr<at::@SCALAR@>());
  const auto vptr=reinterpret_cast<const Element*>(v.data_ptr<at::@SCALAR@>());
  auto optr=reinterpret_cast<Element*>(o.data_ptr<at::@SCALAR@>());
  if (natural) {
    const int shared=sizeof(GroupedRSPrefillSharedStorage);
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        streamattn_natural_wgmma_micro_prefill_partial_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
    streamattn_natural_wgmma_micro_prefill_partial_kernel<<<groups*splits,128,shared,stream>>>(
        qptr,kptr,vptr,po.data_ptr<float>(),pl.data_ptr<float>(),
        B,M,N,H,HK,G,splits,qp.data_ptr<int64_t>(),kp.data_ptr<int64_t>());
    streamattn_natural_wgmma_micro_prefill_merge_kernel<<<B*M*H,128,0,stream>>>(
        po.data_ptr<float>(),pl.data_ptr<float>(),optr,B,M,H,HK,G,splits);
  } else {
    streamattn_transposed_wgmma_exact_partial_kernel<0><<<groups*splits,128,0,stream>>>(
        qptr,kptr,vptr,po.data_ptr<float>(),pl.data_ptr<float>(),groups,N,splits,G,
        nullptr,0,HK,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,M,
        qp.data_ptr<int64_t>(),kp.data_ptr<int64_t>());
    streamattn_transposed_wgmma_exact_merge_warp_kernel<<<groups*G,32,0,stream>>>(
        po.data_ptr<float>(),pl.data_ptr<float>(),optr,groups,splits,G);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""
