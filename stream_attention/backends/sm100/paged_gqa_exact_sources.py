"""PyTorch binding source for the native SM100 paged GQA backend."""

CPP_SOURCE = r"""
#include <torch/extension.h>

void streamattn_sm100_paged_gqa_exact_decode_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor padded_page_table,
    torch::Tensor sequence_lengths,
    torch::Tensor output_group,
    int64_t max_pages,
    int64_t num_splits);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("paged_exact_decode_out",
        &streamattn_sm100_paged_gqa_exact_decode_out_cuda,
        "StreamAttn SM100 direct-NHD paged GQA exact decode");
}
"""


CUDA_SOURCE = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>

#include "tgv_gqa_paged.cuh"

namespace {

using Element = cutlass::bfloat16_t;

constexpr int kKvHeads = 2;
constexpr int kGroupSize = 8;
constexpr int kHeadDim = 128;
constexpr int kPageSize = 16;
constexpr int kMetadataPadding = 64;

void check_cuda_bf16(torch::Tensor tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.scalar_type() == at::ScalarType::BFloat16,
              name, " must be bfloat16");
}

template <int NumSplits, int NumReductionCTA = NumSplits>
void launch_sm100_paged_gqa(
    torch::Tensor q_group,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor padded_page_table,
    torch::Tensor sequence_lengths,
    torch::Tensor output_group,
    int64_t max_pages) {
  const int64_t batch = q_group.size(0);
  const int kv_len = static_cast<int>(max_pages * kPageSize);
  const int q_stride_group = static_cast<int>(q_group.stride(1));
  const int q_stride_head = static_cast<int>(q_group.stride(2));
  const int q_stride_dim = static_cast<int>(q_group.stride(3));
  const int o_stride_group = static_cast<int>(output_group.stride(0));
  const int o_stride_head = static_cast<int>(output_group.stride(1));
  const int o_stride_dim = static_cast<int>(output_group.stride(2));
  auto stream = at::cuda::getCurrentCUDAStream(q_group.get_device()).stream();

  TGV::gqa_paged::gqa_paged_separate_host<
      Element, Element, float,
      kGroupSize, 1, 128, kHeadDim,
      kPageSize,
      3, 3,
      2, 64,
      NumSplits, NumReductionCTA>(
      reinterpret_cast<Element*>(k_pages.data_ptr<at::BFloat16>()),
      reinterpret_cast<Element*>(v_pages.data_ptr<at::BFloat16>()),
      reinterpret_cast<Element*>(q_group.data_ptr<at::BFloat16>()),
      reinterpret_cast<Element*>(output_group.data_ptr<at::BFloat16>()),
      nullptr,
      sequence_lengths.data_ptr<int>(),
      padded_page_table.data_ptr<int>(),
      kKvHeads, kGroupSize, 1, kv_len, kHeadDim, static_cast<int>(batch),
      static_cast<int>(k_pages.stride(0)),
      static_cast<int>(k_pages.stride(1)),
      static_cast<int>(k_pages.stride(2)),
      static_cast<int>(k_pages.stride(3)),
      static_cast<int>(v_pages.stride(0)),
      static_cast<int>(v_pages.stride(1)),
      static_cast<int>(v_pages.stride(2)),
      static_cast<int>(v_pages.stride(3)),
      q_stride_group, q_stride_head, static_cast<int>(q_group.stride(0)),
      q_stride_dim, static_cast<int>(q_group.stride(0)),
      o_stride_group, o_stride_head,
      static_cast<int>(batch * kKvHeads * o_stride_group),
      o_stride_dim, kKvHeads * o_stride_group,
      1, static_cast<int>(padded_page_table.stride(0)),
      1.0f / std::sqrt(static_cast<float>(kHeadDim)),
      0,
      false, -1,
      stream);
}

}  // namespace

void streamattn_sm100_paged_gqa_exact_decode_out_cuda(
    torch::Tensor q_group,
    torch::Tensor k_pages,
    torch::Tensor v_pages,
    torch::Tensor padded_page_table,
    torch::Tensor sequence_lengths,
    torch::Tensor output_group,
    int64_t max_pages,
    int64_t num_splits) {
  check_cuda_bf16(q_group, "q_group");
  check_cuda_bf16(k_pages, "k_pages");
  check_cuda_bf16(v_pages, "v_pages");
  check_cuda_bf16(output_group, "output_group");
  TORCH_CHECK(q_group.is_contiguous(), "q_group must be contiguous");
  TORCH_CHECK(k_pages.is_contiguous(), "k_pages must be contiguous NHD");
  TORCH_CHECK(v_pages.is_contiguous(), "v_pages must be contiguous NHD");
  TORCH_CHECK(output_group.is_contiguous(), "output_group must be contiguous");
  TORCH_CHECK(q_group.dim() == 4 && q_group.size(1) == kKvHeads &&
                  q_group.size(2) == kGroupSize &&
                  q_group.size(3) == kHeadDim,
              "q_group must have shape [B,2,8,128]");
  TORCH_CHECK(k_pages.sizes() == v_pages.sizes(), "K/V shape mismatch");
  TORCH_CHECK(k_pages.dim() == 4 && k_pages.size(1) == kPageSize &&
                  k_pages.size(2) == kKvHeads &&
                  k_pages.size(3) == kHeadDim,
              "K/V must have shape [pages,16,2,128]");
  const int64_t batch = q_group.size(0);
  TORCH_CHECK(output_group.dim() == 3 &&
                  output_group.size(0) == batch * kKvHeads &&
                  output_group.size(1) == kGroupSize &&
                  output_group.size(2) == kHeadDim,
              "output_group must have shape [B*2,8,128]");
  TORCH_CHECK(max_pages > 0 && max_pages * batch <= k_pages.size(0),
              "max_pages exceeds K/V storage");
  TORCH_CHECK(padded_page_table.is_cuda() &&
                  padded_page_table.scalar_type() == at::ScalarType::Int &&
                  padded_page_table.is_contiguous() &&
                  padded_page_table.dim() == 2 &&
                  padded_page_table.size(0) == batch &&
                  padded_page_table.size(1) >= max_pages + kMetadataPadding,
              "padded_page_table must be int32 [B,max_pages+64]");
  TORCH_CHECK(sequence_lengths.is_cuda() &&
                  sequence_lengths.scalar_type() == at::ScalarType::Int &&
                  sequence_lengths.is_contiguous() &&
                  sequence_lengths.numel() == batch,
              "sequence_lengths must be contiguous int32 [B]");
  TORCH_CHECK(num_splits == 2 || num_splits == 4 || num_splits == 8 ||
                  num_splits == 16,
              "num_splits must be 2, 4, 8, or 16");

  c10::cuda::CUDAGuard guard(q_group.device());
  if (num_splits == 2) {
    launch_sm100_paged_gqa<2>(q_group, k_pages, v_pages, padded_page_table,
                              sequence_lengths, output_group, max_pages);
  } else if (num_splits == 4) {
    launch_sm100_paged_gqa<4>(q_group, k_pages, v_pages, padded_page_table,
                              sequence_lengths, output_group, max_pages);
  } else if (num_splits == 8) {
    launch_sm100_paged_gqa<8>(q_group, k_pages, v_pages, padded_page_table,
                              sequence_lengths, output_group, max_pages);
  } else {
    launch_sm100_paged_gqa<16, 8>(q_group, k_pages, v_pages, padded_page_table,
                                  sequence_lengths, output_group, max_pages);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""
