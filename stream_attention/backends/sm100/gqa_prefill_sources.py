"""PyTorch binding source for native SM100 causal GQA prefill."""

CPP_SOURCE = r"""
#include <torch/extension.h>

void streamattn_sm100_gqa_prefill_out_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor sequence_lengths,
    torch::Tensor output,
    int64_t tile_variant);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prefill_out", &streamattn_sm100_gqa_prefill_out_cuda,
        "StreamAttn SM100 contiguous causal GQA prefill");
}
"""


CUDA_SOURCE = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>

#include "tgv_gqa.cuh"

namespace {

using Element = cutlass::bfloat16_t;

constexpr int kQueryHeads = 16;
constexpr int kKvHeads = 2;
constexpr int kGroupSize = 8;
constexpr int kHeadDim = 128;

void check_cuda_bf16(torch::Tensor tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.scalar_type() == at::ScalarType::BFloat16,
              name, " must be bfloat16");
}

template <
    int TileQueryHeads,
    int TileQueryLength,
    int Bmm1Stages = 3,
    int Bmm2Stages = 3>
void launch_sm100_gqa_prefill(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor sequence_lengths,
    torch::Tensor output) {
  const int batch = static_cast<int>(query.size(0));
  const int query_length = static_cast<int>(query.size(1));
  const int kv_length = static_cast<int>(key.size(1));
  auto stream = at::cuda::getCurrentCUDAStream(query.get_device()).stream();

  TGV::gqa::gqa_host<
      Element, Element, float,
      TileQueryHeads, TileQueryLength, 128, kHeadDim,
      Bmm1Stages, Bmm2Stages,
      1, 1,
      true>(
      reinterpret_cast<Element*>(key.data_ptr<at::BFloat16>()),
      reinterpret_cast<Element*>(query.data_ptr<at::BFloat16>()),
      reinterpret_cast<Element*>(value.data_ptr<at::BFloat16>()),
      reinterpret_cast<Element*>(output.data_ptr<at::BFloat16>()),
      sequence_lengths.data_ptr<int>(),
      nullptr,
      kKvHeads, kGroupSize, query_length, kv_length, kHeadDim, batch,
      static_cast<int>(key.stride(2)),
      static_cast<int>(key.stride(1)),
      static_cast<int>(key.stride(3)),
      static_cast<int>(key.stride(0)),
      static_cast<int>(kGroupSize * query.stride(2)),
      static_cast<int>(query.stride(2)),
      static_cast<int>(query.stride(1)),
      static_cast<int>(query.stride(3)),
      static_cast<int>(query.stride(0)),
      static_cast<int>(value.stride(2)),
      static_cast<int>(value.stride(1)),
      static_cast<int>(value.stride(3)),
      static_cast<int>(value.stride(0)),
      static_cast<int>(kGroupSize * output.stride(2)),
      static_cast<int>(output.stride(2)),
      static_cast<int>(output.stride(1)),
      static_cast<int>(output.stride(3)),
      static_cast<int>(output.stride(0)),
      1.0f / std::sqrt(static_cast<float>(kHeadDim)),
      0,
      false, -1,
      stream);
}

}  // namespace

void streamattn_sm100_gqa_prefill_out_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor sequence_lengths,
    torch::Tensor output,
    int64_t tile_variant) {
  check_cuda_bf16(query, "query");
  check_cuda_bf16(key, "key");
  check_cuda_bf16(value, "value");
  check_cuda_bf16(output, "output");
  TORCH_CHECK(query.is_contiguous(), "query must be contiguous BSHD");
  TORCH_CHECK(key.is_contiguous(), "key must be contiguous BSHD");
  TORCH_CHECK(value.is_contiguous(), "value must be contiguous BSHD");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous BSHD");
  TORCH_CHECK(query.dim() == 4 && query.size(2) == kQueryHeads &&
                  query.size(3) == kHeadDim,
              "query must have shape [B,S,16,128]");
  TORCH_CHECK(key.dim() == 4 && key.size(2) == kKvHeads &&
                  key.size(3) == kHeadDim,
              "key must have shape [B,S,2,128]");
  TORCH_CHECK(key.sizes() == value.sizes(), "K/V shape mismatch");
  TORCH_CHECK(query.size(0) == key.size(0) &&
                  query.size(1) == key.size(1),
              "causal prefill requires matching Q and K/V lengths");
  TORCH_CHECK(output.sizes() == query.sizes(), "output shape mismatch");
  TORCH_CHECK(sequence_lengths.is_cuda() &&
                  sequence_lengths.scalar_type() == at::ScalarType::Int &&
                  sequence_lengths.is_contiguous() &&
                  sequence_lengths.numel() == query.size(0),
              "sequence_lengths must be contiguous int32 [B]");
  TORCH_CHECK(tile_variant >= 0 && tile_variant <= 2,
              "tile_variant must be 0, 1, or 2");

  c10::cuda::CUDAGuard guard(query.device());
  if (tile_variant == 0) {
    launch_sm100_gqa_prefill<8, 1>(
        query, key, value, sequence_lengths, output);
  } else if (tile_variant == 1) {
    launch_sm100_gqa_prefill<8, 2>(
        query, key, value, sequence_lengths, output);
  } else {
    launch_sm100_gqa_prefill<8, 4, 2, 2>(
        query, key, value, sequence_lengths, output);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""
