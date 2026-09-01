import torch

from stream_attention.backends.sm90.grouped_prefill_cluster_floor import (
    RESOURCE_FIELDS,
    decode_cluster_resource_info,
)
from stream_attention.backends.sm90.grouped_prefill_cluster_floor_sources import (
    CUDA_SOURCE,
)


def test_cluster_floor_uses_real_two_cta_tma_multicast() -> None:
    assert "SM90_TMA_LOAD_MULTICAST" in CUDA_SOURCE
    assert "ClusterShape = Shape<_2, _1, _1>" in CUDA_SOURCE
    assert "cluster_arrive_relaxed" in CUDA_SOURCE
    assert "cluster_wait" in CUDA_SOURCE
    assert "Layout<ActiveClusterShape>" in CUDA_SOURCE
    assert "multicast_mask" in CUDA_SOURCE
    assert "launch_kernel_on_cluster" in CUDA_SOURCE
    assert "dim3(2, 1, 1)" in CUDA_SOURCE
    assert "warpgroup_reg_dealloc<56>" in CUDA_SOURCE
    assert "warpgroup_reg_alloc<256>" in CUDA_SOURCE


def test_cluster_floor_has_fair_independent_baseline_and_spill_telemetry() -> None:
    assert "cluster_transport_floor_kernel<" in CUDA_SOURCE
    assert "false, decltype(tma_k), decltype(tma_v)" in CUDA_SOURCE
    assert "true, decltype(tma_k), decltype(tma_v)" in CUDA_SOURCE
    assert "localSizeBytes" in CUDA_SOURCE
    assert "cudaOccupancyMaxActiveBlocksPerMultiprocessor" in CUDA_SOURCE


def test_decode_cluster_resource_info() -> None:
    width = len(RESOURCE_FIELDS)
    decoded = decode_cluster_resource_info(torch.arange(2 * width))
    assert tuple(decoded) == ("independent", "multicast")
    assert decoded["independent"]["registers_per_thread"] == 0
    assert decoded["multicast"]["registers_per_thread"] == width
