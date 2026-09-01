from stream_attention.backends.sm90.grouped_prefill_epoch_floor import (
    KERNEL_NAMES,
    KERNEL_RESOURCE_FIELDS,
    RESOURCE_FIELDS,
    decode_cluster2_epoch_resource_info,
    decode_grouped2_resource_info,
)
from stream_attention.backends.sm90.grouped_prefill_epoch_floor_sources import (
    CUDA_SOURCE,
)


def test_epoch_floor_contains_paired_ss_and_rs_paths() -> None:
    assert "GMMA::ss_op_selector" in CUDA_SOURCE
    assert "GMMA::rs_op_selector" in CUDA_SOURCE
    assert "SM90_U32x4_STSM_N" in CUDA_SOURCE
    assert "convert_layout_acc_aregs" in CUDA_SOURCE
    assert "NamedBarrier::sync(kThreads, 0)" in CUDA_SOURCE
    assert "if constexpr (RegisterPV)" in CUDA_SOURCE
    assert "softmax_in_place" in CUDA_SOURCE
    assert "scale_output" in CUDA_SOURCE


def test_epoch_floor_is_component_only_and_reports_spill_surface() -> None:
    assert KERNEL_NAMES == (
        "qk",
        "qk_softmax",
        "pv_ss",
        "pv_rs",
        "epoch_ss",
        "epoch_rs",
        "epoch_rs_reuse_q",
    )
    assert "local_bytes_per_thread" in KERNEL_RESOURCE_FIELDS
    assert RESOURCE_FIELDS[-3:] == (
        "epoch_ss_shared_bytes",
        "epoch_rs_shared_bytes",
        "tma_epoch_shared_bytes",
    )
    assert "cudaFuncGetAttributes" in CUDA_SOURCE
    assert "cudaOccupancyMaxActiveBlocksPerMultiprocessor" in CUDA_SOURCE


def test_epoch_floor_contains_stage_local_tma_rs_path() -> None:
    assert "PipelineTmaAsync<kKStages>" in CUDA_SOURCE
    assert "PipelineTmaAsync<kVStages>" in CUDA_SOURCE
    assert "warpgroup_reg_dealloc<56>" in CUDA_SOURCE
    assert "warpgroup_reg_alloc<256>" in CUDA_SOURCE
    assert "producer_get_barrier" in CUDA_SOURCE
    assert "pipeline_k.consumer_release" in CUDA_SOURCE
    assert "pipeline_v.consumer_release" in CUDA_SOURCE
    assert "epoch_rs_tma_floor_kernel" in CUDA_SOURCE

    kernel = CUDA_SOURCE.split("void epoch_rs_tma_floor_kernel", 1)[1].split(
        "void epoch_rs_grouped2_tma_floor_kernel", 1
    )[0]
    steady_state = kernel.split("for (int tile = begin; tile < end; ++tile)", 2)[2]
    assert "__syncthreads()" not in steady_state


def test_epoch_floor_contains_bounded_two_consumer_reuse_path() -> None:
    assert "epoch_rs_grouped2_serial_floor_kernel" in CUDA_SOURCE
    assert "epoch_rs_grouped2_tma_floor_kernel" in CUDA_SOURCE
    assert "kGroupedTmaThreads = 384" in CUDA_SOURCE
    assert "kGroupedConsumerThreads = 256" in CUDA_SOURCE
    assert "warpgroup_reg_dealloc<24>" in CUDA_SOURCE
    assert "warpgroup_reg_alloc<240>" in CUDA_SOURCE
    assert "params_k.num_consumers = kGroupedConsumerThreads" in CUDA_SOURCE
    assert "params_v.num_consumers = kGroupedConsumerThreads" in CUDA_SOURCE

    kernel = CUDA_SOURCE.split(
        "void epoch_rs_grouped2_tma_floor_kernel", 1
    )[1].split("void epoch_rs_cluster2_floor_kernel", 1)[0]
    steady_state = kernel.split("for (int tile = begin; tile < end; ++tile)", 2)[2]
    assert "__syncthreads()" not in steady_state


def test_decode_grouped2_resource_info() -> None:
    import torch

    width = len(KERNEL_RESOURCE_FIELDS)
    decoded = decode_grouped2_resource_info(torch.arange(2 * width))
    assert tuple(decoded) == ("serial_grouped2", "tma_grouped2")
    assert decoded["serial_grouped2"]["registers_per_thread"] == 0
    assert decoded["tma_grouped2"]["registers_per_thread"] == width


def test_epoch_floor_contains_two_cta_multicast_attention_path() -> None:
    assert "epoch_rs_cluster2_floor_kernel" in CUDA_SOURCE
    assert "SM90_TMA_LOAD_MULTICAST" in CUDA_SOURCE
    assert "ClusterShape2 = Shape<_2, _1, _1>" in CUDA_SOURCE
    assert "cluster_arrive_relaxed" in CUDA_SOURCE
    assert "launch_kernel_on_cluster" in CUDA_SOURCE
    assert "dim3(2, 1, 1)" in CUDA_SOURCE
    assert "warpgroup_reg_dealloc<56>" in CUDA_SOURCE
    assert "warpgroup_reg_alloc<256>" in CUDA_SOURCE

    kernel = CUDA_SOURCE.split("void epoch_rs_cluster2_floor_kernel", 1)[1].split(
        "static void check_common", 1
    )[0]
    steady_state = kernel.split("for (int tile = begin; tile < end; ++tile)", 2)[2]
    assert "__syncthreads()" not in steady_state


def test_decode_cluster2_epoch_resource_info() -> None:
    import torch

    width = len(KERNEL_RESOURCE_FIELDS)
    decoded = decode_cluster2_epoch_resource_info(torch.arange(2 * width))
    assert tuple(decoded) == ("independent", "multicast")
    assert decoded["independent"]["registers_per_thread"] == 0
    assert decoded["multicast"]["registers_per_thread"] == width
