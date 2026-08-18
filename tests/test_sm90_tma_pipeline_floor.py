from stream_attention.backends.sm90.tma_pipeline_floor import (
    KERNEL_NAMES,
    KERNEL_RESOURCE_FIELDS,
    RESOURCE_FIELDS,
)
from stream_attention.backends.sm90.tma_pipeline_floor_sources import CUDA_SOURCE


def test_tma_floor_keeps_producer_consumer_and_independent_v_pipeline() -> None:
    assert "kTmaThreads = 256" in CUDA_SOURCE
    assert "kConsumers = 128" in CUDA_SOURCE
    assert "PipelineTmaAsync<kKStages>" in CUDA_SOURCE
    assert "PipelineTmaAsync<kVStages>" in CUDA_SOURCE
    assert "producer_get_barrier" in CUDA_SOURCE
    assert "consumer_release" in CUDA_SOURCE
    assert "warpgroup_reg_dealloc<24>" in CUDA_SOURCE
    assert "warpgroup_reg_alloc<160>" in CUDA_SOURCE
    assert "threadIdx.x - 128" in CUDA_SOURCE


def test_tma_floor_reports_topology_and_compiled_resources() -> None:
    assert RESOURCE_FIELDS[:4] == (
        "tile_bytes",
        "cp_storage_bytes",
        "tma_2k_storage_bytes",
        "tma_2k1v_storage_bytes",
    )
    assert "raw_2k2v_bytes" in RESOURCE_FIELDS
    assert KERNEL_NAMES == (
        "cp_async_k",
        "cp_async_kv",
        "tma_k",
        "tma_kv_2k1v",
    )
    assert "registers_per_thread" in KERNEL_RESOURCE_FIELDS
    assert "blocks_per_sm" in KERNEL_RESOURCE_FIELDS
    assert "cudaFuncSetAttribute" in CUDA_SOURCE
    assert "cudaOccupancyMaxActiveBlocksPerMultiprocessor" in CUDA_SOURCE
