"""CPU contracts and opt-in GPU integration for the isolated M128 canary."""

import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace

import pytest
import torch

from benchmarks.profile_sm90_micro_prefill_128 import (
    canary_cases,
    fp32_reference,
    lse_error,
    output_error,
    parse_args,
    profile_case,
    reconstructed_lse,
)
from stream_attention.backends.sm90 import micro_prefill_128 as backend
from benchmarks import sm90_binary_diagnostics as diagnostics
from stream_attention.backends.sm90.micro_prefill import (
    choose_natural_micro_prefill_splits,
    natural_micro_prefill_query_tiles,
)
from stream_attention.backends.sm90.micro_prefill_128_sources import (
    CPP_SOURCE,
    CUDA_SOURCE,
    cuda_source_for_head_dim,
)


@pytest.mark.parametrize("g", [4, 8])
def test_128_rows_are_positions_times_heads_and_cover_each_row_once(g):
    for m in range(2, 65):
        tiles = backend.query_tiles_128(m, g)
        covered = [
            (tile * (128 // g) + row // g, row % g)
            for tile in range(tiles)
            for row in range(128)
            if tile * (128 // g) + row // g < m
        ]
        assert covered == [
            (position, head) for position in range(m) for head in range(g)
        ]
        assert tiles == math.ceil(m * g / 128)
    assert backend.query_tiles_128(64, g) * 2 == natural_micro_prefill_query_tiles(
        query_len=64, group_size=g
    )


@pytest.mark.parametrize("m,g", [(1, 4), (65, 4), (16, 2), (16, 16)])
def test_bad_tile_geometry(m, g):
    with pytest.raises(ValueError):
        backend.query_tiles_128(m, g)


def test_default_splits_match_64_row_family_instead_of_doubling_state():
    for g in (4, 8):
        shape = dict(batch=1, query_len=64, kv_heads=16 // g, group_size=g, kv_len=4096)
        assert (
            backend.choose_micro_prefill_128_splits(**shape)
            == choose_natural_micro_prefill_splits(**shape)
            == 16
        )
        assert (16 // g) * backend.query_tiles_128(64, g) * 16 == 128


@pytest.mark.parametrize("tiles", [1, 2, 5, 7, 11, 19, 65])
def test_balanced_intervals_cover_short_and_irregular_splits(tiles):
    for splits in range(1, tiles + 1):
        bounds = [
            backend.balanced_tile_interval(tiles, splits, s) for s in range(splits)
        ]
        assert all(begin < end for begin, end in bounds)
        assert [t for begin, end in bounds for t in range(begin, end)] == list(
            range(tiles)
        )
        assert (
            max(end - begin for begin, end in bounds)
            - min(end - begin for begin, end in bounds)
            <= 1
        )
    with pytest.raises(ValueError):
        backend.balanced_tile_interval(tiles, tiles + 1, 0)


def _resource_vector(producer_local=0, blocks=2, merge_local=0):
    return torch.tensor(
        [232, 0, 81920, producer_local, blocks, 128, 32, 2056, 0, merge_local, 16, 128],
        dtype=torch.int64,
    )


def test_resource_decoding_and_fail_closed_gate():
    resources = backend.decode_resource_info(_resource_vector())
    assert resources["producer"]["dynamic_shared_bytes"] == 81920
    assert resources["producer"]["registers_per_thread"] == 232
    assert backend.resource_gate(resources, direct=False)["passed"]
    assert not backend.resource_gate(
        backend.decode_resource_info(_resource_vector(blocks=1)), direct=False
    )["passed"]
    assert not backend.resource_gate(
        backend.decode_resource_info(_resource_vector(producer_local=4)), direct=True
    )["passed"]
    merge_spill = backend.decode_resource_info(_resource_vector(merge_local=4))
    assert backend.resource_gate(merge_spill, direct=True)["passed"]
    assert not backend.resource_gate(merge_spill, direct=False)["passed"]
    no_merge = backend.decode_resource_info(_resource_vector())
    no_merge["merge"]["blocks_per_sm"] = 0
    assert not backend.resource_gate(no_merge, direct=False)["passed"]
    assert backend.resource_gate(no_merge, direct=True)["passed"]
    with pytest.raises(ValueError, match="12"):
        backend.decode_resource_info(torch.zeros(6))


@pytest.fixture
def mocked_extension(monkeypatch):
    calls = []
    resources = []

    def info(*args):
        resources.append(args)
        return _resource_vector()

    extension = SimpleNamespace(
        out=lambda *args: calls.append(args), resource_info=info
    )
    monkeypatch.setattr(
        backend, "supports_sm90_micro_prefill", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(
        backend, "compile_micro_prefill_128_extension", lambda **kwargs: extension
    )
    return extension, calls, resources


def _buffers():
    return (
        torch.empty(1, 17, 8, 64, dtype=torch.bfloat16),
        torch.empty(1, 2, 320, 64, dtype=torch.bfloat16),
        torch.empty(1, 2, 320, 64, dtype=torch.bfloat16),
    )


def test_plan_replay_has_no_tensor_allocations_or_resource_queries(
    mocked_extension, monkeypatch
):
    extension, calls, resources = mocked_extension
    plan = backend.Natural128AsyncMicroPrefillPlan.build(*_buffers(), num_splits=4)
    assert plan.partial_output.shape == (2, 4, 128, 64)
    assert plan.partial_lse.shape == (2, 4, 128)
    assert plan.lse.shape == (1, 17, 8)
    assert plan.workspace_bytes == 2 * 4 * 128 * 65 * 4
    assert plan.producer_ctas == 8
    assert len(resources) == 1

    def forbidden(*args, **kwargs):
        raise AssertionError("allocation or resource query during replay")

    monkeypatch.setattr(torch, "empty", forbidden)
    monkeypatch.setattr(torch, "empty_like", forbidden)
    extension.resource_info = forbidden
    assert plan.run() is plan.output
    plan.run_component("producer")
    plan.run_component("merge")
    assert [call[-3] for call in calls] == [0, 1, 2]
    assert all(call[-4] == 4 and call[-2] == 0 and call[-1] is False for call in calls)


def test_s1_direct_has_zero_scratch_and_partial_control_is_available(mocked_extension):
    _, calls, _ = mocked_extension
    plan = backend.Natural128AsyncMicroPrefillPlan.build(
        *_buffers(), num_splits=1, protocol="serial"
    )
    assert plan.direct and plan.workspace_bytes == 0
    plan.run()
    assert calls[-1][-4:] == (1, 0, 1, True)
    with pytest.raises(ValueError, match="no merge"):
        plan.run_component("merge")
    with pytest.raises(ValueError, match="no partial"):
        reconstructed_lse(plan)
    control = backend.Natural128AsyncMicroPrefillPlan.build(
        *_buffers(), num_splits=1, direct=False
    )
    assert control.workspace_bytes > 0 and not control.direct


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(num_splits=0),
        dict(num_splits=6),
        dict(num_splits=True),
        dict(num_splits=1.5),
        dict(protocol="tma"),
        dict(num_splits=2, direct=True),
    ],
)
def test_invalid_plan_options_fail_before_dispatch(mocked_extension, kwargs):
    with pytest.raises(ValueError):
        backend.Natural128AsyncMicroPrefillPlan.build(*_buffers(), **kwargs)
    assert not mocked_extension[1]


def test_invalid_output_buffers_and_cpu_device_are_rejected(
    mocked_extension, monkeypatch
):
    q, k, v = _buffers()
    with pytest.raises(ValueError, match="output"):
        backend.Natural128AsyncMicroPrefillPlan.build(q, k, v, output=q.float())
    with pytest.raises(ValueError, match="lse"):
        backend.Natural128AsyncMicroPrefillPlan.build(q, k, v, lse=torch.empty(1))
    monkeypatch.setattr(
        backend, "supports_sm90_micro_prefill", lambda *args, **kwargs: False
    )
    with pytest.raises(ValueError, match="SM90"):
        backend.Natural128AsyncMicroPrefillPlan.build(q, k, v)


@pytest.mark.parametrize("m,g", [(7, 8), (9, 8), (17, 4), (33, 4), (63, 8)])
def test_partial_lse_reconstruction_does_not_mix_halves_heads_or_batches(m, g):
    b, hk, splits = 2, 3, 4
    tiles = backend.query_tiles_128(m, g)
    state = torch.arange(b * hk * tiles * 128).float().view(b * hk * tiles, 128) / 100
    plan = SimpleNamespace(
        direct=False,
        query=torch.empty(b, m, hk * g, 64),
        key_cache=torch.empty(b, hk, 64, 64),
        query_tiles=tiles,
        partial_lse=state[:, None].expand(-1, splits, -1).clone(),
    )
    actual = reconstructed_lse(plan)
    for bi in range(b):
        for qi in range(m):
            for head in range(hk * g):
                work = (bi * hk + head // g) * tiles + qi // (128 // g)
                row = (qi % (128 // g)) * g + head % g
                assert float(actual[bi, qi, head]) == pytest.approx(
                    float(state[work, row]) * math.log(2) + math.log(splits), abs=1e-5
                )


def test_online_recurrence_and_split_merge_match_full_attention_with_late_maxima():
    generator = torch.Generator().manual_seed(27)
    scores = torch.randn(128, 320, dtype=torch.float64, generator=generator)
    scores[:64, -64:] += 20
    scores[64:, :64] += 15
    values = torch.randn(320, 8, dtype=torch.float64, generator=generator)
    partial_o, partial_lse = [], []
    for split in range(4):
        begin, end = backend.balanced_tile_interval(5, 4, split)
        m, denominator = (
            torch.full((128,), -torch.inf, dtype=torch.float64),
            torch.zeros(128, dtype=torch.float64),
        )
        numerator = torch.zeros(128, 8, dtype=torch.float64)
        for tile in range(begin, end):
            z = scores[:, tile * 64 : (tile + 1) * 64]
            next_m = torch.maximum(m, z.max(1).values)
            alpha = torch.exp(m - next_m)
            probability = torch.exp(z - next_m[:, None])
            numerator = (
                alpha[:, None] * numerator
                + probability @ values[tile * 64 : (tile + 1) * 64]
            )
            denominator = alpha * denominator + probability.sum(1)
            m = next_m
        partial_o.append(numerator / denominator[:, None])
        partial_lse.append((m + denominator.log()) / math.log(2))
    state_lse = torch.stack(partial_lse)
    weights = torch.exp2(state_lse - state_lse.max(0).values)
    actual = (weights[:, :, None] * torch.stack(partial_o)).sum(0) / weights.sum(0)[
        :, None
    ]
    torch.testing.assert_close(actual, scores.softmax(1) @ values)
    torch.testing.assert_close(
        torch.logsumexp(state_lse * math.log(2), dim=0), torch.logsumexp(scores, dim=1)
    )


@pytest.mark.parametrize("tiles", [1, 2, 5])
@pytest.mark.parametrize("drained", [False, True])
def test_committed_group_order_releases_each_operand_before_reuse(tiles, drained):
    pending = []

    def wait(count):
        del pending[: max(0, len(pending) - count)]

    for tile in range(tiles):
        assert all(op[0] == "pv_b" for op in pending)
        pending.extend([("qk_a", tile), ("qk_b", tile)])
        wait(1)
        assert pending == [("qk_b", tile)]  # V and P are reusable; A scores are ready.
        pending.append(("pv_a", tile))
        wait(1)
        assert pending == [("pv_a", tile)]  # B scores/output are free for softmax.
        wait(0)
        assert not pending  # Packed P(A) can now become P(B).
        pending.append(("pv_b", tile))
        if drained:
            wait(0)
            assert not pending
    wait(0)
    assert not pending


def test_drained_protocol_changes_only_the_footer_wait_boundary_not_math():
    footer = """    if constexpr (DrainLoop) {
      // Retire the last PV(B) at the footer, preserving intra-iteration overlap.
      warpgroup_wait<0>();
      warpgroup_fence_operand(p_regs);
      warpgroup_fence_operand(output_a);
      warpgroup_fence_operand(output_b);
    }
"""
    for dim in (64, 128):
        body = (
            cuda_source_for_head_dim(dim)
            .split("void streamattn_micro128_kernel(", 1)[1]
            .split("\n__global__", 1)[0]
        )
        assert body.count(footer) == 1
        assert body.count("DrainLoop") == 1
        assert body.count("if constexpr (Overlap) { warpgroup_wait<1>(); }") == 2
        assert (
            "warpgroup_commit_batch();\n    if constexpr (!Overlap) { warpgroup_wait<0>(); }\n"
            + footer
            + "    read_pipe ^= 1;"
            in body
        )
        # Frozen original serial/overlap producer body, before adding the footer.
        assert hashlib.sha256(body.replace(footer, "").encode()).hexdigest() == (
            "1f0e0f8dc66909a69c111b9181e5d631b96ea8cd716bdafbab614c07c5551868"
        )
    assert backend.PROTOCOLS == {"overlap": 0, "serial": 1, "overlap_drained": 2}
    assert CUDA_SOURCE.count("else if (protocol == 2)") == 2
    for direct in ("true", "false"):
        assert f"micro128_launch<GroupSize, true, {direct}, true>" in CUDA_SOURCE
        assert (
            f"streamattn_micro128_kernel<GroupSize, true, {direct}, true>, sizeof"
            in CUDA_SOURCE
        )


@pytest.mark.parametrize("protocol", ["serial", "overlap", "overlap_drained"])
@pytest.mark.parametrize("direct", [False, True])
def test_plan_names_match_exact_launch_and_resource_specialization(
    mocked_extension, protocol, direct
):
    _, calls, resources = mocked_extension
    plan = backend.Natural128AsyncMicroPrefillPlan.build(
        *_buffers(),
        num_splits=1,
        protocol=protocol,
        direct=direct,
        diagnostic_build=True,
        lineinfo=True,
    )
    assert plan.kernel_names["producer"] == (
        f"streamattn_micro128_kernel<4, {str(protocol != 'serial').lower()}, "
        f"{str(direct).lower()}, {str(protocol == 'overlap_drained').lower()}>"
    )
    assert ("merge" in plan.kernel_names) is (not direct)
    plan.run()
    assert calls[-1][-2:] == (backend.PROTOCOLS[protocol], direct)
    assert resources[-1][-2:] == (backend.PROTOCOLS[protocol], direct)


def test_source_composition_keeps_existing_kernels_and_bindings_out():
    assert "streamattn_grouped_wgmma_prefill_kernel(" not in CUDA_SOURCE
    assert "streamattn_natural_wgmma_micro_prefill_partial_kernel(" not in CUDA_SOURCE
    assert CUDA_SOURCE.count("__global__") == 2
    assert "kPrefillRowsPerWarpGroup = 64" in CUDA_SOURCE
    assert "kMicro128Rows = 128" in CUDA_SOURCE
    assert "sizeof(Micro128SharedStorage) == 640 * kHeadDim" in CUDA_SOURCE
    assert "warpgroup_reg_alloc" not in CUDA_SOURCE
    assert "warpgroup_wait<1>()" in CUDA_SOURCE
    assert "at::assert_no_overlap" in CUDA_SOURCE
    assert "CUDAGuard guard(q.device())" in CUDA_SOURCE
    assert 'm.def("out"' in CPP_SOURCE and 'm.def("resource_info"' in CPP_SOURCE
    assert "kHeadDim = 128;" in cuda_source_for_head_dim(128)
    assert backend.source_fingerprint(64) != backend.source_fingerprint(128)
    with pytest.raises(ValueError):
        cuda_source_for_head_dim(256)


def test_launch_path_has_no_resource_queries_or_dynamic_allocations():
    launch = CUDA_SOURCE.split("void streamattn_micro128_out_cuda(", 1)[1].split(
        "template <class Kernel>", 1
    )[0]
    assert "cudaFuncSetAttribute" not in launch
    assert "cudaFuncGetAttributes" not in launch
    assert "torch::empty" not in launch
    assert "cudaMalloc" not in launch


def test_compile_cache_is_specialization_and_source_keyed(
    mocked_extension, monkeypatch, tmp_path
):
    # Test the real compiler wrapper without nvcc or GPU initialization.
    import importlib

    original = importlib.reload(backend)
    import torch.utils.cpp_extension

    calls = []
    monkeypatch.setattr(original, "resolve_cutlass_root", lambda root: tmp_path)
    monkeypatch.setattr(
        torch.utils.cpp_extension,
        "load_inline",
        lambda **kwargs: calls.append(kwargs) or SimpleNamespace(),
    )
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.0")
    a = original.compile_micro_prefill_128_extension(head_dim=64, build_dir=tmp_path)
    assert a is original.compile_micro_prefill_128_extension(
        head_dim=64, build_dir=tmp_path
    )
    original.compile_micro_prefill_128_extension(head_dim=128, build_dir=tmp_path)
    assert len(calls) == 2 and calls[0]["name"] != calls[1]["name"]
    assert calls[0]["build_directory"] != calls[1]["build_directory"]
    assert "--ptxas-options=-v,--warn-on-spills" in calls[0]["extra_cuda_cflags"]
    assert "-lineinfo" not in calls[0]["extra_cuda_cflags"]
    assert "--keep" not in calls[0]["extra_cuda_cflags"]
    assert calls[0]["keep_intermediates"] is True  # load_inline's generated sources
    diagnostic = original.compile_micro_prefill_128_extension(
        head_dim=64, build_dir=tmp_path, lineinfo=True, diagnostic_build=True
    )
    assert diagnostic is original.compile_micro_prefill_128_extension(
        head_dim=64, build_dir=tmp_path, lineinfo=True, diagnostic_build=True
    )
    assert len(calls) == 3
    assert len({call["name"] for call in calls}) == 3
    flags = calls[-1]["extra_cuda_cflags"]
    assert "-lineinfo" in flags and "--keep" in flags
    assert Path(flags[flags.index("--keep-dir") + 1]).is_dir()
    assert diagnostic._streamattn_build_metadata["head_dim"] == 64
    assert diagnostic._streamattn_build_metadata["extra_cuda_cflags"] == flags
    assert diagnostic._streamattn_build_metadata[
        "source_sha256"
    ] == backend.source_fingerprint(64)
    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "8.0"


def test_diagnostic_build_uses_torch_cache_if_no_build_directory(monkeypatch, tmp_path):
    import torch.utils.cpp_extension as cpp

    calls = []
    monkeypatch.setattr(backend, "_EXTENSIONS", {})
    monkeypatch.setattr(backend, "resolve_cutlass_root", lambda root: tmp_path)
    monkeypatch.setattr(
        cpp, "_get_build_directory", lambda *args: str(tmp_path / "cache")
    )
    monkeypatch.setattr(
        cpp, "load_inline", lambda **kwargs: calls.append(kwargs) or SimpleNamespace()
    )
    extension = backend.compile_micro_prefill_128_extension(
        head_dim=64, diagnostic_build=True
    )
    assert extension is backend.compile_micro_prefill_128_extension(
        head_dim=64, diagnostic_build=True
    )
    assert len(calls) == 1 and calls[0]["build_directory"] == str(tmp_path / "cache")
    assert "-lineinfo" not in calls[0]["extra_cuda_cflags"]


def test_compiler_failure_restores_arch_and_is_not_cached(monkeypatch, tmp_path):
    import torch.utils.cpp_extension as cpp

    monkeypatch.setattr(backend, "_EXTENSIONS", {})
    monkeypatch.setattr(backend, "resolve_cutlass_root", lambda root: tmp_path)
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)

    def fail(**kwargs):
        assert os.environ["TORCH_CUDA_ARCH_LIST"] == "9.0a"
        raise RuntimeError("mocked compiler failure")

    monkeypatch.setattr(cpp, "load_inline", fail)
    with pytest.raises(RuntimeError, match="mocked compiler"):
        backend.compile_micro_prefill_128_extension(head_dim=64, build_dir=tmp_path)
    assert "TORCH_CUDA_ARCH_LIST" not in os.environ and not backend._EXTENSIONS


def test_plan_passes_diagnostic_build_options_to_compiler(
    mocked_extension, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        backend,
        "compile_micro_prefill_128_extension",
        lambda **kwargs: calls.append(kwargs) or mocked_extension[0],
    )
    backend.Natural128AsyncMicroPrefillPlan.build(
        *_buffers(), diagnostic_build=True, lineinfo=True
    )
    assert calls[0]["diagnostic_build"] and calls[0]["lineinfo"]


@pytest.fixture
def binary_fixture(monkeypatch, tmp_path):
    binary = tmp_path / "extension.so"
    binary.write_bytes(b"exact loaded binary")
    ptx = tmp_path / "cuda.ptx"
    ptx.write_text(""".visible .entry _Zchosen() {
  wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {};
  wgmma.commit_group.sync.aligned;
  wgmma.wait_group.sync.aligned 1;
  wgmma.wait_group.sync.aligned 0;
}
.visible .entry _Zneighbor() { wgmma.wait_group.sync.aligned 7; }
""")
    (tmp_path / "cuda.sm_90a.cubin").write_bytes(b"retained cubin")
    commands = []
    resource_dump = """Resource usage:
 Common:
  GLOBAL:0
 Function _Zchosen:
  REG:232 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:480
 Function _Zneighbor:
  REG:99 STACK:0 SHARED:0 LOCAL:16
"""
    sass = """Function : _Zchosen
 /*0000*/ HGMMA.64x64x16.F32 R0, R1;
 /*0010*/ WARPGROUP.DEPBAR.LE SB0, 0x1;
 /*0020*/ @P0 BRA 0x0;
"""

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs["check"] and kwargs["timeout"] > 0
        if command == ["cu++filt", "--version"]:
            raise subprocess.CalledProcessError(
                1, command, stderr="unsupported cu++filt option: --version"
            )
        if command == ["cu++filt", "-v"]:
            out = "cu++filt V12.8"
        elif "--version" in command:
            out = "CUDA tool 12.8"
        elif "--dump-resource-usage" in command:
            out = resource_dump
        elif command[0] == "cu++filt":
            out = "void producer<(int)4, (bool)1, (bool)0, (bool)1>(int*)\nvoid producer<(int)4, (bool)1, (bool)0, (bool)0>(int*)\n"
        elif "--dump-sass" in command:
            assert command[-2] in ("_Zchosen", "_Zneighbor")
            out = (
                sass
                if command[-2] == "_Zchosen"
                else sass.replace("_Zchosen", "_Zneighbor")
            )
        else:
            raise AssertionError(command)
        return SimpleNamespace(stdout=out)

    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: name)
    monkeypatch.setattr(diagnostics.subprocess, "run", run)
    extension = SimpleNamespace(
        __file__=str(binary),
        _streamattn_build_metadata={
            "keep_intermediates": True,
            "intermediates_dir": str(tmp_path),
            "head_dim": 128,
        },
    )
    return extension, {"producer": "producer<4, true, false, true>"}, commands


def test_binary_diagnostics_uses_demangler_specific_version_flag(binary_fixture):
    extension, names, commands = binary_fixture
    report = diagnostics.inspect_cuda_binary(extension.__file__, kernel_names=names)
    assert ["cu++filt", "-v"] in commands
    assert ["cu++filt", "--version"] not in commands
    assert report["tools"]["cu++filt"]["version"] == "cu++filt V12.8"
    for tool in ("nvcc", "ptxas", "cuobjdump"):
        assert [tool, "--version"] in commands


@pytest.mark.parametrize("group", [4, 8])
@pytest.mark.parametrize(
    "overlap,direct,drain",
    [(1, 0, 0), (1, 0, 1), (0, 0, 0), (1, 1, 0), (1, 1, 1), (0, 1, 0)],
)
def test_actual_cuda128_demangler_template_spelling(group, overlap, direct, drain):
    # Spelling observed by running cu++filt on the saved D128 GPU binary's symbols.
    actual = (
        f"void streamattn_micro128_kernel<(int){group}, (bool){overlap}, "
        f"(bool){direct}, (bool){drain}>(const cutlass::bfloat16_t *, float *, int)"
    )
    expected = (
        f"streamattn_micro128_kernel<{group}, {str(bool(overlap)).lower()}, "
        f"{str(bool(direct)).lower()}, {str(bool(drain)).lower()}>"
    )
    assert diagnostics._kernel_identity(actual) == diagnostics._kernel_identity(
        expected
    )
    assert diagnostics._kernel_name(actual).endswith(f"(bool){drain}>")


@pytest.mark.parametrize("value", [False, True])
def test_cuda_demangler_temporal_bool_and_merge_identity(value):
    actual = f"void streamattn_temporal_micro_prefill_partial_kernel<(bool){int(value)}>(float *)"
    expected = f"streamattn_temporal_micro_prefill_partial_kernel<{str(value).lower()}>"
    assert diagnostics._kernel_identity(actual) == diagnostics._kernel_identity(
        expected
    )
    assert (
        diagnostics._kernel_identity(
            "streamattn_micro128_merge_kernel(const float *, int)"
        )
        == "streamattn_micro128_merge_kernel"
    )
    assert diagnostics._kernel_identity(actual) != diagnostics._kernel_identity(
        f"streamattn_temporal_micro_prefill_partial_kernel<{int(value)}>"
    )


def test_binary_archive_keeps_exact_symbol_resources_ptx_and_counts(
    binary_fixture, tmp_path
):
    extension, names, commands = binary_fixture
    report = diagnostics.inspect_extension(
        extension,
        tmp_path / "diagnostics",
        kernel_names=names,
        runtime_resources={
            "producer": {"blocks_per_sm": 2, "dynamic_shared_bytes": 81920}
        },
    )
    row = report["kernels"]["producer"]
    assert row["mangled_symbol"] == "_Zchosen"
    assert row["binary_resources"]["REG"] == 232
    assert row["binary_resources"]["LOCAL"] == 0
    assert row["runtime_resources"]["blocks_per_sm"] == 2
    assert row["binary"]["sha256"] == hashlib.sha256(b"exact loaded binary").hexdigest()
    counts = row["instruction_counts"]
    assert counts["ptx_wgmma_wait_group"] == {"0": 1, "1": 1}
    assert counts["ptx_wgmma_mma_async"] == 1
    assert counts["sass_relevant"]["WARPGROUP.DEPBAR.LE"] == 1
    assert counts["sass_relevant"]["HGMMA.64x64x16.F32"] == 1
    assert "HGMMA." not in counts["sass_opcodes"]
    assert counts["sass_opcodes"]["BRA"] == 1
    assert "_Zneighbor" not in row["sass_command"]
    blob = base64.b64decode(report["archive"]["data"])
    assert hashlib.sha256(blob).hexdigest() == report["archive"]["sha256"]
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        files = archive.getnames()
        assert any(name.endswith("manifest.json") for name in files)
        assert any(name.endswith("extension.so") for name in files)
        assert any(name.endswith(".cubin") for name in files)
        for identity in report["artifacts"].values():
            assert (
                hashlib.sha256(
                    archive.extractfile(identity["archive_member"]).read()
                ).hexdigest()
                == identity["sha256"]
            )
    assert (
        json.loads(Path(report["manifest"]["path"]).read_text())["selection"]
        == "exact_kernel_names"
    )
    assert commands and all(
        "--version" in command
        or "--dump" in " ".join(command)
        or command[0] == "cu++filt"
        for command in commands
    )


def test_binary_diagnostics_reject_nonexact_symbol_and_wrong_sass(
    binary_fixture, monkeypatch
):
    extension, names, _ = binary_fixture
    with pytest.raises(ValueError, match="exactly one"):
        diagnostics.inspect_cuda_binary(
            extension.__file__, kernel_names={"producer": "producer"}
        )
    original = diagnostics._run

    def wrong_sass(command, timeout):
        if "--dump-sass" in command:
            return "Function : _Zneighbor\n /*0000*/ EXIT;\n"
        return original(command, timeout)

    monkeypatch.setattr(diagnostics, "_run", wrong_sass)
    with pytest.raises(ValueError, match="only the exact"):
        diagnostics.inspect_cuda_binary(extension.__file__, kernel_names=names)


def test_binary_diagnostics_reject_missing_ptx_and_propagate_timeout(
    binary_fixture, monkeypatch, tmp_path
):
    extension, names, _ = binary_fixture
    with pytest.raises(ValueError, match="none was found"):
        diagnostics.inspect_cuda_binary(
            extension.__file__,
            kernel_names=names,
            build_metadata={
                "keep_intermediates": True,
                "intermediates_dir": str(tmp_path / "missing"),
            },
        )

    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(diagnostics.subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        diagnostics.inspect_cuda_binary(
            extension.__file__, kernel_names=names, timeout=1
        )


def test_binary_diagnostics_all_symbols_is_explicitly_not_timing_attribution(
    binary_fixture, tmp_path
):
    extension, _, _ = binary_fixture
    report = diagnostics.inspect_extension(
        extension, tmp_path / "all", include_archive=False
    )
    assert report["selection"] == "all_device_symbols"
    assert len(report["kernels"]) == 2
    assert "data" not in report["archive"]
    assert Path(report["archive"]["path"]).is_file()


def test_binary_diagnostics_reject_replaced_binary_and_ambiguous_resources(
    binary_fixture, monkeypatch
):
    extension, names, _ = binary_fixture
    with pytest.raises(ValueError, match="recorded at load time"):
        diagnostics.inspect_cuda_binary(
            extension.__file__,
            kernel_names=names,
            build_metadata={
                "loaded_binary": {"path": extension.__file__, "sha256": "old"}
            },
        )
    original = diagnostics._run

    def duplicate_resources(command, timeout):
        out = original(command, timeout)
        if "--dump-resource-usage" in command:
            out += "\nFunction _Zchosen:\n REG:232 STACK:0 LOCAL:0\n"
        return out

    monkeypatch.setattr(diagnostics, "_run", duplicate_resources)
    with pytest.raises(ValueError, match="ambiguous resource"):
        diagnostics.inspect_cuda_binary(extension.__file__, kernel_names=names)


def test_canary_is_bounded_and_exercises_direct_and_irregular_splits():
    smoke = canary_cases("smoke")
    assert len(smoke) == 3 and any(c["splits"] == 1 for c in smoke)
    assert any(c["n"] // 64 % c["splits"] for c in smoke)
    assert len(canary_cases("canary")) == 32
    for suite in ("smoke", "canary", "boundary"):
        assert all(1 <= c["splits"] <= c["n"] // 64 for c in canary_cases(suite))
    args = parse_args(["--mode", "resources", "--baselines", "none"])
    assert args.matches_splits and args.protocol == "both"
    assert not parse_args(["--no-matches-splits"]).matches_splits


def test_no_canary_cases_is_not_success():
    with pytest.raises(ValueError, match="no executable"):
        canary_cases("canary", (128,))


def test_resource_only_does_not_initialize_or_launch_kernels(monkeypatch):
    import benchmarks.profile_sm90_micro_prefill_128 as benchmark

    empty = torch.empty

    def cpu_empty(*args, **kwargs):
        kwargs["device"] = "cpu"
        return empty(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("resource mode must not initialize or launch kernels")

    monkeypatch.setattr(torch, "empty", cpu_empty)
    monkeypatch.setattr(torch, "Generator", forbidden)
    monkeypatch.setattr(benchmark, "fp32_reference", forbidden)

    def build(*args, **kwargs):
        return SimpleNamespace(
            resources=backend.decode_resource_info(_resource_vector()),
            direct=kwargs["direct"],
            protocol=kwargs["protocol"],
            resource_pass=True,
            producer_ctas=4,
            query_tiles=1,
            workspace_bytes=0,
            num_splits=kwargs["num_splits"],
        )

    monkeypatch.setattr(
        benchmark, "Natural128AsyncMicroPrefillPlan", SimpleNamespace(build=build)
    )
    args = parse_args(["--mode", "resources", "--baselines", "none"])
    result = profile_case(canary_cases("smoke")[0], args)
    assert result["status"] == "resources_only" and result["resource_pass"]


def test_resource_mode_collects_every_smoke_case_after_rejection(monkeypatch, tmp_path):
    import json
    import benchmarks.profile_sm90_micro_prefill_128 as benchmark

    seen = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(name="test H100", multi_processor_count=132),
    )

    def resources(case, args):
        seen.append(case)
        return dict(status="rejected_resources" if len(seen) == 1 else "resources_only")

    monkeypatch.setattr(benchmark, "profile_case", resources)
    destination = tmp_path / "resources.json"
    assert (
        benchmark.main(
            ["--mode", "resources", "--baselines", "none", "--output", str(destination)]
        )
        == 1
    )
    assert seen == canary_cases("smoke")
    result = json.loads(destination.read_text())
    assert len(result["rows"]) == 3 and not result["passed"]


def test_references_and_error_gates():
    q = torch.zeros(2, 3, 8, 4)
    k = torch.ones(2, 2, 5, 4)
    v = torch.arange(80).float().view(2, 2, 5, 4)
    out, lse = fp32_reference(q, k, v)
    for b in range(2):
        for h in range(8):
            torch.testing.assert_close(out[b, :, h], v[b, h // 4].mean(0).expand(3, -1))
    torch.testing.assert_close(lse, torch.full_like(lse, math.log(5)))
    assert not output_error(torch.tensor([float("nan")]), torch.ones(1))["passed"]
    assert not lse_error(torch.tensor([float("inf")]), torch.ones(1))["passed"]


@pytest.mark.skipif(
    os.environ.get("STREAMATTN_RUN_SM90_CANARY") != "1",
    reason="opt-in SM90 compilation/integration",
)
@pytest.mark.parametrize("d,g", [(64, 4), (128, 8)])
def test_gpu_complete_outputs_lse_and_s1_direct(d, g):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    q = torch.randn(1, 17, 16, d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, 16 // g, 320, d, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        expected, expected_lse = fp32_reference(q, k, v)
        for protocol in ("serial", "overlap", "overlap_drained"):
            for splits, direct in ((4, False), (1, False), (1, True)):
                plan = backend.Natural128AsyncMicroPrefillPlan.build(
                    q, k, v, num_splits=splits, protocol=protocol, direct=direct
                )
                assert plan.resource_pass, plan.resources
                assert output_error(plan.run(), expected)["passed"]
                assert lse_error(plan.lse, expected_lse)["passed"]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old
