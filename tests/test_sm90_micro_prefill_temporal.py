import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from stream_attention.backends.sm90 import micro_prefill_temporal as backend
from stream_attention.backends.sm90 import micro_prefill_temporal_sources as sources
from stream_attention.backends.sm90.micro_prefill import (
    choose_natural_micro_prefill_splits,
)
from stream_attention.backends.sm90.transposed_gqa_exact_sources import (
    cuda_source_for_head_dim as original_source,
)


@pytest.mark.parametrize("g", [4, 8])
def test_r64_mapping_covers_every_position_head_and_batch(g):
    for m in range(1, 65):
        tiles = backend.query_tiles_temporal(m, g)
        covered = [
            (b, tile * (64 // g) + row // g, kv * g + row % g)
            for b in range(2)
            for kv in range(3)
            for tile in range(tiles)
            for row in range(64)
            if tile * (64 // g) + row // g < m
        ]
        expected = {(b, q, h) for b in range(2) for q in range(m) for h in range(3 * g)}
        assert len(covered) == len(expected) and set(covered) == expected


@pytest.mark.parametrize("tiles", [1, 2, 5, 7, 19, 64, 65, 513])
def test_balanced_split_boundaries_are_nonempty_and_complete(tiles):
    for splits in {1, min(tiles, 24), min(tiles, 512)}:
        intervals = [
            backend.balanced_tile_interval(tiles, splits, s) for s in range(splits)
        ]
        assert all(begin < end for begin, end in intervals)
        assert [t for begin, end in intervals for t in range(begin, end)] == list(
            range(tiles)
        )
        lengths = [end - begin for begin, end in intervals]
        assert max(lengths) - min(lengths) <= 1
    with pytest.raises(ValueError):
        backend.balanced_tile_interval(tiles, tiles + 1, 0)


@pytest.mark.parametrize("g,d", [(4, 64), (4, 128), (8, 64), (8, 128)])
def test_partial_abi_and_verbatim_merge_indexing(g, d):
    b, m, hk, splits = 2, 17, 2, 4
    tiles = backend.query_tiles_temporal(m, g)
    state = torch.arange(b * hk * tiles * splits * 64).reshape(
        b * hk * tiles, splits, 64
    )
    for bi in range(b):
        for qi in range(m):
            for head in range(hk * g):
                work = (bi * hk + head // g) * tiles + qi // (64 // g)
                row = (qi % (64 // g)) * g + head % g
                for split in range(splits):
                    flat = (work * splits + split) * 64 + row
                    assert state[work, split, row] == flat
    assert state.numel() * (d + 1) * 4 == b * hk * tiles * splits * 64 * (d + 1) * 4


def _stats(scores, row_max, row_sum, scale):
    next_max = torch.maximum(row_max, scores.max(dim=1).values)
    alpha = torch.where(
        row_max == -torch.inf, 0.0, torch.exp2((row_max - next_max) * scale)
    )
    probability = torch.exp2(scores * scale - (next_max * scale)[:, None])
    return next_max, row_sum * alpha + probability.sum(1), alpha, probability


def _partial(scores, values, *, temporal, drained=False, quantized=True):
    scale = 0.18033688011112042 if values.shape[1] == 64 else 0.12751743082459868
    maximum = torch.full((64,), -torch.inf, dtype=torch.float64)
    denominator = torch.zeros(64, dtype=torch.float64)
    numerator = torch.zeros(64, values.shape[1], dtype=torch.float64)

    def pack(p):
        return p.to(torch.bfloat16).double() if quantized else p

    if not temporal:
        for tile in range(scores.shape[1] // 64):
            maximum, denominator, alpha, probability = _stats(
                scores[:, tile * 64 : (tile + 1) * 64], maximum, denominator, scale
            )
            numerator *= alpha[:, None]
            numerator += pack(probability) @ values[tile * 64 : (tile + 1) * 64]
    else:
        maximum, denominator, _, probability = _stats(
            scores[:, :64], maximum, denominator, scale
        )
        current_p = pack(probability)
        for tile in range(scores.shape[1] // 64 - 1):
            current_v = values[tile * 64 : (tile + 1) * 64]
            # A full-drain control retires PV before softmax, not after it.
            if drained:
                numerator += current_p @ current_v
            old_n, old_p = numerator.clone(), current_p.clone()
            maximum, denominator, alpha, next_p = _stats(
                scores[:, (tile + 1) * 64 : (tile + 2) * 64],
                maximum,
                denominator,
                scale,
            )
            assert torch.equal(numerator, old_n) and torch.equal(current_p, old_p)
            if not drained:
                numerator += current_p @ current_v
            numerator *= alpha[:, None]
            current_p = pack(next_p)
        numerator += current_p @ values[-64:]
    return numerator / denominator[:, None], maximum * scale + torch.log2(denominator)


@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("tiles,splits", [(1, 1), (2, 1), (5, 4), (7, 3), (16, 1)])
def test_temporal_and_drained_match_serial_state_with_late_maxima(d, tiles, splits):
    generator = torch.Generator().manual_seed(812)
    scores = torch.randn(64, tiles * 64, dtype=torch.float64, generator=generator)
    scores[:32, -64:] += 90
    scores[32:, :64] += 70
    values = torch.randn(tiles * 64, d, dtype=torch.float64, generator=generator)
    for split in range(splits):
        begin, end = backend.balanced_tile_interval(tiles, splits, split)
        z, v = scores[:, begin * 64 : end * 64], values[begin * 64 : end * 64]
        expected = _partial(z, v, temporal=False)
        for drained in (False, True):
            actual = _partial(z, v, temporal=True, drained=drained)
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("d", [64, 128])
def test_split_state_merge_matches_complete_attention_and_log2_lse(d):
    generator = torch.Generator().manual_seed(81)
    scores = torch.randn(64, 320, dtype=torch.float64, generator=generator)
    scores[:32, -64:] += 80
    values = torch.randn(320, d, dtype=torch.float64, generator=generator)
    partials = []
    for split in range(4):
        begin, end = backend.balanced_tile_interval(5, 4, split)
        partials.append(
            _partial(
                scores[:, begin * 64 : end * 64],
                values[begin * 64 : end * 64],
                temporal=True,
                quantized=False,
            )
        )
    outputs = torch.stack([p[0] for p in partials])
    lses = torch.stack([p[1] for p in partials])
    weights = torch.exp2(lses - lses.max(0).values)
    actual = (outputs * weights[:, :, None]).sum(0) / weights.sum(0)[:, None]
    scale = 0.18033688011112042 if d == 64 else 0.12751743082459868
    logits = scores * (scale * math.log(2))
    torch.testing.assert_close(
        actual, logits.softmax(1) @ values, atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(
        torch.logsumexp(lses * math.log(2), dim=0), torch.logsumexp(logits, dim=1)
    )


@pytest.mark.parametrize("drained", [False, True])
@pytest.mark.parametrize("length", [1, 2, 3, 4, 16])
def test_schedule_owns_one_output_and_releases_p_v_and_k_before_reuse(drained, length):
    pending = []
    k_slots = {0: 0}
    v_tile = 0

    def wait(count):
        del pending[: max(0, len(pending) - count)]

    if length > 1:
        k_slots[1] = 1
    for tile in range(length - 1):
        assert not pending  # Full footer drain removes loop-carried GMMA state.
        if tile + 2 < length:
            k_slots[tile % 2] = tile + 2
        assert k_slots[(tile + 1) % 2] == tile + 1 and v_tile == tile
        pending.extend([("scores",), ("output_acc", "p_regs", "sV")])
        wait(0 if drained else 1)
        assert all("scores" not in operands for operands in pending)
        if not drained:
            assert pending == [("output_acc", "p_regs", "sV")]
        # Softmax is allowed to change scores/max/sum/alpha only.
        assert not {"scores", "row_max", "row_sum", "row_alpha"}.intersection(
            operand for operands in pending for operand in operands
        )
        wait(0)
        assert not pending
        v_tile = tile + 1
    assert v_tile == length - 1
    pending.append(("output_acc", "p_regs", "sV"))
    wait(0)
    assert not pending


def _resources(local=0, blocks=2, merge_local=0, merge_blocks=16):
    return torch.tensor(
        [160, 0, 65536, local, blocks, 128, 32, 2056, 0, merge_local, merge_blocks, 128]
    )


@pytest.fixture
def fake_extension(monkeypatch):
    calls, queries, builds = [], [], []

    def resource_info(*args):
        queries.append(args)
        return _resources()

    extension = SimpleNamespace(
        out=lambda *args: calls.append(args), resource_info=resource_info
    )
    monkeypatch.setattr(backend, "supports_sm90_temporal", lambda *args: True)
    monkeypatch.setattr(
        backend,
        "compile_micro_prefill_temporal_extension",
        lambda **kwargs: builds.append(kwargs) or extension,
    )
    return extension, calls, queries, builds


def _buffers(*, b=1, m=17, g=4, d=64, n=320):
    return (
        torch.empty(b, m, 2 * g, d, dtype=torch.bfloat16),
        torch.empty(b, 2, n, d, dtype=torch.bfloat16),
        torch.empty(b, 2, n, d, dtype=torch.bfloat16),
    )


@pytest.mark.parametrize("protocol", ["temporal", "drained"])
@pytest.mark.parametrize("splits", [1, 4])
def test_plan_abi_components_and_allocation_free_replay(
    fake_extension, monkeypatch, protocol, splits
):
    extension, calls, queries, builds = fake_extension
    q, k, v = _buffers()
    plan = backend.TemporalMicroPrefillPlan.build(
        q, k, v, num_splits=splits, protocol=protocol, diagnostic_build=True
    )
    assert plan.output.shape == q.shape and not hasattr(plan, "lse")
    assert plan.partial_output.shape == (4, splits, 64, 64)
    assert plan.partial_lse.shape == (4, splits, 64)
    assert plan.workspace_bytes == 4 * splits * 64 * 65 * 4
    assert plan.producer_ctas == 4 * splits and plan.resource_pass
    assert queries[0][1] == backend.PROTOCOLS[protocol] and len(queries) == 1
    assert builds[0]["diagnostic_build"]

    def forbidden(*args, **kwargs):
        raise AssertionError("allocation/compilation/resource query during replay")

    pointers = [
        t.data_ptr() for t in (plan.output, plan.partial_output, plan.partial_lse)
    ]
    monkeypatch.setattr(torch, "empty", forbidden)
    monkeypatch.setattr(torch, "empty_like", forbidden)
    monkeypatch.setattr(backend, "compile_micro_prefill_temporal_extension", forbidden)
    extension.resource_info = forbidden
    assert plan.run() is plan.output
    plan.run_component("producer")
    plan.run_component("merge")
    assert [call[-3:] for call in calls] == [
        (splits, component, backend.PROTOCOLS[protocol]) for component in (0, 1, 2)
    ]
    assert pointers == [
        t.data_ptr() for t in (plan.output, plan.partial_output, plan.partial_lse)
    ]
    assert len(queries) == 1


def test_anchor_geometry_and_default_splits_match_original(fake_extension):
    for b, n in ((1, 4096), (1, 16384), (2, 4096)):
        q, k, v = _buffers(b=b, m=64, g=8, d=128, n=n)
        plan = backend.TemporalMicroPrefillPlan.build(q, k, v, num_splits=16)
        assert plan.producer_ctas == b * 256
        assert plan.workspace_bytes == b * 16 * 16 * 64 * 129 * 4
        lengths = [
            end - begin
            for begin, end in (
                backend.balanced_tile_interval(n // 64, 16, s) for s in range(16)
            )
        ]
        assert lengths == [n // 1024] * 16
        default = backend.TemporalMicroPrefillPlan.build(q, k, v)
        assert default.num_splits == choose_natural_micro_prefill_splits(
            batch=b, query_len=64, kv_heads=2, group_size=8, kv_len=n
        )
    single = backend.TemporalMicroPrefillPlan.build(*_buffers(m=1, n=64), num_splits=1)
    assert single.query_tiles == 1 and single.workspace_bytes > 0


@pytest.mark.parametrize(
    "options",
    [
        {"num_splits": 0},
        {"num_splits": 6},
        {"num_splits": True},
        {"num_splits": 1.5},
        {"protocol": "overlap"},
        {"target_producer_ctas": 0},
    ],
)
def test_invalid_options_fail_before_compilation(fake_extension, options):
    with pytest.raises(ValueError):
        backend.TemporalMicroPrefillPlan.build(*_buffers(), **options)
    assert not fake_extension[1] and not fake_extension[3]


def test_shape_and_output_contracts_are_not_expanded_to_other_dtypes(
    fake_extension, monkeypatch
):
    q, k, v = _buffers()
    for bad in (
        (q.half(), k.half(), v.half()),
        (q[:, :0], k, v),
        (q, k[:, :, :-1], v[:, :, :-1]),
    ):
        with pytest.raises(ValueError, match="unsupported"):
            backend.TemporalMicroPrefillPlan.build(*bad)
    with pytest.raises(ValueError, match="output"):
        backend.TemporalMicroPrefillPlan.build(q, k, v, output=q.float())
    monkeypatch.setattr(backend, "supports_sm90_temporal", lambda *args: False)
    with pytest.raises(ValueError, match="SM90"):
        backend.TemporalMicroPrefillPlan.build(q, k, v)


@pytest.mark.parametrize(
    "kwargs", [{"local": 4}, {"blocks": 1}, {"merge_local": 4}, {"merge_blocks": 0}]
)
def test_resource_gate_rejects_spills_and_missing_residency(kwargs):
    assert not backend.resource_gate(
        backend.decode_resource_info(_resources(**kwargs))
    )["passed"]
    assert backend.resource_gate(backend.decode_resource_info(_resources()))["passed"]
    with pytest.raises(ValueError, match="12"):
        backend.decode_resource_info(torch.zeros(6))


@pytest.mark.parametrize("d", [64, 128])
def test_composed_source_contains_only_new_producer_and_verbatim_original_merge(d):
    source = sources.cuda_source_for_head_dim(d)
    merge = sources.original_merge_source(d)
    assert merge in original_source(d) and merge in source
    assert source.count("__global__") == 2
    assert "void streamattn_natural_wgmma_micro_prefill_partial_kernel(" not in source
    assert "void streamattn_grouped_wgmma_prefill_kernel(" not in source
    assert "sizeof(GroupedRSPrefillSharedStorage) == 512 * kHeadDim" in source
    assert source.count("Tensor output_acc = partition_fragment_C(") == 1
    assert "exp2f(score_rows(row, col) * kTemporalScaleLog2 - max_scaled)" in source
    assert "cutlass::NumericArrayConverter<Element, Accum, count>" in source
    assert "warpgroup_reg_alloc" not in source and "__launch_bounds__(128)" in source
    assert "static_cast<int64_t>(split) * num_tiles / num_splits" in source


def test_overlap_window_cannot_access_output_or_current_probability():
    source = sources.CUDA_SOURCE
    helper = source.split("void temporal_softmax_next(", 1)[1].split(
        "template <class Scores, class Probability>", 1
    )[0]
    assert "output_acc" not in helper and "p_regs" not in helper
    window = source.split("else { warpgroup_wait<1>(); }", 1)[1].split(
        "warpgroup_wait<0>();", 1
    )[0]
    code = "\n".join(
        line for line in window.splitlines() if not line.strip().startswith("//")
    )
    assert "output_acc" not in code and "p_regs" not in code
    assert "temporal_softmax_next(scores, row_max, row_sum, row_alpha);" in code
    footer = source.split("// PV[t] must retire", 1)[1].split("read_pipe ^= 1;", 1)[0]
    assert footer.index("warpgroup_wait<0>();") < footer.index(
        "output_rows(row, col) *= row_alpha(row)"
    )
    assert footer.index("warpgroup_fence_operand(p_regs);") < footer.index(
        "temporal_pack_p(scores, p_regs);"
    )
    assert footer.index("__syncthreads();") < footer.index("copy_v_tile(tile + 1);")


def test_launches_have_no_resource_queries_or_allocations():
    launch = sources.CUDA_SOURCE.split("static void temporal_launch(", 1)[1].split(
        "template <class Kernel>", 1
    )[0]
    for forbidden in (
        "cudaFuncSetAttribute",
        "cudaFuncGetAttributes",
        "cudaMalloc",
        "torch::empty",
    ):
        assert forbidden not in launch
    assert "CUDAGuard guard(q.device())" in launch and "at::assert_no_overlap" in launch
    assert "if (component != 1)" in launch and "if (component != 2)" in launch


def test_composition_anchor_changes_fail_closed(monkeypatch):
    monkeypatch.setattr(sources, "_base_source", lambda d: "unexpected source")
    with pytest.raises(ValueError, match="anchor"):
        sources.cuda_source_for_head_dim(64)
    with pytest.raises(ValueError, match="anchor"):
        sources.original_merge_source(64)


def test_compiler_diagnostics_and_cache_are_isolated(monkeypatch, tmp_path):
    import torch.utils.cpp_extension as cpp

    calls = []
    monkeypatch.setattr(backend, "_EXTENSIONS", {})
    monkeypatch.setattr(backend, "resolve_cutlass_root", lambda path: tmp_path)
    monkeypatch.setattr(
        cpp,
        "load_inline",
        lambda **kwargs: (
            calls.append(kwargs)
            or SimpleNamespace(
                __file__=str(Path(kwargs["build_directory"]) / "canary.so")
            )
        ),
    )
    monkeypatch.setattr(
        cpp, "get_default_build_root", lambda: str(tmp_path / "default")
    )
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.0")
    compile = backend.compile_micro_prefill_temporal_extension
    plain = compile(head_dim=64, build_dir=tmp_path)
    assert compile(head_dim=64, build_dir=tmp_path) is plain
    diagnostic = compile(head_dim=64, build_dir=tmp_path, diagnostic_build=True)
    assert compile(head_dim=64, build_dir=tmp_path, diagnostic_build=True) is diagnostic
    compile(head_dim=128, build_dir=tmp_path, diagnostic_build=True)
    compile(head_dim=64)
    assert len(calls) == 4 and len({row["build_directory"] for row in calls}) == 4
    assert calls[0]["name"] != calls[1]["name"] != calls[2]["name"]
    assert "--keep" not in calls[0]["extra_cuda_cflags"]
    assert {"-lineinfo", "--keep", "--keep-dir=."}.issubset(
        calls[1]["extra_cuda_cflags"]
    )
    assert "--ptxas-options=-v,--warn-on-spills" in calls[0]["extra_cuda_cflags"]
    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "8.0"
    assert backend.source_fingerprint(64) != backend.source_fingerprint(128)
    assert Path(diagnostic.__file__).parent.is_dir()


def test_compile_failure_restores_arch_and_is_not_cached(monkeypatch, tmp_path):
    import torch.utils.cpp_extension as cpp

    monkeypatch.setattr(backend, "_EXTENSIONS", {})
    monkeypatch.setattr(backend, "resolve_cutlass_root", lambda path: tmp_path)
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)

    def fail(**kwargs):
        assert os.environ["TORCH_CUDA_ARCH_LIST"] == "9.0a"
        raise RuntimeError("mock compiler failure")

    monkeypatch.setattr(cpp, "load_inline", fail)
    with pytest.raises(RuntimeError, match="mock compiler"):
        backend.compile_micro_prefill_temporal_extension(
            head_dim=64, build_dir=tmp_path
        )
    assert not backend._EXTENSIONS and "TORCH_CUDA_ARCH_LIST" not in os.environ
