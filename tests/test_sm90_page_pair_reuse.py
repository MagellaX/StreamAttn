import pytest

from benchmarks.profile_sm90_micro_prefill_mixed import experiment_cases, page_table, workload
from benchmarks.sm90_mixed_attribution import variants, geometry
from stream_attention.backends.sm90.micro_prefill_ragged_sources import ragged_cuda_source
from stream_attention.backends.sm90.micro_prefill_semantics_sources import _between


@pytest.mark.parametrize("nhd", [False, True])
@pytest.mark.parametrize("heads", [2, 8])
@pytest.mark.parametrize("tile", [0, 7])
def test_page_pair_addresses_and_guarded_tails(nhd, heads, tile):
    for permutation in (list(range(40)), list(reversed(range(40)))):
        for tail in range(65):
            length = tile * 64 + tail
            table = [page if i * 16 < length else None for i, page in enumerate(permutation)]
            for thread in range(128):
                for fragment in range(4):
                    token0 = fragment * 16 + thread // 16
                    j0, dim = tile * 64 + token0, (thread % 16) * 8
                    valid0, valid1 = j0 < length, j0 + 8 < length
                    assert not valid1 or valid0
                    source0 = source1 = None
                    if valid0:
                        page = table[tile * 4 + fragment]
                        assert page is not None
                        row = ((page * 16 + thread // 16) * heads + heads - 1 if nhd
                               else (page * heads + heads - 1) * 16 + thread // 16)
                        source0 = row * 128 + dim
                        if valid1:
                            source1 = source0 + 8 * 128 * (heads if nhd else 1)
                    for u, address in enumerate((source0, source1)):
                        vector = thread + 128 * (2 * fragment + u)
                        token, old_dim = vector // 16, (vector % 16) * 8
                        logical = tile * 64 + token
                        assert (token, old_dim) == (token0 + 8 * u, dim)
                        expected = None
                        if logical < length:
                            page, offset = table[logical // 16], logical % 16
                            row = ((page * 16 + offset) * heads + heads - 1 if nhd
                                   else (page * heads + heads - 1) * 16 + offset)
                            expected = row * 128 + old_dim
                        assert address == expected


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("affine", ["none", "index", "interior"])
def test_pair_source_scope_and_pipeline(dtype, affine):
    control = ragged_cuda_source(128, dtype, True, affine, True)
    pair = ragged_cuda_source(128, dtype, True, affine, True, True)
    assert control != pair
    assert "if (valid1) source1 = source0 + half_page_stride" in pair
    assert "target0), valid0" in pair and "target1), valid1" in pair
    for transposed in ("false", "true"):
        assert f"streamattn_micro_load_page16_pair<kNHD, {transposed}>" in pair
    # The page-alias caller and the original helper remain byte-identical.
    for start in ("__forceinline__ __device__ void streamattn_micro_load_page16(",
                  "__forceinline__ __device__ void streamattn_copy_paged16_tile("):
        assert pair.split(start, 1)[1].split("\n}\n", 1)[0] == control.split(start, 1)[1].split("\n}\n", 1)[0]
    for marker in ("  copy_k_tile(tile_begin, sK0);",):
        stop = "\nvoid micro_components_out("
        assert _between(pair, marker, stop) == _between(control, marker, stop)
    assert ragged_cuda_source(64, dtype, True, affine, True, True) == ragged_cuda_source(64, dtype, True, affine, True)


def test_pair_requires_vector_control():
    with pytest.raises(ValueError, match="vector Q control"):
        ragged_cuda_source(128, "bf16", True, page_pair_reuse=True)


def test_pair_variants_and_predeclared_holdout():
    from types import SimpleNamespace

    configs = variants(SimpleNamespace(page_pair=True))
    assert list(configs) == ["interior_q_vector", "interior_q_vector_page_pair"]
    assert all(c["q_vector_copy"] for c in configs.values())
    old = experiment_cases("causal") + experiment_cases("holdout")
    holdout = experiment_cases("pair_holdout")
    assert len(holdout) == 24 and all(c not in old for c in holdout)
    for case in holdout:
        assert geometry(case, "interior_q_vector") == geometry(case, "interior_q_vector_page_pair")
    edges = experiment_cases("pair_edges")
    assert len(edges) == 16
    assert {n % 64 for c in edges for n in c["kv_lengths"]} == set(range(64))


def test_machine_counts_preserve_widths_and_do_not_invent_missing_metrics():
    from benchmarks.summarize_sm90_paged_source import machine_instructions

    def pc(instruction, count, source=""):
        return dict(instruction=instruction, metrics={"Instructions Executed": count},
                    correlations=[dict(source=source)])
    counts = machine_instructions([
        pc("@P0 LDG.E R0, [R2]", 7, "const int page = table[index];"),
        pc("LDG.E.128 R4, [R6]", 3), pc("LDGSTS.E.BYPASS.LTC128B.128 [R0], [R2]", 9),
        pc("HGMMA.64x64x16.F32.BF16 R0", 5), pc("HGMMA.64x128x16.F32.BF16 R0", 4)])
    assert counts["source_correlated_page_load_warp_instructions"] == 7
    assert counts["kv_copy_warp_instructions"] == 9
    assert len(counts["opcodes"]) == 5
    missing = pc("LDG.E R0, [R2]", 1)
    missing["metrics"].clear()
    assert machine_instructions([missing]) is None


@pytest.mark.parametrize("suite", ["pair_regression", "pair_holdout"])
def test_every_pair_case_has_valid_workload_metadata(suite):
    for case in experiment_cases(suite):
        batch = workload(case, page_table(case, 59093))
        assert len(batch.requests) == len(case["query_lengths"])
        assert len({request.phase for request in batch.requests}) == 2
        for request in batch.requests:
            assert request.shared_prefix_len <= min(case["kv_lengths"])


def test_pair_timing_summary_uses_vector_control_and_separates_d64():
    from benchmarks.summarize_sm90_post_affine_attribution import summarize

    control, pair = "interior_q_vector", "interior_q_vector_page_pair"
    rows = []
    for dim in (64, 128):
        case = dict(d=dim, trace="test")
        trials = [dict(us={f"{control}/packed": 10, f"{pair}/packed": 10 if dim == 64 else 5,
                           "flashinfer_fa2/packed": 2})]
        rows.append(dict(case=case, passed=True,
            loaded_binary_provenance={name: dict(resolved=True) for name in (control, pair)},
            graphs=dict(packed=dict(paired_trials=trials,
                fastest_tested_baseline=dict(baseline_id="flashinfer_fa2", correctness_passed=True))),
            attribution=dict(native={name: dict(geometry={}, attributes={}, workspace_allocated_bytes=0,
                                                isolated_median_us={}) for name in (control, pair)})))
    result = summarize(dict(schema="streamattn.sm90_micro_prefill_mixed.v1",
        experiment="page_pair_reuse", rows=rows, planned_cases=2, complete=True,
        suite="pair_holdout", environment={}, seed=1))
    assert result["control"] == control and not result["promotion"]
    data = result["interfaces"]["packed"]["warm"][pair]["by_dimension"]
    assert data["64"]["control_geomean"] == 1
    assert data["128"]["control_geomean"] == 2
    assert data["128"]["baseline_geomean"] == pytest.approx(0.4)
