from types import SimpleNamespace

import pytest

from benchmarks.profile_sm90_micro_prefill_mixed import experiment_cases, page_table, workload
from benchmarks.sm90_mixed_attribution import geometry, variants
from benchmarks.sm90_static_sass import normalize_capture, producer_sass
from stream_attention.backends.sm90.micro_prefill_ragged_sources import ragged_cuda_source


@pytest.mark.parametrize("nhd", [False, True])
def test_unsigned_offsets_widen_before_multiply(nhd):
    # Exercise >4GiB offsets and non-power-of-two head counts without allocating them.
    for page in (0, 1, 1048575, 1048576, 2**31 - 1):
        for heads in (1, 2, 3, 8, 31, 256):
            for head in (0, heads - 1):
                for row in range(8):
                    expected = ((page * 16 + row) * heads + head if nhd
                                else (page * heads + head) * 16 + row)
                    page_heads = page * heads
                    observed = (page_heads * 16 + row * heads + head if nhd
                                else (page_heads + head) * 16 + row)
                    for dim in (0, 120):
                        for member in (0, 1):
                            stride = member * 8 * 128 * (heads if nhd else 1)
                            assert observed * 128 + dim + stride == expected * 128 + dim + stride
                            assert (observed * 128 + dim + stride) * 2 < 2**63


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("affine", ["none", "index", "interior"])
def test_unsigned_scope_preserves_entire_source_except_row_expression(dtype, affine):
    control = ragged_cuda_source(128, dtype, True, affine, True, True)
    candidate = ragged_cuda_source(128, dtype, True, affine, True, True, True)
    start = """      const int64_t row = kNHD
          ? (static_cast<int64_t>(page) * 16 + row_in_half) * kv_heads + head
          : (static_cast<int64_t>(page) * kv_heads + head) * 16 + row_in_half;
"""
    end = "      source0 = base"
    replacement = candidate.split("      // Active page IDs", 1)[1].split(end, 1)[0]
    assert control.count(start) == 1
    expected = control.replace(start, "      // Active page IDs" + replacement)
    assert candidate == expected
    assert "static_cast<uint64_t>(static_cast<uint32_t>(page))" in candidate
    assert "if (valid0) {" in candidate and "if (valid1) source1" in candidate
    assert ragged_cuda_source(64, dtype, True, affine, True, True, True) == ragged_cuda_source(64, dtype, True, affine, True, True)


def test_unsigned_requires_page_pair():
    with pytest.raises(ValueError, match="page-pair control"):
        ragged_cuda_source(128, "bf16", True, unsigned_page_address=True)


def test_address_cases_and_fixed_geometry():
    configs = variants(SimpleNamespace(page_address=True))
    assert list(configs) == ["interior_q_vector_page_pair", "interior_page_address_unsigned"]
    old = sum((experiment_cases(s) for s in ("causal", "holdout", "pair_holdout")), [])
    assert len(experiment_cases("address_canary")) == 4
    fresh = experiment_cases("address_holdout")
    assert len(fresh) == 24 and all(c not in old for c in fresh)
    for c in fresh + experiment_cases("address_canary"):
        workload(c, page_table(c, 79099))
        assert geometry(c, list(configs)[0]) == geometry(c, list(configs)[1])


def test_static_sass_is_not_dynamic_evidence():
    fixture = ""
    for layout in (0, 1):
        fixture += f"Function : streamattn_natural_wgmma_micro_prefill_partial_kernel_{layout}\n"
        fixture += "/*0010*/ @!PT LDGSTS.E.BYPASS.LTC128B.128 [R0], [R2];\n"
        fixture += "/*0020*/ IMAD.WIDE.U32 R0, R1, R2, R3;\n"
        fixture += "/*0030*/ HGMMA.64x128x16.F32.BF16 R0, gdesc[UR4], gdesc[UR6], !P0;\n"
    rows = producer_sass(fixture)
    assert len(rows) == 2 and all(r["instruction_sites"] == 3 for r in rows.values())
    assert all(r["opcodes"]["HGMMA.64x128x16.F32.BF16"] == 1 for r in rows.values())
    with pytest.raises(ValueError, match="expected HND and NHD"):
        producer_sass("")
    binary = dict(binary_sha256="example", functions=rows)
    payload = dict(rows=[dict(static_producer_sass=dict(control=binary, candidate=binary),
                              graphs={"packed": {"median_us": {"control": 3, "candidate": 2}}})])
    normalized = normalize_capture(payload)
    assert len(normalized["static_producer_sass"]) == 1
    assert normalized["rows"][0]["graphs"] == payload["rows"][0]["graphs"]
    assert normalize_capture(normalized) == normalized
    assert isinstance(payload["rows"][0]["static_producer_sass"]["control"], dict)
