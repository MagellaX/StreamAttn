import csv
import io

import pytest

from benchmarks.summarize_sm90_micro_source_counters import source_pcs, summarize


def exported_source(duplicate_stall=7):
    output = io.StringIO()
    writer = csv.writer(output)
    for name, stall in (("kernel.cu", 7), ("inlined.hpp", duplicate_stall)):
        writer.writerow(["File Path", name])
        writer.writerow(["Function Name", "kernel"])
        writer.writerow(["Line No", "Source", "Address", "Source", "# Samples", "stall_long_sb"])
        writer.writerow(["12", "sQ(i) = q[i];", "-", "-", "10", "7"])
        writer.writerow(["", "", "0x123", "ST.E.U16 R0, R1", "10", str(stall)])
        writer.writerow(["", "", "...", "...", "-", "-"])
    return output.getvalue()


def test_inline_duplicates_and_cuda_totals_are_not_counted_twice():
    pcs = source_pcs(exported_source())
    assert len(pcs) == 1
    assert pcs[0]["metrics"] == {"# Samples": 10, "stall_long_sb": 7}
    assert pcs[0]["instruction"] == "ST.E.U16 R0, R1"
    assert len(pcs[0]["correlations"]) == 2


def test_conflicting_duplicate_pc_is_rejected():
    with pytest.raises(ValueError, match="conflicting"):
        source_pcs(exported_source(8))


def test_empty_source_is_not_evidence():
    with pytest.raises(ValueError, match="no source"):
        source_pcs("no kernels profiled")


@pytest.mark.parametrize("samples", [10, 11])
def test_source_sample_totals_must_match_kernel_aggregate(samples):
    payload = dict(schema="streamattn.sm90_micro_prefill_counters.v1", source_correlated=True,
                   complete=True, rows=[dict(batch=1, n=4096, source_csv=exported_source(),
                   profiler_csv='"smsp__pcsamp_sample_count","smsp__pcsamp_warps_issue_stalled_long_scoreboard"\n'
                                f'"{samples}","7"\n')])
    if samples == 10:
        assert summarize(payload)["rows"][0]["aggregate_totals_match"]
    else:
        with pytest.raises(ValueError, match="totals disagree"):
            summarize(payload)
