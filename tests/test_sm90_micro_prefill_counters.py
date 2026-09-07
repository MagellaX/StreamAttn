from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import profile_sm90_micro_prefill_counters as counters


@pytest.mark.parametrize("outcome", ["missing", "denied", "no_kernels", "collected"])
def test_counter_collection_never_promotes_missing_evidence(monkeypatch, tmp_path, outcome):
    monkeypatch.setattr(counters.torch.cuda, "get_device_name", lambda: "test H100")
    monkeypatch.setattr(counters.shutil, "which", lambda _: None if outcome == "missing" else "/ncu")
    monkeypatch.setattr(counters.subprocess, "check_output", lambda *a, **kw: "test ncu")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        raw = {
            "denied": "ERR_NVGPUCTRPERM",
            "no_kernels": '"Metric Name"\n',
            "collected": '"Metric Name"\nstreamattn_natural_wgmma_micro_prefill_partial_kernel\n',
        }[outcome]
        Path(command[command.index("--log-file") + 1]).write_text(raw)
        return SimpleNamespace(returncode=1 if outcome == "denied" else 0, stdout="", stderr="")

    monkeypatch.setattr(counters.subprocess, "run", run)
    args = SimpleNamespace(build_dir=tmp_path, cutlass_root=tmp_path)
    result = counters.collect(args)
    assert result["dynamic_counters_collected"] == (outcome == "collected")
    assert result["complete"] == (outcome != "no_kernels")
    assert len(calls) == (6 if outcome == "collected" else 0 if outcome == "missing" else 1)
    if calls:
        assert "streamattn_counter/" in calls[0]
        assert "--launch-count" in calls[0]
        assert calls[0][calls[0].index("--clock-control") + 1] == "none"


def test_source_counter_mode_exports_original_producer_only(monkeypatch, tmp_path):
    monkeypatch.setattr(counters.torch.cuda, "get_device_name", lambda: "test H100")
    monkeypatch.setattr(counters.shutil, "which", lambda _: "/ncu")
    monkeypatch.setattr(counters.subprocess, "check_output", lambda *a, **kw: "test ncu")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if "--import" in command:
            return SimpleNamespace(returncode=0, stdout="source metrics", stderr="")
        Path(command[command.index("--log-file") + 1]).write_text(
            '"Kernel Name","smsp__pcsamp_warps_issue_stalled_long_scoreboard"\n'
            'streamattn_natural_wgmma_micro_prefill_partial_kernel,42\n')
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(counters.subprocess, "run", run)
    result = counters.collect(SimpleNamespace(build_dir=tmp_path, cutlass_root=tmp_path,
                                             source_correlated=True))
    assert result["complete"] and result["source_correlated"]
    assert len(calls) == 6
    assert len(result["rows"]) == 3
    assert all(r["variant"] == "control" and r["source_csv"] for r in result["rows"])
    assert "SourceCounters" in calls[0] and "--import-source" in calls[0]
    assert "cuda,sass" in calls[1]
