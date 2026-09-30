from argparse import Namespace
from pathlib import Path

import pytest

from benchmarks.profile_adaptive_group_frontier import summarize
from benchmarks.run_lightning_adaptive_two_gate import job_command, upload_capture


def test_summary_does_not_call_more_omissions_a_traffic_win():
    base = dict(radius="origin", decision="two_gate", head_group_size=1,
                query_tile_size=16, max_error=0.0, max_bound=0.0)
    records = [dict(results=[dict(base, kv_read_ratio_vs_full_group=7.8,
                                 pv_regions_saved_fraction=0.05)]),
               dict(results=[dict(base, kv_read_ratio_vs_full_group=8.0,
                                 pv_regions_saved_fraction=0.0)])]
    row, = summarize(records)
    assert row["captures"] == 2
    assert row["captures_below_full_group_reads"] == 0
    assert row["mean_kv_read_ratio"] == 7.9


def test_frontier_job_pins_revision_and_exports_capture_without_secrets(monkeypatch):
    monkeypatch.setattr("benchmarks.run_lightning_adaptive_two_gate.subprocess.check_output",
                        lambda *a, **kw: "a" * 40)
    args = Namespace(experiment="group_frontier", model="test/model", revision="pinned-revision",
                     max_seq=8192, layers=[0, 16], teamspace_id="teamspace", cloud_account="cloud",
                     output_json=Path("frontier.json"))
    command, source = job_command(args)
    assert source["schema"] == "streamattn.adaptive_group_frontier.v1"
    assert "--revision pinned-revision" in command
    assert "--max-seq 8192" in command
    assert "--layers 0 16" in command
    assert "upload_capture(p," in command
    assert "LIGHTNING_AUTH_TOKEN" not in command
    assert "lightning-sdk==2026.9.18.post1" in command
    assert source["capture_remote_path"] == "uploads/streamattn/frontier.captures.pt"


def test_existing_physical_canary_command_is_still_default(monkeypatch):
    monkeypatch.setattr("benchmarks.run_lightning_adaptive_two_gate.subprocess.check_output",
                        lambda *a, **kw: "a" * 40)
    command, source = job_command()
    assert source["schema"] == "streamattn.adaptive_two_gate.v2"
    assert "profile_adaptive_two_gate.py --provider lightning" in command
    assert "TeamspaceApi" not in command


def test_capture_upload_delegates_to_current_sdk(tmp_path):
    source = tmp_path / "capture.pt"
    source.write_bytes(b"capture")
    def upload(**kwargs):
        assert kwargs == dict(file_path=str(source), teamspace_id="team", cloud_account="cloud",
                              remote_path="uploads/run.pt", progress_bar=False)
    upload_capture(source, api=Namespace(upload_file=upload), teamspace_id="team",
                   cloud_account="cloud", remote_path="uploads/run.pt")


def test_capture_upload_redacts_signed_url_in_transport_error(tmp_path):
    source = tmp_path / "capture.pt"
    source.write_bytes(b"capture")
    def fail(*args, **kwargs):
        raise ConnectionError("https://example.invalid/?token=test-secret")
    with pytest.raises(RuntimeError, match="upload failed: ConnectionError") as error:
        upload_capture(source, api=Namespace(upload_file=fail), teamspace_id="team",
                       cloud_account="cloud", remote_path="uploads/run.pt")
    assert "test-secret" not in str(error.value)
    assert error.value.__suppress_context__
