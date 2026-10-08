import importlib.metadata
import json
import sys
import types

import pytest

from benchmarks import run_lightning_adaptive_two_gate as runner
from benchmarks.run_lightning_sm90_grouped_rs_prefill_canary import (
    COMPLETED_STATES,
    TERMINAL_STATES,
)


@pytest.mark.parametrize("state", ["complete", "completed"])
def test_success_aliases_are_terminal(state):
    assert state in COMPLETED_STATES
    assert state in TERMINAL_STATES


@pytest.mark.parametrize("state", ["creating", "pending", "running", "stopping"])
def test_active_states_are_not_terminal(state):
    assert state not in TERMINAL_STATES


@pytest.mark.parametrize("state", ["complete", "completed", "failed", "stopped"])
def test_adaptive_runner_finishes_and_cleans_up(monkeypatch, tmp_path, state):
    output = tmp_path / "result.json"
    calls = []
    job = types.SimpleNamespace(
        id="test-job", state=state, started_at=None, total_cost=0, message="",
    )

    class FakeApi:
        def submit_job(self, **kwargs):
            calls.append("submit")
            assert kwargs["max_run_attempts"] == 1
            assert kwargs["max_runtime"] == 60
            return job

        def get_job(self, **kwargs):
            return job

        def get_logs_finished(self, **kwargs):
            return json.dumps({"schema": "test.v1", "complete": True})

        def stop_job(self, **kwargs):
            pytest.fail("A terminal job must not be stopped again")

        def delete_job(self, **kwargs):
            calls.append("delete")

    sdk = types.ModuleType("lightning_sdk")
    sdk_api = types.ModuleType("lightning_sdk.api")
    sdk_jobs = types.ModuleType("lightning_sdk.api.job_api")
    sdk_jobs.JobApiV2 = FakeApi
    monkeypatch.setitem(sys.modules, "lightning_sdk", sdk)
    monkeypatch.setitem(sys.modules, "lightning_sdk.api", sdk_api)
    monkeypatch.setitem(sys.modules, "lightning_sdk.api.job_api", sdk_jobs)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "2026.9.18.post1")
    monkeypatch.setattr(runner, "job_command", lambda args: (
        "test-command", {"schema": "test.v1"},
    ))
    monkeypatch.setattr(runner.time, "sleep", lambda delay: pytest.fail(
        "Polling must stop as soon as a terminal state is returned",
    ))
    monkeypatch.setattr(sys, "argv", [
        "runner", "--output-json", str(output), "--max-runtime", "60",
    ])

    if state in COMPLETED_STATES:
        runner.main()
    else:
        with pytest.raises(RuntimeError, match="Adaptive canary failed"):
            runner.main()

    assert calls == ["submit", "delete"]
    assert json.loads(output.read_text())["execution"]["platform_state"] == state
