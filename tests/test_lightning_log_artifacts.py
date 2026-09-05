import json

from benchmarks.lightning_log_artifacts import result_from_logs


def test_transport_chunking_inside_json_string_and_number():
    payload = {"schema": "test.v1", "value": "x" * 40000, "number": 123456.789}
    line = json.dumps(payload)
    logs = "build log\n" + "\n".join(line[i:i + 16384] for i in range(0, len(line), 16384))
    assert result_from_logs(logs, schema="test.v1") == payload


def test_last_matching_schema_without_joining_ordinary_lines():
    logs = 'noise\n{"schema":"test.v1","complete":false}\n{"schema":"other"}\n'
    logs += '{"schema":"test.v1","complete":true}\n'
    assert result_from_logs(logs, schema="test.v1") == {"schema": "test.v1", "complete": True}
    assert result_from_logs(logs, schema="absent") is None


def test_incomplete_artifact_is_not_repaired_by_guessing():
    assert result_from_logs('{"schema":"test.v1",\n"complete":true}', schema="test.v1") is None
