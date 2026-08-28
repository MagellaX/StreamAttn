import hashlib
import json

from benchmarks.compile_universal_exact_phase_db import (
    _preserved_database_entries,
)


def test_partial_compile_preserves_other_architecture_index_entries(tmp_path):
    sm90 = tmp_path / "sm90.json"
    sm90.write_text(
        json.dumps({"source_commit": "sm90-commit"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "index.json").write_text(
        json.dumps(
            {
                "databases": [
                    {
                        "architecture": "sm90",
                        "path": "sm90.json",
                        "sha256": "stale",
                    },
                    {
                        "architecture": "sm80",
                        "path": "sm80.json",
                        "sha256": "replaced",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    preserved = _preserved_database_entries(tmp_path, {"sm80"})

    assert set(preserved) == {"sm90"}
    assert preserved["sm90"]["source_commit"] == "sm90-commit"
    assert preserved["sm90"]["sha256"] == hashlib.sha256(
        sm90.read_bytes()
    ).hexdigest()


def test_partial_compile_recovers_unindexed_architecture_database(tmp_path):
    database = {
        "architecture": "sm100",
        "source_commit": "sm100-commit",
        "acceptance": {"compiler_v1_pass": False},
        "entries": [{"cell_id": "one"}],
        "evidence": [{"evidence_id": "one"}, {"evidence_id": "two"}],
    }
    (tmp_path / "sm100.json").write_text(
        json.dumps(database) + "\n", encoding="utf-8"
    )

    preserved = _preserved_database_entries(tmp_path, {"sm80"})

    assert preserved["sm100"]["source_commit"] == "sm100-commit"
    assert preserved["sm100"]["entry_count"] == 1
    assert preserved["sm100"]["evidence_count"] == 2
