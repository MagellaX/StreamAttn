"""Compile architecture phase databases from strict exact-attention evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _source_commit(explicit: str | None) -> str:
    if explicit:
        return explicit
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    from stream_attention.exact_compiler import load_universal_exact_manifest
    from stream_attention.phase_database import (
        compile_phase_database,
        load_backend_evidence,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", nargs="+", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--selection-json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("phase_db"))
    parser.add_argument("--source-commit")
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=("sm80", "sm90", "sm100"),
        default=("sm80", "sm90", "sm100"),
        help="compile only the listed architecture databases",
    )
    args = parser.parse_args()

    manifest = load_universal_exact_manifest(args.manifest)
    evidence = tuple(
        row for artifact in args.evidence for row in load_backend_evidence(artifact)
    )
    selected = {}
    if args.selection_json is not None:
        selected = {
            str(key): str(value)
            for key, value in json.loads(
                args.selection_json.read_text(encoding="utf-8")
            ).items()
        }
    commit = _source_commit(args.source_commit)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    databases = []
    for architecture in args.architectures:
        database = compile_phase_database(
            manifest,
            evidence,
            architecture=architecture,
            source_commit=commit,
            selected_evidence_ids=selected,
        )
        path = args.output_dir / f"{architecture}.json"
        database.write_json(path)
        databases.append(
            {
                "architecture": architecture,
                "path": path.name,
                "sha256": _sha256(path),
                "acceptance": dict(database.acceptance),
                "entry_count": len(database.entries),
                "evidence_count": len(database.evidence),
            }
        )

    index = {
        "schema_version": 1,
        "manifest_id": manifest.manifest_id,
        "source_commit": commit,
        "databases": databases,
    }
    index_path = args.output_dir / "index.json"
    index_path.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(index, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
