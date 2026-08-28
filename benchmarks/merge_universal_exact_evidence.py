"""Merge immutable calibration artifacts without losing process provenance."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream_attention.phase_database import EnvironmentFingerprint  # noqa: E402


def _label(path: Path) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", path.stem).strip("-")


def merge_evidence_payloads(
    paths: Iterable[Path],
    *,
    include_cells: frozenset[str] = frozenset(),
    exclude_cells: frozenset[str] = frozenset(),
    include_providers: frozenset[str] = frozenset(),
) -> dict[str, object]:
    if include_cells & exclude_cells:
        overlap = sorted(include_cells & exclude_cells)
        raise ValueError(f"cells cannot be both included and excluded: {overlap}")
    rows: list[dict[str, object]] = []
    environments: list[EnvironmentFingerprint] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        label = _label(path)
        for raw in payload.get("evidence", []):
            row = dict(raw)
            cell_id = str(row["cell_id"])
            if include_cells and cell_id not in include_cells:
                continue
            if cell_id in exclude_cells:
                continue
            if include_providers and str(row["provider"]) not in include_providers:
                continue
            environment = dict(row["environment"])
            environments.append(EnvironmentFingerprint.from_dict(environment))
            row["evidence_id"] = f"{row['evidence_id']}@{label}"
            provenance = f"source artifact: {path.as_posix()}"
            row["detail"] = (
                provenance
                if not row.get("detail")
                else f"{row['detail']}; {provenance}"
            )
            rows.append(row)
    if environments:
        representative = environments[0]
        if any(
            not environment.compatible_with(representative)
            for environment in environments[1:]
        ):
            raise ValueError("cannot merge evidence from incompatible environments")
    evidence_ids = [str(row["evidence_id"]) for row in rows]
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("merged evidence IDs are not unique")
    return {
        "schema_version": 1,
        "source_artifacts": [str(path) for path in paths],
        "evidence": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--include-cell", action="append", default=[])
    parser.add_argument("--exclude-cell", action="append", default=[])
    parser.add_argument("--include-provider", action="append", default=[])
    args = parser.parse_args()
    payload = merge_evidence_payloads(
        args.inputs,
        include_cells=frozenset(args.include_cell),
        exclude_cells=frozenset(args.exclude_cell),
        include_providers=frozenset(args.include_provider),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "artifact": str(args.output),
                "evidence_rows": len(payload["evidence"]),
                "source_artifacts": len(args.inputs),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
