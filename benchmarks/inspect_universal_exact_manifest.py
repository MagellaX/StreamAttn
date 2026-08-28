"""Validate and summarize the Universal Exact Attention v1 manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main() -> None:
    from stream_attention.exact_compiler import (
        load_universal_exact_manifest,
        matching_kernel_families,
        registered_exact_kernel_families,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    manifest = load_universal_exact_manifest(args.manifest)
    families = registered_exact_kernel_families()
    family_match_counts = {family.family_id: 0 for family in families}
    cells_without_native: list[str] = []
    cell_rows: list[dict[str, object]] = []
    for cell in manifest.cells:
        matches = matching_kernel_families(cell, families)
        native_matches = tuple(family for family in matches if family.native)
        if not native_matches:
            cells_without_native.append(cell.cell_id)
        for family in matches:
            family_match_counts[family.family_id] += 1
        cell_rows.append(
            {
                "cell_id": cell.cell_id,
                "surface": cell.surface,
                "architecture": cell.architecture,
                "phase": cell.phase,
                "native_families": [family.family_id for family in native_matches],
                "fallback_families": [
                    family.family_id for family in matches if not family.native
                ],
            }
        )

    output = {
        **manifest.summary(),
        "acceptance": {
            "semantic_coverage": manifest.acceptance.semantic_coverage,
            "telemetry_coverage": manifest.acceptance.telemetry_coverage,
            "p90_routing_regret": manifest.acceptance.p90_routing_regret,
            "zero_timed_loop_allocations": (
                manifest.acceptance.zero_timed_loop_allocations
            ),
            "retain_negative_cells": manifest.acceptance.retain_negative_cells,
        },
        "kernel_family_count": len(families),
        "family_match_counts": family_match_counts,
        "cells_without_native_family": cells_without_native,
        "cells": cell_rows,
    }
    encoded = json.dumps(output, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
