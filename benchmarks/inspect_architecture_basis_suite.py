"""Validate and summarize an architecture basis-operation suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from stream_attention.basis_evidence import load_basis_suite

    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    suite = load_basis_suite(args.manifest)
    output = {
        "suite_id": suite.suite_id,
        "architecture": suite.architecture,
        "anchor_count": len(suite.anchors),
        "operation_count": len(suite.operations),
        "case_count": len(suite.cases),
        "anchors": [anchor.anchor_id for anchor in suite.anchors],
        "operations": [operation.value for operation in suite.operations],
        "ncu_metric_count": len(suite.required_ncu_metrics),
        "graph_replay": suite.graph_replay,
        "timing": {
            "warmup": suite.warmup,
            "iterations": suite.iterations,
            "repeats": suite.repeats,
        },
    }
    encoded = json.dumps(output, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
