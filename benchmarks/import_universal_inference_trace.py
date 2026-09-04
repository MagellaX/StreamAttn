"""Validate raw serving traces and freeze their calibration/holdout split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from stream_attention.inference_trace import (
        freeze_trace_records,
        read_raw_trace_jsonl,
        read_trace_jsonl,
        summarize_trace,
        write_trace_jsonl,
    )
    from stream_attention.inference_workload import load_universal_inference_manifest

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="raw JSONL records")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--summary-json", type=Path)
    args = parser.parse_args()

    manifest = load_universal_inference_manifest(args.manifest)
    prior = read_trace_jsonl(args.output) if args.output.is_file() else ()
    imported = freeze_trace_records(
        read_raw_trace_jsonl(args.input),
        existing=prior,
        calibration_fraction=manifest.calibration_fraction,
        salt=manifest.split_salt,
    )
    by_id = {record.record_id: record for record in prior}
    by_id.update({record.record_id: record for record in imported})
    records = tuple(by_id[key] for key in sorted(by_id))
    write_trace_jsonl(args.output, records)
    summary = {
        "manifest_id": manifest.manifest_id,
        "output": str(args.output),
        **summarize_trace(records),
    }
    encoded = json.dumps(summary, indent=2, sort_keys=True)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
