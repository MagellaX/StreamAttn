"""Inspect the v2 workload contract and direct exact-baseline coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from stream_attention.baseline_resolver import load_exact_baseline_descriptors
    from stream_attention.inference_trace import read_trace_jsonl, summarize_trace
    from stream_attention.inference_workload import load_universal_inference_manifest

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--baseline-manifest", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    manifest = load_universal_inference_manifest(args.manifest)
    baselines = load_exact_baseline_descriptors(args.baseline_manifest)
    output: dict[str, object] = {
        "manifest_id": manifest.manifest_id,
        "schema_version": manifest.schema_version,
        "architectures": sorted(manifest.architectures),
        "sources": [source.source_id for source in manifest.sources],
        "query_regimes": [regime.name for regime in manifest.regimes],
        "calibration_fraction": manifest.calibration_fraction,
        "holdout_p90_routing_regret": manifest.acceptance.holdout_p90_routing_regret,
        "baseline_ids": [baseline.baseline_id for baseline in baselines],
    }
    if args.trace is not None:
        output["trace"] = summarize_trace(read_trace_jsonl(args.trace))
    encoded = json.dumps(output, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
