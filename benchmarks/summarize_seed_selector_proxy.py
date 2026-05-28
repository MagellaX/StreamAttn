"""Summarize seed selector proxy diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.summarize_seed_policy_attention_coverage import summarize_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json", help="Seed selector proxy JSON output.")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    summary = summarize_payload(Path(args.json))
    summary["schema"] = "streamattn.seed_selector_proxy_summary.v1"
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
