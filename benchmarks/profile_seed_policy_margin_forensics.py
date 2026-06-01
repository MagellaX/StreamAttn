"""Convenience wrapper for margin-aware route-bundle decode forensics.

This intentionally delegates to ``profile_seed_only_route_bundle_decode.py`` so
the safety gate, bucket routing, and product fast-path presets stay in one
place.  Any arguments after this wrapper's own options are passed through.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k-report", type=int, default=10)
    parser.add_argument("--max-rows", type=int, default=24)
    parser.add_argument(
        "route_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to profile_seed_only_route_bundle_decode.py.",
    )
    args = parser.parse_args()

    route_args = list(args.route_args)
    if route_args and route_args[0] == "--":
        route_args = route_args[1:]
    cmd = [
        sys.executable,
        str(REPO_ROOT / "benchmarks" / "profile_seed_only_route_bundle_decode.py"),
        "--margin-forensics",
        "--margin-forensics-top-k",
        str(args.top_k_report),
        "--margin-forensics-max-rows",
        str(args.max_rows),
        *route_args,
    ]
    raise SystemExit(subprocess.call(cmd, cwd=str(REPO_ROOT)))


if __name__ == "__main__":
    main()
