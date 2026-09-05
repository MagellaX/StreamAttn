"""Recover JSON artifacts from Lightning's 16 KiB log-line chunking."""

from __future__ import annotations

import json


def result_from_logs(logs: str, *, schema: str) -> dict | None:
    # The transport inserts newlines even inside JSON strings and numbers.
    # Only join exact-sized transport chunks; ordinary log lines stay separate.
    lines = []
    pending = ""
    for line in logs.splitlines():
        pending += line
        if len(line.encode("utf-8")) != 16384:
            lines.append(pending)
            pending = ""
    if pending:
        lines.append(pending)
    decoder = json.JSONDecoder()
    result = None
    for line in lines:
        for start, char in enumerate(line):
            if char != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(line[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("schema") == schema:
                result = payload
                break
    return result
