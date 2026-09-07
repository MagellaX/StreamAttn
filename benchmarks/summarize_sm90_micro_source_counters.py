"""Deduplicate SASS PCs before attributing sampled stalls to CUDA lines.

Nsight repeats PCs under inlined source files. CUDA aggregate rows and SASS
rows must not be summed together. Samples are evidence of stalls, not time.
"""

import argparse
import csv
import io
import json
from pathlib import Path


def source_pcs(text):
    header, filename, function, source, line = None, None, None, None, None
    pcs = {}
    for fields in csv.reader(io.StringIO(text)):
        if not fields:
            continue
        if fields[0] == "File Path":
            filename, source, line = fields[1], None, None
        elif fields[0] == "Function Name":
            function = fields[1]
        elif fields[0] == "Line No":
            header = fields
        elif header and len(fields) == len(header):
            if fields[0].isdigit():
                line, source = int(fields[0]), fields[1]
            address = fields[2]
            if not address.startswith("0x"):
                continue
            metrics = {name: int(value.replace(",", "")) for name, value in zip(header[4:], fields[4:])
                       if (name.startswith("stall_") or name == "# Samples")
                       and value.replace(",", "").isdigit()}
            key = (function, address)
            correlation = dict(file=filename, line=line, source=source)
            if key in pcs:
                if pcs[key]["metrics"] != metrics:
                    raise ValueError("conflicting duplicate PC metrics")
                if correlation not in pcs[key]["correlations"]:
                    pcs[key]["correlations"].append(correlation)
            else:
                pcs[key] = dict(function=function, address=address, instruction=fields[3].strip(),
                                metrics=metrics, correlations=[correlation])
    if not pcs:
        raise ValueError("no source-correlated SASS PCs")
    return list(pcs.values())


def summarize(payload):
    if payload.get("schema") != "streamattn.sm90_micro_prefill_counters.v1" or not payload.get("source_correlated"):
        raise ValueError("not a source-counter capture")
    rows = []
    for row in payload["rows"]:
        pcs = source_pcs(row["source_csv"])
        totals = {metric: sum(pc["metrics"].get(metric, 0) for pc in pcs)
                  for metric in ("# Samples", "stall_long_sb", "stall_long_sb (Not Issued)",
                                 "stall_wait", "stall_short_sb", "stall_barrier", "stall_membar")}
        raw = list(csv.DictReader(line for line in row["profiler_csv"].splitlines()
                                  if line.startswith('"')))
        aggregate = raw[-1]
        for source_name, raw_name in (
            ("# Samples", "smsp__pcsamp_sample_count"),
            ("stall_long_sb", "smsp__pcsamp_warps_issue_stalled_long_scoreboard"),
        ):
            if totals[source_name] != int(aggregate[raw_name].replace(",", "")):
                raise ValueError("source PC totals disagree with kernel aggregate")
        top = sorted(pcs, key=lambda p: p["metrics"].get("stall_long_sb", 0), reverse=True)[:12]
        rows.append(dict(batch=row["batch"], n=row["n"], unique_pcs=len(pcs),
                         totals=totals, aggregate_totals_match=True, top_long_scoreboard_pcs=top))
    return dict(schema="streamattn.sm90_micro_source_summary.v1", complete=payload["complete"],
                rows=rows, contract="distinct exported PCs; samples, not elapsed-time fractions")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--output-json", type=Path)
    args = p.parse_args()
    result = summarize(json.loads(args.input.read_text(encoding="utf-8")))
    for row in result["rows"]:
        print(f"B{row['batch']} N{row['n']}: {row['totals']}")
        for pc in row["top_long_scoreboard_pcs"][:5]:
            print(pc["metrics"]["stall_long_sb"], pc["instruction"], pc["correlations"])
    if args.output_json:
        if args.output_json.exists():
            raise FileExistsError("preserve existing evidence")
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
