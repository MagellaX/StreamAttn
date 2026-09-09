"""Static code evidence for the loaded producer, not executed instruction counts."""

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess


def producer_sass(text):
    functions = {}
    for block in text.split("Function : ")[1:]:
        symbol = block.splitlines()[0].strip()
        if "streamattn_natural_wgmma_micro_prefill_partial_kernel" not in symbol:
            continue
        ops = Counter(re.findall(
            r"/\*[0-9a-fA-F]+\*/\s+(?:@!?[A-Z0-9]+\s+)?([A-Z][A-Za-z0-9_.]*)\b", block))
        if not ops:
            raise ValueError("producer SASS has no decoded instructions")
        functions[symbol] = dict(instruction_sites=sum(ops.values()), opcodes=dict(sorted(ops.items())),
            sass_sha256=hashlib.sha256(block.encode()).hexdigest(), sass=block)
    if len(functions) != 2:
        raise ValueError(f"expected HND and NHD producers, found {len(functions)}")
    return functions


def capture_producer_sass(extension, cache):
    path = Path(extension.__file__)
    key = str(path)
    if key not in cache:
        result = subprocess.run(["cuobjdump", "--dump-sass", str(path)],
                                capture_output=True, text=True, check=True, timeout=120)
        cache[key] = dict(binary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            functions=producer_sass(result.stdout),
            contract="static sites only; not executed work, stall cycles, or latency")
    return cache[key]


def normalize_capture(payload):
    """Deduplicate saved disassembly and reparse it without touching GPU measurements."""
    result = copy.deepcopy(payload)
    binaries = {b["binary_sha256"]: b for b in result.get("static_producer_sass", [])}
    for row in result["rows"]:
        for variant, evidence in row.get("static_producer_sass", {}).items():
            if isinstance(evidence, dict):
                sha = evidence["binary_sha256"]
                if sha in binaries and binaries[sha] != evidence:
                    raise ValueError("conflicting disassembly for the same binary")
                binaries[sha] = evidence
                row["static_producer_sass"][variant] = sha
    for binary in binaries.values():
        raw = "".join("Function : " + f["sass"] for f in binary["functions"].values())
        binary["functions"] = producer_sass(raw)
    for row in result["rows"]:
        if any(sha not in binaries for sha in row.get("static_producer_sass", {}).values()):
            raise ValueError("missing referenced binary disassembly")
    result["static_producer_sass"] = list(binaries.values())
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError("preserve existing evidence")
    raw = args.input.read_bytes()
    result = normalize_capture(json.loads(raw))
    result["static_sass_normalization"] = dict(input_sha256=hashlib.sha256(raw).hexdigest(),
        source=str(args.input), changes="deduplicate full disassembly; preserve lowercase x in matrix opcodes; no GPU measurement changes")
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
