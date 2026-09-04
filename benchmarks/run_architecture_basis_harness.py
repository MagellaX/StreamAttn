"""Run one architecture basis adapter and emit canonical immutable evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ADAPTER_SCHEMA = "streamattn.basis_adapter.v1"


def _source_commit(explicit: str | None) -> str:
    if explicit:
        return explicit
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _ncu_version(binary: str) -> str:
    result = subprocess.run(
        [binary, "--version"], check=True, capture_output=True, text=True
    )
    return result.stdout.strip().splitlines()[-1]


def _materialize_command(
    command: list[str], *, case_json: Path, result_json: Path
) -> list[str]:
    return [
        value.replace("{case_json}", str(case_json)).replace(
            "{result_json}", str(result_json)
        )
        for value in command
    ]


def main() -> None:
    from stream_attention.basis_evidence import (
        BasisCounter,
        BasisEvidence,
        capture_local_basis_environment,
        load_basis_suite,
        missing_required_counters,
        parse_ncu_csv,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case", required=True, dest="case_id")
    parser.add_argument("--adapter-id", required=True)
    parser.add_argument("--adapter-revision", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--ncu", action="store_true")
    parser.add_argument("--ncu-bin", default="ncu")
    parser.add_argument("--kernel-name")
    parser.add_argument("--allow-missing-counters", action="store_true")
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="adapter command; use {case_json} and {result_json} placeholders",
    )
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("an adapter command is required after --")

    suite = load_basis_suite(args.manifest)
    case = suite.case(args.case_id)
    ncu_binary = shutil.which(args.ncu_bin) if args.ncu else None
    if args.ncu and ncu_binary is None:
        raise RuntimeError(f"could not find Nsight Compute binary {args.ncu_bin!r}")

    with tempfile.TemporaryDirectory(prefix="streamattn-basis-") as temporary:
        workdir = Path(temporary)
        case_json = workdir / "case.json"
        raw_json = workdir / "adapter-result.json"
        ncu_csv = workdir / "ncu.csv"
        case_json.write_text(
            json.dumps(case.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        adapter_command = _materialize_command(
            command, case_json=case_json, result_json=raw_json
        )
        subprocess.run(adapter_command, cwd=ROOT, check=True)
        raw_bytes = raw_json.read_bytes()
        raw = json.loads(raw_bytes)
        if raw.get("schema") != ADAPTER_SCHEMA:
            raise ValueError("basis adapter returned an unsupported schema")
        if raw.get("case_sha256") != case.fingerprint:
            raise ValueError("basis adapter result does not match the requested case")

        counters: tuple[BasisCounter, ...] = ()
        ncu_version = None
        if ncu_binary is not None:
            ncu_version = _ncu_version(ncu_binary)
            ncu_command = [
                ncu_binary,
                "--csv",
                "--page",
                "raw",
                "--target-processes",
                "all",
                "--metrics",
                ",".join(suite.required_ncu_metrics),
                "--log-file",
                str(ncu_csv),
            ]
            if args.kernel_name:
                ncu_command.extend(["--kernel-name", args.kernel_name])
            ncu_command.extend(adapter_command)
            subprocess.run(ncu_command, cwd=ROOT, check=True)
            counters = parse_ncu_csv(ncu_csv.read_text(encoding="utf-8"))

        missing = missing_required_counters(suite.required_ncu_metrics, counters)
        if args.ncu and missing and not args.allow_missing_counters:
            raise RuntimeError(
                "Nsight Compute did not return required metrics: " + ", ".join(missing)
            )
        environment = capture_local_basis_environment(
            source_commit=_source_commit(args.source_commit),
            provider=args.provider,
            ncu_version=ncu_version,
        )
        evidence = BasisEvidence(
            case=case,
            environment=environment,
            adapter_id=args.adapter_id,
            adapter_revision=args.adapter_revision,
            timing_mode=str(raw["timing_mode"]),
            samples_ms=tuple(float(value) for value in raw["samples_ms"]),
            correctness_passed=bool(raw["correctness_passed"]),
            resources=tuple(
                sorted((str(key), float(value)) for key, value in raw["resources"].items())
            ),
            counters=counters,
            missing_counters=missing,
            raw_artifact_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(evidence.as_dict(), indent=2, sort_keys=True)
    args.output_json.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
