"""Summarize the SM90 grouped-prefill execution-state experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ratios(payload: dict[str, Any], key: str) -> list[float]:
    return [float(cell["ratios"][key]) for cell in payload["cells"]]


def _resource(payload: dict[str, Any], key: str) -> dict[str, Any]:
    resources = payload["resources"]
    if key in resources:
        return dict(resources[key])
    return dict(resources["kernels"][key])


def _range(values: list[float]) -> str:
    return f"{min(values):.4f}x..{max(values):.4f}x"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", type=Path, required=True)
    parser.add_argument("--tma", type=Path, required=True)
    parser.add_argument("--dual-consumer", type=Path, required=True)
    parser.add_argument("--cluster-transport", type=Path, required=True)
    parser.add_argument("--cluster-epoch", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    serial = _read(args.serial)
    tma = _read(args.tma)
    dual = _read(args.dual_consumer)
    cluster_transport = _read(args.cluster_transport)
    cluster_epoch = _read(args.cluster_epoch)

    pv_rs = _ratios(serial, "pv_rs_vs_ss")
    epoch_rs = _ratios(serial, "epoch_rs_vs_ss")
    tma_vs_serial = _ratios(tma, "tma_rs_vs_serial_rs")
    dual_vs_serial = _ratios(dual, "tma_grouped2_vs_serial_grouped2")
    transport = _ratios(cluster_transport, "multicast_vs_independent")
    cluster_vs_tma = _ratios(cluster_epoch, "multicast_vs_independent_tma")
    cluster_vs_serial = _ratios(cluster_epoch, "multicast_vs_serial")

    serial_resource = _resource(serial, "epoch_rs")
    tma_resource = _resource(tma, "tma_rs")
    dual_resource = _resource(dual, "tma_grouped2")
    cluster_resource = _resource(cluster_epoch, "multicast")
    paired_correct = all(
        (
            max(float(value) for value in values) == 0.0
            if values
            else False
        )
        for values in (
            [
                cell["correctness"]["serial_vs_tma_max_abs"]
                for cell in tma["cells"]
            ],
            [
                cell["correctness"]["serial_vs_tma_max_abs"]
                for cell in dual["cells"]
            ],
            [
                cell["correctness"]["independent_vs_multicast_max_abs"]
                for cell in cluster_transport["cells"]
            ],
            [
                cell["correctness"]["all_methods_max_abs"]
                for cell in cluster_epoch["cells"]
            ],
        )
    )
    zero_local_except_dual = all(
        int(resource["local_bytes_per_thread"]) == 0
        for resource in (serial_resource, tma_resource, cluster_resource)
    )

    summary = {
        "schema": "streamattn.sm90_grouped_prefill_execution_state_summary.v1",
        "paired_correct": paired_correct,
        "serial_rs": {
            "pv_rs_vs_ss": {"min": min(pv_rs), "max": max(pv_rs)},
            "epoch_rs_vs_ss": {"min": min(epoch_rs), "max": max(epoch_rs)},
            "resources": serial_resource,
            "decision": "retain_rs_pv_dataflow",
        },
        "producer_topologies": {
            "one_producer_one_consumer": {
                "vs_serial": {"min": min(tma_vs_serial), "max": max(tma_vs_serial)},
                "resources": tma_resource,
                "decision": "reject",
            },
            "one_producer_two_consumers_same_cta": {
                "vs_serial": {"min": min(dual_vs_serial), "max": max(dual_vs_serial)},
                "resources": dual_resource,
                "decision": "reject_spills_and_slowdown",
            },
            "two_cta_multicast_transport": {
                "vs_independent_tma": {"min": min(transport), "max": max(transport)},
                "decision": "retain_transport_primitive",
            },
            "two_cta_multicast_attention_epoch": {
                "vs_independent_tma": {
                    "min": min(cluster_vs_tma),
                    "max": max(cluster_vs_tma),
                },
                "vs_serial": {
                    "min": min(cluster_vs_serial),
                    "max": max(cluster_vs_serial),
                },
                "resources": cluster_resource,
                "decision": "reject",
            },
        },
        "zero_local_bytes_except_rejected_dual_consumer": zero_local_except_dual,
        "recommendation": (
            "integrate RS-PV into the lean 128-thread consumer-owned cp.async "
            "prefill path; do not build another producer-heavy TMA topology"
        ),
    }

    print("SM90 grouped-prefill execution-state summary")
    print(f"paired correctness: {paired_correct}")
    print(f"PV RS/SS: {_range(pv_rs)}")
    print(f"serial RS epoch/SS: {_range(epoch_rs)}")
    print(f"1+1 TMA/serial: {_range(tma_vs_serial)}")
    print(f"same-CTA 1+2/serial: {_range(dual_vs_serial)}")
    print(f"cluster transport/independent: {_range(transport)}")
    print(f"cluster epoch/independent TMA: {_range(cluster_vs_tma)}")
    print(f"cluster epoch/serial: {_range(cluster_vs_serial)}")
    print("decision: retain RS-PV; reject producer-heavy attention topologies")
    print(f"next: {summary['recommendation']}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
