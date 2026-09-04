from __future__ import annotations

from pathlib import Path

import pytest

from stream_attention.basis_evidence import (
    BasisCounter,
    BasisEnvironment,
    BasisEvidence,
    load_basis_suite,
    missing_required_counters,
    parse_ncu_csv,
)


ROOT = Path(__file__).resolve().parents[1]


def test_sm90_basis_suite_crosses_six_anchors_with_operation_grammar():
    suite = load_basis_suite(ROOT / "benchmarks/manifests/sm90_basis_v1.yaml")
    assert suite.architecture == "sm90"
    assert len(suite.anchors) == 6
    assert len(suite.operations) == 14
    assert len(suite.cases) == 84
    assert suite.case("m16_b1_n32k_g8_d128.qk").anchor.query_len == 16


def test_ncu_csv_parser_keeps_metric_units_and_values():
    text = '''==PROF== Connected
"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit","Metric Value"
"0","1","python","host","kernel","1","7","128","120","NVIDIA H100","9.0","LaunchStats","launch__registers_per_thread","register/thread","168"
"0","1","python","host","kernel","1","7","128","120","NVIDIA H100","9.0","MemoryWorkloadAnalysis","dram__bytes_read.sum","byte","1,048,576"
'''
    counters = {counter.metric: counter for counter in parse_ncu_csv(text)}
    assert counters["launch__registers_per_thread"].value == 168
    assert counters["dram__bytes_read.sum"].value == 1048576
    assert counters["dram__bytes_read.sum"].unit == "byte"


def test_missing_counter_contract_is_explicit():
    counters = (BasisCounter("a", 1.0, "count"),)
    assert missing_required_counters(("a", "b"), counters) == ("b",)


def test_basis_evidence_records_environment_and_raw_artifact_identity():
    suite = load_basis_suite(ROOT / "benchmarks/manifests/sm90_basis_v1.yaml")
    case = suite.case("m4_b1_n32k_g8_d128.qk")
    environment = BasisEnvironment(
        source_commit="a" * 40,
        provider="test",
        device_name="NVIDIA H100 80GB HBM3",
        compute_capability="9.0",
        gpu_count=1,
        python_version="3.11",
        torch_version="2.7.1",
        cuda_version="12.8",
        driver_version="570.00",
        ncu_version="2025.1",
    )
    evidence = BasisEvidence(
        case=case,
        environment=environment,
        adapter_id="sm90_qk_floor",
        adapter_revision="v1",
        timing_mode="graph_device_floor",
        samples_ms=(0.01, 0.011, 0.009),
        correctness_passed=True,
        resources=(("registers_per_thread", 168.0),),
        counters=(BasisCounter("launch__registers_per_thread", 168.0, "register/thread"),),
        missing_counters=(),
        raw_artifact_sha256="b" * 64,
    )
    assert evidence.median_ms == 0.01
    assert evidence.as_dict()["case_sha256"] == case.fingerprint
    assert len(environment.fingerprint) == 64


def test_suite_rejects_unknown_case():
    suite = load_basis_suite(ROOT / "benchmarks/manifests/sm90_basis_v1.yaml")
    with pytest.raises(KeyError, match="unknown basis case"):
        suite.case("does-not-exist")
