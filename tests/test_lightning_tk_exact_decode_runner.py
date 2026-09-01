import json

from benchmarks.run_lightning_tk_tensor_core_exact_decode import _results_from_logs


def test_results_from_logs_accepts_both_phased_gate_matrix_schemas():
    v1 = {"schema": "streamattn.sm80_d128_phased_kv_gate.matrix.v1", "cells": []}
    v2 = {"schema": "streamattn.sm80_d128_phased_kv_gate.matrix.v2", "cells": []}
    logs = f"setup output\n{json.dumps(v1, indent=2)}\nmore output\n{json.dumps(v2, indent=2)}\n"

    assert _results_from_logs(logs) == [v1, v2]
