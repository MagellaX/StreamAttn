import json

import torch

from benchmarks.profile_real_llm_gate1_heads import (
    _error_summary,
    _load_prompts,
    _parse_int_list,
)


class _Args:
    prompt = None
    prompt_file = None
    max_prompts = 8


def test_parse_int_list_accepts_empty_and_csv():
    assert _parse_int_list(None) is None
    assert _parse_int_list("") is None
    assert _parse_int_list("0, 2,5") == {0, 2, 5}


def test_load_prompts_from_jsonl(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        json.dumps({"text": "alpha"}) + "\n" + json.dumps({"prompt": "beta"}) + "\n",
        encoding="utf-8",
    )
    args = _Args()
    args.prompt_file = str(path)
    args.max_prompts = 1

    assert _load_prompts(args) == ["alpha"]


def test_error_summary_reports_zero_for_identical_tensors():
    x = torch.randn(2, 3, 4)
    summary = _error_summary(x, x.clone())

    assert summary["max_abs_error"] == 0.0
    assert summary["relative_l2_error"] == 0.0
