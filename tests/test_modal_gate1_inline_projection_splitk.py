from pathlib import Path

from benchmarks.modal_gate1_inline_projection_splitk import _read_prompt_file


def test_read_prompt_file_collapses_lines_for_repeated_capture(tmp_path: Path):
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("first line\n\nsecond line\n", encoding="utf-8")

    assert _read_prompt_file(str(prompt_path)) == "first line second line"
