"""Dependency-free helpers shared by benchmark launchers and CPU tests."""

from __future__ import annotations

from pathlib import Path


def read_prompt_file(path: str) -> str:
    """Read a text fixture as one prompt while preserving word boundaries."""
    return " ".join(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
