from __future__ import annotations

from pathlib import Path

from src.utils.io import ensure_dir


def write_tex(path: str | Path, content: str) -> None:
    ensure_dir(Path(path).parent)
    Path(path).write_text(content, encoding="utf-8")
