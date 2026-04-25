from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .base import Document


CODE_SUFFIXES = {".py", ".js", ".ts", ".go", ".rs", ".java", ".cpp", ".c", ".rb"}


def from_directory(
    root: Path,
    suffixes: Iterable[str] | None = None,
    max_bytes_per_file: int = 200_000,
) -> list[Document]:
    """Walk a directory and ingest every code file as a Document. Skips files
    larger than `max_bytes_per_file` (typically generated code or data dumps
    that pollute the index)."""
    root = Path(root).resolve()
    suffixes = set(suffixes) if suffixes is not None else CODE_SUFFIXES
    docs: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in suffixes:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > max_bytes_per_file:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = path.relative_to(root).as_posix()
        docs.append(Document(id=rel, text=text, metadata={"path": rel, "bytes": size}))
    return docs


def from_jsonl(path: Path) -> list[Document]:
    """Load a corpus from JSONL. Each line must contain `id` and `text`;
    everything else lands in `metadata`."""
    docs: list[Document] = []
    with Path(path).open() as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "id" not in row or "text" not in row:
                raise ValueError(f"{path}:{n} missing id or text")
            meta = {k: v for k, v in row.items() if k not in {"id", "text"}}
            docs.append(Document(id=str(row["id"]), text=row["text"], metadata=meta))
    return docs
