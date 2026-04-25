from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .base import Example


@dataclass
class JsonlSource:
    """Generic Source over a JSONL file. Each line must contain at least
    `id`, `context`, `target`. Used for law/compliance datasets, custom
    code corpora, or anything that doesn't naturally come from git.

    Optional fields are forwarded into `Example.metadata`."""

    path: Path

    def iter_examples(self, **_filters: Any) -> Iterator[Example]:
        path = Path(self.path)
        with path.open() as f:
            for n, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                missing = {"id", "context", "target"} - row.keys()
                if missing:
                    raise ValueError(f"{path}:{n} missing fields: {missing}")
                meta = {k: v for k, v in row.items() if k not in {"id", "context", "target"}}
                yield Example(id=row["id"], context=row["context"], target=row["target"], metadata=meta)
