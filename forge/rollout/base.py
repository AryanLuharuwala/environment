from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..verifiers import Score


@dataclass
class RolloutResult:
    example_id: str
    config_label: str
    prediction: str
    scores: list[Score]
    reward: float
    retrieved_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def extract_python_code(text: str) -> str:
    """Pull the first ```python ... ``` block from `text`. Falls back to the
    raw text if no fenced block exists."""
    lines = text.splitlines()
    in_block = False
    block: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not in_block and stripped.startswith("```"):
            in_block = True
            continue
        if in_block and stripped.startswith("```"):
            return "\n".join(block)
        if in_block:
            block.append(line)
    return text
