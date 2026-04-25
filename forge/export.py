from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .datasets import Example
from .rollout import RolloutResult


def pair_to_dpo(
    example: Example,
    a: RolloutResult,
    b: RolloutResult,
    instruction_prefix: str = "",
) -> dict[str, Any] | None:
    """Build a DPO training row from two rollouts of the same example.

    Schema mirrors `ale/rlaif/export.py` exactly: `prompt`, `chosen`,
    `rejected`, `metadata`. Emits None on ties so `ale rlaif train` doesn't
    receive degenerate pairs.

    Note: the prompt is the example context (same input both rollouts saw),
    NOT including any retrieved snippets — the policy being trained should
    learn to handle the bare prompt.
    """
    if a.reward == b.reward:
        return None
    chosen, rejected = (a, b) if a.reward > b.reward else (b, a)
    prompt = (instruction_prefix + example.context).strip()
    return {
        "prompt": prompt,
        "chosen": chosen.prediction,
        "rejected": rejected.prediction,
        "metadata": {
            "example_id": example.id,
            "chosen_label": chosen.config_label,
            "rejected_label": rejected.config_label,
            "reward_chosen": chosen.reward,
            "reward_rejected": rejected.reward,
            "scores_chosen": [
                {"name": s.name, "value": s.value, "detail": s.detail} for s in chosen.scores
            ],
            "scores_rejected": [
                {"name": s.name, "value": s.value, "detail": s.detail} for s in rejected.scores
            ],
            "retrieved_chosen": chosen.retrieved_ids,
            "retrieved_rejected": rejected.retrieved_ids,
            **{k: v for k, v in example.metadata.items() if k != "sources"},  # sources are large
        },
    }


def write_dpo_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
