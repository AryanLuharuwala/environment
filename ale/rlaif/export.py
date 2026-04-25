from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..harness import Trace
from .preference import Preference
from .rollout import RolloutPair


def trajectory_text(trace: Trace) -> str:
    """Render a trace as a plain-text transcript for DPO training. Each step
    shows the model's intent (if any), the tool call, and the observation."""
    lines: list[str] = []
    for e in trace.events:
        if e.kind == "message" and e.data.get("text"):
            lines.append(f"think: {e.data['text'].strip()}")
        elif e.kind == "tool_use":
            args = json.dumps(e.data.get("input", {}), separators=(",", ":"), sort_keys=True)
            lines.append(f"call: {e.data.get('name', '')}({args})")
        elif e.kind == "observation" and not e.data.get("initial"):
            text = str(e.data.get("text", ""))
            lines.append(f"obs:  {text}")
    return "\n".join(lines)


def prompt_text(trace: Trace, task_description: str) -> str:
    initial = next(
        (e.data.get("text", "") for e in trace.events if e.kind == "observation" and e.data.get("initial")),
        "",
    )
    plan = next((e.data.get("text", "") for e in trace.events if e.kind == "plan"), "")
    parts = [f"Task: {task_description}"]
    if plan:
        parts.append(f"Plan:\n{plan}")
    parts.append(f"Initial observation: {initial}")
    return "\n\n".join(parts)


def pair_to_dpo(
    pair: RolloutPair,
    preference: Preference,
    task_description: str,
) -> dict[str, Any] | None:
    """Serialize one preference as a DPO training example.

    Schema mirrors TRL's DPOTrainer expected format: `prompt`, `chosen`,
    `rejected`. Ties are skipped (return None). Also records `metadata` for
    inspection/filtering downstream.
    """
    if preference.winner == "tie":
        return None
    chosen_trace = pair.a if preference.winner == "A" else pair.b
    rejected_trace = pair.b if preference.winner == "A" else pair.a
    chosen_label = pair.a_label if preference.winner == "A" else pair.b_label
    rejected_label = pair.b_label if preference.winner == "A" else pair.a_label

    return {
        "prompt": prompt_text(chosen_trace, task_description),
        "chosen": trajectory_text(chosen_trace),
        "rejected": trajectory_text(rejected_trace),
        "metadata": {
            "task": pair.task_name,
            "seed": pair.seed,
            "chosen_label": chosen_label,
            "rejected_label": rejected_label,
            "reward_chosen": chosen_trace.final_reward,
            "reward_rejected": rejected_trace.final_reward,
            "preference_source": preference.source,
            "reason": preference.reason,
        },
    }


def write_dpo_jsonl(examples: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
