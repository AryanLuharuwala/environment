from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TraceEvent:
    kind: str  # "plan", "tool_use", "observation", "message", "reward", "critique", "error"
    data: dict[str, Any]
    t: float = field(default_factory=time.time)


@dataclass
class Trace:
    task: str
    seed: int
    mode: str  # "single" or "multi"
    model: str
    events: list[TraceEvent] = field(default_factory=list)
    final_reward: float = 0.0
    steps: int = 0

    def log(self, kind: str, **data: Any) -> None:
        self.events.append(TraceEvent(kind=kind, data=data))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "seed": self.seed,
            "mode": self.mode,
            "model": self.model,
            "final_reward": self.final_reward,
            "steps": self.steps,
            "events": [asdict(e) for e in self.events],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))

    def summary(self, limit: int = 40) -> str:
        """A compact rendering of the trace, suitable for feeding to the Critic."""
        lines = []
        for e in self.events[:limit]:
            if e.kind == "tool_use":
                lines.append(f"-> {e.data.get('name')}({e.data.get('input')})")
            elif e.kind == "observation":
                text = str(e.data.get("text", ""))
                if len(text) > 120:
                    text = text[:117] + "..."
                lines.append(f"<- {text}")
            elif e.kind == "message":
                text = str(e.data.get("text", ""))
                if text:
                    lines.append(f"   [say] {text[:120]}")
            elif e.kind == "plan":
                lines.append(f"[plan]\n{e.data.get('text', '')}")
            elif e.kind == "critique":
                lines.append(f"[critique]\n{e.data.get('text', '')}")
        if len(self.events) > limit:
            lines.append(f"... ({len(self.events) - limit} more events)")
        return "\n".join(lines)
