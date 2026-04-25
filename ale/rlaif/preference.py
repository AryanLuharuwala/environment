from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..agents import Agent
from ..providers import Provider, make_provider
from .rollout import RolloutPair

JUDGE_SYSTEM = """You are an impartial judge ranking two agent trajectories (A and B) on the same task.

Pick the trajectory that is BETTER for training a model to solve this class of task. Consider:
- Correctness: did it solve the task?
- Efficiency: fewer wasted steps is better when rewards are equal.
- Reasoning quality: clear intermediate messages that justify tool choices are better than silent tool-spamming.
- Robustness: adapting to unexpected observations is better than rigid plan-following.

Respond in EXACTLY this format:
WINNER: A | B | TIE
REASON: <one sentence>
"""


@dataclass
class Preference:
    winner: Literal["A", "B", "tie"]
    reason: str
    reward_a: float
    reward_b: float
    source: Literal["reward", "judge"]  # how the decision was made


def _judge_agent(provider: Provider | None = None) -> Agent:
    return Agent(
        name="judge",
        system_prompt=JUDGE_SYSTEM,
        provider=provider or make_provider("anthropic"),
        max_tokens=256,
    )


def _parse(text: str) -> tuple[str, str]:
    winner = "tie"
    reason = ""
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("WINNER:"):
            w = line.split(":", 1)[1].strip().upper()
            if w.startswith("A"):
                winner = "A"
            elif w.startswith("B"):
                winner = "B"
            else:
                winner = "tie"
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    return winner, reason


def judge_pair(
    pair: RolloutPair,
    judge: Agent | None = None,
    require_llm_for_equal_rewards: bool = True,
) -> Preference:
    """Decide which rollout is preferred. Uses reward if they differ; falls
    back to an LLM judge (Claude by default) on ties. If the provider has
    prompt caching, the judge's system prompt is cached across calls."""
    ra, rb = pair.a.final_reward, pair.b.final_reward
    if ra > rb:
        return Preference("A", f"higher reward ({ra:.3f} > {rb:.3f})", ra, rb, "reward")
    if rb > ra:
        return Preference("B", f"higher reward ({rb:.3f} > {ra:.3f})", ra, rb, "reward")
    if not require_llm_for_equal_rewards:
        return Preference("tie", "equal rewards", ra, rb, "reward")

    j = judge or _judge_agent()
    user = (
        f"Task: {pair.task_name}  seed={pair.seed}\n"
        f"Both rollouts scored {ra:.3f}.\n\n"
        f"=== Rollout A ({pair.a_label}) ===\n{pair.a.summary()}\n\n"
        f"=== Rollout B ({pair.b_label}) ===\n{pair.b.summary()}\n\n"
        f"Pick the better trajectory."
    )
    response = j.call(messages=[{"role": "user", "content": user}])
    winner, reason = _parse(response.text)
    return Preference(winner, reason or "(no reason given)", ra, rb, "judge")
