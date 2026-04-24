from __future__ import annotations

from .base import Agent

CRITIC_SYSTEM = """You are the Critic in a multi-agent team. After a task run, you assess the Executor's trajectory.

Given the task description, the plan, the trace of tool calls and observations, and the final reward, return a brief critique:
- One sentence on what went well.
- One sentence on what to improve next time.
- A confidence score from 0.0 to 1.0 that the Executor would do better on a re-run with this critique applied.

Format strictly as:
WELL: <sentence>
IMPROVE: <sentence>
CONFIDENCE: <0.0-1.0>
"""


def critic(model: str | None = None) -> Agent:
    kwargs = {"name": "critic", "system_prompt": CRITIC_SYSTEM, "max_tokens": 512}
    if model:
        kwargs["model"] = model
    return Agent(**kwargs)


def review(c: Agent, task_name: str, task_description: str, plan: str, trace_summary: str, reward: float) -> str:
    user = (
        f"Task: {task_name}\n"
        f"Description: {task_description}\n\n"
        f"Plan:\n{plan or '(none)'}\n\n"
        f"Trace:\n{trace_summary}\n\n"
        f"Final reward: {reward:.3f}\n\n"
        f"Write your critique."
    )
    response = c.call(messages=[{"role": "user", "content": user}])
    return c.text_of(response).strip()
