from __future__ import annotations

from .base import Agent

PLANNER_SYSTEM = """You are the Planner in a multi-agent team. Your teammate, the Executor, will act on a task by calling tools.

Given a task description and the available tools, write a SHORT plan (3-5 numbered steps) that the Executor should follow. The plan should:
- Identify the goal and the key sub-decisions
- Suggest which tool to call first and what to attend to in the response
- Avoid step-by-step micromanagement; the Executor will adapt

Respond with the plan only. No preamble.
"""


def planner(model: str | None = None) -> Agent:
    kwargs = {"name": "planner", "system_prompt": PLANNER_SYSTEM, "max_tokens": 1024}
    if model:
        kwargs["model"] = model
    return Agent(**kwargs)


def make_plan(p: Agent, task_name: str, task_description: str, tool_names: list[str]) -> str:
    user = (
        f"Task: {task_name}\n\n"
        f"Description: {task_description}\n\n"
        f"Available tools: {', '.join(tool_names)}\n\n"
        f"Write the plan."
    )
    response = p.call(messages=[{"role": "user", "content": user}])
    return p.text_of(response).strip()
