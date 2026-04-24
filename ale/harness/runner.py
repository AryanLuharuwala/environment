from __future__ import annotations

from typing import Any

from ..agents import Agent, critic, executor, make_plan, planner, review
from ..tasks import Task
from .trace import Trace


def _user_msg(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def _assistant_blocks(blocks: list[Any]) -> dict[str, Any]:
    """Echo Claude's response content back so tool_use IDs line up with tool_results."""
    content = []
    for b in blocks:
        t = getattr(b, "type", "")
        if t == "text":
            if b.text:
                content.append({"type": "text", "text": b.text})
        elif t == "tool_use":
            content.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
    return {"role": "assistant", "content": content}


def _tool_results(results: list[tuple[str, str]]) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": tid, "content": content}
            for tid, content in results
        ],
    }


def run_single(task: Task, ex: Agent | None = None) -> Trace:
    """Single-agent loop: Executor calls tools until done or max_steps reached."""
    ex = ex or executor()
    trace = Trace(task=task.name, seed=task.seed, mode="single", model=ex.model)

    obs = task.reset()
    trace.log("observation", text=obs, initial=True)

    messages: list[dict[str, Any]] = [
        _user_msg(f"Task: {task.description}\n\nInitial observation: {obs}")
    ]

    for step in range(task.max_steps):
        response = ex.call(
            messages=messages,
            tools=task.tools,
            tool_choice={"type": "any"},
        )
        text = ex.text_of(response)
        if text:
            trace.log("message", text=text)
        tool_uses = ex.tool_uses(response)
        if not tool_uses:
            trace.log("error", reason="no tool call returned")
            break

        messages.append(_assistant_blocks(response.content))

        results = []
        done = False
        for tu in tool_uses:
            trace.log("tool_use", name=tu.name, input=tu.input, id=tu.id)
            step_result = task.step(tu.name, tu.input)
            trace.log(
                "observation",
                text=step_result.observation,
                reward=step_result.reward,
                done=step_result.done,
            )
            results.append((tu.id, step_result.observation))
            if step_result.done:
                done = True
        messages.append(_tool_results(results))
        trace.steps = step + 1
        if done:
            break

    trace.final_reward = task.episode_reward()
    trace.log("reward", value=trace.final_reward)
    return trace


def run_multi(
    task: Task,
    p: Agent | None = None,
    ex: Agent | None = None,
    c: Agent | None = None,
) -> Trace:
    """Multi-agent loop: Planner -> Executor tool loop -> Critic review."""
    p = p or planner()
    ex = ex or executor()
    c = c or critic()

    trace = Trace(task=task.name, seed=task.seed, mode="multi", model=ex.model)

    tool_names = [t["name"] for t in task.tools]
    plan = make_plan(p, task.name, task.description, tool_names)
    trace.log("plan", text=plan, by=p.name)

    obs = task.reset()
    trace.log("observation", text=obs, initial=True)

    messages: list[dict[str, Any]] = [
        _user_msg(
            f"Task: {task.description}\n\nPlan from the Planner:\n{plan}\n\n"
            f"Initial observation: {obs}"
        )
    ]

    for step in range(task.max_steps):
        response = ex.call(
            messages=messages,
            tools=task.tools,
            tool_choice={"type": "any"},
        )
        text = ex.text_of(response)
        if text:
            trace.log("message", text=text)
        tool_uses = ex.tool_uses(response)
        if not tool_uses:
            trace.log("error", reason="no tool call returned")
            break

        messages.append(_assistant_blocks(response.content))

        results = []
        done = False
        for tu in tool_uses:
            trace.log("tool_use", name=tu.name, input=tu.input, id=tu.id)
            step_result = task.step(tu.name, tu.input)
            trace.log(
                "observation",
                text=step_result.observation,
                reward=step_result.reward,
                done=step_result.done,
            )
            results.append((tu.id, step_result.observation))
            if step_result.done:
                done = True
        messages.append(_tool_results(results))
        trace.steps = step + 1
        if done:
            break

    trace.final_reward = task.episode_reward()
    trace.log("reward", value=trace.final_reward)

    critique = review(c, task.name, task.description, plan, trace.summary(), trace.final_reward)
    trace.log("critique", text=critique, by=c.name)
    return trace
