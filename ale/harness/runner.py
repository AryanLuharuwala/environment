from __future__ import annotations

from typing import Any

from ..agents import Agent, critic, executor, make_plan, planner, review
from ..tasks import Task
from .trace import Trace


def _user_msg(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def _run_tool_loop(ex: Agent, task: Task, trace: Trace, kickoff: str) -> None:
    messages: list[dict[str, Any]] = [_user_msg(kickoff)]
    for step in range(task.max_steps):
        response = ex.call(messages=messages, tools=task.tools, tool_choice="any")
        if response.text:
            trace.log("message", text=response.text)
        if not response.tool_calls:
            trace.log("error", reason="no tool call returned")
            break
        messages.append(ex.provider.echo_assistant(response))

        results: list[tuple[str, str]] = []
        done = False
        for tc in response.tool_calls:
            trace.log("tool_use", name=tc.name, input=tc.input, id=tc.id)
            step_result = task.step(tc.name, tc.input)
            trace.log(
                "observation",
                text=step_result.observation,
                reward=step_result.reward,
                done=step_result.done,
            )
            results.append((tc.id, step_result.observation))
            if step_result.done:
                done = True
        messages.extend(ex.provider.tool_results(results))
        trace.steps = step + 1
        if done:
            break


def run_single(task: Task, ex: Agent | None = None) -> Trace:
    """Single-agent loop: Executor calls tools until done or max_steps reached."""
    ex = ex or executor()
    trace = Trace(task=task.name, seed=task.seed, mode="single", model=ex.model)

    obs = task.reset()
    trace.log("observation", text=obs, initial=True)
    kickoff = f"Task: {task.description}\n\nInitial observation: {obs}"

    _run_tool_loop(ex, task, trace, kickoff)

    trace.final_reward = task.episode_reward()
    trace.log("reward", value=trace.final_reward)
    return trace


def run_multi(
    task: Task,
    p: Agent | None = None,
    ex: Agent | None = None,
    c: Agent | None = None,
    with_critic: bool = True,
) -> Trace:
    """Multi-agent loop: Planner -> Executor tool loop -> optional Critic."""
    p = p or planner()
    ex = ex or executor()

    trace = Trace(task=task.name, seed=task.seed, mode="multi", model=ex.model)

    tool_names = [t["name"] for t in task.tools]
    plan = make_plan(p, task.name, task.description, tool_names)
    trace.log("plan", text=plan, by=p.name)

    obs = task.reset()
    trace.log("observation", text=obs, initial=True)
    kickoff = (
        f"Task: {task.description}\n\nPlan from the Planner:\n{plan}\n\n"
        f"Initial observation: {obs}"
    )

    _run_tool_loop(ex, task, trace, kickoff)

    trace.final_reward = task.episode_reward()
    trace.log("reward", value=trace.final_reward)

    if with_critic:
        c = c or critic()
        critique = review(c, task.name, task.description, plan, trace.summary(), trace.final_reward)
        trace.log("critique", text=critique, by=c.name)
    return trace
