from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..agents import Agent, executor, planner
from ..harness import Trace, run_multi, run_single
from ..providers import Provider
from ..tasks import Task, make


@dataclass
class RolloutConfig:
    """One way to run a task. The RLAIF loop generates pairs of rollouts with
    *different* configs per (task, seed) so there's real variance to rank."""

    label: str
    mode: str  # "single" or "multi"
    # Optional hook to customize agents (swap executor prompt, etc.).
    build_executor: Callable[[Provider], Agent] | None = None
    build_planner: Callable[[Provider], Agent] | None = None


def rollout(task: Task, cfg: RolloutConfig, provider: Provider) -> Trace:
    ex = cfg.build_executor(provider) if cfg.build_executor else executor(provider=provider)
    if cfg.mode == "single":
        return run_single(task, ex=ex)
    pl = cfg.build_planner(provider) if cfg.build_planner else planner(provider=provider)
    # Skip critic during RLAIF collection — we have a dedicated judge.
    return run_multi(task, p=pl, ex=ex, with_critic=False)


def default_pair() -> tuple[RolloutConfig, RolloutConfig]:
    """Sensible default: 'multi' (planner + executor) vs 'single' (executor only).

    Across most tasks the planner-guided run wins, producing good preference
    signal. Swap in your own configs for more interesting variance (different
    executor prompts, different temperatures on providers that support them,
    etc.)."""
    return (
        RolloutConfig(label="with_plan", mode="multi"),
        RolloutConfig(label="no_plan", mode="single"),
    )


@dataclass
class RolloutPair:
    task_name: str
    seed: int
    a: Trace
    a_label: str
    b: Trace
    b_label: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task_name,
            "seed": self.seed,
            "a": {"label": self.a_label, **self.a.to_dict()},
            "b": {"label": self.b_label, **self.b.to_dict()},
        }


def collect_pair(
    task_name: str,
    seed: int,
    provider: Provider,
    configs: tuple[RolloutConfig, RolloutConfig] | None = None,
) -> RolloutPair:
    cfg_a, cfg_b = configs or default_pair()
    task_a = make(task_name, seed=seed)
    task_b = make(task_name, seed=seed)
    trace_a = rollout(task_a, cfg_a, provider)
    trace_b = rollout(task_b, cfg_b, provider)
    return RolloutPair(
        task_name=task_name, seed=seed,
        a=trace_a, a_label=cfg_a.label,
        b=trace_b, b_label=cfg_b.label,
    )
