from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from .agents import DEFAULT_MODEL, critic, executor, planner
from .harness import aggregate, render_table, run_multi, run_single
from .tasks import REGISTRY, make


def _cmd_list(_args: argparse.Namespace) -> int:
    print("Available tasks:")
    for name, cls in REGISTRY.items():
        desc = (cls.description or "").split("\n")[0]
        print(f"  {name:<14} {desc[:70]}")
    print(f"\nDefault model: {DEFAULT_MODEL}  (override with ALE_MODEL env var or --model)")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    task = make(args.task, seed=args.seed)
    if args.mode == "single":
        trace = run_single(task, ex=executor(model=args.model))
    else:
        trace = run_multi(
            task,
            p=planner(model=args.model),
            ex=executor(model=args.model),
            c=critic(model=args.model),
        )

    print(f"\n== {task.name}  seed={task.seed}  mode={args.mode}  model={args.model} ==")
    print(trace.summary())
    print(f"\nfinal_reward={trace.final_reward:.3f}  steps={trace.steps}")

    if args.save:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = Path("traces") / f"{task.name}-{args.mode}-s{args.seed}-{ts}.json"
        trace.save(path)
        print(f"\ntrace saved to {path}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    task_names = args.tasks or list(REGISTRY.keys())
    seeds = list(range(args.seeds))
    traces = []

    for name in task_names:
        for seed in seeds:
            task = make(name, seed=seed)
            if args.mode == "single":
                trace = run_single(task, ex=executor(model=args.model))
            else:
                trace = run_multi(
                    task,
                    p=planner(model=args.model),
                    ex=executor(model=args.model),
                    c=critic(model=args.model),
                )
            print(
                f"{name:<14} seed={seed}  reward={trace.final_reward:.3f}  "
                f"steps={trace.steps}"
            )
            traces.append(trace)

            if args.save:
                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                path = Path("traces") / f"{name}-{args.mode}-s{seed}-{ts}.json"
                trace.save(path)

    print("\n" + render_table(aggregate(traces)))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ale", description="Agentic Learning Environment"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List available tasks").set_defaults(func=_cmd_list)

    common = {
        "--model": dict(default=DEFAULT_MODEL, help=f"Claude model (default: {DEFAULT_MODEL})"),
        "--mode": dict(choices=["single", "multi"], default="multi", help="Agent configuration"),
        "--save": dict(action="store_true", help="Save traces to ./traces/"),
    }

    p_run = sub.add_parser("run", help="Run one task")
    p_run.add_argument("task", choices=list(REGISTRY.keys()))
    p_run.add_argument("--seed", type=int, default=0)
    for flag, kw in common.items():
        p_run.add_argument(flag, **kw)
    p_run.set_defaults(func=_cmd_run)

    p_bench = sub.add_parser("bench", help="Run the benchmark suite")
    p_bench.add_argument("--tasks", nargs="*", choices=list(REGISTRY.keys()))
    p_bench.add_argument("--seeds", type=int, default=3, help="Seeds per task")
    for flag, kw in common.items():
        p_bench.add_argument(flag, **kw)
    p_bench.set_defaults(func=_cmd_bench)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
