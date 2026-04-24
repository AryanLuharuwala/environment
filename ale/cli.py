from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from .agents import critic, executor, planner
from .harness import aggregate, render_table, run_multi, run_single
from .providers import Provider, make_provider
from .rlaif import (
    RolloutConfig,
    collect_pair,
    default_pair,
    judge_pair,
    pair_to_dpo,
    write_dpo_jsonl,
)
from .tasks import REGISTRY, make


def _provider(args: argparse.Namespace, role: str = "main") -> Provider:
    """Build a provider respecting --provider/--model. `role='judge'` always
    uses Anthropic so Claude judges even when the Executor is OpenAI-format."""
    if role == "judge":
        return make_provider("anthropic", model=args.judge_model)
    return make_provider(args.provider, model=args.model)


def _cmd_list(_args: argparse.Namespace) -> int:
    print("Available tasks:")
    for name, cls in REGISTRY.items():
        desc = (cls.description or "").split("\n")[0]
        print(f"  {name:<14} {desc[:70]}")
    print("\nProviders: anthropic (default, cached), openai (OpenAI-format, any compat server)")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    provider = _provider(args)
    task = make(args.task, seed=args.seed)
    if args.mode == "single":
        trace = run_single(task, ex=executor(provider=provider))
    else:
        trace = run_multi(
            task,
            p=planner(provider=provider),
            ex=executor(provider=provider),
            c=critic(provider=_provider(args, role="judge")),
        )

    print(f"\n== {task.name}  seed={task.seed}  mode={args.mode}  model={provider.model} ==")
    print(trace.summary())
    print(f"\nfinal_reward={trace.final_reward:.3f}  steps={trace.steps}")

    if args.save:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = Path("traces") / f"{task.name}-{args.mode}-s{args.seed}-{ts}.json"
        trace.save(path)
        print(f"\ntrace saved to {path}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    provider = _provider(args)
    task_names = args.tasks or list(REGISTRY.keys())
    seeds = list(range(args.seeds))
    traces = []

    for name in task_names:
        for seed in seeds:
            task = make(name, seed=seed)
            if args.mode == "single":
                trace = run_single(task, ex=executor(provider=provider))
            else:
                trace = run_multi(
                    task,
                    p=planner(provider=provider),
                    ex=executor(provider=provider),
                    c=critic(provider=_provider(args, role="judge")),
                )
            print(f"{name:<14} seed={seed}  reward={trace.final_reward:.3f}  steps={trace.steps}")
            traces.append(trace)
            if args.save:
                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                path = Path("traces") / f"{name}-{args.mode}-s{seed}-{ts}.json"
                trace.save(path)

    print("\n" + render_table(aggregate(traces)))
    return 0


def _cmd_rlaif_collect(args: argparse.Namespace) -> int:
    provider = _provider(args)
    judge_provider = _provider(args, role="judge")

    task_names = args.tasks or list(REGISTRY.keys())
    seeds = list(range(args.seeds))

    configs = None
    if args.configs:
        configs = (
            RolloutConfig(label=args.configs[0], mode=args.configs[0].split("=")[1] if "=" in args.configs[0] else args.configs[0]),
            RolloutConfig(label=args.configs[1], mode=args.configs[1].split("=")[1] if "=" in args.configs[1] else args.configs[1]),
        )

    examples: list[dict] = []
    ties = 0

    for name in task_names:
        for seed in seeds:
            pair = collect_pair(name, seed, provider=provider, configs=configs)
            pref = judge_pair(
                pair,
                judge=None if args.no_judge else _make_judge(judge_provider),
                require_llm_for_equal_rewards=not args.no_judge,
            )
            print(
                f"{name:<14} seed={seed}  "
                f"A[{pair.a_label}]={pref.reward_a:.2f}  B[{pair.b_label}]={pref.reward_b:.2f}  "
                f"-> {pref.winner}  ({pref.source}: {pref.reason[:50]})"
            )
            example = pair_to_dpo(pair, pref, task_description=REGISTRY[name].description)
            if example is None:
                ties += 1
                continue
            examples.append(example)

    out_path = Path(args.out)
    write_dpo_jsonl(examples, out_path)
    print(f"\nwrote {len(examples)} preference pairs to {out_path}  ({ties} ties skipped)")
    return 0


def _make_judge(provider):
    from .rlaif.preference import _judge_agent  # type: ignore
    return _judge_agent(provider=provider)


def _cmd_rlaif_train(args: argparse.Namespace) -> int:
    from .rlaif.train_dpo import DPOArgs, train
    train(
        DPOArgs(
            pairs_path=Path(args.pairs),
            base_model=args.base_model,
            out_dir=Path(args.out),
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
        )
    )
    return 0


def _add_provider_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"],
                   help="LLM backend for Executor/Planner (default: anthropic)")
    p.add_argument("--model", default=None,
                   help="Model ID for --provider (defaults to provider's env var)")
    p.add_argument("--judge-model", default=None,
                   help="Claude model for Critic/Judge role (always Anthropic, stays cached)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ale", description="Agentic Learning Environment")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List available tasks").set_defaults(func=_cmd_list)

    p_run = sub.add_parser("run", help="Run one task")
    p_run.add_argument("task", choices=list(REGISTRY.keys()))
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--mode", choices=["single", "multi"], default="multi")
    p_run.add_argument("--save", action="store_true")
    _add_provider_flags(p_run)
    p_run.set_defaults(func=_cmd_run)

    p_bench = sub.add_parser("bench", help="Run the benchmark suite")
    p_bench.add_argument("--tasks", nargs="*", choices=list(REGISTRY.keys()))
    p_bench.add_argument("--seeds", type=int, default=3)
    p_bench.add_argument("--mode", choices=["single", "multi"], default="multi")
    p_bench.add_argument("--save", action="store_true")
    _add_provider_flags(p_bench)
    p_bench.set_defaults(func=_cmd_bench)

    p_rlaif = sub.add_parser("rlaif", help="RLAIF collection + training")
    rlaif_sub = p_rlaif.add_subparsers(dest="rlaif_cmd", required=True)

    p_collect = rlaif_sub.add_parser(
        "collect", help="Generate preference pairs by comparing two rollout configs per seed"
    )
    p_collect.add_argument("--tasks", nargs="*", choices=list(REGISTRY.keys()))
    p_collect.add_argument("--seeds", type=int, default=4)
    p_collect.add_argument("--out", default="traces/pairs.jsonl")
    p_collect.add_argument(
        "--configs", nargs=2, metavar=("A", "B"),
        help="Two rollout modes to compare (e.g. 'multi' 'single'). Default: multi vs single.",
    )
    p_collect.add_argument("--no-judge", action="store_true",
                           help="Skip LLM judge on ties; drop ties instead")
    _add_provider_flags(p_collect)
    p_collect.set_defaults(func=_cmd_rlaif_collect)

    p_train = rlaif_sub.add_parser("train", help="DPO-train a base model on preference pairs")
    p_train.add_argument("--pairs", default="traces/pairs.jsonl")
    p_train.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p_train.add_argument("--out", default="runs/dpo")
    p_train.add_argument("--epochs", type=int, default=1)
    p_train.add_argument("--lr", type=float, default=5e-6)
    p_train.add_argument("--batch-size", type=int, default=1)
    p_train.add_argument("--grad-accum", type=int, default=8)
    p_train.set_defaults(func=_cmd_rlaif_train)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
