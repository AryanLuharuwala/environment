from __future__ import annotations

import argparse
import json
import sys
from itertools import islice
from pathlib import Path

from ale.providers import make_provider

from .datasets import Example, GitCommitSource, JsonlSource
from .export import pair_to_dpo, write_dpo_jsonl
from .retrieval import TfidfRetriever, from_directory, from_jsonl
from .rollout import CoderConfig, CoderRollout, ReasoningConfig, ReasoningRollout


def _cmd_index(args: argparse.Namespace) -> int:
    docs: list = []
    if args.corpus:
        docs = from_directory(Path(args.corpus), suffixes=args.suffixes)
        print(f"loaded {len(docs)} files from {args.corpus}")
    elif args.jsonl:
        docs = from_jsonl(Path(args.jsonl))
        print(f"loaded {len(docs)} snippets from {args.jsonl}")
    else:
        print("error: pass --corpus DIR or --jsonl FILE", file=sys.stderr)
        return 2
    retriever = TfidfRetriever().index(docs)
    out = Path(args.out)
    retriever.save(out)
    print(f"wrote TF-IDF index to {out}  ({len(docs)} docs)")
    return 0


def _build_provider(args: argparse.Namespace):
    return make_provider(args.provider, model=args.model)


def _cmd_collect_coder(args: argparse.Namespace) -> int:
    src = GitCommitSource(
        repo=Path(args.repo),
        max_commits=args.max_commits,
        branch=args.branch,
        suffix=args.suffix,
    )
    retriever = TfidfRetriever.load(Path(args.rag)) if args.rag else None
    provider = _build_provider(args)

    cfg_a = CoderConfig(label="with_rag", use_rag=True, k=args.k, provider=provider)
    cfg_b = CoderConfig(label="no_rag", use_rag=False, k=args.k, provider=provider)
    rollout_a = CoderRollout.default(cfg_a, retriever=retriever, with_tests=args.with_tests)
    rollout_b = CoderRollout.default(cfg_b, retriever=retriever, with_tests=args.with_tests)

    rows: list[dict] = []
    ties = 0
    for example in islice(src.iter_examples(), args.limit):
        a = rollout_a.run(example)
        b = rollout_b.run(example)
        scores_a = " ".join(f"{s.name}={s.value:.2f}" for s in a.scores)
        scores_b = " ".join(f"{s.name}={s.value:.2f}" for s in b.scores)
        print(
            f"{example.id}\n"
            f"  A[{a.config_label}] reward={a.reward:.3f}  {scores_a}\n"
            f"  B[{b.config_label}] reward={b.reward:.3f}  {scores_b}"
        )
        row = pair_to_dpo(example, a, b)
        if row is None:
            ties += 1
            continue
        rows.append(row)

    out = Path(args.out)
    write_dpo_jsonl(rows, out)
    print(f"\nwrote {len(rows)} DPO pairs to {out}  ({ties} ties skipped)")
    return 0


def _cmd_collect_reasoning(args: argparse.Namespace) -> int:
    src = JsonlSource(path=Path(args.questions))
    retriever = TfidfRetriever.load(Path(args.rag)) if args.rag else None
    answer_provider = _build_provider(args)
    judge_provider = make_provider("anthropic", model=args.judge_model)

    from .verifiers import GroundingVerifier
    grounding = GroundingVerifier(provider=judge_provider)

    cfg_a = ReasoningConfig(label="grounded_rag", use_rag=True, k=args.k, provider=answer_provider)
    cfg_b = ReasoningConfig(label="ungrounded", use_rag=False, k=args.k, provider=answer_provider)
    rollout_a = ReasoningRollout(config=cfg_a, retriever=retriever, grounding=grounding)
    rollout_b = ReasoningRollout(config=cfg_b, retriever=retriever, grounding=grounding)

    rows: list[dict] = []
    ties = 0
    for example in islice(src.iter_examples(), args.limit):
        a = rollout_a.run(example)
        b = rollout_b.run(example)
        print(
            f"{example.id}\n"
            f"  A[{a.config_label}] reward={a.reward:.3f}\n"
            f"  B[{b.config_label}] reward={b.reward:.3f}"
        )
        row = pair_to_dpo(example, a, b, instruction_prefix=args.instruction)
        if row is None:
            ties += 1
            continue
        rows.append(row)

    out = Path(args.out)
    write_dpo_jsonl(rows, out)
    print(f"\nwrote {len(rows)} DPO pairs to {out}  ({ties} ties skipped)")
    return 0


def _add_provider_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    p.add_argument("--model", default=None,
                   help="Model ID (defaults to provider's env var)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forge", description="forge — reasoning-from-history data factory")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_idx = sub.add_parser("index", help="Build a TF-IDF corpus index")
    p_idx.add_argument("--corpus", help="Directory of code files to index")
    p_idx.add_argument("--jsonl", help="JSONL of {id, text, ...} snippets to index")
    p_idx.add_argument("--out", required=True, help="Path for the index JSON")
    p_idx.add_argument("--suffixes", nargs="*", default=None,
                       help="Override default file suffixes (.py .js .ts ...)")
    p_idx.set_defaults(func=_cmd_index)

    p_cc = sub.add_parser("collect-coder", help="Mine commits and emit DPO pairs scored by AST + diff + (optional) tests")
    p_cc.add_argument("--repo", required=True, help="Path to a local git repo")
    p_cc.add_argument("--max-commits", type=int, default=50)
    p_cc.add_argument("--branch", default=None)
    p_cc.add_argument("--suffix", default=".py")
    p_cc.add_argument("--rag", help="Path to a TF-IDF index built by `forge index`")
    p_cc.add_argument("--k", type=int, default=3, help="Top-K retrieved snippets")
    p_cc.add_argument("--with-tests", action="store_true",
                      help="Run pytest on predicted code (requires test_source in metadata; opt-in for safety)")
    p_cc.add_argument("--limit", type=int, default=20)
    p_cc.add_argument("--out", default="traces/forge_coder_pairs.jsonl")
    _add_provider_flags(p_cc)
    p_cc.set_defaults(func=_cmd_collect_coder)

    p_cr = sub.add_parser("collect-reasoning",
                          help="Generic reason-then-ground pipeline (law/compliance) emitting DPO pairs")
    p_cr.add_argument("--questions", required=True, help="JSONL of {id, context, target, ...}")
    p_cr.add_argument("--rag", help="Path to a TF-IDF index over authoritative sources")
    p_cr.add_argument("--k", type=int, default=4)
    p_cr.add_argument("--limit", type=int, default=50)
    p_cr.add_argument("--out", default="traces/forge_reasoning_pairs.jsonl")
    p_cr.add_argument("--instruction", default="",
                      help="Optional prefix prepended to the prompt in DPO output")
    p_cr.add_argument("--judge-model", default=None,
                      help="Claude model for the grounding judge (always Anthropic, kept cached)")
    _add_provider_flags(p_cr)
    p_cr.set_defaults(func=_cmd_collect_reasoning)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
