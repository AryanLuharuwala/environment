# ALE — Agentic Learning Environment

A multi-agent playground plus RLAIF pipeline for Claude, with pluggable
OpenAI-format backends for the student model. Built on the Anthropic Python
SDK (and optionally the `openai` SDK).

## Install

```bash
pip install -e .                 # core: anthropic only
pip install -e '.[openai]'       # add OpenAI-format backend
pip install -e '.[train]'        # add DPO training deps (torch/trl/peft/datasets)
export ANTHROPIC_API_KEY=sk-ant-...
```

Python 3.10+.

## Tour

```bash
ale list                                    # show tasks and providers
ale run guess_number --mode multi           # planner + executor + critic (all Claude)
ale run code_fix --provider openai \
      --model gpt-4o-mini                   # executor on OpenAI, judge stays Claude
ale bench --seeds 3 --save                  # sweep every task, save traces
```

The Critic / Judge always uses Anthropic so the cached system prompt keeps
earning cache reads regardless of which provider runs the Executor.

## Providers

Two backends share one `Provider` interface
(`ale/providers/base.py`). Pick with `--provider` or `ALE_PROVIDER`:

| Provider   | Good for                            | Prompt caching | Tool schema |
|------------|-------------------------------------|----------------|-------------|
| `anthropic`| Claude API (default)                | Yes (ephemeral)| Anthropic native |
| `openai`   | OpenAI, vLLM, Ollama, Together, OpenRouter, Groq, LM Studio | No | auto-converted from the task schema |

Environment: `OPENAI_API_KEY`, `OPENAI_BASE_URL` for the OpenAI-format path.
Point `OPENAI_BASE_URL` at a local vLLM server to swap in your own model.

## Multi-agent core

- **`executor`** — runs the tool-use loop.
- **`planner`** — writes a short plan; seeds the Executor's context.
- **`critic`** — post-hoc WELL / IMPROVE / CONFIDENCE critique.

All share `Agent`, a thin wrapper that just forwards to its `Provider`. Swap
roles onto different backends by passing a `provider=` to each factory —
e.g. a cheap executor on Haiku + a strict critic on Opus.

## Tasks

| Task           | Tools                                      | Reward                                  |
|----------------|--------------------------------------------|-----------------------------------------|
| `guess_number` | `guess`                                    | 1.0 at 1 guess, decays over 10          |
| `text_nav`     | `move` (n/s/e/w)                           | `optimal_path / actual_steps`           |
| `code_fix`     | `submit`                                   | fraction of unit tests passing          |
| `tool_math`    | `add`, `subtract`, `multiply`, `final_answer` | 1.0 if correct else 0.0              |

Add one by subclassing `Task` and registering in `ale/tasks/__init__.py`.

## RLAIF

End-to-end loop: generate contrasting rollouts → Claude judges → DPO-train a
student model → serve the student behind the OpenAI provider.

### 1. Collect preference pairs

```bash
ale rlaif collect --tasks guess_number tool_math --seeds 8 \
    --out traces/pairs.jsonl
```

For each (task, seed) this runs **two rollouts with different configs**
(default: `multi` with planner vs `single` without) and records a preference:

- Higher-reward rollout wins automatically.
- On ties, Claude (cached system prompt) judges on correctness / efficiency /
  reasoning quality and returns `WINNER: A | B | TIE`.

Output is a TRL-ready JSONL: `{prompt, chosen, rejected, metadata}`. Ties are
dropped.

Use `--provider openai --model ...` to collect rollouts from a student model
while Claude stays in the judge seat.

### 2. Train DPO

```bash
pip install -e '.[train]'
ale rlaif train --pairs traces/pairs.jsonl \
    --base-model Qwen/Qwen2.5-0.5B-Instruct \
    --out runs/dpo-qwen --epochs 1
```

LoRA adapter via `peft`, DPO loop via `trl`. Defaults are conservative
(batch 1, grad-accum 8, LR 5e-6, beta 0.1) so it runs on a single small GPU.
Tune from there.

### 3. Serve the student

Any OpenAI-format server works. vLLM with the adapter:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --enable-lora --lora-modules dpo=runs/dpo-qwen
```

Then point ALE at it:

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=sk-local
ale bench --provider openai --model dpo --seeds 3
```

Re-running `rlaif collect` on the student closes the loop: each generation's
pairs become training data for the next.

### Customizing the rollout pair

`default_pair()` returns `(RolloutConfig('with_plan', mode='multi'),
RolloutConfig('no_plan', mode='single'))`. Swap to your own configs by passing
them to `collect_pair(...)` in Python, or wire them through the CLI with
`--configs multi single`. For richer variance, supply different `build_executor`
factories (e.g. a concise vs verbose system prompt).

## Tests

```bash
pytest                              # 15 tests, all offline (no API calls)
```

- `test_tasks.py` — task mechanics
- `test_providers.py` — both providers against scripted fake clients
- `test_rlaif.py` — preference decisions + DPO export

## Cost

Each multi-agent rollout is roughly 3 Claude calls (plan + executor turns +
critique). `bench --seeds 3` across 4 tasks ≈ 36 calls. RLAIF `collect
--seeds 8` doubles that plus a judge call per tie. System prompts are cached
so re-runs pay mostly cache-read prices.

Swap the Executor to `--provider openai --model gpt-4o-mini` (or a local
vLLM model) to run cheap rollouts while the Judge / Critic stays on
claude-opus-4-7 with caching.
