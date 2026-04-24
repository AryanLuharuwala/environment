# ALE — Agentic Learning Environment

A multi-agent playground plus RL-style benchmark suite for Claude. Drop agents
into small deterministic environments, run them with tool-use, and score the
trajectories. Built on the Anthropic Python SDK.

## Install

```bash
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...
```

Python 3.10+.

## Quick tour

```bash
ale list                                    # show tasks
ale run guess_number --mode multi           # planner + executor + critic
ale run code_fix --mode single --seed 2     # just an executor
ale bench --seeds 3 --save                  # sweep every task, save traces
```

Override the model globally with `ALE_MODEL=claude-sonnet-4-6 ale ...` or per
command with `--model`.

## What's in the box

### Agents (`ale/agents/`)

- **`executor`** — runs the tool-use loop against a task. System prompt cached.
- **`planner`** — given a task description and the tool surface, writes a short
  plan for the executor to follow.
- **`critic`** — reviews the trace after the run and returns a WELL / IMPROVE /
  CONFIDENCE critique.

All three share `Agent` (in `base.py`), a thin wrapper around
`anthropic.Anthropic` that sends the system prompt with
`cache_control: {"type": "ephemeral"}` so repeated rollouts read from cache.

### Tasks (`ale/tasks/`)

Every task implements `reset() -> observation` and
`step(tool_name, tool_input) -> StepResult(observation, reward, done, info)`,
with a scalar `episode_reward()` in `[0, 1]` at the end.

| Task           | Tools                          | Reward                                  |
|----------------|--------------------------------|-----------------------------------------|
| `guess_number` | `guess`                        | 1.0 at 1 guess, decays to 0 over 10     |
| `text_nav`     | `move` (n/s/e/w)               | `optimal_path / actual_steps`           |
| `code_fix`     | `submit`                       | fraction of unit tests passing          |
| `tool_math`    | `add`, `subtract`, `multiply`, `final_answer` | 1.0 if correct else 0.0 |

Add a task by subclassing `Task` and registering it in `ale/tasks/__init__.py`.

### Harness (`ale/harness/`)

- **`run_single(task)`** — one executor, tool loop until `done` or `max_steps`.
- **`run_multi(task)`** — planner → executor (same loop) → critic review.
- **`Trace`** — structured event log (plan / tool_use / observation / message /
  reward / critique). Serializes to JSON; `.summary()` renders a compact form
  for the critic.
- **`aggregate` / `render_table`** — roll rewards, success rates, and step
  counts into a leaderboard.

## Extending

**New task:** subclass `Task`, define `tools`, `reset`, `step`, and add it to the
registry.

**New agent role:** copy `planner.py`. Give it a fresh system prompt and a
function that feeds it the inputs it needs.

**Different orchestration:** write a new `run_*` in `ale/harness/runner.py`.
The existing ones are ~50 lines each and share the same message-building
helpers.

## Tests

```bash
pytest
```

The tests exercise task mechanics only (no API calls).

## Notes on cost

Each multi-agent rollout is roughly 3 calls (plan + executor turns + critique).
`bench --seeds 3` across 4 tasks = ~36 calls on the default Opus model. Swap
to Sonnet or Haiku via `--model` if you're iterating; system prompts are
cached so re-runs pay close to read prices.
