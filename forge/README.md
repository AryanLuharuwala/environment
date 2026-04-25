# forge — reasoning-from-history training data factory

A sibling package to `ale`. forge takes **real artifacts** — git commits,
legal corpora, compliance documents — and asks the model to reason over
them. A verifier suite scores each prediction with a rollout-based reward:
AST validity, diff similarity, tests, LLM grounding. The output is the
same DPO JSONL schema (`prompt`, `chosen`, `rejected`, `metadata`) that
`ale rlaif train` already consumes.

forge does not import any `ale.*` modules except `ale.providers` (the LLM
backend abstraction). The integration with the training loop is the JSONL
file alone — there's no runtime coupling.

## Two pipelines

### Coder: predict the next commit

```
example  ─►  retrieve K similar snippets  ─►  ask model for post-commit code
                                            │
                                            ▼
                            verify: AST + diff_similarity (+ optional pytest)
                                            │
                                            ▼
                                     composite reward
```

AST failure caps the reward at zero — broken syntax never beats merely
imperfect-but-parseable code. Two rollout configs per commit (with-RAG vs
no-RAG, by default) give the preference signal that DPO needs.

### Reasoning: grounded answers (law / compliance / policy)

```
question  ─►  retrieve K authoritative sources  ─►  ask agent A for an answer
                                                  │
                                                  ▼
                          verify with agent B (Claude judge): SUPPORT score
                                + structural citation check
                                                  │
                                                  ▼
                                          composite reward
```

This is the user's "agent reasons + another grounds it" pattern. The
grounding judge runs on Anthropic so its system prompt is cached across
rollouts.

## Layout

```
forge/
├── datasets/        Source ABC + GitCommitSource + JsonlSource
├── verifiers/       Verifier ABC + AST + diff + tests + LLM grounding
├── retrieval/       TF-IDF (default, zero-deps) + corpus loaders + optional embeddings
├── rollout/         CoderRollout + ReasoningRollout
├── export.py        DPO JSONL writer (matches ale.rlaif format)
└── cli.py           forge index | collect-coder | collect-reasoning
```

## Install

forge ships with the same `pip install -e .` as `ale`. Optional extra:

```bash
pip install -e '.[forge-embed]'    # adds sentence-transformers for semantic retrieval
```

## Quick paths

### Coding pipeline (mine a local repo, train DPO on the result)

```bash
# 1. Build a corpus index from a directory of "awesome" code
forge index --corpus path/to/awesome-python --out idx/awesome.json

# 2. Mine commits, run two contrasting rollouts per commit, write pairs
forge collect-coder \
    --repo path/to/some-py-project \
    --max-commits 50 \
    --rag idx/awesome.json \
    --out traces/coder_pairs.jsonl

# 3. Drop straight into ALE's DPO trainer
ale rlaif train --pairs traces/coder_pairs.jsonl \
    --base-model Qwen/Qwen2.5-0.5B-Instruct --out runs/forge-dpo
```

The default config-pair is `with_rag` vs `no_rag`. Higher composite reward
wins; ties are dropped. Provider can be Claude (`--provider anthropic`) or
any OpenAI-format endpoint (`--provider openai --model your-student
--rag idx/awesome.json` to evaluate the student against itself).

### Reasoning pipeline (law / compliance)

```bash
# 1. Index the authoritative source corpus (laws, regs, policy docs)
forge index --jsonl law_corpus.jsonl --out idx/law.json

# 2. Run grounded rollouts on a question set
#    questions.jsonl: {"id": ..., "context": "<question text>", "target": "<expected answer or summary>"}
forge collect-reasoning \
    --questions questions.jsonl \
    --rag idx/law.json \
    --judge-model claude-opus-4-7 \
    --out traces/law_pairs.jsonl
```

The judge is fixed to Anthropic so the grounding system prompt benefits
from `cache_control`. The answering agent can be any provider.

## Reward composition

| Domain    | Verifier              | Default weight | Notes                                |
|-----------|-----------------------|---------------:|--------------------------------------|
| coder     | `ast`                 | 0.4            | hard fail: 0 → reward zeroed         |
| coder     | `diff_similarity`     | 0.4            | difflib over normalized text         |
| coder     | `tests` (opt-in)      | 0.2            | sandboxed pytest with timeout        |
| reasoning | `grounding`           | 0.7            | LLM judge against retrieved sources  |
| reasoning | `citation_format`     | 0.3            | regex check that `[n]` cites are valid |

Override weights and the hard-fail set when constructing
`CoderRollout(weights=..., hard_fail=...)` if you want different
trade-offs.

## Verifiers

- **`PythonAstVerifier`** — `ast.parse`. Binary 0/1. Hard-failed by default
  in the coder rollout; broken code can't earn reward through partial
  similarity.
- **`DiffSimilarityVerifier`** — `difflib.SequenceMatcher.ratio()` over
  whitespace-normalized text against `example.target`.
- **`PytestVerifier`** — opt-in. Writes the prediction + a test file to
  `tempfile.TemporaryDirectory`, runs pytest with a 30s timeout. **Runs
  arbitrary model output**; only enable when you trust the input domain.
  Provide `module_filename` and `test_source` in `Example.metadata`.
- **`GroundingVerifier`** — Claude (or any provider) scores whether an
  answer is supported by retrieved sources. Returns `[0, 1]`. Used by
  `ReasoningRollout`.
- **`CitationFormatVerifier`** — cheap structural check: did the answer
  cite at least one valid `[n]` source? Catches naked hallucinations
  before they reach the LLM judge.

## Retrieval

- **`TfidfRetriever`** — pure-stdlib TF-IDF + cosine. Tokenizes code by
  splitting `snake_case` and `camelCase` so queries match either form.
  Persists as plain JSON via `.save()` / `.load()`.
- **`from_directory(root, suffixes=...)`** — walks a tree, reads code
  files (.py .js .ts .go .rs .java .cpp .c .rb by default), skips files
  larger than `max_bytes_per_file`.
- **`from_jsonl(path)`** — loads `{id, text, ...}` snippets from JSONL.
  Easier to share corpora across machines.
- **`EmbeddingRetriever`** (optional, `forge[forge-embed]`) — drop-in
  sentence-transformers backend.

## Sources

- **`GitCommitSource`** — local git via subprocess. Per (commit, file) it
  yields an `Example` whose `context` is the pre-commit version + commit
  message and whose `target` is the post-commit version.
- **`JsonlSource`** — generic. Used for non-code domains (law, compliance,
  Q&A datasets). Each line: `{id, context, target, ...}`.
- **`GitHubApiSource`** — stub that raises `NotImplementedError`. The
  abstraction is in place; remote fetching is intentionally deferred.

## Closing the loop with ALE

`forge collect-*` writes JSONL with the same shape as
`ale/rlaif/export.py` produces. Concatenate sources freely:

```bash
cat traces/forge_coder_pairs.jsonl traces/pairs.jsonl > traces/all.jsonl
ale rlaif train --pairs traces/all.jsonl --base-model ...
```

After training, serve the adapter behind any OpenAI-format server (vLLM,
llama.cpp, Ollama) and run forge again with `--provider openai` to
generate the next round's preference data on the new student. That's the
loop.

## Safety

- `PytestVerifier` runs LLM output. Sandboxed (tempdir + 30s timeout) but
  still arbitrary code execution. Off by default; enable per-call with
  `--with-tests` only on inputs you'd run yourself.
- `forge index --corpus DIR` walks any directory you point it at. There's
  no upload — everything stays local — but the resulting index file
  contains the raw source it indexed. Treat it as you'd treat a copy of
  the corpus.
- The grounding judge's system prompt is sent with `cache_control`, so
  per-rollout cost on Claude is mostly cache-read price.

## Tests

Twenty-six offline tests across the four forge modules + the existing 15
ALE tests. All run without API keys.

```bash
pytest -q
```

The `GitCommitSource` test builds a real temp git repo. In environments
that force commit signing (CI sandboxes, etc.) it skips cleanly — the
code under test is environment-agnostic; only the fixture isn't.
