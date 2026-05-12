# traj_generation

**Stage 3 of the synthtools_env pipeline.** Turns **candidate tool sequences** into **multi-turn trajectories**: an LLM-driven *solver* invokes each tool in order, with every call verified by parameter check + simulator + optional judge + environment update. Output is a JSON trajectory capturing the full chat history, per-turn reasoning, and per-call debug log — ready to be consumed by [`traj_audit.summarize`](../traj_audit/summarize.py) and eventually exported as training data.

This stage has **two entry points**:

1. [`build_sequences`](build_sequences.py) — generates the candidate tool orderings (`spec["sequences"]`) from audited tools. Runs **after** `env_audit`, filters by reliability, writes sequences back into the env_spec.
2. [`run`](run.py) / [`generate.py`](generate.py) — takes sequences (from an env_spec, a field of specs, or a raw list) and produces one trajectory per sequence.

Each trajectory is a self-contained per-turn record. To produce a single merged task description per trajectory (needed for training datasets), run [`traj_audit.summarize`](../traj_audit/summarize.py) as the next stage — it batches across multiple trajectory files and writes a `summary` field back into each one.

---

## `build_sequences` — populate `spec["sequences"]` after audit

Why it's a separate stage: sequences depend on which tools are reliable. Running after `env_audit` lets us ask the LLM to chain *only* tools that pass a reliability threshold. The output is written back into the env_spec JSON.

CLI:
```bash
python -m traj_generation.build_sequences \
  --env-specs-dir tool_content/env_specs
  --model GPT-OSS-120B
  (--env-spec path/to/one_spec.json | --field "Aerospace and Defense")   # mutually exclusive
  [--n-sequences 10]          # number of sequences to request per spec
  [--seq-length 8]            # length of each sequence
  [--min-reliability 0.0]     # drop audited tools below this threshold (default: permissive)
  [--no-eval-only]            # include tools with no audit entry (default: exclude)
```

Behaviour:

- If a spec already has a non-empty `sequences` block, it's **skipped** (no LLM call). To force regeneration, clear the `sequences` field on disk first.
- By default (`eval_only=True`), tools missing from the audit block or with `reliability == None` are excluded from the sequence-generation prompt.
- If the filtered tool pool is smaller than `--seq-length`, the spec gets `sequences: {}` and a warning — not an error.
- Post-processing drops (1) sequences with consecutive duplicate tools, (2) exact duplicate sequences, (3) sequences referencing a tool not in the filtered pool. Name comparison normalizes whitespace, hyphens, and underscores (so declared `"Trade-off Analyzer"` still matches a sequence step `"TradeoffAnalyzer"`).
- **Batched**: all specs needing generation go through a single `batch_call` to vLLM, with per-request token usage attributed via `llm.last_usage_per_request`.
- **Provenance recorded**: each processed spec gets a new `{"phase": "sequences_build", ...}` entry appended to `spec["generation_log"]` containing the prompt, response, usage, model, model_config, filter settings (`min_reliability`, `eval_only`, `n_tools_filtered/total`), and `generated_at` timestamp.

Typical pipeline:
```bash
python -m env_generation.run --fields "..." --output-dir tool_content/env_specs --model ...
python -m env_audit.run --env-specs-dir tool_content/env_specs --dataset-path tool_content/tools_dataset.jsonl --eval-logs-dir tool_content/tool_eval_logs --model ...
python -m traj_generation.build_sequences --env-specs-dir tool_content/env_specs --field "..." --min-reliability 0.5 --model ...
python -m traj_generation.run --dataset tool_content/tools_dataset.jsonl --output-dir tool_content/trajectories --env-specs-dir tool_content/env_specs --field "..." --model ...
```

---

## `run` — three input modes

`run` never generates sequences itself. It reads pre-existing ones (from env_specs) or takes an explicit tool list, and produces trajectories.

The three modes are **mutually exclusive**; all share the same dataset/output/model flags.

### Mode A — explicit tool list (legacy)

```bash
python -m traj_generation.run \
  --dataset tool_content/tools_dataset.jsonl
  --output-dir tool_content/trajectories
  --model GPT-OSS-120B
  --tool-ids \
    "aerospace_and_defense_spec_0.MissionParser" \
    "aerospace_and_defense_spec_0.LiftRequirementEstimator" \
    "aerospace_and_defense_spec_0.WingGeometryDesigner"
```
Filename: `{spec_name}_NNN.json` (auto-numbered via `next_artifact_index` — never collides).

### Mode B — one env_spec

Runs one trajectory per sequence in `spec["sequences"]`:
```bash
python -m traj_generation.run \
  --dataset tool_content/tools_dataset.jsonl
  --output-dir tool_content/trajectories
  --model GPT-OSS-120B
  --env-spec tool_content/env_specs/aerospace_and_defense_spec_000.json
  [--max-trajectories 5]
```
Filename: `{spec_id}_{seq_key}.json` — so runs can be **resumed**: if the file already exists, that (spec, sequence) is skipped.

### Mode C — all specs in a field

Walks `--env-specs-dir`, picks every `*_spec_*.json` whose `field` matches, and dispatches Mode B per spec:
```bash
python -m traj_generation.run \
  --dataset tool_content/tools_dataset.jsonl
  --output-dir tool_content/trajectories
  --model GPT-OSS-120B
  --env-specs-dir tool_content/env_specs
  --field "Aerospace and Defense"
  [--max-trajectories 5]      # cap per spec, not per field
```
Same `{spec_id}_{seq_key}.json` filename convention, same skip-existing semantics.

Shared flags (all modes):
```
[--max-solver-turns 5]    # per-tool retries on bad output
[--max-retries 5]         # per-tool restarts (evolver proposes a new task)
[--verifiable]            # run TaskJudge after each solving loop
[--no-debug]              # skip the per-call event log
[--server-url URL]        # OpenAI-compatible endpoint (e.g. http://localhost:8765/v1).
                          #   When set, all LLM calls go over HTTP — vLLM is NOT loaded
                          #   in-process. Required for --concurrency > 1.
[--concurrency N]         # default 1. With N>1, fans out to a ProcessPoolExecutor
                          #   where each worker generates one trajectory at a time
                          #   against the shared server. Mode A (--tool-ids) rejects
                          #   --concurrency>1 since it's a single trajectory.
```

---

## Parallel mode (--server-url + --concurrency)

For production runs, launch one `vllm serve` per node and fan out N worker processes against it.

The wrapper script [`scripts/run_traj_gen.sh`](../scripts/run_traj_gen.sh) handles the full lifecycle:

```bash
bash scripts/run_traj_gen.sh --field "Investment Banking" --concurrency 5
# every default is overridable via flags or env vars; see --help
```

What happens under the hood:
1. Controller process builds the flat work list `(spec, seq_key, task_id, tool_ids, out_path)` for every pending trajectory.
2. **Resume-safe pre-filter**: items whose `{task_id}.json` already exists are dropped before submission.
3. `ProcessPoolExecutor.imap_unordered(workers=N, ...)` distributes work; slow trajectories never block fast ones (no straggler problem).
4. Each worker:
   - Redirects the `synthtools` logger to `<output-dir>/_logs/<task_id>.log` so detail doesn't interleave on stderr.
   - Constructs its own `LLM(server_url=...)` (cheap — just an HTTP client).
   - Calls `generate_trajectory(...)` exactly as in serial mode.
   - Writes outputs **atomically** (`.tmp` + `os.replace`) — a killed worker leaves at most a stray `.tmp`, never a half-written `.json`.

Throughput: typical workload (long prompts, ~7-15k tokens) sees **~3-5× per-node speedup at concurrency=4-6** before KV-cache memory becomes the bottleneck. The bench at [scripts/bench_vllm_serve.py](../scripts/bench_vllm_serve.py) shows the headline number for short prompts (~11× at concurrency=32) but real prompts are memory-bound, not decode-bound. Bump `--gpu-memory-utilization 0.95` (vLLM serve flag) for more KV headroom if you want to push past concurrency=5.

When `concurrency=1` (the default), the existing serial code path runs unchanged — same code, byte-identical behavior, all existing tests still apply.

---

## Input

- **`tools_dataset.jsonl`** — each row has `{id, field, subfield, task, tool_name, tool}`. Typically produced by `env_audit/` but any jsonl matching that shape works.
- **A list of tool IDs** — either supplied directly (mode A) or derived from `spec["sequences"]` (modes B/C) as `f"{spec_id}.{tool_name}"` per step.

---

## Pipeline (4 nested loops per trajectory)

The orchestrator is **not** batched — each tool in the sequence is handled in its own solver loop that chains 4 roles together across multiple LLM calls per turn.

For each tool in `--tool-ids`, up to `--max-retries` attempts:

```
┌─ TaskEvolver ────────────────────────────────┐   proposes a task that exercises this tool,
│  evolve_task_t0() / evolve_task_t1()         │   given successful prior tasks + env state
└───────────────────────────────────────────────┘

┌─ TaskSolver (inner loop, up to max-solver-turns) ─┐
│  produces {reason, tool_call}                      │
│  on no-tool-call: nudge + retry                    │
│  on failed param_check: retry with error feedback  │
│  on success: break                                 │
└────────────────────────────────────────────────────┘

┌─ ToolSimulator ───────────────────────────────────┐
│  parameter_check → simulate (if passed)            │
│  returns sim_json { status_code, response, ... }   │
└────────────────────────────────────────────────────┘

┌─ TaskJudge (only if --verifiable) ────────────────┐
│  judge_task_gen(expected_tool_call,                │
│                 task_description,                  │
│                 agent_tool_calls)                  │
│  → task_solved: bool                               │
└────────────────────────────────────────────────────┘

┌─ EnvironmentSimulator (only on success) ──────────┐
│  update_environment(tool_schema, tool_call,        │
│                     env_state, sim_output)         │
│  → new env_state for the next tool's evolver       │
└────────────────────────────────────────────────────┘
```

### Persistent solver chat

One rolling `solver_messages: List[{role, content}]` is maintained across the whole sequence. On every **successful** tool call, the orchestrator appends `[user_task, assistant_response, tool_response]` to it. The next tool's solver therefore sees the complete chat history from all previous tools — **no `past_task_history` string stuffing, no synthetic IDs, real chat format**.

### Tool → user rewrite for chat-template-strict models

GPT-OSS (and other models with structured `tool_calls`) reject a `role: "tool"` message unless the preceding assistant message carries a structured `tool_calls` field. Our solver puts the tool call as plain JSON inside `content`, not as `tool_calls`. So before sending to the LLM, [`utils.to_llm_messages`](../utils.py) rewrites `role: "tool"` → `role: "user"` with a `"Tool response: "` prefix. The saved trajectory keeps the `role: "tool"` semantic for training clarity.

---

## Output (in `--output-dir`)

### `{spec_name}_NNN.json` — the trajectory

```
task_id            — e.g. "aerospace_and_defense_spec_1_000"
model              — e.g. "GPT-OSS-120B"
config             — {max_solver_turns, max_retries, verifiable}
tool_ids           — input sequence
tools              — full tool schemas used
turns              — [ {tool_idx, attempt, tool_id, task, chat, judge, env_update, env_state_before/after, ...} ]
solver_chat        — the full rolling conversation (without the system prompt)
usage              — aggregate {prompt_tokens, completion_tokens, total_tokens}
generation_time_s  — wall-clock seconds
```

### `{spec_name}_NNN.debug.json` — every LLM call

Flat chronological list of every call made by any role (`TaskEvolver`, `TaskSolver`, `ToolSimulator`, `EnvironmentSimulator`, `TaskJudge`), with `{timestamp, agent, action, turn_ref, prompt, response, parsed, usage}`. `ToolSimulator` emits two events per call (one for `parameter_check`, one for `simulate`) so token accounting stays precise.

---

## Failure-recovery policy

- **Solver can't produce a tool_call**: append a user nudge message ("No tool call detected. Output a JSON with reason and tool_call."), retry up to `--max-solver-turns`.
- **Parameter check FAILs**: append the param-check error as a `role: "tool"` message, retry within the same solver loop.
- **max-solver-turns hit**: treat this as a failed attempt. TaskEvolver is called again (via `evolve_task_t1`) to propose a new task; the unsuccessful attempt is recorded in `unsuccessful_tasks_list` so the evolver can avoid the same trap.
- **max-retries hit for a tool**: stop the whole sequence. Saves whatever has been generated so far.
- **Simulator crashes**: nothing graceful yet — orchestrator bubbles the exception.

Numbering is still safe: `make_task_id` uses `max existing index + 1` (via [`utils.next_artifact_index`](../utils.py)), so a crash that leaves a half-written `.json` doesn't cause the next run to collide.

---

## Roles involved

| Role | Prompt templates | Purpose |
|---|---|---|
| [TaskEvolver](../roles/task_evolver.py) | task_evolver_t0/t1 | Given prior state, invent the next task |
| [TaskSolver](../roles/task_solver.py) | task_solver_gen | Produce one tool-call (JSON) + reasoning |
| [ToolSimulator](../roles/tool_simulator.py) | parameter_check + tool_simulator_template_metadata | Validate + simulate the tool call |
| [TaskJudge](../roles/task_judge.py) | task_judge_gen | Decide `task_solved ∈ {true,false}` (only if `--verifiable`) |
| [EnvironmentSimulator](../roles/env_simulator.py) | env_simulator | Compute the new env_state after this tool runs |

---

## Flags cheat-sheet

| Flag | Default | What it does |
|---|---|---|
| `--max-solver-turns` | 5 | Max retries *within one attempt* on bad outputs / param-check failures |
| `--max-retries` | 5 | Max *attempts* for a tool (each attempt starts over with a fresh evolver task) |
| `--verifiable` | off | Run `TaskJudge` after each solving loop; its verdict becomes `task_solved`. Off: "solved" means "got any successful simulator response". |
| `--no-debug` | off (= debug on) | Skip writing `.debug.json` |

---

## Files

- [generate.py](generate.py) — `generate_trajectory` (mode A) + `generate_trajectories_for_spec` (mode B) + `generate_trajectories_for_field` (mode C)
- [run.py](run.py) — CLI wrapper with mutually-exclusive `--tool-ids` / `--env-spec` / `--field` modes
- [build_sequences.py](build_sequences.py) — CLI + orchestrator for populating `spec["sequences"]` post-audit (separate entry point, reuses `EnvironmentGenerator.generate_sequences` from env_generation's role)

## Tests

- [../tests/test_traj_regression.py](../tests/test_traj_regression.py) — mode-A regression + mode-B/C coverage (skip-existing, field walk)
- [../tests/test_traj_concurrency.py](../tests/test_traj_concurrency.py) — parallel-mode orchestration: `list_pending_work_for_{spec,field}` filter behavior, CLI argument validation (`--concurrency` requires `--server-url`, Mode A rejects it), serial-vs-parallel path selection, per-worker log redirect
- [../tests/test_atomic_writes.py](../tests/test_atomic_writes.py) — `write_json_atomic` round-trip + crash safety
- [../tests/test_llm_server_url.py](../tests/test_llm_server_url.py) — HTTP branch of `LLM`: single + batched calls, usage attribution, vLLM is never loaded
- [../tests/test_build_sequences.py](../tests/test_build_sequences.py) — reliability filter, skip-if-present, separator-style tolerance, field mode
- [../tests/test_prompts_format.py](../tests/test_prompts_format.py) — covers all role + sequence templates

Run with `pytest tests/`.

---

## Gotchas

- **Not batched within a trajectory**: unlike `env_generation` and `env_audit`, this orchestrator is **serial per turn** because each role's input depends on the previous role's output. Trajectories take real wall-clock time per tool (typically 5-30 s each). Cross-trajectory parallelism is achieved at the `run.py` level via `--concurrency N` against a shared vLLM HTTP server (see "Parallel mode" above).
- **Solver system prompt is big (~8 KB)**: it includes a 4-turn CoT example. Each solver call pays that in input tokens. Trimming the example is a potential optimization but hurts reasoning quality.
- **`generation_time_s` is wall-clock**: includes model-load time on cold start — not comparable across runs until the engine is warm.
- **`turn.chat` vs `trajectory.solver_chat`**: `turn.chat` is the per-attempt conversation including any retries. `trajectory.solver_chat` is the *persistent* cross-tool chat with only successful interactions kept — the training-ready view. Use `solver_chat` for training, `turn.chat` for debugging what happened on a specific attempt.
