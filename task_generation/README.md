# Stage 3 — task_generation

Bottom-up task construction. From a validated tool set, this stage builds
solvable, multi-step verifiable tasks. Each task pairs a natural-language
goal with a ground-truth tool-call sequence and an explicit final environment
state, so any rollout against it can be machine-verified.

The stage has two halves:

- **3a — `build_sequences.py`** — populate `spec["sequences"]` with
  candidate tool orderings, post-audit, so only reliable tools are used.
- **3b — `run.py`** — run a per-step orchestration loop (evolver → solver
  → simulator → judge → environment update) over each sequence and write
  one task JSON per `(spec, sequence)` pair.

## 3a · `task_generation.build_sequences`

Why a separate stage: sequences must depend on which tools are reliable.
Running after `env_audit` lets us filter by reliability before chaining.

```bash
python -m task_generation.build_sequences \
    --env-specs-dir <env_specs_dir> \
    --model GPT-OSS-120B \
    [--field "Investment Banking" | --env-spec <one_spec.json>] \
    [--n-sequences 10] [--seq-length 8] \
    [--min-reliability 0.5] [--no-eval-only]
```

Behaviour:

- Tools missing from the audit block, or with reliability below
  `--min-reliability`, are excluded from the sequence-generation prompt.
- Skips specs that already have a non-empty `sequences` block.
- Output is written back into the env_spec JSON; provenance is appended to
  `spec["generation_log"]` as `{"phase": "sequences_build", ...}`.
- Post-processing drops sequences with consecutive duplicates, exact
  duplicates, or references to filtered-out tools.

## 3b · `task_generation.run`

For each sequence in a spec, runs a 4-role per-turn loop:

```
TaskEvolver  → proposes a per-turn task + expected_tool_call + env_metadata
TaskSolver   → emits {reason, tool_call} given the task and tool schema
ToolSimulator→ parameter_check → simulate (if PASS), conditioned on metadata
TaskJudge    → (if --verifiable) compares agent call vs ground truth, scores
EnvironmentSimulator → updates env_state for the next turn (only on success)
```

Three input modes (mutually exclusive):

```bash
# A. explicit ordered tool list (legacy)
python -m task_generation.run \
    --dataset <tools_dataset.jsonl> \
    --output-dir <traj_dir> \
    --model GPT-OSS-120B \
    --tool-ids "<spec_id>.ToolA" "<spec_id>.ToolB" ...

# B. one env_spec → one task per sequence
python -m task_generation.run \
    --dataset <tools_dataset.jsonl> \
    --output-dir <traj_dir> \
    --model GPT-OSS-120B \
    --env-spec <spec.json> \
    [--max-tasks 5]

# C. all specs in a field
python -m task_generation.run \
    --dataset <tools_dataset.jsonl> \
    --output-dir <traj_dir> \
    --model GPT-OSS-120B \
    --env-specs-dir <env_specs_dir> \
    --field "Investment Banking" \
    [--max-tasks 5]
```

Shared flags:
```
[--max-solver-turns 5]    # per-tool retries on bad output
[--max-retries 5]         # per-tool restarts (evolver proposes a new task)
[--verifiable]            # run TaskJudge after each solving loop
[--no-debug]              # skip the per-LLM-call event log
[--server-url URL]        # OpenAI-compatible HTTP endpoint
[--concurrency N]         # parallel workers; requires --server-url
```

Files are named `{spec_id}_{seq_key}.json`. If the file already exists, the
`(spec, sequence)` pair is skipped — runs are resume-safe.

## Output JSON shape

Top-level:
```
{
  "task_id":          "<spec_id>_<seq_key>",
  "model":            "GPT-OSS-120B",
  "config":           {max_solver_turns, max_retries, verifiable},
  "tool_ids":         [<tool name per step>],
  "tools":            [<tool schemas>],
  "turns":            [<one entry per attempt — see below>],
  "solver_chat":      [<rolling chat minus system messages>],
  "usage":            {prompt_tokens, completion_tokens, total_tokens},
  "generation_time_s": <wall clock>
}
```

Each turn:
```
{
  "tool_idx":              0,
  "attempt":               0,                    # outer retry index
  "tool_id":               "<spec_id>.ToolName",
  "env_state_before":      ...,
  "task":                  {task_description, expected_tool_call,
                            env_metadata, edited_metadata, depends_on_previous},
  "env_state_after_task":  ...,
  "chat":                  [<this attempt's user/assistant/tool messages>],
  "judge":                 <verdict dict | None>,
  "env_update":            <state diff | None>,
  "env_state_after":       ...
}
```

A sibling `<task_id>.debug.json` is written when `--no-debug` is not set; it
records every LLM call (prompt, response, parsed JSON, usage) for every role
including the two-phase tool simulator.

## Files

- `run.py` — CLI; serial path (concurrency=1) and parallel path
  (`_worker` + `ProcessPoolExecutor` against a shared vLLM server).
- `generate.py` — `generate_task`, `generate_tasks_for_spec`,
  `generate_tasks_for_field`, plus the `WorkItem` listing helpers used by
  the parallel path.
- `build_sequences.py` — sequence-building post-audit.
- Prompt templates: `../prompt_templates/task_evolver/`,
  `../prompt_templates/task_solver/`, `../prompt_templates/tool_simulator/`,
  `../prompt_templates/task_judge/`.

## Tests

- `tests/test_task_regression.py` — mode-A regression with `FakeLLM`.
- `tests/test_task_concurrency.py` — parallel-mode orchestration:
  pending-work filters, CLI argument validation, serial-vs-parallel path
  selection, per-worker log redirect.
- `tests/test_atomic_writes.py` — `write_json_atomic` round-trip + crash
  safety.
- `tests/test_task_judge.py` — judge prompt / role.
- `tests/test_build_sequences.py` — reliability filter, skip-if-present,
  separator-style tolerance.

## Failure-recovery semantics

- **Solver can't produce a tool_call**: append a nudge, retry up to
  `--max-solver-turns`.
- **Parameter check FAILs**: append the param-check error as a `role:tool`
  message and retry within the same solver loop.
- **`max-solver-turns` hit**: TaskEvolver is called again to propose a new
  task; the unsuccessful attempt is recorded so the evolver can avoid the
  same trap.
- **`max-retries` hit for a tool**: stop the whole sequence, save what's
  generated so far.
