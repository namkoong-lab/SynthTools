# Stage 4 — task_audit

Merges each task's successful turns into one cohesive natural-language
description suitable as a single user request. The output is written back
into the task JSON as a top-level `summary` field.

This is the stage that turns the per-step micro-tasks produced by
`task_generation/` into the unified `summary` field of `task_content.jsonl`
(the release JSONL consumed by `trajectory_generation/`).

## Inputs

A directory of task JSONs from stage 3 (`{spec_id}_{seq_key}.json`).

CLI:
```bash
python -m task_audit.run \
    --tasks-dir <tasks_dir> \
    --env-specs-dir <env_specs_dir> \
    --model GPT-OSS-120B \
    [--task <one_task.json> | --field "Aerospace and Defense"] \
    [--task-content <task_content.jsonl>] \
    [--resummarize] \
    [--shard I --num-shards N] \
    [--server-url http://host:port/v1] \
    [--audit-only [--audit-report <report.jsonl>]] \
    [--no-debug]
```

`--env-specs-dir` is required (the stage needs each task's owning spec to
populate the `field` and `tools` columns of the release JSONL). If neither
`--task` nor `--field` is passed, the stage processes every `*.json` under
`--tasks-dir` (excluding `.debug.json` siblings).

### Flags

- `--task-content`: shared release JSONL (default:
  `<tasks-dir>/../task_content.jsonl`). Multiple concurrent shards append
  to the same file under an exclusive file lock; no per-shard files, no
  merge step.
- `--resummarize`: ignore existing `summary` blocks and re-run the
  summariser for every in-scope task. Critical for catch-up runs after a
  prompt-version rewrite (the 2026-05 rewrite invalidates prior summaries
  at the prompt level). Default behaviour skips tasks that already have a
  truthy `summary` field.
- `--shard I --num-shards N`: each shard processes
  `target_paths[I::N]` of the deterministically sorted in-scope file
  list. Run N concurrent jobs (e.g. one sbatch task per node) sharing
  the same release JSONL.
- `--server-url`: OpenAI-compatible vLLM HTTP endpoint. When set, all
  LLM calls go over HTTP instead of loading vLLM in-process.
- `--audit-only`: skip the LLM entirely; just run cross-task audit
  checks and write a JSONL report to `--audit-report` (default:
  `<tasks-dir>/../audit_report.jsonl`).
- `--no-debug`: skip appending the `TaskSummarizer` event to sibling
  `.debug.json` logs.

`--model` is restricted to entries of `llm.MODEL_REGISTRY` (see
`cli_args.add_model_arg`).

## Outputs

Two artifacts, in this order.

### 1. In-place `summary` block on each task JSON

```
"summary": {
  "generated_at":  "...",
  "model":         "GPT-OSS-120B",
  "model_config":  {temperature, top_p, max_tokens},
  "n_subtasks":    <number of successful turns merged>,
  "prompt":        "<full TaskSummarizer prompt>",
  "response":      "<raw LLM response>",
  "parsed":        {
      "user_supplied_values":  {<param>: <value>, ...},
      "tool_produced_values":  {<param>: <value>, ...},
      "spillover":             [<list of carry-over notes>],
      "task_summarized":       "<merged single user request>"
  },
  "usage":         {prompt_tokens, completion_tokens}
}
```

The 4-field `parsed` block reflects the 2026-05 prompt rewrite. Catch-up
runs over tasks whose summaries pre-date the rewrite MUST pass
`--resummarize`; do NOT filter to "tasks with null summary" because the
stale rows look fine schema-wise but are not produced by the current
template.

A corresponding `TaskSummarizer` event is appended to the sibling
`.debug.json` log when one exists and `--no-debug` is not set.

### 2. Append row to `task_content.jsonl`

The shared release JSONL grows by one row per in-scope task. This is the
artifact consumed by `trajectory_generation/`:

```
{
  "id":              "<spec_id>_<seq_key>",
  "field":           "<field name from env_spec>",
  "summary":         "<task_summarized string>",
  "tools":           [<full tool schemas, from env_spec>],
  "gt_tool_calls":   [<ground-truth call sequence, one per accepted turn>],
  "initial_state":   {<env_state before first accepted turn>},
  "final_state":     {<env_state after last accepted turn>}
}
```

Multiple concurrent shards (`--shard I --num-shards N`) append to the
same file under an exclusive file lock; consumers dedup by `id` with
last-write-wins semantics.

## Filter (which turns count as "successful")

A turn is eligible for the summarizer iff `turn["env_update"]` is non-null
— i.e. the environment simulator actually advanced the state. Per
`tool_idx`, only the **last** attempt is kept (handles retries cleanly).

This filter works for both `verifiable=True` and `verifiable=False` runs,
because `env_update` is populated only when the orchestrator accepts the
turn as solved.

Before passing tool responses to the summarizer, the simulator's
`explanation` field is stripped from each response — that field carries the
simulator's internal prose, not grounded tool output.

## Pipeline

| # | Step                                                                    | LLM? | Batched? |
|---|-------------------------------------------------------------------------|------|----------|
| 1 | Walk tasks, apply skip-if-present, extract successful turns             | no   | none     |
| 2 | Single batched `TaskSummarizer` call across every task needing work     | yes  | yes      |
| 3 | Per task: write back `summary` block + append event to `.debug.json`    | no   | none     |
| 4 | Per task: extract release-row payload, append to `task_content.jsonl`   | no   | none     |

Tasks whose `summary` field is already populated are skipped (unless
`--resummarize` is passed), so re-running the stage is a near no-op. The
release-row append at step 4 runs whenever step 2 ran for that task.

## Files

- `run.py`: CLI entry point.
- `summarize.py`: `summarize_trajectories` (the main worker called from
  `run.py`) plus `audit_corpus` (the `--audit-only` path) and the
  release-row extraction logic.
- `checks.py`: per-task audit checks invoked by `summarize.py` (release
  row validation, cross-task consistency, etc.).
- `__init__.py`: empty package marker.
- Role: `../roles/task_summarizer.py` and prompt
  `../prompt_templates/task_summarizer/task_summarizer_template.yml`.

## Tests

- `tests/test_task_audit_summarize.py`: unit tests covering
  skip-if-present / idempotency, single + batched, empty-task skip,
  debug-log append vs no-debug, field-mode filtering,
  last-attempt-per-`tool_idx` dedup, `explanation` stripping.
- `tests/test_task_audit_sharding.py`: concurrent-shard correctness.
- `tests/test_task_audit_cli.py`: CLI argparse smoke (every documented
  flag parses; `--env-specs-dir` is required).
- `tests/test_task_audit_readme.py`: doc-presence test; asserts the
  README mentions each load-bearing flag and the release JSONL.
