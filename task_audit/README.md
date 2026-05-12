# Stage 4 — task_audit

Merges each task's successful turns into one cohesive natural-language
description suitable as a single user request. The output is written back
into the task JSON as a top-level `summary` field.

This is the stage that turns the per-step micro-tasks produced by
`task_generation/` into the unified `summary` column of `tasks.parquet`.

## Inputs

A directory of task JSONs from stage 3 (`{spec_id}_{seq_key}.json`).

CLI:
```bash
python -m task_audit.summarize \
    --tasks-dir <traj_dir> \
    --model GPT-OSS-120B \
    [--task <one_task.json>  |  --field "Aerospace and Defense" --env-specs-dir <env_specs_dir>] \
    [--no-debug]
```

If neither `--task` nor `--field` is passed, the stage processes every
`*.json` under `--tasks-dir` (excluding `.debug.json` siblings).

## Outputs

Written **in-place** into each task JSON as a top-level `summary` block:

```
"summary": {
  "generated_at":  "...",
  "model":         "GPT-OSS-120B",
  "model_config":  {temperature, top_p, max_tokens},
  "n_subtasks":    <number of successful turns merged>,
  "prompt":        "<full TaskSummarizer prompt>",
  "response":      "<raw LLM response>",
  "parsed":        {"task_summarized": "<merged task description>"},
  "usage":         {prompt_tokens, completion_tokens}
}
```

A corresponding `TaskSummarizer` event is appended to the sibling
`.debug.json` log when one exists and `--no-debug` is not set.

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

| # | Step                                                              | LLM? | Batched? |
|---|-------------------------------------------------------------------|------|----------|
| 1 | Walk tasks, apply skip-if-present, extract successful turns      | no   | —        |
| 2 | Single batched `TaskSummarizer` call across every task needing work | yes | yes    |
| 3 | Per task: write back `summary` block + append event to `.debug.json` | no | —      |

Tasks whose `summary` field is already populated are skipped, so re-running
the stage is a near no-op.

## Files

- `summarize.py` — `summarize_tasks` + `main` CLI.
- `__init__.py` — empty package marker.
- Role: `../roles/task_summarizer.py` and prompt
  `../prompt_templates/task_summarizer/task_summarizer_template.yml`.

## Tests

- `tests/test_task_audit_summarize.py` — 10 unit tests:
  skip-if-present / idempotency, single + batched, empty-task skip,
  debug-log append vs no-debug, field-mode filtering, last-attempt-per-`tool_idx`
  dedup, `explanation` stripping.
