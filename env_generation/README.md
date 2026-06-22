# Stage 1 — env_generation

Top-down generation of tool environments. Given a free-form list of fields
(e.g. `"Investment Banking"`, `"Healthcare"`), this stage hierarchically
expands each field into subdomains, task families, and concrete tools whose
interfaces encode domain-specific constraints and interactions.

```
field → subfield → task family → concrete tools
```

## Inputs

A list of field names (free-form English).

CLI:
```bash
python -m env_generation.run \
    --field "Aerospace and Defense" --field "Healthcare" \
    --output-dir <env_specs_dir> \
    --model GPT-OSS-120B \
    [--max-subfields 1] [--max-tasks-per-subfield 1]
```

`--field` is repeatable. Re-running on a field that already has specs is
a no-op until you raise `--max-subfields` or `--max-tasks-per-subfield`;
the stage tops up to `max_subfields * max_tasks_per_subfield` specs per
field (counted by existing `{field_slug}_spec_NNN.json` files).

## Outputs

Three artifacts per run, all under `<env_specs_dir>/`.

### 1. Per-scenario spec: `<spec_id>.json`

One JSON file per `(field, subfield, task)` triple, written atomically
(tmp + `os.replace`) so a killed writer leaves no half-file on disk.

```
{
  "schema_version":  "env_spec.v1",
  "spec_id":         "<field_slug>_spec_<NNN>",
  "field":           "...",
  "subfield":        "...",
  "task":            "<one-line task family description>",
  "generated_at":    "<ISO-8601 UTC>",
  "model":           "<model name>",
  "model_config":    {"temperature": ..., "top_p": ..., "max_tokens": ...},
  "tools": [
    {
      "tool_name":        "PascalCase",
      "tool_description": "...",
      "parameters":       {<param>: {type, required, description}, ...},
      "usage":            "<free-text usage hint>",
      "error_messages":   [...],
      "output_details":   {<field>: {type, description}, ...}
    },
    ...
  ],
  "sequences":        {},   # populated later by task_generation/build_sequences.py
  "generation_log":   [{"phase": "subfields"|"tasks"|"tools", ...}, ...],
  "usage":            {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...},
  "field_elapsed_s":  <float>
}
```

The `sequences` block is initially empty: sequences are built post-audit
so they only chain tools whose simulator behaviour has been verified
(stage 3a).

### 2. Shared per-field log: `<field_slug>_field_gen.json`

One per field, accumulating phase-1 (subfields) and phase-2 (tasks) LLM
prompts and responses across all runs. Scenario JSONs reference it by
filename via `generation_log[*].shared_with_field_log`, so it must not be
overwritten on re-run; append-merge is built into `_save_field_log`.

### 3. Run manifest: `manifest.jsonl`

Append-only JSONL, one line per saved scenario. Used by downstream
auditors as a fast index without scanning all scenario JSONs.

```
{"schema_version": "env_manifest.v1",
 "spec_id": "<field_slug>_spec_<NNN>",
 "field": "...", "subfield": "...", "task": "...",
 "n_tools": <int>, "n_sequences": 0,
 "generated_at": "<ISO-8601 UTC>", "model": "...",
 "usage": {"prompt_tokens": ..., ...}}
```

## Pipeline

Three LLM phases per field. Phases 2 and 3 are batched (every (subfield,
task) item goes through vLLM in one `engine.chat(...)` call).

| # | Phase            | Role method                              | Granularity                    |
|---|------------------|------------------------------------------|--------------------------------|
| 1 | field → subfields| `EnvironmentGenerator.generate_subfields`| 1 call per field               |
| 2 | (field,subfield)→task | `EnvironmentGenerator.generate_tasks` | N_subfields per field          |
| 3 | task → tool schemas | `EnvironmentGenerator.generate_tools` | N_subfields × M_tasks per field |

Targeted prompting at each level controls diversity, parameter complexity,
and I/O behaviour. Each tool emitted by phase 3 carries the fields
`tool_name`, `tool_description`, `parameters`, `usage`, `error_messages`,
`output_details`.

## Files

- `run.py`: CLI entry point.
- `generate.py`: `generate_environments`, `_generate_for_field`,
  `_append_manifest`, `_existing_spec_count` (drives top-up re-run).
- Prompt templates: `../prompt_templates/env_generator/{subfield,task,tool}.yml`.
- Role: `../roles/env_generator.py`.

## Tests

`tests/test_env_generation.py` — end-to-end against a `FakeLLM`.
`tests/test_prompts_format.py` — verifies every template `.format()`s
cleanly with the expected slots.

## Notes

- Tool names are PascalCase Python identifiers (enforced by the prompt) so
  that downstream tool-call strings are parseable as Python AST literals.
- `parameters` types are restricted to `string | integer | float | boolean |
  array | object`. The prompt explicitly forbids `number` because downstream
  benchmark validation maps `number` to Python `int`, which silently breaks
  decimal-valued schemas.
- `object` parameters must declare `properties`; the simulator evaluates each
  tool schema in isolation and has no visibility into upstream outputs.
