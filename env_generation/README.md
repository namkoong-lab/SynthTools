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
    --fields "Aerospace and Defense" "Healthcare" \
    --output-dir <env_specs_dir> \
    --model GPT-OSS-120B
```

## Outputs

For each `(field, subfield, task)` triple, one JSON file at
`<env_specs_dir>/<spec_id>.json` containing:

```
{
  "spec_id":     "<field-slug>_spec_<NNN>",
  "field":       "...",
  "subfield":    "...",
  "task":        "<one-line task family description>",
  "tools": [
    {
      "tool_name":     "PascalCase",
      "tool_description": "...",
      "parameters":    {<param>: {type, required, description}, ...},
      "usage":         "<free-text usage hint>",
      "failure_modes": [...],
      "output_details": {<field>: {type, description}, ...}
    },
    ...
  ],
  "sequences":   {}   # populated later by task_generation/build_sequences.py
}
```

The `sequences` block is initially empty — sequences are built post-audit so
they only chain tools whose simulator behaviour has been verified
(stage 3a).

## Pipeline

Three LLM phases per field. Phases 2 and 3 are batched (every (subfield,
task) item goes through vLLM in one `engine.chat(...)` call).

| # | Phase            | Role method                              | Granularity                    |
|---|------------------|------------------------------------------|--------------------------------|
| 1 | field → subfields| `EnvironmentGenerator.generate_subfields`| 1 call per field               |
| 2 | (field,subfield)→task | `EnvironmentGenerator.generate_tasks` | N_subfields per field          |
| 3 | task → tool schemas | `EnvironmentGenerator.generate_tools` | N_subfields × M_tasks per field |

Targeted prompting at each level controls diversity, parameter complexity,
and I/O behaviour. The output is always a tool tuple
`(name, description, parameters, usage, failure_modes, output_schema)`.

## Files

- `run.py` — CLI entry point.
- `generate.py` — `generate_environments`, `_generate_for_field`.
- Prompt templates: `../prompt_templates/env_generator/{subfield,task,tool,sequences,metadata}.yml`.
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
