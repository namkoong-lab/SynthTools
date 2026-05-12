# env_generation

**Stage 1 of the synthtools_env pipeline.** Takes one or more field names and produces, for each `(field, subfield, task)` scenario, a self-contained JSON spec holding:

- **tools**: OpenAI-function-calling-style schemas **plus** simulator-extras (`error_messages`, `usage`, `output_details`)
- **sequences**: empty placeholder `{}`. This stage no longer generates tool orderings — that's done post-audit by [`traj_generation.build_sequences`](../traj_generation/build_sequences.py) so sequences can be built from only the reliable tools.
- **provenance**: full prompts + LLM responses that produced the spec, per-phase token usage, timestamp, model config

Each spec is one JSON file; a sibling "field log" carries the two LLM calls that are shared across every scenario in the same field (so we don't duplicate prompts across spec files).

Stage position in the pipeline:
```
env_generation  →  env_audit  →  traj_generation.build_sequences  →  traj_generation.run
   (this)          (reliability)      (sequences)                       (trajectories)
```

---

## Input

One or more **field names** (free-form English — e.g. `"Aerospace and Defense"`, `"Healthcare"`, `"Financial Trading"`).

CLI:
```bash
python -m env_generation.run \
  --fields "Aerospace and Defense,Healthcare"   # comma-separated list
  --output-dir tool_content/env_specs
  --model GPT-OSS-120B
  [--max-subfields 2]             # how many subfields to explore per field (default 1)
  [--max-tasks-per-subfield 2]    # how many tasks per subfield (default 1)
```

Sequence-generation knobs (`--seqs-per-spec`, `--seq-length`) moved to [`traj_generation.build_sequences`](../traj_generation/build_sequences.py).

---

## Pipeline (3 LLM phases, per field)

Phases 2–3 are batched via [`utils.batch_call`](../utils.py) so all items in a phase go through vLLM in a single `engine.chat(...)` call.

| # | Phase | Role method | Prompt template | Calls | Batched? |
|---|---|---|---|---|---|
| 1 | `field` → subfields list | `EnvironmentGenerator.generate_subfields` | [env_generator/subfield.yml](../prompt_templates/env_generator/subfield.yml) | 1 | no |
| 2 | `(field, subfield)` → task list | `EnvironmentGenerator.generate_tasks` | [env_generator/task.yml](../prompt_templates/env_generator/task.yml) | N_subfields | **yes** |
| 3 | `(field, subfield, task)` → tool schemas | `EnvironmentGenerator.generate_tools` | [env_generator/tool.yml](../prompt_templates/env_generator/tool.yml) | N_subfields × M_tasks | **yes** |

### Per-request token tracking

When `batch_call` dispatches a batch to vLLM, the LLM class populates `llm.last_usage_per_request` with per-prompt `Usage(prompt_tokens, completion_tokens)`. That lets us attribute each scenario's share of phase-3 tokens accurately instead of just reporting the field aggregate.

### Post-processing (pure Python)

- **Subfields**: dedup (order-preserving)
- **Tasks**: dedup per subfield
- **Tools**: dedup by `tool_name` per scenario (first wins)

Sequence post-processing (dedup, consecutive-dup, unknown-name guard) now lives in [`traj_generation.build_sequences`](../traj_generation/build_sequences.py).

### Parse-failure skip

If phase 3 returns **0 tools** for a scenario, that scenario is **not written**. No error-placeholder files. Reruns pick up cleanly because numbering uses `next_artifact_index` (max index + 1).

---

## Token + provenance recording

Every LLM call's `{prompt, response, prompt_tokens, completion_tokens}` is persisted to disk, without overwriting prior runs:

| Call | Written to |
|---|---|
| Phase 1 (subfields) | field log: full prompt+response+usage **and** each scenario's `generation_log` (usage only, with `shared_with_field_log` pointer) |
| Phase 2 (tasks, per subfield) | field log: full prompt+response+usage per subfield **and** each scenario's `generation_log` (usage only, `shared_with_field_log` pointer) |
| Phase 3 (tools, per scenario) | scenario `generation_log` entry: full prompt+response+usage (never shared) |

Each scenario's top-level `usage` equals `p1_usage + p2_usage + p3_usage` for that scenario, so per-scenario cost is always self-contained — **you don't need the field log to reconstruct any scenario's token count**.

**Re-running against the same `--output-dir`**:
- New scenarios get fresh `spec_NNN` indices via `next_artifact_index` (max + 1); prior scenario JSONs are preserved.
- The field log is **merged**, not overwritten: new phase-1/phase-2 entries are appended to the prior `generation_log`; `usage` and `elapsed_s` accumulate across runs. This preserves the prompts that earlier scenarios reference via `shared_with_field_log`.
- `manifest.jsonl` is append-only; re-runs add new rows for new scenarios.
- Writes are atomic (`.tmp` + rename) for the field log, so a mid-write crash won't corrupt the prior file.

Crash recovery: there's no per-field checkpoint, but the *granularity* is per-scenario. If the process dies after writing spec_002 but before spec_003, a re-run produces spec_003 onward under a new max-index. Phase-1/phase-2 prompts from the first run are kept in the merged field log. Note however that a restarted run re-does phase-1/phase-2 with fresh LLM output, so you effectively get "two rounds of seeds" in the same field — not corruption, just richer provenance.

---

## Output (in `--output-dir`)

### Per-scenario `{field_slug}_spec_NNN.json`

```
schema_version:    env_spec.v1
spec_id:           aerospace_and_defense_spec_000
field, subfield, task
generated_at:      ISO8601 UTC
model:             e.g. "GPT-OSS-120B"
model_config:      {temperature, top_p, max_tokens}
tools:             [ ...tool schemas (OpenAI fn-calling + simulator extras)... ]
sequences:         {}    # populated later by traj_generation.build_sequences
generation_log:
  - phase: subfields  —> shared_with_field_log: "<field_gen filename>", usage: {...}
  - phase: tasks      —> shared_with_field_log, subfield_idx, usage
  - phase: tools      —> prompt, response, usage  (inline; scenario-specific)
usage:             aggregate for this scenario {prompt_tokens, completion_tokens, total_tokens}
field_elapsed_s:   wall-clock time for the whole field run (NOT per-scenario)
```

### Per-field `{field_slug}_field_gen.json`

```
schema_version:    env_field_gen.v1
field, model, model_config, generated_at
generation_log:    [phase-1 subfields entry, phase-2 tasks entries (one per subfield)]
                   each carries full prompt + response + parsed + usage
usage, elapsed_s
```

### `manifest.jsonl` (append-only)

One line per saved scenario — the queryable index. `n_sequences` is always 0 at this stage; it stays that way until `build_sequences` is run. (We keep the field for backwards-compatibility with existing readers.)

```json
{"schema_version": "env_manifest.v1", "spec_id": "...", "field": "...", "subfield": "...",
 "task": "...", "n_tools": 12, "n_sequences": 0, "generated_at": "...",
 "model": "...", "usage": {...}}
```

---

## Tool schema shape (what phase 3 emits)

Extends standard OpenAI fn-calling with simulator hooks:

| Field | Purpose |
|---|---|
| `tool_name` | PascalCase Python identifier (no spaces/hyphens — enforced by the prompt) |
| `tool_description` | Natural-language purpose |
| `parameters` | `{param_name: {type, required, description, [items/properties]}}` — types limited to `string, integer, float, boolean, array, object` (never `number`) |
| `error_messages` | List of realistic errors the simulator + judge use as references |
| `usage` | Free-text usage instructions for the simulator |
| `output_details` | Shape of the successful response body — same type vocabulary |

Tool-call strings emitted downstream must be Python-literal-parseable: `ToolName(arg1='str', arg2=1.5, arg3=[1,2], arg4={"k": "v"})`. The tool-generation prompt enforces this.

---

## Gotchas

- **ACEBench `number` → `int`**: downstream ACEBench validation maps `"type": "number"` to Python `int`. If a model emits `number` for a decimal, validation fails silently. The tool-generation prompt now forbids `number` and instructs the model to use `integer` or `float` explicitly.
- **Tool names must be PascalCase Python identifiers**: they appear both in the tool list and as the function-call verb in generated tool-call strings (`ToolName(arg=val)`); anything else breaks `ast.parse`. The tool-generation prompt enforces this, and the sequences prompt (used later by `build_sequences`) uses the same rule.
- **`object` parameters MUST declare `properties`**: even when the object is "passed through" from an upstream tool's output. The simulator evaluates each tool schema in isolation and has no visibility into upstream outputs.
- **`field_elapsed_s` is field-level, not per-scenario**: with batching, all scenarios of phase 3 run concurrently inside one vLLM call; the scenario JSON reports the *field's* wall-clock time, not its own. Use `usage` for per-scenario cost.

---

## Schema versions

| Version | Where |
|---|---|
| `env_spec.v1` | every scenario JSON (bumped to `env_spec.v2` by env_audit when an audit block is added) |
| `env_field_gen.v1` | field log |
| `env_manifest.v1` | each `manifest.jsonl` line |

Bump the version whenever the shape changes. Downstream stages should validate the version they're reading.

---

## Where sequences come from now

Sequences (the ordered lists under `spec["sequences"]`) are populated by [`traj_generation.build_sequences`](../traj_generation/build_sequences.py), **after** `env_audit` has scored each tool's reliability. That stage:

1. Reads the audit block on each env_spec
2. Drops tools whose reliability is below a threshold (configurable; default 0.0) and, by default, drops tools with no audit entry (`--no-eval-only` to include them)
3. Asks the LLM to build sequences from the remaining tool pool
4. Writes the cleaned sequences back into `spec["sequences"]`

This means `env_generation → env_audit → build_sequences` is the intended ordering. Running `env_generation` alone leaves `sequences: {}` in every spec.

---

## Files

- [generate.py](generate.py) — orchestrator (`generate_environments`, `_generate_for_field`)
- [run.py](run.py) — CLI wrapper
- Prompt templates live at `../prompt_templates/env_generator/{subfield,task,tool}.yml`
- [roles/env_generator.py](../roles/env_generator.py) — the `EnvironmentGenerator` class. `generate_sequences` stays on this role for reuse by `build_sequences`; this stage only invokes `generate_{subfields,tasks,tools}`.

## Tests

- [../tests/test_env_generation.py](../tests/test_env_generation.py) — end-to-end tests against `FakeLLM`
- [../tests/test_prompts_format.py](../tests/test_prompts_format.py) — verifies every template `.format()`s cleanly + no fancy quotes

Run with `pytest tests/`.
