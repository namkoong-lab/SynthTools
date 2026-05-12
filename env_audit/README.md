# env_audit

**Stage 2 of the synthtools_env pipeline.** Takes the scenario JSONs produced by `env_generation/`, flattens them into a single tools dataset, and scores each tool's **reliability** — the fraction of synthetic test calls whose simulator response a judge finds correct.

Stage position in the pipeline:
```
env_generation  →  env_audit  →  traj_generation.build_sequences  →  traj_generation.run
                     (this)         (filters by reliability)
```

The stage has two internal halves that share one CLI:
- **build** (no LLM): walks env_specs, appends new tool rows to `tools_dataset.jsonl`, seeds per-tool eval logs.
- **evaluate** (LLM-heavy, batched): runs phases 2–7 on un-evaluated tools, writes per-tool eval logs, updates the dataset, and **folds an `audit` block into each owning env_spec** so that env_specs become the single per-environment aggregate (generation cost + tools + sequences + audit results).

You always invoke them together via `python -m env_audit.run`. The split exists so each half is independently testable and so filters (`--ids`, `--fields`) can scope the LLM-heavy half precisely.

---

## Input

A directory of env_generation scenario JSONs (`{spec_id}.json`, e.g. `aerospace_and_defense_spec_000.json`).

CLI:
```bash
python -m env_audit.run \
  --env-specs-dir tool_content/env_specs \
  --dataset-path tool_content/tools_dataset.jsonl \
  --eval-logs-dir tool_content/tool_eval_logs \
  --model GPT-OSS-120B \
  [--fields "Aerospace and Defense,Healthcare"]   # filter for build AND evaluate
  [--ids "id1,id2,id3"]                            # filter for evaluate only
```

Filter semantics:
- `--fields`: subsets which env_specs `build_dataset` reads (skips specs whose `field` isn't listed) AND which tools `evaluate_tools` considers. Both halves get the filter.
- `--ids`: subsets `evaluate_tools` only — exact-match on `tools_dataset.jsonl` row id (`{spec_id}.{ToolName}`). `build_dataset` ignores `--ids`.
- When both are passed: **intersection** in the evaluate phase (a tool must match BOTH).
- Neither passed: process everything not yet at `phase_completed == 7`.

Idempotent: a re-run over the same inputs is a near-no-op — already-evaluated tools are skipped, the audit-block sweep refreshes env_specs from current eval-log state.

---

## Internal split

| Function | Where | Purpose |
|---|---|---|
| `build_dataset(env_specs_dir, dataset_path, eval_logs_dir, llm, fields=None)` | [generate.py](generate.py) | Phase 1 only. No LLM. Walks env_specs, diffs against the existing dataset, appends new rows, seeds new eval logs at `phase_completed=1`. |
| `evaluate_tools(dataset_path, env_specs_dir, eval_logs_dir, llm, ids=None, fields=None)` | [generate.py](generate.py) | Phases 2–7. Loads pending eval logs, applies filters, runs the batched LLM phases, then sweeps audit blocks across all env_specs to keep them in sync with eval_logs. |
| `audit_tools(...)` | [generate.py](generate.py) | Thin wrapper: build then evaluate. What `run.py` actually invokes. |

---

## Pipeline (7 phases)

Phases 2–6 involve LLM calls and are **batched** via [`utils.batch_call`](../utils.py) — each phase processes every eligible tool (or tool × test_call pair) in a single vLLM batch.

| # | Phase | Role / template | Granularity | Batched? |
|---|---|---|---|---|
| 1 | **Build** (`build_dataset`) | *(no LLM)* — walk `env-specs-dir`, flatten to rows | global | — |
| 2 | **Metadata** | `EnvironmentGenerator.generate_metadata` → [env_generator/metadata.yml](../prompt_templates/env_generator/metadata.yml) | per tool (N) | yes |
| 3 | **Test-call generation** | [tool_simulator/test_calls.yml](../prompt_templates/tool_simulator/test_calls.yml) | per tool (N) | yes |
| 4 | **Parameter check** | `ToolSimulator.parameter_check` → [parameter_check.yml](../prompt_templates/tool_simulator/parameter_check.yml) | per (tool, test_call) (N × K) | yes |
| 5 | **Simulate** | `ToolSimulator.simulate_raw` → [tool_simulator_template_metadata.yml](../prompt_templates/tool_simulator/tool_simulator_template_metadata.yml) | only pairs where param_check PASSED | yes |
| 6 | **Judge** | `JudgeSimulator.judge(..., failure_mode=...)` → [judge_template.yml](../prompt_templates/judge_simulator/judge_template.yml) | per (tool, test_call) (N × K) | yes |
| 7 | **Aggregate + write** | *(no LLM)* — compute `reliability` + per-mode breakdown, write per-tool log + dataset row, refresh owning env_specs' `audit` block | per tool, then per affected env_spec | — |
| 7+ | **Sweep** (`_sweep_refresh_all_audit_blocks`) | *(no LLM)* — at end of evaluate, re-walk every eval_log and refresh the audit block of every spec that has any | per env_spec | — |

### Phase 1 — Build

- Walk `env-specs-dir` for `*_spec_*.json`.
- For each tool in each scenario: derive `id = "{spec_id}.{ToolName}"` (e.g. `aerospace_and_defense_spec_000.MissionParser`).
- Apply `--fields` filter at the spec-reading step.
- Diff against the existing `tools_dataset.jsonl`. **Tools already in the dataset are preserved unchanged** — re-runs never clobber prior reliability scores.
- For each new tool: append a row to `tools_dataset.jsonl` with `reliability: null`, and write a fresh `tool_eval_logs/{id}.json` with `phase_completed: 1`.

### Phase 2 — Metadata

Per tool, the LLM generates a "fake world state" JSON (lists of fake records, users, products, etc.) that the simulator can anchor its responses against in phase 5. Without this, the simulator would invent different state on every test call and break fm2 judgments.

### Phase 3 — Test-call generation

The prompt asks for **5–10 tool-call JSONs** labeled by **failure mode**:
- **fm1 (schema/type/range violation)**: missing required param, wrong type, out-of-range value, misspelled name, etc. Expected simulator behaviour: 400.
- **fm2 (semantic-rule violation)**: schema-valid call that violates a documented cross-field or quota rule. Expected simulator behaviour: 400 after consulting metadata.
- **fm3 (happy path)**: a nominal correct call. Expected simulator behaviour: 200 with a valid output matching `output_details`.

Each test call carries `{failure_mode, parameters, tool_call_message}`. The orchestrator's `_classify_failure_mode` heuristic maps the free-text label to one of `fm1 / fm2 / fm3` so we can aggregate per-mode scores later.

### Phase 4 — Parameter check

Per (tool, test_call), the simulator's param-check pass (`"status": "PASS" | "FAIL"`). Fast, cheap, and it determines whether phase 5 runs for this pair.

### Phase 5 — Simulate

Only pairs that PASSed param-check are sent to the simulator's full response generation, which uses the phase-2 metadata to ground the output. Pairs that FAILed param-check skip this phase — their "response" going into the judge is the param-check error.

### Phase 6 — Judge

For every (tool, test_call), the judge sees:
- `tool_details`
- `tool_call_message`
- `response` (either simulator output from phase 5, or param-check error from phase 4)
- `meta_data` (the phase-2 metadata)
- **`failure_mode`** — **the key input** that tells the judge what the test was probing. fm1/fm2 should be errors; fm3 should succeed. Without this label the judge has to guess, and we get fuzzier verdicts. SynthTools omitted this; we pass it.

Judgment is one of: `"correct" | "should_success_got_error" | "should_error_got_success" | "inappropriate_response"`. Only `"correct"` counts as a pass.

### Phase 7 — Aggregate + audit-block update

Per tool reaching phase 7:
- `reliability = n_correct / n_total`  (or `null` if `n_total == 0`)
- `reliability_per_mode = {fm1: 3/4, fm2: 1/2, fm3: 2/2}` or equivalent

Writes to:
1. `tool_eval_logs/{id}.json` — full per-test-call detail
2. `tools_dataset.jsonl` — summary row gets `reliability`, `reliability_per_mode`, `eval_version`, `evaluated_at`, `model`
3. Each touched env_spec's `audit` block — recomputed from all sibling eval_logs via `utils.refresh_env_spec_audit`. Bumps `schema_version` from `env_spec.v1` → `env_spec.v2`.

### Phase 7+ — Sweep

After phase 7, `_sweep_refresh_all_audit_blocks` walks every eval_log and refreshes the audit block of every spec_id it sees. This is **always executed at the end of `evaluate_tools`**, even when the pending set is empty (e.g. on a re-run with everything already at phase 7). It guarantees:

- **Backfill**: env_specs that were audited before the audit-block writer existed get their audit block on the next run.
- **Idempotency**: running with no work to do still keeps env_specs in sync with current eval-log state.
- **Recovery from manual edits**: if you delete an eval_log to force re-evaluation, the next run's sweep will reflect the absence in `audit.aggregate.n_tools_pending`.

The sweep does NO LLM calls and is O(eval_logs).

---

## Crash recovery: per-tool phase checkpoints

Every tool's eval log carries `phase_completed: int`. On each run, evaluate:

1. Scans `eval-logs-dir` for tools with `phase_completed < 7`.
2. Applies `--ids` and `--fields` filters (intersection).
3. For each phase `p ∈ {2..6}`: processes only items where `phase_completed < p`.
4. After each phase completes successfully, writes the updated eval JSON and bumps `phase_completed`.
5. After phase 7, every touched env_spec's `audit` block is recomputed; then the sweep refreshes all the rest.

Consequence: a crash during phase 5 costs at most that batch. The next run resumes from phase 5 for the affected tools. No flags or cleanup needed.

To **force re-evaluation** of a specific tool: delete its `tool_eval_logs/{id}.json` and its row in `tools_dataset.jsonl`. The next run will re-add and re-score it; the sweep will update its env_spec.

To **wipe all evaluation but keep the env_specs**: delete `tool_eval_logs/` and `tools_dataset.jsonl`. The env_specs are inputs from stage 1; they stay.

---

## Token + provenance recording

Every LLM call's usage is accumulated into the per-tool eval log's `usage` field via `_bump_usage` after each phase (2 → 3 → 4 → 5 → 6). The per-tool log carries:

- `usage`: `{prompt_tokens, completion_tokens, total_tokens}` — **cumulative across all phases for that tool**. Re-runs add only the tokens from newly-run phases (idempotent because `phase_completed` guards skip completed phases).
- `metadata` (phase 2), `test_calls` (phase 3), `results[*].{param_check, simulator_response, judgment}` (phases 4-6): full structured content from every LLM call for that tool.

Each env_spec's `audit.tools[i].usage` mirrors its eval log's `usage` at sweep time. The `audit.aggregate.usage_total` sums over all tools in that spec.

**Re-run safety**:
- Per-tool eval log writes are full overwrites, but because `phase_completed` guards re-execution, **you never lose tokens**: a phase that already ran is skipped, so its `usage` entry stays untouched. Atomic writes via `.tmp` + rename protect against mid-write crashes.
- The audit block in each env_spec is fully **regenerated** from eval logs on every sweep (it's derivable state). This is safe because:
  - **Nothing else in the env_spec is modified**: `sequences`, `generation_log` (including any `sequences_build` entry), `tools`, `model_config`, and all other fields from env_generation survive verbatim. See [utils.py:refresh_env_spec_audit](utils.py).
  - The audit block is idempotent: same eval-log state → same audit block content.
- `tools_dataset.jsonl` rows are **preserved on re-build**: Phase 1 (`build_dataset`) diffs discovered tools against existing IDs and only appends new ones. Existing reliability scores are never clobbered by re-running build.
- `manifest.jsonl` from env_generation is not touched by env_audit at all.

**What IS mutated across runs**:
- Each env_spec's `audit` block (regenerated from eval_logs every sweep — lossless and derivable)
- Each env_spec's `schema_version` bumps v1 → v2 on first audit (one-way)
- Each tool's eval_log `phase_completed` advances monotonically (0 → 7)
- `tools_dataset.jsonl` rows get their `reliability`, `reliability_per_mode`, `evaluated_at`, `model` filled in after phase 7

---

## Output

### `tools_dataset.jsonl` row

```json
{
  "schema_version": "tools_dataset.v1",
  "id": "aerospace_and_defense_spec_000.MissionParser",
  "field": "Aerospace and Defense",
  "subfield": "...",
  "task": "...",
  "tool_name": "MissionParser",
  "tool": { ...full schema (OpenAI fn-calling + simulator extras)... },
  "reliability": 0.875,
  "reliability_per_mode": {"fm1": 1.0, "fm2": 0.67, "fm3": 1.0},
  "eval_version": "env_audit.v1",
  "evaluated_at": "2026-04-19T...Z",
  "model": "GPT-OSS-120B"
}
```

Newly-seeded rows (before phase 2 runs) carry `reliability: null`, `reliability_per_mode: null`, `evaluated_at: null`, `model: null`. JSON has no NaN, so we use `null`.

### `tool_eval_logs/{tool_id}.json` — one per tool, full forensic detail

```json
{
  "schema_version": "env_audit_log.v1",
  "tool_id": "aerospace_and_defense_spec_000.MissionParser",
  "spec_id": "aerospace_and_defense_spec_000",
  "field": "Aerospace and Defense",
  "subfield": "...",
  "task": "...",
  "tool": { ...full schema... },
  "generated_at": "...",
  "model": "GPT-OSS-120B",
  "model_config": {"temperature": 0.2, "top_p": 0.95, "max_tokens": 16384},
  "phase_completed": 7,
  "metadata": { ...phase-2 fake world state... },
  "test_calls": [
    {"idx": 0, "failure_mode_group": "fm1", "failure_mode": "Missing required X",
     "parameters": {}, "tool_call_message": "MissionParser()"}
  ],
  "results": [
    {"idx": 0, "param_check": {"status": "FAIL", "error_message": "..."},
     "param_check_passed": false, "simulator_response": null,
     "judgment": {"judgment": "correct", "confidence": 0.9, "reasoning": "..."},
     "correct": true}
  ],
  "reliability": 0.875,
  "reliability_per_mode": {"fm1": 1.0, "fm2": 0.67, "fm3": 1.0},
  "usage": {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}
}
```

### `env_specs/{spec_id}.json` — folded `audit` block (env_spec.v2)

`evaluate_tools` mutates the env_specs *in place*. All v1 fields are preserved verbatim; one new top-level key is added:

```json
{
  "schema_version": "env_spec.v2",

  // ... all env_spec.v1 fields unchanged: spec_id, field, subfield, task,
  //     generated_at, model, model_config, tools, sequences, generation_log,
  //     usage, field_elapsed_s

  "audit": {
    "schema_version": "audit.v1",
    "last_evaluated_at": "2026-04-20T...Z",
    "model": "GPT-OSS-120B",
    "model_config": {"temperature": 0.2, "top_p": 0.95, "max_tokens": 16384},
    "tools": [
      {
        "id": "aerospace_and_defense_spec_000.MissionParser",
        "tool_name": "MissionParser",
        "reliability": 0.875,
        "reliability_per_mode": {"fm1": 1.0, "fm2": 0.67, "fm3": 1.0},
        "n_test_calls": 8,
        "usage": {"prompt_tokens": 45000, "completion_tokens": 12000, "total_tokens": 57000},
        "evaluated_at": "2026-04-19T19:42Z"
      }
    ],
    "aggregate": {
      "n_tools_total": 12,
      "n_tools_evaluated": 12,
      "n_tools_pending": 0,
      "mean_reliability": 0.79,
      "usage_total": {"prompt_tokens": 540000, "completion_tokens": 144000, "total_tokens": 684000}
    }
  }
}
```

`audit.tools` carries one entry per evaluated tool. Tools whose eval_log is at `phase_completed < 7` are counted in `aggregate.n_tools_pending` but not listed (forensic detail still lives in their eval_log file).

---

## Schema versions

| Version | Where | Owner stage |
|---|---|---|
| `tools_dataset.v1` | each row of `tools_dataset.jsonl` | env_audit |
| `env_audit_log.v1` | each per-tool eval log | env_audit |
| `audit.v1` | the `audit` block inside an env_spec | env_audit |
| `env_spec.v2` | env_spec after first evaluate-touch | env_audit (env_generation writes v1) |

---

## Module layout

```
env_audit/
├── __init__.py
├── generate.py       — build_dataset() + evaluate_tools() + audit_tools() + phase 1-7 helpers
├── run.py            — argparse CLI with --fields / --ids
├── utils.py          — env_audit-local helpers (env_spec I/O, audit refresh, timestamps, JSONL I/O)
└── README.md         — this file
```

### `env_audit/utils.py` — what's in there

Local helpers used only by env_audit (cross-stage helpers stay in top-level [`utils.py`](../utils.py)):

```python
now_iso() -> str                                                    # ISO8601 UTC, "Z" suffix
model_config_for(llm) -> dict                                        # {temperature, top_p, max_tokens}

load_jsonl(path) -> list[dict]
write_jsonl(path, rows) -> None
eval_log_path(eval_logs_dir, tool_id) -> Path
load_eval_log(path) -> dict | None
save_eval_log(path, payload) -> None

read_env_spec(env_specs_dir, spec_id) -> dict
write_env_spec(env_specs_dir, spec) -> Path                          # atomic via .tmp rename
refresh_env_spec_audit(env_specs_dir, spec_id, eval_logs_dir,
                       llm_model, llm_model_config) -> None          # rewrites the whole audit block
```

`refresh_env_spec_audit` is the function that makes the audit-block update **idempotent**: it reads every `tool_eval_logs/{spec_id}.*.json` belonging to that spec and recomputes both `audit.tools[]` and `audit.aggregate` from scratch. Tools at `phase_completed < 7` go into `n_tools_pending`; tools at `phase_completed == 7` get a full per-tool entry.

---

## Role / template additions made for this stage

Backwards-compatible extras added to shared roles (used by env_audit but kept additive so other stages aren't disturbed):

- `EnvironmentGenerator.generate_metadata(tool_data)` — uses the `env_generator/metadata.yml` template (new `"metadata"` key in the role's prompt map)
- `ToolSimulator.parameter_check(...)` / `.simulate_raw(...)` — exposed the previously-private halves as public methods so the orchestrator can batch them separately. `ToolSimulator.simulate(...)` still works as before for consumers (e.g. `traj_generation`).
- `JudgeSimulator.judge(..., failure_mode=None)` — new kwarg; the judge template now has a `{failure_mode}` placeholder with short guidance. Default `"none"` keeps existing callers backward-compatible.

---

## Tests

[../tests/test_env_audit.py](../tests/test_env_audit.py) — currently 14 tests, all using `FakeLLM`:

- **Audit-block writes**: env_spec gets `schema_version: env_spec.v2` + populated `audit` block; idempotent on re-run; sweep backfills audit blocks even when no new evaluation happens
- **Filter behaviour**: `--fields` filter on build, `--ids` filter on evaluate, `--fields` filter on evaluate, intersection when both filters supplied
- **Pipeline correctness**: full happy path, resume from a mid-pipeline checkpoint, judge receives the failure_mode label, per-mode reliability breakdown, param_check failures skip simulation
- **Role additions**: `generate_metadata` renders cleanly, `judge(failure_mode=…)` substitutes correctly

[../tests/test_prompts_format.py](../tests/test_prompts_format.py) auto-discovers the new `{failure_mode}` placeholder in the judge template.

Run with `pytest tests/`.

---

## Gotchas

- **`_classify_failure_mode` is a heuristic**: it maps the free-text `failure_mode` label to one of `fm1 / fm2 / fm3` using keyword matching. A model emitting an unfamiliar label defaults to `fm3`. Inspect the heuristic in [generate.py](generate.py) if the per-mode breakdown looks off — the keyword sets for fm1/fm2 are intentionally narrow.
- **Metadata bloat**: phase 2 can produce large JSON payloads (the prompt asks for "complex" metadata that supports many tasks). If eval logs balloon past 50 KB each, consider tightening the metadata template.
- **Tool name → ID coupling**: tool IDs are constructed from `{spec_id}.{tool_name}` verbatim. If env_generation ever emits a tool with a space or hyphen despite the PascalCase rule, the ID will contain that separator and downstream `traj_generation` may have trouble. Current env_generation prompt enforces PascalCase, but worth a quick sanity scan if you see weird IDs.
- **`field_elapsed_s` is field-level (in env_specs), `elapsed_s` (in run summaries) is wall-clock for that invocation**: don't compare across runs with different sized batches without thinking about cold-start model load.
- **The sweep is always run at the end of evaluate**, even when there's no pending work. This keeps env_specs in sync but means a `--ids "single_id"` invocation still touches every env_spec on disk for re-write. The writes are atomic (`.tmp` + rename), idempotent (same content if no change), and no-LLM, so this is cheap; just be aware mtimes will tick.
