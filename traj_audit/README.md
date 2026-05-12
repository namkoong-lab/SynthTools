# traj_audit

**Stage 4 (final) of the synthtools_env pipeline.** Consumes trajectory JSONs produced by `traj_generation.run` and merges each trajectory's successful subtasks into **one cohesive task description** via the `TaskSummarizer` role. Writes the summary back into the same trajectory JSON (as a top-level `summary` field) and appends a corresponding event to the sibling `.debug.json` log.

Stage position:
```
env_generation  →  env_audit  →  traj_generation.build_sequences  →  traj_generation.run  →  traj_audit.summarize
                                                                                                  (this)
```

Why a separate stage:
- The per-turn trajectory is noisy (evolver, solver retries, param-check failures). Downstream training wants **one grounded task description per trajectory** where every tool argument is traceable to the merged text or a previous tool output.
- Summarization is batched across trajectories (one vLLM call per N trajectories), so running it once over a whole directory is cheap.
- It's idempotent: re-running only summarizes trajectories that don't already have a `summary` field.

---

## Input

A directory of trajectory JSONs produced by `traj_generation.run` (`{spec_id}_{seq_key}.json`). Sibling `.debug.json` files are optional — if present, the stage appends a `TaskSummarizer` event to their `events[]`.

CLI:
```bash
python -m traj_audit.summarize \
  --trajectories-dir tool_content/trajectories
  --model GPT-OSS-120B
  (--trajectory PATH  |  --field NAME --env-specs-dir DIR)   # mutually exclusive; omit both = all files
  [--no-debug]                                               # skip .debug.json append
```

Targeting modes (mutually exclusive):
- `--trajectory PATH`: summarize exactly one file.
- `--field NAME` + `--env-specs-dir DIR`: walk `env-specs-dir`, collect spec_ids whose `field == NAME`, then process every trajectory whose filename starts with `{spec_id}_`.
- **Neither flag**: process every `*.json` under `--trajectories-dir` (excluding `.debug.json` siblings).

---

## Pipeline (single LLM phase, batched)

| # | Step | LLM? | Batched? |
|---|---|---|---|
| 1 | Walk trajectories, apply skip-if-present, extract successful turns, build prompts | no | — |
| 2 | Single batched `TaskSummarizer` call across every trajectory needing work | yes | **yes** (one `batch_call`) |
| 3 | Per trajectory: write back `summary` block + append event to `.debug.json` | no | — |

### Turn filter — what counts as "successful"

A turn is eligible for the summarizer iff `turn["env_update"] is not None`. The env-update step runs *only* when the tool simulator produced a parseable response the orchestrator accepted, regardless of `verifiable=True/False` mode. This is the right signal for both:

- **verifiable=True**: `env_update` is populated only when `judge.task_solved == True`.
- **verifiable=False**: `env_update` is populated whenever a tool response came back (no judge).

Per tool_idx, only the **last attempt** is kept (handles retry attempts cleanly).

### Post-processing

Before passing tool responses to the summarizer, the `explanation` key is stripped from each response JSON if present — that field carries the simulator's internal prose and is not useful to the summarizer (matches the legacy `scripts/task_summary.py` behaviour).

---

## Output

### Inline on the trajectory JSON

```json
{
  "task_id": "aerospace_and_defense_spec_000_seq1",
  "model": "GPT-OSS-120B",
  "tools": [...],
  "turns": [...],
  "solver_chat": [...],
  "usage": {...},
  "generation_time_s": 118.4,
  "summary": {                       // ← added by this stage
    "generated_at": "2026-04-20T21:45:00Z",
    "model": "GPT-OSS-120B",
    "model_config": {"temperature": 0.2, "top_p": 0.95, "max_tokens": 16384},
    "n_subtasks": 4,
    "prompt": "...",                 // full TaskSummarizer prompt
    "response": "...",               // raw LLM response
    "parsed": {"task_summarized": "Merged, grounded task description…"},
    "usage": {"prompt_tokens": 6122, "completion_tokens": 1069}
  }
}
```

### Appended event on the `.debug.json` sibling (if it exists)

A new entry is appended to `events[]`:

```json
{
  "timestamp": "2026-04-20T21:45:00+00:00",
  "agent": "TaskSummarizer",
  "action": "summarize",
  "turn_ref": null,
  "prompt": "...",
  "response": "...",
  "parsed": {"task_summarized": "..."},
  "usage": {"prompt_tokens": 6122, "completion_tokens": 1069}
}
```

Both writes are atomic (`.tmp` + rename).

---

## Token + provenance recording

Every LLM call's prompt, response, parsed JSON, and usage land in **two places**:

1. `trajectory["summary"]` on the main JSON (canonical location, always written when a summary is produced)
2. `debug["events"][-1]` on the `.debug.json` log (only if that file already exists and `--no-debug` wasn't passed)

Per-request usage is correctly attributed via `llm.last_usage_per_request` when the LLM is run batched — one `batch_call` of N prompts yields N distinct `usage` dicts, not a split of the aggregate.

Re-run safety:
- A trajectory whose `summary` field is already truthy is **skipped** (no LLM call, no file mutation).
- Running with `--no-debug` leaves debug logs untouched regardless.
- Atomic writes mean a mid-write crash preserves the previous file contents.

Empty trajectories (no successful turns) are skipped silently with a warning. No `summary` block is written.

---

## Crash recovery

There's no per-trajectory checkpoint — trajectories are atomic units. A crash mid-batch:
- Any trajectory already written in the current run keeps its summary.
- Any trajectory whose batch hadn't completed is left untouched.
- Re-run skips everything already done; picks up the rest.

---

## Reused components

- [`roles/task_summarizer.py`](../roles/task_summarizer.py) — `TaskSummarizer` Role. We use `get_prompt("task_summarizer", ...)` to build the prompt directly (bypassing the role's `summarize_tasks()` which fires a single call) so we can batch across trajectories via `utils.batch_call`.
- [`prompt_templates/task_summarizer/task_summarizer_template.yml`](../prompt_templates/task_summarizer/task_summarizer_template.yml) — the merged-task-description prompt.
- [`utils.batch_call`](../utils.py), `extract_json_objects`, `usage_to_dict`, `get_logger`.
- [`env_audit/utils.py`](../env_audit/utils.py): `now_iso`, `model_config_for` — shared timestamp/config helpers used by `build_sequences` too.

The legacy [`scripts/task_summary.py`](../scripts/task_summary.py) is **not used** (YAML-based, one-at-a-time, broken import path). This stage replaces it for JSON trajectories.

---

## Files

- [summarize.py](summarize.py) — `summarize_trajectories()` orchestrator + `main()` CLI
- [__init__.py](__init__.py) — empty package marker

## Tests

[../tests/test_traj_audit_summarize.py](../tests/test_traj_audit_summarize.py) — 10 unit tests against `FakeLLM`:

- Skip-if-present / idempotency
- Single trajectory / batched-across-directory (with per-request usage attribution)
- Empty-trajectory skip (0 successful turns)
- Debug-log append / no-debug-log OK
- Field-mode filtering via env_specs
- Last-attempt-per-tool_idx dedup
- `verifiable=False` trajectories filter correctly on `env_update`
- `explanation` stripped from tool responses before summarization

Run:
```bash
source /pscratch/sd/t/tcaste/envs/burn-gpu/bin/activate
cd /global/homes/t/tcaste/projects/burn-gpu/synthtools_env
pytest tests/test_traj_audit_summarize.py -q
```

---

## Gotchas

- **`.debug.json` discovery is filename-based**: the sibling is `{stem}.debug.json` next to the main file. If you move or rename the main trajectory, the debug log link breaks — the summarize stage silently logs a debug message (not a warning) and skips the append.
- **`--field` needs env_specs_dir**: field is a property of the env_spec, not the trajectory. The stage looks up matching spec_ids by reading env_specs and matching trajectory files by prefix.
- **Batched usage may fall back to even-split**: `utils.batch_call` honours `llm.last_usage_per_request` when present. If the LLM class doesn't populate it, usage per trajectory will be the aggregate split evenly (lossy but never None). The current vLLM adapter in `llm.py` populates it correctly.
- **The summarizer prompt can be large**: each successful turn contributes a task description, a tool call, and a tool response. For trajectories with 10+ tools this can approach the 32k context — consider increasing `max_model_len` if you see truncation.
