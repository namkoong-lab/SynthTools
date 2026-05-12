# Stage 2 — env_audit

Tool simulation and validation. Every generated tool is exercised with a
suite of synthetic test calls covering schema failures, semantic-rule
violations, and happy paths; the simulator's responses are then judged
against the tool specification, and tools whose responses cannot be
reliably emulated are filtered out.

This is the gate between environment generation (stage 1) and task
construction (stages 3–5).

## Inputs

A directory of `env_spec` JSONs from stage 1.

CLI:
```bash
python -m env_audit.run \
    --env-specs-dir <env_specs_dir> \
    --dataset-path  <tools_dataset.jsonl> \
    --eval-logs-dir <tool_eval_logs_dir> \
    --model GPT-OSS-120B \
    [--fields "Aerospace and Defense,Healthcare"]
    [--ids "spec_id.ToolName,..."]
```

## Outputs

1. `tools_dataset.jsonl` — one row per tool with the full schema and the
   computed `reliability` score.
2. `<eval_logs_dir>/<spec_id>.<ToolName>.json` — per-tool forensic log:
   the world-state metadata, every test call, every parameter-check + simulator
   response, the judge verdict per call.
3. An `audit` block folded into each owning `env_spec` JSON, summarising
   `n_tools`, `mean_reliability`, and `usage_total` for that spec.

## Pipeline (7 phases)

Phases 2–6 are LLM-heavy and batched (every eligible item per phase goes to
vLLM in one call). Crash-recovery is per-tool via `phase_completed`
checkpoints.

| # | Phase         | Role / template                              | Batched? |
|---|---------------|----------------------------------------------|----------|
| 1 | Build         | walk env_specs, append rows to `tools_dataset.jsonl`, seed eval logs | — |
| 2 | Metadata      | `EnvironmentGenerator.generate_metadata`     | yes      |
| 3 | Test calls    | `tool_simulator/test_calls.yml`              | yes      |
| 4 | Param check   | `ToolSimulator.parameter_check`              | yes      |
| 5 | Simulate      | `ToolSimulator.simulate_raw` (only PASSed pairs) | yes  |
| 6 | Judge         | `JudgeSimulator.judge(... failure_mode=...)` | yes      |
| 7 | Aggregate     | compute `reliability`, write per-tool log + dataset row + audit block | — |

### Test-call modes

The test-call generator produces 5–10 calls per tool, labelled by mode:

- **fm1 (schema/type/range violation)** — missing required param, wrong type,
  bad enum, etc. Expected sim behaviour: 400.
- **fm2 (semantic-rule violation)** — schema-valid call that violates a
  documented cross-field or quota rule. Expected sim behaviour: 400 after
  consulting metadata.
- **fm3 (happy path)** — nominal valid call. Expected sim behaviour: 200
  with output matching `output_details`.
- **fm4 (happy path with metadata-pinned output)** — like fm3 but the
  expected response is fully determined by the test-call generator's
  metadata, so correctness can be verified programmatically.

### Judge verdict

Per `(tool, test_call)` pair the judge emits one of:

- `correct` — sim matched the expected behaviour for the labelled mode.
- `should_success_got_error` — fm3/fm4 happy path returned 4xx.
- `should_error_got_success` — fm1/fm2 violation returned 200.
- `inappropriate_response` — right status code, wrong-shaped payload.

Only `correct` counts as a pass; `reliability = n_correct / n_total`.

## Files

- `run.py` — CLI entry point.
- `generate.py` — `build_dataset`, `evaluate_tools`, `audit_tools`.
- `utils.py` — `now_iso`, `model_config_for`, `refresh_env_spec_audit`.
- `embed_tools.py` — optional embedding-based diversity check (paper §2).

## Tests

- `tests/test_env_audit.py`, `test_env_audit_modes.py` — end-to-end against
  `FakeLLM`.
- `tests/test_embed_tools.py` — embedding utilities.

## Notes

- Idempotent on re-run: tools at `phase_completed == 7` are skipped.
- Filters: `--fields` scopes both build and evaluate; `--ids` scopes only
  evaluate. Pass both for an intersection.
- Calibration: simulator and judge prompts were tuned manually against 200
  inspected responses each (94% / 97% accuracy on the held-out sample).
