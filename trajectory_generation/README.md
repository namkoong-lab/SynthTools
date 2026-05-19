# Stage 5 — trajectory_generation

Given a verifiable task from `task_content.jsonl`, this stage rolls out an
LLM agent against the existing tool simulator, saves the full trajectory,
and runs a trajectory-level judge that compares the rollout against the
ground-truth call sequence and final state.

```
env_generation → env_audit → task_generation.build_sequences →
task_generation.run → task_audit.run → trajectory_generation.run  ← this
```

Stages 1–4 *produce* the dataset; this stage *consumes* it. Every row of
`task_content.jsonl` is a task this orchestrator can roll out.

## Inputs

The JSONL file produced by `task_audit.run` (default path:
`/pscratch/sd/t/tcaste/tool_content/task_content.jsonl`). Override with
`--dataset <path>` for a custom slice. The file is one JSON object per
line with these fields:

| field           | type           | description                                              |
|-----------------|----------------|----------------------------------------------------------|
| `id`            | `string`       | task identifier                                          |
| `field`         | `string`       | application domain                                       |
| `summary`       | `string`       | natural-language task description                        |
| `tools`         | `list[dict]`   | tool schemas the agent has access to                     |
| `gt_tool_calls` | `list[string]` | ground-truth ordered call sequence                       |
| `initial_state` | `dict \| null` | env state at the start                                   |
| `final_state`   | `dict \| null` | env state after the ground-truth solution                |

If `task_audit.run` was invoked with `--resummarize`, the JSONL may contain
multiple rows for the same id (the resummarize flow appends new rows
alongside old ones); the loader deduplicates last-write-wins.

CLI:
```bash
# Single task by id
python -m trajectory_generation.run \
    --task-id aerospace_and_defense_spec_007_seq11 \
    --output-dir /tmp/traj_smoke \
    --model GPT-OSS-120B \
    --max-solver-turns 12

# All tasks in a field, parallel against a vLLM HTTP server
python -m trajectory_generation.run \
    --field "Investment Banking" --limit 50 \
    --server-url http://localhost:8765/v1 --concurrency 4 \
    --output-dir <output>
```

Resume-safe: if `<output-dir>/{task_id}.json` already exists, that task is
skipped.

## Pipeline

For each task, in order:

1. **Load** the task row (`loader.Task` dataclass; tools and state come
   in pre-parsed from the JSONL).
2. **Initialise the rolling chat** with the multi-turn solver system
   prompt (`task_solver_trajectory_template.yml`) and one user message:
   the task `summary` plus the full tool catalogue.
3. **Solver loop** (up to `--max-solver-turns`):
   - LLM emits `{reason, tool_call}`.
   - Empty / unparseable call → user nudge, retry.
   - `<STOP>` → terminate the loop.
   - Otherwise resolve which tool the call refers to (by leading
     `ToolName(...)` identifier); if unknown, append a 400 error and
     continue.
   - `ToolSimulator.simulate(tool_data, call, metadata=task.initial_state)`
     runs the parameter check + simulation; the response is appended to
     the chat as a `role: "tool"` message.
4. **Save** the trajectory JSON to `<output-dir>/{task_id}.json`.
5. **TrajectoryJudge** (unless `--no-judge`): compares the agent's
   actually-executed calls against `task.gt_tool_calls` and the implied
   final state against `task.final_state`, emits a single verdict per
   trajectory.

The trajectory file is written atomically before the judge runs and again
after, so a crash between step 4 and step 5 leaves a usable
(judge-less) artefact.

## Output JSON shape

```
{
  "task_id":            <id>,
  "model":              <model name>,
  "config":             {max_solver_turns, run_judge, stop_reason},
  "tool_ids":           [<tool names from task.tools>],
  "tools":              [<full tool schemas>],
  "turns": [
    {"turn_idx", "tool_name", "tool_call",
     "param_check", "passed", "tool_output"},
    ...
  ],
  "solver_chat":        [<rolling chat minus the system message>],
  "ground_truth": {
    "summary":          <task.summary>,
    "field":            <task.field>,
    "gt_tool_calls":    [...],
    "initial_state":    {...},
    "final_state":      {...}
  },
  "trajectory_judge": {
    "per_call":            [{gt, agent, tool_name_match,
                             argument_match, notes}, ...],
    "tool_call_match_rate": <float>,
    "missing_calls":       [...],
    "extra_calls":         [...],
    "final_state_match":   <bool>,
    "final_state_diff":    "...",
    "trajectory_solved":   <bool>,
    "confidence":          <float>,
    "reasoning":           "..."
  },
  "usage":               {prompt_tokens, completion_tokens, total_tokens},
  "generation_time_s":   <wall clock>
}
```

`stop_reason` is one of `stop_emitted` (agent emitted `<STOP>`),
`max_solver_turns` (turn budget exhausted), or `no_tool_call` (agent failed
to emit a parseable call after the nudge).

## Layout

```
trajectory_generation/
  __init__.py
  loader.py          # JSONL → Task dataclass; load_task / iter_tasks / list_task_ids
  orchestrator.py    # generate_trajectory(task, llm, ...)
  run.py             # CLI
roles/
  trajectory_judge.py    # TrajectoryJudge role (shared Role base)
prompt_templates/
  task_solver/
    task_solver_trajectory_template.yml   # multi-turn solver prompt
  trajectory_judge/
    trajectory_judge_template.yml          # whole-rollout judge prompt
tests/
  test_trajectory_loader.py
  test_trajectory_orchestrator.py
  test_trajectory_judge_prompt.py
```

## Reused components

| Component              | Source                                                  |
|------------------------|---------------------------------------------------------|
| Tool agent             | `roles.task_solver.TaskSolver(llm, mode="trajectory")`  |
| Tool simulator         | `roles.tool_simulator.ToolSimulator`                    |
| Trajectory judge       | `roles.trajectory_judge.TrajectoryJudge`                |
| LLM client             | `llm.LLM`                                               |
| Atomic JSON write etc. | `utils.write_json_atomic`, `RunLog`, `UsageTracker`,    |
|                        | `extract_json_objects`, `to_llm_messages`               |

## Trajectory-judge prompt

`prompt_templates/trajectory_judge/trajectory_judge_template.yml`. Inputs:
`task_summary`, `gt_tool_calls`, `agent_tool_calls`, `initial_state`,
`final_state_gt`. The judge proceeds in four steps:

1. Per-call alignment between `gt_tool_calls` and `agent_tool_calls`
   (tolerant of independent reordering); emits `tool_name_match` and
   `argument_match` per pair.
2. Coverage: flag `missing_calls` and `extra_calls`, compute
   `tool_call_match_rate`.
3. Final-state contrast: reason from `initial_state` plus the agent's
   executed calls about whether the resulting state matches
   `final_state_gt`.
4. Verdict: `trajectory_solved` true iff coverage is complete and final
   state matches.

## Tests

```bash
pytest tests/test_trajectory_loader.py        # JSONL round-trip
pytest tests/test_trajectory_orchestrator.py  # FakeLLM end-to-end
pytest tests/test_trajectory_judge_prompt.py  # prompt-format / role
```
