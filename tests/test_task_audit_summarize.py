"""Unit tests for task_audit.summarize against a FakeLLM.

Covers:
- Skip-if-present idempotency
- Single-task and batched-across-directory paths
- Debug-log append (when present) and graceful no-op when absent
- Field-mode filtering via env_specs lookup
- Extraction: last-attempt-per-tool-idx dedup, verifiable=False handling,
  `explanation` stripping from tool responses
- Empty-task skip (no successful turns)
"""

import json
from pathlib import Path

import pytest

from task_audit.summarize import summarize_trajectories as _raw_summarize_trajectories


def summarize_trajectories(*, tasks_dir, llm, env_specs_dir=None, task_content_path=None, **kwargs):
    """Test wrapper: supplies the new required env_specs_dir and
    task_content_path defaults so existing call sites keep working
    unchanged. Tests that exercise the release JSONL pass these
    explicitly."""
    if env_specs_dir is None:
        env_specs_dir = tasks_dir
    if task_content_path is None:
        task_content_path = tasks_dir / "task_content.jsonl"
    return _raw_summarize_trajectories(
        tasks_dir=tasks_dir,
        llm=llm,
        env_specs_dir=env_specs_dir,
        task_content_path=task_content_path,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _summary_response(text: str = "Merged task description.") -> str:
    return "```json\n" + json.dumps({"task_summarized": text}) + "\n```"


def _summary_response_v2(
    text: str = "Merged task description.",
    user_supplied_values=None,
    tool_produced_values=None,
    spillover=None,
) -> str:
    """The current production prompt emits a 4-field JSON: the three
    show-your-work fields plus task_summarized. The pipeline must
    handle this shape without losing the summary string and must
    preserve the extra fields in the task's parsed-summary block."""
    payload = {
        "user_supplied_values": user_supplied_values or [],
        "tool_produced_values": tool_produced_values or [],
        "spillover": spillover or [],
        "task_summarized": text,
    }
    return "```json\n" + json.dumps(payload) + "\n```"


def _all_content(prompt) -> str:
    """Return concatenated content across all messages in a chat-format
    prompt, or the prompt itself if it's a plain string (legacy shape)."""
    if isinstance(prompt, list):
        return "\n".join(
            (m.get("content") or "")
            for m in prompt
            if isinstance(m, dict)
        )
    return prompt or ""


def _turn(
    tool_idx: int,
    attempt: int = 0,
    task_description: str = "Do the thing",
    tool_call: str = "ToolA(x=1)",
    tool_response: dict = None,
    env_update=True,
    include_explanation: bool = False,
):
    resp = tool_response if tool_response is not None else {"status_code": 200, "response": {"result": "ok"}}
    if include_explanation:
        resp = {**resp, "explanation": "simulator internal notes — should be stripped"}
    chat = [
        {"role": "user", "content": task_description},
        {"role": "assistant", "content": json.dumps({"reason": "r", "tool_call": tool_call})},
        {"role": "tool", "content": json.dumps(resp)},
    ]
    return {
        "tool_idx": tool_idx,
        "attempt": attempt,
        "tool_id": f"mock_spec_1.Tool{tool_idx}",
        "env_state_before": None,
        "task": {
            "task_description": task_description,
            "expected_tool_call": tool_call,
            "env_metadata": None,
            "edited_metadata": None,
            "depends_on_previous": False,
        },
        "env_state_after_task": None,
        "chat": chat,
        "judge": None,
        "env_update": {"full_metadata": {"x": 1}} if env_update else None,
        "env_state_after": None,
    }


def _make_trajectory(task_id: str = "mock_spec_1_seq1", turns=None):
    if turns is None:
        turns = [_turn(0), _turn(1, task_description="Second sub-task", tool_call="ToolB(y=2)")]
    return {
        "task_id": task_id,
        "model": "fake-model",
        "config": {"max_solver_turns": 5, "max_retries": 5, "verifiable": False},
        "tool_ids": [t["tool_id"] for t in turns],
        "tools": [],
        "turns": turns,
        "solver_chat": [],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "generation_time_s": 0.0,
    }


def _write_trajectory(dir_: Path, task: dict) -> Path:
    p = dir_ / f"{task['task_id']}.json"
    p.write_text(json.dumps(task, indent=2))
    return p


def _write_debug_log(dir_: Path, task_id: str, n_events: int = 2) -> Path:
    path = dir_ / f"{task_id}.debug.json"
    payload = {
        "task_id": task_id,
        "events": [
            {"timestamp": "2026-04-20T00:00:00Z", "agent": "TaskSolver", "action": "solve",
             "turn_ref": {"tool_idx": 0, "attempt": 0}, "prompt": "...", "response": "...",
             "parsed": None, "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
            for _ in range(n_events)
        ],
    }
    path.write_text(json.dumps(payload))
    return path


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# 1 — Skip-if-present
# ---------------------------------------------------------------------------

def test_summarize_skips_if_present(fake_llm, tmp_output):
    task = _make_trajectory()
    task["summary"] = {"parsed": {"task_summarized": "already done"}}
    path = _write_trajectory(tmp_output, task)

    summarize_trajectories(
        tasks_dir=tmp_output,
        llm=fake_llm,
        task_path=path,
    )

    assert fake_llm.calls == []
    assert _read(path)["summary"]["parsed"] == {"task_summarized": "already done"}


# ---------------------------------------------------------------------------
# 2 — Single task, full summary block
# ---------------------------------------------------------------------------

def test_summarize_single_trajectory(fake_llm, tmp_output):
    task = _make_trajectory()
    path = _write_trajectory(tmp_output, task)

    fake_llm.queue(_summary_response("Merged desc: do A then B."))

    summarize_trajectories(
        tasks_dir=tmp_output,
        llm=fake_llm,
        task_path=path,
    )

    saved = _read(path)
    summary = saved["summary"]
    assert summary["n_subtasks"] == 2
    assert summary["parsed"] == {"task_summarized": "Merged desc: do A then B."}
    assert summary["usage"]["prompt_tokens"] == 10
    assert summary["usage"]["completion_tokens"] == 20
    assert summary["model"] == "fake-model"
    assert summary["generated_at"].endswith("Z")
    assert "prompt" in summary and "response" in summary


# ---------------------------------------------------------------------------
# 3 — Batched across directory
# ---------------------------------------------------------------------------

def test_summarize_batched_across_directory(fake_llm, tmp_output):
    for i in range(3):
        _write_trajectory(tmp_output, _make_trajectory(task_id=f"mock_spec_1_seq{i}"))

    fake_llm.queue_batch([_summary_response(f"Summary {i}") for i in range(3)])

    summarize_trajectories(
        tasks_dir=tmp_output,
        llm=fake_llm,
    )

    # Exactly ONE LLM call (batched), three prompts inside it
    assert len(fake_llm.calls) == 1
    assert fake_llm.calls[0]["batched"] is True
    assert len(fake_llm.calls[0]["messages"]) == 3

    # Per-request usage attributed correctly (FakeLLM uses 10+i, 20+i)
    for i in range(3):
        saved = _read(tmp_output / f"mock_spec_1_seq{i}.json")
        u = saved["summary"]["usage"]
        assert (u["prompt_tokens"], u["completion_tokens"]) == (10 + i, 20 + i)


# ---------------------------------------------------------------------------
# 4 — Empty task (all turns failed)
# ---------------------------------------------------------------------------

def test_summarize_skips_trajectories_with_no_successful_turns(fake_llm, tmp_output):
    turns = [_turn(0, env_update=False), _turn(1, env_update=False)]
    path = _write_trajectory(tmp_output, _make_trajectory(turns=turns))

    summarize_trajectories(
        tasks_dir=tmp_output,
        llm=fake_llm,
        task_path=path,
    )

    # No LLM call, no summary field added
    assert fake_llm.calls == []
    assert "summary" not in _read(path)


# ---------------------------------------------------------------------------
# 5 — Debug log append
# ---------------------------------------------------------------------------

def test_summarize_writes_debug_log_append(fake_llm, tmp_output):
    task = _make_trajectory()
    path = _write_trajectory(tmp_output, task)
    debug_path = _write_debug_log(tmp_output, task["task_id"], n_events=3)

    fake_llm.queue(_summary_response())

    summarize_trajectories(
        tasks_dir=tmp_output,
        llm=fake_llm,
        task_path=path,
    )

    debug = _read(debug_path)
    assert len(debug["events"]) == 4  # 3 pre-existing + 1 summarize
    last = debug["events"][-1]
    assert last["agent"] == "TaskSummarizer"
    assert last["action"] == "summarize"
    assert "prompt" in last and "response" in last
    assert last["usage"]["prompt_tokens"] == 10
    assert last["usage"]["completion_tokens"] == 20


# ---------------------------------------------------------------------------
# 6 — No debug log present is fine
# ---------------------------------------------------------------------------

def test_summarize_no_debug_log_ok(fake_llm, tmp_output):
    task = _make_trajectory()
    path = _write_trajectory(tmp_output, task)
    # No debug file created

    fake_llm.queue(_summary_response())

    summarize_trajectories(
        tasks_dir=tmp_output,
        llm=fake_llm,
        task_path=path,
    )

    # Main task still got its summary — no crash
    assert "summary" in _read(path)
    assert not (tmp_output / f"{task['task_id']}.debug.json").exists()


# ---------------------------------------------------------------------------
# 7 — Field mode filters via env_specs
# ---------------------------------------------------------------------------

def test_summarize_field_mode_filters_via_env_specs(fake_llm, tmp_output):
    env_specs_dir = tmp_output / "env_specs"
    tasks_dir = tmp_output / "tasks"
    env_specs_dir.mkdir()
    tasks_dir.mkdir()

    # env_specs: two fields
    for spec_id, field in [
        ("aerospace_spec_000", "Aerospace"),
        ("aerospace_spec_001", "Aerospace"),
        ("healthcare_spec_000", "Healthcare"),
    ]:
        (env_specs_dir / f"{spec_id}.json").write_text(json.dumps({
            "spec_id": spec_id, "field": field,
        }))

    # Tasks: four files, two per field
    for spec_id in ["aerospace_spec_000", "aerospace_spec_001",
                    "healthcare_spec_000"]:
        _write_trajectory(tasks_dir,
                          _make_trajectory(task_id=f"{spec_id}_seq1"))

    # Only the two Aerospace tasks should be processed
    fake_llm.queue_batch([_summary_response(), _summary_response()])

    summarize_trajectories(
        tasks_dir=tasks_dir,
        llm=fake_llm,
        field="Aerospace",
        env_specs_dir=env_specs_dir,
    )

    assert "summary" in _read(tasks_dir / "aerospace_spec_000_seq1.json")
    assert "summary" in _read(tasks_dir / "aerospace_spec_001_seq1.json")
    assert "summary" not in _read(tasks_dir / "healthcare_spec_000_seq1.json")


# ---------------------------------------------------------------------------
# 8 — Last-attempt-per-tool-idx dedup
# ---------------------------------------------------------------------------

def test_summarize_extracts_only_last_successful_attempt_per_tool_idx(fake_llm, tmp_output):
    # tool_idx=0: attempt 0 failed, attempt 1 succeeded with a distinctive task
    turns = [
        _turn(0, attempt=0, env_update=False,
              task_description="FAIL ATTEMPT — should be excluded"),
        _turn(0, attempt=1, env_update=True,
              task_description="SUCCESS ATTEMPT — keeper"),
        _turn(1, attempt=0, env_update=True,
              task_description="Second tool task"),
    ]
    path = _write_trajectory(tmp_output, _make_trajectory(turns=turns))

    fake_llm.queue(_summary_response())
    summarize_trajectories(tasks_dir=tmp_output, llm=fake_llm, task_path=path)

    saved = _read(path)
    # n_subtasks reflects 2 successful tool_idx entries, not 3 raw turns
    assert saved["summary"]["n_subtasks"] == 2

    # The prompt passed to the LLM should include the success attempt, not the failed one
    prompt_text = _all_content(fake_llm.calls[0]["messages"])
    assert "SUCCESS ATTEMPT" in prompt_text
    assert "FAIL ATTEMPT" not in prompt_text


# ---------------------------------------------------------------------------
# 9 — verifiable=False tasks still work
# ---------------------------------------------------------------------------

def test_summarize_skips_verifiable_false_non_env_update_turns(fake_llm, tmp_output):
    # All turns have judge=None (verifiable=False), but mix env_update presence
    turns = [
        _turn(0, env_update=True, task_description="Solved turn"),
        _turn(1, env_update=False, task_description="Failed tool call — no env update"),
    ]
    task = _make_trajectory(turns=turns)
    task["config"]["verifiable"] = False
    path = _write_trajectory(tmp_output, task)

    fake_llm.queue(_summary_response())
    summarize_trajectories(tasks_dir=tmp_output, llm=fake_llm, task_path=path)

    saved = _read(path)
    assert saved["summary"]["n_subtasks"] == 1
    prompt_text = _all_content(fake_llm.calls[0]["messages"])
    assert "Solved turn" in prompt_text
    assert "Failed tool call" not in prompt_text


# ---------------------------------------------------------------------------
# 10 — `explanation` stripped from tool responses
# ---------------------------------------------------------------------------

def test_summarize_strips_explanation_from_tool_response(fake_llm, tmp_output):
    turns = [_turn(0, include_explanation=True,
                   task_description="Task using StripExplanation")]
    path = _write_trajectory(tmp_output, _make_trajectory(turns=turns))

    fake_llm.queue(_summary_response())
    summarize_trajectories(tasks_dir=tmp_output, llm=fake_llm, task_path=path)

    prompt_text = _all_content(fake_llm.calls[0]["messages"])
    # The explanation string MUST NOT leak into the summarizer prompt
    assert "simulator internal notes" not in prompt_text
    # But the rest of the response should still be there
    assert "status_code" in prompt_text


# ---------------------------------------------------------------------------
# 11 — tool_call source comes from the AGENT's chat, NOT expected_tool_call
# ---------------------------------------------------------------------------

def test_summarize_uses_agent_call_from_chat_not_expected_tool_call(fake_llm, tmp_output):
    """The evolver's `expected_tool_call` is what was PROPOSED; the agent's
    actual successful call (in `chat[i].assistant`) is what worked. The
    summariser must use the AGENT's call."""
    # Build a turn where expected_tool_call != the agent's chat call
    turn = _turn(0,
                 task_description="Compute X using the right tool",
                 tool_call="EVOLVER_PROPOSED_CALL(x=1)")  # both expected and chat set to this
    # Now override the chat's assistant message to a DIFFERENT call
    turn["chat"][1]["content"] = json.dumps({
        "reason": "I corrected the format",
        "tool_call": "AGENT_REAL_CALL(x=1)",
    })
    path = _write_trajectory(tmp_output, _make_trajectory(turns=[turn]))

    fake_llm.queue(_summary_response())
    summarize_trajectories(tasks_dir=tmp_output, llm=fake_llm, task_path=path)

    prompt_text = _all_content(fake_llm.calls[0]["messages"])
    assert "AGENT_REAL_CALL(x=1)" in prompt_text, \
        "summariser must use the agent's chat tool_call"
    assert "EVOLVER_PROPOSED_CALL" not in prompt_text, \
        "summariser must NOT use the evolver's expected_tool_call"


# ---------------------------------------------------------------------------
# 12 — within-turn 400 → fix → 200: only the 200 exchange is used
# ---------------------------------------------------------------------------

def test_summarize_takes_only_last_2xx_exchange_within_turn(fake_llm, tmp_output):
    """A turn may contain a 400 failure then a 200 retry inside the same
    attempt. The summariser must take only the FINAL 200 exchange (call +
    response), ignoring the earlier failed call."""
    turn = _turn(0, task_description="Compute Y")
    # Replace the chat with: 400 attempt, then 200 retry
    turn["chat"] = [
        {"role": "user", "content": "Compute Y"},
        {"role": "assistant", "content": json.dumps({"reason": "first", "tool_call": "FAILED_FIRST(z=0)"})},
        {"role": "tool", "content": json.dumps({"status_code": 400, "response": "bad arg"})},
        {"role": "assistant", "content": json.dumps({"reason": "fixed", "tool_call": "SUCCEEDED_SECOND(z=1)"})},
        {"role": "tool", "content": json.dumps({"status_code": 200, "response": {"value": 42}})},
    ]
    path = _write_trajectory(tmp_output, _make_trajectory(turns=[turn]))

    fake_llm.queue(_summary_response())
    summarize_trajectories(tasks_dir=tmp_output, llm=fake_llm, task_path=path)

    prompt_text = _all_content(fake_llm.calls[0]["messages"])
    assert "SUCCEEDED_SECOND" in prompt_text, "must include the successful call"
    assert "FAILED_FIRST" not in prompt_text, "must NOT include the failed call"
    assert "bad arg" not in prompt_text, "must NOT include the failed response"
    assert "42" in prompt_text, "must include the successful response"


# ---------------------------------------------------------------------------
# 13 — Production 4-field response: pipeline plucks task_summarized and
# preserves the show-your-work fields end-to-end (task JSON + release JSONL).
# ---------------------------------------------------------------------------

def test_summarize_handles_four_field_response_end_to_end(
    fake_llm, tmp_output, tmp_path
):
    """The production prompt now emits a 4-field JSON:
        user_supplied_values, tool_produced_values, spillover, task_summarized.
    The pipeline must:
      - Pick up `task_summarized` correctly (as it always has).
      - Preserve the other three fields in the task's parsed-summary block.
      - Write the release JSONL with only the string `summary` column
        (the show-your-work fields are not in the release schema)."""

    # Build a task with a spec_id-compatible task_id and a matching env_spec
    # so _extract_release_row actually emits a row.
    env_specs_dir = tmp_path / "env_specs"
    env_specs_dir.mkdir()
    (env_specs_dir / "mock_spec_1.json").write_text(json.dumps({
        "spec_id": "mock_spec_1",
        "field": "Mock Field",
        "tools": [{"tool_name": "ToolA", "tool_description": "..."},
                  {"tool_name": "ToolB", "tool_description": "..."}],
    }))

    task = _make_trajectory()
    path = _write_trajectory(tmp_output, task)

    fake_llm.queue(_summary_response_v2(
        text="Process the refund for order 1234.",
        user_supplied_values=["order 1234", "damaged"],
        tool_produced_values=["delivered", "89.50", "ret_77a3"],
        spillover=[],
    ))

    task_content_path = tmp_path / "task_content.jsonl"
    summarize_trajectories(
        tasks_dir=tmp_output,
        llm=fake_llm,
        env_specs_dir=env_specs_dir,
        task_content_path=task_content_path,
        task_path=path,
    )

    # 1. The task JSON keeps the FULL parsed dict — show-your-work
    # fields survive into the on-disk summary block (debug / inspection).
    saved = _read(path)
    parsed = saved["summary"]["parsed"]
    assert parsed["task_summarized"] == "Process the refund for order 1234."
    assert parsed["user_supplied_values"] == ["order 1234", "damaged"]
    assert parsed["tool_produced_values"] == ["delivered", "89.50", "ret_77a3"]
    assert parsed["spillover"] == []

    # 2. The release JSONL gets only the `summary` string — no leak of
    # the show-your-work fields into the released schema.
    rows = [json.loads(line) for line in task_content_path.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["summary"] == "Process the refund for order 1234."
    assert "user_supplied_values" not in row
    assert "tool_produced_values" not in row
    assert "spillover" not in row
    # Spot-check the rest of the row schema.
    assert row["id"] == task["task_id"]
    assert row["field"] == "Mock Field"
    assert isinstance(row["tools"], list)
    assert isinstance(row["gt_tool_calls"], list)


# ---------------------------------------------------------------------------
# 14 — Empty / malformed model output: pipeline degrades gracefully when
# the model fails to emit the 4-field JSON.
# ---------------------------------------------------------------------------

def test_summarize_tolerates_malformed_response(fake_llm, tmp_output):
    """If the model returns garbage that doesn't contain a JSON object,
    `parsed` is None and the summary block is still written (no crash).
    Downstream code reading `parsed.task_summarized` should default to
    empty without raising."""
    task = _make_trajectory()
    path = _write_trajectory(tmp_output, task)

    fake_llm.queue("I tried but I couldn't produce structured output.")
    summarize_trajectories(tasks_dir=tmp_output, llm=fake_llm, task_path=path)

    saved = _read(path)
    assert "summary" in saved
    assert saved["summary"]["parsed"] is None
    # `_extract_release_row` reads parsed.task_summarized; with None it
    # falls back to "" and the row is skipped (no `summary` text → drop).
