"""Unit tests for the chat-format task_evolver t1 refactor.

These tests pin the SHAPE of the message list the evolver sends to the LLM:
- prior successful turns serialised as (user, assistant, tool) triples
- the new request as a final user message
- tool-returned values living under role=tool, never copied into a later
  user-role message by the chat builder.

t0 (cold start) must still pass a plain string to the runner. The legacy
single-`template:` YAML must be rejected by the loader so a half-migrated
deployment fails loudly.
"""

import json
from pathlib import Path

import pytest
import yaml

# Pulled in via conftest's sys.path manipulation.
from roles.task_evolver import TaskEvolver


class _RecordingRunner:
    """Callable that records every argument it was called with."""

    def __init__(self, canned_response: str = '{"tool_call": "X()"}'):
        self.canned = canned_response
        self.calls = []

    def __call__(self, arg):
        self.calls.append(arg)
        return self.canned


def _one_successful_turn():
    return [{
        "task_description": "Estimate the lift requirements for the mission.",
        "tool_call": "LiftRequirementEstimator(required_range_km=1500, payload_kg=800)",
        "tool_simulated": {
            "status_code": 200,
            "response": {"target_lift_coefficient": 0.56, "target_wing_loading_kg_per_m2": 71.3},
        },
    }]


def _next_tool_details():
    return {
        "tool_name": "WingGeometryDesigner",
        "tool_description": "Design a wing.",
        "parameters": {"wing_loading_kg_per_m2": {"type": "number"}},
    }


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_t1_builds_system_plus_history_plus_final_user():
    """One prior turn → 5 messages: system, user, assistant, tool, user."""
    runner = _RecordingRunner()
    e = TaskEvolver(runner)
    msgs = e._build_t1_messages(
        successful_task=_one_successful_turn(),
        unsuccessful_tasks=[],
        tool_details=_next_tool_details(),
        environment_state={"mission": "x"},
    )
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool", "user"]
    assert "Task Evolver" in msgs[0]["content"]
    assert "Estimate the lift requirements" in msgs[1]["content"]
    assert "LiftRequirementEstimator" in msgs[2]["content"]
    # tool message is JSON-encoded simulator payload
    tool_payload = json.loads(msgs[3]["content"])
    assert tool_payload["status_code"] == 200
    assert tool_payload["response"]["target_wing_loading_kg_per_m2"] == 71.3


def test_t1_tool_response_lives_in_tool_role_message_only():
    """A tool-returned value must appear in the `tool` role's message and
    must NOT be copied into any later user-role message by the chat builder.
    """
    runner = _RecordingRunner()
    e = TaskEvolver(runner)
    msgs = e._build_t1_messages(
        successful_task=_one_successful_turn(),
        unsuccessful_tasks=[],
        tool_details=_next_tool_details(),
        environment_state={"mission": "x"},
    )
    leak_value = "71.3"  # the wing-loading number returned by the tool
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert leak_value in tool_msgs[0]["content"]
    # NB: 71.3 may legitimately reappear inside the FINAL user message
    # because environment_state can contain values produced by earlier
    # tools — that's a known constraint of the per-turn pipeline. What we
    # guard against here is the chat builder copying it into the
    # PRIOR-TURN user messages (the mini-task descriptions).
    history_user_msgs = [m for m in msgs[1:-1] if m["role"] == "user"]
    for m in history_user_msgs:
        assert leak_value not in m["content"], (
            "tool-returned value leaked into a prior-turn user message"
        )


def test_t1_final_user_contains_tool_details_and_env_state():
    """The final user message must contain the rendered tool_details and
    environment_state — that's how the LLM learns what the next step is."""
    runner = _RecordingRunner()
    e = TaskEvolver(runner)
    msgs = e._build_t1_messages(
        successful_task=_one_successful_turn(),
        unsuccessful_tasks=[],
        tool_details=_next_tool_details(),
        environment_state={"mission": "EnvelopeForX"},
    )
    final = msgs[-1]
    assert final["role"] == "user"
    assert "WingGeometryDesigner" in final["content"]
    assert "EnvelopeForX" in final["content"]


def test_t1_unsuccessful_attempts_rendered_in_final_user():
    """Judge feedback (`judge_explanation`) on prior failed attempts MUST
    reach the final user message; otherwise the evolver can't address it."""
    runner = _RecordingRunner()
    e = TaskEvolver(runner)
    failed = [
        {
            "task_description": "Bad first try.",
            "tool_call": "WingGeometryDesigner(wing_loading_kg_per_m2='UNGROUNDED')",
            "judge_explanation": "Value 'UNGROUNDED' was not in the task description.",
        }
    ]
    msgs = e._build_t1_messages(
        successful_task=_one_successful_turn(),
        unsuccessful_tasks=failed,
        tool_details=_next_tool_details(),
        environment_state={"mission": "x"},
    )
    final = msgs[-1]["content"]
    assert "UNGROUNDED" in final
    assert "not in the task description" in final


def test_t1_with_empty_history_is_just_system_plus_user():
    """A t1 invocation with no prior successful turns → 2-message chat."""
    runner = _RecordingRunner()
    e = TaskEvolver(runner)
    msgs = e._build_t1_messages(
        successful_task=[],
        unsuccessful_tasks=[],
        tool_details=_next_tool_details(),
        environment_state={},
    )
    assert [m["role"] for m in msgs] == ["system", "user"]


def test_t1_normalises_json_string_successful_task():
    """Back-compat: if a caller passes successful_task as a JSON-encoded
    string (instead of a Python list), the builder should still produce
    a valid chat."""
    runner = _RecordingRunner()
    e = TaskEvolver(runner)
    encoded = json.dumps(_one_successful_turn())
    msgs = e._build_t1_messages(
        successful_task=encoded,
        unsuccessful_tasks=None,  # also accept None for unsuccessful_tasks
        tool_details=_next_tool_details(),
        environment_state={"x": 1},
    )
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool", "user"]


def test_t1_drops_malformed_history_entries_silently():
    """Non-dict entries in successful_task should be skipped (defensive)."""
    runner = _RecordingRunner()
    e = TaskEvolver(runner)
    history = [
        "not a dict — should be skipped",
        _one_successful_turn()[0],
        None,
    ]
    msgs = e._build_t1_messages(
        successful_task=history,
        unsuccessful_tasks=[],
        tool_details=_next_tool_details(),
        environment_state={},
    )
    # one valid entry → 5 messages
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool", "user"]


# ---------------------------------------------------------------------------
# Runner-shape tests (what actually gets passed to the LLM)
# ---------------------------------------------------------------------------

def test_evolve_task_t1_calls_runner_with_list():
    """The integration: calling evolve_task_t1 should pass a list of message
    dicts to the runner, not a single string."""
    runner = _RecordingRunner(canned_response=json.dumps({
        "tool_name": "WingGeometryDesigner",
        "tool_call": "WingGeometryDesigner(...)",
        "edited_metadata": {},
        "env_metadata": {},
        "task_description": "next step",
        "depends_on_previous": True,
    }))
    e = TaskEvolver(runner)
    result = e.evolve_task_t1(
        successful_task=_one_successful_turn(),
        unsuccessful_tasks=[],
        tool_details=_next_tool_details(),
        environment_state={"x": 1},
    )
    assert len(runner.calls) == 1
    sent = runner.calls[0]
    assert isinstance(sent, list), f"expected list of messages, got {type(sent).__name__}"
    assert all(isinstance(m, dict) and "role" in m and "content" in m for m in sent)
    # And the result still carries a `prompt` string (now JSON-encoded chat)
    assert isinstance(result["prompt"], str)
    decoded = json.loads(result["prompt"])
    assert isinstance(decoded, list) and decoded == sent


def test_evolve_task_t0_unchanged_string_prompt():
    """Regression guard: t0 must still pass a string to the runner."""
    runner = _RecordingRunner(canned_response='{"tool_name":"X","tool_call":"X()","task_description":"y","env_metadata":{}}')
    e = TaskEvolver(runner)
    e.evolve_task_t0({"tool_name": "X", "parameters": {}})
    assert len(runner.calls) == 1
    sent = runner.calls[0]
    assert isinstance(sent, str), f"t0 should pass a string, got {type(sent).__name__}"
    assert "Task Evolver" in sent


# ---------------------------------------------------------------------------
# Template loader tests
# ---------------------------------------------------------------------------

def test_load_prompts_rejects_legacy_t1_yaml(tmp_path, monkeypatch):
    """If the t1 YAML still has only the old `template:` field (legacy
    single-shot format), the loader must raise — silently falling back
    would mask a half-migrated deployment."""
    legacy_yaml = tmp_path / "task_evolver_t1_template.yml"
    legacy_yaml.write_text(yaml.safe_dump({
        "schema": {"type": "object"},
        "template": "You are a Task Evolver. ... {successful_task} ...",
    }))

    # Also need a working t0 file in the same dir for the loader to find.
    t0_yaml = tmp_path / "task_evolver_t0_template.yml"
    t0_yaml.write_text(yaml.safe_dump({
        "schema": {"type": "object"},
        "template": "t0 prompt {tool_details}",
    }))

    monkeypatch.setattr("roles.task_evolver.TASK_EVOLVER_T0_TEMPLATE_FILE", t0_yaml)
    monkeypatch.setattr("roles.task_evolver.TASK_EVOLVER_T1_TEMPLATE_FILE", legacy_yaml)

    with pytest.raises(ValueError) as exc:
        TaskEvolver(lambda x: "")
    err = str(exc.value)
    assert "system_template" in err and "final_user_template" in err


def test_load_prompts_accepts_current_yaml():
    """Sanity: the real t1 YAML in the repo loads cleanly into both keys."""
    e = TaskEvolver(lambda x: "")
    assert "task_evolver_t1_system" in e.prompts
    assert "task_evolver_t1_final_user" in e.prompts
    assert "Task Evolver" in e.prompts["task_evolver_t1_system"]
    assert "{tool_details}" in e.prompts["task_evolver_t1_final_user"]
