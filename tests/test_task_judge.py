"""Tests for roles/task_judge.py — grounding-first judge with information asymmetry.

The judge must:
  - run arguments_grounded check ALWAYS (independent of tool_call_equality)
  - have access to the same world-information as the agent (current task + prior_chat)
  - NOT accept env_metadata / env_state — asymmetry enforced by API
  - emit feedback consumed by evolver retries
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from roles.task_judge import TaskJudge  # noqa: E402


def _runner_from_fake(fake_llm):
    """Wrap FakeLLM as the simple string-runner the role expects."""
    def runner(prompt: str) -> str:
        return fake_llm(prompt)
    runner.last_usage = None  # not used by judge; role pulls usage off the LLM
    return runner


def _make_judge(fake_llm):
    """Build a TaskJudge that delegates to FakeLLM. The role expects a callable
    runner with a `last_usage` attribute; we attach it dynamically per call."""
    def runner(prompt: str) -> str:
        out = fake_llm(prompt)
        runner.last_usage = fake_llm.last_usage
        return out
    runner.last_usage = None
    return TaskJudge(runner)


# --- API-shape: signature does NOT accept env_metadata / env_state ----------


def test_judge_signature_rejects_env_metadata(fake_llm):
    """Asymmetry-by-API: the judge_task_gen function must not accept
    `env_metadata`, `env_state`, or anything env-related as a kwarg."""
    judge = _make_judge(fake_llm)
    sig = inspect.signature(judge.judge_task_gen)
    forbidden = {"env_metadata", "env_state", "environment_state", "environment_metadata"}
    params = set(sig.parameters)
    leaked = params & forbidden
    assert not leaked, f"judge_task_gen must not accept env-related params; leaked: {leaked}"
    # Required slots present:
    assert "true_tool_call" in params
    assert "current_task_description" in params
    assert "prior_chat" in params
    assert "agent_tool_calls" in params


# --- Grounded: value verbatim in current task -------------------------------


def test_judge_grounded_when_value_in_current_task(fake_llm):
    judge = _make_judge(fake_llm)
    fake_llm.queue(json.dumps({
        "argument_citations": [{
            "argument_name": "altitude_ft",
            "value": "15000",
            "source_quote": "cruise altitude of 15000 ft",
            "source_location": "current_task",
            "derivation": "verbatim",
            "grounded": True,
        }],
        "arguments_grounded": True,
        "tool_call_equality": True,
        "task_solved": True,
        "task_solved_confidence": 0.95,
        "task_solvability": 1.0,
        "feedback": "",
    }))

    result = judge.judge_task_gen(
        true_tool_call="LiftRequirementEstimator(altitude_ft=15000)",
        current_task_description="Estimate the lift coefficient at a cruise altitude of 15000 ft.",
        prior_chat=[],
        agent_tool_calls=[{"tool_call": "LiftRequirementEstimator(altitude_ft=15000)",
                           "tool_output": {"status_code": 200, "response": {"lift": 0.42}}}],
    )
    parsed = result["parsed"]
    assert parsed["arguments_grounded"] is True
    assert parsed["task_solved"] is True
    assert parsed["argument_citations"][0]["source_location"] == "current_task"


# --- Grounded: value comes from a prior tool response -----------------------


def test_judge_grounded_when_value_in_prior_response(fake_llm):
    judge = _make_judge(fake_llm)
    fake_llm.queue(json.dumps({
        "argument_citations": [{
            "argument_name": "return_request_id",
            "value": "RET002",
            "source_quote": '"return_request_id":"RET002"',
            "source_location": "prior_response_0",
            "derivation": "verbatim",
            "grounded": True,
        }],
        "arguments_grounded": True,
        "tool_call_equality": True,
        "task_solved": True,
        "task_solved_confidence": 0.92,
        "task_solvability": 1.0,
        "feedback": "",
    }))

    prior_chat = [{
        "task_description": "Validate the return request.",
        "tool_call": "ReturnRequestValidator(return_request_id='RET002')",
        "tool_response": {"status_code": 200,
                          "response": {"return_request_id": "RET002", "status": "approved"}},
    }]
    result = judge.judge_task_gen(
        true_tool_call="ReturnLabelGenerator(return_request_id='RET002', carrier='ups')",
        current_task_description="Generate a UPS return label for the approved return.",
        prior_chat=prior_chat,
        agent_tool_calls=[{"tool_call": "ReturnLabelGenerator(return_request_id='RET002', carrier='ups')",
                           "tool_output": {"status_code": 200, "response": {}}}],
    )
    parsed = result["parsed"]
    assert parsed["arguments_grounded"] is True
    assert parsed["argument_citations"][0]["source_location"] == "prior_response_0"


# --- Ungrounded: value invented (the confabulation case) --------------------


def test_judge_ungrounded_when_value_only_in_expected_tool_call(fake_llm):
    """Value LAYOUT-X appears only in the true_tool_call — no source in chat."""
    judge = _make_judge(fake_llm)
    fake_llm.queue(json.dumps({
        "argument_citations": [{
            "argument_name": "layout_design",
            "value": "LAYOUT-20231101-001",
            "source_quote": "",
            "source_location": "none",
            "derivation": "none",
            "grounded": False,
        }],
        "arguments_grounded": False,
        "tool_call_equality": True,
        "task_solved": False,
        "task_solved_confidence": 1.0,
        "task_solvability": 0.0,
        "feedback": "Value 'LAYOUT-20231101-001' is not present in the current task description nor in any prior tool response. Either name it verbatim in the task description, or thread it from a prior tool response.",
    }))

    result = judge.judge_task_gen(
        true_tool_call='SimulationRunner(layout_design="LAYOUT-20231101-001")',
        current_task_description="Run a simulation using the existing layout.",
        prior_chat=[],
        agent_tool_calls=[{"tool_call": 'SimulationRunner(layout_design="LAYOUT-20231101-001")',
                           "tool_output": {"status_code": 200, "response": {}}}],
    )
    parsed = result["parsed"]
    assert parsed["arguments_grounded"] is False
    assert parsed["task_solved"] is False
    assert "LAYOUT-20231101-001" in parsed["feedback"]


# --- Regression: ungrounded must override matching tool_call ---------------


def test_judge_ungrounded_overrides_tool_call_equality(fake_llm):
    """Solver and evolver hallucinate the same ID. Calls match → old logic
    would skip the grounding check. New logic: grounding always runs first;
    arguments_grounded=False forces task_solved=False."""
    judge = _make_judge(fake_llm)
    fake_llm.queue(json.dumps({
        "argument_citations": [{
            "argument_name": "session_id",
            "value": "SESSION-999",
            "source_quote": "",
            "source_location": "none",
            "derivation": "none",
            "grounded": False,
        }],
        "arguments_grounded": False,
        "tool_call_equality": True,   # they match — but it's still bad
        "task_solved": False,
        "task_solved_confidence": 1.0,
        "task_solvability": 0.0,
        "feedback": "Value 'SESSION-999' is not in the task description or any prior tool response.",
    }))

    result = judge.judge_task_gen(
        true_tool_call='Foo(session_id="SESSION-999")',
        current_task_description="Use the active session to do X.",
        prior_chat=[],
        agent_tool_calls=[{"tool_call": 'Foo(session_id="SESSION-999")',
                           "tool_output": {"status_code": 200, "response": {}}}],
    )
    parsed = result["parsed"]
    assert parsed["tool_call_equality"] is True
    assert parsed["arguments_grounded"] is False
    assert parsed["task_solved"] is False, \
        "task_solved must be False when arguments_grounded=False, even if tool_call_equality=True"


# --- Output schema parse ----------------------------------------------------


def test_judge_returns_parsed_dict_with_all_fields(fake_llm):
    judge = _make_judge(fake_llm)
    fake_llm.queue(json.dumps({
        "argument_citations": [],
        "arguments_grounded": True,
        "tool_call_equality": True,
        "task_solved": True,
        "task_solved_confidence": 0.99,
        "task_solvability": 1.0,
        "feedback": "",
    }))
    result = judge.judge_task_gen(
        true_tool_call="X()",
        current_task_description="Do X.",
        prior_chat=[],
        agent_tool_calls=[],
    )
    assert "prompt" in result and "response" in result and "parsed" in result
    parsed = result["parsed"]
    for key in ("argument_citations", "arguments_grounded", "tool_call_equality",
                "task_solved", "task_solvability", "feedback"):
        assert key in parsed, f"missing field {key} in judge output"


# --- Prompt rendering: prior_chat actually appears in the rendered prompt ---


def test_judge_prompt_renders_prior_chat(fake_llm):
    """The rendered prompt must contain the prior_chat content so the LLM can use
    it. Regression check that we don't drop prior_chat into the void."""
    judge = _make_judge(fake_llm)
    fake_llm.queue(json.dumps({
        "argument_citations": [],
        "arguments_grounded": True,
        "tool_call_equality": True,
        "task_solved": True,
        "task_solved_confidence": 0.9,
        "task_solvability": 1.0,
        "feedback": "",
    }))
    prior_chat = [{
        "task_description": "Validate XYZ-123",
        "tool_call": "Validator(id='XYZ-123')",
        "tool_response": {"status_code": 200, "response": {"id": "XYZ-123", "ok": True}},
    }]
    result = judge.judge_task_gen(
        true_tool_call="Followup(id='XYZ-123')",
        current_task_description="Do the followup on the validated record.",
        prior_chat=prior_chat,
        agent_tool_calls=[],
    )
    rendered = result["prompt"]
    assert "XYZ-123" in rendered
    assert "Validator" in rendered
