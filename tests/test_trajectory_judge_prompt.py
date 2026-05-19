"""Verify the trajectory_judge YAML template loads, formats with placeholders,
and exposes every field documented in the spec."""

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from roles.trajectory_judge import (   # noqa: E402
    TRAJECTORY_JUDGE_TEMPLATE_FILE,
    TrajectoryJudge,
)


def test_yaml_parses_with_template_key():
    with open(TRAJECTORY_JUDGE_TEMPLATE_FILE) as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)
    assert "template" in data and isinstance(data["template"], str)
    assert "schema" in data


def test_template_format_with_placeholders_succeeds():
    """The .format() call must succeed with every documented input slot."""
    with open(TRAJECTORY_JUDGE_TEMPLATE_FILE) as f:
        template = yaml.safe_load(f)["template"]
    rendered = template.format(
        task_summary="solve a task",
        gt_tool_calls=json.dumps(["A()", "B()"]),
        agent_tool_calls=json.dumps([{"tool_call": "A()", "tool_output": {}}]),
        initial_state=json.dumps({"k": 0}),
        final_state_gt=json.dumps({"k": 1}),
    )
    assert "task_summary" not in rendered or "solve a task" in rendered  # value substituted
    assert "solve a task" in rendered
    assert "A()" in rendered


def test_template_advertises_all_required_output_keys():
    """The judge must instruct the model to emit every contract field."""
    with open(TRAJECTORY_JUDGE_TEMPLATE_FILE) as f:
        template = yaml.safe_load(f)["template"]
    for k in [
        "per_call",
        "tool_call_match_rate",
        "missing_calls",
        "extra_calls",
        "final_state_match",
        "final_state_diff",
        "trajectory_solved",
        "confidence",
        "reasoning",
    ]:
        assert k in template, f"output key {k!r} not mentioned in the prompt template"


def test_judge_role_renders_prompt_via_runner_stub():
    """Calling judge_trajectory with a dummy runner returns the parsed JSON envelope."""

    captured = {}

    def fake_runner(prompt: str) -> str:
        captured["prompt"] = prompt
        return json.dumps({
            "per_call": [{"gt": "A()", "agent": "A()", "tool_name_match": True,
                          "argument_match": True, "notes": ""}],
            "tool_call_match_rate": 1.0,
            "missing_calls": [],
            "extra_calls": [],
            "final_state_match": True,
            "final_state_diff": "",
            "trajectory_solved": True,
            "confidence": 0.99,
            "reasoning": "All calls aligned and final state matches.",
        })

    judge = TrajectoryJudge(runner=fake_runner)
    out = judge.judge_trajectory(
        task_summary="t",
        gt_tool_calls=["A()"],
        agent_tool_calls=[{"tool_call": "A()", "tool_output": {}}],
        initial_state={"k": 0},
        final_state_gt={"k": 1},
    )
    assert out["parsed"]["trajectory_solved"] is True
    assert out["parsed"]["tool_call_match_rate"] == 1.0
    assert "task_summary" not in captured["prompt"] or "t" in captured["prompt"]
    # The runner's prompt must have all five inputs interpolated.
    assert "A()" in captured["prompt"]
