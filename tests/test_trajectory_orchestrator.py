"""Tests for trajectory_generation.orchestrator.generate_trajectory.

Uses a fake LLM with canned responses to verify the loop terminates correctly,
the chat shape matches expectations, and the judge runs only when requested.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trajectory_generation.loader import Task               # noqa: E402
from trajectory_generation.orchestrator import generate_trajectory   # noqa: E402


# --- canned LLM ---

class FakeLLM:
    """Plays back a fixed list of responses; tracks calls and usage."""

    def __init__(self, scripted: List[str]):
        self._iter = iter(scripted)
        self.model = "fake"
        self.last_usage = None
        self.last_usage_per_request = None
        self.prompts: List[Any] = []

    def __call__(self, messages):
        self.prompts.append(messages)
        try:
            r = next(self._iter)
        except StopIteration:
            r = json.dumps({"reason": "fallback", "tool_call": "<STOP>"})
        # Mimic the public API: assignable last_usage attribute (Usage-shaped dict).
        self.last_usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        return r


def _toy_task() -> Task:
    return Task(
        id="toy_field_spec_000_seq1",
        field="Toy Field",
        summary="Greet the user, then say goodbye.",
        tools=[
            {"tool_name": "Hello", "tool_description": "Says hi.",
             "parameters": {"name": {"type": "string", "required": True}},
             "output_details": {"greeting": {"type": "string"}}},
            {"tool_name": "Goodbye", "tool_description": "Says bye.",
             "parameters": {"name": {"type": "string", "required": True}},
             "output_details": {"farewell": {"type": "string"}}},
        ],
        gt_tool_calls=["Hello(name='alice')", "Goodbye(name='alice')"],
        initial_state={"greeted": False},
        final_state={"greeted": True, "said_goodbye": True},
    )


# --- responses the fake LLM will play back ---

# Sequence: Hello → simulator (param_check + simulate) → Goodbye → simulator → STOP → judge
def _scripted_responses() -> List[str]:
    hello_call    = json.dumps({"reason": "Greet alice.", "tool_call": "Hello(name='alice')"})
    goodbye_call  = json.dumps({"reason": "Bid farewell.", "tool_call": "Goodbye(name='alice')"})
    stop_call     = json.dumps({"reason": "Done.",      "tool_call": "<STOP>"})

    # Each ToolSimulator.simulate call is two LLM calls: parameter_check and simulate_raw.
    pc_pass = json.dumps({"status": "PASS", "status_code": 200})
    sim_ok  = json.dumps({"status_code": 200, "response": {"ok": True}})

    judge_verdict = json.dumps({
        "per_call": [
            {"gt": "Hello(name='alice')",   "agent": "Hello(name='alice')",
             "tool_name_match": True, "argument_match": True, "notes": ""},
            {"gt": "Goodbye(name='alice')", "agent": "Goodbye(name='alice')",
             "tool_name_match": True, "argument_match": True, "notes": ""},
        ],
        "tool_call_match_rate": 1.0,
        "missing_calls": [],
        "extra_calls": [],
        "final_state_match": True,
        "final_state_diff": "",
        "trajectory_solved": True,
        "confidence": 0.99,
        "reasoning": "All aligned.",
    })

    return [
        hello_call,                    # solver turn 0
        pc_pass, sim_ok,               # ToolSimulator on Hello
        goodbye_call,                  # solver turn 1
        pc_pass, sim_ok,               # ToolSimulator on Goodbye
        stop_call,                     # solver turn 2 → <STOP>
        judge_verdict,                 # TrajectoryJudge
    ]


# --- tests ---

def test_generate_trajectory_full_run_with_judge(tmp_path):
    llm = FakeLLM(_scripted_responses())
    task = _toy_task()
    out = generate_trajectory(task, llm, output_dir=tmp_path,
                              max_solver_turns=10, debug=False, run_judge=True)

    # Trajectory schema sanity
    assert out["task_id"] == task.id
    assert out["model"] == "fake"
    assert out["config"]["stop_reason"] == "stop_emitted"
    assert out["tool_ids"] == ["Hello", "Goodbye"]
    assert len(out["turns"]) == 2
    assert out["turns"][0]["tool_name"] == "Hello"
    assert out["turns"][1]["tool_name"] == "Goodbye"
    assert all(t["passed"] is True for t in out["turns"])

    # Ground-truth echoed
    assert out["ground_truth"]["gt_tool_calls"] == task.gt_tool_calls
    assert out["ground_truth"]["final_state"] == task.final_state

    # Judge populated
    j = out["trajectory_judge"]
    assert j is not None
    assert j["trajectory_solved"] is True
    assert j["tool_call_match_rate"] == 1.0

    # File written and self-consistent on disk
    saved = json.loads((tmp_path / f"{task.id}.json").read_text())
    assert saved["task_id"] == task.id
    assert saved["trajectory_judge"]["trajectory_solved"] is True


def test_generate_trajectory_skips_judge_when_disabled(tmp_path):
    # Drop the judge response from the script — we should never reach it.
    responses = _scripted_responses()[:-1]  # exclude judge_verdict
    llm = FakeLLM(responses)
    task = _toy_task()
    out = generate_trajectory(task, llm, output_dir=tmp_path,
                              max_solver_turns=10, debug=False, run_judge=False)

    assert out["trajectory_judge"] is None
    assert out["config"]["stop_reason"] == "stop_emitted"
    # All scripted responses consumed except none left over for the judge.
    assert llm._iter.__next__ is not None  # generator still exists


def test_unknown_tool_call_records_error_and_continues(tmp_path):
    """If the agent emits a tool name not in the catalogue, we record an error
    and proceed (no simulator call). The next assistant message can recover."""
    responses = [
        json.dumps({"reason": "wrong tool", "tool_call": "Nonexistent(x=1)"}),
        json.dumps({"reason": "ok stop", "tool_call": "<STOP>"}),
    ]
    llm = FakeLLM(responses)
    task = _toy_task()
    out = generate_trajectory(task, llm, output_dir=tmp_path,
                              max_solver_turns=5, debug=False, run_judge=False)

    assert len(out["turns"]) == 1
    assert out["turns"][0]["error"] == "unknown_tool_name"
    assert out["turns"][0]["passed"] is False


def test_param_check_failure_records_400_in_tool_output(tmp_path):
    """When parameter_check fails, tool_output must carry the 400 payload the
    agent actually received, not None. simulate_raw is never called on a fail,
    so the simulator contributes exactly one (param_check) LLM response."""
    hello = json.dumps({"reason": "bad args", "tool_call": "Hello(name=123)"})
    pc_fail = json.dumps({"status": "FAIL", "status_code": 400,
                          "error_message": "Invalid type for name: expected string."})
    stop = json.dumps({"reason": "give up", "tool_call": "<STOP>"})
    responses = [hello, pc_fail, stop]

    llm = FakeLLM(responses)
    task = _toy_task()
    out = generate_trajectory(task, llm, output_dir=tmp_path,
                              max_solver_turns=5, debug=False, run_judge=False)

    t0 = out["turns"][0]
    assert t0["passed"] is False
    assert t0["tool_output"] is not None, "param-fail tool_output must not be None"
    assert t0["tool_output"].get("status_code") == 400
    assert "error_message" in t0["tool_output"]


def test_stop_reason_max_turns_when_no_stop(tmp_path):
    """If the agent never emits <STOP>, the loop terminates on the turn budget."""
    # Three valid Hello calls in a row, no STOP — budget=3 so we stop at the limit.
    hello = json.dumps({"reason": "again", "tool_call": "Hello(name='alice')"})
    pc_pass = json.dumps({"status": "PASS", "status_code": 200})
    sim_ok  = json.dumps({"status_code": 200, "response": {"ok": True}})
    responses = [hello, pc_pass, sim_ok] * 3  # 3 solver turns, all simulated

    llm = FakeLLM(responses)
    task = _toy_task()
    out = generate_trajectory(task, llm, output_dir=tmp_path,
                              max_solver_turns=3, debug=False, run_judge=False)

    assert out["config"]["stop_reason"] == "max_solver_turns"
    assert len(out["turns"]) == 3
