"""Tests for the running_summary plumbing in TaskEvolver.

The LLM compliance with the new prompt rules (e.g. "must contain every
value in running_summary_prev") is enforced by the judge at runtime;
these tests only pin the CODE plumbing:

- t0 reads `running_summary` from the parsed output.
- t1 accepts a `running_summary_prev` kwarg and renders it into the
  final user message so the LLM can see it.
- t1 reads `running_summary` from the parsed output.
"""

from __future__ import annotations

import json

import pytest

from roles.task_evolver import TaskEvolver


def _t0_response(task_description: str, running_summary: str) -> str:
    return "```json\n" + json.dumps({
        "tool_name": "Tool",
        "tool_call": "Tool(x=1)",
        "env_metadata": {"x": 1},
        "task_description": task_description,
        "running_summary": running_summary,
    }) + "\n```"


def _t1_response(task_description: str, running_summary: str) -> str:
    return "```json\n" + json.dumps({
        "tool_name": "Tool",
        "tool_call": "Tool(y=2)",
        "edited_metadata": {},
        "env_metadata": {"x": 1, "y": 2},
        "task_description": task_description,
        "running_summary": running_summary,
        "depends_on_previous": True,
    }) + "\n```"


# ---------------------------------------------------------------------------
# t0
# ---------------------------------------------------------------------------

def test_t0_parses_running_summary(fake_llm):
    """evolve_task_t0 surfaces running_summary from the parsed output."""
    evolver = TaskEvolver(fake_llm)
    fake_llm.queue(_t0_response("do step 1", "do step 1"))

    result = evolver.evolve_task_t0({"tool_name": "Tool"})
    parsed = result["parsed"]
    assert parsed["task_description"] == "do step 1"
    assert parsed["running_summary"] == "do step 1"


def test_t0_running_summary_can_diverge_if_llm_does(fake_llm):
    """The role does not enforce equality at t0; it trusts the LLM.

    Enforcement of "running_summary equals task_description at t0" is a
    prompt-level rule. The role just parses what the LLM emits.
    """
    evolver = TaskEvolver(fake_llm)
    fake_llm.queue(_t0_response("do step 1", "DIFFERENT TEXT"))

    result = evolver.evolve_task_t0({"tool_name": "Tool"})
    assert result["parsed"]["running_summary"] == "DIFFERENT TEXT"


# ---------------------------------------------------------------------------
# t1
# ---------------------------------------------------------------------------

def test_t1_threads_running_summary_prev_into_final_user_message(fake_llm):
    """evolve_task_t1 must render running_summary_prev into the final user
    chat message so the LLM sees the cumulative-summary-so-far text."""
    evolver = TaskEvolver(fake_llm)
    fake_llm.queue(_t1_response(
        task_description="do step 2",
        running_summary="step 1 prose. step 2 prose.",
    ))

    prev = "step 1 prose."
    successful_task = [
        {"task_description": "do step 1", "tool_call": "Tool(x=1)",
         "tool_simulated": {"status_code": 200, "response": {"ok": True}}},
    ]
    result = evolver.evolve_task_t1(
        successful_task=successful_task,
        unsuccessful_tasks=[],
        tool_details={"tool_name": "Tool"},
        environment_state={"x": 1},
        running_summary_prev=prev,
    )

    # The role serialises the chat into result["prompt"] (a JSON string of
    # the message list). The running_summary_prev must appear inside it.
    rendered = result["prompt"]
    assert prev in rendered, "running_summary_prev was not threaded into the chat"

    parsed = result["parsed"]
    assert parsed["task_description"] == "do step 2"
    assert parsed["running_summary"] == "step 1 prose. step 2 prose."


def test_t1_defaults_running_summary_prev_to_empty(fake_llm):
    """Omitting running_summary_prev is the same as passing ""."""
    evolver = TaskEvolver(fake_llm)
    fake_llm.queue(_t1_response(task_description="do step 1", running_summary="do step 1"))

    result = evolver.evolve_task_t1(
        successful_task=[],
        unsuccessful_tasks=[],
        tool_details={"tool_name": "Tool"},
        environment_state={},
    )
    # No exception, parsed available.
    assert result["parsed"]["running_summary"] == "do step 1"


def test_t1_running_summary_prev_is_distinct_field_in_chat(fake_llm):
    """The role must put running_summary_prev under the labelled slot in the
    final user template (so it's NOT confused with the prior-tool messages)."""
    evolver = TaskEvolver(fake_llm)
    fake_llm.queue(_t1_response("step 2", "rs prev. step 2"))

    distinct_prev = "RUNNING_SUMMARY_PREV_CANARY_STRING_42"
    evolver.evolve_task_t1(
        successful_task=[],
        unsuccessful_tasks=[],
        tool_details={"tool_name": "Tool"},
        environment_state={},
        running_summary_prev=distinct_prev,
    )

    # The runner saw exactly one call; the final user message in that call
    # must contain the canary string.
    assert len(fake_llm.calls) == 1
    messages = fake_llm.calls[0]["messages"]
    final_user = messages[-1]
    assert final_user["role"] == "user"
    assert distinct_prev in final_user["content"]
