"""Tests for the running_summary check plumbing in TaskJudge.

Same shape as test_task_evolver_running_summary: we don't test the LLM's
verdict compliance, we test that the role threads running_summary into
the prompt and parses running_summary_grounded out of the response.
"""

from __future__ import annotations

import json

import pytest

from roles.task_judge import TaskJudge


def _judge_response(
    arguments_grounded: bool = True,
    running_summary_grounded: bool = True,
    tool_call_equality: bool = True,
    task_solved: bool = True,
) -> str:
    return "```json\n" + json.dumps({
        "argument_citations": [],
        "arguments_grounded": arguments_grounded,
        "running_summary_grounded": running_summary_grounded,
        "tool_call_equality": tool_call_equality,
        "task_solved": task_solved,
        "task_solved_confidence": 1.0 if task_solved else 0.0,
        "task_solvability": 1.0 if (arguments_grounded and running_summary_grounded) else 0.0,
        "feedback": "",
    }) + "\n```"


def test_judge_renders_running_summary_into_prompt(fake_llm):
    """judge_task_gen must thread its `running_summary` arg into the prompt
    so the LLM can see what a rollout agent would see."""
    judge = TaskJudge(fake_llm)
    fake_llm.queue(_judge_response())

    canary = "RUNNING_SUMMARY_CANARY_xyz789"
    result = judge.judge_task_gen(
        true_tool_call="Tool(journal='Journal of Medical Informatics')",
        current_task_description="now narrow by the journal mentioned earlier",
        prior_chat=[],
        agent_tool_calls=[{"tool_call": "Tool(journal='Journal of Medical Informatics')",
                           "tool_output": {"status_code": 200, "response": {}}}],
        running_summary=canary,
    )

    rendered_prompt = result["prompt"]
    assert canary in rendered_prompt, "running_summary was not rendered into the judge prompt"


def test_judge_parses_running_summary_grounded(fake_llm):
    """The role surfaces running_summary_grounded out of the parsed JSON."""
    judge = TaskJudge(fake_llm)
    fake_llm.queue(_judge_response(running_summary_grounded=False, task_solved=False))

    result = judge.judge_task_gen(
        true_tool_call="Tool(x=42)",
        current_task_description="do the thing",
        prior_chat=[],
        agent_tool_calls=[],
        running_summary="",
    )
    parsed = result["parsed"]
    assert parsed["running_summary_grounded"] is False
    assert parsed["task_solved"] is False


def test_judge_defaults_running_summary_to_empty(fake_llm):
    """Calling without running_summary is OK; the role passes "" through."""
    judge = TaskJudge(fake_llm)
    fake_llm.queue(_judge_response())

    result = judge.judge_task_gen(
        true_tool_call="Tool(x=1)",
        current_task_description="do step 1",
        prior_chat=[],
        agent_tool_calls=[],
    )
    # No exception. Parsed is the canned response.
    assert result["parsed"]["task_solved"] is True


def test_judge_running_summary_grounded_false_independent_of_chat_grounded(fake_llm):
    """The two grounding fields are independent: chat-grounded can be true
    while running-summary-grounded is false (the rollout-leak case)."""
    judge = TaskJudge(fake_llm)
    fake_llm.queue(_judge_response(
        arguments_grounded=True,
        running_summary_grounded=False,
        task_solved=False,
    ))

    result = judge.judge_task_gen(
        true_tool_call="Tool(journal='X')",
        current_task_description="filter by the journal",
        prior_chat=[{"task_description": "search", "tool_call": "Search()",
                     "tool_response": {"journals": ["X", "Y"]}}],
        agent_tool_calls=[],
        running_summary="filter by the journal",   # implicit, "X" missing
    )
    parsed = result["parsed"]
    assert parsed["arguments_grounded"] is True
    assert parsed["running_summary_grounded"] is False
    assert parsed["task_solved"] is False
