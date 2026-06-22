"""Unit tests for the chat-format task_summarizer refactor.

These pin the SHAPE of the message list the summarizer sends to the LLM:
- prior turns serialised as (user, assistant, tool) triples
- tool-returned content lives under role=tool, not copied into user-role
  messages by the builder
- a final user message asks for a summary (not a continuation)
- t-style runner gets a list of messages, not a single string
- the loader rejects a legacy single-`template:` YAML
"""

import json
from pathlib import Path

import pytest
import yaml

from roles.task_summarizer import TaskSummarizer


class _RecordingRunner:
    def __init__(self, canned_response: str = '{"task_summarized": "ok"}'):
        self.canned = canned_response
        self.calls = []

    def __call__(self, arg):
        self.calls.append(arg)
        return self.canned


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_build_messages_shape_for_one_turn():
    s = TaskSummarizer(_RecordingRunner())
    msgs = s.build_messages(
        tasks=["Estimate lift requirements."],
        tool_calls=["LiftRequirementEstimator(required_range_km=1500)"],
        tool_responses=['{"status_code":200,"response":{"target_wing_loading_kg_per_m2":71.3}}'],
    )
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool", "user"]


def test_build_messages_shape_for_N_turns():
    s = TaskSummarizer(_RecordingRunner())
    n = 4
    msgs = s.build_messages(
        tasks=[f"task {i}" for i in range(n)],
        tool_calls=[f"Tool{i}()" for i in range(n)],
        tool_responses=[f'{{"status_code":200,"response":{{"v":{i}}}}}' for i in range(n)],
    )
    # 1 system + 3*N turn messages + 1 final user = 1 + 12 + 1 = 14
    assert len(msgs) == 1 + 3 * n + 1
    roles = [m["role"] for m in msgs]
    assert roles[0] == "system"
    assert roles[-1] == "user"
    for i in range(n):
        base = 1 + 3 * i
        assert roles[base:base + 3] == ["user", "assistant", "tool"]


def test_tool_response_lives_only_in_tool_role():
    """A literal value inside a tool response must appear in the `tool`
    message at the same index and NOT leak into any other user-role
    content (other than possibly the final user meta-instruction)."""
    s = TaskSummarizer(_RecordingRunner())
    # A distinctive token only present in tool_responses[1]
    leak = "ZUNICORN_42"
    msgs = s.build_messages(
        tasks=["alpha", "beta"],
        tool_calls=["A()", "B()"],
        tool_responses=[
            '{"status_code":200,"response":{}}',
            json.dumps({"status_code": 200, "response": {"id": leak}}),
        ],
    )
    # Find the tool messages
    tool_idx = [i for i, m in enumerate(msgs) if m["role"] == "tool"]
    assert len(tool_idx) == 2
    # leak must appear in the second tool message
    assert leak in msgs[tool_idx[1]]["content"]
    # leak must NOT appear in any prior-turn user message (the per-turn
    # mini-task descriptions). The final user message is a meta-instruction
    # and is content-free w.r.t. fixture data — skip it from the assertion.
    prior_turn_user_msgs = [m for i, m in enumerate(msgs)
                            if m["role"] == "user" and 0 < i < len(msgs) - 1]
    for m in prior_turn_user_msgs:
        assert leak not in m["content"]


def test_final_user_is_meta_instruction():
    """The final user message must read like 'now write the summary',
    not like another mini-task that continues the chat."""
    s = TaskSummarizer(_RecordingRunner())
    msgs = s.build_messages(tasks=["t"], tool_calls=["T()"], tool_responses=["{}"])
    final = msgs[-1]
    assert final["role"] == "user"
    content = final["content"].lower()
    # Heuristic but robust phrasing checks
    assert "single user-facing request" in content
    assert "chain of operations" in content


# ---------------------------------------------------------------------------
# Integration / runner shape
# ---------------------------------------------------------------------------

def test_summarize_tasks_calls_runner_with_list():
    runner = _RecordingRunner(canned_response=json.dumps({"task_summarized": "ok"}))
    s = TaskSummarizer(runner)
    result = s.summarize_tasks(
        tasks=["t1", "t2"],
        tool_calls=["A()", "B()"],
        tool_responses=["{}", "{}"],
    )
    assert len(runner.calls) == 1
    sent = runner.calls[0]
    assert isinstance(sent, list), f"runner should get a list, got {type(sent).__name__}"
    assert all(isinstance(m, dict) and "role" in m and "content" in m for m in sent)
    assert isinstance(result["prompt"], str)
    decoded = json.loads(result["prompt"])
    assert decoded == sent
    assert result["parsed"] == {"task_summarized": "ok"}


# ---------------------------------------------------------------------------
# Loader behaviour
# ---------------------------------------------------------------------------

def test_load_prompts_rejects_legacy_template_only_yaml(tmp_path, monkeypatch):
    """A legacy single-`template:` summariser YAML must raise."""
    legacy = tmp_path / "task_summarizer_template.yml"
    legacy.write_text(yaml.safe_dump({
        "schema": {"type": "object"},
        "template": "You are a Task Summarizer. {tasks} {tool_calls} {tool_responses}",
    }))
    monkeypatch.setattr(
        "roles.task_summarizer.TASK_SUMMARIZER_TEMPLATE_FILE",
        legacy,
    )
    with pytest.raises(ValueError) as exc:
        TaskSummarizer(lambda x: "")
    assert "system_template" in str(exc.value)
    assert "final_user_template" in str(exc.value)


def test_load_prompts_accepts_current_yaml():
    """The real YAML in the repo loads with both keys populated."""
    s = TaskSummarizer(lambda x: "")
    assert "task_summarizer_system" in s.prompts
    assert "task_summarizer_final_user" in s.prompts
    assert "Task Author" in s.prompts["task_summarizer_system"]
    assert "single user-facing request" in s.prompts["task_summarizer_final_user"].lower()
