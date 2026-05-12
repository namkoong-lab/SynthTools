"""Unit tests for utils.py helpers."""

import json
import logging
from pathlib import Path

import pytest

from utils import (
    RunLog,
    UsageTracker,
    get_logger,
    next_artifact_index,
    to_llm_messages,
    usage_str,
    usage_to_dict,
)


# --- usage_to_dict / usage_str ------------------------------------------------

class _U:
    def __init__(self, p, c):
        self.prompt_tokens = p
        self.completion_tokens = c


def test_usage_to_dict_none():
    assert usage_to_dict(None) is None


def test_usage_to_dict_dict_passthrough():
    d = {"prompt_tokens": 1, "completion_tokens": 2}
    assert usage_to_dict(d) is d


def test_usage_to_dict_dataclass():
    assert usage_to_dict(_U(3, 7)) == {"prompt_tokens": 3, "completion_tokens": 7}


def test_usage_to_dict_unknown_returns_none():
    assert usage_to_dict("not a usage") is None


def test_usage_str_empty_for_none():
    assert usage_str(None) == ""


def test_usage_str_formats_counts():
    assert usage_str(_U(4, 8)) == "[prompt=4, completion=8]"


# --- get_logger ---------------------------------------------------------------

def test_get_logger_idempotent():
    a = get_logger("test_synthtools_logger")
    n_before = len(a.handlers)
    b = get_logger("test_synthtools_logger")
    assert a is b
    assert len(b.handlers) == n_before


def test_get_logger_default_name_level():
    lg = get_logger()
    assert lg.name == "synthtools"
    assert lg.level == logging.INFO


# --- next_artifact_index ------------------------------------------------------

def test_next_index_empty_dir(tmp_path: Path):
    assert next_artifact_index("spec", tmp_path) == 0


def test_next_index_contiguous(tmp_path: Path):
    for i in range(3):
        (tmp_path / f"spec_{i:03d}.json").touch()
    assert next_artifact_index("spec", tmp_path) == 3


def test_next_index_gap_uses_max_plus_one(tmp_path: Path):
    (tmp_path / "spec_000.json").touch()
    (tmp_path / "spec_002.json").touch()
    assert next_artifact_index("spec", tmp_path) == 3


def test_next_index_counts_orphan_debug(tmp_path: Path):
    (tmp_path / "spec_000.json").touch()
    (tmp_path / "spec_007.debug.json").touch()
    assert next_artifact_index("spec", tmp_path) == 8


def test_next_index_ignores_other_prefixes(tmp_path: Path):
    (tmp_path / "spec_000.json").touch()
    (tmp_path / "other_999.json").touch()
    assert next_artifact_index("spec", tmp_path) == 1


def test_next_index_nonexistent_dir(tmp_path: Path):
    missing = tmp_path / "does_not_exist"
    assert next_artifact_index("spec", missing) == 0


# --- to_llm_messages ----------------------------------------------------------

def test_to_llm_messages_rewrites_tool_role():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
        {"role": "tool", "content": "t"},
        {"role": "user", "content": "u2"},
    ]
    out = to_llm_messages(msgs)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "user", "user"]
    assert out[3]["content"] == "Tool response: t"
    # Originals untouched
    assert msgs[3]["role"] == "tool"


def test_to_llm_messages_empty():
    assert to_llm_messages([]) == []


# --- RunLog -------------------------------------------------------------------

def test_runlog_record_and_save_roundtrip(tmp_path: Path):
    log = RunLog("task_abc")
    log.record(
        agent="TaskSolver",
        action="solve_turn_0",
        turn_ref={"tool_idx": 0, "attempt": 0},
        result={"prompt": "p", "response": "r", "parsed": {"k": 1}, "usage": _U(5, 6)},
    )
    path = log.save(tmp_path)
    assert path.name == "task_abc.debug.json"
    with open(path) as f:
        data = json.load(f)
    assert data["task_id"] == "task_abc"
    assert len(data["events"]) == 1
    ev = data["events"][0]
    assert ev["agent"] == "TaskSolver"
    assert ev["action"] == "solve_turn_0"
    assert ev["turn_ref"] == {"tool_idx": 0, "attempt": 0}
    assert ev["usage"] == {"prompt_tokens": 5, "completion_tokens": 6}


# --- UsageTracker -------------------------------------------------------------

def test_tracker_noop_on_none():
    t = UsageTracker()
    t.track(None)
    assert t.total == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def test_tracker_accumulates_dataclasses():
    t = UsageTracker()
    t.track({"usage": _U(3, 4)})
    t.track({"usage": _U(10, 20)})
    assert t.total == {"prompt_tokens": 13, "completion_tokens": 24, "total_tokens": 37}


def test_tracker_handles_nested_check_simulation():
    """ToolSimulator shape: {'usage': {'check': Usage, 'simulation': Usage}}."""
    t = UsageTracker()
    t.track({"usage": {"check": _U(1, 2), "simulation": _U(5, 10)}})
    assert t.total == {"prompt_tokens": 6, "completion_tokens": 12, "total_tokens": 18}


def test_tracker_tracks_bare_usage_objects():
    t = UsageTracker()
    t.track(_U(1, 1))
    assert t.total == {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


# --- batch_call ---------------------------------------------------------------

def test_batch_call_empty_prompts_list():
    from utils import batch_call
    assert batch_call(object(), []) == []


def test_batch_call_single_prompt_uses_non_batched_path(fake_llm):
    from utils import batch_call
    fake_llm.queue("solo response")
    out = batch_call(fake_llm, ["just one"])
    assert len(out) == 1
    assert out[0]["response"] == "solo response"
    # single path → per_request stays None
    assert fake_llm.last_usage_per_request is None
    # And the recorded call was NOT a batched shape
    assert fake_llm.calls[-1]["batched"] is False


def test_batch_call_many_prompts_batches_and_returns_per_request_usage(fake_llm):
    from utils import batch_call
    fake_llm.queue_batch(["r1", "r2", "r3"])
    out = batch_call(fake_llm, ["p1", "p2", "p3"])
    assert [d["response"] for d in out] == ["r1", "r2", "r3"]
    # Each dict has a non-None usage derived from last_usage_per_request
    for d in out:
        assert d["usage"] is not None
        assert hasattr(d["usage"], "prompt_tokens")
    # Batched shape was used
    assert fake_llm.calls[-1]["batched"] is True
    assert len(fake_llm.calls[-1]["messages"]) == 3


# --- batch_call chat-format path -------------------------------------------

def _chat(text: str):
    return [{"role": "user", "content": f"sys-{text}"},
            {"role": "tool", "content": f"output for {text}"}]


def test_batch_call_chat_format_single_prompt(fake_llm):
    """A single chat (List[Dict]) goes through the non-batched path and
    is forwarded to the LLM verbatim (NOT re-wrapped as one user msg)."""
    from utils import batch_call
    fake_llm.queue("solo chat response")
    out = batch_call(fake_llm, [_chat("alpha")])
    assert len(out) == 1
    assert out[0]["response"] == "solo chat response"
    assert fake_llm.calls[-1]["batched"] is False
    sent = fake_llm.calls[-1]["messages"]
    # Original chat preserved — both messages there, in order, untouched
    assert sent == _chat("alpha")


def test_batch_call_chat_format_multiple(fake_llm):
    """Multiple chats batch via the LLM's batched path (List[List[Dict]])."""
    from utils import batch_call
    fake_llm.queue_batch(["r1", "r2"])
    out = batch_call(fake_llm, [_chat("p1"), _chat("p2")])
    assert [d["response"] for d in out] == ["r1", "r2"]
    assert fake_llm.calls[-1]["batched"] is True
    # The batched payload must be a list of two chats (each a list of dicts)
    batched_payload = fake_llm.calls[-1]["messages"]
    assert len(batched_payload) == 2
    assert batched_payload[0] == _chat("p1")
    assert batched_payload[1] == _chat("p2")


def test_batch_call_chat_format_per_request_usage(fake_llm):
    """Per-request usage tracking works the same on the chat-format path."""
    from utils import batch_call
    fake_llm.queue_batch(["r1", "r2", "r3"])
    out = batch_call(fake_llm, [_chat("a"), _chat("b"), _chat("c")])
    for d in out:
        assert d["usage"] is not None
        assert hasattr(d["usage"], "prompt_tokens")
        assert hasattr(d["usage"], "completion_tokens")
