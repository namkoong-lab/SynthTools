"""Tests for scripts/build_parquet.py — turns summarised task JSONs into
the `tasks.parquet` release artefact. Previously had zero coverage.

The script reads a tasks_dir + env_specs_dir, drops legacy imports, drops
tasks without a `summary`, and emits one parquet row per fresh task.
We exercise the row extractor directly (no parquet write needed)."""

import json
import sys
from pathlib import Path

import pytest

# scripts/ is not a package — pull build_parquet onto sys.path explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_parquet  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _write_task(dir_: Path, task_id: str, task: dict) -> Path:
    p = dir_ / f"{task_id}.json"
    p.write_text(json.dumps(task))
    return p


def _write_spec(dir_: Path, spec_id: str, field: str, tools: list) -> Path:
    p = dir_ / f"{spec_id}.json"
    p.write_text(json.dumps({"spec_id": spec_id, "field": field, "tools": tools}))
    return p


def _good_chat_turn(call: str, response_payload: dict):
    return {
        "tool_id": f"x_spec_000.{call.split('(')[0]}",
        "task": {"task_description": "do the thing",
                 "expected_tool_call": call,
                 "env_metadata": {"seed": "init"}},
        "chat": [
            {"role": "user", "content": "do the thing"},
            {"role": "assistant",
             "content": json.dumps({"reason": "r", "tool_call": call})},
            {"role": "tool",
             "content": json.dumps({"status_code": 200, "response": response_payload})},
        ],
        "env_state_after": {"computed": "value"},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_extract_row_happy_path(tmp_path: Path):
    """A task with summary + turns yields a row with all 7 columns."""
    env_specs = tmp_path / "env_specs"; env_specs.mkdir()
    tasks_dir = tmp_path / "tasks"; tasks_dir.mkdir()

    _write_spec(env_specs, "x_spec_000", "Aerospace and Defense", [
        {"tool_name": "ToolA", "tool_description": "...", "parameters": {}},
    ])
    task_path = _write_task(tasks_dir, "x_spec_000_seq1", {
        "task_id": "x_spec_000_seq1",
        "summary": {"parsed": {"task_summarized": "merged user request"}},
        "turns": [_good_chat_turn("ToolA(x=1)", {"result": "ok"})],
    })

    row = build_parquet.extract_row((str(task_path), str(env_specs)))

    assert row is not None
    assert row["id"] == "x_spec_000_seq1"
    assert row["field"] == "Aerospace and Defense"
    assert row["summary"] == "merged user request"
    assert len(row["tools"]) == 1
    assert "ToolA" in row["tools"][0]
    assert row["gt_tool_calls"] == ["ToolA(x=1)"]
    assert row["initial_state"] is not None
    assert "seed" in row["initial_state"]
    assert row["final_state"] is not None
    assert "computed" in row["final_state"]


def test_extract_row_drops_legacy_imports(tmp_path: Path):
    """A task with `imported_from` is rejected as legacy."""
    env_specs = tmp_path / "env_specs"; env_specs.mkdir()
    tasks_dir = tmp_path / "tasks"; tasks_dir.mkdir()
    _write_spec(env_specs, "x_spec_000", "F", [{"tool_name": "T"}])

    task_path = _write_task(tasks_dir, "x_spec_000_seq1", {
        "task_id": "x_spec_000_seq1",
        "imported_from": "legacy_archive",   # marks it as legacy
        "summary": {"parsed": {"task_summarized": "would be a good summary"}},
        "turns": [_good_chat_turn("T()", {"result": "ok"})],
    })

    assert build_parquet.extract_row((str(task_path), str(env_specs))) is None


def test_extract_row_drops_when_summary_missing(tmp_path: Path):
    """No `summary` field → row dropped (un-summarised tasks aren't released)."""
    env_specs = tmp_path / "env_specs"; env_specs.mkdir()
    tasks_dir = tmp_path / "tasks"; tasks_dir.mkdir()
    _write_spec(env_specs, "x_spec_000", "F", [{"tool_name": "T"}])

    task_path = _write_task(tasks_dir, "x_spec_000_seq1", {
        "task_id": "x_spec_000_seq1",
        # no summary
        "turns": [_good_chat_turn("T()", {"result": "ok"})],
    })

    assert build_parquet.extract_row((str(task_path), str(env_specs))) is None


def test_extract_row_drops_when_summary_text_empty(tmp_path: Path):
    """`summary.parsed.task_summarized == ""` → drop (no point training on it)."""
    env_specs = tmp_path / "env_specs"; env_specs.mkdir()
    tasks_dir = tmp_path / "tasks"; tasks_dir.mkdir()
    _write_spec(env_specs, "x_spec_000", "F", [{"tool_name": "T"}])

    task_path = _write_task(tasks_dir, "x_spec_000_seq1", {
        "task_id": "x_spec_000_seq1",
        "summary": {"parsed": {"task_summarized": ""}},
        "turns": [_good_chat_turn("T()", {"result": "ok"})],
    })

    assert build_parquet.extract_row((str(task_path), str(env_specs))) is None


def test_extract_row_includes_only_tools_actually_used(tmp_path: Path):
    """The `tools` column lists ONLY the tools that appear in turns,
    in invocation order, dedup'd by name."""
    env_specs = tmp_path / "env_specs"; env_specs.mkdir()
    tasks_dir = tmp_path / "tasks"; tasks_dir.mkdir()
    _write_spec(env_specs, "x_spec_000", "F", [
        {"tool_name": "ToolA", "tool_description": "A"},
        {"tool_name": "ToolB", "tool_description": "B"},
        {"tool_name": "ToolC", "tool_description": "C-unused"},
    ])

    task_path = _write_task(tasks_dir, "x_spec_000_seq1", {
        "task_id": "x_spec_000_seq1",
        "summary": {"parsed": {"task_summarized": "merged"}},
        "turns": [
            _good_chat_turn("ToolA(p=1)", {"a": 1}),
            _good_chat_turn("ToolB(q=2)", {"b": 2}),
            _good_chat_turn("ToolA(p=3)", {"a": 3}),  # dedup'd
        ],
    })

    row = build_parquet.extract_row((str(task_path), str(env_specs)))
    assert row is not None
    tools_str = " | ".join(row["tools"])
    assert "ToolA" in tools_str
    assert "ToolB" in tools_str
    assert "ToolC" not in tools_str
    # 2 distinct tools used → 2 entries in `tools`
    assert len(row["tools"]) == 2
    # 3 turns → 3 gt_tool_calls (no dedup of calls)
    assert len(row["gt_tool_calls"]) == 3
