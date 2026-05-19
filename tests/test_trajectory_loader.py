"""Round-trip tests for the JSONL loader in trajectory_generation/loader.py."""

import json
import sys
from pathlib import Path

import pytest

# Make the parent package importable when running tests from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trajectory_generation.loader import (   # noqa: E402
    Task,
    iter_tasks,
    list_task_ids,
    load_task,
)


def _build_test_jsonl(path: Path) -> None:
    """Create a tiny 2-row JSONL matching the task_content release schema.

    Native JSONL form uses parsed dicts/lists (no double-encoded JSON strings)."""
    rows = [
        {
            "id": "test_field_spec_000_seq1",
            "field": "Test Field",
            "summary": "Do thing A then thing B.",
            "tools": [
                {"tool_name": "ToolA", "parameters": {"x": {"type": "string"}}},
                {"tool_name": "ToolB", "parameters": {"y": {"type": "integer"}}},
            ],
            "gt_tool_calls": ["ToolA(x='hello')", "ToolB(y=42)"],
            "initial_state": {"counter": 0},
            "final_state":   {"counter": 1, "last_x": "hello"},
        },
        {
            "id": "other_field_spec_007_seq3",
            "field": "Other Field",
            "summary": "A second task.",
            "tools": [{"tool_name": "ToolC", "parameters": {}}],
            "gt_tool_calls": ["ToolC()"],
            "initial_state": None,
            "final_state":   None,
        },
    ]
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


@pytest.fixture
def test_jsonl(tmp_path: Path) -> Path:
    p = tmp_path / "task_content.jsonl"
    _build_test_jsonl(p)
    return p


def test_load_task_returns_dataclass_with_parsed_fields(test_jsonl: Path) -> None:
    t = load_task(test_jsonl, "test_field_spec_000_seq1")
    assert isinstance(t, Task)
    assert t.id == "test_field_spec_000_seq1"
    assert t.field == "Test Field"
    assert t.summary == "Do thing A then thing B."

    assert isinstance(t.tools, list) and len(t.tools) == 2
    assert all(isinstance(tool, dict) for tool in t.tools)
    assert t.tools[0]["tool_name"] == "ToolA"
    assert t.tools[1]["tool_name"] == "ToolB"

    assert t.gt_tool_calls == ["ToolA(x='hello')", "ToolB(y=42)"]

    assert t.initial_state == {"counter": 0}
    assert t.final_state == {"counter": 1, "last_x": "hello"}


def test_load_task_raises_on_missing_id(test_jsonl: Path) -> None:
    with pytest.raises(KeyError):
        load_task(test_jsonl, "does_not_exist")


def test_iter_tasks_yields_all_rows(test_jsonl: Path) -> None:
    tasks = list(iter_tasks(test_jsonl))
    assert len(tasks) == 2
    assert {t.id for t in tasks} == {
        "test_field_spec_000_seq1",
        "other_field_spec_007_seq3",
    }


def test_iter_tasks_field_filter(test_jsonl: Path) -> None:
    tasks = list(iter_tasks(test_jsonl, field="Other Field"))
    assert len(tasks) == 1 and tasks[0].id == "other_field_spec_007_seq3"


def test_iter_tasks_limit(test_jsonl: Path) -> None:
    tasks = list(iter_tasks(test_jsonl, limit=1))
    assert len(tasks) == 1


def test_iter_tasks_ids_filter(test_jsonl: Path) -> None:
    tasks = list(iter_tasks(test_jsonl, ids=["other_field_spec_007_seq3"]))
    assert len(tasks) == 1 and tasks[0].field == "Other Field"


def test_null_state_fields_become_None(test_jsonl: Path) -> None:
    t = load_task(test_jsonl, "other_field_spec_007_seq3")
    assert t.initial_state is None
    assert t.final_state is None


def test_list_task_ids(test_jsonl: Path) -> None:
    ids = list_task_ids(test_jsonl)
    assert set(ids) == {"test_field_spec_000_seq1", "other_field_spec_007_seq3"}
    only_other = list_task_ids(test_jsonl, field="Other Field")
    assert only_other == ["other_field_spec_007_seq3"]


def test_duplicate_id_uses_last_write(tmp_path: Path) -> None:
    """task_audit's --resummarize can append a second row for the same id;
    the loader deduplicates last-write-wins (matching consumer convention)."""
    p = tmp_path / "task_content.jsonl"
    with p.open("w") as f:
        f.write(json.dumps({
            "id": "dup_spec_000_seq1", "field": "F",
            "summary": "OLD summary",
            "tools": [], "gt_tool_calls": [],
            "initial_state": None, "final_state": None,
        }) + "\n")
        f.write(json.dumps({
            "id": "dup_spec_000_seq1", "field": "F",
            "summary": "NEW summary",
            "tools": [], "gt_tool_calls": [],
            "initial_state": None, "final_state": None,
        }) + "\n")
    t = load_task(p, "dup_spec_000_seq1")
    assert t.summary == "NEW summary"
    # iter_tasks also dedupes
    tasks = list(iter_tasks(p))
    assert len(tasks) == 1 and tasks[0].summary == "NEW summary"


def test_blank_lines_and_invalid_lines_are_skipped(tmp_path: Path) -> None:
    p = tmp_path / "task_content.jsonl"
    p.write_text(
        "\n"
        + json.dumps({"id": "a_spec_000_seq1", "field": "F", "summary": "ok",
                      "tools": [], "gt_tool_calls": [],
                      "initial_state": None, "final_state": None}) + "\n"
        + "this is not json at all\n"
        + "\n"
    )
    ids = list_task_ids(p)
    assert ids == ["a_spec_000_seq1"]
