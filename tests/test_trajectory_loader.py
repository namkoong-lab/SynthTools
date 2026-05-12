"""Round-trip tests for the parquet loader in trajectory_generation/loader.py."""

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# Make the parent package importable when running tests from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trajectory_generation.loader import (   # noqa: E402
    Task,
    iter_tasks,
    list_task_ids,
    load_task,
)


def _build_test_parquet(path: Path) -> None:
    """Create a tiny 2-row parquet matching the published schema."""
    rows = [
        {
            "id": "test_field_spec_000_seq1",
            "field": "Test Field",
            "summary": "Do thing A then thing B.",
            "tools": [
                json.dumps({"tool_name": "ToolA", "parameters": {"x": {"type": "string"}}}),
                json.dumps({"tool_name": "ToolB", "parameters": {"y": {"type": "integer"}}}),
            ],
            "gt_tool_calls": ["ToolA(x='hello')", "ToolB(y=42)"],
            "initial_state": json.dumps({"counter": 0}),
            "final_state":   json.dumps({"counter": 1, "last_x": "hello"}),
        },
        {
            "id": "other_field_spec_007_seq3",
            "field": "Other Field",
            "summary": "A second task.",
            "tools": [json.dumps({"tool_name": "ToolC", "parameters": {}})],
            "gt_tool_calls": ["ToolC()"],
            "initial_state": None,
            "final_state":   None,
        },
    ]
    schema = pa.schema([
        ("id",            pa.string()),
        ("field",         pa.string()),
        ("summary",       pa.string()),
        ("tools",         pa.list_(pa.string())),
        ("gt_tool_calls", pa.list_(pa.string())),
        ("initial_state", pa.string()),
        ("final_state",   pa.string()),
    ])
    cols = {k: [r[k] for r in rows] for k in schema.names}
    pq.write_table(pa.table(cols, schema=schema), path)


@pytest.fixture
def test_parquet(tmp_path: Path) -> Path:
    p = tmp_path / "test_tasks.parquet"
    _build_test_parquet(p)
    return p


def test_load_task_returns_dataclass_with_parsed_fields(test_parquet: Path) -> None:
    t = load_task(test_parquet, "test_field_spec_000_seq1")
    assert isinstance(t, Task)
    assert t.id == "test_field_spec_000_seq1"
    assert t.field == "Test Field"
    assert t.summary == "Do thing A then thing B."

    # tools were JSON strings — they must come back parsed as dicts
    assert isinstance(t.tools, list) and len(t.tools) == 2
    assert all(isinstance(tool, dict) for tool in t.tools)
    assert t.tools[0]["tool_name"] == "ToolA"
    assert t.tools[1]["tool_name"] == "ToolB"

    # gt_tool_calls stays as raw strings
    assert t.gt_tool_calls == ["ToolA(x='hello')", "ToolB(y=42)"]

    # state was JSON-encoded — parsed back to dict
    assert t.initial_state == {"counter": 0}
    assert t.final_state == {"counter": 1, "last_x": "hello"}


def test_load_task_raises_on_missing_id(test_parquet: Path) -> None:
    with pytest.raises(KeyError):
        load_task(test_parquet, "does_not_exist")


def test_iter_tasks_yields_all_rows(test_parquet: Path) -> None:
    tasks = list(iter_tasks(test_parquet))
    assert len(tasks) == 2
    assert {t.id for t in tasks} == {
        "test_field_spec_000_seq1",
        "other_field_spec_007_seq3",
    }


def test_iter_tasks_field_filter(test_parquet: Path) -> None:
    tasks = list(iter_tasks(test_parquet, field="Other Field"))
    assert len(tasks) == 1 and tasks[0].id == "other_field_spec_007_seq3"


def test_iter_tasks_limit(test_parquet: Path) -> None:
    tasks = list(iter_tasks(test_parquet, limit=1))
    assert len(tasks) == 1


def test_iter_tasks_ids_filter(test_parquet: Path) -> None:
    tasks = list(iter_tasks(test_parquet, ids=["other_field_spec_007_seq3"]))
    assert len(tasks) == 1 and tasks[0].field == "Other Field"


def test_null_state_fields_become_None(test_parquet: Path) -> None:
    t = load_task(test_parquet, "other_field_spec_007_seq3")
    assert t.initial_state is None
    assert t.final_state is None


def test_list_task_ids(test_parquet: Path) -> None:
    ids = list_task_ids(test_parquet)
    assert set(ids) == {"test_field_spec_000_seq1", "other_field_spec_007_seq3"}
    only_other = list_task_ids(test_parquet, field="Other Field")
    assert only_other == ["other_field_spec_007_seq3"]
