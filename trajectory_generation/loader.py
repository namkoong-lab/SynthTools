"""Load verifiable tasks from a parquet file.

Schema of `tasks.parquet`:
  id            string
  field         string
  summary       string
  tools         list[string]   (each item is a JSON-encoded tool schema)
  gt_tool_calls list[string]   (raw call strings)
  initial_state string         (JSON-encoded dict, may be null)
  final_state   string         (JSON-encoded dict, may be null)

Use `load_task(path, task_id)` for a single row, `iter_tasks(path, ...)` for streaming,
and `list_task_ids(path, ...)` to enumerate IDs without parsing JSON columns.

This module reads parquet via pyarrow only (no pandas dependency).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


@dataclass
class Task:
    """One verifiable task: a user goal plus its ground-truth solution."""
    id: str
    field: str
    summary: str
    tools: List[Dict[str, Any]]            # parsed from JSON strings
    gt_tool_calls: List[str]               # ground-truth call strings, in order
    initial_state: Optional[Dict[str, Any]]
    final_state: Optional[Dict[str, Any]]


# --- internal parse helpers ---

def _parse_state(raw: Any) -> Optional[Dict[str, Any]]:
    if raw is None or raw == "":
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    return None


def _parse_tools(rows: Any) -> List[Dict[str, Any]]:
    if rows is None:
        return []
    out: List[Dict[str, Any]] = []
    for item in rows:
        if isinstance(item, dict):
            out.append(item)
            continue
        if isinstance(item, str):
            try:
                out.append(json.loads(item))
            except json.JSONDecodeError:
                continue
    return out


def _row_to_task(row: Dict[str, Any]) -> Task:
    return Task(
        id=row["id"],
        field=row.get("field") or "",
        summary=row.get("summary") or "",
        tools=_parse_tools(row.get("tools") or []),
        gt_tool_calls=list(row.get("gt_tool_calls") or []),
        initial_state=_parse_state(row.get("initial_state")),
        final_state=_parse_state(row.get("final_state")),
    )


REQUIRED_COLUMNS = ["id", "field", "summary", "tools", "gt_tool_calls",
                    "initial_state", "final_state"]


def _read_table(parquet_path: Path, columns: Optional[List[str]] = None):
    return pq.read_table(Path(parquet_path), columns=columns or REQUIRED_COLUMNS)


# --- public API ---

def load_task(parquet_path: Path, task_id: str) -> Task:
    """Load a single task by id. Raises KeyError if not found."""
    table = _read_table(parquet_path)
    mask = pc.equal(table["id"], task_id)
    matches = table.filter(mask)
    if matches.num_rows == 0:
        raise KeyError(f"task_id {task_id!r} not found in {parquet_path}")
    if matches.num_rows > 1:
        raise KeyError(f"task_id {task_id!r} matched {matches.num_rows} rows in {parquet_path}")
    rows = matches.to_pylist()
    return _row_to_task(rows[0])


def iter_tasks(
    parquet_path: Path,
    field: Optional[str] = None,
    limit: Optional[int] = None,
    ids: Optional[List[str]] = None,
) -> Iterator[Task]:
    """Yield tasks from the parquet file, optionally filtered by field or explicit ids."""
    table = _read_table(parquet_path)
    if field is not None:
        table = table.filter(pc.equal(table["field"], field))
    if ids is not None:
        table = table.filter(pc.is_in(table["id"], value_set=pa.array(list(ids))))
    if limit is not None:
        table = table.slice(0, int(limit))
    for row in table.to_pylist():
        yield _row_to_task(row)


def list_task_ids(parquet_path: Path, field: Optional[str] = None) -> List[str]:
    """Return every task id (optionally filtered by field) without parsing JSON columns."""
    cols = ["id"] if field is None else ["id", "field"]
    table = _read_table(parquet_path, columns=cols)
    if field is not None:
        table = table.filter(pc.equal(table["field"], field))
    return table["id"].to_pylist()
