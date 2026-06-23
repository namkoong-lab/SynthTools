"""Load verifiable tasks from `task_content.jsonl`.

The release JSONL is produced by `task_audit.run` and is the canonical input
to trajectory_generation. One row per line; each line is a JSON object with
the seven release columns:

  id              string
  field           string
  summary         string
  tools           list[dict]              (parsed tool schemas)
  gt_tool_calls   list[string]            (raw call strings)
  initial_state   dict | null             (env state at the start)
  final_state     dict | null             (env state after the ground truth)

Use `load_task(path, task_id)` for a single row, `iter_tasks(path, ...)` for
streaming, and `list_task_ids(path, ...)` to enumerate ids without parsing
the full payload.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class Task:
    """One verifiable task: a user goal plus its ground-truth solution."""
    id: str
    field: str
    summary: str
    tools: List[Dict[str, Any]]
    gt_tool_calls: List[str]
    initial_state: Optional[Dict[str, Any]]
    final_state: Optional[Dict[str, Any]]


# --- internal parse helpers ---

def _parse_state(raw: Any) -> Optional[Dict[str, Any]]:
    """Accept either a dict (JSONL native) or a JSON-encoded string.
    Returns None for null/empty/invalid input."""
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
    """Accept either a list of dicts (JSONL native) or a list of JSON-encoded
    strings. Skip anything that can't be parsed."""
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


def _iter_rows(jsonl_path: Path) -> Iterator[Dict[str, Any]]:
    """Stream JSON objects from the JSONL file, skipping blank/invalid lines."""
    with Path(jsonl_path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


# --- public API ---

def load_task(jsonl_path: Path, task_id: str) -> Task:
    """Load a single task by id. Raises KeyError if not found.

    Latest occurrence wins if the JSONL contains multiple rows with the same
    id (e.g. after a `--resummarize` run that left duplicate rows; consumers
    deduplicate last-write-wins by convention)."""
    last_row: Optional[Dict[str, Any]] = None
    for row in _iter_rows(jsonl_path):
        if row.get("id") == task_id:
            last_row = row
    if last_row is None:
        raise KeyError(f"task_id {task_id!r} not found in {jsonl_path}")
    return _row_to_task(last_row)


def iter_tasks(
    jsonl_path: Path,
    field: Optional[str] = None,
    limit: Optional[int] = None,
    ids: Optional[List[str]] = None,
) -> Iterator[Task]:
    """Yield tasks from the JSONL, optionally filtered by field or explicit ids.

    Duplicate-id rows are deduplicated last-write-wins (matching `load_task`).
    """
    id_filter = set(ids) if ids is not None else None
    by_id: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for row in _iter_rows(jsonl_path):
        tid = row.get("id")
        if not tid:
            continue
        if field is not None and row.get("field") != field:
            continue
        if id_filter is not None and tid not in id_filter:
            continue
        if tid not in by_id:
            order.append(tid)
        by_id[tid] = row

    n = 0
    for tid in order:
        if limit is not None and n >= int(limit):
            return
        yield _row_to_task(by_id[tid])
        n += 1


def select_random_ids(
    ids: List[str],
    seed: int,
    sample: Optional[int] = None,
) -> List[str]:
    """Deterministically shuffle `ids` by `seed` and take the first `sample`.

    Pure and reproducible: the same (ids, seed) always yields the same order,
    so distinct seeds across jobs/nodes draw distinct (overlapping) subsets.
    `sample=None` returns a full permutation. `sample` larger than the input
    just returns the whole permutation.
    """
    rng = random.Random(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    if sample is None:
        return shuffled
    return shuffled[: max(0, int(sample))]


def list_task_ids(jsonl_path: Path, field: Optional[str] = None) -> List[str]:
    """Return every task id (optionally filtered by field), deduplicated and
    in chronological JSONL order (first occurrence wins for ordering)."""
    seen: set = set()
    out: List[str] = []
    for row in _iter_rows(jsonl_path):
        tid = row.get("id")
        if not tid or tid in seen:
            continue
        if field is not None and row.get("field") != field:
            continue
        seen.add(tid)
        out.append(tid)
    return out
