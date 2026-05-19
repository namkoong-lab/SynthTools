"""Sharding & concurrent-append safety tests for task_audit.

Three concerns when 5 jobs run --shard 0/5 .. --shard 4/5 against the same
output JSONL:

  1. Partitioning: the 5 shards together cover the full task list with
     no overlap (so no duplicate LLM calls and no missed tasks).
  2. JSONL append safety: concurrent flock-protected appends produce
     intact lines (no byte interleaving of one row across two writers).
  3. Resume safety: a second run of the same shard does NOT re-write
     rows whose id already exists in the JSONL.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

import pytest

# Make the parent package importable when tests run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from task_audit.summarize import (   # noqa: E402
    _existing_release_ids,
    _flush_release_rows,
    _iter_trajectory_paths,
)


# ---------------------------------------------------------------------------
# 1 — Partitioning: shard 0/N .. shard N-1/N covers every task exactly once
# ---------------------------------------------------------------------------

def test_shard_slicing_partitions_tasks_disjoint_and_complete(tmp_path: Path):
    """The same sorted list, sliced by [shard::num_shards] across all
    shards, must cover every element exactly once."""
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    env_specs_dir = tmp_path / "env_specs"
    env_specs_dir.mkdir()

    # 23 tasks (prime, to exercise uneven slicing across 5 shards).
    expected_ids = []
    for i in range(23):
        task_id = f"mock_spec_{i:03d}_seq1"
        (tasks_dir / f"{task_id}.json").write_text("{}")
        expected_ids.append(task_id)

    # All target paths, deterministic order (must match what each shard sees).
    all_paths = _iter_trajectory_paths(tasks_dir, field=None,
                                       env_specs_dir=env_specs_dir)
    # Two independent calls must agree (deterministic across processes).
    again = _iter_trajectory_paths(tasks_dir, field=None,
                                   env_specs_dir=env_specs_dir)
    assert all_paths == again, "task ordering not deterministic across calls"

    num_shards = 5
    shards = {i: all_paths[i::num_shards] for i in range(num_shards)}

    # Union covers all
    union: list = []
    for paths in shards.values():
        union.extend(paths)
    assert sorted(union) == all_paths, "shards do not cover the full task list"

    # Disjoint
    seen: set = set()
    for shard_idx, paths in shards.items():
        for p in paths:
            assert p not in seen, (
                f"task {p.name} assigned to multiple shards "
                f"(double-assignment in shard {shard_idx})"
            )
            seen.add(p)
    assert len(seen) == len(all_paths) == 23

    # And the union of ids equals the expected set.
    ids = {p.stem for p in seen}
    assert ids == set(expected_ids)


def test_shard_slicing_handles_count_smaller_than_shards(tmp_path: Path):
    """If there are fewer tasks than shards, the trailing shards get
    empty slices and nobody double-processes."""
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    env_specs_dir = tmp_path / "env_specs"
    env_specs_dir.mkdir()

    for i in range(3):
        (tasks_dir / f"mock_spec_{i:03d}_seq1.json").write_text("{}")

    all_paths = _iter_trajectory_paths(tasks_dir, field=None,
                                       env_specs_dir=env_specs_dir)
    shards = {i: all_paths[i::5] for i in range(5)}
    # 3 tasks, 5 shards → shards 0..2 get one each, 3..4 get none.
    assert sum(len(p) for p in shards.values()) == 3
    assert len(shards[3]) == 0 and len(shards[4]) == 0
    # Disjoint
    seen: set = set()
    for paths in shards.values():
        for p in paths:
            assert p not in seen
            seen.add(p)


# ---------------------------------------------------------------------------
# 2 — Concurrent JSONL append safety under fcntl.flock
# ---------------------------------------------------------------------------

def _worker_append(worker_id: int, jsonl_path: str, n_rows: int,
                   payload_size: int, resummarize: bool) -> int:
    """Worker entry point — must be top-level for pickling."""
    # Re-import inside the child process (forked workers inherit modules
    # but spawned workers don't, and pytest may switch start methods).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from task_audit.summarize import _flush_release_rows as _flush

    # Build rows with disjoint IDs across workers and a large payload to
    # exceed PIPE_BUF (typically 4096 bytes), so byte interleaving is a
    # real failure mode if locking were absent.
    big = "x" * payload_size
    rows = [
        {
            "id": f"w{worker_id}_task_{i:03d}",
            "field": "F",
            "summary": big,
            "tools": [],
            "gt_tool_calls": [],
            "initial_state": None,
            "final_state": None,
        }
        for i in range(n_rows)
    ]
    counters = _flush(Path(jsonl_path), rows, resummarize=resummarize)
    return counters["appended"]


def test_concurrent_appends_produce_intact_lines(tmp_path: Path):
    """5 processes each appending 100 rows (~5 KB each) to the same
    JSONL must produce 500 fully-parseable lines. flock guarantees
    that a single _flush_release_rows write is not interleaved with
    another process's write."""
    jsonl = tmp_path / "task_content.jsonl"

    n_workers = 5
    n_rows_per_worker = 100
    payload_size = 5000   # > PIPE_BUF; without flock, interleaving is likely.

    ctx = mp.get_context("fork")
    procs = [
        ctx.Process(target=_worker_append,
                    args=(wid, str(jsonl), n_rows_per_worker,
                          payload_size, True))
        for wid in range(n_workers)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0, f"worker {p.pid} exited with {p.exitcode}"

    lines = jsonl.read_text().splitlines()
    assert len(lines) == n_workers * n_rows_per_worker, (
        f"expected {n_workers * n_rows_per_worker} lines, got {len(lines)}"
    )

    # Every line must parse cleanly — no byte-level interleaving.
    ids_seen: set = set()
    for i, line in enumerate(lines):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            pytest.fail(
                f"line {i} is not valid JSON (likely byte interleaving): "
                f"{exc} — line starts: {line[:200]!r}"
            )
        ids_seen.add(obj["id"])

    # Every worker contributed n_rows distinct ids.
    assert len(ids_seen) == n_workers * n_rows_per_worker, (
        f"expected {n_workers * n_rows_per_worker} unique ids, "
        f"got {len(ids_seen)}"
    )
    # And every expected id is present.
    expected = {f"w{wid}_task_{i:03d}"
                for wid in range(n_workers)
                for i in range(n_rows_per_worker)}
    assert ids_seen == expected


def test_concurrent_appends_disjoint_ids_no_dedup_needed(tmp_path: Path):
    """When 5 workers have disjoint ids (the shard invariant), running
    them with resummarize=False (the normal mode) still produces all
    500 rows — none get dropped by the dedup check because no worker's
    ids match any other worker's ids."""
    jsonl = tmp_path / "task_content.jsonl"
    ctx = mp.get_context("fork")
    procs = [
        ctx.Process(target=_worker_append,
                    args=(wid, str(jsonl), 50, 2000, False))
        for wid in range(5)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0

    lines = [line for line in jsonl.read_text().splitlines() if line.strip()]
    assert len(lines) == 5 * 50

    ids = {json.loads(line)["id"] for line in lines}
    assert len(ids) == 5 * 50, "dedup wrongly dropped disjoint-id rows"


# ---------------------------------------------------------------------------
# 3 — Resume safety: a second flush of the same ids is a no-op when
# resummarize=False; --resummarize=True writes the rows again (duplicates
# allowed, downstream dedupes by last-write-wins).
# ---------------------------------------------------------------------------

def test_flush_release_rows_skips_existing_ids_on_resume(tmp_path: Path):
    """Default (resummarize=False): second flush of the same rows
    writes nothing — the existing-id check skips them."""
    jsonl = tmp_path / "task_content.jsonl"
    rows = [
        {"id": "task_001", "field": "F", "summary": "first",
         "tools": [], "gt_tool_calls": [],
         "initial_state": None, "final_state": None},
        {"id": "task_002", "field": "F", "summary": "first",
         "tools": [], "gt_tool_calls": [],
         "initial_state": None, "final_state": None},
    ]

    counters = _flush_release_rows(jsonl, rows, resummarize=False)
    assert counters == {"appended": 2, "skipped": 0}
    assert _existing_release_ids(jsonl) == {"task_001", "task_002"}

    # Re-run with same rows → both skipped.
    counters = _flush_release_rows(jsonl, rows, resummarize=False)
    assert counters == {"appended": 0, "skipped": 2}
    # File still only has the original 2 lines.
    lines = [line for line in jsonl.read_text().splitlines() if line.strip()]
    assert len(lines) == 2


def test_flush_release_rows_resummarize_appends_duplicates(tmp_path: Path):
    """resummarize=True: rows are written regardless of existing ids,
    producing duplicate lines that downstream loaders dedup by
    last-write-wins."""
    jsonl = tmp_path / "task_content.jsonl"
    rows = [
        {"id": "task_001", "field": "F", "summary": "first",
         "tools": [], "gt_tool_calls": [],
         "initial_state": None, "final_state": None},
    ]
    _flush_release_rows(jsonl, rows, resummarize=False)

    # Write again with a different summary under --resummarize semantics.
    rows[0]["summary"] = "second"
    counters = _flush_release_rows(jsonl, rows, resummarize=True)
    assert counters == {"appended": 1, "skipped": 0}

    lines = [line for line in jsonl.read_text().splitlines() if line.strip()]
    assert len(lines) == 2, "resummarize should append, not dedup"
    objs = [json.loads(line) for line in lines]
    assert [o["summary"] for o in objs] == ["first", "second"]


def test_resume_after_partial_crash_writes_only_missing_ids(tmp_path: Path):
    """Simulate a shard that crashed mid-write (only 2 of 5 rows landed).
    On resume the same shard reads existing ids and writes only the 3
    missing ones — no duplicates."""
    jsonl = tmp_path / "task_content.jsonl"
    full_rows = [
        {"id": f"task_{i:03d}", "field": "F", "summary": f"s{i}",
         "tools": [], "gt_tool_calls": [],
         "initial_state": None, "final_state": None}
        for i in range(5)
    ]

    # First "run" stops after 2 rows.
    _flush_release_rows(jsonl, full_rows[:2], resummarize=False)
    assert _existing_release_ids(jsonl) == {"task_000", "task_001"}

    # Resume: shard hands the FULL row list back; only the 3 missing
    # ids should be appended.
    counters = _flush_release_rows(jsonl, full_rows, resummarize=False)
    assert counters == {"appended": 3, "skipped": 2}

    lines = [line for line in jsonl.read_text().splitlines() if line.strip()]
    assert len(lines) == 5
    ids = {json.loads(line)["id"] for line in lines}
    assert ids == {f"task_{i:03d}" for i in range(5)}
