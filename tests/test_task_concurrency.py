"""Tests for parallel task generation orchestration.

Strategy: test the orchestration logic (work-queue building, CLI argument
validation, the serial-path fast-track) without actually spawning child
processes. Real ProcessPoolExecutor + FakeLLM doesn't pickle cleanly, and
the task-generation behavior itself is already covered by
test_task_regression.py — what's new here is the controller logic.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from task_generation.generate import (
    list_pending_work_for_spec,
    list_pending_work_for_field,
)


# ---------------------------------------------------------------------------
# Helpers: minimal env_spec builder
# ---------------------------------------------------------------------------

def _write_spec(path: Path, spec_id: str, field: str, sequences: dict) -> None:
    path.write_text(json.dumps({
        "spec_id": spec_id,
        "field": field,
        "tools": [],
        "sequences": sequences,
    }))


# ---------------------------------------------------------------------------
# list_pending_work_for_spec
# ---------------------------------------------------------------------------

def test_list_pending_work_for_spec_returns_all_when_none_done(tmp_path: Path):
    spec_path = tmp_path / "spec.json"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _write_spec(spec_path, "ab_spec_000", "AB", {
        "seq1": ["ToolA", "ToolB"],
        "seq2": ["ToolC"],
    })

    items = list_pending_work_for_spec(spec_path, output_dir)

    assert len(items) == 2
    assert {i.task_id for i in items} == {"ab_spec_000_seq1", "ab_spec_000_seq2"}
    assert items[0].tool_ids == ["ab_spec_000.ToolA", "ab_spec_000.ToolB"]


def test_list_pending_work_for_spec_skips_existing(tmp_path: Path):
    spec_path = tmp_path / "spec.json"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _write_spec(spec_path, "ab_spec_000", "AB", {
        "seq1": ["ToolA"],
        "seq2": ["ToolB"],
        "seq3": ["ToolC"],
    })
    # seq1 already done
    (output_dir / "ab_spec_000_seq1.json").write_text("{}")

    items = list_pending_work_for_spec(spec_path, output_dir)

    assert {i.task_id for i in items} == {"ab_spec_000_seq2", "ab_spec_000_seq3"}


def test_list_pending_work_for_spec_skips_invalid_sequences(tmp_path: Path):
    spec_path = tmp_path / "spec.json"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _write_spec(spec_path, "ab_spec_000", "AB", {
        "seq1": [],                # empty list — invalid
        "seq2": "not a list",      # wrong type — invalid
        "seq3": ["ToolC"],         # valid
    })

    items = list_pending_work_for_spec(spec_path, output_dir)

    assert len(items) == 1
    assert items[0].task_id == "ab_spec_000_seq3"


def test_list_pending_work_for_spec_respects_max_tasks(tmp_path: Path):
    spec_path = tmp_path / "spec.json"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _write_spec(spec_path, "ab_spec_000", "AB", {
        f"seq{i}": ["ToolA"] for i in range(1, 6)  # 5 sequences
    })

    items = list_pending_work_for_spec(spec_path, output_dir, max_tasks=2)

    assert len(items) == 2


def test_list_pending_work_for_spec_no_sequences_returns_empty(tmp_path: Path):
    spec_path = tmp_path / "spec.json"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _write_spec(spec_path, "ab_spec_000", "AB", {})

    items = list_pending_work_for_spec(spec_path, output_dir)

    assert items == []


def test_list_pending_work_ignores_tmp_files(tmp_path: Path):
    """A leftover .tmp from a crashed worker should not satisfy the resume check."""
    spec_path = tmp_path / "spec.json"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _write_spec(spec_path, "ab_spec_000", "AB", {
        "seq1": ["ToolA"],
    })
    # A stray .tmp from a crashed worker
    (output_dir / "ab_spec_000_seq1.json.tmp").write_text("partial")

    items = list_pending_work_for_spec(spec_path, output_dir)

    # The work item is still pending — only the final .json blocks resume.
    assert len(items) == 1
    assert items[0].task_id == "ab_spec_000_seq1"


# ---------------------------------------------------------------------------
# list_pending_work_for_field
# ---------------------------------------------------------------------------

def test_list_pending_work_for_field_aggregates_specs(tmp_path: Path):
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    _write_spec(specs_dir / "ab_spec_000.json", "ab_spec_000", "AB",
                {"seq1": ["X"], "seq2": ["Y"]})
    _write_spec(specs_dir / "ab_spec_001.json", "ab_spec_001", "AB",
                {"seq1": ["Z"]})
    _write_spec(specs_dir / "cd_spec_000.json", "cd_spec_000", "CD",
                {"seq1": ["W"]})

    items = list_pending_work_for_field(specs_dir, "AB", output_dir)

    task_ids = {i.task_id for i in items}
    assert task_ids == {
        "ab_spec_000_seq1",
        "ab_spec_000_seq2",
        "ab_spec_001_seq1",
    }


def test_list_pending_work_for_field_excludes_other_fields(tmp_path: Path):
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    _write_spec(specs_dir / "ab_spec_000.json", "ab_spec_000", "AB",
                {"seq1": ["X"]})
    _write_spec(specs_dir / "cd_spec_000.json", "cd_spec_000", "CD",
                {"seq1": ["Y"]})

    items = list_pending_work_for_field(specs_dir, "CD", output_dir)
    assert len(items) == 1
    assert items[0].task_id == "cd_spec_000_seq1"


# ---------------------------------------------------------------------------
# CLI argument validation
# ---------------------------------------------------------------------------

def _run_cli(argv):
    """Invoke run.main() with the given argv; return SystemExit code."""
    from task_generation import run as run_module
    saved = sys.argv
    sys.argv = ["task_generation.run"] + argv
    try:
        run_module.main()
    finally:
        sys.argv = saved


def test_run_concurrency_requires_server_url(tmp_path: Path, capsys):
    spec_path = tmp_path / "spec.json"
    _write_spec(spec_path, "ab_spec_000", "AB", {"seq1": ["X"]})
    dataset = tmp_path / "tools.jsonl"
    dataset.write_text("")

    with pytest.raises(SystemExit) as excinfo:
        _run_cli([
            "--dataset", str(dataset),
            "--output-dir", str(tmp_path / "out"),
            "--model", "Qwen3-32B",
            "--env-spec", str(spec_path),
            "--concurrency", "4",
        ])
    assert excinfo.value.code != 0
    err = capsys.readouterr().err
    assert "--server-url" in err


def test_run_mode_a_rejects_concurrency(tmp_path: Path, capsys):
    dataset = tmp_path / "tools.jsonl"
    dataset.write_text("")

    with pytest.raises(SystemExit) as excinfo:
        _run_cli([
            "--dataset", str(dataset),
            "--output-dir", str(tmp_path / "out"),
            "--model", "Qwen3-32B",
            "--tool-ids", "tool1",
            "--server-url", "http://fake/v1",
            "--concurrency", "2",
        ])
    assert excinfo.value.code != 0
    err = capsys.readouterr().err
    assert "--tool-ids" in err


def test_run_concurrency_zero_rejected(tmp_path: Path, capsys):
    dataset = tmp_path / "tools.jsonl"
    dataset.write_text("")
    spec_path = tmp_path / "spec.json"
    _write_spec(spec_path, "ab_spec_000", "AB", {"seq1": ["X"]})

    with pytest.raises(SystemExit):
        _run_cli([
            "--dataset", str(dataset),
            "--output-dir", str(tmp_path / "out"),
            "--model", "Qwen3-32B",
            "--env-spec", str(spec_path),
            "--concurrency", "0",
        ])


def test_concurrency_one_uses_serial_path(tmp_path: Path):
    """With --concurrency 1, the serial path is taken (no run_parallel call)."""
    spec_path = tmp_path / "spec.json"
    _write_spec(spec_path, "ab_spec_000", "AB", {"seq1": ["X"]})
    dataset = tmp_path / "tools.jsonl"
    dataset.write_text("")
    out_dir = tmp_path / "out"

    with patch("task_generation.run.run_parallel") as mock_parallel, \
         patch("task_generation.run.LLM") as mock_llm, \
         patch("task_generation.run.generate_trajectories_for_spec") as mock_gen:
        mock_gen.return_value = []
        _run_cli([
            "--dataset", str(dataset),
            "--output-dir", str(out_dir),
            "--model", "Qwen3-32B",
            "--env-spec", str(spec_path),
            "--concurrency", "1",
        ])
    assert mock_parallel.call_count == 0
    # Serial path goes through generate_trajectories_for_spec
    assert mock_gen.call_count == 1


def test_redirect_synthtools_logger_to_file_writes_to_path(tmp_path: Path):
    """The redirect helper swaps handlers; subsequent log lines land in the file."""
    import logging
    from utils import redirect_synthtools_logger_to_file, get_logger

    log_path = tmp_path / "_logs" / "task_abc.log"
    redirect_synthtools_logger_to_file(log_path)

    logger = get_logger("synthtools")
    logger.info("hello from worker")
    # Force flush by closing the handler.
    for h in logger.handlers:
        h.flush()

    assert log_path.exists()
    content = log_path.read_text()
    assert "hello from worker" in content


def test_redirect_synthtools_logger_does_not_affect_run_logger(tmp_path: Path):
    """The task_generation.run logger keeps its own stderr handler — parent stays clean."""
    import logging
    from utils import redirect_synthtools_logger_to_file, get_logger

    parent_logger = get_logger("task_generation.run")
    parent_handlers_before = list(parent_logger.handlers)

    redirect_synthtools_logger_to_file(tmp_path / "task.log")

    assert list(parent_logger.handlers) == parent_handlers_before


def test_concurrency_greater_than_one_uses_parallel_path(tmp_path: Path):
    """With --concurrency 2 and --server-url, the parallel path is taken."""
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    spec_path = specs_dir / "ab_spec_000.json"
    _write_spec(spec_path, "ab_spec_000", "AB", {"seq1": ["X"], "seq2": ["Y"]})
    dataset = tmp_path / "tools.jsonl"
    dataset.write_text("")
    out_dir = tmp_path / "out"

    with patch("task_generation.run._run_parallel") as mock_parallel, \
         patch("task_generation.run.generate_trajectories_for_spec") as mock_serial:
        mock_parallel.return_value = []
        _run_cli([
            "--dataset", str(dataset),
            "--output-dir", str(out_dir),
            "--model", "Qwen3-32B",
            "--env-spec", str(spec_path),
            "--server-url", "http://fake/v1",
            "--concurrency", "2",
        ])
    assert mock_parallel.call_count == 1
    assert mock_serial.call_count == 0
    items_arg = mock_parallel.call_args.kwargs["items"]
    assert len(items_arg) == 2
    assert mock_parallel.call_args.kwargs["concurrency"] == 2
