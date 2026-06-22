"""CLI tests for the new --max-solver-retries-in-place flag.

The inner solver-retry budget was previously unreachable from the CLI
(only the `generate_trajectory` kwarg existed). These tests pin the
new behaviour:

  - argparse accepts the flag and stores it as an int.
  - Default value is sourced from `config.DEFAULT_MAX_SOLVER_RETRIES_IN_PLACE`.
  - Value reaches `generate_trajectory` via task_kwargs unchanged.
  - The flag composes with `--max-solver-turns`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest


def _patch_main(monkeypatch, captured: Dict[str, Any]):
    """Replace LLM constructor and generate_trajectory so main() returns fast."""
    import task_generation.run as run_mod

    monkeypatch.setattr(run_mod, "LLM", lambda *_a, **_k: object())

    def fake_generate_trajectory(**kwargs):
        captured.update(kwargs)
        return {"usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}

    monkeypatch.setattr(run_mod, "generate_trajectory", fake_generate_trajectory)
    monkeypatch.setattr(run_mod, "generate_trajectories_for_spec", lambda **kw: None)
    monkeypatch.setattr(run_mod, "generate_trajectories_for_field", lambda **kw: None)
    return run_mod


def _base_argv(tmp_path: Path):
    """Mode A invocation (single explicit tool list).

    `--concurrency 1` forces the serial path that calls generate_trajectory
    directly; without it the parallel path requires --server-url.
    """
    return [
        "run.py",
        "--dataset", str(tmp_path / "tools_dataset.jsonl"),
        "--output-dir", str(tmp_path / "tasks"),
        "--model", "GPT-OSS-120B",
        "--tool-ids", "spec_id.Alpha",
        "--concurrency", "1",
    ]


def test_flag_default_is_two(monkeypatch, tmp_path):
    """Omitting --max-solver-retries-in-place uses default 2."""
    captured: Dict[str, Any] = {}
    run_mod = _patch_main(monkeypatch, captured)
    monkeypatch.setattr(sys, "argv", _base_argv(tmp_path))
    run_mod.main()
    assert captured.get("max_solver_retries_in_place") == 2


def test_flag_explicit_value_is_forwarded(monkeypatch, tmp_path):
    """--max-solver-retries-in-place 7 reaches generate_trajectory as kwarg=7."""
    captured: Dict[str, Any] = {}
    run_mod = _patch_main(monkeypatch, captured)
    argv = _base_argv(tmp_path) + ["--max-solver-retries-in-place", "7"]
    monkeypatch.setattr(sys, "argv", argv)
    run_mod.main()
    assert captured.get("max_solver_retries_in_place") == 7


def test_flag_rejects_non_int(monkeypatch, tmp_path):
    """argparse rejects a non-integer value with SystemExit."""
    import task_generation.run as run_mod
    argv = _base_argv(tmp_path) + ["--max-solver-retries-in-place", "abc"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        run_mod.main()


def test_flag_composes_with_max_solver_turns(monkeypatch, tmp_path):
    """Both retry-budget flags can be set on the same invocation; both reach kwargs."""
    captured: Dict[str, Any] = {}
    run_mod = _patch_main(monkeypatch, captured)
    argv = _base_argv(tmp_path) + [
        "--max-solver-turns", "8",
        "--max-solver-retries-in-place", "3",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    run_mod.main()
    assert captured.get("max_solver_turns") == 8
    assert captured.get("max_solver_retries_in_place") == 3


def test_flag_default_matches_config_constant(monkeypatch, tmp_path):
    """The argparse default value is sourced from DEFAULT_MAX_SOLVER_RETRIES_IN_PLACE."""
    from config import DEFAULT_MAX_SOLVER_RETRIES_IN_PLACE
    captured: Dict[str, Any] = {}
    run_mod = _patch_main(monkeypatch, captured)
    monkeypatch.setattr(sys, "argv", _base_argv(tmp_path))
    run_mod.main()
    assert captured.get("max_solver_retries_in_place") == DEFAULT_MAX_SOLVER_RETRIES_IN_PLACE
