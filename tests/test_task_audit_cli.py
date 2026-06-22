"""CLI argparse smoke tests for task_audit/run.py.

Confirms every flag documented in the README parses without error and
that omitting required flags raises SystemExit. No LLM calls, no real
work; LLM and the two worker functions are monkeypatched out.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest


def _base_argv(tmp_path: Path):
    """The minimum required args for task_audit.run.main()."""
    return [
        "run.py",
        "--tasks-dir", str(tmp_path / "tasks"),
        "--env-specs-dir", str(tmp_path / "specs"),
        "--model", "GPT-OSS-120B",
    ]


def _patch(monkeypatch):
    """Stub LLM, summarize_trajectories, and audit_corpus inside task_audit.run."""
    import task_audit.run as run_mod

    captured: Dict[str, Any] = {}
    monkeypatch.setattr(run_mod, "LLM", lambda *_a, **_k: object())
    monkeypatch.setattr(run_mod, "summarize_trajectories",
                        lambda **kw: (captured.update(kw), None)[1])
    monkeypatch.setattr(run_mod, "audit_corpus",
                        lambda **kw: (captured.update(kw), None)[1])
    return run_mod, captured


def test_minimum_required_args_parses(monkeypatch, tmp_path):
    """Only --tasks-dir + --env-specs-dir + --model is enough to run."""
    run_mod, captured = _patch(monkeypatch)
    monkeypatch.setattr(sys, "argv", _base_argv(tmp_path))
    run_mod.main()
    assert captured["tasks_dir"] == tmp_path / "tasks"
    assert captured["env_specs_dir"] == tmp_path / "specs"


def test_env_specs_dir_is_required(monkeypatch, tmp_path):
    """Omitting --env-specs-dir is a hard error."""
    import task_audit.run as run_mod
    argv = [
        "run.py",
        "--tasks-dir", str(tmp_path / "tasks"),
        "--model", "GPT-OSS-120B",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        run_mod.main()


def test_resummarize_flag_parses(monkeypatch, tmp_path):
    """--resummarize sets the kwarg true; default is false."""
    run_mod, captured = _patch(monkeypatch)
    monkeypatch.setattr(sys, "argv", _base_argv(tmp_path) + ["--resummarize"])
    run_mod.main()
    assert captured["resummarize"] is True


def test_resummarize_default_is_false(monkeypatch, tmp_path):
    """Omitting --resummarize leaves the kwarg false."""
    run_mod, captured = _patch(monkeypatch)
    monkeypatch.setattr(sys, "argv", _base_argv(tmp_path))
    run_mod.main()
    assert captured["resummarize"] is False


def test_shard_and_num_shards_parse(monkeypatch, tmp_path):
    """--shard / --num-shards parse and reach the worker."""
    run_mod, captured = _patch(monkeypatch)
    argv = _base_argv(tmp_path) + ["--shard", "2", "--num-shards", "4"]
    monkeypatch.setattr(sys, "argv", argv)
    run_mod.main()
    assert captured["shard"] == 2
    assert captured["num_shards"] == 4


def test_shard_out_of_range_rejected(monkeypatch, tmp_path):
    """--shard >= --num-shards is rejected by run.py guard."""
    import task_audit.run as run_mod
    monkeypatch.setattr(run_mod, "LLM", lambda *_a, **_k: object())
    argv = _base_argv(tmp_path) + ["--shard", "4", "--num-shards", "4"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        run_mod.main()


def test_task_content_flag_parses(monkeypatch, tmp_path):
    """--task-content overrides the default release JSONL path."""
    run_mod, captured = _patch(monkeypatch)
    custom = tmp_path / "elsewhere" / "task_content.jsonl"
    argv = _base_argv(tmp_path) + ["--task-content", str(custom)]
    monkeypatch.setattr(sys, "argv", argv)
    run_mod.main()
    assert captured["task_content_path"] == custom


def test_audit_only_routes_to_audit_corpus(monkeypatch, tmp_path):
    """--audit-only skips the LLM path and calls audit_corpus instead."""
    import task_audit.run as run_mod

    routed = {"summarize": 0, "audit": 0}
    monkeypatch.setattr(run_mod, "LLM", lambda *_a, **_k: object())
    monkeypatch.setattr(
        run_mod, "summarize_trajectories",
        lambda **kw: (routed.__setitem__("summarize", routed["summarize"] + 1), None)[1],
    )
    monkeypatch.setattr(
        run_mod, "audit_corpus",
        lambda **kw: (routed.__setitem__("audit", routed["audit"] + 1), None)[1],
    )

    argv = _base_argv(tmp_path) + ["--audit-only"]
    monkeypatch.setattr(sys, "argv", argv)
    run_mod.main()
    assert routed == {"summarize": 0, "audit": 1}


def test_server_url_flag_parses(monkeypatch, tmp_path):
    """--server-url is accepted and forwarded to LLM construction."""
    import task_audit.run as run_mod

    seen_kwargs: Dict[str, Any] = {}

    def fake_llm(model, *, server_url=None, **kw):
        seen_kwargs["model"] = model
        seen_kwargs["server_url"] = server_url
        return object()

    monkeypatch.setattr(run_mod, "LLM", fake_llm)
    monkeypatch.setattr(run_mod, "summarize_trajectories", lambda **kw: None)
    argv = _base_argv(tmp_path) + ["--server-url", "http://localhost:8765/v1"]
    monkeypatch.setattr(sys, "argv", argv)
    run_mod.main()
    assert seen_kwargs["server_url"] == "http://localhost:8765/v1"


def test_no_debug_flag_parses(monkeypatch, tmp_path):
    """--no-debug flips write_debug to False; default is True."""
    run_mod, captured = _patch(monkeypatch)
    monkeypatch.setattr(sys, "argv", _base_argv(tmp_path) + ["--no-debug"])
    run_mod.main()
    assert captured["write_debug"] is False


def test_invalid_model_rejected(monkeypatch, tmp_path):
    """--model BOGUS is rejected (MODEL_REGISTRY restriction)."""
    import task_audit.run as run_mod
    argv = [a if a != "GPT-OSS-120B" else "BOGUS_MODEL" for a in _base_argv(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        run_mod.main()
