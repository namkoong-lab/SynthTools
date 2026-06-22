"""CLI smoke tests for env_generation/run.py.

Confirms the CLI's argparse accepts the README-documented invocation
(`--field A --field B`) and rejects the old wrong form (`--fields A,B`).
Also verifies top-up re-run idempotency through the CLI path: running
twice with the same flags produces the same set of spec files.
"""

import json
import sys

import pytest


def _patch_llm_and_generate(monkeypatch, fake_llm):
    """Replace LLM and generate_environments inside env_generation.run."""
    import env_generation.run as run_mod

    monkeypatch.setattr(run_mod, "LLM", lambda *_a, **_k: fake_llm)
    return run_mod


def test_cli_field_flag_repeatable(monkeypatch, tmp_path):
    """`--field A --field B` parses; both names reach generate_environments."""
    import env_generation.run as run_mod

    captured = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(run_mod, "LLM", lambda *_a, **_k: object())
    monkeypatch.setattr(run_mod, "generate_environments", fake_generate)
    monkeypatch.setattr(sys, "argv", [
        "run.py",
        "--field", "Aerospace and Defense",
        "--field", "Healthcare",
        "--output-dir", str(tmp_path / "specs"),
        "--model", "GPT-OSS-120B",
    ])

    run_mod.main()
    assert captured["fields"] == ["Aerospace and Defense", "Healthcare"]


def test_cli_rejects_legacy_fields_form(monkeypatch, tmp_path):
    """The old README form `--fields A,B` is rejected (argparse has no such flag)."""
    import env_generation.run as run_mod

    monkeypatch.setattr(sys, "argv", [
        "run.py",
        "--fields", "Healthcare,Insurance",
        "--output-dir", str(tmp_path / "specs"),
        "--model", "GPT-OSS-120B",
    ])
    with pytest.raises(SystemExit):
        run_mod.main()


def _fenced(items):
    return "```json\n" + json.dumps(items) + "\n```"


def _tool_blocks(tools):
    return "\n".join("```json\n" + json.dumps(t) + "\n```" for t in tools)


def _stub_tool(name):
    return {
        "tool_name": name,
        "tool_description": f"desc for {name}",
        "parameters": {"x": {"type": "string", "required": True}},
        "error_messages": [],
        "usage": "usage",
        "output_details": {"y": {"type": "string"}},
    }


def test_cli_topup_rerun_is_idempotent(monkeypatch, tmp_path, fake_llm):
    """Running the CLI twice with the same flags produces the same spec files."""
    output_dir = tmp_path / "specs"
    run_mod = _patch_llm_and_generate(monkeypatch, fake_llm)

    # Run 1: phase 1 -> 1 subfield, phase 2 -> 1 task, phase 3 -> 1 tool.
    fake_llm.queue(_fenced(["S"]))
    fake_llm.queue(_fenced(["T"]))
    fake_llm.queue(_tool_blocks([_stub_tool("Tool")]))

    monkeypatch.setattr(sys, "argv", [
        "run.py",
        "--field", "Smoke",
        "--output-dir", str(output_dir),
        "--model", "GPT-OSS-120B",
    ])
    run_mod.main()

    specs_after_run1 = sorted(p.name for p in output_dir.iterdir() if p.name.startswith("smoke_spec_"))
    assert specs_after_run1 == ["smoke_spec_000.json"]

    # Run 2: same flags, no responses queued. Top-up should make this a no-op.
    calls_before_run2 = len(fake_llm.calls)
    monkeypatch.setattr(sys, "argv", [
        "run.py",
        "--field", "Smoke",
        "--output-dir", str(output_dir),
        "--model", "GPT-OSS-120B",
    ])
    run_mod.main()

    specs_after_run2 = sorted(p.name for p in output_dir.iterdir() if p.name.startswith("smoke_spec_"))
    assert specs_after_run2 == ["smoke_spec_000.json"], "second run must be a no-op"
    assert len(fake_llm.calls) == calls_before_run2, "no LLM calls expected on no-op"

    # The original spec is still parseable JSON (atomic write left no half-file).
    payload = json.loads((output_dir / "smoke_spec_000.json").read_text())
    assert payload["spec_id"] == "smoke_spec_000"
