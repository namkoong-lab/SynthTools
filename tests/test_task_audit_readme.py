"""Doc-presence test for task_audit/README.md.

The README must mention every load-bearing flag and the
`task_content.jsonl` release artifact. Catches future drift where
someone edits the README but drops critical doc that consumers depend
on (the 2026-05 prompt rewrite history is encoded in `--resummarize`,
the Perlmutter sbatch flow depends on `--shard`/`--num-shards`, etc).
"""
from __future__ import annotations

from pathlib import Path

import pytest

README = Path(__file__).resolve().parent.parent / "task_audit" / "README.md"


def _text():
    return README.read_text()


@pytest.mark.parametrize("needle", [
    "python -m task_audit.run",
    "--tasks-dir",
    "--env-specs-dir",
    "--model",
    "--resummarize",
    "--shard",
    "--num-shards",
    "--task-content",
    "--audit-only",
    "--audit-report",
    "--server-url",
    "--no-debug",
    "task_content.jsonl",
    "user_supplied_values",
    "tool_produced_values",
    "spillover",
    "task_summarized",
])
def test_readme_mentions(needle):
    """README must contain each load-bearing token."""
    assert needle in _text(), f"task_audit/README.md missing reference to {needle!r}"


def test_readme_does_not_advertise_old_entrypoint():
    """README must not still call out the old `python -m task_audit.summarize` form."""
    assert "python -m task_audit.summarize" not in _text(), \
        "stale entrypoint still in README; should be `python -m task_audit.run`"
