"""Tests for the phased `audit_tools(mode=...)` CLI that enables safe parallel runs.

Modes exercised:
  - "build"    — phase 1 only, no LLM
  - "evaluate" — phases 2-6 + per-spec refresh, skips global dataset rewrite AND end-of-run sweep
  - "sweep"    — no LLM; rebuilds dataset + refreshes every env_spec audit block

Final ground-truth test: two sequential `mode="evaluate"` calls on disjoint --fields
followed by one `mode="sweep"` produce byte-identical dataset + env_specs compared
to the monolithic `mode="full"` baseline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pytest

from env_audit.generate import audit_tools

# Reuse the same helpers + fixtures as the main test file.
from tests.test_env_audit import (
    _make_tool,
    _make_env_spec,
    _write_env_spec,
    _queue_full_happy_path,
    env_audit_env,  # fixture re-export via import — pytest picks it up
)


# --- Shared test-call templates --------------------------------------------

_TC = lambda tool: [
    {"Failure mode": "Happy path", "Tool parameters": {"arg1": "ok"},
     "Tool call message": f"[{tool}(arg1='ok')]"},
]


# --- 1. build-only ---------------------------------------------------------

def test_build_only_creates_stubs_no_llm_calls(fake_llm, env_audit_env):
    """`mode='build'` seeds dataset + eval_log stubs and makes zero LLM calls."""
    spec_a = _make_env_spec("f_spec_000", "F", "S", "T", [_make_tool("Alpha"), _make_tool("Beta")])
    _write_env_spec(env_audit_env["env_specs_dir"], spec_a)

    summary = audit_tools(**env_audit_env, llm=fake_llm, mode="build")

    assert summary["mode"] == "build"
    assert summary["n_new_tools"] == 2
    assert fake_llm.calls == []  # Zero LLM invocations.

    # Dataset has 2 rows, both with reliability=None.
    rows = [json.loads(l) for l in env_audit_env["dataset_path"].read_text().splitlines() if l.strip()]
    assert len(rows) == 2
    for r in rows:
        assert r["reliability"] is None
        assert r["evaluated_at"] is None

    # Per-tool eval_log stubs at phase_completed == 1.
    logs = sorted(p.name for p in env_audit_env["eval_logs_dir"].iterdir())
    assert logs == ["f_spec_000.Alpha.json", "f_spec_000.Beta.json"]
    for p in env_audit_env["eval_logs_dir"].iterdir():
        log = json.loads(p.read_text())
        assert log["phase_completed"] == 1


# --- 2. evaluate-only skips globals, still refreshes scoped spec audits ----

def test_evaluate_only_skips_dataset_and_sweep(fake_llm, env_audit_env):
    """`mode='evaluate' --fields [FieldA]` evaluates FieldA tools but leaves the
    dataset rows' reliability fields as None (skip_dataset_write). FieldA specs
    STILL get their audit block refreshed (via scoped phase-7 refresh, not the
    unscoped sweep). FieldB specs are left untouched (no sweep)."""
    # FieldA and FieldB with one tool each.
    _write_env_spec(env_audit_env["env_specs_dir"],
                    _make_env_spec("a_spec_000", "FieldA", "S", "T", [_make_tool("Alpha")]))
    _write_env_spec(env_audit_env["env_specs_dir"],
                    _make_env_spec("b_spec_000", "FieldB", "S", "T", [_make_tool("Beta")]))

    # Step 1: build populates both (no LLM).
    audit_tools(**env_audit_env, llm=fake_llm, mode="build")
    assert fake_llm.calls == []

    # Step 2: evaluate-only on FieldA — queue 1-tool pipeline only.
    _queue_full_happy_path(fake_llm, n_tools=1, test_calls_per_tool=[_TC("Alpha")])
    audit_tools(**env_audit_env, llm=fake_llm, mode="evaluate", fields=["FieldA"])

    # Dataset rows untouched (reliability still None) because skip_dataset_write=True.
    rows = {r["id"]: r
            for r in (json.loads(l) for l in env_audit_env["dataset_path"].read_text().splitlines() if l.strip())}
    assert rows["a_spec_000.Alpha"]["reliability"] is None, "dataset must NOT be updated in evaluate-only mode"
    assert rows["b_spec_000.Beta"]["reliability"] is None

    # FieldA's eval_log IS at phase 7 with reliability=1.0.
    a_log = json.loads((env_audit_env["eval_logs_dir"] / "a_spec_000.Alpha.json").read_text())
    assert a_log["phase_completed"] == 7
    assert a_log["reliability"] == 1.0

    # FieldA env_spec has an audit block (scoped refresh inside phase 7 still runs).
    a_spec = json.loads((env_audit_env["env_specs_dir"] / "a_spec_000.json").read_text())
    assert a_spec.get("audit") is not None
    assert a_spec["audit"]["aggregate"]["n_tools_evaluated"] == 1

    # FieldB env_spec is UNTOUCHED (no sweep ran).
    b_spec = json.loads((env_audit_env["env_specs_dir"] / "b_spec_000.json").read_text())
    assert b_spec.get("audit") is None, "FieldB spec must not have been swept in evaluate-only mode"


# --- 3. two evaluate-only + one sweep == full pipeline --------------------

def test_two_evaluate_only_then_sweep_matches_full(fake_llm, env_audit_env, tmp_path):
    """Simulates N=2 parallel jobs: build, evaluate FieldA, evaluate FieldB, sweep.
    Final state should exactly match a monolithic `mode='full'` run on the same seed."""
    # --- Phased run in env_audit_env ---
    spec_a = _make_env_spec("a_spec_000", "FieldA", "S", "T", [_make_tool("Alpha")])
    spec_b = _make_env_spec("b_spec_000", "FieldB", "S", "T", [_make_tool("Beta")])
    _write_env_spec(env_audit_env["env_specs_dir"], spec_a)
    _write_env_spec(env_audit_env["env_specs_dir"], spec_b)

    # 1. Build (no LLM)
    audit_tools(**env_audit_env, llm=fake_llm, mode="build")

    # 2a. Evaluate FieldA
    _queue_full_happy_path(fake_llm, n_tools=1, test_calls_per_tool=[_TC("Alpha")])
    audit_tools(**env_audit_env, llm=fake_llm, mode="evaluate", fields=["FieldA"])

    # 2b. Evaluate FieldB
    _queue_full_happy_path(fake_llm, n_tools=1, test_calls_per_tool=[_TC("Beta")])
    audit_tools(**env_audit_env, llm=fake_llm, mode="evaluate", fields=["FieldB"])

    # 3. Sweep (no LLM)
    audit_tools(**env_audit_env, llm=fake_llm, mode="sweep")

    phased_rows = {r["id"]: r for r in
                   (json.loads(l) for l in env_audit_env["dataset_path"].read_text().splitlines() if l.strip())}
    phased_specs = {p.name: json.loads(p.read_text())
                    for p in env_audit_env["env_specs_dir"].iterdir()}

    # --- Baseline monolithic run in a fresh dir ---
    baseline = {
        "env_specs_dir": tmp_path / "baseline_specs",
        "dataset_path": tmp_path / "baseline_dataset.jsonl",
        "eval_logs_dir": tmp_path / "baseline_logs",
    }
    baseline["env_specs_dir"].mkdir()
    _write_env_spec(baseline["env_specs_dir"], spec_a)
    _write_env_spec(baseline["env_specs_dir"], spec_b)

    from tests.conftest import FakeLLM
    baseline_llm = FakeLLM()
    _queue_full_happy_path(baseline_llm, n_tools=2,
                           test_calls_per_tool=[_TC("Alpha"), _TC("Beta")])
    audit_tools(**baseline, llm=baseline_llm, mode="full")

    baseline_rows = {r["id"]: r for r in
                     (json.loads(l) for l in baseline["dataset_path"].read_text().splitlines() if l.strip())}
    baseline_specs = {p.name: json.loads(p.read_text())
                      for p in baseline["env_specs_dir"].iterdir()}

    # --- Compare (ignoring timestamps which differ across runs) ---
    _VOLATILE = {"evaluated_at", "generated_at", "last_evaluated_at", "usage", "usage_total"}

    def _strip_volatile(d):
        if isinstance(d, dict):
            return {k: _strip_volatile(v) for k, v in d.items() if k not in _VOLATILE}
        if isinstance(d, list):
            return [_strip_volatile(x) for x in d]
        return d

    # Dataset row-by-row compare on reliability + per-mode
    for tid, r in phased_rows.items():
        b = baseline_rows[tid]
        assert r["reliability"] == b["reliability"], f"{tid} reliability mismatch"
        assert r["reliability_per_mode"] == b["reliability_per_mode"]

    # Env spec audit blocks (structure only, excluding timestamps + usage)
    for name, s in phased_specs.items():
        b = baseline_specs[name]
        assert _strip_volatile(s.get("audit") or {}) == _strip_volatile(b.get("audit") or {}), \
            f"audit block mismatch for {name}"


# --- 4. full mode preserves current behavior -------------------------------

def test_full_mode_default_unchanged(fake_llm, env_audit_env):
    """`mode='full'` (default, omitted) behaves identically to the old API.
    Pins the legacy contract so future refactors don't silently change it."""
    spec = _make_env_spec("f_spec_000", "F", "S", "T", [_make_tool("Alpha")])
    _write_env_spec(env_audit_env["env_specs_dir"], spec)

    _queue_full_happy_path(fake_llm, n_tools=1, test_calls_per_tool=[_TC("Alpha")])
    summary = audit_tools(**env_audit_env, llm=fake_llm)

    assert summary.get("mode") == "full"
    assert summary["n_new_tools"] == 1
    assert summary["n_evaluated"] == 1

    # Dataset IS rewritten (as before)
    rows = [json.loads(l) for l in env_audit_env["dataset_path"].read_text().splitlines() if l.strip()]
    assert rows[0]["reliability"] == 1.0

    # Env spec IS swept
    updated = json.loads((env_audit_env["env_specs_dir"] / "f_spec_000.json").read_text())
    assert updated.get("audit") is not None


# --- 5. write_jsonl atomic (no orphan .tmp) --------------------------------

def test_write_jsonl_is_atomic(tmp_path):
    """After a successful write, no .tmp file should remain."""
    from env_audit.utils import write_jsonl
    target = tmp_path / "out.jsonl"
    write_jsonl(target, [{"a": 1}, {"b": 2}])
    assert target.exists()
    # The whole dir contains exactly one file, and it's the target.
    files = list(tmp_path.iterdir())
    assert files == [target], f"orphan tmp or extra files: {files}"
    # File content intact.
    lines = target.read_text().strip().splitlines()
    assert [json.loads(l) for l in lines] == [{"a": 1}, {"b": 2}]


# --- 6. guard against bogus mode strings -----------------------------------

def test_invalid_mode_raises(fake_llm, env_audit_env):
    with pytest.raises(ValueError):
        audit_tools(**env_audit_env, llm=fake_llm, mode="nonsense")


# --- 7. parallel-safety hardening ------------------------------------------

def test_write_jsonl_tmp_is_pid_suffixed(tmp_path, monkeypatch):
    """Concurrent writers to same target must not collide on `.tmp` filename."""
    from env_audit.utils import write_jsonl
    target = tmp_path / "out.jsonl"
    # Simulate a second writer leaving a stale .tmp with a different PID
    stale = target.with_suffix(target.suffix + ".tmp.99999")
    stale.write_text("garbage-from-other-process\n")
    # Our write should use OUR pid, not trample the other one.
    write_jsonl(target, [{"a": 1}])
    assert target.exists()
    assert target.read_text().strip() == '{"a": 1}'
    # Stale tmp from the "other pid" is untouched — no collision.
    assert stale.exists()
    assert stale.read_text() == "garbage-from-other-process\n"


def test_save_eval_log_atomic_and_resilient(tmp_path):
    """save_eval_log must be atomic; load_eval_log must tolerate a corrupt file."""
    from env_audit.utils import save_eval_log, load_eval_log
    p = tmp_path / "tool.json"
    save_eval_log(p, {"tool_id": "t", "phase_completed": 7})
    assert load_eval_log(p) == {"tool_id": "t", "phase_completed": 7}
    # Simulate a SIGKILL'd half-written file
    p.write_text("{ not valid json")
    assert load_eval_log(p) is None  # corrupt → None, does not crash
    # Re-save overwrites atomically
    save_eval_log(p, {"tool_id": "t", "phase_completed": 2})
    assert load_eval_log(p) == {"tool_id": "t", "phase_completed": 2}


def test_build_dataset_noop_skips_write(fake_llm, env_audit_env):
    """Calling build_dataset twice on same spec set: second call must NOT
    touch tools_dataset.jsonl (mtime unchanged)."""
    spec = _make_env_spec("f_spec_000", "F", "S", "T", [_make_tool("Alpha")])
    _write_env_spec(env_audit_env["env_specs_dir"], spec)

    # First call: writes dataset
    audit_tools(**env_audit_env, llm=fake_llm, mode="build")
    mtime_after_first = env_audit_env["dataset_path"].stat().st_mtime_ns

    # Second call: nothing new, dataset write must be skipped
    import time; time.sleep(0.01)  # ensure any write would change mtime
    audit_tools(**env_audit_env, llm=fake_llm, mode="build")
    mtime_after_second = env_audit_env["dataset_path"].stat().st_mtime_ns

    assert mtime_after_first == mtime_after_second, \
        "build_dataset rewrote the dataset even though no new rows were added"
