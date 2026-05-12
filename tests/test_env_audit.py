"""Unit tests for env_audit against a FakeLLM.

Covers the 7-phase pipeline:
- build diff / skip existing
- full happy path
- resume from mid-pipeline
- failure_mode reaches the judge
- per-mode reliability aggregation
- param_check FAIL skips simulation

See /path/to/data for the plan.
"""

import json
from pathlib import Path
from typing import Dict, List

import pytest

from env_audit.generate import audit_tools, build_dataset, evaluate_tools


# --- Helpers to build FakeLLM responses for each phase --------------------


def _make_tool(name: str, params: Dict[str, Dict] = None) -> Dict:
    return {
        "tool_name": name,
        "tool_description": f"Description for {name}",
        "parameters": params or {"arg1": {"type": "string", "required": True, "description": "a string"}},
        "error_messages": [f"Missing required parameter: arg1"],
        "usage": f"Call {name} with arg1.",
        "output_details": {"result": {"type": "string", "description": "outcome"}},
    }


def _make_env_spec(spec_id: str, field: str, subfield: str, task: str, tools: List[Dict]) -> Dict:
    return {
        "schema_version": "env_spec.v1",
        "spec_id": spec_id,
        "field": field,
        "subfield": subfield,
        "task": task,
        "model": "fake",
        "tools": tools,
        "sequences": {},
    }


def _write_env_spec(env_specs_dir: Path, spec: Dict) -> Path:
    path = env_specs_dir / f"{spec['spec_id']}.json"
    path.write_text(json.dumps(spec, ensure_ascii=False, default=str))
    return path


def _metadata_response(payload: Dict) -> str:
    return "```json\n" + json.dumps(payload) + "\n```"


def _test_calls_response(calls: List[Dict]) -> str:
    """Format a test-calls response matching what generate_testing_tool_calls.yml asks for."""
    body = {f"Tool call {i+1}": c for i, c in enumerate(calls)}
    return json.dumps(body)


def _param_check_response(status: str) -> str:
    code = 200 if status == "PASS" else 400
    body = {"status": status, "status_code": code, "error_message": None if status == "PASS" else "missing param"}
    return json.dumps(body)


def _sim_response(payload: Dict) -> str:
    return json.dumps(payload)


def _judge_response(judgment: str) -> str:
    return json.dumps({"judgment": judgment, "confidence": 0.9, "reasoning": "fake"})


@pytest.fixture
def env_audit_env(tmp_path: Path):
    env_specs_dir = tmp_path / "env_specs"
    env_specs_dir.mkdir()
    dataset_path = tmp_path / "tools_dataset.jsonl"
    eval_logs_dir = tmp_path / "tool_eval_logs"
    return {"env_specs_dir": env_specs_dir, "dataset_path": dataset_path, "eval_logs_dir": eval_logs_dir}


# --- Split-API filter tests (Step 3 of the restructure plan) --------------


def test_build_dataset_fields_filter(fake_llm, env_audit_env):
    """Specs whose field isn't in the --fields set are skipped by build_dataset."""
    _write_env_spec(
        env_audit_env["env_specs_dir"],
        _make_env_spec("aero_spec_000", "Aerospace", "S", "T", [_make_tool("Alpha")]),
    )
    _write_env_spec(
        env_audit_env["env_specs_dir"],
        _make_env_spec("health_spec_000", "Healthcare", "S", "T", [_make_tool("Beta")]),
    )

    summary = build_dataset(
        env_specs_dir=env_audit_env["env_specs_dir"],
        dataset_path=env_audit_env["dataset_path"],
        eval_logs_dir=env_audit_env["eval_logs_dir"],
        llm=fake_llm,
        fields=["Aerospace"],
    )

    assert summary["n_discovered"] == 2
    assert summary["n_skipped_by_field"] == 1
    assert summary["n_new"] == 1
    # Only Aerospace tool's eval log exists
    logs = sorted(p.name for p in env_audit_env["eval_logs_dir"].iterdir())
    assert logs == ["aero_spec_000.Alpha.json"]


def test_evaluate_sweep_backfills_audit_blocks(fake_llm, env_audit_env):
    """If env_spec is missing the audit block but eval_logs exist, the sweep adds it."""
    spec = _make_env_spec("f_spec_000", "F", "S", "T", [_make_tool("Alpha")])
    _write_env_spec(env_audit_env["env_specs_dir"], spec)

    test_calls = [{"Failure mode": "Happy path", "Tool parameters": {"arg1": "ok"}, "Tool call message": "[Alpha(arg1='ok')]"}]
    _queue_full_happy_path(fake_llm, n_tools=1, test_calls_per_tool=[test_calls])
    audit_tools(**env_audit_env, llm=fake_llm)

    # Strip the audit block manually (simulates: env_spec was written by prior code without audit support)
    spec_path = env_audit_env["env_specs_dir"] / "f_spec_000.json"
    spec_data = json.loads(spec_path.read_text())
    del spec_data["audit"]
    spec_data["schema_version"] = "env_spec.v1"
    spec_path.write_text(json.dumps(spec_data, indent=2))

    # Re-run evaluate — pending is empty (tool already at phase 7) but the sweep
    # should still refresh the audit block from existing eval_logs.
    evaluate_tools(
        dataset_path=env_audit_env["dataset_path"],
        env_specs_dir=env_audit_env["env_specs_dir"],
        eval_logs_dir=env_audit_env["eval_logs_dir"],
        llm=fake_llm,
    )

    refreshed = json.loads(spec_path.read_text())
    assert refreshed["schema_version"] == "env_spec.v2"
    assert "audit" in refreshed
    assert refreshed["audit"]["aggregate"]["n_tools_evaluated"] == 1


def test_evaluate_ids_filter(fake_llm, env_audit_env):
    """Only the listed ids are evaluated; the rest stay at phase_completed < 7."""
    spec = _make_env_spec("f_spec_000", "F", "S", "T", [_make_tool("Alpha"), _make_tool("Beta")])
    _write_env_spec(env_audit_env["env_specs_dir"], spec)
    # First, build the dataset (seeds both eval logs at phase 1)
    build_dataset(llm=fake_llm, **env_audit_env)

    # Queue phases for only Alpha (1 tool, K=1 test calls)
    test_calls = [{"Failure mode": "Happy path", "Tool parameters": {"arg1": "ok"}, "Tool call message": "[Alpha(arg1='ok')]"}]
    _queue_full_happy_path(fake_llm, n_tools=1, test_calls_per_tool=[test_calls])

    evaluate_tools(
        dataset_path=env_audit_env["dataset_path"],
        env_specs_dir=env_audit_env["env_specs_dir"],
        eval_logs_dir=env_audit_env["eval_logs_dir"],
        llm=fake_llm,
        ids=["f_spec_000.Alpha"],
    )

    alpha = json.loads((env_audit_env["eval_logs_dir"] / "f_spec_000.Alpha.json").read_text())
    beta = json.loads((env_audit_env["eval_logs_dir"] / "f_spec_000.Beta.json").read_text())
    assert alpha["phase_completed"] == 7
    assert beta["phase_completed"] == 1  # unchanged


def test_evaluate_fields_filter(fake_llm, env_audit_env):
    """Only tools from fields in the filter set are evaluated."""
    _write_env_spec(
        env_audit_env["env_specs_dir"],
        _make_env_spec("aero_spec_000", "Aerospace", "S", "T", [_make_tool("Alpha")]),
    )
    _write_env_spec(
        env_audit_env["env_specs_dir"],
        _make_env_spec("health_spec_000", "Healthcare", "S", "T", [_make_tool("Beta")]),
    )
    build_dataset(llm=fake_llm, **env_audit_env)

    test_calls = [{"Failure mode": "Happy path", "Tool parameters": {"arg1": "ok"}, "Tool call message": "[Alpha(arg1='ok')]"}]
    _queue_full_happy_path(fake_llm, n_tools=1, test_calls_per_tool=[test_calls])

    evaluate_tools(
        dataset_path=env_audit_env["dataset_path"],
        env_specs_dir=env_audit_env["env_specs_dir"],
        eval_logs_dir=env_audit_env["eval_logs_dir"],
        llm=fake_llm,
        fields=["Aerospace"],
    )

    alpha = json.loads((env_audit_env["eval_logs_dir"] / "aero_spec_000.Alpha.json").read_text())
    beta = json.loads((env_audit_env["eval_logs_dir"] / "health_spec_000.Beta.json").read_text())
    assert alpha["phase_completed"] == 7
    assert beta["phase_completed"] == 1


def test_evaluate_ids_and_fields_intersect(fake_llm, env_audit_env):
    """When both filters given, pending = intersection."""
    spec = _make_env_spec("aero_spec_000", "Aerospace", "S", "T",
                         [_make_tool("Alpha"), _make_tool("Beta"), _make_tool("Gamma")])
    _write_env_spec(env_audit_env["env_specs_dir"], spec)
    build_dataset(llm=fake_llm, **env_audit_env)

    # Filter: ids=[Alpha, Beta]  fields=[Aerospace]  → intersection = [Alpha, Beta]
    # Queue for 2 tools
    test_calls = [{"Failure mode": "Happy path", "Tool parameters": {"arg1": "ok"}, "Tool call message": "[X(arg1='ok')]"}]
    _queue_full_happy_path(fake_llm, n_tools=2, test_calls_per_tool=[test_calls, test_calls])

    evaluate_tools(
        dataset_path=env_audit_env["dataset_path"],
        env_specs_dir=env_audit_env["env_specs_dir"],
        eval_logs_dir=env_audit_env["eval_logs_dir"],
        llm=fake_llm,
        ids=["aero_spec_000.Alpha", "aero_spec_000.Beta"],
        fields=["Aerospace"],
    )

    alpha = json.loads((env_audit_env["eval_logs_dir"] / "aero_spec_000.Alpha.json").read_text())
    beta = json.loads((env_audit_env["eval_logs_dir"] / "aero_spec_000.Beta.json").read_text())
    gamma = json.loads((env_audit_env["eval_logs_dir"] / "aero_spec_000.Gamma.json").read_text())
    assert alpha["phase_completed"] == 7
    assert beta["phase_completed"] == 7
    assert gamma["phase_completed"] == 1


# --- 1. Build skips existing tools ----------------------------------------


def test_build_skips_existing_tools(fake_llm, env_audit_env):
    """A tool already in tools_dataset.jsonl must NOT be re-processed."""
    spec = _make_env_spec("f_spec_000", "F", "S", "T", [_make_tool("ExistingTool"), _make_tool("NewTool")])
    _write_env_spec(env_audit_env["env_specs_dir"], spec)

    # Pre-seed dataset with ExistingTool already evaluated
    env_audit_env["dataset_path"].write_text(json.dumps({
        "schema_version": "tools_dataset.v1",
        "id": "f_spec_000.ExistingTool",
        "field": "F", "subfield": "S", "task": "T",
        "tool_name": "ExistingTool", "tool": _make_tool("ExistingTool"),
        "reliability": 1.0,
        "reliability_per_mode": {"fm1": 1.0, "fm2": 1.0, "fm3": 1.0},
        "eval_version": "env_audit.v1",
        "evaluated_at": "2026-01-01T00:00:00Z",
        "model": "GPT-OSS-120B",
    }) + "\n")

    # Queue phase responses for NewTool only (1 tool × 3 test calls)
    _queue_full_happy_path(fake_llm, n_tools=1, test_calls_per_tool=[
        [{"Failure mode": "Missing required", "Tool parameters": {}, "Tool call message": "[NewTool()]"},
         {"Failure mode": "Cross-field", "Tool parameters": {"arg1": "x"}, "Tool call message": "[NewTool(arg1='x')]"},
         {"Failure mode": "Happy path", "Tool parameters": {"arg1": "ok"}, "Tool call message": "[NewTool(arg1='ok')]"}],
    ])

    summary = audit_tools(
        env_specs_dir=env_audit_env["env_specs_dir"],
        dataset_path=env_audit_env["dataset_path"],
        eval_logs_dir=env_audit_env["eval_logs_dir"],
        llm=fake_llm,
    )

    assert summary["n_new_tools"] == 1
    # Only NewTool got an eval log; ExistingTool was skipped entirely
    log_files = sorted(p.name for p in env_audit_env["eval_logs_dir"].iterdir())
    assert log_files == ["f_spec_000.NewTool.json"]

    # ExistingTool's row is preserved unchanged
    rows = [json.loads(l) for l in env_audit_env["dataset_path"].read_text().splitlines() if l.strip()]
    existing = [r for r in rows if r["id"] == "f_spec_000.ExistingTool"][0]
    assert existing["reliability"] == 1.0
    assert existing["evaluated_at"] == "2026-01-01T00:00:00Z"


# --- 2. Full pipeline happy path ------------------------------------------


def test_env_spec_gets_audit_block(fake_llm, env_audit_env):
    """After evaluate runs, the owning env_spec gets schema v2 with a populated audit block."""
    spec = _make_env_spec("f_spec_000", "F", "S", "T", [_make_tool("Alpha"), _make_tool("Beta")])
    _write_env_spec(env_audit_env["env_specs_dir"], spec)

    test_calls = [
        {"Failure mode": "Happy path", "Tool parameters": {"arg1": "ok"}, "Tool call message": "[Alpha(arg1='ok')]"},
    ]
    _queue_full_happy_path(fake_llm, n_tools=2, test_calls_per_tool=[test_calls, test_calls])

    audit_tools(**env_audit_env, llm=fake_llm)

    spec_path = env_audit_env["env_specs_dir"] / "f_spec_000.json"
    updated = json.loads(spec_path.read_text())
    assert updated["schema_version"] == "env_spec.v2"
    assert "audit" in updated
    audit = updated["audit"]
    assert audit["schema_version"] == "audit.v1"
    assert {t["tool_name"] for t in audit["tools"]} == {"Alpha", "Beta"}
    assert audit["aggregate"]["n_tools_evaluated"] == 2
    assert audit["aggregate"]["n_tools_total"] == 2
    assert audit["aggregate"]["n_tools_pending"] == 0
    assert audit["aggregate"]["mean_reliability"] == 1.0
    assert audit["model"] == "fake-model"

    # Running again (idempotent) — no queued responses needed since all phases are at 7
    audit_tools(**env_audit_env, llm=fake_llm)
    re_updated = json.loads(spec_path.read_text())
    # Per-tool entries identical structurally
    assert [t["id"] for t in re_updated["audit"]["tools"]] == [t["id"] for t in audit["tools"]]


def test_full_pipeline_small(fake_llm, env_audit_env):
    """1 scenario × 2 tools, all test calls pass param check and judge as correct."""
    spec = _make_env_spec("f_spec_000", "F", "S", "T", [_make_tool("Alpha"), _make_tool("Beta")])
    _write_env_spec(env_audit_env["env_specs_dir"], spec)

    test_calls_alpha = [
        {"Failure mode": "Missing required", "Tool parameters": {}, "Tool call message": "[Alpha()]"},
        {"Failure mode": "Happy path", "Tool parameters": {"arg1": "ok"}, "Tool call message": "[Alpha(arg1='ok')]"},
    ]
    test_calls_beta = [
        {"Failure mode": "Cross-field", "Tool parameters": {"arg1": "bad"}, "Tool call message": "[Beta(arg1='bad')]"},
        {"Failure mode": "Happy path", "Tool parameters": {"arg1": "ok"}, "Tool call message": "[Beta(arg1='ok')]"},
    ]
    _queue_full_happy_path(fake_llm, n_tools=2, test_calls_per_tool=[test_calls_alpha, test_calls_beta])

    summary = audit_tools(**env_audit_env, llm=fake_llm)

    assert summary["n_new_tools"] == 2
    assert summary["n_evaluated"] == 2

    # Per-tool eval JSONs
    alpha = json.loads((env_audit_env["eval_logs_dir"] / "f_spec_000.Alpha.json").read_text())
    beta = json.loads((env_audit_env["eval_logs_dir"] / "f_spec_000.Beta.json").read_text())
    assert alpha["phase_completed"] == 7
    assert beta["phase_completed"] == 7
    assert alpha["reliability"] == 1.0
    assert beta["reliability"] == 1.0
    assert len(alpha["test_calls"]) == 2
    assert len(alpha["results"]) == 2

    # Dataset row updated
    rows = [json.loads(l) for l in env_audit_env["dataset_path"].read_text().splitlines() if l.strip()]
    assert len(rows) == 2
    for row in rows:
        assert row["reliability"] == 1.0
        assert row["evaluated_at"] is not None


# --- 3. Resume from mid-pipeline -----------------------------------------


def test_resume_from_phase_4(fake_llm, env_audit_env):
    """Pre-seed an eval log at phase_completed=3; orchestrator skips phases 2-3."""
    spec = _make_env_spec("f_spec_000", "F", "S", "T", [_make_tool("Alpha")])
    _write_env_spec(env_audit_env["env_specs_dir"], spec)
    # Seed the dataset so phase 1 doesn't try to re-add (but do *not* mark evaluated)
    env_audit_env["dataset_path"].write_text(json.dumps({
        "schema_version": "tools_dataset.v1",
        "id": "f_spec_000.Alpha",
        "field": "F", "subfield": "S", "task": "T",
        "tool_name": "Alpha", "tool": _make_tool("Alpha"),
        "reliability": None, "reliability_per_mode": None,
        "eval_version": None, "evaluated_at": None, "model": None,
    }) + "\n")

    # Seed eval log already at phase 3 with metadata + test_calls
    env_audit_env["eval_logs_dir"].mkdir(parents=True, exist_ok=True)
    (env_audit_env["eval_logs_dir"] / "f_spec_000.Alpha.json").write_text(json.dumps({
        "schema_version": "env_audit_log.v1",
        "tool_id": "f_spec_000.Alpha",
        "spec_id": "f_spec_000", "field": "F", "subfield": "S", "task": "T",
        "tool": _make_tool("Alpha"),
        "generated_at": "2026-01-01T00:00:00Z",
        "model": "fake", "model_config": {},
        "phase_completed": 3,
        "metadata": {"items": [{"id": 1}]},
        "test_calls": [
            {"idx": 0, "failure_mode_group": "fm3", "failure_mode": "Happy path",
             "parameters": {"arg1": "ok"}, "tool_call_message": "Alpha(arg1='ok')"},
        ],
        "results": [],
        "reliability": None, "reliability_per_mode": None,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }))

    # Queue only phase 4-6 responses (1 test call → 1 param_check + 1 simulate + 1 judge)
    fake_llm.queue(_param_check_response("PASS"))
    fake_llm.queue(_sim_response({"status_code": 200, "response": {"result": "ok"}}))
    fake_llm.queue(_judge_response("correct"))

    audit_tools(**env_audit_env, llm=fake_llm)

    # No phase 2/3 calls should have happened — only 3 calls queued and all consumed
    assert len(fake_llm.calls) == 3

    log = json.loads((env_audit_env["eval_logs_dir"] / "f_spec_000.Alpha.json").read_text())
    assert log["phase_completed"] == 7
    assert log["reliability"] == 1.0


# --- 4. Judge receives failure_mode --------------------------------------


def test_judge_receives_failure_mode(fake_llm, env_audit_env):
    spec = _make_env_spec("f_spec_000", "F", "S", "T", [_make_tool("Alpha")])
    _write_env_spec(env_audit_env["env_specs_dir"], spec)

    test_calls = [
        {"Failure mode": "THIS IS THE FAILURE MODE LABEL", "Tool parameters": {}, "Tool call message": "[Alpha()]"},
    ]
    _queue_full_happy_path(fake_llm, n_tools=1, test_calls_per_tool=[test_calls])

    audit_tools(**env_audit_env, llm=fake_llm)

    # Phase 6 is the last batched call. Inspect its prompt.
    last_call = fake_llm.calls[-1]
    # Unwrap (single or batched)
    if last_call["batched"]:
        prompt_text = last_call["messages"][0][0]["content"]
    else:
        prompt_text = last_call["messages"][0]["content"]
    assert "THIS IS THE FAILURE MODE LABEL" in prompt_text


# --- 5. Per-mode breakdown -----------------------------------------------


def test_reliability_per_mode_breakdown(fake_llm, env_audit_env):
    """3 test calls, one per group. fm1 correct, fm2 incorrect, fm3 correct → reliability 2/3."""
    spec = _make_env_spec("f_spec_000", "F", "S", "T", [_make_tool("Alpha")])
    _write_env_spec(env_audit_env["env_specs_dir"], spec)

    test_calls = [
        {"Failure mode": "Missing required param", "Tool parameters": {}, "Tool call message": "[Alpha()]"},
        {"Failure mode": "Cross-field dependency violated", "Tool parameters": {"arg1": "x"}, "Tool call message": "[Alpha(arg1='x')]"},
        {"Failure mode": "Nominal happy path", "Tool parameters": {"arg1": "ok"}, "Tool call message": "[Alpha(arg1='ok')]"},
    ]
    # Phase 2 — metadata
    fake_llm.queue(_metadata_response({"items": []}))
    # Phase 3 — test calls
    fake_llm.queue(_test_calls_response(test_calls))
    # Phase 4 — param_check × 3 (fm1 FAIL, fm2 PASS, fm3 PASS)
    fake_llm.queue_batch([_param_check_response("FAIL"), _param_check_response("PASS"), _param_check_response("PASS")])
    # Phase 5 — simulate for the 2 passing tests
    fake_llm.queue_batch([
        _sim_response({"status_code": 400, "response": None, "explanation": "violates cross-field"}),
        _sim_response({"status_code": 200, "response": {"result": "ok"}}),
    ])
    # Phase 6 — judge × 3, outcomes: fm1 correct, fm2 incorrect, fm3 correct
    fake_llm.queue_batch([_judge_response("correct"), _judge_response("should_success_got_error"), _judge_response("correct")])

    audit_tools(**env_audit_env, llm=fake_llm)

    log = json.loads((env_audit_env["eval_logs_dir"] / "f_spec_000.Alpha.json").read_text())
    assert log["reliability"] == pytest.approx(2/3)
    assert log["reliability_per_mode"]["fm1"] == 1.0
    assert log["reliability_per_mode"]["fm2"] == 0.0
    assert log["reliability_per_mode"]["fm3"] == 1.0


# --- 6. param_check FAIL skips simulation --------------------------------


def test_param_check_fail_skips_simulation(fake_llm, env_audit_env):
    """When a test call fails param_check, phase 5 must NOT simulate it, but phase 6 still judges it."""
    spec = _make_env_spec("f_spec_000", "F", "S", "T", [_make_tool("Alpha")])
    _write_env_spec(env_audit_env["env_specs_dir"], spec)

    test_calls = [
        {"Failure mode": "Missing required", "Tool parameters": {}, "Tool call message": "[Alpha()]"},
        {"Failure mode": "Happy path", "Tool parameters": {"arg1": "ok"}, "Tool call message": "[Alpha(arg1='ok')]"},
    ]
    # Phase 2 — metadata (1 tool, 1 prompt → single call)
    fake_llm.queue(_metadata_response({"items": []}))
    # Phase 3 — test calls (1 tool, 1 prompt → single call)
    fake_llm.queue(_test_calls_response(test_calls))
    # Phase 4 — param_check × 2 (batched)
    fake_llm.queue_batch([_param_check_response("FAIL"), _param_check_response("PASS")])
    # Phase 5 — simulate for ONE passing test call (single, since only 1 item)
    fake_llm.queue(_sim_response({"status_code": 200, "response": {"result": "ok"}}))
    # Phase 6 — judge × 2 (batched, both still judged)
    fake_llm.queue_batch([_judge_response("correct"), _judge_response("correct")])

    audit_tools(**env_audit_env, llm=fake_llm)

    log = json.loads((env_audit_env["eval_logs_dir"] / "f_spec_000.Alpha.json").read_text())
    # The failed one has no simulator_response
    assert log["results"][0]["simulator_response"] is None
    # The passing one does
    assert log["results"][1]["simulator_response"] is not None
    # Both got judged
    assert log["results"][0]["judgment"] is not None
    assert log["results"][1]["judgment"] is not None


# --- Shared queue helper: "happy path" = PASS → simulate → correct ---------

def _queue_full_happy_path(fake_llm, n_tools: int, test_calls_per_tool: List[List[Dict]]) -> None:
    """Queue all 5 LLM phases for a scenario where every test call PASSes and is judged correct."""
    # Phase 2 — metadata (N tools)
    if n_tools == 1:
        fake_llm.queue(_metadata_response({"items": [{"id": 1}]}))
    else:
        fake_llm.queue_batch([_metadata_response({"items": [{"id": 1}]}) for _ in range(n_tools)])
    # Phase 3 — test calls
    if n_tools == 1:
        fake_llm.queue(_test_calls_response(test_calls_per_tool[0]))
    else:
        fake_llm.queue_batch([_test_calls_response(tcs) for tcs in test_calls_per_tool])
    # Phase 4 — param_check per (tool, test_call). Total = sum of K across tools.
    total_pairs = sum(len(tcs) for tcs in test_calls_per_tool)
    if total_pairs == 1:
        fake_llm.queue(_param_check_response("PASS"))
    else:
        fake_llm.queue_batch([_param_check_response("PASS") for _ in range(total_pairs)])
    # Phase 5 — simulate (all PASS in this helper)
    if total_pairs == 1:
        fake_llm.queue(_sim_response({"status_code": 200, "response": {"result": "ok"}}))
    else:
        fake_llm.queue_batch([_sim_response({"status_code": 200, "response": {"result": "ok"}}) for _ in range(total_pairs)])
    # Phase 6 — judge all as correct
    if total_pairs == 1:
        fake_llm.queue(_judge_response("correct"))
    else:
        fake_llm.queue_batch([_judge_response("correct") for _ in range(total_pairs)])


# --- 7. Role-level tests ---------------------------------------------------


def test_envgen_metadata_method_renders(fake_llm):
    from roles.env_generator import EnvironmentGenerator
    from env_audit.generate import _env_generator_template_files
    role = EnvironmentGenerator(_env_generator_template_files(), fake_llm)
    fake_llm.queue("```json\n" + json.dumps({"items": [{"id": 1}]}) + "\n```")
    result = role.generate_metadata({"field_name": "F", "tools": [_make_tool("Alpha")]})
    assert result["parsed"] == {"items": [{"id": 1}]}
    assert "{Data}" not in result["prompt"]  # placeholder got substituted


def test_judge_failure_mode_placeholder(fake_llm):
    from roles.judge_simulator import JudgeSimulator
    role = JudgeSimulator(fake_llm)
    fake_llm.queue(_judge_response("correct"))
    result = role.judge(
        tool_details={"tool_name": "X"},
        message="X()",
        response={"status_code": 400},
        meta_data={},
        failure_mode="MY UNIQUE LABEL FOR THIS TEST",
    )
    assert "MY UNIQUE LABEL FOR THIS TEST" in result["prompt"]
