"""End-to-end unit tests for env_generation against a FakeLLM.

Covers:
- The 3-phase pipeline (subfields → tasks → tools) runs correctly with canned responses.
- Phases 2-3 use batched LLM calls (single call with list-of-lists messages).
- Per-scenario JSON files produced with the expected shape.
- Sibling `{field_slug}_field_gen.json` carries phase-1 and phase-2 content.
- spec_id auto-numbering advances across scenarios.
- Token totals match sum of per-phase token counts.
- Scenarios save `sequences: {}` as a placeholder (sequences are populated later
  by `traj_generation.build_sequences`).
"""

import json
from pathlib import Path

import pytest

from env_generation.generate import generate_environments


def _subfields_response(items):
    import json as _json
    return "```json\n" + _json.dumps(items) + "\n```"


def _tasks_response(items):
    import json as _json
    return "```json\n" + _json.dumps(items) + "\n```"


def _tools_response(tools):
    """Emit tools as individual fenced JSON blocks (matches generate_tools output)."""
    import json as _json
    parts = []
    for t in tools:
        parts.append("```json\n" + _json.dumps(t) + "\n```")
    return "\n".join(parts)


def _make_tool(name: str):
    return {
        "tool_name": name,
        "tool_description": f"desc for {name}",
        "parameters": {"x": {"type": "string", "required": True}},
        "error_messages": [],
        "usage": "usage",
        "output_details": {"y": {"type": "string"}},
    }


def test_full_pipeline_one_field_two_subfields_two_tasks(fake_llm, tmp_output):
    """1 field × 2 subfields × 2 tasks = 4 scenarios. Verify shape + batching."""
    subfields = ["Subfield A", "Subfield B"]
    tasks_A = ["Task A1", "Task A2"]
    tasks_B = ["Task B1", "Task B2"]
    tools = [_make_tool("Alpha"), _make_tool("Beta")]

    # Phase 1: subfields (single)
    fake_llm.queue(_subfields_response(subfields))
    # Phase 2: tasks (batched, one per subfield)
    fake_llm.queue_batch([_tasks_response(tasks_A), _tasks_response(tasks_B)])
    # Phase 3: tools (batched, one per scenario = 4)
    fake_llm.queue_batch([_tools_response(tools)] * 4)

    saved = generate_environments(
        fields=["MyField"],
        output_dir=tmp_output,
        llm=fake_llm,
        max_subfields=2,
        max_tasks_per_subfield=2,
    )

    assert len(saved) == 4, "4 scenarios expected (2 subfields × 2 tasks)"
    # Spot-check batching: phases 2-3 must be single batched calls, not 2/4 separate calls.
    batched_calls = [c for c in fake_llm.calls if c["batched"]]
    assert len(batched_calls) == 2
    assert len(batched_calls[0]["messages"]) == 2    # phase 2: 2 subfields
    assert len(batched_calls[1]["messages"]) == 4    # phase 3: 4 scenarios

    # Check files exist with expected names
    files = sorted(p.name for p in tmp_output.iterdir())
    assert "myfield_field_gen.json" in files
    spec_files = [f for f in files if f.startswith("myfield_spec_")]
    assert len(spec_files) == 4
    assert spec_files == [f"myfield_spec_{i:03d}.json" for i in range(4)]

    # Verify scenario shape
    with open(tmp_output / "myfield_spec_000.json") as f:
        scenario = json.load(f)
    assert scenario["field"] == "MyField"
    assert scenario["subfield"] in subfields
    assert scenario["task"] in tasks_A + tasks_B
    assert {t["tool_name"] for t in scenario["tools"]} == {"Alpha", "Beta"}
    # Sequences placeholder — populated later by traj_generation.build_sequences
    assert scenario["sequences"] == {}

    # generation_log has exactly 3 phase entries
    phases = [e["phase"] for e in scenario["generation_log"]]
    assert phases == ["subfields", "tasks", "tools"]

    # Phase 1 and 2 entries reference the field log; phase 3 contains prompt+response
    assert scenario["generation_log"][0]["shared_with_field_log"] == "myfield_field_gen.json"
    assert scenario["generation_log"][1]["shared_with_field_log"] == "myfield_field_gen.json"
    assert "prompt" in scenario["generation_log"][2]
    assert "response" in scenario["generation_log"][2]

    # Token aggregation: scenario usage == sum of its 3 phase usages
    per_phase_p = sum(e["usage"]["prompt_tokens"] for e in scenario["generation_log"])
    per_phase_c = sum(e["usage"]["completion_tokens"] for e in scenario["generation_log"])
    assert scenario["usage"]["prompt_tokens"] == per_phase_p
    assert scenario["usage"]["completion_tokens"] == per_phase_c
    assert scenario["usage"]["total_tokens"] == per_phase_p + per_phase_c

    # Reproducibility stamp + schema version
    assert scenario["schema_version"] == "env_spec.v1"
    assert "generated_at" in scenario and scenario["generated_at"].endswith("Z")
    assert "model_config" in scenario
    assert "field_elapsed_s" in scenario
    assert "generation_time_s" not in scenario  # renamed

    # Field log has its own schema_version + reproducibility stamp
    with open(tmp_output / "myfield_field_gen.json") as f:
        flog = json.load(f)
    assert flog["schema_version"] == "env_field_gen.v1"
    assert "generated_at" in flog and flog["generated_at"].endswith("Z")
    assert "model_config" in flog
    assert "elapsed_s" in flog

    # Manifest: one line per saved scenario
    manifest_path = tmp_output / "manifest.jsonl"
    assert manifest_path.exists(), "manifest.jsonl not emitted"
    lines = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 4, f"expected 4 manifest lines, got {len(lines)}"
    for entry in lines:
        assert entry["schema_version"] == "env_manifest.v1"
        assert set(["spec_id", "field", "subfield", "task", "n_tools", "n_sequences", "generated_at", "model", "usage"]).issubset(entry)
        assert entry["n_sequences"] == 0  # env_generation never writes sequences anymore
    assert {e["spec_id"] for e in lines} == {f"myfield_spec_{i:03d}" for i in range(4)}


def test_field_log_contains_phase_1_and_2(fake_llm, tmp_output):
    # 1 subfield × 1 task → phases 2, 3 each have exactly 1 prompt → single calls
    fake_llm.queue(_subfields_response(["S1"]))
    fake_llm.queue(_tasks_response(["T1"]))
    fake_llm.queue(_tools_response([_make_tool("X")]))

    generate_environments(
        fields=["F"],
        output_dir=tmp_output,
        llm=fake_llm,
        max_subfields=1,
        max_tasks_per_subfield=1,
    )

    log_path = tmp_output / "f_field_gen.json"
    assert log_path.exists()
    with open(log_path) as f:
        flog = json.load(f)
    phases = [e["phase"] for e in flog["generation_log"]]
    assert phases == ["subfields", "tasks"]
    # Full prompt and response carried in field log
    assert flog["generation_log"][0]["prompt"]
    assert flog["generation_log"][0]["response"]


def test_skip_scenario_when_tools_parse_empty(fake_llm, tmp_output):
    """A scenario whose phase-3 tools parse as empty is skipped — no JSON written.

    Next run can retry without manual cleanup (next_artifact_index handles it).
    """
    fake_llm.queue(_subfields_response(["S1"]))
    fake_llm.queue(_tasks_response(["T1"]))
    fake_llm.queue("no valid tools in this response")  # phase 3 returns nothing parseable

    saved = generate_environments(
        fields=["Fx"],
        output_dir=tmp_output,
        llm=fake_llm,
        max_subfields=1,
        max_tasks_per_subfield=1,
    )

    assert saved == []
    spec_files = [p.name for p in tmp_output.iterdir() if p.name.startswith("fx_spec_")]
    assert spec_files == []
    # Field log is still written (phases 1-2 succeeded)
    assert (tmp_output / "fx_field_gen.json").exists()


def test_partial_skip_keeps_healthy_scenarios(fake_llm, tmp_output):
    """Two scenarios under one subfield; one has bad tools, the other is healthy.

    Only the healthy scenario is written to disk.
    """
    tools = [_make_tool("Good"), _make_tool("Better")]

    # 1 subfield × 2 tasks → phase 2 is a single call, phase 3 is batched (size 2)
    fake_llm.queue(_subfields_response(["SA"]))
    fake_llm.queue(_tasks_response(["T_bad", "T_good"]))
    fake_llm.queue_batch(["no tools here", _tools_response(tools)])

    saved = generate_environments(
        fields=["Fmix"],
        output_dir=tmp_output,
        llm=fake_llm,
        max_subfields=1,
        max_tasks_per_subfield=2,
    )
    assert len(saved) == 1
    assert saved[0]["task"] == "T_good"
    spec_files = sorted(p.name for p in tmp_output.iterdir() if p.name.startswith("fmix_spec_"))
    assert spec_files == ["fmix_spec_000.json"]


def test_field_log_appends_across_reruns(fake_llm, tmp_output):
    """Re-running env_generation for the same field merges phase-1/phase-2 entries.

    Scenario files from the first run reference `{field_slug}_field_gen.json` via
    `shared_with_field_log`, so overwriting it would lose the prompts they point to.
    """
    # Run 1 — 1 subfield × 1 task
    fake_llm.queue(_subfields_response(["S_R1"]))
    fake_llm.queue(_tasks_response(["T_R1"]))
    fake_llm.queue(_tools_response([_make_tool("X")]))
    generate_environments(
        fields=["F"],
        output_dir=tmp_output,
        llm=fake_llm,
        max_subfields=1,
        max_tasks_per_subfield=1,
    )

    log_path = tmp_output / "f_field_gen.json"
    run1 = json.loads(log_path.read_text())
    assert [e["phase"] for e in run1["generation_log"]] == ["subfields", "tasks"]
    run1_usage_p = run1["usage"]["prompt_tokens"]
    run1_usage_c = run1["usage"]["completion_tokens"]

    # Run 2 — 1 more subfield × 1 task
    fake_llm.queue(_subfields_response(["S_R2"]))
    fake_llm.queue(_tasks_response(["T_R2"]))
    fake_llm.queue(_tools_response([_make_tool("Y")]))
    generate_environments(
        fields=["F"],
        output_dir=tmp_output,
        llm=fake_llm,
        max_subfields=1,
        max_tasks_per_subfield=1,
    )

    run2 = json.loads(log_path.read_text())
    # Run 1 entries preserved; Run 2 entries appended → 4 total
    phases = [e["phase"] for e in run2["generation_log"]]
    assert phases == ["subfields", "tasks", "subfields", "tasks"]
    # Usage accumulated across runs
    assert run2["usage"]["prompt_tokens"] > run1_usage_p
    assert run2["usage"]["completion_tokens"] > run1_usage_c
    # Elapsed accumulated across runs
    assert run2["elapsed_s"] >= run1["elapsed_s"]


def test_spec_id_advances_when_output_dir_has_existing(fake_llm, tmp_output):
    (tmp_output / "existing_spec_005.json").touch()  # unrelated prefix, ignored
    # 1 subfield × 1 task → single calls for phases 2, 3
    fake_llm.queue(_subfields_response(["S"]))
    fake_llm.queue(_tasks_response(["T"]))
    fake_llm.queue(_tools_response([_make_tool("Z")]))

    saved = generate_environments(
        fields=["NewField"],
        output_dir=tmp_output,
        llm=fake_llm,
        max_subfields=1,
        max_tasks_per_subfield=1,
    )
    assert saved[0]["spec_id"] == "newfield_spec_000"
