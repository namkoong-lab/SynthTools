"""Regression tests for task_generation after the utils refactor.

Ensures:
- Public symbols still importable.
- `generate_trajectory` runs end-to-end against a FakeLLM with canned responses.
- Output JSON shape (keys + nested structure) matches the committed snapshot.
- Numbering (from the moved `next_artifact_index`) behaves exactly as before.
"""

import json
from pathlib import Path

import pytest


# --- Public symbols --------------------------------------------------------

def test_public_symbols_importable():
    from task_generation import generate
    assert hasattr(generate, "generate_trajectory")
    assert hasattr(generate, "make_task_id")
    assert hasattr(generate, "load_tools_dataset")


# --- make_task_id numbering via the shared next_artifact_index --------------

def test_make_task_id_numbering(tmp_path: Path):
    from task_generation.generate import make_task_id
    spec = "aerospace_and_defense_tool_spec_1"
    ids = [f"{spec}.Tool"]

    # empty dir -> _000
    assert make_task_id(ids, tmp_path) == f"{spec}_000"

    # contiguous
    for i in range(3):
        (tmp_path / f"{spec}_{i:03d}.json").touch()
        (tmp_path / f"{spec}_{i:03d}.debug.json").touch()
    assert make_task_id(ids, tmp_path) == f"{spec}_003"

    # gap -> still max+1
    (tmp_path / f"{spec}_001.json").unlink()
    assert make_task_id(ids, tmp_path) == f"{spec}_003"

    # orphan debug at higher index
    (tmp_path / f"{spec}_007.debug.json").touch()
    assert make_task_id(ids, tmp_path) == f"{spec}_008"

    # unrelated spec ignored
    (tmp_path / "unrelated_tool_spec_2_999.json").touch()
    assert make_task_id(ids, tmp_path) == f"{spec}_008"


# --- End-to-end run with canned FakeLLM responses --------------------------

_TOOL_SPEC = {
    "id": "mock_spec_1.MockTool",
    "field": "Mock",
    "subfield": "SubMock",
    "task": "Do Mock Things",
    "tool_name": "MockTool",
    "tool": {
        "tool_name": "MockTool",
        "tool_description": "Mock tool for regression tests.",
        "parameters": {
            "arg1": {"type": "string", "required": True, "description": "arg"},
        },
        "error_messages": ["Missing required parameter: arg1"],
        "usage": "pass arg1",
        "output_details": {"result": {"type": "string"}},
    },
}


def _canned_traj_responses():
    """Canned responses for a 1-tool task, in call order:
       evolver.t0 -> solver.turn_0 -> sim.parameter_check -> sim.simulate -> env_sim.update
    """
    evolver = json.dumps({
        "tool_name": "MockTool",
        "tool_call": "MockTool(arg1='hello')",
        "env_metadata": {"seed": "x"},
        "task_description": "Invoke MockTool with argument hello.",
    })
    solver = json.dumps({
        "reason": "Single-tool task; arg1='hello' from the task.",
        "tool_call": "MockTool(arg1='hello')",
    })
    param_check = json.dumps({"status": "PASS", "status_code": 200, "error_message": None})
    simulate = json.dumps({"status_code": 200, "response": {"result": "ok"}, "explanation": "mocked"})
    env_update = json.dumps({"edited_metadata": {"seed": "x"}, "full_metadata": {"seed": "x"}})
    return [evolver, solver, param_check, simulate, env_update]


def test_solver_user_message_filters_to_openai_subset():
    """Solver must only see {tool_name, tool_description, parameters} — not
    simulator-only fields like `error_messages`, `usage`, `output_details`."""
    from roles.task_solver import TaskSolver

    full_schema = {
        "tool_name": "Foo",
        "tool_description": "do foo",
        "parameters": {"x": {"type": "string", "required": True}},
        "error_messages": ["Missing required parameter: x"],
        "usage": "Provide x as a string",
        "output_details": {"result": {"type": "string"}},
    }
    msg = TaskSolver.build_user_message("do it", full_schema)
    assert "Foo" in msg
    assert "do foo" in msg
    assert '"x"' in msg
    # Simulator-only fields must NOT leak through
    assert "error_messages" not in msg
    assert "Missing required parameter" not in msg
    assert "Provide x as a string" not in msg
    assert "output_details" not in msg


def test_solver_user_message_renders_multi_tool_list():
    """When passed a list of tools, the user message lists all of them with
    'Tools available for this task (select one):' header and numbered
    blocks. Each tool's OpenAI-subset is preserved; simulator-only fields are
    stripped from every tool."""
    from roles.task_solver import TaskSolver

    tools = [
        {
            "tool_name": "Foo",
            "tool_description": "do foo",
            "parameters": {"x": {"type": "string", "required": True}},
            "error_messages": ["leak1"],
        },
        {
            "tool_name": "Bar",
            "tool_description": "do bar",
            "parameters": {"y": {"type": "integer", "required": True}},
            "usage": "leak2",
        },
        {
            "tool_name": "Baz",
            "tool_description": "do baz",
            "parameters": {"z": {"type": "boolean", "required": False}},
            "output_details": {"leak3": "shape"},
        },
    ]
    msg = TaskSolver.build_user_message("pick one", tools)
    assert "Tools available for this task (select one):" in msg
    assert "[1] " in msg and "[2] " in msg and "[3] " in msg
    for name in ("Foo", "Bar", "Baz"):
        assert name in msg
    # Simulator-only fields stripped from each tool entry
    for leak in ("error_messages", "leak1", "usage", "leak2", "output_details", "leak3"):
        assert leak not in msg
    # Single-tool legacy path still works (backward compat)
    legacy = TaskSolver.build_user_message("do it", tools[0])
    assert "Tool to use:" in legacy
    assert "Tools available" not in legacy


def test_non_2xx_simulator_response_is_not_success(fake_llm, tmp_output):
    """A simulator 4xx response must NOT mark task_solved; the attempt should be retried."""
    from task_generation.generate import generate_trajectory

    dataset_path = tmp_output / "tools_dataset.jsonl"
    with open(dataset_path, "w") as f:
        f.write(json.dumps(_TOOL_SPEC) + "\n")
    out_dir = tmp_output / "task_out"
    out_dir.mkdir()

    evolver = json.dumps({
        "tool_name": "MockTool",
        "tool_call": "MockTool(arg1='hello')",
        "env_metadata": {"seed": "x"},
        "task_description": "Invoke MockTool with arg1='hello'.",
    })
    solver = json.dumps({"reason": "direct call", "tool_call": "MockTool(arg1='hello')"})
    param_check = json.dumps({"status": "PASS", "status_code": 200, "error_message": None})
    sim_404 = json.dumps({"status_code": 404, "response": {"error": "not found"}, "explanation": "x"})
    sim_200 = json.dumps({"status_code": 200, "response": {"result": "ok"}, "explanation": "x"})
    env_update = json.dumps({"edited_metadata": {"seed": "x"}, "full_metadata": {"seed": "x"}})

    # Attempt 0: evolver, solver, param_check, sim_404       → task_solved=False (new: tighter check)
    # Attempt 1: evolver, solver, param_check, sim_200, env  → task_solved=True
    for r in [evolver, solver, param_check, sim_404,
              evolver, solver, param_check, sim_200, env_update]:
        fake_llm.queue(r)

    task = generate_trajectory(
        tool_ids=["mock_spec_1.MockTool"],
        tools_dataset_path=dataset_path,
        output_dir=out_dir,
        llm=fake_llm,
        max_solver_turns=5,
        max_retries=5,
        verifiable=False,
        debug=False,
    )

    # Two turns recorded for tool_idx=0 (attempt 0 and attempt 1)
    assert len(task["turns"]) == 2
    t0, t1 = task["turns"]
    assert t0["tool_idx"] == 0 and t0["attempt"] == 0
    assert t0["env_update"] is None, "4xx should NOT populate env_update"
    assert t1["tool_idx"] == 0 and t1["attempt"] == 1
    assert t1["env_update"] is not None, "200 on retry should populate env_update"


def test_verifiable_judge_solved_but_non_2xx_blocks_env_update(fake_llm, tmp_output):
    """verifiable=True: even if the judge returns task_solved=True, a non-2xx
    tool response (here 500) must NOT populate env_update. The deterministic
    status-code gate overrides the LLM verdict."""
    from task_generation.generate import generate_trajectory

    dataset_path = tmp_output / "tools_dataset.jsonl"
    with open(dataset_path, "w") as f:
        f.write(json.dumps(_TOOL_SPEC) + "\n")
    out_dir = tmp_output / "task_out"
    out_dir.mkdir()

    evolver = json.dumps({
        "tool_name": "MockTool",
        "tool_call": "MockTool(arg1='hello')",
        "env_metadata": {"seed": "x"},
        "task_description": "Invoke MockTool with arg1='hello'.",
    })
    solver = json.dumps({"reason": "direct call", "tool_call": "MockTool(arg1='hello')"})
    param_check = json.dumps({"status": "PASS", "status_code": 200, "error_message": None})
    sim_500 = json.dumps({"status_code": 500, "response": {"error": "server error"}, "explanation": "x"})
    sim_200 = json.dumps({"status_code": 200, "response": {"result": "ok"}, "explanation": "x"})
    judge_solved = json.dumps({
        "argument_citations": [], "arguments_grounded": True, "tool_call_equality": True,
        "tool_succeeded": True, "task_solved": True, "task_solved_confidence": 1.0,
        "task_solvability": 1.0, "feedback": "",
    })
    env_update = json.dumps({"edited_metadata": {"seed": "x"}, "full_metadata": {"seed": "x"}})

    # attempt 0: evolver, solver, param_check, sim_500, judge(solved=True) → gate blocks → re-roll
    # attempt 1: evolver, solver, param_check, sim_200, judge(solved=True), env_update → solved
    for r in [evolver, solver, param_check, sim_500, judge_solved,
              evolver, solver, param_check, sim_200, judge_solved, env_update]:
        fake_llm.queue(r)

    task = generate_trajectory(
        tool_ids=["mock_spec_1.MockTool"],
        tools_dataset_path=dataset_path,
        output_dir=out_dir,
        llm=fake_llm,
        max_solver_turns=5,
        max_retries=5,
        max_solver_retries_in_place=1,
        verifiable=True,
        debug=False,
    )

    assert len(task["turns"]) == 2
    t0, t1 = task["turns"]
    assert t0["env_update"] is None, "judge said solved but 500 must block env_update"
    assert t1["env_update"] is not None, "200 + judge solved must populate env_update"


def test_end_to_end_with_fake_llm(fake_llm, tmp_output):
    """Write a mock tools_dataset.jsonl, run one task, verify shape."""
    from task_generation.generate import generate_trajectory

    # Prepare tools_dataset.jsonl
    dataset_path = tmp_output / "tools_dataset.jsonl"
    with open(dataset_path, "w") as f:
        f.write(json.dumps(_TOOL_SPEC) + "\n")

    out_dir = tmp_output / "task_out"
    out_dir.mkdir()

    # Queue every call the orchestrator makes (in order, single calls)
    for r in _canned_traj_responses():
        fake_llm.queue(r)

    task = generate_trajectory(
        tool_ids=["mock_spec_1.MockTool"],
        tools_dataset_path=dataset_path,
        output_dir=out_dir,
        llm=fake_llm,
        max_solver_turns=5,
        max_retries=5,
        verifiable=False,
        debug=True,
    )

    # Files written
    files = sorted(p.name for p in out_dir.iterdir())
    assert files == ["mock_spec_1_000.debug.json", "mock_spec_1_000.json"]

    # Top-level shape (use keys, not values, to dodge timing flakiness)
    assert set(task.keys()) == {
        "task_id", "model", "config", "tool_ids", "tools", "turns",
        "solver_chat", "usage", "generation_time_s",
    }
    assert task["task_id"] == "mock_spec_1_000"
    assert task["model"] == "fake-model"
    assert len(task["turns"]) == 1
    turn = task["turns"][0]
    assert set(turn.keys()) == {
        "tool_idx", "attempt", "tool_id", "env_state_before",
        "task", "env_state_after_task", "chat", "judge",
        "env_update", "env_state_after",
    }
    assert turn["tool_idx"] == 0
    assert turn["attempt"] == 0

    # solver_chat: exactly [user, assistant, tool] (one successful tool call)
    assert [m["role"] for m in task["solver_chat"]] == ["user", "assistant", "tool"]

    # usage aggregated (FakeLLM gives 10/20 per single call; 5 calls → 50/100 total)
    # NOTE: the param_check response and simulate are recorded as separate events
    # but share usage tracking on the ToolSimulator side, so we just assert positive sum.
    assert task["usage"]["total_tokens"] > 0
    assert task["usage"]["prompt_tokens"] > 0
    assert task["usage"]["completion_tokens"] > 0

    # Debug file round-trip
    with open(out_dir / "mock_spec_1_000.debug.json") as f:
        debug = json.load(f)
    assert debug["task_id"] == "mock_spec_1_000"
    assert len(debug["events"]) >= 4   # evolver + solver + sim (split into param_check+simulate) + env_update


# --- Verifiable mode: judge gating (grounding-first, re-roll vs in-place) ---


def test_evolver_rerolls_on_arguments_grounded_false(fake_llm, tmp_output):
    """Judge says arguments_grounded=False on attempt 0 → outer loop must
    re-roll the evolver. Judge feedback must reach the next evolver call via
    `unsuccessful_tasks_list[*].judge_explanation`."""
    from task_generation.generate import generate_trajectory

    dataset_path = tmp_output / "tools_dataset.jsonl"
    with open(dataset_path, "w") as f:
        f.write(json.dumps(_TOOL_SPEC) + "\n")
    out_dir = tmp_output / "task_out"
    out_dir.mkdir()

    evolver_attempt0 = json.dumps({
        "tool_name": "MockTool",
        "tool_call": "MockTool(arg1='UNGROUNDED-VALUE')",
        "env_metadata": {"seed": "x"},
        "task_description": "Use the existing thing.",
    })
    evolver_attempt1 = json.dumps({
        "tool_name": "MockTool",
        "tool_call": "MockTool(arg1='hello')",
        "env_metadata": {"seed": "x"},
        "task_description": "Invoke MockTool with arg1='hello'.",
    })
    solver = json.dumps({"reason": "...", "tool_call": "MockTool(arg1='hello')"})
    param_check = json.dumps({"status": "PASS", "status_code": 200, "error_message": None})
    sim_200 = json.dumps({"status_code": 200, "response": {"result": "ok"}, "explanation": "x"})
    judge_ungrounded = json.dumps({
        "argument_citations": [],
        "arguments_grounded": False,
        "tool_call_equality": True,
        "task_solved": False,
        "task_solved_confidence": 1.0,
        "task_solvability": 0.0,
        "feedback": "Value 'UNGROUNDED-VALUE' is not in the task description.",
    })
    judge_grounded = json.dumps({
        "argument_citations": [],
        "arguments_grounded": True,
        "tool_call_equality": True,
        "task_solved": True,
        "task_solved_confidence": 0.95,
        "task_solvability": 1.0,
        "feedback": "",
    })
    env_update = json.dumps({"edited_metadata": {"seed": "x"}, "full_metadata": {"seed": "x"}})

    # attempt 0: evolver, solver, param_check, sim_200, judge_ungrounded   → re-roll
    # attempt 1: evolver, solver, param_check, sim_200, judge_grounded, env_update → solved
    for r in [
        evolver_attempt0, solver, param_check, sim_200, judge_ungrounded,
        evolver_attempt1, solver, param_check, sim_200, judge_grounded, env_update,
    ]:
        fake_llm.queue(r)

    task = generate_trajectory(
        tool_ids=["mock_spec_1.MockTool"],
        tools_dataset_path=dataset_path,
        output_dir=out_dir,
        llm=fake_llm,
        max_solver_turns=5,
        max_retries=5,
        max_solver_retries_in_place=2,
        verifiable=True,
        debug=False,
    )

    # Two turns recorded: attempt 0 (re-rolled) + attempt 1 (solved)
    assert len(task["turns"]) == 2
    t0, t1 = task["turns"]
    assert t0["attempt"] == 0 and t0["env_update"] is None
    assert t1["attempt"] == 1 and t1["env_update"] is not None

    # The evolver's second call must have received the judge feedback as the
    # `judge_explanation` of the prior unsuccessful attempt. After the
    # chat-format refactor t1 calls pass a list of messages, not a single
    # string — so detect "this is an evolver call" by scanning content
    # across whichever shape was used.
    def _is_evolver_call(call):
        msgs = call["messages"]
        if isinstance(msgs, str):
            return "Task Evolver" in msgs
        if isinstance(msgs, list):
            return any("Task Evolver" in (m.get("content", "") or "") for m in msgs)
        return False

    def _haystack(call):
        msgs = call["messages"]
        if isinstance(msgs, str):
            return msgs
        return "\n".join((m.get("content", "") or "") for m in msgs)

    evolver_calls = [c for c in fake_llm.calls if _is_evolver_call(c)]
    assert len(evolver_calls) == 2, f"expected 2 evolver calls, got {len(evolver_calls)}"
    assert "UNGROUNDED-VALUE" in _haystack(evolver_calls[1]), \
        "judge feedback must reach the second evolver call"


def test_solver_retries_in_place_when_grounded_but_not_solved(fake_llm, tmp_output):
    """Judge says arguments_grounded=True AND task_solved=False on attempt 0 →
    solver retries in place WITHOUT re-rolling the evolver. Same task, same
    expected_tool_call, fresh solver attempt."""
    from task_generation.generate import generate_trajectory

    dataset_path = tmp_output / "tools_dataset.jsonl"
    with open(dataset_path, "w") as f:
        f.write(json.dumps(_TOOL_SPEC) + "\n")
    out_dir = tmp_output / "task_out"
    out_dir.mkdir()

    evolver = json.dumps({
        "tool_name": "MockTool",
        "tool_call": "MockTool(arg1='hello')",
        "env_metadata": {"seed": "x"},
        "task_description": "Invoke MockTool with arg1='hello'.",
    })
    solver_wrong = json.dumps({"reason": "miss", "tool_call": "MockTool(arg1='wrong')"})
    solver_right = json.dumps({"reason": "got it", "tool_call": "MockTool(arg1='hello')"})
    param_check = json.dumps({"status": "PASS", "status_code": 200, "error_message": None})
    sim_200 = json.dumps({"status_code": 200, "response": {"result": "ok"}, "explanation": "x"})
    judge_grounded_unsolved = json.dumps({
        "argument_citations": [],
        "arguments_grounded": True,
        "tool_call_equality": False,
        "task_solved": False,
        "task_solved_confidence": 1.0,
        "task_solvability": 1.0,
        "feedback": "solver used 'wrong' instead of 'hello'",
    })
    judge_solved = json.dumps({
        "argument_citations": [],
        "arguments_grounded": True,
        "tool_call_equality": True,
        "task_solved": True,
        "task_solved_confidence": 0.95,
        "task_solvability": 1.0,
        "feedback": "",
    })
    env_update = json.dumps({"edited_metadata": {"seed": "x"}, "full_metadata": {"seed": "x"}})

    # ONE evolver call. TWO solver attempts. TWO judge calls. Then env_update on solved.
    for r in [
        evolver,
        solver_wrong, param_check, sim_200, judge_grounded_unsolved,
        solver_right, param_check, sim_200, judge_solved,
        env_update,
    ]:
        fake_llm.queue(r)

    task = generate_trajectory(
        tool_ids=["mock_spec_1.MockTool"],
        tools_dataset_path=dataset_path,
        output_dir=out_dir,
        llm=fake_llm,
        max_solver_turns=5,
        max_retries=5,
        max_solver_retries_in_place=2,
        verifiable=True,
        debug=False,
    )

    # ONE turn recorded — same evolver attempt, but solver retried internally.
    assert len(task["turns"]) == 1
    t = task["turns"][0]
    assert t["attempt"] == 0
    assert t["env_update"] is not None
    # Final judge verdict must be the SOLVED one.
    assert t["judge"]["task_solved"] is True


# --- Mode B: single env_spec -----------------------------------------------

def _write_dataset(path: Path, tool_specs):
    with open(path, "w") as f:
        for spec in tool_specs:
            f.write(json.dumps(spec) + "\n")


def _mock_tool_spec(spec_id: str, tool_name: str):
    return {
        "id": f"{spec_id}.{tool_name}",
        "field": "Mock",
        "subfield": "SubMock",
        "task": "Do Mock Things",
        "tool_name": tool_name,
        "tool": {
            "tool_name": tool_name,
            "tool_description": f"Mock tool {tool_name}.",
            "parameters": {"arg1": {"type": "string", "required": True, "description": "arg"}},
            "error_messages": ["Missing required parameter: arg1"],
            "usage": "pass arg1",
            "output_details": {"result": {"type": "string"}},
        },
    }


def _mock_env_spec(spec_id: str, field: str, tool_names, sequences):
    return {
        "schema_version": "env_spec.v2",
        "spec_id": spec_id,
        "field": field,
        "subfield": "Sub",
        "task": "Task",
        "tools": [{"tool_name": n} for n in tool_names],
        "sequences": sequences,
    }


def test_mode_b_iterates_sequences_and_skips_existing(fake_llm, tmp_output):
    """Mode B: run one task per sequence, skip any {spec_id}_{seq_key}.json that exists."""
    from task_generation.generate import generate_trajectories_for_spec

    spec_id = "mock_spec_1"
    dataset_path = tmp_output / "tools_dataset.jsonl"
    _write_dataset(dataset_path, [_mock_tool_spec(spec_id, "MockTool")])

    env_specs_dir = tmp_output / "env_specs"
    env_specs_dir.mkdir()
    env_spec = _mock_env_spec(
        spec_id, "Mock", ["MockTool"],
        {"seq1": ["MockTool"], "seq2": ["MockTool"]},
    )
    env_spec_path = env_specs_dir / f"{spec_id}.json"
    env_spec_path.write_text(json.dumps(env_spec))

    out_dir = tmp_output / "task_out"
    out_dir.mkdir()

    # Pre-create seq1's file so it gets skipped
    (out_dir / f"{spec_id}_seq1.json").write_text('{"pre-existing": true}')

    # Queue responses for seq2 only (one task)
    for r in _canned_traj_responses():
        fake_llm.queue(r)

    tasks = generate_trajectories_for_spec(
        env_spec_path=env_spec_path,
        tools_dataset_path=dataset_path,
        output_dir=out_dir,
        llm=fake_llm,
        verifiable=False,
        debug=False,
    )

    assert len(tasks) == 1
    assert tasks[0]["task_id"] == f"{spec_id}_seq2"

    # seq1 file preserved as pre-existing sentinel
    pre = json.loads((out_dir / f"{spec_id}_seq1.json").read_text())
    assert pre == {"pre-existing": True}

    # seq2 task written with the expected name
    assert (out_dir / f"{spec_id}_seq2.json").exists()


def test_mode_b_no_llm_calls_when_all_skipped(fake_llm, tmp_output):
    """If every {spec_id}_{seq_key}.json already exists, no LLM call is made."""
    from task_generation.generate import generate_trajectories_for_spec

    spec_id = "mock_spec_1"
    dataset_path = tmp_output / "tools_dataset.jsonl"
    _write_dataset(dataset_path, [_mock_tool_spec(spec_id, "MockTool")])

    env_specs_dir = tmp_output / "env_specs"
    env_specs_dir.mkdir()
    env_spec = _mock_env_spec(
        spec_id, "Mock", ["MockTool"],
        {"seq1": ["MockTool"]},
    )
    env_spec_path = env_specs_dir / f"{spec_id}.json"
    env_spec_path.write_text(json.dumps(env_spec))

    out_dir = tmp_output / "task_out"
    out_dir.mkdir()
    (out_dir / f"{spec_id}_seq1.json").write_text("{}")

    tasks = generate_trajectories_for_spec(
        env_spec_path=env_spec_path,
        tools_dataset_path=dataset_path,
        output_dir=out_dir,
        llm=fake_llm,
        verifiable=False,
        debug=False,
    )
    assert tasks == []
    assert fake_llm.calls == []


def test_mode_b_warns_when_sequences_empty(fake_llm, tmp_output):
    """Empty sequences block → nothing to do, no LLM calls."""
    from task_generation.generate import generate_trajectories_for_spec

    spec_id = "mock_spec_1"
    dataset_path = tmp_output / "tools_dataset.jsonl"
    _write_dataset(dataset_path, [_mock_tool_spec(spec_id, "MockTool")])

    env_specs_dir = tmp_output / "env_specs"
    env_specs_dir.mkdir()
    env_spec = _mock_env_spec(spec_id, "Mock", ["MockTool"], {})
    env_spec_path = env_specs_dir / f"{spec_id}.json"
    env_spec_path.write_text(json.dumps(env_spec))

    out_dir = tmp_output / "task_out"
    out_dir.mkdir()

    tasks = generate_trajectories_for_spec(
        env_spec_path=env_spec_path,
        tools_dataset_path=dataset_path,
        output_dir=out_dir,
        llm=fake_llm,
        verifiable=False,
        debug=False,
    )
    assert tasks == []
    assert fake_llm.calls == []


# --- Mode C: all specs in a field ------------------------------------------

def test_mode_c_iterates_all_specs_in_field(fake_llm, tmp_output):
    """Mode C picks every `*_spec_*.json` whose `field` matches, skipping others."""
    from task_generation.generate import generate_trajectories_for_field

    dataset_path = tmp_output / "tools_dataset.jsonl"
    _write_dataset(dataset_path, [
        _mock_tool_spec("mock_spec_1", "MockTool"),
        _mock_tool_spec("mock_spec_2", "MockTool"),
        _mock_tool_spec("other_spec_1", "MockTool"),
    ])

    env_specs_dir = tmp_output / "env_specs"
    env_specs_dir.mkdir()
    for spec_id, field in [
        ("mock_spec_1", "Mock"),
        ("mock_spec_2", "Mock"),
        ("other_spec_1", "OtherField"),
    ]:
        env_spec = _mock_env_spec(spec_id, field, ["MockTool"], {"seq1": ["MockTool"]})
        (env_specs_dir / f"{spec_id}.json").write_text(json.dumps(env_spec))

    out_dir = tmp_output / "task_out"
    out_dir.mkdir()

    # Queue responses for 2 tasks (mock_spec_1 + mock_spec_2)
    for _ in range(2):
        for r in _canned_traj_responses():
            fake_llm.queue(r)

    tasks = generate_trajectories_for_field(
        env_specs_dir=env_specs_dir,
        field="Mock",
        tools_dataset_path=dataset_path,
        output_dir=out_dir,
        llm=fake_llm,
        verifiable=False,
        debug=False,
    )

    assert len(tasks) == 2
    assert {t["task_id"] for t in tasks} == {"mock_spec_1_seq1", "mock_spec_2_seq1"}
    assert (out_dir / "mock_spec_1_seq1.json").exists()
    assert (out_dir / "mock_spec_2_seq1.json").exists()
    # OtherField spec not touched
    assert not (out_dir / "other_spec_1_seq1.json").exists()
