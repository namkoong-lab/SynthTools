"""Regression tests for two trajectory-orchestrator fixes:
  - state threading: the simulator sees prior successful calls/outputs each turn
  - explanation strip: the simulator's 'explanation' never reaches the agent
"""
import json
import tempfile
from pathlib import Path

from trajectory_generation.loader import Task
from trajectory_generation.orchestrator import generate_trajectory


class _RecordingLLM:
    """Plays scripted responses; records the text of every prompt received."""
    def __init__(self, scripted):
        self._it = iter(scripted)
        self.model = "fake"
        self.last_usage = None
        self.last_usage_per_request = None
        self.prompts = []

    def __call__(self, messages):
        self.prompts.append(messages if isinstance(messages, str)
                            else json.dumps(messages, default=str))
        try:
            r = next(self._it)
        except StopIteration:
            r = json.dumps({"reason": "x", "tool_call": "<STOP>"})
        self.last_usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        return r


def _task():
    return Task(
        id="t_spec_000_seq1", field="F", summary="create then use",
        tools=[
            {"tool_name": "Create", "tool_description": "create",
             "parameters": {"name": {"type": "string", "required": True}},
             "output_details": {"id": {"type": "string"}}},
            {"tool_name": "Use", "tool_description": "use",
             "parameters": {"id": {"type": "string", "required": True}},
             "output_details": {"ok": {"type": "boolean"}}},
        ],
        gt_tool_calls=["Create(name='x')", "Use(id='id_1')"],
        initial_state={"items": []}, final_state=None,
    )


def _script():
    pc = json.dumps({"status": "PASS", "status_code": 200})
    sim_create = json.dumps({"status_code": 200, "response": {"id": "id_1"},
                             "explanation": "SECRET backend reasoning"})
    sim_use = json.dumps({"status_code": 200, "response": {"ok": True},
                          "explanation": "more secret"})
    return [
        json.dumps({"reason": "c", "tool_call": "Create(name='x')"}), pc, sim_create,
        json.dumps({"reason": "u", "tool_call": "Use(id='id_1')"}), pc, sim_use,
        json.dumps({"reason": "done", "tool_call": "<STOP>"}),
    ]


def _run():
    llm = _RecordingLLM(_script())
    with tempfile.TemporaryDirectory() as d:
        out = generate_trajectory(_task(), llm, output_dir=Path(d),
                                  max_solver_turns=6, debug=False, run_judge=False)
    return llm, out


def test_simulator_sees_prior_calls_and_outputs():
    llm, _ = _run()
    sim_prompts = [p for p in llm.prompts if "API simulator" in p]
    assert len(sim_prompts) == 2
    # the 2nd simulate prompt must carry the 1st call + its output
    assert "previous_tool_calls" in sim_prompts[1]
    assert "Create(name='x')" in sim_prompts[1]
    assert "id_1" in sim_prompts[1]


def test_explanation_never_reaches_the_agent():
    _, out = _run()
    chat_txt = json.dumps(out["solver_chat"], default=str)
    assert "SECRET backend reasoning" not in chat_txt
    assert "more secret" not in chat_txt
    for t in out["turns"]:
        to = t.get("tool_output")
        if isinstance(to, dict):
            assert "explanation" not in to
