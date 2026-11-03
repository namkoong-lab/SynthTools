import json
import pytest
from dataset_export.assemble import assemble, read_envelope, read_tool_reply

TOOLS = [{"tool_name": "Search", "tool_description": "d", "parameters": {"q": {"type": "string", "required": True}}},
         {"tool_name": "Save", "tool_description": "d", "parameters": {"items": {"type": "array", "required": True, "items": {"type": "string"}}}}]
RETRY = "No tool call detected. Reply with a single JSON object containing 'reason' and 'tool_call' (use \"<STOP>\" to finish)."

def asst(reason, call): return {"role": "assistant", "content": json.dumps({"reason": reason, "tool_call": call})}
def tool(status, **extra): return {"role": "tool", "content": json.dumps({"status_code": status, "response": {"ok": True}, **extra})}
def traj(chat, summary="Do it."):
    return {"task_id": "demo_spec_000_seq1", "tools": TOOLS, "ground_truth": {"summary": summary},
            "solver_chat": [{"role": "user", "content": "Task: ..."}] + chat}
HAPPY = [asst("search", "Search(q='x')"), tool(200), asst("save", "Save(items=['a'])"), tool(201), asst("Done: saved a.", "<STOP>")]


def test_happy_path():
    row, reason = assemble(traj(HAPPY))
    assert reason is None and row["n_calls"] == 2 and row["n_failed"] == 0
    roles = [m["role"] for m in row["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert row["messages"][0]["content"] == "Do it."
    assert row["messages"][1] == {"role": "assistant", "content": "", "reasoning_content": "search",
                                  "tool_calls": [{"type": "function", "function": {"name": "Search", "arguments": {"q": "x"}}}]}
    assert row["messages"][-1] == {"role": "assistant", "content": "Done: saved a."}
    assert [t["function"]["name"] for t in row["tools"]] == ["Search", "Save"]


def test_semantic_failure_kept_and_counted():
    chat = [asst("try", "Search(q='')"), {"role": "tool", "content": json.dumps({"status": "FAIL", "status_code": 400, "error_message": "empty"})},
            asst("retry", "Search(q='x')"), tool(200), asst("Done.", "<STOP>")]
    row, _ = assemble(traj(chat))
    assert row["n_failed"] == 1 and row["n_calls"] == 2


def test_envelope_read_like_the_orchestrator():
    chat = [{"role": "assistant", "content": json.dumps({"reason": "r", "tool_call": "Search(q='x')"}) + '"}'}, tool(200),
            {"role": "assistant", "content": "```json\n" + json.dumps({"reason": "r2", "tool_call": "Save(items=[])"}) + "\n```"}, tool(200),
            {"role": "assistant", "content": "Answer:\n" + json.dumps({"reason": "Done.", "tool_call": " <STOP> now"})}]
    row, reason = assemble(traj(chat))
    assert reason is None and row["n_calls"] == 2 and row["messages"][-1]["content"] == "Done."


def test_explanation_is_stripped_only_when_present():
    raw = json.dumps({"status_code": 200, "response": {"a": 1}})
    assert read_tool_reply(raw) == (raw, None)
    stripped, _ = read_tool_reply(json.dumps({"status_code": 200, "response": {"a": 1}, "explanation": "secret"}))
    assert json.loads(stripped) == {"status_code": 200, "response": {"a": 1}}


@pytest.mark.parametrize("chat,reason", [
    ([{"role": "assistant", "content": "plain text"}, {"role": "user", "content": RETRY}] + HAPPY, "envelope_unreadable"),
    ([{"role": "assistant", "content": json.dumps({"reason": "r"})}, {"role": "user", "content": RETRY}] + HAPPY, "no_tool_call"),
    ([asst("r", None), {"role": "user", "content": RETRY}] + HAPPY, "no_tool_call"),
    ([{"role": "assistant", "content": json.dumps({"reason": ["x"], "tool_call": "Search(q='x')"})}] + HAPPY[1:], "reason_not_str"),
    ([asst("r", "Search(q='x'")] + HAPPY[1:], "malformed_call"),
    ([asst("r", "Search(q={'a': true})")] + HAPPY[1:], "malformed_call"),
    ([asst("r", "Search('x')")] + HAPPY[1:], "malformed_call"),
    ([asst("r", "Delete(q='x')"), tool(400)] + HAPPY, "unknown_tool"),
    ([asst("r", "Search(q='x')"), {"role": "tool", "content": "null"}] + HAPPY[2:], "tool_reply_no_status"),
    ([asst("r", "Search(q='x')"), {"role": "tool", "content": "not json"}] + HAPPY[2:], "tool_reply_not_json"),
    ([asst("r", "Search(q='x')"), {"role": "tool", "content": json.dumps({"status_code": 400, "response": {"error": "Unknown tool name in call. Choose one of: []"}})}] + HAPPY[2:], "unknown_tool_reply"),
    (HAPPY[:2] + [{"role": "user", "content": RETRY}] + HAPPY[2:], "unexpected_role"),
    (HAPPY[:4], "no_final_stop"),
    ([asst("", "<STOP>")], "empty_final_answer"),
    ([asst("Nothing to do.", "<STOP>")], "no_tool_calls"),
])
def test_drop_reasons(chat, reason):
    assert assemble(traj(chat)) == (None, reason)


def test_missing_summary_or_tools_or_chat():
    t = traj(HAPPY, summary=" "); assert assemble(t) == (None, "no_summary")
    t = traj(HAPPY); t["tools"] = []; assert assemble(t) == (None, "no_tools")
    t = traj(HAPPY); t["solver_chat"] = t["solver_chat"][1:]; assert assemble(t) == (None, "no_solver_chat")
    t = traj(HAPPY); t["tools"] = TOOLS + [TOOLS[0]]; assert assemble(t) == (None, "duplicate_tool_names")


def test_messages_after_stop_are_ignored():
    row, reason = assemble(traj(HAPPY + [tool(200)]))
    assert reason is None and row["messages"][-1]["role"] == "assistant"
