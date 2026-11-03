"""One rollout -> the messages and tools of one dataset row, or a drop reason.

Reads each turn the way the orchestrator did: the assistant envelope is the first JSON object
`utils.extract_json_objects` finds; a falsy `tool_call` is what made the orchestrator nudge;
`"<STOP>" in tool_call` ends the chat; a call is converted with `tool_calls.convert_call` and kept
only if it is a plain-literal call; a tool reply must be a JSON object with an int status_code.
Any violation drops the whole trajectory with a reason; nothing is truncated or repaired.

Verified on 2026-11-03 over all 210,925 SUCCESSFUL trajectories of VALID tasks: 167,586 kept, 43,339
dropped (23,612 unreadable assistant replies, 9,738 replies without a tool_call, 9,078 malformed
calls, 806 without <STOP>, 105 other), zero invariant violations on the kept rows.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from utils import extract_json_objects
from dataset_export.row_schema import (
    STOP, assistant_call_message, final_message, tool_message, user_message,
)
from dataset_export.tool_calls import convert_call, is_plain_literal_call
from dataset_export.tool_schemas import tools_to_functions

UNKNOWN_TOOL_PREFIX = "Unknown tool name"

DROP_REASONS = (
    "no_summary", "no_tools", "duplicate_tool_names", "no_solver_chat",
    "unexpected_role",        # a nudge ("No tool call detected") or a missing reply
    "envelope_unreadable",    # no JSON object in the assistant reply
    "no_tool_call",           # envelope without a tool_call (the orchestrator nudged)
    "tool_call_not_str", "reason_not_str",
    "malformed_call",         # not a plain-literal call (syntax error, bareword, positional)
    "unknown_tool",
    "tool_reply_not_json", "tool_reply_no_status", "unknown_tool_reply",
    "empty_final_answer", "no_tool_calls", "no_final_stop",
)


def read_envelope(content: Any) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """(envelope, None) or (None, reason). Mirrors orchestrator.py: objs[0] if it is a dict."""
    objs = extract_json_objects(content) if isinstance(content, str) else []
    if not objs or not isinstance(objs[0], dict):
        return None, "envelope_unreadable"
    env = objs[0]
    if not env.get("tool_call"):
        return None, "no_tool_call"
    if not isinstance(env["tool_call"], str):
        return None, "tool_call_not_str"
    reason = env.get("reason")
    if reason is None:
        reason = ""
    if not isinstance(reason, str):
        return None, "reason_not_str"
    return {"reason": reason.strip(), "tool_call": env["tool_call"].strip()}, None


def read_tool_reply(content: Any) -> Tuple[Optional[str], Optional[str]]:
    """(content string to store, None) or (None, reason). `explanation` removed if present."""
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return None, "tool_reply_not_json"
    if not isinstance(payload, dict) or not isinstance(payload.get("status_code"), int):
        return None, "tool_reply_no_status"
    resp = payload.get("response")
    if isinstance(resp, dict) and str(resp.get("error", "")).startswith(UNKNOWN_TOOL_PREFIX):
        return None, "unknown_tool_reply"
    if "explanation" in payload:
        payload = {k: v for k, v in payload.items() if k != "explanation"}
        return json.dumps(payload, ensure_ascii=False), None
    return content, None


def assemble(traj: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Returns ({"messages", "tools", "n_calls", "n_failed"}, None) or (None, drop reason)."""
    summary = ((traj.get("ground_truth") or {}).get("summary") or "").strip()
    if not summary:
        return None, "no_summary"
    raw_tools = traj.get("tools") or []
    if not raw_tools:
        return None, "no_tools"
    tools = tools_to_functions(raw_tools)
    names = [t["function"]["name"] for t in tools]
    if len(set(names)) != len(names):
        return None, "duplicate_tool_names"
    chat = traj.get("solver_chat") or []
    if not chat or chat[0].get("role") != "user":
        return None, "no_solver_chat"

    messages: List[Dict[str, Any]] = [user_message(summary)]
    expected = "assistant"
    n_calls = n_failed = 0
    for m in chat[1:]:
        role = m.get("role")
        if role != expected:
            return None, "unexpected_role"
        if role == "assistant":
            env, reason = read_envelope(m.get("content"))
            if env is None:
                return None, reason
            if STOP in env["tool_call"]:
                if not env["reason"]:
                    return None, "empty_final_answer"
                if n_calls == 0:
                    return None, "no_tool_calls"
                messages.append(final_message(env["reason"]))
                return {"messages": messages, "tools": tools, "n_calls": n_calls, "n_failed": n_failed}, None
            c = convert_call(env["tool_call"])
            if not is_plain_literal_call(c):
                return None, "malformed_call"
            if c.name not in names:
                return None, "unknown_tool"
            messages.append(assistant_call_message(env["reason"], c.name, c.arguments))
            n_calls += 1
            expected = "tool"
        else:
            content, reason = read_tool_reply(m.get("content"))
            if content is None:
                return None, reason
            if not 200 <= json.loads(content)["status_code"] < 300:
                n_failed += 1
            messages.append(tool_message(content))
            expected = "assistant"
    return None, "no_final_stop"
