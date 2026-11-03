"""The contract of one dataset entry: one row per kept SUCCESSFUL trajectory.

Stored as parquet (HF shard naming); `tools` and `messages` are JSON strings because their nested
keys differ per row and do not fit one Arrow struct. One `json.loads` each gives the exact input of
`tokenizer.apply_chat_template(messages, tools=tools)`.

Row:
    id                        "<task_id>.s<seed>" = the rollout file stem (task_id, seed = split at ".s")
    field                     from task_content.valid.jsonl
    gt_tool_calls             JSON list of the task's ground-truth calls, as released (Python call strings)
    n_target_tokens           Qwen3 tokens of all assistant blocks, thinking on (see tokens.py)
    n_target_tokens_nothink   same with reasoning_content removed
    n_total_tokens            the whole thinking-on render
    tools                     JSON: [{"type":"function","function":{"name","description","parameters":<JSON Schema>}}]
    messages                  JSON: the conversation below

messages:
    {"role":"user",      "content": <task summary>}
    {"role":"assistant", "content":"", "reasoning_content": <reason>,
                         "tool_calls":[{"type":"function","function":{"name":<Tool>,"arguments":<dict>}}]}
    {"role":"tool",      "content": <simulator reply JSON string, "explanation" removed>}
    ... one (assistant, tool) pair per call ...
    {"role":"assistant", "content": <final answer = the <STOP> reason>}

Thinking off at training time = `strip_reasoning(messages)`.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

COLUMNS = (
    "id", "field", "gt_tool_calls",
    "n_target_tokens", "n_target_tokens_nothink", "n_total_tokens",
    "tools", "messages",
)

STOP = "<STOP>"


def arrow_schema():
    import pyarrow as pa
    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("field", pa.string()),
        pa.field("gt_tool_calls", pa.string()),
        pa.field("n_target_tokens", pa.int32()),
        pa.field("n_target_tokens_nothink", pa.int32()),
        pa.field("n_total_tokens", pa.int32()),
        pa.field("tools", pa.string()),
        pa.field("messages", pa.string()),
    ])


def user_message(summary: str) -> Dict[str, Any]:
    return {"role": "user", "content": summary}


def assistant_call_message(reason: str, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": reason,
        "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}],
    }


def tool_message(reply_json: str) -> Dict[str, Any]:
    return {"role": "tool", "content": reply_json}


def final_message(answer: str) -> Dict[str, Any]:
    return {"role": "assistant", "content": answer}


def strip_reasoning(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The same conversation rendered as if thinking were off."""
    return [{k: v for k, v in m.items() if k != "reasoning_content"} for m in messages]


def make_row(id_: str, field: str, gt_tool_calls: List[str], tools: List[Dict[str, Any]],
             messages: List[Dict[str, Any]], counts: Dict[str, int]) -> Dict[str, Any]:
    return {
        "id": id_,
        "field": field,
        "gt_tool_calls": json.dumps(gt_tool_calls, ensure_ascii=False),
        "n_target_tokens": counts["n_target_tokens"],
        "n_target_tokens_nothink": counts["n_target_tokens_nothink"],
        "n_total_tokens": counts["n_total_tokens"],
        "tools": json.dumps(tools, ensure_ascii=False),
        "messages": json.dumps(messages, ensure_ascii=False),
    }
