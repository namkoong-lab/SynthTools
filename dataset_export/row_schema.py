"""The contract of one dataset entry: one row per kept SUCCESSFUL trajectory.

Stored as parquet (HF shard naming) written by `datasets`, so `load_dataset` gets typed columns:
`messages` is the Hugging Face conversational format with tool calls in the OpenAI convention
(`function.arguments` is a JSON string), and `tools` is a JSON string of the function-schema list.
Both strings exist because the nested keys differ per row and the `datasets.Json` feature decodes
floats imprecisely (ujson: 0.6 -> 0.6000000000000001), which would alter call arguments on load.
The Qwen3 template renders string arguments verbatim, byte-identical to dict arguments, so
`tokenizer.apply_chat_template(row["messages"], tools=json.loads(row["tools"]))` is the row's render.

Row:
    id                        "<task_id>.s<seed>" = the rollout file stem (task_id, seed = split at ".s")
    field                     from task_content.valid.jsonl
    gt_tool_calls             list of the task's ground-truth calls, as released (Python call strings)
    n_target_tokens           Qwen3 tokens of all assistant blocks, thinking on (see tokens.py)
    n_target_tokens_nothink   same with reasoning_content removed
    n_total_tokens            the whole thinking-on render
    tools                     JSON string: [{"type":"function","function":{"name","description","parameters":<JSON Schema>}}]
    messages                  the conversation below; `reasoning_content` and `tool_calls` are null where absent

messages:
    {"role":"user",      "content": <task summary>}
    {"role":"assistant", "content":"", "reasoning_content": <reason>,
                         "tool_calls":[{"type":"function","function":{"name":<Tool>,"arguments":<JSON string>}}]}
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


def features():
    """The `datasets.Features` of a row."""
    from datasets import Features, List as HFList, Value
    tool_call = {"type": Value("string"), "function": {"name": Value("string"), "arguments": Value("string")}}
    message = {"role": Value("string"), "content": Value("string"),
               "reasoning_content": Value("string"), "tool_calls": HFList(tool_call)}
    return Features({
        "id": Value("string"),
        "field": Value("string"),
        "gt_tool_calls": HFList(Value("string")),
        "n_target_tokens": Value("int32"),
        "n_target_tokens_nothink": Value("int32"),
        "n_total_tokens": Value("int32"),
        "tools": Value("string"),
        "messages": HFList(message),
    })


def dumps(obj: Any) -> str:
    """The one JSON serialisation of the dataset (what the Qwen3 template's `tojson` would render)."""
    return json.dumps(obj, ensure_ascii=False)


def user_message(summary: str) -> Dict[str, Any]:
    return {"role": "user", "content": summary}


def assistant_call_message(reason: str, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": reason,
        "tool_calls": [{"type": "function", "function": {"name": name, "arguments": dumps(arguments)}}],
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
        "gt_tool_calls": list(gt_tool_calls),
        "n_target_tokens": counts["n_target_tokens"],
        "n_target_tokens_nothink": counts["n_target_tokens_nothink"],
        "n_total_tokens": counts["n_total_tokens"],
        "tools": dumps(tools),
        "messages": messages,
    }
