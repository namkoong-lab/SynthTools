"""Build a multi-turn SFT parquet from SynthTools task JSONs.

One row per task. All env-advancing turns concatenated into a single
`messages` array, with the agent's failed retries kept (the point of the
multi-turn split is the model learning recovery).

Format is ACEBench-style: tools embedded in the system message, tool
calls emitted as `[ToolName(key='value', ...)]` Python-parseable
bracketed strings. No chat-template `tools=` parameter is used — the
system message carries everything.

Usage:
    python -m dataset_export.build_multi_turn \\
        --tasks-dir       /path/to/tasks \\
        --env-specs-dir   /path/to/env_specs \\
        --output          /path/to/multi_turn.parquet \\
        [--limit N] [--shard i --num-shards N] [--verbose]
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from task_audit.summarize import _extract_successful_turns

logger = logging.getLogger("synthtools.dataset_export")

SPEC_ID_RE = re.compile(r"(.+_spec_\d+)_")

SYSTEM_PROMPT_TEMPLATE = """You are an AI assistant with the role name "assistant." Based on the provided API specifications and conversation history, generate the API requests that the assistant should call. The API requests should be output in the format [ApiName(key1='value1', key2='value2', ...)], replacing ApiName with the actual API name. The output should start with a square bracket "[" and end with a square bracket "]".

If there are multiple API requests, separate them with commas, e.g.: [ApiName(...), ApiName(...)]. Do not include any other explanations, prompts, or API call results in the output.

Role Descriptions:
user: User
assistant: The AI assistant role that makes API requests
tool: Provides the results returned from tool calls

API Specifications:
{tools_json}"""


def _to_oai_tool(spec_tool: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a SynthTools tool schema → OpenAI function-call schema."""
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for name, p in (spec_tool.get("parameters") or {}).items():
        prop: Dict[str, Any] = {"type": p.get("type", "string")}
        if p.get("description"):
            prop["description"] = p["description"]
        if "default" in p:
            prop["default"] = p["default"]
        properties[name] = prop
        if p.get("required"):
            required.append(name)
    return {
        "type": "function",
        "function": {
            "name": spec_tool.get("tool_name", ""),
            "description": spec_tool.get("tool_description", ""),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _build_system_message(tools_oai: List[Dict[str, Any]]) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        tools_json=json.dumps(tools_oai, ensure_ascii=False, indent=2)
    )


def _strip_explanation(tool_content: str) -> str:
    """Drop the simulator's `explanation` key from a tool response if present."""
    try:
        parsed = json.loads(tool_content)
    except (TypeError, ValueError):
        return tool_content
    if isinstance(parsed, dict) and "explanation" in parsed:
        cleaned = {k: v for k, v in parsed.items() if k != "explanation"}
        return json.dumps(cleaned, ensure_ascii=False)
    return tool_content


def _parse_solver_assistant(content: str) -> Tuple[str, str]:
    """Solver assistant content is JSON-encoded `{reason, tool_call}`.
    Return `(reason, raw_call_str)`. Empty strings if unparseable.
    """
    try:
        obj = json.loads(content or "")
    except (TypeError, ValueError):
        return "", ""
    if not isinstance(obj, dict):
        return "", ""
    reason = obj.get("reason") or ""
    if not isinstance(reason, str):
        reason = str(reason)
    tc = obj.get("tool_call")
    if isinstance(tc, str):
        call = tc
    elif tc is None:
        call = ""
    else:
        call = json.dumps(tc, ensure_ascii=False)
    return reason, call


def _format_assistant_call(reason: str, raw_call_str: str) -> str:
    """Wrap `raw_call_str` into ACEBench bracket form, prefix reason if any.

    Verifies that the wrapped form parses as a Python expression. Raises
    `SyntaxError` (or `ValueError`) if not — callers treat that as a
    task-level skip so unparseable strings never reach training data.
    """
    wrapped = f"[{raw_call_str.strip()}]"
    ast.parse(wrapped, mode="eval")
    if reason.strip():
        return f"{reason.strip()}\n\n{wrapped}"
    return wrapped


def _build_turn_messages(turn: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    """Convert one task turn into a list of messages starting with `user`.

    Returns the list of messages on success, or None if the turn's chat
    contains a malformed solver message, an unparseable tool-call
    string, or an unexpected role. The caller treats None as a turn-level
    failure and stops accumulating further turns (truncation at turn
    boundary).
    """
    task_block = turn.get("task") or {}
    task_desc = task_block.get("task_description") or ""
    if not task_desc:
        return None
    msgs: List[Dict[str, str]] = [{"role": "user", "content": task_desc}]
    chat = turn.get("chat") or []
    # chat[0] is the solver's bundled user message (task + tool schemas).
    # We've already emitted the bare task_description above and the tool
    # schemas live in the system message, so skip it.
    for m in chat[1:]:
        role = m.get("role")
        content = m.get("content", "") or ""
        if role == "assistant":
            reason, raw_call = _parse_solver_assistant(content)
            if not raw_call:
                return None
            try:
                formatted = _format_assistant_call(reason, raw_call)
            except (SyntaxError, ValueError):
                return None
            msgs.append({"role": "assistant", "content": formatted})
        elif role == "tool":
            msgs.append({"role": "tool", "content": _strip_explanation(content)})
        else:
            return None
    return msgs


def _validate_turn_messages(
    turn_messages: List[Dict[str, str]], tool_names: Set[str]
) -> Optional[str]:
    """Verify one turn's messages (`user → (assistant, tool)+`). Returns None
    on success or a short reason string on failure.

    Checks:
      1. Strict role alternation: `user → assistant → tool → ...`,
         ending in tool. Length ≥ 3 (user + ≥1 pair). Pairs are
         (assistant, tool) only.
      2. Every assistant call is `[ToolName(...)]` and `ToolName` is in
         `tool_names`.
      3. Every tool response is JSON with an int `status_code`.
      4. Intermediate tool responses are NOT 2xx (they're failed
         retries within the same successful turn). The LAST tool
         response IS 2xx.
      5. No empty content anywhere.
    """
    if len(turn_messages) < 3:
        return "turn_too_short"
    if turn_messages[0]["role"] != "user":
        return "first_msg_not_user"
    if not turn_messages[0]["content"].strip():
        return "empty_user"

    rest = turn_messages[1:]
    if len(rest) % 2 != 0:
        return "unpaired_assistant_tool"

    for j in range(0, len(rest), 2):
        asst = rest[j]
        tool = rest[j + 1]
        if asst["role"] != "assistant":
            return f"expected_assistant_at_offset_{j+1}"
        if tool["role"] != "tool":
            return f"expected_tool_at_offset_{j+2}"
        if not asst["content"].strip():
            return "empty_assistant"
        if not tool["content"].strip():
            return "empty_tool"

        # Locate the trailing [Name(...)] block.
        sep = asst["content"].rfind("\n\n")
        tail = (asst["content"][sep + 2:] if sep >= 0 else asst["content"]).strip()
        try:
            tree = ast.parse(tail, mode="eval")
        except SyntaxError:
            return "assistant_call_not_parseable"
        if not (isinstance(tree.body, ast.List) and tree.body.elts):
            return "assistant_not_bracketed_list"
        call_node = tree.body.elts[0]
        if not (isinstance(call_node, ast.Call) and isinstance(call_node.func, ast.Name)):
            return "assistant_not_simple_call"
        if call_node.func.id not in tool_names:
            return f"unknown_tool_{call_node.func.id}"

        try:
            parsed = json.loads(tool["content"])
        except (TypeError, ValueError):
            return "tool_content_not_json"
        if not isinstance(parsed, dict):
            return "tool_content_not_dict"
        sc = parsed.get("status_code")
        if not isinstance(sc, int):
            return "tool_missing_int_status_code"

        is_last = (j + 2 == len(rest))
        if is_last:
            if not (200 <= sc < 300):
                return f"last_tool_status_code_{sc}_not_2xx"
        else:
            if 200 <= sc < 300:
                return f"intermediate_tool_status_code_{sc}_was_2xx"

    return None


def _task_to_row(
    task: Dict[str, Any], env_specs_dir: Path
) -> Optional[Dict[str, Any]]:
    """Build one row from a task JSON, or return None if no valid turns.

    Validation is per-turn. On the first turn that fails validation we
    STOP and keep only the prefix of valid turns (truncation at turn
    boundary). If zero turns survive → return None.
    """
    if task.get("imported_from"):
        return None
    task_id = task.get("task_id")
    if not task_id:
        return None
    m = SPEC_ID_RE.match(task_id)
    if not m:
        return None
    spec_id = m.group(1)
    try:
        spec = json.loads((env_specs_dir / f"{spec_id}.json").read_text())
    except (OSError, json.JSONDecodeError):
        logger.debug("%s: spec %s.json missing/unreadable", task_id, spec_id)
        return None

    field = spec.get("field") or ""
    successful = _extract_successful_turns(task)
    if not successful:
        return None

    # Universe of tool names = every tool that appears in ANY successful
    # turn. We validate against this superset so that legitimate calls
    # don't get flagged just because a later turn introduces a new tool.
    by_name = {t.get("tool_name"): t for t in (spec.get("tools") or [])}
    superset_tool_names: Set[str] = set()
    for t in successful:
        nm = (t.get("tool_id") or "").split(".")[-1]
        if nm in by_name:
            superset_tool_names.add(nm)
    if not superset_tool_names:
        return None

    kept_turn_msgs: List[List[Dict[str, str]]] = []
    kept_tool_names: List[str] = []
    seen_tools: Set[str] = set()
    n_total_turns = len(successful)
    for idx, turn in enumerate(successful):
        turn_msgs = _build_turn_messages(turn)
        if turn_msgs is None:
            logger.debug("%s: turn %d malformed during message build", task_id, idx)
            break
        fail_reason = _validate_turn_messages(turn_msgs, superset_tool_names)
        if fail_reason is not None:
            logger.debug(
                "%s: turn %d failed validation (%s); truncating after turn %d",
                task_id, idx, fail_reason, len(kept_turn_msgs),
            )
            break
        kept_turn_msgs.append(turn_msgs)
        nm = (turn.get("tool_id") or "").split(".")[-1]
        if nm and nm not in seen_tools:
            seen_tools.add(nm)
            kept_tool_names.append(nm)

    if not kept_turn_msgs:
        return None

    # Recompute tools_oai from kept turns only — never advertise a tool
    # the kept trajectory doesn't use.
    spec_tools = [by_name[n] for n in kept_tool_names if n in by_name]
    if not spec_tools:
        return None
    tools_oai = [_to_oai_tool(t) for t in spec_tools]

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _build_system_message(tools_oai)},
    ]
    for tm in kept_turn_msgs:
        messages.extend(tm)

    if len(kept_turn_msgs) < n_total_turns:
        logger.debug(
            "%s: truncated %d → %d turns",
            task_id, n_total_turns, len(kept_turn_msgs),
        )

    return {
        "id": task_id,
        "field": field,
        "messages": messages,
        "tools_oai": tools_oai,
        "n_turns": len(kept_turn_msgs),
        "n_messages": len(messages),
        "n_tools": len(tools_oai),
    }


def _iter_task_paths(tasks_dir: Path) -> List[Path]:
    return sorted(
        p for p in tasks_dir.glob("*.json")
        if not p.name.endswith(".debug.json") and not p.name.endswith(".tmp")
    )


def _build_rows(
    tasks_dir: Path,
    env_specs_dir: Path,
    limit: Optional[int],
    shard: int,
    num_shards: int,
) -> List[Dict[str, Any]]:
    paths = _iter_task_paths(tasks_dir)
    if num_shards > 1:
        paths = paths[shard::num_shards]
    if limit is not None:
        paths = paths[:limit]

    rows: List[Dict[str, Any]] = []
    skipped = 0
    for i, path in enumerate(paths):
        if i and i % 500 == 0:
            logger.info(
                "processed %d/%d (rows=%d skipped=%d)",
                i, len(paths), len(rows), skipped,
            )
        try:
            task = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("unreadable %s: %s", path.name, exc)
            skipped += 1
            continue
        row = _task_to_row(task, env_specs_dir)
        if row is None:
            skipped += 1
            continue
        rows.append(row)
    logger.info(
        "done: rows=%d skipped=%d total=%d",
        len(rows), skipped, len(paths),
    )
    return rows


_ROW_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("field", pa.string()),
    pa.field("messages", pa.list_(pa.struct([
        pa.field("role", pa.string()),
        pa.field("content", pa.string()),
    ]))),
    # OAI tool dicts are JSON-encoded per element — parameter properties
    # vary per tool, so a uniform pyarrow struct doesn't fit.
    pa.field("tools_oai", pa.list_(pa.string())),
    pa.field("n_turns", pa.int32()),
    pa.field("n_messages", pa.int32()),
    pa.field("n_tools", pa.int32()),
])


def _rows_to_arrow(rows: List[Dict[str, Any]]) -> pa.Table:
    serialized = [
        {
            "id": r["id"],
            "field": r["field"],
            "messages": r["messages"],
            "tools_oai": [json.dumps(t, ensure_ascii=False) for t in r["tools_oai"]],
            "n_turns": r["n_turns"],
            "n_messages": r["n_messages"],
            "n_tools": r["n_tools"],
        }
        for r in rows
    ]
    return pa.Table.from_pylist(serialized, schema=_ROW_SCHEMA)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build a multi-turn SFT parquet from SynthTools task JSONs."
    )
    p.add_argument("--tasks-dir", type=Path, required=True)
    p.add_argument("--env-specs-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    rows = _build_rows(
        tasks_dir=args.tasks_dir,
        env_specs_dir=args.env_specs_dir,
        limit=args.limit,
        shard=args.shard,
        num_shards=args.num_shards,
    )
    if not rows:
        logger.error("no rows produced — nothing to write")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table = _rows_to_arrow(rows)
    pq.write_table(table, args.output)
    logger.info("wrote %d rows to %s", len(rows), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
