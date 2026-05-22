"""Audit checks for task_audit/.

Three audit boundaries, run as gates around the summariser:

  A · pre-summarize    — `audit_task_pre_summarize(task, spec)` →
                          (clean_prefix_of_successful_turns, issues)
  B · post-summarize   — `audit_summary_output(parsed)` → issues
  C · release-row      — `audit_release_row(row)` → issues

Issue levels:
  BLOCK  — caller MUST drop / not write
  WARN   — caller proceeds; issue is observable for telemetry

The pre-summarize audit also performs prefix-truncation: it walks the
successful turns in tool_idx order and stops at the first turn that
fails a per-turn invariant, returning the clean prefix as well as a
WARN issue describing the cut.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from task_audit.summarize import _extract_successful_turns, _successful_call_and_response

_SPEC_ID_RE = re.compile(r"(.+_spec_\d+)_")

BLOCK = "BLOCK"
WARN = "WARN"


@dataclass
class Issue:
    level: str
    code: str
    where: str
    message: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def _is_2xx(sc: Any) -> bool:
    return isinstance(sc, int) and 200 <= sc < 300


# ---------------------------------------------------------------------------
# A · pre-summarize
# ---------------------------------------------------------------------------

def audit_task_pre_summarize(
    task: Dict[str, Any], spec: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Issue]]:
    """Validate a task's inputs to the summariser.

    Returns (clean_prefix, issues). clean_prefix is the contiguous prefix
    of successful turns that pass every per-turn check. Issues describe
    why we truncated or why the whole task is unusable.

    Caller policy:
      - any BLOCK issue       → drop the task (do not call LLM, do not write release row)
      - empty clean_prefix    → skip task (nothing to summarise)
      - otherwise             → use clean_prefix
    """
    issues: List[Issue] = []

    # A1 — structural
    task_id = task.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        issues.append(Issue(BLOCK, "A1_missing_task_id", "task", "task has no task_id"))
        return [], issues
    if not _SPEC_ID_RE.match(task_id):
        issues.append(Issue(BLOCK, "A1_bad_task_id_pattern", "task",
                            f"task_id {task_id!r} does not match <spec_id>_seq<N> pattern"))
        return [], issues
    if not isinstance(task.get("turns"), list) or not task["turns"]:
        issues.append(Issue(BLOCK, "A1_empty_turns", "task", "task.turns is missing or empty"))
        return [], issues
    if not isinstance(spec, dict) or not spec:
        issues.append(Issue(BLOCK, "A1_no_spec", "task",
                            "env_spec for this task is missing or unreadable"))
        return [], issues

    # All env_update-set turns, deduped by tool_idx.
    successful = _extract_successful_turns(task)
    if not successful:
        return [], issues  # no LLM call needed; not an error per se

    spec_tools_by_name = {
        t.get("tool_name"): t for t in (spec.get("tools") or [])
    }

    # A2 — per-turn invariants; stop at first failure (truncate the prefix).
    clean: List[Dict[str, Any]] = []
    prev_env_after: Any = None
    for i, turn in enumerate(successful):
        where = f"turn[tool_idx={turn.get('tool_idx')}]"
        td = (turn.get("task") or {}).get("task_description") or ""
        if not td.strip():
            issues.append(Issue(WARN, "A2_empty_task_description", where,
                                "task_description is empty/null on this turn"))
            break

        chat = turn.get("chat") or []
        if not chat or chat[0].get("role") != "user":
            issues.append(Issue(WARN, "A2_chat_no_leading_user", where,
                                "chat[0] is missing or not a user message"))
            break

        last_tool_sc: Any = None
        for m in reversed(chat):
            if m.get("role") == "tool":
                try:
                    last_tool_sc = json.loads(m.get("content", "") or "").get("status_code")
                except (TypeError, ValueError):
                    last_tool_sc = None
                break
        if not _is_2xx(last_tool_sc):
            issues.append(Issue(WARN, "A2_final_tool_not_2xx", where,
                                f"last chat tool status_code={last_tool_sc!r}, not 2xx"))
            break

        call_str, _ = _successful_call_and_response(chat)
        if not call_str:
            issues.append(Issue(WARN, "A2_no_extractable_call", where,
                                "could not extract agent's tool_call from chat"))
            break
        try:
            ast.parse(call_str, mode="eval")
        except SyntaxError:
            issues.append(Issue(WARN, "A2_call_not_parseable", where,
                                f"agent's tool_call is not a valid Python expression: {call_str[:120]!r}"))
            break

        tool_name = (turn.get("tool_id") or "").split(".")[-1]
        if tool_name and tool_name not in spec_tools_by_name:
            issues.append(Issue(WARN, "A2_tool_not_in_spec", where,
                                f"turn.tool_id={turn.get('tool_id')!r} not in env_spec tools"))
            break

        # A3 — env_state chain continuity (skip on first turn).
        if i > 0 and prev_env_after is not None and turn.get("env_state_before") is not None:
            if prev_env_after != turn.get("env_state_before"):
                issues.append(Issue(WARN, "A3_env_state_chain_break", where,
                                    "env_state_before != prev turn's env_state_after"))
                break

        clean.append(turn)
        prev_env_after = turn.get("env_state_after")

    if not clean:
        return [], issues

    # A4 — kept-prefix tool catalogue is resolvable.
    used = []
    seen = set()
    for t in clean:
        nm = (t.get("tool_id") or "").split(".")[-1]
        if nm and nm not in seen:
            seen.add(nm)
            used.append(nm)
    resolved = [n for n in used if n in spec_tools_by_name]
    if not resolved:
        issues.append(Issue(BLOCK, "A4_no_resolvable_tools", "task",
                            "none of the tools used in successful turns resolve to a schema in the env_spec"))
        return [], issues

    return clean, issues


# ---------------------------------------------------------------------------
# B · post-summarize
# ---------------------------------------------------------------------------

REQUIRED_SUMMARY_FIELDS = (
    "user_supplied_values", "tool_produced_values", "spillover", "task_summarized",
)

MIN_SUMMARY_CHARS = 30


def audit_summary_output(parsed: Any) -> List[Issue]:
    """Validate the LLM's JSON response. Caller drops the summary if any BLOCK."""
    issues: List[Issue] = []
    if not isinstance(parsed, dict):
        issues.append(Issue(BLOCK, "B1_not_a_dict", "summary",
                            f"parsed summary is not a dict (got {type(parsed).__name__})"))
        return issues

    for f in REQUIRED_SUMMARY_FIELDS:
        if f not in parsed:
            issues.append(Issue(BLOCK, "B1_missing_field", "summary",
                                f"required field {f!r} missing from parsed summary"))

    ts = parsed.get("task_summarized")
    if not isinstance(ts, str) or not ts.strip():
        issues.append(Issue(BLOCK, "B2_empty_task_summarized", "summary",
                            "task_summarized is missing, empty, or not a string"))
    elif len(ts.strip()) < MIN_SUMMARY_CHARS:
        issues.append(Issue(BLOCK, "B2_task_summarized_too_short", "summary",
                            f"task_summarized is {len(ts.strip())} chars, < {MIN_SUMMARY_CHARS}"))

    sp = parsed.get("spillover")
    if isinstance(sp, list) and sp:
        issues.append(Issue(WARN, "B3_nonempty_spillover", "summary",
                            f"spillover has {len(sp)} entries (model failed to fully hide tool-produced values)"))

    return issues


# ---------------------------------------------------------------------------
# C · release-row
# ---------------------------------------------------------------------------

_REQUIRED_ROW_COLS = ("id", "field", "summary", "tools",
                      "gt_tool_calls", "initial_state", "final_state")


def audit_release_row(row: Dict[str, Any]) -> List[Issue]:
    """Validate a row before it goes into task_content.jsonl."""
    issues: List[Issue] = []
    for c in _REQUIRED_ROW_COLS:
        if c not in row:
            issues.append(Issue(BLOCK, "C1_missing_column", "row",
                                f"required column {c!r} missing"))

    if not row.get("id"):
        issues.append(Issue(BLOCK, "C1_empty_id", "row", "id is empty"))
    if not row.get("summary"):
        issues.append(Issue(BLOCK, "C1_empty_summary", "row", "summary is empty"))
    tools = row.get("tools") or []
    if not tools:
        issues.append(Issue(BLOCK, "C1_empty_tools", "row", "tools list is empty"))
    gt = row.get("gt_tool_calls") or []
    if not gt:
        issues.append(Issue(BLOCK, "C1_empty_gt_tool_calls", "row", "gt_tool_calls is empty"))

    if isinstance(tools, list) and isinstance(gt, list) and tools and gt:
        tool_names = {t.get("tool_name") for t in tools if isinstance(t, dict)}
        for i, call in enumerate(gt):
            m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", call or "")
            if not m:
                issues.append(Issue(WARN, "C3_gt_call_no_name", f"gt_tool_calls[{i}]",
                                    f"could not extract tool name from call: {(call or '')[:120]!r}"))
                continue
            name = m.group(1)
            if name not in tool_names:
                issues.append(Issue(BLOCK, "C3_unknown_tool_in_gt", f"gt_tool_calls[{i}]",
                                    f"call uses tool {name!r}, not in tools column"))

    return issues


# ---------------------------------------------------------------------------
# Helpers for callers
# ---------------------------------------------------------------------------

def any_block(issues: List[Issue]) -> bool:
    return any(i.level == BLOCK for i in issues)
