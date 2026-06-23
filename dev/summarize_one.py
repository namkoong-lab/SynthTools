#!/usr/bin/env python3
"""Run the task summarizer on one task and inspect the 4-field output.

Loads a task from scratch, runs the real audit + TaskSummarizer pipeline,
prints the parsed JSON (user_supplied_values, tool_produced_values,
spillover, task_summarized), and flags any value in tool_produced_values
that also appears in a later user message: the over-hiding signature.

Usage:
  python dev/summarize_one.py --task-id academic_publishing_and_citations_spec_000_seq11
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from llm import LLM, MODEL_REGISTRY  # noqa: E402
from roles.task_summarizer import TaskSummarizer  # noqa: E402
from task_audit.checks import audit_task_pre_summarize, any_block  # noqa: E402
from task_audit.summarize import _triples_for_summarizer, _SPEC_ID_RE  # noqa: E402
from utils import extract_json_objects  # noqa: E402

TASKS_DIR = Path("/pscratch/sd/t/tcaste/tool_content/tasks")
ENV_SPECS_DIR = Path("/pscratch/sd/t/tcaste/tool_content/env_specs")


def _spec_for(task_id: str) -> Dict[str, Any]:
    m = _SPEC_ID_RE.match(task_id)
    if not m:
        sys.exit(f"cannot parse spec_id from task_id {task_id!r}")
    spec_id = m.group(0)
    p = ENV_SPECS_DIR / f"{spec_id}.json"
    if not p.exists():
        sys.exit(f"missing env_spec {p}")
    return json.loads(p.read_text())


def _user_texts_after(chat: List[Dict[str, Any]], k: int) -> List[str]:
    out = []
    for msg in chat[k:]:
        if isinstance(msg, dict) and msg.get("role") == "user":
            c = msg.get("content")
            if isinstance(c, str):
                out.append(c)
    return out


def overhide_candidates(parsed: Dict[str, Any], clean: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Values bucketed as tool_produced that also appear verbatim in any later
    user message anywhere in the cleaned chat. Signature of summarizer
    over-hiding when the value is actually a user-chosen criterion.
    """
    tool_vals = parsed.get("tool_produced_values") or []
    hits = []
    all_user_text_parts: List[str] = []
    for turn in clean:
        chat = turn.get("chat") or []
        for msg in chat:
            if isinstance(msg, dict) and msg.get("role") == "user":
                c = msg.get("content")
                if isinstance(c, str):
                    all_user_text_parts.append(c)
    joined = "\n".join(all_user_text_parts)
    for v in tool_vals:
        if not isinstance(v, str) or not v.strip():
            continue
        if v in joined:
            hits.append({"value": v, "mentioned_in_user_text": True})
    return hits


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", required=True)
    p.add_argument("--model", default="GPT-OSS-120B", choices=list(MODEL_REGISTRY))
    p.add_argument("--server-url", default="http://localhost:8765/v1")
    p.add_argument("--tasks-dir", default=str(TASKS_DIR))
    args = p.parse_args()

    raw = json.loads((Path(args.tasks_dir) / f"{args.task_id}.json").read_text())
    spec = _spec_for(args.task_id)

    clean, issues = audit_task_pre_summarize(raw, spec)
    if any_block(issues):
        sys.exit(f"pre-audit BLOCK: {[i.to_dict() for i in issues]}")
    if not clean:
        sys.exit("no clean successful turns in task")

    triples = _triples_for_summarizer(clean, spec=spec)

    llm = LLM(args.model, server_url=args.server_url)
    role = TaskSummarizer(llm)
    messages = role.build_messages(
        tasks=triples["tasks"], tool_calls=triples["tool_calls"],
        tool_responses=triples["tool_responses"], tools=triples["tools"],
    )
    response = llm(messages)
    objs = extract_json_objects(response)
    parsed = objs[0] if objs and isinstance(objs[0], dict) else None

    print(f"task   : {args.task_id}")
    print(f"server : {args.server_url}")
    print(f"model  : {args.model}")
    print(f"turns  : {len(clean)} clean")
    print("=" * 70)
    if not parsed:
        print("\nRAW RESPONSE (no JSON parsed):")
        print(response)
        return

    for k in ("user_supplied_values", "tool_produced_values", "spillover"):
        vals = parsed.get(k) or []
        print(f"\n{k} ({len(vals)}):")
        for v in vals:
            print(f"  - {v!r}")

    print("\ntask_summarized:")
    print(f"  {parsed.get('task_summarized', '')}")

    flagged = overhide_candidates(parsed, clean)
    print("\n" + "=" * 70)
    print(f"over-hide candidates (tool_produced AND appears in some user msg): {len(flagged)}")
    for h in flagged:
        print(f"  - {h['value']!r}")


if __name__ == "__main__":
    main()
