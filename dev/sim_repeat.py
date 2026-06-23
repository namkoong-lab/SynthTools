#!/usr/bin/env python3
"""Repeat one simulator call N times against a running vLLM server to
expose non-determinism.

Loads the recorded ground-truth (tool, call, env state) from a task JSON
on scratch, calls ToolSimulator.simulate(...) N times, prints each
response and a distinct-output count.

Usage:
  python dev/sim_repeat.py --task-id academic_publishing_and_citations_spec_000_seq1
  python dev/sim_repeat.py --task-id <id> --turn 2 --n 5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from llm import LLM, MODEL_REGISTRY  # noqa: E402
from roles.tool_simulator import ToolSimulator  # noqa: E402
from task_audit.summarize import _successful_call_and_response  # noqa: E402

TASKS_DIR = Path("/pscratch/sd/t/tcaste/tool_content/tasks")


def short(s: str, n: int = 600) -> str:
    return s if len(s) <= n else s[:n] + "..."


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", required=True)
    p.add_argument("--turn", type=int, default=0)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--model", default="GPT-OSS-120B", choices=list(MODEL_REGISTRY))
    p.add_argument("--server-url", default="http://localhost:8765/v1")
    p.add_argument("--tasks-dir", default=str(TASKS_DIR))
    args = p.parse_args()

    raw = json.loads((Path(args.tasks_dir) / f"{args.task_id}.json").read_text())
    turns = raw.get("turns") or []
    if not 0 <= args.turn < len(turns):
        sys.exit(f"turn {args.turn} out of range (task has {len(turns)} turns)")
    turn = turns[args.turn]

    tool_name = (turn.get("tool_id") or "").split(".")[-1]
    by_name = {t.get("tool_name"): t for t in (raw.get("tools") or [])}
    tool_data = by_name.get(tool_name)
    if not tool_data:
        sys.exit(f"tool {tool_name!r} not in task tools (have {list(by_name)})")

    call, recorded = _successful_call_and_response(turn.get("chat") or [])
    if not call:
        call = (turn.get("task") or {}).get("expected_tool_call") or ""
    if not call:
        sys.exit("no tool_call found for this turn")
    metadata = turn.get("env_state_before") or {}

    print(f"task    : {args.task_id}")
    print(f"turn    : {args.turn}")
    print(f"tool    : {tool_name}")
    print(f"call    : {call}")
    print(f"server  : {args.server_url}")
    print(f"model   : {args.model}")
    print("\nrecorded GT output:")
    print(short(recorded or "<none>", 800))
    print("=" * 70)

    llm = LLM(args.model, server_url=args.server_url)
    simulator = ToolSimulator(runner=llm)

    outputs = []
    for i in range(args.n):
        res = simulator.simulate(tool_data, call, metadata=metadata)
        sim = (res.get("parsed") or {}).get("simulation")
        text = json.dumps(sim, sort_keys=True, ensure_ascii=False) if sim is not None else "<no parse>"
        outputs.append(text)
        print(f"\n--- run {i+1} ---")
        print(short(text, 800))

    digests = [hashlib.sha1(o.encode()).hexdigest()[:8] for o in outputs]
    counts = Counter(digests)
    print("\n" + "=" * 70)
    print(f"distinct outputs : {len(counts)} / {args.n}")
    print(f"hash counts      : {dict(counts)}")


if __name__ == "__main__":
    main()
