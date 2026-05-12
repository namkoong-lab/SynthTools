"""For each task_id in an AST-quoting bucket, check whether a later turn
re-calls the same tool with valid AST and a 2xx status. Recovered ones encode
the failure→fix training pattern and should be KEPT. Uncorrected ones leave
the agent stuck and should be DROPPED.

Read-only — no trajectory files are modified.

Usage:
    python -m scripts.audit_recovery_check \\
        --task-ids-file /tmp/audit_buckets/ast_quoting.txt \\
        --trajectories-dir /pscratch/sd/t/tcaste/tool_content/trajectories_cot_regen \\
        --out-dir /tmp/audit_recovery/

Writes:
    <out_dir>/ast_quoting_recovered.txt    (one task_id per line)
    <out_dir>/ast_quoting_uncorrected.txt
    <out_dir>/ast_quoting_no_ast_issue.txt (judge mistake — for manual review)
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils import get_logger  # noqa: E402

logger = get_logger("synthtools")


def is_ast_parseable(tc: str) -> bool:
    if not isinstance(tc, str) or not tc.strip():
        return False
    try:
        ast.parse(tc, mode="eval")
        return True
    except SyntaxError:
        return False


def tool_name_from_call(tc: str) -> Optional[str]:
    if not isinstance(tc, str):
        return None
    m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", tc)
    return m.group(1) if m else None


def turn_status(turn: dict) -> Optional[int]:
    chat = turn.get("chat") or []
    if len(chat) < 3:
        return None
    try:
        return json.loads(chat[2].get("content") or "{}").get("status_code")
    except Exception:
        return None


def turn_tool_call(turn: dict) -> Optional[str]:
    """Agent's tool_call is in chat[1].content as JSON {"reason":..., "tool_call":...}."""
    chat = turn.get("chat") or []
    if len(chat) < 2:
        return None
    try:
        ap = json.loads(chat[1].get("content") or "{}")
        return ap.get("tool_call")
    except Exception:
        return None


def classify(traj: dict) -> str:
    """Find the first AST-failing tool_call. If a later turn calls the same
    tool with valid AST and 2xx status → recovered. Else → uncorrected."""
    turns = traj.get("turns") or []
    bad_idx = None
    bad_tool = None
    for i, t in enumerate(turns):
        tc = turn_tool_call(t) or t.get("task", {}).get("expected_tool_call")
        if tc and not is_ast_parseable(tc):
            bad_idx = i
            bad_tool = tool_name_from_call(tc)
            break
    if bad_idx is None or bad_tool is None:
        return "no_ast_issue"

    for j in range(bad_idx + 1, len(turns)):
        tj = turns[j]
        tcj = turn_tool_call(tj) or tj.get("task", {}).get("expected_tool_call")
        if not tcj:
            continue
        if tool_name_from_call(tcj) != bad_tool:
            continue
        if not is_ast_parseable(tcj):
            continue
        st = turn_status(tj)
        if isinstance(st, int) and 200 <= st < 300:
            return "recovered"
    return "uncorrected"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-ids-file", type=Path, required=True)
    ap.add_argument("--trajectories-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/audit_recovery"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    task_ids: list[str] = []
    with open(args.task_ids_file) as f:
        for line in f:
            tid = line.strip()
            if tid:
                task_ids.append(tid)
    logger.info(f"checking {len(task_ids):,} candidate trajectories")

    buckets = {"recovered": [], "uncorrected": [], "no_ast_issue": []}
    for tid in task_ids:
        path = args.trajectories_dir / f"{tid}.json"
        if not path.exists():
            logger.warning(f"missing: {tid}")
            continue
        try:
            traj = json.loads(path.read_text())
        except Exception as e:
            logger.warning(f"unreadable {tid}: {e}")
            continue
        verdict = classify(traj)
        buckets[verdict].append(tid)

    print()
    print("=== AST-quoting recovery check ===")
    print(f"  total checked:                {len(task_ids):,}")
    for k, v in buckets.items():
        pct = 100 * len(v) / max(len(task_ids), 1)
        print(f"  {k:<16}: {len(v):>4} ({pct:5.1f}%)")

    for k, v in buckets.items():
        out = args.out_dir / f"ast_quoting_{k}.txt"
        out.write_text("\n".join(sorted(v)) + ("\n" if v else ""))
        print(f"  wrote {len(v):>4} → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
