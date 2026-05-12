"""Build a parquet dataset from cleaned + summarised tasks.

One row per fresh task (legacy imports are dropped). Columns:
  id              - task_id (e.g. "aerospace_and_defense_spec_007_seq11")
  field           - human-readable field (e.g. "Aerospace and Defense")
  summary         - task_summarized from summary_clean.v1
  tools           - list of full tool schemas (JSON strings) for tools actually used
  gt_tool_calls   - per-turn successful tool calls the agent made
                    (the same calls fed to the summariser)
  initial_state   - env_state_before of turn 0 (JSON string, may be null)
  final_state     - env_state_after of last turn (JSON string, may be null)

Usage:
  python scripts/build_parquet.py \\
      --tasks-dir     /pscratch/.../tasks_clean \\
      --env-specs-dir /pscratch/.../env_specs \\
      --output        /pscratch/.../tasks.parquet \\
      [--limit N]    # for testing
      [--workers N]  # default 16
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


# Re-use the same call/response extractor as summary_clean.py for consistency.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from summary_clean import _successful_call_and_response  # noqa: E402


def _to_json_str(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    return json.dumps(obj, ensure_ascii=False, default=str)


def _is_legacy(traj: Dict[str, Any]) -> bool:
    return bool(traj.get("imported_from"))


def extract_row(args: Tuple[str, str]) -> Optional[Dict[str, Any]]:
    fp, env_specs_dir = args
    try:
        with open(fp) as f:
            traj = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    # Drop legacy imports
    if _is_legacy(traj):
        return None

    # Must have a fresh summary
    summary_block = traj.get("summary") or {}
    if not isinstance(summary_block, dict):
        return None
    src = summary_block.get("source") or ""
    if not src.startswith("summary_clean"):
        return None
    summary_text = (summary_block.get("parsed") or {}).get("task_summarized") or ""
    if not summary_text:
        return None

    task_id = traj.get("task_id") or Path(fp).stem
    spec_id_match = re.match(r"(.+_spec_\d+)_", task_id)
    if not spec_id_match:
        return None
    spec_id = spec_id_match.group(1)

    # Load the spec for the field name and full tool schemas
    spec_path = Path(env_specs_dir) / f"{spec_id}.json"
    try:
        with open(spec_path) as f:
            spec = json.load(f)
    except (OSError, json.JSONDecodeError):
        spec = {}
    field = spec.get("field") or ""

    # Tools the trajectory actually used (in order, dedup by tool_name)
    used_tool_names: List[str] = []
    seen = set()
    for t in traj.get("turns") or []:
        nm = (t.get("tool_id") or "").split(".")[-1]
        if nm and nm not in seen:
            seen.add(nm)
            used_tool_names.append(nm)
    spec_tools_by_name = {t.get("tool_name"): t for t in (spec.get("tools") or [])}
    tools_used_schemas: List[str] = []
    for nm in used_tool_names:
        s = spec_tools_by_name.get(nm)
        if s is not None:
            tools_used_schemas.append(json.dumps(s, ensure_ascii=False, default=str))

    # Per-turn agent successful tool calls (the same ones fed to the summariser)
    gt_tool_calls: List[str] = []
    for turn in traj.get("turns") or []:
        call, _resp = _successful_call_and_response(turn.get("chat") or [])
        gt_tool_calls.append(call or "")

    # Initial state = the TaskEvolver's `env_metadata` for turn 0 (fake-world
    # grounding the simulator); fall back to `edited_metadata` if present.
    # Final state = env_state_after of the last turn.
    turns = traj.get("turns") or []
    initial_state = None
    if turns:
        t0_task = turns[0].get("task") or {}
        initial_state = _to_json_str(
            t0_task.get("edited_metadata") or t0_task.get("env_metadata")
        )
    final_state = _to_json_str(turns[-1].get("env_state_after")) if turns else None

    return {
        "id": task_id,
        "field": field,
        "summary": summary_text,
        "tools": tools_used_schemas,
        "gt_tool_calls": gt_tool_calls,
        "initial_state": initial_state,
        "final_state": final_state,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks-dir", type=Path, required=True)
    p.add_argument("--env-specs-dir",    type=Path, required=True)
    p.add_argument("--output",           type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    files = sorted(p_ for p_ in args.tasks_dir.iterdir()
                   if p_.suffix == ".json" and not p_.name.endswith(".debug.json"))
    if args.limit:
        files = files[: args.limit]
    print(f"scanning {len(files):,} task files ...", flush=True)

    rows: List[Dict[str, Any]] = []
    n_legacy = n_skipped = 0
    work = [(str(p_), str(args.env_specs_dir)) for p_ in files]

    if args.workers <= 1:
        for w in work:
            r = extract_row(w)
            if r is None: n_skipped += 1
            else: rows.append(r)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(extract_row, w) for w in work]
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                if r is None: n_skipped += 1
                else: rows.append(r)
                if i % 5000 == 0:
                    print(f"  ... {i}/{len(work)}  (rows kept so far: {len(rows):,})", flush=True)

    print(f"\nrows extracted: {len(rows):,}  skipped (legacy / no fresh summary): {n_skipped:,}")

    # Build parquet table with explicit schema
    schema = pa.schema([
        ("id",            pa.string()),
        ("field",         pa.string()),
        ("summary",       pa.string()),
        ("tools",         pa.list_(pa.string())),
        ("gt_tool_calls", pa.list_(pa.string())),
        ("initial_state", pa.string()),  # JSON-encoded; nullable
        ("final_state",   pa.string()),
    ])
    cols = {k: [r[k] for r in rows] for k in schema.names}
    table = pa.table(cols, schema=schema)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.output, compression="snappy")
    size_mb = args.output.stat().st_size / 1e6
    print(f"\nwrote {args.output}  rows={len(rows):,}  size={size_mb:.1f} MB")


if __name__ == "__main__":
    main()
