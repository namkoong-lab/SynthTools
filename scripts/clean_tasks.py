"""Clean synthtools tasks: keep only judge-approved turns, rebuild solver_chat.

Per-turn filter (fresh runs only):
    judge.task_solved AND judge.tool_call_equality AND judge.arguments_grounded
Per tool_idx, the LAST kept turn wins (most recent successful attempt).

Legacy imports (no .debug.json sibling, all turns have judge=null) are passed
through unchanged — they were collapsed during import and already have a
populated `summary` field.

Outputs are written atomically (.tmp + replace) into a sibling directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def is_clean_turn(turn: Dict[str, Any]) -> bool:
    """Strict 3-judge filter for fresh runs."""
    j = turn.get("judge") or {}
    return (
        j.get("task_solved") is True
        and j.get("tool_call_equality") is True
        and j.get("arguments_grounded") is True
    )


def is_legacy_import(task: Dict[str, Any]) -> bool:
    """Legacy imports have no judge data on any turn."""
    if task.get("imported_from"):
        return True
    turns = task.get("turns") or []
    if not turns:
        return False
    return all(t.get("judge") is None for t in turns)


def clean_trajectory(task: Dict[str, Any]) -> Dict[str, Any]:
    """Filter `turns`, rebuild solver_chat, retighten tool_ids/tools.

    Returns a NEW dict; original is not mutated.
    """
    turns: List[Dict[str, Any]] = task.get("turns") or []

    # Per tool_idx, keep the LAST clean turn (highest attempt index).
    by_idx: Dict[int, Dict[str, Any]] = {}
    for t in turns:
        if not is_clean_turn(t):
            continue
        idx = t.get("tool_idx")
        if idx is None:
            continue
        prev = by_idx.get(idx)
        if prev is None or (t.get("attempt") or 0) >= (prev.get("attempt") or 0):
            by_idx[idx] = t

    kept = [by_idx[i] for i in sorted(by_idx)]

    # Rebuild solver_chat from the kept turns' chats. We keep each turn's full
    # in-attempt chat (so within-attempt 400 → fix → 200 recovery is preserved
    # as training signal).
    rebuilt_chat: List[Dict[str, Any]] = []
    for t in kept:
        for m in t.get("chat") or []:
            if m.get("role") in ("user", "assistant", "tool"):
                rebuilt_chat.append(m)

    # Map original tools by tool_id, then take only the kept ones in order.
    orig_tool_ids: List[str] = task.get("tool_ids") or []
    orig_tools: List[Any] = task.get("tools") or []
    tools_by_id: Dict[str, Any] = dict(zip(orig_tool_ids, orig_tools))
    kept_tool_ids = [t.get("tool_id") for t in kept if t.get("tool_id") is not None]
    kept_tools = [tools_by_id.get(tid) for tid in kept_tool_ids]

    cleaned = dict(task)  # shallow copy
    cleaned["tool_ids"] = kept_tool_ids
    cleaned["tools"] = kept_tools
    cleaned["turns"] = kept
    cleaned["solver_chat"] = rebuilt_chat
    cleaned["clean_meta"] = {
        "filter": "task_solved AND tool_call_equality AND arguments_grounded; last attempt per tool_idx",
        "original_turns": len(turns),
        "kept_turns": len(kept),
        "original_tools": len(orig_tool_ids),
        "kept_tools": len(kept_tool_ids),
        "dropped_legacy": False,
    }
    return cleaned


def passthrough_legacy(task: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy imports are already clean; just annotate provenance."""
    out = dict(task)
    out["clean_meta"] = {
        "filter": "passthrough_legacy_import",
        "original_turns": len(task.get("turns") or []),
        "kept_turns": len(task.get("turns") or []),
        "original_tools": len(task.get("tool_ids") or []),
        "kept_tools": len(task.get("tool_ids") or []),
        "dropped_legacy": False,
    }
    return out


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write `payload` to `path` atomically. Lustre-safe: unique tmp name (PID+
    randint) so multiple workers can never collide, fsync before rename, and one
    bounded retry on ENOENT (Lustre metadata-server occasionally lags).
    """
    import os, random
    tmp = path.with_name(f"{path.stem}.{os.getpid()}_{random.randint(0, 1<<30):x}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.replace(tmp, path)
        except FileNotFoundError:
            # Lustre metadata flake — fall back to plain open+write (no rename).
            with open(path, "w") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
                f.flush()
                os.fsync(f.fileno())
    except BaseException:
        try: os.remove(tmp)
        except FileNotFoundError: pass
        raise


def process_one(in_path: Path, out_dir: Path, dry_run: bool = False) -> Dict[str, Any]:
    """Read, clean, write. Returns a small summary dict."""
    with in_path.open() as f:
        task = json.load(f)

    if is_legacy_import(task):
        cleaned = passthrough_legacy(task)
        action = "passthrough_legacy"
    else:
        cleaned = clean_trajectory(task)
        action = "filtered_fresh"

    n_in = len(task.get("turns") or [])
    n_out = len(cleaned.get("turns") or [])
    has_summary = bool(task.get("summary"))
    drop = (action == "filtered_fresh" and n_out == 0)

    summary = {
        "task_id": task.get("task_id") or in_path.stem,
        "action": action if not drop else "dropped_no_clean_turns",
        "n_turns_in": n_in,
        "n_turns_out": n_out,
        "n_tools_in": len(task.get("tool_ids") or []),
        "n_tools_out": len(cleaned.get("tool_ids") or []),
        "has_summary": has_summary,
    }

    if dry_run:
        return summary

    if drop:
        return summary  # don't write empty tasks

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / in_path.name
    atomic_write_json(out_path, cleaned)
    summary["wrote"] = str(out_path)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir",  type=Path, required=True, help="tasks source dir")
    p.add_argument("--out-dir", type=Path, required=True, help="cleaned tasks dest dir")
    p.add_argument("--limit",   type=int, default=None, help="cap number of files (for testing)")
    p.add_argument("--ids",     nargs="*", help="explicit task_ids to process (without .json)")
    p.add_argument("--dry-run", action="store_true", help="don't write outputs")
    p.add_argument("--workers", type=int, default=1, help="parallel worker processes (default 1)")
    args = p.parse_args()

    in_dir: Path = args.in_dir
    out_dir: Path = args.out_dir

    if args.ids:
        paths = [in_dir / f"{tid}.json" for tid in args.ids]
        paths = [p for p in paths if p.exists()]
    else:
        paths = sorted(p for p in in_dir.iterdir()
                       if p.suffix == ".json" and not p.name.endswith(".debug.json"))
        if args.limit:
            paths = paths[: args.limit]

    summaries: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    if args.workers <= 1:
        for i, p in enumerate(paths, 1):
            try:
                summaries.append(process_one(p, out_dir, dry_run=args.dry_run))
            except Exception as e:
                errors.append({"path": str(p), "error": f"{type(e).__name__}: {e}"})
            if i % 1000 == 0:
                print(f"  ...processed {i}/{len(paths)}  errors={len(errors)}", flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from functools import partial
        worker = partial(process_one, out_dir=out_dir, dry_run=args.dry_run)
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(worker, p): p for p in paths}
            for i, fut in enumerate(as_completed(futs), 1):
                src = futs[fut]
                try:
                    summaries.append(fut.result())
                except Exception as e:
                    errors.append({"path": str(src), "error": f"{type(e).__name__}: {e}"})
                if i % 1000 == 0:
                    print(f"  ...processed {i}/{len(paths)}  errors={len(errors)}", flush=True)

    if errors:
        err_log = out_dir / "_clean_errors.jsonl"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(err_log, "a") as f:
            for e in errors:
                f.write(json.dumps(e) + "\n")
        print(f"[!] {len(errors)} task(s) failed — logged to {err_log}", flush=True)

    # Aggregate
    n = len(summaries)
    n_drop_no_clean = sum(1 for s in summaries if s["action"] == "dropped_no_clean_turns")
    n_legacy        = sum(1 for s in summaries if s["action"] == "passthrough_legacy")
    n_fresh         = sum(1 for s in summaries if s["action"] == "filtered_fresh")
    n_with_summary  = sum(1 for s in summaries if s["has_summary"])

    print()
    print(f"=== summary over {n} tasks ===")
    print(f"  legacy passthrough (kept as-is)  : {n_legacy}")
    print(f"  fresh, filtered                  : {n_fresh}")
    print(f"  fresh, dropped (0 clean turns)   : {n_drop_no_clean}")
    print(f"  inputs that had a `summary` field: {n_with_summary}")

    # Per-task printout
    print()
    print(f"{'task_id':<55}{'action':<22}{'turns':>10}{'tools':>10}{'summary':>9}")
    print("-" * 110)
    for s in summaries:
        print(f"{s['task_id'][:54]:<55}{s['action']:<22}"
              f"{s['n_turns_in']:>4}→{s['n_turns_out']:<5}"
              f"{s['n_tools_in']:>4}→{s['n_tools_out']:<5}"
              f"{'✓' if s['has_summary'] else '·':>9}")


if __name__ == "__main__":
    main()
