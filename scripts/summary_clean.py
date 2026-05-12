"""Summarize *cleaned* tasks using the agent's actual successful calls.

Differences from `task_audit.summarize`:
  1. Tasks in the input folder are already filtered to judge-approved turns
     (by `scripts/clean_tasks.py`), so every turn is summarized — no per-turn
     filter here.
  2. `tool_call` strings come from the AGENT's successful assistant message in
     the kept turn's chat (the last 2xx exchange), NOT from
     `turn.task.expected_tool_call`.
  3. Tool responses are the last 2xx tool message of each kept turn, with the
     simulator's `explanation` key stripped.

Idempotent: tasks whose `summary` field is already truthy are skipped.

Run shape:
  - Sweep entire input folder, send to vLLM in chunks of `--batch-size`.
  - Each chunk = one `batch_call(llm, prompts)`. Per-chunk writeback so a
    crash never loses more than one batch.
  - With `--server-url`, no in-process vLLM load; otherwise loads in-process.

Usage:
  python scripts/summary_clean.py \\
      --tasks-dir /pscratch/.../tasks_clean \\
      --model GPT-OSS-120B \\
      --batch-size 64 \\
      [--limit 5]               # process at most N tasks
      [--ids id1 id2 ...]       # explicit task_ids
      [--dry-run-prompt]        # build prompts, print first one, no LLM call
      [--server-url URL]        # use vLLM HTTP server instead of in-process
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make `import roles.*` work the same way summarize.py does.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env_audit.utils import model_config_for, now_iso
from llm import LLM, MODEL_REGISTRY
from roles.task_summarizer import TaskSummarizer
from utils import batch_call, extract_json_objects, get_logger, usage_to_dict

logger = get_logger("synthtools")


# ---------------------------------------------------------------------------
# Per-turn extractors (the only thing that differs from task_audit.summarize)
# ---------------------------------------------------------------------------

def _parse_tool_msg(content: str):
    try:
        return json.loads(content or "")
    except (TypeError, json.JSONDecodeError):
        return content


def _agent_tool_call_str(asst_content: str) -> str:
    """Solver emits {"reason": "...", "tool_call": "ToolName(...)"} as JSON;
    return the tool_call string verbatim (or "" on parse failure)."""
    try:
        obj = json.loads(asst_content or "")
        tc = obj.get("tool_call") if isinstance(obj, dict) else None
        if isinstance(tc, str):
            return tc
        if tc is not None:
            return json.dumps(tc, ensure_ascii=False)
    except (TypeError, json.JSONDecodeError):
        pass
    return ""


def _successful_call_and_response(chat: List[Dict[str, Any]]):
    """Within a turn (which may have within-attempt 400 → fix → 200 recovery),
    return (agent_tool_call_str, last_tool_response_str) for the FINAL
    successful exchange. Strips the simulator's `explanation` from the tool
    response."""
    if not chat:
        return "", ""
    last_tool_i = None
    for i in range(len(chat) - 1, -1, -1):
        m = chat[i]
        if m.get("role") != "tool":
            continue
        parsed = _parse_tool_msg(m.get("content", ""))
        sc = parsed.get("status_code") if isinstance(parsed, dict) else None
        if isinstance(sc, int) and 200 <= sc < 300:
            last_tool_i = i
            break
    if last_tool_i is None:
        return "", ""
    asst_content = ""
    for j in range(last_tool_i - 1, -1, -1):
        if chat[j].get("role") == "assistant":
            asst_content = chat[j].get("content", "") or ""
            break
    tool_str = chat[last_tool_i].get("content", "") or ""
    parsed = _parse_tool_msg(tool_str)
    if isinstance(parsed, dict) and "explanation" in parsed:
        parsed = {k: v for k, v in parsed.items() if k != "explanation"}
        tool_str = json.dumps(parsed, ensure_ascii=False)
    return _agent_tool_call_str(asst_content), tool_str


def _triples_for_summarizer(turns: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    tasks: List[str] = []
    tool_calls: List[str] = []
    tool_responses: List[str] = []
    for turn in turns:
        task_block = turn.get("task") or {}
        tasks.append(task_block.get("task_description") or "")
        call, resp = _successful_call_and_response(turn.get("chat") or [])
        tool_calls.append(call)
        tool_responses.append(resp)
    return {"tasks": tasks, "tool_calls": tool_calls, "tool_responses": tool_responses}


def build_prompt(traj: Dict[str, Any], role: TaskSummarizer) -> Optional[str]:
    """Return the prompt the summarizer would receive, or None if there is
    nothing to summarize (e.g. zero turns)."""
    turns = traj.get("turns") or []
    if not turns:
        return None
    triples = _triples_for_summarizer(turns)
    return role.get_prompt(
        "task_summarizer",
        tasks=role._fmt(triples["tasks"]),
        tool_calls=role._fmt(triples["tool_calls"]),
        tool_responses=role._fmt(triples["tool_responses"]),
    )


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, payload: Any) -> None:
    """Lustre-safe atomic write."""
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
            with open(path, "w") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
                f.flush()
                os.fsync(f.fileno())
    except BaseException:
        try: os.remove(tmp)
        except FileNotFoundError: pass
        raise


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _build_pending(paths: List[Path], role: TaskSummarizer) -> List[Dict[str, Any]]:
    """Load each trajectory, skip already-summarized, build prompts."""
    pending: List[Dict[str, Any]] = []
    for path in paths:
        try:
            traj = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"  {path.name}: unreadable ({exc}) — skipping")
            continue
        task_id = traj.get("task_id") or path.stem
        if traj.get("summary"):
            continue  # idempotent skip — already summarized
        prompt = build_prompt(traj, role)
        if prompt is None:
            logger.warning(f"  {task_id}: 0 turns to summarize — skipping")
            continue
        pending.append({
            "path": path,
            "task_id": task_id,
            "traj": traj,
            "prompt": prompt,
            "n_subtasks": len(traj.get("turns") or []),
        })
    return pending


def _writeback_chunk(chunk: List[Dict[str, Any]], results: List[Dict[str, Any]],
                     llm, generated_at: str) -> None:
    for item, r in zip(chunk, results):
        traj = item["traj"]
        path: Path = item["path"]
        response = r["response"]
        usage = r["usage"]
        objs = extract_json_objects(response)
        parsed = objs[0] if objs and isinstance(objs[0], dict) else None
        usage_dict = usage_to_dict(usage)

        traj["summary"] = {
            "generated_at": generated_at,
            "model": getattr(llm, "model", None),
            "model_config": model_config_for(llm),
            "n_subtasks": item["n_subtasks"],
            "prompt": item["prompt"],
            "response": response,
            "parsed": parsed,
            "usage": usage_dict,
            "source": "summary_clean.v1 (agent's actual tool_call from chat)",
        }
        _atomic_write_json(path, traj)


def _parse_shard(shard: str) -> tuple[int, int]:
    try:
        k_str, n_str = shard.split("/")
        k, n = int(k_str), int(n_str)
    except (ValueError, AttributeError):
        raise SystemExit(f"--shard must be 'k/N' with integer k, N (got {shard!r})")
    if n < 1 or not (0 <= k < n):
        raise SystemExit(f"--shard must satisfy 0 <= k < N (got {k}/{n})")
    return k, n


def summarize_clean(
    tasks_dir: Path,
    llm,
    ids: Optional[List[str]] = None,
    limit: Optional[int] = None,
    batch_size: int = 64,
    dry_run_prompt: bool = False,
    shard: Optional[str] = None,
) -> None:
    role = TaskSummarizer(llm)
    if hasattr(llm, "_ensure_engine") and not dry_run_prompt:
        llm._ensure_engine()

    if ids:
        paths = [tasks_dir / f"{tid}.json" for tid in ids]
        paths = [p for p in paths if p.exists()]
    else:
        paths = sorted(p for p in tasks_dir.iterdir()
                       if p.suffix == ".json" and not p.name.endswith(".debug.json"))
        if shard:
            k, n = _parse_shard(shard)
            paths = [p for i, p in enumerate(paths) if i % n == k]
            logger.info(f"summary_clean: shard {k}/{n} → {len(paths)} files")
        if limit:
            paths = paths[:limit]
    logger.info(f"summary_clean: scanning {len(paths)} target file(s)")

    pending = _build_pending(paths, role)
    n_total = len(pending)
    logger.info(f"summary_clean: {n_total} prompt(s) to send "
                f"(others were already summarized)")

    if dry_run_prompt:
        if not pending:
            print("Nothing pending — every trajectory already summarized.")
            return
        item = pending[0]
        print()
        print("=" * 78)
        print(f"PROMPT PREVIEW for: {item['task_id']}  ({item['n_subtasks']} subtasks)")
        print("=" * 78)
        print(item["prompt"])
        print("=" * 78)
        print(f"(prompt length: {len(item['prompt']):,} chars)")
        print(f"(would send {n_total} prompts in batches of size requested)")
        return

    if not pending:
        return

    # Process in chunks
    n_done = 0
    n_chunks = (n_total + batch_size - 1) // batch_size
    t_start = time.time()
    for ci in range(n_chunks):
        chunk = pending[ci * batch_size : (ci + 1) * batch_size]
        prompts = [it["prompt"] for it in chunk]
        t0 = time.time()
        results = batch_call(llm, prompts)
        _writeback_chunk(chunk, results, llm, generated_at=now_iso())
        n_done += len(chunk)
        elapsed = time.time() - t0
        eta = (time.time() - t_start) / n_done * (n_total - n_done)
        logger.info(
            f"chunk {ci+1}/{n_chunks}  size={len(chunk)}  "
            f"{elapsed:.1f}s  done {n_done}/{n_total}  eta {eta/60:.1f}m"
        )

    logger.info(
        f"summary_clean: completed {n_done} trajectory/ies in "
        f"{time.time() - t_start:.1f}s"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks-dir", type=Path, required=True,
                   help="Directory of CLEANED tasks (output of clean_tasks.py)")
    p.add_argument("--model", default="GPT-OSS-120B", choices=list(MODEL_REGISTRY))
    p.add_argument("--batch-size", type=int, default=64,
                   help="Number of prompts per vLLM batch call (default 64)")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap number of trajectories (for testing)")
    p.add_argument("--ids", nargs="*", help="Explicit task_ids (without .json)")
    p.add_argument("--dry-run-prompt", action="store_true",
                   help="Build prompts; print the first one and exit (no LLM call)")
    p.add_argument("--server-url", default=None,
                   help="vLLM HTTP server URL; if set, no in-process model load")
    p.add_argument("--shard", default=None,
                   help="Process only a slice of files: 'k/N' (e.g. 0/4) — use to "
                        "split work across multiple parallel jobs")
    args = p.parse_args()

    if args.dry_run_prompt:
        llm = LLM(args.model)
    else:
        llm = LLM(args.model, server_url=args.server_url) if args.server_url else LLM(args.model)

    summarize_clean(
        tasks_dir=args.tasks_dir,
        llm=llm,
        ids=args.ids,
        limit=args.limit,
        batch_size=args.batch_size,
        dry_run_prompt=args.dry_run_prompt,
        shard=args.shard,
    )


if __name__ == "__main__":
    main()
