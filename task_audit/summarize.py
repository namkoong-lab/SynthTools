"""Task summarization — merges a task's successful subtasks into one
cohesive task description.

Why a separate stage:
    task_generation writes per-turn detail (evolver → solver → simulator → judge
    → env update). Downstream training needs ONE task description per task
    that is provably solvable from the merged text + tool outputs alone. The
    existing `TaskSummarizer` role already does this; we wrap it here with:
        - batched LLM calls across N tasks
        - skip-if-present idempotency
        - atomic writeback to both the main task JSON and its .debug.json

Pipeline position:
    env_generation → env_audit → build_sequences → task_generation.run → **this**

Targeting modes (mutually exclusive; if none, processes every *_spec_*_*.json
under `tasks_dir`):
    - task_path: one specific task file
    - field + env_specs_dir: walks env_specs, collects matching spec_ids, then
      picks every task whose filename starts with one of those spec_ids

Idempotency / resume: a task is skipped if `task["summary"]` is
truthy. Clear that key to force re-summarization.

Usage:
    from llm import LLM
    from task_audit.summarize import summarize_trajectories

    llm = LLM("GPT-OSS-120B")
    summarize_trajectories(
        tasks_dir=Path("tool_content/tasks"),
        llm=llm,
        field="Aerospace and Defense",
        env_specs_dir=Path("tool_content/env_specs"),
    )
"""

import fcntl
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from env_audit.utils import model_config_for, now_iso
from roles.task_summarizer import TaskSummarizer
from utils import batch_call, extract_json_objects, get_logger, usage_to_dict, write_json_atomic

logger = get_logger("synthtools")


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _parse_tool_msg(content: str):
    """Parse a `role: tool` message content as JSON; fall back to the raw string."""
    try:
        return json.loads(content or "")
    except (TypeError, json.JSONDecodeError):
        return content


def _agent_tool_call_str(asst_content: str) -> str:
    """Solver emits {"reason": ..., "tool_call": "ToolName(...)"} as JSON;
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
    response.

    Critical: the tool_call returned here is the AGENT'S actual successful
    call (from its assistant message), NOT `turn.task.expected_tool_call`
    (the evolver's proposed call). The agent's real call is what training
    should learn from."""
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


def _extract_successful_turns(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per tool_idx, keep the LAST turn whose `env_update is not None`.

    `env_update` is populated only on successful tool calls (by
    `task_generation.generate`), so this filter is correct for both
    verifiable=True and verifiable=False runs.
    """
    successful_by_idx: Dict[int, Dict[str, Any]] = {}
    for turn in task.get("turns") or []:
        if turn.get("env_update") is None:
            continue
        idx = turn.get("tool_idx")
        if idx is None:
            continue
        successful_by_idx[idx] = turn
    return [successful_by_idx[i] for i in sorted(successful_by_idx)]


def _triples_for_summarizer(successful: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Build the three parallel lists the TaskSummarizer template expects.

    tool_calls come from the agent's last successful assistant message in
    `turn.chat` (see `_successful_call_and_response`), NOT from the
    evolver's `turn.task.expected_tool_call`. tool_responses come from the
    matching last 2xx tool message, with `explanation` stripped.
    """
    tasks: List[str] = []
    tool_calls: List[str] = []
    tool_responses: List[str] = []
    for turn in successful:
        task_block = turn.get("task") or {}
        tasks.append(task_block.get("task_description") or "")
        call, resp = _successful_call_and_response(turn.get("chat") or [])
        tool_calls.append(call)
        tool_responses.append(resp)
    return {"tasks": tasks, "tool_calls": tool_calls, "tool_responses": tool_responses}


# ---------------------------------------------------------------------------
# Release-row extraction (the JSONL output written alongside summarisation)
# ---------------------------------------------------------------------------

_SPEC_ID_RE = re.compile(r"(.+_spec_\d+)_")


def _extract_release_row(
    task: Dict[str, Any], env_specs_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Build the per-task release row from an in-memory task dict.

    Values are parsed JSON structures (lists of dicts, dict-or-None) so
    consumers like `trajectory_generation/loader.py` can use them directly
    without re-parsing. Returns None for legacy imports, missing summaries,
    or tasks whose id does not match the *_spec_NNN_* convention.

    Columns:
      id              - task_id
      field           - spec.field
      summary         - summary.parsed.task_summarized
      tools           - list of tool schemas (dicts) actually used, in
                        invocation order, deduplicated by tool_name
      gt_tool_calls   - per-turn AGENT successful tool call (string)
      initial_state   - turn 0 edited_metadata or env_metadata (dict|None)
      final_state     - last turn env_state_after (dict|None)
    """
    if task.get("imported_from"):
        return None
    summary_block = task.get("summary") or {}
    if not isinstance(summary_block, dict):
        return None
    summary_text = (summary_block.get("parsed") or {}).get("task_summarized") or ""
    if not summary_text:
        return None
    task_id = task.get("task_id")
    if not task_id:
        return None
    m = _SPEC_ID_RE.match(task_id)
    if not m:
        return None
    spec_id = m.group(1)

    try:
        spec = json.loads((env_specs_dir / f"{spec_id}.json").read_text())
    except (OSError, json.JSONDecodeError):
        spec = {}
    field = spec.get("field") or ""

    used_names: List[str] = []
    seen: Set[str] = set()
    for t in task.get("turns") or []:
        nm = (t.get("tool_id") or "").split(".")[-1]
        if nm and nm not in seen:
            seen.add(nm)
            used_names.append(nm)
    by_name = {t.get("tool_name"): t for t in (spec.get("tools") or [])}
    tools = [by_name[n] for n in used_names if n in by_name]

    gt_tool_calls: List[str] = []
    for turn in task.get("turns") or []:
        call, _ = _successful_call_and_response(turn.get("chat") or [])
        gt_tool_calls.append(call or "")

    turns = task.get("turns") or []
    initial_state: Optional[Dict[str, Any]] = None
    if turns:
        t0 = turns[0].get("task") or {}
        initial_state = t0.get("edited_metadata") or t0.get("env_metadata")
    final_state: Optional[Dict[str, Any]] = (
        turns[-1].get("env_state_after") if turns else None
    )

    return {
        "id": task_id,
        "field": field,
        "summary": summary_text,
        "tools": tools,
        "gt_tool_calls": gt_tool_calls,
        "initial_state": initial_state,
        "final_state": final_state,
    }


def _existing_release_ids(jsonl_path: Path) -> Set[str]:
    """Return the set of task ids already present in the JSONL release file."""
    if not jsonl_path.exists():
        return set()
    ids: Set[str] = set()
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except (TypeError, ValueError):
                continue
            tid = obj.get("id")
            if tid:
                ids.add(tid)
    return ids


def _flush_release_rows(
    jsonl_path: Path,
    rows: List[Dict[str, Any]],
    resummarize: bool,
) -> Dict[str, int]:
    """Append release rows to `jsonl_path` under an exclusive file lock,
    so multiple processes writing to the same JSONL never interleave
    bytes mid-line.

    - resummarize=False (default): rows whose `id` is already present
      in the file are skipped (resume-safe, idempotent).
    - resummarize=True: rows are written regardless of whether their id
      is already present (allows fresh re-runs at the cost of duplicate
      lines for re-summarised tasks; consumers should dedup by id,
      last-write-wins).

    Returns {"appended": int, "skipped": int}.

    Lock semantics: fcntl.flock(LOCK_EX) on the JSONL file fd. Held for
    the duration of one batch of writes. The lock is released when the
    `with` block exits (or when the process dies, by the kernel).
    """
    counters = {"appended": 0, "skipped": 0}
    if not rows:
        return counters

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    seen_ids: Set[str] = set() if resummarize else _existing_release_ids(jsonl_path)

    to_write = [r for r in rows if resummarize or r["id"] not in seen_ids]
    counters["skipped"] = len(rows) - len(to_write)
    if not to_write:
        return counters

    payload = "".join(
        json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in to_write
    )
    with jsonl_path.open("a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(payload)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    counters["appended"] = len(to_write)
    return counters


# ---------------------------------------------------------------------------
# Targeting
# ---------------------------------------------------------------------------

def _iter_trajectory_paths(
    tasks_dir: Path,
    field: Optional[str],
    env_specs_dir: Optional[Path],
) -> List[Path]:
    """Return every `{spec_id}_*.json` task whose spec_id matches the field."""
    all_paths = sorted(
        p for p in tasks_dir.glob("*.json")
        if not p.name.endswith(".debug.json") and not p.name.endswith(".tmp")
    )
    if field is None:
        return all_paths
    if env_specs_dir is None:
        raise ValueError("field filter requires env_specs_dir")

    matching_spec_ids: set = set()
    for spec_path in sorted(env_specs_dir.glob("*_spec_*.json")):
        if spec_path.name.endswith(".tmp"):
            continue
        try:
            spec = json.loads(spec_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if spec.get("field") == field:
            sid = spec.get("spec_id")
            if sid:
                matching_spec_ids.add(sid)

    return [p for p in all_paths if any(p.stem.startswith(f"{sid}_") for sid in matching_spec_ids)]


# ---------------------------------------------------------------------------
# Debug-log append helper (atomic write via utils.write_json_atomic)
# ---------------------------------------------------------------------------

def _append_debug_event(debug_path: Path, event: Dict[str, Any]) -> bool:
    """Append to `events[]` in a debug log if it exists. Returns True on append."""
    if not debug_path.exists():
        return False
    try:
        debug = json.loads(debug_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"  {debug_path.name}: unreadable debug log ({exc}) — skipping append")
        return False
    events = debug.setdefault("events", [])
    events.append(event)
    write_json_atomic(debug, debug_path)
    return True


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def summarize_trajectories(
    tasks_dir: Path,
    llm,
    env_specs_dir: Path,
    task_content_path: Path,
    task_path: Optional[Path] = None,
    field: Optional[str] = None,
    write_debug: bool = True,
    resummarize: bool = False,
    shard: int = 0,
    num_shards: int = 1,
) -> List[Dict[str, Any]]:
    """Populate `summary` on task JSONs and append release rows to
    `task_content_path`.

    Args:
        tasks_dir: Directory containing task JSONs. Used as the base
            for walking files and for resolving `.debug.json` siblings.
        llm: LLM instance.
        env_specs_dir: Directory with env_spec JSONs. Required: it
            supplies the `field` and `tools` columns of each release row,
            and (when `field` is set) the spec_id filter.
        task_content_path: Path to the shared JSONL release file. All
            concurrent jobs append to the SAME path under an exclusive
            file lock (no shard suffix). Resume-safe.
        task_path: Optional path to a single task file. Mutually
            exclusive with `field`.
        field: Optional field name. Walks env_specs to collect matching
            spec_ids, then picks every task whose filename starts with
            one of those spec_ids.
        write_debug: If True (default), append a summarize event to each
            sibling `.debug.json` (when present). Set False to leave
            debug logs untouched.
        resummarize: If True, ignore existing `summary` blocks and
            re-run the LLM for every in-scope task. Release rows are
            appended regardless of whether the id already exists in
            the JSONL (consumers dedup by id, last-write-wins). Default
            False: tasks with a summary are skipped and existing JSONL
            ids are not re-written.
        shard: Shard index for this process (0-based). Must satisfy
            0 <= shard < num_shards.
        num_shards: Total number of concurrent shards. When > 1, this
            process processes only target_paths[shard::num_shards] of
            the deterministically-sorted in-scope file list. Combined
            with file locking on the shared JSONL, this lets multiple
            jobs run concurrently against the same output path safely.

    Returns:
        The list of tasks that were (or would be — for skipped ones) in
        scope FOR THIS SHARD, with their `summary` field populated
        where applicable.
    """
    if task_path is not None and field is not None:
        raise ValueError("task_path and field are mutually exclusive")
    if not (0 <= shard < num_shards):
        raise ValueError(f"shard must be in [0, num_shards) — got shard={shard}, num_shards={num_shards}")

    tasks_dir = Path(tasks_dir)
    env_specs_dir = Path(env_specs_dir)
    task_content_path = Path(task_content_path)

    role = TaskSummarizer(llm)
    if hasattr(llm, "_ensure_engine"):
        llm._ensure_engine()

    if task_path is not None:
        target_paths = [Path(task_path)]
    else:
        target_paths = _iter_trajectory_paths(tasks_dir, field, env_specs_dir)
        scope_label = f"field '{field}'" if field else "all tasks"
        if num_shards > 1:
            target_paths = target_paths[shard::num_shards]
            scope_label += f" [shard {shard}/{num_shards}]"
        logger.info(f"{scope_label}: found {len(target_paths)} task file(s)")

    start = time.time()
    updated: List[Dict[str, Any]] = []
    to_process: List[Dict[str, Any]] = []

    # Phase 1: load each task, filter, build prompt (no LLM).
    for path in target_paths:
        try:
            task = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"  {path.name}: unreadable ({exc}) — skipping")
            continue
        task_id = task.get("task_id") or path.stem

        if task.get("summary") and not resummarize:
            logger.info(f"  {task_id}: summary already present — skipping")
            updated.append(task)
            continue

        successful = _extract_successful_turns(task)
        if not successful:
            logger.warning(f"  {task_id}: 0 successful turns — skipping")
            updated.append(task)
            continue

        triples = _triples_for_summarizer(successful)
        prompt = role.build_messages(
            tasks=triples["tasks"],
            tool_calls=triples["tool_calls"],
            tool_responses=triples["tool_responses"],
        )
        to_process.append({
            "path": path,
            "task_id": task_id,
            "task": task,
            "prompt": prompt,
            "n_subtasks": len(successful),
        })

    # Phase 2: one batched LLM call across every task that needs it.
    if to_process:
        prompts = [item["prompt"] for item in to_process]
        logger.info(f"summarize: batched LLM call for {len(prompts)} task/ies")
        results = batch_call(llm, prompts)

        for item, r in zip(to_process, results):
            task = item["task"]
            path: Path = item["path"]
            task_id = item["task_id"]
            response = r["response"]
            usage = r["usage"]

            objs = extract_json_objects(response)
            parsed = objs[0] if objs and isinstance(objs[0], dict) else None

            generated_at = now_iso()
            usage_dict = usage_to_dict(usage)
            summary_block = {
                "generated_at": generated_at,
                "model": getattr(llm, "model", None),
                "model_config": model_config_for(llm),
                "n_subtasks": item["n_subtasks"],
                "prompt": item["prompt"],
                "response": response,
                "parsed": parsed,
                "usage": usage_dict,
            }
            task["summary"] = summary_block
            write_json_atomic(task, path)
            logger.info(
                f"  {task_id}: summarized {item['n_subtasks']} subtasks "
                f"(usage: {usage_dict})"
            )

            if write_debug:
                debug_path = path.with_name(path.stem + ".debug.json")
                event = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "agent": "TaskSummarizer",
                    "action": "summarize",
                    "turn_ref": None,
                    "prompt": item["prompt"],
                    "response": response,
                    "parsed": parsed,
                    "usage": usage_dict,
                }
                appended = _append_debug_event(debug_path, event)
                if not appended:
                    logger.debug(f"  {task_id}: no debug log at {debug_path.name}")

            updated.append(task)

    release_rows: List[Dict[str, Any]] = []
    for task in updated:
        row = _extract_release_row(task, env_specs_dir)
        if row is not None:
            release_rows.append(row)
    if release_rows:
        counters = _flush_release_rows(task_content_path, release_rows, resummarize=resummarize)
        logger.info(
            f"task_content: appended={counters['appended']} "
            f"skipped={counters['skipped']} -> {task_content_path}"
        )
    else:
        logger.info(f"task_content: no eligible rows in scope -> {task_content_path} unchanged")

    logger.info(
        f"summarize_trajectories: processed {len(updated)} task/ies "
        f"in {time.time() - start:.1f}s"
    )
    return updated
