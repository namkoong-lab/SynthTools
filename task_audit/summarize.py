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

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from env_audit.utils import model_config_for, now_iso
from roles.task_summarizer import TaskSummarizer
from utils import batch_call, extract_json_objects, get_logger, usage_to_dict

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
# Atomic file writers
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    tmp.replace(path)


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
    _atomic_write_json(debug_path, debug)
    return True


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def summarize_trajectories(
    tasks_dir: Path,
    llm,
    task_path: Optional[Path] = None,
    field: Optional[str] = None,
    env_specs_dir: Optional[Path] = None,
    write_debug: bool = True,
) -> List[Dict[str, Any]]:
    """Populate `summary` on task JSONs that don't already have one.

    Args:
        tasks_dir: Directory containing task JSONs. Used as the base
            for walking files and for resolving `.debug.json` siblings.
        llm: LLM instance.
        task_path: Optional path to a single task file. Mutually
            exclusive with `field`.
        field: Optional field name. Requires `env_specs_dir`. Walks env_specs to
            collect matching spec_ids, then picks every task whose filename
            starts with one of those spec_ids.
        env_specs_dir: Directory with env_spec JSONs — needed for `field` lookup.
        write_debug: If True (default), append a summarize event to each sibling
            `.debug.json` (when present). Set False to leave debug logs untouched.

    Returns:
        The list of tasks that were (or would be — for skipped ones) in
        scope, with their `summary` field populated where applicable.
    """
    if task_path is not None and field is not None:
        raise ValueError("task_path and field are mutually exclusive")

    tasks_dir = Path(tasks_dir)

    role = TaskSummarizer(llm)
    if hasattr(llm, "_ensure_engine"):
        llm._ensure_engine()

    if task_path is not None:
        target_paths = [Path(task_path)]
    else:
        target_paths = _iter_trajectory_paths(
            tasks_dir,
            field,
            Path(env_specs_dir) if env_specs_dir is not None else None,
        )
        scope_label = f"field '{field}'" if field else "all tasks"
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

        if task.get("summary"):
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
            _atomic_write_json(path, task)
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

    logger.info(
        f"summarize_trajectories: processed {len(updated)} task/ies "
        f"in {time.time() - start:.1f}s"
    )
    return updated
