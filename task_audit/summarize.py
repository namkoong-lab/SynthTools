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

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env_audit.utils import model_config_for, now_iso
from llm import LLM, MODEL_REGISTRY
from roles.task_summarizer import TaskSummarizer
from utils import batch_call, extract_json_objects, get_logger, usage_to_dict

logger = get_logger("synthtools")


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _last_tool_response_from_chat(chat: List[Dict[str, Any]]) -> Any:
    """Return the last `role: tool` message's content, parsed as JSON.

    The legacy `scripts/task_summary.py` drops the `explanation` key before
    handing the tool response to the summarizer — keep that behaviour so the
    summarizer sees grounded tool outputs rather than the simulator's prose.
    """
    last = None
    for msg in chat or []:
        if msg.get("role") == "tool":
            last = msg.get("content")
    if last is None:
        return None
    try:
        parsed = json.loads(last)
    except (TypeError, json.JSONDecodeError):
        return last
    if isinstance(parsed, dict) and "explanation" in parsed:
        parsed = {k: v for k, v in parsed.items() if k != "explanation"}
    return parsed


def _last_tool_call_from_chat(chat: List[Dict[str, Any]]) -> Optional[str]:
    """Find the assistant message that produced a `tool_call` — fall back when
    `turn.task.expected_tool_call` is missing."""
    for msg in reversed(chat or []):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content") or ""
        objs = extract_json_objects(content)
        if objs and isinstance(objs[0], dict) and "tool_call" in objs[0]:
            call = objs[0]["tool_call"]
            return call if isinstance(call, str) else json.dumps(call, ensure_ascii=False)
    return None


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
    """Build the three parallel lists the TaskSummarizer template expects."""
    tasks: List[str] = []
    tool_calls: List[str] = []
    tool_responses: List[str] = []
    for turn in successful:
        task_block = turn.get("task") or {}
        tasks.append(task_block.get("task_description") or "")
        tc = task_block.get("expected_tool_call") or _last_tool_call_from_chat(turn.get("chat") or [])
        if tc is None:
            tc = ""
        if not isinstance(tc, str):
            tc = json.dumps(tc, ensure_ascii=False)
        tool_calls.append(tc)
        tr = _last_tool_response_from_chat(turn.get("chat") or [])
        if not isinstance(tr, str):
            tr = json.dumps(tr, ensure_ascii=False, default=str) if tr is not None else ""
        tool_responses.append(tr)
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Summarize tasks produced by task_generation.run."
    )
    parser.add_argument("--tasks-dir", type=Path, required=True,
                        help="Directory containing task JSONs")
    parser.add_argument("--model", default="GPT-OSS-120B", choices=list(MODEL_REGISTRY))

    target = parser.add_mutually_exclusive_group()
    target.add_argument("--task", type=Path,
                        help="Path to a single task JSON")
    target.add_argument("--field", type=str,
                        help="Field name; requires --env-specs-dir")

    parser.add_argument("--env-specs-dir", type=Path,
                        help="env_specs directory — required with --field")
    parser.add_argument("--no-debug", dest="write_debug", action="store_false",
                        help="Skip appending to sibling .debug.json files")
    parser.set_defaults(write_debug=True)

    args = parser.parse_args()

    if args.field and args.env_specs_dir is None:
        parser.error("--field requires --env-specs-dir")

    llm = LLM(args.model)
    summarize_trajectories(
        tasks_dir=args.tasks_dir,
        llm=llm,
        task_path=args.task,
        field=args.field,
        env_specs_dir=args.env_specs_dir,
        write_debug=args.write_debug,
    )


if __name__ == "__main__":
    main()
