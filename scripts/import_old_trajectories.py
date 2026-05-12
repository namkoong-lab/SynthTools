"""
One-off migration: import old-pipeline dataset.jsonl rows (v0) into the new
trajectory.json format.

Consumes both tc_2 and tc_3 source datasets; routes tc_2 rows to
`tc2_{seq_key}` output keys so they don't collide with tc_3's `seq{N}` keys
(see Part A of the noble-discovering-flamingo plan).

Prerequisite: env_specs must already be imported via `scripts/import_old_envs.py`.
The importer builds a reverse lookup `old_filename -> new_spec_id` from the
`imported_from.source` block on each env_spec and lifts the simulator-format
`tools` list directly from the env_spec (dataset.jsonl's `tools` column is
OpenAI-function-format and lacks simulator extras like `error_messages`).

Run:
  python -m scripts.import_old_trajectories \
      --sources /pscratch/sd/t/tcaste/synthtools_dataset/tool_content_2/dataset.jsonl \
                /pscratch/sd/t/tcaste/synthtools_dataset/tool_content_3/dataset.jsonl \
      --env-specs-dir /pscratch/sd/t/tcaste/tool_content/env_specs \
      --output-dir /pscratch/sd/t/tcaste/tool_content/trajectories
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PASCAL_RE = re.compile(r"[A-Za-z0-9]+")


def _normalize_name(raw: str) -> str:
    """Match scripts/import_old_envs.normalize_tool_name — strip non-alnum, PascalCase."""
    s = unicodedata.normalize("NFKC", raw or "")
    parts = _PASCAL_RE.findall(s)
    if not parts:
        return ""
    out = "".join(w[0].upper() + w[1:] for w in parts)
    if out and out[0].isdigit():
        out = "T" + out
    return out

IMPORT_MODEL = "GPT-OSS-120B"
IMPORT_MODEL_CONFIG = {"temperature": 0.2, "top_p": 0.95, "max_tokens": 16384}
IMPORT_CONFIG = {"max_solver_turns": 10, "max_retries": 10, "verifiable": True}
SOURCE_SCHEMA = "v0"

# Fields the solver sees — must match roles/task_solver.py:_OPENAI_TOOL_FIELDS.
_OPENAI_TOOL_FIELDS = ("tool_name", "tool_description", "parameters")


def build_solver_user_message(task_description: str, tool_schema: Dict[str, Any]) -> str:
    """Mirror roles/task_solver.py::TaskSolver.build_user_message exactly."""
    filtered = {k: v for k, v in tool_schema.items() if k in _OPENAI_TOOL_FIELDS}
    schema_json = json.dumps(filtered, ensure_ascii=False)
    return f"Task: {task_description}\n\nTool to use:\n{schema_json}"

ID_RE = re.compile(r"^tool_content_(\d)_(.+)\.(seq\d+)$")
TOOL_NAME_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")
ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

# Matches dataset/utils.py:_STOP_TOKENS_RE — strip trailing STOP/END markers.
_STOP_TOKENS_RE = re.compile(r"\s*</?(?:STOP|stop|END|end)>\s*$|<\|(?:endoftext|im_end|eot_id)\|>\s*$")


def _clean_text(s: str) -> str:
    """Strip trailing STOP-style tokens and whitespace (mirror canonical dataset util)."""
    if not isinstance(s, str):
        return ""
    return _STOP_TOKENS_RE.sub("", s).strip()


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_spec_index(env_specs_dir: Path) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    """Walk imported env_specs to produce:
      filename_to_spec_id: old_yaml_filename -> new_spec_id
      spec_index:          new_spec_id      -> {tools, tool_name_set, field, subfield, task}
    """
    filename_to_spec_id: Dict[str, str] = {}
    spec_index: Dict[str, Dict[str, Any]] = {}
    for p in sorted(env_specs_dir.glob("*_spec_*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        imported = d.get("imported_from") or {}
        src = imported.get("source") or ""
        stem = Path(src).name  # e.g. "aerospace_and_defense_tool_spec_10.yml"
        if stem:
            filename_to_spec_id[stem] = d["spec_id"]
        tools_list = d.get("tools") or []
        by_name = {t.get("tool_name"): t for t in tools_list if t.get("tool_name")}
        spec_index[d["spec_id"]] = {
            "tools": tools_list,
            "tool_name_set": set(by_name.keys()),
            "tool_by_name": by_name,
            "field": d.get("field", ""),
            "subfield": d.get("subfield", ""),
            "task": d.get("task", ""),
        }
    return filename_to_spec_id, spec_index


def build_turn(
    user_msg: Dict[str, Any],
    asst_msg: Dict[str, Any],
    tool_msg: Dict[str, Any],
    *,
    tool_idx: int,
    spec_id: str,
    tool_by_name: Dict[str, Dict[str, Any]],
    warn_ctx: str,
    warnings: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    tool_call_str = _clean_text(asst_msg.get("tool_call") or "")
    m = TOOL_NAME_RE.match(tool_call_str)
    if not m:
        warnings.append({"ctx": warn_ctx, "turn": tool_idx, "issue": "malformed_tool_call", "value": tool_call_str[:80]})
        return None
    raw_name = m.group(1)
    if raw_name in tool_by_name:
        tool_name = raw_name
    else:
        normalized = _normalize_name(raw_name)
        if normalized in tool_by_name:
            tool_name = normalized
            tool_call_str = tool_call_str.replace(raw_name, normalized, 1)
        else:
            warnings.append({"ctx": warn_ctx, "turn": tool_idx, "issue": "unknown_tool", "value": raw_name})
            return None

    tool_content = tool_msg.get("content") or "{}"
    try:
        parsed_tool = json.loads(tool_content)
        status = parsed_tool.get("status_code")
    except Exception:
        # Malformed tool JSON = structural error → drop.
        warnings.append({"ctx": warn_ctx, "turn": tool_idx, "issue": "malformed_tool_response"})
        return None

    # Keep failed attempts (4xx/5xx are semantic mistakes, valuable training signal).
    # Only drop if status_code is entirely missing (structural, not a real response).
    if not isinstance(status, int):
        warnings.append({"ctx": warn_ctx, "turn": tool_idx, "issue": "no_status_code", "value": status})
        return None
    is_success = 200 <= status < 300

    task_description = _clean_text(user_msg.get("content") or "")
    tool_schema = tool_by_name[tool_name]
    wrapped_user_content = build_solver_user_message(task_description, tool_schema)

    asst_content = json.dumps({
        "reason": _clean_text(asst_msg.get("reasoning_content") or ""),
        "tool_call": tool_call_str,
    })

    return {
        "tool_idx": tool_idx,
        "attempt": 0,
        "tool_id": f"{spec_id}.{tool_name}",
        "env_state_before": None,
        "task": {
            "task_description": task_description,
            "expected_tool_call": tool_call_str,
            "env_metadata": None,
            "edited_metadata": None,
            "depends_on_previous": tool_idx > 0,
        },
        "env_state_after_task": None,
        "chat": [
            {"role": "user", "content": wrapped_user_content},
            {"role": "assistant", "content": asst_content},
            {"role": "tool", "content": tool_content},
        ],
        "judge": None,
        # env_update is {} for successful calls (preserves the "successful turn"
        # signal that traj_audit.summarize and other consumers check for), and
        # None for failed calls (4xx/5xx) — matches the live pipeline's shape.
        "env_update": {} if is_success else None,
        "env_state_after": None,
    }


def row_to_trajectory(
    row: Dict[str, Any],
    *,
    source_tag: str,
    filename_to_spec_id: Dict[str, str],
    spec_index: Dict[str, Dict[str, Any]],
    import_ts: str,
    warnings: List[Dict[str, Any]],
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Return (task_id, trajectory_json) or None if the row should be skipped."""
    row_id = row.get("id") or ""
    m = ID_RE.match(row_id)
    if not m:
        warnings.append({"ctx": row_id, "issue": "malformed_row_id"})
        return None
    _, stem, seq_key = m.group(1), m.group(2), m.group(3)
    old_filename = f"{stem}.yml"
    new_spec_id = filename_to_spec_id.get(old_filename)
    if new_spec_id is None:
        warnings.append({"ctx": row_id, "issue": "no_matching_env_spec", "value": old_filename})
        return None

    effective_seq = f"tc2_{seq_key}" if source_tag == "2" else seq_key
    task_id = f"{new_spec_id}_{effective_seq}"
    spec_info = spec_index[new_spec_id]
    tool_by_name = spec_info["tool_by_name"]

    messages = row.get("messages") or []
    turns: List[Dict[str, Any]] = []
    tool_ids: List[str] = []
    tool_idx = 0
    current_user = {"content": ""}
    i = 0
    while i < len(messages):
        role = messages[i].get("role")
        if role == "user":
            current_user = messages[i]
            i += 1
            continue
        if role == "assistant" and i + 1 < len(messages) and messages[i + 1].get("role") == "tool":
            turn = build_turn(
                current_user, messages[i], messages[i + 1],
                tool_idx=tool_idx, spec_id=new_spec_id, tool_by_name=tool_by_name,
                warn_ctx=row_id, warnings=warnings,
            )
            if turn is None:
                # Structural break (malformed tool call / unknown tool / malformed tool response).
                # Truncate here — keep every turn produced so far, drop everything after.
                warnings.append({"ctx": row_id, "issue": "truncated_at_turn", "value": tool_idx})
                break
            turns.append(turn)
            tool_ids.append(turn["tool_id"])
            tool_idx += 1
            i += 2
            continue
        i += 1

    if not turns:
        warnings.append({"ctx": row_id, "issue": "no_valid_turns"})
        return None

    # Trim tools list to only those actually called, in order of first appearance,
    # to match the live trajectory shape (tool_ids ~ tools 1:1 by unique name).
    seen = set()
    called_tools: List[Dict[str, Any]] = []
    for tid in tool_ids:
        name = tid.split(".", 1)[1]
        if name not in seen:
            seen.add(name)
            called_tools.append(tool_by_name[name])

    trajectory = {
        "task_id": task_id,
        "model": IMPORT_MODEL,
        "config": dict(IMPORT_CONFIG),
        "tool_ids": tool_ids,
        "tools": called_tools,
        "turns": turns,
        "solver_chat": [],
        "usage": dict(ZERO_USAGE),
        "generation_time_s": None,
        "summary": {
            "generated_at": import_ts,
            "model": IMPORT_MODEL,
            "model_config": dict(IMPORT_MODEL_CONFIG),
            "n_subtasks": len(turns),
            "prompt": None,
            "response": None,
            "parsed": {"task_summarized": row.get("summary") or ""},
            "usage": dict(ZERO_USAGE),
        },
        "imported_from": {
            "source": f"tool_content_{source_tag}/dataset.jsonl",
            "source_row_id": row_id,
            "source_schema": SOURCE_SCHEMA,
        },
    }
    return task_id, trajectory


def atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    tmp.replace(path)


def source_tag_from_path(p: Path) -> str:
    """Infer '2' / '3' from a dataset.jsonl path under .../tool_content_N/..."""
    for part in p.parts:
        if part.startswith("tool_content_"):
            tail = part.split("_", 2)[-1]
            if tail.isdigit():
                return tail
    return "X"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sources",
        nargs="+",
        default=[
            "/pscratch/sd/t/tcaste/synthtools_dataset/tool_content_3/dataset.jsonl",
            "/pscratch/sd/t/tcaste/synthtools_dataset/tool_content_2/dataset.jsonl",
        ],
        help="One or more dataset.jsonl paths. tc_3 should come first so its seq keys win over tc_2's.",
    )
    ap.add_argument(
        "--env-specs-dir",
        default="/pscratch/sd/t/tcaste/tool_content/env_specs",
    )
    ap.add_argument(
        "--output-dir",
        default="/pscratch/sd/t/tcaste/tool_content/trajectories",
    )
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit (per source)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    env_dir = Path(args.env_specs_dir)
    out_dir = Path(args.output_dir)
    if not env_dir.is_dir():
        print(f"ERROR: env specs dir {env_dir} not found", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Indexing env_specs in {env_dir} ...")
    filename_to_spec_id, spec_index = build_spec_index(env_dir)
    print(f"  {len(filename_to_spec_id)} old filenames, {len(spec_index)} spec_ids")

    import_ts = now_iso()
    warnings: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    n_written = 0
    n_collisions = 0

    for src in args.sources:
        src_path = Path(src)
        if not src_path.is_file():
            print(f"  !! source not found: {src_path}", file=sys.stderr)
            continue
        source_tag = source_tag_from_path(src_path)
        print(f"--- processing {src_path.name} (source_tag={source_tag}) ---")
        n_src_rows = 0
        n_src_written = 0
        with src_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_src_rows += 1
                if args.limit and n_src_rows > args.limit:
                    break
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    skipped.append({"source": str(src_path), "n": n_src_rows, "reason": f"json: {e}"})
                    continue
                result = row_to_trajectory(
                    row, source_tag=source_tag,
                    filename_to_spec_id=filename_to_spec_id,
                    spec_index=spec_index,
                    import_ts=import_ts,
                    warnings=warnings,
                )
                if result is None:
                    skipped.append({"source": str(src_path), "row_id": row.get("id", "?"), "reason": "skipped_by_converter"})
                    continue
                task_id, traj = result
                out_path = out_dir / f"{task_id}.json"
                if out_path.exists():
                    n_collisions += 1
                    continue
                if args.dry_run:
                    n_written += 1
                    n_src_written += 1
                    continue
                atomic_write_json(out_path, traj)
                n_written += 1
                n_src_written += 1
        print(f"  rows_read={n_src_rows}  written={n_src_written}")

    if not args.dry_run:
        warn_path = out_dir / "_import_warnings.jsonl"
        with warn_path.open("w") as f:
            for w in warnings:
                f.write(json.dumps(w, ensure_ascii=False) + "\n")
        skip_path = out_dir / "_import_skipped.jsonl"
        with skip_path.open("w") as f:
            for s in skipped:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print()
    print(f"Wrote:       {n_written} trajectory JSONs to {out_dir}")
    print(f"Warnings:    {len(warnings)}  →  _import_warnings.jsonl")
    print(f"Skipped:     {len(skipped)}  →  _import_skipped.jsonl")
    print(f"Collisions:  {n_collisions}  (same task_id appeared twice; later ignored)")
    if args.dry_run:
        print("(dry-run — nothing written to disk)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
