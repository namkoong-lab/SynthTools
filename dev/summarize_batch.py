#!/usr/bin/env python3
"""Run the summarizer on a set of tasks against a vLLM server and write
per-task YAML results + an index.md under dev/results/summarize_<ts>/.

Task selection (one of):
  --task-ids id1,id2,id3
  --task-list path/to/file.txt   (one task_id per line)
  --field "Cybersecurity" --n 10
  --n 10                          (random eligible tasks across all fields)

Usage:
  python dev/summarize_batch.py --task-ids \\
    academic_publishing_and_citations_spec_000_seq11,\\
    academic_publishing_and_citations_spec_000_seq1
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

from llm import LLM, MODEL_REGISTRY  # noqa: E402
from roles.task_summarizer import TaskSummarizer  # noqa: E402
from task_audit.checks import audit_task_pre_summarize, any_block  # noqa: E402
from task_audit.summarize import _triples_for_summarizer, _SPEC_ID_RE  # noqa: E402
from utils import extract_json_objects  # noqa: E402

TASKS_DIR = Path("/pscratch/sd/t/tcaste/tool_content/tasks")
ENV_SPECS_DIR = Path("/pscratch/sd/t/tcaste/tool_content/env_specs")
RESULTS_ROOT = REPO / "dev" / "results"


def _load_spec(task_id: str) -> Optional[Dict[str, Any]]:
    m = _SPEC_ID_RE.match(task_id)
    if not m:
        return None
    p = ENV_SPECS_DIR / f"{m.group(1)}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _select_task_ids(args: argparse.Namespace) -> List[str]:
    if args.task_ids:
        return [t.strip() for t in args.task_ids.split(",") if t.strip()]
    if args.task_list:
        return [
            ln.strip() for ln in Path(args.task_list).read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
    candidates = sorted(
        p.stem for p in TASKS_DIR.glob("*.json")
        if not p.name.endswith(".debug.json")
    )
    if args.field:
        # Field is in the spec, not the task filename. Filter via spec lookup.
        field_lower = args.field.lower()
        kept = []
        seen_spec: Dict[str, Optional[str]] = {}
        for tid in candidates:
            m = _SPEC_ID_RE.match(tid)
            if not m:
                continue
            sid = m.group(1)
            if sid not in seen_spec:
                spec = _load_spec(tid)
                seen_spec[sid] = (spec or {}).get("field", "")
            if (seen_spec[sid] or "").lower() == field_lower:
                kept.append(tid)
        candidates = kept
    if not candidates:
        sys.exit("no candidate tasks matched")
    if args.n and args.n < len(candidates):
        random.seed(args.seed)
        candidates = random.sample(candidates, args.n)
    return candidates


def _overhide_candidates(parsed: Dict[str, Any], clean: List[Dict[str, Any]]) -> List[str]:
    tool_vals = parsed.get("tool_produced_values") or []
    user_texts: List[str] = []
    for turn in clean:
        for msg in turn.get("chat") or []:
            if isinstance(msg, dict) and msg.get("role") == "user":
                c = msg.get("content")
                if isinstance(c, str):
                    user_texts.append(c)
    joined = "\n".join(user_texts)
    return [v for v in tool_vals if isinstance(v, str) and v.strip() and v in joined]


def run_one(task_id: str, llm, role: TaskSummarizer) -> Dict[str, Any]:
    p = TASKS_DIR / f"{task_id}.json"
    if not p.exists():
        return {"task_id": task_id, "status": "missing", "error": f"no file {p}"}
    raw = json.loads(p.read_text())
    spec = _load_spec(task_id)
    if spec is None:
        return {"task_id": task_id, "status": "no_spec"}

    clean, issues = audit_task_pre_summarize(raw, spec)
    if any_block(issues):
        return {"task_id": task_id, "status": "pre_audit_BLOCK",
                "issues": [i.to_dict() for i in issues]}
    if not clean:
        return {"task_id": task_id, "status": "no_clean_turns"}

    triples = _triples_for_summarizer(clean, spec=spec)
    messages = role.build_messages(
        tasks=triples["tasks"], tool_calls=triples["tool_calls"],
        tool_responses=triples["tool_responses"], tools=triples["tools"],
    )
    response = llm(messages)
    objs = extract_json_objects(response)
    parsed = objs[0] if objs and isinstance(objs[0], dict) else None

    out: Dict[str, Any] = {
        "task_id": task_id,
        "status": "ok" if parsed else "no_parse",
        "n_clean_turns": len(clean),
        "n_tools": len(triples["tools"]),
        "prompt_chars": sum(len(m["content"]) for m in messages),
    }
    if parsed:
        out["user_supplied_values"] = parsed.get("user_supplied_values") or []
        out["tool_produced_values"] = parsed.get("tool_produced_values") or []
        out["spillover"] = parsed.get("spillover") or []
        out["task_summarized"] = parsed.get("task_summarized") or ""
        out["overhide_candidates"] = _overhide_candidates(parsed, clean)
    else:
        out["raw_response"] = response
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-ids", help="comma-separated task ids")
    ap.add_argument("--task-list", help="path to file with one task id per line")
    ap.add_argument("--field", help="select tasks whose spec.field matches")
    ap.add_argument("--n", type=int, default=0,
                    help="random sample size when not using --task-ids/--task-list")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="GPT-OSS-120B", choices=list(MODEL_REGISTRY))
    ap.add_argument("--server-url", default="http://localhost:8765/v1")
    ap.add_argument("--out-dir", default=None,
                    help="override the dev/results/summarize_<ts>/ destination")
    args = ap.parse_args()

    task_ids = _select_task_ids(args)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_ROOT / f"summarize_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"results dir : {out_dir}")
    print(f"model       : {args.model}")
    print(f"server      : {args.server_url}")
    print(f"tasks       : {len(task_ids)}")
    print("-" * 60, flush=True)

    llm = LLM(args.model, server_url=args.server_url)
    role = TaskSummarizer(llm)

    index_rows: List[Dict[str, Any]] = []
    for i, tid in enumerate(task_ids, 1):
        print(f"[{i}/{len(task_ids)}] {tid}", flush=True)
        res = run_one(tid, llm, role)
        (out_dir / f"{tid}.yml").write_text(
            yaml.safe_dump(res, allow_unicode=True, sort_keys=False, width=1000)
        )
        index_rows.append({
            "task_id": tid,
            "status": res.get("status"),
            "overhide_n": len(res.get("overhide_candidates") or []),
            "summary_chars": len(res.get("task_summarized") or ""),
            "n_turns": res.get("n_clean_turns"),
            "n_tools": res.get("n_tools"),
            "prompt_chars": res.get("prompt_chars"),
        })

    md = ["# summarize batch", f"", f"- timestamp: {ts}",
          f"- model: {args.model}", f"- server: {args.server_url}",
          f"- tasks: {len(task_ids)}", "",
          "| task_id | status | overhide | summary_chars | turns | tools | prompt_chars |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for r in index_rows:
        md.append(f"| {r['task_id']} | {r['status']} | {r['overhide_n']} | "
                  f"{r['summary_chars']} | {r['n_turns']} | {r['n_tools']} | "
                  f"{r['prompt_chars']} |")
    (out_dir / "index.md").write_text("\n".join(md) + "\n")
    print("-" * 60)
    print(f"wrote index.md and {len(index_rows)} per-task YAML files in {out_dir}")


if __name__ == "__main__":
    main()
