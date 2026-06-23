#!/usr/bin/env python3
"""End-to-end inspection pipeline for SynthTools tasks.

For each selected task, runs all stages against a vLLM server and writes a
fully-formatted Markdown transcript (every prompt + every LLM output) plus a
structured JSON to /global/homes/t/tcaste/projects/trajectories/:

  1. select task(s)
  2. (optionally) launch a vLLM server
  3. summarize the task (fresh, via the real audit + TaskSummarizer)
  4. roll out the agent against the ToolSimulator
  5. judge the trajectory
  6. write the report

Examples (server already up on localhost:8765):
  python try_trajectory.py --task-id academic_publishing_and_citations_spec_000_seq1
  python try_trajectory.py --task-ids a,b,c
  python try_trajectory.py --field "Cybersecurity" --n 5
  python try_trajectory.py --n 3                      # 3 random eligible tasks

Launch the server as part of the run:
  python try_trajectory.py --n 3 --launch-server
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path("/global/homes/t/tcaste/projects/SynthTools")
sys.path.insert(0, str(REPO))

from llm import LLM, MODEL_REGISTRY                                    # noqa: E402
from roles.task_summarizer import TaskSummarizer                       # noqa: E402
from roles.task_solver import TaskSolver                               # noqa: E402
from roles.tool_simulator import ToolSimulator                         # noqa: E402
from roles.trajectory_judge import TrajectoryJudge                     # noqa: E402
from trajectory_generation.loader import Task                          # noqa: E402
from task_audit.summarize import (                                     # noqa: E402
    _triples_for_summarizer, _successful_call_and_response, _SPEC_ID_RE,
)
from task_audit.checks import audit_task_pre_summarize, any_block      # noqa: E402
from utils import extract_json_objects, to_llm_messages                # noqa: E402

TASKS_DIR_DEFAULT = Path("/pscratch/sd/t/tcaste/tool_content/tasks")
ENV_SPECS_DIR_DEFAULT = Path("/pscratch/sd/t/tcaste/tool_content/env_specs")
REPORT_DIR_DEFAULT = Path("/global/homes/t/tcaste/projects/trajectories")

FENCE = "~~~"


# ---------------------------------------------------------------------------
# vLLM server (optional in-pipeline launch)
# ---------------------------------------------------------------------------

def _server_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def launch_server(model: str, port: int, tp: int, ready_timeout: int) -> subprocess.Popen:
    serve_model = MODEL_REGISTRY[model].model_id
    cmd = [
        "vllm", "serve", serve_model,
        "--port", str(port),
        "--tensor-parallel-size", str(tp),
        "--max-model-len", "32768",
        "--gpu-memory-utilization", "0.90",
        "--dtype", "bfloat16",
        "--trust-remote-code",
        "--reasoning-parser", "openai_gptoss",
    ]
    print(f"[server] launching: {' '.join(cmd)}", flush=True)
    log = open("/tmp/inspect_vllm.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)

    def _cleanup():
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    atexit.register(_cleanup)

    url = f"http://localhost:{port}/v1/models"
    for i in range(ready_timeout):
        if _server_ready(url):
            print(f"[server] ready after {i}s", flush=True)
            return proc
        if proc.poll() is not None:
            raise SystemExit("[server] vllm exited before becoming ready; see /tmp/inspect_vllm.log")
        time.sleep(1)
    raise SystemExit(f"[server] not ready after {ready_timeout}s; see /tmp/inspect_vllm.log")


# ---------------------------------------------------------------------------
# Task selection
# ---------------------------------------------------------------------------

def _is_eligible(raw: Dict[str, Any], min_turns: int) -> bool:
    n_succ = sum(1 for t in (raw.get("turns") or []) if t.get("env_update") is not None)
    return n_succ >= min_turns


def _select_paths(args) -> List[Path]:
    tasks_dir: Path = args.tasks_dir
    if args.task_id:
        p = tasks_dir / f"{args.task_id}.json"
        if not p.exists():
            raise SystemExit(f"no task json at {p}")
        return [p]
    if args.task_ids:
        out = []
        for tid in [x.strip() for x in args.task_ids.split(",") if x.strip()]:
            p = tasks_dir / f"{tid}.json"
            if not p.exists():
                raise SystemExit(f"no task json at {p}")
            out.append(p)
        return out

    # random selection (optionally restricted to a field)
    spec_ids_in_field: Optional[set] = None
    if args.field:
        spec_ids_in_field = set()
        for sp in args.env_specs_dir.glob("*_spec_*.json"):
            if sp.name.endswith(".tmp"):
                continue
            try:
                s = json.loads(sp.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if s.get("field") == args.field and s.get("spec_id"):
                spec_ids_in_field.add(s["spec_id"])
        if not spec_ids_in_field:
            raise SystemExit(f"no env_specs found for field {args.field!r}")

    names = []
    for entry in os.scandir(tasks_dir):
        if not entry.name.endswith(".json") or entry.name.endswith(".debug.json"):
            continue
        if spec_ids_in_field is not None:
            m = _SPEC_ID_RE.match(entry.name)
            if not m or m.group(1) not in spec_ids_in_field:
                continue
        names.append(entry.name)
    random.shuffle(names)

    chosen: List[Path] = []
    for fn in names:
        try:
            raw = json.loads((tasks_dir / fn).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if _is_eligible(raw, args.min_turns):
            chosen.append(tasks_dir / fn)
        if len(chosen) >= args.n:
            break
    if not chosen:
        raise SystemExit("no eligible task found for the given selection")
    return chosen


# ---------------------------------------------------------------------------
# Stage 3 — summarize (fresh)
# ---------------------------------------------------------------------------

def summarize(raw: Dict[str, Any], spec: Dict[str, Any], llm) -> Dict[str, Any]:
    clean, issues = audit_task_pre_summarize(raw, spec)
    if any_block(issues):
        return {"ok": False, "reason": "pre-audit BLOCK",
                "issues": [i.to_dict() for i in issues], "clean": []}
    if not clean:
        return {"ok": False, "reason": "no clean successful turns",
                "issues": [i.to_dict() for i in issues], "clean": []}

    triples = _triples_for_summarizer(clean, spec=spec)
    role = TaskSummarizer(llm)
    messages = role.build_messages(
        tasks=triples["tasks"], tool_calls=triples["tool_calls"],
        tool_responses=triples["tool_responses"], tools=triples["tools"],
    )
    response = llm(messages)
    objs = extract_json_objects(response)
    parsed = objs[0] if objs and isinstance(objs[0], dict) else None
    return {
        "ok": parsed is not None and bool(parsed.get("task_summarized")),
        "clean": clean,
        "triples": triples,
        "prompt": messages,
        "response": response,
        "parsed": parsed,
        "issues": [i.to_dict() for i in issues],
    }


def build_task(raw: Dict[str, Any], spec: Dict[str, Any],
               clean: List[Dict[str, Any]], summary_text: str) -> Task:
    used_names: List[str] = []
    seen = set()
    for t in clean:
        nm = (t.get("tool_id") or "").split(".")[-1]
        if nm and nm not in seen:
            seen.add(nm); used_names.append(nm)
    by_name = {t.get("tool_name"): t for t in (spec.get("tools") or [])}
    tools = [by_name[n] for n in used_names if n in by_name]

    gt_tool_calls: List[str] = []
    for turn in clean:
        call, _ = _successful_call_and_response(turn.get("chat") or [])
        if not call:
            call = (turn.get("task") or {}).get("expected_tool_call") or ""
        if call:
            gt_tool_calls.append(call)

    t0 = (clean[0].get("task") or {}) if clean else {}
    initial_state = t0.get("edited_metadata") or t0.get("env_metadata")
    final_state = clean[-1].get("env_state_after") if clean else None

    return Task(
        id=raw.get("task_id") or "",
        field=spec.get("field") or "",
        summary=summary_text,
        tools=tools,
        gt_tool_calls=gt_tool_calls,
        initial_state=initial_state,
        final_state=final_state,
    )


# ---------------------------------------------------------------------------
# Stage 4 — rollout
# ---------------------------------------------------------------------------

def build_replay_map(clean: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Per-tool list of recorded {tool_call, tool_output} from the GT chat."""
    m: Dict[str, List[Dict[str, Any]]] = {}
    for turn in clean:
        call, out = _successful_call_and_response(turn.get("chat") or [])
        if not call:
            continue
        mm = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", call)
        if not mm:
            continue
        try:
            out_parsed = json.loads(out)
        except (TypeError, ValueError):
            out_parsed = out
        m.setdefault(mm.group(1), []).append({"tool_call": call, "tool_output": out_parsed})
    return m


def rollout(task: Task, llm, max_solver_turns: int,
            replay_map: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> Dict[str, Any]:
    solver = TaskSolver(runner=llm, mode="trajectory")
    simulator = ToolSimulator(runner=llm)

    initial_user = TaskSolver.build_user_message(task.summary, task.tools)
    chat: List[Dict[str, str]] = [
        {"role": "system", "content": solver.system_prompt()},
        {"role": "user", "content": initial_user},
    ]

    turns: List[Dict[str, Any]] = []
    observed_calls: List[Dict[str, Any]] = []
    stop_reason = "max_solver_turns"
    start = time.time()

    for turn_idx in range(max_solver_turns):
        solver_msgs = to_llm_messages(chat)
        response = llm(solver_msgs)
        chat.append({"role": "assistant", "content": response})
        objs = extract_json_objects(response)
        parsed = objs[0] if objs and isinstance(objs[0], dict) else None
        tool_call = parsed.get("tool_call") if parsed else None
        reason = parsed.get("reason") if parsed else None

        rec: Dict[str, Any] = {
            "turn_idx": turn_idx,
            "solver_prompt": solver_msgs,
            "solver_response": response,
            "reason": reason,
            "tool_call": tool_call,
        }

        if not tool_call:
            nudge = ("No tool call detected. Reply with a single JSON object "
                     "containing 'reason' and 'tool_call' (use \"<STOP>\" to finish).")
            chat.append({"role": "user", "content": nudge})
            rec["event"] = "empty_call_nudge"
            turns.append(rec)
            continue

        if "<STOP>" in str(tool_call):
            stop_reason = "stop_emitted"
            rec["event"] = "stop"
            turns.append(rec)
            break

        m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", tool_call or "")
        name = m.group(1) if m else None
        tool_data = next((t for t in task.tools if t.get("tool_name") == name), None)
        if tool_data is None:
            err = {"status_code": 400,
                   "response": {"error": f"Unknown tool name. Choose from: "
                                          f"{[t.get('tool_name') for t in task.tools]}"}}
            chat.append({"role": "tool", "content": json.dumps(err)})
            rec["event"] = "unknown_tool"
            rec["tool_reply"] = err
            turns.append(rec)
            continue

        replay_examples = replay_map.get(name) if replay_map else None
        sim_res = simulator.simulate(tool_data, tool_call, metadata=task.initial_state,
                                     replay_examples=replay_examples)
        sim_parsed = sim_res.get("parsed") or {}
        passed = bool(sim_parsed.get("passed"))
        payload = sim_parsed.get("simulation") if passed else sim_parsed.get("parameter_check")

        rec["event"] = "tool_call"
        rec["tool_name"] = tool_data.get("tool_name")
        rec["passed"] = passed
        rec["replayed"] = bool(replay_examples)
        rec["ast_result"] = sim_parsed.get("ast_result")
        rec["param_check_prompt"] = sim_res.get("prompt")
        rec["param_check_response"] = sim_res.get("response")
        rec["param_check_parsed"] = sim_parsed.get("parameter_check")
        rec["simulate_prompt"] = sim_parsed.get("simulation_prompt")
        rec["simulate_response"] = sim_parsed.get("simulation_response")
        rec["tool_reply"] = payload
        turns.append(rec)

        chat.append({"role": "tool", "content": json.dumps(payload, ensure_ascii=False, default=str)})
        if passed:
            observed_calls.append({"tool_call": tool_call, "tool_output": sim_parsed.get("simulation")})

    return {
        "initial_user": initial_user,
        "turns": turns,
        "observed_calls": observed_calls,
        "stop_reason": stop_reason,
        "elapsed_s": round(time.time() - start, 2),
    }


# ---------------------------------------------------------------------------
# Stage 5 — judge
# ---------------------------------------------------------------------------

def judge(task: Task, observed_calls: List[Dict[str, Any]], llm) -> Dict[str, Any]:
    j = TrajectoryJudge(runner=llm)
    return j.judge_trajectory(
        task_summary=task.summary,
        gt_tool_calls=task.gt_tool_calls,
        agent_tool_calls=observed_calls,
        initial_state=task.initial_state,
        final_state_gt=task.final_state,
    )


# ---------------------------------------------------------------------------
# Stage 6 — report
# ---------------------------------------------------------------------------

def _fence(content: Any) -> str:
    if not isinstance(content, str):
        content = json.dumps(content, indent=2, ensure_ascii=False, default=str)
    return f"{FENCE}\n{content}\n{FENCE}"


def _fmt_messages(messages: Any) -> str:
    """Render a chat-message list (or a raw string) as labeled fenced blocks."""
    if isinstance(messages, str):
        return _fence(messages)
    out = []
    for msg in messages or []:
        if isinstance(msg, dict):
            role = msg.get("role", "?")
            content = msg.get("content", "")
        else:
            role, content = "?", msg
        out.append(f"**[{role}]**\n{_fence(content)}")
    return "\n\n".join(out)


def write_report(out_dir: Path, task: Task, summ: Dict[str, Any],
                 roll: Dict[str, Any], judge_cap: Dict[str, Any],
                 stamp: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    md: List[str] = []
    md.append(f"# Trajectory report — {task.id}\n")
    md.append(f"- field: {task.field}")
    md.append(f"- tools: {[t.get('tool_name') for t in task.tools]}")
    md.append(f"- generated: {stamp}")
    md.append("")

    # Stage 1 — summarization
    md.append("## Stage 1 — Summarization\n")
    md.append("### Successful turns fed to the summarizer\n")
    triples = summ.get("triples") or {"tasks": [], "tool_calls": [], "tool_responses": []}
    for i in range(len(triples["tasks"])):
        md.append(f"**Subtask {i}**")
        md.append(f"- task_description:\n{_fence(triples['tasks'][i])}")
        md.append(f"- tool_call:\n{_fence(triples['tool_calls'][i])}")
        md.append(f"- tool_response:\n{_fence(triples['tool_responses'][i])}")
        md.append("")
    md.append("### Summarizer prompt\n")
    md.append(_fmt_messages(summ.get("prompt")))
    md.append("\n### Summarizer raw response\n")
    md.append(_fence(summ.get("response")))
    md.append("\n### Parsed summary\n")
    parsed = summ.get("parsed") or {}
    md.append(f"- user_supplied_values:\n{_fence(parsed.get('user_supplied_values'))}")
    md.append(f"- tool_produced_values:\n{_fence(parsed.get('tool_produced_values'))}")
    md.append(f"- spillover:\n{_fence(parsed.get('spillover'))}")
    md.append(f"- task_summarized:\n{_fence(parsed.get('task_summarized'))}")
    md.append("")

    # Stage 2 — rollout
    md.append("## Stage 2 — Trajectory rollout\n")
    md.append(f"- stop_reason: {roll['stop_reason']}")
    md.append(f"- turns: {len(roll['turns'])}")
    md.append(f"- elapsed_s: {roll['elapsed_s']}\n")
    md.append("### Initial user message (summary + tool catalogue)\n")
    md.append(_fence(roll["initial_user"]))
    md.append("")
    for rec in roll["turns"]:
        md.append(f"### Turn {rec['turn_idx']} — {rec.get('event')}\n")
        md.append("**Solver prompt**")
        md.append(_fmt_messages(rec.get("solver_prompt")))
        md.append("\n**Solver response**")
        md.append(_fence(rec.get("solver_response")))
        if rec.get("tool_call"):
            md.append(f"\n**Parsed tool_call:** `{rec.get('tool_call')}`")
        if rec.get("ast_result") is not None:
            md.append("\n**AST result**")
            md.append(_fence(rec.get("ast_result")))
        if rec.get("param_check_prompt"):
            md.append("\n**Tool param-check prompt**")
            md.append(_fence(rec.get("param_check_prompt")))
            md.append("\n**Tool param-check response**")
            md.append(_fence(rec.get("param_check_response")))
        if rec.get("simulate_prompt"):
            md.append("\n**Tool simulate prompt**")
            md.append(_fence(rec.get("simulate_prompt")))
            md.append("\n**Tool simulate response**")
            md.append(_fence(rec.get("simulate_response")))
        if rec.get("tool_reply") is not None:
            md.append("\n**Resolved tool reply**")
            md.append(_fence(rec.get("tool_reply")))
        md.append("")

    # Stage 3 — judge
    md.append("## Stage 3 — Trajectory judge\n")
    if judge_cap:
        md.append("**Judge prompt**")
        md.append(_fence(judge_cap.get("prompt")))
        md.append("\n**Judge raw response**")
        md.append(_fence(judge_cap.get("response")))
        md.append("\n**Parsed verdict**")
        md.append(_fence(judge_cap.get("parsed")))
    else:
        md.append("_(judge skipped)_")
    md.append("")

    # Ground truth
    md.append("## Ground truth\n")
    md.append("**gt_tool_calls** (chat-call-first extraction)")
    md.append(_fence(task.gt_tool_calls))
    md.append("\n**initial_state**")
    md.append(_fence(task.initial_state))
    md.append("\n**final_state**")
    md.append(_fence(task.final_state))
    md.append("")

    report_md = out_dir / f"{task.id}.md"
    report_md.write_text("\n".join(md))

    # structured sidecar
    (out_dir / f"{task.id}.json").write_text(json.dumps({
        "task_id": task.id,
        "field": task.field,
        "summary": task.summary,
        "gt_tool_calls": task.gt_tool_calls,
        "rollout": {
            "stop_reason": roll["stop_reason"],
            "n_turns": len(roll["turns"]),
            "observed_calls": roll["observed_calls"],
        },
        "judge": judge_cap.get("parsed") if judge_cap else None,
        "generated": stamp,
    }, indent=2, ensure_ascii=False, default=str))
    return report_md


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _load(task_path: Path, env_specs_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    raw = json.loads(task_path.read_text())
    tid = raw.get("task_id") or task_path.stem
    m = _SPEC_ID_RE.match(tid)
    if not m:
        raise SystemExit(f"task_id {tid!r} does not match *_spec_NNN_* convention")
    spec = json.loads((env_specs_dir / f"{m.group(1)}.json").read_text())
    return raw, spec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sel = ap.add_argument_group("task selection")
    sel.add_argument("--task-id", type=str, default=None)
    sel.add_argument("--task-ids", type=str, default=None, help="comma-separated ids")
    sel.add_argument("--field", type=str, default=None)
    sel.add_argument("--n", type=int, default=1, help="number of random tasks")
    sel.add_argument("--min-turns", type=int, default=2)
    sel.add_argument("--seed", type=int, default=42)

    ap.add_argument("--tasks-dir", type=Path, default=TASKS_DIR_DEFAULT)
    ap.add_argument("--env-specs-dir", type=Path, default=ENV_SPECS_DIR_DEFAULT)
    ap.add_argument("--report-dir", type=Path, default=REPORT_DIR_DEFAULT)
    ap.add_argument("--model", default="GPT-OSS-120B", choices=list(MODEL_REGISTRY))
    ap.add_argument("--server-url", default="http://localhost:8765/v1")
    ap.add_argument("--launch-server", action="store_true",
                    help="Popen `vllm serve` and tear it down on exit")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--ready-timeout", type=int, default=1800)
    ap.add_argument("--max-solver-turns", type=int, default=12)
    ap.add_argument("--replay", action="store_true",
                    help="attach recorded GT call/output pairs to the simulator so it "
                         "returns the recorded output when the agent's call matches")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--write-summary", action="store_true",
                    help="persist the freshly-generated summary into the task JSON")
    args = ap.parse_args()

    random.seed(args.seed)

    if args.launch_server:
        launch_server(args.model, args.port, args.tp, args.ready_timeout)
        args.server_url = f"http://localhost:{args.port}/v1"

    paths = _select_paths(args)
    print(f"selected {len(paths)} task(s): {[p.stem for p in paths]}", flush=True)

    llm = LLM(args.model, server_url=args.server_url)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    written = []
    for path in paths:
        raw, spec = _load(path, args.env_specs_dir)
        tid = raw.get("task_id") or path.stem
        print(f"\n=== {tid} ===", flush=True)

        print("  [1/3] summarizing ...", flush=True)
        summ = summarize(raw, spec, llm)
        if not summ.get("ok"):
            print(f"  SKIP: summarization unusable ({summ.get('reason')})", flush=True)
            continue
        summary_text = (summ["parsed"] or {}).get("task_summarized") or ""

        if args.write_summary:
            raw["summary"] = {"parsed": summ["parsed"], "response": summ["response"]}
            path.write_text(json.dumps(raw, indent=2, ensure_ascii=False, default=str))

        task = build_task(raw, spec, summ["clean"], summary_text)

        print("  [2/3] rolling out trajectory ...", flush=True)
        roll = rollout(task, llm, args.max_solver_turns)

        judge_cap: Dict[str, Any] = {}
        if not args.no_judge:
            print("  [3/3] judging ...", flush=True)
            judge_cap = judge(task, roll["observed_calls"], llm)

        report = write_report(args.report_dir, task, summ, roll, judge_cap, stamp)
        verdict = (judge_cap.get("parsed") or {}) if judge_cap else {}
        print(f"  report: {report}", flush=True)
        print(f"  solved={verdict.get('trajectory_solved')} "
              f"match_rate={verdict.get('tool_call_match_rate')} "
              f"final_state_match={verdict.get('final_state_match')}", flush=True)
        written.append(report)

    print(f"\nwrote {len(written)} report(s) to {args.report_dir}", flush=True)


if __name__ == "__main__":
    main()
