"""Roll out an LLM agent against a single Task, save the trajectory, and judge it.

Reuses the existing TaskSolver (in trajectory mode) and ToolSimulator from
roles/. Drops TaskEvolver/per-turn TaskJudge/EnvironmentSimulator — the
release JSONL already supplies the ground truth and TrajectoryJudge runs
once at the end.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from roles.task_solver import TaskSolver
from roles.tool_simulator import ToolSimulator
from roles.trajectory_judge import TrajectoryJudge
from trajectory_generation.loader import Task
from utils import (
    RunLog,
    UsageTracker,
    extract_json_objects,
    get_logger,
    to_llm_messages,
    write_json_atomic,
)

logger = get_logger("synthtools.trajectory_generation")


# --- helpers ---

def _tool_name_from_call(tool_call_str: str) -> Optional[str]:
    """Extract the leading identifier from a "Name(...)" tool-call string."""
    m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", tool_call_str or "")
    return m.group(1) if m else None


def _match_tool_data(call_str: str, tools: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Find the tool schema in `tools` that matches the call's leading name."""
    name = _tool_name_from_call(call_str)
    if not name:
        return None
    for t in tools:
        if t.get("tool_name") == name:
            return t
    return None


def _build_user_message(task: Task) -> str:
    """One initial user message: task summary + numbered tool catalogue."""
    return TaskSolver.build_user_message(task.summary, task.tools)


def _last_tool_payload(sim_parsed: Dict[str, Any]) -> Any:
    """Pick the most informative payload to record as the tool message content."""
    if sim_parsed.get("simulation"):
        return sim_parsed["simulation"]
    return sim_parsed.get("parameter_check")


def _strip_explanation(payload: Any) -> Any:
    """Drop the simulator's internal 'explanation' (backend reasoning the agent
    must never see) from a response payload."""
    if isinstance(payload, dict) and "explanation" in payload:
        return {k: v for k, v in payload.items() if k != "explanation"}
    return payload


# --- main orchestrator ---

def generate_trajectory(
    task: Task,
    llm,
    output_dir: Path,
    max_solver_turns: int = 12,
    debug: bool = True,
    run_judge: bool = True,
    judge_llm = None,
    nudge_on_no_call: bool = True,
    rollout_tag: str = "",
) -> Dict[str, Any]:
    """Run the agent against `task`, save the trajectory JSON, optionally judge.

    Returns the trajectory dict (with the judge verdict if run_judge).
    Output schema is structurally compatible with task_generation/generate.py:457.

    `rollout_tag` is appended to the output stem (e.g. ".s5"), so the same
    task can be rolled out multiple times (different seeds) without the
    trajectory and debug files colliding.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{task.id}{rollout_tag}"
    output_path = output_dir / f"{stem}.json"

    solver = TaskSolver(runner=llm, mode="trajectory")
    simulator = ToolSimulator(runner=llm)

    event_log = RunLog(stem) if debug else None
    usage_tracker = UsageTracker()
    start = time.time()

    def _record(agent: str, action: str, turn_ref: Any, result: Dict[str, Any]) -> None:
        usage_tracker.track(result)
        if event_log is None:
            return
        usage = result.get("usage")
        if isinstance(usage, dict) and "check" in usage:
            event_log.record(agent, "parameter_check", turn_ref, {
                "prompt": result.get("prompt"),
                "response": result.get("response"),
                "parsed": result["parsed"].get("parameter_check") if result.get("parsed") else None,
                "usage": usage.get("check"),
            })
            if result.get("parsed", {}).get("passed"):
                event_log.record(agent, "simulate", turn_ref, {
                    "prompt": result["parsed"].get("simulation_prompt"),
                    "response": result["parsed"].get("simulation_response"),
                    "parsed": result["parsed"].get("simulation"),
                    "usage": usage.get("simulation"),
                })
        else:
            event_log.record(agent, action, turn_ref, result)

    # Build the rolling chat: system + initial user message.
    chat: List[Dict[str, str]] = [
        {"role": "system",  "content": solver.system_prompt()},
        {"role": "user",    "content": _build_user_message(task)},
    ]

    turns: List[Dict[str, Any]] = []
    observed_calls: List[Dict[str, Any]] = []
    stop_reason = "max_solver_turns"

    for turn_idx in range(max_solver_turns):
        turn_ref = {"turn_idx": turn_idx}

        # 1) Agent step.
        solver_msgs = to_llm_messages(chat)
        response = llm(solver_msgs)
        usage_solver = getattr(llm, "last_usage", None)
        chat.append({"role": "assistant", "content": response})
        objs = extract_json_objects(response)
        parsed = objs[0] if objs and isinstance(objs[0], dict) else None
        _record("TaskSolver", "solve", turn_ref,
                {"prompt": solver_msgs, "response": response, "parsed": parsed,
                 "usage": usage_solver})

        tool_call = parsed.get("tool_call") if parsed else None

        # 2) Empty / malformed call → nudge once and continue.
        if not tool_call:
            if nudge_on_no_call:
                nudge = ("No tool call detected. Reply with a single JSON object "
                         "containing 'reason' and 'tool_call' (use \"<STOP>\" to finish).")
                chat.append({"role": "user", "content": nudge})
                continue
            stop_reason = "no_tool_call"
            break

        # 3) Termination signal.
        if "<STOP>" in str(tool_call):
            stop_reason = "stop_emitted"
            break

        # 4) Resolve which tool spec the call refers to.
        tool_data = _match_tool_data(tool_call, task.tools)
        if tool_data is None:
            err = json.dumps({
                "status_code": 400,
                "response": {
                    "error": (f"Unknown tool name in call. Choose one of: "
                              f"{[t.get('tool_name') for t in task.tools]}")
                },
            })
            chat.append({"role": "tool", "content": err})
            turns.append({
                "turn_idx": turn_idx,
                "tool_call": tool_call,
                "param_check": None,
                "passed": False,
                "tool_output": None,
                "tool_name": None,
                "error": "unknown_tool_name",
            })
            continue

        # 5) Simulate (parameter check + simulate). Thread the evolving state:
        # the simulator sees the initial state PLUS every prior successful
        # call/response, so resources created mid-rollout exist on later turns
        # (mirrors what the evolver/env_simulator give task generation).
        sim_metadata = {
            "initial_state": task.initial_state,
            "previous_tool_calls": observed_calls,
        }
        sim_res = simulator.simulate(tool_data, tool_call, metadata=sim_metadata)
        _record("ToolSimulator", "simulate", turn_ref, sim_res)
        sim_parsed = sim_res.get("parsed") or {}
        passed = bool(sim_parsed.get("passed"))
        # Strip the simulator's 'explanation' before the agent sees it.
        payload = _strip_explanation(_last_tool_payload(sim_parsed))
        chat.append({"role": "tool", "content": json.dumps(payload, ensure_ascii=False, default=str)})
        turns.append({
            "turn_idx": turn_idx,
            "tool_name": tool_data.get("tool_name"),
            "tool_call": tool_call,
            "param_check": sim_parsed.get("parameter_check"),
            "passed": passed,
            # Record what the agent actually received: the simulation on a
            # pass, else the parameter_check payload (a 400) on failure. These
            # were previously dropped to None, hiding the real status_code.
            "tool_output": payload,
        })
        if passed:
            observed_calls.append({
                "tool_call": tool_call,
                "tool_output": _strip_explanation(sim_parsed.get("simulation")),
            })

    elapsed = round(time.time() - start, 2)

    trajectory: Dict[str, Any] = {
        "task_id": task.id,
        "model":   getattr(llm, "model", None),
        "config":  {
            "max_solver_turns": max_solver_turns,
            "run_judge":         run_judge,
            "stop_reason":       stop_reason,
        },
        "tool_ids": [t.get("tool_name") for t in task.tools],
        "tools":    task.tools,
        "turns":    turns,
        "solver_chat": [m for m in chat if m["role"] != "system"],
        "ground_truth": {
            "summary":         task.summary,
            "field":           task.field,
            "gt_tool_calls":   task.gt_tool_calls,
            "initial_state":   task.initial_state,
            "final_state":     task.final_state,
        },
        "trajectory_judge": None,
        "usage":            usage_tracker.total,
        "generation_time_s": elapsed,
    }

    write_json_atomic(trajectory, output_path)
    logger.info(f"Trajectory saved: {output_path} (stop_reason={stop_reason}, turns={len(turns)})")

    # 6) Optional trajectory-level judge.
    if run_judge:
        judge = TrajectoryJudge(runner=judge_llm or llm)
        verdict = judge.judge_trajectory(
            task_summary=task.summary,
            gt_tool_calls=task.gt_tool_calls,
            agent_tool_calls=observed_calls,
            initial_state=task.initial_state,
            final_state_gt=task.final_state,
        )
        usage_tracker.track(verdict)
        if event_log:
            event_log.record("TrajectoryJudge", "judge_trajectory", None, verdict)
        trajectory["trajectory_judge"] = verdict.get("parsed")
        trajectory["usage"] = usage_tracker.total
        write_json_atomic(trajectory, output_path)
        logger.info(f"Trajectory judged: trajectory_solved="
                    f"{(trajectory['trajectory_judge'] or {}).get('trajectory_solved')}")

    if event_log:
        event_log.save(output_dir)

    return trajectory
