"""Task generation orchestrator.

Usage:
    from llm import LLM
    from task_generation.generate import generate_trajectory

    llm = LLM("GPT-OSS-120B")
    generate_trajectory(
        tool_ids=["aerospace_and_defense_tool_spec_1.MissionParser",
                  "aerospace_and_defense_tool_spec_1.LiftRequirementEstimator"],
        tools_dataset_path=Path("tool_content/tools_dataset.jsonl"),
        output_dir=Path("tool_content/test_output"),
        llm=llm,
    )
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from roles.task_evolver import TaskEvolver
from roles.task_solver import TaskSolver
from roles.tool_simulator import ToolSimulator
from roles.env_simulator import EnvironmentSimulator
from roles.task_judge import TaskJudge
from utils import (
    extract_json_objects,
    get_logger,
    next_artifact_index,
    RunLog,
    to_llm_messages,
    UsageTracker,
    usage_str,
    write_json_atomic,
)

logger = get_logger("synthtools")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_tools_dataset(path: Path) -> Dict[str, Dict]:
    """Load tools_dataset.jsonl into {id: row} dict."""
    index = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            index[row["id"]] = row
    return index


def make_task_id(tool_ids: List[str], output_dir: Path) -> str:
    """Generate task_id = {spec_name}_{counter:03d}."""
    spec_name = tool_ids[0].rsplit(".", 1)[0]
    return f"{spec_name}_{next_artifact_index(spec_name, output_dir):03d}"


def _extract_judge_input(chat: List[Dict]) -> List[Dict]:
    """Extract assistant/tool pairs for judge input."""
    pairs = []
    for i in range(len(chat) - 1):
        msg = chat[i]
        nxt = chat[i + 1]
        if msg.get("role") != "assistant" or nxt.get("role") != "tool":
            continue
        if "<STOP>" in (msg.get("content") or ""):
            continue

        call_content = msg.get("content", "")
        call_objs = extract_json_objects(call_content)
        call_parsed = None
        if call_objs and isinstance(call_objs[0], dict) and "tool_call" in call_objs[0]:
            call_parsed = call_objs[0]["tool_call"]

        tool_content = nxt.get("content", "")
        tool_objs = extract_json_objects(tool_content)
        tool_parsed = tool_objs[0] if tool_objs and isinstance(tool_objs[0], dict) else None

        pairs.append({
            "tool_call": call_parsed or call_content,
            "tool_output": tool_parsed or tool_content,
        })
    return pairs


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def generate_trajectory(
    tool_ids: List[str],
    tools_dataset_path: Path,
    output_dir: Path,
    llm,
    max_solver_turns: int = 5,
    max_retries: int = 5,
    max_solver_retries_in_place: int = 2,
    verifiable: bool = False,
    debug: bool = True,
    task_id: Optional[str] = None,
) -> Dict:
    """Generate a task for a sequence of tools.

    Args:
        tool_ids: Ordered list of tool IDs from tools_dataset.
        tools_dataset_path: Path to tools_dataset.jsonl.
        output_dir: Directory for output files.
        llm: LLM instance (callable, accepts strings).
        max_solver_turns: Max turns per solving loop.
        max_retries: Max evolver re-roll attempts per tool. Re-rolls happen when
            the judge marks `arguments_grounded=False` (the task itself is broken).
        max_solver_retries_in_place: Max solver retries with the SAME task when
            the judge marks `arguments_grounded=True AND task_solved=False` (task
            is fine, solver botched it). Only applies when verifiable=True.
        verifiable: If True, run TaskJudge after solving. If False, skip judging.
        debug: If True, save debug event log.
        task_id: Optional explicit task_id for output filename. If None, one is
            auto-generated via `make_task_id`. Used by the spec/field wrappers
            to produce `{spec_id}_{seq_key}.json` filenames for skip detection.

    Returns:
        The task dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tools_index = load_tools_dataset(tools_dataset_path)
    tools = []
    for tid in tool_ids:
        if tid not in tools_index:
            raise ValueError(f"Tool ID '{tid}' not found in {tools_dataset_path}")
        tools.append(tools_index[tid])

    seen_names = set()
    task_tool_set: List[Dict[str, Any]] = []
    for row in tools:
        name = (row.get("tool") or {}).get("tool_name")
        if name and name not in seen_names:
            seen_names.add(name)
            task_tool_set.append(row["tool"])

    if task_id is None:
        task_id = make_task_id(tool_ids, output_dir)

    task_evolver = TaskEvolver(llm)
    task_solver = TaskSolver(llm)
    tool_simulator = ToolSimulator(llm)
    env_simulator = EnvironmentSimulator(llm)
    task_judge = TaskJudge(llm)

    # Trigger model load before starting so engine init doesn't interleave with run logs.
    llm._ensure_engine()
    logger.info(f"Model: {llm.model}")
    logger.info(f"Starting task: {task_id}")
    logger.info(f"Tools: {[t['tool_name'] for t in tools]}")
    logger.info(f"Config: max_solver_turns={max_solver_turns}, max_retries={max_retries}, verifiable={verifiable}")

    start_time = time.time()

    event_log = RunLog(task_id) if debug else None

    turns = []
    env_state = None
    successful_tasks_list: List[Dict[str, Any]] = []
    unsuccessful_tasks_list: List[Dict[str, Any]] = []
    solver_messages: List[Dict[str, str]] = [
        {"role": "system", "content": task_solver.system_prompt()}
    ]
    usage_tracker = UsageTracker()

    def _record(agent_name, action, turn_ref, result):
        """Record to event log + track usage."""
        usage_tracker.track(result)
        if event_log:
            usage = result.get("usage")
            # ToolSimulator emits two LLM calls; split into two events so per-call usage is preserved.
            if isinstance(usage, dict) and "check" in usage:
                event_log.record(agent_name, "parameter_check", turn_ref, {
                    "prompt": result.get("prompt"),
                    "response": result.get("response"),
                    "parsed": result["parsed"].get("parameter_check") if result.get("parsed") else None,
                    "usage": usage.get("check"),
                })
                if result.get("parsed", {}).get("passed"):
                    event_log.record(agent_name, "simulate", turn_ref, {
                        "prompt": result["parsed"].get("simulation_prompt"),
                        "response": result["parsed"].get("simulation_response"),
                        "parsed": result["parsed"].get("simulation"),
                        "usage": usage.get("simulation"),
                    })
            else:
                event_log.record(agent_name, action, turn_ref, result)

    for tool_idx, (tool_id, tool_row) in enumerate(zip(tool_ids, tools)):
        tool_data = tool_row["tool"]
        tool_name = tool_data.get("tool_name", tool_id)
        solved = False

        for attempt in range(max_retries):
            turn_ref = {"tool_idx": tool_idx, "attempt": attempt}
            logger.info(f"Tool {tool_idx}/{len(tool_ids)}: {tool_name} (attempt {attempt})")

            turn = {
                "tool_idx": tool_idx,
                "attempt": attempt,
                "tool_id": tool_id,
                "env_state_before": env_state,
            }

            # --- Step 1: Task Creation ---
            if tool_idx == 0 and attempt == 0:
                task_result = task_evolver.evolve_task_t0(tool_data)
            else:
                task_result = task_evolver.evolve_task_t1(
                    successful_task=successful_tasks_list,
                    unsuccessful_tasks=unsuccessful_tasks_list,
                    tool_details=tool_data,
                    environment_state=env_state,
                )
            _record("TaskEvolver", "evolve_task", turn_ref, task_result)

            task_parsed = task_result.get("parsed") or {}
            task_description = task_parsed.get("task_description")
            expected_tool_call = task_parsed.get("tool_call")
            env_metadata = task_parsed.get("env_metadata")
            edited_metadata = task_parsed.get("edited_metadata")
            # t0 has no predecessor; for t1 trust the evolver's self-report, default to True if missing.
            depends_on_previous = (
                False if tool_idx == 0
                else bool(task_parsed.get("depends_on_previous", True))
            )

            turn["task"] = {
                "task_description": task_description,
                "expected_tool_call": expected_tool_call,
                "env_metadata": env_metadata,
                "edited_metadata": edited_metadata,
                "depends_on_previous": depends_on_previous,
            }

            env_state_after_task = env_metadata or env_state
            turn["env_state_after_task"] = env_state_after_task

            logger.info(f"Task created: {(task_description or '')[:80]}")

            # --- Step 2 + 3: Solver + Judge, with in-place solver retries ---
            #
            # When verifiable=True:
            #   - arguments_grounded=False  → break out of the inner loop, the
            #                                 outer loop re-rolls the evolver
            #                                 (the task itself is broken).
            #   - arguments_grounded=True
            #     AND task_solved=False     → solver botched a derivable task;
            #                                 retry the solver in place with the
            #                                 SAME task, up to max_solver_retries_in_place.
            #   - task_solved=True          → success, fall through to env update.
            #
            # When verifiable=False, in-place solver retries are disabled (the
            # outer max_retries evolver loop is the only retry mechanism).
            in_place_budget = max_solver_retries_in_place if verifiable else 1
            arguments_grounded = True   # default for non-verifiable mode
            task_solved = False
            judge_parsed = None
            attempt_chat = []
            last_tool_call = None
            last_tool_output = None
            last_assistant_msg = None
            last_tool_msg = None

            for solver_attempt in range(in_place_budget):
                user_msg = {
                    "role": "user",
                    "content": task_solver.build_user_message(task_description or "", task_tool_set),
                }
                messages = solver_messages + [user_msg]
                attempt_chat = [user_msg]
                last_tool_call = None
                last_tool_output = None
                last_assistant_msg = None
                last_tool_msg = None

                for solver_turn in range(max_solver_turns):
                    response = llm(to_llm_messages(messages))
                    usage = llm.last_usage
                    usage_info = usage_str(usage)

                    objs = extract_json_objects(response)
                    solver_parsed = objs[0] if objs else None

                    debug_prompt = messages[-1] if solver_turn > 0 else {"user": user_msg["content"]}
                    solver_result = {"prompt": debug_prompt, "response": response, "parsed": solver_parsed, "usage": usage}
                    _record("TaskSolver", f"solve_attempt_{solver_attempt}_turn_{solver_turn}", turn_ref, solver_result)

                    tool_call = solver_parsed.get("tool_call") if isinstance(solver_parsed, dict) else None
                    logger.info(f"Solver attempt {solver_attempt + 1}/{in_place_budget} turn {solver_turn + 1}: tool_call={tool_call} {usage_info}")

                    assistant_msg = {"role": "assistant", "content": response}
                    messages.append(assistant_msg)
                    attempt_chat.append(assistant_msg)

                    if not tool_call:
                        nudge = {"role": "user", "content": "No tool call detected. Output a JSON with reason and tool_call."}
                        messages.append(nudge)
                        attempt_chat.append(nudge)
                        continue

                    tool_call_str = tool_call if isinstance(tool_call, str) else json.dumps(tool_call, ensure_ascii=False)

                    sim_result = tool_simulator.simulate(
                        tool_data=tool_data,
                        tool_call_message=tool_call_str,
                        metadata=env_state_after_task,
                    )
                    _record("ToolSimulator", "simulate", turn_ref, sim_result)

                    sim_parsed = sim_result.get("parsed") or {}
                    sim_json = sim_parsed.get("simulation")
                    passed = sim_parsed.get("passed", False)

                    if passed and sim_json:
                        last_tool_call = tool_call_str
                        last_tool_output = sim_json
                        last_assistant_msg = assistant_msg
                        last_tool_msg = {"role": "tool", "content": json.dumps(sim_json, ensure_ascii=False)}
                        status = sim_json.get("status_code", "?")
                        logger.info(f"Tool simulated: status_code={status}")
                        attempt_chat.append(last_tool_msg)
                        break
                    else:
                        param_check = sim_parsed.get("parameter_check") or {}
                        error = param_check.get("error_message", "param check failed")
                        logger.info(f"Tool param check failed: {error}")

                        error_content = json.dumps(sim_parsed.get("parameter_check"), ensure_ascii=False)
                        error_tool = {"role": "tool", "content": error_content}
                        messages.append(error_tool)
                        attempt_chat.append(error_tool)

                # --- Judging (only if verifiable) ---
                if verifiable:
                    # Build prior_chat from successful_tasks_list — exactly what the
                    # agent will see at inference time. NO env_state, NO env_metadata.
                    prior_chat = [
                        {
                            "task_description": s.get("task_description"),
                            "tool_call": s.get("tool_call"),
                            "tool_response": (s.get("tool_simulated") or {}).get("response")
                            if isinstance(s.get("tool_simulated"), dict)
                            else s.get("tool_simulated"),
                        }
                        for s in successful_tasks_list
                    ]
                    judge_input = _extract_judge_input(attempt_chat)
                    judge_result = task_judge.judge_task_gen(
                        true_tool_call=expected_tool_call,
                        current_task_description=task_description,
                        prior_chat=prior_chat,
                        agent_tool_calls=judge_input,
                    )
                    _record("TaskJudge", "judge", turn_ref, judge_result)

                    judge_parsed = judge_result.get("parsed") or {}
                    arguments_grounded = bool(judge_parsed.get("arguments_grounded", True))
                    task_solved = bool(judge_parsed.get("task_solved", False))
                    logger.info(
                        f"Judge: arguments_grounded={arguments_grounded}, task_solved={task_solved}"
                    )
                else:
                    status_code = last_tool_output.get("status_code") if isinstance(last_tool_output, dict) else None
                    task_solved = last_tool_call is not None and isinstance(status_code, int) and 200 <= status_code < 300
                    arguments_grounded = True   # we don't probe this without the judge
                    logger.info(f"Skipping judge (verifiable=False). status_code={status_code} task_solved={task_solved}")

                # Bail out of the in-place solver loop on either success or
                # ungrounded (the outer evolver-reroll loop handles ungrounded).
                if task_solved or not arguments_grounded:
                    break
                if solver_attempt + 1 < in_place_budget:
                    logger.info(
                        f"Grounded but solver botched. Retrying solver in place "
                        f"({solver_attempt + 1}/{in_place_budget})."
                    )

            turn["chat"] = attempt_chat
            turn["judge"] = judge_parsed if verifiable else None

            # --- Step 4: Environment Update ---
            if task_solved and last_tool_call and last_tool_output:
                env_result = env_simulator.update_environment(
                    tool_schema=tool_data,
                    tool_call_message=last_tool_call,
                    environment_state=env_state_after_task,
                    tool_simulation_output=last_tool_output,
                )
                _record("EnvironmentSimulator", "update", turn_ref, env_result)

                env_parsed = env_result.get("parsed") or {}
                turn["env_update"] = env_parsed

                if isinstance(env_parsed, dict) and "full_metadata" in env_parsed:
                    env_state = env_parsed["full_metadata"]
                elif env_parsed:
                    env_state = env_parsed
                turn["env_state_after"] = env_state

                successful_tasks_list.append({
                    "task_description": task_description,
                    "tool_call": last_tool_call,
                    "tool_simulated": {
                        "status_code": last_tool_output.get("status_code"),
                        "response": last_tool_output.get("response"),
                    } if isinstance(last_tool_output, dict) else last_tool_output,
                })
                unsuccessful_tasks_list = []
                if last_assistant_msg and last_tool_msg:
                    historical_user = {"role": "user", "content": task_description or ""}
                    solver_messages.extend([historical_user, last_assistant_msg, last_tool_msg])
                solved = True
                logger.info("Solved. Advancing to next tool.")
            else:
                turn["env_update"] = None
                turn["env_state_after"] = env_state

                # Surface the judge's feedback (consumed by task_evolver_t1 on
                # the next attempt). When verifiable=False there is no judge,
                # so the field stays None.
                feedback = (
                    judge_parsed.get("feedback") if isinstance(judge_parsed, dict) else None
                )
                unsuccessful_tasks_list.append({
                    "task_description": task_description,
                    "tool_call": last_tool_call,
                    "judge_explanation": feedback,
                })
                if feedback:
                    logger.info(f"Not solved. Judge feedback: {feedback}")
                logger.info(f"Not solved. Re-rolling evolver ({attempt + 1}/{max_retries}).")

            turns.append(turn)

            if solved:
                break

        if not solved:
            logger.warning(f"Max retries reached for {tool_name}. Stopping sequence.")
            break

    elapsed = time.time() - start_time

    task = {
        "task_id": task_id,
        "model": llm.model,
        "config": {
            "max_solver_turns": max_solver_turns,
            "max_retries": max_retries,
            "verifiable": verifiable,
        },
        "tool_ids": tool_ids,
        "tools": [t["tool"] for t in tools],
        "turns": turns,
        "solver_chat": [m for m in solver_messages if m["role"] != "system"],
        "usage": usage_tracker.total,
        "generation_time_s": round(elapsed, 2),
    }

    output_path = output_dir / f"{task_id}.json"
    write_json_atomic(task, output_path)
    logger.info(f"Task saved: {output_path}")

    if event_log:
        debug_path = event_log.save(output_dir)
        logger.info(f"Debug log saved: {debug_path}")

    totals = usage_tracker.total
    logger.info(f"Total usage: prompt={totals['prompt_tokens']}, completion={totals['completion_tokens']}, total={totals['total_tokens']}")
    logger.info(f"Generation time: {elapsed:.1f}s")
    return task


# ---------------------------------------------------------------------------
# Work-item listing (resume-safe pre-filter)
# ---------------------------------------------------------------------------

@dataclass
class WorkItem:
    """One task-to-generate, fully resolved.

    Produced by `list_pending_work_for_spec` / `list_pending_work_for_field`
    after pre-filtering already-finished outputs and skipping invalid
    sequences. Picklable so it can be sent across `ProcessPoolExecutor`.
    """
    spec_id: str
    seq_key: str
    task_id: str
    tool_ids: List[str]
    out_path: Path


def list_pending_work_for_spec(
    env_spec_path: Path,
    output_dir: Path,
    max_tasks: Optional[int] = None,
) -> List[WorkItem]:
    """Return WorkItems for every sequence in the spec that still needs to run.

    Filters out:
      - sequences whose `{spec_id}_{seq_key}.json` already exists in output_dir
      - non-list / empty sequence values

    `max_tasks` is a TOTAL cap (existing on disk + new) — once the spec
    has that many tasks committed, no new work is yielded. Idempotent
    across re-runs.
    """
    env_spec_path = Path(env_spec_path)
    output_dir = Path(output_dir)
    with open(env_spec_path) as f:
        spec = json.load(f)

    spec_id = spec["spec_id"]
    sequences = spec.get("sequences") or {}
    if not sequences:
        logger.warning(f"{spec_id}: no sequences populated — run build_sequences first")
        return []

    existing = sum(
        1 for k in sequences if (output_dir / f"{spec_id}_{k}.json").exists()
    )
    budget = (max_tasks - existing) if max_tasks is not None else None
    if budget is not None and budget <= 0:
        return []

    items: List[WorkItem] = []
    for seq_key, seq in sequences.items():
        if budget is not None and len(items) >= budget:
            break
        task_id = f"{spec_id}_{seq_key}"
        out_path = output_dir / f"{task_id}.json"
        if out_path.exists():
            logger.info(f"  {task_id}: already exists, skipping")
            continue
        if not isinstance(seq, list) or not seq:
            logger.warning(f"  {task_id}: invalid sequence, skipping")
            continue
        tool_ids = [f"{spec_id}.{name}" for name in seq]
        items.append(WorkItem(
            spec_id=spec_id,
            seq_key=seq_key,
            task_id=task_id,
            tool_ids=tool_ids,
            out_path=out_path,
        ))
    return items


def list_pending_work_for_field(
    env_specs_dir: Path,
    field: str,
    output_dir: Path,
    max_tasks_per_spec: Optional[int] = None,
) -> List[WorkItem]:
    """Walk env_specs_dir, gather pending WorkItems for every spec matching `field`."""
    env_specs_dir = Path(env_specs_dir)
    output_dir = Path(output_dir)

    matched: List[Path] = []
    for p in sorted(env_specs_dir.glob("*_spec_*.json")):
        if p.name.endswith(".tmp"):
            continue
        try:
            with open(p) as f:
                spec = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if spec.get("field") == field:
            matched.append(p)

    logger.info(f"Field '{field}': {len(matched)} spec(s) match")
    per_spec: List[List[WorkItem]] = [
        list_pending_work_for_spec(
            env_spec_path=p,
            output_dir=output_dir,
            max_tasks=max_tasks_per_spec,
        )
        for p in matched
    ]
    from itertools import zip_longest
    items: List[WorkItem] = [
        w for layer in zip_longest(*per_spec) for w in layer if w is not None
    ]
    return items


# ---------------------------------------------------------------------------
# Mode B: all sequences in one env_spec
# ---------------------------------------------------------------------------

def generate_trajectories_for_spec(
    env_spec_path: Path,
    tools_dataset_path: Path,
    output_dir: Path,
    llm,
    max_tasks: Optional[int] = None,
    **task_kwargs,
) -> List[Dict]:
    """Run one task per sequence in the env_spec.

    Files are named `{spec_id}_{seq_key}.json`. If that file already exists in
    output_dir, the (spec_id, seq_key) pair is skipped — supports resuming
    interrupted runs without rework.
    """
    env_spec_path = Path(env_spec_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = list_pending_work_for_spec(env_spec_path, output_dir, max_tasks)
    if not items:
        return []
    spec_id = items[0].spec_id
    logger.info(f"{spec_id}: {len(items)} sequence(s) to process")

    tasks: List[Dict] = []
    for item in items:
        logger.info(f"  {item.task_id}: {len(item.tool_ids)} tools")
        task = generate_trajectory(
            tool_ids=item.tool_ids,
            tools_dataset_path=tools_dataset_path,
            output_dir=output_dir,
            llm=llm,
            task_id=item.task_id,
            **task_kwargs,
        )
        tasks.append(task)
    return tasks


# ---------------------------------------------------------------------------
# Mode C: all sequences across every spec in a field
# ---------------------------------------------------------------------------

def generate_trajectories_for_field(
    env_specs_dir: Path,
    field: str,
    tools_dataset_path: Path,
    output_dir: Path,
    llm,
    max_tasks_per_spec: Optional[int] = None,
    **task_kwargs,
) -> List[Dict]:
    """Walk env_specs_dir, dispatch `generate_trajectories_for_spec` per matching spec."""
    env_specs_dir = Path(env_specs_dir)
    output_dir = Path(output_dir)

    matched: List[Path] = []
    for p in sorted(env_specs_dir.glob("*_spec_*.json")):
        if p.name.endswith(".tmp"):
            continue
        try:
            with open(p) as f:
                spec = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if spec.get("field") == field:
            matched.append(p)

    logger.info(f"Field '{field}': {len(matched)} spec(s) match")
    tasks: List[Dict] = []
    for p in matched:
        tasks.extend(generate_trajectories_for_spec(
            env_spec_path=p,
            tools_dataset_path=tools_dataset_path,
            output_dir=output_dir,
            llm=llm,
            max_tasks=max_tasks_per_spec,
            **task_kwargs,
        ))
    return tasks
