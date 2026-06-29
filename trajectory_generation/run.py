"""CLI entry point for trajectory_generation.

Loads tasks from `task_content.jsonl` (produced by `task_audit.run`), runs an
agent end-to-end against the tool simulator, judges the rollout, and writes
one trajectory JSON per task.

Three input modes (mutually exclusive):

  --task-id ID                     # one task by id
  --field "Investment Banking"     # every task in a field, optionally capped by --limit
  (default)                        # walk every task in the JSONL, optionally capped by --limit

Examples:
  python -m trajectory_generation.run \\
      --task-id aerospace_and_defense_spec_007_seq11 \\
      --dataset /pscratch/.../tool_content/task_content.jsonl \\
      --output-dir /tmp/traj_smoke

  python -m trajectory_generation.run \\
      --field "Investment Banking" --limit 50 \\
      --dataset /pscratch/.../tool_content/task_content.jsonl \\
      --server-url http://localhost:8765/v1 --concurrency 4 \\
      --output-dir /path/to/data/trajectories_rollout

Trajectory generation is resume-safe: if `<output-dir>/{task_id}.json` already
exists, that task is skipped.
"""

from __future__ import annotations

import argparse
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make sibling packages importable when running as `python -m trajectory_generation.run`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli_args import add_model_arg, add_server_url_arg                  # noqa: E402
from config import DEFAULT_CONCURRENCY                                  # noqa: E402
from llm import LLM                                                # noqa: E402
from trajectory_generation.loader import (                             # noqa: E402
    Task, iter_tasks, load_task, list_task_ids, select_random_ids,
)
from trajectory_generation.orchestrator import generate_trajectory     # noqa: E402
from utils import get_logger, run_parallel                          # noqa: E402

logger = get_logger("synthtools.trajectory_generation.run")

# Default JSONL path — kept symmetrical with task_audit's default
# (<tasks-dir>/../task_content.jsonl).
DEFAULT_DATASET = Path("/pscratch/sd/t/tcaste/tool_content/task_content.jsonl")


# --- worker for parallel mode ---

def _worker(
    task: Task,
    output_dir: str,
    max_solver_turns: int,
    debug: bool,
    run_judge: bool,
    llm_kwargs: Dict[str, Any],
    rollout_tag: str = "",
) -> Dict[str, Any]:
    out = Path(output_dir)
    if (out / f"{task.id}{rollout_tag}.json").exists():
        return {"task_id": task.id, "status": "skipped"}
    llm = LLM(**llm_kwargs)
    try:
        traj = generate_trajectory(
            task=task,
            llm=llm,
            output_dir=out,
            max_solver_turns=max_solver_turns,
            debug=debug,
            run_judge=run_judge,
            rollout_tag=rollout_tag,
        )
        verdict = (traj.get("trajectory_judge") or {}).get("trajectory_solved")
        return {"task_id": task.id, "status": "ok", "trajectory_solved": verdict}
    except Exception as exc:    # pragma: no cover — surfaced in the parent
        return {"task_id": task.id, "status": "error", "error": f"{type(exc).__name__}: {exc}"}


def _run_parallel(
    tasks: List[Task],
    output_dir: Path,
    max_solver_turns: int,
    debug: bool,
    run_judge: bool,
    llm_kwargs: Dict[str, Any],
    concurrency: int,
    rollout_tag: str = "",
) -> None:
    work = partial(
        _worker,
        output_dir=str(output_dir),
        max_solver_turns=max_solver_turns,
        debug=debug,
        run_judge=run_judge,
        llm_kwargs=llm_kwargs,
        rollout_tag=rollout_tag,
    )
    logger.info(f"dispatching {len(tasks)} tasks to {concurrency} workers")

    def _log(done: int, total: int, r: Dict[str, Any]) -> None:
        # Log every 10th success, and EVERY non-ok result.
        if done % 10 == 0 or r.get("status") != "ok":
            logger.info(f"[{done}/{total}] {r}")

    run_parallel(tasks, work, concurrency, on_result=_log)


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help=f"JSONL of tasks (default: {DEFAULT_DATASET})")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where trajectory JSONs (and debug logs) are written.")
    add_model_arg(parser)
    parser.add_argument("--max-solver-turns", type=int, default=12,
                        help="Cap on the agent's loop (default 12).")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature override (default: registry 0.2). "
                             "gpt-oss spec is 1.0 for diverse rollouts.")
    parser.add_argument("--top-p", type=float, default=None,
                        help="top_p override (default: registry 0.95). gpt-oss spec is 1.0.")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip the TrajectoryJudge pass.")
    parser.add_argument("--no-debug", action="store_true",
                        help="Skip writing the per-LLM-call event log.")
    add_server_url_arg(parser)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help="Number of tasks to roll out in parallel. Requires --server-url > 1.")

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--task-id", type=str,
                           help="Run a single task by id.")
    selection.add_argument("--field", type=str,
                           help="Run every task whose field equals this value.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of tasks (with --field or default mode).")
    parser.add_argument("--sample", type=int, default=None,
                        help="Draw this many tasks at random (requires --seed for "
                             "reproducibility; distinct seeds draw distinct subsets).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for --sample, and the source of the rollout tag "
                             "'.s<seed>' appended to output filenames so the same task "
                             "can be rolled out under multiple seeds without collision.")
    parser.add_argument("--rollout-tag", type=str, default=None,
                        help="Override the output-filename tag (default: '.s<seed>' when "
                             "--seed is set, else empty).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve and print the selected task ids, then exit "
                             "(no vLLM, no rollout). Useful for testing selection.")

    args = parser.parse_args()

    # Validate
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    if args.concurrency > 1 and not args.server_url and not args.dry_run:
        parser.error("--concurrency > 1 requires --server-url (in-process vLLM cannot be forked).")
    if args.concurrency > 1 and args.task_id:
        parser.error("--concurrency > 1 is not supported with a single --task-id.")
    if args.sample is not None and args.task_id:
        parser.error("--sample cannot be combined with --task-id.")
    if args.sample is not None and args.seed is None:
        parser.error("--sample requires --seed (for reproducible, non-overlapping draws).")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.dataset.exists():
        parser.error(
            f"--dataset {args.dataset} does not exist. Produce it by running "
            f"task_audit (writes <tasks-dir>/../task_content.jsonl), or pass "
            f"--dataset explicitly."
        )

    # The rollout tag keeps multiple seeds of the same task from colliding.
    if args.rollout_tag is not None:
        rollout_tag = args.rollout_tag
    elif args.seed is not None:
        rollout_tag = f".s{args.seed}"
    else:
        rollout_tag = ""

    # Resolve the task set. A seed switches on random mode: the whole
    # (field-filtered) id list is shuffled by the seed and processed in that
    # order, optionally capped by --sample. Distinct seeds -> distinct, only
    # incidentally-overlapping orders, so 50 independent nodes cover the
    # dataset at random and the same task gets multiple rollouts across seeds.
    if args.task_id:
        tasks: List[Task] = [load_task(args.dataset, args.task_id)]
    elif args.seed is not None:
        all_ids = list_task_ids(args.dataset, field=args.field)
        chosen = select_random_ids(all_ids, seed=args.seed, sample=args.sample)
        logger.info(f"random mode: {len(chosen)}/{len(all_ids)} task ids "
                    f"(seed={args.seed}, sample={args.sample}, tag='{rollout_tag}')")
        # iter_tasks yields JSONL order; re-order to the shuffled `chosen` order.
        by_id = {t.id: t for t in iter_tasks(args.dataset, ids=chosen)}
        tasks = [by_id[i] for i in chosen if i in by_id]
    else:
        tasks = list(iter_tasks(args.dataset, field=args.field, limit=args.limit))
    if not tasks:
        logger.warning("no tasks matched the selection — nothing to do")
        return

    if args.dry_run:
        ids = [t.id for t in tasks]
        logger.info(f"[dry-run] {len(ids)} task(s) selected; tag='{rollout_tag}'")
        for tid in ids[:20]:
            logger.info(f"[dry-run]   {tid}{rollout_tag}")
        if len(ids) > 20:
            logger.info(f"[dry-run]   ... and {len(ids) - 20} more")
        return

    llm_kwargs = dict(model=args.model)
    if args.server_url:
        llm_kwargs["server_url"] = args.server_url
    if args.temperature is not None:
        llm_kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        llm_kwargs["top_p"] = args.top_p

    if args.concurrency == 1:
        # Serial path — single LLM, single trajectory at a time.
        llm = LLM(**llm_kwargs)
        for i, task in enumerate(tasks, 1):
            out_path = args.output_dir / f"{task.id}{rollout_tag}.json"
            if out_path.exists():
                logger.info(f"[{i}/{len(tasks)}] {task.id}{rollout_tag}: already exists — skipping")
                continue
            logger.info(f"[{i}/{len(tasks)}] generating {task.id}{rollout_tag}")
            generate_trajectory(
                task=task,
                llm=llm,
                output_dir=args.output_dir,
                max_solver_turns=args.max_solver_turns,
                debug=not args.no_debug,
                run_judge=not args.no_judge,
                rollout_tag=rollout_tag,
            )
        return

    # Parallel path
    _run_parallel(
        tasks=tasks,
        output_dir=args.output_dir,
        max_solver_turns=args.max_solver_turns,
        debug=not args.no_debug,
        run_judge=not args.no_judge,
        llm_kwargs=llm_kwargs,
        concurrency=args.concurrency,
        rollout_tag=rollout_tag,
    )


if __name__ == "__main__":
    main()
