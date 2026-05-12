"""CLI entry point for trajectory_generation.

Loads tasks from `tasks.parquet`, runs an agent end-to-end against the tool
simulator, judges the rollout, and writes one trajectory JSON per task.

Three input modes (mutually exclusive):

  --task-id ID                     # one task by id
  --field "Investment Banking"     # every task in a field, optionally capped by --limit
  (default)                        # walk every task in the parquet, optionally capped by --limit

Examples:
  python -m trajectory_generation.run \\
      --task-id aerospace_and_defense_spec_007_seq11 \\
      --output-dir /tmp/traj_smoke

  python -m trajectory_generation.run \\
      --field "Investment Banking" --limit 50 \\
      --server-url http://localhost:8765/v1 --concurrency 4 \\
      --output-dir /path/to/data/trajectories_rollout

Trajectory generation is resume-safe: if `<output-dir>/{task_id}.json` already
exists, that task is skipped.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make synthtools_nips26 importable when running as `python -m trajectory_generation.run`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm import LLM, MODEL_REGISTRY                                # noqa: E402
from trajectory_generation.loader import Task, iter_tasks, load_task   # noqa: E402
from trajectory_generation.orchestrator import generate_trajectory     # noqa: E402
from utils import get_logger                                       # noqa: E402

logger = get_logger("synthtools.trajectory_generation.run")

# The parquet ships at the repo root; this is the default --dataset.
DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "tasks.parquet"

# Hugging Face fallback used by `_ensure_dataset_present` when the local
# `--dataset` path does not exist.
HF_DATASET_REPO = "SynthTools/SynthTools-Tasks"
HF_DATASET_FILE = "tasks.parquet"


def _ensure_dataset_present(path: Path) -> Path:
    """If `path` is missing, download the parquet from the SynthTools dataset
    repository on Hugging Face into `path`. Returns the resolved local path.
    """
    path = Path(path)
    if path.exists():
        return path
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            f"{path} is missing and `huggingface_hub` is not installed; "
            "either install it (`uv pip install huggingface_hub`) or place "
            "tasks.parquet at the path manually."
        ) from exc
    logger.info(f"{path} not found locally — downloading from "
                f"https://huggingface.co/datasets/{HF_DATASET_REPO}")
    path.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=HF_DATASET_FILE,
        repo_type="dataset",
    )
    # The HF cache stores files under its own tree; copy into the requested
    # location so subsequent runs find it without hitting the network again.
    import shutil
    shutil.copy2(cached, path)
    logger.info(f"saved dataset to {path}")
    return path


# --- worker for parallel mode ---

def _worker(
    task: Task,
    output_dir: str,
    max_solver_turns: int,
    debug: bool,
    run_judge: bool,
    llm_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    out = Path(output_dir)
    if (out / f"{task.id}.json").exists():
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
) -> None:
    work = partial(
        _worker,
        output_dir=str(output_dir),
        max_solver_turns=max_solver_turns,
        debug=debug,
        run_judge=run_judge,
        llm_kwargs=llm_kwargs,
    )
    n = len(tasks)
    logger.info(f"dispatching {n} tasks to {concurrency} workers")
    with ProcessPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(work, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if i % 10 == 0 or r.get("status") != "ok":
                logger.info(f"[{i}/{n}] {r}")


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help=f"Parquet of tasks (default: {DEFAULT_DATASET})")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where trajectory JSONs (and debug logs) are written.")
    parser.add_argument("--model", default="Qwen3-32B", choices=list(MODEL_REGISTRY))
    parser.add_argument("--max-solver-turns", type=int, default=12,
                        help="Cap on the agent's loop (default 12).")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip the TrajectoryJudge pass.")
    parser.add_argument("--no-debug", action="store_true",
                        help="Skip writing the per-LLM-call event log.")
    parser.add_argument("--server-url", type=str, default=None,
                        help="OpenAI-compatible base URL (e.g. http://localhost:8765/v1). "
                             "Required for --concurrency > 1.")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of tasks to roll out in parallel. Requires --server-url > 1.")

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--task-id", type=str,
                           help="Run a single task by id.")
    selection.add_argument("--field", type=str,
                           help="Run every task whose field equals this value.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of tasks (with --field or default mode).")

    args = parser.parse_args()

    # Validate
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    if args.concurrency > 1 and not args.server_url:
        parser.error("--concurrency > 1 requires --server-url (in-process vLLM cannot be forked).")
    if args.concurrency > 1 and args.task_id:
        parser.error("--concurrency > 1 is not supported with a single --task-id.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-download tasks.parquet from Hugging Face if the local file is missing.
    args.dataset = _ensure_dataset_present(args.dataset)

    # Resolve the task set
    if args.task_id:
        tasks: List[Task] = [load_task(args.dataset, args.task_id)]
    else:
        tasks = list(iter_tasks(args.dataset, field=args.field, limit=args.limit))
    if not tasks:
        logger.warning("no tasks matched the selection — nothing to do")
        return

    llm_kwargs = dict(model=args.model)
    if args.server_url:
        llm_kwargs["server_url"] = args.server_url

    if args.concurrency == 1:
        # Serial path — single LLM, single trajectory at a time.
        llm = LLM(**llm_kwargs)
        for i, task in enumerate(tasks, 1):
            out_path = args.output_dir / f"{task.id}.json"
            if out_path.exists():
                logger.info(f"[{i}/{len(tasks)}] {task.id}: already exists — skipping")
                continue
            logger.info(f"[{i}/{len(tasks)}] generating {task.id}")
            generate_trajectory(
                task=task,
                llm=llm,
                output_dir=args.output_dir,
                max_solver_turns=args.max_solver_turns,
                debug=not args.no_debug,
                run_judge=not args.no_judge,
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
    )


if __name__ == "__main__":
    main()
