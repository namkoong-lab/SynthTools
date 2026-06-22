"""CLI entry point for task generation.

Three mutually-exclusive input modes:

    Mode A — explicit list of tools (legacy):
        python -m task_generation.run \
            --dataset tool_content/tools_dataset.jsonl \
            --output-dir tool_content/tasks \
            --model GPT-OSS-120B \
            --tool-ids "id1" "id2" "id3"

    Mode B — every sequence in one env_spec:
        python -m task_generation.run \
            --dataset tool_content/tools_dataset.jsonl \
            --output-dir tool_content/tasks \
            --model GPT-OSS-120B \
            --env-spec tool_content/env_specs/aerospace_and_defense_spec_000.json \
            [--max-tasks 5]

    Mode C — every sequence of every spec in a field:
        python -m task_generation.run \
            --dataset tool_content/tools_dataset.jsonl \
            --output-dir tool_content/tasks \
            --model GPT-OSS-120B \
            --env-specs-dir tool_content/env_specs \
            --field "Aerospace and Defense"

For modes B/C the output filename is `{spec_id}_{seq_key}.json`; if that file
already exists the (spec_id, seq_key) is skipped — so runs can be resumed.

For parallel generation against a `vllm serve` process, add:
    --server-url http://localhost:8765/v1
    --concurrency N
Each worker is its own OS process with its own HTTP client; all workers
share the single vLLM server.

`task_generation.run` does not generate sequences. Run
`task_generation.build_sequences` first to populate the `sequences` block on
each env_spec.
"""

import argparse
import logging
from functools import partial
from pathlib import Path
from typing import Any, Dict, List

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli_args import add_model_arg, add_server_url_arg
from config import (
    DEFAULT_CONCURRENCY,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_SOLVER_RETRIES_IN_PLACE,
    DEFAULT_MAX_SOLVER_TURNS,
)
from llm import LLM
from task_generation.generate import (
    WorkItem,
    generate_trajectory,
    generate_trajectories_for_spec,
    generate_trajectories_for_field,
    list_pending_work_for_spec,
    list_pending_work_for_field,
)
from utils import get_logger, redirect_synthtools_logger_to_file, run_parallel

logger = get_logger("task_generation.run")


# ---------------------------------------------------------------------------
# Worker (top-level so it pickles cleanly across ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _worker(item: WorkItem, llm_kwargs: Dict[str, Any], task_kwargs: Dict[str, Any],
            tools_dataset_path: str, output_dir: str) -> Dict[str, Any]:
    """Generate one task in a child process.

    First thing the worker does: redirect the `synthtools` logger to a
    per-task file at `<output_dir>/_logs/<task_id>.log` so detailed logs
    don't interleave on the parent's stderr. The parent's orchestration
    logger (`task_generation.run`) continues to write to stderr.

    Each worker constructs its own LLM (cheap when `server_url` is set —
    just an HTTP client), then calls `generate_trajectory`.
    """
    log_path = Path(output_dir) / "_logs" / f"{item.task_id}.log"
    redirect_synthtools_logger_to_file(log_path)

    llm = LLM(**llm_kwargs)
    task = generate_trajectory(
        tool_ids=item.tool_ids,
        tools_dataset_path=Path(tools_dataset_path),
        output_dir=Path(output_dir),
        llm=llm,
        task_id=item.task_id,
        **task_kwargs,
    )
    return {
        "task_id": item.task_id,
        "log_path": str(log_path),
        "usage": task.get("usage"),
        "generation_time_s": task.get("generation_time_s"),
    }


def _run_parallel(items: List[WorkItem], llm_kwargs: Dict[str, Any], task_kwargs: Dict[str, Any],
                  tools_dataset_path: Path, output_dir: Path, concurrency: int) -> List[Dict[str, Any]]:
    """Dispatch all WorkItems to a process pool via utils.run_parallel."""
    if not items:
        logger.info("No pending work — all tasks already exist.")
        return []
    log_dir = Path(output_dir) / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Submitting {len(items)} tasks to a pool of {concurrency} workers.")
    logger.info(f"Per-worker logs: {log_dir}/<task_id>.log  (tail -f any to follow)")

    bound_worker = partial(
        _worker,
        llm_kwargs=llm_kwargs,
        task_kwargs=task_kwargs,
        tools_dataset_path=str(tools_dataset_path),
        output_dir=str(output_dir),
    )

    def _log(done: int, total: int, r: Dict[str, Any]) -> None:
        if "error" in r:
            logger.error(f"[{done}/{total}] worker failed: {r['error']}")
        else:
            logger.info(
                f"[{done}/{total}] done {r['task_id']} in {r.get('generation_time_s')}s"
            )

    results = run_parallel(items, bound_worker, concurrency, on_result=_log)
    n_ok = sum(1 for r in results if "error" not in r)
    logger.info(f"Pool finished: {n_ok}/{len(items)} succeeded.")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate tasks from tool sequences.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to tools_dataset.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    add_model_arg(parser)
    parser.add_argument("--max-solver-turns", type=int, default=DEFAULT_MAX_SOLVER_TURNS,
                        help="Solver turns inside a single in-place attempt.")
    parser.add_argument("--max-solver-retries-in-place", type=int,
                        default=DEFAULT_MAX_SOLVER_RETRIES_IN_PLACE,
                        help="In-place solver retry budget per attempt. "
                             "Each retry runs up to --max-solver-turns turns; "
                             "if it still fails the outer --max-retries reroll kicks in.")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                        help="Outer evolver-reroll budget per (spec, sequence) attempt.")
    parser.add_argument("--verifiable", action="store_true", help="Enable task judging")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug event log")
    add_server_url_arg(parser)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help="Number of tasks to generate in parallel. Requires --server-url "
                             "when > 1 (multiple in-process vLLM engines would OOM the GPUs).")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tool-ids", nargs="+", help="Mode A: explicit ordered list of tool IDs")
    mode.add_argument("--env-spec", type=Path, help="Mode B: path to a single env_spec JSON")
    mode.add_argument("--field", type=str, help="Mode C: field name (requires --env-specs-dir)")

    parser.add_argument("--env-specs-dir", type=Path,
                        help="Directory containing env_spec JSONs (required with --field)")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="(Mode B/C) Cap the number of tasks per spec")

    args = parser.parse_args()

    if args.field and args.env_specs_dir is None:
        parser.error("--field requires --env-specs-dir")
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    if args.concurrency > 1 and not args.server_url:
        parser.error("--concurrency > 1 requires --server-url (in-process vLLM cannot be forked).")
    if args.concurrency > 1 and args.tool_ids:
        parser.error("--concurrency > 1 is not supported with --tool-ids (single task).")

    task_kwargs = dict(
        max_solver_turns=args.max_solver_turns,
        max_solver_retries_in_place=args.max_solver_retries_in_place,
        max_retries=args.max_retries,
        verifiable=args.verifiable,
        debug=not args.no_debug,
    )

    # LLM kwargs — used for both the controller's LLM (concurrency=1) and the
    # workers' LLMs (concurrency>1). server_url=None falls through to the
    # in-process vLLM path.
    llm_kwargs = dict(model=args.model, server_url=args.server_url)

    if args.concurrency == 1:
        # Serial path — unchanged from before. Keeps existing tests / behavior.
        llm = LLM(**llm_kwargs)
        if args.tool_ids:
            generate_trajectory(
                tool_ids=args.tool_ids,
                tools_dataset_path=args.dataset,
                output_dir=args.output_dir,
                llm=llm,
                **task_kwargs,
            )
        elif args.env_spec:
            generate_trajectories_for_spec(
                env_spec_path=args.env_spec,
                tools_dataset_path=args.dataset,
                output_dir=args.output_dir,
                llm=llm,
                max_tasks=args.max_tasks,
                **task_kwargs,
            )
        else:
            generate_trajectories_for_field(
                env_specs_dir=args.env_specs_dir,
                field=args.field,
                tools_dataset_path=args.dataset,
                output_dir=args.output_dir,
                llm=llm,
                max_tasks_per_spec=args.max_tasks,
                **task_kwargs,
            )
        return

    # Parallel path — build flat work list, dispatch to ProcessPoolExecutor.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.env_spec:
        items = list_pending_work_for_spec(
            env_spec_path=args.env_spec,
            output_dir=args.output_dir,
            max_tasks=args.max_tasks,
        )
    else:
        items = list_pending_work_for_field(
            env_specs_dir=args.env_specs_dir,
            field=args.field,
            output_dir=args.output_dir,
            max_tasks_per_spec=args.max_tasks,
        )
    _run_parallel(
        items=items,
        llm_kwargs=llm_kwargs,
        task_kwargs=task_kwargs,
        tools_dataset_path=args.dataset,
        output_dir=args.output_dir,
        concurrency=args.concurrency,
    )


if __name__ == "__main__":
    main()
