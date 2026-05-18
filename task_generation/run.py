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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_CONCURRENCY, DEFAULT_MAX_RETRIES, DEFAULT_MAX_SOLVER_TURNS, DEFAULT_MODEL
from llm import LLM, MODEL_REGISTRY
from task_generation.generate import (
    WorkItem,
    generate_trajectory,
    generate_trajectories_for_spec,
    generate_trajectories_for_field,
    list_pending_work_for_spec,
    list_pending_work_for_field,
)
from utils import get_logger, redirect_synthtools_logger_to_file

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
    """Submit all WorkItems to a ProcessPoolExecutor; drain results as they complete."""
    if not items:
        logger.info("No pending work — all tasks already exist.")
        return []
    log_dir = Path(output_dir) / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Submitting {len(items)} tasks to a pool of {concurrency} workers.")
    logger.info(f"Per-worker logs: {log_dir}/<task_id>.log  (tail -f any to follow)")
    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=concurrency) as ex:
        futures = {
            ex.submit(_worker, item, llm_kwargs, task_kwargs,
                      str(tools_dataset_path), str(output_dir)): item
            for item in items
        }
        for fut in as_completed(futures):
            item = futures[fut]
            try:
                summary = fut.result()
                results.append(summary)
                logger.info(
                    f"[{len(results)}/{len(items)}] done {summary['task_id']} "
                    f"in {summary.get('generation_time_s')}s"
                )
            except Exception as e:
                logger.error(f"Worker failed for {item.task_id}: {type(e).__name__}: {e}")
    logger.info(f"Pool finished: {len(results)}/{len(items)} succeeded.")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate tasks from tool sequences.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to tools_dataset.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODEL_REGISTRY))
    parser.add_argument("--max-solver-turns", type=int, default=DEFAULT_MAX_SOLVER_TURNS)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--verifiable", action="store_true", help="Enable task judging")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug event log")
    parser.add_argument("--server-url", type=str, default=None,
                        help="OpenAI-compatible base URL (e.g. http://localhost:8765/v1). "
                             "When set, all LLM calls go over HTTP instead of loading vLLM in-process.")
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
