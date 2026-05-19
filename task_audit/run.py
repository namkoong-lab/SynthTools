"""CLI entry point for task_audit.

Usage:
    python -m task_audit.run \\
        --tasks-dir     /pscratch/.../tasks \\
        --env-specs-dir /pscratch/.../env_specs \\
        --model GPT-OSS-120B \\
        [--task PATH | --field NAME]
        [--task-content PATH]                  # default: tasks_dir.parent/task_content.jsonl
        [--resummarize]                        # ignore existing summaries, redo
        [--shard I --num-shards N]             # split work across N concurrent jobs
        [--server-url http://...]              # vLLM HTTP server
        [--no-debug]                           # skip .debug.json append

The summariser mutates each task JSON in place, adding a `summary` block,
AND appends one release row per in-scope task to the SHARED JSONL at
`--task-content`. Multiple concurrent jobs (sharded via --shard/--num-shards)
all append to the same file under an exclusive file lock — no per-shard
files, no merge step. Skips tasks that already have a truthy `summary`
field unless --resummarize is passed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sibling packages importable when run as `python -m task_audit.run`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli_args import add_model_arg, add_server_url_arg
from llm import LLM
from task_audit.summarize import summarize_trajectories


def main():
    parser = argparse.ArgumentParser(
        description="Summarize tasks produced by task_generation.run."
    )
    parser.add_argument(
        "--tasks-dir", type=Path, required=True,
        help="Directory containing task JSONs",
    )
    parser.add_argument(
        "--env-specs-dir", type=Path, required=True,
        help="env_specs directory (always required: supplies `field` and "
             "`tools` columns for the release JSONL)",
    )
    add_model_arg(parser)

    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--task", type=Path,
        help="Path to a single task JSON",
    )
    target.add_argument(
        "--field", type=str,
        help="Field name (filters tasks via env_specs lookup)",
    )

    parser.add_argument(
        "--task-content", type=Path, default=None,
        help="Path to the shared release JSONL "
             "(default: <tasks-dir>/../task_content.jsonl)",
    )
    parser.add_argument(
        "--resummarize", action="store_true",
        help="Ignore existing `summary` blocks and re-run the LLM for every "
             "in-scope task. Release rows are appended regardless of whether "
             "the id is already in the JSONL (consumers dedup by id, "
             "last-write-wins). Default: skip tasks with existing summaries.",
    )
    parser.add_argument(
        "--shard", type=int, default=0,
        help="Shard index for this process (0-based). Use with --num-shards "
             "to run multiple concurrent jobs against the same output JSONL.",
    )
    parser.add_argument(
        "--num-shards", type=int, default=1,
        help="Total number of concurrent shards (default 1, no sharding). "
             "Each shard processes target_paths[shard::num_shards] of the "
             "deterministically-sorted in-scope file list.",
    )
    add_server_url_arg(parser)
    parser.add_argument(
        "--no-debug", dest="write_debug", action="store_false",
        help="Skip appending to sibling .debug.json files",
    )
    parser.set_defaults(write_debug=True)

    args = parser.parse_args()

    if not (0 <= args.shard < args.num_shards):
        parser.error(f"--shard must be in [0, --num-shards) — got shard={args.shard}, num_shards={args.num_shards}")

    task_content_path = args.task_content or (args.tasks_dir.parent / "task_content.jsonl")

    llm = LLM(args.model, server_url=args.server_url)
    summarize_trajectories(
        tasks_dir=args.tasks_dir,
        llm=llm,
        env_specs_dir=args.env_specs_dir,
        task_content_path=task_content_path,
        task_path=args.task,
        field=args.field,
        write_debug=args.write_debug,
        resummarize=args.resummarize,
        shard=args.shard,
        num_shards=args.num_shards,
    )


if __name__ == "__main__":
    main()
