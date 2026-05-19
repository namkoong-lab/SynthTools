"""CLI entry point for task_audit.

Usage:
    python -m task_audit.run \\
        --tasks-dir /pscratch/.../tasks \\
        --model GPT-OSS-120B \\
        [--task PATH | --field NAME --env-specs-dir DIR]
        [--server-url http://...]    # use vLLM HTTP server
        [--no-debug]                  # skip .debug.json append

The summariser mutates each task JSON in place, adding a `summary` block.
Skips tasks that already have a truthy `summary` field (resume-safe).
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
    add_model_arg(parser)

    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--task", type=Path,
        help="Path to a single task JSON",
    )
    target.add_argument(
        "--field", type=str,
        help="Field name; requires --env-specs-dir",
    )

    parser.add_argument(
        "--env-specs-dir", type=Path,
        help="env_specs directory — required with --field",
    )
    add_server_url_arg(parser)
    parser.add_argument(
        "--no-debug", dest="write_debug", action="store_false",
        help="Skip appending to sibling .debug.json files",
    )
    parser.set_defaults(write_debug=True)

    args = parser.parse_args()

    if args.field and args.env_specs_dir is None:
        parser.error("--field requires --env-specs-dir")

    llm = LLM(args.model, server_url=args.server_url)
    summarize_trajectories(
        tasks_dir=args.tasks_dir,
        llm=llm,
        task_path=args.task,
        field=args.field,
        env_specs_dir=args.env_specs_dir,
        write_debug=args.write_debug,
    )


if __name__ == "__main__":
    main()
