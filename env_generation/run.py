"""CLI entry point for environment generation.

Usage:
    python -m env_generation.run \
        --field "Aerospace and Defense" --field "Healthcare" \
        --output-dir tool_content/env_specs \
        --model GPT-OSS-120B \
        [--max-subfields 1] [--max-tasks-per-subfield 1]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli_args import add_model_arg
from llm import LLM
from env_generation.generate import generate_environments


def main():
    parser = argparse.ArgumentParser(description="Generate environment specs for one or more fields.")
    parser.add_argument("--field", action="append", required=True, dest="fields",
                        help="Field name (repeatable: --field 'Aerospace and Defense' --field 'Healthcare')")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for JSON spec files")
    add_model_arg(parser)
    parser.add_argument("--max-subfields", type=int, default=1)
    parser.add_argument("--max-tasks-per-subfield", type=int, default=1)
    args = parser.parse_args()

    fields = [f.strip() for f in (args.fields or []) if f and f.strip()]
    if not fields:
        parser.error("At least one --field is required")

    llm = LLM(args.model)
    generate_environments(
        fields=fields,
        output_dir=args.output_dir,
        llm=llm,
        max_subfields=args.max_subfields,
        max_tasks_per_subfield=args.max_tasks_per_subfield,
    )


if __name__ == "__main__":
    main()
