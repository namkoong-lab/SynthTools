"""CLI entry point for environment generation.

Usage:
    python -m env_generation.run \
        --fields "Aerospace and Defense,Healthcare" \
        --output-dir tool_content/env_specs \
        --model GPT-OSS-120B \
        [--max-subfields 1] [--max-tasks-per-subfield 1]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_MODEL
from llm import LLM, MODEL_REGISTRY
from env_generation.generate import generate_environments


def main():
    parser = argparse.ArgumentParser(description="Generate environment specs for one or more fields.")
    parser.add_argument("--fields", required=True,
                        help="Comma-separated field names (e.g. 'Aerospace and Defense,Healthcare')")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for JSON spec files")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODEL_REGISTRY))
    parser.add_argument("--max-subfields", type=int, default=1)
    parser.add_argument("--max-tasks-per-subfield", type=int, default=1)
    args = parser.parse_args()

    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    if not fields:
        parser.error("At least one field name is required")

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
