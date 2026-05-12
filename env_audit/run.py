"""CLI entry point for environment audit.

One CLI runs both stages (build → evaluate). Filters:
  --fields  applies to BOTH build (skip specs with non-matching field) AND evaluate.
  --ids     applies to evaluate only (intersected with --fields if both given).

Usage:
    python -m env_audit.run \
        --env-specs-dir tool_content/env_specs \
        --dataset-path tool_content/tools_dataset.jsonl \
        --eval-logs-dir tool_content/tool_eval_logs \
        --model GPT-OSS-120B \
        [--fields "Aerospace and Defense,Healthcare"] \
        [--ids "id1,id2,id3"]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm import LLM, MODEL_REGISTRY
from env_audit.generate import audit_tools


def _csv(arg: str) -> list[str]:
    return [x.strip() for x in arg.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Build + evaluate tools dataset from env_generation scenarios.")
    parser.add_argument("--env-specs-dir", type=Path, required=True,
                        help="Directory of env_generation scenario JSONs")
    parser.add_argument("--dataset-path", type=Path, required=True,
                        help="Path to tools_dataset.jsonl (read + rewrite in place)")
    parser.add_argument("--eval-logs-dir", type=Path, required=True,
                        help="Directory for per-tool eval JSON logs")
    parser.add_argument("--model", default="GPT-OSS-120B", choices=list(MODEL_REGISTRY))
    parser.add_argument("--fields", type=_csv, default=None,
                        help="Comma-separated field names to include (build + evaluate)")
    parser.add_argument("--ids", type=_csv, default=None,
                        help="Comma-separated tool ids to include (evaluate only; intersected with --fields)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build-only", action="store_true",
                      help="Phase 1 only. No LLM. Seeds dataset + eval_logs. Run once before parallel evaluate jobs.")
    mode.add_argument("--evaluate-only", action="store_true",
                      help="Phases 2-6 + per-spec refresh. SKIPS global dataset rewrite + end-of-run sweep. "
                           "Safe for concurrent sbatch jobs with disjoint --fields.")
    mode.add_argument("--sweep-only", action="store_true",
                      help="No LLM. Rebuild tools_dataset.jsonl from eval_logs + refresh every env_spec audit block.")
    args = parser.parse_args()

    mode_str = (
        "build"    if args.build_only
        else "evaluate" if args.evaluate_only
        else "sweep"    if args.sweep_only
        else "full"
    )

    llm = LLM(args.model)
    summary = audit_tools(
        env_specs_dir=args.env_specs_dir,
        dataset_path=args.dataset_path,
        eval_logs_dir=args.eval_logs_dir,
        llm=llm,
        fields=args.fields,
        ids=args.ids,
        mode=mode_str,
    )
    print(summary)


if __name__ == "__main__":
    main()
