"""Shared argparse helpers for pipeline CLIs.

Tiny module by design — only hoist flags that have IDENTICAL semantics
across multiple `run.py` entry points. Flags with stage-specific shape
(`--output-dir` vs `--output`, `--no-debug` vs `--write_debug`, the
mode-mutex groups, etc.) stay in their own `run.py` to avoid leaky
abstractions.

Currently hoisted:
  add_model_arg(parser)       --model
  add_server_url_arg(parser)  --server-url
"""

from __future__ import annotations

import argparse

from config import DEFAULT_MODEL
from llm import MODEL_REGISTRY


def add_model_arg(parser: argparse.ArgumentParser, *, default: str = DEFAULT_MODEL) -> None:
    """Add the standard `--model` flag, choices from MODEL_REGISTRY."""
    parser.add_argument(
        "--model",
        default=default,
        choices=list(MODEL_REGISTRY),
        help="LLM model name (registered in llm.MODEL_REGISTRY).",
    )


def add_server_url_arg(parser: argparse.ArgumentParser) -> None:
    """Add the standard `--server-url` flag for vLLM HTTP backend."""
    parser.add_argument(
        "--server-url",
        type=str,
        default=None,
        help="OpenAI-compatible base URL (e.g. http://localhost:8765/v1). "
             "When set, all LLM calls go over HTTP instead of loading vLLM "
             "in-process.",
    )
