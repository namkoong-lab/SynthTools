"""Role base class with shared helpers.

A Role is an LLM-driven persona (Solver, Simulator, Judge, …) that owns a set
of named prompt templates and exposes a `run()` entry point.

The base class provides utilities every concrete role needs:

  get_prompt(name, **fmt_kwargs)              — render a template
  _fmt(obj) -> str                            — JSON-or-string serialiser
  _load_single_template(path) -> str          — read a YAML with `template:` field
  _load_chat_template(path, *fields) -> dict  — read a YAML with named template fields
  _chat_turn(user, asst, tool) -> List[dict]  — emit one (user, assistant, tool) slice
  _get_usage()                                — last_usage off the runner

Subclasses still own their own prompt-file paths and their own `__init__`
(because the runner arg + which templates to load are per-role).
"""

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

__all__ = ["Role"]


class Role(ABC):
    def __init__(self, prompts: Dict[str, str]):
        if not prompts:
            raise ValueError("prompts must not be empty")
        self.prompts = prompts

    def get_prompt(self, name: str, **kwargs) -> str:
        template = self.prompts.get(name)
        if template is None:
            raise KeyError(f"Prompt '{name}' is not defined")
        return template.format(**kwargs)

    # -----------------------------------------------------------------
    # Shared helpers (hoisted from per-role duplicates)
    # -----------------------------------------------------------------

    @staticmethod
    def _fmt(obj: Any) -> str:
        """Stringify `obj`. Strings pass through; everything else gets
        JSON-encoded (falls back to ``str(obj)`` if not JSON-serialisable)."""
        if isinstance(obj, str):
            return obj
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return str(obj)

    @staticmethod
    def _load_single_template(path: Path) -> str:
        """Load a YAML file with a single top-level `template:` field; return
        the template string. Raises ValueError if the field is missing."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict) and "template" in data:
            return data["template"]
        raise ValueError(f"Template missing or invalid in {path}")

    @staticmethod
    def _load_chat_template(path: Path, *fields: str) -> Dict[str, str]:
        """Load a YAML file with multiple top-level template fields (e.g.
        `system_template` and `final_user_template`). Return a dict keyed
        by the requested field names. Raises ValueError if any field is
        missing."""
        if not fields:
            raise ValueError("must pass at least one field name")
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Template root is not a mapping in {path}")
        missing = [k for k in fields if k not in data]
        if missing:
            raise ValueError(
                f"Template missing required fields {missing} (expected "
                f"{list(fields)}): {path}"
            )
        return {k: data[k] for k in fields}

    @staticmethod
    def _chat_turn(user_content: str, assistant_content: str, tool_content: str) -> List[Dict[str, str]]:
        """Emit one (user, assistant, tool) turn as a list of three message
        dicts. Used by chat-format prompts (task_evolver_t1, task_summarizer)."""
        return [
            {"role": "user", "content": user_content or ""},
            {"role": "assistant", "content": assistant_content or ""},
            {"role": "tool", "content": tool_content or ""},
        ]

    def _get_usage(self):
        return getattr(self.runner, "last_usage", None) if hasattr(self, "runner") else None

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError
