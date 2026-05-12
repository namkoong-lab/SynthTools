"""Role base class with shared prompt handling.

A Role is an LLM-driven persona (Solver, Simulator, Judge, …) that owns a set
of named prompt templates and exposes a `run()` entry point. Subclasses supply
their own prompts and implement `run()`.
"""

from abc import ABC, abstractmethod
from typing import Dict

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

    def _get_usage(self):
        return getattr(self.runner, "last_usage", None) if hasattr(self, "runner") else None

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError
