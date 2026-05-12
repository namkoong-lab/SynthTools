"""
TaskSummarizer agent: loads the task-summarizer prompt and generates one summarized task
from a sequence of tasks and their tool calls/responses.
"""

from typing import Callable, Dict, Any
import json
from pathlib import Path

import yaml

from . import Role
from utils import extract_json_objects

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompt_templates" / "task_summarizer"
TASK_SUMMARIZER_TEMPLATE_FILE = PROMPT_DIR / "task_summarizer_template.yml"


class TaskSummarizer(Role):
    def __init__(self, runner: Callable[[str], str]):
        prompts = self._load_prompts()
        super().__init__(prompts)
        self.runner = runner

    def summarize_tasks(
        self,
        tasks: Any,
        tool_calls: Any,
        tool_responses: Any,
    ) -> Dict[str, Any]:
        """
        Summarize a sequence of tasks into one cohesive task description.
        """
        prompt = self.get_prompt(
            "task_summarizer",
            tasks=self._fmt(tasks),
            tool_calls=self._fmt(tool_calls),
            tool_responses=self._fmt(tool_responses),
        )
        response = self.runner(prompt)
        usage = self._get_usage()
        objs = extract_json_objects(response)
        parsed = objs[0] if objs else None
        return {"prompt": prompt, "response": response, "parsed": parsed, "usage": usage}

    def run(self, action: str, **kwargs):
        actions = {
            "summarize_tasks": self.summarize_tasks,
        }
        if action not in actions:
            raise ValueError(f"Unsupported action '{action}'. Valid: {list(actions)}")
        return actions[action](**kwargs)

    @staticmethod
    def _fmt(obj: Any) -> str:
        if isinstance(obj, str):
            return obj
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return str(obj)

    @staticmethod
    def _load_prompts() -> Dict[str, str]:
        def load_template(path: Path) -> str:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and "template" in data:
                return data["template"]
            raise ValueError(f"Template missing or invalid in {path}")

        return {
            "task_summarizer": load_template(TASK_SUMMARIZER_TEMPLATE_FILE),
        }