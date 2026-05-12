"""
TaskEvolver agent: loads the task-evolver prompt and generates one evolved task
given tool details.
"""

from typing import Callable, Dict, Any
import json
from pathlib import Path

import yaml

from . import Role
from utils import extract_json_objects

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompt_templates" / "task_evolver"
TASK_EVOLVER_T0_TEMPLATE_FILE = PROMPT_DIR / "task_evolver_t0_template.yml"
TASK_EVOLVER_T1_TEMPLATE_FILE = PROMPT_DIR / "task_evolver_t1_template.yml"


class TaskEvolver(Role):
    def __init__(self, runner: Callable[[str], str]):
        prompts = self._load_prompts()
        super().__init__(prompts)
        self.runner = runner

    def evolve_task_t0(self, tool_details: Any) -> Dict[str, Any]:
        """
        Generate a single evolved task (tool call, env metadata, task description)
        for the provided tool details.
        """
        prompt = self.get_prompt("task_evolver_t0", tool_details=self._fmt(tool_details))
        response = self.runner(prompt)
        usage = self._get_usage()
        objs = extract_json_objects(response)
        parsed = objs[0] if objs else None
        return {"prompt": prompt, "response": response, "parsed": parsed, "usage": usage}

    def evolve_task_t1(
        self,
        successful_task: Any,
        unsuccessful_tasks: Any,
        tool_details: Any,
        environment_state: Any,
    ) -> Dict[str, Any]:
        """
        Generate a follow-up task using past successes/failures, keeping it solvable in one tool call.
        """
        prompt = self.get_prompt(
            "task_evolver_t1",
            successful_task=self._fmt(successful_task),
            unsuccessful_tasks=self._fmt(unsuccessful_tasks),
            tool_details=self._fmt(tool_details),
            environment_state=self._fmt(environment_state),
        )
        response = self.runner(prompt)
        usage = self._get_usage()
        objs = extract_json_objects(response)
        parsed = objs[0] if objs else None
        return {"prompt": prompt, "response": response, "parsed": parsed, "usage": usage}

    def run(self, action: str, **kwargs):
        actions = {
            "evolve_task_t0": self.evolve_task_t0,
            "evolve_task_t1": self.evolve_task_t1,
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
            "task_evolver_t0": load_template(TASK_EVOLVER_T0_TEMPLATE_FILE),
            "task_evolver_t1": load_template(TASK_EVOLVER_T1_TEMPLATE_FILE),
        }