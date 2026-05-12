"""
TaskSolver agent: loads the task-solver prompt and produces a single turn
solution attempt given a tool schema and a task.
"""

from typing import Callable, Dict, Any
import json
from pathlib import Path

import yaml

from . import Role
from utils import extract_json_objects

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompt_templates" / "task_solver"
TASK_SOLVER_TEMPLATE_FILE = PROMPT_DIR / "task_solver_gen_template.yml"
TASK_SOLVER_EVAL_TEMPLATE_FILE = PROMPT_DIR / "task_solver_eval_template.yml"
TASK_SOLVER_TRAJECTORY_TEMPLATE_FILE = PROMPT_DIR / "task_solver_trajectory_template.yml"

# Fields the solver is allowed to see — the OpenAI function-calling subset.
# Simulator-only extras (`error_messages`, `usage`, `output_details`) are filtered
# out so the solver can't cheat off pre-documented error strings or output shapes.
_OPENAI_TOOL_FIELDS = ("tool_name", "tool_description", "parameters")


class TaskSolver(Role):
    def __init__(self, runner: Callable[[str], str], mode: str = "gen"):
        """`mode` selects which system prompt to use:
          - "gen"        : one-tool-per-step generator (default; legacy behaviour)
          - "trajectory" : multi-step solver that calls tools turn by turn
                            until it emits ``<STOP>`` (used by the
                            `trajectory_generation` package).
        """
        if mode not in ("gen", "trajectory"):
            raise ValueError(f"Unknown mode {mode!r}; expected 'gen' or 'trajectory'.")
        self.mode = mode
        prompts = self._load_prompts()
        super().__init__(prompts)
        self.runner = runner

    def system_prompt(self) -> str:
        """Return the static solver system prompt (no per-call placeholders)."""
        key = "task_solver_trajectory" if self.mode == "trajectory" else "task_solver_gen"
        return self.get_prompt(key)

    @staticmethod
    def _filter_tool(tool: Any) -> Any:
        if isinstance(tool, dict):
            return {k: v for k, v in tool.items() if k in _OPENAI_TOOL_FIELDS}
        return tool

    @staticmethod
    def build_user_message(task_description: str, tool_schemas: Any) -> str:
        """Format a user message exposing the OpenAI-subset of one or more tools.

        `tool_schemas` may be:
          - a single tool dict / schema string  → renders as "Tool to use:" (legacy)
          - a list of tool dicts                → renders as "Tools available
            for this task (select one):" with each tool numbered.
        """
        if isinstance(tool_schemas, list):
            blocks = []
            for i, tool in enumerate(tool_schemas, start=1):
                f = TaskSolver._filter_tool(tool)
                blocks.append(f"[{i}] {json.dumps(f, ensure_ascii=False)}")
            joined = "\n".join(blocks)
            return (
                f"Task: {task_description}\n\n"
                f"Tools available for this task (select one):\n{joined}"
            )

        if isinstance(tool_schemas, str):
            schema_json = tool_schemas
        elif isinstance(tool_schemas, dict):
            schema_json = json.dumps(TaskSolver._filter_tool(tool_schemas), ensure_ascii=False)
        else:
            schema_json = json.dumps(tool_schemas, ensure_ascii=False)
        return f"Task: {task_description}\n\nTool to use:\n{schema_json}"

    def interact(self, prompt: str) -> Dict[str, Any]:
        """Send a raw prompt to the LLM without templating; return prompt, response, parsed JSON if any."""
        response = self.runner(prompt)
        usage = self._get_usage()
        objs = extract_json_objects(response)
        parsed = objs[0] if objs else None
        return {"prompt": prompt, "response": response, "parsed": parsed, "usage": usage}

    def solve_task_eval(
        self,
        tools: Any,
        task: Any,
    ) -> Dict[str, Any]:
        """Generate the initial prompt for multi-turn task solving with a set of tools."""
        prompt = self.get_prompt(
            "task_solver_eval",
            tools=self._fmt(tools),
            task=self._fmt(task),
        )
        response = self.runner(prompt)
        usage = self._get_usage()
        objs = extract_json_objects(response)
        parsed = objs[0] if objs else None
        return {"prompt": prompt, "response": response, "parsed": parsed, "usage": usage}

    def run(self, action: str, **kwargs):
        actions = {
            "interact": self.interact,
            "solve_task_eval": self.solve_task_eval,
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
            "task_solver_gen": load_template(TASK_SOLVER_TEMPLATE_FILE),
            "task_solver_eval": load_template(TASK_SOLVER_EVAL_TEMPLATE_FILE),
            "task_solver_trajectory": load_template(TASK_SOLVER_TRAJECTORY_TEMPLATE_FILE),
        }