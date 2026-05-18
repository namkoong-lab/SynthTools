"""
EnvironmentSimulator agent: loads the environment-update prompt, runs the LLM,
and returns the edited and full environment metadata updates.

Expected hard-coded template file:
- update_environment: prompt_templates/env_simulator/env_simulator_template.yml
"""

from typing import Callable, Dict, Any, Optional
from pathlib import Path

from . import Role
from utils import extract_json_objects

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompt_templates" / "env_simulator"
ENV_SIM_TEMPLATE_FILE = PROMPT_DIR / "env_simulator_template.yml"


class EnvironmentSimulator(Role):
    def __init__(self, runner: Callable[[str], str]):
        prompts = self._load_prompts()
        super().__init__(prompts)
        self.runner = runner

    def update_environment(
        self,
        tool_schema: Dict[str, Any],
        tool_call_message: str,
        environment_state: Any = None,
        tool_simulation_output: Any = None,
    ) -> Dict[str, Any]:
        """
        Apply a tool call to the environment metadata and return only the
        parsed JSON (edited_metadata and full_metadata).
        """
        prompt, update_resp = self._update_environment_raw(
            tool_schema=tool_schema,
            tool_call_message=tool_call_message,
            environment_state=environment_state,
            tool_simulation_output=tool_simulation_output,
        )
        usage = self._get_usage()
        objs = extract_json_objects(update_resp)
        update_json = objs[0] if objs else None

        return {"prompt": prompt, "response": update_resp, "parsed": update_json, "usage": usage}

    def run(self, action: str, **kwargs):
        actions = {
            "update_environment": self.update_environment,
        }
        if action not in actions:
            raise ValueError(f"Unsupported action '{action}'. Valid: {list(actions)}")
        return actions[action](**kwargs)

    @classmethod
    def _load_prompts(cls) -> Dict[str, str]:
        return {
            "update_environment": cls._load_single_template(ENV_SIM_TEMPLATE_FILE),
        }

    def _update_environment_raw(
        self,
        tool_schema: Dict[str, Any],
        tool_call_message: str,
        environment_state: Any = None,
        tool_simulation_output: Any = None,
    ) -> tuple[str, str]:
        prompt = self.get_prompt(
            "update_environment",
            tool_schema=self._fmt(tool_schema),
            tool_call=self._fmt(tool_call_message),
            environment_state=self._fmt(environment_state or {}),
            tool_simulation_output=self._fmt(tool_simulation_output or {}),
        )
        return prompt, self.runner(prompt)

    # JSON extraction centralized via utils.extract_first_json

