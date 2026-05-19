"""
JudgeSimulator agent: loads judge prompt from hard-coded template and returns cleaned JSON judgment.

Expected hard-coded template file:
- judge: prompt_templates/judge_simulator/judge_simulator_template.yml
"""

from typing import Callable, Dict, Any, Optional
from pathlib import Path

from . import Role
from utils import extract_json_objects

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompt_templates" / "judge_simulator"
JUDGE_TEMPLATE_FILE = PROMPT_DIR / "judge_simulator_template.yml"


class JudgeSimulator(Role):
    def __init__(self, runner: Callable[[str], str]):
        prompts = self._load_prompts()
        super().__init__(prompts)
        self.runner = runner

    def judge(
        self,
        tool_details: Any,
        message: Any,
        response: Any,
        meta_data: Any = None,
        failure_mode: Any = None,
    ) -> Dict[str, Any]:
        """
        Run the judge prompt and return parsed JSON judgment.
        """
        prompt, raw = self._judge_raw(
            tool_details=tool_details,
            message=message,
            response=response,
            meta_data=meta_data,
            failure_mode=failure_mode,
        )
        usage = self._get_usage()
        objs = extract_json_objects(raw)
        parsed = objs[0] if objs else None
        parsed_or_error = parsed if parsed is not None else {"error": "could_not_parse", "raw": raw}
        return {"prompt": prompt, "response": raw, "parsed": parsed_or_error, "usage": usage}

    def run(self, action: str, **kwargs):
        actions = {
            "judge": self.judge,
        }
        if action not in actions:
            raise ValueError(f"Unsupported action '{action}'. Valid: {list(actions)}")
        return actions[action](**kwargs)

    @classmethod
    def _load_prompts(cls) -> Dict[str, str]:
        return {
            "judge": cls._load_single_template(JUDGE_TEMPLATE_FILE),
        }

    def _judge_raw(
        self,
        tool_details: Any,
        message: Any,
        response: Any,
        meta_data: Any = None,
        failure_mode: Any = None,
    ) -> tuple[str, str]:
        prompt = self.get_prompt(
            "judge",
            tool_details=self._fmt(tool_details),
            message=self._fmt(message),
            response=self._fmt(response),
            meta_data=self._fmt(meta_data or {}),
            failure_mode=self._fmt(failure_mode or "none"),
        )
        return prompt, self.runner(prompt)

    # JSON extraction centralized via utils.extract_first_json

