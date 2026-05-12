"""
JudgeSimulator agent: loads judge prompt from hard-coded template and returns cleaned JSON judgment.

Expected hard-coded template file:
- judge: prompt_templates/judge_simulator/judge_template.yml
"""

from typing import Callable, Dict, Any, Optional
import json
from pathlib import Path
import re

import yaml

from . import Role
from utils import extract_json_objects

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompt_templates" / "judge_simulator"
JUDGE_TEMPLATE_FILE = PROMPT_DIR / "judge_template.yml"


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
            "judge": load_template(JUDGE_TEMPLATE_FILE),
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

