"""
TaskSummarizer role: fuses a multi-step task's per-turn artefacts into one
user-facing request.

Chat-format input: the prior turns are serialised as (user, assistant, tool)
triples, then a final user message asks the LLM to write the summary. The
`tool` role makes tool outputs structurally distinct from user intent, which
discourages the LLM from copying tool-returned values into the summary.
"""

from typing import Callable, Dict, Any, List
import json
from pathlib import Path

import yaml

from . import Role
from utils import extract_json_objects

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompt_templates" / "task_summarizer"
TASK_SUMMARIZER_TEMPLATE_FILE = PROMPT_DIR / "task_summarizer_template.yml"


class TaskSummarizer(Role):
    def __init__(self, runner: Callable[[Any], str]):
        prompts = self._load_prompts()
        super().__init__(prompts)
        self.runner = runner

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def summarize_tasks(
        self,
        tasks: Any,
        tool_calls: Any,
        tool_responses: Any,
    ) -> Dict[str, Any]:
        """Summarize a sequence of mini-tasks into one user-facing request."""
        messages = self.build_messages(tasks, tool_calls, tool_responses)
        response = self.runner(messages)
        usage = self._get_usage()
        objs = extract_json_objects(response)
        parsed = objs[0] if objs else None
        # Persist the full chat in the `prompt` slot so the debug log /
        # summary block captures exactly what the LLM saw.
        prompt_for_record = json.dumps(messages, ensure_ascii=False, default=str)
        return {"prompt": prompt_for_record, "response": response, "parsed": parsed, "usage": usage}

    def build_messages(
        self,
        tasks: Any,
        tool_calls: Any,
        tool_responses: Any,
    ) -> List[Dict[str, str]]:
        """Build the (system, prior turns, final user) chat.

        Accepts the three inputs either as Python lists or as already-
        JSON-serialised strings (back-compat for callers that pre-format).
        Lists are aligned by index; if they have different lengths, the
        shorter ones pad with empty strings (defensive — the orchestrator
        always produces parallel lists of equal length).
        """
        tasks_l = self._coerce_to_list(tasks)
        calls_l = self._coerce_to_list(tool_calls)
        resps_l = self._coerce_to_list(tool_responses)
        n = max(len(tasks_l), len(calls_l), len(resps_l))

        msgs: List[Dict[str, str]] = [
            {"role": "system", "content": self.prompts["task_summarizer_system"]},
        ]
        for i in range(n):
            td = self._stringify(tasks_l[i] if i < len(tasks_l) else "")
            tc = self._stringify(calls_l[i] if i < len(calls_l) else "")
            tr = self._stringify(resps_l[i] if i < len(resps_l) else "")
            msgs.append({"role": "user", "content": td})
            msgs.append({"role": "assistant", "content": tc})
            msgs.append({"role": "tool", "content": tr})
        msgs.append({"role": "user", "content": self.prompts["task_summarizer_final_user"]})
        return msgs

    def run(self, action: str, **kwargs):
        actions = {
            "summarize_tasks": self.summarize_tasks,
        }
        if action not in actions:
            raise ValueError(f"Unsupported action '{action}'. Valid: {list(actions)}")
        return actions[action](**kwargs)

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _coerce_to_list(obj: Any) -> List[Any]:
        if isinstance(obj, list):
            return obj
        if isinstance(obj, str):
            try:
                parsed = json.loads(obj)
            except json.JSONDecodeError:
                return [obj]  # single element if it's not valid JSON
            return parsed if isinstance(parsed, list) else [parsed]
        if obj is None:
            return []
        return [obj]

    @staticmethod
    def _stringify(obj: Any) -> str:
        if isinstance(obj, str):
            return obj
        if obj is None:
            return ""
        try:
            return json.dumps(obj, ensure_ascii=False, default=str)
        except Exception:
            return str(obj)

    @staticmethod
    def _fmt(obj: Any) -> str:
        """Back-compat helper kept for callers that still pre-format inputs."""
        if isinstance(obj, str):
            return obj
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return str(obj)

    @staticmethod
    def _load_prompts() -> Dict[str, str]:
        def load_yaml(path: Path) -> dict:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                raise ValueError(f"Template root is not a mapping in {path}")
            return data

        data = load_yaml(TASK_SUMMARIZER_TEMPLATE_FILE)
        missing = [k for k in ("system_template", "final_user_template") if k not in data]
        if missing:
            raise ValueError(
                f"Task-summarizer template missing required fields {missing} "
                f"(expected both `system_template` and `final_user_template`): "
                f"{TASK_SUMMARIZER_TEMPLATE_FILE}"
            )
        return {
            "task_summarizer_system": data["system_template"],
            "task_summarizer_final_user": data["final_user_template"],
        }
