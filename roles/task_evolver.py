"""
TaskEvolver role: proposes one mini-task at a time for the per-turn pipeline.

Two entry points:

  evolve_task_t0: cold start. Single-shot string prompt — only a tool schema,
                  no history, no leakage possible at this step.

  evolve_task_t1: every subsequent tool. Chat-format prompt — prior successful
                  turns are serialised as (user, assistant, tool) triples and
                  the new request goes in a final user message. The `tool`
                  role makes prior outputs structurally distinct from user
                  intent, which discourages the LLM from inlining
                  tool-returned values into the next mini-task description.
"""

from typing import Callable, Dict, Any, List
import json
from pathlib import Path

import yaml

from . import Role
from utils import extract_json_objects

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompt_templates" / "task_evolver"
TASK_EVOLVER_T0_TEMPLATE_FILE = PROMPT_DIR / "task_evolver_t0_template.yml"
TASK_EVOLVER_T1_TEMPLATE_FILE = PROMPT_DIR / "task_evolver_t1_template.yml"


class TaskEvolver(Role):
    def __init__(self, runner: Callable[[Any], str]):
        prompts = self._load_prompts()
        super().__init__(prompts)
        self.runner = runner

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def evolve_task_t0(self, tool_details: Any) -> Dict[str, Any]:
        """Generate the first task in a sequence (single-shot, no history)."""
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
        """Generate a follow-up task using past successes/failures as a chat.

        The runner receives a list of messages:
          [system]
            <full evolver instructions>
          [user, assistant, tool] per previously successful turn (in order)
            user: task_description
            assistant: tool_call
            tool: json.dumps(tool_simulated)
          [user]
            "Now produce the next mini-task" + tool_details + environment_state
            + unsuccessful_tasks (any prior failed attempts at THIS step)
        """
        messages = self._build_t1_messages(
            successful_task=successful_task,
            unsuccessful_tasks=unsuccessful_tasks,
            tool_details=tool_details,
            environment_state=environment_state,
        )
        response = self.runner(messages)
        usage = self._get_usage()
        objs = extract_json_objects(response)
        parsed = objs[0] if objs else None
        # `prompt` is what gets stored in the RunLog event for this call; the
        # full message list is the most useful artefact to debug a leak from.
        prompt_for_record = json.dumps(messages, ensure_ascii=False, default=str)
        return {"prompt": prompt_for_record, "response": response, "parsed": parsed, "usage": usage}

    def run(self, action: str, **kwargs):
        actions = {
            "evolve_task_t0": self.evolve_task_t0,
            "evolve_task_t1": self.evolve_task_t1,
        }
        if action not in actions:
            raise ValueError(f"Unsupported action '{action}'. Valid: {list(actions)}")
        return actions[action](**kwargs)

    # -----------------------------------------------------------------
    # Chat-message construction (t1)
    # -----------------------------------------------------------------

    def _build_t1_messages(
        self,
        successful_task: Any,
        unsuccessful_tasks: Any,
        tool_details: Any,
        environment_state: Any,
    ) -> List[Dict[str, str]]:
        """Build the (system, prior turns, final user) chat for the t1 evolver.

        Tolerates `successful_task` either as a Python list-of-dicts (the
        normal pipeline shape) or a JSON-encoded string (back-compat for
        callers that already serialise it). Malformed entries are dropped
        rather than raising — keeps the orchestrator resilient.
        """
        system_msg = {"role": "system", "content": self.prompts["task_evolver_t1_system"]}
        prior = self._normalise_successful(successful_task)
        history_msgs = self._interleave_prior_turns(prior)
        final_user_content = self.prompts["task_evolver_t1_final_user"].format(
            tool_details=self._fmt(tool_details),
            environment_state=self._fmt(environment_state),
            unsuccessful_tasks=self._fmt(unsuccessful_tasks if unsuccessful_tasks is not None else []),
        )
        final_user_msg = {"role": "user", "content": final_user_content}
        return [system_msg, *history_msgs, final_user_msg]

    @staticmethod
    def _normalise_successful(successful_task: Any) -> List[Dict[str, Any]]:
        if isinstance(successful_task, str):
            try:
                parsed = json.loads(successful_task)
            except json.JSONDecodeError:
                return []
            return parsed if isinstance(parsed, list) else []
        if isinstance(successful_task, list):
            return successful_task
        return []

    @staticmethod
    def _interleave_prior_turns(prior: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """For each prior successful turn emit (user, assistant, tool) messages."""
        out: List[Dict[str, str]] = []
        for entry in prior:
            if not isinstance(entry, dict):
                continue
            user_content = entry.get("task_description") or ""
            tc = entry.get("tool_call")
            asst_content = tc if isinstance(tc, str) else json.dumps(tc or "", ensure_ascii=False)
            tool_payload = entry.get("tool_simulated")
            if tool_payload is None:
                tool_payload = {}
            tool_content = json.dumps(tool_payload, ensure_ascii=False, default=str)
            out.append({"role": "user", "content": user_content})
            out.append({"role": "assistant", "content": asst_content})
            out.append({"role": "tool", "content": tool_content})
        return out

    # -----------------------------------------------------------------
    # Misc helpers
    # -----------------------------------------------------------------

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
        def load_yaml(path: Path) -> dict:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                raise ValueError(f"Template root is not a mapping in {path}")
            return data

        t0 = load_yaml(TASK_EVOLVER_T0_TEMPLATE_FILE)
        if "template" not in t0:
            raise ValueError(
                f"Task-evolver t0 template missing `template` field: {TASK_EVOLVER_T0_TEMPLATE_FILE}"
            )

        t1 = load_yaml(TASK_EVOLVER_T1_TEMPLATE_FILE)
        missing = [k for k in ("system_template", "final_user_template") if k not in t1]
        if missing:
            raise ValueError(
                f"Task-evolver t1 template missing required fields {missing} "
                f"(expected both `system_template` and `final_user_template`): "
                f"{TASK_EVOLVER_T1_TEMPLATE_FILE}"
            )

        return {
            "task_evolver_t0": t0["template"],
            "task_evolver_t1_system": t1["system_template"],
            "task_evolver_t1_final_user": t1["final_user_template"],
        }
