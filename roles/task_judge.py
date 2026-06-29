"""
TaskJudge agent: judges whether an agent's tool call solved the task
against the true tool call and task description.
"""

from typing import Callable, Dict, Any
from pathlib import Path

from . import Role
from utils import extract_json_objects

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompt_templates" / "task_judge"
TASK_JUDGE_TEMPLATE_FILE = PROMPT_DIR / "task_judge_gen_template.yml"
TASK_JUDGE_EVAL_TEMPLATE_FILE = PROMPT_DIR / "task_judge_eval_template.yml"


class TaskJudge(Role):
    def __init__(self, runner: Callable[[str], str]):
        prompts = self._load_prompts()
        super().__init__(prompts)
        self.runner = runner

    def judge_task_gen(
        self,
        true_tool_call: Any,
        current_task_description: Any,
        prior_chat: Any,
        agent_tool_calls: Any,
        running_summary: str = "",
    ) -> Dict[str, Any]:
        """
        Judge whether the agent's attempts solve the task AND whether every argument
        in the ground-truth tool_call is grounded in the agent's view (current task
        description + prior_chat). The judge has the SAME world-information as the
        agent (no env_state / env_metadata) — this asymmetry is the correctness
        guarantee.

        Also runs a second INDEPENDENT grounding check against `running_summary`:
        the cumulative natural-language user request a rollout agent would read at
        test time WITHOUT chat history. Both `arguments_grounded` (chat) and
        `running_summary_grounded` (rollout) must be true for a task to be
        considered solvable end-to-end.

        prior_chat: list of {"task_description", "tool_call", "tool_response"} from
                    prior SUCCESSFUL turns, in chronological order. Empty list for
                    the first turn.

        Returns prompt, raw response, and parsed JSON (if any).
        """
        prompt = self.get_prompt(
            "task_judge_gen",
            true_tool_call=self._fmt(true_tool_call),
            current_task_description=self._fmt(current_task_description),
            prior_chat=self._fmt(prior_chat if prior_chat is not None else []),
            agent_tool_calls=self._fmt(agent_tool_calls),
            running_summary=running_summary or "",
        )
        response = self.runner(prompt)
        usage = self._get_usage()
        objs = extract_json_objects(response)
        parsed = objs[0] if objs else None
        return {"prompt": prompt, "response": response, "parsed": parsed, "usage": usage}

    def judge_task_eval(
        self,
        true_env_state: Any,
        eval_env_state: Any,
        true_tool_calls: Any,
        eval_tool_calls: Any,
        task_description: Any,
    ) -> Dict[str, Any]:
        """
        Judge evaluation results by comparing true and evaluation environment states and tool calls.
        Returns prompt, raw response, and parsed JSON (if any).
        """
        prompt = self.get_prompt(
            "task_judge_eval",
            true_env_state=self._fmt(true_env_state),
            eval_env_state=self._fmt(eval_env_state),
            true_tool_calls=self._fmt(true_tool_calls),
            eval_tool_calls=self._fmt(eval_tool_calls),
            task_description=self._fmt(task_description),
        )
        response = self.runner(prompt)
        usage = self._get_usage()
        objs = extract_json_objects(response)
        parsed = objs[0] if objs else None
        return {"prompt": prompt, "response": response, "parsed": parsed, "usage": usage}

    def run(self, action: str, **kwargs):
        actions = {
            "judge_task_gen": self.judge_task_gen,
            "judge_task_eval": self.judge_task_eval,
        }
        if action not in actions:
            raise ValueError(f"Unsupported action '{action}'. Valid: {list(actions)}")
        return actions[action](**kwargs)

    @classmethod
    def _load_prompts(cls) -> Dict[str, str]:
        return {
            "task_judge_gen": cls._load_single_template(TASK_JUDGE_TEMPLATE_FILE),
            "task_judge_eval": cls._load_single_template(TASK_JUDGE_EVAL_TEMPLATE_FILE),
        }