"""TrajectoryJudge: full-rollout judge for trajectory_generation.

Given a verifiable task and an agent's rollout, the TrajectoryJudge decides
whether the rollout covers the ground-truth call sequence with semantically
equivalent arguments and whether the resulting environment state matches the
ground-truth final state.

Mirrors the role/prompt convention used by roles.task_judge; the only
difference is that the judge operates on the WHOLE trajectory at once
instead of per-turn.
"""

from typing import Any, Callable, Dict
from pathlib import Path

from . import Role
from utils import extract_json_objects


PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompt_templates" / "trajectory_judge"
TRAJECTORY_JUDGE_TEMPLATE_FILE = PROMPT_DIR / "trajectory_judge_template.yml"


class TrajectoryJudge(Role):
    def __init__(self, runner: Callable[[str], str]):
        prompts = self._load_prompts()
        super().__init__(prompts)
        self.runner = runner

    def judge_trajectory(
        self,
        task_summary: Any,
        gt_tool_calls: Any,
        agent_tool_calls: Any,
        initial_state: Any,
        final_state_gt: Any,
    ) -> Dict[str, Any]:
        """Render the prompt, invoke the runner, return prompt/response/parsed/usage."""
        prompt = self.get_prompt(
            "trajectory_judge",
            task_summary=self._fmt(task_summary),
            gt_tool_calls=self._fmt(gt_tool_calls),
            agent_tool_calls=self._fmt(agent_tool_calls),
            initial_state=self._fmt(initial_state if initial_state is not None else {}),
            final_state_gt=self._fmt(final_state_gt if final_state_gt is not None else {}),
        )
        response = self.runner(prompt)
        usage = self._get_usage()
        objs = extract_json_objects(response)
        parsed = objs[0] if objs and isinstance(objs[0], dict) else None
        return {"prompt": prompt, "response": response, "parsed": parsed, "usage": usage}

    def run(self, action: str, **kwargs):
        actions = {"judge_trajectory": self.judge_trajectory}
        if action not in actions:
            raise ValueError(f"Unsupported action '{action}'. Valid: {list(actions)}")
        return actions[action](**kwargs)

    @classmethod
    def _load_prompts(cls) -> Dict[str, str]:
        return {
            "trajectory_judge": cls._load_single_template(TRAJECTORY_JUDGE_TEMPLATE_FILE),
        }
