"""Task validation: is a released task fit to roll out and train on?

Runs on a release row (`task_content.jsonl`) alone — no LLM, no simulator,
no rollout. Every check is a structural impossibility proof, so a BLOCK
here is sound in a way an "accept" verdict never is: we can prove a task
cannot be solved, we cannot prove one was.

Use `validate_task(row, max_solver_turns=...)` → `TaskReport`.
"""

from task_validation.validator import (  # noqa: F401
    BLOCK,
    WARN,
    Finding,
    TaskReport,
    validate_task,
)
