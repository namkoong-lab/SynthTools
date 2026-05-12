"""Trajectory generation: roll out an LLM agent on a verifiable task and judge the result.

A *task* is one row of `tasks.parquet` (id, summary, tools, gt_tool_calls, initial_state,
final_state). Calling `generate_trajectory(task, llm, ...)` runs the existing
TaskSolver against the existing ToolSimulator over multiple turns until the agent
emits `<STOP>` or hits a turn budget, then verifies the rollout with the
`TrajectoryJudge` against the ground-truth call sequence and final state.
"""
