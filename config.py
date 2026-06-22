"""Pipeline-wide defaults.

Single source of truth for values that previously drifted across run.py
argparse defaults, function signatures, and HPC sbatch heredocs. Edit
here once; every stage and HPC script imports from this module.
"""

# LLM default for every stage that calls argparse(--model). Override at
# call time with --model on the CLI.
DEFAULT_MODEL = "GPT-OSS-120B"

# Sequence generation (task_generation/build_sequences).
DEFAULT_N_SEQUENCES = 50
DEFAULT_SEQ_LENGTH = 10
DEFAULT_MIN_RELIABILITY = 0.33

# Per-task generation loop (task_generation/run).
DEFAULT_MAX_SOLVER_TURNS = 5
DEFAULT_MAX_RETRIES = 5
DEFAULT_MAX_SOLVER_RETRIES_IN_PLACE = 2

# Batched LLM calls (task_audit/summarize, env_audit phases 2+).
DEFAULT_BATCH_SIZE = 64

# Process-pool concurrency for task_generation and trajectory_generation.
DEFAULT_CONCURRENCY = 8

# vLLM in-process engine tuning. Override per call via LLM(...) kwargs.
DEFAULT_TENSOR_PARALLEL = 4
DEFAULT_MAX_MODEL_LEN = 32768
DEFAULT_GPU_MEM_UTIL = 0.90
