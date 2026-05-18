#!/bin/bash
#SBATCH -t 08:00:00
#SBATCH --account=m5091
#SBATCH --job-name=build-sequences
#SBATCH --output=/global/homes/t/tcaste/projects/logs/build_sequences_%j.out
#SBATCH --error=/global/homes/t/tcaste/projects/logs/build_sequences_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --constraint "gpu&hbm40g"
#SBATCH --qos=premium

set -euo pipefail

cd /global/homes/t/tcaste/projects/SynthTools
source /pscratch/sd/t/tcaste/envs/burn-gpu/bin/activate

python3 << 'PYEOF'
import sys
sys.path.insert(0, ".")
from pathlib import Path
from config import DEFAULT_MIN_RELIABILITY, DEFAULT_MODEL, DEFAULT_N_SEQUENCES, DEFAULT_SEQ_LENGTH
from llm import LLM
from task_generation.build_sequences import build_sequences

llm = LLM(DEFAULT_MODEL, max_tokens=32768)
ENV_SPECS = Path("/pscratch/sd/t/tcaste/tool_content/env_specs")

build_sequences(
    env_specs_dir=ENV_SPECS,
    llm=llm,
    n_sequences=DEFAULT_N_SEQUENCES,
    seq_length=DEFAULT_SEQ_LENGTH,
    min_reliability=DEFAULT_MIN_RELIABILITY,
    eval_only=True,
)
print("\nall done", flush=True)
PYEOF
