#!/bin/bash
#SBATCH -t 06:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --account=m5091
#SBATCH --job-name=summary-clean
#SBATCH --output=/global/homes/t/tcaste/projects/logs/summary_clean_%x_%j.out
#SBATCH --error=/global/homes/t/tcaste/projects/logs/summary_clean_%x_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --constraint "gpu&hbm40g"
#SBATCH --qos=premium

# Usage:
#   sbatch summary_clean.sbatch <SHARD_K> <SHARD_N>
#   e.g.  sbatch summary_clean.sbatch 0 5    (one of 5 disjoint shards)
#
# Each job loads its own GPT-OSS-120B in-process across the 4 node GPUs (TP=4)
# and processes only the files where (idx % SHARD_N == SHARD_K). Submit N jobs
# total to cover the whole corpus in parallel.

set -euo pipefail

SHARD_K="${1:?usage: sbatch summary_clean.sbatch SHARD_K SHARD_N}"
SHARD_N="${2:?usage: sbatch summary_clean.sbatch SHARD_K SHARD_N}"

cd /global/homes/t/tcaste/projects/SynthTools
source /pscratch/sd/t/tcaste/envs/burn-gpu/bin/activate
module load cudatoolkit/12.9

INPUT=/pscratch/sd/t/tcaste/tool_content/trajectories_clean
MODEL=GPT-OSS-120B
BATCH_SIZE=128

echo "=========================================="
echo "summary_clean — shard ${SHARD_K}/${SHARD_N}"
echo "input        : $INPUT"
echo "model        : $MODEL"
echo "batch size   : $BATCH_SIZE"
echo "started      : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "node         : $(hostname)"
echo "=========================================="

t0=$(date +%s)

python -m scripts.summary_clean \
    --trajectories-dir "$INPUT" \
    --model            "$MODEL" \
    --batch-size       "$BATCH_SIZE" \
    --shard            "${SHARD_K}/${SHARD_N}"

t1=$(date +%s)
elapsed=$((t1 - t0))

echo ""
echo "=========================================="
echo "ended        : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "elapsed      : ${elapsed}s ($(printf '%dh%02dm%02ds' $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))))"
echo "=========================================="
