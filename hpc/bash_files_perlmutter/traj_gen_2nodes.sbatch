#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH --account=m5091
#SBATCH --job-name=traj-gen-2n
#SBATCH --output=/global/homes/t/tcaste/projects/logs/traj_gen_2n_%j.out
#SBATCH --error=/global/homes/t/tcaste/projects/logs/traj_gen_2n_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --constraint "gpu&hbm40g"
#SBATCH --qos=premium

set -euo pipefail

cd /global/homes/t/tcaste/projects/SynthTools
source /pscratch/sd/t/tcaste/envs/burn-gpu/bin/activate
module load cudatoolkit/12.9

export DATASET=/pscratch/sd/t/tcaste/tool_content/tools_dataset.jsonl
export ENV_SPECS=/pscratch/sd/t/tcaste/tool_content/env_specs
export OUTPUT=/pscratch/sd/t/tcaste/tool_content/tasks
export MODEL=GPT-OSS-120B

LOG_DIR=/global/homes/t/tcaste/projects/logs
mkdir -p "$LOG_DIR"

# IMPORTANT: this is a HARDCODED slice of 20 fields, split 10/10 across the
# 2 allocated nodes. If new fields appear in env_specs/ (currently 100 fields
# total) this script will SILENTLY miss them. Update the two FIELDS_NODE_*
# arrays below to keep coverage. A future helper could derive these from
# `ls env_specs/*_spec_000.json` — for now they're frozen by hand.
FIELDS_NODE_A=(
  "Investment Banking"
  "Legal Services"
  "Manufacturing Industrial IoT Robot Control"
  "Marketing"
  "Media And Entertainment"
  "Mining And Metals"
  "Nonprofit And Philanthropy"
  "Private Banking"
  "Public Sector And Government"
  "Real Estate"
)

FIELDS_NODE_B=(
  "Research And Development"
  "Semiconductors"
  "Software Apps"
  "Sports And Fitness"
  "Telecommunications Networks"
  "Transportation Logistics"
  "Travel And Hospitality"
  "Waste Management Recycling"
  "Water And Sanitation"
  "Weather And Climate Services"
)

# Resolve the two nodes Slurm gave us.
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
NODE_A=${NODES[0]}
NODE_B=${NODES[1]}

echo "=========================================="
echo "Trajectory generation, 2-node split"
echo "started  : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "node A   : $NODE_A  (${#FIELDS_NODE_A[@]} fields)"
echo "node B   : $NODE_B  (${#FIELDS_NODE_B[@]} fields)"
echo "model    : $MODEL"
echo "output   : $OUTPUT"
echo "=========================================="

run_fields_on_node () {
    local node=$1
    local tag=$2
    shift 2
    local fields=("$@")
    local i=0
    for field in "${fields[@]}"; do
        i=$((i+1))
        echo ""
        echo "=== [$tag $i/${#fields[@]}] $field ==="
        python -m task_generation.run \
            --dataset       "$DATASET" \
            --env-specs-dir "$ENV_SPECS" \
            --output-dir    "$OUTPUT" \
            --model         "$MODEL" \
            --field         "$field" \
            --verifiable
    done
    echo "[$tag] done."
}

# Launch one task per node in parallel. Each gets all 4 GPUs on its node.
srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus-per-node=4 \
     --nodelist="$NODE_A" --exclusive \
     --output="$LOG_DIR/traj_gen_2n_${SLURM_JOB_ID}_nodeA.log" \
     bash -c "$(declare -f run_fields_on_node); \
              run_fields_on_node '$NODE_A' 'A' $(printf '%q ' "${FIELDS_NODE_A[@]}")" &
PID_A=$!

srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus-per-node=4 \
     --nodelist="$NODE_B" --exclusive \
     --output="$LOG_DIR/traj_gen_2n_${SLURM_JOB_ID}_nodeB.log" \
     bash -c "$(declare -f run_fields_on_node); \
              run_fields_on_node '$NODE_B' 'B' $(printf '%q ' "${FIELDS_NODE_B[@]}")" &
PID_B=$!

# Wait for both, propagate failure.
wait $PID_A; rc_a=$?
wait $PID_B; rc_b=$?

echo ""
echo "=========================================="
echo "ended    : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "node A rc: $rc_a"
echo "node B rc: $rc_b"
echo "trajectories on disk: $(ls "$OUTPUT"/*.json 2>/dev/null | wc -l)"
echo "=========================================="

exit $(( rc_a | rc_b ))
