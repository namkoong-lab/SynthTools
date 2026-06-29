#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH --account=m5091
#SBATCH --job-name=traj-gen
#SBATCH --output=/global/homes/t/tcaste/projects/logs/traj_gen_%j.out
#SBATCH --error=/global/homes/t/tcaste/projects/logs/traj_gen_%j.err
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --constraint "gpu&hbm40g"
#SBATCH --qos=premium
#SBATCH --no-kill

# Trajectory generation, no judge. Each node is an INDEPENDENT rollout
# stream: it launches its own vLLM and processes its own random permutation
# of the dataset (seeded by JOB_SEED*1000 + node_index) until the walltime.
# Output files are seed-tagged ({task_id}.s<nodeseed>.json), so the same
# task rolled out by different nodes/jobs never collides and accumulates
# multiple trajectories.
#
# Node count is read from the allocation, so the SAME file serves a 10-node
# or a 20-node job: override at submit time with `sbatch --nodes=20 ...`.
#
# Required env (set via --export):
#   JOB_SEED     integer base seed for this job (distinct per job)
#   OUTPUT_DIR   where trajectory JSONs land
# Optional env (defaults shown):
#   SAMPLE=        per-node cap on tasks (empty = uncapped, run till walltime)
#   CONCURRENCY=16 rollouts in flight per node against its vLLM
#   MAX_SOLVER_TURNS=12
#   DATASET=/pscratch/sd/t/tcaste/tool_content/task_content.jsonl

set -euo pipefail

REPO=/global/homes/t/tcaste/projects/SynthTools
VENV=/pscratch/sd/t/tcaste/envs/burn-gpu/bin/activate

DATASET=${DATASET:-/pscratch/sd/t/tcaste/tool_content/task_content.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR}
JOB_SEED=${JOB_SEED:?set JOB_SEED}
SAMPLE=${SAMPLE:-}
CONCURRENCY=${CONCURRENCY:-16}
MAX_SOLVER_TURNS=${MAX_SOLVER_TURNS:-20}
TEMPERATURE=${TEMPERATURE:-0.2}
TOP_P=${TOP_P:-0.95}

LOG_DIR=/global/homes/t/tcaste/projects/logs
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

SERVE_MODEL=${SERVE_MODEL:-openai/gpt-oss-120b}
PORT=${PORT:-8765}
TP=${TP:-4}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
REASONING_PARSER=${REASONING_PARSER:-openai_gptoss}
DTYPE=${DTYPE:-bfloat16}
READY_TIMEOUT=${READY_TIMEOUT:-1800}
MODEL=${MODEL:-GPT-OSS-120B}

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
NUM_NODES=${#NODES[@]}

echo "=========================================="
echo "trajectory generation — $NUM_NODES nodes, 1 independent stream per node"
echo "started   : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "job id    : $SLURM_JOB_ID"
echo "job seed  : $JOB_SEED  (node seeds: $((JOB_SEED*1000))..$((JOB_SEED*1000+NUM_NODES-1)))"
echo "nodes     : ${NODES[*]}"
echo "dataset   : $DATASET"
echo "output    : $OUTPUT_DIR"
echo "sample    : ${SAMPLE:-<uncapped>}   concurrency: $CONCURRENCY   max_turns: $MAX_SOLVER_TURNS   temperature: $TEMPERATURE   top_p: $TOP_P"
echo "=========================================="

run_stream_on_node () {
    local node=$1
    local node_seed=$2
    local repo=$3
    local venv=$4
    local dataset=$5
    local output_dir=$6
    local sample=$7
    local concurrency=$8
    local max_solver_turns=$9
    local log_dir=${10}
    local model=${11}
    local serve_model=${12}
    local port=${13}
    local tp=${14}
    local max_model_len=${15}
    local gpu_mem_util=${16}
    local reasoning_parser=${17}
    local dtype=${18}
    local ready_timeout=${19}
    local temperature=${20}
    local top_p=${21}

    cd "$repo"
    # shellcheck disable=SC1090
    source "$venv"
    module load cudatoolkit/12.9

    local vllm_log="$log_dir/traj_gen_${SLURM_JOB_ID}_seed${node_seed}_vllm.log"

    cleanup () {
        if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
            kill "$VLLM_PID" 2>/dev/null || true
            sleep 3
            kill -9 "$VLLM_PID" 2>/dev/null || true
        fi
    }
    trap cleanup EXIT INT TERM

    echo "[$node seed $node_seed] launching vllm ..."
    vllm serve "$serve_model" \
        --port "$port" \
        --tensor-parallel-size "$tp" \
        --max-model-len "$max_model_len" \
        --gpu-memory-utilization "$gpu_mem_util" \
        --dtype "$dtype" \
        --trust-remote-code \
        --reasoning-parser "$reasoning_parser" \
        > "$vllm_log" 2>&1 &
    VLLM_PID=$!
    echo "[$node seed $node_seed] vllm pid: $VLLM_PID, log: $vllm_log"

    echo "[$node seed $node_seed] waiting for server (timeout ${ready_timeout}s) ..."
    local url="http://localhost:${port}/v1/models"
    for i in $(seq 1 "$ready_timeout"); do
        if curl -fsS "$url" > /dev/null 2>&1; then
            echo "[$node seed $node_seed] server ready after ${i}s"
            break
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "[$node seed $node_seed] vllm exited prematurely"
            tail -80 "$vllm_log" || true
            exit 1
        fi
        sleep 1
        if [[ "$i" == "$ready_timeout" ]]; then
            echo "[$node seed $node_seed] vllm never became ready"
            tail -80 "$vllm_log" || true
            exit 1
        fi
    done

    local sample_arg=()
    if [[ -n "$sample" ]]; then
        sample_arg=(--sample "$sample")
    fi

    echo "[$node seed $node_seed] rolling out trajectories (no judge) ..."
    python -m trajectory_generation.run \
        --dataset       "$dataset" \
        --output-dir    "$output_dir" \
        --model         "$model" \
        --server-url    "http://localhost:${port}/v1" \
        --concurrency   "$concurrency" \
        --max-solver-turns "$max_solver_turns" \
        --temperature   "$temperature" \
        --top-p         "$top_p" \
        --no-judge \
        --seed          "$node_seed" \
        "${sample_arg[@]}"

    echo "[$node seed $node_seed] done at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

PIDS=()
for i in $(seq 0 $((NUM_NODES - 1))); do
    NODE="${NODES[$i]}"
    NODE_SEED=$((JOB_SEED * 1000 + i))
    NODE_LOG="$LOG_DIR/traj_gen_${SLURM_JOB_ID}_seed${NODE_SEED}.log"
    srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus-per-node=4 \
         --nodelist="$NODE" --exclusive \
         --output="$NODE_LOG" \
         bash -c "$(declare -f run_stream_on_node); \
                  run_stream_on_node '$NODE' '$NODE_SEED' \
                                     '$REPO' '$VENV' '$DATASET' '$OUTPUT_DIR' \
                                     '$SAMPLE' '$CONCURRENCY' '$MAX_SOLVER_TURNS' \
                                     '$LOG_DIR' '$MODEL' \
                                     '$SERVE_MODEL' '$PORT' '$TP' '$MAX_MODEL_LEN' \
                                     '$GPU_MEM_UTIL' '$REASONING_PARSER' '$DTYPE' \
                                     '$READY_TIMEOUT' '$TEMPERATURE' '$TOP_P'" &
    PIDS+=($!)
done

overall_rc=0
for pid in "${PIDS[@]}"; do
    wait "$pid"
    rc=$?
    if [[ $rc -ne 0 ]]; then overall_rc=$rc; fi
done

echo "=========================================="
echo "all streams exited at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "overall rc       : $overall_rc"
echo "trajectory files : $(find "$OUTPUT_DIR" -maxdepth 1 -name '*.json' ! -name '*.debug.json' 2>/dev/null | wc -l)"
echo "=========================================="

exit $overall_rc
