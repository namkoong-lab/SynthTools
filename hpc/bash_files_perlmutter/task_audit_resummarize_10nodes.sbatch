#!/bin/bash
#SBATCH -t 04:00:00
#SBATCH --account=m5091
#SBATCH --job-name=ta-resum-10n
#SBATCH --output=/global/homes/t/tcaste/projects/logs/ta_resum_10n_%j.out
#SBATCH --error=/global/homes/t/tcaste/projects/logs/ta_resum_10n_%j.err
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --constraint "gpu&hbm40g"
#SBATCH --qos=premium

# 10-node single-job version: each node runs vLLM locally + one shard of
# task_audit.run. All 10 shards append to the same task_content.jsonl
# under fcntl.flock. Walltime 4h to fit before the 8am SF maintenance.

set -euo pipefail

REPO=/global/homes/t/tcaste/projects/SynthTools
VENV=/pscratch/sd/t/tcaste/envs/burn-gpu/bin/activate

TASKS_DIR=/pscratch/sd/t/tcaste/tool_content/tasks
ENV_SPECS_DIR=/pscratch/sd/t/tcaste/tool_content/env_specs
TASK_CONTENT=/pscratch/sd/t/tcaste/tool_content/task_content.jsonl

LOG_DIR=/global/homes/t/tcaste/projects/logs
mkdir -p "$LOG_DIR"

SERVE_MODEL=${SERVE_MODEL:-openai/gpt-oss-120b}
PORT=${PORT:-8765}
TP=${TP:-4}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
REASONING_PARSER=${REASONING_PARSER:-openai_gptoss}
DTYPE=${DTYPE:-bfloat16}
READY_TIMEOUT=${READY_TIMEOUT:-1800}
MODEL=${MODEL:-GPT-OSS-120B}

NUM_SHARDS=10

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
echo "=========================================="
echo "task_audit resummarize — 10 nodes, 1 shard per node"
echo "started  : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "job id   : $SLURM_JOB_ID"
echo "nodes    : ${NODES[*]}"
echo "tasks    : $TASKS_DIR"
echo "envspecs : $ENV_SPECS_DIR"
echo "output   : $TASK_CONTENT (all 10 shards append here)"
echo "=========================================="

run_shard_on_node () {
    local node=$1
    local shard=$2
    local num_shards=$3
    local repo=$4
    local venv=$5
    local tasks_dir=$6
    local env_specs_dir=$7
    local task_content=$8
    local log_dir=$9
    local model=${10}
    local serve_model=${11}
    local port=${12}
    local tp=${13}
    local max_model_len=${14}
    local gpu_mem_util=${15}
    local reasoning_parser=${16}
    local dtype=${17}
    local ready_timeout=${18}

    cd "$repo"
    # shellcheck disable=SC1090
    source "$venv"
    module load cudatoolkit/12.9

    local vllm_log="$log_dir/ta_resum_10n_${SLURM_JOB_ID}_shard${shard}_vllm.log"

    cleanup () {
        if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
            kill "$VLLM_PID" 2>/dev/null || true
            sleep 3
            kill -9 "$VLLM_PID" 2>/dev/null || true
        fi
    }
    trap cleanup EXIT INT TERM

    echo "[$node shard $shard] launching vllm ..."
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
    echo "[$node shard $shard] vllm pid: $VLLM_PID, log: $vllm_log"

    echo "[$node shard $shard] waiting for server (timeout ${ready_timeout}s) ..."
    local url="http://localhost:${port}/v1/models"
    for i in $(seq 1 "$ready_timeout"); do
        if curl -fsS "$url" > /dev/null 2>&1; then
            echo "[$node shard $shard] server ready after ${i}s"
            break
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "[$node shard $shard] vllm exited prematurely"
            tail -80 "$vllm_log" || true
            exit 1
        fi
        sleep 1
        if [[ "$i" == "$ready_timeout" ]]; then
            echo "[$node shard $shard] vllm never became ready"
            tail -80 "$vllm_log" || true
            exit 1
        fi
    done

    echo "[$node shard $shard] running task_audit shard $shard/$num_shards"
    python -m task_audit.run \
        --tasks-dir     "$tasks_dir" \
        --env-specs-dir "$env_specs_dir" \
        --task-content  "$task_content" \
        --model         "$model" \
        --server-url    "http://localhost:${port}/v1" \
        --resummarize \
        --shard         "$shard" \
        --num-shards    "$num_shards"

    echo "[$node shard $shard] done at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

PIDS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    NODE="${NODES[$i]}"
    NODE_LOG="$LOG_DIR/ta_resum_10n_${SLURM_JOB_ID}_shard${i}.log"
    srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus-per-node=4 \
         --nodelist="$NODE" --exclusive \
         --output="$NODE_LOG" \
         bash -c "$(declare -f run_shard_on_node); \
                  run_shard_on_node '$NODE' '$i' '$NUM_SHARDS' \
                                    '$REPO' '$VENV' \
                                    '$TASKS_DIR' '$ENV_SPECS_DIR' '$TASK_CONTENT' \
                                    '$LOG_DIR' '$MODEL' \
                                    '$SERVE_MODEL' '$PORT' '$TP' '$MAX_MODEL_LEN' \
                                    '$GPU_MEM_UTIL' '$REASONING_PARSER' '$DTYPE' \
                                    '$READY_TIMEOUT'" &
    PIDS+=($!)
done

overall_rc=0
for pid in "${PIDS[@]}"; do
    wait "$pid"
    rc=$?
    if [[ $rc -ne 0 ]]; then overall_rc=$rc; fi
done

echo "=========================================="
echo "all shards exited at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "overall rc: $overall_rc"
echo "task_content rows: $(wc -l < "$TASK_CONTENT" 2>/dev/null || echo 0)"
echo "=========================================="

exit $overall_rc
