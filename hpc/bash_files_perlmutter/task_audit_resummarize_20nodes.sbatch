#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH --account=m5091
#SBATCH --job-name=ta-resum-20n
#SBATCH --output=/global/homes/t/tcaste/projects/logs/ta_resum_20n_%j.out
#SBATCH --error=/global/homes/t/tcaste/projects/logs/ta_resum_20n_%j.err
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --constraint "gpu&hbm40g"
#SBATCH --qos=premium

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

NUM_SHARDS=20

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
echo "=========================================="
echo "task_audit resummarize — 20 nodes, 1 shard per node"
echo "started  : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "job id   : $SLURM_JOB_ID"
echo "nodes    : ${NODES[*]}"
echo "tasks    : $TASKS_DIR"
echo "envspecs : $ENV_SPECS_DIR"
echo "output   : $TASK_CONTENT (all 20 shards append here)"
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

    local vllm_log="$log_dir/ta_resum_20n_${SLURM_JOB_ID}_shard${shard}_vllm.log"

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
    NODE_LOG="$LOG_DIR/ta_resum_20n_${SLURM_JOB_ID}_shard${i}.log"
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

# On success, chain the trajectory-generation jobs. Premium QOS allows 5
# concurrent submitted jobs; this summary job still occupies 1 slot while it
# submits, so we submit 4 trajectory jobs (1 + 4 = 5). The 4 jobs total 50
# nodes (1 x 20 nodes + 3 x 10 nodes) = 50 independent rollout streams, each
# with a distinct JOB_SEED so per-node seeds never collide.
#
# SUBMIT_TRAJ=0 skips chaining. TRAJ_SUBMIT_CMD=echo prints the submissions
# without submitting (dry-run the chaining).
SUBMIT_TRAJ=${SUBMIT_TRAJ:-1}
TRAJ_SUBMIT_CMD=${TRAJ_SUBMIT_CMD:-sbatch}
TRAJ_SBATCH=${TRAJ_SBATCH:-$REPO/hpc/bash_files_perlmutter/trajectory_generation_premium.sbatch}
TRAJ_OUTPUT_DIR=${TRAJ_OUTPUT_DIR:-/pscratch/sd/t/tcaste/tool_content/trajectories}

if [[ "$overall_rc" -eq 0 && "$SUBMIT_TRAJ" -eq 1 ]]; then
    mkdir -p "$TRAJ_OUTPUT_DIR"
    echo "chaining trajectory jobs -> $TRAJ_OUTPUT_DIR"
    common_export="ALL,OUTPUT_DIR=${TRAJ_OUTPUT_DIR},DATASET=${TASK_CONTENT}"
    # job 1: 20 nodes, seed base 1 (node seeds 1000..1019)
    $TRAJ_SUBMIT_CMD --nodes=20 --export="${common_export},JOB_SEED=1" "$TRAJ_SBATCH"
    # jobs 2-4: 10 nodes each, seed bases 2..4 (node seeds 2000.., 3000.., 4000..)
    for seed in 2 3 4; do
        $TRAJ_SUBMIT_CMD --nodes=10 --export="${common_export},JOB_SEED=${seed}" "$TRAJ_SBATCH"
    done
    echo "submitted 4 trajectory jobs (20+10+10+10 = 50 nodes)"
else
    echo "skipping trajectory chaining (overall_rc=$overall_rc, SUBMIT_TRAJ=$SUBMIT_TRAJ)"
fi

exit $overall_rc
