#!/bin/bash
#SBATCH -t 06:00:00
#SBATCH --account=m5091
#SBATCH --job-name=ta-resum
#SBATCH --output=/global/homes/t/tcaste/projects/logs/ta_resum_%A_%a.out
#SBATCH --error=/global/homes/t/tcaste/projects/logs/ta_resum_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --constraint "gpu&hbm40g"
#SBATCH --qos=premium
#SBATCH --array=0-4

# ---------------------------------------------------------------------------
# Resummarise the 80k task corpus across 5 concurrent shards.
#
# Each array task is one independent node with 4× A100 80GB. It launches its
# own vLLM server (port 8765 on its own node, no cross-node coordination)
# and runs `task_audit.run` with --shard $SLURM_ARRAY_TASK_ID --num-shards 5.
# All 5 shards APPEND to the SAME task_content.jsonl under fcntl.flock —
# this is the path on /pscratch (Lustre); flock is supported. Disjoint task
# slicing means no two shards ever write the same row.
#
# Submit:
#   sbatch hpc/bash_files_perlmutter/task_audit_resummarize.sbatch
#
# Resume (if a shard times out, just resubmit JUST that shard array task):
#   sbatch --array=2 hpc/bash_files_perlmutter/task_audit_resummarize.sbatch
#   # With --resummarize OFF the resumed shard would skip everything; with
#   # --resummarize ON it redoes the shard's work. To do a clean re-run of
#   # one shard after a partial completion, see the IMPORTANT note below.
#
# IMPORTANT: --resummarize means tasks with an existing summary are
# re-summarised. If a shard times out partway, on resubmit the same shard
# will start over (re-doing the tasks it already finished). The JSONL still
# gets duplicate id rows (consumer deduplicates last-write-wins), and the
# task JSONs get their summary block overwritten — no corruption, just
# wasted compute. To avoid that, replace --resummarize with no flag on
# retries (skip-if-summary-present then takes effect).
# ---------------------------------------------------------------------------

set -euo pipefail

# --- paths ---
REPO=/global/homes/t/tcaste/projects/SynthTools
VENV=/pscratch/sd/t/tcaste/envs/burn-gpu/bin/activate

TASKS_DIR=/pscratch/sd/t/tcaste/tool_content/tasks
ENV_SPECS_DIR=/pscratch/sd/t/tcaste/tool_content/env_specs
TASK_CONTENT=/pscratch/sd/t/tcaste/tool_content/task_content.jsonl

LOG_DIR=/global/homes/t/tcaste/projects/logs
mkdir -p "$LOG_DIR"

# --- vLLM ---
SERVE_MODEL=${SERVE_MODEL:-openai/gpt-oss-120b}
PORT=${PORT:-8765}
TP=${TP:-4}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
REASONING_PARSER=${REASONING_PARSER:-openai_gptoss}
DTYPE=${DTYPE:-bfloat16}
READY_TIMEOUT=${READY_TIMEOUT:-1800}

# --- pipeline ---
MODEL=${MODEL:-GPT-OSS-120B}
NUM_SHARDS=5
SHARD=${SLURM_ARRAY_TASK_ID:-0}

# --- env ---
cd "$REPO"
# shellcheck disable=SC1090
source "$VENV"
module load cudatoolkit/12.9

VLLM_LOG="$LOG_DIR/ta_resum_vllm_${SLURM_JOB_ID}_shard${SHARD}.log"

echo "=========================================="
echo "task_audit resummarize — shard $SHARD/$NUM_SHARDS"
echo "host           : $(hostname)"
echo "started        : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "tasks dir      : $TASKS_DIR"
echo "env specs dir  : $ENV_SPECS_DIR"
echo "task content   : $TASK_CONTENT  (all 5 shards append here)"
echo "vllm model     : $SERVE_MODEL"
echo "port / tp      : $PORT / $TP"
echo "vllm log       : $VLLM_LOG"
echo "=========================================="

# ---------------------------------------------------------------------------
# 1. Launch vLLM on this node
# ---------------------------------------------------------------------------

cleanup() {
    if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[cleanup] killing vllm (pid $VLLM_PID)"
        kill "$VLLM_PID" 2>/dev/null || true
        sleep 3
        kill -9 "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "[step 1] launching vllm serve ..."
vllm serve "$SERVE_MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --dtype "$DTYPE" \
    --trust-remote-code \
    --reasoning-parser "$REASONING_PARSER" \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "[step 1] vllm pid: $VLLM_PID"

# ---------------------------------------------------------------------------
# 2. Wait for server ready
# ---------------------------------------------------------------------------

echo "[step 2] waiting for server (timeout ${READY_TIMEOUT}s) ..."
URL="http://localhost:${PORT}/v1/models"
for i in $(seq 1 "$READY_TIMEOUT"); do
    if curl -fsS "$URL" > /dev/null 2>&1; then
        echo "[step 2] server ready after ${i}s"
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[step 2] vllm exited prematurely. last 80 lines of log:"
        tail -80 "$VLLM_LOG" || true
        exit 1
    fi
    sleep 1
    if [[ "$i" == "$READY_TIMEOUT" ]]; then
        echo "[step 2] vllm never became ready within ${READY_TIMEOUT}s"
        tail -80 "$VLLM_LOG" || true
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# 3. Run this shard's task_audit
# ---------------------------------------------------------------------------

echo "[step 3] task_audit shard $SHARD/$NUM_SHARDS"
python -m task_audit.run \
    --tasks-dir     "$TASKS_DIR" \
    --env-specs-dir "$ENV_SPECS_DIR" \
    --task-content  "$TASK_CONTENT" \
    --model         "$MODEL" \
    --server-url    "http://localhost:${PORT}/v1" \
    --resummarize \
    --shard         "$SHARD" \
    --num-shards    "$NUM_SHARDS"

echo "=========================================="
echo "shard $SHARD finished : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "task_content rows now: $(wc -l < "$TASK_CONTENT" 2>/dev/null || echo 0)"
echo "=========================================="
