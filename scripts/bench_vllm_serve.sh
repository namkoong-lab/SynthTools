#!/bin/bash
# Start a vLLM OpenAI-compatible server, run the throughput benchmark, kill the server.
# Run inside an active salloc with --gpus-per-node=4. Assumes the venv is activated
# OR sourced inside the script.
#
# Usage:
#   bash scripts/bench_vllm_serve.sh [MODEL_ID] [PORT]
#
# Defaults:
#   MODEL_ID = openai/gpt-oss-120b
#   PORT     = 8765   (avoid 8000 to dodge anything else listening)
#
# Output:
#   /tmp/bench_vllm_serve.log   (vllm server log)
#   /tmp/bench_results.json     (benchmark results)

set -uo pipefail

MODEL_ID=${1:-openai/gpt-oss-120b}
PORT=${2:-8765}
TP=${TP:-4}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}

VENV=/pscratch/sd/t/tcaste/envs/burn-gpu/bin/activate
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

LOG=/tmp/bench_vllm_serve.log
OUT=/tmp/bench_results.json

if [[ ! -f "$VENV" ]]; then
    echo "venv not found at $VENV" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$VENV"

echo "=== bench_vllm_serve ==="
echo "host        : $(hostname)"
echo "model       : $MODEL_ID"
echo "port        : $PORT"
echo "tp_size     : $TP"
echo "max_model_len: $MAX_MODEL_LEN"
echo "gpu_mem_util : $GPU_MEM_UTIL"
echo "log         : $LOG"
echo "out         : $OUT"
echo "========================"

cleanup() {
    if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[cleanup] killing vllm (pid $VLLM_PID)"
        kill "$VLLM_PID" 2>/dev/null || true
        sleep 2
        kill -9 "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# Step 1: start vllm serve in the background.
# --reasoning-parser openai_gptoss is the GPT-OSS-specific Harmony parser.
# --enable-auto-tool-choice + --tool-call-parser are NOT enabled here
# because the pipeline uses plain JSON-in-content, not OpenAI-style tool calls.
# We just need to verify reasoning_content comes through.
echo "[step 1] launching vllm serve ..."
vllm serve "$MODEL_ID" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --dtype bfloat16 \
    --trust-remote-code \
    --reasoning-parser openai_gptoss \
    > "$LOG" 2>&1 &
VLLM_PID=$!
echo "[step 1] vllm pid: $VLLM_PID"

# Step 2: poll /v1/models until ready.
echo "[step 2] waiting for server to come up ..."
URL="http://localhost:${PORT}/v1/models"
for i in $(seq 1 600); do
    if curl -fsS "$URL" > /dev/null 2>&1; then
        echo "[step 2] server ready after ${i}s"
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[step 2] vllm process exited prematurely. last 60 lines of log:"
        tail -60 "$LOG"
        exit 2
    fi
    sleep 1
done

if ! curl -fsS "$URL" > /dev/null 2>&1; then
    echo "[step 2] timed out waiting for server. last 60 lines of log:"
    tail -60 "$LOG"
    exit 3
fi

# Step 3: run the benchmark.
echo "[step 3] running benchmark ..."
cd "$PROJ_DIR"
python -m scripts.bench_vllm_serve \
    --base-url "http://localhost:${PORT}/v1" \
    --model "$MODEL_ID" \
    --concurrency "1,4,16,32" \
    --max-tokens 512 \
    --warmup 2 \
    --out "$OUT"
RC=$?

echo "[done] benchmark rc=$RC; results at $OUT"
echo "[done] vllm log at $LOG"
exit $RC
