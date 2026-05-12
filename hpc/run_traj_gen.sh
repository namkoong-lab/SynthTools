#!/bin/bash
# Run task generation against a freshly-launched vLLM server.
#
# Usage:
#   bash hpc/run_traj_gen.sh --field "Investment Banking" --concurrency 5
#
# All flags below have sensible defaults for Perlmutter (NERSC), but every
# value is overridable so this script works on any cluster as long as the
# venv has vllm + openai installed.
#
# Run this *inside an active allocation* (salloc'd compute node with the
# requested GPUs). It launches vllm serve in the background, waits for it
# to become ready, runs the task generation pipeline, then cleans
# the server up on exit.

set -uo pipefail

# ---------------------------------------------------------------------------
# Defaults (override with --flag value)
# ---------------------------------------------------------------------------

# vLLM server
SERVE_MODEL=${SERVE_MODEL:-openai/gpt-oss-120b}
PORT=${PORT:-8765}
TP=${TP:-4}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
REASONING_PARSER=${REASONING_PARSER:-openai_gptoss}
DTYPE=${DTYPE:-bfloat16}

# Pipeline
MODEL=${MODEL:-GPT-OSS-120B}
FIELD=""
ENV_SPEC=""
CONCURRENCY=${CONCURRENCY:-5}
MAX_TASKS=""
MAX_SOLVER_TURNS=${MAX_SOLVER_TURNS:-5}
MAX_RETRIES=${MAX_RETRIES:-5}
VERIFIABLE=${VERIFIABLE:-1}

# Paths (Perlmutter defaults; override on other clusters)
DATASET=${DATASET:-/pscratch/sd/t/tcaste/tool_content/tools_dataset.jsonl}
ENV_SPECS_DIR=${ENV_SPECS_DIR:-/pscratch/sd/t/tcaste/tool_content/env_specs}
OUTPUT_DIR=${OUTPUT_DIR:-/pscratch/sd/t/tcaste/tool_content/tasks}
VENV=${VENV:-/pscratch/sd/t/tcaste/envs/burn-gpu/bin/activate}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR=${LOG_DIR:-/tmp}

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: bash $0 [options]

Required (one of):
  --field FIELD                  Mode C: process every spec in FIELD (e.g. "Investment Banking")
  --env-spec PATH                Mode B: process a single env_spec JSON

Optional (defaults shown in [...]):
  --concurrency N                Number of parallel workers     [$CONCURRENCY]
  --max-tasks N           Cap tasks per spec      [unlimited]
  --max-solver-turns N           Max turns per task             [$MAX_SOLVER_TURNS]
  --max-retries N                Max evolver re-rolls           [$MAX_RETRIES]
  --no-verifiable                Disable task judging
  --no-debug                     Disable debug event log

vLLM server:
  --serve-model NAME             vllm model id                  [$SERVE_MODEL]
  --port N                       Port                           [$PORT]
  --tp N                         Tensor parallel size           [$TP]
  --max-model-len N              Max model length               [$MAX_MODEL_LEN]
  --gpu-memory-utilization F     KV-cache budget                [$GPU_MEM_UTIL]
  --reasoning-parser NAME        Reasoning parser               [$REASONING_PARSER]
  --dtype NAME                   Dtype                          [$DTYPE]

Pipeline model:
  --model NAME                   MODEL_REGISTRY key             [$MODEL]

Paths (cluster-specific):
  --dataset PATH                 tools_dataset.jsonl            [$DATASET]
  --env-specs-dir PATH           Directory of env_spec JSONs    [$ENV_SPECS_DIR]
  --output-dir PATH              Task output directory    [$OUTPUT_DIR]
  --venv PATH                    Path to bin/activate           [$VENV]
  --log-dir PATH                 Where to put vllm log          [$LOG_DIR]

Every flag above is also settable via env var of the same name (uppercase,
underscores). Run inside an active GPU allocation.
EOF
}

NO_DEBUG=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --field)                  FIELD="$2"; shift 2;;
        --env-spec)               ENV_SPEC="$2"; shift 2;;
        --concurrency)            CONCURRENCY="$2"; shift 2;;
        --max-tasks)       MAX_TASKS="$2"; shift 2;;
        --max-solver-turns)       MAX_SOLVER_TURNS="$2"; shift 2;;
        --max-retries)            MAX_RETRIES="$2"; shift 2;;
        --no-verifiable)          VERIFIABLE=0; shift;;
        --no-debug)               NO_DEBUG=1; shift;;
        --serve-model)            SERVE_MODEL="$2"; shift 2;;
        --port)                   PORT="$2"; shift 2;;
        --tp)                     TP="$2"; shift 2;;
        --max-model-len)          MAX_MODEL_LEN="$2"; shift 2;;
        --gpu-memory-utilization) GPU_MEM_UTIL="$2"; shift 2;;
        --reasoning-parser)       REASONING_PARSER="$2"; shift 2;;
        --dtype)                  DTYPE="$2"; shift 2;;
        --model)                  MODEL="$2"; shift 2;;
        --dataset)                DATASET="$2"; shift 2;;
        --env-specs-dir)          ENV_SPECS_DIR="$2"; shift 2;;
        --output-dir)             OUTPUT_DIR="$2"; shift 2;;
        --venv)                   VENV="$2"; shift 2;;
        --log-dir)                LOG_DIR="$2"; shift 2;;
        -h|--help)                usage; exit 0;;
        *)                        echo "Unknown flag: $1"; usage; exit 1;;
    esac
done

if [[ -z "$FIELD" && -z "$ENV_SPEC" ]]; then
    echo "Error: --field or --env-spec is required." >&2
    usage; exit 1
fi
if [[ -n "$FIELD" && -n "$ENV_SPEC" ]]; then
    echo "Error: pass --field OR --env-spec, not both." >&2
    exit 1
fi
if [[ ! -f "$VENV" ]]; then
    echo "Error: venv activate not found at $VENV" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$VENV"

VLLM_LOG="$LOG_DIR/vllm_serve_${PORT}.log"
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "run_traj_gen"
echo "host             : $(hostname)"
echo "started          : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "------------------------------------------------------------"
echo "vllm model       : $SERVE_MODEL"
echo "port             : $PORT"
echo "tp / max_len     : $TP / $MAX_MODEL_LEN"
echo "gpu_mem_util     : $GPU_MEM_UTIL"
echo "reasoning parser : $REASONING_PARSER"
echo "------------------------------------------------------------"
echo "pipeline model   : $MODEL"
[[ -n "$FIELD" ]]    && echo "field            : $FIELD"
[[ -n "$ENV_SPEC" ]] && echo "env-spec         : $ENV_SPEC"
echo "concurrency      : $CONCURRENCY"
[[ -n "$MAX_TASKS" ]] && echo "max tasks : $MAX_TASKS"
echo "verifiable       : $VERIFIABLE   debug : $((1-NO_DEBUG))"
echo "------------------------------------------------------------"
echo "dataset          : $DATASET"
echo "env_specs_dir    : $ENV_SPECS_DIR"
echo "output_dir       : $OUTPUT_DIR"
echo "venv             : $VENV"
echo "vllm log         : $VLLM_LOG"
echo "============================================================"

# ---------------------------------------------------------------------------
# Step 1: launch vllm serve in the background
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
# Step 2: wait for the server to become ready
# ---------------------------------------------------------------------------

echo "[step 2] waiting for server to come up ..."
URL="http://localhost:${PORT}/v1/models"
READY_TIMEOUT=${READY_TIMEOUT:-1800}
for i in $(seq 1 "$READY_TIMEOUT"); do
    if curl -fsS "$URL" > /dev/null 2>&1; then
        echo "[step 2] server ready after ${i}s"
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[step 2] vllm exited prematurely. last 80 lines of log:"
        tail -80 "$VLLM_LOG"
        exit 2
    fi
    sleep 1
done
if ! curl -fsS "$URL" > /dev/null 2>&1; then
    echo "[step 2] timed out after ${READY_TIMEOUT}s. last 80 lines:"
    tail -80 "$VLLM_LOG"
    exit 3
fi

# ---------------------------------------------------------------------------
# Step 3: run task generation
# ---------------------------------------------------------------------------

cd "$PROJ_DIR"

PIPELINE_ARGS=(
    --dataset "$DATASET"
    --output-dir "$OUTPUT_DIR"
    --model "$MODEL"
    --server-url "http://localhost:${PORT}/v1"
    --concurrency "$CONCURRENCY"
    --max-solver-turns "$MAX_SOLVER_TURNS"
    --max-retries "$MAX_RETRIES"
)
if [[ -n "$FIELD" ]]; then
    PIPELINE_ARGS+=(--env-specs-dir "$ENV_SPECS_DIR" --field "$FIELD")
else
    PIPELINE_ARGS+=(--env-spec "$ENV_SPEC")
fi
if [[ -n "$MAX_TASKS" ]]; then
    PIPELINE_ARGS+=(--max-tasks "$MAX_TASKS")
fi
if [[ "$VERIFIABLE" == "1" ]]; then
    PIPELINE_ARGS+=(--verifiable)
fi
if [[ "$NO_DEBUG" == "1" ]]; then
    PIPELINE_ARGS+=(--no-debug)
fi

echo "[step 3] running task_generation.run ..."
python -m task_generation.run "${PIPELINE_ARGS[@]}"
RC=$?

echo "============================================================"
echo "ended            : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "pipeline rc      : $RC"
echo "tasks            : $(ls "$OUTPUT_DIR"/*.json 2>/dev/null | grep -v debug | wc -l)"
echo "vllm log         : $VLLM_LOG"
echo "============================================================"
exit $RC
