#!/bin/bash
# Per-node ssh launcher for trajectory generation in field-mode.
# Reads $PBS_NODEFILE for allocated nodes and $FIELDS env var for the list of
# human-form field names (one per node, newline-separated). For each
# (node_i, field_i) pair, ssh into node_i and start its own vllm server +
# pipeline run via hpc/run_traj_gen.sh.
#
# Designed to be invoked from a PBS batch script. Output goes to the shared
# trajectories directory ($OUTPUT_DIR); the pipeline self-skips already-done
# trajectories so resume after preemption is automatic.

set -uo pipefail

REPO=${REPO:-/home/tommicaste/projects/SynthTools}
VENV=${VENV:-/lus/eagle/projects/CausalAlign/tommicaste/envs/synthtools/bin/activate}
TOOL_CONTENT=${TOOL_CONTENT:-/lus/eagle/projects/CausalAlign/tommicaste/tool_content}
OUTPUT_DIR=${OUTPUT_DIR:-$TOOL_CONTENT/trajectories}
RUNLOG_BASE=${RUNLOG_BASE:-$TOOL_CONTENT/_runlogs}

SERVE_MODEL=${SERVE_MODEL:-openai/gpt-oss-120b}
PIPELINE_MODEL=${PIPELINE_MODEL:-GPT-OSS-120B}
TP=${TP:-4}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
CONCURRENCY=${CONCURRENCY:-10}
PORT=${PORT:-8765}
VLLM_LOG_LEVEL=${VLLM_LOG_LEVEL:-INFO}
MAX_TRAJ_PER_SPEC=${MAX_TRAJ_PER_SPEC:-10}

ENV_FILE=${ENV_FILE:-/home/tommicaste/projects/SynthTools/.env}
if [[ -f "$ENV_FILE" ]]; then
  set -a; source "$ENV_FILE"; set +a
fi

if [[ -z "${PBS_NODEFILE:-}" ]]; then
  echo "ERROR: PBS_NODEFILE not set. Run inside a PBS allocation." >&2; exit 1
fi
if [[ -z "${FIELDS:-}" ]]; then
  echo "ERROR: FIELDS env var not set (newline-separated field names)." >&2; exit 1
fi
if [[ ! -f "$VENV" ]]; then
  echo "ERROR: venv activate not found at $VENV" >&2; exit 1
fi

mapfile -t NODES < <(sort -u "$PBS_NODEFILE")
mapfile -t FIELDS_ARR < <(printf '%s\n' "$FIELDS" | sed '/^$/d')

N=${#NODES[@]}
(( ${#FIELDS_ARR[@]} < N )) && N=${#FIELDS_ARR[@]}

JOBID_RAW=${PBS_JOBID:-local-$$}
JOBID=${JOBID_RAW%%.*}
JOBNAME=${PBS_JOBNAME:-traj_batch}
RUNLOG_DIR="$RUNLOG_BASE/${JOBNAME}_${JOBID}"
mkdir -p "$OUTPUT_DIR" "$RUNLOG_DIR"

echo "============================================================"
echo "run_traj_gen_field_per_node"
echo "started     : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "head node   : $(hostname)"
echo "PBS jobid   : $JOBID  name: $JOBNAME"
echo "nodes       : $N"
printf '  %s\n' "${NODES[@]:0:$N}"
echo "fields      : $N"
printf '  %s\n' "${FIELDS_ARR[@]:0:$N}"
echo "output dir  : $OUTPUT_DIR  (shared)"
echo "runlog dir  : $RUNLOG_DIR"
echo "model       : $SERVE_MODEL  (pipeline key: $PIPELINE_MODEL)"
echo "tp/max_len  : $TP / $MAX_MODEL_LEN"
echo "concurrency : $CONCURRENCY  per node"
echo "vllm port   : $PORT  (per-node localhost; no cross-node conflict)"
echo "============================================================"

cleanup() {
  echo "[cleanup] propagating kill to remote vllm/run_traj_gen on each node"
  for node in "${NODES[@]:0:$N}"; do
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$node" \
      'pkill -f "vllm serve|run_traj_gen.sh" 2>/dev/null; \
       sleep 2; pkill -9 -f "vllm serve|run_traj_gen.sh" 2>/dev/null' &
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

PIDS=()
LOGS=()
for ((i=0; i<N; i++)); do
  node="${NODES[$i]}"
  field="${FIELDS_ARR[$i]}"
  short_host="${node%%.*}"
  log="$RUNLOG_DIR/node${i}_${short_host}.log"
  vllm_log_dir="$RUNLOG_DIR/node${i}_${short_host}_vllm"
  mkdir -p "$vllm_log_dir"

  echo "[node${i}] $node"
  echo "         field=\"$field\""
  echo "         log=$log"

  field_q=$(printf '%q' "$field")

  ssh -o StrictHostKeyChecking=no -o BatchMode=yes "$node" \
    "export VLLM_LOGGING_LEVEL=$VLLM_LOG_LEVEL PYTHONUNBUFFERED=1 \
            NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1 \
            HF_HOME=/lus/eagle/projects/CausalAlign/tommicaste/.cache/huggingface \
            HF_TOKEN='${HF_TOKEN:-}' && \
     cd $REPO && bash hpc/run_traj_gen.sh \
       --field $field_q \
       --concurrency $CONCURRENCY \
       --max-trajectories $MAX_TRAJ_PER_SPEC \
       --dataset '$TOOL_CONTENT/tools_dataset.jsonl' \
       --env-specs-dir '$TOOL_CONTENT/env_specs' \
       --output-dir '$OUTPUT_DIR' \
       --venv '$VENV' \
       --log-dir '$vllm_log_dir' \
       --serve-model '$SERVE_MODEL' \
       --port $PORT \
       --tp $TP \
       --max-model-len $MAX_MODEL_LEN \
       --gpu-memory-utilization $GPU_MEM_UTIL \
       --model '$PIPELINE_MODEL'" \
    > "$log" 2>&1 &
  PIDS+=($!)
  LOGS+=("$log")
done

echo
echo "Launched $N ssh jobs."
echo

RC=0
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    echo "[node${i}] FAILED — see ${LOGS[$i]}"
    RC=1
  else
    echo "[node${i}] done"
  fi
done

echo
echo "ended       : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "overall rc  : $RC"
exit $RC
