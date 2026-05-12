#!/bin/bash
# Launch one run_traj_gen.sh per allocated PBS node.
# Each node spins up its own vllm server and runs the pipeline against it.
# Run this inside an active PBS allocation (interactive or batch).
#
# Defaults are tuned for a small smoke test: 1 env_spec per node,
# --max-tasks 2, --concurrency 2.
#
# Override via env vars (see "Tunables" below).

set -uo pipefail

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
REPO=${REPO:-/home/tommicaste/projects/SynthTools}
VENV=${VENV:-/lus/eagle/projects/CausalAlign/tommicaste/envs/synthtools/bin/activate}
TOOL_CONTENT=${TOOL_CONTENT:-/lus/eagle/projects/CausalAlign/tommicaste/tool_content}
SERVE_MODEL=${SERVE_MODEL:-openai/gpt-oss-120b}
PIPELINE_MODEL=${PIPELINE_MODEL:-GPT-OSS-120B}
TP=${TP:-4}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
CONCURRENCY=${CONCURRENCY:-2}
MAX_TASKS=${MAX_TASKS:-2}

# One env_spec per node. Add more entries if you allocate >2 nodes.
SPECS_DEFAULT=(
  "$TOOL_CONTENT/env_specs/academic_publishing_and_citations_spec_000.json"
  "$TOOL_CONTENT/env_specs/academic_publishing_and_citations_spec_001.json"
  "$TOOL_CONTENT/env_specs/academic_publishing_and_citations_spec_002.json"
  "$TOOL_CONTENT/env_specs/academic_publishing_and_citations_spec_003.json"
)
# If SPECS env var is set (whitespace-separated list), it overrides the default.
if [[ -n "${SPECS:-}" ]]; then
  read -r -a SPECS_ARR <<< "$SPECS"
else
  SPECS_ARR=("${SPECS_DEFAULT[@]}")
fi

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if [[ -z "${PBS_NODEFILE:-}" ]]; then
  echo "ERROR: PBS_NODEFILE not set. Run this inside a PBS allocation." >&2
  exit 1
fi
if [[ ! -f "$VENV" ]]; then
  echo "ERROR: venv activate not found at $VENV" >&2
  exit 1
fi
if [[ ! -d "$REPO" ]]; then
  echo "ERROR: repo dir not found: $REPO" >&2
  exit 1
fi

mapfile -t NODES < <(sort -u "$PBS_NODEFILE")
N=${#NODES[@]}
if (( N > ${#SPECS_ARR[@]} )); then
  echo "ERROR: $N nodes but only ${#SPECS_ARR[@]} specs in SPECS list. Add more." >&2
  exit 1
fi

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$TOOL_CONTENT/test_runs/$STAMP"
mkdir -p "$RUN_DIR"

echo "============================================================"
echo "run_traj_gen_per_node"
echo "started     : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "head node   : $(hostname)"
echo "nodes       : $N"
printf '  %s\n' "${NODES[@]}"
echo "run dir     : $RUN_DIR"
echo "model       : $SERVE_MODEL  (pipeline key: $PIPELINE_MODEL)"
echo "tp / max_len: $TP / $MAX_MODEL_LEN"
echo "concurrency : $CONCURRENCY  per node"
echo "max tasks  : $MAX_TASKS per spec"
echo "============================================================"

# ---------------------------------------------------------------------------
# Launch one run_traj_gen.sh per node via ssh
# ---------------------------------------------------------------------------
PIDS=()
LOGS=()
OUTS=()
for i in "${!NODES[@]}"; do
  node="${NODES[$i]}"
  spec="${SPECS_ARR[$i]}"
  out="$RUN_DIR/node${i}_${node}"
  log="$RUN_DIR/node${i}_${node}.log"
  port=$((8765 + i))
  mkdir -p "$out"

  echo "[node${i}] $node  spec=$(basename "$spec")  port=$port"
  echo "         out=$out"
  echo "         log=$log"

  ssh -o StrictHostKeyChecking=no -o BatchMode=yes "$node" \
    "export VLLM_LOGGING_LEVEL=DEBUG PYTHONUNBUFFERED=1 && \
     cd $REPO && bash hpc/run_traj_gen.sh \
      --env-spec '$spec' \
      --max-tasks $MAX_TASKS \
      --concurrency $CONCURRENCY \
      --dataset '$TOOL_CONTENT/tools_dataset.jsonl' \
      --env-specs-dir '$TOOL_CONTENT/env_specs' \
      --output-dir '$out' \
      --venv '$VENV' \
      --log-dir '$out/_vllm_log' \
      --serve-model '$SERVE_MODEL' \
      --port $port \
      --tp $TP \
      --max-model-len $MAX_MODEL_LEN \
      --gpu-memory-utilization $GPU_MEM_UTIL \
      --model '$PIPELINE_MODEL'" \
    > "$log" 2>&1 &

  PIDS+=($!)
  LOGS+=("$log")
  OUTS+=("$out")
done

echo
echo "Launched $N ssh jobs. Tail any of:"
for log in "${LOGS[@]}"; do echo "  tail -f $log"; done
echo

# ---------------------------------------------------------------------------
# Wait for all and report
# ---------------------------------------------------------------------------
RC=0
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    echo "[node${i}] FAILED (see ${LOGS[$i]})"
    RC=1
  else
    echo "[node${i}] done"
  fi
done

echo
echo "=== Per-node trajectory counts ==="
for i in "${!NODES[@]}"; do
  out="${OUTS[$i]}"
  count=$(ls "$out"/*.json 2>/dev/null | grep -v debug | wc -l)
  echo "node${i} ${NODES[$i]}: $count tasks  ($out)"
done
echo
echo "ended       : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "overall rc  : $RC"
exit $RC
