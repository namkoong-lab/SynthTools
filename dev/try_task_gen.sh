#!/bin/bash
# Run task generation against a vLLM server that is ALREADY running.
# Use this to smoke-check the iterative running_summary pipeline.
#
# Usage:
#   bash dev/try_task_gen.sh --env-spec /path/to/spec.json
#   bash dev/try_task_gen.sh --field "Investment Banking"
#
# Defaults assume vllm serve is up at http://localhost:8765/v1.

set -uo pipefail

SERVER_URL=${SERVER_URL:-http://localhost:8765/v1}
MODEL=${MODEL:-GPT-OSS-120B}
DATASET=${DATASET:-/pscratch/sd/t/tcaste/tool_content/tools_dataset.jsonl}
ENV_SPECS_DIR=${ENV_SPECS_DIR:-/pscratch/sd/t/tcaste/tool_content/env_specs}
OUTPUT_DIR=${OUTPUT_DIR:-/pscratch/sd/t/tcaste/tool_content/dev_new_gen}
CONCURRENCY=${CONCURRENCY:-1}
MAX_TASKS=${MAX_TASKS:-1}
MAX_SOLVER_TURNS=${MAX_SOLVER_TURNS:-5}
MAX_RETRIES=${MAX_RETRIES:-3}
VERIFIABLE=${VERIFIABLE:-1}
ENV_SPEC=""
FIELD=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --server-url)     SERVER_URL=$2; shift 2;;
        --model)          MODEL=$2; shift 2;;
        --dataset)        DATASET=$2; shift 2;;
        --env-specs-dir)  ENV_SPECS_DIR=$2; shift 2;;
        --env-spec)       ENV_SPEC=$2; shift 2;;
        --field)          FIELD=$2; shift 2;;
        --output-dir)     OUTPUT_DIR=$2; shift 2;;
        --concurrency)    CONCURRENCY=$2; shift 2;;
        --max-tasks)      MAX_TASKS=$2; shift 2;;
        --max-solver-turns) MAX_SOLVER_TURNS=$2; shift 2;;
        --max-retries)    MAX_RETRIES=$2; shift 2;;
        --verifiable)     VERIFIABLE=1; shift 1;;
        --no-verifiable)  VERIFIABLE=0; shift 1;;
        -h|--help)        sed -n '2,12p' "$0"; exit 0;;
        *) echo "Unknown flag: $1" >&2; exit 2;;
    esac
done

if [[ -z "$ENV_SPEC" && -z "$FIELD" ]]; then
    echo "Specify either --env-spec PATH or --field NAME" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

ARGS=(
    --dataset "$DATASET"
    --output-dir "$OUTPUT_DIR"
    --model "$MODEL"
    --server-url "$SERVER_URL"
    --concurrency "$CONCURRENCY"
    --max-solver-turns "$MAX_SOLVER_TURNS"
    --max-retries "$MAX_RETRIES"
)
if [[ "$VERIFIABLE" == "1" ]]; then
    ARGS+=(--verifiable)
fi
if [[ -n "$ENV_SPEC" ]]; then
    ARGS+=(--env-spec "$ENV_SPEC")
elif [[ -n "$FIELD" ]]; then
    ARGS+=(--field "$FIELD" --env-specs-dir "$ENV_SPECS_DIR")
fi
if [[ "$MAX_TASKS" != "0" && "$MAX_TASKS" != "" ]]; then
    ARGS+=(--max-tasks "$MAX_TASKS")
fi

echo "[try_task_gen] server: $SERVER_URL"
echo "[try_task_gen] model:  $MODEL"
echo "[try_task_gen] output: $OUTPUT_DIR"
echo "[try_task_gen] args:   ${ARGS[*]}"
echo

cd "$PROJ_DIR"
python -m task_generation.run "${ARGS[@]}"
PIPE_RC=$?
if [[ $PIPE_RC -ne 0 ]]; then
    echo "[try_task_gen] pipeline exited with $PIPE_RC"
    exit $PIPE_RC
fi

echo
echo "[try_task_gen] running_summary per produced task:"
shopt -s nullglob
for f in "$OUTPUT_DIR"/*.json; do
    case "$f" in *.debug.json) continue ;; esac
    echo "----- $f -----"
    python -c "
import json
t = json.loads(open('$f').read())
turns = t.get('turns') or []
for i, turn in enumerate(turns):
    rs = ((turn.get('task') or {}).get('running_summary') or '').strip()
    suffix = ' ...' if len(rs) > 200 else ''
    print(f'  turn {i} ({turn.get(\"tool_id\")}): {rs[:200]}{suffix}')
final = ''
for turn in reversed(turns):
    if turn.get('env_update') is None:
        continue
    rs = ((turn.get('task') or {}).get('running_summary') or '').strip()
    if rs:
        final = rs
        break
print(f'  FINAL running_summary: {final}')
"
done
