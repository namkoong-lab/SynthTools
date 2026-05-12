#!/bin/bash
# Launch N parallel summary_clean jobs, one per disjoint shard.
#
# Usage:
#   ./submit_summary_clean.sh             # default: 5 jobs (5 nodes × 4 GPUs each)
#   ./submit_summary_clean.sh 8           # 8 jobs

set -euo pipefail

N="${1:-5}"
SBATCH_FILE="$(dirname "$0")/summary_clean.sbatch"

echo "submitting $N parallel summary_clean shards (1 node × 4 GPUs each, premium QoS)"
echo "sbatch file : $SBATCH_FILE"
echo

for k in $(seq 0 $((N - 1))); do
    JOBID=$(sbatch --parsable \
        --job-name="summary-${k}of${N}" \
        "$SBATCH_FILE" "$k" "$N")
    echo "  shard ${k}/${N}: jobid=$JOBID"
done

echo
echo "all submitted. Watch progress with:"
echo "  squeue -u \$USER -o '%.12i %.20j %.2t %.10M %R'"
echo "  tail -f /global/homes/t/tcaste/projects/logs/summary_clean_summary-*.out"
