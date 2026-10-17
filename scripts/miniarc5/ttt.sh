#!/bin/bash
# Test-time training of one checkpoint on MiniARC tasks under the shared GPU lock.
# usage: ttt.sh <checkpoint.pth> <suffix> <hydra overrides for evaluate_ttt.py...>   (eval=ttt_mini is the default recipe)
# Writes <run_dir>/eval_preds/step_<N>_<suffix>/{tasks/*.json,ttt_metrics.json} and logs eval/<suffix>/* to the checkpoint's W&B run.
set -u
CKPT=$1; SUFFIX=$2; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
WT=$(cd "$HERE/../.." && pwd)
LOCK=${GPU_LOCK:-$HOME/ARC-agi-testbed/autoresearch_runs/mcts_bench/gpu.lock}
LOG=$(dirname "$(dirname "$CKPT")")/ttt_$SUFFIX.log
START=$(date +%s)
cd "$WT"
flock "$LOCK" env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 WANDB_SILENT=true \
  /usr/bin/time -v venv/bin/python evaluate_ttt.py checkpoint="$CKPT" eval=ttt_mini suffix="$SUFFIX" "$@" > "$LOG" 2>&1
echo "exit=$? seconds=$(( $(date +%s) - START ))" >> "$LOG"
grep -E "tasks: |Maximum resident set size|^exit=|Error|Traceback" "$LOG" | tail -4
