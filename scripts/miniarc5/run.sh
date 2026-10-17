#!/bin/bash
# One MiniARC 5x5 training run under the shared GPU lock. usage: run.sh <name> <config-name> <hydra overrides...>
# Logs to $RUNS/<name>/train.log (W&B online unless +wandb_mode=... is among the overrides), with the wall-clock and the
# process's peak RSS appended; keeps the checkpoints. See REPORTS_ARC.md.
set -u
NAME=$1; CFG=$2; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
WT=$(cd "$HERE/../.." && pwd)
RUNS=${MINIARC5_RUNS:-$HOME/ARC-agi-testbed/autoresearch_runs/miniarc5}
LOCK=${GPU_LOCK:-$HOME/ARC-agi-testbed/autoresearch_runs/mcts_bench/gpu.lock}
RUNDIR=$RUNS/$NAME; mkdir -p "$RUNDIR"
cd "$RUNDIR"
START=$(date +%s)
flock "$LOCK" env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OUTPUT_ROOT="$RUNDIR" WANDB_SILENT=true PYTHONPATH="$WT" \
  /usr/bin/time -v "$WT/venv/bin/python" "$WT/pretrain.py" --config-path "$WT/config" --config-name "$CFG" \
  dataset.grid_size=5 checkpoint_interval_steps=null heavy_metrics_log_interval=100000 steps_hist_log_interval_steps=100000 flops_log_interval_steps=100000 \
  "$@" > "$RUNDIR/train.log" 2>&1
echo "exit=$? seconds=$(( $(date +%s) - START ))" >> "$RUNDIR/train.log"
find "$RUNDIR" -type d -name code -prune -exec rm -rf {} +
grep -E "Maximum resident set size|^exit=" "$RUNDIR/train.log" | tail -2
