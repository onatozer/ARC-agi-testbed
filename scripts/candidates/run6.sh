#!/bin/bash
# One 6x6 Sudoku (unique solutions) training run under the shared GPU lock. usage: run6.sh <name> <hydra overrides...>
# Logs to $RUNS/<name>/train.log (W&B online to the user's project unless +wandb_mode=... is among the overrides; the
# local .wandb copy is what summary.py reads), appends the wall-clock, drops checkpoints. See REPORTS_CANDIDATES.md.
set -u
NAME=$1; shift
HERE=$(cd "$(dirname "$0")" && pwd)
WT=$(cd "$HERE/../.." && pwd)
RUNS=${CANDIDATES_RUNS:-$HOME/ARC-agi-testbed/autoresearch_runs/candidates6}
LOCK=${GPU_LOCK:-$HOME/ARC-agi-testbed/autoresearch_runs/mcts_bench/gpu.lock}
RUNDIR=$RUNS/$NAME; rm -rf "$RUNDIR"; mkdir -p "$RUNDIR"
cd "$RUNDIR"
START=$(date +%s)
flock -o "$LOCK" env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OUTPUT_ROOT="$RUNDIR" WANDB_SILENT=true PYTHONPATH="$WT" \
  /usr/bin/time -v "$WT/venv/bin/python" "$WT/pretrain.py" --config-path "$WT/config" --config-name cfg_pretrain \
  dataset.data_path="$WT/data/sudoku-6u" dataset.grid_size=6 dataset.name=Sudoku-6u \
  checkpoint_interval_steps=null heavy_metrics_log_interval=100000 steps_hist_log_interval_steps=100000 flops_log_interval_steps=100000 \
  +project_name=ARC-agi-testbed +entity=ozeronatwanb +run_name="cand6_$NAME" "$@" > "$RUNDIR/train.log" 2>&1
echo "exit=$? seconds=$(( $(date +%s) - START ))" >> "$RUNDIR/train.log"
find "$RUNDIR" -name "*.pth" -delete
find "$RUNDIR" -type d -name code -prune -exec rm -rf {} +
"$WT/venv/bin/python" "$HERE/summary.py" "$RUNDIR"
