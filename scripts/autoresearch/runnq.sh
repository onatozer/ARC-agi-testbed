#!/bin/bash
# Confirmation tier: 8x8 N-Queens. usage: runnq.sh <name> <hydra overrides...>
# Logs to $RUNS/<name>/train.log (offline W&B inside), prints the final eval summary. See REPORTS.md.
NAME=$1; shift
HERE=$(cd "$(dirname "$0")" && pwd)
WT=$(cd "$HERE/../.." && pwd)
RUNS=${AUTORESEARCH_RUNS:-$HOME/ARC-agi-testbed/autoresearch_runs/nqueens8}
RUNDIR=$RUNS/$NAME; rm -rf "$RUNDIR"; mkdir -p "$RUNDIR"
cd "$RUNDIR"
START=$(date +%s)
OUTPUT_ROOT="$RUNDIR" WANDB_SILENT=true PYTHONPATH="$WT" /home/onat_ozer/ARC-agi-testbed/venv/bin/python "$WT/pretrain.py" --config-path "$WT/config" --config-name cfg_pretrain_nqueens \
  dataset.data_path=/home/onat_ozer/ARC-agi-testbed/data/nqueens-8 dataset.board_size=8 \
  checkpoint_interval_steps=null heavy_metrics_log_interval=100000 steps_hist_log_interval_steps=100000 flops_log_interval_steps=100000 \
  +wandb_mode=offline "$@" > "$RUNDIR/train.log" 2>&1
echo "exit=$? seconds=$(( $(date +%s) - START ))" >> "$RUNDIR/train.log"
find "$RUNDIR" -name "*.pth" -delete
find "$RUNDIR" -type d -name code -prune -exec rm -rf {} +
tail -1 "$RUNDIR/train.log"
/home/onat_ozer/ARC-agi-testbed/venv/bin/python "$HERE/summary.py" "$RUNDIR"
