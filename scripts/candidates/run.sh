#!/bin/bash
# One training run of a tier under the shared GPU lock. usage: run.sh <sudoku6|nqueens8|nqueens8ho|sudoku4|sudoku4u> <name> <hydra overrides...>
# Logs to $RUNS/<tier>/<name>/train.log (W&B online to the user's project as cand_<tier>_<name> unless +wandb_mode=...
# is among the overrides; the local .wandb copy is what summary.py reads), appends the wall-clock, drops checkpoints
# unless KEEP_CKPT is set.
set -u
TIER=$1; NAME=$2; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
WT=$(cd "$HERE/../.." && pwd)
RUNS=${CANDIDATES_RUNS:-$HOME/ARC-agi-testbed/autoresearch_runs/candidates}/$TIER
LOCK=${GPU_LOCK:-$HOME/ARC-agi-testbed/autoresearch_runs/mcts_bench/gpu.lock}
case "$TIER" in
  sudoku6)  CFG=cfg_pretrain; DATA="dataset.data_path=$WT/data/sudoku-6u dataset.grid_size=6 dataset.name=Sudoku-6u";;
  nqueens8) CFG=cfg_pretrain_nqueens; DATA="dataset.data_path=$WT/data/nqueens-8 dataset.board_size=8";;
  nqueens8ho) CFG=cfg_pretrain_nqueens; DATA="dataset.data_path=$WT/data/nqueens-8ho dataset.board_size=8";;
  sudoku4)  CFG=cfg_pretrain; DATA="dataset.data_path=$WT/data/sudoku-4ho dataset.grid_size=4 dataset.name=Sudoku-4ho";;
  sudoku4u) CFG=cfg_pretrain; DATA="dataset.data_path=$WT/data/sudoku-4hu dataset.grid_size=4 dataset.name=Sudoku-4hu";;
  *) echo "unknown tier $TIER" >&2; exit 2;;
esac
RUNDIR=$RUNS/$NAME; rm -rf "$RUNDIR"; mkdir -p "$RUNDIR"
cd "$RUNDIR"
START=$(date +%s)
flock -o "$LOCK" env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OUTPUT_ROOT="$RUNDIR" WANDB_SILENT=true PYTHONPATH="$WT" \
  /usr/bin/time -v "$WT/venv/bin/python" "$WT/pretrain.py" --config-path "$WT/config" --config-name "$CFG" $DATA \
  checkpoint_interval_steps=null heavy_metrics_log_interval=100000 steps_hist_log_interval_steps=100000 flops_log_interval_steps=100000 \
  +project_name=ARC-agi-testbed +entity=ozeronatwanb +run_name="cand_${TIER}_$NAME" "$@" > "$RUNDIR/train.log" 2>&1
echo "exit=$? seconds=$(( $(date +%s) - START ))" >> "$RUNDIR/train.log"
[ -n "${KEEP_CKPT:-}" ] || find "$RUNDIR" -name "*.pth" -delete
find "$RUNDIR" -type d -name code -prune -exec rm -rf {} +
"$WT/venv/bin/python" "$HERE/summary.py" "$RUNDIR"
