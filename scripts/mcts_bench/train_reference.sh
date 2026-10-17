#!/bin/bash
# Trains a reference TreeRM checkpoint with the autoresearch recipe (REPORTS.md section 0) and keeps its .pth and
# final full-test-set eval. usage: train_reference.sh <sudoku6u|nqueens6> [extra hydra overrides...]
# Output: $MCTS_BENCH_RUNS/ref_<env>/ (default ~/ARC-agi-testbed/autoresearch_runs/mcts_bench), train.log inside.
set -u
ENV=$1; shift
HERE=$(cd "$(dirname "$0")" && pwd)
WT=$(cd "$HERE/../.." && pwd)
RUNS=${MCTS_BENCH_RUNS:-$HOME/ARC-agi-testbed/autoresearch_runs/mcts_bench}
RUNDIR=$RUNS/ref_$ENV; rm -rf "$RUNDIR"; mkdir -p "$RUNDIR"
case $ENV in
  sudoku6u) CFG=cfg_pretrain; DS="dataset.data_path=$WT/data/sudoku-6u dataset.grid_size=6 dataset.name=Sudoku-6u"; STEPS=2000; LR=1e-3;;
  nqueens6) CFG=cfg_pretrain_nqueens; DS="dataset.data_path=$WT/data/nqueens-6 dataset.board_size=6"; STEPS=1000; LR=3e-4;;
  *) echo "unknown env $ENV"; exit 2;;
esac
# The REPORTS.md recipe (flag candidates, no candidate loss; arch/treerm.yaml holds K=4, 8 / 16 simulations, value
# bootstrap and the 0.1x head lr): the reference checkpoints of REFERENCE.md were trained with it
RECIPE="arch=treerm arch.hidden_size=128 arch.num_heads=4 arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 arch.halt_max_steps=8 \
arch.candidate_mode=flags arch.candidate_loss=none lr=$LR lr_warmup_steps=50 global_batch_size=64 max_steps=$STEPS epochs=100000 train_epochs_per_iter=100000 \
eval_interval_steps=$STEPS max_eval_steps=16 checkpoint_interval_steps=null heavy_metrics_log_interval=100000 steps_hist_log_interval_steps=100000 flops_log_interval_steps=100000"
cd "$RUNDIR"
START=$(date +%s)
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OUTPUT_ROOT="$RUNDIR" WANDB_SILENT=true PYTHONPATH="$WT" "$WT/venv/bin/python" "$WT/pretrain.py" \
  --config-path "$WT/config" --config-name $CFG $DS $RECIPE +wandb_mode=offline "$@" > "$RUNDIR/train.log" 2>&1
echo "exit=$? seconds=$(( $(date +%s) - START ))" >> "$RUNDIR/train.log"
find "$RUNDIR" -type d -name code -prune -exec rm -rf {} +
tail -1 "$RUNDIR/train.log"
"$WT/venv/bin/python" "$HERE/run_summary.py" "$RUNDIR"
