#!/bin/bash
# Evaluates one TreeRM checkpoint under several search budgets (config/eval/treerm_sims.yaml) under the shared GPU
# lock. usage: sims_sweep.sh <checkpoint.pth> '<sims list, e.g. [4,8,16,32]>' [seed] [batch] [dataset dir]
# (the dataset dir overrides the checkpoint's, which may point into a worktree that is gone).
# Results: <run dir>/eval_preds/step_<n>_sims_arch_eval_mcts_simulations-<N>/eval_metrics_step_<n>.json
set -u
CKPT=$1; SIMS=$2; SEED=${3:-0}; BATCH=${4:-250}; DATA=${5:-null}
HERE=$(cd "$(dirname "$0")" && pwd)
WT=$(cd "$HERE/../.." && pwd)
LOCK=${GPU_LOCK:-$HOME/ARC-agi-testbed/autoresearch_runs/mcts_bench/gpu.lock}
flock -o "$LOCK" env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$WT" "$WT/venv/bin/python" "$WT/evaluate.py" \
  --config-path "$WT/config" --config-name cfg_eval eval=treerm_sims \
  ckpt="$CKPT" sims="$SIMS" seed="$SEED" global_batch_size="$BATCH" dataset_data_path="$DATA"
