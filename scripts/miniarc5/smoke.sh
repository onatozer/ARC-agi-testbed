#!/bin/bash
# Phase-0 smoke: 200 steps + one full eval per architecture (W&B disabled), then a 2-task TTT on the TRM and TreeRM
# checkpoints (compiled search on and off, one freeze override). usage: smoke.sh
source "$(dirname "$0")/recipes.sh"
SMOKE="max_steps=200 eval_interval_steps=200 +wandb_mode=disabled"
export MINIARC5_RUNS=$HOME/ARC-agi-testbed/autoresearch_runs/miniarc5/smoke
for arch in trm varc treerm; do
  echo "== smoke $arch"; run_$arch "smoke_$arch" "$DATA_STD" $SMOKE
  "$WT/venv/bin/python" "$HERE/summarize.py" "$MINIARC5_RUNS/smoke_$arch" 2>&1 | tail -3
done
for arch in trm treerm; do
  ckpt=$(ckpt_of "smoke/smoke_$arch"); echo "== ttt smoke $arch: $ckpt"
  bash "$HERE/ttt.sh" "$ckpt" smoke_full max_tasks=2 epochs=2 num_aux_tasks=3 wandb_mode=disabled
  bash "$HERE/ttt.sh" "$ckpt" smoke_embed max_tasks=2 epochs=2 num_aux_tasks=3 wandb_mode=disabled train_overrides.freeze_weights=true
done
ckpt=$(ckpt_of "smoke/smoke_treerm")
bash "$HERE/ttt.sh" "$ckpt" smoke_eager max_tasks=2 epochs=2 num_aux_tasks=3 wandb_mode=disabled train_overrides.arch.search_compile=false
