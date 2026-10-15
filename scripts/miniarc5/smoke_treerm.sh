#!/bin/bash
# Phase-0 smoke for TreeRM only: 200 steps + one full eval, then 2-task TTT variants (W&B disabled).
source "$(dirname "$0")/recipes.sh"
SMOKE="max_steps=200 eval_interval_steps=200 +wandb_mode=disabled"
export MINIARC5_RUNS=$HOME/ARC-agi-testbed/autoresearch_runs/miniarc5/smoke
rm -rf "$MINIARC5_RUNS/smoke_treerm"
echo "== smoke treerm"; run_treerm smoke_treerm "$DATA_STD" $SMOKE
"$WT/venv/bin/python" "$HERE/summarize.py" "$MINIARC5_RUNS/smoke_treerm" 2>&1 | tail -3
ckpt=$(ckpt_of "smoke/smoke_treerm"); echo "== ttt smoke treerm: $ckpt"
bash "$HERE/ttt.sh" "$ckpt" smoke_full  max_tasks=2 epochs=4 num_aux_tasks=3 wandb_mode=disabled
bash "$HERE/ttt.sh" "$ckpt" smoke_embed max_tasks=2 epochs=4 num_aux_tasks=3 wandb_mode=disabled train_overrides.freeze_weights=true
bash "$HERE/ttt.sh" "$ckpt" smoke_heads max_tasks=2 epochs=4 num_aux_tasks=3 wandb_mode=disabled "train_overrides.freeze_param_substrings=[candidate_,v_head]" train_overrides.freeze_alternation_epochs=2
bash "$HERE/ttt.sh" "$ckpt" smoke_eager max_tasks=2 epochs=4 num_aux_tasks=3 wandb_mode=disabled +train_overrides.arch.search_compile=false
grep -h "override " "$(dirname "$(dirname "$ckpt")")"/ttt_smoke_heads.log | head -5
