#!/bin/bash
# Rerun of the alternation + heads-frozen continuation (its first attempt was terminated by mistake while stopping the
# full-horizon run).
source "$(dirname "$0")/recipes.sh"
ckpt=$(ckpt_of treerm_std); echo "== continuation from $ckpt"
CONT="+load_checkpoint=$ckpt +load_weights_only=true max_steps=1000 eval_interval_steps=1000 lr_warmup_steps=50"
rm -rf "$HOME/ARC-agi-testbed/autoresearch_runs/miniarc5/treerm_cont_alt_heads"
run_treerm treerm_cont_alt_heads "$DATA_STD" $CONT freeze_alternation_epochs=25 "freeze_param_substrings=[candidate_,v_head]"
"$WT/venv/bin/python" "$HERE/summarize.py" "$HOME/ARC-agi-testbed/autoresearch_runs/miniarc5/treerm_cont_alt_heads"
